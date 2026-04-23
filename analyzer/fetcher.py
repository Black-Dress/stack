#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据获取模块：负责所有数据获取（baostock、新浪、akshare情绪指标）、状态管理及缓存。
"""
import os
import json
import datetime
import logging
import requests
import hashlib
import time
import math
import pandas as pd
import numpy as np
import baostock as bs
from contextlib import redirect_stdout
from typing import Dict, Tuple, Optional, List

from analyzer.ai import AIClient

from .config import (
    STATE_FILE,
    CACHE_FILE,
    POSITION_FILE,
    RSI_PERIOD,
    get_email_config,
)
from .utils import discretize

logger = logging.getLogger(__name__)

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("未安装 akshare，情绪指标功能将不可用。请执行 pip install akshare")


class DataFetcher:
    """负责所有数据获取（baostock, sina, akshare 情绪指标）"""

    def __init__(self):
        self._logged_in = False
        self._sentiment_history: List[float] = []      # 平滑后的历史情绪值（滚动）
        self._raw_sentiment_history: List[float] = []  # 原始值（用于动态阈值）
        self._last_trade_date: Optional[str] = None
        self._breadth_cache: Optional[Tuple[float, float, float]] = None  # (timestamp, up, down)
        self._load_histories_from_cache()

    # ---------- 历史序列与缓存整合 ----------
    def _load_histories_from_cache(self):
        """从 CACHE_FILE 中加载滚动历史列表（用于动态阈值）"""
        cache = self._load_cache()
        if "_sentiment_history" in cache:
            self._sentiment_history = cache["_sentiment_history"]
        if "_raw_sentiment_history" in cache:
            self._raw_sentiment_history = cache["_raw_sentiment_history"]

    def _save_histories_to_cache(self):
        """将滚动历史列表写入缓存文件（同时保留每日归档记录）"""
        cache = self._load_cache()
        cache["_sentiment_history"] = self._sentiment_history[-60:]
        cache["_raw_sentiment_history"] = self._raw_sentiment_history[-60:]
        self._save_cache(cache)

    def login(self) -> bool:
        """登录 baostock"""
        with open(os.devnull, "w") as f, redirect_stdout(f):
            lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"登录失败: {lg.error_msg}")
            return False
        self._logged_in = True
        return True

    def logout(self):
        """登出 baostock"""
        if self._logged_in:
            with open(os.devnull, "w") as f, redirect_stdout(f):
                bs.logout()
            self._logged_in = False

    # ---------- 状态管理 ----------
    def load_state(self) -> Dict:
        if not os.path.exists(STATE_FILE):
            return {}
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("状态文件损坏，重新初始化")
            return {}

    def save_state(self, state: Dict):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ---------- 缓存管理（权重/市场/情绪统一缓存）----------
    def _get_daily_cache_key(self) -> str:
        """返回当日日期字符串，作为环境缓存的键"""
        return datetime.date.today().strftime("%Y-%m-%d")

    def _load_cache(self) -> Dict:
        """加载整个缓存文件"""
        if not os.path.exists(CACHE_FILE):
            return {}
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            # 清理过期的权重缓存条目（保留日期键和特殊键）
            now = time.time()
            keys_to_delete = []
            for k, v in cache.items():
                if isinstance(v, dict) and "timestamp" in v:
                    # 只清理有过期时间戳且超过10分钟的条目（排除滚动历史等特殊键）
                    if k not in ["_sentiment_history", "_raw_sentiment_history"]:
                        if v["timestamp"] < now - 600:
                            keys_to_delete.append(k)
            for k in keys_to_delete:
                del cache[k]
            return cache
        except:
            return {}

    def _save_cache(self, cache: Dict):
        """保存整个缓存文件"""
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def get_cached_environment(self) -> Optional[Dict]:
        """
        获取当日缓存的环境数据（市场状态、因子、权重、情绪）。
        若缓存存在且未超过10分钟，则返回数据；否则返回 None。
        """
        cache = self._load_cache()
        key = self._get_daily_cache_key()
        if key not in cache:
            return None
        data = cache[key]
        if not isinstance(data, dict):
            return None
        ts = data.get("timestamp", 0)
        if time.time() - ts < 600:   # 10分钟内有效
            return data
        return None

    def save_environment_cache(self, market_state: str, market_factor: float,
                               buy_weights: Dict, sell_weights: Dict,
                               sentiment: float):
        """
        保存当日环境数据到缓存（覆盖）。
        同时更新滚动情绪历史列表。
        """
        cache = self._load_cache()
        key = self._get_daily_cache_key()
        cache[key] = {
            "market_state": market_state,
            "market_factor": market_factor,
            "buy_weights": buy_weights,
            "sell_weights": sell_weights,
            "sentiment": sentiment,
            "timestamp": time.time()
        }
        # 持久化滚动历史
        cache["_sentiment_history"] = self._sentiment_history[-60:]
        cache["_raw_sentiment_history"] = self._raw_sentiment_history[-60:]
        self._save_cache(cache)

    # ---------- 市场状态缓存辅助（兼容旧逻辑）----------
    def _get_market_cache_key(self, market_df: pd.DataFrame) -> str:
        """根据市场数据生成缓存键（保留用于兼容）"""
        recent = market_df.tail(20)
        data_str = recent[["close", "volume", "macd_dif", "rsi"]].round(4).to_json()
        key = hashlib.md5(data_str.encode()).hexdigest()
        today = datetime.date.today().strftime("%Y-%m-%d")
        return f"market_state_{today}_{key}"

    def get_market_state(
        self, market_df: pd.DataFrame, ai_client: Optional["AIClient"] = None
    ) -> Tuple[str, float]:
        """获取市场状态（优先使用当日统一缓存）"""
        # 优先尝试统一缓存
        cached_env = self.get_cached_environment()
        if cached_env:
            return cached_env["market_state"], cached_env["market_factor"]

        # 若无统一缓存，尝试旧的独立缓存
        cache = self._load_cache()
        cache_key = self._get_market_cache_key(market_df)
        if cache_key in cache and isinstance(cache[cache_key], dict):
            entry = cache[cache_key]
            if "state" in entry and "factor" in entry:
                logger.info(f"使用旧缓存市场状态: {entry['state']}, 因子: {entry['factor']}")
                return entry["state"], entry["factor"]

        # 计算新状态
        if ai_client:
            state, factor = ai_client.refine_market_state(market_df)
        else:
            last = market_df.iloc[-1]
            above_ma20 = last["close"] > last["ma_short"]
            above_ma60 = last["close"] > last.get("ma_long", last["ma_short"])
            if above_ma20 and above_ma60:
                state, factor = "正常牛市", 1.2
            elif not above_ma20 and not above_ma60:
                state, factor = "熊市下跌", 0.8
            else:
                state, factor = "震荡偏弱", 1.0

        # 保存到旧缓存（可选）
        cache[cache_key] = {"state": state, "factor": factor, "timestamp": time.time()}
        self._save_cache(cache)
        logger.info(f"市场状态已缓存: {state}, 因子: {factor}")
        return state, factor

    def get_daily_data(
        self, code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取日线数据（原始数据，未计算指标）"""
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
        )
        if rs.error_code != "0":
            return None
        data = []
        while rs.next():
            data.append(rs.get_row_data())
        if not data:
            return None
        df = pd.DataFrame(data, columns=rs.fields)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        for col in ["open", "high", "low", "close", "volume", "amount"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_weekly_data(
        self, code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """获取周线数据（基于日线重采样）"""
        from .config import WEEKLY_MA
        df = self.get_daily_data(code, start_date, end_date)
        if df is None:
            return None
        weekly = df.resample("W-FRI").last()
        weekly["ma_short"] = weekly["close"].rolling(window=WEEKLY_MA).mean()
        return weekly

    def get_realtime_price(self, code: str) -> Optional[float]:
        """获取新浪实时价格"""
        try:
            for domain in ["hq.sinajs.cn", "hq2.sinajs.cn", "hq3.sinajs.cn"]:
                url = f"http://{domain}/list={code.replace('.','')}"
                r = requests.get(
                    url, headers={"Referer": "http://finance.sina.com.cn"}, timeout=3
                )
                if r.status_code == 200:
                    parts = r.text.split('"')[1].split(",")
                    if len(parts) > 3 and parts[3]:
                        return float(parts[3])
            return None
        except Exception:
            return None

    def load_positions(self) -> pd.DataFrame:
        """加载持仓列表（CSV）"""
        try:
            df = pd.read_csv(POSITION_FILE, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(POSITION_FILE, encoding="gbk")
        return df[["代码", "名称"]]

    # ---------- 辅助：获取最近交易日 ----------
    def _get_latest_trade_date(self) -> str:
        if self._last_trade_date:
            return self._last_trade_date
        today = datetime.date.today()
        for i in range(5):
            check_date = today - datetime.timedelta(days=i)
            if check_date.weekday() < 5:
                self._last_trade_date = check_date.strftime("%Y%m%d")
                break
        if not self._last_trade_date:
            self._last_trade_date = today.strftime("%Y%m%d")
        return self._last_trade_date

    # ---------- 情绪指标获取 ----------
    def fetch_sentiment_indicators(self) -> dict:
        raw = {}
        if not AKSHARE_AVAILABLE:
            return raw

        # 1. 北向资金（过滤NaN）
        try:
            df_north = ak.stock_hsgt_hist_em(symbol="北向资金")
            if not df_north.empty:
                north_series = df_north["当日成交净买额"].dropna()
                if len(north_series) > 0:
                    latest = north_series.iloc[-1]
                    raw["north_net"] = float(latest)
                    raw["north_net_20d_avg"] = float(north_series.tail(20).mean())
        except Exception as e:
            logger.debug(f"北向资金获取失败: {e}")

        # 2. 主力资金（候选列名）
        try:
            df_fund = ak.stock_market_fund_flow()
            if not df_fund.empty:
                latest = df_fund.iloc[-1]
                for col in ["主力净流入-净占比", "主力净流入-净占比(%)", "主力净流入净占比"]:
                    if col in latest.index:
                        raw["main_net_pct"] = float(latest[col])
                        break
        except Exception as e:
            logger.debug(f"主力资金流向获取失败: {e}")

        # 3. 涨跌停比（最多回退3天）
        try:
            trade_date = self._get_latest_trade_date()
            zt_count = dt_count = 0
            for offset in range(4):
                dt = (datetime.datetime.strptime(trade_date, "%Y%m%d") - datetime.timedelta(days=offset)).strftime("%Y%m%d")
                df_zt = ak.stock_zt_pool_em(date=dt)
                df_dt = ak.stock_zt_pool_dtgc_em(date=dt)
                zt_count = len(df_zt) if not df_zt.empty else 0
                dt_count = len(df_dt) if not df_dt.empty else 0
                if zt_count > 0 or dt_count > 0:
                    break
            raw["zt_count"] = zt_count
            raw["dt_count"] = dt_count
            raw["zt_dt_ratio"] = zt_count / max(dt_count, 1)
        except Exception as e:
            logger.debug(f"涨跌停数据获取失败: {e}")

        # 4. 上涨下跌家数比（缓存5分钟）
        try:
            now = time.time()
            if self._breadth_cache and (now - self._breadth_cache[0]) < 300:
                up_count, down_count = self._breadth_cache[1], self._breadth_cache[2]
            else:
                df_spot = ak.stock_zh_a_spot()
                up_count = down_count = 0
                if not df_spot.empty and "涨跌幅" in df_spot.columns:
                    up_count = len(df_spot[df_spot["涨跌幅"] > 0])
                    down_count = len(df_spot[df_spot["涨跌幅"] < 0])
                self._breadth_cache = (now, up_count, down_count)
            raw["up_count"] = up_count
            raw["down_count"] = down_count
            raw["up_down_ratio"] = up_count / max(down_count, 1)
        except Exception as e:
            logger.debug(f"上涨下跌家数获取失败: {e}")

        # 5. 历史波动率
        try:
            df_300 = ak.stock_zh_index_daily(symbol="sh000300")
            if not df_300.empty:
                df_300 = df_300.sort_values("date")
                df_300["ret"] = df_300["close"].pct_change()
                hv = df_300["ret"].rolling(20).std() * math.sqrt(252) * 100
                latest_hv = hv.iloc[-1]
                if pd.isna(latest_hv):
                    latest_hv = 20.0
                raw["hv"] = float(latest_hv)
                raw["hv_ma20"] = float(hv.tail(20).mean()) if len(hv) >= 20 else raw["hv"]
        except Exception as e:
            logger.debug(f"历史波动率获取失败: {e}")

        # 6. 融资余额变化率
        try:
            start = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y%m%d")
            end = datetime.datetime.now().strftime("%Y%m%d")
            df_margin = ak.stock_margin_sse(start_date=start, end_date=end)
            if not df_margin.empty:
                col = "融资余额" if "融资余额" in df_margin.columns else "rzye"
                if col in df_margin.columns:
                    df_margin[col] = pd.to_numeric(df_margin[col], errors="coerce")
                    recent = df_margin[col].dropna().tail(5)
                    if len(recent) >= 2:
                        change = (recent.iloc[-1] / recent.mean() - 1.0) * 100
                        raw["margin_change"] = change
        except Exception as e:
            logger.debug(f"融资余额获取失败: {e}")

        return raw

    # ---------- 标准化函数 ----------
    @staticmethod
    def _normalize_north(north_net: float, north_avg: float) -> float:
        if north_avg is None or north_avg == 0:
            return 1.0
        ratio = north_net / (abs(north_avg) + 1e-8)
        ratio = max(-3.0, min(3.0, ratio))
        score = 1.0 + ratio * 0.3
        return max(0.6, min(1.4, score))

    @staticmethod
    def _normalize_main(main_pct: float) -> float:
        score = 1.0 + main_pct / 100 * 8
        return max(0.6, min(1.4, score))

    @staticmethod
    def _normalize_zt_dt(ratio: float) -> float:
        if ratio <= 0.1:
            return 0.6
        score = 1.0 + math.log10(ratio) * 0.3
        return max(0.6, min(1.4, score))

    @staticmethod
    def _normalize_up_down(ratio: float) -> float:
        if ratio <= 0.1:
            return 0.6
        score = 1.0 + math.log10(ratio) * 0.3
        return max(0.6, min(1.4, score))

    @staticmethod
    def _normalize_volatility(hv: float, hv_ma: float) -> float:
        if hv_ma is None or hv_ma == 0:
            return 1.0
        ratio = hv / hv_ma
        score = 1.0 - (ratio - 1.0) * 0.8
        return max(0.4, min(1.6, score))

    @staticmethod
    def _normalize_margin(change_pct: float) -> float:
        change_pct = max(-10.0, min(10.0, change_pct))
        score = 1.0 + change_pct / 5 * 0.3
        return max(0.6, min(1.4, score))

    # ---------- 情绪因子合成（含动态权重）----------
    def compute_sentiment_factor(self, indicators: dict) -> float:
        if not indicators:
            return 1.0

        weights = {
            "north": 0.25,
            "main": 0.20,
            "zt_dt": 0.15,
            "up_down": 0.20,
            "volatility": 0.10,
            "margin": 0.10,
        }

        # 动态调整：高波动时降低资金流权重，提高波动率权重
        hv = indicators.get("hv", 20.0)
        if hv > 30:
            weights["north"] *= 0.7
            weights["main"] *= 0.7
            weights["volatility"] = 0.20
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}

        scores = {}
        if "north_net" in indicators and "north_net_20d_avg" in indicators:
            scores["north"] = self._normalize_north(indicators["north_net"], indicators["north_net_20d_avg"])
        else:
            scores["north"] = 1.0

        if "main_net_pct" in indicators:
            scores["main"] = self._normalize_main(indicators["main_net_pct"])
        else:
            scores["main"] = 1.0

        if "zt_dt_ratio" in indicators:
            scores["zt_dt"] = self._normalize_zt_dt(indicators["zt_dt_ratio"])
        else:
            scores["zt_dt"] = 1.0

        if "up_down_ratio" in indicators:
            scores["up_down"] = self._normalize_up_down(indicators["up_down_ratio"])
        else:
            scores["up_down"] = 1.0

        if "hv" in indicators and "hv_ma20" in indicators:
            scores["volatility"] = self._normalize_volatility(indicators["hv"], indicators["hv_ma20"])
        else:
            scores["volatility"] = 1.0

        if "margin_change" in indicators:
            scores["margin"] = self._normalize_margin(indicators["margin_change"])
        else:
            scores["margin"] = 1.0

        raw_sentiment = sum(scores[k] * weights[k] for k in weights)
        raw_sentiment = max(0.6, min(1.5, raw_sentiment))

        self._raw_sentiment_history.append(raw_sentiment)
        if len(self._raw_sentiment_history) > 60:
            self._raw_sentiment_history.pop(0)

        # EWMA 平滑 (alpha=0.3)
        if self._sentiment_history:
            smoothed = 0.7 * self._sentiment_history[-1] + 0.3 * raw_sentiment
        else:
            smoothed = raw_sentiment
        self._sentiment_history.append(smoothed)
        if len(self._sentiment_history) > 60:
            self._sentiment_history.pop(0)

        self._save_histories_to_cache()
        return self._sentiment_adjustment(smoothed)

    def _sentiment_adjustment(self, sentiment: float) -> float:
        x = sentiment - 1.0
        if x >= 0:
            adj = 1.0 + 1.2 * math.tanh(3.0 * x) * math.exp(-0.8 * x)
        else:
            adj = 1.0 + 1.2 * math.tanh(3.0 * x)
        return max(0.6, min(1.5, adj))

    # ---------- 风险提示（动态阈值）----------
    def get_sentiment_risk_tip(self, sentiment_factor: float) -> str:
        if len(self._raw_sentiment_history) >= 20:
            recent = sorted(self._raw_sentiment_history[-20:])
            overheat = recent[int(len(recent) * 0.8)]
            panic = recent[int(len(recent) * 0.2)]
        else:
            overheat, panic = 1.25, 0.75

        if sentiment_factor >= overheat:
            return f"⚠️ 市场情绪过热(>{overheat:.2f})，短期回调风险较高"
        elif sentiment_factor >= 1.10:
            return "市场情绪偏乐观"
        elif sentiment_factor >= 0.85:
            return "市场情绪平稳"
        elif sentiment_factor >= panic:
            return "💡 市场情绪偏悲观，可关注错杀机会"
        else:
            return f"💡💡 市场情绪极度恐慌(<{panic:.2f})，左侧布局良机"

    # ---------- 后备情绪因子（连续映射+平滑）----------
    def get_sentiment_factor_simple(self, macro_df: pd.DataFrame) -> float:
        if len(macro_df) < RSI_PERIOD + 1:
            return 1.0
        delta = macro_df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        latest_rsi = rsi.iloc[-1]
        raw_sentiment = 1.0 + (50 - latest_rsi) / 100
        raw_sentiment = max(0.6, min(1.4, raw_sentiment))

        self._raw_sentiment_history.append(raw_sentiment)
        if len(self._raw_sentiment_history) > 60:
            self._raw_sentiment_history.pop(0)

        if self._sentiment_history:
            smoothed = 0.7 * self._sentiment_history[-1] + 0.3 * raw_sentiment
        else:
            smoothed = raw_sentiment
        self._sentiment_history.append(smoothed)
        if len(self._sentiment_history) > 60:
            self._sentiment_history.pop(0)

        self._save_histories_to_cache()
        return self._sentiment_adjustment(smoothed)