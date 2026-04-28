#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
数据获取模块：负责所有数据获取（baostock、新浪、akshare情绪指标）、状态管理及缓存。
深度融合后增加 extra_sentiment 传递和 ai_params_advice 缓存。
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

from .ai import AIClient
from .config import (
    STATE_FILE,
    CACHE_FILE,
    POSITION_FILE,
    RSI_PERIOD,
    SENTIMENT_LOWER_BOUND,
    get_email_config,
)
from .utils import discretize, apply_sentiment_adjustment, fallback_market_state

logger = logging.getLogger(__name__)

try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.warning("未安装 akshare，情绪指标功能将不可用。请执行 pip install akshare")

MAX_HISTORY_DAYS = 60


class DataFetcher:
    """负责所有数据获取（baostock, sina, akshare 情绪指标）"""

    def __init__(self):
        self._logged_in = False
        self._sentiment_history: List[float] = []
        self._raw_sentiment_history: List[float] = []
        self._last_trade_date: Optional[str] = None
        self._breadth_cache: Optional[Tuple[float, float, float]] = None
        self._rebuild_histories_from_cache()

    def _rebuild_histories_from_cache(self):
        """启动时从缓存文件恢复情绪历史"""
        cache = self._load_cache()
        date_list = cache.get("_env_date_list", [])
        if not date_list:
            env_keys = sorted([k for k in cache.keys() if k.startswith("env_")])
            date_list = [k[4:] for k in env_keys]
        for date_str in date_list[-MAX_HISTORY_DAYS:]:
            key = f"env_{date_str}"
            entry = cache.get(key)
            if isinstance(entry, dict):
                if "sentiment" in entry:
                    self._sentiment_history.append(entry["sentiment"])
                if "sentiment_raw" in entry:
                    self._raw_sentiment_history.append(entry["sentiment_raw"])
        self._sentiment_history = self._sentiment_history[-MAX_HISTORY_DAYS:]
        self._raw_sentiment_history = self._raw_sentiment_history[-MAX_HISTORY_DAYS:]

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

    def _load_cache(self) -> Dict:
        if not os.path.exists(CACHE_FILE):
            return {}
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def _save_cache(self, cache: Dict):
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _get_env_key(self, date_str: str = None) -> str:
        if date_str is None:
            date_str = datetime.date.today().strftime("%Y-%m-%d")
        return f"env_{date_str}"

    def get_cached_environment(self) -> Optional[Dict]:
        cache = self._load_cache()
        key = self._get_env_key()
        if key not in cache:
            return None
        data = cache[key]
        if not isinstance(data, dict):
            return None
        ts = data.get("timestamp", 0)
        if time.time() - ts < 600:
            return data
        return None

    def save_environment_cache(self, market_state: str, market_factor: float,
                               buy_weights: Dict, sell_weights: Dict,
                               sentiment: float, sentiment_raw: float,
                               ai_params_advice: Optional[Dict] = None):
        """保存当天的市场环境、权重、情绪以及AI参数建议到缓存"""
        cache = self._load_cache()
        key = self._get_env_key()
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        is_new_day = key not in cache

        cache[key] = {
            "market_state": market_state,
            "market_factor": market_factor,
            "buy_weights": buy_weights,
            "sell_weights": sell_weights,
            "sentiment": sentiment,
            "sentiment_raw": sentiment_raw,
            "timestamp": time.time(),
            "ai_params_advice": ai_params_advice,  # 新增
        }

        date_list = cache.get("_env_date_list", [])
        if today_str not in date_list:
            date_list.append(today_str)
        if len(date_list) > MAX_HISTORY_DAYS * 2:
            date_list = date_list[-MAX_HISTORY_DAYS * 2:]
        cache["_env_date_list"] = date_list

        if is_new_day:
            self._sentiment_history.append(sentiment)
            self._raw_sentiment_history.append(sentiment_raw)
            if len(self._sentiment_history) > MAX_HISTORY_DAYS:
                self._sentiment_history.pop(0)
            if len(self._raw_sentiment_history) > MAX_HISTORY_DAYS:
                self._raw_sentiment_history.pop(0)

        self._save_cache(cache)

    def _get_market_cache_key(self, market_df: pd.DataFrame) -> str:
        recent = market_df.tail(20)
        data_str = recent[["close", "volume", "macd_dif", "rsi"]].round(4).to_json()
        key = hashlib.md5(data_str.encode()).hexdigest()
        today = datetime.date.today().strftime("%Y-%m-%d")
        return f"market_state_{today}_{key}"

    def get_market_state(
        self, market_df: pd.DataFrame, ai_client: Optional["AIClient"] = None,
        extra_sentiment: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """获取市场状态和因子（可接受额外情绪指标）"""
        # 如果提供了额外情绪指标，则直接向AI请求最新分析，不使用缓存
        if extra_sentiment is not None:
            if ai_client:
                return ai_client.refine_market_state(market_df, extra_sentiment=extra_sentiment)
            else:
                last = market_df.iloc[-1]
                above_ma20 = last["close"] > last["ma_short"]
                above_ma60 = last["close"] > last.get("ma_long", last["ma_short"])
                return fallback_market_state(above_ma20, above_ma60)

        # 常规路线：先查环境缓存，再查独立缓存
        cached = self.get_cached_environment()
        if cached:
            return cached["market_state"], cached["market_factor"]

        cache = self._load_cache()
        cache_key = self._get_market_cache_key(market_df)
        if cache_key in cache and isinstance(cache[cache_key], dict):
            entry = cache[cache_key]
            if "state" in entry and "factor" in entry:
                return entry["state"], entry["factor"]

        if ai_client:
            state, factor = ai_client.refine_market_state(market_df)
        else:
            last = market_df.iloc[-1]
            above_ma20 = last["close"] > last["ma_short"]
            above_ma60 = last["close"] > last.get("ma_long", last["ma_short"])
            state, factor = fallback_market_state(above_ma20, above_ma60)

        cache[cache_key] = {"state": state, "factor": factor, "timestamp": time.time()}
        self._save_cache(cache)
        return state, factor

    def get_daily_data(
        self, code: str, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
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
        from .config import WEEKLY_MA
        df = self.get_daily_data(code, start_date, end_date)
        if df is None:
            return None
        weekly = df.resample("W-FRI").last()
        weekly["ma_short"] = weekly["close"].rolling(window=WEEKLY_MA).mean()
        return weekly

    def get_realtime_price(self, code: str) -> Optional[float]:
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
        try:
            df = pd.read_csv(POSITION_FILE, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(POSITION_FILE, encoding="gbk")
        return df[["代码", "名称"]]

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

        try:
            df_north = ak.stock_hsgt_hist_em(symbol="北向资金")
            if not df_north.empty:
                north_series = df_north["当日成交净买额"].dropna()
                if len(north_series) > 0:
                    raw["north_net"] = float(north_series.iloc[-1])
                    raw["north_net_20d_avg"] = float(north_series.tail(20).mean())
        except Exception as e:
            logger.debug(f"北向资金获取失败: {e}")

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

    def compute_sentiment_factor(self, indicators: dict) -> Tuple[float, float]:
        if not indicators:
            return 1.0, 1.0

        weights = {
            "north": 0.25,
            "main": 0.20,
            "zt_dt": 0.15,
            "up_down": 0.20,
            "volatility": 0.10,
            "margin": 0.10,
        }

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

        scores["main"] = self._normalize_main(indicators.get("main_net_pct", 0.0)) if "main_net_pct" in indicators else 1.0
        scores["zt_dt"] = self._normalize_zt_dt(indicators.get("zt_dt_ratio", 1.0)) if "zt_dt_ratio" in indicators else 1.0
        scores["up_down"] = self._normalize_up_down(indicators.get("up_down_ratio", 1.0)) if "up_down_ratio" in indicators else 1.0

        if "hv" in indicators and "hv_ma20" in indicators:
            scores["volatility"] = self._normalize_volatility(indicators["hv"], indicators["hv_ma20"])
        else:
            scores["volatility"] = 1.0

        scores["margin"] = self._normalize_margin(indicators.get("margin_change", 0.0)) if "margin_change" in indicators else 1.0

        raw_sentiment = sum(scores[k] * weights[k] for k in weights)
        raw_sentiment = max(0.6, min(1.5, raw_sentiment))

        if self._sentiment_history:
            smoothed = 0.7 * self._sentiment_history[-1] + 0.3 * raw_sentiment
        else:
            smoothed = raw_sentiment

        smoothed = apply_sentiment_adjustment(smoothed)
        smoothed = max(SENTIMENT_LOWER_BOUND, smoothed)
        return smoothed, raw_sentiment

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

    def get_sentiment_factor_simple(self, macro_df: pd.DataFrame) -> Tuple[float, float]:
        if len(macro_df) < RSI_PERIOD + 1:
            return 1.0, 1.0
        delta = macro_df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        latest_rsi = rsi.iloc[-1]
        raw_sentiment = 1.0 + (50 - latest_rsi) / 100
        raw_sentiment = max(0.6, min(1.4, raw_sentiment))

        if self._sentiment_history:
            smoothed = 0.7 * self._sentiment_history[-1] + 0.3 * raw_sentiment
        else:
            smoothed = raw_sentiment

        smoothed = apply_sentiment_adjustment(smoothed)
        smoothed = max(SENTIMENT_LOWER_BOUND, smoothed)
        return smoothed, raw_sentiment