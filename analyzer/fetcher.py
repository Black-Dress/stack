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
import baostock as bs
from contextlib import redirect_stdout
from typing import Dict, Tuple, Optional

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

# 尝试导入 akshare
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
    @staticmethod
    def load_state(self) -> Dict:
        if not os.path.exists(STATE_FILE):
            return {}
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("状态文件损坏，重新初始化")
            return {}

    @staticmethod
    def save_state(self, state: Dict):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    # ---------- 缓存管理（权重缓存）----------
    def _get_cache_key_fuzzy(
        self,
        macro_status: str,
        sentiment_factor: float,
        market_above_ma20: bool,
        market_above_ma60: bool,
        market_amount_above_ma20: bool,
        volatility: float,
        cache_type: str = "weights",
    ) -> str:
        vol_bins = [0.01, 0.02, 0.03, 0.04]
        vol_level = discretize(volatility, vol_bins)
        sent_bins = [0.7, 0.85, 1.0, 1.15, 1.25]
        sent_level = discretize(sentiment_factor, sent_bins)
        today = datetime.date.today().strftime("%Y-%m-%d")
        key_str = f"{cache_type}_{today}_{macro_status}_{sent_level}_{market_above_ma20}_{market_above_ma60}_{market_amount_above_ma20}_{vol_level}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def _load_cache(self) -> Dict:
        if not os.path.exists(CACHE_FILE):
            return {}
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            now = time.time()
            expired = [
                k
                for k, v in cache.items()
                if isinstance(v, dict) and v.get("timestamp", 0) < now - 600
            ]
            for k in expired:
                del cache[k]
            if expired:
                self._save_cache(cache)
            return cache
        except:
            return {}

    def _save_cache(self, cache: Dict):
        for val in cache.values():
            if isinstance(val, dict) and "timestamp" not in val:
                val["timestamp"] = time.time()
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _get_market_cache_key(self, market_df: pd.DataFrame) -> str:
        """根据市场数据生成缓存键"""
        recent = market_df.tail(20)
        data_str = recent[["close", "volume", "macd_dif", "rsi"]].round(4).to_json()
        key = hashlib.md5(data_str.encode()).hexdigest()
        today = datetime.date.today().strftime("%Y-%m-%d")
        return f"market_state_{today}_{key}"

    def get_market_state(
        self, market_df: pd.DataFrame, ai_client: Optional["AIClient"] = None
    ) -> Tuple[str, float]:
        """获取市场状态，优先从缓存读取，否则调用AI或简单规则"""
        cache = self._load_cache()
        cache_key = self._get_market_cache_key(market_df)

        if cache_key in cache and isinstance(cache[cache_key], dict):
            entry = cache[cache_key]
            if "state" in entry and "factor" in entry:
                logger.info(f"使用缓存市场状态: {entry['state']}, 因子: {entry['factor']}")
                return entry["state"], entry["factor"]

        if ai_client:
            state, factor = ai_client.refine_market_state(market_df)
        else:
            # 后备简单规则
            last = market_df.iloc[-1]
            above_ma20 = last["close"] > last["ma_short"]
            above_ma60 = last["close"] > last.get("ma_long", last["ma_short"])
            if above_ma20 and above_ma60:
                state, factor = "正常牛市", 1.2
            elif not above_ma20 and not above_ma60:
                state, factor = "熊市下跌", 0.8
            else:
                state, factor = "震荡偏弱", 1.0

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

    # ---------- 情绪指标（akshare）----------
    def fetch_sentiment_indicators(self) -> dict:
        """获取多个情绪指标"""
        indicators = {}
        if not AKSHARE_AVAILABLE:
            return indicators

        try:
            df_north = ak.stock_hsgt_hist_em(symbol="北向资金")
            if not df_north.empty:
                latest = df_north.iloc[-1]
                indicators["north_net_inflow"] = float(latest["净流入"])
                indicators["north_net_inflow_20d_avg"] = float(
                    df_north["净流入"].tail(20).mean()
                )
        except Exception as e:
            logger.debug(f"北向资金获取失败: {e}")

        try:
            df_fund = ak.stock_market_fund_flow()
            if not df_fund.empty:
                latest = df_fund.iloc[-1]
                indicators["main_net_inflow_pct"] = float(latest["主力净流入-净占比"])
        except Exception as e:
            logger.debug(f"大盘资金流向获取失败: {e}")

        try:
            today_str = datetime.datetime.now().strftime("%Y%m%d")
            df_zt = ak.stock_zt_pool_em(date=today_str)
            df_dt = ak.stock_zt_pool_dtgc_em(date=today_str)
            zt_count = len(df_zt)
            dt_count = len(df_dt)
            indicators["zt_count"] = zt_count
            indicators["dt_count"] = dt_count
            indicators["zt_dt_ratio"] = zt_count / max(dt_count, 1)
        except Exception as e:
            logger.debug(f"涨跌停数据获取失败: {e}")

        try:
            df_summary = ak.stock_sse_summary()
            if "项目" in df_summary.columns:
                up_row = df_summary[df_summary["项目"] == "上涨家数"]
                down_row = df_summary[df_summary["项目"] == "下跌家数"]
                if not up_row.empty and not down_row.empty:
                    up_count = int(up_row.iloc[0, 1])
                    down_count = int(down_row.iloc[0, 1])
                    indicators["up_count"] = up_count
                    indicators["down_count"] = down_count
                    indicators["up_down_ratio"] = up_count / max(down_count, 1)
        except Exception as e:
            logger.debug(f"市场总貌获取失败: {e}")

        try:
            df_hot = ak.stock_hot_search_baidu(
                symbol="A股", date=datetime.datetime.now().strftime("%Y%m%d")
            )
            if "rank" in df_hot.columns:
                avg_rank = df_hot["rank"].head(10).mean()
                indicators["baidu_hot_avg_rank"] = avg_rank
        except Exception as e:
            logger.debug(f"百度热搜获取失败: {e}")

        try:
            df_vix = ak.index_option_50etf_qvix()
            if not df_vix.empty:
                latest_vix = df_vix.iloc[-1]["close"]
                vix_ma20 = df_vix["close"].tail(20).mean()
                indicators["vix"] = float(latest_vix)
                indicators["vix_ma20"] = float(vix_ma20)
        except Exception as e:
            logger.debug(f"VIX获取失败: {e}")

        return indicators

    def compute_sentiment_factor(self, indicators: dict) -> float:
        """根据情绪指标计算综合情绪因子"""
        if not indicators:
            return 1.0
        score = 0.0
        weight_total = 0.0

        # 北向资金
        north_net = indicators.get("north_net_inflow")
        north_avg = indicators.get("north_net_inflow_20d_avg")
        if north_net is not None and north_avg is not None and north_avg != 0:
            north_ratio = north_net / abs(north_avg)
            north_score = 1.0 + north_ratio * 0.5
        else:
            north_score = 1.0
        score += north_score * 0.25
        weight_total += 0.25

        # 主力资金
        main_pct = indicators.get("main_net_inflow_pct", 0.0)
        main_score = 1.0 + main_pct / 100 * 5
        score += main_score * 0.20
        weight_total += 0.20

        # 涨跌停比
        zt_dt_ratio = indicators.get("zt_dt_ratio", 1.0)
        zt_score = 1.0 + (zt_dt_ratio - 1.0) * 0.3
        score += zt_score * 0.15
        weight_total += 0.15

        # 上涨下跌比
        up_down_ratio = indicators.get("up_down_ratio", 1.0)
        ud_score = 1.0 + (up_down_ratio - 1.0) * 0.3
        score += ud_score * 0.15
        weight_total += 0.15

        # 百度热搜
        avg_rank = indicators.get("baidu_hot_avg_rank", 50.0)
        if avg_rank < 30:
            rank_score = 1.15
        elif avg_rank > 70:
            rank_score = 0.85
        else:
            rank_score = 1.0
        score += rank_score * 0.10
        weight_total += 0.10

        # VIX
        vix = indicators.get("vix")
        vix_ma = indicators.get("vix_ma20")
        if vix is not None and vix_ma is not None and vix_ma > 0:
            vix_ratio = vix / vix_ma
            vix_score = 1.0 - (vix_ratio - 1.0) * 0.8
        else:
            vix_score = 1.0
        score += vix_score * 0.15
        weight_total += 0.15

        if weight_total > 0:
            score /= weight_total
        return self._sentiment_adjustment(max(0.6, min(1.4, score)))

    def _sentiment_adjustment(self, sentiment: float) -> float:
        """情绪因子非线性调整"""
        x = sentiment - 1.0
        if x >= 0:
            adj = 1.0 + 1.2 * math.tanh(3.0 * x) * math.exp(-0.8 * x)
        else:
            adj = 1.0 + 1.2 * math.tanh(3.0 * x)
        return max(0.6, min(1.5, adj))

    def get_sentiment_risk_tip(self, sentiment_factor: float) -> str:
        """根据情绪因子生成风险提示文字"""
        if sentiment_factor >= 1.25:
            return "⚠️  市场情绪过热，短期回调风险较高，追高需谨慎"
        elif sentiment_factor >= 1.10:
            return "市场情绪偏乐观，警惕过热迹象"
        elif sentiment_factor >= 0.85:
            return "市场情绪平稳"
        elif sentiment_factor >= 0.70:
            return "💡  市场情绪偏悲观，可关注错杀机会"
        else:
            return "💡💡  市场情绪极度恐慌，往往是左侧布局良机"

    def get_sentiment_factor_simple(self, macro_df: pd.DataFrame) -> float:
        """后备情绪因子（基于RSI）"""
        if len(macro_df) < RSI_PERIOD + 1:
            return 1.0
        delta = macro_df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
        rsi = 100 - 100 / (1 + gain / loss)
        latest_rsi = rsi.iloc[-1]
        if latest_rsi < 30:
            return 0.6
        if latest_rsi < 50:
            return 0.8
        if latest_rsi < 70:
            return 1.0
        return 0.9