#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""数据层：行情获取、指标计算、持仓历史记录 + 状态管理（合并 state_manager）"""
import os
import json
import datetime
import logging
import time
import random
import requests
import pandas as pd
import numpy as np
import baostock as bs
from contextlib import redirect_stdout
from typing import Dict, Tuple, Optional, List, Any
from .config import BUY_WEIGHTS_BULL, BUY_WEIGHTS_RANGE, BUY_WEIGHTS_BEAR, SELL_WEIGHTS_BULL, SELL_WEIGHTS_RANGE, SELL_WEIGHTS_BEAR
from .config import *
from .utils import calc_rsi, calc_macd, calculate_atr, calculate_adx

logger = logging.getLogger(__name__)


class DataLayer:
    def __init__(self):
        self._logged_in = False
        self._price_cache = {}

    def login(self) -> bool:
        with open(os.devnull, "w") as f, redirect_stdout(f):
            lg = bs.login()
        if lg.error_code != "0":
            logger.error(f"登录失败: {lg.error_msg}")
            return False
        self._logged_in = True
        return True

    def logout(self):
        if self._logged_in:
            with open(os.devnull, "w") as f, redirect_stdout(f):
                bs.logout()
            self._logged_in = False

    # ---------- 行情获取 ----------
    def get_daily_data(self, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        rs = bs.query_history_k_data_plus(
            code, "date,code,open,high,low,close,volume,amount",
            start_date=start_date, end_date=end_date, frequency="d"
        )
        if rs is None:
            return None
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

    def get_weekly_data(self, code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        df = self.get_daily_data(code, start_date, end_date)
        if df is None:
            return None
        weekly = df.resample("W-FRI").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum"
        })
        today_ts = pd.Timestamp(datetime.date.today())
        weekly = weekly[weekly.index < today_ts]
        if weekly.empty:
            return None
        weekly["ma_short"] = weekly["close"].rolling(ETF_MA).mean()
        return weekly

    def get_realtime_price(self, code: str, cache_ttl_seconds: int = 60) -> Optional[float]:
        now = datetime.datetime.now()
        cache_key = (code, now.strftime("%Y-%m-%d %H:%M"))
        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                for domain in ["hq.sinajs.cn", "hq2.sinajs.cn", "hq3.sinajs.cn"]:
                    url = f"http://{domain}/list={code.replace('.', '')}"
                    r = requests.get(url, headers={"Referer": "http://finance.sina.com.cn"}, timeout=3)
                    if r.status_code == 200:
                        parts = r.text.split('"')[1].split(",")
                        if len(parts) > 3 and parts[3]:
                            price = float(parts[3])
                            self._price_cache[cache_key] = price
                            return price
                return None
            except Exception:
                if attempt < max_retries:
                    time.sleep(random.uniform(0.5, 1.5))
                else:
                    return None
        return None

    def load_positions(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(POSITION_FILE, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(POSITION_FILE, encoding="gbk")
        if "成本" not in df.columns:
            df["成本"] = None
        if "份额" not in df.columns:
            df["份额"] = 0
        df["成本"] = pd.to_numeric(df["成本"], errors="coerce")
        df.loc[df["成本"] <= 0, "成本"] = None
        df["份额"] = pd.to_numeric(df["份额"], errors="coerce").fillna(0)
        return df[["代码", "名称", "成本", "份额"]]

    # ---------- 持仓历史记录 ----------
    def load_position_history(self) -> List[Dict]:
        if not os.path.exists(POSITION_HISTORY_FILE):
            return []
        try:
            with open(POSITION_HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("持仓历史文件损坏，重置")
            return []

    def save_position_history(self, history: List[Dict]):
        with open(POSITION_HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def append_daily_snapshot(self, date: str, positions_df: pd.DataFrame):
        history = self.load_position_history()
        history = [rec for rec in history if rec.get("date") != date]
        snapshot = {
            "date": date,
            "positions": []
        }
        for _, row in positions_df.iterrows():
            if row["份额"] > 0:
                snapshot["positions"].append({
                    "code": row["代码"],
                    "name": row["名称"],
                    "shares": int(row["份额"]),
                    "cost": row["成本"] if pd.notna(row["成本"]) else None
                })
        history.append(snapshot)
        history.sort(key=lambda x: x["date"])
        self.save_position_history(history)

    def get_position_change(self, code: str, current_shares: int) -> Tuple[int, float]:
        history = self.load_position_history()
        if len(history) < 2:
            return (0, 0.0)
        latest = history[-1]
        prev = history[-2]
        prev_shares = 0
        for p in prev.get("positions", []):
            if p["code"] == code:
                prev_shares = p["shares"]
                break
        delta = current_shares - prev_shares
        pct_change = (delta / prev_shares * 100) if prev_shares > 0 else 100.0 if delta > 0 else 0.0
        return (delta, pct_change)

    # ---------- 状态管理 (原 state_manager 功能) ----------
    def load_state(self) -> Dict:
        if not os.path.exists(STATE_FILE):
            return {}
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("状态文件损坏，已重置")
            return {}

    def save_state(self, state: Dict):
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def sync_user_operations(self, state: Dict, current_etf_data: Dict[str, Dict]) -> Dict:
        """
        同步用户操作：比较 state 中记录的 last_known_shares 与当前实际份额，
        更新内部计数器（trend_add_count, dip_add_count, overheat 等）
        返回更新后的 state。
        """
        for code, data in current_etf_data.items():
            current_shares = data.get("shares", 0)
            last_known = state.get(code, {}).get("last_known_shares", 0)
            if current_shares == last_known:
                continue
            if code not in state:
                state[code] = {}
            # 更新份额和成本
            state[code]["last_known_shares"] = current_shares
            if data.get("cost_price") is not None:
                state[code]["last_known_cost"] = data["cost_price"]
            # 处理份额归零
            if current_shares == 0:
                state[code]["trend_add_count"] = 0
                state[code]["dip_add_count"] = 0
                state[code]["overheat_triggered"] = False
                state[code]["overheat_count"] = 0
                state[code]["position_state"] = "CLEARED"
                continue
            # 新买入（从0到>0）
            if last_known == 0 and current_shares > 0:
                state[code]["trend_add_count"] = 0
                state[code]["dip_add_count"] = 0
                state[code]["overheat_triggered"] = False
                state[code]["overheat_count"] = 0
                state[code]["position_state"] = "BASE_HOLD"
                continue
            # 份额增加（用户手动加仓）
            if current_shares > last_known:
                added_shares = current_shares - last_known
                # 估算加仓比例（相对于当前持仓），粗略认为一次加仓操作对应 trend_add_count +1
                # 为避免过度增加，限制最多2次
                new_count = state[code].get("trend_add_count", 0) + 1
                state[code]["trend_add_count"] = min(new_count, 2)
            # 份额减少（用户手动减仓）
            elif current_shares < last_known:
                # 减仓后重置加仓计数，允许重新加仓
                state[code]["trend_add_count"] = 0
                state[code]["dip_add_count"] = 0
                # 如果减仓幅度很大，可重置 overheat
                if current_shares < last_known / 2:
                    state[code]["overheat_triggered"] = False
                    state[code]["overheat_count"] = 0
        return state

    def update_state_counters(self, state: Dict, code: str, event: str):
        """根据触发的事件更新内部计数器"""
        if code not in state:
            state[code] = {}
        if event == "trend_reversal":
            state[code]["trend_add_count"] = state[code].get("trend_add_count", 0) + 1
        elif event == "trend_confirm":
            state[code]["trend_add_count"] = state[code].get("trend_add_count", 0) + 1
        elif event == "dip":
            state[code]["dip_add_count"] = state[code].get("dip_add_count", 0) + 1
        elif event == "overheat":
            state[code]["overheat_triggered"] = True
            state[code]["overheat_count"] = state[code].get("overheat_count", 0) + 1
        elif event in ("sell_prelim", "sell_confirm"):
            # 减仓后重置加仓计数
            state[code]["trend_add_count"] = 0
            state[code]["dip_add_count"] = 0
        elif event == "clear":
            state[code]["trend_add_count"] = 0
            state[code]["dip_add_count"] = 0
            state[code]["overheat_triggered"] = False
            state[code]["overheat_count"] = 0
            state[code]["position_state"] = "CLEARED"
        elif event == "buy":
            state[code]["position_state"] = "BASE_HOLD"
            state[code]["trend_add_count"] = 0
            state[code]["dip_add_count"] = 0

    def calculate_indicators(self, df: pd.DataFrame,
                             need_amount_ma=False,
                             recent_high_window=10,
                             recent_low_window=20) -> pd.DataFrame:
        df = df.copy()
        df["ma_short"] = df["close"].rolling(ETF_MA).mean()
        df["ma5"] = df["close"].rolling(5).mean()
        df["ma10"] = df["close"].rolling(10).mean()
        df["vol_ma"] = df["volume"].rolling(ETF_VOL_MA).mean()
        if need_amount_ma:
            df["amount_ma"] = df["amount"].rolling(ETF_VOL_MA).mean()
        df["ma30"] = df["close"].rolling(MA30_WINDOW).mean()

        df["macd_dif"], df["macd_dea"], df["macd_hist"] = calc_macd(df["close"])

        low_n = df["low"].rolling(KDJ_N).min()
        high_n = df["high"].rolling(KDJ_N).max()
        rsv = (df["close"] - low_n) / (high_n - low_n) * 100
        df["kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(alpha=1/3, adjust=False).mean()

        df["boll_mid"] = df["close"].rolling(BOLL_WINDOW).mean()
        df["boll_std"] = df["close"].rolling(BOLL_WINDOW).std()
        df["boll_up"] = df["boll_mid"] + BOLL_STD_MULT * df["boll_std"]
        df["boll_low"] = df["boll_mid"] - BOLL_STD_MULT * df["boll_std"]

        high_14 = df["high"].rolling(WILLIAMS_WINDOW).max()
        low_14 = df["low"].rolling(WILLIAMS_WINDOW).min()
        df["williams_r"] = (high_14 - df["close"]) / (high_14 - low_14) * -100

        df["rsi"] = calc_rsi(df["close"])
        df["atr"] = calculate_atr(df, ATR_PERIOD)
        adx_df = calculate_adx(df)
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]
        df["adx"] = adx_df["adx"]

        df["downside_momentum_raw"] = np.where(
            (df["close"] < df["ma_short"]) & (df["minus_di"] > df["plus_di"]),
            (df["ma_short"] - df["close"]) / df["ma_short"] * (df["volume"] / df["vol_ma"]).clip(0, 3),
            0
        )
        df[f"recent_high_{recent_high_window}"] = df["high"].rolling(recent_high_window).max()
        df[f"recent_low_{recent_low_window}"] = df["low"].rolling(recent_low_window).min()
        df["boll_width"] = df["boll_std"] / df["boll_mid"]
        df["boll_width_ma20"] = df["boll_width"].rolling(20).mean()
        df["low_close_20"] = df["close"].rolling(20).min()
        df["rsi_prev"] = df["rsi"].shift(1)
        df["close_open_ratio"] = df["close"] / df["open"]
        return df

    def get_market_environment(self, market_df: pd.DataFrame) -> Dict:
        d = market_df.iloc[-1]
        if "ma5" not in market_df.columns:
            market_df["ma5"] = market_df["close"].rolling(5).mean()
            d = market_df.iloc[-1]
        if "ma10" not in market_df.columns:
            market_df["ma10"] = market_df["close"].rolling(10).mean()
            d = market_df.iloc[-1]
        ma5 = d["ma5"]
        ma20 = d["ma_short"]
        ma60 = d.get("ma30", ma20)
        above_5 = d["close"] > ma5
        above_20 = d["close"] > ma20
        above_60 = d["close"] > ma60
        atr_pct = (d["atr"] / d["close"]) if d["close"] > 0 else 0
        vol_ratio = d["volume"] / d["vol_ma"] if d["vol_ma"] > 0 else 1.0

        if above_60 and above_20:
            state = "强牛" if atr_pct < 0.02 else "弱牛"
        elif not above_60 and above_20:
            state = "弱牛"
        elif above_60 and not above_20:
            state = "震荡"
        elif not above_60 and not above_20:
            state = "强熊" if atr_pct > 0.03 else "弱熊"
        else:
            state = "震荡"

        # 使用 MA5 降级
        if len(market_df) >= 2:
            prev_ma5 = market_df["ma5"].iloc[-2]
            ma5_slope = (ma5 - prev_ma5) / prev_ma5 if prev_ma5 != 0 else 0
        else:
            ma5_slope = 0
        if not above_5 and ma5_slope < 0:
            downgrade_map = {
                "强牛": "弱牛",
                "弱牛": "震荡",
                "震荡": "弱熊",
                "弱熊": "强熊",
                "强熊": "强熊"
            }
            state = downgrade_map.get(state, state)

        # 环境因子计算
        pos_score_ma20 = ((d["close"] / ma20 - 1) * 5).clip(-1, 1) if ma20 > 0 else 0
        pos_score_ma5 = ((d["close"] / ma5 - 1) * 5).clip(-1, 1) if ma5 > 0 else 0
        pos_score = (pos_score_ma20 + pos_score_ma5) / 2
        rsi_norm = (d["rsi"] - 50) / 50
        vol_norm = (atr_pct - 0.02) / 0.02
        fund_norm = np.tanh((vol_ratio - 1) * 2)
        env_factor = 1.0 + 0.1 * (pos_score + rsi_norm + vol_norm + fund_norm)
        env_factor = max(0.8, min(1.2, env_factor))

        risk_tip = ""
        if state in ("强熊", "弱熊"):
            risk_tip = "市场偏弱，注意风险控制"
        elif env_factor > 1.1:
            risk_tip = "环境偏暖，但警惕过热"
        elif not above_5 and ma5_slope < 0:
            risk_tip = "短期转弱，注意回调"

        return {
            "state": state,
            "factor": env_factor,
            "volatility": atr_pct,
            "above_ma20": above_20,
            "above_ma60": above_60,
            "above_ma5": above_5,
            "ma5_slope": ma5_slope,
            "amount_above_ma20": d.get("amount_ma", 0) > 0 and d["amount"] > d["amount_ma"],
            "ret_market_5d": (d["close"] / market_df.iloc[-5]["close"] - 1)
                             if len(market_df) >= 5 else 0,
            "risk_tip": risk_tip
        }

    def select_weights(self, env_state: str) -> Tuple[Dict, Dict]:
        if "牛" in env_state:
            return BUY_WEIGHTS_BULL.copy(), SELL_WEIGHTS_BULL.copy()
        elif "熊" in env_state:
            return BUY_WEIGHTS_BEAR.copy(), SELL_WEIGHTS_BEAR.copy()
        else:
            return BUY_WEIGHTS_RANGE.copy(), SELL_WEIGHTS_RANGE.copy()