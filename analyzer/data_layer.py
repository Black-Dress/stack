#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一数据与环境层：获取行情、计算指标、评估市场状态（无外部情绪爬虫）"""
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
from typing import Dict, Tuple, Optional

from .config import *
from .utils import calc_rsi, calc_macd, calculate_atr, calculate_adx

logger = logging.getLogger(__name__)


class DataLayer:
    def __init__(self):
        self._logged_in = False
        self._state_file = STATE_FILE

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
        weekly = df.resample("W-FRI").last()
        today_ts = pd.Timestamp(datetime.date.today())
        weekly = weekly[weekly.index < today_ts]
        if weekly.empty:
            return None
        weekly["ma_short"] = weekly["close"].rolling(ETF_MA).mean()
        return weekly

    def get_realtime_price(self, code: str) -> Optional[float]:
        """增加简单重试与延迟，避免限流"""
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                for domain in ["hq.sinajs.cn", "hq2.sinajs.cn", "hq3.sinajs.cn"]:
                    url = f"http://{domain}/list={code.replace('.', '')}"
                    r = requests.get(url, headers={"Referer": "http://finance.sina.com.cn"}, timeout=3)
                    if r.status_code == 200:
                        parts = r.text.split('"')[1].split(",")
                        if len(parts) > 3 and parts[3]:
                            return float(parts[3])
                return None
            except Exception:
                if attempt < max_retries:
                    time.sleep(random.uniform(0.5, 1.5))
                else:
                    return None
        return None

    def load_positions(self) -> pd.DataFrame:
        """读取持仓文件，返回包含 [代码, 名称, 成本] 的DataFrame。
           成本价为 None 表示未持有。
        """
        try:
            df = pd.read_csv(POSITION_FILE, encoding="utf-8-sig")
        except UnicodeDecodeError:
            df = pd.read_csv(POSITION_FILE, encoding="gbk")
        if "成本" not in df.columns:
            df["成本"] = None
        df["成本"] = pd.to_numeric(df["成本"], errors="coerce")
        df.loc[df["成本"] <= 0, "成本"] = None
        return df[["代码", "名称", "成本"]]

    def load_state(self) -> Dict:
        if not os.path.exists(self._state_file):
            return {}
        try:
            with open(self._state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("状态文件损坏，已重置")
            return {}

    def save_state(self, state: Dict):
        with open(self._state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def calculate_indicators(self, df: pd.DataFrame,
                             need_amount_ma=False,
                             recent_high_window=10,
                             recent_low_window=20) -> pd.DataFrame:
        df = df.copy()
        df["ma_short"] = df["close"].rolling(ETF_MA).mean()
        df["vol_ma"] = df["volume"].rolling(ETF_VOL_MA).mean()
        if need_amount_ma:
            df["amount_ma"] = df["amount"].rolling(ETF_VOL_MA).mean()
        df["ma30"] = df["close"].rolling(MA30_WINDOW).mean()

        df["macd_dif"], df["macd_dea"], _ = calc_macd(df["close"])

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
        ma20 = d["ma_short"]
        ma60 = d.get("ma30", ma20)
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

        pos_score = ((d["close"] / ma20 - 1) * 5).clip(-1, 1)
        rsi_norm = (d["rsi"] - 50) / 50
        vol_norm = (atr_pct - 0.02) / 0.02
        fund_norm = np.tanh((vol_ratio - 1) * 2)
        env_factor = 1.0 + 0.1 * pos_score + 0.1 * rsi_norm + 0.1 * vol_norm + 0.1 * fund_norm
        env_factor = max(0.8, min(1.2, env_factor))

        risk_tip = ""
        if state in ("强熊", "弱熊"):
            risk_tip = "市场偏弱，注意风险控制"
        elif env_factor > 1.1:
            risk_tip = "环境偏暖，但警惕过热"

        return {
            "state": state,
            "factor": env_factor,
            "volatility": atr_pct,
            "above_ma20": above_20,
            "above_ma60": above_60,
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