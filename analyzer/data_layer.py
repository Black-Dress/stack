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
from typing import Dict, Tuple, Optional, List

from .config import *
from .utils import calc_rsi, calc_macd, calculate_atr, calculate_adx

logger = logging.getLogger(__name__)


class PositionManager:
    def __init__(self, ai_client=None):
        self.ai_client = ai_client

    def get_unified_advice(self, ctx, final_score: float, risk_str: str) -> str:
        """
        根据是否有持仓返回唯一建议：
        - 有持仓：调用 AI 仓位建议（后备硬编码）
        - 无持仓：调用 AI 买入力度建议（后备 evaluate_buy_level）
        """
        if ctx.shares > 0 and ctx.cost_price is not None and ctx.real_price is not None:
            # 有持仓 -> 仓位管理建议
            advice = ""
            if self.ai_client and AI_ENABLE_POSITION_ADVICE:
                advice = self.ai_client.get_position_advice(ctx, final_score, risk_str)
            if not advice:
                advice = self._fallback_position_advice(ctx, final_score, risk_str)
            return advice
        else:
            # 无持仓 -> 买入力度建议
            if self.ai_client and AI_ENABLE_BUY_LEVEL:
                # 构造 scan_info 字典
                scan_info = {
                    "final_score": final_score,
                    "rsi": ctx.rsi,
                    "vol_ratio": getattr(ctx, 'vol_ratio', 1.0),
                    "has_weak_ma_text": ctx.is_weak_ma,
                    "has_clear_stop_text": "清仓止盈" in risk_str or "止损卖出" in risk_str,
                    "has_strong_sell_text": "连续低分" in risk_str,
                    "change_pct": ctx.change_pct / 100.0,
                }
                advice = self.ai_client.get_buy_level(scan_info)
            if not advice:
                # 后备：使用原来的 evaluate_buy_level
                scan_info = {
                    "final_score": final_score,
                    "rsi": ctx.rsi,
                    "change_pct": ctx.change_pct / 100.0,
                    "has_weak_ma_text": ctx.is_weak_ma,
                    "has_clear_stop_text": False,
                    "has_strong_sell_text": False,
                    "cost_profit_pct": None,
                }
                advice = evaluate_buy_level(scan_info)
            return advice

    def _fallback_position_advice(self, ctx, final_score: float, risk_str: str) -> str:
        """硬编码仓位建议（后备）"""
        profit_pct = (ctx.real_price - ctx.cost_price) / ctx.cost_price
        if final_score >= POSITION_ADD_THRESHOLD:
            if profit_pct < 0.05:
                return f"🔼 加仓{int(POSITION_ADD_RATIO*100)}%"
            elif profit_pct < 0.15:
                return f"📈 小幅加仓{int(POSITION_ADD_RATIO*100)}%"
            else:
                return f"💰 浮盈{profit_pct:.1%}，加仓{int(POSITION_ADD_RATIO*50)}%"
        elif final_score <= POSITION_CLEAR_THRESHOLD:
            return "⛔ 清仓离场"
        elif final_score <= POSITION_REDUCE_THRESHOLD:
            return f"🔽 减仓{int(POSITION_REDUCE_RATIO*100)}%"
        else:
            base = "⚪ 持有"
            if "清仓止盈" in risk_str or "移动止盈" in risk_str:
                return "⚠️ " + base + "，注意止盈"
            elif "止损卖出" in risk_str:
                return "⛔ 强制止损"
            elif "连续低分" in risk_str:
                return "🔽 强烈减仓"
            return base

class DataLayer:
    def __init__(self):
        self._logged_in = False
        self._state_file = STATE_FILE
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
        """将当日持仓快照写入历史（若当天已有则覆盖）"""
        history = self.load_position_history()
        # 移除同一天的旧记录
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
        # 按日期排序
        history.sort(key=lambda x: x["date"])
        self.save_position_history(history)

    def get_position_change(self, code: str, current_shares: int) -> Tuple[int, float]:
        """返回 (变化份额, 变化百分比) 与前一日快照对比"""
        history = self.load_position_history()
        if len(history) < 2:
            return (0, 0.0)
        latest = history[-1]  # 今日已保存的快照
        prev = history[-2]    # 昨日快照
        prev_shares = 0
        for p in prev.get("positions", []):
            if p["code"] == code:
                prev_shares = p["shares"]
                break
        delta = current_shares - prev_shares
        pct_change = (delta / prev_shares * 100) if prev_shares > 0 else 100.0 if delta > 0 else 0.0
        return (delta, pct_change)

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