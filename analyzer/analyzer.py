#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心分析模块：负责技术指标计算、评分、信号确认及批量分析入口。
依赖 fetcher.py 中的数据获取功能。
"""
import datetime
import logging
import re
import numpy as np
import pandas as pd
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

from .config import (
    ETF_MA,
    ETF_VOL_MA,
    ATR_PERIOD,
    ATR_STOP_MULT,
    ATR_TRAILING_MULT,
    PROFIT_TARGET,
    RISK_WARNING_DAYS,
    RISK_WARNING_THRESHOLD,
    DEFAULT_BUY_WEIGHTS,
    DEFAULT_SELL_WEIGHTS,
    DEFAULT_PARAMS,
    MACRO_INDEX,
    MARKET_INDEX,
    MACRO_MA_LONG,
    get_email_config,
)
from .utils import pad_display, validate_and_filter_weights, send_email
from .ai import AIClient
from .fetcher import DataFetcher, AKSHARE_AVAILABLE

logger = logging.getLogger(__name__)


class DataAnalyzer:
    """负责所有技术指标计算、评分、信号确认、状态管理、缓存等"""

    # ---------- 涨跌幅计算 ----------
    def calc_change_pct(
        self, real_price: float, hist_df: pd.DataFrame, today: datetime.date
    ) -> float:
        """计算实时价格相对于上一个交易日收盘价的涨跌幅"""
        if real_price is None or hist_df is None or len(hist_df) < 1:
            return 0.0
        last_date = hist_df.index[-1].date()
        if last_date == today and len(hist_df) >= 2:
            base_close = hist_df.iloc[-2]["close"]
        else:
            base_close = hist_df.iloc[-1]["close"]
        if base_close > 0:
            return (real_price - base_close) / base_close * 100
        return 0.0

    # ---------- 技术指标计算 ----------
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        tr = pd.concat(
            [
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift()),
                abs(df["low"] - df["close"].shift()),
            ],
            axis=1,
        ).max(1)
        return tr.rolling(period).mean()

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = self.calculate_atr(df, 1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return pd.DataFrame(
            {"plus_di": plus_di, "minus_di": minus_di, "adx": adx}, index=df.index
        )

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        need_amount_ma: bool = True,
        recent_high_window: int = 10,
        recent_low_window: int = 20,
    ) -> pd.DataFrame:
        """计算全部技术指标"""
        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=ETF_MA).mean()
        df["vol_ma"] = df["volume"].rolling(window=ETF_VOL_MA).mean()
        if need_amount_ma:
            df["amount_ma"] = df["amount"].rolling(window=ETF_VOL_MA).mean()
        # MACD
        exp1 = df["close"].ewm(span=12, adjust=False).mean()
        exp2 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd_dif"] = exp1 - exp2
        df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
        # KDJ
        low_n = df["low"].rolling(9).min()
        high_n = df["high"].rolling(9).max()
        rsv = (df["close"] - low_n) / (high_n - low_n) * 100
        df["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
        # 布林
        df["boll_mid"] = df["close"].rolling(20).mean()
        df["boll_std"] = df["close"].rolling(20).std()
        df["boll_up"] = df["boll_mid"] + 2 * df["boll_std"]
        df["boll_low"] = df["boll_mid"] - 2 * df["boll_std"]
        # 威廉
        high_14 = df["high"].rolling(14).max()
        low_14 = df["low"].rolling(14).min()
        df["williams_r"] = (high_14 - df["close"]) / (high_14 - low_14) * -100
        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df["rsi"] = 100 - 100 / (1 + gain / loss)
        # ATR
        df["atr"] = self.calculate_atr(df, ATR_PERIOD)
        # ADX
        adx_df = self.calculate_adx(df)
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]
        df["adx"] = adx_df["adx"]
        # 下跌动量
        df["downside_momentum_raw"] = np.where(
            (df["close"] < df["ma_short"]) & (df["minus_di"] > df["plus_di"]),
            (df["ma_short"] - df["close"])
            / df["ma_short"]
            * (df["volume"] / df["vol_ma"]).clip(0, 3),
            0,
        )
        # 近期高低点
        df[f"recent_high_{recent_high_window}"] = (
            df["high"].rolling(recent_high_window).max()
        )
        df[f"recent_low_{recent_low_window}"] = (
            df["low"].rolling(recent_low_window).min()
        )
        return df

    def compute_tmsv(self, df: pd.DataFrame) -> pd.Series:
        """计算 TMSV 复合指标"""
        if df is None or len(df) < 20:
            return (
                pd.Series([50.0] * max(1, len(df)))
                if len(df) > 0
                else pd.Series([50.0])
            )
        df = df.copy()
        # 确保所需列存在
        if "ma20" not in df.columns:
            df["ma20"] = df["close"].rolling(20).mean()
        if "ma60" not in df.columns:
            df["ma60"] = df["close"].rolling(60).mean()
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df["rsi"] = 100 - 100 / (1 + gain / loss)
        if "macd_hist" not in df.columns:
            exp12 = df["close"].ewm(span=12, adjust=False).mean()
            exp26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp12 - exp26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
        if "atr" not in df.columns:
            df["atr"] = self.calculate_atr(df, 14)
        if "vol_ma" not in df.columns:
            df["vol_ma"] = df["volume"].rolling(20).mean()

        price_above_ma20 = (
            ((df["close"] - df["ma20"]) / (df["ma20"].replace(0, np.nan) * 0.1))
            .clip(0, 1)
            .fillna(0)
        )
        price_above_ma60 = (
            ((df["close"] - df["ma60"]) / (df["ma60"].replace(0, np.nan) * 0.1))
            .clip(0, 1)
            .fillna(0)
        )
        ma20_slope = df["ma20"].diff(5) / df["ma20"].shift(5).replace(0, np.nan)
        slope_score = (ma20_slope * 10).clip(0, 1).fillna(0)
        trend_score = (
            price_above_ma20 * 0.5 + price_above_ma60 * 0.3 + slope_score * 0.2
        ) * 100

        rsi_score = ((df["rsi"] - 50) * 3.33).clip(0, 100).fillna(50)
        macd_change = df["macd_hist"].diff() / (df["macd_hist"].shift(1).abs() + 0.001)
        macd_score = (macd_change * 100).clip(0, 100).fillna(50)
        mom_score = rsi_score * 0.6 + macd_score * 0.4

        vol_ratio = df["volume"] / df["vol_ma"].replace(0, np.nan)
        vol_ratio_score = ((vol_ratio - 0.8) / 1.2 * 100).clip(0, 100).fillna(50)
        price_up = df["close"] > df["close"].shift(1)
        vol_up = df["volume"] > df["vol_ma"]
        consistency = np.where(price_up == vol_up, 100, 0)
        vol_score = vol_ratio_score * 0.7 + consistency * 0.3

        atr_pct = df["atr"] / df["close"].replace(0, np.nan)
        vol_factor = np.select(
            [atr_pct < 0.01, atr_pct > 0.03],
            [1.5, 0.6],
            default=1.2 - (atr_pct - 0.01) / 0.02 * 0.6,
        )
        vol_factor = np.nan_to_num(vol_factor, nan=1.0)

        tmsv = (trend_score * 0.3 + mom_score * 0.3 + vol_score * 0.2) * vol_factor
        return tmsv.clip(0, 100).fillna(50)

    # ---------- 权重处理 ----------
    def compute_dynamic_trust(self, ai_weights: Dict, default_weights: Dict) -> float:
        base_trust = 0.75
        zero_count = sum(1 for v in ai_weights.values() if v < 0.01)
        if zero_count > len(ai_weights) * 0.3:
            base_trust = min(base_trust, 0.5)
        max_w = max(ai_weights.values())
        if max_w > 0.45:
            base_trust = min(base_trust, 0.6)
        dot = sum(
            ai_weights.get(k, 0) * default_weights.get(k, 0) for k in default_weights
        )
        norm_ai = math.sqrt(sum(v**2 for v in ai_weights.values()))
        norm_def = math.sqrt(sum(v**2 for v in default_weights.values()))
        if norm_ai > 0 and norm_def > 0:
            similarity = dot / (norm_ai * norm_def)
            if similarity < 0.6:
                base_trust = min(base_trust, 0.55)
        return base_trust

    def blend_weights(self, ai_w: Dict, def_w: Dict, trust: float) -> Dict:
        blended = {k: ai_w.get(k, 0) * trust + def_w[k] * (1 - trust) for k in def_w}
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    def apply_correlation_penalty(
        self,
        weights: Dict,
        factor_names: List[str],
        corr_matrix: pd.DataFrame,
        penalty_threshold: float = 0.7,
    ) -> Dict:
        w = weights.copy()
        high_corr_pairs = []
        for i, f1 in enumerate(factor_names):
            for j, f2 in enumerate(factor_names):
                if i >= j:
                    continue
                if corr_matrix.loc[f1, f2] > penalty_threshold:
                    if w.get(f1, 0) > 0.12 and w.get(f2, 0) > 0.12:
                        high_corr_pairs.append((f1, f2))
        if not high_corr_pairs:
            return w
        for f1, f2 in high_corr_pairs:
            penalty = 0.2
            w[f1] *= 1 - penalty
            w[f2] *= 1 - penalty
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        return w

    def compute_factor_correlation(
        self, df: pd.DataFrame, factor_names: List[str]
    ) -> pd.DataFrame:
        """简化版相关性矩阵"""
        return pd.DataFrame(
            np.eye(len(factor_names)), index=factor_names, columns=factor_names
        )

    # ---------- 因子计算（公共方法）----------
    def _compute_factors(
        self,
        price: float,
        ma20: float,
        volume: float,
        vol_ma: float,
        macd_golden: int,
        kdj_golden: int,
        rsi: float,
        boll_up: float,
        boll_low: float,
        williams_r: float,
        ret_etf_5d: float,
        ret_market_5d: float,
        weekly_above: bool,
        weekly_below: bool,
        recent_high: float,
        recent_low: float,
        atr_pct: float,
        market_above_ma20: bool,
        market_above_ma60: bool,
        market_amount_above_ma20: bool,
        tmsv_strength: float,
        downside_momentum: float,
        max_drawdown_pct: float,
    ) -> Tuple[Dict, Dict]:
        """返回买入因子字典和卖出因子字典"""

        def cap(x):
            return max(0.0, min(1.0, x))

        buy_factors = {
            "price_above_ma20": (
                cap((price - ma20) / (ma20 * 0.1)) if price > ma20 else 0
            ),
            "volume_above_ma5": cap(volume / vol_ma) if volume > vol_ma else 0,
            "macd_golden_cross": macd_golden,
            "kdj_golden_cross": kdj_golden,
            "bollinger_break_up": (
                cap((price - boll_up) / boll_up) if price > boll_up else 0
            ),
            "williams_oversold": (
                cap((-80 - williams_r) / 20) if williams_r < -80 else 0
            ),
            "market_above_ma20": 1 if market_above_ma20 else 0,
            "market_above_ma60": 1 if market_above_ma60 else 0,
            "market_amount_above_ma20": 1 if market_amount_above_ma20 else 0,
            "outperform_market": (
                cap((ret_etf_5d - ret_market_5d) / 0.05)
                if ret_etf_5d > ret_market_5d
                else 0
            ),
            "weekly_above_ma20": 1 if weekly_above else 0,
            "tmsv_score": tmsv_strength,
        }

        sell_factors = {
            "price_below_ma20": (
                cap((ma20 - price) / (ma20 * 0.1)) if price < ma20 else 0
            ),
            "bollinger_break_down": (
                cap((boll_low - price) / boll_low) if price < boll_low else 0
            ),
            "williams_overbought": (
                cap((20 - williams_r) / 20) if williams_r < 20 else 0
            ),
            "rsi_overbought": cap((rsi - 70) / 30) if rsi > 70 else 0,
            "underperform_market": (
                cap((ret_market_5d - ret_etf_5d) / 0.05)
                if ret_etf_5d < ret_market_5d
                else 0
            ),
            "stop_loss_ma_break": (
                cap((ma20 - price) / (ma20 * 0.05)) if price < ma20 else 0
            ),
            "trailing_stop_clear": (
                cap((recent_high - price) / recent_high / (ATR_STOP_MULT * atr_pct))
                if recent_high > 0
                and atr_pct > 0
                and (recent_high - price) / recent_high >= ATR_STOP_MULT * atr_pct
                else 0
            ),
            "trailing_stop_half": (
                cap((recent_high - price) / recent_high / (ATR_TRAILING_MULT * atr_pct))
                if recent_high > 0
                and atr_pct > 0
                and (recent_high - price) / recent_high >= ATR_TRAILING_MULT * atr_pct
                else 0
            ),
            "profit_target_hit": (
                cap((price - recent_low) / recent_low / PROFIT_TARGET)
                if recent_low > 0 and (price - recent_low) / recent_low >= PROFIT_TARGET
                else 0
            ),
            "weekly_below_ma20": 1 if weekly_below else 0,
            "downside_momentum": cap(downside_momentum),
            "max_drawdown_stop": (
                cap(max_drawdown_pct / 0.08) if max_drawdown_pct >= 0.08 else 0
            ),
        }
        return buy_factors, sell_factors

    def strength(
        self, *args, is_buy: bool, buy_weights: Dict, sell_weights: Dict, **kwargs
    ) -> float:
        """计算买入或卖出评分"""
        factors = self._compute_factors(*args, **kwargs)[0 if is_buy else 1]
        weights = buy_weights if is_buy else sell_weights
        return sum(weights.get(k, 0) * factors.get(k, 0) for k in factors)

    # ---------- 信号确认与参数调整 ----------
    def get_dynamic_history_days(self, volatility: float) -> int:
        if volatility > 0.04:
            return 5
        if volatility > 0.025:
            return 8
        if volatility > 0.015:
            return 12
        return 20

    def get_action(
        self,
        score: float,
        score_history: List[Dict],
        params: Dict,
        atr_pct: float = None,
    ) -> str:
        hist_scores = [s["score"] for s in score_history]
        if len(hist_scores) < 2:
            return (
                "BUY"
                if score > params["BUY_THRESHOLD"]
                else ("SELL" if score < params["SELL_THRESHOLD"] else "HOLD")
            )
        window = self.get_dynamic_history_days(atr_pct) if atr_pct else 12
        window = min(window, len(hist_scores))
        recent = hist_scores[-window:]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        up_days = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        down_days = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])

        if score > params["BUY_THRESHOLD"]:
            if (
                score > params["QUICK_BUY_THRESHOLD"]
                and slope > 0.05
                and avg > params["BUY_THRESHOLD"] + 0.1
            ):
                return "BUY"
            if len(hist_scores) >= params["CONFIRM_DAYS"]:
                confirm = hist_scores[-params["CONFIRM_DAYS"] :]
                if all(s > params["BUY_THRESHOLD"] for s in confirm) and slope >= 0:
                    return "BUY"
            if down_days >= 2 and slope > 0.02 and score > avg + 0.1:
                return "BUY"
        if score < params["SELL_THRESHOLD"]:
            if (
                score < params["SELL_THRESHOLD"] - 0.1
                and slope < -0.05
                and avg < params["SELL_THRESHOLD"] - 0.1
            ):
                return "SELL"
            if len(hist_scores) >= params["CONFIRM_DAYS"]:
                confirm = hist_scores[-params["CONFIRM_DAYS"] :]
                if all(s < params["SELL_THRESHOLD"] for s in confirm) and slope <= 0:
                    return "SELL"
            if up_days >= 2 and slope < -0.02 and score < avg - 0.1:
                return "SELL"
        if atr_pct and atr_pct > 0.04:
            if score > params["BUY_THRESHOLD"] + 0.15 and slope > 0.1 and up_days >= 4:
                return "BUY"
            if (
                score < params["SELL_THRESHOLD"] - 0.05
                and slope < -0.08
                and down_days >= 3
            ):
                return "SELL"
            return "HOLD"
        if atr_pct and atr_pct > 0.03:
            if score > params["BUY_THRESHOLD"] + 0.1 and slope > 0.08 and up_days >= 3:
                return "BUY"
            if (
                score < params["SELL_THRESHOLD"] - 0.1
                and slope < -0.08
                and down_days >= 3
            ):
                return "SELL"
        return "HOLD"

    def get_action_level(self, score: float) -> str:
        thresholds = [0.8, 0.7, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8]
        levels = [
            "极度看好",
            "强烈买入",
            "买入",
            "谨慎买入",
            "偏多持有",
            "中性偏多",
            "中性偏空",
            "偏空持有",
            "谨慎卖出",
            "卖出",
        ]
        for th, level in zip(thresholds, levels):
            if score >= th:
                return level
        return "强烈卖出"

    def adjust_params_based_on_history(
        self,
        params: Dict,
        score_history: List[Dict],
        volatility: float,
        market_factor: float,
    ) -> Dict:
        if len(score_history) < 10:
            return params
        window = min(self.get_dynamic_history_days(volatility), len(score_history))
        recent = [s["score"] for s in score_history[-window:]]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        short_recent = [
            s["score"] for s in score_history[-min(3, len(score_history)) :]
        ]
        short_slope = (
            np.polyfit(range(len(short_recent)), short_recent, 1)[0]
            if len(short_recent) >= 2
            else 0
        )
        adjust_mult = 1.2 / market_factor
        adjusted = params.copy()

        if slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = min(0.03, abs(avg - params["BUY_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["BUY_THRESHOLD"] = max(0.35, params["BUY_THRESHOLD"] - delta)
        elif slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = min(0.03, abs(avg - params["BUY_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["BUY_THRESHOLD"] = min(0.65, params["BUY_THRESHOLD"] + delta)
        else:
            adjusted["BUY_THRESHOLD"] = (
                params["BUY_THRESHOLD"] * 0.9 + DEFAULT_PARAMS["BUY_THRESHOLD"] * 0.1
            )

        if slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = min(0.03, abs(avg - params["SELL_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["SELL_THRESHOLD"] = max(-0.45, params["SELL_THRESHOLD"] - delta)
        elif slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = min(0.03, abs(avg - params["SELL_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["SELL_THRESHOLD"] = min(-0.15, params["SELL_THRESHOLD"] + delta)
        else:
            adjusted["SELL_THRESHOLD"] = (
                params["SELL_THRESHOLD"] * 0.9 + DEFAULT_PARAMS["SELL_THRESHOLD"] * 0.1
            )

        if volatility > 0.04:
            adjusted["CONFIRM_DAYS"] = min(5, int(round(params["CONFIRM_DAYS"] * 1.1)))
        elif volatility < 0.01:
            adjusted["CONFIRM_DAYS"] = max(2, int(round(params["CONFIRM_DAYS"] * 0.9)))
        else:
            adjusted["CONFIRM_DAYS"] = int(
                round(
                    params["CONFIRM_DAYS"] * 0.95
                    + DEFAULT_PARAMS["CONFIRM_DAYS"] * 0.05
                )
            )
        return adjusted

    # ---------- 单只 ETF 简要分析 ----------
    def analyze_single_etf(
        self,
        code: str,
        name: str,
        real_price: Optional[float],
        hist_df: Optional[pd.DataFrame],
        weekly_df: Optional[pd.DataFrame],
        market: Dict,
        today: datetime.date,
        state: Dict,
        buy_w: Dict,
        sell_w: Dict,
        params: Dict,
    ) -> Tuple[str, Optional[Dict], Dict, float]:
        change_pct = (
            self.calc_change_pct(real_price, hist_df, today)
            if real_price is not None
            else 0.0
        )

        if real_price is None:
            out = f"{pad_display(name, 16)} {pad_display(code, 12)} {'获取失败':<8} {pad_display('0.00%', 8, 'right')} {'0.00':<6}  {'价格缺失':<10}"
            return out, None, state, 0.0
        if hist_df is None or len(hist_df) < 20:
            out = f"{pad_display(name, 16)} {pad_display(code, 12)} {real_price:<8.3f} {pad_display(f'{change_pct:+.2f}%', 8, 'right')} {0.00:<6}  {'数据不足':<10}"
            return out, None, state, 0.0

        d = hist_df.iloc[-1]
        ma20, vol_ma, volume = d["ma_short"], d["vol_ma"], d["volume"]
        rsi, boll_up, boll_low, williams_r = (
            d["rsi"],
            d["boll_up"],
            d["boll_low"],
            d["williams_r"],
        )
        atr_pct = d["atr"] / real_price if real_price > 0 else 0
        recent_high = d.get(
            f"recent_high_{params['RECENT_HIGH_WINDOW']}",
            hist_df["high"].rolling(params["RECENT_HIGH_WINDOW"]).max().iloc[-1],
        )
        recent_low = d.get(
            f"recent_low_{params['RECENT_LOW_WINDOW']}",
            hist_df["low"].rolling(params["RECENT_LOW_WINDOW"]).min().iloc[-1],
        )

        if len(hist_df) >= 2:
            prev = hist_df.iloc[-2]
            macd_golden = (
                1
                if (
                    d["macd_dif"] > d["macd_dea"]
                    and prev["macd_dif"] <= prev["macd_dea"]
                )
                else 0
            )
            kdj_golden = (
                1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0
            )
        else:
            macd_golden = kdj_golden = 0

        ret_etf_5d = (
            (real_price / hist_df.iloc[-5]["close"] - 1) if len(hist_df) >= 5 else 0
        )
        weekly_above = weekly_below = False
        if weekly_df is not None and not weekly_df.empty:
            w = weekly_df.iloc[-1]
            if "ma_short" in w.index and not pd.isna(w["ma_short"]):
                weekly_above = w["close"] > w["ma_short"]
                weekly_below = w["close"] < w["ma_short"]

        try:
            tmsv_series = self.compute_tmsv(hist_df)
            tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
            tmsv = 50.0 if np.isnan(tmsv) else tmsv
        except Exception:
            tmsv = 50.0
        tmsv_strength = tmsv / 100.0
        downside_momentum = d.get("downside_momentum_raw", 0.0)
        max_drawdown_pct = (
            (recent_high - real_price) / recent_high if recent_high > 0 else 0.0
        )

        buy_factors, sell_factors = self._compute_factors(
            real_price,
            ma20,
            volume,
            vol_ma,
            macd_golden,
            kdj_golden,
            rsi,
            boll_up,
            boll_low,
            williams_r,
            ret_etf_5d,
            market["ret_market_5d"],
            weekly_above,
            weekly_below,
            recent_high,
            recent_low,
            atr_pct,
            market["market_above_ma20"],
            market["market_above_ma60"],
            market["market_amount_above_ma20"],
            tmsv_strength,
            downside_momentum,
            max_drawdown_pct,
        )

        buy_score = sum(buy_w.get(k, 0) * buy_factors[k] for k in buy_factors)
        sell_score = sum(sell_w.get(k, 0) * sell_factors[k] for k in sell_factors)
        raw = buy_score - sell_score
        env_factor = market["market_factor"] * market["sentiment_factor"]
        env_factor = max(0.60, min(1.30, env_factor))
        final = max(-1.0, min(1.0, raw * env_factor))

        today_str = today.strftime("%Y-%m-%d")
        if "score_history" not in state:
            state["score_history"] = []
        found = False
        for item in state["score_history"]:
            if item["date"] == today_str:
                item["score"] = final
                found = True
                break
        if not found:
            state["score_history"].append({"date": today_str, "score": final})
        state["score_history"].sort(key=lambda x: x["date"])

        if len(state["score_history"]) >= 7:
            params = self.adjust_params_based_on_history(
                params, state["score_history"], atr_pct, market["market_factor"]
            )

        action = self.get_action(final, state["score_history"], params, atr_pct)
        action_level = self.get_action_level(final)

        risk_warning = ""
        if len(state["score_history"]) >= RISK_WARNING_DAYS:
            recent_scores = [
                s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]
            ]
            if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
                risk_warning = f"风险提示:连续{RISK_WARNING_DAYS}天评分低于{RISK_WARNING_THRESHOLD}"
            elif final < -0.5 or final > 0.8:
                risk_warning = f"风险提示:极端评分{final:.2f}"
            elif atr_pct > 0.03:
                risk_warning = f"风险提示:高波动{atr_pct:.3f}"

        output = (
            f"{pad_display(name, 16)} {pad_display(code, 12)} "
            f"{pad_display(f'{real_price:.3f}', 8, 'right')} "
            f"{pad_display(f'{change_pct:+.2f}%', 8, 'right')} "
            f"{pad_display(f'{final:.2f}', 6, 'right')}  "
            f"{pad_display(action_level, 10)}"
        )
        if risk_warning:
            output += f"  {risk_warning}"
        signal = (
            {"action": action, "name": name, "code": code, "score": final}
            if action in ("BUY", "SELL")
            else None
        )
        return output, signal, state, final

    # ---------- 详细分析报告 ----------
    def detailed_analysis(
        self,
        code: str,
        name: str,
        real_price: Optional[float],
        hist_df: Optional[pd.DataFrame],
        weekly_df: Optional[pd.DataFrame],
        market: Dict,
        today: datetime.date,
        state: Dict,
        buy_w: Dict,
        sell_w: Dict,
        params: Dict,
        ai_client: Optional[AIClient] = None,
    ) -> str:
        if real_price is None:
            return f"【{name} ({code})】实时价格获取失败，无法分析。"
        if hist_df is None or len(hist_df) < 20:
            return f"【{name} ({code})】历史数据不足，无法分析。"

        change_pct = self.calc_change_pct(real_price, hist_df, today)

        d = hist_df.iloc[-1]
        ma20, vol_ma, volume = d["ma_short"], d["vol_ma"], d["volume"]
        rsi, boll_up, boll_low, williams_r = (
            d["rsi"],
            d["boll_up"],
            d["boll_low"],
            d["williams_r"],
        )
        atr_pct = d["atr"] / real_price if real_price > 0 else 0
        recent_high = d.get(
            f"recent_high_{params['RECENT_HIGH_WINDOW']}",
            hist_df["high"].rolling(params["RECENT_HIGH_WINDOW"]).max().iloc[-1],
        )
        recent_low = d.get(
            f"recent_low_{params['RECENT_LOW_WINDOW']}",
            hist_df["low"].rolling(params["RECENT_LOW_WINDOW"]).min().iloc[-1],
        )

        if len(hist_df) >= 2:
            prev = hist_df.iloc[-2]
            macd_golden = (
                1
                if (
                    d["macd_dif"] > d["macd_dea"]
                    and prev["macd_dif"] <= prev["macd_dea"]
                )
                else 0
            )
            kdj_golden = (
                1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0
            )
        else:
            macd_golden = kdj_golden = 0

        ret_etf_5d = (
            (real_price / hist_df.iloc[-5]["close"] - 1) if len(hist_df) >= 5 else 0
        )
        weekly_above = weekly_below = False
        if weekly_df is not None and not weekly_df.empty:
            w = weekly_df.iloc[-1]
            if "ma_short" in w.index and not pd.isna(w["ma_short"]):
                weekly_above = w["close"] > w["ma_short"]
                weekly_below = w["close"] < w["ma_short"]

        try:
            tmsv_series = self.compute_tmsv(hist_df)
            tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
            tmsv = 50.0 if np.isnan(tmsv) else tmsv
        except Exception:
            tmsv = 50.0
        tmsv_strength = tmsv / 100.0
        downside_momentum = d.get("downside_momentum_raw", 0.0)
        max_drawdown_pct = (
            (recent_high - real_price) / recent_high if recent_high > 0 else 0.0
        )

        buy_factors, sell_factors = self._compute_factors(
            real_price,
            ma20,
            volume,
            vol_ma,
            macd_golden,
            kdj_golden,
            rsi,
            boll_up,
            boll_low,
            williams_r,
            ret_etf_5d,
            market["ret_market_5d"],
            weekly_above,
            weekly_below,
            recent_high,
            recent_low,
            atr_pct,
            market["market_above_ma20"],
            market["market_above_ma60"],
            market["market_amount_above_ma20"],
            tmsv_strength,
            downside_momentum,
            max_drawdown_pct,
        )

        buy_score = sum(buy_w.get(k, 0) * buy_factors[k] for k in buy_factors)
        sell_score = sum(sell_w.get(k, 0) * sell_factors[k] for k in sell_factors)
        raw = buy_score - sell_score
        final = max(
            -1.0, min(1.0, raw * market["market_factor"] * market["sentiment_factor"])
        )
        action_level = self.get_action_level(final)

        lines = []
        lines.append("=" * 70)
        lines.append(f"ETF详细分析报告 - {name} ({code})")
        lines.append(
            f"分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        lines.append("=" * 70)
        lines.append(f"实时价格：{real_price:.3f}")
        lines.append(f"涨跌幅：{change_pct:+.2f}%")
        lines.append(
            f"市场状态：{market['macro_status']}，市场因子：{market['market_factor']:.2f}，情绪因子：{market['sentiment_factor']:.2f}"
        )
        if market.get("sentiment_risk_tip"):
            lines.append(f"情绪风险提示：{market['sentiment_risk_tip']}")
        lines.append(f"波动率(ATR%)：{atr_pct*100:.2f}%")
        lines.append(f"TMSV复合强度：{tmsv:.1f} (强度系数 {tmsv_strength:.3f})")
        lines.append(f"最大回撤：{max_drawdown_pct*100:.2f}%")
        lines.append("")

        col_name, col_strength, col_weight, col_contrib = 25, 8, 8, 8

        def row_line(items):
            return "".join(
                [
                    pad_display(items[0], col_name),
                    pad_display(items[1], col_strength, "right"),
                    pad_display(items[2], col_weight, "right"),
                    pad_display(items[3], col_contrib, "right"),
                ]
            )

        lines.append("【买入因子详情】")
        lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
        lines.append("-" * 50)
        buy_contribs = sorted(
            [
                (k, buy_factors[k], buy_w.get(k, 0), buy_w.get(k, 0) * buy_factors[k])
                for k in buy_factors
            ],
            key=lambda x: x[3],
            reverse=True,
        )
        for name_f, s, w, contrib in buy_contribs:
            lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
        lines.append(row_line(["买入总分", "", "", f"{buy_score:.3f}"]))
        lines.append("")
        lines.append("【卖出因子详情】")
        lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
        lines.append("-" * 50)
        sell_contribs = sorted(
            [
                (
                    k,
                    sell_factors[k],
                    sell_w.get(k, 0),
                    sell_w.get(k, 0) * sell_factors[k],
                )
                for k in sell_factors
            ],
            key=lambda x: x[3],
            reverse=True,
        )
        for name_f, s, w, contrib in sell_contribs:
            lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
        lines.append(row_line(["卖出总分", "", "", f"{sell_score:.3f}"]))
        lines.append("")
        lines.append("【评分合成】")
        lines.append(
            f"原始净分 = 买入总分 - 卖出总分 = {buy_score:.3f} - {sell_score:.3f} = {raw:.3f}"
        )
        lines.append("最终评分 = 原始净分 × 市场因子 × 情绪因子")
        lines.append(
            f"        = {raw:.3f} × {market['market_factor']:.2f} × {market['sentiment_factor']:.2f} = {final:.3f}"
        )
        lines.append(f"操作等级：{action_level}")

        if ai_client:
            lines.append("")
            lines.append("【AI 专业点评】")
            ai_comment = ai_client.comment_on_etf(
                code,
                name,
                final,
                action_level,
                market["macro_status"],
                market["market_factor"],
                market["sentiment_factor"],
                buy_w,
                sell_w,
                buy_factors,
                sell_factors,
                tmsv,
                atr_pct,
            )
            lines.append(ai_comment)
        else:
            lines.append("")
            lines.append("【AI 专业点评】未配置 API_KEY，无法生成。")
        lines.append("=" * 70)
        return "\n".join(lines)


# ========================== 批量分析入口 ==========================
def run_batch_analysis(
    api_key: Optional[str] = None, target_code: Optional[str] = None
):
    fetcher = DataFetcher()
    analyzer = DataAnalyzer()

    if not fetcher.login():
        return
    try:
        etf_list = fetcher.load_positions()
    except Exception as e:
        print(f"请准备 positions.csv (代码,名称)，错误: {e}")
        return

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=200)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    market_df = fetcher.get_daily_data(MARKET_INDEX, start, today_str)
    macro_df = fetcher.get_daily_data(MACRO_INDEX, start, today_str)
    if market_df is None or macro_df is None:
        print("获取宏观数据失败")
        return

    macro_df = analyzer.calculate_indicators(macro_df, need_amount_ma=False)
    macro_df["ma_long"] = macro_df["close"].rolling(MACRO_MA_LONG).mean()
    market_df = analyzer.calculate_indicators(market_df, need_amount_ma=True)
    market_df["atr"] = analyzer.calculate_atr(market_df, ATR_PERIOD)
    volatility = (market_df["atr"] / market_df["close"]).iloc[-20:].mean()

    ai_client = None
    if api_key:
        ai_client = AIClient(api_key)
        market_state, market_factor = fetcher.get_market_state(market_df, ai_client)
    else:
        last = market_df.iloc[-1]
        above_ma20 = last["close"] > last["ma_short"]
        above_ma60 = last["close"] > last.get("ma_long", last["ma_short"])
        if above_ma20 and above_ma60:
            market_state, market_factor = "正常牛市", 1.2
        elif not above_ma20 and not above_ma60:
            market_state, market_factor = "熊市下跌", 0.8
        else:
            market_state, market_factor = "震荡偏弱", 1.0
        logger.info(f"简单规则市场状态: {market_state}, 因子: {market_factor}")

    sentiment = 1.0
    sentiment_risk_tip = ""
    if AKSHARE_AVAILABLE:
        try:
            sentiment_indicators = fetcher.fetch_sentiment_indicators()
            sentiment = fetcher.compute_sentiment_factor(sentiment_indicators)
            sentiment_risk_tip = fetcher.get_sentiment_risk_tip(sentiment)
            logger.info(f"综合情绪因子: {sentiment:.3f}")
        except Exception as e:
            logger.warning(f"获取情绪指标失败，使用后备RSI情绪: {e}")
            sentiment = fetcher.get_sentiment_factor_simple(macro_df)
    else:
        sentiment = fetcher.get_sentiment_factor_simple(macro_df)
        logger.info(f"使用后备RSI情绪因子: {sentiment:.3f}")

    mkt = market_df.iloc[-1]
    market_info = {
        "macro_status": market_state,
        "market_factor": market_factor,
        "sentiment_factor": sentiment,
        "sentiment_risk_tip": sentiment_risk_tip,
        "market_above_ma20": mkt["close"] > mkt["ma_short"],
        "market_above_ma60": mkt["close"] > mkt.get("ma_long", mkt["ma_short"]),
        "market_amount_above_ma20": mkt["amount"] > mkt["amount_ma"],
        "ret_market_5d": (
            (mkt["close"] / market_df.iloc[-5]["close"] - 1)
            if len(market_df) >= 5
            else 0
        ),
    }

    params = DEFAULT_PARAMS.copy()
    if volatility > 0.04:
        params.update(
            {
                "BUY_THRESHOLD": 0.65,
                "SELL_THRESHOLD": -0.35,
                "CONFIRM_DAYS": 5,
                "QUICK_BUY_THRESHOLD": 0.75,
            }
        )
    elif volatility > 0.02:
        params.update(
            {
                "BUY_THRESHOLD": 0.6,
                "SELL_THRESHOLD": -0.3,
                "CONFIRM_DAYS": 4,
                "QUICK_BUY_THRESHOLD": 0.7,
            }
        )
    elif volatility < 0.01:
        params.update(
            {
                "BUY_THRESHOLD": 0.4,
                "SELL_THRESHOLD": -0.1,
                "CONFIRM_DAYS": 2,
                "QUICK_BUY_THRESHOLD": 0.5,
            }
        )

    state = fetcher.load_state()
    buy_w, sell_w = DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()

    if ai_client:
        # ----- 内部函数：从 AI 生成权重并处理后返回 (buy_w, sell_w) -----
        def _generate_weights_from_ai():
            ai_buy, ai_sell = ai_client.generate_weights(
                market_state,
                sentiment,
                market_info["market_above_ma20"],
                market_info["market_above_ma60"],
                market_info["market_amount_above_ma20"],
                volatility,
            )
            trust = min(
                analyzer.compute_dynamic_trust(ai_buy, DEFAULT_BUY_WEIGHTS),
                analyzer.compute_dynamic_trust(ai_sell, DEFAULT_SELL_WEIGHTS),
            )
            corr_buy = analyzer.compute_factor_correlation(
                None, list(DEFAULT_BUY_WEIGHTS.keys())
            )
            corr_sell = analyzer.compute_factor_correlation(
                None, list(DEFAULT_SELL_WEIGHTS.keys())
            )
            ai_buy = analyzer.apply_correlation_penalty(
                ai_buy, list(DEFAULT_BUY_WEIGHTS.keys()), corr_buy
            )
            ai_sell = analyzer.apply_correlation_penalty(
                ai_sell, list(DEFAULT_SELL_WEIGHTS.keys()), corr_sell
            )
            buy_w = analyzer.blend_weights(ai_buy, DEFAULT_BUY_WEIGHTS, trust)
            sell_w = analyzer.blend_weights(ai_sell, DEFAULT_SELL_WEIGHTS, trust)

            # 情绪过热时调整卖出止盈权重
            if sentiment >= 1.25:
                boost = 0.1
                sell_w["profit_target_hit"] = min(
                    0.5, sell_w.get("profit_target_hit", 0) + boost
                )
                other_keys = [k for k in sell_w if k != "profit_target_hit"]
                if other_keys:
                    reduce_each = boost / len(other_keys)
                    for k in other_keys:
                        sell_w[k] = max(0.02, sell_w[k] - reduce_each)
                total = sum(sell_w.values())
                sell_w = {k: v / total for k, v in sell_w.items()}
            return buy_w, sell_w

        cache_key = fetcher._get_cache_key_fuzzy(
            market_state,
            sentiment,
            market_info["market_above_ma20"],
            market_info["market_above_ma60"],
            market_info["market_amount_above_ma20"],
            volatility,
        )
        cache = fetcher._load_cache()

        if (
            cache_key in cache
            and "buy" in cache[cache_key]
            and "sell" in cache[cache_key]
        ):
            cached_buy = validate_and_filter_weights(
                cache[cache_key]["buy"], DEFAULT_BUY_WEIGHTS.keys(), "缓存买入"
            )
            cached_sell = validate_and_filter_weights(
                cache[cache_key]["sell"], DEFAULT_SELL_WEIGHTS.keys(), "缓存卖出"
            )
            if cached_buy and cached_sell:
                buy_w, sell_w = cached_buy, cached_sell
            else:
                buy_w, sell_w = _generate_weights_from_ai()
                cache[cache_key] = {"buy": buy_w, "sell": sell_w}
                fetcher._save_cache(cache)
        else:
            buy_w, sell_w = _generate_weights_from_ai()
            cache[cache_key] = {"buy": buy_w, "sell": sell_w}
            fetcher._save_cache(cache)

    if target_code:
        target = etf_list[etf_list["代码"] == target_code]
        if target.empty:
            print(f"未找到代码 {target_code}，请检查 positions.csv")
            fetcher.logout()
            return
        row = target.iloc[0]
        code, name = row["代码"], row["名称"]
        hist = fetcher.get_daily_data(code, start, today_str)
        if hist is not None:
            hist = analyzer.calculate_indicators(
                hist,
                need_amount_ma=False,
                recent_high_window=params["RECENT_HIGH_WINDOW"],
                recent_low_window=params["RECENT_LOW_WINDOW"],
            )
        weekly = fetcher.get_weekly_data(code, start, today_str)
        etf_state = state.get(code, {})
        real_price = fetcher.get_realtime_price(code)
        report = analyzer.detailed_analysis(
            code,
            name,
            real_price,
            hist,
            weekly,
            market_info,
            today,
            etf_state,
            buy_w,
            sell_w,
            params,
            ai_client,
        )
        print(report)
        fetcher.logout()
        return
    else:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ETF 分析报告")
        print(f"市场状态: {market_state}, 市场因子: {market_factor:.2f}")
        if sentiment_risk_tip:
            print(f"情绪因子: {sentiment:.3f} - {sentiment_risk_tip}")
        else:
            print(f"情绪因子: {sentiment:.3f}")

        print(
            pad_display("名称", 16),
            pad_display("代码", 12),
            pad_display("价格", 8, "right"),
            pad_display("涨跌幅", 8, "right"),
            pad_display("评分", 6, "right"),
            "  " + pad_display("操作", 10),
        )
        print("-" * 80)

        output_lines = []
        results = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = []
            for _, row in etf_list.iterrows():
                code, name = row["代码"], row["名称"]
                hist = fetcher.get_daily_data(code, start, today_str)
                if hist is not None:
                    hist = analyzer.calculate_indicators(
                        hist,
                        need_amount_ma=False,
                        recent_high_window=params["RECENT_HIGH_WINDOW"],
                        recent_low_window=params["RECENT_LOW_WINDOW"],
                    )
                weekly = fetcher.get_weekly_data(code, start, today_str)
                s = state.get(code, {})
                futures.append(
                    ex.submit(
                        analyzer.analyze_single_etf,
                        code,
                        name,
                        fetcher.get_realtime_price(code),
                        hist,
                        weekly,
                        market_info,
                        today,
                        s,
                        buy_w,
                        sell_w,
                        params,
                    )
                )
            for f in futures:
                out, _, new_state, score = f.result()
                results.append((out, score))
                m = re.search(r"【.*?\((.*?)\)】", out)
                if m:
                    state[m.group(1)] = new_state
        results.sort(key=lambda x: x[1], reverse=True)
        for out, _ in results:
            print(out)
            output_lines.append(out)
        fetcher.save_state(state)
    fetcher.logout()

    email_cfg = get_email_config()
    if email_cfg["send_email"]:
        subject = f"ETF分析报告 - {today_str}"
        send_email(subject, "\n".join(output_lines))
