#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心分析模块：负责技术指标计算、评分、信号确认及批量分析入口。
"""
import datetime
import logging
import re
import time
import numpy as np
import pandas as pd
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .config import *
from .utils import (
    pad_display,
    validate_and_filter_weights,
    send_email,
    sigmoid_normalize,
    nonlinear_score_transform,
    clip_env_factor,
    cap,
    weighted_sum,
    get_dynamic_history_days,
    get_dynamic_confirm_days,
    # 因子强度函数
    factor_buy_price_above_ma20,
    factor_buy_volume_above_ma5,
    factor_buy_bollinger_break_up,
    factor_buy_williams_oversold,
    factor_buy_rsi_oversold,
    factor_buy_outperform_market,
    factor_sell_price_below_ma20,
    factor_sell_bollinger_break_down,
    factor_sell_williams_overbought,
    factor_sell_rsi_overbought,
    factor_sell_underperform_market,
    factor_sell_stop_loss_ma_break,
    factor_sell_trailing_stop_clear,
    factor_sell_trailing_stop_half,
    factor_sell_downside_momentum,
    factor_sell_max_drawdown_stop,
)
from .ai import AIClient
from .fetcher import DataFetcher, AKSHARE_AVAILABLE

logger = logging.getLogger(__name__)


# ========================== 数据类 ==========================
@dataclass
class ETFContext:
    """单只 ETF 的完整分析上下文，承载所有中间计算结果。"""
    code: str
    name: str
    real_price: Optional[float]
    hist_df: Optional[pd.DataFrame]
    weekly_df: Optional[pd.DataFrame]
    today: datetime.date
    market: Dict[str, Any]
    params: Dict[str, Any]

    change_pct: float = 0.0
    atr_pct: float = 0.0
    tmsv: float = 50.0
    tmsv_strength: float = 0.5
    downside_momentum: float = 0.0
    max_drawdown_pct: float = 0.0
    weekly_above: bool = False
    weekly_below: bool = False
    buy_factors: Dict = field(default_factory=dict)
    sell_factors: Dict = field(default_factory=dict)
    buy_score: float = 0.0
    sell_score: float = 0.0
    raw_score: float = 0.0
    final_score: float = 0.0
    error: Optional[str] = None

    recent_low_price: float = 0.0
    profit_pct_from_low: float = 0.0
    should_take_profit: bool = False
    effective_profit_threshold: float = TAKE_PROFIT_WARNING_THRESHOLD
    rsi: float = 50.0

    recent_high_price: float = 0.0                # 近期高点（用于移动止盈计算）
    trailing_profit_level: Optional[str] = None   # 'clear', 'half', 或 None
    trailing_profit_detail: str = ""              # 止盈提示文本（预留）

    # 新增：低点涨幅止盈提示级别
    profit_level: Optional[str] = None            # 'clear', 'half', 'watch', None
    take_profit_summary: str = ""                 # 综合提示文本


# ========================== 公共指标计算 ==========================
def calc_rsi(series: pd.Series, period: int = RSI_WINDOW) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


def calc_macd(series, fast=MACD_FAST, slow=MACD_SLOW, signal=MACD_SIGNAL):
    exp_fast = series.ewm(span=fast, adjust=False).mean()
    exp_slow = series.ewm(span=slow, adjust=False).mean()
    dif = exp_fast - exp_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


# ========================== 核心分析类 ==========================
class DataAnalyzer:
    """主要分析引擎，包含技术指标计算、因子评分、信号判定、权重管理等功能。"""
    def __init__(self, buy_weights=None, sell_weights=None, params=None):
        self.buy_weights = buy_weights or DEFAULT_BUY_WEIGHTS.copy()
        self.sell_weights = sell_weights or DEFAULT_SELL_WEIGHTS.copy()
        self.params = params or DEFAULT_PARAMS.copy()
        self.market_info = {}
        self._indicator_cache = {}

    def set_market_info(self, market_info):
        self.market_info = market_info

    def set_weights(self, buy_w, sell_w):
        self.buy_weights = buy_w
        self.sell_weights = sell_w

    # ---------- 内部工具方法 ----------
    @staticmethod
    def _nonlinear_score_transform(raw, market_status="震荡偏弱"):
        return nonlinear_score_transform(raw, market_status, NONLINEAR_SCALE_BULL, NONLINEAR_SCALE_RANGE)

    def _get_tmsv_weights(self, market_status, volatility):
        if "牛" in market_status:
            w = {"trend": 0.40, "momentum": 0.25, "volume": 0.15}
        elif "熊" in market_status:
            w = {"trend": 0.20, "momentum": 0.30, "volume": 0.25}
        else:
            w = {"trend": 0.25, "momentum": 0.40, "volume": 0.15}
        if volatility > TMSV_HIGH_VOL_THRESH:
            w["trend"] = max(TMSV_MIN_TREND_WEIGHT, w["trend"] - TMSV_TREND_REDUCE)
            w["momentum"] += TMSV_TREND_REDUCE
        return w

    def calc_change_pct(self, real_price, hist_df, today):
        if real_price is None or hist_df is None or len(hist_df) < 1:
            return 0.0
        last_date = hist_df.index[-1].date()
        if last_date == today and len(hist_df) >= 2:
            base_close = hist_df.iloc[-2]["close"]
        else:
            base_close = hist_df.iloc[-1]["close"]
        return (real_price - base_close) / base_close * 100 if base_close > 0 else 0.0

    # ---------- 技术指标计算 ----------
    def calculate_atr(self, df, period=14):
        tr = pd.concat([df["high"] - df["low"],
                         abs(df["high"] - df["close"].shift()),
                         abs(df["low"] - df["close"].shift())], axis=1).max(1)
        return tr.rolling(period).mean()

    def calculate_adx(self, df, period=14):
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
        return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx}, index=df.index)

    def _get_cache_key(self, code, start_date, end_date):
        today_str = datetime.date.today().strftime("%Y%m%d")
        raw = f"{code}_{start_date}_{end_date}_{today_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def calculate_indicators(self, df, need_amount_ma=True,
                             recent_high_window=10, recent_low_window=20,
                             use_cache=False, cache_key=None):
        if use_cache and cache_key and cache_key in self._indicator_cache:
            cached_df, cache_time = self._indicator_cache[cache_key]
            if (time.time() - cache_time) < CACHE_EXPIRE_SECONDS:
                return cached_df.copy()

        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=ETF_MA).mean()
        df["vol_ma"] = df["volume"].rolling(window=ETF_VOL_MA).mean()
        if need_amount_ma:
            df["amount_ma"] = df["amount"].rolling(window=ETF_VOL_MA).mean()
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
        df["atr"] = self.calculate_atr(df, ATR_PERIOD)
        adx_df = self.calculate_adx(df)
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]
        df["adx"] = adx_df["adx"]
        df["downside_momentum_raw"] = np.where(
            (df["close"] < df["ma_short"]) & (df["minus_di"] > df["plus_di"]),
            (df["ma_short"] - df["close"]) / df["ma_short"] * (df["volume"] / df["vol_ma"]).clip(0, 3), 0)
        df[f"recent_high_{recent_high_window}"] = df["high"].rolling(recent_high_window).max()
        df[f"recent_low_{recent_low_window}"] = df["low"].rolling(recent_low_window).min()

        if use_cache and cache_key:
            self._indicator_cache[cache_key] = (df.copy(), time.time())
        return df

    def compute_tmsv(self, df, market_status="震荡偏弱", volatility=0.02):
        if df is None or len(df) < 20:
            return pd.Series([50.0] * max(1, len(df))) if len(df) > 0 else pd.Series([50.0])

        df = df.copy()
        if "ma20" not in df.columns: df["ma20"] = df["close"].rolling(TMSV_MA20_WINDOW).mean()
        if "ma60" not in df.columns: df["ma60"] = df["close"].rolling(TMSV_MA60_WINDOW).mean()
        if "rsi" not in df.columns: df["rsi"] = calc_rsi(df["close"])
        if "macd_hist" not in df.columns: _, _, df["macd_hist"] = calc_macd(df["close"])
        if "atr" not in df.columns: df["atr"] = self.calculate_atr(df, TMSV_ATR_WINDOW)
        if "vol_ma" not in df.columns: df["vol_ma"] = df["volume"].rolling(TMSV_VOL_MA_WINDOW).mean()

        price_above_ma20 = ((df["close"] - df["ma20"]) / (df["ma20"].replace(0, np.nan) * 0.1)).clip(0,1).fillna(0)
        price_above_ma60 = ((df["close"] - df["ma60"]) / (df["ma60"].replace(0, np.nan) * 0.1)).clip(0,1).fillna(0)
        ma20_slope = df["ma20"].diff(5) / df["ma20"].shift(5).replace(0, np.nan)
        slope_score = (ma20_slope * 10).clip(0,1).fillna(0)
        trend_score = (price_above_ma20 * TMSV_TREND_MA20_WEIGHT +
                       price_above_ma60 * TMSV_TREND_MA60_WEIGHT +
                       slope_score * TMSV_TREND_SLOPE_WEIGHT) * 100

        rsi_score = ((df["rsi"] - 50) * 3.33).clip(0,100).fillna(50)
        macd_change = df["macd_hist"].diff() / (df["macd_hist"].shift(1).abs() + 0.001)
        macd_score = (macd_change * 100).clip(0,100).fillna(50)
        mom_score = rsi_score * TMSV_MOM_RSI_WEIGHT + macd_score * TMSV_MOM_MACD_WEIGHT

        vol_ratio = df["volume"] / df["vol_ma"].replace(0, np.nan)
        vol_ratio_score = ((vol_ratio - 0.8) / 1.2 * 100).clip(0,100).fillna(50)
        price_up = df["close"] > df["close"].shift(1)
        vol_up = df["volume"] > df["vol_ma"]
        consistency = np.where(price_up == vol_up, 100, 0)
        vol_score = vol_ratio_score * TMSV_VOL_RATIO_WEIGHT + consistency * TMSV_VOL_CONSIST_WEIGHT

        atr_pct = df["atr"] / df["close"].replace(0, np.nan)
        vol_factor = np.select(
            [atr_pct < TMSV_VOL_LOW_THRESH, atr_pct > TMSV_VOL_HIGH_THRESH],
            [TMSV_VOL_LOW_FACTOR, TMSV_VOL_HIGH_FACTOR],
            default=TMSV_VOL_MID_FACTOR_BASE - (atr_pct - TMSV_VOL_LOW_THRESH) / TMSV_VOL_BAND_WIDTH * TMSV_VOL_MID_FACTOR_SLOPE)
        vol_factor = np.nan_to_num(vol_factor, nan=1.0)

        w = self._get_tmsv_weights(market_status, volatility)
        tmsv = (trend_score * w['trend'] + mom_score * w['momentum'] + vol_score * w['volume']) * vol_factor
        return tmsv.clip(0,100).fillna(50)

    # ---------- 权重处理 ----------
    def compute_dynamic_trust(self, ai_weights, default_weights):
        base_trust = 0.75
        zero_count = sum(1 for v in ai_weights.values() if v < 0.01)
        if zero_count > len(ai_weights) * 0.3: base_trust = min(base_trust, 0.5)
        max_w = max(ai_weights.values())
        if max_w > 0.45: base_trust = min(base_trust, 0.6)
        dot = sum(ai_weights.get(k,0) * default_weights.get(k,0) for k in default_weights)
        norm_ai = math.sqrt(sum(v**2 for v in ai_weights.values()))
        norm_def = math.sqrt(sum(v**2 for v in default_weights.values()))
        if norm_ai > 0 and norm_def > 0 and (similarity := dot / (norm_ai * norm_def)) < 0.6:
            base_trust = min(base_trust, 0.55)
        return base_trust

    def blend_weights(self, ai_w, def_w, trust):
        blended = {k: ai_w.get(k,0) * trust + def_w[k] * (1 - trust) for k in def_w}
        total = sum(blended.values())
        return {k: v/total for k, v in blended.items()} if total > 0 else blended

    def apply_correlation_penalty(self, weights, factor_names, corr_matrix, penalty_threshold=0.7):
        w = weights.copy()
        high_pairs = []
        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i+1:]:
                if corr_matrix.loc[f1, f2] > penalty_threshold and w.get(f1,0) > 0.12 and w.get(f2,0) > 0.12:
                    high_pairs.append((f1, f2))
        for f1, f2 in high_pairs:
            w[f1] *= 0.8; w[f2] *= 0.8
        total = sum(w.values())
        return {k: v/total for k, v in w.items()} if total > 0 else w

    def compute_factor_correlation(self, df, factor_names):
        return pd.DataFrame(np.eye(len(factor_names)), index=factor_names, columns=factor_names)

    def generate_ai_weights(self, ai_client, market_state, sentiment,
                            market_above_ma20, market_above_ma60, market_amount_above_ma20, volatility):
        ai_buy, ai_sell = ai_client.generate_weights(market_state, sentiment,
                                                     market_above_ma20, market_above_ma60,
                                                     market_amount_above_ma20, volatility)
        trust = min(self.compute_dynamic_trust(ai_buy, DEFAULT_BUY_WEIGHTS),
                    self.compute_dynamic_trust(ai_sell, DEFAULT_SELL_WEIGHTS))
        corr_buy = self.compute_factor_correlation(None, BUY_FACTOR_NAMES)
        corr_sell = self.compute_factor_correlation(None, SELL_FACTOR_NAMES)
        ai_buy = self.apply_correlation_penalty(ai_buy, BUY_FACTOR_NAMES, corr_buy)
        ai_sell = self.apply_correlation_penalty(ai_sell, SELL_FACTOR_NAMES, corr_sell)
        buy_w = self.blend_weights(ai_buy, DEFAULT_BUY_WEIGHTS, trust)
        sell_w = self.blend_weights(ai_sell, DEFAULT_SELL_WEIGHTS, trust)
        return buy_w, sell_w

    # ---------- 因子计算 ----------
    def _compute_factors(self, ctx, d):
        price = ctx.real_price
        ma20 = d["ma_short"]
        volume = d["volume"]
        vol_ma = d["vol_ma"]
        rsi = d["rsi"]
        boll_up = d["boll_up"]
        boll_low = d["boll_low"]
        williams_r = d["williams_r"]

        # 金叉/死叉
        macd_golden = kdj_golden = 0
        if len(ctx.hist_df) >= 2:
            prev = ctx.hist_df.iloc[-2]
            macd_golden = 1 if (d["macd_dif"] > d["macd_dea"] and prev["macd_dif"] <= prev["macd_dea"]) else 0
            kdj_golden = 1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0

        ret_etf_5d = (price / ctx.hist_df.iloc[-5]["close"] - 1) if len(ctx.hist_df) >= 5 else 0
        ret_market_5d = ctx.market.get("ret_market_5d", 0)
        weekly_above = ctx.weekly_above
        weekly_below = ctx.weekly_below
        recent_high = d.get(f"recent_high_{ctx.params['RECENT_HIGH_WINDOW']}",
                            ctx.hist_df["high"].rolling(ctx.params["RECENT_HIGH_WINDOW"]).max().iloc[-1])
        recent_low = d.get(f"recent_low_{ctx.params['RECENT_LOW_WINDOW']}",
                           ctx.hist_df["low"].rolling(ctx.params["RECENT_LOW_WINDOW"]).min().iloc[-1])
        atr_pct = ctx.atr_pct
        tmsv_strength = ctx.tmsv_strength
        downside_momentum = ctx.downside_momentum
        max_drawdown_pct = ctx.max_drawdown_pct
        mkt_ma20 = ctx.market.get("market_above_ma20", False)
        mkt_ma60 = ctx.market.get("market_above_ma60", False)
        mkt_amt = ctx.market.get("market_amount_above_ma20", False)

        buy_factors = {
            "price_above_ma20":      factor_buy_price_above_ma20(price, ma20),
            "volume_above_ma5":      factor_buy_volume_above_ma5(volume, vol_ma),
            "macd_golden_cross":     macd_golden,
            "kdj_golden_cross":      kdj_golden,
            "bollinger_break_up":    factor_buy_bollinger_break_up(price, boll_up),
            "williams_oversold":     factor_buy_williams_oversold(williams_r),
            "market_above_ma20":     1 if mkt_ma20 else 0,
            "market_above_ma60":     1 if mkt_ma60 else 0,
            "market_amount_above_ma20": 1 if mkt_amt else 0,
            "outperform_market":     factor_buy_outperform_market(ret_etf_5d, ret_market_5d),
            "weekly_above_ma20":     1 if weekly_above else 0,
            "tmsv_score":            tmsv_strength,
            "rsi_oversold":          factor_buy_rsi_oversold(rsi),
        }

        sell_factors = {
            "price_below_ma20":      factor_sell_price_below_ma20(price, ma20),
            "bollinger_break_down":  factor_sell_bollinger_break_down(price, boll_low),
            "williams_overbought":   factor_sell_williams_overbought(williams_r),
            "rsi_overbought":        factor_sell_rsi_overbought(rsi),
            "underperform_market":   factor_sell_underperform_market(ret_etf_5d, ret_market_5d),
            "stop_loss_ma_break":    factor_sell_stop_loss_ma_break(price, ma20),
            # 注意：不再包含 trailing_stop_clear 和 trailing_stop_half
            "weekly_below_ma20":     1 if weekly_below else 0,
            "downside_momentum":     factor_sell_downside_momentum(downside_momentum),
            "max_drawdown_stop":     factor_sell_max_drawdown_stop(max_drawdown_pct),
        }
        return buy_factors, sell_factors
    

    # ---------- 信号确认 ----------
    def get_action(self, score, score_history, params, atr_pct=None) -> Tuple[str, str]:
        hist_scores = [s["score"] for s in score_history]
        buy_thresh = params["BUY_THRESHOLD"]
        sell_thresh = params["SELL_THRESHOLD"]

        def _get_level(s):
            for th, level in zip(ACTION_LEVEL_THRESHOLDS, ACTION_LEVEL_NAMES):
                if s >= th:
                    return level
            return "强烈卖出"

        action_level = _get_level(score)

        if len(hist_scores) < 2:
            if score > buy_thresh: return "BUY", action_level
            elif score < sell_thresh: return "SELL", action_level
            else: return "HOLD", action_level

        confirm_days = get_dynamic_confirm_days(atr_pct, params["CONFIRM_DAYS"])
        window = get_dynamic_history_days(atr_pct) if atr_pct else 12
        window = min(window, len(hist_scores))
        recent = hist_scores[-window:]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        up_days = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        down_days = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])

        if score > buy_thresh:
            if score > params["QUICK_BUY_THRESHOLD"] and slope > SIGNAL_SLOPE_BUY_THRESH and avg > buy_thresh + SIGNAL_AVG_OFFSET:
                return "BUY", action_level
            if len(hist_scores) >= confirm_days and all(s > buy_thresh for s in hist_scores[-confirm_days:]) and slope >= 0:
                return "BUY", action_level
            if down_days >= 2 and slope > SIGNAL_SLOPE_WEAK and score > avg + SIGNAL_AVG_OFFSET:
                return "BUY", action_level
            return "PREP_BUY", action_level

        if score < sell_thresh:
            if score < sell_thresh - SIGNAL_AVG_OFFSET and slope < SIGNAL_SELL_SLOPE and avg < sell_thresh - SIGNAL_AVG_OFFSET:
                return "SELL", action_level
            if len(hist_scores) >= confirm_days and all(s < sell_thresh for s in hist_scores[-confirm_days:]) and slope <= 0:
                return "SELL", action_level
            if up_days >= 2 and slope < SIGNAL_SELL_WEAK_SLOPE and score < avg - SIGNAL_AVG_OFFSET:
                return "SELL", action_level
            return "PREP_SELL", action_level

        if atr_pct and atr_pct > VOL_HIGH_CONFIRM:
            if score > buy_thresh + 0.15 and slope > SIGNAL_HIGH_VOL_BUY_SLOPE and up_days >= SIGNAL_HIGH_VOL_DAYS:
                return "BUY", action_level
            if score < sell_thresh - 0.05 and slope < -0.08 and down_days >= 3:
                return "SELL", action_level
            return "HOLD", action_level
        if atr_pct and atr_pct > VOL_MID_CONFIRM:
            if score > buy_thresh + 0.1 and slope > SIGNAL_MID_VOL_BUY_SLOPE and up_days >= SIGNAL_MID_VOL_DAYS:
                return "BUY", action_level
            if score < sell_thresh - 0.1 and slope < -0.08 and down_days >= 3:
                return "SELL", action_level
        return "HOLD", action_level

    def adjust_params_based_on_history(self, params, score_history, volatility, market_factor):
        if len(score_history) < 10:
            return params
        window = min(get_dynamic_history_days(volatility), len(score_history))
        recent = [s["score"] for s in score_history[-window:]]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        short_recent = [s["score"] for s in score_history[-min(3, len(score_history)):]]
        short_slope = np.polyfit(range(len(short_recent)), short_recent, 1)[0] if len(short_recent) >= 2 else 0
        adjust_mult = ADJUST_MULT_BASE / market_factor
        adjusted = params.copy()

        if slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = min(ADJUST_BUY_DELTA_MAX, abs(avg - params["BUY_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["BUY_THRESHOLD"] = max(0.35, params["BUY_THRESHOLD"] - delta)
        elif slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = min(ADJUST_BUY_DELTA_MAX, abs(avg - params["BUY_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["BUY_THRESHOLD"] = min(0.65, params["BUY_THRESHOLD"] + delta)
        else:
            adjusted["BUY_THRESHOLD"] = params["BUY_THRESHOLD"] * 0.9 + DEFAULT_PARAMS["BUY_THRESHOLD"] * 0.1

        if slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = min(ADJUST_SELL_DELTA_MAX, abs(avg - params["SELL_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["SELL_THRESHOLD"] = max(-0.45, params["SELL_THRESHOLD"] - delta)
        elif slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = min(ADJUST_SELL_DELTA_MAX, abs(avg - params["SELL_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["SELL_THRESHOLD"] = min(-0.15, params["SELL_THRESHOLD"] + delta)
        else:
            adjusted["SELL_THRESHOLD"] = params["SELL_THRESHOLD"] * 0.9 + DEFAULT_PARAMS["SELL_THRESHOLD"] * 0.1

        if volatility > VOL_HIGH_CONFIRM:
            adjusted["CONFIRM_DAYS"] = min(5, int(round(params["CONFIRM_DAYS"] * 1.1)))
        elif volatility < 0.01:
            adjusted["CONFIRM_DAYS"] = max(2, int(round(params["CONFIRM_DAYS"] * 0.9)))
        else:
            adjusted["CONFIRM_DAYS"] = int(round(params["CONFIRM_DAYS"] * 0.95 + DEFAULT_PARAMS["CONFIRM_DAYS"] * 0.05))
        return adjusted

    def _get_effective_take_profit_threshold(self, ctx):
        base = TAKE_PROFIT_WARNING_THRESHOLD
        market_status = ctx.market.get("macro_status", "")
        if "牛" in market_status: base *= 1.10
        elif "熊" in market_status: base *= 0.90
        if ctx.atr_pct > 0.03: base *= 0.85
        return base

    # ---------- 核心分析 ----------
    def _core_analysis(self, ctx):
        if ctx.real_price is None:
            ctx.error = "实时价格获取失败"; return ctx
        if ctx.hist_df is None or len(ctx.hist_df) < 20:
            ctx.error = "历史数据不足"; return ctx

        ctx.change_pct = self.calc_change_pct(ctx.real_price, ctx.hist_df, ctx.today)
        d = ctx.hist_df.iloc[-1]
        atr_pct = d["atr"] / ctx.real_price if ctx.real_price > 0 else 0
        ctx.atr_pct = atr_pct
        ctx.rsi = d["rsi"]

        weekly_above = weekly_below = False
        if ctx.weekly_df is not None and not ctx.weekly_df.empty:
            w = ctx.weekly_df.iloc[-1]
            if "ma_short" in w.index and not pd.isna(w["ma_short"]):
                weekly_above = w["close"] > w["ma_short"]
                weekly_below = w["close"] < w["ma_short"]
        ctx.weekly_above = weekly_above
        ctx.weekly_below = weekly_below

        market_status = ctx.market.get("macro_status", "震荡偏弱")
        try:
            tmsv_series = self.compute_tmsv(ctx.hist_df, market_status, atr_pct)
            tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
            tmsv = 50.0 if np.isnan(tmsv) else tmsv
        except Exception:
            logger.error("TMSV 计算失败", exc_info=True)
            tmsv = 50.0
        ctx.tmsv = tmsv
        ctx.tmsv_strength = tmsv / 100.0

        ctx.downside_momentum = d.get("downside_momentum_raw", 0.0)
        recent_high = d.get(f"recent_high_{ctx.params['RECENT_HIGH_WINDOW']}",
                            ctx.hist_df["high"].rolling(ctx.params["RECENT_HIGH_WINDOW"]).max().iloc[-1])
        ctx.recent_high_price = recent_high
        ctx.max_drawdown_pct = (recent_high - ctx.real_price) / recent_high if recent_high > 0 else 0.0

        recent_low = d.get(f"recent_low_{ctx.params['RECENT_LOW_WINDOW']}",
                           ctx.hist_df["low"].rolling(ctx.params["RECENT_LOW_WINDOW"]).min().iloc[-1])
        ctx.recent_low_price = recent_low
        if recent_low > 0:
            ctx.profit_pct_from_low = (ctx.real_price - recent_low) / recent_low
            ctx.effective_profit_threshold = self._get_effective_take_profit_threshold(ctx)
            if ctx.profit_pct_from_low >= ctx.effective_profit_threshold:
                ctx.should_take_profit = True

        # --- 移动止盈判断（不参与评分）---
        from .utils import get_trailing_profit_signals
        ctx.trailing_profit_level = get_trailing_profit_signals(
            ctx.real_price, recent_high, ctx.atr_pct
        )

        ctx.buy_factors, ctx.sell_factors = self._compute_factors(ctx, d)
        ctx.buy_score = weighted_sum(ctx.buy_factors, self.buy_weights)
        ctx.sell_score = weighted_sum(ctx.sell_factors, self.sell_weights)

        if ctx.buy_factors.get("macd_golden_cross",0) == 0 and ctx.buy_factors.get("kdj_golden_cross",0) == 0:
            ctx.buy_score *= MOMENTUM_MISSING_PENALTY

        if ctx.sell_factors.get("max_drawdown_stop",0) > 0 or ctx.sell_factors.get("stop_loss_ma_break",0) > 0:
            ctx.final_score = -1.0
            ctx.raw_score = ctx.buy_score - ctx.sell_score
            # 强制卖出逻辑保留，但止盈提示不参与此处分值
            return ctx

        sentiment = ctx.market.get("sentiment_factor", 1.0)
        if sentiment >= SENTIMENT_OVERHEAT_THRESHOLD:
            ctx.buy_score *= SENTIMENT_PENALTY_FACTOR

        ctx.raw_score = ctx.buy_score - ctx.sell_score
        transformed_raw = self._nonlinear_score_transform(ctx.raw_score, market_status)
        env_factor = clip_env_factor(ctx.market["market_factor"], sentiment)
        ctx.final_score = max(-1.0, min(1.0, transformed_raw * env_factor))

        # ---------- 纯提示止盈判断（不修改评分）----------
        ctx.profit_level = None
        if ctx.profit_pct_from_low >= ctx.effective_profit_threshold * PROFIT_LOW_CLEAR_MULT:
            ctx.profit_level = 'clear'
        elif ctx.profit_pct_from_low >= ctx.effective_profit_threshold * PROFIT_LOW_HALF_MULT:
            ctx.profit_level = 'half'
        elif ctx.profit_pct_from_low >= ctx.effective_profit_threshold * PROFIT_LOW_WATCH_MULT:
            ctx.profit_level = 'watch'

        # 综合提示文本（可选，供详细报告使用）
        parts = []
        if ctx.trailing_profit_level == 'clear':
            parts.append("🚨 移动清仓")
        elif ctx.trailing_profit_level == 'half':
            parts.append("📈 移动半仓")
        else:
            pct = ctx.profit_pct_from_low * 100
            if ctx.profit_level == 'clear':
                parts.append(f"🚨  低点涨+{pct:.1f}%")
            elif ctx.profit_level == 'half':
                parts.append(f"📈  低点涨+{pct:.1f}%")
            else:
                parts.append(f"💡  低点涨+{pct:.1f}%")
        ctx.take_profit_summary = " ".join(parts)

        return ctx
    

    # ---------- 简要分析 ----------
    def analyze_single_etf(self, code, name, real_price, hist_df, weekly_df,
                        market, today, state, ai_client=None):
        """
        单只 ETF 完整分析流水线。
        返回: (output, signal, state, final_score)
        其中操作信号(action)仅由技术评分决定，止盈风险通过 risk_str 提示，
        既不假设持仓，也不强制覆盖卖出。
        """
        # 1. 构建上下文并执行核心计算
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy())
        ctx = self._core_analysis(ctx)

        # 2. 处理数据错误
        if ctx.error:
            if "实时价格" in ctx.error:
                out = (f"{pad_display(name, 14)} {pad_display(code, 12)} "
                    f"{pad_display('获取失败', 8)} {pad_display('0.00%', 8, 'right')} "
                    f"{pad_display('0.00', 6, 'right')}  {pad_display('价格缺失', 16)}")
            else:
                price_str = f"{real_price:.3f}" if real_price else "N/A"
                change_str = f"{ctx.change_pct:+.2f}%"
                out = (f"{pad_display(name, 14)} {pad_display(code, 12)} "
                    f"{pad_display(price_str, 8, 'right')} "
                    f"{pad_display(change_str, 8, 'right')} "
                    f"{pad_display('0.00', 6, 'right')}  {pad_display('数据不足', 16)}")
            return out, None, state, 0.0

        final = ctx.final_score
        today_str = today.strftime("%Y-%m-%d")

        # 3. 更新评分历史并动态调整参数
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
            self.params = self.adjust_params_based_on_history(
                self.params, state["score_history"], ctx.atr_pct, market["market_factor"]
            )

        # 4. 获取纯技术操作建议（不包含止盈）
        action, action_level = self.get_action(final, state["score_history"], self.params, ctx.atr_pct)

        # 5. 构建风险/机会提示文本 (risk_str)
        risk_parts = []

        # 5.1 原有风险提示（连续低分、极端评分、高波动）
        if len(state["score_history"]) >= RISK_WARNING_DAYS:
            recent_scores = [s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]]
            if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
                risk_parts.append(f"连续{RISK_WARNING_DAYS}日低评分")
            elif final < -0.5 or final > 0.8:
                risk_parts.append(f"极端评分 {final:.2f}")
            elif ctx.atr_pct > 0.03:
                risk_parts.append(f"高波动 {ctx.atr_pct*100:.1f}%")

        # 5.2 止盈风险描述（仅供提醒，不修改操作信号）
        tp_detail = []
        if ctx.trailing_profit_level:
            if ctx.recent_high_price > 0 and ctx.real_price:
                fall = (ctx.recent_high_price - ctx.real_price) / ctx.recent_high_price
                tp_detail.append(f"高点回落{fall:.1%}")
        if ctx.profit_level:
            tp_detail.append(f"低点涨{ctx.profit_pct_from_low:.1%}")
        if tp_detail:
            risk_parts.append("止盈风险:" + "，".join(tp_detail))

        # 5.3 低位机会提示
        if ctx.profit_pct_from_low is not None and ctx.profit_pct_from_low <= 0.03:
            risk_parts.append("低位区间")

        # 合并风险提示
        risk_str = " ".join(risk_parts) if risk_parts else ""

        # 6. 根据止盈风险调整操作等级（降级，但不变更 action 类别）
        #    例如：原等级“强烈买入”，止盈清仓级→降为“买入（高位风险）”
        adjusted_level = action_level
        # 定义降级映射，只针对偏多方向（从高到低）
        if ctx.trailing_profit_level == 'clear' or ctx.profit_level == 'clear':
            # 清仓级止盈：强烈降级
            downgrade_map = {
                "极度看好": "谨慎买入(高估)",
                "强烈买入": "谨慎买入(高估)",
                "买入": "谨慎买入(高估)",
                "谨慎买入": "偏多持有(高估)",
                "偏多持有": "中性偏多(高估)",
            }
            adjusted_level = downgrade_map.get(action_level, action_level)
        elif ctx.trailing_profit_level == 'half' or ctx.profit_level == 'half':
            # 半仓级止盈：温和降级
            downgrade_map = {
                "极度看好": "强烈买入(注意止盈)",
                "强烈买入": "买入(注意止盈)",
                "买入": "谨慎买入(注意止盈)",
                "谨慎买入": "偏多持有",
                "偏多持有": "中性偏多",
            }
            adjusted_level = downgrade_map.get(action_level, action_level)

        # 如果原始已经是偏空/卖出方向，不再降级（保留原样）
        sell_or_neutral = {"中性偏空", "偏空持有", "谨慎卖出", "卖出", "强烈卖出"}
        if action_level in sell_or_neutral:
            adjusted_level = action_level

        # 7. 格式输出（使用调整后的等级）
        from .utils import format_etf_output_line
        output = format_etf_output_line(
            name=name,
            code=code,
            price=real_price,
            change_pct=ctx.change_pct,
            final_score=final,
            action_level=adjusted_level,
            atr_pct=ctx.atr_pct,
            trailing_profit_level=None,   # 已整合到 risk_str，避免重复图标
            recent_high_price=ctx.recent_high_price,
            risk_str=risk_str,
            profit_level=None,            # 同上
            profit_pct_from_low=None,
        )

        # 8. 信号输出（基于原始 action，如 BUY、SELL）
        signal = None
        if action in ("BUY", "SELL"):
            signal = {"action": action, "name": name, "code": code, "score": final}

        return output, signal, state, final
        
        
        
        # ---------- 详细分析报告 ----------
        def detailed_analysis(self, code, name, real_price, hist_df, weekly_df,
                            market, today, state, ai_client=None):
            ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy())
            ctx = self._core_analysis(ctx)
            if ctx.error:
                return f"【{name} ({code})】{ctx.error}，无法分析。"

            final = ctx.final_score
            _, action_level = self.get_action(final, state.get("score_history", []), self.params, ctx.atr_pct)

            lines = ["=" * 70,
                    f"ETF详细分析报告 - {name} ({code})",
                    f"分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    "=" * 70,
                    f"实时价格：{real_price:.3f}",
                    f"涨跌幅：{ctx.change_pct:+.2f}%",
                    f"市场状态：{market['macro_status']}，市场因子：{market['market_factor']:.2f}，情绪因子：{market['sentiment_factor']:.2f}"]
            if market.get("sentiment_risk_tip"): lines.append(f"情绪风险提示：{market['sentiment_risk_tip']}")
            lines += [f"波动率(ATR%)：{ctx.atr_pct*100:.2f}%",
                    f"TMSV复合强度：{ctx.tmsv:.1f} (强度系数 {ctx.tmsv_strength:.3f})",
                    f"最大回撤：{ctx.max_drawdown_pct*100:.2f}%", ""]

            
            # 止盈观察区块（展示移动止盈 + 低点涨幅止盈）
            if ctx.trailing_profit_level or ctx.profit_level:
                lines.append("【止盈观察 (仅供参考)】")
                if ctx.trailing_profit_level:
                    recent_high = ctx.recent_high_price
                    from_high_pct = (recent_high - ctx.real_price) / recent_high if recent_high > 0 else 0
                    lines.append(f"  从{ctx.params['RECENT_HIGH_WINDOW']}日高点 {recent_high:.3f} 回落 {from_high_pct:.1%}")
                    level_text = "清仓级" if ctx.trailing_profit_level == 'clear' else "半仓级"
                    lines.append(f"  移动止盈信号：{level_text}")
                if ctx.profit_level:
                    lines.append(f"  距{ctx.params['RECENT_LOW_WINDOW']}日低点 {ctx.recent_low_price:.3f} 涨幅 {ctx.profit_pct_from_low:.1%}")
                    level_map = {'clear': '清仓级', 'half': '半仓级', 'watch': '关注级'}
                    lines.append(f"  低点涨幅信号：{level_map.get(ctx.profit_level, '')}")
                lines.append("  *以上提示不构成自动卖出指令，请结合其他因素决策。")
                lines.append("")

            col_name, col_strength, col_weight, col_contrib = 25, 8, 8, 8
            def row_line(items):
                return "".join([pad_display(items[0], col_name), pad_display(items[1], col_strength, "right"),
                                pad_display(items[2], col_weight, "right"), pad_display(items[3], col_contrib, "right")])

            lines.append("【买入因子详情】")
            lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
            lines.append("-" * 50)
            buy_contribs = sorted([(k, ctx.buy_factors[k], self.buy_weights.get(k,0), self.buy_weights.get(k,0)*ctx.buy_factors[k])
                                for k in ctx.buy_factors], key=lambda x: x[3], reverse=True)
            for name_f, s, w, contrib in buy_contribs:
                lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
            lines.append(row_line(["买入总分", "", "", f"{ctx.buy_score:.3f}"]))
            lines.append("")

            lines.append("【卖出因子详情】")
            lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
            lines.append("-" * 50)
            sell_contribs = sorted([(k, ctx.sell_factors[k], self.sell_weights.get(k,0), self.sell_weights.get(k,0)*ctx.sell_factors[k])
                                    for k in ctx.sell_factors], key=lambda x: x[3], reverse=True)
            for name_f, s, w, contrib in sell_contribs:
                lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
            lines.append(row_line(["卖出总分", "", "", f"{ctx.sell_score:.3f}"]))
            lines.append("")

            scale = NONLINEAR_SCALE_BULL if ("牛" in market["macro_status"] or "熊" in market["macro_status"]) else NONLINEAR_SCALE_RANGE
            env_factor = clip_env_factor(market["market_factor"], market["sentiment_factor"])
            lines += ["【评分合成】",
                    f"原始净分 = {ctx.buy_score:.3f} - {ctx.sell_score:.3f} = {ctx.raw_score:.3f}",
                    f"非线性变换 × 环境因子 ({env_factor:.2f}) → {final:.3f}",
                    f"操作等级：{action_level}"]

            if ai_client:
                lines += ["", "【AI 专业点评】"]
                ai_comment = ai_client.comment_on_etf(code, name, final, action_level,
                                                    market["macro_status"], market["market_factor"],
                                                    market["sentiment_factor"], self.buy_weights, self.sell_weights,
                                                    ctx.buy_factors, ctx.sell_factors, ctx.tmsv, ctx.atr_pct)
                lines.append(ai_comment)
            else:
                lines += ["", "【AI 专业点评】未配置 API_KEY，无法生成。"]

            # AI 止盈建议（当存在止盈提示时显示）
            if ai_client and (ctx.trailing_profit_level or ctx.profit_level):
                lines += ["", "【AI 止盈建议】"]
                try:
                    ai_tp_advice = ai_client.take_profit_advice(
                        code, name, ctx.profit_pct_from_low,
                        ctx.recent_low_price, real_price,
                        ctx.tmsv, ctx.rsi, ctx.atr_pct,
                        market["macro_status"], market["sentiment_factor"]
                    )
                    lines.append(ai_tp_advice)
                except Exception as e:
                    logger.error(f"AI止盈建议生成失败: {e}")
                    lines.append("（止盈建议生成失败）")

            lines.append("=" * 70)
            return "\n".join(lines)


# ========================== 辅助函数 ==========================
def _prepare_etf_data(code, fetcher, analyzer, start, today_str, params):
    cache_key_hist = analyzer._get_cache_key(code, start, today_str)
    hist = fetcher.get_daily_data(code, start, today_str)
    if hist is not None:
        hist = analyzer.calculate_indicators(hist, need_amount_ma=False,
                                            recent_high_window=params["RECENT_HIGH_WINDOW"],
                                            recent_low_window=params["RECENT_LOW_WINDOW"],
                                            use_cache=True, cache_key=cache_key_hist)
    weekly = fetcher.get_weekly_data(code, start, today_str)
    real_price = fetcher.get_realtime_price(code)
    return hist, weekly, real_price


def _get_or_create_environment(fetcher, analyzer, market_df, macro_df, volatility, market_info_basic, api_key):
    cached_env = fetcher.get_cached_environment()
    if cached_env:
        return (cached_env["market_state"], cached_env["market_factor"],
                cached_env["sentiment"], cached_env["buy_weights"],
                cached_env["sell_weights"], fetcher.get_sentiment_risk_tip(cached_env["sentiment"]))

    ai_client = AIClient(api_key) if api_key else None
    if ai_client:
        market_state, market_factor = fetcher.get_market_state(market_df, ai_client)
    else:
        last = market_df.iloc[-1]
        above_ma20 = last["close"] > last["ma_short"]
        above_ma60 = last["close"] > last.get("ma_long", last["ma_short"])
        if above_ma20 and above_ma60: market_state, market_factor = "正常牛市", 1.2
        elif not above_ma20 and not above_ma60: market_state, market_factor = "熊市下跌", 0.8
        else: market_state, market_factor = "震荡偏弱", 1.0

    sentiment, sentiment_raw = 1.0, 1.0
    if AKSHARE_AVAILABLE:
        try:
            ind = fetcher.fetch_sentiment_indicators()
            sentiment, sentiment_raw = fetcher.compute_sentiment_factor(ind)
        except Exception as e:
            logger.warning(f"获取情绪失败，使用后备: {e}")
            sentiment, sentiment_raw = fetcher.get_sentiment_factor_simple(macro_df)
    else:
        sentiment, sentiment_raw = fetcher.get_sentiment_factor_simple(macro_df)
    sentiment_risk_tip = fetcher.get_sentiment_risk_tip(sentiment)

    buy_w, sell_w = DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()
    if api_key:
        ai_temp = AIClient(api_key)
        buy_w, sell_w = analyzer.generate_ai_weights(ai_temp, market_state, sentiment,
                                                    market_info_basic["market_above_ma20"],
                                                    market_info_basic["market_above_ma60"],
                                                    market_info_basic["market_amount_above_ma20"], volatility)
    fetcher.save_environment_cache(market_state, market_factor, buy_w, sell_w, sentiment, sentiment_raw)
    return market_state, market_factor, sentiment, buy_w, sell_w, sentiment_risk_tip


def run_batch_analysis(api_key=None, target_code=None):
    fetcher = DataFetcher()
    analyzer = DataAnalyzer()
    if not fetcher.login(): return
    try:
        etf_list = fetcher.load_positions()
    except Exception as e:
        print(f"请准备 positions.csv (代码,名称)，错误: {e}"); return

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=200)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    market_df = fetcher.get_daily_data(MARKET_INDEX, start, today_str)
    macro_df = fetcher.get_daily_data(MACRO_INDEX, start, today_str)
    if market_df is None or macro_df is None:
        print("获取宏观数据失败"); return

    macro_df = analyzer.calculate_indicators(macro_df, need_amount_ma=False)
    macro_df["ma_long"] = macro_df["close"].rolling(MACRO_MA_LONG).mean()
    market_df = analyzer.calculate_indicators(market_df, need_amount_ma=True)
    market_df["atr"] = analyzer.calculate_atr(market_df, ATR_PERIOD)
    volatility = (market_df["atr"] / market_df["close"]).iloc[-20:].mean()

    mkt = market_df.iloc[-1]
    market_info_basic = {
        "market_above_ma20": mkt["close"] > mkt["ma_short"],
        "market_above_ma60": mkt["close"] > mkt.get("ma_long", mkt["ma_short"]),
        "market_amount_above_ma20": mkt["amount"] > mkt["amount_ma"],
        "ret_market_5d": (mkt["close"] / market_df.iloc[-5]["close"] - 1) if len(market_df) >= 5 else 0,
    }

    market_state, market_factor, sentiment, buy_w, sell_w, risk_tip = \
        _get_or_create_environment(fetcher, analyzer, market_df, macro_df, volatility, market_info_basic, api_key)

    market_info = {"macro_status": market_state, "market_factor": market_factor,
                   "sentiment_factor": sentiment, "sentiment_risk_tip": risk_tip, **market_info_basic}
    analyzer.set_market_info(market_info)
    analyzer.set_weights(buy_w, sell_w)

    params = DEFAULT_PARAMS.copy()
    if volatility > 0.04:
        params.update({"BUY_THRESHOLD": 0.65, "SELL_THRESHOLD": -0.35, "CONFIRM_DAYS": 5, "QUICK_BUY_THRESHOLD": 0.75})
    elif volatility > 0.02:
        params.update({"BUY_THRESHOLD": 0.6, "SELL_THRESHOLD": -0.3, "CONFIRM_DAYS": 4, "QUICK_BUY_THRESHOLD": 0.7})
    elif volatility < 0.01:
        params.update({"BUY_THRESHOLD": 0.4, "SELL_THRESHOLD": -0.1, "CONFIRM_DAYS": 2, "QUICK_BUY_THRESHOLD": 0.5})
    analyzer.params = params

    state = fetcher.load_state()
    ai_client = AIClient(api_key) if api_key else None

    if target_code:
        target = etf_list[etf_list["代码"] == target_code]
        if target.empty:
            print(f"未找到代码 {target_code}"); fetcher.logout(); return
        code, name = target.iloc[0]["代码"], target.iloc[0]["名称"]
        hist, weekly, real_price = _prepare_etf_data(code, fetcher, analyzer, start, today_str, params)
        etf_state = state.get(code, {})
        report = analyzer.detailed_analysis(code, name, real_price, hist, weekly, market_info, today, etf_state, ai_client)
        print(report)
        fetcher.logout()
        return

    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ETF 分析报告")
    print(f"市场状态: {market_state}, 市场因子: {market_factor:.2f}")
    if risk_tip: print(f"情绪因子: {sentiment:.3f} - {risk_tip}")
    else: print(f"情绪因子: {sentiment:.3f}")

    print(pad_display("名称", 14), pad_display("代码", 12), pad_display("价格", 10, "right"),
          pad_display("涨跌", 10, "right"), pad_display("评分", 16, "right"), " " + pad_display("操作", 22), " 信号/提示")
    print("-" * 95)

    output_lines, results = [], []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = []
        for _, row in etf_list.iterrows():
            code, name = row["代码"], row["名称"]
            hist, weekly, real_price = _prepare_etf_data(code, fetcher, analyzer, start, today_str, params)
            s = state.get(code, {})
            futures.append(ex.submit(analyzer.analyze_single_etf, code, name, real_price, hist, weekly, market_info, today, s, ai_client))
        for f in futures:
            out, _, new_state, score = f.result()
            results.append((out, score))
            m = re.search(r"【.*?\((.*?)\)】", out)
            if m: state[m.group(1)] = new_state

    results.sort(key=lambda x: x[1], reverse=True)
    for out, _ in results:
        print(out); output_lines.append(out)

    fetcher.save_state(state)
    fetcher.logout()

    email_cfg = get_email_config()
    if email_cfg["send_email"]:
        send_email(f"ETF分析报告 - {today_str}", "\n".join(output_lines))
