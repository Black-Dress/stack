#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子强度计算模块：包含所有买入/卖出因子强度函数、移动止盈信号，
以及 TMSV 复合强度计算函数。
"""
import math
import numpy as np
import pandas as pd
from .utils import sigmoid_normalize, cap, calc_rsi, calc_macd, calculate_atr

# ---------- 因子计算常量 ----------
PRICE_DEVIATION_MA_MULT = 0.1
VOLUME_RATIO_CENTER = 0.2
SIGMOID_STEEPNESS_VOLUME = 3.0
WILLIAMS_OVERSOLD_THRESH = -80
WILLIAMS_OVERBOUGHT_THRESH = -20
WILLIAMS_NORMALIZE_DIV = 20
RSI_OVERBOUGHT_THRESH = 70
RSI_OVERBOUGHT_DIV = 30
RSI_OVERSOLD_THRESH = 30
OUTPERFORM_MARKET_DIV = 0.05
HARD_STOP_MA_BREAK_PCT = 0.05
MAX_DRAWDOWN_STOP_DIV = 0.08
ATR_STOP_MULT = 3.0          # 移动止损倍数
ATR_TRAILING_MULT = 2.0      # 移动止盈倍数


# ========================== 买入因子 ==========================
def factor_buy_price_above_ma20(price: float, ma20: float) -> float:
    if price <= ma20 or ma20 <= 0:
        return 0.0
    deviation = (price - ma20) / (ma20 * PRICE_DEVIATION_MA_MULT)
    return sigmoid_normalize(deviation, center=0.2)


def factor_buy_volume_above_ma5(volume: float, vol_ma: float) -> float:
    if volume <= vol_ma or vol_ma <= 0:
        return 0.0
    ratio = volume / vol_ma - 1.0
    return sigmoid_normalize(ratio, center=VOLUME_RATIO_CENTER, steepness=SIGMOID_STEEPNESS_VOLUME)


def factor_buy_bollinger_break_up(price: float, boll_up: float) -> float:
    if price <= boll_up:
        return 0.0
    return sigmoid_normalize((price - boll_up) / boll_up, center=0.01)


def factor_buy_williams_oversold(williams_r: float) -> float:
    if williams_r < WILLIAMS_OVERSOLD_THRESH:
        return 0.0
    return sigmoid_normalize(
        (WILLIAMS_OVERSOLD_THRESH - williams_r) / WILLIAMS_NORMALIZE_DIV, center=0.5
    )


def factor_buy_rsi_oversold(rsi: float) -> float:
    if rsi < RSI_OVERSOLD_THRESH:
        return max(0.0, (RSI_OVERSOLD_THRESH - rsi) / RSI_OVERSOLD_THRESH)
    return 0.0


def factor_buy_outperform_market(ret_etf_5d: float, ret_market_5d: float) -> float:
    if ret_etf_5d > ret_market_5d:
        return sigmoid_normalize((ret_etf_5d - ret_market_5d) / OUTPERFORM_MARKET_DIV, center=0.2)
    return 0.0


# ========================== 卖出因子 ==========================
def factor_sell_price_below_ma20(price: float, ma20: float) -> float:
    if price >= ma20 or ma20 <= 0:
        return 0.0
    deviation = (price - ma20) / (ma20 * PRICE_DEVIATION_MA_MULT)
    return sigmoid_normalize(-deviation, center=0.2)


def factor_sell_bollinger_break_down(price: float, boll_low: float) -> float:
    if price >= boll_low:
        return 0.0
    return sigmoid_normalize((boll_low - price) / boll_low, center=0.01)


def factor_sell_williams_overbought(williams_r: float) -> float:
    if williams_r >= WILLIAMS_OVERBOUGHT_THRESH:
        return 0.0
    return sigmoid_normalize(
        (WILLIAMS_OVERBOUGHT_THRESH - williams_r) / WILLIAMS_NORMALIZE_DIV, center=0.5
    )


def factor_sell_rsi_overbought(rsi: float) -> float:
    if rsi > RSI_OVERBOUGHT_THRESH:
        return sigmoid_normalize((rsi - RSI_OVERBOUGHT_THRESH) / RSI_OVERBOUGHT_DIV, center=0.2)
    return 0.0


def factor_sell_underperform_market(ret_etf_5d: float, ret_market_5d: float) -> float:
    if ret_etf_5d < ret_market_5d:
        return sigmoid_normalize((ret_market_5d - ret_etf_5d) / OUTPERFORM_MARKET_DIV, center=0.2)
    return 0.0


def factor_sell_stop_loss_ma_break(price: float, ma20: float) -> float:
    if price < ma20 and ma20 > 0:
        return cap((ma20 - price) / (ma20 * HARD_STOP_MA_BREAK_PCT))
    return 0.0


def factor_sell_trailing_stop_clear(price: float, recent_high: float, atr_pct: float) -> float:
    if recent_high > 0 and atr_pct > 0 and (recent_high - price) / recent_high >= ATR_STOP_MULT * atr_pct:
        return cap((recent_high - price) / recent_high / (ATR_STOP_MULT * atr_pct))
    return 0.0


def factor_sell_trailing_stop_half(price: float, recent_high: float, atr_pct: float) -> float:
    if recent_high > 0 and atr_pct > 0 and (recent_high - price) / recent_high >= ATR_TRAILING_MULT * atr_pct:
        return cap((recent_high - price) / recent_high / (ATR_TRAILING_MULT * atr_pct))
    return 0.0


def factor_sell_downside_momentum(downside: float) -> float:
    return cap(downside)


def factor_sell_max_drawdown_stop(max_drawdown_pct: float) -> float:
    if max_drawdown_pct >= MAX_DRAWDOWN_STOP_DIV:
        return cap(max_drawdown_pct / MAX_DRAWDOWN_STOP_DIV)
    return 0.0


# ========================== 移动止盈信号 ==========================
def get_trailing_profit_signals(price: float, recent_high: float, atr_pct: float) -> str:
    """
    返回移动止盈提醒级别：'clear'（建议清仓）、'half'（建议半仓止盈）或 None
    """
    if factor_sell_trailing_stop_clear(price, recent_high, atr_pct) > 0:
        return "clear"
    if factor_sell_trailing_stop_half(price, recent_high, atr_pct) > 0:
        return "half"
    return None


# ========================== TMSV 复合强度 ==========================
def _get_tmsv_weights(market_status, volatility):
    """根据市场状态和波动率返回 TMSV 子成分的动态权重"""
    from .config import TMSV_HIGH_VOL_THRESH, TMSV_TREND_REDUCE, TMSV_MIN_TREND_WEIGHT

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


def compute_tmsv(df, market_status="震荡偏弱", volatility=0.02):
    """
    计算 TMSV 复合强度指标 (0-100)
    所需 DataFrame 至少包含 close, volume 列，若缺失部分指标会自动计算。
    """
    from .config import (
        TMSV_MA20_WINDOW,
        TMSV_MA60_WINDOW,
        TMSV_ATR_WINDOW,
        TMSV_VOL_MA_WINDOW,
        TMSV_PRICE_DIVISOR,
        TMSV_SLOPE_SCALE,
        TMSV_RSI_SCALE,
        TMSV_MACD_DIFF_EPS,
        TMSV_MACD_CHANGE_SCALE,
        TMSV_VOL_RATIO_BASE,
        TMSV_VOL_RATIO_DIVISOR,
        TMSV_VOL_CONSIST_SCORE,
        TMSV_TREND_MA20_WEIGHT,
        TMSV_TREND_MA60_WEIGHT,
        TMSV_TREND_SLOPE_WEIGHT,
        TMSV_MOM_RSI_WEIGHT,
        TMSV_MOM_MACD_WEIGHT,
        TMSV_VOL_RATIO_WEIGHT,
        TMSV_VOL_CONSIST_WEIGHT,
        TMSV_VOL_LOW_THRESH,
        TMSV_VOL_HIGH_THRESH,
        TMSV_VOL_LOW_FACTOR,
        TMSV_VOL_HIGH_FACTOR,
        TMSV_VOL_MID_FACTOR_BASE,
        TMSV_VOL_MID_FACTOR_SLOPE,
        TMSV_VOL_BAND_WIDTH,
    )

    if df is None or len(df) < 20:
        return pd.Series([50.0] * max(1, len(df))) if len(df) > 0 else pd.Series([50.0])

    df = df.copy()
    if "ma20" not in df.columns:
        df["ma20"] = df["close"].rolling(TMSV_MA20_WINDOW).mean()
    if "ma60" not in df.columns:
        df["ma60"] = df["close"].rolling(TMSV_MA60_WINDOW).mean()
    if "rsi" not in df.columns:
        df["rsi"] = calc_rsi(df["close"])
    if "macd_hist" not in df.columns:
        _, _, df["macd_hist"] = calc_macd(df["close"])
    if "atr" not in df.columns:
        df["atr"] = calculate_atr(df, TMSV_ATR_WINDOW)
    if "vol_ma" not in df.columns:
        df["vol_ma"] = df["volume"].rolling(TMSV_VOL_MA_WINDOW).mean()

    price_above_ma20 = (
        (
            (df["close"] - df["ma20"])
            / (df["ma20"].replace(0, np.nan) * TMSV_PRICE_DIVISOR)
        )
        .clip(0, 1)
        .fillna(0)
    )
    price_above_ma60 = (
        (
            (df["close"] - df["ma60"])
            / (df["ma60"].replace(0, np.nan) * TMSV_PRICE_DIVISOR)
        )
        .clip(0, 1)
        .fillna(0)
    )
    ma20_slope = df["ma20"].diff(5) / df["ma20"].shift(5).replace(0, np.nan)
    slope_score = (ma20_slope * TMSV_SLOPE_SCALE).clip(0, 1).fillna(0)
    trend_score = (
        price_above_ma20 * TMSV_TREND_MA20_WEIGHT
        + price_above_ma60 * TMSV_TREND_MA60_WEIGHT
        + slope_score * TMSV_TREND_SLOPE_WEIGHT
    ) * 100

    rsi_score = ((df["rsi"] - 50) * TMSV_RSI_SCALE).clip(0, 100).fillna(50)
    macd_change = df["macd_hist"].diff() / (
        df["macd_hist"].shift(1).abs() + TMSV_MACD_DIFF_EPS
    )
    macd_score = (macd_change * TMSV_MACD_CHANGE_SCALE).clip(0, 100).fillna(50)
    mom_score = rsi_score * TMSV_MOM_RSI_WEIGHT + macd_score * TMSV_MOM_MACD_WEIGHT

    vol_ratio = df["volume"] / df["vol_ma"].replace(0, np.nan)
    vol_ratio_score = (
        ((vol_ratio - TMSV_VOL_RATIO_BASE) / TMSV_VOL_RATIO_DIVISOR * 100)
        .clip(0, 100)
        .fillna(50)
    )
    price_up = df["close"] > df["close"].shift(1)
    vol_up = df["volume"] > df["vol_ma"]
    consistency = np.where(price_up == vol_up, TMSV_VOL_CONSIST_SCORE, 0)
    vol_score = (
        vol_ratio_score * TMSV_VOL_RATIO_WEIGHT + consistency * TMSV_VOL_CONSIST_WEIGHT
    )

    atr_pct = df["atr"] / df["close"].replace(0, np.nan)
    vol_factor = np.select(
        [atr_pct < TMSV_VOL_LOW_THRESH, atr_pct > TMSV_VOL_HIGH_THRESH],
        [TMSV_VOL_LOW_FACTOR, TMSV_VOL_HIGH_FACTOR],
        default=TMSV_VOL_MID_FACTOR_BASE
        - (atr_pct - TMSV_VOL_LOW_THRESH)
        / TMSV_VOL_BAND_WIDTH
        * TMSV_VOL_MID_FACTOR_SLOPE,
    )
    vol_factor = np.nan_to_num(vol_factor, nan=1.0)

    w = _get_tmsv_weights(market_status, volatility)
    tmsv = (
        trend_score * w["trend"] + mom_score * w["momentum"] + vol_score * w["volume"]
    ) * vol_factor
    return tmsv.clip(0, 100).fillna(50)
