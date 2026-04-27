#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
因子强度计算模块：包含所有买入/卖出因子强度函数、移动止盈信号，
以及因子计算所需的专用常量。
"""
import math
from .utils import sigmoid_normalize, cap

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