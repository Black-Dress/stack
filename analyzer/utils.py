#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""通用工具函数：显示、数学、指标计算、报告格式化"""
import unicodedata
import math
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from .config import *

def get_display_width(text):
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))

def pad_display(text, width, align="left"):
    text = str(text)
    cur = get_display_width(text)
    if cur >= width:
        return text
    pad = width - cur
    if align == "right":
        return " " * pad + text
    elif align == "center":
        left = pad // 2
        return " " * left + text + " " * (pad - left)
    return text + " " * pad

def sigmoid_normalize(x: float, center: float = 0.0, steepness: float = 5.0) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

def nonlinear_score_transform(raw: float, market_status: str,
                              bull_scale: float = 2.5, range_scale: float = 1.5) -> float:
    status_lower = market_status.lower()
    scale = bull_scale if ("牛" in status_lower or "熊" in status_lower) else range_scale
    return math.tanh(scale * raw)

def clip_env_factor(market_factor: float, sentiment_factor: float = 1.0) -> float:
    raw = market_factor * sentiment_factor
    center = 1.0
    scale = 2.0
    mapped = 0.95 + 0.35 * math.tanh(scale * (raw - center))
    return max(0.60, min(1.30, mapped))

def cap(x: float) -> float:
    return max(0.0, min(1.0, x))

def weighted_sum(factors: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(weights.get(k, 0) * factors[k] for k in factors)

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)

def calc_macd(series, fast=12, slow=26, signal=9):
    exp_fast = series.ewm(span=fast, adjust=False).mean()
    exp_slow = series.ewm(span=slow, adjust=False).mean()
    dif = exp_fast - exp_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist

def calculate_atr(df, period=14) -> pd.Series:
    tr = pd.concat([
        df["high"] - df["low"],
        abs(df["high"] - df["close"].shift()),
        abs(df["low"] - df["close"].shift()),
    ], axis=1).max(1)
    return tr.rolling(period).mean()

def calculate_adx(df, period=14) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = calculate_atr(df, 1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx}, index=df.index)

def fallback_market_state(above_ma20: bool, above_ma60: bool) -> Tuple[str, float]:
    if above_ma20 and above_ma60:
        return "正常牛市", 1.2
    if not above_ma20 and not above_ma60:
        return "熊市下跌", 0.8
    return "震荡偏弱", 1.0

def format_etf_output_line(name, code, price, change_pct, final_score, action_level,
                           risk_str="", signal_action=None):
    price_str = f"{price:.3f}" if price is not None else "N/A"
    change_str = f"{change_pct:+.2f}%" if change_pct is not None else "0.00%"
    score_str = f"{final_score:.1f}" if final_score is not None else "0.0"
    out = (f"{pad_display(name, DISPLAY_NAME_WIDTH)} "
           f"{pad_display(code, DISPLAY_CODE_WIDTH)} "
           f"{pad_display(price_str, DISPLAY_PRICE_WIDTH, 'right')} "
           f"{pad_display(change_str, DISPLAY_CHANGE_WIDTH, 'right')} "
           f"{pad_display(score_str, DISPLAY_SCORE_WIDTH, 'right')}  "
           f"{pad_display(action_level, DISPLAY_ACTION_WIDTH)}")
    parts = []
    if risk_str:
        parts.append(risk_str)
    if signal_action:
        parts.append(f"[{signal_action}]")
    if parts:
        out += "  " + " ".join(parts)
    return out

def format_detailed_report(ctx, market, params, action_level, ai_comment):
    lines = []
    lines.append(f"  {ctx.name} ({ctx.code})  评分: {ctx.final_score:.1f}  等级: {action_level}")
    lines.append(f"  价格: {ctx.real_price:.3f}  涨跌: {ctx.change_pct:+.2f}%  ATR: {ctx.atr_pct*100:.2f}%")
    lines.append(f"  市场状态: {market['state']}  环境因子: {market['factor']:.2f}")
    
    # ---- 持仓成本信息（如果有） ----
    if ctx.cost_price is not None and ctx.cost_price > 0:
        profit_pct = ctx.cost_profit_pct
        cost_str = f"持仓成本: {ctx.cost_price:.3f}"
        if profit_pct is not None:
            cost_str += f"  浮动盈亏: {profit_pct:+.2%}"
        lines.append(f"  💰 {cost_str}")

    # ---- 止盈/止损状态（只显示有效级别） ----
    if ctx.trailing_profit_level and ctx.trailing_profit_level != "None":
        fall = (ctx.recent_high_price - ctx.real_price) / ctx.recent_high_price if ctx.recent_high_price > 0 else 0
        label = ctx.trailing_profit_level
        if label == "clear":
            label = "清仓"
        elif label == "half":
            label = "半仓"
        lines.append(f"  ⚠️ 移动止盈: {label}级，高点回落{fall:.1%}")

    if ctx.profit_pct_from_low >= 0.12:
        if ctx.profit_level == 'clear':
            lines.append(f"  ⛔ 低点涨幅止盈: {ctx.profit_pct_from_low:.1%} (清仓级)")
        elif ctx.profit_level == 'half':
            lines.append(f"  ⚠️ 低点涨幅止盈: {ctx.profit_pct_from_low:.1%} (半仓级)")
        else:
            lines.append(f"  止盈关注: 低点涨幅{ctx.profit_pct_from_low:.1%}")

    # ---- AI 评论 ----
    if ai_comment:
        lines.append(f"  💬 AI点评: {ai_comment}")
    return "\n".join(lines)