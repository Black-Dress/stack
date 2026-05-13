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
           f"{pad_display(action_level, DISPLAY_LEVEL_WIDTH)}")
    parts = []
    if risk_str:
        parts.append(risk_str)
    if signal_action:
        parts.append(f"[{signal_action}]")
    if parts:
        out += "  " + " ".join(parts)
    return out

def format_detailed_report(ctx, market, params, action_level, ai_comment, signal_action=None):
    lines = []
    # 头部
    lines.append(f"  {ctx.name} ({ctx.code})  评分: {ctx.final_score:.1f}  等级: {action_level}")
    if signal_action:
        lines.append(f"  信号: {signal_action}")
    lines.append(f"  价格: {ctx.real_price:.3f}  涨跌: {ctx.change_pct:+.2f}%  ATR: {ctx.atr_pct*100:.2f}%")
    lines.append(f"  市场状态: {market['state']}  环境因子: {market['factor']:.2f}")

    # 持仓成本
    if ctx.cost_price is not None and ctx.cost_price > 0:
        profit_pct = ctx.cost_profit_pct
        cost_str = f"持仓成本: {ctx.cost_price:.3f}"
        if profit_pct is not None:
            cost_str += f"  浮动盈亏: {profit_pct:+.2%}"
        lines.append(f"  💰 {cost_str}")

    # 技术指标板块
    indicators = format_indicators(ctx)
    if indicators:
        lines.append("  " + "─" * 50)
        lines.append("  【技术指标】")
        lines.append("  " + indicators)

    # 因子明细
    factor_detail = format_factor_details(ctx)
    if factor_detail:
        lines.append("  " + "─" * 50)
        lines.append("  【因子评分】")
        lines.append("  " + factor_detail)

    # 止盈/止损状态
    if ctx.trailing_profit_level and ctx.trailing_profit_level != "None":
        fall = (ctx.recent_high_price - ctx.real_price) / ctx.recent_high_price if ctx.recent_high_price > 0 else 0
        label = ctx.trailing_profit_level
        if label == "clear":
            label = "清仓"
        elif label == "half":
            label = "半仓"
        lines.append(f"  💸 移动止盈: {label}级，高点回落{fall:.1%}")

    if ctx.profit_pct_from_low >= 0.12:
        if ctx.profit_level == 'clear':
            lines.append(f"  ⛔ 低点涨幅止盈: {ctx.profit_pct_from_low:.1%} (清仓级)")
        elif ctx.profit_level == 'half':
            lines.append(f"  💸 低点涨幅止盈: {ctx.profit_pct_from_low:.1%} (半仓级)")
        else:
            lines.append(f"  🤭 止盈关注: 低点涨幅{ctx.profit_pct_from_low:.1%}")

    # AI 评论
    if ai_comment:
        lines.append("  " + "─" * 50)
        lines.append("  💬 AI短评:")
        # 自动换行美化
        max_width = 70
        comment = ai_comment.strip()
        while len(comment) > max_width:
            split_at = comment.rfind(' ', 0, max_width)
            if split_at == -1:
                split_at = max_width
            lines.append(f"     {comment[:split_at]}")
            comment = comment[split_at:].lstrip()
        if comment:
            lines.append(f"     {comment}")

    return "\n".join(lines)




def format_indicators(ctx) -> str:
    """生成 ETF 技术指标摘要，用于详细报告"""
    if ctx.hist_df is None or ctx.hist_df.empty:
        return ""
    d = ctx.hist_df.iloc[-1]
    price = ctx.real_price
    lines = []  # 每行一个字符串

    # 均线及偏离
    ma20 = d.get("ma_short", None)
    ma30 = d.get("ma30", None)
    ma20_val = f"{ma20:.3f}" if pd.notna(ma20) else "N/A"
    if price and ma20 and ma20 > 0:
        dev_pct = (price - ma20) / ma20 * 100
        dev_str = f"{dev_pct:+.2f}%"
    else:
        dev_str = "N/A"
    lines.append(f"MA20: {ma20_val}  偏离: {dev_str}")

    # RSI / TMSV
    rsi_val = d.get("rsi", None)
    rsi_str = f"{rsi_val:.1f}" if pd.notna(rsi_val) else "N/A"
    tmsv_str = f"{ctx.tmsv:.1f}" if ctx.tmsv is not None else "N/A"
    lines.append(f"RSI: {rsi_str}  TMSV: {tmsv_str}")

    # MACD 状态、成交量比
    macd_dif = d.get("macd_dif", None)
    macd_dea = d.get("macd_dea", None)
    if pd.notna(macd_dif) and pd.notna(macd_dea):
        macd_status = "金叉" if macd_dif > macd_dea else "死叉"
    else:
        macd_status = "N/A"
    vol = d.get("volume", None)
    vol_ma = d.get("vol_ma", None)
    if vol and vol_ma and vol_ma > 0:
        vol_ratio = vol / vol_ma
        vol_str = f"{vol_ratio:.2f}"
    else:
        vol_str = "N/A"
    lines.append(f"MACD: {macd_status}  成交量比: {vol_str}")

    # 布林带
    boll_up = d.get("boll_up", None)
    boll_low = d.get("boll_low", None)
    boll_up_str = f"{boll_up:.3f}" if pd.notna(boll_up) else "N/A"
    boll_low_str = f"{boll_low:.3f}" if pd.notna(boll_low) else "N/A"
    lines.append(f"布林上轨: {boll_up_str}  下轨: {boll_low_str}")

    # 周线状态
    if ctx.weekly_above:
        weekly_str = "周线多头"
    elif ctx.weekly_below:
        weekly_str = "周线空头"
    else:
        weekly_str = "周线不明"
    lines.append(f"周线: {weekly_str}")

    # 近期高低点及回撤
    rh = ctx.recent_high_price
    rl = ctx.recent_low_price
    rh_str = f"{rh:.3f}" if rh > 0 else "N/A"
    rl_str = f"{rl:.3f}" if rl > 0 else "N/A"
    dd = ctx.max_drawdown_pct
    dd_str = f"{dd*100:.2f}%" if dd else "N/A"
    low_profit = ctx.profit_pct_from_low
    lp_str = f"{low_profit*100:.2f}%" if low_profit is not None else "N/A"
    lines.append(f"近期高点: {rh_str}  低点: {rl_str}  最大回撤: {dd_str}  低点涨幅: {lp_str}")

    return "\n  ".join(lines)


def format_factor_details(ctx) -> str:
    """生成买入/卖出因子明细"""
    parts = []
    # 买入因子
    if ctx.buy_factors:
        buy_items = [f"{k}={v:.2f}" for k, v in ctx.buy_factors.items()]
        parts.append("买入因子: " + ", ".join(buy_items[:8]) + ("..." if len(buy_items) > 8 else ""))
    # 卖出因子
    if ctx.sell_factors:
        sell_items = [f"{k}={v:.2f}" for k, v in ctx.sell_factors.items()]
        parts.append("卖出因子: " + ", ".join(sell_items[:8]) + ("..." if len(sell_items) > 8 else ""))
    # 评分合成
    parts.append(f"买入总分: {ctx.buy_score:.3f}  卖出总分: {ctx.sell_score:.3f}  原始差值: {ctx.raw_score:.3f}")
    return "\n  ".join(parts)
