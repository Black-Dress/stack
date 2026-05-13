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
        # 如果内容过长，截断并添加省略号
        if cur > width:
            # 尝试截断
            truncated = ""
            for ch in text:
                if get_display_width(truncated + ch) <= width - 2:
                    truncated += ch
                else:
                    break
            text = truncated + ".."
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

def cap(x: float) -> float:
    return max(0.0, min(1.0, x))

def resolve_real_price(real_price: Optional[float], hist_df: Optional[pd.DataFrame]) -> Tuple[Optional[float], bool]:
    """如果实时价不可用，则使用前一日收盘价回退。"""
    if real_price is not None:
        return real_price, False
    if hist_df is not None and not hist_df.empty:
        return hist_df.iloc[-1]["close"], True
    return None, False

def safe_ratio(numerator: float, denominator: float, default: Optional[float] = 1.0) -> Optional[float]:
    return numerator / denominator if denominator else default

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

def print_unified_table(rows, title=None, env=None, today_str=None, table_type="main", print_header=True):
    """
    统一表格打印函数
    table_type: "main" - 主表格（特征标签）, "position" - 持仓表格, "trend" - 趋势扫描综合表格
    print_header: 是否打印表头
    """
    # 打印标题或报告头部
    if title:
        print(f"\n{'='*90}")
        print(f"  {title}")
        print(f"{'='*90}")
    elif env is not None and today_str is not None:
        print(f"\n{'='*90}")
        print(f"  ETF 分析报告 - {today_str}  市场状态: {env['state']}  环境因子: {env['factor']:.2f}")
        if env.get("risk_tip"):
            print(f"  {env['risk_tip']}")
        print(f"{'='*90}")
    # 如果都没有，则不打印任何头部

    if not rows:
        print("  无数据")
        return

    # 定义列宽和对齐方式
    if table_type == "main":
        cols = [
            ("名称", "name", DISPLAY_NAME_WIDTH, "left"),
            ("代码", "code", DISPLAY_CODE_WIDTH, "left"),
            ("价格", "price", DISPLAY_PRICE_WIDTH, "left"),
            ("涨跌", "change_pct", DISPLAY_CHANGE_WIDTH, "left"),
            ("评分", "final_score", DISPLAY_SCORE_WIDTH, "left"),
            ("特征标签", "risk_str", DISPLAY_TAGS_WIDTH, "left")
        ]
    elif table_type == "position":
        cols = [
            ("名称", "name", DISPLAY_NAME_WIDTH, "left"),
            ("代码", "code", DISPLAY_CODE_WIDTH, "left"),
            ("份额", "shares", DISPLAY_PRICE_WIDTH, "left"),
            ("成本", "cost", DISPLAY_PRICE_WIDTH, "left"),
            ("现价", "price", DISPLAY_PRICE_WIDTH, "left"),
            ("盈亏%", "profit_pct", DISPLAY_PRICE_WIDTH, "left"),
            ("变化", "change", DISPLAY_PRICE_WIDTH, "left"),
            ("评分", "score", DISPLAY_SCORE_WIDTH, "left"),
            ("建议", "advice", DISPLAY_TAGS_WIDTH, "left")
        ]
    elif table_type == "trend":
        cols = [
            ("名称", "name", DISPLAY_NAME_WIDTH, "left"),
            ("代码", "code", DISPLAY_CODE_WIDTH, "left"),
            ("价格", "price", DISPLAY_PRICE_WIDTH, "left"),
            ("涨跌", "change_pct", DISPLAY_CHANGE_WIDTH, "left"),
            ("评分", "final_score", DISPLAY_SCORE_WIDTH, "left"),
            ("建议", "advice", DISPLAY_TAGS_WIDTH, "left")
        ]
    else:
        raise ValueError(f"未知表格类型: {table_type}")

    # 打印表头（根据列类型决定对齐方式）
    if print_header:
        header_parts = []
        for col_name, col_key, width, align in cols:
            header_parts.append(pad_display(col_name, width, align))
        header_line = " ".join(header_parts)
        print(header_line)
        # 计算总宽度（包括列之间的空格）
        total_width = len(header_line)
        print("-" * total_width)

    # 打印数据行
    for row in rows:
        row_parts = []
        for col_name, col_key, width, align in cols:
            val = row.get(col_key, "")
            if col_key in ("price", "cost"):
                if isinstance(val, (int, float)):
                    val_str = f"{val:.3f}"
                else:
                    val_str = str(val)
            elif col_key == "change_pct":
                if isinstance(val, (int, float)):
                    val_str = f"{val:+.2f}%"
                else:
                    val_str = str(val)
            elif col_key in ("final_score", "score"):
                if isinstance(val, (int, float)):
                    val_str = f"{val:.1f}"
                else:
                    val_str = str(val)
            elif col_key == "profit_pct":
                if isinstance(val, (int, float)):
                    val_str = f"{val:+.1f}%"
                else:
                    val_str = str(val)
            elif col_key == "shares":
                if isinstance(val, (int, float)):
                    val_str = f"{int(val)}"
                else:
                    val_str = str(val)
            else:
                val_str = str(val)
            row_parts.append(pad_display(val_str, width, align))
        print(" ".join(row_parts))




def format_detailed_report(ctx, market, params, action_level, ai_comment, signal_action=None):
    lines = []
    lines.append(f"  {ctx.name} ({ctx.code})  评分: {ctx.final_score:.1f}  等级: {action_level}")
    if signal_action:
        lines.append(f"  信号: {signal_action}")
    lines.append(f"  价格: {ctx.real_price:.3f}  涨跌: {ctx.change_pct:+.2f}%  ATR: {ctx.atr_pct*100:.2f}%")
    lines.append(f"  市场状态: {market['state']}  环境因子: {market['factor']:.2f}")

    if ctx.cost_price is not None and ctx.cost_price > 0:
        profit_pct = ctx.cost_profit_pct
        cost_str = f"持仓成本: {ctx.cost_price:.3f}"
        if profit_pct is not None:
            cost_str += f"  浮动盈亏: {profit_pct:+.2%}"
        lines.append(f"  💰 {cost_str}")

    indicators = _format_indicators(ctx)
    if indicators:
        lines.append("  " + "─" * 50)
        lines.append("  【技术指标】")
        lines.append("  " + indicators)

    factor_detail = _format_factor_details(ctx)
    if factor_detail:
        lines.append("  " + "─" * 50)
        lines.append("  【因子评分】")
        lines.append("  " + factor_detail)

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

    if ai_comment:
        lines.append("  " + "─" * 50)
        lines.append("  💬 AI短评:")
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

def _format_indicators(ctx) -> str:
    if ctx.hist_df is None or ctx.hist_df.empty:
        return ""
    d = ctx.hist_df.iloc[-1]
    price = ctx.real_price
    lines = []

    ma20 = d.get("ma_short", None)
    ma20_val = f"{ma20:.3f}" if pd.notna(ma20) else "N/A"
    if price and ma20 and ma20 > 0:
        dev_pct = (price - ma20) / ma20 * 100
        dev_str = f"{dev_pct:+.2f}%"
    else:
        dev_str = "N/A"
    lines.append(f"MA20: {ma20_val}  偏离: {dev_str}")

    rsi_val = d.get("rsi", None)
    rsi_str = f"{rsi_val:.1f}" if pd.notna(rsi_val) else "N/A"
    tmsv_str = f"{ctx.tmsv:.1f}" if ctx.tmsv is not None else "N/A"
    lines.append(f"RSI: {rsi_str}  TMSV: {tmsv_str}")

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

    boll_up = d.get("boll_up", None)
    boll_low = d.get("boll_low", None)
    boll_up_str = f"{boll_up:.3f}" if pd.notna(boll_up) else "N/A"
    boll_low_str = f"{boll_low:.3f}" if pd.notna(boll_low) else "N/A"
    lines.append(f"布林上轨: {boll_up_str}  下轨: {boll_low_str}")

    if ctx.weekly_above:
        weekly_str = "周线多头"
    elif ctx.weekly_below:
        weekly_str = "周线空头"
    else:
        weekly_str = "周线不明"
    lines.append(f"周线: {weekly_str}")

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

def _format_factor_details(ctx) -> str:
    parts = []
    if ctx.buy_factors:
        buy_items = [f"{k}={v:.2f}" for k, v in ctx.buy_factors.items()]
        parts.append("买入因子: " + ", ".join(buy_items[:8]) + ("..." if len(buy_items) > 8 else ""))
    if ctx.sell_factors:
        sell_items = [f"{k}={v:.2f}" for k, v in ctx.sell_factors.items()]
        parts.append("卖出因子: " + ", ".join(sell_items[:8]) + ("..." if len(sell_items) > 8 else ""))
    parts.append(f"买入总分: {ctx.buy_score:.3f}  卖出总分: {ctx.sell_score:.3f}  原始差值: {ctx.raw_score:.3f}")
    return "\n  ".join(parts)