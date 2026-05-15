#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""事件检测与仓位计算模块"""
import datetime
from typing import Dict, Any, Optional, Tuple
from .config import *


def detect_events(
    code: str,
    data: Dict[str, Any],
    market_state: str,
    state: Dict[str, Any],
) -> Tuple[str, str, bool]:
    """
    事件检测主函数。
    返回 (event_type, advice_text, need_update_state)
    """
    # 读取状态计数器
    trend_add_count = state.get("trend_add_count", 0)
    dip_add_count = state.get("dip_add_count", 0)
    overheat_triggered = state.get("overheat_triggered", False)
    overheat_count = state.get("overheat_count", 0)

    # 从 data 中提取指标
    price = data.get("price", 0.0)
    score = data.get("final_score", 50.0)
    rsi = data.get("rsi", 50.0)
    vol_ratio = data.get("vol_ratio", 1.0)
    profit_pct_from_low = data.get("profit_pct_from_low", 0.0)
    atr_pct = data.get("atr_pct", 0.02)
    ma5 = data.get("ma5")
    ma10 = data.get("ma10")
    ma20 = data.get("ma20")
    macd_hist = data.get("macd_hist", 0.0)
    # MACD 连续缩短需要历史值，这里简单处理：使用 state 中存储的 last_macd_hist
    last_macd_hist = state.get("last_macd_hist", 0.0)
    # 计算今日是否缩短
    macd_shrinking = (macd_hist > last_macd_hist)
    # 连续天数计数器
    macd_shrink_days = state.get("macd_shrink_days", 0)
    if macd_shrinking:
        macd_shrink_days += 1
    else:
        macd_shrink_days = 0
    # 更新 state 中的 macd 历史值（在调用前需要保存）
    # 注意：state 会在外部更新，此处仅读取

    # 回撤计算
    recent_high = data.get("recent_high_10", price)
    pullback = (recent_high - price) / recent_high if recent_high > 0 else 0.0

    # 大盘过滤
    if market_state in ("弱熊", "强熊"):
        if data.get("shares", 0) > 0:
            base_shares = int(BASE_POSITION_RATIO * TOTAL_CAPITAL / price / 100) * 100
            current_shares = data.get("shares", 0)
            if current_shares <= base_shares:
                return ("hold", "持有底仓", False)
            else:
                return ("sell_confirm", f"减仓至底仓 | 价位：{price:.3f}附近（大盘走弱）", True)
        else:
            return ("none", "", False)

    # 已持仓
    if data.get("shares", 0) > 0:
        # 1. 清仓（动态止损）
        recent_high_long = data.get("recent_high_20", price)
        stop_price = recent_high_long - ATR_STOP_MULT * atr_pct * price
        if price <= stop_price:
            return ("clear", f"清仓 | 价位：{price:.3f}附近（止损）", True)

        # 2. 过热止盈
        overheat_conditions = 0
        if rsi > OVERHEAT_RSI_THRESHOLD:
            overheat_conditions += 1
        if ma20 and (price / ma20 - 1) > OVERHEAT_MA20_DEVIATION:
            overheat_conditions += 1
        if vol_ratio > OVERHEAT_VOL_RATIO:
            overheat_conditions += 1
        if profit_pct_from_low > OVERHEAT_3DAY_GAIN:
            overheat_conditions += 1
        cost_profit = data.get("cost_profit_pct")
        if cost_profit is not None and cost_profit > OVERHEAT_PROFIT_PCT:
            overheat_conditions += 1
        if overheat_conditions >= OVERHEAT_MIN_CONDITIONS:
            if not overheat_triggered:
                return ("overheat", f"止盈{int(OVERHEAT_SELL_PCT*100)}% | 价位：{price:.3f}附近（过热）", True)
            elif overheat_count == 1:
                return ("overheat", f"再次止盈{int(OVERHEAT_SELL_PCT2*100)}% | 价位：{price:.3f}附近（持续过热）", True)

        # 3. 确认减仓（趋势破位）
        if ma20 and price < ma20 and (data.get("ma20_trend", 0) <= 0):
            base_shares = int(BASE_POSITION_RATIO * TOTAL_CAPITAL / price / 100) * 100
            current_shares = data.get("shares", 0)
            if current_shares <= base_shares:
                return ("hold", "持有底仓", False)
            else:
                return ("sell_confirm", f"减仓至底仓 | 价位：{price:.3f}附近（趋势破位）", True)

        # 4. 初步减仓
        ma10_trend = data.get("ma10_trend", 0)
        if (ma10 and price < ma10 and ma10_trend <= 0) or (pullback >= SELL_PRELIMINARY_PULLBACK):
            reduce_pct = SELL_PRELIMINARY_REDUCE_PCT
            return ("sell_prelim", f"减仓{int(reduce_pct*100)}% | 价位：{price:.3f}附近（趋势转弱）", True)

        # 5. 深度回撤补仓
        dip_threshold = BULL_MARKET_DIP_THRESHOLD if market_state == "强牛" else DIP_THRESHOLD
        max_dip_value = MAX_ADD_RATIO * TOTAL_CAPITAL
        current_dip_value = dip_add_count * ADD_BASE_PCT * BASE_POSITION_RATIO * TOTAL_CAPITAL
        if pullback >= dip_threshold and rsi <= DIP_RSI_THRESHOLD and current_dip_value < max_dip_value:
            add_pct = ADD_BASE_PCT * 100
            return ("dip", f"补仓{int(add_pct)}% | 价位：{price:.3f}附近（超卖反弹）", True)

        # 6. 确认加仓
        if trend_add_count == 1 and ma10 and ma20 and ma10 > ma20 and score >= CONFIRN_SCORE_MIN:
            add_pct = TREND_ADD_PCT * 100
            return ("trend_confirm", f"加仓{int(add_pct)}% | 价位：{price:.3f}附近（趋势确认）", True)

        # 7. 初步加仓
        if (ma10 and price > ma10) and vol_ratio > REVERSAL_VOL_RATIO and rsi > REVERSAL_RSI_MIN and trend_add_count < 2:
            # 还需要MACD柱连续两天缩短或由负转正
            # 使用 macd_shrink_days 计数
            if macd_shrink_days >= REVERSAL_MACD_HIST_CONSECUTIVE or (macd_hist > 0 and last_macd_hist <= 0):
                add_pct = TREND_ADD_PCT * 100
                return ("trend_reversal", f"加仓{int(add_pct)}% | 价位：{price:.3f}附近（趋势启动）", True)

        # 默认持有
        pos_state = state.get("position_state", "BASE_HOLD")
        if pos_state == "BASE_HOLD" and data.get("shares", 0) > 0:
            return ("hold", "持有底仓，等待信号", False)
        else:
            return ("hold", "持有不动", False)

    else:
        # 未持仓：仅检测买入信号
        if market_state in ("弱熊", "强熊", "震荡"):
            return ("none", "", False)
        # 需要连续2日评分 >=65
        score_history = state.get("score_history", [])
        if len(score_history) < 2:
            return ("none", "", False)
        today_str = datetime.date.today().strftime("%Y-%m-%d")
        yesterday_score = None
        for rec in score_history:
            if rec["date"] != today_str:
                yesterday_score = rec["score"]
                break
        if not (score >= 65 and yesterday_score and yesterday_score >= 65):
            return ("none", "", False)

        # 买入条件同初步加仓（去掉持仓限制）
        if (ma10 and price > ma10) and vol_ratio > REVERSAL_VOL_RATIO and rsi > REVERSAL_RSI_MIN:
            if macd_shrink_days >= REVERSAL_MACD_HIST_CONSECUTIVE or (macd_hist > 0 and last_macd_hist <= 0):
                return ("buy", f"买入底仓 | 价位：{price:.3f}附近（趋势启动）", True)
        return ("none", "", False)


def calculate_position(event: str, current_shares: int, price: float, 
                       base_ratio: float = BASE_POSITION_RATIO) -> Tuple[int, str]:
    """
    根据事件类型和当前持仓计算具体操作数量。
    返回 (操作份额, 建议文本补充) 等。
    简化起见，直接在事件检测时已生成建议文本，此函数保留作为扩展。
    """
    # 本函数在新架构中已由事件检测直接返回建议文本，无需额外实现。
    pass