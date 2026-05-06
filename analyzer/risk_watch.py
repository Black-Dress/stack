#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立风险观察模块：基于ATR的动态止盈/止损参考，以及基于持仓成本的硬止盈止损研判"""
from typing import Dict, Optional, Union
from .config import (
    ATR_STOP_MULT,
    ATR_TRAILING_PROFIT_MULT,
    RISK_ALERT_DISTANCE_ATR,
    COST_TAKE_PROFIT_CLEAR,
    COST_TAKE_PROFIT_HALF,
    COST_STOP_LOSS_PCT,
)


def generate_risk_alerts(
    price: float,
    recent_high: float,
    recent_low: float,
    atr: float
) -> Dict[str, Optional[Union[float, str]]]:
    """
    返回：
        stop_loss: 动态止损价 (float or None)
        trail_profit: 移动止盈价 (float or None)
        alert: 提醒文本 (str or None)
    """
    if atr <= 0:
        return {"stop_loss": None, "trail_profit": None, "alert": None}

    stop_price = recent_high - ATR_STOP_MULT * atr
    trail_price = recent_high - ATR_TRAILING_PROFIT_MULT * atr
    alert: Optional[str] = None

    if price <= stop_price:
        alert = f"🔴 已触发动态止损（{stop_price:.3f}）"
    elif price - stop_price < RISK_ALERT_DISTANCE_ATR * atr:
        alert = f"🟡 接近动态止损线 {stop_price:.3f}（距 {price - stop_price:.3f}）"
    elif trail_price and price - trail_price < RISK_ALERT_DISTANCE_ATR * atr:
        alert = f"🟢 接近动态止盈线 {trail_price:.3f}（距 {price - trail_price:.3f}）"

    return {"stop_loss": stop_price, "trail_profit": trail_price, "alert": alert}


def evaluate_cost_based_stop_profit(
    price: float,
    cost: float,
    recent_high: float,
    atr: float,
    trailing_profit_level: str,
) -> Dict[str, Optional[str]]:
    """
    基于持仓成本与移动止盈信号，判断是否触发硬性清仓/减仓。
    返回字典：
        - action_override: "SELL" 或 None
        - level_override:   "清仓止盈" / "半仓止盈" / "止损卖出" / None
    """
    if cost is None or cost <= 0:
        return {"action_override": None, "level_override": None}

    profit_pct = (price - cost) / cost

    # 硬止损：亏损达到阈值，无条件卖出
    if profit_pct <= COST_STOP_LOSS_PCT:
        return {"action_override": "SELL", "level_override": "止损卖出"}

    # 止盈：盈利丰厚且移动止盈信号为clear
    if profit_pct >= COST_TAKE_PROFIT_CLEAR and trailing_profit_level == "clear":
        return {"action_override": "SELL", "level_override": "清仓止盈"}

    # 半仓止盈提醒：盈利中等且移动止盈 half
    if profit_pct >= COST_TAKE_PROFIT_HALF and trailing_profit_level in ("half", "clear"):
        return {"action_override": None, "level_override": "半仓止盈"}

    return {"action_override": None, "level_override": None}