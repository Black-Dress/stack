#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立风险观察模块：基于 ATR 的动态止盈/止损参考，不产生交易信号"""
from typing import Dict, Optional, Union
from .config import ATR_STOP_MULT, ATR_TRAILING_PROFIT_MULT, RISK_ALERT_DISTANCE_ATR


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