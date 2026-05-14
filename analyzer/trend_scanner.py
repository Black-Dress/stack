#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立趋势扫描模块：基于结构化数据识别形态，不依赖评分或权重"""
from typing import List, Dict, Optional


def select_trend_buy(
    scan_list: List[Dict],
    max_count: int = 3,
    low_profit_min: float = 0.05,
    low_profit_max: float = 0.25,   
    max_pullback: float = 0.05,
    daily_gain_min: float = 0.005,
    daily_gain_max: float = 0.06,
    prefer_signal: bool = True,
) -> List[int]:
    candidates = []
    for idx, s in enumerate(scan_list):
        if s.get("has_strong_sell_text") or s.get("has_clear_stop_text"):
            continue
        if s.get("has_sell_signal"):
            continue

        low_pct = s.get("profit_pct_from_low")
        if low_pct is None or low_pct < low_profit_min or low_pct > low_profit_max:
            continue

        pullback = s.get("max_drawdown_pct")
        if pullback is not None and pullback > max_pullback:
            continue

        change_pct = s.get("change_pct")
        if change_pct is None:
            continue
        if abs(change_pct) < 1e-6:
            continue
        if change_pct < daily_gain_min or change_pct > daily_gain_max:
            continue

        has_buy = s.get("has_buy_signal", False)
        profit_score = 1.0 - abs(low_pct - 0.10) / 0.10
        sort_key = (not has_buy, -(10.0 if has_buy else 0.0) - profit_score)
        candidates.append((sort_key, idx))

    candidates.sort(key=lambda x: x[0])
    return [idx for _, idx in candidates[:max_count]]


def select_trend_sell(
    scan_list: List[Dict],
    max_count: int = 3,
    min_daily_loss: float = -0.03,
    min_pullback: float = 0.06,
    min_low_profit: float = 0.18,
    include_weak_ma: bool = True,
    include_clear_stop: bool = True,
) -> List[int]:
    def risk_score(s: Dict) -> int:
        risk = 0
        if s.get("has_sell_signal") and s.get("has_strong_sell_text"):
            risk = 100
        elif s.get("has_sell_signal"):
            risk = 90
        elif s.get("has_clear_stop_text"):
            risk = 80
        elif s.get("has_weak_ma_text"):
            pullback = s.get("max_drawdown_pct")
            change = s.get("change_pct")
            if (pullback and pullback > 0.08) or (change and change < -0.02):
                risk = 75
            else:
                risk = 60
        elif s.get("has_strong_sell_text"):
            risk = 50
        else:
            pullback = s.get("max_drawdown_pct")
            change = s.get("change_pct")
            if pullback and pullback > 0.10:
                risk = 70
            elif change and change < -0.03:
                risk = 65
            elif pullback and pullback > 0.06:
                risk = 40
            elif change and change < -0.01:
                risk = 30
            else:
                risk = 10
        return risk

    candidates = []
    for idx, s in enumerate(scan_list):
        cond = False
        change = s.get("change_pct")
        if change is not None and change < min_daily_loss:
            cond = True
        pullback = s.get("max_drawdown_pct")
        if pullback is not None and pullback > min_pullback:
            cond = True
        low_pct = s.get("profit_pct_from_low")
        if low_pct is not None and low_pct > min_low_profit and pullback is not None and pullback > 0.05:
            cond = True
        if include_weak_ma and s.get("has_weak_ma_text"):
            cond = True
        if include_clear_stop and s.get("has_clear_stop_text"):
            cond = True
        if s.get("has_strong_sell_text"):
            cond = True
        if s.get("has_sell_signal"):
            cond = True
        if cond:
            candidates.append((idx, risk_score(s)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, _ in candidates[:max_count]]

def select_left_buy(
    scan_list: List[Dict],
    max_count: int = 3,
    daily_gain_min: float = -0.02,
    daily_gain_max: float = 0.03,
    low_profit_min: float = 0.0,
    low_profit_max: float = 0.08,
    max_pullback: float = 0.05,
    min_score: float = 55,
    rsi_max: float = 50,
    require_below_ma: bool = True,
) -> List[int]:
    candidates = []
    for idx, s in enumerate(scan_list):
        if s.get("has_strong_sell_text") or s.get("has_clear_stop_text"):
            continue
        if s.get("has_sell_signal"):
            continue

        score = s.get("final_score", 0)
        if score < min_score:
            continue

        change = s.get("change_pct")
        if change is None or change < daily_gain_min or change > daily_gain_max:
            continue

        low_pct = s.get("profit_pct_from_low")
        if low_pct is None or low_pct < low_profit_min or low_pct > low_profit_max:
            continue

        pullback = s.get("max_drawdown_pct")
        if pullback is not None and pullback > max_pullback:
            continue

        rsi = s.get("rsi")
        if rsi is not None and rsi > rsi_max:
            continue

        if require_below_ma and not s.get("has_weak_ma_text"):
            continue

        has_buy = s.get("has_buy_signal", False)
        sort_key = (not has_buy, low_pct if low_pct else 999)
        candidates.append((sort_key, idx))

    candidates.sort(key=lambda x: x[0])
    return [idx for _, idx in candidates[:max_count]]