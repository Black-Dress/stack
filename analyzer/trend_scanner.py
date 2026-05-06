#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""独立趋势扫描模块：基于价格形态识别，不依赖评分或权重。输出标签为 [趋势扫描]"""
import re
from typing import List, Tuple, Optional


def _extract_pct(out: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, out)
    return float(m.group(1)) if m else None


def select_trend_buy(results, max_count=3, low_profit_min=5.0, low_profit_max=15.0,
                     max_pullback=5.0, daily_gain_min=0.5, daily_gain_max=6.0,
                     prefer_signal=True) -> List[str]:
    candidates = []
    for out, _ in results:
        if any(x in out for x in ["弱于中期均线", "强烈卖出", "连续3日低评分", "清仓级"]):
            continue
        low_pct = _extract_pct(out, r"低点涨(\d+\.?\d*)%")
        if low_pct is None or low_pct < low_profit_min or low_pct > low_profit_max:
            continue
        pullback = _extract_pct(out, r"高点回落(\d+\.?\d*)%")
        if pullback is not None and pullback > max_pullback:
            continue
        change_pct = _extract_pct(out, r"([+-]\d+\.?\d*)%")
        if change_pct is None or change_pct < daily_gain_min or change_pct > daily_gain_max:
            continue
        has_buy = "[BUY]" in out
        profit_score = 1.0 - abs(low_pct - 10) / 10
        sort_key = (not has_buy, -(10.0 if has_buy else 0.0) - profit_score)
        candidates.append((sort_key, out))
    candidates.sort(key=lambda x: x[0])
    return [out for _, out in candidates[:max_count]]


def select_trend_sell(results, max_count=3, min_daily_loss=-3.0, min_pullback=6.0,
                      min_low_profit=18.0, include_weak_ma=True, include_clear_stop=True) -> List[str]:
    def risk_score(out):
        risk = 0
        if "[SELL]" in out and "连续3日低评分" in out:
            risk = 100
        elif "[SELL]" in out:
            risk = 90
        elif "清仓级" in out:
            risk = 80
        elif "弱于中期均线" in out:
            pullback = _extract_pct(out, r"高点回落(\d+\.?\d*)%")
            change = _extract_pct(out, r"([+-]\d+\.?\d*)%")
            if (pullback and pullback > 8) or (change and change < -2):
                risk = 75
            else:
                risk = 60
        elif "强烈卖出" in out:
            risk = 50
        else:
            pullback = _extract_pct(out, r"高点回落(\d+\.?\d*)%")
            change = _extract_pct(out, r"([+-]\d+\.?\d*)%")
            if pullback and pullback > 10:
                risk = 70
            elif change and change < -3:
                risk = 65
            elif pullback and pullback > 6:
                risk = 40
            elif change and change < -1:
                risk = 30
            else:
                risk = 10
        return risk

    candidates = []
    for out, _ in results:
        cond = False
        change = _extract_pct(out, r"([+-]\d+\.?\d*)%")
        if change is not None and change < min_daily_loss:
            cond = True
        pullback = _extract_pct(out, r"高点回落(\d+\.?\d*)%")
        if pullback is not None and pullback > min_pullback:
            cond = True
        low_pct = _extract_pct(out, r"低点涨(\d+\.?\d*)%")
        if low_pct is not None and low_pct > min_low_profit and pullback is not None and pullback > 5:
            cond = True
        if include_weak_ma and "弱于中期均线" in out:
            cond = True
        if include_clear_stop and "清仓级" in out:
            cond = True
        if "强烈卖出" in out or "连续3日低评分" in out:
            cond = True
        if "[SELL]" in out:
            cond = True
        if cond:
            candidates.append((out, risk_score(out)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [out for out, _ in candidates[:max_count]]