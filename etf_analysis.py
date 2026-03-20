# etf_analysis.py
# 单个ETF分析逻辑

import numpy as np
from config import (
    STRATEGY_WEIGHTS, BUY_THRESHOLD, SELL_THRESHOLD, CONFIRM_DAYS,
    TRAILING_STOP_HALF, TRAILING_STOP_CLEAR, PROFIT_TARGETS,
    RECENT_HIGH_WINDOW, RECENT_LOW_WINDOW
)

def calculate_score(
    real_price, ma20, volume, vol_ma, macd_golden, kdj_golden, rsi,
    boll_up, boll_low, williams_r,
    market_above_ma20, market_above_ma60, market_amount_above_ma20,
    ret_etf_5d, ret_market_5d,
    break_ma, trailing_half, trailing_clear, profit_hit,
    weights=None
):
    """计算基础评分，支持动态权重"""
    if weights is None:
        weights = STRATEGY_WEIGHTS
    score = 0.0

    # ETF自身指标
    if real_price > ma20:
        score += weights.get("price_above_ma20", 0)
    if volume > vol_ma:
        score += weights.get("volume_above_ma5", 0)
    if macd_golden:
        score += weights.get("macd_golden_cross", 0)
    if kdj_golden:
        score += weights.get("kdj_golden_cross", 0)
    if real_price > boll_up:
        score += weights.get("bollinger_break_up", 0)
    if williams_r > 80:
        score += weights.get("williams_oversold", 0)
    if real_price < ma20:
        score += weights.get("price_below_ma20", 0)
    if real_price < boll_low:
        score += weights.get("bollinger_break_down", 0)
    if williams_r < 20:
        score += weights.get("williams_overbought", 0)
    if rsi > 70:
        score += weights.get("rsi_overbought", 0)

    # 大盘指标
    if market_above_ma20:
        score += weights.get("market_above_ma20", 0)
    if market_above_ma60:
        score += weights.get("market_above_ma60", 0)
    if market_amount_above_ma20:
        score += weights.get("market_amount_above_ma20", 0)
    if ret_etf_5d > ret_market_5d:
        score += weights.get("outperform_market", 0)
    else:
        score += weights.get("underperform_market", 0)

    # 止盈止损条件
    if break_ma:
        score += weights.get("stop_loss_ma_break", 0)
    if trailing_clear:
        score += weights.get("trailing_stop_clear", 0)
    elif trailing_half:
        score += weights.get("trailing_stop_half", 0)
    if profit_hit:
        score += weights.get("profit_target_hit", 0)

    return score

def map_score_to_position(score):
    if score >= 1.0:
        return 1.0
    elif score >= 0.8:
        return 0.8
    elif score >= 0.6:
        return 0.6
    elif score >= 0.4:
        return 0.4
    elif score >= 0.2:
        return 0.2
    else:
        return 0.0

def analyze_etf_signal(
    code, name, real_price, hist_df,
    macro_status, market_factor, sentiment_factor,
    market_above_ma20, market_above_ma60, market_amount_above_ma20,
    ret_market_5d, today, state, weights=None
):
    """
    分析单个ETF，返回状态字符串和操作建议
    state: 该ETF的状态字典，包含 score_history（列表，元素为 {"date": str, "score": float}）
    """
    if hist_df is None or len(hist_df) < 20:
        return f"【{name} ({code})】\n  历史数据不足", None, state

    latest = hist_df.iloc[-1]
    ma20 = latest["ma_short"]
    vol_ma = latest["vol_ma"]
    volume = latest["volume"]
    macd_dif = latest["macd_dif"]
    macd_dea = latest["macd_dea"]
    kdj_k = latest["kdj_k"]
    kdj_d = latest["kdj_d"]
    rsi = latest["rsi"]
    boll_up = latest["boll_up"]
    boll_low = latest["boll_low"]
    williams_r = latest["williams_r"]

    # 判断金叉
    if len(hist_df) >= 2:
        prev = hist_df.iloc[-2]
        macd_golden = (macd_dif > macd_dea) and (prev["macd_dif"] <= prev["macd_dea"])
        kdj_golden = (kdj_k > kdj_d) and (prev["kdj_k"] <= prev["kdj_d"])
    else:
        macd_golden = kdj_golden = False

    # 近5日涨幅
    if len(hist_df) >= 5:
        ret_etf_5d = (real_price / hist_df.iloc[-5]["close"]) - 1
    else:
        ret_etf_5d = 0

    # 近期高点和低点
    recent_high = hist_df['high'].rolling(window=RECENT_HIGH_WINDOW).max().iloc[-1]
    recent_low = hist_df['low'].rolling(window=RECENT_LOW_WINDOW).min().iloc[-1]
    drawdown = (recent_high - real_price) / recent_high if recent_high > 0 else 0
    gain = (real_price - recent_low) / recent_low if recent_low > 0 else 0

    break_ma = real_price < ma20
    trailing_clear = drawdown >= TRAILING_STOP_CLEAR
    trailing_half = drawdown >= TRAILING_STOP_HALF and not trailing_clear
    profit_hit = any(gain >= threshold for threshold, _ in PROFIT_TARGETS)

    # 计算基础评分
    base_score = calculate_score(
        real_price, ma20, volume, vol_ma,
        macd_golden, kdj_golden, rsi,
        boll_up, boll_low, williams_r,
        market_above_ma20, market_above_ma60, market_amount_above_ma20,
        ret_etf_5d, ret_market_5d,
        break_ma, trailing_half, trailing_clear, profit_hit,
        weights
    )
    final_score = base_score * market_factor * sentiment_factor
    target_position = map_score_to_position(final_score)

    # 更新评分历史（带日期）
    today_str = today.strftime('%Y-%m-%d')
    if "score_history" not in state:
        state["score_history"] = []
    found = False
    for item in state["score_history"]:
        if item.get("date") == today_str:
            item["score"] = final_score
            found = True
            break
    if not found:
        state["score_history"].append({"date": today_str, "score": final_score})
    state["score_history"] = sorted(state["score_history"], key=lambda x: x["date"])
    if len(state["score_history"]) > CONFIRM_DAYS:
        state["score_history"] = state["score_history"][-CONFIRM_DAYS:]

    # 构建输出行
    lines = [f"【{name} ({code})】"]
    vol_m = volume / 1e6
    vol_ma_m = vol_ma / 1e6
    lines.append(f"  价格:{real_price:.3f} | 20日线:{ma20:.3f} | 量:{vol_m:.2f}M/5日均:{vol_ma_m:.2f}M")
    macd_symbol = '✓' if macd_golden else '✗'
    kdj_symbol = '✓' if kdj_golden else '✗'
    lines.append(f"  MACD:{macd_symbol} KDJ:{kdj_symbol} RSI:{rsi:.1f} | 布林:{boll_up:.3f}/{boll_low:.3f} | 威廉:{williams_r:.1f}")
    lines.append(f"  大盘:{macro_status}({market_factor:.2f}) 情绪:{sentiment_factor:.2f} | 最终评分:{final_score:.2f} | 仓位建议:{target_position*100:.0f}%")

    signal = None
    if len(state["score_history"]) >= CONFIRM_DAYS:
        recent_scores = [item["score"] for item in state["score_history"]]
        if all(s > BUY_THRESHOLD for s in recent_scores):
            lines.append(f"  🟢 买入{target_position*100:.0f}%:连续{CONFIRM_DAYS}天>{BUY_THRESHOLD}")
            signal = {"action": "BUY", "ratio": target_position, "reason": f"连续{CONFIRM_DAYS}天评分>{BUY_THRESHOLD}"}
        elif all(s < SELL_THRESHOLD for s in recent_scores):
            lines.append(f"  🔴 卖出50%:连续{CONFIRM_DAYS}天<{SELL_THRESHOLD}")
            signal = {"action": "SELL", "ratio": 0.5, "reason": f"连续{CONFIRM_DAYS}天评分<{SELL_THRESHOLD}", "is_clear": False}
        else:
            lines.append("  ⚪ 观望")
    else:
        days = len(state["score_history"])
        last_score = state["score_history"][-1]["score"]
        if last_score > BUY_THRESHOLD:
            lines.append(f"  🟢 偏买入确认中({days}/{CONFIRM_DAYS})")
        elif last_score < SELL_THRESHOLD:
            lines.append(f"  🔴 偏卖出确认中({days}/{CONFIRM_DAYS})")
        else:
            lines.append(f"  ⚪ 中性确认中({days}/{CONFIRM_DAYS})")

    return "\n".join(lines), signal, state