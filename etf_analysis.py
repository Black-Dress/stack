# etf_analysis.py
# 单个ETF分析逻辑

import numpy as np
from config import (
    STRATEGY_WEIGHTS, BUY_THRESHOLD, SELL_THRESHOLD, CONFIRM_DAYS,
    PROFIT_TARGETS, RECENT_HIGH_WINDOW, RECENT_LOW_WINDOW,
    ATR_STOP_MULT, ATR_TRAILING_MULT, QUICK_BUY_THRESHOLD,
    RISK_WARNING_DAYS, RISK_WARNING_THRESHOLD
)

def _get_recent_high_low(hist_df, row_idx):
    if 'recent_high_10' in hist_df.columns and 'recent_low_20' in hist_df.columns:
        return hist_df.iloc[row_idx]['recent_high_10'], hist_df.iloc[row_idx]['recent_low_20']
    else:
        high_series = hist_df['high'].rolling(window=RECENT_HIGH_WINDOW).max()
        low_series = hist_df['low'].rolling(window=RECENT_LOW_WINDOW).min()
        return high_series.iloc[row_idx], low_series.iloc[row_idx]

def _check_signal_confirm(score_history, target_position):
    if len(score_history) < CONFIRM_DAYS:
        return None, None
    recent_scores = [item["score"] for item in score_history]
    if all(s > BUY_THRESHOLD for s in recent_scores):
        signal = {
            "action": "BUY",
            "ratio": target_position,
            "reason": f"连续{CONFIRM_DAYS}天评分>{BUY_THRESHOLD}",
            "text": f"  🟢 买入{target_position*100:.0f}%:连续{CONFIRM_DAYS}天>{BUY_THRESHOLD}"
        }
        return "BUY", signal
    elif all(s < SELL_THRESHOLD for s in recent_scores):
        signal = {
            "action": "SELL",
            "ratio": 0.5,
            "reason": f"连续{CONFIRM_DAYS}天评分<{SELL_THRESHOLD}",
            "is_clear": False,
            "text": f"  🔴 卖出50%:连续{CONFIRM_DAYS}天<{SELL_THRESHOLD}"
        }
        return "SELL", signal
    return None, None

def _check_quick_signal(score_history, last_score):
    if len(score_history) >= 2:
        prev_score = score_history[-2]["score"]
        if last_score > prev_score and last_score > QUICK_BUY_THRESHOLD:
            return True
    return False

def _check_risk_warning(score_history):
    if len(score_history) >= RISK_WARNING_DAYS:
        recent = [item["score"] for item in score_history[-RISK_WARNING_DAYS:]]
        if all(s < RISK_WARNING_THRESHOLD for s in recent):
            return True
    return False

def _format_output(name, code, real_price, ma20, volume, vol_ma,
                   macd_golden, kdj_golden, rsi, boll_up, boll_low, williams_r,
                   macro_status, market_factor, sentiment_factor,
                   final_score, target_position, confirm_info, signal_info, risk_warning):
    vol_m = volume / 1e6
    vol_ma_m = vol_ma / 1e6
    lines = [f"【{name} ({code})】"]
    lines.append(f"  价格:{real_price:.3f} | 20日线:{ma20:.3f} | 量:{vol_m:.2f}M/5日均:{vol_ma_m:.2f}M")
    macd_symbol = '✓' if macd_golden else '✗'
    kdj_symbol = '✓' if kdj_golden else '✗'
    lines.append(f"  MACD:{macd_symbol} KDJ:{kdj_symbol} RSI:{rsi:.1f} | 布林:{boll_up:.3f}/{boll_low:.3f} | 威廉:{williams_r:.1f}")
    lines.append(f"  大盘:{macro_status}({market_factor:.2f}) 情绪:{sentiment_factor:.2f} | 最终评分:{final_score:.2f} | 仓位建议:{target_position*100:.0f}%")
    if risk_warning:
        lines.append(f"  ⚠️ 风险提示：连续{RISK_WARNING_DAYS}天评分低于{RISK_WARNING_THRESHOLD}")
    if signal_info:
        # signal_info 现在包含 'text' 键
        lines.append(signal_info['text'])
    else:
        days, last_score = confirm_info['days'], confirm_info['last_score']
        if last_score > BUY_THRESHOLD:
            lines.append(f"  🟢 偏买入确认中({days}/{CONFIRM_DAYS})")
        elif last_score < SELL_THRESHOLD:
            lines.append(f"  🔴 偏卖出确认中({days}/{CONFIRM_DAYS})")
        else:
            lines.append(f"  ⚪ 中性确认中({days}/{CONFIRM_DAYS})")
    return "\n".join(lines)

def calculate_score(
    real_price, ma20, volume, vol_ma, macd_golden, kdj_golden, rsi,
    boll_up, boll_low, williams_r,
    market_above_ma20, market_above_ma60, market_amount_above_ma20,
    ret_etf_5d, ret_market_5d,
    break_ma, trailing_half, trailing_clear, profit_hit,
    weekly_above_ma, weekly_below_ma,
    weights=None
):
    if weights is None:
        weights = STRATEGY_WEIGHTS

    conditions = {
        "price_above_ma20": real_price > ma20,
        "volume_above_ma5": volume > vol_ma,
        "macd_golden_cross": macd_golden,
        "kdj_golden_cross": kdj_golden,
        "bollinger_break_up": real_price > boll_up,
        "williams_oversold": williams_r > 80,
        "price_below_ma20": real_price < ma20,
        "bollinger_break_down": real_price < boll_low,
        "williams_overbought": williams_r < 20,
        "rsi_overbought": rsi > 70,
        "market_above_ma20": market_above_ma20,
        "market_above_ma60": market_above_ma60,
        "market_amount_above_ma20": market_amount_above_ma20,
        "outperform_market": ret_etf_5d > ret_market_5d,
        "underperform_market": not (ret_etf_5d > ret_market_5d),
        "stop_loss_ma_break": break_ma,
        "trailing_stop_clear": trailing_clear,
        "trailing_stop_half": trailing_half,
        "profit_target_hit": profit_hit,
        "weekly_above_ma20": weekly_above_ma,
        "weekly_below_ma20": weekly_below_ma,
    }
    score = 0.0
    for key, cond in conditions.items():
        if cond:
            weight = weights.get(key, 0)
            score += weight
            if key not in weights:
                print(f"警告：权重字典缺少键 '{key}'，使用0")
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
    code, name, real_price, hist_df, weekly_df,
    macro_status, market_factor, sentiment_factor,
    market_above_ma20, market_above_ma60, market_amount_above_ma20,
    ret_market_5d, today, state, weights=None
):
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
    atr = latest["atr"]

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
    recent_high, recent_low = _get_recent_high_low(hist_df, -1)
    if np.isnan(recent_high):
        recent_high = hist_df['high'].max()
    if np.isnan(recent_low):
        recent_low = hist_df['low'].min()

    drawdown = (recent_high - real_price) / recent_high if recent_high > 0 else 0
    gain = (real_price - recent_low) / recent_low if recent_low > 0 else 0

    # ATR动态止损
    atr_pct = atr / real_price if real_price > 0 else 0
    trailing_clear = drawdown >= (ATR_STOP_MULT * atr_pct)
    trailing_half = drawdown >= (ATR_TRAILING_MULT * atr_pct) and not trailing_clear

    break_ma = real_price < ma20
    profit_hit = any(gain >= threshold for threshold, _ in PROFIT_TARGETS)

    # 周线判断
    weekly_above_ma = False
    weekly_below_ma = False
    if weekly_df is not None and not weekly_df.empty:
        weekly_latest = weekly_df.iloc[-1]
        weekly_close = weekly_latest['close']
        weekly_ma = weekly_latest.get('ma_short', np.nan)
        if not np.isnan(weekly_ma):
            weekly_above_ma = weekly_close > weekly_ma
            weekly_below_ma = weekly_close < weekly_ma

    # 计算基础评分
    base_score = calculate_score(
        real_price, ma20, volume, vol_ma,
        macd_golden, kdj_golden, rsi,
        boll_up, boll_low, williams_r,
        market_above_ma20, market_above_ma60, market_amount_above_ma20,
        ret_etf_5d, ret_market_5d,
        break_ma, trailing_half, trailing_clear, profit_hit,
        weekly_above_ma, weekly_below_ma,
        weights
    )
    final_score = base_score * market_factor * sentiment_factor
    if np.isnan(final_score):
        final_score = 0.0
    target_position = map_score_to_position(final_score)

    # 更新评分历史
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

    # 信号确认
    signal_type, signal = _check_signal_confirm(state["score_history"], target_position)
    if not signal:
        if len(state["score_history"]) >= 2:
            last_score = state["score_history"][-1]["score"]
            if _check_quick_signal(state["score_history"], last_score):
                quick_ratio = min(target_position * 0.5, 0.5)
                signal = {
                    "action": "BUY",
                    "ratio": quick_ratio,
                    "reason": f"快速信号（评分上升且>{QUICK_BUY_THRESHOLD}）",
                    "text": f"  🟢 买入（快速）{quick_ratio*100:.0f}%:快速信号"
                }
                signal_type = "BUY_QUICK"

    risk_warning = _check_risk_warning(state["score_history"])

    confirm_info = {
        'days': len(state["score_history"]),
        'last_score': state["score_history"][-1]["score"]
    }
    output = _format_output(
        name, code, real_price, ma20, volume, vol_ma,
        macd_golden, kdj_golden, rsi, boll_up, boll_low, williams_r,
        macro_status, market_factor, sentiment_factor,
        final_score, target_position, confirm_info, signal, risk_warning
    )

    return output, signal, state