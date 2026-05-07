#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统主入口
用法：
    python main.py                # 批量分析所有 ETF
    python main.py --code sh.512800  # 详细分析单只 ETF
"""
import argparse
import datetime
import logging
import os
import re
import time
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any

from analyzer.analyzer import DataAnalyzer
from analyzer.ai import AIClient
from analyzer.config import (
    DEFAULT_PARAMS,
    DEFAULT_BUY_WEIGHTS,
    DEFAULT_SELL_WEIGHTS,
    MARKET_INDEX,
    MACRO_INDEX,
    MACRO_MA_LONG,
    ATR_PERIOD,
    HISTORY_DAYS,
    MAX_WORKERS,
    VOL_HIGH_CONFIRM,
    VOL_MID_CONFIRM,
    AI_PARAMS_ADVISE,
    AI_MARKET_STATE_WITH_SENTIMENT,
    get_email_config,
    DISPLAY_ACTION_WIDTH,
    DISPLAY_CHANGE_WIDTH,
    DISPLAY_CODE_WIDTH,
    DISPLAY_NAME_WIDTH,
    DISPLAY_PRICE_WIDTH,
    DISPLAY_SCORE_WIDTH,
    TREND_BUY_MAX_COUNT,
    TREND_BUY_LOW_PROFIT_MIN,
    TREND_BUY_LOW_PROFIT_MAX,
    TREND_BUY_MAX_PULLBACK,
    TREND_BUY_DAILY_GAIN_MIN,
    TREND_BUY_DAILY_GAIN_MAX,
    TREND_BUY_PREFER_SIGNAL,
    TREND_SELL_MAX_COUNT,
    TREND_SELL_MIN_DAILY_LOSS,
    TREND_SELL_MIN_PULLBACK,
    TREND_SELL_MIN_LOW_PROFIT,
    TREND_SELL_INCLUDE_WEAK_MA,
    TREND_SELL_INCLUDE_CLEAR_STOP,
)
from analyzer.fetcher import DataFetcher, AKSHARE_AVAILABLE
from analyzer.utils import (
    pad_display,
    get_dynamic_confirm_days,
    get_dynamic_history_days,
    fallback_market_state,
    send_email,
    calculate_atr,          # ★ 新增导入
)

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# 内部辅助函数（原在 analyzer.py 底部）
# ----------------------------------------------------------------------
def _prepare_etf_data(code, fetcher, analyzer, start, today_str, params):
    """获取单个 ETF 的历史数据、周线及实时价格并计算指标"""
    cache_key_hist = analyzer._get_cache_key(code, start, today_str)
    hist = fetcher.get_daily_data(code, start, today_str)
    if hist is not None:
        hist = analyzer.calculate_indicators(
            hist,
            need_amount_ma=False,
            recent_high_window=params["RECENT_HIGH_WINDOW"],
            recent_low_window=params["RECENT_LOW_WINDOW"],
            use_cache=True,
            cache_key=cache_key_hist,
        )
    weekly = fetcher.get_weekly_data(code, start, today_str)
    real_price = fetcher.get_realtime_price(code)
    return hist, weekly, real_price


def _get_or_create_environment(fetcher, analyzer, market_df, macro_df, volatility,
                               market_info_basic, api_key):
    """获取或计算市场环境、权重、情绪因子、AI 参数建议"""
    cached_env = fetcher.get_cached_environment()
    if cached_env:
        ai_params = cached_env.get("ai_params_advice")
        return (
            cached_env["market_state"],
            cached_env["market_factor"],
            cached_env["sentiment"],
            cached_env["buy_weights"],
            cached_env["sell_weights"],
            fetcher.get_sentiment_risk_tip(cached_env["sentiment"]),
            ai_params,
        )

    ai_client = AIClient(api_key) if api_key else None

    extra_sentiment = None
    if ai_client and AI_MARKET_STATE_WITH_SENTIMENT and AKSHARE_AVAILABLE:
        try:
            ind = fetcher.fetch_sentiment_indicators()
            extra_sentiment = {
                k: ind[k]
                for k in ["north_net", "main_net_pct", "zt_dt_ratio", "up_down_ratio"]
                if k in ind
            }
        except Exception:
            pass

    if ai_client:
        market_state, market_factor = fetcher.get_market_state(
            market_df, ai_client, extra_sentiment=extra_sentiment
        )
    else:
        state, factor = fallback_market_state(
            market_info_basic["market_above_ma20"], market_info_basic["market_above_ma60"]
        )
        market_state, market_factor = state, factor

    sentiment, sentiment_raw = 1.0, 1.0
    if AKSHARE_AVAILABLE:
        try:
            ind = fetcher.fetch_sentiment_indicators()
            sentiment, sentiment_raw = fetcher.compute_sentiment_factor(ind)
        except Exception as e:
            logger.warning(f"获取情绪失败，使用后备: {e}")
            sentiment, sentiment_raw = fetcher.get_sentiment_factor_simple(macro_df)
    else:
        sentiment, sentiment_raw = fetcher.get_sentiment_factor_simple(macro_df)
    sentiment_risk_tip = fetcher.get_sentiment_risk_tip(sentiment)

    buy_w, sell_w = DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()
    if api_key:
        ai_temp = AIClient(api_key)
        buy_w, sell_w = analyzer.generate_ai_weights(
            ai_temp,
            market_state,
            sentiment,
            market_info_basic["market_above_ma20"],
            market_info_basic["market_above_ma60"],
            market_info_basic["market_amount_above_ma20"],
            volatility,
        )

    ai_params_advice = None
    if api_key and AI_PARAMS_ADVISE:
        try:
            ai_temp = AIClient(api_key)
            ai_params_advice = ai_temp.generate_params_advice(
                market_state, sentiment, volatility
            )
        except Exception as e:
            logger.warning(f"AI参数建议获取失败: {e}")

    fetcher.save_environment_cache(
        market_state,
        market_factor,
        buy_w,
        sell_w,
        sentiment,
        sentiment_raw,
        ai_params_advice=ai_params_advice,
    )
    return market_state, market_factor, sentiment, buy_w, sell_w, sentiment_risk_tip, ai_params_advice


def _extract_pct(out: str, pattern: str) -> Optional[float]:
    """从输出行中按指定正则提取第一个百分比数字，失败返回 None"""
    m = re.search(pattern, out)
    if m:
        return float(m.group(1))
    return None


def select_trend_buy(results, max_count=3,
                     low_profit_min=5.0, low_profit_max=15.0,
                     max_pullback=5.0,
                     daily_gain_min=0.5, daily_gain_max=6.0,
                     prefer_signal=True) -> list:
    """
    趋势型买入推荐：基于技术形态和今日表现，不使用评分。
    """
    candidates = []
    for out, _ in results:
        if "弱于中期均线" in out or "强烈卖出" in out or f"连续{str(DEFAULT_PARAMS.get('CONFIRM_DAYS'))}日低评分" in out or "清仓级" in out:
            continue
        
        low_pct = _extract_pct(out, r"低点涨(\d+\.?\d*)%")
        if low_pct is None or low_pct < low_profit_min or low_pct > low_profit_max:
            continue
        
        pullback = _extract_pct(out, r"高点回落(\d+\.?\d*)%")
        if pullback is not None and pullback > max_pullback:
            continue
        
        change_pct = _extract_pct(out, r"([+-]\d+\.?\d*)%")
        if change_pct is None:
            continue
        if change_pct < daily_gain_min or change_pct > daily_gain_max:
            continue
        
        has_buy = "[BUY]" in out
        ideal_profit = 10.0
        profit_score = 1.0 - abs(low_pct - ideal_profit) / ideal_profit
        sort_key = (not has_buy, - (10.0 if has_buy else 0.0) - profit_score)
        candidates.append((sort_key, out))
    
    candidates.sort(key=lambda x: x[0])
    return [out for _, out in candidates[:max_count]]


def select_trend_sell(results, max_count=3,
                      min_daily_loss=-3.0,
                      min_pullback=6.0,
                      min_low_profit=18.0,
                      include_weak_ma=True,
                      include_clear_stop=True) -> list:
    """
    趋势型卖出警示：基于技术形态和今日表现，不使用评分。
    """
    def risk_score(out: str) -> int:
        risk = 0
        if "[SELL]" in out and f"连续{str(DEFAULT_PARAMS.get('CONFIRM_DAYS'))}日低评分" in out:
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
        if low_pct is not None and low_pct > min_low_profit:
            if pullback is not None and pullback > 5:
                cond = True
        
        if include_weak_ma and "弱于中期均线" in out:
            cond = True
        if include_clear_stop and "清仓级" in out:
            cond = True
        if "强烈卖出" in out or f"连续{str(DEFAULT_PARAMS.get('CONFIRM_DAYS'))}日低评分" in out:
            cond = True
        if "[SELL]" in out:
            cond = True
        
        if cond:
            candidates.append((out, risk_score(out)))
    
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [out for out, _ in candidates[:max_count]]


# ----------------------------------------------------------------------
# 批量分析主流程
# ----------------------------------------------------------------------
def run_batch_analysis(api_key=None, target_code=None):
    fetcher = DataFetcher()
    analyzer = DataAnalyzer()
    if not fetcher.login():
        return
    try:
        etf_list = fetcher.load_positions()
    except Exception as e:
        print(f"请准备 positions.csv (代码,名称)，错误: {e}")
        return

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    market_df = fetcher.get_daily_data(MARKET_INDEX, start, today_str)
    macro_df = fetcher.get_daily_data(MACRO_INDEX, start, today_str)
    if market_df is None or macro_df is None:
        print("获取宏观数据失败")
        return

    macro_df = analyzer.calculate_indicators(macro_df, need_amount_ma=False)
    macro_df["ma_long"] = macro_df["close"].rolling(MACRO_MA_LONG).mean()
    market_df = analyzer.calculate_indicators(market_df, need_amount_ma=True)
    market_df["atr"] = calculate_atr(market_df, ATR_PERIOD)
    volatility = (market_df["atr"] / market_df["close"]).iloc[-20:].mean()

    mkt = market_df.iloc[-1]
    market_info_basic = {
        "market_above_ma20": mkt["close"] > mkt["ma_short"],
        "market_above_ma60": mkt["close"] > mkt.get("ma_long", mkt["ma_short"]),
        "market_amount_above_ma20": mkt["amount"] > mkt["amount_ma"],
        "ret_market_5d": (mkt["close"] / market_df.iloc[-5]["close"] - 1)
        if len(market_df) >= 5
        else 0,
    }

    (
        market_state,
        market_factor,
        sentiment,
        buy_w,
        sell_w,
        risk_tip,
        ai_params_advice,
    ) = _get_or_create_environment(
        fetcher, analyzer, market_df, macro_df, volatility, market_info_basic, api_key
    )

    market_info = {
        "macro_status": market_state,
        "market_factor": market_factor,
        "sentiment_factor": sentiment,
        "sentiment_risk_tip": risk_tip,
        "market_volatility": volatility, 
        **market_info_basic,
    }
    analyzer.set_market_info(market_info)
    analyzer.set_weights(buy_w, sell_w)
    analyzer.set_ai_params_advice(ai_params_advice or {})

    params = DEFAULT_PARAMS.copy()
    params["CONFIRM_DAYS"] = get_dynamic_confirm_days(volatility, params["CONFIRM_DAYS"])
    params = analyzer._apply_ai_params_advice(params)
    analyzer.params = params

    state = fetcher.load_state()
    ai_client = AIClient(api_key) if api_key else None

    # 单只详细分析
    if target_code:
        target = etf_list[etf_list["代码"] == target_code]
        if target.empty:
            print(f"未找到代码 {target_code}")
            fetcher.logout()
            return
        code, name = target.iloc[0]["代码"], target.iloc[0]["名称"]
        hist, weekly, real_price = _prepare_etf_data(
            code, fetcher, analyzer, start, today_str, params
        )
        etf_state = state.get(code, {})
        report = analyzer.detailed_analysis(
            code, name, real_price, hist, weekly, market_info, today, etf_state, ai_client
        )
        print(report)
        fetcher.logout()
        return

    # 批量分析
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ETF 分析报告")
    print(f"市场状态: {market_state}, 市场因子: {market_factor:.2f}")
    if risk_tip:
        print(f"情绪因子: {sentiment:.3f} - {risk_tip}")
    else:
        print(f"情绪因子: {sentiment:.3f}")

    print(
        pad_display("名称", DISPLAY_NAME_WIDTH),
        pad_display("代码", DISPLAY_CODE_WIDTH),
        pad_display("价格", DISPLAY_PRICE_WIDTH, "right"),
        pad_display("涨跌", DISPLAY_CHANGE_WIDTH, "right"),
        pad_display("评分", DISPLAY_SCORE_WIDTH, "right"),
        " " + pad_display("操作", DISPLAY_ACTION_WIDTH),
        " 信号/提示",
    )
    print("-" * 120)

    output_lines, results = [], []
    contexts = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for _, row in etf_list.iterrows():
            code, name = row["代码"], row["名称"]
            hist, weekly, real_price = _prepare_etf_data(
                code, fetcher, analyzer, start, today_str, params
            )
            s = state.get(code, {})
            futures.append((
                code,
                ex.submit(
                    analyzer.analyze_single_etf,
                    code, name, real_price, hist, weekly,
                    market_info, today, s, ai_client,
                )
            ))
        for code, f in futures:
            out, signal, new_state, score = f.result()
            results.append((out, score))
            state[code] = new_state

    results.sort(key=lambda x: x[1], reverse=True)
    for out, _ in results:
        print(out)
        output_lines.append(out)

    # 趋势买入参考
    trend_buys = select_trend_buy(
        results,
        max_count=TREND_BUY_MAX_COUNT,
        low_profit_min=TREND_BUY_LOW_PROFIT_MIN,
        low_profit_max=TREND_BUY_LOW_PROFIT_MAX,
        max_pullback=TREND_BUY_MAX_PULLBACK,
        daily_gain_min=TREND_BUY_DAILY_GAIN_MIN,
        daily_gain_max=TREND_BUY_DAILY_GAIN_MAX,
        prefer_signal=TREND_BUY_PREFER_SIGNAL
    )
    if trend_buys:
        print("\n📈 趋势型买入参考（上涨中继/回调再启动）")
        print("-" * 60)
        for line in trend_buys:
            print(line)

    # 趋势卖出警示
    trend_sells = select_trend_sell(
        results,
        max_count=TREND_SELL_MAX_COUNT,
        min_daily_loss=TREND_SELL_MIN_DAILY_LOSS,
        min_pullback=TREND_SELL_MIN_PULLBACK,
        min_low_profit=TREND_SELL_MIN_LOW_PROFIT,
        include_weak_ma=TREND_SELL_INCLUDE_WEAK_MA,
        include_clear_stop=TREND_SELL_INCLUDE_CLEAR_STOP
    )
    if trend_sells:
        print("\n📉 趋势型卖出警示（趋势转弱/风险累积）")
        print("-" * 60)
        for line in trend_sells:
            print(line)


    fetcher.save_state(state)
    fetcher.logout()

    email_cfg = get_email_config()
    if email_cfg["send_email"]:
        send_email(f"ETF分析报告 - {today_str}", "\n".join(output_lines))




# ----------------------------------------------------------------------
# 入口
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ETF智能分析系统")
    parser.add_argument(
        "--code",
        type=str,
        help="指定分析某个ETF代码（例如 sh.512800），不指定则批量分析所有",
    )
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    run_batch_analysis(api_key=api_key, target_code=args.code)


if __name__ == "__main__":
    main()