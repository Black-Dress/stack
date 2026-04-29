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


def select_unlaunched_etfs(results, max_count=3,
                           min_score=0.5,
                           max_profit_pct=10.0,
                           require_uptrend=True) -> list:
    """
    筛选未启动板块（低点涨幅小、评级不差、无清仓级风险）。
    参数:
        results: (output_str, score) 列表
        max_count: 最多推荐个数
        min_score: 最低评分，低于此值视为动能不足
        max_profit_pct: 最大低点涨幅，超过则视为已启动
        require_uptrend: 是否要求站上中期均线（避免下跌趋势的板块）
    返回: 推荐行字符串列表（不超过 max_count 个）
    """
    candidates = []
    for out, score in results:
        # 硬性排除：强烈卖出、连续低评分、清仓止盈
        if "强烈卖出" in out or "连续3日低评分" in out:
            continue
        if "清仓级" in out:
            continue
        # 评分过滤器
        if score < min_score:
            continue
        # 中期趋势过滤器（若要求趋势向上）
        if require_uptrend and "弱于中期均线" in out:
            continue
        # 低点涨幅过滤器（仅处理明确显示涨幅的行）
        low_match = re.search(r"低点涨(\d+\.?\d*)%", out)
        if low_match:
            low_pct = float(low_match.group(1))
            if low_pct > max_profit_pct:
                continue
        # 符合条件
        candidates.append((score, out))

    # 按评分降序，取前 max_count 个
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [out for _, out in candidates[:max_count]]


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
            futures.append(
                ex.submit(
                    analyzer.analyze_single_etf,
                    code,
                    name,
                    real_price,
                    hist,
                    weekly,
                    market_info,
                    today,
                    s,
                    ai_client,
                )
            )
        for f in futures:
            out, signal, new_state, score = f.result()
            results.append((out, score))
            m = re.search(r"【.*?\((.*?)\)】", out)
            if m:
                state[m.group(1)] = new_state

    results.sort(key=lambda x: x[1], reverse=True)
    for out, _ in results:
        print(out)
        output_lines.append(out)

    # ---------- 每日精选推荐（≤3个）----------
    unlaunched = select_unlaunched_etfs(
        results,
        max_count=3,
        min_score=0.5,
        max_profit_pct=12.0,     # 涨幅 >12% 视为已启动
        require_uptrend=True     # 避开弱势板块
    )
    if unlaunched:
        print("\n" + "★" * 51)
        print("★★ 今日未启动潜力板块（评级较好、涨幅小、趋势稳）★★")
        print("★" * 51)
        for line in unlaunched:
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