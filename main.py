#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统（重构版）
用法：
    python main.py                # 批量分析所有 ETF
    python main.py --code sh.512800  # 详细分析单只 ETF
"""
import argparse
import datetime
import logging
import os
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from analyzer.data_layer import DataLayer
from analyzer.analyzer import DataAnalyzer
from analyzer.ai import AIClient
from analyzer.trend_scanner import select_trend_buy, select_trend_sell
from analyzer.risk_watch import generate_risk_alerts
from analyzer.config import (
    HISTORY_DAYS,
    MAX_WORKERS,
    DISPLAY_NAME_WIDTH,
    DISPLAY_CODE_WIDTH,
    DISPLAY_PRICE_WIDTH,
    DISPLAY_CHANGE_WIDTH,
    DISPLAY_SCORE_WIDTH,
    DISPLAY_ACTION_WIDTH,
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
from analyzer.utils import pad_display

logger = logging.getLogger(__name__)


def run_batch_analysis(api_key=None, target_code=None):
    dl = DataLayer()
    if not dl.login():
        print("登录 baostock 失败")
        return

    analyzer = DataAnalyzer()

    # 加载 ETF 列表
    try:
        etf_list = dl.load_positions()
    except Exception as e:
        print(f"加载持仓文件失败: {e}")
        dl.logout()
        return

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    # ---------- 市场环境 ----------
    market_df = dl.get_daily_data("sh.000001", start, today_str)
    if market_df is None:
        print("获取大盘数据失败")
        dl.logout()
        return
    market_df = dl.calculate_indicators(market_df, need_amount_ma=True)
    env = dl.get_market_environment(market_df)
    buy_w, sell_w = dl.select_weights(env["state"])
    analyzer.set_environment(env, buy_w, sell_w)

    state = dl.load_state()
    ai_client = AIClient(api_key) if api_key else None

    # ---------- 单只详细分析 ----------
    if target_code:
        row = etf_list[etf_list["代码"] == target_code]
        if row.empty:
            print(f"未找到代码 {target_code}")
            dl.logout()
            return
        code, name = row.iloc[0]["代码"], row.iloc[0]["名称"]
        hist = dl.get_daily_data(code, start, today_str)
        if hist is not None:
            hist = dl.calculate_indicators(hist, need_amount_ma=False)
        weekly = dl.get_weekly_data(code, start, today_str)
        real_price = dl.get_realtime_price(code)
        s = state.get(code, {})
        report = analyzer.detailed_analysis(
            code, name, real_price, hist, weekly, env, today, s,
            ai_comment=(
                ai_client.comment_on_etf(
                    code, name, 0, "详细", env["state"], env["factor"], 50, 0.02
                ) if ai_client else None
            )
        )
        print(report)
        dl.logout()
        return

    # ---------- 批量分析 ----------
    print(f"\n{'='*90}")
    print(f"  ETF 分析报告 - {today_str}  市场状态: {env['state']}  环境因子: {env['factor']:.2f}")
    if env.get("risk_tip"):
        print(f"  {env['risk_tip']}")
    print(f"{'='*90}")
    print(
        pad_display("名称", DISPLAY_NAME_WIDTH),
        pad_display("代码", DISPLAY_CODE_WIDTH),
        pad_display("价格", DISPLAY_PRICE_WIDTH, "right"),
        pad_display("涨跌", DISPLAY_CHANGE_WIDTH, "right"),
        pad_display("评分", DISPLAY_SCORE_WIDTH, "right"),
        " " + pad_display("操作", DISPLAY_ACTION_WIDTH),
        " 信号/提示"
    )
    print("-" * 90)

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for _, row in etf_list.iterrows():
            code, name = row["代码"], row["名称"]
            hist = dl.get_daily_data(code, start, today_str)
            if hist is not None:
                hist = dl.calculate_indicators(hist, need_amount_ma=False)
            weekly = dl.get_weekly_data(code, start, today_str)
            real_price = dl.get_realtime_price(code)
            s = state.get(code, {})
            futures.append(
                (code, ex.submit(analyzer.analyze_single_etf,
                                 code, name, real_price, hist, weekly, env, today, s))
            )
        for code, f in futures:
            out, signal, new_state, score = f.result()
            results.append((out, score))
            state[code] = new_state

    results.sort(key=lambda x: x[1], reverse=True)
    for out, _ in results:
        print(out)

    # ---------- 趋势扫描（独立模块） ----------
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
        print("\n📈 [趋势扫描] 上涨中继/回调再启动形态：")
        print("-" * 60)
        for line in trend_buys:
            print(line)

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
        print("\n📉 [趋势扫描] 趋势转弱/风险累积形态：")
        print("-" * 60)
        for line in trend_sells:
            print(line)

    # ---------- 风险观察（前5只） ----------
    print("\n🛡️ [风险观察] 动态止盈/止损参考（前5只）：")
    for out, score in results[:5]:
        # 简单解析代码和价格（实际可从原始数据获取，这里略作示意）
        parts = out.split()
        if len(parts) >= 4:
            code = parts[1]
            price_str = parts[2]
            try:
                price = float(price_str)
            except:
                continue
            # 这里缺少 hist 和 atr 数据，实际使用时需从分析结果中传入更多信息
            # 暂时用占位提示
            print(f"  {code} : 动态止损/止盈计算需完整行情数据，请在详细报告中查看。")

    dl.save_state(state)
    dl.logout()


def main():
    parser = argparse.ArgumentParser(description="ETF智能分析系统（重构版）")
    parser.add_argument("--code", type=str, help="指定分析某个ETF代码")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    run_batch_analysis(api_key=api_key, target_code=args.code)


if __name__ == "__main__":
    main()