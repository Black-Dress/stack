#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统
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

    raw_outputs = []          # (out_str, score) 按提交顺序
    scan_info_list = []       # 趋势扫描数据（与 raw_outputs 顺序一致）

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
                (code, name, ex.submit(analyzer.analyze_single_etf,
                                 code, name, real_price, hist, weekly, env, today, s))
            )
        for code, name, f in futures:
            out, signal, new_state, score, risk_data, scan_info = f.result()
            raw_outputs.append((out, score))
            state[code] = new_state
            if scan_info:
                scan_info_list.append(scan_info)

    # 按评分排序输出
    sorted_results = sorted(raw_outputs, key=lambda x: x[1], reverse=True)
    for out, _ in sorted_results:
        print(out)

    # 趋势扫描（参数转换为小数）
    scan_to_out = [out for out, _ in raw_outputs]

    buy_indices = select_trend_buy(
        scan_info_list,
        max_count=TREND_BUY_MAX_COUNT,
        low_profit_min=TREND_BUY_LOW_PROFIT_MIN / 100.0,
        low_profit_max=TREND_BUY_LOW_PROFIT_MAX / 100.0,
        max_pullback=TREND_BUY_MAX_PULLBACK / 100.0,
        daily_gain_min=TREND_BUY_DAILY_GAIN_MIN / 100.0,
        daily_gain_max=TREND_BUY_DAILY_GAIN_MAX / 100.0,
        prefer_signal=TREND_BUY_PREFER_SIGNAL,
    )
    if buy_indices:
        print("\n📈 [趋势扫描] 上涨中继/回调再启动形态：")
        print("-" * 60)
        for idx in buy_indices:
            print(scan_to_out[idx])

    sell_indices = select_trend_sell(
        scan_info_list,
        max_count=TREND_SELL_MAX_COUNT,
        min_daily_loss=TREND_SELL_MIN_DAILY_LOSS / 100.0,
        min_pullback=TREND_SELL_MIN_PULLBACK / 100.0,
        min_low_profit=TREND_SELL_MIN_LOW_PROFIT / 100.0,
        include_weak_ma=TREND_SELL_INCLUDE_WEAK_MA,
        include_clear_stop=TREND_SELL_INCLUDE_CLEAR_STOP,
    )
    if sell_indices:
        print("\n📉 [趋势扫描] 趋势转弱/风险累积形态：")
        print("-" * 60)
        for idx in sell_indices:
            print(scan_to_out[idx])

    # 不再需要单独的风险观察区块

    dl.save_state(state)
    dl.logout()


def main():
    parser = argparse.ArgumentParser(description="ETF智能分析系统")
    parser.add_argument("--code", type=str, help="指定分析某个ETF代码")
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    run_batch_analysis(api_key=api_key, target_code=args.code)


if __name__ == "__main__":
    main()