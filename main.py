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
from analyzer.analyzer import DataAnalyzer, AIClient
from analyzer.trend_scanner import (
    select_left_buy, select_trend_buy, select_trend_sell, evaluate_buy_level
)
from analyzer.config import *
from analyzer.utils import format_etf_output_line, pad_display

logger = logging.getLogger(__name__)


def print_table(rows, env, today_str):
    """统一打印报告表头及所有数据行"""
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
        " " + pad_display("操作", DISPLAY_LEVEL_WIDTH),
        " 信号/提示"
    )
    print("-" * 90)
    for out, score in rows:
        print(out)


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

    if target_code:
        row = etf_list[etf_list["代码"] == target_code]
        if row.empty:
            print(f"未找到代码 {target_code}")
            dl.logout()
            return
        code, name, cost, shares = row.iloc[0]["代码"], row.iloc[0]["名称"], row.iloc[0]["成本"], row.iloc[0]["份额"]
        hist = dl.get_daily_data(code, start, today_str)
        if hist is not None:
            hist = dl.calculate_indicators(hist, need_amount_ma=False)
        weekly = dl.get_weekly_data(code, start, today_str)
        real_price = dl.get_realtime_price(code)
        # 实时价获取失败时使用昨日收盘价
        if real_price is None and hist is not None and not hist.empty:
            real_price = hist.iloc[-1]["close"]
            logger.warning(f"{code} 实时价获取失败，使用昨日收盘 {real_price:.3f}")
        s = state.get(code, {})
        report = analyzer.detailed_analysis(
            code, name, real_price, hist, weekly, env, today, s,
            ai_client=ai_client,
            cost_price=cost if pd.notna(cost) else None,
            shares=int(shares) if pd.notna(shares) else 0
        )
        print(report)
        dl.logout()
        return

    raw_outputs = []
    scan_info_list = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = []
        for _, row in etf_list.iterrows():
            code, name = row["代码"], row["名称"]
            cost = row["成本"] if pd.notna(row["成本"]) else None
            shares = int(row["份额"]) if pd.notna(row["份额"]) else 0
            hist = dl.get_daily_data(code, start, today_str)
            if hist is not None:
                hist = dl.calculate_indicators(hist, need_amount_ma=False)
            weekly = dl.get_weekly_data(code, start, today_str)
            real_price = dl.get_realtime_price(code)
            # 实时价获取失败时使用昨日收盘价
            if real_price is None and hist is not None and not hist.empty:
                real_price = hist.iloc[-1]["close"]
                logger.warning(f"{code} 实时价获取失败，使用昨日收盘 {real_price:.3f}")
            s = state.get(code, {})
            futures.append(
                (code, name, ex.submit(analyzer.analyze_single_etf,
                                 code, name, real_price, hist, weekly, env, today, s,
                                 cost_price=cost, shares=shares))
            )

        for code, name, f in futures:
            out, signal, new_state, score, risk_data, scan_info = f.result()
            # 移除主表格中的买卖信号标签，仅保留风险提示
            out_clean = out.replace(" [BUY]", "").replace(" [SELL]", "")
            raw_outputs.append((out_clean, score))
            state[code] = new_state
            if scan_info is None:
                scan_info = {}
            scan_info_list.append(scan_info)

    # ---------- 买入力度建议（统一在 main 中生成） ----------
    buy_advice_map = {}
    if BUY_ADVICE_ENABLE:
        for idx, si in enumerate(scan_info_list):
            advice = evaluate_buy_level(si)
            if advice:
                buy_advice_map[idx] = advice

    # ---------- 表格输出 ----------
    sorted_results = sorted(raw_outputs, key=lambda x: x[1], reverse=True)
    print_table(sorted_results, env, today_str)

    # 用于趋势扫描的原始输出行（不包含买入建议）
    scan_to_out = [out for out, _ in raw_outputs]

    # ---------- 右侧趋势扫描 ----------
    right_buy_indices = select_trend_buy(
        scan_info_list,
        max_count=TREND_BUY_MAX_COUNT,
        low_profit_min=TREND_BUY_LOW_PROFIT_MIN,
        low_profit_max=TREND_BUY_LOW_PROFIT_MAX,
        max_pullback=TREND_BUY_MAX_PULLBACK,
        daily_gain_min=TREND_BUY_DAILY_GAIN_MIN,
        daily_gain_max=TREND_BUY_DAILY_GAIN_MAX,
        prefer_signal=TREND_BUY_PREFER_SIGNAL,
    )

    # ---------- 左侧趋势扫描 ----------
    left_buy_indices = []
    if LEFT_BUY_ENABLE:
        left_buy_indices = select_left_buy(
            scan_info_list,
            max_count=LEFT_BUY_MAX_COUNT,
            daily_gain_min=LEFT_BUY_DAILY_GAIN_MIN,
            daily_gain_max=LEFT_BUY_DAILY_GAIN_MAX,
            low_profit_min=LEFT_BUY_LOW_PROFIT_MIN,
            low_profit_max=LEFT_BUY_LOW_PROFIT_MAX,
            max_pullback=LEFT_BUY_MAX_PULLBACK,
            min_score=LEFT_BUY_MIN_SCORE,
            rsi_max=LEFT_BUY_RSI_MAX,
            require_below_ma=LEFT_BUY_REQUIRE_BELOW_MA,
        )

    # ---------- 打印右侧推荐 ----------
    if right_buy_indices:
        print("\n📈 [趋势扫描] 上涨中继/回调再启动形态：")
        print("-" * 60)
        for idx in right_buy_indices:
            disp = scan_info_list[idx]["display"]
            advice = buy_advice_map.get(idx, "")
            # 将买入建议附加到风险提示中
            risk_with_advice = disp["risk_str"]
            if advice:
                risk_with_advice += f"  {advice}" if risk_with_advice else advice
            # 重新生成固定宽度行
            line = format_etf_output_line(
                name=disp["name"],
                code=disp["code"],
                price=disp["price"],
                change_pct=disp["change_pct"],
                final_score=disp["final_score"],
                action_level=disp["action_level"],
                risk_str=risk_with_advice,
                signal_action=None,  # 不显示买卖信号
            )
            print(line)

    # 打印左侧推荐同理
    if left_buy_indices:
        print("\n📉 [左侧扫描] 潜在低吸/反转形态：")
        print("-" * 60)
        for idx in left_buy_indices:
            disp = scan_info_list[idx]["display"]
            advice = buy_advice_map.get(idx, "")
            risk_with_advice = disp["risk_str"]
            if advice:
                risk_with_advice += f"  {advice}" if risk_with_advice else advice
            line = format_etf_output_line(
                name=disp["name"],
                code=disp["code"],
                price=disp["price"],
                change_pct=disp["change_pct"],
                final_score=disp["final_score"],
                action_level=disp["action_level"],
                risk_str=risk_with_advice,
            )
            print(line)

    # ---------- 卖出扫描 ----------
    sell_indices = select_trend_sell(
        scan_info_list,
        max_count=TREND_SELL_MAX_COUNT,
        min_daily_loss=TREND_SELL_MIN_DAILY_LOSS,
        min_pullback=TREND_SELL_MIN_PULLBACK,
        min_low_profit=TREND_SELL_MIN_LOW_PROFIT,
        include_weak_ma=TREND_SELL_INCLUDE_WEAK_MA,
        include_clear_stop=TREND_SELL_INCLUDE_CLEAR_STOP,
    )
    if sell_indices:
        print("\n📉 [趋势扫描] 趋势转弱/风险累积形态：")
        print("-" * 60)
        for idx in sell_indices:
            print(scan_to_out[idx])

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