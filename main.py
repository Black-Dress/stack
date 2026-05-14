#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统
主表格简洁，新增持仓表格，趋势扫描整合为单一表格并支持AI
（纯AI版：无后备规则）
"""
import argparse
import datetime
import logging
import os
from typing import Optional

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from analyzer.data_layer import DataLayer
from analyzer.analyzer import DataAnalyzer
from analyzer.ai import AIClient
from analyzer.config import *
from analyzer.utils import print_unified_table, resolve_real_price
from analyzer.trend_scanner import select_trend_buy, select_left_buy, select_trend_sell

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

    # 记录当日持仓快照
    dl.append_daily_snapshot(today_str, etf_list)

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
        raw_price = dl.get_realtime_price(code)
        real_price, use_fallback = resolve_real_price(raw_price, hist)
        if use_fallback:
            logger.warning(f"{code} 实时价获取失败，使用昨日收盘 {real_price:.3f}")
        s = state.get(code, {})
        report = analyzer.detailed_analysis(
            code, name, real_price, hist, weekly, env, today, s,
            ai_client=ai_client, cost_price=cost if pd.notna(cost) else None, shares=int(shares) if pd.notna(shares) else 0
        )
        print(report)
        dl.logout()
        return

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
            raw_price = dl.get_realtime_price(code)
            real_price, use_fallback = resolve_real_price(raw_price, hist)
            if use_fallback:
                logger.warning(f"{code} 实时价获取失败，使用昨日收盘 {real_price:.3f}")
            s = state.get(code, {})
            futures.append(
                (code, name, ex.submit(analyzer.analyze_single_etf,
                                 code, name, real_price, hist, weekly, env, today, s,
                                 cost_price=cost, shares=shares))
            )
        for code, name, f in futures:
            out, signal, new_state, score, risk_data, scan_info = f.result()
            state[code] = new_state
            if scan_info is None:
                scan_info = {}
            scan_info_list.append(scan_info)

    # ---------- 构建主表格行 ----------
    main_rows = []
    for si in scan_info_list:
        d = si.get("display", {})
        main_rows.append({
            "name": d.get("name", ""),
            "code": d.get("code", ""),
            "price": d.get("price", 0),
            "change_pct": d.get("change_pct", 0),
            "final_score": si.get("final_score", 0),
            "risk_str": d.get("risk_str", "")
        })
    main_rows_sorted = sorted(main_rows, key=lambda x: x["final_score"], reverse=True)
    print_unified_table(main_rows_sorted, env=env, today_str=today_str, table_type="main")

    # ---------- 构建持仓表格（暂不含建议） ----------
    position_rows = []
    position_change_map = {}
    for si in scan_info_list:
        if si.get("shares", 0) > 0 and si.get("cost_price") is not None:
            cost = si["cost_price"]
            price = si.get("display", {}).get("price", 0)
            profit_pct = (price - cost)/cost*100 if cost>0 else 0
            delta, delta_pct = dl.get_position_change(si.get("display",{}).get("code",""), si["shares"])
            change_str = f"+{delta}(+{delta_pct:.0f}%)" if delta>0 else (f"{delta}({delta_pct:.0f}%)" if delta!=0 else "0")
            position_rows.append({
                "name": si.get("display",{}).get("name",""),
                "code": si.get("display",{}).get("code",""),
                "shares": si["shares"],
                "cost": cost,
                "price": price,
                "profit_pct": profit_pct,
                "change": change_str,
                "score": si["final_score"],
                "advice": ""
            })
            position_change_map[si.get("display",{}).get("code","")] = change_str

    # ---------- 趋势扫描候选生成 ----------
    BUY_MAX_COUNT = 6
    BUY_LOW_PROFIT_MIN = 0.02
    BUY_LOW_PROFIT_MAX = 0.35
    BUY_MAX_PULLBACK = 0.08
    BUY_DAILY_GAIN_MIN = 0.0
    BUY_DAILY_GAIN_MAX = 0.09

    LEFT_MAX_COUNT = 4
    LEFT_DAILY_GAIN_MIN = -0.04
    LEFT_DAILY_GAIN_MAX = 0.04
    LEFT_LOW_PROFIT_MIN = 0.0
    LEFT_LOW_PROFIT_MAX = 0.12
    LEFT_MAX_PULLBACK = 0.10
    LEFT_MIN_SCORE = 45
    LEFT_RSI_MAX = 55
    LEFT_REQUIRE_BELOW_MA = False

    SELL_MAX_COUNT = 5
    SELL_MIN_DAILY_LOSS = -0.02
    SELL_MIN_PULLBACK = 0.05
    SELL_MIN_LOW_PROFIT = 0.15
    SELL_INCLUDE_WEAK_MA = True
    SELL_INCLUDE_CLEAR_STOP = True

    def to_py_bool(val):
        return bool(val) if isinstance(val, (bool, np.bool_)) else val
    def to_py_float(val):
        return float(val) if isinstance(val, (float, np.floating)) else val

    right_buy_indices = select_trend_buy(
        scan_info_list,
        max_count=BUY_MAX_COUNT,
        low_profit_min=BUY_LOW_PROFIT_MIN,
        low_profit_max=BUY_LOW_PROFIT_MAX,
        max_pullback=BUY_MAX_PULLBACK,
        daily_gain_min=BUY_DAILY_GAIN_MIN,
        daily_gain_max=BUY_DAILY_GAIN_MAX,
        prefer_signal=True
    )
    right_buy_indices = [idx for idx in right_buy_indices if scan_info_list[idx].get("shares", 0) == 0]

    left_buy_indices = select_left_buy(
        scan_info_list,
        max_count=LEFT_MAX_COUNT,
        daily_gain_min=LEFT_DAILY_GAIN_MIN,
        daily_gain_max=LEFT_DAILY_GAIN_MAX,
        low_profit_min=LEFT_LOW_PROFIT_MIN,
        low_profit_max=LEFT_LOW_PROFIT_MAX,
        max_pullback=LEFT_MAX_PULLBACK,
        min_score=LEFT_MIN_SCORE,
        rsi_max=LEFT_RSI_MAX,
        require_below_ma=LEFT_REQUIRE_BELOW_MA
    )
    left_buy_indices = [idx for idx in left_buy_indices if scan_info_list[idx].get("shares", 0) == 0]

    sell_indices = select_trend_sell(
        scan_info_list,
        max_count=SELL_MAX_COUNT,
        min_daily_loss=SELL_MIN_DAILY_LOSS,
        min_pullback=SELL_MIN_PULLBACK,
        min_low_profit=SELL_MIN_LOW_PROFIT,
        include_weak_ma=SELL_INCLUDE_WEAK_MA,
        include_clear_stop=SELL_INCLUDE_CLEAR_STOP
    )
    sell_indices = [idx for idx in sell_indices if scan_info_list[idx].get("shares", 0) > 0]

    # 构建用于AI的候选字典
    all_need_recommend = {}
    for si in scan_info_list:
        if si.get("shares", 0) > 0:
            code = si["display"]["code"]
            all_need_recommend[code] = si
    for idx in right_buy_indices + left_buy_indices:
        si = scan_info_list[idx]
        code = si["display"]["code"]
        if code not in all_need_recommend:
            all_need_recommend[code] = si
    for idx in sell_indices:
        si = scan_info_list[idx]
        code = si["display"]["code"]
        if code not in all_need_recommend:
            all_need_recommend[code] = si

    # 仅AI调用，无后备规则
    batch_advice = {}
    if ai_client and AI_ENABLE:
        etf_dict_for_ai = {}
        for code, si in all_need_recommend.items():
            d = si["display"]
            etf_dict_for_ai[code] = {
                "name": d["name"],
                "final_score": to_py_float(si["final_score"]),
                "rsi": to_py_float(si.get("rsi", 50)),
                "vol_ratio": to_py_float(si.get("vol_ratio", 1.0)),
                "change_pct": to_py_float(si.get("change_pct", 0)),
                "above_ma": to_py_bool(si.get("above_ma", False)),
                "profit_pct_from_low": to_py_float(si.get("profit_pct_from_low", 0)),
                "shares": si.get("shares", 0),
                "cost_price": si.get("cost_price"),
                "risk_str": d.get("risk_str", ""),
                "price": d["price"],
                "change_pct_display": d["change_pct"]
            }
        batch_advice = ai_client.get_batch_recommendations(etf_dict_for_ai, env["state"])

    # 更新持仓表格建议
    for row in position_rows:
        code = row["code"]
        if code in batch_advice:
            original_advice = batch_advice[code]
            change_str = position_change_map.get(code, "")
            is_reduced_today = change_str.startswith("-") if change_str else False
            if is_reduced_today and any(kw in original_advice for kw in ["减仓", "卖出", "止损"]):
                row["advice"] = "✅ 已执行减仓"
            else:
                row["advice"] = original_advice

    if position_rows:
        print_unified_table(position_rows, title="📋 当前持仓详情", table_type="position")

    # 趋势扫描显示
    buy_rows = []
    sell_rows = []
    for code, advice_text in batch_advice.items():
        if code not in all_need_recommend:
            continue
        si = all_need_recommend[code]
        d = si["display"]
        is_holding = si.get("shares", 0) > 0
        change_str = position_change_map.get(code, "")
        is_reduced_today = change_str.startswith("-") if change_str else False
        if not is_holding and any(kw in advice_text for kw in ["买入", "吸筹", "低吸"]):
            buy_rows.append({
                "name": d["name"],
                "code": d["code"],
                "price": d["price"],
                "change_pct": d["change_pct"],
                "final_score": si["final_score"],
                "advice": advice_text
            })
        elif is_holding and any(kw in advice_text for kw in ["卖出", "止损", "止盈", "减仓", "清仓"]):
            if not is_reduced_today:
                sell_rows.append({
                    "name": d["name"],
                    "code": d["code"],
                    "price": d["price"],
                    "change_pct": d["change_pct"],
                    "final_score": si["final_score"],
                    "advice": advice_text
                })

    buy_rows.sort(key=lambda x: x["final_score"], reverse=True)
    sell_rows.sort(key=lambda x: x["final_score"], reverse=True)

    if buy_rows:
        print_unified_table(buy_rows, title="📊 [趋势扫描] 买入推荐", table_type="trend")
    if buy_rows and sell_rows:
        print()
    if sell_rows:
        print_unified_table(sell_rows, title="📊 [趋势扫描] 卖出警示", table_type="trend")

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