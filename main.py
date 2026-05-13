#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统
主表格简洁，新增持仓表格，趋势扫描整合为单一表格并支持AI
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
from analyzer.data_layer import PositionManager
from analyzer.trend_scanner import (
    select_left_buy, select_trend_buy, select_trend_sell
)
from analyzer.config import *
from analyzer.utils import print_unified_table

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
    pos_mgr = PositionManager(ai_client)

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
        real_price = dl.get_realtime_price(code)
        if real_price is None and hist is not None and not hist.empty:
            real_price = hist.iloc[-1]["close"]
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
            real_price = dl.get_realtime_price(code)
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

    # ---------- 持仓表格 ----------
    position_rows = []
    for si in scan_info_list:
        if si.get("shares", 0) > 0 and si.get("cost_price") is not None:
            cost = si["cost_price"]
            price = si.get("display", {}).get("price", 0)
            profit_pct = (price - cost)/cost*100 if cost>0 else 0
            delta, delta_pct = dl.get_position_change(si.get("display",{}).get("code",""), si["shares"])
            change_str = f"+{delta}(+{delta_pct:.0f}%)" if delta>0 else (f"{delta}({delta_pct:.0f}%)" if delta!=0 else "0")
            # 构建虚拟上下文
            class SimpleCtx:
                pass
            ctx = SimpleCtx()
            ctx.name = si.get("display",{}).get("name","")
            ctx.shares = si["shares"]
            ctx.cost_price = cost
            ctx.real_price = price
            ctx.rsi = si.get("rsi", 50)
            ctx.is_weak_ma = si.get("has_weak_ma_text", False)
            ctx.change_pct = si.get("change_pct", 0) * 100
            ctx.hist_df = None
            advice = pos_mgr.get_unified_advice(ctx, si["final_score"], si.get("display",{}).get("risk_str",""))
            if not advice:
                advice = "持有"
            position_rows.append({
                "name": si.get("display",{}).get("name",""),
                "code": si.get("display",{}).get("code",""),
                "shares": si["shares"],
                "cost": cost,
                "price": price,
                "profit_pct": profit_pct,
                "change": change_str,
                "score": si["final_score"],
                "advice": advice
            })
    if position_rows:
        print_unified_table(position_rows, title="📋 当前持仓详情", table_type="position")

    # ---------- 趋势扫描：调用 AI 或后备 ----------
    buy_candidates = []
    sell_candidates = []
    for si in scan_info_list:
        # 买入候选：未持仓 且 不弱于均线
        if si.get("shares", 0) == 0 and not si.get("has_weak_ma_text", False):
            buy_candidates.append({
                "code": si.get("display",{}).get("code",""),
                "name": si.get("display",{}).get("name",""),
                "final_score": si["final_score"],
                "rsi": si.get("rsi", 50),
                "vol_ratio": si.get("vol_ratio", 1.0),
                "change_pct": si.get("change_pct", 0),
                "above_ma": si.get("above_ma", False),
                "profit_pct_from_low": si.get("profit_pct_from_low", 0),
                "max_drawdown_pct": si.get("max_drawdown_pct", 0)
            })
        # 卖出候选：所有ETF
        sell_candidates.append({
            "code": si.get("display",{}).get("code",""),
            "name": si.get("display",{}).get("name",""),
            "final_score": si["final_score"],
            "rsi": si.get("rsi", 50),
            "change_pct": si.get("change_pct", 0),
            "has_weak_ma_text": si.get("has_weak_ma_text", False),
            "has_clear_stop_text": si.get("has_clear_stop_text", False),
            "has_strong_sell_text": si.get("has_strong_sell_text", False)
        })
    trend_result = {"buy": [], "sell": []}
    if ai_client and AI_ENABLE:
        trend_result = ai_client.get_trend_recommendations(buy_candidates, sell_candidates, env["state"])
    # 后备：如果AI没有返回或失败，使用硬编码
    if not trend_result["buy"] and not trend_result["sell"]:
        buy_indices = select_trend_buy(scan_info_list, max_count=TREND_BUY_MAX_COUNT,
                                       low_profit_min=TREND_BUY_LOW_PROFIT_MIN, low_profit_max=TREND_BUY_LOW_PROFIT_MAX,
                                       max_pullback=TREND_BUY_MAX_PULLBACK, daily_gain_min=TREND_BUY_DAILY_GAIN_MIN,
                                       daily_gain_max=TREND_BUY_DAILY_GAIN_MAX, prefer_signal=TREND_BUY_PREFER_SIGNAL)
        left_indices = select_left_buy(scan_info_list, max_count=LEFT_BUY_MAX_COUNT,
                                       daily_gain_min=LEFT_BUY_DAILY_GAIN_MIN, daily_gain_max=LEFT_BUY_DAILY_GAIN_MAX,
                                       low_profit_min=LEFT_BUY_LOW_PROFIT_MIN, low_profit_max=LEFT_BUY_LOW_PROFIT_MAX,
                                       max_pullback=LEFT_BUY_MAX_PULLBACK, min_score=LEFT_BUY_MIN_SCORE,
                                       rsi_max=LEFT_BUY_RSI_MAX, require_below_ma=LEFT_BUY_REQUIRE_BELOW_MA)
        sell_indices = select_trend_sell(scan_info_list, max_count=TREND_SELL_MAX_COUNT,
                                         min_daily_loss=TREND_SELL_MIN_DAILY_LOSS, min_pullback=TREND_SELL_MIN_PULLBACK,
                                         min_low_profit=TREND_SELL_MIN_LOW_PROFIT,
                                         include_weak_ma=TREND_SELL_INCLUDE_WEAK_MA,
                                         include_clear_stop=TREND_SELL_INCLUDE_CLEAR_STOP)
        # 过滤掉弱于均线的买入推荐
        for idx in buy_indices[:2]:
            if not scan_info_list[idx].get("has_weak_ma_text", False):
                d = scan_info_list[idx].get("display",{})
                trend_result["buy"].append({"code": d.get("code",""), "direction": "right_buy", "advice_text": "🔥 右侧买入"})
        for idx in left_indices[:2]:
            if not scan_info_list[idx].get("has_weak_ma_text", False):
                d = scan_info_list[idx].get("display",{})
                trend_result["buy"].append({"code": d.get("code",""), "direction": "left_buy", "advice_text": "📉 左侧低吸"})
        for idx in sell_indices[:3]:
            d = scan_info_list[idx].get("display",{})
            trend_result["sell"].append({"code": d.get("code",""), "advice_text": "❗ 卖出信号"})
    # 构建趋势扫描表格行
    trend_rows = []
    for rec in trend_result.get("buy", []):
        code = rec["code"]
        si = next((s for s in scan_info_list if s.get("display",{}).get("code")==code), None)
        if si:
            d = si["display"]
            trend_rows.append({
                "name": d["name"],
                "code": d["code"],
                "price": d["price"],
                "change_pct": d["change_pct"],
                "final_score": si["final_score"],
                "advice": rec["advice_text"]
            })
    for rec in trend_result.get("sell", []):
        code = rec["code"]
        si = next((s for s in scan_info_list if s.get("display",{}).get("code")==code), None)
        if si:
            d = si["display"]
            trend_rows.append({
                "name": d["name"],
                "code": d["code"],
                "price": d["price"],
                "change_pct": d["change_pct"],
                "final_score": si["final_score"],
                "advice": rec["advice_text"]
            })
    if trend_rows:
        print_unified_table(trend_rows, title="📊 [趋势扫描] 综合推荐", table_type="trend")

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