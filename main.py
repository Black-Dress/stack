#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""ETF 事件驱动仓位管理引擎 - 主入口"""
import argparse
import datetime
import logging
import os
import sys
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import numpy as np

from analyzer.data_layer import DataLayer
from analyzer.analyzer import DataAnalyzer
from analyzer.ai import AIClient
from analyzer.event_detector import detect_events
from analyzer.config import *
from analyzer.utils import print_unified_table, resolve_real_price

logger = logging.getLogger(__name__)


def run_batch_analysis(api_key: str = None, target_code: str = None):
    if not api_key:
        print("错误：未设置 DEEPSEEK_API_KEY 环境变量，AI不可用，程序退出。")
        sys.exit(1)

    dl = DataLayer()
    if not dl.login():
        print("登录 baostock 失败")
        sys.exit(1)

    analyzer = DataAnalyzer()
    try:
        etf_list = dl.load_positions()
    except Exception as e:
        print(f"加载持仓文件失败: {e}")
        dl.logout()
        sys.exit(1)

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=HISTORY_DAYS)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    market_df = dl.get_daily_data("sh.000001", start, today_str)
    if market_df is None:
        print("获取大盘数据失败")
        dl.logout()
        sys.exit(1)
    market_df = dl.calculate_indicators(market_df, need_amount_ma=True)
    env = dl.get_market_environment(market_df)
    buy_w, sell_w = dl.select_weights(env["state"])
    analyzer.set_environment(env, buy_w, sell_w)

    state = dl.load_state()
    ai_client = AIClient(api_key)

    dl.append_daily_snapshot(today_str, etf_list)

    if target_code:
        row = etf_list[etf_list["代码"] == target_code]
        if row.empty:
            print(f"未找到代码 {target_code}")
            dl.logout()
            sys.exit(1)
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
        # 详细分析（仅打印评分）
        _, _, new_state, score, _, scan_info = analyzer.analyze_single_etf(
            code, name, real_price, hist, weekly, env, today, s,
            cost_price=cost if pd.notna(cost) else None, shares=int(shares) if pd.notna(shares) else 0
        )
        print(f"详细分析 - 代码: {code}, 评分: {score:.1f}")
        dl.save_state(new_state)
        dl.logout()
        return

    # 并发获取所有ETF的技术指标
    scan_info_list: List[Dict] = []
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

    # 同步用户操作（比较当前份额与上次记录的份额）
    for si in scan_info_list:
        code = si["display"]["code"]
        current_shares = si.get("shares", 0)
        last_known = state.get(code, {}).get("last_known_shares", 0)
        if current_shares != last_known:
            if code not in state:
                state[code] = {}
            state[code]["last_known_shares"] = current_shares
            if si.get("cost_price") is not None:
                state[code]["last_known_cost"] = si["cost_price"]
            if current_shares == 0:
                state[code]["trend_add_count"] = 0
                state[code]["dip_add_count"] = 0
                state[code]["overheat_triggered"] = False
                state[code]["overheat_count"] = 0
                state[code]["position_state"] = "CLEARED"
            elif last_known == 0 and current_shares > 0:
                state[code]["trend_add_count"] = 0
                state[code]["dip_add_count"] = 0
                state[code]["position_state"] = "BASE_HOLD"

    # 主表格
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

    # 事件检测
    event_results: Dict[str, Dict] = {}
    for si in scan_info_list:
        code = si["display"]["code"]
        current_state = state.get(code, {})
        # 准备数据字典
        data_dict = {
            "price": si["display"]["price"],
            "final_score": si.get("final_score", 50),
            "rsi": si.get("rsi", 50),
            "vol_ratio": si.get("vol_ratio", 1.0),
            "profit_pct_from_low": si.get("profit_pct_from_low", 0.0),
            "atr_pct": si.get("atr_pct", 0.02),
            "ma5": si.get("ma5"),
            "ma10": si.get("ma10"),
            "ma20": si.get("ma20"),
            "ma10_trend": si.get("ma10_trend", 0),
            "ma20_trend": si.get("ma20_trend", 0),
            "recent_high_10": si.get("recent_high_10", si["display"]["price"]),
            "recent_high_20": si.get("recent_high_20", si["display"]["price"]),
            "macd_hist": si.get("macd_hist", 0.0),
            "cost_profit_pct": si.get("cost_profit_pct"),
            "shares": si.get("shares", 0),
        }
        # 更新 MACD 历史
        last_macd_hist = current_state.get("last_macd_hist", 0.0)
        current_state["last_macd_hist"] = data_dict["macd_hist"]
        macd_shrink_days = current_state.get("macd_shrink_days", 0)
        if data_dict["macd_hist"] > last_macd_hist:
            macd_shrink_days += 1
        else:
            macd_shrink_days = 0
        current_state["macd_shrink_days"] = macd_shrink_days
        # 调用事件检测
        event, advice, need_update = detect_events(
            code, data_dict, env["state"], current_state
        )
        if need_update:
            # 更新计数器
            if event == "trend_reversal":
                current_state["trend_add_count"] = current_state.get("trend_add_count", 0) + 1
            elif event == "trend_confirm":
                current_state["trend_add_count"] = current_state.get("trend_add_count", 0) + 1
            elif event == "dip":
                current_state["dip_add_count"] = current_state.get("dip_add_count", 0) + 1
            elif event == "overheat":
                current_state["overheat_triggered"] = True
                current_state["overheat_count"] = current_state.get("overheat_count", 0) + 1
            elif event in ("sell_prelim", "sell_confirm"):
                current_state["trend_add_count"] = 0
                current_state["dip_add_count"] = 0
            elif event == "clear":
                current_state["trend_add_count"] = 0
                current_state["dip_add_count"] = 0
                current_state["overheat_triggered"] = False
                current_state["overheat_count"] = 0
                current_state["position_state"] = "CLEARED"
            elif event == "buy":
                current_state["position_state"] = "BASE_HOLD"
                current_state["trend_add_count"] = 0
                current_state["dip_add_count"] = 0
        event_results[code] = {
            "advice": advice,
            "name": si["display"]["name"],
            "score": si["final_score"],
            "event": event,
        }

    # AI 润色
    if ai_client and AI_ENABLE:
        try:
            # 提取建议映射 {code: advice}
            advice_map = {code: data["advice"] for code, data in event_results.items()}
            # 调用 AI 润色
            enhanced_advice = ai_client.get_batch_recommendations(advice_map, env["state"])
            # 更新 event_results 中的 advice
            for code, new_advice in enhanced_advice.items():
                if code in event_results:
                    event_results[code]["advice"] = new_advice
        except Exception as e:
            logger.warning(f"AI润色失败: {e}")

    # 持仓表格
    position_rows = []
    for si in scan_info_list:
        if si.get("shares", 0) > 0 and si.get("cost_price") is not None:
            code = si["display"]["code"]
            cost = si["cost_price"]
            price = si["display"]["price"]
            profit_pct = (price - cost)/cost*100 if cost>0 else 0
            delta, delta_pct = dl.get_position_change(code, si["shares"])
            change_str = f"+{delta}(+{delta_pct:.0f}%)" if delta>0 else (f"{delta}({delta_pct:.0f}%)" if delta!=0 else "0")
            advice = event_results.get(code, {}).get("advice", "")
            position_rows.append({
                "name": si["display"]["name"],
                "code": code,
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

    # 趋势扫描买入推荐
    buy_rows = []
    for si in scan_info_list:
        code = si["display"]["code"]
        if si.get("shares", 0) == 0:
            event = event_results.get(code, {}).get("event", "")
            if event in ("trend_reversal", "trend_confirm", "buy"):
                advice = event_results[code].get("advice", "")
                if "买入" in advice or "加仓" in advice:
                    display_advice = "🔥 强烈推荐买入" if event == "trend_confirm" else "📈 推荐买入"
                    buy_rows.append({
                        "name": si["display"]["name"],
                        "code": code,
                        "price": si["display"]["price"],
                        "change_pct": si["display"]["change_pct"],
                        "final_score": si["final_score"],
                        "advice": display_advice
                    })
    if buy_rows:
        buy_rows.sort(key=lambda x: x["final_score"], reverse=True)
        print_unified_table(buy_rows, title="📊 [趋势扫描] 买入推荐", table_type="trend")

    # 保存状态
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