#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统
ATR仓位管理 + 全局仓位约束（按评分分配总仓位上限100%）
趋势扫描使用5日涨跌幅 + 连续2日评分过滤
删除卖出警示表格
"""
import argparse
import datetime
import logging
import os
import sys
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


def calculate_atr_advice(
    score: float,
    atr_pct: float,
    price: float,
    shares: int,
    cost_price: Optional[float] = None,
    daily_change_shares: int = 0,
    target_ratio: Optional[float] = None,   # 外部传入的目标占比（已考虑全局约束）
) -> str:
    """
    基于 ATR 和实际持仓占比的仓位管理建议。
    如果 target_ratio 不为 None，则直接使用该目标占比；否则根据 score 计算。
    """
    is_holding = shares > 0

    def _calc_normal_advice(use_target_ratio: float) -> str:
        current_value = shares * price
        current_ratio = current_value / TOTAL_CAPITAL
        gap = use_target_ratio - current_ratio
        if abs(gap) < 0.01:
            return "持有不动"
        vol_factor = max(0.5, min(1.5, 0.02 / max(atr_pct, 0.01)))
        max_step = 0.10 * vol_factor
        if gap > 0:
            step = min(gap, max_step)
            if current_value > 0:
                pct_of_current = (step * TOTAL_CAPITAL) / current_value * 100
                pct_of_current = min(pct_of_current, 50.0)
                return f"加仓{pct_of_current:.0f}% | 价位：当前价附近"
            else:
                return "持有不动"
        else:
            step = min(-gap, max_step)
            if current_value > 0:
                pct_of_current = (step * TOTAL_CAPITAL) / current_value * 100
                pct_of_current = min(pct_of_current, 50.0)
                if pct_of_current >= 80:
                    return f"清仓 | 价位：{price:.3f}附近"
                return f"减仓{pct_of_current:.0f}% | 价位：{price:.3f}附近"
            else:
                return "持仓异常"

    # 未持仓
    if not is_holding:
        if target_ratio is not None and target_ratio <= 0:
            return "推荐但无仓位（总仓位已满）"
        if score < 50:
            return "买入 | 程度：谨慎"
        elif score < 70:
            return "买入 | 程度：推荐"
        else:
            return "买入 | 程度：强烈推荐"

    # 已持仓：确定目标占比
    if target_ratio is None:
        target_ratio = max(0.0, min(MAX_POSITION_PCT, (score - 30) / 70 * MAX_POSITION_PCT))
    normal_advice = _calc_normal_advice(target_ratio)

    if daily_change_shares == 0:
        return normal_advice

    op_text = f"已加仓{daily_change_shares}股" if daily_change_shares > 0 else f"已减仓{-daily_change_shares}股"
    if normal_advice == "持有不动":
        return op_text
    else:
        return f"{op_text}，{normal_advice}"


def run_batch_analysis(api_key=None, target_code=None):
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

    # ---------- 主表格 ----------
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
            position_change_map[si.get("display",{}).get("code","")] = delta

    # ---------- 趋势扫描候选（使用5日涨跌幅 + 连续2日评分过滤） ----------
    BUY_MAX_COUNT = 6
    BUY_LOW_PROFIT_MIN = 0.02
    BUY_LOW_PROFIT_MAX = 0.35
    BUY_MAX_PULLBACK = 0.08
    BUY_DAILY_GAIN_MIN = -0.15
    BUY_DAILY_GAIN_MAX = 0.30

    LEFT_MAX_COUNT = 4
    LEFT_DAILY_GAIN_MIN = -0.15
    LEFT_DAILY_GAIN_MAX = 0.30
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

    # 复制列表并将 change_pct 替换为 change_5d
    scan_info_list_5d = []
    for si in scan_info_list:
        new_si = si.copy()
        new_si["change_pct"] = si.get("change_5d", 0.0)
        scan_info_list_5d.append(new_si)

    right_buy_indices = select_trend_buy(
        scan_info_list_5d,
        max_count=BUY_MAX_COUNT,
        low_profit_min=BUY_LOW_PROFIT_MIN,
        low_profit_max=BUY_LOW_PROFIT_MAX,
        max_pullback=BUY_MAX_PULLBACK,
        daily_gain_min=BUY_DAILY_GAIN_MIN,
        daily_gain_max=BUY_DAILY_GAIN_MAX,
        prefer_signal=True
    )
    right_buy_indices = [idx for idx in right_buy_indices if scan_info_list[idx].get("shares", 0) == 0]

    scan_info_list_5d_left = []
    for si in scan_info_list:
        new_si = si.copy()
        new_si["change_pct"] = si.get("change_5d", 0.0)
        scan_info_list_5d_left.append(new_si)
    left_buy_indices = select_left_buy(
        scan_info_list_5d_left,
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

    # 评分连续性过滤（连续2日评分≥65且历史记录≥2天）
    SCORE_CONSECUTIVE_DAYS = 2
    SCORE_MIN_THRESHOLD = 65
    def filter_by_consecutive_score(indices, is_buy=True):
        filtered = []
        for idx in indices:
            code = scan_info_list[idx]["display"]["code"]
            hist_scores = state.get(code, {}).get("score_history", [])
            if len(hist_scores) < SCORE_CONSECUTIVE_DAYS:
                continue
            today_score = scan_info_list[idx]["final_score"]
            today_str_date = today.strftime("%Y-%m-%d")
            yesterday_score = None
            for rec in hist_scores:
                if rec["date"] != today_str_date:
                    yesterday_score = rec["score"]
                    break
            if (today_score >= SCORE_MIN_THRESHOLD and 
                yesterday_score is not None and 
                yesterday_score >= SCORE_MIN_THRESHOLD):
                filtered.append(idx)
        return filtered

    right_buy_indices = filter_by_consecutive_score(right_buy_indices)
    left_buy_indices = filter_by_consecutive_score(left_buy_indices)

    # 构建所有需要关注的ETF（用于全局仓位分配）
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

    # ---------- 全局仓位约束（方案A） ----------
    # 收集所有需要参与分配的目标（持仓ETF + 评分>=50的未持仓候选）
    constraint_items = []
    for code, si in all_need_recommend.items():
        score = si["final_score"]
        shares = si.get("shares", 0)
        price = si["display"]["price"]
        current_value = shares * price
        current_ratio = current_value / TOTAL_CAPITAL
        raw_target = max(0.0, min(MAX_POSITION_PCT, (score - 30) / 70 * MAX_POSITION_PCT))
        # 只考虑持仓ETF 或 评分>=50的未持仓ETF（潜在买入）
        if shares > 0 or score >= 50:
            constraint_items.append({
                "code": code,
                "score": score,
                "current_ratio": current_ratio,
                "raw_target": raw_target,
                "shares": shares,
                "si": si,
            })
    # 按评分降序排序
    constraint_items.sort(key=lambda x: x["score"], reverse=True)
    # 分配仓位：总上限100%
    total_limit = 1.0
    remaining = total_limit
    adjusted_targets = {}
    for item in constraint_items:
        # 持仓ETF至少保留当前仓位，但目标不能低于当前（减仓另行处理），这里分配上限为目标值
        alloc = min(item["raw_target"], remaining)
        adjusted_targets[item["code"]] = alloc
        remaining -= alloc
        if remaining <= 0:
            break
    # 对于未分配到的，若当前仓位>0则保持当前仓位，否则目标为0
    for item in constraint_items:
        if item["code"] not in adjusted_targets:
            adjusted_targets[item["code"]] = item["current_ratio"] if item["shares"] > 0 else 0.0

    # 准备AI数据，并计算系统建议（使用调整后的目标占比）
    etf_dict_for_ai = {}
    sys_advice_map = {}
    for code, si in all_need_recommend.items():
        d = si["display"]
        score = si["final_score"]
        atr_pct = si.get("atr_pct", 0.02)
        price = d["price"]
        shares = si.get("shares", 0)
        cost = si.get("cost_price")
        daily_change = position_change_map.get(code, 0)
        target_ratio = adjusted_targets.get(code)
        sys_advice = calculate_atr_advice(
            score=score,
            atr_pct=atr_pct,
            price=price,
            shares=shares,
            cost_price=cost,
            daily_change_shares=daily_change,
            target_ratio=target_ratio,
        )
        sys_advice_map[code] = sys_advice

        etf_dict_for_ai[code] = {
            "name": d["name"],
            "final_score": to_py_float(score),
            "rsi": to_py_float(si.get("rsi", 50)),
            "vol_ratio": to_py_float(si.get("vol_ratio", 1.0)),
            "change_pct": to_py_float(si.get("change_5d", 0.0)),
            "above_ma": to_py_bool(si.get("above_ma", False)),
            "profit_pct_from_low": to_py_float(si.get("profit_pct_from_low", 0)),
            "shares": shares,
            "cost_price": cost,
            "risk_str": d.get("risk_str", ""),
            "price": price,
            "change_pct_display": d["change_pct"],
            "atr_pct": to_py_float(atr_pct),
            "sys_advice": sys_advice,
            "daily_change": daily_change,
        }

    # 调用AI
    try:
        batch_advice = ai_client.get_batch_recommendations(etf_dict_for_ai, env["state"])
    except Exception as e:
        print(f"AI调用失败，程序退出: {e}")
        dl.logout()
        sys.exit(1)

    final_advice_map = batch_advice

    # 更新持仓表格建议
    for row in position_rows:
        code = row["code"]
        if code in final_advice_map:
            row["advice"] = final_advice_map[code]

    if position_rows:
        print_unified_table(position_rows, title="📋 当前持仓详情", table_type="position")

    # 趋势扫描买入推荐（仅输出）
    buy_rows = []
    for code, advice_text in final_advice_map.items():
        if code not in all_need_recommend:
            continue
        si = all_need_recommend[code]
        d = si["display"]
        is_holding = si.get("shares", 0) > 0
        if not is_holding and "买入" in advice_text:
            if "强烈推荐" in advice_text:
                display_advice = "🔥 强烈推荐买入"
            elif "推荐" in advice_text:
                display_advice = "📈 推荐买入"
            else:
                display_advice = "💡 谨慎买入"
            buy_rows.append({
                "name": d["name"],
                "code": d["code"],
                "price": d["price"],
                "change_pct": d["change_pct"],
                "final_score": si["final_score"],
                "advice": display_advice
            })

    buy_rows.sort(key=lambda x: x["final_score"], reverse=True)

    if buy_rows:
        print_unified_table(buy_rows, title="📊 [趋势扫描] 买入推荐", table_type="trend")

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