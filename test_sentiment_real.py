#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情绪因子真实数据测试
用法：python test_sentiment_real.py
将连接 akshare 获取最新市场情绪指标，逐步展示计算过程，
帮助诊断情绪因子是否被异常压制在 0.70。
"""
import sys
import os
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from analyzer.fetcher import DataFetcher, AKSHARE_AVAILABLE
from analyzer.utils import apply_sentiment_adjustment
from analyzer.config import SENTIMENT_LOWER_BOUND

# ---------- 辅助函数：模拟计算过程（从 DataFetcher 中提取） ----------
def compute_sentiment_with_details(fetcher, indicators):
    """复制 fetcher.compute_sentiment_factor 的核心步骤，打印所有中间值"""
    if not indicators:
        print("未获取到任何情绪指标，使用默认值 1.0")
        return 1.0, 1.0

    # 基础权重
    weights = {
        "north": 0.25,
        "main": 0.20,
        "zt_dt": 0.15,
        "up_down": 0.20,
        "volatility": 0.10,
        "margin": 0.10,
    }

    # 高波动调整
    hv = indicators.get("hv", 20.0)
    print(f"\n波动率 hv = {hv:.2f}%")
    if hv > 30:
        print("触发高波动调整：北向、主力权重降低，波动率权重提升")
        weights["north"] *= 0.7
        weights["main"] *= 0.7
        weights["volatility"] = 0.20
        total = sum(weights.values())
        weights = {k: v/total for k, v in weights.items()}
        print(f"调整后权重：{ {k: f'{v:.3f}' for k, v in weights.items()} }")

    # 计算各子得分
    scores = {}
    if "north_net" in indicators and "north_net_20d_avg" in indicators:
        scores["north"] = fetcher._normalize_north(indicators["north_net"], indicators["north_net_20d_avg"])
        print(f"北向得分：净买额 {indicators['north_net']:.1f} 亿, 20日均 {indicators['north_net_20d_avg']:.1f} 亿 -> {scores['north']:.3f}")
    else:
        scores["north"] = 1.0
        print("北向得分：无数据，默认 1.000")

    if "main_net_pct" in indicators:
        scores["main"] = fetcher._normalize_main(indicators["main_net_pct"])
        print(f"主力得分：净流入占比 {indicators['main_net_pct']:.2f}% -> {scores['main']:.3f}")
    else:
        scores["main"] = 1.0
        print("主力得分：无数据，默认 1.000")

    if "zt_dt_ratio" in indicators:
        scores["zt_dt"] = fetcher._normalize_zt_dt(indicators["zt_dt_ratio"])
        print(f"涨跌停比得分：涨跌停比 {indicators['zt_dt_ratio']:.2f} -> {scores['zt_dt']:.3f}")
    else:
        scores["zt_dt"] = 1.0
        print("涨跌停比得分：无数据，默认 1.000")

    if "up_down_ratio" in indicators:
        scores["up_down"] = fetcher._normalize_up_down(indicators["up_down_ratio"])
        print(f"上涨下跌家数比得分：比例 {indicators['up_down_ratio']:.2f} -> {scores['up_down']:.3f}")
    else:
        scores["up_down"] = 1.0
        print("上涨下跌家数比得分：无数据，默认 1.000")

    if "hv" in indicators and "hv_ma20" in indicators:
        scores["volatility"] = fetcher._normalize_volatility(indicators["hv"], indicators["hv_ma20"])
        print(f"波动率得分：hv={indicators['hv']:.1f}%, ma20={indicators['hv_ma20']:.1f}% -> {scores['volatility']:.3f}")
    else:
        scores["volatility"] = 1.0
        print("波动率得分：无数据，默认 1.000")

    if "margin_change" in indicators:
        scores["margin"] = fetcher._normalize_margin(indicators["margin_change"])
        print(f"融资余额得分：变化率 {indicators['margin_change']:.2f}% -> {scores['margin']:.3f}")
    else:
        scores["margin"] = 1.0
        print("融资余额得分：无数据，默认 1.000")

    # 原始情绪
    raw_sentiment = sum(scores[k] * weights[k] for k in weights)
    raw_sentiment = max(0.6, min(1.5, raw_sentiment))
    print(f"\n加权原始情绪 (已截断至[0.6,1.5])：{raw_sentiment:.4f}")

    # 平滑处理
    if fetcher._sentiment_history:
        smoothed = 0.7 * fetcher._sentiment_history[-1] + 0.3 * raw_sentiment
        print(f"平滑处理：历史最后值={fetcher._sentiment_history[-1]:.4f}，平滑后={smoothed:.4f}")
    else:
        smoothed = raw_sentiment
        print("无历史数据，跳过平滑")

    # 非线性调整
    adjusted = apply_sentiment_adjustment(smoothed)
    print(f"非线性调整 (apply_sentiment_adjustment)：{smoothed:.4f} -> {adjusted:.4f}")

    # 最终下限保护
    final_sentiment = max(SENTIMENT_LOWER_BOUND, adjusted)
    if final_sentiment != adjusted:
        print(f"下限保护生效，调整为 {final_sentiment:.4f} (下限={SENTIMENT_LOWER_BOUND})")

    return final_sentiment, raw_sentiment


def print_adjust_curve():
    """打印调整函数在常见区间的映射"""
    print("\n" + "="*60)
    print("调整函数曲线 (apply_sentiment_adjustment)")
    print(" sentiment -> adjusted")
    for s in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00,
              1.05, 1.10, 1.20, 1.30]:
        adj = apply_sentiment_adjustment(s)
        flag = ""
        if adj <= SENTIMENT_LOWER_BOUND + 0.001 and s < 0.9:
            flag = " <-- 可能被压制在下限"
        print(f"  {s:.2f} -> {adj:.4f}{flag}")
    print("="*60)


def main():
    print("情绪因子真实数据测试")
    print("=" * 60)

    if not AKSHARE_AVAILABLE:
        print("未安装 akshare，无法获取真实数据。请执行 pip install akshare")
        return

    fetcher = DataFetcher()
    print("正在从 akshare 获取情绪指标...")
    indicators = fetcher.fetch_sentiment_indicators()

    if not indicators:
        print("获取失败或未返回任何数据，请检查网络或 akshare 版本。")
        return

    print("\n获取到的原始指标：")
    for k, v in indicators.items():
        print(f"  {k}: {v}")

    final, raw = compute_sentiment_with_details(fetcher, indicators)

    print(f"\n最终情绪因子：{final:.4f}")
    print(f"风险提示：{fetcher.get_sentiment_risk_tip(final)}")

    print_adjust_curve()

    # 简单诊断
    if final <= SENTIMENT_LOWER_BOUND + 0.01:
        print("\n⚠️ 警告：最终情绪因子触及下限，可能因调整函数或数据异常导致。")
        if raw < 0.9:
            print("   原始情绪偏低，建议检查各子指标得分是否反映真实市场。")
            print("   若原始情绪在 0.8~0.9 仍被压到 0.70，请调整 apply_sentiment_adjustment 悲观侧参数。")
    else:
        print("\n情绪因子未触及下限，正常。")


if __name__ == "__main__":
    main()