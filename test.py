#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
情绪指标数据源可用性测试脚本
运行：python test_sentiment_sources.py
"""

import datetime
import pandas as pd

try:
    import akshare as ak
    print("✅ akshare 已安装，版本：", ak.__version__)
except ImportError:
    print("❌ 未安装 akshare，请执行 pip install akshare")
    exit(1)

def test_north_flow():
    """测试北向资金"""
    print("\n📊 测试 北向资金 (stock_hsgt_hist_em) ...")
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        if df.empty:
            print("   ⚠️ 返回空 DataFrame")
            return
        print(f"   数据行数: {len(df)}")
        print(f"   列名: {list(df.columns)}")
        latest = df.iloc[-1]
        print(f"   最新日期: {latest.get('日期', 'N/A')}")
        print(f"   当日成交净买额: {latest.get('当日成交净买额', 'N/A')}")
        # 检查关键字段是否存在
        if '当日成交净买额' in df.columns:
            print("   ✅ 字段 '当日成交净买额' 存在")
        else:
            print("   ❌ 字段 '当日成交净买额' 不存在")
    except Exception as e:
        print(f"   ❌ 接口异常: {e}")

def test_main_fund_flow():
    """测试主力资金流向"""
    print("\n📊 测试 主力资金流向 (stock_market_fund_flow) ...")
    try:
        df = ak.stock_market_fund_flow()
        if df.empty:
            print("   ⚠️ 返回空 DataFrame")
            return
        print(f"   数据行数: {len(df)}")
        print(f"   列名: {list(df.columns)}")
        latest = df.iloc[-1]
        print(f"   最新日期: {latest.get('日期', 'N/A')}")
        # 检查主力净流入-净占比字段
        col_candidates = ["主力净流入-净占比", "主力净流入-净占比(%)", "主力净流入净占比"]
        found = False
        for col in col_candidates:
            if col in latest.index:
                print(f"   ✅ 字段 '{col}' 存在，值: {latest[col]}")
                found = True
                break
        if not found:
            print(f"   ❌ 未找到主力净流入占比字段，现有字段: {list(latest.index)}")
    except Exception as e:
        print(f"   ❌ 接口异常: {e}")

def test_limit_pool():
    """测试涨跌停池"""
    print("\n📊 测试 涨跌停池 (stock_zt_pool_em / stock_zt_pool_dtgc_em) ...")
    today_str = datetime.datetime.now().strftime("%Y%m%d")
    # 尝试最近几个交易日
    for offset in [0, 1, 2]:
        date_str = (datetime.datetime.now() - datetime.timedelta(days=offset)).strftime("%Y%m%d")
        try:
            df_zt = ak.stock_zt_pool_em(date=date_str)
            df_dt = ak.stock_zt_pool_dtgc_em(date=date_str)
            zt_cnt = len(df_zt) if not df_zt.empty else 0
            dt_cnt = len(df_dt) if not df_dt.empty else 0
            print(f"   日期 {date_str}: 涨停 {zt_cnt} 只, 跌停 {dt_cnt} 只")
            if zt_cnt > 0 or dt_cnt > 0:
                print(f"   ✅ 该日期数据有效")
                # 显示列名示例
                if not df_zt.empty:
                    print(f"      涨停池列名: {list(df_zt.columns)}")
                break
        except Exception as e:
            print(f"   日期 {date_str} 异常: {e}")

def test_market_breadth():
    """测试全市场涨跌家数（通过 stock_zh_a_spot）"""
    print("\n📊 测试 全市场涨跌家数 (stock_zh_a_spot) ...")
    try:
        df = ak.stock_zh_a_spot()
        if df.empty:
            print("   ⚠️ 返回空 DataFrame")
            return
        print(f"   股票总数: {len(df)}")
        print(f"   列名: {list(df.columns)}")
        if '涨跌幅' in df.columns:
            up = len(df[df['涨跌幅'] > 0])
            down = len(df[df['涨跌幅'] < 0])
            print(f"   上涨家数: {up}, 下跌家数: {down}")
            print("   ✅ 可正常计算上涨下跌比")
        else:
            print("   ❌ 缺少 '涨跌幅' 字段")
    except Exception as e:
        print(f"   ❌ 接口异常: {e}")

def test_margin_balance():
    """测试融资余额"""
    print("\n📊 测试 融资余额 (stock_margin_sse) ...")
    try:
        start = (datetime.datetime.now() - datetime.timedelta(days=10)).strftime("%Y%m%d")
        end = datetime.datetime.now().strftime("%Y%m%d")
        df = ak.stock_margin_sse(start_date=start, end_date=end)
        if df.empty:
            print("   ⚠️ 返回空 DataFrame")
            return
        print(f"   数据行数: {len(df)}")
        print(f"   列名: {list(df.columns)}")
        # 检查融资余额字段
        col_candidates = ['融资余额', 'rzye']
        found = False
        for col in col_candidates:
            if col in df.columns:
                print(f"   ✅ 字段 '{col}' 存在，最新值: {df[col].iloc[-1]}")
                found = True
                break
        if not found:
            print(f"   ❌ 未找到融资余额字段")
    except Exception as e:
        print(f"   ❌ 接口异常: {e}")

def test_alternative_volatility():
    """测试替代波动率（沪深300历史波动率）"""
    print("\n📊 测试 沪深300历史波动率 (stock_zh_index_daily) ...")
    try:
        df = ak.stock_zh_index_daily(symbol="sh000300")
        if df.empty:
            print("   ⚠️ 返回空 DataFrame")
            return
        print(f"   数据行数: {len(df)}")
        print(f"   列名: {list(df.columns)}")
        if 'close' in df.columns:
            df = df.sort_values('date')
            df['ret'] = df['close'].pct_change()
            hv = df['ret'].rolling(20).std().iloc[-1] * (252**0.5) * 100
            print(f"   最新20日年化波动率: {hv:.2f}%")
            print("   ✅ 可正常计算历史波动率")
        else:
            print("   ❌ 缺少 'close' 字段")
    except Exception as e:
        print(f"   ❌ 接口异常: {e}")

if __name__ == "__main__":
    test_north_flow()
    test_main_fund_flow()
    test_limit_pool()
    test_market_breadth()
    test_margin_balance()
    test_alternative_volatility()

    print("\n🎯 测试完成，请根据输出选择可用的接口组合。")