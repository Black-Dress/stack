import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
import json

def test_akshare_sentiment_apis_final():
    """
    AKShare 情绪指标接口最终测试版
    """
    print("=" * 70)
    print("AKShare 情绪指标接口测试 (最终版)")
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    results = {}

    # ---------- 1. 直接的情绪指标 ----------
    print("\n【1. 直接情绪指标】")
    
    # 1.1 A股新闻情绪指数
    # 接口存在，但数据源可能不稳定。捕获更详细的异常信息。
    try:
        df = ak.index_news_sentiment_scope()
        print(f"  ✓ index_news_sentiment_scope: 获取成功, 共 {len(df)} 条记录")
        print(f"    字段: {list(df.columns)}")
        print(f"    最近数据:\n{df.tail(3).to_string(index=False)}")
        results["news_sentiment"] = df
    except Exception as e:
        error_msg = str(e)
        if "Expecting value" in error_msg:
            print(f"  ✗ index_news_sentiment_scope: 数据源返回无效JSON，可能暂时不可用")
        elif "Connection" in error_msg:
            print(f"  ✗ index_news_sentiment_scope: 网络连接失败")
        else:
            print(f"  ✗ index_news_sentiment_scope: 失败 - {error_msg[:60]}")

    # 1.2 恐惧贪婪指数 (已失效，使用替代指标)
    # 原接口 index_fear_greed_funddb 已从 AKShare 移除
    print("  ⚠ index_fear_greed_funddb: AKShare 已移除该接口")
    # 替代指标：机构调研热度
    try:
        df = ak.stock_comment_detail_zlkp_jgcyd_em(symbol="300750") # 示例: 宁德时代
        print(f"  ✓ stock_comment_detail_zlkp_jgcyd_em (机构调研): 获取成功, 共 {len(df)} 条记录")
        print(f"    字段: {list(df.columns)}")
        results["institution_research"] = df
    except Exception as e:
        print(f"  ✗ stock_comment_detail_zlkp_jgcyd_em: 失败 - {str(e)[:60]}")

    # ---------- 2. 资金流向类 ----------
    print("\n【2. 资金流向指标】")
    
    # 2.1 北向资金实时净流入
    # 接口名正确，但直接调用可能失败，增加备选调用方式
    try:
        df = ak.stock_hsgt_north_net_flow_in_em(symbol="北上")
        print(f"  ✓ stock_hsgt_north_net_flow_in_em: 获取成功, 共 {len(df)} 条记录")
        print(f"    字段: {list(df.columns)}")
        print(f"    最近数据:\n{df.tail(3).to_string(index=False)}")
        results["north_flow"] = df
    except AttributeError:
        # 如果直接调用失败，尝试通过模块别名调用
        try:
            import akshare.stock.hsgt as hsgt
            df = hsgt.stock_hsgt_north_net_flow_in_em(symbol="北上")
            print(f"  ✓ (别名调用) stock_hsgt_north_net_flow_in_em: 获取成功, 共 {len(df)} 条记录")
            results["north_flow"] = df
        except Exception as e2:
            print(f"  ✗ stock_hsgt_north_net_flow_in_em (别名调用): 失败 - {str(e2)[:60]}")
    except Exception as e:
        print(f"  ✗ stock_hsgt_north_net_flow_in_em: 失败 - {str(e)[:60]}")

    # 2.2 北向资金历史数据
    try:
        df = ak.stock_hsgt_hist_em(symbol="北向资金")
        print(f"  ✓ stock_hsgt_hist_em: 获取成功, 共 {len(df)} 条记录")
        results["north_flow_hist"] = df
    except Exception as e:
        print(f"  ✗ stock_hsgt_hist_em: 失败 - {str(e)[:60]}")

    # 2.3 大盘资金流向
    try:
        df = ak.stock_market_fund_flow()
        print(f"  ✓ stock_market_fund_flow: 获取成功, 共 {len(df)} 条记录")
        results["market_fund_flow"] = df
    except Exception as e:
        print(f"  ✗ stock_market_fund_flow: 失败 - {str(e)[:60]}")

    # ---------- 3. 市场行为类 ----------
    print("\n【3. 市场行为指标】")
    
    # 3.1 市场总貌（上证）
    try:
        df = ak.stock_sse_summary()
        print(f"  ✓ stock_sse_summary: 获取成功, 共 {len(df)} 条记录")
        results["sse_summary"] = df
    except Exception as e:
        print(f"  ✗ stock_sse_summary: 失败 - {str(e)[:60]}")

    # 3.2 融资融券汇总
    try:
        df = ak.stock_margin_sse(start_date="2024-01-01")
        if df is not None and not df.empty:
            if '融资余额' not in df.columns:
                print(f"  ⚠ stock_margin_sse: 数据格式可能已变更，返回列: {list(df.columns)}")
            else:
                print(f"  ✓ stock_margin_sse: 获取成功, 共 {len(df)} 条记录")
            results["margin_sse"] = df
        else:
            print(f"  ✗ stock_margin_sse: 返回空数据")
    except Exception as e:
        print(f"  ✗ stock_margin_sse: 失败 - {str(e)[:60]}")

    # 3.3 涨停板池
    try:
        df = ak.stock_zt_pool_em(date=datetime.now().strftime("%Y%m%d"))
        print(f"  ✓ stock_zt_pool_em: 获取成功, 共 {len(df)} 条记录")
        results["zt_pool"] = df
    except Exception as e:
        print(f"  ✗ stock_zt_pool_em: 失败 - {str(e)[:60]}")

    # 3.4 跌停板池 (更新接口名)
    try:
        df = ak.stock_zt_pool_dtgc_em(date=datetime.now().strftime("%Y%m%d"))
        print(f"  ✓ stock_zt_pool_dtgc_em: 获取成功, 共 {len(df)} 条记录")
        results["dt_pool"] = df
    except Exception as e:
        print(f"  ✗ stock_zt_pool_dtgc_em: 失败 - {str(e)[:60]}")

    # ---------- 4. 舆情与关注度 ----------
    print("\n【4. 舆情与关注度指标】")
    
    # 4.1 个股人气榜
    try:
        df = ak.stock_hot_rank_detail_realtime_em(symbol="SZ000665")
        print(f"  ✓ stock_hot_rank_detail_realtime_em: 获取成功, 共 {len(df)} 条记录")
        results["hot_rank"] = df
    except Exception as e:
        print(f"  ✗ stock_hot_rank_detail_realtime_em: 失败 - {str(e)[:60]}")

    # 4.2 百度热搜股票
    try:
        df = ak.stock_hot_search_baidu(symbol="A股", date=datetime.now().strftime("%Y%m%d"))
        print(f"  ✓ stock_hot_search_baidu: 获取成功, 共 {len(df)} 条记录")
        results["baidu_hot"] = df
    except Exception as e:
        print(f"  ✗ stock_hot_search_baidu: 失败 - {str(e)[:60]}")

    # 4.3 雪球热榜 (已失效，跳过)
    print("  ⚠ stock_hot_tweet_xueqiu: AKShare 已移除相关接口")

    # ---------- 5. 波动率类 ----------
    print("\n【5. 波动率指标】")
    
    # 5.1 50ETF 期权波动率指数 (更新接口名)
    try:
        df = ak.index_option_50etf_qvix()
        print(f"  ✓ index_option_50etf_qvix: 获取成功, 共 {len(df)} 条记录")
        print(f"    字段: {list(df.columns)}")
        print(f"    最近数据:\n{df.tail(3).to_string(index=False)}")
        results["vix_50etf"] = df
    except Exception as e:
        print(f"  ✗ index_option_50etf_qvix: 失败 - {str(e)[:60]}")

    # ---------- 测试汇总 ----------
    print("\n" + "=" * 70)
    print(f"【测试汇总】共测试 14 个接口，成功 {len(results)} 个")
    print("成功的接口:")
    for name, df in results.items():
        print(f"  - {name}: {len(df)} 条记录")
    print("=" * 70)

    return results

# 运行测试
if __name__ == "__main__":
    test_results = test_akshare_sentiment_apis_final()