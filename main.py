# main.py
# 主程序入口

import datetime
import sys
import config

from config import (
    POSITION_FILE, STATE_FILE, INITIAL_CAPITAL, MAX_SINGLE_POSITION,
    BACKTEST_START, BACKTEST_END, STRATEGY_WEIGHTS
)
from data import (
    silent_login, silent_logout, load_positions, load_state, save_state,
    get_daily_data, calculate_indicators, get_realtime_price_sina, get_realtime_index_sina
)
from etf_analysis import analyze_etf_signal
from backtest import optimize_parameters
from ai import deepseek_generate_weights

def real_time_analysis():
    """实时分析模式"""
    print("正在加载ETF列表...")
    try:
        etf_list = load_positions(POSITION_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=300)).strftime('%Y-%m-%d')

    # 获取宏观指数和大盘数据
    macro_df = get_daily_data(config.MACRO_INDEX, start_date, today_str)
    if macro_df is None or macro_df.empty:
        print("无法获取宏观指数数据")
        return
    macro_df = calculate_indicators(macro_df, ma_short=config.MACRO_MA_SHORT)
    macro_df['ma_long'] = macro_df['close'].rolling(window=config.MACRO_MA_LONG).mean()

    market_df = get_daily_data(config.MARKET_INDEX, start_date, today_str)
    if market_df is None or market_df.empty:
        print("无法获取大盘指数数据")
        return
    market_df = calculate_indicators(market_df, ma_short=20, vol_ma=20)

    # 计算宏观状态和情绪
    from backtest import get_macro_status, get_sentiment_factor
    macro_status, market_factor = get_macro_status(macro_df)
    sentiment_factor = get_sentiment_factor(macro_df)

    market_latest = market_df.iloc[-1]
    market_close = market_latest['close']
    market_ma20 = market_latest['ma_short']
    market_ma60 = market_latest.get('ma_long', market_ma20)
    market_above_ma20 = market_close > market_ma20
    market_above_ma60 = market_close > market_ma60
    market_amount = market_latest['amount']
    market_amount_ma20 = market_latest['amount_ma']
    market_amount_above_ma20 = market_amount > market_amount_ma20

    if len(market_df) >= 5:
        ret_market_5d = (market_close / market_df.iloc[-5]['close']) - 1
    else:
        ret_market_5d = 0

    # 动态权重（可选）
    use_ai = input("是否使用AI动态权重？(y/n): ").strip().lower() == 'y'
    if use_ai:
        api_key = input("请输入DeepSeek API Key: ").strip()
        weights = deepseek_generate_weights(
            macro_status, sentiment_factor,
            market_above_ma20, market_above_ma60, market_amount_above_ma20,
            api_key
        )
        if weights:
            global STRATEGY_WEIGHTS
            STRATEGY_WEIGHTS.update(weights)
            print("已更新动态权重")
        else:
            print("使用默认权重")

    # 加载状态
    all_state = load_state(STATE_FILE)
    for _, row in etf_list.iterrows():
        code = row['代码']
        if code not in all_state:
            all_state[code] = {}

    # 获取所有ETF历史数据
    etf_hist_cache = {}
    for code in etf_list['代码'].unique():
        df = get_daily_data(code, start_date, today_str)
        if df is not None and not df.empty:
            df = calculate_indicators(df, ma_short=config.ETF_MA, vol_ma=config.ETF_VOL_MA)
            etf_hist_cache[code] = df

    real_index = get_realtime_index_sina("sh000001")
    index_info = f"实时大盘: {real_index:.2f}" if real_index else "实时大盘: 获取失败"
    print(f"\n日期：{today_str} | {index_info}")
    print(f"宏观状态：{macro_status.upper()} | 市场因子：{market_factor:.2f} | 情绪系数：{sentiment_factor:.2f}")
    print(f"大盘站上20日线：{'是' if market_above_ma20 else '否'} | 站上60日线：{'是' if market_above_ma60 else '否'}")
    print(f"市场成交额：{market_amount/1e8:.2f}亿 | 20日均额：{market_amount_ma20/1e8:.2f}亿 | 大于均额：{'是' if market_amount_above_ma20 else '否'}")
    print("=" * 70)

    signals = []
    for _, row in etf_list.iterrows():
        code = row['代码']
        name = row['名称']
        hist_df = etf_hist_cache.get(code)
        state = all_state.get(code, {})
        # 获取实时价格（这里使用新浪接口，可替换为其他）
        real_price = get_realtime_price_sina(code)
        if real_price is None:
            print(f"{name} ({code}) 获取实时价格失败，跳过")
            continue

        status, signal, new_state = analyze_etf_signal(
            code, name, real_price, hist_df,
            macro_status, market_factor, sentiment_factor,
            market_above_ma20, market_above_ma60, market_amount_above_ma20,
            ret_market_5d, today, state,
            weights=STRATEGY_WEIGHTS
        )
        print(status)
        print("-" * 40)
        all_state[code] = new_state
        if signal:
            signals.append({**signal, 'code': code, 'name': name})

    # 保存状态
    save_state(all_state, STATE_FILE)

    # 信号汇总
    if signals:
        print("\n" + "=" * 70)
        print("操作信号汇总")
        print("=" * 70)
        for sig in signals:
            if sig['action'] == 'BUY':
                print(f"买入 {sig['name']} ({sig['code']}) - 建议仓位比例: {sig['ratio']*100:.0f}% (原因: {sig['reason']})")
            else:
                if sig.get('is_clear'):
                    print(f"清仓 {sig['name']} ({sig['code']}) - 建议卖出全部持仓 (原因: {sig['reason']})")
                else:
                    print(f"卖出 {sig['name']} ({sig['code']}) - 建议卖出比例: {sig['ratio']*100:.0f}% (原因: {sig['reason']})")
    else:
        print("\n⚪ 无操作信号，所有ETF建议观望")

def backtest_optimization():
    """回测优化模式"""
    test_codes = input("请输入要回测的ETF代码（逗号分隔，如 sh.512480,sh.510300）: ").split(',')
    test_codes = [c.strip() for c in test_codes]
    if not test_codes:
        print("未输入ETF代码，退出")
        return

    param_grid = {
        'etf_ma': [20, 30, 40],
        'etf_vol_ma': [5, 10],
        'confirm_days': [3, 5],
        'buy_threshold': [0.5, 0.6, 0.7],
        'sell_threshold': [-0.2, -0.3, -0.4],
        'trailing_stop_half': [0.05, 0.07],
        'trailing_stop_clear': [0.08, 0.10],
        'profit_targets': [[(0.20,0.3), (0.40,0.3)], [(0.15,0.2), (0.30,0.3)]],
        'macro_ma_short': [20],
        'macro_ma_long': [60],
        'recent_high_window': [10],
        'recent_low_window': [20],
        'weights': [{k: v for k, v in STRATEGY_WEIGHTS.items()}]  # 暂不优化权重
    }

    best_params, best_score = optimize_parameters(
        test_codes, BACKTEST_START, BACKTEST_END, param_grid,
        INITIAL_CAPITAL, MAX_SINGLE_POSITION
    )
    if best_params:
        print(f"\n最优参数组合 (夏普比率 {best_score:.3f}):")
        import json
        print(json.dumps(best_params, indent=2, default=str))
    else:
        print("未找到有效参数组合")

def main():
    if not silent_login():
        return

    print("请选择模式:")
    print("1 - 实时分析")
    print("2 - 回测优化")
    choice = input("输入1或2: ").strip()
    if choice == '1':
        real_time_analysis()
    elif choice == '2':
        backtest_optimization()
    else:
        print("无效输入")

    silent_logout()

if __name__ == '__main__':
    main()