# main.py
# 主程序入口

import datetime
from config import (
    POSITION_FILE, STATE_FILE, MACRO_INDEX, MARKET_INDEX,
    MACRO_MA_SHORT, MACRO_MA_LONG, ETF_MA, ETF_VOL_MA,
    STRATEGY_WEIGHTS, ATR_PERIOD
)
from data import (
    silent_login, silent_logout, load_positions, load_state, save_state,
    get_daily_data, calculate_indicators, get_realtime_price_sina, get_realtime_index_sina,
    get_weekly_data
)
from etf_analysis import analyze_etf_signal

def real_time_analysis():
    print("正在加载ETF列表...")
    try:
        etf_list = load_positions(POSITION_FILE)
    except FileNotFoundError as e:
        print(e)
        return

    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=300)).strftime('%Y-%m-%d')

    # 获取大盘指数数据，用于判断交易日
    market_df = get_daily_data(MARKET_INDEX, start_date, today_str)
    if market_df is None or market_df.empty:
        print("无法获取大盘指数数据")
        return

    if today not in market_df.index:
        print(f"今天 {today_str} 不是交易日，程序退出。")
        return

    # 获取宏观指数数据
    macro_df = get_daily_data(MACRO_INDEX, start_date, today_str)
    if macro_df is None or macro_df.empty:
        print("无法获取宏观指数数据")
        return
    macro_df = calculate_indicators(macro_df, ma_short=MACRO_MA_SHORT)
    macro_df['ma_long'] = macro_df['close'].rolling(window=MACRO_MA_LONG).mean()

    market_df = calculate_indicators(market_df, ma_short=20, vol_ma=20)

    # 宏观状态和情绪函数（内嵌）
    from data import calculate_rsi
    from config import MARKET_STATES, RSI_PERIOD

    def get_macro_status(macro_df):
        latest = macro_df.iloc[-1]
        close = latest["close"]
        ma20 = latest["ma_short"]
        ma60 = latest.get("ma_long", ma20)
        if close > ma20 and close > ma60:
            return "bull", MARKET_STATES["bull"]
        elif close < ma20 and close < ma60:
            return "bear", MARKET_STATES["bear"]
        else:
            return "oscillate", MARKET_STATES["oscillate"]

    def get_sentiment_factor(macro_df):
        if macro_df is None or len(macro_df) < RSI_PERIOD + 1:
            return 1.0
        rsi_series = calculate_rsi(macro_df["close"], RSI_PERIOD)
        latest_rsi = rsi_series.iloc[-1]
        if latest_rsi < 30:
            return 0.6
        elif latest_rsi < 50:
            return 0.8
        elif latest_rsi < 70:
            return 1.0
        else:
            return 0.9

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

    # 加载状态
    all_state = load_state(STATE_FILE)
    for _, row in etf_list.iterrows():
        code = row['代码']
        if code not in all_state:
            all_state[code] = {}

    # 获取所有ETF历史数据（日线）
    etf_hist_cache = {}
    for code in etf_list['代码'].unique():
        df = get_daily_data(code, start_date, today_str)
        if df is not None and not df.empty:
            df = calculate_indicators(df, ma_short=ETF_MA, vol_ma=ETF_VOL_MA)
            etf_hist_cache[code] = df

    # 获取周线数据（用于每个ETF）
    weekly_cache = {}
    for code in etf_list['代码'].unique():
        weekly = get_weekly_data(code, start_date, today_str)
        weekly_cache[code] = weekly

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
        weekly_df = weekly_cache.get(code)
        state = all_state.get(code, {})
        real_price = get_realtime_price_sina(code)
        if real_price is None:
            print(f"{name} ({code}) 获取实时价格失败，跳过")
            continue

        status, signal, new_state = analyze_etf_signal(
            code, name, real_price, hist_df, weekly_df,
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

    save_state(all_state, STATE_FILE)

    if signals:
        print("\n" + "=" * 70)
        print("操作信号汇总")
        print("=" * 70)
        for sig in signals:
            if sig['action'] == 'BUY':
                print(f"买入 {sig['name']} ({sig['code']}) - 建议仓位比例: {sig['ratio']*100:.0f}% (原因: {sig['reason']})")
            elif sig['action'] == 'BUY_QUICK':
                print(f"买入（快速） {sig['name']} ({sig['code']}) - 建议仓位比例: {sig['ratio']*100:.0f}% (原因: {sig['reason']})")
            else:
                if sig.get('is_clear'):
                    print(f"清仓 {sig['name']} ({sig['code']}) - 建议卖出全部持仓 (原因: {sig['reason']})")
                else:
                    print(f"卖出 {sig['name']} ({sig['code']}) - 建议卖出比例: {sig['ratio']*100:.0f}% (原因: {sig['reason']})")
    else:
        print("\n⚪ 无操作信号，所有ETF建议观望")

def main():
    if not silent_login():
        return
    real_time_analysis()
    silent_logout()

if __name__ == '__main__':
    main()