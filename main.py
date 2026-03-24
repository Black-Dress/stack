# main.py
# 主程序入口

import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor

from ai import deepseek_generate_weights
from config import (
    POSITION_FILE, STATE_FILE, MACRO_INDEX, MARKET_INDEX,
    MACRO_MA_SHORT, MACRO_MA_LONG, ETF_MA, ETF_VOL_MA,
    STRATEGY_WEIGHTS, RSI_PERIOD, MARKET_STATES
)
from data import (
    silent_login, silent_logout, load_positions, load_state, save_state,
    get_daily_data, calculate_indicators, get_realtime_price_sina, get_realtime_index_sina,
    get_weekly_data, get_macro_status, get_sentiment_factor
)
from etf_analysis import analyze_etf_signal

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _get_market_data(start_date, today_str):
    market_df = get_daily_data(MARKET_INDEX, start_date, today_str)
    if market_df is None or market_df.empty:
        raise ValueError("无法获取大盘指数数据")
    
    macro_df = get_daily_data(MACRO_INDEX, start_date, today_str)
    if macro_df is None or macro_df.empty:
        raise ValueError("无法获取宏观指数数据")
    macro_df = calculate_indicators(macro_df, ma_short=MACRO_MA_SHORT)
    macro_df['ma_long'] = macro_df['close'].rolling(window=MACRO_MA_LONG).mean()

    market_df = calculate_indicators(market_df, ma_short=20, vol_ma=20)

    return macro_df, market_df

def _get_market_status(macro_df, market_df):
    macro_status, market_factor = get_macro_status(macro_df, MARKET_STATES)
    sentiment_factor = get_sentiment_factor(macro_df, RSI_PERIOD)

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

    return {
        "macro_status": macro_status,
        "market_factor": market_factor,
        "sentiment_factor": sentiment_factor,
        "market_above_ma20": market_above_ma20,
        "market_above_ma60": market_above_ma60,
        "market_amount_above_ma20": market_amount_above_ma20,
        "ret_market_5d": ret_market_5d,
        "market_close": market_close,
        "market_amount": market_amount,
        "market_amount_ma20": market_amount_ma20,
    }

def _fetch_etf_data(etf_list, start_date, today_str):
    etf_hist_cache = {}
    weekly_cache = {}
    for code in etf_list['代码'].unique():
        df = get_daily_data(code, start_date, today_str)
        if df is not None and not df.empty:
            df = calculate_indicators(df, ma_short=ETF_MA, vol_ma=ETF_VOL_MA)
            etf_hist_cache[code] = df
        weekly = get_weekly_data(code, start_date, today_str)
        weekly_cache[code] = weekly
    return etf_hist_cache, weekly_cache

def _analyze_etf_parallel(etf_list, etf_hist_cache, weekly_cache, state_dict, market_status, today, weights):
    signals = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for _, row in etf_list.iterrows():
            code = row['代码']
            name = row['名称']
            hist_df = etf_hist_cache.get(code)
            weekly_df = weekly_cache.get(code)
            state = state_dict.get(code, {})
            futures.append(executor.submit(
                _analyze_single_etf,
                code, name, hist_df, weekly_df, state,
                market_status, today, weights
            ))
        for future in futures:
            try:
                status, signal, code, name, new_state = future.result()
                print(status)
                print("-" * 40)
                state_dict[code] = new_state
                if signal:
                    signal['name'] = name
                    signal['code'] = code
                    signals.append(signal)
            except Exception as e:
                logger.error(f"分析ETF时出错: {e}")
    return signals, state_dict

def _analyze_single_etf(code, name, hist_df, weekly_df, state, market_status, today, weights):
    from etf_analysis import analyze_etf_signal
    real_price = get_realtime_price_sina(code)
    if real_price is None:
        return f"【{name} ({code})】\n  获取实时价格失败", None, code, name, state

    status, signal, new_state = analyze_etf_signal(
        code, name, real_price, hist_df, weekly_df,
        market_status['macro_status'],
        market_status['market_factor'],
        market_status['sentiment_factor'],
        market_status['market_above_ma20'],
        market_status['market_above_ma60'],
        market_status['market_amount_above_ma20'],
        market_status['ret_market_5d'],
        today, state, weights
    )
    return status, signal, code, name, new_state

def _output_signals(signals):
    if not signals:
        print("\n⚪ 无操作信号，所有ETF建议观望")
        return
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

def real_time_analysis():
    logger.info("开始实时分析")
    try:
        etf_list = load_positions(POSITION_FILE)
    except FileNotFoundError as e:
        logger.error(e)
        return

    today = datetime.date.today()
    today_str = today.strftime('%Y-%m-%d')
    start_date = (today - datetime.timedelta(days=300)).strftime('%Y-%m-%d')

    try:
        macro_df, market_df = _get_market_data(start_date, today_str)
    except ValueError as e:
        logger.error(e)
        return

    market_status = _get_market_status(macro_df, market_df)

    all_state = load_state(STATE_FILE)
    for _, row in etf_list.iterrows():
        code = row['代码']
        if code not in all_state:
            all_state[code] = {}

    etf_hist_cache, weekly_cache = _fetch_etf_data(etf_list, start_date, today_str)

    # 获取动态权重
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        weights = deepseek_generate_weights(
            market_status['macro_status'],
            market_status['sentiment_factor'],
            market_status['market_above_ma20'],
            market_status['market_above_ma60'],
            market_status['market_amount_above_ma20'],
            api_key, use_cache=True, model="deepseek-lite"
        )
        if weights:
            STRATEGY_WEIGHTS.update(weights)
            logger.info("已更新动态权重")
        else:
            logger.warning("使用默认权重")
    else:
        logger.warning("未设置DEEPSEEK_API_KEY，使用默认权重")

    real_index = get_realtime_index_sina("sh000001")
    index_info = f"实时大盘: {real_index:.2f}" if real_index else "实时大盘: 获取失败"
    print(f"\n日期：{today_str} | {index_info}")
    print(f"宏观状态：{market_status['macro_status'].upper()} | 市场因子：{market_status['market_factor']:.2f} | 情绪系数：{market_status['sentiment_factor']:.2f}")
    print(f"大盘站上20日线：{'是' if market_status['market_above_ma20'] else '否'} | 站上60日线：{'是' if market_status['market_above_ma60'] else '否'}")
    print(f"市场成交额：{market_status['market_amount']/1e8:.2f}亿 | 20日均额：{market_status['market_amount_ma20']/1e8:.2f}亿 | 大于均额：{'是' if market_status['market_amount_above_ma20'] else '否'}")
    print("=" * 70)

    signals, all_state = _analyze_etf_parallel(
        etf_list, etf_hist_cache, weekly_cache, all_state,
        market_status, today, STRATEGY_WEIGHTS
    )

    save_state(all_state, STATE_FILE)
    _output_signals(signals)

def main():
    if not silent_login():
        return
    real_time_analysis()
    silent_logout()

if __name__ == '__main__':
    main()