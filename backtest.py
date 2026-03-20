# backtest.py
# 回测功能模块

import pandas as pd
import numpy as np
import itertools
from data import get_daily_data, calculate_indicators
from etf_analysis import calculate_score, map_score_to_position
from config import (
    MACRO_INDEX, MARKET_INDEX, MACRO_MA_SHORT, MACRO_MA_LONG,
    RSI_PERIOD, CONFIRM_DAYS, BUY_THRESHOLD, SELL_THRESHOLD,
    TRAILING_STOP_HALF, TRAILING_STOP_CLEAR, PROFIT_TARGETS,
    RECENT_HIGH_WINDOW, RECENT_LOW_WINDOW, MARKET_STATES
)

def get_macro_status(macro_df):
    """根据沪深300与均线关系判断宏观状态"""
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
    from data import calculate_rsi
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

def backtest_single_etf(etf_code, start_date, end_date, params, initial_capital, max_single_position):
    """对单个ETF进行回测，返回绩效指标字典"""
    etf_df = get_daily_data(etf_code, start_date, end_date)
    if etf_df is None or etf_df.empty:
        return None

    macro_df = get_daily_data(MACRO_INDEX, start_date, end_date)
    if macro_df is None or macro_df.empty:
        return None
    macro_df = calculate_indicators(macro_df, ma_short=params['macro_ma_short'])
    macro_df['ma_long'] = macro_df['close'].rolling(window=params['macro_ma_long']).mean()

    market_df = get_daily_data(MARKET_INDEX, start_date, end_date)
    if market_df is None or market_df.empty:
        return None
    market_df = calculate_indicators(market_df, ma_short=20, vol_ma=20)

    etf_df = calculate_indicators(etf_df, ma_short=params['etf_ma'], vol_ma=params['etf_vol_ma'])

    common_dates = etf_df.index.intersection(macro_df.index).intersection(market_df.index)
    if len(common_dates) < params['confirm_days']:
        return None

    # 模拟交易
    capital = initial_capital
    shares = 0
    cost = 0.0
    cash = capital
    trades = []
    daily_equity = []
    score_history = []
    date_list = sorted(common_dates)

    for i in range(len(date_list)):
        date = date_list[i]
        etf_row = etf_df.loc[date]
        macro_row = macro_df.loc[date]
        market_row = market_df.loc[date]

        real_price = etf_row['close']
        ma20 = etf_row['ma_short']
        vol_ma = etf_row['vol_ma']
        volume = etf_row['volume']
        macd_dif = etf_row['macd_dif']
        macd_dea = etf_row['macd_dea']
        kdj_k = etf_row['kdj_k']
        kdj_d = etf_row['kdj_d']
        rsi = etf_row['rsi']
        boll_up = etf_row['boll_up']
        boll_low = etf_row['boll_low']
        williams_r = etf_row['williams_r']

        if i >= 1:
            prev = etf_df.loc[date_list[i-1]]
            macd_golden = (macd_dif > macd_dea) and (prev['macd_dif'] <= prev['macd_dea'])
            kdj_golden = (kdj_k > kdj_d) and (prev['kdj_k'] <= prev['kdj_d'])
        else:
            macd_golden = kdj_golden = False

        if i >= 4:
            ret_etf_5d = (real_price / etf_df.loc[date_list[i-4], 'close']) - 1
        else:
            ret_etf_5d = 0

        market_close = market_row['close']
        market_ma20 = market_row['ma_short']
        market_ma60 = market_row.get('ma_long', market_ma20)
        market_above_ma20 = market_close > market_ma20
        market_above_ma60 = market_close > market_ma60
        market_amount = market_row['amount']
        market_amount_ma20 = market_row['amount_ma']
        market_amount_above_ma20 = market_amount > market_amount_ma20

        if i >= 4:
            ret_market_5d = (market_close / market_df.loc[date_list[i-4], 'close']) - 1
        else:
            ret_market_5d = 0

        # 近期高低点
        if i >= params['recent_high_window']:
            recent_high = etf_df['high'].iloc[i-params['recent_high_window']+1:i+1].max()
        else:
            recent_high = etf_df['high'].iloc[:i+1].max()
        if i >= params['recent_low_window']:
            recent_low = etf_df['low'].iloc[i-params['recent_low_window']+1:i+1].min()
        else:
            recent_low = etf_df['low'].iloc[:i+1].min()
        drawdown = (recent_high - real_price) / recent_high if recent_high > 0 else 0
        gain = (real_price - recent_low) / recent_low if recent_low > 0 else 0

        break_ma = real_price < ma20
        trailing_clear = drawdown >= params['trailing_stop_clear']
        trailing_half = drawdown >= params['trailing_stop_half'] and not trailing_clear
        profit_hit = any(gain >= threshold for threshold, _ in params['profit_targets'])

        macro_status, market_factor = get_macro_status(macro_df.loc[:date])
        sentiment_factor = get_sentiment_factor(macro_df.loc[:date])

        base_score = calculate_score(
            real_price, ma20, volume, vol_ma,
            macd_golden, kdj_golden, rsi,
            boll_up, boll_low, williams_r,
            market_above_ma20, market_above_ma60, market_amount_above_ma20,
            ret_etf_5d, ret_market_5d,
            break_ma, trailing_half, trailing_clear, profit_hit,
            params['weights']
        )
        final_score = base_score * market_factor * sentiment_factor
        target_position = map_score_to_position(final_score)

        score_history.append(final_score)
        if len(score_history) > params['confirm_days']:
            score_history = score_history[-params['confirm_days']:]

        if len(score_history) >= params['confirm_days']:
            if all(s > params['buy_threshold'] for s in score_history):
                target_value = capital * min(target_position, max_single_position)
                current_value = shares * real_price
                if target_value > current_value:
                    buy_value = target_value - current_value
                    buy_shares = int(buy_value / real_price)
                    if buy_shares > 0:
                        cost = (shares * cost + buy_shares * real_price) / (shares + buy_shares) if (shares + buy_shares) > 0 else 0
                        shares += buy_shares
                        cash -= buy_shares * real_price
                        trades.append({'date': date, 'action': 'BUY', 'shares': buy_shares, 'price': real_price})
            elif all(s < params['sell_threshold'] for s in score_history):
                sell_shares = int(shares * 0.5)
                if sell_shares > 0:
                    cash += sell_shares * real_price
                    shares -= sell_shares
                    trades.append({'date': date, 'action': 'SELL', 'shares': sell_shares, 'price': real_price})
                    if shares == 0:
                        cost = 0.0

        daily_equity.append(cash + shares * real_price)

    # 绩效计算
    equity_series = pd.Series(daily_equity, index=common_dates)
    rets = equity_series.pct_change().dropna()
    total_return = (equity_series.iloc[-1] / initial_capital) - 1
    annual_return = (1 + total_return) ** (252 / len(common_dates)) - 1 if len(common_dates) > 0 else 0
    max_drawdown = (equity_series / equity_series.cummax() - 1).min()
    sharpe_ratio = rets.mean() / rets.std() * np.sqrt(252) if rets.std() != 0 else 0
    win_rate = len([t for t in trades if t['action'] == 'SELL' and t['price'] > cost]) / max(1, len([t for t in trades if t['action'] == 'SELL']))

    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'trades': trades,
        'equity_curve': equity_series,
    }

def run_backtest(etf_codes, start_date, end_date, params, initial_capital, max_single_position):
    results = []
    for code in etf_codes:
        res = backtest_single_etf(code, start_date, end_date, params, initial_capital, max_single_position)
        if res:
            results.append(res)
    if not results:
        return None
    return {
        'total_return': np.mean([r['total_return'] for r in results]),
        'annual_return': np.mean([r['annual_return'] for r in results]),
        'max_drawdown': np.mean([r['max_drawdown'] for r in results]),
        'sharpe_ratio': np.mean([r['sharpe_ratio'] for r in results]),
        'win_rate': np.mean([r['win_rate'] for r in results]),
    }

def optimize_parameters(etf_codes, start_date, end_date, param_grid, initial_capital, max_single_position):
    best_score = -np.inf
    best_params = None
    results = []
    keys = param_grid.keys()
    values = param_grid.values()
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        # 处理嵌套权重
        if 'weights' in params:
            from config import STRATEGY_WEIGHTS
            weights = STRATEGY_WEIGHTS.copy()
            for k, v in params['weights'].items():
                weights[k] = v
            params['weights'] = weights
        else:
            from config import STRATEGY_WEIGHTS
            params['weights'] = STRATEGY_WEIGHTS.copy()
        print(f"正在测试参数组合: {params}")
        avg_perf = run_backtest(etf_codes, start_date, end_date, params, initial_capital, max_single_position)
        if avg_perf is None:
            continue
        score = avg_perf['sharpe_ratio']
        if score > best_score:
            best_score = score
            best_params = params
        results.append((params, avg_perf))

    sorted_results = sorted(results, key=lambda x: x[1]['sharpe_ratio'], reverse=True)
    print("\n参数优化结果（按夏普比率排序）:")
    for i, (p, perf) in enumerate(sorted_results[:5]):
        print(f"{i+1}. 夏普={perf['sharpe_ratio']:.3f}, 年化={perf['annual_return']:.2%}, 最大回撤={perf['max_drawdown']:.2%}, 胜率={perf['win_rate']:.2%}")
        print(f"   参数: {p}")
    return best_params, best_score