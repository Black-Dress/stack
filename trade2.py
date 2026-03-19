import baostock as bs
import pandas as pd
import numpy as np
import datetime
import os
import requests
from contextlib import redirect_stdout

# ==================== 全局可调参数 ====================

MACRO_INDEX = 'sh.000300'               
MARKET_INDEX = 'sh.000001'               
MACRO_MA_SHORT = 20                       
MACRO_MA_LONG = 60                         

ETF_MA = 20                                
ETF_VOL_MA = 5                              

MACD_SHORT = 12
MACD_LONG = 26
MACD_SIGNAL = 9

KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3

RSI_PERIOD = 14                            
SENTIMENT_FACTOR = {
    'panic': 0.6,      
    'pessimistic': 0.8, 
    'optimistic': 1.0,  
    'frenzy': 0.9       
}

STOP_LOSS = 0.10                             
TRAILING_STOP = 0.08                          
PROFIT_TARGETS = [0.20, 0.40]                  

BUY_SCORE_WEIGHTS = {
    'macro_not_bear': 0.15,          
    'market_amount_above_ma': 0.15,   
    'price_above_ma20_and_vol': 0.20, 
    'outperform_macro': 0.15,          
    'macd_golden_cross': 0.15,         
    'kdj_golden_cross': 0.20            
}

def map_score_to_ratio(score):
    if score < 0.2:
        return 0.0
    elif score < 0.4:
        return 0.25
    elif score < 0.6:
        return 0.4
    elif score < 0.8:
        return 0.6
    else:
        return 0.8

BUY_COOLDOWN = 1      
SELL_COOLDOWN = 3     

EXTREME_MARKET_DROP = 0.03                    
EXTREME_ETF_DEVIATION = 0.05                   

POSITION_LIMIT = {
    'bull': 0.7,      
    'oscillate': 0.5, 
    'bear': 0.2       
}

TOTAL_CAPITAL = 1000000   

POSITION_FILE = 'positions.csv'               

# ==================== 数据获取函数 ====================

def get_realtime_price_sina(code):
    try:
        sina_code = code.replace('.', '')
        url = f'http://hq.sinajs.cn/list={sina_code}'
        headers = {'Referer': 'http://finance.sina.com.cn'}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.text
            parts = data.split('"')[1].split(',')
            if len(parts) > 3:
                return float(parts[3])  
        return None
    except Exception as e:
        print(f"获取实时价格失败 {code}: {e}")
        return None

def get_realtime_index_sina(code='sh000001'):
    try:
        url = f'http://hq.sinajs.cn/list={code}'
        headers = {'Referer': 'http://finance.sina.com.cn'}
        r = requests.get(url, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.text
            parts = data.split('"')[1].split(',')
            if len(parts) > 1:
                return float(parts[1])
        return None
    except Exception as e:
        print(f"获取实时指数失败: {e}")
        return None

def silent_login():
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        lg = bs.login()
    if lg.error_code != '0':
        print(f'登录失败: {lg.error_msg}')
        return False
    return True

def silent_logout():
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        bs.logout()

def get_daily_data(code, start_date, end_date, fields="date,code,open,high,low,close,volume,amount"):
    rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, end_date=end_date, frequency="d")
    if rs.error_code != '0':
        print(f'获取数据失败 {code}: {rs.error_msg}')
        return None
    data_list = []
    while (rs.error_code == '0') & rs.next():
        data_list.append(rs.get_row_data())
    if not data_list:
        return None
    df = pd.DataFrame(data_list, columns=rs.fields)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def calculate_macd(df, short=12, long_=26, signal=9):
    exp1 = df['close'].ewm(span=short, adjust=False).mean()
    exp2 = df['close'].ewm(span=long_, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    df['macd_dif'] = macd
    df['macd_dea'] = signal_line
    df['macd_hist'] = histogram
    return df

def calculate_kdj(df, n=9, m1=3, m2=3):
    low_list = df['low'].rolling(window=n).min()
    low_list.fillna(value=df['low'].expanding().min(), inplace=True)
    high_list = df['high'].rolling(window=n).max()
    high_list.fillna(value=df['high'].expanding().max(), inplace=True)
    rsv = (df['close'] - low_list) / (high_list - low_list) * 100
    df['kdj_k'] = rsv.ewm(alpha=1/m1, adjust=False).mean()
    df['kdj_d'] = df['kdj_k'].ewm(alpha=1/m2, adjust=False).mean()
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    return df

def calculate_indicators(df, ma_short=20, ma_long=60, vol_ma=5):
    df = df.copy()
    df['ma_short'] = df['close'].rolling(window=ma_short).mean()
    df['ma_long'] = df['close'].rolling(window=ma_long).mean()
    df['vol_ma'] = df['volume'].rolling(window=vol_ma).mean()
    df['amount_ma'] = df['amount'].rolling(window=vol_ma).mean()
    df = calculate_macd(df, MACD_SHORT, MACD_LONG, MACD_SIGNAL)
    df = calculate_kdj(df, KDJ_N, KDJ_M1, KDJ_M2)
    return df

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def check_macro_status(df_macro):
    latest = df_macro.iloc[-1]
    close = latest['close']
    ma20 = latest['ma_short']
    ma60 = latest['ma_long']
    if close > ma20 and close > ma60:
        return 'bull', POSITION_LIMIT['bull']
    elif close < ma20 and close < ma60:
        return 'bear', POSITION_LIMIT['bear']
    else:
        return 'oscillate', POSITION_LIMIT['oscillate']

def get_sentiment_level(macro_df):
    if macro_df is None or len(macro_df) < RSI_PERIOD + 1:
        return 'optimistic', 1.0
    rsi = calculate_rsi(macro_df['close'], RSI_PERIOD)
    latest_rsi = rsi.iloc[-1]
    if latest_rsi < 30:
        return 'panic', SENTIMENT_FACTOR['panic']
    elif latest_rsi < 50:
        return 'pessimistic', SENTIMENT_FACTOR['pessimistic']
    elif latest_rsi < 70:
        return 'optimistic', SENTIMENT_FACTOR['optimistic']
    else:
        return 'frenzy', SENTIMENT_FACTOR['frenzy']

def load_positions():
    if not os.path.exists(POSITION_FILE):
        raise FileNotFoundError(f"持仓文件 {POSITION_FILE} 不存在，请手动创建。")
    df = pd.read_csv(POSITION_FILE, encoding='utf-8-sig')
    required_cols = ['代码', '名称', '份额', '成本', '止盈档位', '买入日期', '满仓金额', '上次买入日期', '上次卖出日期']
    # 如果缺少某些列，填充默认值以便程序运行
    for col in required_cols:
        if col not in df.columns:
            if '日期' in col:
                df[col] = pd.NaT
            else:
                df[col] = 0.0
    # 类型转换
    df['份额'] = pd.to_numeric(df['份额'], errors='coerce').fillna(0).astype(int)
    df['成本'] = pd.to_numeric(df['成本'], errors='coerce')
    df['止盈档位'] = pd.to_numeric(df['止盈档位'], errors='coerce').fillna(0.0)
    df['买入日期'] = pd.to_datetime(df['买入日期'], errors='coerce')
    df['满仓金额'] = pd.to_numeric(df['满仓金额'], errors='coerce').fillna(0.0)
    df['上次买入日期'] = pd.to_datetime(df['上次买入日期'], errors='coerce')
    df['上次卖出日期'] = pd.to_datetime(df['上次卖出日期'], errors='coerce')
    return df

def calculate_buy_score(macro_status, market_amount, market_amount_ma20,
                        real_price, ma20, volume, vol_ma,
                        ret_etf_5d, ret_macro_5d,
                        macd_golden, kdj_golden):
    score = 0.0
    if macro_status != 'bear':
        score += BUY_SCORE_WEIGHTS['macro_not_bear']
    if market_amount > market_amount_ma20:
        score += BUY_SCORE_WEIGHTS['market_amount_above_ma']
    if real_price > ma20 and volume > vol_ma:
        score += BUY_SCORE_WEIGHTS['price_above_ma20_and_vol']
    if ret_etf_5d > ret_macro_5d:
        score += BUY_SCORE_WEIGHTS['outperform_macro']
    if macd_golden:
        score += BUY_SCORE_WEIGHTS['macd_golden_cross']
    if kdj_golden:
        score += BUY_SCORE_WEIGHTS['kdj_golden_cross']
    return score

def is_extreme_situation(ret, real_price, ma60, market_drop):
    if ret is not None and ret <= -STOP_LOSS:
        return True
    if market_drop is not None and market_drop >= EXTREME_MARKET_DROP:
        return True
    if ma60 is not None and real_price < ma60 * (1 - EXTREME_ETF_DEVIATION):
        return True
    return False

def analyze_etf(row, macro_status, macro_position_limit, sentiment_factor,
                market_amount, market_amount_ma20,
                macro_df, etf_history, real_index, prev_index_close, today,
                total_capital, total_market_value):
    """
    分析单个ETF，返回状态字符串和可能的操作信号
    信号格式：{'action': 'BUY'/'SELL', 'reason': str, 'shares': int, 'price': float, 
              'new_shares': int, 'new_cost': float, 'new_tp_level': float}  （new_* 为执行后的状态）
    """
    code = row['代码']
    name = row['名称']
    shares = row['份额']
    cost = row['成本']
    tp_level = row['止盈档位']
    buy_date = row['买入日期']
    full_amount = row['满仓金额']
    last_buy_date = row['上次买入日期']
    last_sell_date = row['上次卖出日期']

    real_price = get_realtime_price_sina(code)
    if real_price is None:
        return f"{name} ({code}): 无法获取实时价格", None

    if etf_history is None or etf_history.empty or len(etf_history) < ETF_MA:
        return f"{name} ({code}): 历史数据不足", None

    latest_hist = etf_history.iloc[-1]
    ma20 = latest_hist['ma_short']
    ma60 = latest_hist['ma_long'] if 'ma_long' in latest_hist else None
    volume = latest_hist['volume']
    vol_ma = latest_hist['vol_ma']

    # MACD/KDJ金叉
    if len(etf_history) >= 2:
        macd_golden = (latest_hist['macd_dif'] > latest_hist['macd_dea'] and
                       etf_history.iloc[-2]['macd_dif'] <= etf_history.iloc[-2]['macd_dea'])
        kdj_golden = (latest_hist['kdj_k'] > latest_hist['kdj_d'] and
                      etf_history.iloc[-2]['kdj_k'] <= etf_history.iloc[-2]['kdj_d'])
    else:
        macd_golden = kdj_golden = False

    # 近5日涨幅对比
    if len(etf_history) >= 5 and len(macro_df) >= 5:
        ret_etf_5d = (real_price / etf_history.iloc[-5]['close']) - 1
        ret_macro_5d = (macro_df.iloc[-1]['close'] / macro_df.iloc[-5]['close']) - 1
    else:
        ret_etf_5d = 0
        ret_macro_5d = 0

    # 大盘实时跌幅
    market_drop = None
    if real_index is not None and prev_index_close is not None:
        market_drop = (prev_index_close - real_index) / prev_index_close

    # 冷静期判断
    can_buy = True
    can_sell = True
    if pd.notna(last_sell_date):
        days_since_sell = (today - last_sell_date.date()).days
        if days_since_sell < BUY_COOLDOWN:
            can_buy = False
    if pd.notna(buy_date):
        days_since_buy = (today - buy_date.date()).days
        if days_since_buy < SELL_COOLDOWN:
            can_sell = False

    extreme = is_extreme_situation((real_price-cost)/cost if shares>0 else None, real_price, ma60, market_drop)
    if extreme:
        can_buy = can_sell = True

    lines = []
    lines.append(f"【{name} ({code})】")
    lines.append(f"  实时价: {real_price:.3f} | 20日线: {ma20:.3f} | 60日线: {ma60:.3f}" if ma60 else f"  实时价: {real_price:.3f} | 20日线: {ma20:.3f}")
    lines.append(f"  成交量: {volume:.0f} (5日均: {vol_ma:.0f}) {'✅ 达标' if volume > vol_ma else '❌ 不足'}")
    lines.append(f"  近5日涨幅: {ret_etf_5d*100:+.2f}% | 沪深300同期: {ret_macro_5d*100:+.2f}% {'✅ 跑赢' if ret_etf_5d > ret_macro_5d else '❌ 跑输'}")
    lines.append(f"  MACD: {'✅ 金叉' if macd_golden else '❌ 非金叉'} | KDJ: {'✅ 金叉' if kdj_golden else '❌ 非金叉'}")
    if not can_buy:
        lines.append(f"  ⏳ 买入冷静期: 上次卖出距今 {days_since_sell if pd.notna(last_sell_date) else 'N/A'}天")
    if not can_sell:
        lines.append(f"  ⏳ 卖出冷静期: 上次买入距今 {days_since_buy if pd.notna(buy_date) else 'N/A'}天")

    signal = None  # 信号字典

    # ========== 卖出信号（持仓时） ==========
    if shares > 0:
        ret = (real_price - cost) / cost
        lines.append(f"  持仓: {shares}份 | 成本: {cost:.3f} | 收益率: {ret*100:+.2f}%")

        # 计算自买入以来的最高价（用于移动止盈）
        if pd.notna(buy_date):
            hist_since_buy = etf_history[etf_history.index >= buy_date]
            if not hist_since_buy.empty:
                peak_price = hist_since_buy['high'].max()
            else:
                peak_price = real_price
        else:
            peak_price = real_price

        drawdown = (peak_price - real_price) / peak_price if peak_price > 0 else 0

        # 1. 硬止损
        if ret <= -STOP_LOSS:
            if not can_sell:
                lines.append("  ⚠️  硬止损触发，但卖出冷静期内（可考虑手动干预）")
            else:
                new_shares = 0
                new_cost = 0.0
                new_tp_level = 0.0
                signal = {
                    'action': 'SELL', 'reason': '硬止损', 'shares': shares,
                    'new_shares': new_shares, 'new_cost': new_cost, 'new_tp_level': new_tp_level
                }
                lines.append(f"  🔴  建议卖出全部 {shares}份 (硬止损)")
                lines.append(f"     执行后: 份额 0, 成本 0, 止盈档位 0")
        # 2. 移动止盈
        elif ret >= 0.10 and drawdown >= TRAILING_STOP:
            if not can_sell:
                lines.append("  ⚠️  移动止盈触发，但卖出冷静期内")
            else:
                new_shares = 0
                new_cost = 0.0
                new_tp_level = 0.0
                signal = {
                    'action': 'SELL', 'reason': '移动止盈', 'shares': shares,
                    'new_shares': new_shares, 'new_cost': new_cost, 'new_tp_level': new_tp_level
                }
                lines.append(f"  🟡  建议卖出全部 {shares}份 (移动止盈)")
                lines.append(f"     执行后: 份额 0, 成本 0, 止盈档位 0")
        # 3. 跌破20日线
        elif real_price < ma20:
            if not can_sell:
                lines.append("  ⚠️  跌破20日线，但卖出冷静期内")
            else:
                new_shares = 0
                new_cost = 0.0
                new_tp_level = 0.0
                signal = {
                    'action': 'SELL', 'reason': '跌破20日线', 'shares': shares,
                    'new_shares': new_shares, 'new_cost': new_cost, 'new_tp_level': new_tp_level
                }
                lines.append(f"  🔴  建议卖出全部 {shares}份 (跌破20日线)")
                lines.append(f"     执行后: 份额 0, 成本 0, 止盈档位 0")
        else:
            # 4. 分批止盈
            for target in PROFIT_TARGETS:
                if ret >= target and tp_level < target:
                    sell_shares = int(shares * 0.333)
                    if sell_shares == 0:
                        sell_shares = shares
                    if not can_sell:
                        lines.append(f"  ⚠️  止盈{int(target*100)}%触发，但卖出冷静期内")
                        break
                    else:
                        new_shares = shares - sell_shares
                        new_cost = cost  # 部分卖出成本不变
                        new_tp_level = target
                        signal = {
                            'action': 'SELL', 'reason': f'止盈{int(target*100)}%', 'shares': sell_shares,
                            'new_shares': new_shares, 'new_cost': new_cost, 'new_tp_level': new_tp_level
                        }
                        lines.append(f"  🟢  建议卖出 {sell_shares}份 (止盈{int(target*100)}%)")
                        lines.append(f"     执行后: 份额 {new_shares}, 成本 {new_cost:.3f}, 止盈档位 {new_tp_level*100:.0f}%")
                        break
            else:
                lines.append("  ✅ 无卖出信号")

    # ========== 买入信号 ==========
    if signal is None:
        score = calculate_buy_score(
            macro_status, market_amount, market_amount_ma20,
            real_price, ma20, volume, vol_ma,
            ret_etf_5d, ret_macro_5d,
            macd_golden, kdj_golden
        )
        raw_ratio = map_score_to_ratio(score)
        adjusted_ratio = raw_ratio * macro_position_limit * sentiment_factor
        lines.append(f"  买入加权评分: {score:.3f} → 理论仓位: {raw_ratio*100:.1f}%")
        lines.append(f"  宏观限制{macro_position_limit*100:.0f}% * 情绪系数{sentiment_factor:.2f} → 实际仓位: {adjusted_ratio*100:.1f}%")

        if adjusted_ratio > 0 and full_amount > 0:
            target_value = full_amount * adjusted_ratio
            current_value = shares * real_price
            if target_value > current_value:
                buy_value = target_value - current_value
                remaining_capital = total_capital - total_market_value
                feasible_buy_value = min(buy_value, remaining_capital)
                suggested_shares = int(feasible_buy_value / real_price)
                if suggested_shares > 0:
                    if not can_buy:
                        lines.append(f"  ⏳ 建议买入 {suggested_shares}份，但买入冷静期内")
                    else:
                        new_shares = shares + suggested_shares
                        total_cost = shares * cost + suggested_shares * real_price
                        new_cost = total_cost / new_shares if new_shares > 0 else 0
                        new_tp_level = 0.0  # 重置止盈档位
                        signal = {
                            'action': 'BUY', 'reason': f'评分{score:.2f}宏观{macro_status}情绪{score:.2f}',
                            'shares': suggested_shares, 'price': real_price,
                            'new_shares': new_shares, 'new_cost': new_cost, 'new_tp_level': new_tp_level
                        }
                        lines.append(f"  🟢 建议买入 {suggested_shares}份 (目标仓位，剩余资金允许)")
                        lines.append(f"     执行后: 份额 {new_shares}, 成本 {new_cost:.3f}, 止盈档位 0%")
                else:
                    lines.append("  ℹ️ 剩余资金不足或无需买入")
            else:
                lines.append("  ℹ️ 当前持仓已达或超过目标仓位，无需买入")
        elif adjusted_ratio > 0 and full_amount == 0:
            if not can_buy:
                lines.append("  ⏳ 建议买入，但买入冷静期内")
            else:
                # 无满仓金额限制，只提示，不计算具体份额
                signal = {
                    'action': 'BUY', 'reason': f'评分{score:.2f}', 'shares': None, 'price': real_price,
                    'new_shares': None, 'new_cost': None, 'new_tp_level': None
                }
                lines.append("  🟢 建议买入 (请根据资金自行决定数量)")
        else:
            lines.append("  ⚪ 不满足买入条件")

    return "\n".join(lines), signal

def main():
    if not silent_login():
        return
    try:
        positions_df = load_positions()

        today = datetime.date.today()
        today_str = today.strftime('%Y-%m-%d')
        start_date = (today - datetime.timedelta(days=300)).strftime('%Y-%m-%d')

        # 获取宏观指数
        macro_df = get_daily_data(MACRO_INDEX, start_date, today_str)
        if macro_df is None or macro_df.empty:
            print('无法获取宏观指数数据')
            return
        macro_df = calculate_indicators(macro_df, ma_short=MACRO_MA_SHORT, ma_long=MACRO_MA_LONG)

        # 获取市场指数
        market_df = get_daily_data(MARKET_INDEX, start_date, today_str)
        if market_df is None or market_df.empty:
            print('无法获取市场指数数据')
            return
        market_df = calculate_indicators(market_df, ma_short=20, ma_long=60, vol_ma=20)

        # 获取所有ETF历史数据
        etf_hist_cache = {}
        for code in positions_df['代码'].unique():
            df = get_daily_data(code, start_date, today_str)
            if df is not None and not df.empty:
                df = calculate_indicators(df, ma_short=ETF_MA, ma_long=60, vol_ma=ETF_VOL_MA)
                etf_hist_cache[code] = df

        # 实时大盘
        real_index = get_realtime_index_sina('sh000001')
        prev_index_close = None
        if market_df is not None and not market_df.empty:
            prev_index_close = market_df.iloc[-1]['close']

        # 宏观状态与情绪
        macro_status, macro_position_limit = check_macro_status(macro_df)
        sentiment_level, sentiment_factor = get_sentiment_level(macro_df)

        market_latest = market_df.iloc[-1]
        market_amount = market_latest['amount']
        market_amount_ma20 = market_latest['amount_ma']

        # 计算当前总持仓市值
        total_market_value = 0.0
        for idx, row in positions_df.iterrows():
            code = row['代码']
            shares = row['份额']
            if shares > 0:
                price = get_realtime_price_sina(code)
                if price is not None:
                    total_market_value += shares * price

        # 输出宏观信息
        print(f"\n日期：{today_str}")
        print(f"宏观状态：{macro_status.upper()}，建议仓位上限：{macro_position_limit*100:.0f}%")
        print(f"情绪等级：{sentiment_level}，情绪系数：{sentiment_factor:.2f}")
        print(f"市场成交额：{market_amount/1e8:.2f}亿，20日均额：{market_amount_ma20/1e8:.2f}亿")
        if real_index is not None and prev_index_close is not None:
            index_change = (real_index - prev_index_close) / prev_index_close * 100
            print(f"实时大盘：{real_index:.2f} ({index_change:+.2f}%)")
        else:
            print("实时大盘：获取失败")
        print(f"当前总持仓市值：{total_market_value:.2f} 元，剩余资金：{TOTAL_CAPITAL - total_market_value:.2f} 元")

        # 分析每个ETF
        print("\n" + "="*60)
        signals = []

        for idx, row in positions_df.iterrows():
            code = row['代码']
            hist_df = etf_hist_cache.get(code)
            status, signal = analyze_etf(
                row, macro_status, macro_position_limit, sentiment_factor,
                market_amount, market_amount_ma20,
                macro_df, hist_df, real_index, prev_index_close, today,
                TOTAL_CAPITAL, total_market_value
            )
            print(status)
            print("-" * 40)
            if signal:
                signals.append({**signal, 'code': code, 'name': row['名称']})

        # 输出操作信号汇总
        print("\n" + "="*60)
        print("操作信号汇总（执行后的新状态）")
        print("="*60)
        if not signals:
            print("无操作建议")
        else:
            for sig in signals:
                if sig['action'] == 'BUY':
                    if sig['shares'] is not None:
                        print(f"买入 {sig['name']} ({sig['code']}) - 原因：{sig['reason']}，建议买入 {sig['shares']}份，现价：{sig['price']:.3f}")
                        print(f"  执行后: 份额 {sig['new_shares']}, 成本 {sig['new_cost']:.3f}, 止盈档位 0%")
                    else:
                        print(f"买入 {sig['name']} ({sig['code']}) - 原因：{sig['reason']}，现价：{sig['price']:.3f}（请自行决定数量）")
                else:
                    print(f"卖出 {sig['name']} ({sig['code']}) - 原因：{sig['reason']}，建议卖出 {sig['shares']}份")
                    print(f"  执行后: 份额 {sig['new_shares']}, 成本 {sig['new_cost']:.3f}, 止盈档位 {sig['new_tp_level']*100:.0f}%")

    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        silent_logout()

if __name__ == '__main__':
    main()