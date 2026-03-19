# -*- coding: utf-8 -*-
"""
ETF 买卖决策脚本（综合评分10分制 + 实时价格混合模式 + 批量处理）
用法：
  单只模式: python trade.py <ETF代码> <收益率/成本价> <持仓天数> <当前持仓金额> [可用资金]
  批量模式: python trade.py <Excel文件路径>

Excel格式要求（共6列）：
  代码、名称、当前收益率、持有日期、当前仓位、当前剩余可用资金
示例：
  代码       名称      当前收益率 持有日期 当前仓位 当前剩余可用资金
  sz.159326  有色金属ETF -12.36%  30     20000    100000
"""

import baostock as bs
import pandas as pd
import numpy as np
import sys
import os
import traceback
from datetime import datetime, timedelta

# 尝试导入akshare，若失败则情绪指标使用默认值
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False
    print("警告：未安装akshare库，情绪指标将使用中性值，无法获取实时价格。")
    print("请运行：pip install akshare")

# ======================== 全局策略参数 ========================
# 止损参数（已适当放宽）
HARD_STOP_LOSS = -0.10          # 硬止损阈值（-10%）
TIME_STOP_DAYS = 10             # 时间止损最低持仓天数
TIME_STOP_RANGE = 0.02          # 时间止损收益率范围（±2%）

# 止盈参数
TAKE_PROFIT_THRESHOLD = 0.15    # 触发移动止盈的最低收益率
TAKE_PROFIT_LEVELS = [0.15, 0.20, 0.25]  # 分档止盈阈值
TAKE_PROFIT_RATIOS = [0.3, 0.3, 0.4]     # 每档减仓比例（总和1）

# 移动止盈参数（基础容忍度已适当放宽）
DRAWDOWN_TOLERANCE_BASE = {
    '进攻区': 0.10,
    '震荡区': 0.07,
    '防御区': 0.04,
    '未知': 0.04
}
DRAWDOWN_ADJUST_BY_ATR = True
ATR_PERIOD = 14

# 成交量信号阈值（动态分位数）
VOLUME_HISTORY_DAYS = 60
VOLUME_SURGE_PERCENTILE = 80
VOLUME_SHRINK_PERCENTILE = 20

# 技术指标参数
MA_SHORT = 5
MA_MEDIUM = 10
MA_LONG = 20
MA_TREND = 60
ADX_PERIOD = 14
ADX_THRESHOLD = 25

# MACD参数
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9

# 仓位管理参数
BASE_POSITION_RATIO = 0.2
ATR_POSITION_ADJUST = True
MAX_POSITION_RATIO = 0.3

# 买入信号强度对应的基准加仓比例
BUY_ADD_RATIO_BASE = {
    'strong': 0.5,
    'cautious': 1/3,
    'weak': 0.25
}

# 减仓比例映射
SELL_REDUCE_RATIO = {
    '硬止损': 1.0,
    '时间止损（夏普负）': 0.5,
    '移动止盈': 0.5,
    '分档止盈': 0.3,
    '成交量危险信号+防御': 1.0,
    '成交量危险信号+震荡': 0.5,
    '情绪高潮卖出': 0.5,
}

# ======================== 评分归一化参数 ========================
MAX_BUY_SCORE = 480       # 买入条件理论最大总分
MAX_SELL_SCORE = 1000      # 卖出条件理论最大总分（不含硬止损）
BUY_THRESHOLD = 5         # 买入阈值（10分制）
SELL_THRESHOLD = 5        # 卖出阈值（10分制）
# =============================================================

# ======================== 情绪指标参数 ========================
SENTIMENT_WEIGHTS = {
    'margin_trade_ratio': 0.4,
    'turnover_rate': 0.3,
    'limit_up_count': 0.3
}

SENTIMENT_THRESHOLDS = {
    'margin_trade_ratio': {'low': 8, 'high': 12},
    'turnover_rate': {'low': 1.0, 'high': 2.5},
    'limit_up_count': {'low': 30, 'high': 80}
}

SENTIMENT_ADJUST = {
    'panic': 0.7,
    'cautious': 0.9,
    'neutral': 1.0,
    'optimistic': 1.1,
    'euphoric': 1.2
}
# =============================================================

# ------------------------- 静默登录/登出 -------------------------
def quiet_login():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    lg = bs.login()
    sys.stdout.close()
    sys.stdout = old_stdout
    if lg.error_code != '0':
        print('登录失败:', lg.error_msg)
        return False
    return True

def quiet_logout():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    bs.logout()
    sys.stdout.close()
    sys.stdout = old_stdout

# ------------------------- 数据获取与指标计算 -------------------------
def get_k_data(code, days=250, min_required=20):
    if not quiet_login():
        return None

    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=days*1.5)).strftime('%Y-%m-%d')

    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,volume,amount",
        start_date=start_date, end_date=end_date,
        frequency="d", adjustflag="3"
    )

    data_list = []
    while (rs.error_code == '0') and rs.next():
        data_list.append(rs.get_row_data())
    quiet_logout()

    if not data_list:
        print(f"警告：{code} 未返回任何数据")
        return None

    df = pd.DataFrame(data_list, columns=rs.fields)
    df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0)
    for col in ['open','high','low','close','amount']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    if len(df) < min_required:
        print(f"警告：{code} 只有 {len(df)} 天数据，少于策略最小要求 {min_required}，分析可能不准确。")
    return df.tail(min(len(df), days))

def add_indicators(df):
    """添加技术指标：均线、成交量分位数、ATR、ADX、MACD"""
    df['ma5'] = df['close'].rolling(MA_SHORT).mean()
    df['ma10'] = df['close'].rolling(MA_MEDIUM).mean()
    df['ma20'] = df['close'].rolling(MA_LONG).mean()
    df['ma60'] = df['close'].rolling(MA_TREND).mean() if len(df) >= MA_TREND else np.nan

    df['vol_ma5'] = df['volume'].rolling(MA_SHORT).mean()
    df['pct_change'] = df['close'].pct_change() * 100
    df['high_20'] = df['high'].rolling(MA_LONG).max()

    # ATR
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift()),
            abs(df['low'] - df['close'].shift())
        )
    )
    df['atr'] = df['tr'].rolling(ATR_PERIOD).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # 成交量分位数
    if len(df) >= VOLUME_HISTORY_DAYS:
        df['vol_percentile_80'] = df['volume'].rolling(VOLUME_HISTORY_DAYS).quantile(0.8)
        df['vol_percentile_20'] = df['volume'].rolling(VOLUME_HISTORY_DAYS).quantile(0.2)
    else:
        df['vol_percentile_80'] = np.nan
        df['vol_percentile_20'] = np.nan

    # ADX
    df['up_move'] = df['high'] - df['high'].shift()
    df['down_move'] = df['low'].shift() - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

    df['plus_di'] = 100 * df['plus_dm'].rolling(ATR_PERIOD).mean() / df['atr']
    df['minus_di'] = 100 * df['minus_dm'].rolling(ATR_PERIOD).mean() / df['atr']
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(ATR_PERIOD).mean()

    # MACD
    exp12 = df['close'].ewm(span=MACD_FAST, adjust=False).mean()
    exp26 = df['close'].ewm(span=MACD_SLOW, adjust=False).mean()
    df['macd'] = exp12 - exp26
    df['macd_signal'] = df['macd'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']

    return df

# ------------------------- 大盘状态判断 -------------------------
def get_market_status(index_code='sh.000001'):
    df = get_k_data(index_code, days=120, min_required=20)
    if df is None or len(df) < 20:
        return '未知'
    df = add_indicators(df)
    last = df.iloc[-1]
    close = last['close']
    ma20 = last['ma20']
    ma60 = last['ma60']
    adx = last['adx']

    if pd.isna(ma60):
        if close > ma20:
            trend = '进攻区'
        elif close < ma20:
            trend = '防御区'
        else:
            trend = '震荡区'
    else:
        if close > ma20 and ma20 > ma60:
            trend = '进攻区'
        elif close < ma20 and ma20 < ma60:
            trend = '防御区'
        else:
            trend = '震荡区'

    if not pd.isna(adx) and adx < ADX_THRESHOLD:
        trend = '震荡区'

    return trend

# ------------------------- 成交量信号判断 -------------------------
def get_volume_signal(df):
    if len(df) < 5:
        return '数据不足', {}
    last = df.iloc[-1]
    vol = last['volume']
    vol_80 = last['vol_percentile_80']
    vol_20 = last['vol_percentile_20']
    pct = last['pct_change']

    details = {
        'volume': vol,
        'vol_80': vol_80,
        'vol_20': vol_20,
        'pct_change': pct,
        'vol_ratio': vol / last['vol_ma5'] if last['vol_ma5'] > 0 else 0
    }

    if pd.isna(vol_80) or pd.isna(vol_20):
        vol_ma5 = last['vol_ma5']
        if vol > vol_ma5 * 1.5:
            if pct > 2:
                return '放量突破', details
            elif pct < -2:
                return '放量下跌', details
            elif abs(pct) <= 1:
                return '放量滞涨', details
            else:
                return '放量正常', details
        elif vol < vol_ma5 * 0.8:
            if pct > 0:
                return '缩量上涨', details
            else:
                return '缩量下跌/调整', details
        else:
            return '量能健康', details

    if vol > vol_80:
        if pct > 2:
            return '放量突破', details
        elif pct < -2:
            return '放量下跌', details
        elif abs(pct) <= 1:
            return '放量滞涨', details
        else:
            return '放量正常', details
    elif vol < vol_20:
        if pct > 0:
            return '缩量上涨', details
        else:
            return '缩量下跌/调整', details
    else:
        return '量能健康', details

# ------------------------- 情绪指标获取 -------------------------
def get_market_sentiment():
    if not AKSHARE_AVAILABLE:
        return 50, 'neutral', {}

    margin_ratio = 10.0
    turnover = 2.0
    limit_up_count = 50

    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        margin_df = ak.stock_margin_sse(start_date=yesterday, end_date=yesterday)
        if margin_df is not None and not margin_df.empty and '融资买入额' in margin_df.columns:
            pass
    except Exception:
        pass

    try:
        all_a_df = ak.stock_zh_index_daily(symbol="sh000001")
        if len(all_a_df) > 5:
            vol_ma5 = all_a_df['volume'].rolling(5).mean().iloc[-1]
            turnover = all_a_df['volume'].iloc[-1] / vol_ma5 * 2
    except Exception:
        pass

    try:
        today = datetime.now().strftime('%Y%m%d')
        limit_up_df = ak.stock_zt_pool_em(date=today)
        if limit_up_df is not None and not limit_up_df.empty:
            limit_up_count = len(limit_up_df)
    except Exception:
        pass

    indicators = {
        'margin_trade_ratio': margin_ratio,
        'turnover_rate': turnover,
        'limit_up_count': limit_up_count
    }

    scores = {}
    for key, value in indicators.items():
        low = SENTIMENT_THRESHOLDS[key]['low']
        high = SENTIMENT_THRESHOLDS[key]['high']
        if value <= low:
            scores[key] = 0
        elif value >= high:
            scores[key] = 100
        else:
            scores[key] = (value - low) / (high - low) * 100

    total_score = sum(scores[k] * SENTIMENT_WEIGHTS[k] for k in SENTIMENT_WEIGHTS.keys())

    if total_score < 20:
        sentiment_state = 'panic'
    elif total_score < 40:
        sentiment_state = 'cautious'
    elif total_score < 60:
        sentiment_state = 'neutral'
    elif total_score < 80:
        sentiment_state = 'optimistic'
    else:
        sentiment_state = 'euphoric'

    return total_score, sentiment_state, indicators

# ------------------------- 实时价格获取 -------------------------
def get_realtime_price(code):
    """使用AKShare获取实时最新价，失败返回None"""
    if not AKSHARE_AVAILABLE:
        return None
    try:
        df = ak.stock_zh_a_spot_em()
        code_num = code.split('.')[-1]
        row = df[df['代码'] == code_num]
        if row.empty:
            row = df[df['代码'] == code]
        if row.empty:
            return None
        price = row['最新价'].values[0]
        return float(price)
    except Exception:
        return None

# ------------------------- 买入形态判断（返回分数） -------------------------
def check_breakout_buy(df, verbose=False):
    if len(df) < 20:
        return 0
    last = df.iloc[-1]
    high_20 = last['high_20']
    close = last['close']
    vol_signal, _ = get_volume_signal(df)
    prev_close = df.iloc[-2]['close'] if len(df) > 1 else 0
    cond_stand = prev_close > high_20

    score = 0
    if close > high_20:
        score += 40
        if vol_signal == '放量突破':
            score += 40
        elif vol_signal in ['放量正常', '量能健康']:
            score += 20
        if cond_stand:
            score += 20
    if verbose:
        if score >= 80:
            print(f"  ✅ 突破买入得分 {score}: 强烈信号")
        elif score >= 50:
            print(f"  ⚠️ 突破买入得分 {score}: 一般信号")
        else:
            print(f"  ❌ 突破买入得分 {score}: 不满足")
    return score

def check_pullback_buy(df, verbose=False):
    if len(df) < 30:
        return 0
    last = df.iloc[-1]
    close = last['close']
    ma20 = last['ma20']
    vol_signal, _ = get_volume_signal(df)
    macd_hist = last['macd_hist']

    score = 0
    near_ma20 = abs(close - ma20) / ma20 < 0.015 if ma20 > 0 else False
    if near_ma20:
        score += 40
    else:
        deviation = abs(close - ma20) / ma20
        if deviation < 0.03:
            score += 20

    vol_shrink = vol_signal in ['缩量下跌/调整', '量能健康'] and last['volume'] < last['vol_ma5'] * 0.9
    if vol_shrink:
        score += 30

    if len(df) > 5:
        price_lower = close < df.iloc[-2]['close']
        macd_higher = macd_hist > df.iloc[-2]['macd_hist']
        if price_lower and macd_higher:
            score += 30

    if verbose:
        if score >= 70:
            print(f"  ✅ 回调买入得分 {score}: 强烈信号")
        elif score >= 40:
            print(f"  ⚠️ 回调买入得分 {score}: 一般信号")
        else:
            print(f"  ❌ 回调买入得分 {score}: 不满足")
    return score

def check_bottom_buy(df, verbose=False):
    if len(df) < 10:
        return 0
    last = df.iloc[-1]
    close = last['close']
    ma5 = last['ma5']
    ma10 = last['ma10']
    vol_signal, _ = get_volume_signal(df)
    pct = last['pct_change']
    macd = last['macd']
    macd_signal = last['macd_signal']
    min_10 = df['low'].tail(10).min()

    score = 0
    if vol_signal == '放量突破' and pct > 3:
        score += 40
    elif vol_signal == '放量正常' and pct > 2:
        score += 20

    if close > ma5 and close > ma10:
        score += 30

    if close > min_10 * 1.05:
        score += 15

    if macd > macd_signal:
        score += 15

    if verbose:
        if score >= 70:
            print(f"  ✅ 底部试错得分 {score}: 强烈信号")
        elif score >= 40:
            print(f"  ⚠️ 底部试错得分 {score}: 一般信号")
        else:
            print(f"  ❌ 底部试错得分 {score}: 不满足")
    return score

def check_ma_bullish(df, verbose=False):
    if len(df) < 20:
        return 0
    last = df.iloc[-1]
    ma5 = last['ma5']
    ma10 = last['ma10']
    ma20 = last['ma20']
    close = last['close']

    score = 0
    if ma5 > ma10 > ma20:
        score += 50
        if close > ma5:
            score += 30
        else:
            score += 10
    else:
        if ma5 > ma10:
            score += 20
        if ma10 > ma20:
            score += 20
        if close > ma5:
            score += 10

    if verbose:
        if score >= 70:
            print(f"  ✅ 均线多头得分 {score}: 强烈信号")
        elif score >= 40:
            print(f"  ⚠️ 均线多头得分 {score}: 一般信号")
        else:
            print(f"  ❌ 均线多头得分 {score}: 不满足")
    return score

def check_macd_golden_cross(df, verbose=False):
    if len(df) < 2:
        return 0
    last = df.iloc[-1]
    prev = df.iloc[-2]
    macd = last['macd']
    signal = last['macd_signal']
    prev_macd = prev['macd']
    prev_signal = prev['macd_signal']

    score = 0
    if prev_macd <= prev_signal and macd > signal:
        score += 80
        if macd > 0:
            score += 20
    else:
        if macd > signal and prev_macd > prev_signal:
            score += 30

    if verbose:
        if score >= 80:
            print(f"  ✅ MACD金叉得分 {score}: 强烈信号")
        elif score >= 40:
            print(f"  ⚠️ MACD金叉得分 {score}: 一般信号")
        else:
            print(f"  ❌ MACD金叉得分 {score}: 不满足")
    return score

# ------------------------- 卖出条件判断（包含所有条件） -------------------------
def check_sell_conditions(df_etf, market_status, curr_return, hold_days, current_price, high_since_buy, atr_pct, sentiment_state, verbose=True):
    """
    检查所有卖出条件，返回卖出总分和触发的主要卖出类型（用于减仓比例）
    分数已优化，最大理论总分约1000（不含硬止损）
    """
    vol_signal, vol_details = get_volume_signal(df_etf)
    last = df_etf.iloc[-1]
    prev_vol_signal, _ = get_volume_signal(df_etf.iloc[:-1]) if len(df_etf) > 1 else ('无', {})
    sell_score = 0
    primary_sell_type = None

    # 1. 硬止损（强制卖出，得分1000不计入归一化，但作为触发信号）
    if curr_return <= HARD_STOP_LOSS:
        sell_score += 1000
        primary_sell_type = '硬止损'
        if verbose: print(f"  ✅ 硬止损：收益率 {curr_return*100:.2f}% ≤ {HARD_STOP_LOSS*100:.0f}%，得分 1000")
    else:
        if verbose: print(f"  ❌ 硬止损未触发：收益率 {curr_return*100:.2f}% > {HARD_STOP_LOSS*100:.0f}%")

    # 2. 时间止损 + 夏普
    if hold_days >= TIME_STOP_DAYS:
        recent_returns = df_etf['pct_change'].tail(min(10, len(df_etf))).dropna() / 100
        if len(recent_returns) >= 5:
            sharpe = recent_returns.mean() / (recent_returns.std() + 1e-6) * np.sqrt(252)
            if sharpe < 0 and abs(curr_return) <= TIME_STOP_RANGE:
                score = 100  # 提高权重
                sell_score += score
                if not primary_sell_type: primary_sell_type = '时间止损（夏普负）'
                if verbose: print(f"  ✅ 时间止损（夏普负）：夏普{sharpe:.2f}<0，收益率在±{TIME_STOP_RANGE*100:.0f}%内，得分 {score}")
            else:
                if verbose: print(f"  ❌ 时间止损未触发：夏普{sharpe:.2f}，收益率{curr_return*100:.2f}%")
        else:
            if verbose: print(f"  ❌ 时间止损未触发：数据不足")
    else:
        if verbose: print(f"  ❌ 时间止损未触发：持仓 {hold_days}天 < {TIME_STOP_DAYS}天")

    # 3. 分档止盈（提高权重）
    if curr_return >= TAKE_PROFIT_THRESHOLD:
        reached_levels = [lvl for lvl in TAKE_PROFIT_LEVELS if curr_return >= lvl]
        if reached_levels:
            max_level = max(reached_levels)
            idx = TAKE_PROFIT_LEVELS.index(max_level)
            # 第一档80，第二档100，第三档120
            score = 60 + idx * 20
            sell_score += score
            if not primary_sell_type: primary_sell_type = f'分档止盈_{int(max_level*100)}%'
            if verbose: print(f"  ✅ 分档止盈：收益率达到 {max_level*100:.0f}% 档，得分 {score}")
        else:
            if verbose: print(f"  ❌ 分档止盈未触发：未达到任何档位")
    else:
        if verbose: print(f"  ❌ 分档止盈未触发：收益率 {curr_return*100:.2f}% < {TAKE_PROFIT_THRESHOLD*100:.0f}%")

    # 4. 移动止盈（提高权重）
    if curr_return >= TAKE_PROFIT_THRESHOLD:
        base_tol = DRAWDOWN_TOLERANCE_BASE.get(market_status, 0.04)
        if DRAWDOWN_ADJUST_BY_ATR and atr_pct > 0:
            atr_factor = atr_pct / 2.0
            atr_factor = np.clip(atr_factor, 0.5, 1.5)
            tolerance = base_tol * atr_factor
        else:
            tolerance = base_tol

        if high_since_buy > 0:
            drawdown = (high_since_buy - current_price) / high_since_buy
            if drawdown >= tolerance:
                score = 100  # 原80
                sell_score += score
                if not primary_sell_type: primary_sell_type = '移动止盈'
                if verbose: print(f"  ✅ 移动止盈：回撤 {drawdown*100:.2f}% ≥ {tolerance*100:.1f}%，得分 {score}")
            else:
                if verbose: print(f"  ❌ 移动止盈未触发：回撤 {drawdown*100:.2f}% < {tolerance*100:.1f}%")
        else:
            if verbose: print(f"  ❌ 移动止盈：高点数据无效")
    else:
        if verbose: print(f"  ❌ 移动止盈未触发：收益率未达阈值")

    # 5. 成交量危险信号（提高权重）
    if vol_signal in ['放量下跌', '放量滞涨', '缩量上涨'] and prev_vol_signal in ['放量下跌', '放量滞涨', '缩量上涨']:
        base_score = 80
        if market_status == '防御区':
            score = base_score + 40  # 120
            if not primary_sell_type: primary_sell_type = '成交量危险信号+防御'
        elif market_status == '震荡区':
            score = base_score + 20  # 100
            if not primary_sell_type: primary_sell_type = '成交量危险信号+震荡'
        else:
            score = base_score       # 80
        sell_score += score
        if verbose: print(f"  ✅ 成交量危险信号：连续两日 {vol_signal}，大盘 {market_status}，得分 {score}")
    else:
        if verbose:
            if vol_signal in ['放量下跌', '放量滞涨', '缩量上涨']:
                print(f"  ❌ 成交量危险信号未触发：需要连续两日确认")
            else:
                print(f"  ❌ 成交量无危险信号：{vol_signal}")

    # 6. 情绪高潮卖出（提高权重）
    if sentiment_state == 'euphoric':
        score = 80  # 原60
        sell_score += score
        if not primary_sell_type: primary_sell_type = '情绪高潮卖出'
        if verbose: print(f"  ✅ 情绪高潮卖出：市场情绪狂热，得分 {score}")
    else:
        if verbose: print(f"  ❌ 情绪高潮未触发：当前情绪 {sentiment_state}")

    # 7. 短期均线跌破（5日、10日、20日）——提高权重
    if not pd.isna(last['ma5']) and current_price < last['ma5']:
        score = 30  # 原20
        sell_score += score
        if verbose: print(f"  ✅ 跌破5日均线：收盘价 {current_price:.4f} < MA5({last['ma5']:.4f})，得分 {score}")
    else:
        if verbose and not pd.isna(last['ma5']):
            print(f"  ❌ 未跌破5日均线：收盘价 {current_price:.4f} >= MA5({last['ma5']:.4f})")

    if not pd.isna(last['ma10']) and current_price < last['ma10']:
        score = 50  # 原30
        sell_score += score
        if verbose: print(f"  ✅ 跌破10日均线：收盘价 {current_price:.4f} < MA10({last['ma10']:.4f})，得分 {score}")
    else:
        if verbose and not pd.isna(last['ma10']):
            print(f"  ❌ 未跌破10日均线：收盘价 {current_price:.4f} >= MA10({last['ma10']:.4f})")

    if not pd.isna(last['ma20']) and current_price < last['ma20']:
        score = 80  # 原50
        sell_score += score
        if verbose: print(f"  ✅ 跌破20日均线：收盘价 {current_price:.4f} < MA20({last['ma20']:.4f})，得分 {score}")
    else:
        if verbose and not pd.isna(last['ma20']):
            print(f"  ❌ 未跌破20日均线：收盘价 {current_price:.4f} >= MA20({last['ma20']:.4f})")

    # 8. 均线止损（60日均线）——提高权重
    if not pd.isna(last['ma60']):
        ma60 = last['ma60']
        ma60_slope = ma60 - df_etf.iloc[-2]['ma60'] if len(df_etf) > 1 else 0
        if current_price < ma60 and ma60_slope <= 0:
            score = 100  # 原60
            sell_score += score
            if not primary_sell_type:
                primary_sell_type = '均线止损'
            if verbose:
                print(f"  ✅ 均线止损：收盘价 {current_price:.4f} < MA60({ma60:.4f})，均线走平/向下，得分 {score}")
        else:
            if verbose:
                print(f"  ❌ 均线止损未触发：收盘价 {current_price:.4f} >= MA60({ma60:.4f}) 或均线向上")
    else:
        if verbose:
            print("  ❌ 均线止损未触发：MA60数据不足")

    # 9. 新增：跌破20日低点
    if len(df_etf) >= 20:
        low_20 = df_etf['low'].tail(20).min()
        if current_price < low_20:
            score = 60
            sell_score += score
            if verbose: print(f"  ✅ 跌破20日低点：收盘价 {current_price:.4f} < 20日低点 {low_20:.4f}，得分 {score}")
        else:
            if verbose: print(f"  ❌ 未跌破20日低点：收盘价 {current_price:.4f} >= 20日低点 {low_20:.4f}")

    # 10. 新增：MACD死叉
    if len(df_etf) >= 2:
        last = df_etf.iloc[-1]
        prev = df_etf.iloc[-2]
        if prev['macd'] >= prev['macd_signal'] and last['macd'] < last['macd_signal']:
            score = 60
            sell_score += score
            if verbose: print(f"  ✅ MACD死叉成立：MACD({last['macd']:.4f}) 下穿 Signal({last['macd_signal']:.4f})，得分 {score}")
        else:
            if verbose: print(f"  ❌ MACD死叉未触发")

    return sell_score, primary_sell_type

# ------------------------- 仓位计算辅助函数 -------------------------
def calculate_add_amount(current_amount, ratio, atr_pct=None, base_ratio=None):
    if current_amount <= 0 or ratio <= 0:
        return 0
    if ATR_POSITION_ADJUST and atr_pct is not None and base_ratio is not None:
        atr_factor = base_ratio / (atr_pct + 1e-6) * 0.02
        atr_factor = np.clip(atr_factor, 0.5, 2.0)
        ratio_adj = ratio * atr_factor
    else:
        ratio_adj = ratio
    return current_amount * ratio_adj

def calculate_reduce_amount(current_amount, ratio):
    if current_amount <= 0 or ratio <= 0:
        return 0
    return current_amount * ratio

def suggest_first_buy(available_cash, signal_strength, atr_pct=None):
    if signal_strength == 'strong':
        base = 0.3
    elif signal_strength == 'cautious':
        base = 0.2
    else:
        base = 0.1
    if ATR_POSITION_ADJUST and atr_pct is not None:
        atr_factor = 2.0 / (atr_pct + 1e-6) * 0.5
        atr_factor = np.clip(atr_factor, 0.5, 1.5)
        base *= atr_factor
    max_amount = available_cash * MAX_POSITION_RATIO
    return min(base * available_cash, max_amount)

def score_to_strength(buy_score):
    if buy_score >= 80:
        return 'strong'
    elif buy_score >= 50:
        return 'cautious'
    else:
        return 'weak'

# ------------------------- 单只ETF处理函数（支持实时价格） -------------------------
def process_single_etf(etf_code, name, user_return, hold_days, current_amount, available_cash=0, verbose=False):
    sentiment_score, sentiment_state, _ = get_market_sentiment()

    df_etf = get_k_data(etf_code, days=250, min_required=20)
    if df_etf is None or len(df_etf) < 20:
        return f"{etf_code}\t{name}\t0\t0\t数据不足\t无数据"

    df_etf = add_indicators(df_etf)
    market_status = get_market_status()

    # 获取日线数据中的最近收盘价
    last = df_etf.iloc[-1]
    atr_pct = last['atr_pct'] if not pd.isna(last['atr_pct']) else 2.0
    high_since_buy = df_etf['high'].tail(min(MA_LONG, len(df_etf))).max()

    # 获取实时价格（仅当verbose=True的单只模式时打印信息，批量模式静默）
    current_price = last['close']
    real_price = get_realtime_price(etf_code)
    if real_price is not None:
        current_price = real_price
        # 此函数在批量模式中调用，不涉及实时收益率更新

    # 计算买入总分（仍基于日线数据）
    buy_score = 0
    buy_score += check_breakout_buy(df_etf, verbose=verbose)
    buy_score += check_pullback_buy(df_etf, verbose=verbose)
    buy_score += check_bottom_buy(df_etf, verbose=verbose)
    buy_score += check_ma_bullish(df_etf, verbose=verbose)
    buy_score += check_macd_golden_cross(df_etf, verbose=verbose)

    # 计算卖出总分（使用传入的user_return和current_price）
    sell_score, primary_sell_type = check_sell_conditions(
        df_etf, market_status, user_return, hold_days, current_price, high_since_buy, atr_pct, sentiment_state, verbose=verbose
    )

    # 归一化到10分制
    norm_buy = min(10, round(buy_score * 10 / MAX_BUY_SCORE))
    norm_sell = min(10, round(sell_score * 10 / MAX_SELL_SCORE))

    # 决策逻辑
    if norm_sell >= SELL_THRESHOLD and norm_sell > norm_buy:
        reduce_ratio = SELL_REDUCE_RATIO.get(primary_sell_type, 0.5) if primary_sell_type else 0.5
        if current_amount > 0:
            reduce_amount = calculate_reduce_amount(current_amount, reduce_ratio)
            new_amount = current_amount - reduce_amount
            if reduce_ratio >= 1.0:
                action = "清仓"
                detail = f"减仓{reduce_amount:.0f}->{new_amount:.0f}"
            else:
                fraction = {0.5: '1/2', 1/3: '1/3', 0.25: '1/4', 0.3: '30%', 0.4: '40%'}.get(reduce_ratio, f'{reduce_ratio*100:.0f}%')
                action = f"减仓{fraction}"
                detail = f"减仓{reduce_amount:.0f}->{new_amount:.0f}"
        else:
            action = "卖出"
            detail = "持仓为0，无需操作"
        return f"{etf_code}\t{name}\t{norm_buy}\t{norm_sell}\t卖出\t{detail}"
    elif norm_buy >= BUY_THRESHOLD and norm_buy > norm_sell:
        strength = score_to_strength(buy_score)
        base_ratio = BUY_ADD_RATIO_BASE.get(strength, 0.25)
        if current_amount <= 0:
            if available_cash > 0:
                first_amount = suggest_first_buy(available_cash, strength, atr_pct)
                detail = f"首次建仓建议{first_amount:.0f}"
            else:
                detail = f"信号强度{strength}，参考仓位{base_ratio*100:.0f}%"
        else:
            add_amount = calculate_add_amount(current_amount, base_ratio, atr_pct, base_ratio)
            if add_amount > 0:
                new_amount = current_amount + add_amount
                fraction = {0.5: '1/2', 1/3: '1/3', 0.25: '1/4'}.get(base_ratio, f'{base_ratio*100:.0f}%')
                detail = f"加仓{fraction}: {add_amount:.0f}->{new_amount:.0f}"
            else:
                detail = "加仓金额为0"
        return f"{etf_code}\t{name}\t{norm_buy}\t{norm_sell}\t买入\t{detail}"
    else:
        if current_amount > 0:
            detail = "持有"
        else:
            detail = "空仓"
        return f"{etf_code}\t{name}\t{norm_buy}\t{norm_sell}\t持有\t{detail}"

# ------------------------- 主函数 -------------------------
def main():
    if len(sys.argv) == 1:
        print('用法:')
        print('  单只模式: python trade.py <ETF代码> <收益率/成本价> <持仓天数> <当前持仓金额> [可用资金]')
        print('  批量模式: python trade.py <Excel文件路径>')
        sys.exit()

    # 批量模式
    if len(sys.argv) == 2 and os.path.isfile(sys.argv[1]) and (sys.argv[1].endswith('.xlsx') or sys.argv[1].endswith('.xls')):
        excel_file = sys.argv[1]
        print(f'========== 批量分析：{excel_file} ==========')
        try:
            df = pd.read_excel(excel_file)
        except Exception as e:
            print(f'读取Excel失败: {e}')
            sys.exit()

        if df.shape[1] < 6:
            print('Excel列数不足，需要至少6列：代码、名称、当前收益率、持有日期、当前仓位、当前剩余可用资金')
            sys.exit()

        df.columns = ['code', 'name', 'return_str', 'hold_days', 'current_amount', 'available_cash'][:df.shape[1]]
        print("序号\t\t名称\t\t代码\t\t收益率\t\t持有\t\t仓位\t\t可用资金\t\t买入分\t\t卖出分\t\t建议\t\t详情")
        results = []
        for idx, row in df.iterrows():
            code = str(row['code']).strip()
            name = str(row['name']).strip()
            return_str = str(row['return_str']).strip()
            hold_days = int(row['hold_days'])
            current_amount = float(row['current_amount'])
            if pd.isna(row['available_cash']):
                available_cash = 0.0
            else:
                available_cash = float(row['available_cash'])

            if return_str.endswith('%'):
                user_return = float(return_str.strip('%')) / 100
            else:
                try:
                    user_return = float(return_str)
                except:
                    print(f"第{idx+2}行收益率格式错误: {return_str}，跳过")
                    continue

            if '.' not in code:
                if code.startswith(('5', '6')):
                    code = 'sh.' + code
                else:
                    code = 'sz.' + code

            result = process_single_etf(code, name, user_return, hold_days, current_amount, available_cash=available_cash, verbose=False)
            results.append(result)

            parts = result.split('\t')
            if len(parts) == 6:
                res_code, res_name, buy_score, sell_score, action, detail = parts
                print(f"{idx+2}\t\t{res_name}\t\t{res_code}\t\t{return_str}\t\t{hold_days}\t\t{current_amount:.2f}\t\t{available_cash:.2f}\t\t{buy_score}\t\t{sell_score}\t\t{action}\t\t{detail}")
            else:
                print(result)

        print('\n========== 批量分析完成 ==========')
        output_file = excel_file.replace('.xlsx', '_result.txt').replace('.xls', '_result.txt')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("代码\t名称\t买入总分\t卖出总分\t建议\t详情\n")
            for r in results:
                f.write(r + '\n')
        print(f'结果已保存至 {output_file}')
        return

    # 单只模式
    if len(sys.argv) not in [5, 6]:
        print('用法: python trade.py <ETF代码> <收益率/成本价> <持仓天数> <当前持仓金额> [可用资金]')
        print('示例: python trade.py sz.159326 0 0 10000')
        sys.exit()

    etf_code = sys.argv[1]
    arg2 = sys.argv[2]
    hold_days = int(sys.argv[3])
    try:
        current_amount = float(sys.argv[4])
    except ValueError:
        print('错误：持仓金额必须是数字')
        sys.exit()
    available_cash = float(sys.argv[5]) if len(sys.argv) == 6 else 0

    # 解析收益率/成本价
    if arg2.endswith('%'):
        user_return = float(arg2.strip('%')) / 100
        cost_price = None
        print(f'解析为收益率: {user_return*100:.2f}%')
    else:
        try:
            user_return = float(arg2)
            cost_price = None
            print(f'解析为收益率: {user_return*100:.2f}%')
        except ValueError:
            cost_price = float(arg2)
            user_return = None
            print(f'解析为成本价: {cost_price}')

    print(f'当前持仓金额: {current_amount:.2f}')
    if available_cash > 0:
        print(f'可用资金: {available_cash:.2f}')
    print(f'========== 策略分析（10分制）：{etf_code} ==========')

    sentiment_score, sentiment_state, _ = get_market_sentiment()
    print(f'\n市场情绪评分: {sentiment_score:.1f} (状态: {sentiment_state})')

    df_etf = get_k_data(etf_code, days=250, min_required=20)
    if df_etf is None or len(df_etf) < 20:
        print('无法获取足够的ETF数据（至少需要20天），分析终止。')
        return
    df_etf = add_indicators(df_etf)

    market_status = get_market_status()
    print(f'大盘状态: {market_status}')

    # 获取日线数据中的最近收盘价（作为后备）
    last = df_etf.iloc[-1]
    current_price = last['close']
    lookback = min(MA_LONG, len(df_etf))
    high_since_buy = df_etf['high'].tail(lookback).max()
    atr_pct = last['atr_pct'] if not pd.isna(last['atr_pct']) else 2.0

    # 获取实时价格
    real_price = get_realtime_price(etf_code)
    if real_price is not None:
        current_price = real_price
        print(f'实时价格: {current_price:.4f}')
        if cost_price is not None:
            curr_return = (current_price - cost_price) / cost_price
            print(f'实时收益率: {curr_return*100:.2f}%')
        else:
            curr_return = user_return
            print('注：您输入的是收益率，无法计算实时收益率，仍使用原收益率。')
    else:
        print('获取实时价格失败，使用昨日收盘价。')
        if user_return is not None:
            curr_return = user_return
        else:
            curr_return = (current_price - cost_price) / cost_price

    print(f'\n最新交易日: {last["date"].strftime("%Y-%m-%d")}')
    print(f'当前价格: {current_price:.4f}, 当前收益率: {curr_return*100:.2f}%')
    print(f'近期最高价(过去{lookback}天): {high_since_buy:.4f}')
    print(f'ATR百分比: {atr_pct:.2f}%')

    vol_signal, vol_details = get_volume_signal(df_etf)
    vol_80_display = f"{vol_details.get('vol_80', 0):.2e}" if not pd.isna(vol_details.get('vol_80', np.nan)) else "数据不足"
    vol_20_display = f"{vol_details.get('vol_20', 0):.2e}" if not pd.isna(vol_details.get('vol_20', np.nan)) else "数据不足"
    print(f'成交量: {last["volume"]:.2e}, 80分位: {vol_80_display}, 20分位: {vol_20_display}')
    print(f'成交量信号: {vol_signal}')

    print('\n--- 检查买入条件 ---')
    buy_score = 0
    buy_score += check_breakout_buy(df_etf, verbose=True)
    buy_score += check_pullback_buy(df_etf, verbose=True)
    buy_score += check_bottom_buy(df_etf, verbose=True)
    buy_score += check_ma_bullish(df_etf, verbose=True)
    buy_score += check_macd_golden_cross(df_etf, verbose=True)
    print(f'\n买入总分（原始）: {buy_score}')

    print('\n--- 检查卖出条件 ---')
    sell_score, primary_sell_type = check_sell_conditions(
        df_etf, market_status, curr_return, hold_days, current_price, high_since_buy, atr_pct, sentiment_state, verbose=True
    )
    print(f'\n卖出总分（原始）: {sell_score}')

    norm_buy = min(10, round(buy_score * 10 / MAX_BUY_SCORE))
    norm_sell = min(10, round(sell_score * 10 / MAX_SELL_SCORE))
    print(f'\n归一化后（10分制）：买入 {norm_buy}，卖出 {norm_sell}')

    print('\n--- 最终结论 ---')
    if norm_sell >= SELL_THRESHOLD and norm_sell > norm_buy:
        print(f'✅ 卖出信号占优 (卖出 {norm_sell} > 买入 {norm_buy})')
        reduce_ratio = SELL_REDUCE_RATIO.get(primary_sell_type, 0.5) if primary_sell_type else 0.5
        if current_amount > 0:
            reduce_amount = calculate_reduce_amount(current_amount, reduce_ratio)
            new_amount = current_amount - reduce_amount
            if reduce_ratio >= 1.0:
                action_desc = '清仓'
            else:
                fraction = {0.5: '1/2', 1/3: '1/3', 0.25: '1/4', 0.3: '30%', 0.4: '40%'}.get(reduce_ratio, f'{reduce_ratio*100:.0f}%')
                action_desc = f'减仓 {fraction}'
            print(f'操作建议: {action_desc}')
            print(f'减仓金额: {reduce_amount:.2f}, 剩余持仓: {new_amount:.2f}')
        else:
            print('当前持仓为0，无需卖出')
    elif norm_buy >= BUY_THRESHOLD and norm_buy > norm_sell:
        print(f'✅ 买入信号占优 (买入 {norm_buy} > 卖出 {norm_sell})')
        strength = score_to_strength(buy_score)
        base_ratio = BUY_ADD_RATIO_BASE.get(strength, 0.25)
        if current_amount <= 0:
            if available_cash > 0:
                first_amount = suggest_first_buy(available_cash, strength, atr_pct)
                print(f'⚠️ 当前持仓为0，建议首次建仓')
                print(f'建议建仓金额: {first_amount:.2f} (基于可用资金 {available_cash:.2f} 和信号强度 {strength})')
            else:
                print(f'当前持仓为0，未提供可用资金，无法计算建仓金额。信号强度 {strength}，可参考 {base_ratio*100:.0f}% 仓位。')
        else:
            add_amount = calculate_add_amount(current_amount, base_ratio, atr_pct, base_ratio)
            if add_amount > 0:
                new_amount = current_amount + add_amount
                fraction = {0.5: '1/2', 1/3: '1/3', 0.25: '1/4'}.get(base_ratio, f'{base_ratio*100:.0f}%')
                print(f'操作建议: 加仓 {fraction}')
                print(f'加仓金额: {add_amount:.2f}, 新持仓: {new_amount:.2f}')
            else:
                print('操作建议: 持有（加仓金额为0）')
    else:
        print(f'买入 {norm_buy}，卖出 {norm_sell}，均未显著占优')
        if current_amount > 0:
            print('综合建议：持有。')
        else:
            print('综合建议：空仓观望。')

if __name__ == '__main__':
    main()