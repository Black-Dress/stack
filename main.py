#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统（优化版）- 市场状态精细化 + 信号优先级排序
功能：实时分析 ETF，输出评分和操作建议（买入/卖出/持有）。
依赖：baostock, pandas, numpy, requests, openai
"""

import os
import json
import datetime
import logging
import requests
import numpy as np
import pandas as pd
import baostock as bs
import hashlib
import openai
import re
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor

# ---------------------------- 日志配置（屏蔽 HTTP 请求日志） ----------------------------
# 设置根日志级别为 INFO，保留自定义 INFO 日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 屏蔽 OpenAI 及其底层 HTTP 库的 INFO 日志（去除 "HTTP Request: POST ..." 输出）
for lib in ["openai", "httpx", "httpcore"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# ---------------------------- 配置 ----------------------------
ETF_MA = 20
ETF_VOL_MA = 5
MACRO_INDEX = "sh.000300"
MARKET_INDEX = "sh.000001"
MACRO_MA_SHORT = 20
MACRO_MA_LONG = 60
RSI_PERIOD = 14
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0
ATR_TRAILING_MULT = 1.0
PROFIT_TARGET = 0.15
CONFIRM_DAYS = 3
BUY_THRESHOLD = 0.5
SELL_THRESHOLD = -0.2
QUICK_BUY_THRESHOLD = 0.6
RECENT_HIGH_WINDOW = 10
RECENT_LOW_WINDOW = 20
WEEKLY_MA = 20
RISK_WARNING_DAYS = 3
RISK_WARNING_THRESHOLD = -0.1
POSITION_FILE = "positions.csv"
STATE_FILE = "etf_state.json"
CACHE_FILE = "weight_cache.json"

# 默认权重（会被AI覆盖）
DEFAULT_BUY_WEIGHTS = {
    "price_above_ma20": 0.30, "volume_above_ma5": 0.20, "macd_golden_cross": 0.15,
    "kdj_golden_cross": 0.15, "bollinger_break_up": 0.10, "williams_oversold": 0.10,
    "market_above_ma20": 0.10, "market_above_ma60": 0.10, "market_amount_above_ma20": 0.10,
    "outperform_market": 0.20, "weekly_above_ma20": 0.20,
}
DEFAULT_SELL_WEIGHTS = {
    "price_below_ma20": 0.40, "bollinger_break_down": 0.20, "williams_overbought": 0.10,
    "rsi_overbought": 0.15, "underperform_market": 0.20, "stop_loss_ma_break": 1.00,
    "trailing_stop_clear": 1.00, "trailing_stop_half": 0.50, "profit_target_hit": 0.30,
    "weekly_below_ma20": 0.20,
}

# 默认参数（会被AI覆盖）
DEFAULT_PARAMS = {
    "CONFIRM_DAYS": 3,
    "BUY_THRESHOLD": 0.5,
    "SELL_THRESHOLD": -0.2,
    "QUICK_BUY_THRESHOLD": 0.6,
    "RECENT_HIGH_WINDOW": 10,
    "RECENT_LOW_WINDOW": 20,
}

# ---------------------------- 数据获取 ----------------------------
def silent_login():
    with open(os.devnull, "w") as f, redirect_stdout(f):
        lg = bs.login()
    if lg.error_code != "0":
        print(f"登录失败: {lg.error_msg}")
        return False
    return True

def silent_logout():
    with open(os.devnull, "w") as f, redirect_stdout(f):
        bs.logout()

def get_daily_data(code, start_date, end_date):
    rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close,volume,amount",
                                      start_date=start_date, end_date=end_date, frequency="d")
    if rs.error_code != "0":
        return None
    data = []
    while rs.next():
        data.append(rs.get_row_data())
    if not data:
        return None
    df = pd.DataFrame(data, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    for col in ["open","high","low","close","volume","amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def get_weekly_data(code, start_date, end_date):
    df = get_daily_data(code, start_date, end_date)
    if df is None: return None
    weekly = df.resample("W-FRI").last()
    weekly["ma_short"] = weekly["close"].rolling(window=WEEKLY_MA).mean()
    return weekly

def get_realtime_price_sina(code):
    try:
        for domain in ["hq.sinajs.cn", "hq2.sinajs.cn", "hq3.sinajs.cn"]:
            url = f"http://{domain}/list={code.replace('.','')}"
            r = requests.get(url, headers={"Referer": "http://finance.sina.com.cn"}, timeout=3)
            if r.status_code == 200:
                parts = r.text.split('"')[1].split(",")
                if len(parts) > 3 and parts[3]:
                    return float(parts[3])
        return None
    except:
        return None

def get_realtime_index_sina(code="sh000001"):
    return get_realtime_price_sina(code)

def load_positions():
    df = pd.read_csv(POSITION_FILE, encoding="utf-8-sig")
    return df[["代码", "名称"]]

def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ---------------------------- 技术指标 ----------------------------
def calculate_atr(df, period=14):
    tr = pd.concat([df["high"]-df["low"], abs(df["high"]-df["close"].shift()), abs(df["low"]-df["close"].shift())], axis=1).max(1)
    return tr.rolling(period).mean()


def calculate_indicators(
    df, need_amount_ma=True, recent_high_window=10, recent_low_window=20
):
    df = df.copy()
    df["ma_short"] = df["close"].rolling(window=ETF_MA).mean()
    df["vol_ma"] = df["volume"].rolling(window=ETF_VOL_MA).mean()
    if need_amount_ma:
        df["amount_ma"] = df["amount"].rolling(window=ETF_VOL_MA).mean()
    # MACD
    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd_dif"] = exp1 - exp2
    df["macd_dea"] = df["macd_dif"].ewm(span=9, adjust=False).mean()
    # KDJ
    low_n = df["low"].rolling(9).min()
    high_n = df["high"].rolling(9).max()
    rsv = (df["close"] - low_n) / (high_n - low_n) * 100
    df["kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1/3, adjust=False).mean()
    # 布林
    df["boll_mid"] = df["close"].rolling(20).mean()
    df["boll_std"] = df["close"].rolling(20).std()
    df["boll_up"] = df["boll_mid"] + 2*df["boll_std"]
    df["boll_low"] = df["boll_mid"] - 2*df["boll_std"]
    # 威廉
    high_14 = df["high"].rolling(14).max()
    low_14 = df["low"].rolling(14).min()
    df["williams_r"] = (high_14 - df["close"]) / (high_14 - low_14) * -100
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta>0,0).rolling(14).mean()
    loss = (-delta.where(delta<0,0)).rolling(14).mean()
    df["rsi"] = 100 - 100/(1+gain/loss)
    # ATR
    df["atr"] = calculate_atr(df, ATR_PERIOD)
    df[f"recent_high_{recent_high_window}"] = (
        df["high"].rolling(recent_high_window).max()
    )
    df[f"recent_low_{recent_low_window}"] = df["low"].rolling(recent_low_window).min()
    return df


def get_sentiment_factor(macro_df):
    if len(macro_df) < RSI_PERIOD+1:
        return 1.0
    delta = macro_df["close"].diff()
    gain = delta.where(delta>0,0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta<0,0)).rolling(RSI_PERIOD).mean()
    rsi = 100 - 100/(1+gain/loss)
    latest_rsi = rsi.iloc[-1]
    if latest_rsi < 30: return 0.6
    if latest_rsi < 50: return 0.8
    if latest_rsi < 70: return 1.0
    return 0.9

# ---------------------------- AI 权重生成（优化版） ----------------------------
def _get_cache_key(macro_status, sentiment_factor, market_above_ma20, market_above_ma60, market_amount_above_ma20, volatility):
    return hashlib.md5(f"{macro_status}_{sentiment_factor}_{market_above_ma20}_{market_above_ma60}_{market_amount_above_ma20}_{volatility:.3f}".encode()).hexdigest()

def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            return json.load(open(CACHE_FILE, "r", encoding="utf-8"))
        except:
            return {}
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _validate_and_filter_weights(weights, expected_keys, name):
    """校验并过滤权重：只保留expected_keys中的键，缺失补0，然后归一化"""
    if not isinstance(weights, dict):
        logger.error(f"{name} 不是字典，使用默认权重")
        return None
    filtered = {k: weights.get(k, 0.0) for k in expected_keys}
    for k, v in filtered.items():
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            filtered[k] = max(0.0, min(1.0, v))
    total = sum(filtered.values())
    if total == 0:
        for k in filtered:
            filtered[k] = 1.0 / len(filtered)
    elif abs(total - 1.0) > 1e-6:
        for k in filtered:
            filtered[k] /= total
    return filtered

def build_optimized_prompt(macro_status, sentiment_factor, market_above_ma20, market_above_ma60,
                           market_amount_above_ma20, volatility):
    buy_keys = ["price_above_ma20", "volume_above_ma5", "macd_golden_cross", "kdj_golden_cross",
                "bollinger_break_up", "williams_oversold", "market_above_ma20", "market_above_ma60",
                "market_amount_above_ma20", "outperform_market", "weekly_above_ma20"]
    sell_keys = ["price_below_ma20", "bollinger_break_down", "williams_overbought", "rsi_overbought",
                 "underperform_market", "stop_loss_ma_break", "trailing_stop_clear", "trailing_stop_half",
                 "profit_target_hit", "weekly_below_ma20"]
    prompt = f"""
你是一个量化交易策略专家。当前市场环境如下：
- 宏观状态：{macro_status}（bull牛市、oscillate震荡市、bear熊市）
- 情绪系数：{sentiment_factor}（0.6=恐慌，0.8=偏弱，1.0=中性，0.9=偏热）
- 大盘站上20日均线：{"是" if market_above_ma20 else "否"}
- 大盘站上60日均线：{"是" if market_above_ma60 else "否"}
- 市场成交额高于20日均额：{"是" if market_amount_above_ma20 else "否"}
- 市场波动率(ATR/收盘价)：{volatility:.3f}（<0.01低波动，>0.02高波动）

请根据以下规则分配买入权重和卖出权重，所有权重为0-1之间的浮点数，每个部分总和为1。

【买入权重规则】
1. 牛市且大盘站上60日线：趋势类因子(price_above_ma20, market_above_ma60, weekly_above_ma20)总权重应≥0.5，单个因子0.15-0.30。
2. 震荡市：反转类因子(williams_oversold, bollinger_break_up)总权重应≥0.4，单个因子0.20-0.40。
3. 熊市：买入总权重建议≤0.3，可分配给超跌反弹因子(williams_oversold, bollinger_break_up)，其他趋势因子设为0。
4. 高波动时(>0.02)：降低price_above_ma20权重，提高williams_oversold和volume_above_ma5。
5. 低波动时(<0.01)：可适当提高macd_golden_cross和kdj_golden_cross。

【卖出权重规则】
1. 熊市且大盘跌破60日线：止损类因子(stop_loss_ma_break, trailing_stop_clear)总权重应≥0.6，单个因子可达0.4-0.8。
2. 牛市：止盈类因子(profit_target_hit)和超买因子(rsi_overbought)总权重应≥0.5。
3. 震荡市：平衡止损和止盈，各约0.5。
4. 高波动时：提高profit_target_hit权重（因容易触发止盈），降低trailing_stop_clear阈值效应。
5. 任何市场下，若认为应完全空仓：买入所有权重设为0，卖出权重全部给stop_loss_ma_break（设为1.0）。

【约束条件】
- 禁止添加任何未列出的键（如“空仓”、“观望”等）。
- 单个因子权重不得超过0.6（除熊市卖出止损因子可到0.8外）。
- 每个部分所有权重必须为正数（可0），总和为1（允许浮点误差）。

【输出格式示例】
牛市示例：
{{"buy": {{"price_above_ma20":0.25,"market_above_ma60":0.20,"weekly_above_ma20":0.15,"volume_above_ma5":0.10,"outperform_market":0.10,"macd_golden_cross":0.10,"kdj_golden_cross":0.10}}, "sell": {{"profit_target_hit":0.40,"rsi_overbought":0.30,"underperform_market":0.30}}}}
熊市示例：
{{"buy": {{"williams_oversold":0.20,"bollinger_break_up":0.10}}, "sell": {{"stop_loss_ma_break":0.70,"trailing_stop_clear":0.30}}}}

请严格按照JSON格式输出，不要包含任何解释文字。
"""
    return prompt

def deepseek_generate_weights(macro_status, sentiment_factor, market_above_ma20, market_above_ma60,
                              market_amount_above_ma20, volatility, api_key, use_cache=True):
    cache_key = _get_cache_key(macro_status, sentiment_factor, market_above_ma20, market_above_ma60,
                               market_amount_above_ma20, volatility)
    if use_cache:
        cache = _load_cache()
        if cache_key in cache:
            cached = cache[cache_key]
            if "buy" in cached and "sell" in cached:
                buy = _validate_and_filter_weights(cached["buy"], DEFAULT_BUY_WEIGHTS.keys(), "缓存买入权重")
                sell = _validate_and_filter_weights(cached["sell"], DEFAULT_SELL_WEIGHTS.keys(), "缓存卖出权重")
                if buy and sell:
                    return buy, sell
                else:
                    logger.warning("缓存权重无效，重新生成")

    prompt = build_optimized_prompt(macro_status, sentiment_factor, market_above_ma20, market_above_ma60,
                                    market_amount_above_ma20, volatility)
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"system","content":"你是一个量化交易专家，输出严格符合要求的JSON。"},
                      {"role":"user","content":prompt}],
            max_tokens=800, temperature=0.0, timeout=10
        )
        import re
        content = resp.choices[0].message.content
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ValueError("未找到JSON")
        data = json.loads(json_match.group())
        if "buy" not in data or "sell" not in data:
            raise ValueError("JSON缺少buy或sell字段")
        buy_weights = _validate_and_filter_weights(data["buy"], DEFAULT_BUY_WEIGHTS.keys(), "AI买入权重")
        sell_weights = _validate_and_filter_weights(data["sell"], DEFAULT_SELL_WEIGHTS.keys(), "AI卖出权重")
        if buy_weights is None or sell_weights is None:
            raise ValueError("权重校验失败")
        if use_cache:
            cache = _load_cache()
            cache[cache_key] = {"buy": buy_weights, "sell": sell_weights}
            _save_cache(cache)
        return buy_weights, sell_weights
    except Exception as e:
        logger.error(f"AI权重生成失败: {e}，使用默认权重")
        return DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()


def build_optimized_prompt_for_params(
    macro_status,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    volatility,
):
    prompt = f"""
你是一个量化交易策略专家。当前市场环境如下：
- 宏观状态：{macro_status}（bull牛市、oscillate震荡市、bear熊市）
- 情绪系数：{sentiment_factor}（0.6=恐慌，0.8=偏弱，1.0=中性，0.9=偏热）
- 大盘站上20日均线：{"是" if market_above_ma20 else "否"}
- 大盘站上60日均线：{"是" if market_above_ma60 else "否"}
- 市场成交额高于20日均额：{"是" if market_amount_above_ma20 else "否"}
- 市场波动率(ATR/收盘价)：{volatility:.3f}（<0.01低波动，>0.02高波动）

请根据市场环境调整以下交易参数，所有参数为整数或浮点数。

【参数调整规则】
1. CONFIRM_DAYS（确认天数，1-5）：高波动时增加到4-5以避免假信号，低波动时减少到2-3以快速响应。
2. BUY_THRESHOLD（买入阈值，0.3-0.7）：牛市降低到0.4-0.5，熊市提高到0.6-0.7，震荡市0.5左右。
3. SELL_THRESHOLD（卖出阈值，-0.5到-0.1）：熊市降低到-0.3到-0.2，牛市提高到-0.1左右。
4. QUICK_BUY_THRESHOLD（快速买入阈值，0.5-0.8）：高波动时提高到0.7-0.8，低波动时降低到0.5-0.6。
5. RECENT_HIGH_WINDOW（近期高点窗口，5-20）：高波动时缩短到5-10，低波动时延长到15-20。
6. RECENT_LOW_WINDOW（近期低点窗口，10-30）：高波动时缩短到10-15，低波动时延长到20-30。

【输出格式示例】
{{"CONFIRM_DAYS":3,"BUY_THRESHOLD":0.5,"SELL_THRESHOLD":-0.2,"QUICK_BUY_THRESHOLD":0.6,"RECENT_HIGH_WINDOW":10,"RECENT_LOW_WINDOW":20}}

请严格按照JSON格式输出，不要包含任何解释文字。
"""
    return prompt


def _validate_and_filter_params(params, expected_keys, name):
    """校验并过滤参数：只保留expected_keys中的键，缺失补默认值"""
    if not isinstance(params, dict):
        logger.error(f"{name} 不是字典，使用默认参数")
        return None
    filtered = {}
    defaults = DEFAULT_PARAMS
    for k in expected_keys:
        v = params.get(k, defaults[k])
        if k == "CONFIRM_DAYS" and not (1 <= v <= 5):
            v = defaults[k]
        elif k in ["BUY_THRESHOLD", "SELL_THRESHOLD", "QUICK_BUY_THRESHOLD"] and not (
            -1 <= v <= 1
        ):
            v = defaults[k]
        elif k == "RECENT_HIGH_WINDOW" and not (5 <= v <= 20):
            v = defaults[k]
        elif k == "RECENT_LOW_WINDOW" and not (10 <= v <= 30):
            v = defaults[k]
        filtered[k] = v
    return filtered


def deepseek_generate_params(
    macro_status,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    volatility,
    api_key,
    use_cache=True,
):
    import re

    cache_key = (
        _get_cache_key(
            macro_status,
            sentiment_factor,
            market_above_ma20,
            market_above_ma60,
            market_amount_above_ma20,
            volatility,
        )
        + "_params"
    )
    if use_cache:
        cache = _load_cache()
        if cache_key in cache and "params" in cache[cache_key]:
            params = _validate_and_filter_params(
                cache[cache_key]["params"], DEFAULT_PARAMS.keys(), "缓存参数"
            )
            if params:
                return params
            else:
                logger.warning("缓存参数无效，重新生成")

    prompt = build_optimized_prompt_for_params(
        macro_status,
        sentiment_factor,
        market_above_ma20,
        market_above_ma60,
        market_amount_above_ma20,
        volatility,
    )
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个量化交易专家，输出严格符合要求的JSON。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=400,
            temperature=0.0,
            timeout=10,
        )
        content = resp.choices[0].message.content
        import re

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ValueError("未找到JSON")
        data = json.loads(json_match.group())
        params = _validate_and_filter_params(data, DEFAULT_PARAMS.keys(), "AI参数")
        if params is None:
            raise ValueError("参数校验失败")
        if use_cache:
            cache = _load_cache()
            cache[cache_key] = {"params": params}
            _save_cache(cache)
        return params
    except Exception as e:
        logger.error(f"AI参数生成失败: {e}，使用默认参数")
        return DEFAULT_PARAMS.copy()


# ---------------------------- 市场状态精细化 ----------------------------
def refine_market_state(market_df, api_key, use_cache=True):
    """调用AI分析大盘数据，返回精细市场状态和推荐市场因子"""
    cache_key = f"market_state_{hashlib.md5(market_df.tail(20).to_json().encode()).hexdigest()}"
    if use_cache:
        cache = _load_cache()
        if cache_key in cache and "market_state" in cache[cache_key]:
            state = cache[cache_key]["market_state"]
            factor = cache[cache_key]["market_factor"]
            return state, factor

    recent = market_df.tail(20)
    close_pct = recent['close'].pct_change().mean()
    vol_pct = recent['volume'].pct_change().mean()
    volatility = (recent['close'].pct_change().std()) * 100
    above_ma20 = recent['close'].iloc[-1] > recent['ma_short'].iloc[-1]
    above_ma60 = recent['close'].iloc[-1] > recent['ma_long'].iloc[-1] if 'ma_long' in recent else above_ma20

    prompt = f"""
你是一个市场分析专家。请根据以下大盘指数(上证)最近20日的数据，判断当前市场状态，并给出一个推荐的市场因子系数（0.6-1.4，1.0为中性，>1.0为积极，<1.0为防御）。
数据摘要：
- 最近20日平均日涨跌幅：{close_pct:.4f}
- 最近20日平均成交量变化率：{vol_pct:.4f}
- 日波动率（标准差%）：{volatility:.2f}%
- 当前价格是否站上20日均线：{"是" if above_ma20 else "否"}
- 当前价格是否站上60日均线：{"是" if above_ma60 else "否"}

请输出JSON格式：{{"state": "市场状态标签", "factor": 市场因子系数}}
可选市场状态标签：强势牛市、正常牛市、震荡偏强、震荡偏弱、弱势反弹、熊市下跌中继、熊市加速下跌。
不要包含其他解释。
"""
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"system","content":"输出严格JSON。"},{"role":"user","content":prompt}],
            max_tokens=200, temperature=0.0, timeout=10
        )
        import re
        content = resp.choices[0].message.content
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            state = data.get("state", "震荡偏弱")
            factor = float(data.get("factor", 1.0))
            factor = max(0.6, min(1.4, factor))
            if use_cache:
                cache = _load_cache()
                cache[cache_key] = {"market_state": state, "market_factor": factor}
                _save_cache(cache)
            return state, factor
    except Exception as e:
        logger.error(f"市场状态AI分析失败: {e}")
    # 降级
    if above_ma20 and above_ma60:
        return "正常牛市", 1.2
    elif not above_ma20 and not above_ma60:
        return "熊市下跌", 0.8
    else:
        return "震荡偏弱", 1.0

# ---------------------------- 信号优先级排序 ----------------------------
def rank_signals(signals, etf_hist_cache, api_key):
    if not signals:
        return signals
    signal_details = []
    for sig in signals:
        code = sig['code']
        hist = etf_hist_cache.get(code)
        if hist is not None and len(hist) >= 5:
            last = hist.iloc[-1]
            vol_ratio = last['volume'] / last['vol_ma'] if last['vol_ma']>0 else 1
            rsi = last['rsi']
            williams = last['williams_r']
            ret_5d = (last['close'] / hist.iloc[-5]['close'] - 1) if len(hist)>=5 else 0
        else:
            vol_ratio, rsi, williams, ret_5d = 1, 50, -50, 0
        signal_details.append({
            "code": code, "name": sig['name'], "action": sig['action'], "score": sig['score'],
            "vol_ratio": vol_ratio, "rsi": rsi, "williams": williams, "ret_5d": ret_5d
        })
    prompt = f"""
你是一个交易策略顾问。以下是一些ETF的交易信号，请根据信号强度、技术指标和近期表现，分别对BUY信号和SELL信号进行优先级排序（从高到低）。
每个信号包含以下信息：
- 代码/名称
- 操作(BUY/SELL)
- 综合评分(score)
- 成交量比率(vol_ratio，>1放量)
- RSI值
- 威廉指标(williams)
- 近5日涨幅(ret_5d)

数据：
{json.dumps(signal_details, indent=2, ensure_ascii=False)}

请输出JSON数组，每个元素为{{"code": "代码", "priority": 排序序号(1最高)}}，先列出BUY信号，再列出SELL信号。如果某类信号为空，则输出空数组。
不要包含其他解释。
"""
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role":"system","content":"输出严格JSON数组。"},{"role":"user","content":prompt}],
            max_tokens=500, temperature=0.0, timeout=10
        )
        content = resp.choices[0].message.content
        import re
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            ranked = json.loads(json_match.group())
            order_map = {item['code']: item['priority'] for item in ranked}
            signals.sort(key=lambda x: order_map.get(x['code'], 999))
    except Exception as e:
        logger.error(f"信号排序失败: {e}")
    return signals

# ---------------------------- 评分与信号（已修复威廉超卖） ----------------------------
def strength(price, ma20, volume, vol_ma, macd_golden, kdj_golden, rsi, boll_up, boll_low, williams_r,
            ret_etf_5d, ret_market_5d, weekly_above, weekly_below, recent_high, recent_low, atr_pct,
            market_above_ma20, market_above_ma60, market_amount_above_ma20, is_buy, buy_weights, sell_weights):
    def cap(x): return max(0.0, min(1.0, x))
    if is_buy:
        # 修复：威廉超卖条件 williams_r < -80
        williams_oversold = cap((-80 - williams_r) / 20) if williams_r < -80 else 0
        factors = {
            "price_above_ma20": cap((price-ma20)/(ma20*0.1)) if price>ma20 else 0,
            "volume_above_ma5": cap(volume/vol_ma) if volume>vol_ma else 0,
            "macd_golden_cross": macd_golden,
            "kdj_golden_cross": kdj_golden,
            "bollinger_break_up": cap((price-boll_up)/boll_up) if price>boll_up else 0,
            "williams_oversold": williams_oversold,
            "market_above_ma20": 1 if market_above_ma20 else 0,
            "market_above_ma60": 1 if market_above_ma60 else 0,
            "market_amount_above_ma20": 1 if market_amount_above_ma20 else 0,
            "outperform_market": cap((ret_etf_5d-ret_market_5d)/0.05) if ret_etf_5d>ret_market_5d else 0,
            "weekly_above_ma20": 1 if weekly_above else 0,
        }
        weights = buy_weights
    else:
        factors = {
            "price_below_ma20": cap((ma20-price)/(ma20*0.1)) if price<ma20 else 0,
            "bollinger_break_down": cap((boll_low-price)/boll_low) if price<boll_low else 0,
            "williams_overbought": cap((20-williams_r)/20) if williams_r<20 else 0,
            "rsi_overbought": cap((rsi-70)/30) if rsi>70 else 0,
            "underperform_market": cap((ret_market_5d-ret_etf_5d)/0.05) if ret_etf_5d<ret_market_5d else 0,
            "stop_loss_ma_break": cap((ma20-price)/(ma20*0.05)) if price<ma20 else 0,
            "trailing_stop_clear": cap((recent_high-price)/recent_high/(ATR_STOP_MULT*atr_pct)) if recent_high>0 and atr_pct>0 and (recent_high-price)/recent_high >= ATR_STOP_MULT*atr_pct else 0,
            "trailing_stop_half": cap((recent_high-price)/recent_high/(ATR_TRAILING_MULT*atr_pct)) if recent_high>0 and atr_pct>0 and (recent_high-price)/recent_high >= ATR_TRAILING_MULT*atr_pct else 0,
            "profit_target_hit": cap((price-recent_low)/recent_low/PROFIT_TARGET) if recent_low>0 and (price-recent_low)/recent_low >= PROFIT_TARGET else 0,
            "weekly_below_ma20": 1 if weekly_below else 0,
        }
        weights = sell_weights
    return sum(weights.get(k,0)*factors.get(k,0) for k in weights)


def get_action(
    score,
    score_history,
    confirm_days,
    buy_threshold,
    sell_threshold,
    quick_buy_threshold,
):
    if len(score_history) >= confirm_days:
        recent = [s["score"] for s in score_history[-confirm_days:]]
        if all(s > buy_threshold for s in recent):
            return "BUY"
        if all(s < sell_threshold for s in recent):
            return "SELL"
    if len(score_history) >= 2:
        if (
            score_history[-1]["score"] > quick_buy_threshold
            and score_history[-1]["score"] > score_history[-2]["score"]
        ):
            return "BUY"
    if score > buy_threshold:
        return "BUY"
    elif score < sell_threshold:
        return "SELL"
    else:
        return "HOLD"


def analyze_etf(
    code,
    name,
    real_price,
    hist_df,
    weekly_df,
    market,
    today,
    state,
    buy_w,
    sell_w,
    params,
):
    if hist_df is None or len(hist_df) < 20:
        return f"{name}({code}) 数据不足", None, state
    d = hist_df.iloc[-1]
    ma20, vol_ma, volume = d["ma_short"], d["vol_ma"], d["volume"]
    rsi, boll_up, boll_low, williams_r = d["rsi"], d["boll_up"], d["boll_low"], d["williams_r"]
    atr_pct = d["atr"]/real_price if real_price>0 else 0
    recent_high_window = params["RECENT_HIGH_WINDOW"]
    recent_low_window = params["RECENT_LOW_WINDOW"]
    recent_high = d.get(
        f"recent_high_{recent_high_window}",
        hist_df["high"].rolling(recent_high_window).max().iloc[-1],
    )
    recent_low = d.get(
        f"recent_low_{recent_low_window}",
        hist_df["low"].rolling(recent_low_window).min().iloc[-1],
    )
    if len(hist_df) >= 2:
        prev = hist_df.iloc[-2]
        macd_golden = 1 if (d["macd_dif"] > d["macd_dea"] and prev["macd_dif"] <= prev["macd_dea"]) else 0
        kdj_golden = 1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0
    else:
        macd_golden = kdj_golden = 0
    ret_etf_5d = (real_price / hist_df.iloc[-5]["close"]) - 1 if len(hist_df)>=5 else 0
    weekly_above = weekly_below = False
    if weekly_df is not None and not weekly_df.empty:
        w = weekly_df.iloc[-1]
        if not np.isnan(w["ma_short"]):
            weekly_above = w["close"] > w["ma_short"]
            weekly_below = w["close"] < w["ma_short"]
    buy_score = strength(real_price, ma20, volume, vol_ma, macd_golden, kdj_golden, rsi, boll_up, boll_low, williams_r,
                        ret_etf_5d, market["ret_market_5d"], weekly_above, weekly_below, recent_high, recent_low, atr_pct,
                        market["market_above_ma20"], market["market_above_ma60"], market["market_amount_above_ma20"],
                        True, buy_w, sell_w)
    sell_score = strength(real_price, ma20, volume, vol_ma, macd_golden, kdj_golden, rsi, boll_up, boll_low, williams_r,
                         ret_etf_5d, market["ret_market_5d"], weekly_above, weekly_below, recent_high, recent_low, atr_pct,
                         market["market_above_ma20"], market["market_above_ma60"], market["market_amount_above_ma20"],
                         False, buy_w, sell_w)
    raw = buy_score - sell_score
    final = raw * market["market_factor"] * market["sentiment_factor"]
    today_str = today.strftime("%Y-%m-%d")
    if "score_history" not in state:
        state["score_history"] = []
    found = False
    for item in state["score_history"]:
        if item["date"] == today_str:
            item["score"] = final
            found = True
            break
    if not found:
        state["score_history"].append({"date": today_str, "score": final})
    state["score_history"] = sorted(state["score_history"], key=lambda x: x["date"])[
        -params["CONFIRM_DAYS"] :
    ]
    action = get_action(
        final,
        state["score_history"],
        params["CONFIRM_DAYS"],
        params["BUY_THRESHOLD"],
        params["SELL_THRESHOLD"],
        params["QUICK_BUY_THRESHOLD"],
    )
    risk_warning = False
    if len(state["score_history"]) >= RISK_WARNING_DAYS:
        recent_scores = [s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]]
        if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
            risk_warning = True
    output = f"【{name}({code})】 {real_price:.3f} 评分:{final:.2f} 操作:{action}"
    if risk_warning:
        output += f" 风险提示:连续{RISK_WARNING_DAYS}天评分低于{RISK_WARNING_THRESHOLD}"
    signal = {"action": action, "name": name, "code": code, "score": final} if action in ("BUY","SELL") else None
    return output, signal, state


# ---------------------------- 主程序 ----------------------------
def main():
    if not silent_login(): return
    try:
        etf_list = load_positions()
    except Exception as e:
        print(f"请准备 positions.csv (代码,名称)，错误: {e}")
        return
    today = datetime.date.today()
    start = (today - datetime.timedelta(days=300)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")
    # 宏观数据
    market_df = get_daily_data(MARKET_INDEX, start, today_str)
    macro_df = get_daily_data(MACRO_INDEX, start, today_str)
    if market_df is None or macro_df is None:
        print("获取宏观数据失败")
        return
    macro_df = calculate_indicators(
        macro_df, need_amount_ma=False, recent_high_window=10, recent_low_window=20
    )
    macro_df["ma_long"] = macro_df["close"].rolling(MACRO_MA_LONG).mean()
    market_df = calculate_indicators(
        market_df, need_amount_ma=True, recent_high_window=10, recent_low_window=20
    )
    market_df["atr"] = calculate_atr(market_df, ATR_PERIOD)
    volatility = (market_df["atr"] / market_df["close"]).iloc[-20:].mean()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        market_state, market_factor = refine_market_state(market_df, api_key)
        logger.info(f"AI市场状态: {market_state}, 因子: {market_factor}")
    else:
        if market_df.iloc[-1]["close"] > market_df.iloc[-1]["ma_short"] and market_df.iloc[-1]["close"] > market_df.iloc[-1].get("ma_long", market_df.iloc[-1]["ma_short"]):
            market_state, market_factor = "正常牛市", 1.2
        elif market_df.iloc[-1]["close"] < market_df.iloc[-1]["ma_short"] and market_df.iloc[-1]["close"] < market_df.iloc[-1].get("ma_long", market_df.iloc[-1]["ma_short"]):
            market_state, market_factor = "熊市下跌", 0.8
        else:
            market_state, market_factor = "震荡偏弱", 1.0
        logger.info(f"简单规则市场状态: {market_state}, 因子: {market_factor}")

    sentiment = get_sentiment_factor(macro_df)
    mkt = market_df.iloc[-1]
    market_info = {
        "macro_status": market_state,
        "market_factor": market_factor,
        "sentiment_factor": sentiment,
        "market_above_ma20": mkt["close"] > mkt["ma_short"],
        "market_above_ma60": mkt["close"] > mkt.get("ma_long", mkt["ma_short"]),
        "market_amount_above_ma20": mkt["amount"] > mkt["amount_ma"],
        "ret_market_5d": (mkt["close"]/market_df.iloc[-5]["close"]-1) if len(market_df)>=5 else 0,
    }
    state = load_state()
    buy_w, sell_w = DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()
    params = DEFAULT_PARAMS.copy()
    if api_key:
        bw, sw = deepseek_generate_weights(
            market_state,
            sentiment,
            market_info["market_above_ma20"],
            market_info["market_above_ma60"],
            market_info["market_amount_above_ma20"],
            volatility,
            api_key,
            use_cache=False,
        )
        if bw and sw:
            buy_w, sell_w = bw, sw
            logger.info("使用AI动态权重")
        else:
            logger.warning("AI权重生成失败，使用默认权重")
        p = deepseek_generate_params(
            market_state,
            sentiment,
            market_info["market_above_ma20"],
            market_info["market_above_ma60"],
            market_info["market_amount_above_ma20"],
            volatility,
            api_key,
            use_cache=False,
        )
        if p:
            params = p
            logger.info(f"使用AI动态参数: {params}")
        else:
            logger.warning("AI参数生成失败，使用默认参数")
    else:
        logger.info("未设置DEEPSEEK_API_KEY，使用默认权重和参数")

    signals = []
    etf_hist_cache = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = []
        for _, row in etf_list.iterrows():
            code, name = row["代码"], row["名称"]
            hist = get_daily_data(code, start, today_str)
            hist = (
                calculate_indicators(
                    hist,
                    need_amount_ma=False,
                    recent_high_window=params["RECENT_HIGH_WINDOW"],
                    recent_low_window=params["RECENT_LOW_WINDOW"],
                )
                if hist is not None
                else None
            )
            etf_hist_cache[code] = hist
            weekly = get_weekly_data(code, start, today_str)
            s = state.get(code, {})
            futures.append(
                ex.submit(
                    analyze_etf,
                    code,
                    name,
                    get_realtime_price_sina(code),
                    hist,
                    weekly,
                    market_info,
                    today,
                    s,
                    buy_w,
                    sell_w,
                    params,
                )
            )
        for f in futures:
            out, sig, new_state = f.result()
            print(out)
            import re
            m = re.search(r'【.*?\((.*?)\)】', out)
            if m:
                code = m.group(1)
                state[code] = new_state
            if sig:
                signals.append(sig)
    save_state(state)

    if signals and api_key:
        signals = rank_signals(signals, etf_hist_cache, api_key)
        print("\n【信号优先级排序】")
        for sig in signals:
            print(f"{sig['action']}: {sig['name']}({sig['code']}) 评分:{sig['score']:.2f}")
    elif signals:
        print("\n【信号汇总】")
        for sig in signals:
            print(f"{sig['action']}: {sig['name']}({sig['code']}) 评分:{sig['score']:.2f}")

    silent_logout()

if __name__ == "__main__":
    main()
