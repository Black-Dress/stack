#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统（最终版）
功能：实时分析 ETF，输出评分和操作等级（强烈买入/买入/偏多持有/中性观望/卖出/强烈卖出）。
TMSV 复合指标已融入评分体系，仅作为内部因子。
支持 AI 动态权重、动态参数、市场状态精细化。
输出为对齐表格，无额外信号汇总。
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
import unicodedata
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor

# ---------------------------- 日志配置（屏蔽 HTTP 请求日志） ----------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
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

DEFAULT_BUY_WEIGHTS = {
    "price_above_ma20": 0.22,
    "volume_above_ma5": 0.14,
    "macd_golden_cross": 0.08,
    "kdj_golden_cross": 0.08,
    "bollinger_break_up": 0.08,
    "williams_oversold": 0.08,
    "market_above_ma20": 0.05,
    "market_above_ma60": 0.08,
    "market_amount_above_ma20": 0.05,
    "outperform_market": 0.10,
    "weekly_above_ma20": 0.10,
    "tmsv_score": 0.18,
}
DEFAULT_SELL_WEIGHTS = {
    "price_below_ma20": 0.40,
    "bollinger_break_down": 0.20,
    "williams_overbought": 0.10,
    "rsi_overbought": 0.15,
    "underperform_market": 0.20,
    "stop_loss_ma_break": 1.00,
    "trailing_stop_clear": 1.00,
    "trailing_stop_half": 0.50,
    "profit_target_hit": 0.30,
    "weekly_below_ma20": 0.20,
}
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
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,volume,amount",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
    )
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
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_weekly_data(code, start_date, end_date):
    df = get_daily_data(code, start_date, end_date)
    if df is None:
        return None
    weekly = df.resample("W-FRI").last()
    weekly["ma_short"] = weekly["close"].rolling(window=WEEKLY_MA).mean()
    return weekly


def get_realtime_price_sina(code):
    try:
        for domain in ["hq.sinajs.cn", "hq2.sinajs.cn", "hq3.sinajs.cn"]:
            url = f"http://{domain}/list={code.replace('.','')}"
            r = requests.get(
                url, headers={"Referer": "http://finance.sina.com.cn"}, timeout=3
            )
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
    tr = pd.concat(
        [
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift()),
        ],
        axis=1,
    ).max(1)
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
    df["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
    # 布林
    df["boll_mid"] = df["close"].rolling(20).mean()
    df["boll_std"] = df["close"].rolling(20).std()
    df["boll_up"] = df["boll_mid"] + 2 * df["boll_std"]
    df["boll_low"] = df["boll_mid"] - 2 * df["boll_std"]
    # 威廉
    high_14 = df["high"].rolling(14).max()
    low_14 = df["low"].rolling(14).min()
    df["williams_r"] = (high_14 - df["close"]) / (high_14 - low_14) * -100
    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi"] = 100 - 100 / (1 + gain / loss)
    # ATR
    df["atr"] = calculate_atr(df, ATR_PERIOD)
    df[f"recent_high_{recent_high_window}"] = (
        df["high"].rolling(recent_high_window).max()
    )
    df[f"recent_low_{recent_low_window}"] = df["low"].rolling(recent_low_window).min()
    return df


def get_sentiment_factor(macro_df):
    if len(macro_df) < RSI_PERIOD + 1:
        return 1.0
    delta = macro_df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(RSI_PERIOD).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(RSI_PERIOD).mean()
    rsi = 100 - 100 / (1 + gain / loss)
    latest_rsi = rsi.iloc[-1]
    if latest_rsi < 30:
        return 0.6
    if latest_rsi < 50:
        return 0.8
    if latest_rsi < 70:
        return 1.0
    return 0.9


# ---------------------------- TMSV 复合指标（仅内部使用） ----------------------------
def compute_tmsv(df):
    if df is None or len(df) < 20:
        return pd.Series([50.0] * max(1, len(df))) if len(df) > 0 else pd.Series([50.0])
    df = df.copy()
    try:
        if "ma20" not in df.columns:
            df["ma20"] = df["close"].rolling(20).mean()
        if "ma60" not in df.columns:
            df["ma60"] = df["close"].rolling(60).mean()
        if "rsi" not in df.columns:
            delta = df["close"].diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            df["rsi"] = 100 - 100 / (1 + gain / loss)
        if "macd_hist" not in df.columns:
            exp12 = df["close"].ewm(span=12, adjust=False).mean()
            exp26 = df["close"].ewm(span=26, adjust=False).mean()
            df["macd"] = exp12 - exp26
            df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
            df["macd_hist"] = df["macd"] - df["macd_signal"]
        if "atr" not in df.columns:
            df["atr"] = calculate_atr(df, 14)
        if "vol_ma" not in df.columns:
            df["vol_ma"] = df["volume"].rolling(20).mean()
    except Exception as e:
        logger.warning(f"TMSV 列计算失败: {e}")
        return pd.Series([50.0] * len(df))

    # 趋势得分
    price_above_ma20 = (
        ((df["close"] - df["ma20"]) / (df["ma20"].replace(0, np.nan) * 0.1))
        .clip(0, 1)
        .fillna(0)
    )
    price_above_ma60 = (
        ((df["close"] - df["ma60"]) / (df["ma60"].replace(0, np.nan) * 0.1))
        .clip(0, 1)
        .fillna(0)
    )
    ma20_slope = df["ma20"].diff(5) / df["ma20"].shift(5).replace(0, np.nan)
    slope_score = (ma20_slope * 10).clip(0, 1).fillna(0)
    trend_score = (
        price_above_ma20 * 0.5 + price_above_ma60 * 0.3 + slope_score * 0.2
    ) * 100

    # 动量得分
    rsi_score = ((df["rsi"] - 50) * 3.33).clip(0, 100).fillna(50)
    macd_change = df["macd_hist"].diff() / (df["macd_hist"].shift(1).abs() + 0.001)
    macd_score = (macd_change * 100).clip(0, 100).fillna(50)
    mom_score = rsi_score * 0.6 + macd_score * 0.4

    # 量价得分
    vol_ratio = df["volume"] / df["vol_ma"].replace(0, np.nan)
    vol_ratio_score = ((vol_ratio - 0.8) / 1.2 * 100).clip(0, 100).fillna(50)
    price_up = df["close"] > df["close"].shift(1)
    vol_up = df["volume"] > df["vol_ma"]
    consistency = np.where(price_up == vol_up, 100, 0)
    vol_score = vol_ratio_score * 0.7 + consistency * 0.3

    # 波动率因子
    atr_pct = df["atr"] / df["close"].replace(0, np.nan)
    vol_factor = np.select(
        [atr_pct < 0.01, atr_pct > 0.03],
        [1.5, 0.6],
        default=1.2 - (atr_pct - 0.01) / 0.02 * 0.6,
    )
    vol_factor = np.nan_to_num(vol_factor, nan=1.0)

    tmsv = (trend_score * 0.3 + mom_score * 0.3 + vol_score * 0.2) * vol_factor
    tmsv = tmsv.clip(0, 100).fillna(50)
    return tmsv


# ---------------------------- AI 权重生成（优化版） ----------------------------
def _get_cache_key(
    macro_status,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    volatility,
    cache_type="weights",
):
    """生成更精细的缓存键，包含日期和类型"""
    today = datetime.date.today().strftime("%Y-%m-%d")
    return hashlib.md5(
        f"{cache_type}_{today}_{macro_status}_{sentiment_factor:.2f}_{market_above_ma20}_{market_above_ma60}_{market_amount_above_ma20}_{volatility:.3f}".encode()
    ).hexdigest()


def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                cache = json.load(f)
            # 清理过期缓存（超过7天）
            current_time = datetime.datetime.now().timestamp()
            expired_keys = []
            for key, value in cache.items():
                if isinstance(value, dict) and "timestamp" in value:
                    if current_time - value["timestamp"] > 7 * 24 * 3600:  # 7天
                        expired_keys.append(key)
            for key in expired_keys:
                del cache[key]
                logger.info(f"清理过期缓存: {key}")
            if expired_keys:
                _save_cache(cache)
            return cache
        except Exception as e:
            logger.warning(f"缓存文件损坏: {e}，使用空缓存")
            return {}
    return {}


def _save_cache(cache):
    try:
        # 添加时间戳
        for key, value in cache.items():
            if isinstance(value, dict) and "timestamp" not in value:
                value["timestamp"] = datetime.datetime.now().timestamp()
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存缓存失败: {e}")


def _validate_and_filter_weights(weights, expected_keys, name):
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


def build_optimized_prompt(
    macro_status,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    volatility,
):
    buy_keys = [
        "price_above_ma20",
        "volume_above_ma5",
        "macd_golden_cross",
        "kdj_golden_cross",
        "bollinger_break_up",
        "williams_oversold",
        "market_above_ma20",
        "market_above_ma60",
        "market_amount_above_ma20",
        "outperform_market",
        "weekly_above_ma20",
        "tmsv_score",
    ]
    sell_keys = [
        "price_below_ma20",
        "bollinger_break_down",
        "williams_overbought",
        "rsi_overbought",
        "underperform_market",
        "stop_loss_ma_break",
        "trailing_stop_clear",
        "trailing_stop_half",
        "profit_target_hit",
        "weekly_below_ma20",
    ]
    prompt = f"""
你是一个量化交易策略专家。当前市场环境如下：
- 宏观状态：{macro_status}（bull牛市、oscillate震荡市、bear熊市）
- 情绪系数：{sentiment_factor}（0.6=恐慌，0.8=偏弱，1.0=中性，0.9=偏热）
- 大盘站上20日均线：{"是" if market_above_ma20 else "否"}
- 大盘站上60日均线：{"是" if market_above_ma60 else "否"}
- 市场成交额高于20日均额：{"是" if market_amount_above_ma20 else "否"}
- 市场波动率(ATR/收盘价)：{volatility:.3f}（<0.01低波动，>0.02高波动）

买入因子说明：
- tmsv_score 是一个复合指标（0-100，越高越看涨），其强度为 tmsv/100。
- 在牛市中应给予 tmsv_score 较高权重（0.15-0.25），震荡市中等（0.10-0.15），熊市较低（0.05-0.10）。

请根据以下规则分配买入权重和卖出权重，所有权重为0-1之间的浮点数，每个部分总和为1。

【买入权重规则】
1. 牛市且大盘站上60日线：趋势类因子(price_above_ma20, market_above_ma60, weekly_above_ma20)总权重应≥0.5，tmsv_score 0.15-0.25。
2. 震荡市：反转类因子(williams_oversold, bollinger_break_up)总权重应≥0.4，tmsv_score 0.10-0.15。
3. 熊市：买入总权重建议≤0.3，tmsv_score 0.05-0.10，其他趋势因子设为0。
4. 高波动时(>0.02)：降低price_above_ma20权重，提高williams_oversold、volume_above_ma5和tmsv_score。
5. 低波动时(<0.01)：可适当提高macd_golden_cross、kdj_golden_cross和tmsv_score。

【卖出权重规则】
1. 熊市且大盘跌破60日线：止损类因子(stop_loss_ma_break, trailing_stop_clear)总权重应≥0.6，单个因子可达0.4-0.8。
2. 牛市：止盈类因子(profit_target_hit)和超买因子(rsi_overbought)总权重应≥0.5。
3. 震荡市：平衡止损和止盈，各约0.5。
4. 高波动时：提高profit_target_hit权重，降低trailing_stop_clear阈值效应。
5. 任何市场下，若认为应完全空仓：买入所有权重设为0，卖出权重全部给stop_loss_ma_break（设为1.0）。

【约束条件】
- 禁止添加任何未列出的键。
- 单个因子权重不得超过0.6（除熊市卖出止损因子可到0.8外）。
- 每个部分总和为1（允许浮点误差）。

【输出格式示例】
牛市示例：
{{"buy": {{"price_above_ma20":0.20,"market_above_ma60":0.15,"weekly_above_ma20":0.10,"tmsv_score":0.20,"volume_above_ma5":0.10,"outperform_market":0.10,"macd_golden_cross":0.05,"kdj_golden_cross":0.05,"williams_oversold":0.05}}, "sell": {{"profit_target_hit":0.40,"rsi_overbought":0.30,"underperform_market":0.30}}}}
熊市示例：
{{"buy": {{"williams_oversold":0.20,"bollinger_break_up":0.10,"tmsv_score":0.05}}, "sell": {{"stop_loss_ma_break":0.70,"trailing_stop_clear":0.30}}}}

请严格按照JSON格式输出，不要包含任何解释文字。
"""
    return prompt


def deepseek_generate_weights(
    macro_status,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    volatility,
    api_key,
    use_cache=True,
):
    cache_key = _get_cache_key(
        macro_status,
        sentiment_factor,
        market_above_ma20,
        market_above_ma60,
        market_amount_above_ma20,
        volatility,
        "weights",
    )
    if use_cache:
        cache = _load_cache()
        if cache_key in cache:
            cached = cache[cache_key]
            if "buy" in cached and "sell" in cached:
                buy = _validate_and_filter_weights(
                    cached["buy"], DEFAULT_BUY_WEIGHTS.keys(), "缓存买入权重"
                )
                sell = _validate_and_filter_weights(
                    cached["sell"], DEFAULT_SELL_WEIGHTS.keys(), "缓存卖出权重"
                )
                if buy and sell:
                    return buy, sell
                else:
                    logger.warning("缓存权重无效，重新生成")

    prompt = build_optimized_prompt(
        macro_status,
        sentiment_factor,
        market_above_ma20,
        market_above_ma60,
        market_amount_above_ma20,
        volatility,
    )
    max_retries = 3
    for attempt in range(max_retries):
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
                max_tokens=800,
                temperature=0.0,
                timeout=15,  # 增加超时时间
            )
            content = resp.choices[0].message.content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                raise ValueError("未找到JSON")
            data = json.loads(json_match.group())
            if "buy" not in data or "sell" not in data:
                raise ValueError("JSON缺少buy或sell字段")
            buy_weights = _validate_and_filter_weights(
                data["buy"], DEFAULT_BUY_WEIGHTS.keys(), "AI买入权重"
            )
            sell_weights = _validate_and_filter_weights(
                data["sell"], DEFAULT_SELL_WEIGHTS.keys(), "AI卖出权重"
            )
            if buy_weights is None or sell_weights is None:
                raise ValueError("权重校验失败")
            if use_cache:
                cache = _load_cache()
                cache[cache_key] = {"buy": buy_weights, "sell": sell_weights}
                _save_cache(cache)
            return buy_weights, sell_weights
        except Exception as e:
            logger.warning(f"AI权重生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time

                time.sleep(2**attempt)  # 指数退避
            else:
                logger.error(f"AI权重生成最终失败，使用默认权重")
                return DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()


# ---------------------------- AI 参数生成 ----------------------------
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
1. CONFIRM_DAYS（确认天数，1-5）：
   - 高波动时(>0.02)增加到4-5以避免假信号
   - 低波动时(<0.01)减少到2-3以快速响应
   - 牛市适当减少到2-3，熊市增加到4-5

2. BUY_THRESHOLD（买入阈值，0.3-0.7）：
   - 牛市降低到0.4-0.5以提高敏感度
   - 熊市提高到0.6-0.7以提高谨慎度
   - 震荡市保持0.5左右

3. SELL_THRESHOLD（卖出阈值，-0.5到-0.1）：
   - 熊市降低到-0.3到-0.2以快速止损
   - 牛市提高到-0.1左右以减少噪音
   - 高波动时适当放宽阈值

4. QUICK_BUY_THRESHOLD（快速买入阈值，0.5-0.8）：
   - 高波动时提高到0.7-0.8以过滤噪音
   - 低波动时降低到0.5-0.6以捕捉机会
   - 牛市可适当降低

5. RECENT_HIGH_WINDOW（近期高点窗口，5-20）：
   - 高波动时缩短到5-10以适应快速变化
   - 低波动时延长到15-20以获得更稳定信号

6. RECENT_LOW_WINDOW（近期低点窗口，10-30）：
   - 高波动时缩短到10-15以快速响应
   - 低波动时延长到20-30以获得更可靠支撑

【历史表现反馈】
- 在高波动熊市环境中，增加确认天数和提高买入阈值可有效减少假信号
- 在低波动牛市环境中，减少确认天数和降低买入阈值可提高胜率
- 震荡市应保持中等参数设置，避免过度交易

【输出格式示例】
{{"CONFIRM_DAYS":3,"BUY_THRESHOLD":0.5,"SELL_THRESHOLD":-0.2,"QUICK_BUY_THRESHOLD":0.6,"RECENT_HIGH_WINDOW":10,"RECENT_LOW_WINDOW":20}}

请严格按照JSON格式输出，不要包含任何解释文字。
"""
    return prompt


def _validate_and_filter_params(params, expected_keys, name):
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
    cache_key = _get_cache_key(
        macro_status,
        sentiment_factor,
        market_above_ma20,
        market_above_ma60,
        market_amount_above_ma20,
        volatility,
        "params",
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
    max_retries = 3
    for attempt in range(max_retries):
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
                timeout=15,  # 增加超时时间
            )
            content = resp.choices[0].message.content
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
            logger.warning(f"AI参数生成失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                import time

                time.sleep(2**attempt)  # 指数退避
            else:
                logger.error(f"AI参数生成最终失败，使用默认参数")
                return DEFAULT_PARAMS.copy()


# ---------------------------- 市场状态精细化 ----------------------------
def refine_market_state(market_df, api_key, use_cache=True):
    cache_key = (
        f"market_state_{hashlib.md5(market_df.tail(20).to_json().encode()).hexdigest()}"
    )
    if use_cache:
        cache = _load_cache()
        if cache_key in cache and "market_state" in cache[cache_key]:
            state = cache[cache_key]["market_state"]
            factor = cache[cache_key]["market_factor"]
            return state, factor

    recent = market_df.tail(20)
    close_pct = recent["close"].pct_change().mean()
    vol_pct = recent["volume"].pct_change().mean()
    volatility = (recent["close"].pct_change().std()) * 100
    above_ma20 = recent["close"].iloc[-1] > recent["ma_short"].iloc[-1]
    above_ma60 = (
        recent["close"].iloc[-1] > recent["ma_long"].iloc[-1]
        if "ma_long" in recent
        else above_ma20
    )

    prompt = f"""
你是一个市场分析专家。请根据以下大盘指数(上证)最近20日的数据，判断当前市场状态，并给出一个推荐的市场因子系数（0.6-1.4，1.0为中性，>1.0为积极，<1.0为防御）。
数据摘要：
- 最近20日平均日涨跌幅：{close_pct:.4f}
- 最近20日平均成交量变化率：{vol_pct:.4f}
- 日波动率（标准差%）：{volatility:.2f}%
- 当前价格是否站上20日均线：{"是" if above_ma20 else "否"}
- 当前价格是否站上60日均线：{"是" if above_ma60 else "否"}
- RSI指标：{recent['rsi'].iloc[-1]:.1f}（<30超卖，>70超买）
- MACD指标：{recent['macd_dif'].iloc[-1]:.3f} vs {recent['macd_dea'].iloc[-1]:.3f}
- 布林带位置：收盘价相对中轨 {(recent['close'].iloc[-1] - recent['boll_mid'].iloc[-1]) / recent['boll_std'].iloc[-1]:.2f} 标准差
- 威廉指标：{recent['williams_r'].iloc[-1]:.1f}（<-80超卖，>-20超买）

请综合以上技术指标判断市场状态：
- 强势牛市：多头排列，RSI>50，MACD金叉，布林上轨
- 正常牛市：站上中长期均线，技术指标向好
- 震荡偏强：站上短期均线，但技术指标中性
- 震荡偏弱：跌破短期均线，但未破长期均线
- 弱势反弹：技术指标转好，但整体偏空
- 熊市下跌中继：空头排列，技术指标恶化
- 熊市加速下跌：连续跌破重要支撑，恐慌性抛售

请输出JSON格式：{{"state": "市场状态标签", "factor": 市场因子系数}}
"""
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "输出严格JSON。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
            temperature=0.0,
            timeout=10,
        )
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
    if above_ma20 and above_ma60:
        return "正常牛市", 1.2
    elif not above_ma20 and not above_ma60:
        return "熊市下跌", 0.8
    else:
        return "震荡偏弱", 1.0


# ---------------------------- 评分与信号（融入 TMSV，输出分级） ----------------------------
def strength(
    price,
    ma20,
    volume,
    vol_ma,
    macd_golden,
    kdj_golden,
    rsi,
    boll_up,
    boll_low,
    williams_r,
    ret_etf_5d,
    ret_market_5d,
    weekly_above,
    weekly_below,
    recent_high,
    recent_low,
    atr_pct,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    is_buy,
    buy_weights,
    sell_weights,
    tmsv_strength=0.0,
):

    def cap(x):
        return max(0.0, min(1.0, x))

    if is_buy:
        williams_oversold = cap((-80 - williams_r) / 20) if williams_r < -80 else 0
        factors = {
            "price_above_ma20": (
                cap((price - ma20) / (ma20 * 0.1)) if price > ma20 else 0
            ),
            "volume_above_ma5": cap(volume / vol_ma) if volume > vol_ma else 0,
            "macd_golden_cross": macd_golden,
            "kdj_golden_cross": kdj_golden,
            "bollinger_break_up": (
                cap((price - boll_up) / boll_up) if price > boll_up else 0
            ),
            "williams_oversold": williams_oversold,
            "market_above_ma20": 1 if market_above_ma20 else 0,
            "market_above_ma60": 1 if market_above_ma60 else 0,
            "market_amount_above_ma20": 1 if market_amount_above_ma20 else 0,
            "outperform_market": (
                cap((ret_etf_5d - ret_market_5d) / 0.05)
                if ret_etf_5d > ret_market_5d
                else 0
            ),
            "weekly_above_ma20": 1 if weekly_above else 0,
            "tmsv_score": tmsv_strength,
        }
        weights = buy_weights
    else:
        factors = {
            "price_below_ma20": (
                cap((ma20 - price) / (ma20 * 0.1)) if price < ma20 else 0
            ),
            "bollinger_break_down": (
                cap((boll_low - price) / boll_low) if price < boll_low else 0
            ),
            "williams_overbought": (
                cap((20 - williams_r) / 20) if williams_r < 20 else 0
            ),
            "rsi_overbought": cap((rsi - 70) / 30) if rsi > 70 else 0,
            "underperform_market": (
                cap((ret_market_5d - ret_etf_5d) / 0.05)
                if ret_etf_5d < ret_market_5d
                else 0
            ),
            "stop_loss_ma_break": (
                cap((ma20 - price) / (ma20 * 0.05)) if price < ma20 else 0
            ),
            "trailing_stop_clear": (
                cap((recent_high - price) / recent_high / (ATR_STOP_MULT * atr_pct))
                if recent_high > 0
                and atr_pct > 0
                and (recent_high - price) / recent_high >= ATR_STOP_MULT * atr_pct
                else 0
            ),
            "trailing_stop_half": (
                cap((recent_high - price) / recent_high / (ATR_TRAILING_MULT * atr_pct))
                if recent_high > 0
                and atr_pct > 0
                and (recent_high - price) / recent_high >= ATR_TRAILING_MULT * atr_pct
                else 0
            ),
            "profit_target_hit": (
                cap((price - recent_low) / recent_low / PROFIT_TARGET)
                if recent_low > 0 and (price - recent_low) / recent_low >= PROFIT_TARGET
                else 0
            ),
            "weekly_below_ma20": 1 if weekly_below else 0,
        }
        weights = sell_weights
    return sum(weights.get(k, 0) * factors.get(k, 0) for k in factors)


def get_action(
    score,
    score_history,
    confirm_days,
    buy_threshold,
    sell_threshold,
    quick_buy_threshold,
    atr_pct=None,
):
    """返回 BUY/SELL/HOLD 用于内部信号判断，包含智能确认机制"""
    # 智能信号确认：结合趋势强度和连续性
    if len(score_history) >= confirm_days:
        recent_scores = [s["score"] for s in score_history[-confirm_days:]]
        avg_score = sum(recent_scores) / len(recent_scores)

        # 买入信号：连续确认且平均分足够高
        if (
            all(s > buy_threshold for s in recent_scores)
            and avg_score > buy_threshold + 0.1
        ):
            return "BUY"
        # 卖出信号：连续确认且平均分足够低
        if (
            all(s < sell_threshold for s in recent_scores)
            and avg_score < sell_threshold - 0.1
        ):
            return "SELL"

    # 快速买入：当前分数显著高于阈值且有改善趋势
    if len(score_history) >= 2:
        last = score_history[-1]["score"]
        prev = score_history[-2]["score"]
        if last > quick_buy_threshold and last > prev and last > buy_threshold + 0.2:
            return "BUY"

    # 单次信号：当前分数超过阈值
    if score > buy_threshold:
        return "BUY"
    if score < sell_threshold:
        return "SELL"

    return "HOLD"


def get_action_level(score):
    """根据评分返回操作等级（用于显示）"""
    if score >= 0.8:
        return "极度看好"
    if score >= 0.7:
        return "强烈买入"
    if score >= 0.6:
        return "买入"
    if score >= 0.4:
        return "谨慎买入"
    if score >= 0.2:
        return "偏多持有"
    if score >= 0.0:
        return "中性偏多"
    if score >= -0.2:
        return "中性偏空"
    if score >= -0.4:
        return "偏空持有"
    if score >= -0.6:
        return "谨慎卖出"
    if score >= -0.8:
        return "卖出"
    return "强烈卖出"


def get_display_width(text):
    text = "" if text is None else str(text)
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def pad_display(text, width, align="left"):
    text = "" if text is None else str(text)
    display_width = get_display_width(text)
    if display_width >= width:
        return text
    pad = width - display_width
    if align == "right":
        return " " * pad + text
    if align == "center":
        left = pad // 2
        return " " * left + text + " " * (pad - left)
    return text + " " * pad


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
    row_format = "{:<16} {:<12} {:>8} {:>6}  {:<8}"
    if hist_df is None or len(hist_df) < 20:
        output = (
            pad_display(name, 16)
            + " "
            + pad_display(code, 12)
            + " "
            + pad_display(f"{real_price:.3f}", 8, "right")
            + " "
            + pad_display("0.00", 6, "right")
            + "  "
            + pad_display("中性观望", 8)
        )
        return output, None, state
    d = hist_df.iloc[-1]
    ma20, vol_ma, volume = d["ma_short"], d["vol_ma"], d["volume"]
    rsi, boll_up, boll_low, williams_r = (
        d["rsi"],
        d["boll_up"],
        d["boll_low"],
        d["williams_r"],
    )
    atr_pct = d["atr"] / real_price if real_price > 0 else 0
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
        macd_golden = (
            1
            if (d["macd_dif"] > d["macd_dea"] and prev["macd_dif"] <= prev["macd_dea"])
            else 0
        )
        kdj_golden = (
            1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0
        )
    else:
        macd_golden = kdj_golden = 0
    ret_etf_5d = (
        (real_price / hist_df.iloc[-5]["close"]) - 1 if len(hist_df) >= 5 else 0
    )
    weekly_above = weekly_below = False
    if weekly_df is not None and not weekly_df.empty:
        w = weekly_df.iloc[-1]
        if not np.isnan(w["ma_short"]):
            weekly_above = w["close"] > w["ma_short"]
            weekly_below = w["close"] < w["ma_short"]

    # 计算 TMSV
    try:
        tmsv_series = compute_tmsv(hist_df)
        tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
        if np.isnan(tmsv):
            tmsv = 50.0
    except Exception as e:
        logger.warning(f"TMSV 计算异常: {e}")
        tmsv = 50.0
    tmsv_strength = tmsv / 100.0

    buy_score = strength(
        real_price,
        ma20,
        volume,
        vol_ma,
        macd_golden,
        kdj_golden,
        rsi,
        boll_up,
        boll_low,
        williams_r,
        ret_etf_5d,
        market["ret_market_5d"],
        weekly_above,
        weekly_below,
        recent_high,
        recent_low,
        atr_pct,
        market["market_above_ma20"],
        market["market_above_ma60"],
        market["market_amount_above_ma20"],
        True,
        buy_w,
        sell_w,
        tmsv_strength,
    )
    sell_score = strength(
        real_price,
        ma20,
        volume,
        vol_ma,
        macd_golden,
        kdj_golden,
        rsi,
        boll_up,
        boll_low,
        williams_r,
        ret_etf_5d,
        market["ret_market_5d"],
        weekly_above,
        weekly_below,
        recent_high,
        recent_low,
        atr_pct,
        market["market_above_ma20"],
        market["market_above_ma60"],
        market["market_amount_above_ma20"],
        False,
        buy_w,
        sell_w,
        tmsv_strength,
    )
    raw = buy_score - sell_score
    final = raw * market["market_factor"] * market["sentiment_factor"]

    # 更新状态
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
    action_level = get_action_level(final)

    risk_warning = False
    risk_level = "normal"
    if len(state["score_history"]) >= RISK_WARNING_DAYS:
        recent_scores = [
            s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]
        ]
        # 连续低分警告
        if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
            risk_warning = True
            risk_level = "连续低分"
        # 单日极端评分警告
        elif final < -0.5 or final > 0.8:
            risk_warning = True
            risk_level = "极端评分"
        # 高波动环境警告
        elif atr_pct and atr_pct > 0.03:
            risk_warning = True
            risk_level = "高波动"

    output = (
        pad_display(name, 16)
        + " "
        + pad_display(code, 12)
        + " "
        + pad_display(f"{real_price:.3f}", 8, "right")
        + " "
        + pad_display(f"{final:.2f}", 6, "right")
        + "  "
        + pad_display(action_level, 10)
    )
    if risk_warning:
        output += "  ⚠️  "
        if risk_level == "连续低分":
            output += f"风险提示({risk_level}): 连续{RISK_WARNING_DAYS}天评分低于{RISK_WARNING_THRESHOLD}"
        elif risk_level == "极端评分":
            output += f"风险提示({risk_level}): 当前评分{final:.2f}处于极端水平"
        elif risk_level == "高波动":
            output += f"风险提示({risk_level}): 市场波动率{atr_pct:.3f}过高"
    signal = (
        {"action": action, "name": name, "code": code, "score": final}
        if action in ("BUY", "SELL")
        else None
    )
    return output, signal, state, final


# ---------------------------- 主程序 ----------------------------
def main():
    if not silent_login():
        return
    try:
        etf_list = load_positions()
    except Exception as e:
        print(f"请准备 positions.csv (代码,名称)，错误: {e}")
        return

    today = datetime.date.today()
    # 优化：减少数据量到200天，提高效率
    start = (today - datetime.timedelta(days=200)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    # 获取宏观数据
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
        if market_df.iloc[-1]["close"] > market_df.iloc[-1][
            "ma_short"
        ] and market_df.iloc[-1]["close"] > market_df.iloc[-1].get(
            "ma_long", market_df.iloc[-1]["ma_short"]
        ):
            market_state, market_factor = "正常牛市", 1.2
        elif market_df.iloc[-1]["close"] < market_df.iloc[-1][
            "ma_short"
        ] and market_df.iloc[-1]["close"] < market_df.iloc[-1].get(
            "ma_long", market_df.iloc[-1]["ma_short"]
        ):
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
        "ret_market_5d": (
            (mkt["close"] / market_df.iloc[-5]["close"] - 1)
            if len(market_df) >= 5
            else 0
        ),
    }

    # 策略适应性优化：根据波动率和市场状态动态调整参数
    params = DEFAULT_PARAMS.copy()
    if volatility > 0.02:  # 高波动环境
        params["BUY_THRESHOLD"] = 0.6
        params["SELL_THRESHOLD"] = -0.3
        params["CONFIRM_DAYS"] = 4
        params["QUICK_BUY_THRESHOLD"] = 0.7
    elif volatility < 0.01:  # 低波动环境
        params["BUY_THRESHOLD"] = 0.4
        params["SELL_THRESHOLD"] = -0.1
        params["CONFIRM_DAYS"] = 2
        params["QUICK_BUY_THRESHOLD"] = 0.5
    else:  # 中等波动
        # 根据市场状态调整确认天数
        if "牛市" in market_state:
            params["CONFIRM_DAYS"] = 2  # 牛市快速响应
        elif "熊市" in market_state:
            params["CONFIRM_DAYS"] = 4  # 熊市谨慎确认
        else:
            params["CONFIRM_DAYS"] = 3  # 震荡市标准确认

    state = load_state()
    buy_w, sell_w = DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()

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
    else:
        logger.info("未设置DEEPSEEK_API_KEY，使用默认权重和参数")

    # 打印表头
    print(
        pad_display("名称", 16)
        + " "
        + pad_display("代码", 12)
        + " "
        + pad_display("价格", 8, "right")
        + " "
        + pad_display("评分", 6, "right")
        + "  "
        + pad_display("操作", 10)
    )
    print("-" * 68)

    # 并行分析并输出
    results = []
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
            out, _, new_state, score = f.result()
            results.append((out, score))
            # 更新状态（从输出中提取代码）
            m = re.search(r"【.*?\((.*?)\)】", out)
            if m:
                code = m.group(1)
                # new_state 已经是更新后的状态，但我们需要把它合并回 state
                # 由于 analyze_etf 返回的 new_state 包含整个 state 字典（包含所有代码的状态？实际上只是单个代码的状态）
                # 这里为了简化，我们直接更新 state[code] = new_state
                state[code] = new_state

    # 按照评分从高到低排序并输出
    results.sort(key=lambda x: x[1], reverse=True)
    for out, _ in results:
        print(out)

    save_state(state)
    silent_logout()


if __name__ == "__main__":
    main()
