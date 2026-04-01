#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统（单文件版）
功能：实时分析 ETF，输出买卖建议，支持 ATR 动态止损、周线过滤、AI 动态权重等。
依赖：baostock, pandas, numpy, requests, openai
运行前请创建 positions.csv（包含代码,名称两列），可选设置环境变量 DEEPSEEK_API_KEY。
"""

import os
import json
import datetime
import logging
import requests
import numpy as np
import pandas as pd
import baostock as bs
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
import openai

# ---------------------------- 配置参数 ----------------------------
# 基础参数
ETF_MA = 20
ETF_VOL_MA = 5
MACRO_INDEX = "sh.000300"
MARKET_INDEX = "sh.000001"
MACRO_MA_SHORT = 20
MACRO_MA_LONG = 60
RSI_PERIOD = 14

# 策略权重（正买入，负卖出）
STRATEGY_WEIGHTS = {
    "price_above_ma20": 0.30,
    "volume_above_ma5": 0.20,
    "macd_golden_cross": 0.15,
    "kdj_golden_cross": 0.15,
    "bollinger_break_up": 0.10,
    "williams_oversold": 0.10,
    "price_below_ma20": -0.40,
    "bollinger_break_down": -0.20,
    "williams_overbought": -0.10,
    "rsi_overbought": -0.15,
    "market_above_ma20": 0.10,
    "market_above_ma60": 0.10,
    "market_amount_above_ma20": 0.10,
    "outperform_market": 0.20,
    "underperform_market": -0.20,
    "stop_loss_ma_break": -1.00,
    "trailing_stop_clear": -1.00,
    "trailing_stop_half": -0.50,
    "profit_target_hit": -0.30,
    "weekly_above_ma20": 0.20,
    "weekly_below_ma20": -0.20,
}

# 止盈止损参数
STOP_LOSS_MA = 20
PROFIT_TARGETS = [(0.20, 0.3), (0.40, 0.3)]

# ATR 动态止损
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0  # 清仓倍数
ATR_TRAILING_MULT = 1.0  # 减半仓倍数

# 信号确认
CONFIRM_DAYS = 3
BUY_THRESHOLD = 0.5
SELL_THRESHOLD = -0.2
QUICK_BUY_THRESHOLD = 0.6

# 大盘状态映射
MARKET_STATES = {"bull": 1.2, "oscillate": 1.0, "bear": 0.8}

# 近期高低点窗口
RECENT_HIGH_WINDOW = 10
RECENT_LOW_WINDOW = 20

# 周线参数
WEEKLY_MA = 20

# 风险提示
RISK_WARNING_DAYS = 3
RISK_WARNING_THRESHOLD = -0.1

# 文件路径
POSITION_FILE = "positions.csv"
STATE_FILE = "etf_state.json"
CACHE_FILE = "weight_cache.json"

# 日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------- 数据获取函数 ----------------------------
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


def get_daily_data(
    code, start_date, end_date, fields="date,code,open,high,low,close,volume,amount"
):
    rs = bs.query_history_k_data_plus(
        code, fields, start_date=start_date, end_date=end_date, frequency="d"
    )
    if rs.error_code != "0":
        print(f"获取数据失败 {code}: {rs.error_msg}")
        return None
    data_list = []
    while (rs.error_code == "0") & rs.next():
        data_list.append(rs.get_row_data())
    if not data_list:
        return None
    df = pd.DataFrame(data_list, columns=rs.fields)
    df["date"] = pd.to_datetime(df["date"])
    df.set_index("date", inplace=True)
    numeric_cols = ["open", "high", "low", "close", "volume", "amount"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def get_weekly_data(code, start_date, end_date):
    df = get_daily_data(code, start_date, end_date)
    if df is None or df.empty:
        return None
    weekly = df.resample("W-FRI").last()
    weekly["ma_short"] = weekly["close"].rolling(window=WEEKLY_MA).mean()
    return weekly


def get_realtime_price_sina(code):
    try:
        domains = ["hq.sinajs.cn", "hq2.sinajs.cn", "hq3.sinajs.cn"]
        for domain in domains:
            sina_code = code.replace(".", "")
            url = f"http://{domain}/list={sina_code}"
            headers = {"Referer": "http://finance.sina.com.cn"}
            r = requests.get(url, headers=headers, timeout=3)
            if r.status_code == 200:
                data = r.text
                parts = data.split('"')[1].split(",")
                if len(parts) > 3 and parts[3]:
                    return float(parts[3])
        return None
    except:
        return None


def get_realtime_index_sina(code="sh000001"):
    try:
        domains = ["hq.sinajs.cn", "hq2.sinajs.cn", "hq3.sinajs.cn"]
        for domain in domains:
            url = f"http://{domain}/list={code}"
            headers = {"Referer": "http://finance.sina.com.cn"}
            r = requests.get(url, headers=headers, timeout=3)
            if r.status_code == 200:
                data = r.text
                parts = data.split('"')[1].split(",")
                if len(parts) > 3 and parts[3]:
                    return float(parts[3])
        return None
    except:
        return None


def load_positions():
    if not os.path.exists(POSITION_FILE):
        raise FileNotFoundError(f"ETF列表文件 {POSITION_FILE} 不存在。")
    df = pd.read_csv(POSITION_FILE, encoding="utf-8-sig")
    if "代码" not in df.columns or "名称" not in df.columns:
        raise ValueError("CSV文件必须包含 '代码' 和 '名称' 列。")
    return df[["代码", "名称"]]


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        for code, value in data.items():
            if isinstance(value, dict) and "score_history" in value:
                if value["score_history"] and all(
                    isinstance(x, (int, float)) for x in value["score_history"]
                ):
                    print(f"检测到旧版评分历史（无日期），已重置 {code} 的评分。")
                    value["score_history"] = []
        return data
    return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


# ---------------------------- 技术指标计算 ----------------------------
def calculate_macd(df, short=12, long_=26, signal=9):
    exp1 = df["close"].ewm(span=short, adjust=False).mean()
    exp2 = df["close"].ewm(span=long_, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    df["macd_dif"] = macd
    df["macd_dea"] = signal_line
    df["macd_hist"] = macd - signal_line
    return df


def calculate_kdj(df, n=9, m1=3, m2=3):
    low_list = df["low"].rolling(window=n).min()
    low_list.fillna(value=df["low"].expanding().min(), inplace=True)
    high_list = df["high"].rolling(window=n).max()
    high_list.fillna(value=df["high"].expanding().max(), inplace=True)
    rsv = (df["close"] - low_list) / (high_list - low_list) * 100
    df["kdj_k"] = rsv.ewm(alpha=1 / m1, adjust=False).mean()
    df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / m2, adjust=False).mean()
    df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
    return df


def calculate_bollinger(df, period=20, width=2):
    df["boll_mid"] = df["close"].rolling(window=period).mean()
    df["boll_std"] = df["close"].rolling(window=period).std()
    df["boll_up"] = df["boll_mid"] + width * df["boll_std"]
    df["boll_low"] = df["boll_mid"] - width * df["boll_std"]
    return df


def calculate_williams(df, period=14):
    high = df["high"].rolling(window=period).max()
    low = df["low"].rolling(window=period).min()
    df["williams_r"] = (high - df["close"]) / (high - low) * -100
    return df


def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(df, period=14):
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr


def calculate_indicators(df, ma_short=20, vol_ma=5):
    df = df.copy()
    df["ma_short"] = df["close"].rolling(window=ma_short).mean()
    if vol_ma is not None:
        df["vol_ma"] = df["volume"].rolling(window=vol_ma).mean()
    df["amount_ma"] = (
        df["amount"].rolling(window=vol_ma).mean() if vol_ma is not None else None
    )
    df = calculate_macd(df)
    df = calculate_kdj(df)
    df = calculate_bollinger(df)
    df = calculate_williams(df)
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    df["atr"] = calculate_atr(df, ATR_PERIOD)
    df["recent_high_10"] = df["high"].rolling(window=10).max()
    df["recent_low_20"] = df["low"].rolling(window=20).min()
    return df


# ---------------------------- 宏观状态 ----------------------------
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


# ---------------------------- AI 动态权重 ----------------------------
def _get_cache_key(
    macro_status,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
):
    data = f"{macro_status}_{sentiment_factor}_{market_above_ma20}_{market_above_ma60}_{market_amount_above_ma20}"
    return hashlib.md5(data.encode()).hexdigest()


def _load_cache():
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


def _save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def _validate_weights(weights, expected_keys):
    if not isinstance(weights, dict):
        return False
    missing = set(expected_keys) - set(weights.keys())
    if missing:
        logger.warning(f"权重缺少键: {missing}，将补0")
        for k in missing:
            weights[k] = 0.0
    for k, v in weights.items():
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            logger.warning(f"权重键 {k} 值 {v} 不在[0,1]内，将裁剪")
            weights[k] = max(0.0, min(1.0, v))
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        logger.warning(f"权重总和 {total} 不为1，将归一化")
        for k in weights:
            weights[k] /= total
    return True


def _build_prompt(
    macro_status,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    indicators,
):
    indicator_desc = "\n".join([f"- {k}: {v}" for k, v in indicators.items()])
    prompt = f"""
你是一个量化交易策略专家。当前市场环境如下：
- 宏观状态：{macro_status}（牛市、震荡市、熊市之一）
- 情绪系数：{sentiment_factor}（0.6-1.2之间，越低越恐慌）
- 大盘站上20日均线：{"是" if market_above_ma20 else "否"}
- 大盘站上60日均线：{"是" if market_above_ma60 else "否"}
- 市场成交额高于20日均额：{"是" if market_amount_above_ma20 else "否"}

请为以下指标分配权重（所有权重之和应为1），输出严格的JSON对象，键为指标名称，值为权重（浮点数，保留2位小数）。示例输出格式：
{{
    "price_above_ma20": 0.25,
    "volume_above_ma5": 0.20,
    ...
}}
指标列表：
{indicator_desc}
"""
    return prompt


def deepseek_generate_weights(
    macro_status,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    api_key,
    model="deepseek-chat",
    use_cache=True,
):
    cache_key = _get_cache_key(
        macro_status,
        sentiment_factor,
        market_above_ma20,
        market_above_ma60,
        market_amount_above_ma20,
    )

    if use_cache:
        cache = _load_cache()
        if cache_key in cache:
            logger.info("使用缓存权重")
            return cache[cache_key]

    indicator_names = list(STRATEGY_WEIGHTS.keys())
    indicator_desc = {k: "" for k in indicator_names}
    prompt = _build_prompt(
        macro_status,
        sentiment_factor,
        market_above_ma20,
        market_above_ma60,
        market_amount_above_ma20,
        indicator_desc,
    )

    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        import time

        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "你是一个量化交易专家，输出严格的JSON格式。不要包含其他解释。",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=800,
            temperature=0.0,
            timeout=10,
        )
        elapsed = time.time() - start_time
        content = response.choices[0].message.content
        logger.info(f"API调用成功，耗时 {elapsed:.2f}s")

        import re

        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            weights = json.loads(json_match.group())
            if _validate_weights(weights, indicator_names):
                if use_cache:
                    cache = _load_cache()
                    cache[cache_key] = weights
                    _save_cache(cache)
                return weights
            else:
                logger.error("权重验证失败")
                return None
        else:
            logger.error("无法解析JSON")
            return None
    except Exception as e:
        logger.error(f"DeepSeek API调用失败: {e}")
        if use_cache:
            cache = _load_cache()
            if cache_key in cache:
                logger.info("使用缓存中的权重（上次成功）")
                return cache[cache_key]
        logger.warning("使用默认权重")
        return STRATEGY_WEIGHTS.copy()


# ---------------------------- ETF 分析核心 ----------------------------
def _get_recent_high_low(hist_df, row_idx):
    if "recent_high_10" in hist_df.columns and "recent_low_20" in hist_df.columns:
        return (
            hist_df.iloc[row_idx]["recent_high_10"],
            hist_df.iloc[row_idx]["recent_low_20"],
        )
    else:
        high_series = hist_df["high"].rolling(window=RECENT_HIGH_WINDOW).max()
        low_series = hist_df["low"].rolling(window=RECENT_LOW_WINDOW).min()
        return high_series.iloc[row_idx], low_series.iloc[row_idx]


def _check_signal_confirm(score_history, target_position):
    if len(score_history) < CONFIRM_DAYS:
        return None, None
    recent_scores = [item["score"] for item in score_history]
    if all(s > BUY_THRESHOLD for s in recent_scores):
        signal = {
            "action": "BUY",
            "ratio": target_position,
            "reason": f"连续{CONFIRM_DAYS}天评分>{BUY_THRESHOLD}",
            "text": f"  🟢 买入{target_position*100:.0f}%:连续{CONFIRM_DAYS}天>{BUY_THRESHOLD}",
        }
        return "BUY", signal
    elif all(s < SELL_THRESHOLD for s in recent_scores):
        signal = {
            "action": "SELL",
            "ratio": 0.5,
            "reason": f"连续{CONFIRM_DAYS}天评分<{SELL_THRESHOLD}",
            "is_clear": False,
            "text": f"  🔴 卖出50%:连续{CONFIRM_DAYS}天<{SELL_THRESHOLD}",
        }
        return "SELL", signal
    return None, None


def _check_quick_signal(score_history, last_score):
    if len(score_history) >= 2:
        prev_score = score_history[-2]["score"]
        if last_score > prev_score and last_score > QUICK_BUY_THRESHOLD:
            return True
    return False


def _check_risk_warning(score_history):
    if len(score_history) >= RISK_WARNING_DAYS:
        recent = [item["score"] for item in score_history[-RISK_WARNING_DAYS:]]
        if all(s < RISK_WARNING_THRESHOLD for s in recent):
            return True
    return False


def _format_output(
    name,
    code,
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
    macro_status,
    market_factor,
    sentiment_factor,
    final_score,
    target_position,
    confirm_info,
    signal_info,
    risk_warning,
):
    vol_m = volume / 1e6
    vol_ma_m = vol_ma / 1e6
    lines = [f"【{name} ({code})】"]
    lines.append(
        f"  价格:{real_price:.3f} | 20日线:{ma20:.3f} | 量:{vol_m:.2f}M/5日均:{vol_ma_m:.2f}M"
    )
    macd_symbol = "✓" if macd_golden else "✗"
    kdj_symbol = "✓" if kdj_golden else "✗"
    lines.append(
        f"  MACD:{macd_symbol} KDJ:{kdj_symbol} RSI:{rsi:.1f} | 布林:{boll_up:.3f}/{boll_low:.3f} | 威廉:{williams_r:.1f}"
    )
    lines.append(
        f"  大盘:{macro_status}({market_factor:.2f}) 情绪:{sentiment_factor:.2f} | 最终评分:{final_score:.2f} | 仓位建议:{target_position*100:.0f}%"
    )
    if risk_warning:
        lines.append(
            f"  ⚠️ 风险提示：连续{RISK_WARNING_DAYS}天评分低于{RISK_WARNING_THRESHOLD}"
        )
    if signal_info:
        lines.append(signal_info["text"])
    else:
        days, last_score = confirm_info["days"], confirm_info["last_score"]
        if last_score > BUY_THRESHOLD:
            lines.append(f"  🟢 偏买入确认中({days}/{CONFIRM_DAYS})")
        elif last_score < SELL_THRESHOLD:
            lines.append(f"  🔴 偏卖出确认中({days}/{CONFIRM_DAYS})")
        else:
            lines.append(f"  ⚪ 中性确认中({days}/{CONFIRM_DAYS})")
    return "\n".join(lines)


def calculate_score(
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
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    ret_etf_5d,
    ret_market_5d,
    break_ma,
    trailing_half,
    trailing_clear,
    profit_hit,
    weekly_above_ma,
    weekly_below_ma,
    weights=None,
):
    if weights is None:
        weights = STRATEGY_WEIGHTS

    conditions = {
        "price_above_ma20": real_price > ma20,
        "volume_above_ma5": volume > vol_ma,
        "macd_golden_cross": macd_golden,
        "kdj_golden_cross": kdj_golden,
        "bollinger_break_up": real_price > boll_up,
        "williams_oversold": williams_r > 80,
        "price_below_ma20": real_price < ma20,
        "bollinger_break_down": real_price < boll_low,
        "williams_overbought": williams_r < 20,
        "rsi_overbought": rsi > 70,
        "market_above_ma20": market_above_ma20,
        "market_above_ma60": market_above_ma60,
        "market_amount_above_ma20": market_amount_above_ma20,
        "outperform_market": ret_etf_5d > ret_market_5d,
        "underperform_market": not (ret_etf_5d > ret_market_5d),
        "stop_loss_ma_break": break_ma,
        "trailing_stop_clear": trailing_clear,
        "trailing_stop_half": trailing_half,
        "profit_target_hit": profit_hit,
        "weekly_above_ma20": weekly_above_ma,
        "weekly_below_ma20": weekly_below_ma,
    }
    score = 0.0
    for key, cond in conditions.items():
        if cond:
            weight = weights.get(key, 0)
            score += weight
            if key not in weights:
                print(f"警告：权重字典缺少键 '{key}'，使用0")
    return score


def map_score_to_position(score):
    if score >= 1.0:
        return 1.0
    elif score >= 0.8:
        return 0.8
    elif score >= 0.6:
        return 0.6
    elif score >= 0.4:
        return 0.4
    elif score >= 0.2:
        return 0.2
    else:
        return 0.0


def analyze_etf_signal(
    code,
    name,
    real_price,
    hist_df,
    weekly_df,
    macro_status,
    market_factor,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    ret_market_5d,
    today,
    state,
    weights=None,
):
    if hist_df is None or len(hist_df) < 20:
        return f"【{name} ({code})】\n  历史数据不足", None, state

    latest = hist_df.iloc[-1]
    ma20 = latest["ma_short"]
    vol_ma = latest["vol_ma"]
    volume = latest["volume"]
    macd_dif = latest["macd_dif"]
    macd_dea = latest["macd_dea"]
    kdj_k = latest["kdj_k"]
    kdj_d = latest["kdj_d"]
    rsi = latest["rsi"]
    boll_up = latest["boll_up"]
    boll_low = latest["boll_low"]
    williams_r = latest["williams_r"]
    atr = latest["atr"]

    # 判断金叉
    if len(hist_df) >= 2:
        prev = hist_df.iloc[-2]
        macd_golden = (macd_dif > macd_dea) and (prev["macd_dif"] <= prev["macd_dea"])
        kdj_golden = (kdj_k > kdj_d) and (prev["kdj_k"] <= prev["kdj_d"])
    else:
        macd_golden = kdj_golden = False

    # 近5日涨幅
    if len(hist_df) >= 5:
        ret_etf_5d = (real_price / hist_df.iloc[-5]["close"]) - 1
    else:
        ret_etf_5d = 0

    # 近期高点和低点
    recent_high, recent_low = _get_recent_high_low(hist_df, -1)
    if np.isnan(recent_high):
        recent_high = hist_df["high"].max()
    if np.isnan(recent_low):
        recent_low = hist_df["low"].min()

    drawdown = (recent_high - real_price) / recent_high if recent_high > 0 else 0
    gain = (real_price - recent_low) / recent_low if recent_low > 0 else 0

    # ATR动态止损
    atr_pct = atr / real_price if real_price > 0 else 0
    trailing_clear = drawdown >= (ATR_STOP_MULT * atr_pct)
    trailing_half = drawdown >= (ATR_TRAILING_MULT * atr_pct) and not trailing_clear

    break_ma = real_price < ma20
    profit_hit = any(gain >= threshold for threshold, _ in PROFIT_TARGETS)

    # 周线判断
    weekly_above_ma = False
    weekly_below_ma = False
    if weekly_df is not None and not weekly_df.empty:
        weekly_latest = weekly_df.iloc[-1]
        weekly_close = weekly_latest["close"]
        weekly_ma = weekly_latest.get("ma_short", np.nan)
        if not np.isnan(weekly_ma):
            weekly_above_ma = weekly_close > weekly_ma
            weekly_below_ma = weekly_close < weekly_ma

    # 计算基础评分
    base_score = calculate_score(
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
        market_above_ma20,
        market_above_ma60,
        market_amount_above_ma20,
        ret_etf_5d,
        ret_market_5d,
        break_ma,
        trailing_half,
        trailing_clear,
        profit_hit,
        weekly_above_ma,
        weekly_below_ma,
        weights,
    )
    final_score = base_score * market_factor * sentiment_factor
    if np.isnan(final_score):
        final_score = 0.0
    target_position = map_score_to_position(final_score)

    # 更新评分历史
    today_str = today.strftime("%Y-%m-%d")
    if "score_history" not in state:
        state["score_history"] = []
    found = False
    for item in state["score_history"]:
        if item.get("date") == today_str:
            item["score"] = final_score
            found = True
            break
    if not found:
        state["score_history"].append({"date": today_str, "score": final_score})
    state["score_history"] = sorted(state["score_history"], key=lambda x: x["date"])
    if len(state["score_history"]) > CONFIRM_DAYS:
        state["score_history"] = state["score_history"][-CONFIRM_DAYS:]

    # 信号确认
    signal_type, signal = _check_signal_confirm(state["score_history"], target_position)
    if not signal:
        if len(state["score_history"]) >= 2:
            last_score = state["score_history"][-1]["score"]
            if _check_quick_signal(state["score_history"], last_score):
                quick_ratio = min(target_position * 0.5, 0.5)
                signal = {
                    "action": "BUY",
                    "ratio": quick_ratio,
                    "reason": f"快速信号（评分上升且>{QUICK_BUY_THRESHOLD}）",
                    "text": f"  🟢 买入（快速）{quick_ratio*100:.0f}%:快速信号",
                }
                signal_type = "BUY_QUICK"

    risk_warning = _check_risk_warning(state["score_history"])

    confirm_info = {
        "days": len(state["score_history"]),
        "last_score": state["score_history"][-1]["score"],
    }
    output = _format_output(
        name,
        code,
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
        macro_status,
        market_factor,
        sentiment_factor,
        final_score,
        target_position,
        confirm_info,
        signal,
        risk_warning,
    )

    return output, signal, state


# ---------------------------- 主程序 ----------------------------
def _get_market_data(start_date, today_str):
    """获取宏观和大盘数据"""
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


def _analyze_single_etf(
    code, name, hist_df, weekly_df, state, market_status, today, weights
):
    real_price = get_realtime_price_sina(code)
    if real_price is None:
        return f"【{name} ({code})】\n  获取实时价格失败", None, code, name, state

    status, signal, new_state = analyze_etf_signal(
        code,
        name,
        real_price,
        hist_df,
        weekly_df,
        market_status["macro_status"],
        market_status["market_factor"],
        market_status["sentiment_factor"],
        market_status["market_above_ma20"],
        market_status["market_above_ma60"],
        market_status["market_amount_above_ma20"],
        market_status["ret_market_5d"],
        today,
        state,
        weights,
    )
    return status, signal, code, name, new_state


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


def real_time_analysis():
    logger.info("开始实时分析")
    try:
        etf_list = load_positions()
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

    all_state = load_state()
    for _, row in etf_list.iterrows():
        code = row['代码']
        if code not in all_state:
            all_state[code] = {}

    etf_hist_cache, weekly_cache = _fetch_etf_data(etf_list, start_date, today_str)

    # 获取动态权重（AI）
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

    save_state(all_state)

def main():
    if not silent_login():
        return
    real_time_analysis()
    silent_logout()

if __name__ == '__main__':
    main()
