#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统（优化版）- 完整功能
功能：实时分析 ETF，输出评分和操作等级，支持AI动态权重、智能信号确认、动态历史天数。
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
import smtplib
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from contextlib import redirect_stdout
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional

# ---------------------------- 日志配置 ----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# 屏蔽第三方库的 INFO 日志
for lib in ["openai", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)


# ---------------------------- 固定参数 ----------------------------
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
WEEKLY_MA = 20
RISK_WARNING_DAYS = 3
RISK_WARNING_THRESHOLD = -0.1
POSITION_FILE = "positions.csv"
STATE_FILE = "etf_state.json"
CACHE_FILE = "weight_cache.json"

# 默认权重（会被AI覆盖）
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
def silent_login() -> bool:
    with open(os.devnull, "w") as f, redirect_stdout(f):
        lg = bs.login()
    if lg.error_code != "0":
        print(f"登录失败: {lg.error_msg}")
        return False
    return True

def silent_logout():
    with open(os.devnull, "w") as f, redirect_stdout(f):
        bs.logout()

def get_daily_data(code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
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

def get_weekly_data(code: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    df = get_daily_data(code, start_date, end_date)
    if df is None:
        return None
    weekly = df.resample("W-FRI").last()
    weekly["ma_short"] = weekly["close"].rolling(window=WEEKLY_MA).mean()
    return weekly

def get_realtime_price_sina(code: str) -> Optional[float]:
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

def load_positions() -> pd.DataFrame:
    try:
        df = pd.read_csv(POSITION_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(POSITION_FILE, encoding="gbk")
    return df[["代码", "名称"]]

def load_state() -> Dict:
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        logger.warning(f"状态文件损坏，重新初始化")
        return {}

def save_state(state: Dict):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ---------------------------- 技术指标 ----------------------------
def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
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
    df: pd.DataFrame, need_amount_ma: bool = True, recent_high_window: int = 10, recent_low_window: int = 20
) -> pd.DataFrame:
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
    df[f"recent_high_{recent_high_window}"] = df["high"].rolling(recent_high_window).max()
    df[f"recent_low_{recent_low_window}"] = df["low"].rolling(recent_low_window).min()
    return df

def get_sentiment_factor(macro_df: pd.DataFrame) -> float:
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

# ---------------------------- TMSV 复合指标（高效版） ----------------------------
def compute_tmsv(df: pd.DataFrame) -> pd.Series:
    """计算 TMSV，优先使用 df 中已有的列"""
    if df is None or len(df) < 20:
        return pd.Series([50.0] * max(1, len(df))) if len(df) > 0 else pd.Series([50.0])

    df = df.copy()
    # 确保必要列存在，若已存在则直接使用
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

    # 趋势得分
    price_above_ma20 = ((df["close"] - df["ma20"]) / (df["ma20"].replace(0, np.nan) * 0.1)).clip(0, 1).fillna(0)
    price_above_ma60 = ((df["close"] - df["ma60"]) / (df["ma60"].replace(0, np.nan) * 0.1)).clip(0, 1).fillna(0)
    ma20_slope = df["ma20"].diff(5) / df["ma20"].shift(5).replace(0, np.nan)
    slope_score = (ma20_slope * 10).clip(0, 1).fillna(0)
    trend_score = (price_above_ma20 * 0.5 + price_above_ma60 * 0.3 + slope_score * 0.2) * 100

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

# ---------------------------- 缓存管理（优化） ----------------------------
def _get_cache_key(macro_status: str, sentiment_factor: float, market_above_ma20: bool,
                   market_above_ma60: bool, market_amount_above_ma20: bool, volatility: float,
                   cache_type: str = "weights") -> str:
    today = datetime.date.today().strftime("%Y-%m-%d")
    return hashlib.md5(
        f"{cache_type}_{today}_{macro_status}_{sentiment_factor:.2f}_{market_above_ma20}_{market_above_ma60}_{market_amount_above_ma20}_{volatility:.3f}".encode()
    ).hexdigest()

def _load_cache() -> Dict:
    if not os.path.exists(CACHE_FILE):
        return {}
    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        # 清理超过7天的缓存（仅清理权重和参数缓存）
        now = time.time()
        expired = []
        for key, val in cache.items():
            if isinstance(val, dict) and val.get("timestamp", 0) < now - 7 * 86400:
                expired.append(key)
        for key in expired:
            del cache[key]
        if expired:
            _save_cache(cache)
        return cache
    except Exception:
        return {}

def _save_cache(cache: Dict):
    # 添加时间戳
    for key, val in cache.items():
        if isinstance(val, dict) and "timestamp" not in val:
            val["timestamp"] = time.time()
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存缓存失败: {e}")

def _validate_and_filter_weights(weights: Dict, expected_keys: List[str], name: str) -> Optional[Dict]:
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

# ---------------------------- AI 权重生成 ----------------------------
def build_weights_prompt(macro_status: str, sentiment_factor: float, market_above_ma20: bool,
                         market_above_ma60: bool, market_amount_above_ma20: bool, volatility: float) -> str:
    buy_keys = [
        "price_above_ma20", "volume_above_ma5", "macd_golden_cross", "kdj_golden_cross",
        "bollinger_break_up", "williams_oversold", "market_above_ma20", "market_above_ma60",
        "market_amount_above_ma20", "outperform_market", "weekly_above_ma20", "tmsv_score"
    ]
    sell_keys = [
        "price_below_ma20", "bollinger_break_down", "williams_overbought", "rsi_overbought",
        "underperform_market", "stop_loss_ma_break", "trailing_stop_clear", "trailing_stop_half",
        "profit_target_hit", "weekly_below_ma20"
    ]
    return f"""
你是一个量化交易策略专家。当前市场环境如下：
- 宏观状态：{macro_status}
- 情绪系数：{sentiment_factor}
- 大盘站上20日均线：{"是" if market_above_ma20 else "否"}
- 大盘站上60日均线：{"是" if market_above_ma60 else "否"}
- 成交额高于20日均额：{"是" if market_amount_above_ma20 else "否"}
- 波动率(ATR/收盘价)：{volatility:.3f}

买入因子说明：tmsv_score 是复合指标(0-100)，强度为 tmsv/100。牛市给予较高权重(0.15-0.25)，震荡市中等(0.10-0.15)，熊市较低(0.05-0.10)。

请输出买入权重和卖出权重，JSON格式：{{"buy":{{...}},"sell":{{...}}}}，每个部分总和为1。禁止添加未列出的键。单个因子权重≤0.6(熊市止损因子可到0.8)。输出示例见说明。严格JSON，无解释。"""

def deepseek_generate_weights(macro_status: str, sentiment_factor: float, market_above_ma20: bool,
                              market_above_ma60: bool, market_amount_above_ma20: bool,
                              volatility: float, api_key: str, use_cache: bool = True) -> Tuple[Dict, Dict]:
    cache_key = _get_cache_key(macro_status, sentiment_factor, market_above_ma20,
                               market_above_ma60, market_amount_above_ma20, volatility, "weights")
    if use_cache:
        cache = _load_cache()
        if cache_key in cache and "buy" in cache[cache_key] and "sell" in cache[cache_key]:
            buy = _validate_and_filter_weights(cache[cache_key]["buy"], DEFAULT_BUY_WEIGHTS.keys(), "缓存买入权重")
            sell = _validate_and_filter_weights(cache[cache_key]["sell"], DEFAULT_SELL_WEIGHTS.keys(), "缓存卖出权重")
            if buy and sell:
                return buy, sell

    prompt = build_weights_prompt(macro_status, sentiment_factor, market_above_ma20,
                                  market_above_ma60, market_amount_above_ma20, volatility)
    for attempt in range(3):
        try:
            client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
            resp = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "输出严格JSON。"}, {"role": "user", "content": prompt}],
                max_tokens=800, temperature=0.0, timeout=15
            )
            content = resp.choices[0].message.content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if not json_match:
                raise ValueError("未找到JSON")
            data = json.loads(json_match.group())
            if "buy" not in data or "sell" not in data:
                raise ValueError("缺少buy/sell字段")
            buy = _validate_and_filter_weights(data["buy"], DEFAULT_BUY_WEIGHTS.keys(), "AI买入权重")
            sell = _validate_and_filter_weights(data["sell"], DEFAULT_SELL_WEIGHTS.keys(), "AI卖出权重")
            if buy and sell:
                if use_cache:
                    cache = _load_cache()
                    cache[cache_key] = {"buy": buy, "sell": sell}
                    _save_cache(cache)
                return buy, sell
        except Exception as e:
            logger.warning(f"AI权重生成失败(尝试{attempt+1}/3): {e}")
            time.sleep(2 ** attempt)
    logger.warning("使用默认权重")
    return DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()

# ---------------------------- 市场状态精细化 ----------------------------
def refine_market_state(market_df: pd.DataFrame, api_key: str, use_cache: bool = True) -> Tuple[str, float]:
    cache_key = f"market_state_{hashlib.md5(market_df.tail(20).to_json().encode()).hexdigest()}"
    if use_cache:
        cache = _load_cache()
        if cache_key in cache and "market_state" in cache[cache_key]:
            return cache[cache_key]["market_state"], cache[cache_key]["market_factor"]

    recent = market_df.tail(20)
    close_pct = recent["close"].pct_change().mean()
    vol_pct = recent["volume"].pct_change().mean()
    volatility = (recent["close"].pct_change().std()) * 100
    above_ma20 = recent["close"].iloc[-1] > recent["ma_short"].iloc[-1]
    above_ma60 = recent["close"].iloc[-1] > recent["ma_long"].iloc[-1] if "ma_long" in recent else above_ma20

    prompt = f"""市场分析专家。根据以下大盘最近20日数据判断市场状态并给出市场因子(0.6-1.4)：
- 平均日涨跌幅:{close_pct:.4f}
- 成交量变化率:{vol_pct:.4f}
- 日波动率:{volatility:.2f}%
- 站上20日线:{"是" if above_ma20 else "否"}
- 站上60日线:{"是" if above_ma60 else "否"}
- RSI:{recent['rsi'].iloc[-1]:.1f}
- MACD:{recent['macd_dif'].iloc[-1]:.3f}/{recent['macd_dea'].iloc[-1]:.3f}
- 布林位置:{(recent['close'].iloc[-1]-recent['boll_mid'].iloc[-1])/recent['boll_std'].iloc[-1]:.2f}σ
- 威廉:{recent['williams_r'].iloc[-1]:.1f}
输出JSON:{{"state":"市场状态标签","factor":因子系数}} 可选标签:强势牛市、正常牛市、震荡偏强、震荡偏弱、弱势反弹、熊市下跌中继、熊市加速下跌。"""
    try:
        client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "system", "content": "输出严格JSON。"}, {"role": "user", "content": prompt}],
            max_tokens=200, temperature=0.0, timeout=10
        )
        data = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content, re.DOTALL).group())
        state = data.get("state", "震荡偏弱")
        factor = max(0.6, min(1.4, float(data.get("factor", 1.0))))
        if use_cache:
            cache = _load_cache()
            cache[cache_key] = {"market_state": state, "market_factor": factor}
            _save_cache(cache)
        return state, factor
    except Exception as e:
        logger.error(f"市场状态AI分析失败: {e}")
        if above_ma20 and above_ma60:
            return "正常牛市", 1.2
        if not above_ma20 and not above_ma60:
            return "熊市下跌", 0.8
        return "震荡偏弱", 1.0

# ---------------------------- 智能信号确认（核心改进） ----------------------------
def get_dynamic_history_days(volatility: float) -> int:
    """根据波动率动态调整历史天数（5~20天）"""
    if volatility > 0.04:
        return 5   # 极高波动，使用更短历史
    if volatility > 0.025:
        return 8
    if volatility > 0.015:
        return 12
    return 20    # 低波动，使用更长历史

def get_action(score: float, score_history: List[Dict], params: Dict, atr_pct: float = None) -> str:
    """智能信号确认，使用动态历史天数"""
    hist_scores = [s["score"] for s in score_history]
    if len(hist_scores) < 2:
        return "BUY" if score > params["BUY_THRESHOLD"] else ("SELL" if score < params["SELL_THRESHOLD"] else "HOLD")

    # 动态确定历史窗口长度
    window = get_dynamic_history_days(atr_pct) if atr_pct else 12
    window = min(window, len(hist_scores))
    recent = hist_scores[-window:]
    avg = sum(recent) / len(recent)
    slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
    up_days = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
    down_days = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])

    # 买入信号
    if score > params["BUY_THRESHOLD"]:
        # 强势买入：高分 + 趋势向上 + 平均分高
        if score > params["QUICK_BUY_THRESHOLD"] and slope > 0.05 and avg > params["BUY_THRESHOLD"] + 0.1:
            return "BUY"
        # 连续确认
        if len(hist_scores) >= params["CONFIRM_DAYS"]:
            confirm = hist_scores[-params["CONFIRM_DAYS"]:]
            if all(s > params["BUY_THRESHOLD"] for s in confirm) and slope >= 0:
                return "BUY"
        # 反弹买入
        if down_days >= 2 and slope > 0.02 and score > avg + 0.1:
            return "BUY"

    # 卖出信号
    if score < params["SELL_THRESHOLD"]:
        if score < params["SELL_THRESHOLD"] - 0.1 and slope < -0.05 and avg < params["SELL_THRESHOLD"] - 0.1:
            return "SELL"
        if len(hist_scores) >= params["CONFIRM_DAYS"]:
            confirm = hist_scores[-params["CONFIRM_DAYS"]:]
            if all(s < params["SELL_THRESHOLD"] for s in confirm) and slope <= 0:
                return "SELL"
        if up_days >= 2 and slope < -0.02 and score < avg - 0.1:
            return "SELL"

    # 高波动保护
    if atr_pct and atr_pct > 0.04:
        if score > params["BUY_THRESHOLD"] + 0.15 and slope > 0.1 and up_days >= 4:
            return "BUY"
        if score < params["SELL_THRESHOLD"] - 0.05 and slope < -0.08 and down_days >= 3:
            return "SELL"
        return "HOLD"
    if atr_pct and atr_pct > 0.03:
        if score > params["BUY_THRESHOLD"] + 0.1 and slope > 0.08 and up_days >= 3:
            return "BUY"
        if score < params["SELL_THRESHOLD"] - 0.1 and slope < -0.08 and down_days >= 3:
            return "SELL"

    return "HOLD"

def get_action_level(score: float) -> str:
    if score >= 0.8: return "极度看好"
    if score >= 0.7: return "强烈买入"
    if score >= 0.6: return "买入"
    if score >= 0.4: return "谨慎买入"
    if score >= 0.2: return "偏多持有"
    if score >= 0.0: return "中性偏多"
    if score >= -0.2: return "中性偏空"
    if score >= -0.4: return "偏空持有"
    if score >= -0.6: return "谨慎卖出"
    if score >= -0.8: return "卖出"
    return "强烈卖出"

# ---------------------------- 评分计算核心 ----------------------------
def strength(
    price: float, ma20: float, volume: float, vol_ma: float, macd_golden: int, kdj_golden: int,
    rsi: float, boll_up: float, boll_low: float, williams_r: float, ret_etf_5d: float, ret_market_5d: float,
    weekly_above: bool, weekly_below: bool, recent_high: float, recent_low: float, atr_pct: float,
    market_above_ma20: bool, market_above_ma60: bool, market_amount_above_ma20: bool,
    is_buy: bool, buy_weights: Dict, sell_weights: Dict, tmsv_strength: float = 0.0
) -> float:
    def cap(x): return max(0.0, min(1.0, x))
    if is_buy:
        williams_oversold = cap((-80 - williams_r) / 20) if williams_r < -80 else 0
        factors = {
            "price_above_ma20": cap((price - ma20) / (ma20 * 0.1)) if price > ma20 else 0,
            "volume_above_ma5": cap(volume / vol_ma) if volume > vol_ma else 0,
            "macd_golden_cross": macd_golden,
            "kdj_golden_cross": kdj_golden,
            "bollinger_break_up": cap((price - boll_up) / boll_up) if price > boll_up else 0,
            "williams_oversold": williams_oversold,
            "market_above_ma20": 1 if market_above_ma20 else 0,
            "market_above_ma60": 1 if market_above_ma60 else 0,
            "market_amount_above_ma20": 1 if market_amount_above_ma20 else 0,
            "outperform_market": cap((ret_etf_5d - ret_market_5d) / 0.05) if ret_etf_5d > ret_market_5d else 0,
            "weekly_above_ma20": 1 if weekly_above else 0,
            "tmsv_score": tmsv_strength,
        }
        weights = buy_weights
    else:
        factors = {
            "price_below_ma20": cap((ma20 - price) / (ma20 * 0.1)) if price < ma20 else 0,
            "bollinger_break_down": cap((boll_low - price) / boll_low) if price < boll_low else 0,
            "williams_overbought": cap((20 - williams_r) / 20) if williams_r < 20 else 0,
            "rsi_overbought": cap((rsi - 70) / 30) if rsi > 70 else 0,
            "underperform_market": cap((ret_market_5d - ret_etf_5d) / 0.05) if ret_etf_5d < ret_market_5d else 0,
            "stop_loss_ma_break": cap((ma20 - price) / (ma20 * 0.05)) if price < ma20 else 0,
            "trailing_stop_clear": cap((recent_high - price) / recent_high / (ATR_STOP_MULT * atr_pct))
                if recent_high > 0 and atr_pct > 0 and (recent_high - price) / recent_high >= ATR_STOP_MULT * atr_pct else 0,
            "trailing_stop_half": cap((recent_high - price) / recent_high / (ATR_TRAILING_MULT * atr_pct))
                if recent_high > 0 and atr_pct > 0 and (recent_high - price) / recent_high >= ATR_TRAILING_MULT * atr_pct else 0,
            "profit_target_hit": cap((price - recent_low) / recent_low / PROFIT_TARGET)
                if recent_low > 0 and (price - recent_low) / recent_low >= PROFIT_TARGET else 0,
            "weekly_below_ma20": 1 if weekly_below else 0,
        }
        weights = sell_weights
    return sum(weights.get(k, 0) * factors.get(k, 0) for k in factors)

def adjust_params_based_on_history(params: Dict, score_history: List[Dict], volatility: float) -> Dict:
    if len(score_history) < 7:
        return params
    recent = [s["score"] for s in score_history[-7:]]
    avg = sum(recent) / len(recent)
    slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
    std = np.std(recent)

    adjusted = params.copy()
    if slope < -0.02 and avg < -0.1:
        adjusted["SELL_THRESHOLD"] = min(params["SELL_THRESHOLD"] + 0.1, -0.1)
        adjusted["CONFIRM_DAYS"] = max(params["CONFIRM_DAYS"] - 1, 2)
    elif slope > 0.02 and avg > 0.1:
        adjusted["BUY_THRESHOLD"] = max(params["BUY_THRESHOLD"] - 0.1, 0.3)
        adjusted["QUICK_BUY_THRESHOLD"] = max(params["QUICK_BUY_THRESHOLD"] - 0.1, 0.5)
    if std > 0.3:
        adjusted["CONFIRM_DAYS"] = min(params["CONFIRM_DAYS"] + 1, 5)
    if volatility > 0.04:
        adjusted["BUY_THRESHOLD"] = max(params["BUY_THRESHOLD"], 0.6)
        adjusted["SELL_THRESHOLD"] = min(params["SELL_THRESHOLD"], -0.3)
        adjusted["CONFIRM_DAYS"] = max(params["CONFIRM_DAYS"], 4)
    elif volatility > 0.02:
        adjusted["BUY_THRESHOLD"] = max(params["BUY_THRESHOLD"], 0.5)
        adjusted["SELL_THRESHOLD"] = min(params["SELL_THRESHOLD"], -0.2)
    return adjusted

# ---------------------------- 单只ETF分析 ----------------------------
def analyze_etf(code: str, name: str, real_price: Optional[float], hist_df: Optional[pd.DataFrame],
                weekly_df: Optional[pd.DataFrame], market: Dict, today: datetime.date,
                state: Dict, buy_w: Dict, sell_w: Dict, params: Dict) -> Tuple[str, Optional[Dict], Dict, float]:
    # 实时价格检查
    if real_price is None:
        out = f"{name:<16} {code:<12} {'获取失败':>8} {'0.00':>6}  {'价格缺失':<10}"
        return out, None, state, 0.0

    if hist_df is None or len(hist_df) < 20:
        out = f"{name:<16} {code:<12} {real_price:>8.3f} {0.00:>6}  {'数据不足':<10}"
        return out, None, state, 0.0

    d = hist_df.iloc[-1]
    ma20, vol_ma, volume = d["ma_short"], d["vol_ma"], d["volume"]
    rsi, boll_up, boll_low, williams_r = d["rsi"], d["boll_up"], d["boll_low"], d["williams_r"]
    atr_pct = d["atr"] / real_price if real_price > 0 else 0
    recent_high_window = params["RECENT_HIGH_WINDOW"]
    recent_low_window = params["RECENT_LOW_WINDOW"]
    recent_high = d.get(f"recent_high_{recent_high_window}", hist_df["high"].rolling(recent_high_window).max().iloc[-1])
    recent_low = d.get(f"recent_low_{recent_low_window}", hist_df["low"].rolling(recent_low_window).min().iloc[-1])

    # 金叉判断
    if len(hist_df) >= 2:
        prev = hist_df.iloc[-2]
        macd_golden = 1 if (d["macd_dif"] > d["macd_dea"] and prev["macd_dif"] <= prev["macd_dea"]) else 0
        kdj_golden = 1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0
    else:
        macd_golden = kdj_golden = 0

    ret_etf_5d = (real_price / hist_df.iloc[-5]["close"]) - 1 if len(hist_df) >= 5 else 0
    weekly_above = weekly_below = False
    if weekly_df is not None and not weekly_df.empty:
        w = weekly_df.iloc[-1]
        if not np.isnan(w["ma_short"]):
            weekly_above = w["close"] > w["ma_short"]
            weekly_below = w["close"] < w["ma_short"]

    # TMSV
    try:
        tmsv_series = compute_tmsv(hist_df)
        tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
        tmsv = 50.0 if np.isnan(tmsv) else tmsv
    except Exception:
        tmsv = 50.0
    tmsv_strength = tmsv / 100.0

    buy_score = strength(real_price, ma20, volume, vol_ma, macd_golden, kdj_golden, rsi, boll_up, boll_low, williams_r,
                         ret_etf_5d, market["ret_market_5d"], weekly_above, weekly_below,
                         recent_high, recent_low, atr_pct, market["market_above_ma20"],
                         market["market_above_ma60"], market["market_amount_above_ma20"],
                         True, buy_w, sell_w, tmsv_strength)
    sell_score = strength(real_price, ma20, volume, vol_ma, macd_golden, kdj_golden, rsi, boll_up, boll_low, williams_r,
                          ret_etf_5d, market["ret_market_5d"], weekly_above, weekly_below,
                          recent_high, recent_low, atr_pct, market["market_above_ma20"],
                          market["market_above_ma60"], market["market_amount_above_ma20"],
                          False, buy_w, sell_w, tmsv_strength)
    raw = buy_score - sell_score
    final = raw * market["market_factor"] * market["sentiment_factor"]
    final = max(-1.0, min(1.0, final))   # 裁剪到[-1,1]

    # 更新状态（保存所有历史）
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
    # 不限制长度，保留所有历史
    state["score_history"].sort(key=lambda x: x["date"])

    # 动态参数调整
    if len(state["score_history"]) >= 7:
        params = adjust_params_based_on_history(params, state["score_history"], atr_pct)

    # 信号判断（传入波动率）
    action = get_action(final, state["score_history"], params, atr_pct)
    action_level = get_action_level(final)

    # 风险提示
    risk_warning = ""
    if len(state["score_history"]) >= RISK_WARNING_DAYS:
        recent = [s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]]
        if all(s < RISK_WARNING_THRESHOLD for s in recent):
            risk_warning = f" ⚠️ 风险提示:连续{RISK_WARNING_DAYS}天评分低于{RISK_WARNING_THRESHOLD}"
        elif final < -0.5 or final > 0.8:
            risk_warning = f" ⚠️ 风险提示:极端评分{final:.2f}"
        elif atr_pct > 0.03:
            risk_warning = f" ⚠️ 风险提示:高波动{atr_pct:.3f}"

    output = f"{name:<16} {code:<12} {real_price:>8.3f} {final:>6.2f}  {action_level:<10}{risk_warning}"
    signal = {"action": action, "name": name, "code": code, "score": final} if action in ("BUY", "SELL") else None
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
    start = (today - datetime.timedelta(days=200)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    # 宏观数据
    market_df = get_daily_data(MARKET_INDEX, start, today_str)
    macro_df = get_daily_data(MACRO_INDEX, start, today_str)
    if market_df is None or macro_df is None:
        print("获取宏观数据失败")
        return

    macro_df = calculate_indicators(macro_df, need_amount_ma=False, recent_high_window=10, recent_low_window=20)
    macro_df["ma_long"] = macro_df["close"].rolling(MACRO_MA_LONG).mean()
    market_df = calculate_indicators(market_df, need_amount_ma=True, recent_high_window=10, recent_low_window=20)
    market_df["atr"] = calculate_atr(market_df, ATR_PERIOD)
    volatility = (market_df["atr"] / market_df["close"]).iloc[-20:].mean()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if api_key:
        market_state, market_factor = refine_market_state(market_df, api_key)
        logger.info(f"AI市场状态: {market_state}, 因子: {market_factor}")
    else:
        # 降级规则
        last = market_df.iloc[-1]
        if last["close"] > last["ma_short"] and last["close"] > last.get("ma_long", last["ma_short"]):
            market_state, market_factor = "正常牛市", 1.2
        elif last["close"] < last["ma_short"] and last["close"] < last.get("ma_long", last["ma_short"]):
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
        "ret_market_5d": (mkt["close"] / market_df.iloc[-5]["close"] - 1) if len(market_df) >= 5 else 0,
    }

    # 根据波动率调整基础参数
    params = DEFAULT_PARAMS.copy()
    if volatility > 0.04:
        params.update({"BUY_THRESHOLD": 0.65, "SELL_THRESHOLD": -0.35, "CONFIRM_DAYS": 5, "QUICK_BUY_THRESHOLD": 0.75})
    elif volatility > 0.02:
        params.update({"BUY_THRESHOLD": 0.6, "SELL_THRESHOLD": -0.3, "CONFIRM_DAYS": 4, "QUICK_BUY_THRESHOLD": 0.7})
    elif volatility < 0.01:
        params.update({"BUY_THRESHOLD": 0.4, "SELL_THRESHOLD": -0.1, "CONFIRM_DAYS": 2, "QUICK_BUY_THRESHOLD": 0.5})

    state = load_state()
    buy_w, sell_w = DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()
    if api_key:
        bw, sw = deepseek_generate_weights(market_state, sentiment, market_info["market_above_ma20"],
                                           market_info["market_above_ma60"], market_info["market_amount_above_ma20"],
                                           volatility, api_key, use_cache=False)
        if bw and sw:
            buy_w, sell_w = bw, sw
            logger.info("使用AI动态权重")
        else:
            logger.warning("AI权重生成失败，使用默认权重")

    # 输出表头
    header = f"\n{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ETF 分析报告"
    print(header)
    print(f"{'名称':<16} {'代码':<12} {'价格':>8} {'评分':>6}  {'操作':<10}")
    print("-" * 68)
    output_lines = [header, f"{'名称':<16} {'代码':<12} {'价格':>8} {'评分':>6}  {'操作':<10}", "-" * 68]

    # 并行分析
    results = []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = []
        for _, row in etf_list.iterrows():
            code, name = row["代码"], row["名称"]
            hist = get_daily_data(code, start, today_str)
            hist = calculate_indicators(hist, need_amount_ma=False,
                                        recent_high_window=params["RECENT_HIGH_WINDOW"],
                                        recent_low_window=params["RECENT_LOW_WINDOW"]) if hist is not None else None
            weekly = get_weekly_data(code, start, today_str)
            s = state.get(code, {})
            futures.append(ex.submit(analyze_etf, code, name, get_realtime_price_sina(code), hist,
                                     weekly, market_info, today, s, buy_w, sell_w, params))
        for f in futures:
            out, _, new_state, score = f.result()
            results.append((out, score))
            # 更新状态（根据输出中的代码）
            m = re.search(r'【.*?\((.*?)\)】', out)
            if m:
                code = m.group(1)
                state[code] = new_state
    # 按评分排序输出
    results.sort(key=lambda x: x[1], reverse=True)
    for out, _ in results:
        print(out)
        output_lines.append(out)

    save_state(state)
    silent_logout()


if __name__ == "__main__":
    main()