# data.py
# 数据获取与存储模块

import os
import json
import pandas as pd
import numpy as np
import datetime
import requests
import baostock as bs
from contextlib import redirect_stdout
from config import RSI_PERIOD

# ---------- 登录/登出 ----------
def silent_login():
    with open(os.devnull, "w") as f, redirect_stdout(f):
        lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return False
    return True

def silent_logout():
    with open(os.devnull, "w") as f, redirect_stdout(f):
        bs.logout()

# ---------- 历史数据 ----------
def get_daily_data(code, start_date, end_date, fields="date,code,open,high,low,close,volume,amount"):
    rs = bs.query_history_k_data_plus(code, fields, start_date=start_date, end_date=end_date, frequency="d")
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

# ---------- 技术指标计算 ----------
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

def calculate_indicators(df, ma_short=20, vol_ma=5):
    df = df.copy()
    df["ma_short"] = df["close"].rolling(window=ma_short).mean()
    df["vol_ma"] = df["volume"].rolling(window=vol_ma).mean()
    df["amount_ma"] = df["amount"].rolling(window=vol_ma).mean()
    df = calculate_macd(df)
    df = calculate_kdj(df)
    df = calculate_bollinger(df)
    df = calculate_williams(df)
    df["rsi"] = calculate_rsi(df["close"], RSI_PERIOD)
    return df

# ---------- 实时价格（新浪备用） ----------
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

# ---------- 状态存储（JSON） ----------
def load_state(state_file):
    if os.path.exists(state_file):
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容旧格式
        for code, value in data.items():
            if isinstance(value, dict) and "score_history" in value:
                if value["score_history"] and all(isinstance(x, (int, float)) for x in value["score_history"]):
                    print(f"检测到旧版评分历史（无日期），已重置 {code} 的评分。")
                    value["score_history"] = []
        return data
    return {}

def save_state(state, state_file):
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

# ---------- ETF列表加载 ----------
def load_positions(position_file):
    if not os.path.exists(position_file):
        raise FileNotFoundError(f"ETF列表文件 {position_file} 不存在。")
    df = pd.read_csv(position_file, encoding="utf-8-sig")
    if "代码" not in df.columns or "名称" not in df.columns:
        raise ValueError("CSV文件必须包含 '代码' 和 '名称' 列。")
    return df[["代码", "名称"]]