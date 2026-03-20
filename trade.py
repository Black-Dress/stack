import os
import json
import pandas as pd
import numpy as np
import datetime
import requests
import baostock as bs
from contextlib import redirect_stdout

# ==================== 可调参数 ====================
ETF_MA = 20  # ETF价格均线周期
ETF_VOL_MA = 5  # ETF成交量均线周期
MACRO_INDEX = "sh.000300"  # 宏观指数（沪深300）
MARKET_INDEX = "sh.000001"  # 大盘指数（上证）
MACRO_MA_SHORT = 20  # 宏观指数短期均线
MACRO_MA_LONG = 60  # 宏观指数长期均线
RSI_PERIOD = 14  # RSI周期（用于情绪判断）

# 策略权重（正值为买入贡献，负值为卖出贡献）
STRATEGY_WEIGHTS = {
    # ETF自身指标
    "price_above_ma20": 0.30,  # 价格站上20日线
    "volume_above_ma5": 0.20,  # 成交量高于5日均量
    "macd_golden_cross": 0.15,  # MACD金叉
    "kdj_golden_cross": 0.15,  # KDJ金叉
    "bollinger_break_up": 0.10,  # 突破布林带上轨
    "williams_oversold": 0.10,  # 威廉超卖 (WR>80)
    "price_below_ma20": -0.40,  # 价格跌破20日线
    "bollinger_break_down": -0.20,  # 跌破布林带下轨
    "williams_overbought": -0.10,  # 威廉超买 (WR<20)
    "rsi_overbought": -0.15,  # RSI超买 (>70)
    # 大盘相关指标
    "market_above_ma20": 0.10,  # 大盘站上20日线
    "market_above_ma60": 0.10,  # 大盘站上60日线
    "market_amount_above_ma20": 0.10,  # 市场成交额高于20日均额
    "outperform_market": 0.20,  # 近5日跑赢大盘
    "underperform_market": -0.20,  # 近5日跑输大盘
}

# 仓位档位（0.2, 0.4, 0.6, 0.8, 1.0）
POSITION_LEVELS = [0.2, 0.4, 0.6, 0.8, 1.0]

# 止盈止损参数（基于价格形态）
STOP_LOSS_MA = 20  # 跌破该均线建议清仓
TRAILING_STOP = 0.05  # 移动止盈回撤阈值（从近期高点回撤5%建议减半仓，回撤8%建议清仓）
PROFIT_TARGETS = [0.20, 0.40]  # 从近期低点上涨20%、40%时建议卖出部分

# 信号确认周期（连续N天评分全部满足条件才触发）
CONFIRM_DAYS = 3
BUY_THRESHOLD = 0.5  # 买入所需最低评分（连续三天 > 该值）
SELL_THRESHOLD = -0.2  # 卖出所需最高评分（连续三天 < 该值）

# 大盘状态划分（用于最终评分调整）
MARKET_STATES = {
    "bull": 1.2,  # 牛市：评分放大20%
    "oscillate": 1.0,  # 震荡市：不变
    "bear": 0.8,  # 熊市：评分打8折
}

POSITION_FILE = "positions.csv"  # ETF列表文件，只需包含代码和名称两列
STATE_FILE = "etf_state.json"  # 用于存储评分历史的JSON文件
# =================================================


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


def get_realtime_price_sina(code):
    """备用新浪接口"""
    try:
        # 尝试不同的新浪接口域名
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
    """备用新浪接口"""
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
    """加载ETF列表（只需代码和名称）"""
    if not os.path.exists(POSITION_FILE):
        raise FileNotFoundError(f"ETF列表文件 {POSITION_FILE} 不存在。")
    df = pd.read_csv(POSITION_FILE, encoding="utf-8-sig")
    if "代码" not in df.columns or "名称" not in df.columns:
        raise ValueError("CSV文件必须包含 '代码' 和 '名称' 列。")
    return df[["代码", "名称"]]


def load_state():
    """加载JSON状态文件，返回字典（评分历史为带日期的列表）"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 兼容旧格式：如果发现列表内是数值，则清空（无法追溯日期）
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
    """保存状态到JSON文件"""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def get_macro_status(macro_df):
    """根据沪深300与均线关系判断宏观状态，并返回状态和调整系数"""
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
    """根据沪深300的RSI获取情绪系数"""
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
):
    """计算基础评分"""
    score = 0.0

    # ETF自身指标
    if real_price > ma20:
        score += STRATEGY_WEIGHTS["price_above_ma20"]
    if volume > vol_ma:
        score += STRATEGY_WEIGHTS["volume_above_ma5"]
    if macd_golden:
        score += STRATEGY_WEIGHTS["macd_golden_cross"]
    if kdj_golden:
        score += STRATEGY_WEIGHTS["kdj_golden_cross"]
    if real_price > boll_up:
        score += STRATEGY_WEIGHTS["bollinger_break_up"]
    if williams_r > 80:
        score += STRATEGY_WEIGHTS["williams_oversold"]
    if real_price < ma20:
        score += STRATEGY_WEIGHTS["price_below_ma20"]
    if real_price < boll_low:
        score += STRATEGY_WEIGHTS["bollinger_break_down"]
    if williams_r < 20:
        score += STRATEGY_WEIGHTS["williams_overbought"]
    if rsi > 70:
        score += STRATEGY_WEIGHTS["rsi_overbought"]

    # 大盘指标
    if market_above_ma20:
        score += STRATEGY_WEIGHTS["market_above_ma20"]
    if market_above_ma60:
        score += STRATEGY_WEIGHTS["market_above_ma60"]
    if market_amount_above_ma20:
        score += STRATEGY_WEIGHTS["market_amount_above_ma20"]
    if ret_etf_5d > ret_market_5d:
        score += STRATEGY_WEIGHTS["outperform_market"]
    else:
        score += STRATEGY_WEIGHTS["underperform_market"]

    return score


def map_score_to_position(score):
    """将评分映射到仓位档位（0, 0.2, 0.4, 0.6, 0.8, 1.0）"""
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


def check_stop_profit(real_price, hist_df):
    """基于价格形态检查止盈止损，返回建议卖出比例（0~1）和原因"""
    if hist_df is None or len(hist_df) < 20:
        return 0.0, "数据不足"

    latest = hist_df.iloc[-1]
    ma20 = latest["ma_short"]
    recent_high = hist_df["high"].rolling(window=10).max().iloc[-1]
    recent_low = hist_df["low"].rolling(window=20).min().iloc[-1]

    # 1. 跌破20日均线 -> 清仓
    if real_price < ma20:
        return 1.0, "跌破20日均线"

    # 2. 移动止盈：从近期高点回撤
    drawdown = (recent_high - real_price) / recent_high if recent_high > 0 else 0
    if drawdown >= 0.08:
        return 1.0, f"从高点回撤{drawdown:.1%}，触发清仓"
    elif drawdown >= 0.05:
        return 0.5, f"从高点回撤{drawdown:.1%}，建议减半仓"

    # 3. 分批止盈：从近期低点上涨
    if recent_low > 0:
        gain = (real_price - recent_low) / recent_low
        if gain >= 0.40:
            return 0.3, f"从低点上涨{gain:.1%}，建议卖出30%"
        elif gain >= 0.20:
            return 0.3, f"从低点上涨{gain:.1%}，建议卖出30%"

    return 0.0, "无止盈止损信号"


def analyze_etf(
    code,
    name,
    hist_df,
    macro_status,
    market_factor,
    sentiment_factor,
    market_above_ma20,
    market_above_ma60,
    market_amount_above_ma20,
    ret_market_5d,
    today,
    state,
):
    """
    分析单个ETF，生成状态字符串和操作建议
    state: 该ETF的状态字典，包含 score_history（列表，元素为 {"date": str, "score": float}）
    """
    real_price = get_realtime_price_sina(code)
    if real_price is None:
        return f"【{name} ({code})】\n  获取实时价格失败", None, state

    if hist_df is None or len(hist_df) < ETF_MA:
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

    # 判断金叉
    if len(hist_df) >= 2:
        prev = hist_df.iloc[-2]
        macd_golden = (macd_dif > macd_dea) and (prev["macd_dif"] <= prev["macd_dea"])
        kdj_golden = (kdj_k > kdj_d) and (prev["kdj_k"] <= prev["kdj_d"])
    else:
        macd_golden = kdj_golden = False

    # 计算ETF近5日涨幅
    if len(hist_df) >= 5:
        ret_etf_5d = (real_price / hist_df.iloc[-5]["close"]) - 1
    else:
        ret_etf_5d = 0

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
    )
    final_score = base_score * market_factor * sentiment_factor
    target_position = map_score_to_position(final_score)

    # 更新带日期的评分历史
    today_str = today.strftime("%Y-%m-%d")
    if "score_history" not in state:
        state["score_history"] = []
    # 如果今天已有记录，则替换；否则追加
    found = False
    for item in state["score_history"]:
        if item.get("date") == today_str:
            item["score"] = final_score
            found = True
            break
    if not found:
        state["score_history"].append({"date": today_str, "score": final_score})
    # 按日期排序，并只保留最近 CONFIRM_DAYS 天
    state["score_history"] = sorted(state["score_history"], key=lambda x: x["date"])
    if len(state["score_history"]) > CONFIRM_DAYS:
        state["score_history"] = state["score_history"][-CONFIRM_DAYS:]

    # 检查止盈止损（优先级最高）
    sell_ratio, stop_reason = check_stop_profit(real_price, hist_df)

    # 生成输出行
    lines = [f"【{name} ({code})】"]
    lines.append(f"  实时价格: {real_price:.3f} | 20日均线: {ma20:.3f}")
    lines.append(f"  成交量: {volume:.0f} | 5日均量: {vol_ma:.0f}")
    lines.append(
        f"  MACD: {'金叉' if macd_golden else '非金叉'} | KDJ: {'金叉' if kdj_golden else '非金叉'} | RSI: {rsi:.1f}"
    )
    lines.append(
        f"  布林带: 上轨 {boll_up:.3f} 下轨 {boll_low:.3f} | 威廉: {williams_r:.1f}"
    )
    lines.append(
        f"  大盘状态: {macro_status} (因子{market_factor:.2f}) | 情绪系数: {sentiment_factor:.2f}"
    )
    lines.append(f"  基础评分: {base_score:.2f} | 最终评分: {final_score:.2f}")
    lines.append(f"  建议买入仓位比例: {target_position*100:.0f}%")

    signal = None

    if sell_ratio > 0:
        # 止盈止损信号
        if sell_ratio >= 1.0:
            lines.append(f"  🔴 建议清仓（空仓） ({stop_reason})")
        else:
            lines.append(f"  🟡 建议卖出 {sell_ratio*100:.0f}% 持仓 ({stop_reason})")
        signal = {
            "action": "SELL",
            "ratio": sell_ratio,
            "reason": stop_reason,
            "is_clear": sell_ratio >= 1.0,
        }
    else:
        # 信号确认：检查最近 CONFIRM_DAYS 天的评分是否全部满足条件
        if len(state["score_history"]) >= CONFIRM_DAYS:
            recent_scores = [item["score"] for item in state["score_history"]]
            if all(s > BUY_THRESHOLD for s in recent_scores):
                lines.append(
                    f"  🟢 建议买入 (连续{CONFIRM_DAYS}天评分 > {BUY_THRESHOLD})"
                )
                lines.append(f"     建议仓位比例: {target_position*100:.0f}%")
                signal = {
                    "action": "BUY",
                    "ratio": target_position,
                    "reason": f"连续{CONFIRM_DAYS}天评分>{BUY_THRESHOLD}",
                }
            elif all(s < SELL_THRESHOLD for s in recent_scores):
                lines.append(
                    f"  🔴 建议卖出部分或全部 (连续{CONFIRM_DAYS}天评分 < {SELL_THRESHOLD})"
                )
                lines.append(f"     建议减仓比例: 50% (可根据风险调整)")
                signal = {
                    "action": "SELL",
                    "ratio": 0.5,
                    "reason": f"连续{CONFIRM_DAYS}天评分<{SELL_THRESHOLD}",
                    "is_clear": False,
                }
            else:
                lines.append("  ⚪ 建议观望（可根据现有持仓决定持有或空仓）")
        else:
            lines.append(
                f"  ⚪ 信号确认中（还需{CONFIRM_DAYS - len(state['score_history'])}天）"
            )

    return "\n".join(lines), signal, state


def main():
    if not silent_login():
        return
    try:
        etf_list = load_positions()
        today = datetime.date.today()
        today_str = today.strftime("%Y-%m-%d")
        start_date = (today - datetime.timedelta(days=300)).strftime("%Y-%m-%d")

        # 加载JSON状态
        all_state = load_state()
        # 确保每个ETF在状态中有条目
        for _, row in etf_list.iterrows():
            code = row["代码"]
            if code not in all_state:
                all_state[code] = {}

        # 获取宏观指数（沪深300）
        macro_df = get_daily_data(MACRO_INDEX, start_date, today_str)
        if macro_df is None or macro_df.empty:
            print("无法获取宏观指数数据")
            return
        macro_df = calculate_indicators(macro_df, ma_short=MACRO_MA_SHORT)
        macro_df["ma_long"] = macro_df["close"].rolling(window=MACRO_MA_LONG).mean()

        # 获取大盘指数（上证）
        market_df = get_daily_data(MARKET_INDEX, start_date, today_str)
        if market_df is None or market_df.empty:
            print("无法获取大盘指数数据")
            return
        market_df = calculate_indicators(market_df, ma_short=20, vol_ma=20)  # 20日均额

        # 宏观状态和因子
        macro_status, market_factor = get_macro_status(macro_df)
        sentiment_factor = get_sentiment_factor(macro_df)

        # 大盘指标（用于基础评分）
        market_latest = market_df.iloc[-1]
        market_close = market_latest["close"]
        market_ma20 = market_latest["ma_short"]
        market_ma60 = market_latest.get("ma_long", market_ma20)
        market_above_ma20 = market_close > market_ma20
        market_above_ma60 = market_close > market_ma60
        market_amount = market_latest["amount"]
        market_amount_ma20 = market_latest["amount_ma"]
        market_amount_above_ma20 = market_amount > market_amount_ma20

        # 大盘近5日涨幅
        if len(market_df) >= 5:
            ret_market_5d = (market_close / market_df.iloc[-5]["close"]) - 1
        else:
            ret_market_5d = 0

        # 实时大盘指数
        real_index = get_realtime_index_sina("sh000001")
        index_info = (
            f"实时大盘: {real_index:.2f}" if real_index else "实时大盘: 获取失败"
        )

        print(f"\n日期：{today_str} | {index_info}")
        print(
            f"宏观状态：{macro_status.upper()} | 市场因子：{market_factor:.2f} | 情绪系数：{sentiment_factor:.2f}"
        )
        print(
            f"大盘站上20日线：{'是' if market_above_ma20 else '否'} | 站上60日线：{'是' if market_above_ma60 else '否'}"
        )
        print(
            f"市场成交额：{market_amount/1e8:.2f}亿 | 20日均额：{market_amount_ma20/1e8:.2f}亿 | 大于均额：{'是' if market_amount_above_ma20 else '否'}"
        )
        print("=" * 70)

        # 获取所有ETF的历史数据
        etf_hist_cache = {}
        for code in etf_list["代码"].unique():
            df = get_daily_data(code, start_date, today_str)
            if df is not None and not df.empty:
                df = calculate_indicators(df, ma_short=ETF_MA, vol_ma=ETF_VOL_MA)
                etf_hist_cache[code] = df

        signals = []

        # 逐个分析
        for idx, row in etf_list.iterrows():
            code = row["代码"]
            name = row["名称"]
            hist_df = etf_hist_cache.get(code)
            state = all_state.get(code, {})
            status, signal, new_state = analyze_etf(
                code,
                name,
                hist_df,
                macro_status,
                market_factor,
                sentiment_factor,
                market_above_ma20,
                market_above_ma60,
                market_amount_above_ma20,
                ret_market_5d,
                today,
                state,
            )
            print(status)
            print("-" * 40)
            all_state[code] = new_state
            if signal:
                signals.append({**signal, "code": code, "name": name})

        # 保存更新后的状态
        save_state(all_state)

        # 输出信号汇总
        if signals:
            print("\n" + "=" * 70)
            print("操作信号汇总")
            print("=" * 70)
            for sig in signals:
                if sig["action"] == "BUY":
                    print(
                        f"买入 {sig['name']} ({sig['code']}) - 建议仓位比例: {sig['ratio']*100:.0f}% (原因: {sig['reason']})"
                    )
                else:
                    if sig.get("is_clear"):
                        print(
                            f"清仓 {sig['name']} ({sig['code']}) - 建议卖出全部持仓 (原因: {sig['reason']})"
                        )
                    else:
                        print(
                            f"卖出 {sig['name']} ({sig['code']}) - 建议卖出比例: {sig['ratio']*100:.0f}% (原因: {sig['reason']})"
                        )
        else:
            print("\n⚪ 无操作信号，所有ETF建议观望")

    except Exception as e:
        print(f"程序出错: {e}")
    finally:
        silent_logout()


if __name__ == '__main__':
    main()
