#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块：存放所有固定参数、默认权重、默认参数及邮件配置。
"""
import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
POSITION_FILE = os.path.join(DATA_DIR, "positions.csv")
STATE_FILE = os.path.join(DATA_DIR, "etf_state.json")
CACHE_FILE = os.path.join(DATA_DIR, "weight_cache.json")

# ---------------------------- ETF 技术参数 ----------------------------
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

# ---------------------------- 默认权重 ----------------------------
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
    "downside_momentum": 0.15,
    "max_drawdown_stop": 0.00,
}
DEFAULT_PARAMS = {
    "CONFIRM_DAYS": 3,
    "BUY_THRESHOLD": 0.5,
    "SELL_THRESHOLD": -0.2,
    "QUICK_BUY_THRESHOLD": 0.6,
    "RECENT_HIGH_WINDOW": 10,
    "RECENT_LOW_WINDOW": 20,
}

# ---------------------------- 技术指标参数 ----------------------------
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
KDJ_N = 9
BOLL_WINDOW = 20
BOLL_STD_MULT = 2
WILLIAMS_WINDOW = 14
RSI_WINDOW = 14
TMSV_MA20_WINDOW = 20
TMSV_MA60_WINDOW = 60
TMSV_ATR_WINDOW = 14
TMSV_VOL_MA_WINDOW = 20

# ---------------------------- 动量缺失惩罚 ----------------------------
MOMENTUM_MISSING_PENALTY = 0.95

# ---------------------------- 非线性缩放 ----------------------------
NONLINEAR_SCALE_BULL = 2.5
NONLINEAR_SCALE_RANGE = 1.5

# ---------------------------- Sigmoid 参数 ----------------------------
SIGMOID_STEEPNESS_DEFAULT = 5.0
SIGMOID_STEEPNESS_VOLUME = 3.0

# ---------------------------- 硬止损阈值 ----------------------------
HARD_STOP_DRAWDOWN = 0.08
HARD_STOP_MA_BREAK_PCT = 0.05

# ---------------------------- 情绪过热惩罚 ----------------------------
SENTIMENT_OVERHEAT_THRESHOLD = 1.25
SENTIMENT_PENALTY_FACTOR = 0.8

# ---------------------------- 缓存有效期 ----------------------------
CACHE_EXPIRE_SECONDS = 600

# ---------------------------- 因子计算通用参数 ----------------------------
PRICE_DEVIATION_MA_MULT = 0.1
VOLUME_RATIO_CENTER = 0.2
OUTPERFORM_MARKET_DIV = 0.05
WILLIAMS_OVERBOUGHT_THRESH = -20
WILLIAMS_OVERSOLD_THRESH = -80
RSI_OVERBOUGHT_THRESH = 70
RSI_OVERBOUGHT_DIV = 30
PROFIT_TARGET_DIV = PROFIT_TARGET
MAX_DRAWDOWN_STOP_DIV = 0.08
WILLIAMS_NORMALIZE_DIV = 20

# ---------------------------- TMSV 动态权重与参数 ----------------------------
TMSV_HIGH_VOL_THRESH = 0.03
TMSV_TREND_REDUCE = 0.05
TMSV_MIN_TREND_WEIGHT = 0.15
TMSV_TREND_MA20_WEIGHT = 0.5
TMSV_TREND_MA60_WEIGHT = 0.3
TMSV_TREND_SLOPE_WEIGHT = 0.2
TMSV_MOM_RSI_WEIGHT = 0.6
TMSV_MOM_MACD_WEIGHT = 0.4
TMSV_VOL_RATIO_WEIGHT = 0.7
TMSV_VOL_CONSIST_WEIGHT = 0.3
TMSV_VOL_LOW_THRESH = 0.01
TMSV_VOL_HIGH_THRESH = 0.03
TMSV_VOL_LOW_FACTOR = 1.5
TMSV_VOL_HIGH_FACTOR = 0.6
TMSV_VOL_MID_FACTOR_BASE = 1.2
TMSV_VOL_MID_FACTOR_SLOPE = 0.6
TMSV_VOL_BAND_WIDTH = 0.02

# ---------------------------- 信号确认参数 ----------------------------
SIGNAL_SLOPE_BUY_THRESH = 0.05
SIGNAL_AVG_OFFSET = 0.1
SIGNAL_SLOPE_WEAK = 0.02
SIGNAL_SELL_SLOPE = -0.05
SIGNAL_SELL_WEAK_SLOPE = -0.02
SIGNAL_HIGH_VOL_BUY_SLOPE = 0.1
SIGNAL_HIGH_VOL_DAYS = 4
SIGNAL_MID_VOL_BUY_SLOPE = 0.08
SIGNAL_MID_VOL_DAYS = 3
MIN_CONFIRM_DAYS = 2
MAX_CONFIRM_DAYS = 5

# ---------------------------- 动态确认天数波动率阈值 ----------------------------
VOL_HIGH_CONFIRM = 0.04
VOL_MID_CONFIRM = 0.025

# ---------------------------- 评分等级阈值 ----------------------------
ACTION_LEVEL_THRESHOLDS = [0.8, 0.7, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8]
ACTION_LEVEL_NAMES = [
    "极度看好", "强烈买入", "买入", "谨慎买入", "偏多持有",
    "中性偏多", "中性偏空", "偏空持有", "谨慎卖出", "卖出"
]

# ---------------------------- 参数动态调整 ----------------------------
ADJUST_MULT_BASE = 1.2
ADJUST_BUY_DELTA_MAX = 0.03
ADJUST_SELL_DELTA_MAX = 0.03


def get_email_config():
    """获取邮件配置（从环境变量读取）"""
    return {
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.qq.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "sender_email": os.getenv("SENDER_EMAIL", ""),
        "sender_password": os.getenv("SENDER_PASSWORD", ""),
        "receiver_email": os.getenv("RECEIVER_EMAIL", ""),
        "send_email": os.getenv("SEND_EMAIL", "false").lower() == "true",
    }