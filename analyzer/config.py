#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块：存放全局固定参数、默认权重、默认参数及邮件配置。
因子计算相关常量已主要迁移至 factors.py，此处保留少量跨模块引用常量。
"""
import os

# ---------------------------- 路径配置 ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
PROFIT_TARGET = 0.15
WEEKLY_MA = 20
RISK_WARNING_DAYS = 3
RISK_WARNING_THRESHOLD = -0.1

# ---------------------------- 最终输出是否使用 Unicode 符号 ----------------------------
USE_UNICODE = True

# ---------------------------- 止盈提示控制 ----------------------------
TAKE_PROFIT_DISPLAY_LEVELS = True
PROFIT_LOW_WATCH_MULT = 0.7
PROFIT_LOW_HALF_MULT = 1.0
PROFIT_LOW_CLEAR_MULT = 1.6

# ---------------------------- 默认权重 ----------------------------
DEFAULT_BUY_WEIGHTS = {
    "price_above_ma20": 0.18,
    "volume_above_ma5": 0.14,
    "macd_golden_cross": 0.08,
    "kdj_golden_cross": 0.08,
    "bollinger_break_up": 0.06,
    "williams_oversold": 0.08,
    "market_above_ma20": 0.05,
    "market_above_ma60": 0.08,
    "market_amount_above_ma20": 0.05,
    "outperform_market": 0.10,
    "weekly_above_ma20": 0.10,
    "tmsv_score": 0.18,
    "rsi_oversold": 0.06,
}

DEFAULT_SELL_WEIGHTS = {
    "price_below_ma20": 0.40,
    "bollinger_break_down": 0.20,
    "williams_overbought": 0.10,
    "rsi_overbought": 0.15,
    "underperform_market": 0.20,
    "stop_loss_ma_break": 1.00,
    "weekly_below_ma20": 0.20,
    "downside_momentum": 0.15,
    "max_drawdown_stop": 0.02,
}

DEFAULT_PARAMS = {
    "CONFIRM_DAYS": 3,
    "BUY_THRESHOLD": 0.5,
    "SELL_THRESHOLD": -0.2,
    "QUICK_BUY_THRESHOLD": 0.6,
    "RECENT_HIGH_WINDOW": 10,
    "RECENT_LOW_WINDOW": 14,
}

BUY_FACTOR_NAMES = list(DEFAULT_BUY_WEIGHTS.keys())
SELL_FACTOR_NAMES = list(DEFAULT_SELL_WEIGHTS.keys())

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

# ---------------------------- 硬止损阈值 ----------------------------
HARD_STOP_DRAWDOWN = 0.08

# ---------------------------- 情绪过热惩罚 ----------------------------
SENTIMENT_OVERHEAT_THRESHOLD = 1.25
SENTIMENT_PENALTY_FACTOR = 0.8
SENTIMENT_LOWER_BOUND = 0.70

# ---------------------------- 缓存有效期 ----------------------------
CACHE_EXPIRE_SECONDS = 600

# ---------------------------- 跨模块保留的因子常量（兼顾 ai.py 等旧引用）---------------------------
RSI_OVERSOLD_THRESH = 30  # RSI 超卖阈值
TAKE_PROFIT_WARNING_THRESHOLD = 0.15  # 止盈预警基础阈值（15%）

# 说明：其他因子常量已迁移至 factors.py，新代码请优先从 factors 导入

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
    "极度看好",
    "强烈买入",
    "买入",
    "谨慎买入",
    "偏多持有",
    "中性偏多",
    "中性偏空",
    "偏空持有",
    "谨慎卖出",
    "卖出",
]

# ---------------------------- 参数动态调整 ----------------------------
ADJUST_MULT_BASE = 1.2
ADJUST_BUY_DELTA_MAX = 0.03
ADJUST_SELL_DELTA_MAX = 0.03

# ==================== 新增：AI 深度融合控制常量 ====================
# 是否启用个股权重微调
AI_PER_ETF_WEIGHT_ADJUST = True
# 个股权重微调的最大偏移幅度（相对于全局权重）
AI_PER_ETF_WEIGHT_MAX_DELTA = 0.08
# 是否启用 AI 参数建议（买卖阈值、确认天数）
AI_PARAMS_ADVISE = True
# AI 参数建议对现有参数的影响权重（0:只用规则，1:只用AI，建议0.3~0.5）
AI_PARAMS_ADVISE_TRUST = 0.4
# 批量生成 ETF 评论时每批数量
AI_BATCH_COMMENT_SIZE = 6
# 批量生成止盈建议时每批数量
AI_BATCH_TAKE_PROFIT_SIZE = 6
# AI 市场状态分析时是否传入情绪指标集
AI_MARKET_STATE_WITH_SENTIMENT = True

# ---------------------------- 邮件配置 ----------------------------
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

# ---------------------------- 新增：原 analyzer.py 顶部的常量 ----------------------------
MA30_WINDOW = 30
MA30_WEAKNESS_PENALTY = 0.9

TMSV_PRICE_DIVISOR = 0.1
TMSV_SLOPE_SCALE = 10.0
TMSV_RSI_SCALE = 3.33
TMSV_MACD_DIFF_EPS = 0.001
TMSV_MACD_CHANGE_SCALE = 100.0
TMSV_VOL_RATIO_BASE = 0.8
TMSV_VOL_RATIO_DIVISOR = 1.2
TMSV_VOL_CONSIST_SCORE = 100.0

RISK_EXTREME_LOW = -0.5
RISK_EXTREME_HIGH = 0.8
RISK_HIGH_VOL_THRESH = 0.03

TAKE_PROFIT_BULL_MULT = 1.10
TAKE_PROFIT_BEAR_MULT = 0.90
TAKE_PROFIT_HIGHVOL_MULT = 0.85

HISTORY_DAYS = 200
MAX_WORKERS = 5