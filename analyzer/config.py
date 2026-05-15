#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全局配置：技术参数、事件阈值、资金参数等"""
import os

# ========== 路径 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
POSITION_FILE = os.path.join(DATA_DIR, "positions.csv")
STATE_FILE = os.path.join(DATA_DIR, "etf_state.json")
POSITION_HISTORY_FILE = os.path.join(DATA_DIR, "position_history.json")

# ========== 基础技术参数 ==========
ETF_MA = 20
ETF_VOL_MA = 5
MARKET_INDEX = "sh.000001"
MACRO_INDEX = "sh.000300"
MACRO_MA_LONG = 60
ATR_PERIOD = 14
HISTORY_DAYS = 200
MAX_WORKERS = 5

MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
KDJ_N = 9
BOLL_WINDOW = 20
BOLL_STD_MULT = 2
WILLIAMS_WINDOW = 14
RSI_WINDOW = 14
MA30_WINDOW = 30

NONLINEAR_SCALE_BULL = 2.5
NONLINEAR_SCALE_RANGE = 1.5

# ========== 风险标签参数（用于显示） ==========
RISK_WARNING_DAYS = 3
RISK_WARNING_THRESHOLD = 30
MA30_WEAKNESS_PENALTY = 0.9

# ========== 动态止损（用于风险标签） ==========
ATR_STOP_MULT = 2.0
ATR_TRAILING_PROFIT_MULT = 1.5
RISK_ALERT_DISTANCE_ATR = 0.5

# ========== 成本止盈止损（仅用于提示，不决策） ==========
COST_TAKE_PROFIT_CLEAR = 0.20
COST_TAKE_PROFIT_HALF = 0.15
COST_STOP_LOSS_PCT = -0.08
PROFIT_TAKE_MODE = "soft"

# ========== 全局资金仓位约束 ==========
TOTAL_CAPITAL = 40000      # 总资金（元）
MAX_SINGLE_POSITION_RATIO = 0.30    # 单只ETF最大仓位（25%）
MAX_TOTAL_CAPITAL_RATIO = 0.9       # 所有ETF总仓位上限（100%）

# ========== 底仓参数 ==========
BASE_POSITION_RATIO = 0.07      # 底仓占总资金比例（7%）

# ========== 买入事件参数 ==========
# 早期预警
EARLY_WARNING_MA_SHORT = 5
EARLY_WARNING_MA_MEDIUM = 10
EARLY_WARNING_VOL_RATIO = 1.2

# 初步加仓（趋势反转）
REVERSAL_MA = 10
REVERSAL_MACD_HIST_CONSECUTIVE = 2
REVERSAL_RSI_MIN = 40
REVERSAL_VOL_RATIO = 1.2
TREND_ADD_PCT = 0.5                 # 每次加仓占底仓的比例（50%）
MAX_TREND_ADD_RATIO = 0.10          # 总趋势加仓上限占总资金比例（10%）

# 确认加仓
CONFIRM_MA_SHORT = 10
CONFIRM_MA_LONG = 20
CONFIRN_SCORE_MIN = 70              # 原始评分最低要求

# ========== 卖出事件参数 ==========
# 卖出早期预警
SELL_EARLY_WARNING_MA = 10
SELL_EARLY_WARNING_MACD_HIST = 2
SELL_EARLY_WARNING_VOL_RATIO = 1.2

# 初步减仓
SELL_PRELIMINARY_MA = 10
SELL_PRELIMINARY_PULLBACK = 0.05    # 5%
SELL_PRELIMINARY_REDUCE_PCT = 0.20  # 减仓20%

# 确认减仓
SELL_CONFIRM_MA = 20
SELL_CONFIRM_PULLBACK = 0.10        # 10%
SELL_CONFIRM_SCORE_THRESHOLD = 40

# ========== 深度回撤补仓参数 ==========
DIP_HIGH_PERIOD = 10
DIP_THRESHOLD = 0.06                # 回撤6%
DIP_RSI_THRESHOLD = 35
ADD_BASE_PCT = 0.30                 # 每次补仓占底仓的比例（30%）
MAX_ADD_RATIO = 0.15                # 总补仓上限占总资金比例（15%）

# ========== 过热止盈参数 ==========
OVERHEAT_RSI_THRESHOLD = 80
OVERHEAT_RSI_DAYS = 2
OVERHEAT_MA20_DEVIATION = 0.15
OVERHEAT_MA60_DEVIATION = 0.25
OVERHEAT_VOL_RATIO = 2.0
OVERHEAT_ATR_MULT = 1.5
OVERHEAT_3DAY_GAIN = 0.15
OVERHEAT_PROFIT_PCT = 0.20

OVERHEAT_SELL_PCT = 0.30            # 首次止盈卖出30%
OVERHEAT_SELL_PCT2 = 0.30           # 二次止盈卖出30%
OVERHEAT_MIN_CONDITIONS = 2         # 最少满足条件数

# ========== 大盘状态折扣（用于评分调整，已保留但不再用于事件） ==========
STATE_DISCOUNT = {
    "强牛": 1.0,
    "弱牛": 0.9,
    "震荡": 0.8,
    "弱熊": 0.7,
    "强熊": 0.6
}

# 强牛时补仓回撤阈值降低
BULL_MARKET_DIP_THRESHOLD = 0.05

# ========== AI 配置 ==========
AI_CACHE_TTL = 300
AI_ENABLE = True

# ========== 显示宽度 ==========
DISPLAY_NAME_WIDTH = 22
DISPLAY_CODE_WIDTH = 12
DISPLAY_PRICE_WIDTH = 10
DISPLAY_CHANGE_WIDTH = 14
DISPLAY_SCORE_WIDTH = 10
DISPLAY_TAGS_WIDTH = 60
DISPLAY_NUMS_WIDTH = 8

# ========== TMSV 参数（保留，供 analyzer 使用） ==========
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
TMSV_MA20_WINDOW = 20
TMSV_MA60_WINDOW = 60
TMSV_ATR_WINDOW = 14
TMSV_VOL_MA_WINDOW = 20
TMSV_PRICE_DIVISOR = 0.1
TMSV_SLOPE_SCALE = 10.0
TMSV_RSI_SCALE = 3.33
TMSV_MACD_DIFF_EPS = 0.001
TMSV_MACD_CHANGE_SCALE = 100.0
TMSV_VOL_RATIO_BASE = 0.8
TMSV_VOL_RATIO_DIVISOR = 1.2
TMSV_VOL_CONSIST_SCORE = 100.0


BUY_WEIGHTS_BULL = {
    "price_above_ma20": 0.18,
    "volume_above_ma5": 0.10,
    "macd_golden_cross": 0.10,
    "kdj_golden_cross": 0.08,
    "bollinger_break_up": 0.06,
    "williams_oversold": 0.04,
    "market_above_ma20": 0.08,
    "market_above_ma60": 0.10,
    "market_amount_above_ma20": 0.06,
    "outperform_market": 0.06,
    "weekly_above_ma20": 0.10,
    "tmsv_score": 0.18,
    "rsi_oversold": 0.02,
    "reversal_potential": 0.04,
}
BUY_WEIGHTS_RANGE = {
    "price_above_ma20": 0.12,
    "volume_above_ma5": 0.08,
    "macd_golden_cross": 0.08,
    "kdj_golden_cross": 0.06,
    "bollinger_break_up": 0.04,
    "williams_oversold": 0.12,
    "market_above_ma20": 0.06,
    "market_above_ma60": 0.06,
    "market_amount_above_ma20": 0.04,
    "outperform_market": 0.04,
    "weekly_above_ma20": 0.08,
    "tmsv_score": 0.12,
    "rsi_oversold": 0.10,
    "reversal_potential": 0.12,
}
BUY_WEIGHTS_BEAR = {
    "price_above_ma20": 0.10,
    "volume_above_ma5": 0.04,
    "macd_golden_cross": 0.04,
    "kdj_golden_cross": 0.04,
    "bollinger_break_up": 0.02,
    "williams_oversold": 0.08,
    "market_above_ma20": 0.04,
    "market_above_ma60": 0.04,
    "market_amount_above_ma20": 0.02,
    "outperform_market": 0.02,
    "weekly_above_ma20": 0.04,
    "tmsv_score": 0.08,
    "rsi_oversold": 0.20,
    "reversal_potential": 0.24,
}

SELL_WEIGHTS_BULL = {
    "price_below_ma20": 0.30,
    "bollinger_break_down": 0.15,
    "williams_overbought": 0.15,
    "rsi_overbought": 0.20,
    "underperform_market": 0.10,
    "stop_loss_ma_break": 0.05,
    "weekly_below_ma20": 0.10,
    "downside_momentum": 0.05,
    "max_drawdown_stop": 0.00,
}
SELL_WEIGHTS_RANGE = {
    "price_below_ma20": 0.30,
    "bollinger_break_down": 0.15,
    "williams_overbought": 0.12,
    "rsi_overbought": 0.12,
    "underperform_market": 0.10,
    "stop_loss_ma_break": 0.08,
    "weekly_below_ma20": 0.10,
    "downside_momentum": 0.08,
    "max_drawdown_stop": 0.05,
}
SELL_WEIGHTS_BEAR = {
    "price_below_ma20": 0.30,
    "bollinger_break_down": 0.15,
    "williams_overbought": 0.05,
    "rsi_overbought": 0.05,
    "underperform_market": 0.15,
    "stop_loss_ma_break": 0.15,
    "weekly_below_ma20": 0.15,
    "downside_momentum": 0.10,
    "max_drawdown_stop": 0.10,
}