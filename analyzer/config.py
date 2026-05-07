#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""全局配置、默认权重表（按市场状态）及技术参数"""
import os

# ---------------------------- 路径 ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
POSITION_FILE = os.path.join(DATA_DIR, "positions.csv")
STATE_FILE = os.path.join(DATA_DIR, "etf_state.json")

# ---------------------------- ETF 技术参数 ----------------------------
ETF_MA = 20
ETF_VOL_MA = 5
MARKET_INDEX = "sh.000001"
MACRO_INDEX = "sh.000300"
MACRO_MA_LONG = 60
ATR_PERIOD = 14
HISTORY_DAYS = 200
MAX_WORKERS = 5

# ---------------------------- 技术指标窗长 ----------------------------
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
KDJ_N = 9
BOLL_WINDOW = 20
BOLL_STD_MULT = 2
WILLIAMS_WINDOW = 14
RSI_WINDOW = 14
MA30_WINDOW = 30

# ---------------------------- 评分合成 ----------------------------
NONLINEAR_SCALE_BULL = 2.5          # 趋势市场缩放
NONLINEAR_SCALE_RANGE = 1.5         # 震荡市场缩放

# ---------------------------- 信号确认 ----------------------------
DEFAULT_CONFIRM_DAYS = 2
BUY_THRESHOLD = 50                  # 百分制买入阈值
SELL_THRESHOLD = -20                # 卖出阈值
QUICK_BUY_THRESHOLD = 60

ACTION_LEVEL_THRESHOLDS = [80, 70, 60, 40, 20, 0, -20, -40, -60, -80]
ACTION_LEVEL_NAMES = [
    "极度看好", "强烈买入", "买入", "谨慎买入", "偏多持有",
    "中性偏多", "中性偏空", "偏空持有", "谨慎卖出", "卖出"
]

# ---------------------------- 风险提示参数 ----------------------------
RISK_WARNING_DAYS = 3
RISK_WARNING_THRESHOLD = 30         # 已修正，百分制低分阈值
MA30_WEAKNESS_PENALTY = 0.9

# ---------------------------- 动态风险（ATR 倍数） ----------------------------
ATR_STOP_MULT = 2.0                 # 硬止损
ATR_TRAILING_PROFIT_MULT = 1.5      # 移动止盈
RISK_ALERT_DISTANCE_ATR = 0.5       # 提醒距离

# ---------------------------- 基于成本的止盈止损（硬规则） ----------------------------
COST_TAKE_PROFIT_CLEAR = 0.20       # 浮动盈利≥20% 且移动止盈触发 → 清仓卖出
COST_TAKE_PROFIT_HALF = 0.15        # 浮动盈利≥15% 且移动止盈触发 → 半仓卖出（可配置）
COST_STOP_LOSS_PCT = -0.08          # 浮动亏损≤-8% → 无条件止损卖出
USE_COST_BASED_OVERRIDE = True      # 是否启用成本价覆盖规则
COST_HALF_PROFIT_ACTION = "HOLD"    # 半仓止盈时的动作："SELL" 或 "HOLD"

# ---------------------------- 显示宽度 ----------------------------
DISPLAY_NAME_WIDTH = 16
DISPLAY_CODE_WIDTH = 12
DISPLAY_PRICE_WIDTH = 10
DISPLAY_CHANGE_WIDTH = 10
DISPLAY_SCORE_WIDTH = 8
DISPLAY_ACTION_WIDTH = 22

# ---------------------------- 市场状态权重表 ----------------------------
# 买入权重（牛市 / 震荡 / 熊市）
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

# 卖出权重（牛市 / 震荡 / 熊市）
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

BUY_FACTOR_NAMES = list(BUY_WEIGHTS_BULL.keys())
SELL_FACTOR_NAMES = list(SELL_WEIGHTS_BULL.keys())

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

# ---------------------------- 趋势扫描参数 ----------------------------
TREND_BUY_MAX_COUNT = 3
TREND_BUY_LOW_PROFIT_MIN = 5.0
TREND_BUY_LOW_PROFIT_MAX = 15.0
TREND_BUY_MAX_PULLBACK = 5.0
TREND_BUY_DAILY_GAIN_MIN = 0.5
TREND_BUY_DAILY_GAIN_MAX = 6.0
TREND_BUY_PREFER_SIGNAL = True

TREND_SELL_MAX_COUNT = 5
TREND_SELL_MIN_DAILY_LOSS = -3.0
TREND_SELL_MIN_PULLBACK = 6.0
TREND_SELL_MIN_LOW_PROFIT = 18.0
TREND_SELL_INCLUDE_WEAK_MA = True
TREND_SELL_INCLUDE_CLEAR_STOP = True

# ---------------------------- 邮件配置 ----------------------------
def get_email_config():
    return {
        "smtp_server": os.getenv("SMTP_SERVER", "smtp.qq.com"),
        "smtp_port": int(os.getenv("SMTP_PORT", "587")),
        "sender_email": os.getenv("SENDER_EMAIL", ""),
        "sender_password": os.getenv("SENDER_PASSWORD", ""),
        "receiver_email": os.getenv("RECEIVER_EMAIL", ""),
        "send_email": os.getenv("SEND_EMAIL", "false").lower() == "true",
    }