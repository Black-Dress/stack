#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块：存放所有固定参数、默认权重、默认参数及邮件配置。
"""
import os

# 项目根目录（etf_project）
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

# ---------------------------- 文件路径 ----------------------------
POSITION_FILE = "positions.csv"
STATE_FILE = "etf_state.json"
CACHE_FILE = "weight_cache.json"

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

# ---------------------------- 情绪调节参数 ----------------------------
SENT_ALPHA = 0.8
SENT_BETA = 2.5
SENT_GAMMA = 1.2


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