#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
配置模块：存放所有固定参数、默认权重、默认参数及邮件配置。
"""
import os

# ---------------------------- 路径配置 ----------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
POSITION_FILE = os.path.join(DATA_DIR, "positions.csv")      # 持仓列表
STATE_FILE = os.path.join(DATA_DIR, "etf_state.json")        # 各 ETF 状态
CACHE_FILE = os.path.join(DATA_DIR, "weight_cache.json")     # 环境缓存

# ---------------------------- ETF 技术参数 ----------------------------
ETF_MA = 20                     # ETF 短期均线周期
ETF_VOL_MA = 5                  # ETF 成交量均线周期
MACRO_INDEX = "sh.000300"       # 宏观指数（沪深300）
MARKET_INDEX = "sh.000001"      # 大盘指数（上证综指）
MACRO_MA_SHORT = 20             # 宏观指数短期均线
MACRO_MA_LONG = 60              # 宏观指数长期均线
RSI_PERIOD = 14                 # RSI 周期
ATR_PERIOD = 14                 # ATR 周期
ATR_STOP_MULT = 2.0             # 移动止损倍数
ATR_TRAILING_MULT = 1.0         # 移动止盈倍数
PROFIT_TARGET = 0.15            # 止盈目标比例
WEEKLY_MA = 20                  # 周线短期均线
RISK_WARNING_DAYS = 3           # 连续风险提示天数阈值
RISK_WARNING_THRESHOLD = -0.1   # 风险提示评分阈值

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
    "CONFIRM_DAYS": 3,                  # 信号确认所需连续天数
    "BUY_THRESHOLD": 0.5,               # 买入评分阈值
    "SELL_THRESHOLD": -0.2,             # 卖出评分阈值
    "QUICK_BUY_THRESHOLD": 0.6,         # 快速买入阈值
    "RECENT_HIGH_WINDOW": 10,           # 近期高点计算窗口
    "RECENT_LOW_WINDOW": 20,            # 近期低点计算窗口
}

# ---------------------------- 技术指标参数 ----------------------------
MACD_FAST = 12                      # MACD 快线周期
MACD_SLOW = 26                      # MACD 慢线周期
MACD_SIGNAL = 9                     # MACD 信号线周期
KDJ_N = 9                           # KDJ 参数 N
BOLL_WINDOW = 20                    # 布林带窗口
BOLL_STD_MULT = 2                   # 布林带标准差倍数
WILLIAMS_WINDOW = 14                # 威廉指标窗口
RSI_WINDOW = 14                     # RSI 窗口
TMSV_MA20_WINDOW = 20               # TMSV 中20日均线窗口
TMSV_MA60_WINDOW = 60               # TMSV 中60日均线窗口
TMSV_ATR_WINDOW = 14                # TMSV 中ATR窗口
TMSV_VOL_MA_WINDOW = 20             # TMSV 中成交量均线窗口

# ---------------------------- 动量缺失惩罚 ----------------------------
MOMENTUM_MISSING_PENALTY = 0.95     # 无金叉时买入得分惩罚系数

# ---------------------------- 非线性缩放 ----------------------------
NONLINEAR_SCALE_BULL = 2.5          # 牛市/熊市的 tanh 缩放因子
NONLINEAR_SCALE_RANGE = 1.5         # 震荡市的 tanh 缩放因子

# ---------------------------- Sigmoid 参数 ----------------------------
SIGMOID_STEEPNESS_DEFAULT = 5.0     # sigmoid 函数陡峭度（默认）
SIGMOID_STEEPNESS_VOLUME = 3.0      # 成交量因子 sigmoid 陡峭度

# ---------------------------- 硬止损阈值 ----------------------------
HARD_STOP_DRAWDOWN = 0.08           # 硬止损最大回撤比例
HARD_STOP_MA_BREAK_PCT = 0.05       # 跌破均线百分比（硬止损）

# ---------------------------- 情绪过热惩罚 ----------------------------
SENTIMENT_OVERHEAT_THRESHOLD = 1.25 # 情绪过热阈值
SENTIMENT_PENALTY_FACTOR = 0.8      # 情绪过热时买入得分惩罚因子

# ---------------------------- 缓存有效期 ----------------------------
CACHE_EXPIRE_SECONDS = 600          # 指标缓存有效期（秒）

# ---------------------------- 因子计算通用参数 ----------------------------
PRICE_DEVIATION_MA_MULT = 0.1                   # 价格偏离均线归一化乘数
VOLUME_RATIO_CENTER = 0.2                       # 量比 sigmoid 中心点
OUTPERFORM_MARKET_DIV = 0.05                    # 跑赢市场归一化除数
WILLIAMS_OVERBOUGHT_THRESH = -20                # 威廉超买阈值
WILLIAMS_OVERSOLD_THRESH = -80                  # 威廉超卖阈值
RSI_OVERBOUGHT_THRESH = 70                      # RSI 超买阈值
RSI_OVERBOUGHT_DIV = 30                         # RSI 超买归一化除数
PROFIT_TARGET_DIV = PROFIT_TARGET               # 止盈目标除数（引用前面）
MAX_DRAWDOWN_STOP_DIV = 0.08                    # 最大回撤止损触发除数
WILLIAMS_NORMALIZE_DIV = 20                     # 威廉指标归一化除数

# ---------------------------- TMSV 动态权重与参数 ----------------------------
TMSV_HIGH_VOL_THRESH = 0.03         # 高波动率阈值（ATR/价格）
TMSV_TREND_REDUCE = 0.05            # 趋势权重削减量
TMSV_MIN_TREND_WEIGHT = 0.15        # 趋势权重下限
TMSV_TREND_MA20_WEIGHT = 0.5        # 趋势子项：MA20 权重
TMSV_TREND_MA60_WEIGHT = 0.3        # 趋势子项：MA60 权重
TMSV_TREND_SLOPE_WEIGHT = 0.2       # 趋势子项：均线斜率权重
TMSV_MOM_RSI_WEIGHT = 0.6           # 动量子项：RSI 权重
TMSV_MOM_MACD_WEIGHT = 0.4          # 动量子项：MACD 权重
TMSV_VOL_RATIO_WEIGHT = 0.7         # 成交量子项：量比权重
TMSV_VOL_CONSIST_WEIGHT = 0.3       # 成交量子项：价量一致性权重
TMSV_VOL_LOW_THRESH = 0.01          # 低波动率阈值
TMSV_VOL_HIGH_THRESH = 0.03         # 高波动率阈值
TMSV_VOL_LOW_FACTOR = 1.5           # 低波动时得分放大因子
TMSV_VOL_HIGH_FACTOR = 0.6          # 高波动时得分衰减因子
TMSV_VOL_MID_FACTOR_BASE = 1.2      # 中等波动基准因子
TMSV_VOL_MID_FACTOR_SLOPE = 0.6     # 中等波动因子斜率
TMSV_VOL_BAND_WIDTH = 0.02          # 波动率区间宽度

# ---------------------------- 信号确认参数 ----------------------------
SIGNAL_SLOPE_BUY_THRESH = 0.05      # 买入信号斜率阈值
SIGNAL_AVG_OFFSET = 0.1             # 评分与均线的偏移要求
SIGNAL_SLOPE_WEAK = 0.02            # 弱买入斜率阈值
SIGNAL_SELL_SLOPE = -0.05           # 卖出信号斜率阈值
SIGNAL_SELL_WEAK_SLOPE = -0.02      # 弱卖出斜率阈值
SIGNAL_HIGH_VOL_BUY_SLOPE = 0.1     # 高波动买入斜率
SIGNAL_HIGH_VOL_DAYS = 4            # 高波动买入连续上涨天数要求
SIGNAL_MID_VOL_BUY_SLOPE = 0.08     # 中波动买入斜率
SIGNAL_MID_VOL_DAYS = 3             # 中波动买入连续上涨天数要求
MIN_CONFIRM_DAYS = 2                # 最小确认天数
MAX_CONFIRM_DAYS = 5                # 最大确认天数

# ---------------------------- 动态确认天数波动率阈值 ----------------------------
VOL_HIGH_CONFIRM = 0.04             # 高波动阈值
VOL_MID_CONFIRM = 0.025             # 中波动阈值

# ---------------------------- 评分等级阈值 ----------------------------
ACTION_LEVEL_THRESHOLDS = [0.8, 0.7, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8]
ACTION_LEVEL_NAMES = [
    "极度看好", "强烈买入", "买入", "谨慎买入", "偏多持有",
    "中性偏多", "中性偏空", "偏空持有", "谨慎卖出", "卖出"
]

# ---------------------------- 参数动态调整 ----------------------------
ADJUST_MULT_BASE = 1.2              # 调整乘数基准
ADJUST_BUY_DELTA_MAX = 0.03         # 买入阈值最大调整幅度
ADJUST_SELL_DELTA_MAX = 0.03        # 卖出阈值最大调整幅度


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