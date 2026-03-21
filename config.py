# config.py
# 全局参数与指标权重

ETF_MA = 20
ETF_VOL_MA = 5
MACRO_INDEX = "sh.000300"
MARKET_INDEX = "sh.000001"
MACRO_MA_SHORT = 20
MACRO_MA_LONG = 60
RSI_PERIOD = 14

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
    "weekly_above_ma20": 0.20,   # 周线站上20周线
    "weekly_below_ma20": -0.20,   # 周线跌破20周线
}

STOP_LOSS_MA = 20
PROFIT_TARGETS = [(0.20, 0.3), (0.40, 0.3)]  # 固定止盈

# ATR参数
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0      # 清仓倍数
ATR_TRAILING_MULT = 1.0  # 减半仓倍数

CONFIRM_DAYS = 3
BUY_THRESHOLD = 0.5
SELL_THRESHOLD = -0.2
QUICK_BUY_THRESHOLD = 0.6   # 快速买入阈值

MARKET_STATES = {
    "bull": 1.2,
    "oscillate": 1.0,
    "bear": 0.8,
}

RECENT_HIGH_WINDOW = 10
RECENT_LOW_WINDOW = 20

# 周线参数
WEEKLY_MA = 20
WEEKLY_WEIGHT = 0.20

# 风险提示
RISK_WARNING_DAYS = 3
RISK_WARNING_THRESHOLD = -0.1

POSITION_FILE = "positions.csv"
STATE_FILE = "etf_state.json"