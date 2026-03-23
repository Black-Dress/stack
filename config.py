# config.py
# 全局参数与指标权重

# ==================== 基础参数 ====================
ETF_MA = 20                      # ETF价格均线周期
ETF_VOL_MA = 5                    # ETF成交量均线周期
MACRO_INDEX = "sh.000300"         # 宏观指数（沪深300）
MARKET_INDEX = "sh.000001"        # 大盘指数（上证）
MACRO_MA_SHORT = 20                # 宏观指数短期均线
MACRO_MA_LONG = 60                  # 宏观指数长期均线
RSI_PERIOD = 14                     # RSI周期（用于情绪判断）

# ==================== 策略权重 ====================
STRATEGY_WEIGHTS = {
    # ETF自身指标
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
    # 大盘相关指标
    "market_above_ma20": 0.10,
    "market_above_ma60": 0.10,
    "market_amount_above_ma20": 0.10,
    "outperform_market": 0.20,
    "underperform_market": -0.20,
    # 止盈止损条件
    "stop_loss_ma_break": -1.00,
    "trailing_stop_clear": -1.00,
    "trailing_stop_half": -0.50,
    "profit_target_hit": -0.30,
    # 周线指标
    "weekly_above_ma20": 0.20,
    "weekly_below_ma20": -0.20,
}

# ==================== 止盈止损参数 ====================
STOP_LOSS_MA = 20
PROFIT_TARGETS = [(0.20, 0.3), (0.40, 0.3)]  # 固定止盈 (涨幅阈值, 卖出比例)

# ATR动态止损参数
ATR_PERIOD = 14
ATR_STOP_MULT = 2.0      # 清仓倍数（回撤超过 2*ATR 清仓）
ATR_TRAILING_MULT = 1.0  # 减半仓倍数（回撤超过 1*ATR 减半）

# ==================== 信号确认参数 ====================
CONFIRM_DAYS = 3
BUY_THRESHOLD = 0.5
SELL_THRESHOLD = -0.2
QUICK_BUY_THRESHOLD = 0.6   # 快速买入阈值

# ==================== 大盘状态映射 ====================
MARKET_STATES = {
    "bull": 1.2,
    "oscillate": 1.0,
    "bear": 0.8,
}

# ==================== 近期高低点窗口 ====================
RECENT_HIGH_WINDOW = 10
RECENT_LOW_WINDOW = 20

# ==================== 周线参数 ====================
WEEKLY_MA = 20
WEEKLY_WEIGHT = 0.20

# ==================== 风险提示参数 ====================
RISK_WARNING_DAYS = 3
RISK_WARNING_THRESHOLD = -0.1

# ==================== 文件路径 ====================
POSITION_FILE = "positions.csv"
STATE_FILE = "etf_state.json"