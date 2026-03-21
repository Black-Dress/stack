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
# 正值为买入贡献，负值为卖出贡献
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
}

# ==================== 止盈止损参数 ====================
STOP_LOSS_MA = 20                    # 跌破该均线建议清仓
TRAILING_STOP_HALF = 0.05             # 从近期高点回撤5%建议减半仓
TRAILING_STOP_CLEAR = 0.08            # 从近期高点回撤8%建议清仓
PROFIT_TARGETS = [(0.20, 0.3), (0.40, 0.3)]  # (涨幅阈值, 卖出比例)

# ==================== 信号确认参数 ====================
CONFIRM_DAYS = 3
BUY_THRESHOLD = 0.5
SELL_THRESHOLD = -0.2

# ==================== 大盘状态映射 ====================
MARKET_STATES = {
    "bull": 1.2,
    "oscillate": 1.0,
    "bear": 0.8,
}

# ==================== 近期高低点窗口 ====================
RECENT_HIGH_WINDOW = 10
RECENT_LOW_WINDOW = 20

# ==================== 文件路径 ====================
POSITION_FILE = "positions.csv"
STATE_FILE = "etf_state.json"