#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具模块：显示宽度计算、字符串填充、离散化、权重验证、邮件发送、数学辅助函数，
以及因子计算、动态窗口等可复用逻辑。
"""
import unicodedata
import logging
import math
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


# ========================== 显示 & 辅助 ==========================
def get_display_width(text):
    """计算字符串在等宽字体下的显示宽度（中文占2，英文占1）"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad_display(text, width, align="left"):
    """将字符串补足到指定的显示宽度"""
    text = str(text)
    cur = get_display_width(text)
    if cur >= width:
        return text
    pad = width - cur
    if align == "right":
        return " " * pad + text
    elif align == "center":
        left = pad // 2
        return " " * left + text + " " * (pad - left)
    return text + " " * pad


def discretize(value: float, bins: List[float]) -> int:
    """根据给定的区间边界将值离散化为索引"""
    for i, thresh in enumerate(bins):
        if value < thresh:
            return i
    return len(bins)


def validate_and_filter_weights(weights: dict, expected_keys: List[str], name: str):
    """验证并归一化权重字典：仅保留期望的键，补全缺失的键，归一化到总和为1"""
    if not isinstance(weights, dict):
        return None
    filtered = {k: max(0.0, min(1.0, weights.get(k, 0.0))) for k in expected_keys}
    total = sum(filtered.values())
    if total == 0:
        filtered = {k: 1.0 / len(filtered) for k in filtered}
    elif abs(total - 1.0) > 1e-6:
        filtered = {k: v / total for k, v in filtered.items()}
    return filtered


# ========================== 数学工具 ==========================
def sigmoid_normalize(x: float, center: float = 0.0, steepness: float = 5.0) -> float:
    """Sigmoid 归一化，将任意实数映射到 [0,1]"""
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


def nonlinear_score_transform(
    raw: float, market_status: str, bull_scale: float = 2.5, range_scale: float = 1.5
) -> float:
    """根据市场状态选择缩放因子，对原始净分进行 tanh 非线性变换"""
    status_lower = market_status.lower()
    scale = bull_scale if ("牛" in status_lower or "熊" in status_lower) else range_scale
    return math.tanh(scale * raw)


def apply_sentiment_adjustment(sentiment: float) -> float:
    """情绪因子非线性调整，增强极端区域区分度"""
    x = sentiment - 1.0
    if x >= 0:
        adj = 1.0 + 1.2 * math.tanh(3.0 * x) * math.exp(-0.8 * x)
    else:
        adj = 1.0 + 1.2 * math.tanh(3.0 * x)
    return max(0.6, min(1.5, adj))


def clip_env_factor(market_factor: float, sentiment_factor: float) -> float:
    """环境因子非线性映射：将市场因子与情绪因子的乘积压缩到 [0.6, 1.3] 区间"""
    raw = market_factor * sentiment_factor
    center = 1.0
    scale = 2.0
    mapped = 0.95 + 0.35 * math.tanh(scale * (raw - center))
    return max(0.60, min(1.30, mapped))


def cap(x: float) -> float:
    """将值限制在 [0, 1]"""
    return max(0.0, min(1.0, x))


def weighted_sum(factors: Dict[str, float], weights: Dict[str, float]) -> float:
    """计算因子加权得分"""
    return sum(weights.get(k, 0) * factors[k] for k in factors)


# ========================== 动态窗口工具 ==========================
def get_dynamic_history_days(volatility: float) -> int:
    """根据波动率动态调整用于判断趋势的评分历史窗口天数"""
    from .config import VOL_HIGH_CONFIRM, VOL_MID_CONFIRM
    if volatility > VOL_HIGH_CONFIRM:
        return 5
    if volatility > VOL_MID_CONFIRM:
        return 8
    return 20 if volatility <= 0.015 else 12


def get_dynamic_confirm_days(atr_pct: float, base_days: int) -> int:
    """根据波动率动态调整确认信号的连续天数"""
    from .config import VOL_HIGH_CONFIRM, VOL_MID_CONFIRM, MIN_CONFIRM_DAYS, MAX_CONFIRM_DAYS
    if atr_pct is None:
        return base_days
    if atr_pct > VOL_HIGH_CONFIRM:
        return max(MIN_CONFIRM_DAYS, base_days - 1)
    elif atr_pct > VOL_MID_CONFIRM:
        return base_days
    else:
        return min(MAX_CONFIRM_DAYS, base_days + 1)


# ========================== 因子强度计算函数 ==========================
# 以下函数均接受必要的标量参数，返回 [0, 1] 的因子强度

def factor_buy_price_above_ma20(price: float, ma20: float) -> float:
    from .config import PRICE_DEVIATION_MA_MULT
    if price <= ma20 or ma20 <= 0:
        return 0.0
    deviation = (price - ma20) / (ma20 * PRICE_DEVIATION_MA_MULT)
    return sigmoid_normalize(deviation, center=0.2)


def factor_buy_volume_above_ma5(volume: float, vol_ma: float) -> float:
    from .config import VOLUME_RATIO_CENTER, SIGMOID_STEEPNESS_VOLUME
    if volume <= vol_ma or vol_ma <= 0:
        return 0.0
    ratio = volume / vol_ma - 1.0
    return sigmoid_normalize(ratio, center=VOLUME_RATIO_CENTER, steepness=SIGMOID_STEEPNESS_VOLUME)


def factor_buy_bollinger_break_up(price: float, boll_up: float) -> float:
    if price <= boll_up:
        return 0.0
    return sigmoid_normalize((price - boll_up) / boll_up, center=0.01)


def factor_buy_williams_oversold(williams_r: float) -> float:
    from .config import WILLIAMS_OVERSOLD_THRESH, WILLIAMS_NORMALIZE_DIV
    if williams_r < WILLIAMS_OVERSOLD_THRESH:
        return 0.0
    return sigmoid_normalize(
        (WILLIAMS_OVERSOLD_THRESH - williams_r) / WILLIAMS_NORMALIZE_DIV, center=0.5
    )


def factor_buy_rsi_oversold(rsi: float) -> float:
    from .config import RSI_OVERSOLD_THRESH
    if rsi < RSI_OVERSOLD_THRESH:
        return max(0.0, (RSI_OVERSOLD_THRESH - rsi) / RSI_OVERSOLD_THRESH)
    return 0.0


def factor_buy_outperform_market(ret_etf_5d: float, ret_market_5d: float) -> float:
    from .config import OUTPERFORM_MARKET_DIV
    if ret_etf_5d > ret_market_5d:
        return sigmoid_normalize((ret_etf_5d - ret_market_5d) / OUTPERFORM_MARKET_DIV, center=0.2)
    return 0.0


def factor_sell_price_below_ma20(price: float, ma20: float) -> float:
    from .config import PRICE_DEVIATION_MA_MULT
    if price >= ma20 or ma20 <= 0:
        return 0.0
    deviation = (price - ma20) / (ma20 * PRICE_DEVIATION_MA_MULT)
    return sigmoid_normalize(-deviation, center=0.2)


def factor_sell_bollinger_break_down(price: float, boll_low: float) -> float:
    if price >= boll_low:
        return 0.0
    return sigmoid_normalize((boll_low - price) / boll_low, center=0.01)


def factor_sell_williams_overbought(williams_r: float) -> float:
    from .config import WILLIAMS_OVERBOUGHT_THRESH, WILLIAMS_NORMALIZE_DIV
    if williams_r >= WILLIAMS_OVERBOUGHT_THRESH:
        return 0.0
    return sigmoid_normalize(
        (WILLIAMS_OVERBOUGHT_THRESH - williams_r) / WILLIAMS_NORMALIZE_DIV, center=0.5
    )


def factor_sell_rsi_overbought(rsi: float) -> float:
    from .config import RSI_OVERBOUGHT_THRESH, RSI_OVERBOUGHT_DIV
    if rsi > RSI_OVERBOUGHT_THRESH:
        return sigmoid_normalize((rsi - RSI_OVERBOUGHT_THRESH) / RSI_OVERBOUGHT_DIV, center=0.2)
    return 0.0


def factor_sell_underperform_market(ret_etf_5d: float, ret_market_5d: float) -> float:
    from .config import OUTPERFORM_MARKET_DIV
    if ret_etf_5d < ret_market_5d:
        return sigmoid_normalize((ret_market_5d - ret_etf_5d) / OUTPERFORM_MARKET_DIV, center=0.2)
    return 0.0


def factor_sell_stop_loss_ma_break(price: float, ma20: float) -> float:
    from .config import HARD_STOP_MA_BREAK_PCT
    if price < ma20 and ma20 > 0:
        return cap((ma20 - price) / (ma20 * HARD_STOP_MA_BREAK_PCT))
    return 0.0


def factor_sell_trailing_stop_clear(price: float, recent_high: float, atr_pct: float) -> float:
    from .config import ATR_STOP_MULT
    if recent_high > 0 and atr_pct > 0 and (recent_high - price) / recent_high >= ATR_STOP_MULT * atr_pct:
        return cap((recent_high - price) / recent_high / (ATR_STOP_MULT * atr_pct))
    return 0.0


def factor_sell_trailing_stop_half(price: float, recent_high: float, atr_pct: float) -> float:
    from .config import ATR_TRAILING_MULT
    if recent_high > 0 and atr_pct > 0 and (recent_high - price) / recent_high >= ATR_TRAILING_MULT * atr_pct:
        return cap((recent_high - price) / recent_high / (ATR_TRAILING_MULT * atr_pct))
    return 0.0


def factor_sell_downside_momentum(downside: float) -> float:
    return cap(downside)


def factor_sell_max_drawdown_stop(max_drawdown_pct: float) -> float:
    from .config import MAX_DRAWDOWN_STOP_DIV
    if max_drawdown_pct >= MAX_DRAWDOWN_STOP_DIV:
        return cap(max_drawdown_pct / MAX_DRAWDOWN_STOP_DIV)
    return 0.0


# ========================== 邮件发送 ==========================
def send_email(subject: str, body: str) -> bool:
    """使用 SMTP 发送邮件通知"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.header import Header
    from .config import get_email_config
    logger = logging.getLogger(__name__)

    email_cfg = get_email_config()
    if not email_cfg["send_email"]:
        return False
    if not all([email_cfg["sender_email"], email_cfg["sender_password"], email_cfg["receiver_email"]]):
        logger.error("邮件配置不完整")
        return False
    try:
        msg = MIMEMultipart()
        msg["From"] = email_cfg["sender_email"]
        msg["To"] = email_cfg["receiver_email"]
        msg["Subject"] = Header(subject, "utf-8")
        msg.attach(MIMEText(body, "plain", "utf-8"))
        server = smtplib.SMTP(email_cfg["smtp_server"], email_cfg["smtp_port"])
        server.starttls()
        server.login(email_cfg["sender_email"], email_cfg["sender_password"])
        server.sendmail(email_cfg["sender_email"], [email_cfg["receiver_email"]], msg.as_string())
        server.quit()
        logger.info("邮件发送成功")
        return True
    except Exception as e:
        logger.error(f"邮件发送失败: {e}")
        return False


def get_trailing_profit_signals(
    price: float, recent_high: float, atr_pct: float
) -> Optional[str]:
    """
    返回移动止盈提醒级别：'clear'（建议清仓）、'half'（建议半仓止盈）或 None
    """
    clear_strength = factor_sell_trailing_stop_clear(price, recent_high, atr_pct)
    half_strength = factor_sell_trailing_stop_half(price, recent_high, atr_pct)
    if clear_strength > 0:
        return "clear"
    if half_strength > 0:
        return "half"
    return None


def format_etf_output_line(name, code, price, change_pct, final_score, action_level,
                           atr_pct=None, recent_high_price=None, risk_str="",
                           signal_action=None):
    """
    格式化单只 ETF 的分析输出行，返回字符串。
    止盈提示已整合在 risk_str 中，信号独立显示在行尾。
    """
    price_str = f"{price:.3f}" if price is not None else "N/A"
    change_str = f"{change_pct:+.2f}%" if change_pct is not None else "0.00%"
    final_str = f"{final_score:.2f}" if final_score is not None else "0.00"

    output = (f"{pad_display(name, 14)} {pad_display(code, 12)} "
              f"{pad_display(price_str, 10, 'right')} "
              f"{pad_display(change_str, 10, 'right')} "
              f"{pad_display(final_str, 16, 'right')}  "
              f"{pad_display(action_level, 22)}")

    parts = []
    if risk_str:
        parts.append(risk_str)
    if signal_action:
        parts.append(f"[{signal_action}]")
    if parts:
        output += "  " + " ".join(parts)

    return output
