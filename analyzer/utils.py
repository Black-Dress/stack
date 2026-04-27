#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具模块：显示宽度、字符串填充、数学辅助、邮件发送、动态窗口等。
"""
import unicodedata
import logging
import math
from typing import List, Dict, Optional, Tuple

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
    from .config import SENTIMENT_LOWER_BOUND

    x = sentiment - 1.0
    if x >= 0:
        adj = 1.0 + 1.2 * math.tanh(3.0 * x) * math.exp(-0.8 * x)
    else:
        adj = 1.0 + 1.2 * math.tanh(3.0 * x)
    return max(SENTIMENT_LOWER_BOUND, min(1.5, adj))


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


# ========================== 市场状态回退规则 ==========================
def fallback_market_state(above_ma20: bool, above_ma60: bool) -> Tuple[str, float]:
    """简单的市场状态判断（无 AI 时使用）"""
    if above_ma20 and above_ma60:
        return "正常牛市", 1.2
    if not above_ma20 and not above_ma60:
        return "熊市下跌", 0.8
    return "震荡偏弱", 1.0


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


# 输出格式化函数（保留）
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
