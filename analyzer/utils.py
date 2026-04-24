#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具模块：显示宽度计算、字符串填充、离散化、权重验证、邮件发送、数学辅助函数。
"""
import unicodedata
import logging
import math
from typing import List

logger = logging.getLogger(__name__)


def get_display_width(text):
    """计算字符串在等宽字体下的显示宽度（中文占2，英文占1）"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))


def pad_display(text, width, align="left"):
    """
    将字符串补足到指定的显示宽度

    Args:
        text: 原始字符串
        width: 目标显示宽度
        align: 对齐方式（left/right/center）

    Returns:
        补足空格后的字符串
    """
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
    """
    根据给定的区间边界将值离散化为索引

    Args:
        value: 待离散化的值
        bins: 区间边界（递增）

    Returns:
        所属区间的索引
    """
    for i, thresh in enumerate(bins):
        if value < thresh:
            return i
    return len(bins)


def validate_and_filter_weights(weights: dict, expected_keys: List[str], name: str):
    """
    验证并归一化权重字典：仅保留期望的键，补全缺失的键，归一化到总和为1

    Args:
        weights: 输入的权重字典
        expected_keys: 允许的因子名列表
        name: 用于日志的权重名称

    Returns:
        归一化后的权重字典，若输入无效则返回 None
    """
    if not isinstance(weights, dict):
        return None
    filtered = {k: max(0.0, min(1.0, weights.get(k, 0.0))) for k in expected_keys}
    total = sum(filtered.values())
    if total == 0:
        filtered = {k: 1.0 / len(filtered) for k in filtered}
    elif abs(total - 1.0) > 1e-6:
        filtered = {k: v / total for k, v in filtered.items()}
    return filtered


# ---------------------------- 数学辅助函数 ----------------------------
def sigmoid_normalize(x: float, center: float = 0.0, steepness: float = 5.0) -> float:
    """Sigmoid 归一化，将任意实数映射到 [0,1]"""
    return 1.0 / (1.0 + math.exp(-steepness * (x - center)))


def nonlinear_score_transform(
    raw: float, market_status: str, bull_scale: float = 2.5, range_scale: float = 1.5
) -> float:
    """根据市场状态选择缩放因子，对原始净分进行 tanh 非线性变换"""
    status_lower = market_status.lower()
    scale = (
        bull_scale if ("牛" in status_lower or "熊" in status_lower) else range_scale
    )
    return math.tanh(scale * raw)


def apply_sentiment_adjustment(sentiment: float) -> float:
    """
    情绪因子非线性调整，增强极端区域区分度

    Args:
        sentiment: 原始情绪值（1.0为中性）

    Returns:
        调整后的情绪值
    """
    x = sentiment - 1.0
    if x >= 0:
        adj = 1.0 + 1.2 * math.tanh(3.0 * x) * math.exp(-0.8 * x)
    else:
        adj = 1.0 + 1.2 * math.tanh(3.0 * x)
    return max(0.6, min(1.5, adj))


def clip_env_factor(market_factor: float, sentiment_factor: float) -> float:
    """
    环境因子非线性映射：将市场因子与情绪因子的乘积压缩到 [0.6, 1.3] 区间

    Args:
        market_factor: 市场因子
        sentiment_factor: 情绪因子

    Returns:
        映射后的环境因子
    """
    raw = market_factor * sentiment_factor
    center = 1.0
    scale = 2.0  # 控制陡峭程度，值越大越接近硬裁剪
    # tanh 压缩到 (-0.35, 0.35) 加上基准0.95，得到约 (0.60, 1.30)
    mapped = 0.95 + 0.35 * math.tanh(scale * (raw - center))
    # 最终保障边界
    return max(0.60, min(1.30, mapped))


# ---------------------------- 邮件发送 ----------------------------
def send_email(subject: str, body: str) -> bool:
    """
    使用 SMTP 发送邮件通知

    Args:
        subject: 邮件主题
        body: 邮件正文

    Returns:
        是否发送成功
    """
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.header import Header
    from .config import get_email_config
    import logging
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
