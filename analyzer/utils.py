#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具模块：显示宽度计算、字符串填充、离散化、权重验证、邮件发送等。
"""
import unicodedata
import logging
from typing import List

logger = logging.getLogger(__name__)

def get_display_width(text):
    """计算字符串显示宽度（中文占2，英文占1）"""
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in str(text))

def pad_display(text, width, align="left"):
    """按显示宽度填充字符串"""
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
    """将连续值离散化为区间索引"""
    for i, thresh in enumerate(bins):
        if value < thresh:
            return i
    return len(bins)

def validate_and_filter_weights(weights: dict, expected_keys: List[str], name: str):
    """验证并过滤权重字典，确保总和为1，所有键都在预期内"""
    if not isinstance(weights, dict):
        return None
    filtered = {k: max(0.0, min(1.0, weights.get(k, 0.0))) for k in expected_keys}
    total = sum(filtered.values())
    if total == 0:
        filtered = {k: 1.0 / len(filtered) for k in filtered}
    elif abs(total - 1.0) > 1e-6:
        filtered = {k: v / total for k, v in filtered.items()}
    return filtered

# ---------------------------- 邮件发送（依赖 config 模块） ----------------------------
def send_email(subject: str, body: str) -> bool:
    """发送邮件（配置从 config.get_email_config() 读取）"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from email.header import Header
    from .config import get_email_config   # 注意相对导入
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