#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用工具模块：显示宽度、字符串填充、数学辅助、邮件发送、动态窗口等。
深度融合后新增技术指标计算函数（calc_rsi, calc_macd, calculate_atr, calculate_adx）
以及详细报告格式化功能。
"""
import unicodedata
import logging
import math
from typing import List, Dict, Optional, Tuple

import pandas as pd
import numpy as np

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


# ========================== 技术指标计算函数 ==========================
def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算 RSI 指标"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


def calc_macd(series, fast=12, slow=26, signal=9):
    """计算 MACD 指标，返回 dif, dea, hist"""
    exp_fast = series.ewm(span=fast, adjust=False).mean()
    exp_slow = series.ewm(span=slow, adjust=False).mean()
    dif = exp_fast - exp_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


def calculate_atr(df, period=14) -> pd.Series:
    """计算 ATR 指标"""
    tr = pd.concat(
        [
            df["high"] - df["low"],
            abs(df["high"] - df["close"].shift()),
            abs(df["low"] - df["close"].shift()),
        ],
        axis=1,
    ).max(1)
    return tr.rolling(period).mean()


def calculate_adx(df, period=14) -> pd.DataFrame:
    """计算 ADX 指标，返回 plus_di, minus_di, adx 的 DataFrame"""
    high, low, close = df["high"], df["low"], df["close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    tr = calculate_atr(df, 1)
    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.rolling(period).mean()
    return pd.DataFrame(
        {"plus_di": plus_di, "minus_di": minus_di, "adx": adx}, index=df.index
    )


# ========================== 详细报告格式化 ==========================
def format_detailed_report(ctx, market, params, buy_weights, sell_weights,
                           action_level, ai_comment, ai_tp) -> str:
    """
    根据 ETFContext 生成完整的详细分析报告字符串。
    """
    import datetime
    from .config import DETAIL_COL_NAME, DETAIL_COL_STRENGTH, DETAIL_COL_WEIGHT, DETAIL_COL_CONTRIB
    from .utils import clip_env_factor, pad_display, get_display_width  # 避免循环引用

    final = ctx.final_score

    lines = ["=" * 70,
             f"ETF详细分析报告 - {ctx.name} ({ctx.code})",
             f"分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             "=" * 70,
             f"实时价格：{ctx.real_price:.3f}",
             f"涨跌幅：{ctx.change_pct:+.2f}%",
             f"市场状态：{market['macro_status']}，市场因子：{market['market_factor']:.2f}，情绪因子：{market['sentiment_factor']:.2f}"]
    if market.get("sentiment_risk_tip"):
        lines.append(f"情绪风险提示：{market['sentiment_risk_tip']}")
    lines += [f"波动率(ATR%)：{ctx.atr_pct*100:.2f}%",
              f"TMSV复合强度：{ctx.tmsv:.1f} (强度系数 {ctx.tmsv_strength:.3f})",
              f"最大回撤：{ctx.max_drawdown_pct*100:.2f}%",
              f"中期均线（30日）：{'站上' if ctx.above_ma30 else '跌破'}", ""]

    if ctx.trailing_profit_level or ctx.profit_level:
        lines.append("【止盈观察 (仅供参考)】")
        if ctx.trailing_profit_level:
            recent_high = ctx.recent_high_price
            from_high_pct = (recent_high - ctx.real_price) / recent_high if recent_high > 0 else 0
            lines.append(f"  从{ctx.params['RECENT_HIGH_WINDOW']}日高点 {recent_high:.3f} 回落 {from_high_pct:.1%}")
            level_text = "清仓级" if ctx.trailing_profit_level == 'clear' else "半仓级"
            lines.append(f"  移动止盈信号：{level_text}")
        if ctx.profit_level:
            lines.append(f"  距{ctx.params['RECENT_LOW_WINDOW']}日低点 {ctx.recent_low_price:.3f} 涨幅 {ctx.profit_pct_from_low:.1%}")
            level_map = {'clear': '清仓级', 'half': '半仓级', 'watch': '关注级'}
            lines.append(f"  低点涨幅信号：{level_map.get(ctx.profit_level, '')}")
        lines.append("  *以上提示不构成自动卖出指令，请结合其他因素决策。")
        lines.append("")

    def row_line(items):
        return "".join([pad_display(items[0], DETAIL_COL_NAME),
                        pad_display(items[1], DETAIL_COL_STRENGTH, "right"),
                        pad_display(items[2], DETAIL_COL_WEIGHT, "right"),
                        pad_display(items[3], DETAIL_COL_CONTRIB, "right")])

    lines.append("【买入因子详情】")
    lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
    lines.append("-" * 50)
    buy_contribs = sorted([(k, ctx.buy_factors[k], buy_weights.get(k, 0), buy_weights.get(k, 0) * ctx.buy_factors[k])
                           for k in ctx.buy_factors], key=lambda x: x[3], reverse=True)
    for name_f, s, w, contrib in buy_contribs:
        lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
    lines.append(row_line(["买入总分（含中期过滤）", "", "", f"{ctx.buy_score:.3f}"]))
    lines.append("")

    lines.append("【卖出因子详情】")
    lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
    lines.append("-" * 50)
    sell_contribs = sorted([(k, ctx.sell_factors[k], sell_weights.get(k, 0), sell_weights.get(k, 0) * ctx.sell_factors[k])
                            for k in ctx.sell_factors], key=lambda x: x[3], reverse=True)
    for name_f, s, w, contrib in sell_contribs:
        lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
    lines.append(row_line(["卖出总分", "", "", f"{ctx.sell_score:.3f}"]))
    lines.append("")

    # 评分合成
    env_factor = clip_env_factor(market["market_factor"], market["sentiment_factor"])
    lines += ["【评分合成】",
              f"原始净分 = {ctx.buy_score:.3f} - {ctx.sell_score:.3f} = {ctx.raw_score:.3f}",
              f"非线性变换 × 环境因子 ({env_factor:.2f}) → {final:.3f}",
              f"操作等级：{action_level}"]

    if ai_comment is not None:
        lines += ["", "【AI 专业点评】"]
        lines.append(ai_comment)
    else:
        lines += ["", "【AI 专业点评】未配置 API_KEY，无法生成。"]

    if ai_tp is not None:
        lines += ["", "【AI 止盈建议】"]
        lines.append(ai_tp)
    else:
        lines += ["", "【AI 止盈建议】无需止盈建议。"]

    lines.append("=" * 70)
    return "\n".join(lines)