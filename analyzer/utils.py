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
        adj = 1.0 + 0.8 * math.tanh(2.5 * x)  
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
def format_detailed_report(ctx, market, params, action_level, ai_comment, ai_tp) -> str:
    """
    生成结构紧凑、对齐美观的 ETF 详细分析报告。
    移除全局 vs 微调权重对比，因子表格只显示非零贡献项，整体高度精简。
    """
    import datetime
    from .utils import clip_env_factor, pad_display

    # ---------- 基础行情（一行式）----------
    lines = []
    lines.append("=" * 70)
    lines.append(f"  {ctx.name} ({ctx.code})  分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("-" * 70)
    lines.append(
        f"  价格 {ctx.real_price:>6.3f}  |  涨跌 {ctx.change_pct:>+6.2f}%  |  "
        f"ATR {ctx.atr_pct*100:>4.2f}%  |  RSI {ctx.rsi:>5.1f}  |  "
        f"TMSV {ctx.tmsv:>5.1f} (强度{ctx.tmsv_strength:.2f})  |  "
        f"中期均线(30日): {'站上' if ctx.above_ma30 else '跌破'}"
    )
    lines.append("")

    # ---------- 市场环境 ----------
    lines.append(f"  市场状态：{market['macro_status']}，市场因子 {market['market_factor']:.2f}，情绪因子 {market['sentiment_factor']:.2f}")
    if market.get("sentiment_risk_tip"):
        lines.append(f"  情绪提示：{market['sentiment_risk_tip']}")
    lines.append("")

    # ---------- 止盈观察（仅触发时显示）----------
    if ctx.trailing_profit_level or ctx.profit_level:
        parts = []
        if ctx.trailing_profit_level:
            fall = (ctx.recent_high_price - ctx.real_price) / ctx.recent_high_price if ctx.recent_high_price > 0 else 0
            parts.append(f"高点回落{fall:.1%} ({'清仓级' if ctx.trailing_profit_level == 'clear' else '半仓级'})")
        if ctx.profit_level:
            level_map = {'clear':'清仓级','half':'半仓级','watch':'关注级'}
            parts.append(f"低点涨{ctx.profit_pct_from_low:.1%} ({level_map.get(ctx.profit_level, '')})")
        lines.append("  ⚠ 止盈提示：" + " | ".join(parts))
        lines.append("")

    # ---------- 因子表格辅助函数 ----------
    def _render_factor_table(title, factors_dict, weights_dict, score, extra_note="", hide_zero_contrib=True):
        local = []
        local.append(f"  {title}")
        # 表头
        local.append(f"   {'因子名称':<24s} {'强度':>6s} {'权重':>6s} {'贡献':>7s}")
        local.append("   " + "-" * 46)
        items = []
        for k, strength in factors_dict.items():
            w = weights_dict.get(k, 0.0)
            contrib = strength * w
            # 如果隐藏零贡献，且贡献绝对值 < 0.001，跳过（但保留强度为0但权重非零？我们跳过零贡献）
            if hide_zero_contrib and abs(contrib) < 0.001:
                continue
            items.append((k, strength, w, contrib))
        # 按贡献降序
        items.sort(key=lambda x: abs(x[3]), reverse=True)
        for name, s, w, c in items:
            local.append(f"   {name:<24s} {s:>5.3f} {w:>6.3f} {c:>7.3f}")
        local.append("   " + "-" * 46)
        local.append(f"   {'总分（含过滤）':<24s} {'':>6} {'':>6} {score:>7.3f}")
        if extra_note:
            local.append(f"   ※ {extra_note}")
        local.append("")
        return local

    # 买入因子
    buy_note = ""
    if ctx.buy_factors.get("macd_golden_cross",0)==0 and ctx.buy_factors.get("kdj_golden_cross",0)==0:
        buy_note = "动量缺失惩罚已应用"
    lines.extend(_render_factor_table("【买入因子】", ctx.buy_factors, ctx.buy_weights_used, ctx.buy_score, buy_note))

    # 卖出因子
    lines.extend(_render_factor_table("【卖出因子】", ctx.sell_factors, ctx.sell_weights_used, ctx.sell_score))

    # ---------- 评分合成 ----------
    env = clip_env_factor(market["market_factor"], market["sentiment_factor"])
    lines.append(f"  【评分合成】")
    lines.append(f"  净分 = {ctx.buy_score:.3f} - {ctx.sell_score:.3f} = {ctx.raw_score:.3f}")
    lines.append(f"  非线性变换 × 环境因子({env:.3f}) → 最终得分 {ctx.final_score:.3f} → 操作等级：{action_level}")
    lines.append("")

    # ---------- AI 评论 / 建议 ----------
    if ai_comment:
        lines.append(f"  【AI 专业点评】")
        lines.append(f"  {ai_comment.strip()}")
        lines.append("")
    if ai_tp:
        lines.append(f"  【AI 止盈建议】")
        lines.append(f"  {ai_tp.strip()}")
        lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)




def post_process_weights(ai_weights: Dict[str, float],
                         global_weights: Dict[str, float],
                         min_weight: float = 0.02,
                         max_weight: float = 0.40,
                         min_active_factors: int = 5) -> Dict[str, float]:
    """
    后期融合优化权重：
    1. 将负值置 0
    2. 对全局权重中 ≥ min_weight 的因子，若 AI 权重 < min_weight，则强制设置回 min_weight
    3. 单个权重不超过 max_weight
    4. 归一化
    5. 若非零因子数少于 min_active_factors，则从全局权重中补充
    返回优化后的权重字典（非负，总和 1.0）
    """
    weights = {k: max(0.0, ai_weights.get(k, 0.0)) for k in global_weights}
    
    # 保留核心分散：原本在全局中不低于 min_weight 的因子，AI 不能完全清零
    for k, gw in global_weights.items():
        if gw >= min_weight and weights[k] < min_weight:
            weights[k] = min_weight
    
    # 上限约束
    for k in weights:
        if weights[k] > max_weight:
            weights[k] = max_weight
    
    # 归一化
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        weights = global_weights.copy()
        total = sum(weights.values())
        weights = {k: v / total for k, v in weights.items()}
    
    # 确保足够多的活跃因子
    active = [k for k, v in weights.items() if v > 0.001]
    if len(active) < min_active_factors:
        # 从全局权重中选取原权重最大的因子补足（排除已存在的）
        sorted_g = sorted(global_weights.items(), key=lambda x: x[1], reverse=True)
        added = 0
        for k, _ in sorted_g:
            if k not in active and weights.get(k, 0) < 0.001:
                weights[k] = min_weight
                active.append(k)
                added += 1
                if len(active) >= min_active_factors:
                    break
        # 重新归一化
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
    
    return weights