#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核心分析引擎：ETF 评分与信号生成，权重由环境决定，已纳入成本价止盈止损覆盖（止盈软提示）"""
import datetime
import logging
import numpy as np
import openai
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import json
import re

from .config import *
from .utils import (
    weighted_sum,
    nonlinear_score_transform,
    clip_env_factor,
    cap,
    pad_display,
    format_etf_output_line,
    format_detailed_report,
)
from .factors import (
    factor_buy_price_above_ma20,
    factor_buy_volume_above_ma5,
    factor_buy_bollinger_break_up,
    factor_buy_williams_oversold,
    factor_buy_rsi_oversold,
    factor_buy_outperform_market,
    factor_sell_price_below_ma20,
    factor_sell_bollinger_break_down,
    factor_sell_williams_overbought,
    factor_sell_rsi_overbought,
    factor_sell_underperform_market,
    factor_sell_stop_loss_ma_break,
    factor_sell_downside_momentum,
    factor_sell_max_drawdown_stop,
    get_trailing_profit_signals,
    compute_tmsv,
    factor_buy_reversal_potential,
)

logger = logging.getLogger(__name__)


@dataclass
class ETFContext:
    """单只 ETF 完整分析上下文"""
    code: str
    name: str
    real_price: Optional[float]
    hist_df: Optional[pd.DataFrame]
    weekly_df: Optional[pd.DataFrame]
    today: datetime.date
    market: Dict
    params: Dict
    cost_price: Optional[float] = None
    shares: int = 0                     # 持仓份额

    change_pct: float = 0.0
    atr_pct: float = 0.0
    tmsv: float = 50.0
    tmsv_strength: float = 0.5
    downside_momentum: float = 0.0
    max_drawdown_pct: float = 0.0
    weekly_above: bool = False
    weekly_below: bool = False
    above_ma30: bool = False
    is_weak_ma: bool = False
    buy_factors: Dict = field(default_factory=dict)
    sell_factors: Dict = field(default_factory=dict)
    buy_score: float = 0.0
    sell_score: float = 0.0
    raw_score: float = 0.0
    final_score: float = 0.0
    error: Optional[str] = None

    recent_low_price: float = 0.0
    profit_pct_from_low: float = 0.0
    should_take_profit: bool = False
    effective_profit_threshold: float = 0.15
    rsi: float = 50.0
    recent_high_price: float = 0.0
    trailing_profit_level: Optional[str] = None
    profit_level: Optional[str] = None
    cost_profit_pct: Optional[float] = None

    buy_weights_used: Dict = field(default_factory=dict)
    sell_weights_used: Dict = field(default_factory=dict)


class AIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    @staticmethod
    def _extract_json(content: Optional[str]) -> dict:
        if not content:
            raise ValueError("content is None")
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if not m:
            raise ValueError("未找到JSON")
        return json.loads(m.group())

    def comment_on_etf(self, code: str, name: str, final_score: float,
                       action_level: str, market_state: str, market_factor: float,
                       tmsv: float, atr_pct: float,
                       cost_price: Optional[float] = None,
                       cost_profit_pct: Optional[float] = None,
                       signal_action: Optional[str] = None,
                       risk_tags: str = "",
                       rsi: Optional[float] = None,
                       macd_status: str = "",
                       vol_ratio: Optional[float] = None) -> str:
        extra = []
        if cost_price and cost_price > 0:
            profit_str = f"{cost_profit_pct*100:+.2f}%" if cost_profit_pct is not None else "未知"
            extra.append(f"持仓成本：{cost_price:.3f}，浮动盈亏：{profit_str}")
        if signal_action:
            extra.append(f"系统信号：{signal_action}")
        if risk_tags:
            extra.append(f"风险标签：{risk_tags}")
        tech = []
        if rsi is not None:
            tech.append(f"RSI: {rsi:.1f}")
        if macd_status:
            tech.append(f"MACD: {macd_status}")
        if vol_ratio is not None:
            tech.append(f"成交量比: {vol_ratio:.2f}")
        if tech:
            extra.append("关键指标: " + " | ".join(tech))
        extra_info = "\n".join(extra)

        prompt = f"""你是资深ETF分析师，请基于以下数据给出80~120字点评，要求：
- 结合持仓盈亏、技术指标、风险标签具体分析，不可仅重复数值。
- 若存在风险标签（如近止盈、低位、弱于均线），请重点提示。
- 若系统信号为BUY或SELL，需解释原因。
ETF：{name}（{code}）
综合评分：{final_score:.1f}  等级：{action_level}
市场状态：{market_state}  环境因子：{market_factor:.2f}
TMSV复合强度：{tmsv:.1f}  ATR波动率：{atr_pct*100:.2f}%
{extra_info}
请直接在技术面、风险应对两方面给出具体点评。"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
                timeout=10,
            )
            content = resp.choices[0].message.content
            return content.strip() if content else "（AI评论生成失败）"
        except Exception as e:
            logger.error(f"AI评论生成失败: {e}")
            return "（AI评论生成失败）"

    def batch_comment_on_etfs(self, etf_list: List[Dict], batch_size=6) -> List[str]:
        results = [""] * len(etf_list)
        for i in range(0, len(etf_list), batch_size):
            batch = etf_list[i : i + batch_size]
            prompts = []
            for idx, e in enumerate(batch):
                prompts.append(
                    f"ETF {idx}: {e['name']}({e['code']}) 评分{e['final_score']:.1f} "
                    f"等级{e['action_level']} 市场{e['market_state']} TMSV{e['tmsv']:.1f} "
                    f"ATR{e['atr_pct']*100:.2f}%"
                )
            combined = (
                "为以下ETF分别生成80~120字专业点评，用JSON返回，键为序号字符串。\n"
                + "\n".join(prompts)
                + '\n输出格式：{"0":"...","1":"..."}'
            )
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": combined}],
                    max_tokens=300 * len(batch),
                    temperature=0.3,
                    timeout=20,
                )
                data = self._extract_json(resp.choices[0].message.content)
                for j in range(len(batch)):
                    results[i + j] = data.get(str(j), "（批量评论缺失）").strip()
            except Exception as e:
                logger.error(f"批量评论失败(批次{i}): {e}")
                for j in range(len(batch)):
                    results[i + j] = "（批量评论生成失败）"
        return results


class DataAnalyzer:
    def __init__(self):
        self.market_info = {}
        self.buy_weights = {}
        self.sell_weights = {}
        self.params = {
            "CONFIRM_DAYS": DEFAULT_CONFIRM_DAYS,
            "BUY_THRESHOLD": BUY_THRESHOLD,
            "SELL_THRESHOLD": SELL_THRESHOLD,
            "QUICK_BUY_THRESHOLD": QUICK_BUY_THRESHOLD,
            "RECENT_HIGH_WINDOW": 10,
            "RECENT_LOW_WINDOW": 14,
        }

    def set_environment(self, market_info, buy_w, sell_w):
        self.market_info = market_info
        self.buy_weights = buy_w
        self.sell_weights = sell_w

    # ---------- 辅助计算 ----------
    @staticmethod
    def calc_change_pct(real_price, hist_df, today):
        if real_price is None or hist_df is None or hist_df.empty:
            return 0.0
        today_ts = pd.Timestamp(today)
        finished_bars = hist_df[hist_df.index < today_ts]
        if not finished_bars.empty:
            base_close = finished_bars.iloc[-1]["close"]
        else:
            base_close = hist_df.iloc[-1]["close"]
        return (real_price - base_close) / base_close * 100 if base_close > 0 else 0.0

    @staticmethod
    def generate_risk_alerts(
        price: float,
        recent_high: float,
        recent_low: float,
        atr: float
    ) -> Dict[str, Optional[Union[float, str]]]:
        if atr <= 0:
            return {"stop_loss": None, "trail_profit": None, "alert": None}

        stop_price = recent_high - ATR_STOP_MULT * atr
        trail_price = recent_high - ATR_TRAILING_PROFIT_MULT * atr
        alert: Optional[str] = None

        if price <= stop_price:
            alert = f"🔴 动态止损({stop_price:.3f})"
        elif price - stop_price < RISK_ALERT_DISTANCE_ATR * atr:
            alert = f"🟡 近止损({stop_price:.3f})"
        elif trail_price and price - trail_price < RISK_ALERT_DISTANCE_ATR * atr:
            alert = f"🟢 近止盈({trail_price:.3f})"

        return {"stop_loss": stop_price, "trail_profit": trail_price, "alert": alert}

    @staticmethod
    def evaluate_cost_based_stop_profit(
        price: float,
        cost: float,
        recent_high: float,
        atr: float,
        trailing_profit_level: Optional[str],
    ) -> Dict[str, Optional[str]]:
        """
        成本止盈止损评估。
        止损：强制 SELL。
        止盈：根据 PROFIT_TAKE_MODE 决定是否强制 SELL。
        """
        if cost is None or cost <= 0:
            return {"action_override": None, "level_override": None}

        profit_pct = (price - cost) / cost

        # 止损：硬性卖出
        if profit_pct <= COST_STOP_LOSS_PCT:
            return {"action_override": "SELL", "level_override": "止损卖出"}

        # 止盈逻辑
        is_clear_profit = profit_pct >= COST_TAKE_PROFIT_CLEAR and trailing_profit_level == "clear"
        is_half_profit = profit_pct >= COST_TAKE_PROFIT_HALF and trailing_profit_level in ("half", "clear")

        if PROFIT_TAKE_MODE == "hard":
            # 硬性止盈：强制卖出
            if is_clear_profit:
                return {"action_override": "SELL", "level_override": "清仓止盈"}
            if is_half_profit and COST_HALF_PROFIT_ACTION == "SELL":
                return {"action_override": "SELL", "level_override": "半仓止盈"}
        else:
            # 软性止盈：只改等级，不改动作
            if is_clear_profit:
                return {"action_override": None, "level_override": "清仓止盈(提示)"}
            if is_half_profit:
                return {"action_override": None, "level_override": "半仓止盈(提示)"}

        return {"action_override": None, "level_override": None}

    def _evaluate_take_profit(self, ctx: ETFContext):
        if ctx.real_price is None:
            return
        price = ctx.real_price

        if ctx.hist_df is None or ctx.hist_df.empty:
            return
        d = ctx.hist_df.iloc[-1]
        recent_high = d.get(
            f"recent_high_{ctx.params['RECENT_HIGH_WINDOW']}",
            ctx.hist_df["high"].rolling(ctx.params["RECENT_HIGH_WINDOW"]).max().iloc[-1]
        )
        recent_low = d.get(
            f"recent_low_{ctx.params['RECENT_LOW_WINDOW']}",
            ctx.hist_df["low"].rolling(ctx.params["RECENT_LOW_WINDOW"]).min().iloc[-1]
        )

        ctx.recent_high_price = recent_high
        ctx.recent_low_price = recent_low
        ctx.max_drawdown_pct = (recent_high - price) / recent_high if recent_high > 0 else 0.0

        ctx.trailing_profit_level = get_trailing_profit_signals(
            price, recent_high, ctx.atr_pct
        )

        if recent_low > 0:
            ctx.profit_pct_from_low = (price - recent_low) / recent_low
            threshold = 0.15
            if ctx.profit_pct_from_low >= threshold * 1.6:
                ctx.profit_level = 'clear'
            elif ctx.profit_pct_from_low >= threshold * 1.0:
                ctx.profit_level = 'half'
            elif ctx.profit_pct_from_low >= threshold * 0.7:
                ctx.profit_level = 'watch'
            else:
                ctx.profit_level = None
        else:
            ctx.profit_pct_from_low = 0.0
            ctx.profit_level = None

    def _compute_factors(self, ctx: ETFContext, d: pd.Series):
        hist_df = ctx.hist_df
        assert hist_df is not None
        price = ctx.real_price
        assert price is not None

        ma20 = d["ma_short"]
        volume = d["volume"]
        vol_ma = d["vol_ma"]
        rsi = d["rsi"]
        boll_up = d["boll_up"]
        boll_low = d["boll_low"]
        williams_r = d["williams_r"]

        macd_golden = kdj_golden = 0
        if len(hist_df) >= 2:
            prev = hist_df.iloc[-2]
            macd_golden = 1 if (d["macd_dif"] > d["macd_dea"] and prev["macd_dif"] <= prev["macd_dea"]) else 0
            kdj_golden = 1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0

        ret_etf_5d = (price / hist_df.iloc[-5]["close"] - 1) if len(hist_df) >= 5 else 0
        ret_market_5d = ctx.market.get("ret_market_5d", 0)
        mkt_ma20 = ctx.market.get("above_ma20", False)
        mkt_ma60 = ctx.market.get("above_ma60", False)
        mkt_amt = ctx.market.get("amount_above_ma20", False)

        reversal_score = factor_buy_reversal_potential(
            price, d.get("low_close_20", price), rsi, d.get("rsi_prev", rsi),
            d.get("boll_width", 0), d.get("boll_width_ma20", 0),
            volume, vol_ma, d.get("close_open_ratio", 1.0),
            atr_pct=ctx.atr_pct
        )

        ctx.buy_factors = {
            "price_above_ma20":      factor_buy_price_above_ma20(price, ma20),
            "volume_above_ma5":      factor_buy_volume_above_ma5(volume, vol_ma),
            "macd_golden_cross":     macd_golden,
            "kdj_golden_cross":      kdj_golden,
            "bollinger_break_up":    factor_buy_bollinger_break_up(price, boll_up),
            "williams_oversold":     factor_buy_williams_oversold(williams_r),
            "market_above_ma20":     1 if mkt_ma20 else 0,
            "market_above_ma60":     1 if mkt_ma60 else 0,
            "market_amount_above_ma20": 1 if mkt_amt else 0,
            "outperform_market":     factor_buy_outperform_market(ret_etf_5d, ret_market_5d),
            "weekly_above_ma20":     1 if ctx.weekly_above else 0,
            "tmsv_score":            ctx.tmsv_strength,
            "rsi_oversold":          factor_buy_rsi_oversold(rsi),
            "reversal_potential":    reversal_score,
        }
        ctx.sell_factors = {
            "price_below_ma20":      factor_sell_price_below_ma20(price, ma20),
            "bollinger_break_down":  factor_sell_bollinger_break_down(price, boll_low),
            "williams_overbought":   factor_sell_williams_overbought(williams_r),
            "rsi_overbought":        factor_sell_rsi_overbought(rsi),
            "underperform_market":   factor_sell_underperform_market(ret_etf_5d, ret_market_5d),
            "stop_loss_ma_break":    factor_sell_stop_loss_ma_break(price, ma20),
            "weekly_below_ma20":     1 if ctx.weekly_below else 0,
            "downside_momentum":     factor_sell_downside_momentum(ctx.downside_momentum),
            "max_drawdown_stop":     factor_sell_max_drawdown_stop(ctx.max_drawdown_pct),
        }

    def _core_analysis(self, ctx: ETFContext):
        real_price = ctx.real_price
        hist_df = ctx.hist_df
        if real_price is None or hist_df is None or len(hist_df) < 20:
            ctx.error = "数据不足" if hist_df is not None else "实时价格获取失败"
            return ctx

        ctx.change_pct = self.calc_change_pct(real_price, hist_df, ctx.today)
        d = hist_df.iloc[-1]
        ctx.atr_pct = d["atr"] / real_price if real_price > 0 else 0
        ctx.rsi = d["rsi"]
        ctx.above_ma30 = real_price > d["ma30"] if "ma30" in d and not pd.isna(d["ma30"]) else True
        ctx.is_weak_ma = not ctx.above_ma30

        ctx.weekly_above = ctx.weekly_below = False
        if ctx.weekly_df is not None and not ctx.weekly_df.empty:
            w = ctx.weekly_df.iloc[-1]
            if "ma_short" in w.index and not pd.isna(w["ma_short"]):
                ctx.weekly_above = w["close"] > w["ma_short"]
                ctx.weekly_below = w["close"] < w["ma_short"]

        market_state = ctx.market.get("state", "震荡")
        try:
            tmsv_series = compute_tmsv(hist_df, market_state, ctx.market.get("volatility", 0.02))
            ctx.tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
            ctx.tmsv_strength = ctx.tmsv / 100.0
        except Exception:
            ctx.tmsv = 50.0
            ctx.tmsv_strength = 0.5

        ctx.downside_momentum = d.get("downside_momentum_raw", 0.0)
        self._evaluate_take_profit(ctx)

        if ctx.cost_price is not None and ctx.cost_price > 0:
            ctx.cost_profit_pct = (real_price - ctx.cost_price) / ctx.cost_price

        ctx.buy_weights_used = self.buy_weights.copy()
        ctx.sell_weights_used = self.sell_weights.copy()
        self._compute_factors(ctx, d)

        ctx.buy_score = weighted_sum(ctx.buy_factors, ctx.buy_weights_used)
        ctx.sell_score = weighted_sum(ctx.sell_factors, ctx.sell_weights_used)

        if not ctx.above_ma30:
            ctx.buy_score *= MA30_WEAKNESS_PENALTY
        if ctx.buy_factors.get("macd_golden_cross", 0) == 0 and ctx.buy_factors.get("kdj_golden_cross", 0) == 0:
            ctx.buy_score *= 0.95

        ctx.raw_score = ctx.buy_score - ctx.sell_score
        env_factor = ctx.market["factor"]
        transformed = nonlinear_score_transform(
            ctx.raw_score, market_state,
            NONLINEAR_SCALE_BULL, NONLINEAR_SCALE_RANGE
        )
        ctx.final_score = (transformed * env_factor) * 50 + 50
        ctx.final_score = max(1.0, min(99.0, ctx.final_score))

        if ctx.profit_pct_from_low > 0.12:
            ctx.final_score *= 0.92
        if ctx.rsi > 75:
            ctx.final_score *= 0.95
        ctx.final_score = max(1.0, min(99.0, ctx.final_score))
        return ctx

    def _build_risk_str(self, ctx: ETFContext, state: dict, final_level: str = "") -> str:
        labels = []

        # 量价关系标签
        if ctx.real_price and ctx.hist_df is not None and not ctx.hist_df.empty:
            d = ctx.hist_df.iloc[-1]
            vol_ratio = (d["volume"] / d["vol_ma"]) if d["vol_ma"] > 0 else None
            if vol_ratio is not None:
                if ctx.change_pct > 0 and vol_ratio < 0.8:
                    labels.append("🤔 缩量上涨")
                elif ctx.change_pct > 0 and vol_ratio > 1.5:
                    labels.append("🤑 放量上涨")
                elif ctx.change_pct < 0 and vol_ratio < 0.8:
                    labels.append("😦 缩量下跌")
                elif ctx.change_pct < 0 and vol_ratio > 1.5:
                    labels.append("😭 放量下跌")

        if ctx.cost_price is not None:
            if ctx.cost_profit_pct is not None:
                pct = ctx.cost_profit_pct * 100
                if pct >= COST_TAKE_PROFIT_CLEAR * 100:
                    labels.append(f"💰浮盈 {pct:.1f}%")
                elif pct <= COST_STOP_LOSS_PCT * 100:
                    labels.append(f"🔻浮亏 {pct:.1f}%")

            # 软止盈提示（不再强制SELL）
            if ctx.profit_pct_from_low >= 0.12:
                if ctx.profit_level == 'clear':
                    labels.append("⛔ 清仓止盈(提示)")
                elif ctx.profit_level == 'half':
                    labels.append("💸 半仓止盈(提示)")
                else:
                    labels.append("🤭 止盈关注")
            if ctx.trailing_profit_level == 'clear':
                labels.append("⛔ 移动止盈(提示)")
            elif ctx.trailing_profit_level == 'half':
                labels.append("💸 移动止盈(提示)")

            if "止损卖出" not in final_level and "清仓止盈" not in final_level:
                if ctx.real_price and ctx.recent_high_price and ctx.atr_pct:
                    atr_abs = ctx.atr_pct * ctx.real_price
                    alerts = self.generate_risk_alerts(ctx.real_price, ctx.recent_high_price, 0, atr_abs)
                    alert_text = alerts.get("alert")
                    if isinstance(alert_text, str):
                        labels.append(alert_text)

        if ctx.profit_pct_from_low is not None and ctx.profit_pct_from_low <= 0.03:
            labels.append("📉 低位")
        if ctx.is_weak_ma:
            labels.append("🔽 弱于均线")

        if len(state.get("score_history", [])) >= RISK_WARNING_DAYS:
            recent_scores = [s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]]
            if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
                labels.append("🛑 连续低分")

        return " ".join(labels)

    def get_action(self, score: float, score_history: List[dict]) -> Tuple[str, str]:
        hist_scores = [s["score"] for s in score_history]
        buy_thresh = self.params["BUY_THRESHOLD"]
        sell_thresh = self.params["SELL_THRESHOLD"]

        def _get_level(s):
            for th, lvl in zip(ACTION_LEVEL_THRESHOLDS, ACTION_LEVEL_NAMES):
                if s >= th:
                    return lvl
            return ACTION_LEVEL_NAMES[-1]

        level = _get_level(score)

        if QUICK_BUY_ENABLE and len(hist_scores) >= 1:
            prev_score = hist_scores[-1]
            if score >= QUICK_BUY_THRESHOLD and (score - prev_score) >= QUICK_BUY_SCORE_INCREASE:
                return "BUY", level

        if len(hist_scores) < self.params["CONFIRM_DAYS"]:
            if score > buy_thresh:
                return "BUY", level
            elif score < sell_thresh:
                return "SELL", level
            else:
                return "HOLD", level

        confirm_days = self.params["CONFIRM_DAYS"]
        recent = hist_scores[-confirm_days:]

        if SIGNAL_CONFIRM_MODE == "strict":
            if score > buy_thresh and all(s > buy_thresh for s in recent):
                return "BUY", level
            if score < sell_thresh and all(s < sell_thresh for s in recent):
                return "SELL", level
        elif SIGNAL_CONFIRM_MODE == "majority":
            needed = (confirm_days // 2) + 1
            if score > buy_thresh and sum(1 for s in recent if s > buy_thresh) >= needed:
                return "BUY", level
            if score < sell_thresh and sum(1 for s in recent if s < sell_thresh) >= needed:
                return "SELL", level
        elif SIGNAL_CONFIRM_MODE == "trend_break":
            if score > buy_thresh and all(recent[i] < recent[i+1] for i in range(len(recent)-1)) and recent[-1] > buy_thresh:
                return "BUY", level
            if score < sell_thresh and all(recent[i] > recent[i+1] for i in range(len(recent)-1)) and recent[-1] < sell_thresh:
                return "SELL", level

        return "HOLD", level

    @staticmethod
    def _adjust_action_level_for_profit(action_level, ctx):
        return action_level

    def _apply_cost_based_overrides(self, action: str, action_level: str,
                                    ctx: ETFContext) -> Tuple[str, str]:
        if (not USE_COST_BASED_OVERRIDE
            or ctx.cost_price is None
            or ctx.real_price is None):
            return action, action_level

        atr_abs = ctx.atr_pct * ctx.real_price if ctx.atr_pct else 0.0
        decision = self.evaluate_cost_based_stop_profit(
            price=ctx.real_price,
            cost=ctx.cost_price,
            recent_high=ctx.recent_high_price,
            atr=atr_abs,
            trailing_profit_level=ctx.trailing_profit_level,
        )

        if decision["action_override"] == "SELL":
            return "SELL", decision["level_override"] or action_level
        elif decision["level_override"]:
            # 软止盈：只改等级，不改动作
            return action, decision["level_override"]
        return action, action_level

    @staticmethod
    def generate_position_advice(
        shares: int,
        cost_price: Optional[float],
        real_price: Optional[float],
        final_score: float,
        action_level: str,
        risk_str: str,
    ) -> str:
        """
        根据持仓、评分、风险标签生成仓位管理建议。
        输出格式：仓位建议文本（如“🔼 加仓20%” / “🔽 减仓30%” / “⚪ 持有”等）
        """
        if not POSITION_ADVICE_ENABLE or shares <= 0 or cost_price is None or real_price is None:
            return ""

        # 计算当前市值与盈亏
        current_value = shares * real_price
        cost_value = shares * cost_price
        profit_pct = (real_price - cost_price) / cost_price
        profit_amount = current_value - cost_value

        # 基础状态
        if final_score >= POSITION_ADD_THRESHOLD:
            if profit_pct < 0.05:
                advice = f"🔼 加仓 {int(POSITION_ADD_RATIO*100)}%"
            elif profit_pct < 0.15:
                advice = f"📈 小幅加仓 {int(POSITION_ADD_RATIO*100)}%"
            else:
                advice = f"💰 浮盈{profit_pct:.1%}，可追加上涨 {int(POSITION_ADD_RATIO*50)}%"
        elif final_score <= POSITION_CLEAR_THRESHOLD:
            advice = "⛔ 清仓离场"
        elif final_score <= POSITION_REDUCE_THRESHOLD:
            advice = f"🔽 减仓 {int(POSITION_REDUCE_RATIO*100)}%"
        else:
            advice = "⚪ 持有观察"

        # 叠加风险标签修正
        if "清仓止盈" in risk_str or "移动止盈" in risk_str:
            if "提示" in risk_str:
                advice = "⚠️ " + advice + "，注意止盈"
            else:
                advice = "⛔ 清仓止盈"
        elif "止损卖出" in risk_str:
            advice = "⛔ 强制止损"
        elif "连续低分" in risk_str:
            advice = "🔽 强烈建议减仓"

        return advice

    def analyze_single_etf(self, code, name, real_price, hist_df, weekly_df,
                           market, today, state, cost_price=None, shares=0) -> Tuple[str, Optional[dict], dict, float, Optional[dict], Optional[dict]]:
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy(),
                         cost_price=cost_price, shares=shares)
        ctx = self._core_analysis(ctx)

        if ctx.error:
            out = format_etf_output_line(name, code, real_price, 0.0, 0, "数据不足")
            return out, None, state, -100.0, None, None

        final = ctx.final_score
        today_str = today.strftime("%Y-%m-%d")

        if "score_history" not in state:
            state["score_history"] = []
        found = False
        for item in state["score_history"]:
            if item["date"] == today_str:
                item["score"] = final
                found = True
                break
        if not found:
            state["score_history"].append({"date": today_str, "score": final})
        state["score_history"].sort(key=lambda x: x["date"])

        action, action_level = self.get_action(final, state["score_history"])

        if cost_price is not None:
            action, final_level = self._apply_cost_based_overrides(action, action_level, ctx)
        else:
            final_level = action_level

        risk_str = self._build_risk_str(ctx, state, final_level)

        # 生成仓位管理建议（仅当有持仓且开启时）
        position_advice = self.generate_position_advice(
            shares=shares,
            cost_price=cost_price,
            real_price=real_price,
            final_score=final,
            action_level=final_level,
            risk_str=risk_str,
        )

        # 将仓位建议附加到 risk_str 末尾（便于表格显示）
        if position_advice:
            risk_str = (risk_str + "  " + position_advice) if risk_str else position_advice

        output = format_etf_output_line(
            name=name,
            code=code,
            price=real_price,
            change_pct=ctx.change_pct,
            final_score=final,
            action_level=final_level,
            risk_str=risk_str,
            signal_action=action if action in ("BUY", "SELL") else None,
        )
        signal = {
            "action": action, "name": name, "code": code, "score": final
        } if action in ("BUY", "SELL") else None

        risk_data = {
            "price": ctx.real_price,
            "recent_high": ctx.recent_high_price,
            "recent_low": ctx.recent_low_price,
            "atr": ctx.atr_pct * ctx.real_price if ctx.atr_pct and ctx.real_price else 0.0,
            "cost": ctx.cost_price,
            "cost_profit_pct": ctx.cost_profit_pct,
        }

        scan_info = {
            "profit_pct_from_low": ctx.profit_pct_from_low if ctx.profit_pct_from_low is not None else 0.0,
            "max_drawdown_pct": ctx.max_drawdown_pct if ctx.max_drawdown_pct is not None else 0.0,
            "change_pct": (ctx.change_pct / 100.0) if ctx.change_pct is not None else 0.0,
            "has_weak_ma_text": ctx.is_weak_ma,
            "has_clear_stop_text": "清仓止盈" in risk_str or "止损卖出" in final_level,
            "has_strong_sell_text": "强烈卖出" in final_level or "连续低分" in risk_str,
            "has_buy_signal": action == "BUY",
            "has_sell_signal": action == "SELL",
            "final_score": final,
            "rsi": ctx.rsi,
            "tmsv": ctx.tmsv,
            "cost_profit_pct": ctx.cost_profit_pct,
            "cost_price": ctx.cost_price,
            "shares": shares,
            "position_advice": position_advice,
            "display": {
                "name": name,
                "code": code,
                "price": real_price,
                "change_pct": ctx.change_pct,
                "final_score": final,
                "action_level": final_level,
                "risk_str": risk_str,
            }
        }

        return output, signal, state, final, risk_data, scan_info

    def detailed_analysis(self, code, name, real_price, hist_df, weekly_df,
                          market, today, state, ai_client=None, cost_price=None, shares=0) -> str:
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy(),
                         cost_price=cost_price, shares=shares)
        ctx = self._core_analysis(ctx)
        if ctx.error:
            return f"【{name} ({code})】{ctx.error}，无法分析。"

        final = ctx.final_score
        action, action_level = self.get_action(final, state.get("score_history", []))

        if cost_price is not None:
            action, final_level = self._apply_cost_based_overrides(action, action_level, ctx)
        else:
            final_level = action_level

        risk_str = self._build_risk_str(ctx, state, final_level)
        position_advice = self.generate_position_advice(
            shares=shares,
            cost_price=cost_price,
            real_price=real_price,
            final_score=final,
            action_level=final_level,
            risk_str=risk_str,
        )
        if position_advice:
            risk_str = (risk_str + "  " + position_advice) if risk_str else position_advice

        ai_comment = None
        if ai_client:
            try:
                d = ctx.hist_df.iloc[-1] if ctx.hist_df is not None else None
                macd_status = ""
                if d is not None and pd.notna(d.get("macd_dif")) and pd.notna(d.get("macd_dea")):
                    macd_status = "金叉" if d["macd_dif"] > d["macd_dea"] else "死叉"
                vol_ratio = None
                if d is not None and d.get("volume") and d.get("vol_ma") and d["vol_ma"] > 0:
                    vol_ratio = d["volume"] / d["vol_ma"]

                ai_comment = ai_client.comment_on_etf(
                    code=code,
                    name=name,
                    final_score=ctx.final_score,
                    action_level=final_level,
                    market_state=market["state"],
                    market_factor=market["factor"],
                    tmsv=ctx.tmsv,
                    atr_pct=ctx.atr_pct,
                    cost_price=ctx.cost_price,
                    cost_profit_pct=ctx.cost_profit_pct,
                    signal_action=action if action in ("BUY", "SELL") else None,
                    risk_tags=risk_str,
                    rsi=ctx.rsi,
                    macd_status=macd_status,
                    vol_ratio=vol_ratio,
                )
            except Exception as e:
                logger.warning(f"AI评论生成失败: {e}")
                ai_comment = "（AI评论生成失败）"

        return format_detailed_report(ctx, market, self.params, final_level, ai_comment,
                                      signal_action=action if action in ("BUY", "SELL") else None)