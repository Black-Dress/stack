#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核心分析引擎：ETF 评分与信号生成，权重由环境决定，已纳入成本价止盈止损覆盖"""
import datetime
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from .risk_watch import generate_risk_alerts, evaluate_cost_based_stop_profit

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
    cost_price: Optional[float] = None       # 持仓成本价

    change_pct: float = 0.0
    atr_pct: float = 0.0
    tmsv: float = 50.0
    tmsv_strength: float = 0.5
    downside_momentum: float = 0.0
    max_drawdown_pct: float = 0.0
    weekly_above: bool = False
    weekly_below: bool = False
    above_ma30: bool = False
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
    cost_profit_pct: Optional[float] = None          # 基于成本的浮动盈亏

    buy_weights_used: Dict = field(default_factory=dict)
    sell_weights_used: Dict = field(default_factory=dict)


class DataAnalyzer:
    # 这些降级映射现在已经不再用于成本覆盖的ETF，保留供无成本的ETF使用（但无成本时也不做降级，因此实际上可以废弃，但保留向后兼容）
    PROFIT_DOWNGRADE_CLEAR = {
        "极度看好": "谨慎买入(高估)",
        "强烈买入": "谨慎买入(高估)",
        "买入": "谨慎买入(高估)",
        "谨慎买入": "偏多持有(高估)",
        "偏多持有": "中性偏多(高估)",
    }
    PROFIT_DOWNGRADE_HALF = {
        "极度看好": "强烈买入(注意止盈)",
        "强烈买入": "买入(注意止盈)",
        "买入": "谨慎买入(注意止盈)",
        "谨慎买入": "偏多持有",
        "偏多持有": "中性偏多",
    }

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
        finished_bars = hist_df[hist_df.index.date < today]
        if not finished_bars.empty:
            base_close = finished_bars.iloc[-1]["close"]
        else:
            base_close = hist_df.iloc[-1]["close"]
        return (real_price - base_close) / base_close * 100 if base_close > 0 else 0.0

    def _evaluate_take_profit(self, ctx: ETFContext):
        # 保护：若实时价格缺失，无法计算止盈数据
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

    # ---------- 因子计算 ----------
    def _compute_factors(self, ctx: ETFContext, d: pd.Series):
        hist_df = ctx.hist_df
        assert hist_df is not None, "hist_df must not be None when computing factors"
        price = ctx.real_price
        assert price is not None, "real_price must not be None"

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
            "reversal_potential":    factor_buy_reversal_potential(
                price, d.get("low_close_20", price), rsi, d.get("rsi_prev", rsi),
                d.get("boll_width", 0), d.get("boll_width_ma20", 0),
                volume, vol_ma, d.get("close_open_ratio", 1.0)
            ),
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

    # ---------- 核心分析 ----------
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

        # 周线状态
        ctx.weekly_above = ctx.weekly_below = False
        if ctx.weekly_df is not None and not ctx.weekly_df.empty:
            w = ctx.weekly_df.iloc[-1]
            if "ma_short" in w.index and not pd.isna(w["ma_short"]):
                ctx.weekly_above = w["close"] > w["ma_short"]
                ctx.weekly_below = w["close"] < w["ma_short"]

        # TMSV 强度
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

        # 如果提供了成本价，计算浮动盈亏
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
        env_factor = clip_env_factor(ctx.market["factor"], 1.0)
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

        # 有持仓成本的才添加止盈止损相关标签
        if ctx.cost_price is not None:
            # 浮动盈亏标签
            if ctx.cost_profit_pct is not None:
                pct = ctx.cost_profit_pct * 100
                if pct >= COST_TAKE_PROFIT_CLEAR * 100:
                    labels.append(f"💰浮盈 {pct:.1f}%")
                elif pct <= COST_STOP_LOSS_PCT * 100:
                    labels.append(f"🔻浮亏 {pct:.1f}%")

            # 原有基于低点涨幅的止盈标签、移动止盈标签等，现在仅对持仓者显示
            if ctx.profit_pct_from_low >= 0.12:
                if ctx.profit_level == 'clear':
                    labels.append("⛔ 清仓止盈")
                elif ctx.profit_level == 'half':
                    labels.append("⚠️  半仓止盈")
                else:
                    labels.append("止盈关注")
            if ctx.trailing_profit_level == 'clear':
                labels.append("⛔ 移动止盈")
            elif ctx.trailing_profit_level == 'half':
                labels.append("⚠️  移动止盈")

            # ATR 动态止损/止盈提醒（但若已触发成本止损/止盈，则不再显示，避免冗余）
            if "止损卖出" not in final_level and "清仓止盈" not in final_level:
                if ctx.real_price and ctx.recent_high_price and ctx.atr_pct:
                    atr_abs = ctx.atr_pct * ctx.real_price
                    alerts = generate_risk_alerts(ctx.real_price, ctx.recent_high_price, 0, atr_abs)
                    alert_text = alerts.get("alert")
                    if isinstance(alert_text, str):
                        # 已包含价位信息，直接使用
                        labels.append(alert_text)

        # 位置特征（所有ETF都显示）
        if ctx.profit_pct_from_low is not None and ctx.profit_pct_from_low <= 0.03:
            labels.append("📉 低位")
        if not ctx.above_ma30:
            labels.append("⬇️  弱于均线")

        # 连续低分警告
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
            return "强烈卖出"

        level = _get_level(score)

        if len(hist_scores) < 2:
            if score > buy_thresh:
                return "BUY", level
            elif score < sell_thresh:
                return "SELL", level
            else:
                return "HOLD", level

        confirm_days = self.params["CONFIRM_DAYS"]
        recent = hist_scores[-min(confirm_days, len(hist_scores)):]
        if score > buy_thresh and all(s > buy_thresh for s in recent):
            return "BUY", level
        if score < sell_thresh and all(s < sell_thresh for s in recent):
            return "SELL", level
        return "HOLD", level

    @staticmethod
    def _adjust_action_level_for_profit(action_level, ctx):
        """
        不再使用此方法进行成本相关调整，仅在有成本价时直接由成本覆盖，
        无成本价时也不进行任何止盈降级。
        保留方法以防其他地方调用，但逻辑已废弃。
        """
        return action_level

    def _apply_cost_based_overrides(self, action: str, action_level: str,
                                    ctx: ETFContext) -> Tuple[str, str]:
        # 没有成本价或实时价格，不触发
        if (not USE_COST_BASED_OVERRIDE
            or ctx.cost_price is None
            or ctx.real_price is None):
            return action, action_level

        atr_abs = ctx.atr_pct * ctx.real_price if ctx.atr_pct else 0.0
        decision = evaluate_cost_based_stop_profit(
            price=ctx.real_price,
            cost=ctx.cost_price,
            recent_high=ctx.recent_high_price,
            atr=atr_abs,
            trailing_profit_level=ctx.trailing_profit_level or "",
        )

        if decision["action_override"] == "SELL":
            return "SELL", decision["level_override"] or action_level
        elif decision["level_override"]:
            return action, decision["level_override"]
        return action, action_level

    # ---------- 公开接口 ----------
    def analyze_single_etf(self, code, name, real_price, hist_df, weekly_df,
                           market, today, state, cost_price=None) -> Tuple[str, Optional[dict], dict, float, Optional[dict], Optional[dict]]:
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy(),
                         cost_price=cost_price)
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

        # 有成本价时直接应用成本覆盖规则，不再进行原有的技术面降级
        if cost_price is not None:
            action, final_level = self._apply_cost_based_overrides(action, action_level, ctx)
        else:
            final_level = action_level

        risk_str = self._build_risk_str(ctx, state, final_level)

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
            "has_weak_ma_text": "弱于均线" in risk_str,
            "has_clear_stop_text": "清仓止盈" in risk_str or "止损卖出" in final_level,
            "has_strong_sell_text": "强烈卖出" in final_level or "连续低分" in risk_str,
            "has_buy_signal": action == "BUY",
            "has_sell_signal": action == "SELL",
        }

        return output, signal, state, final, risk_data, scan_info

    def detailed_analysis(self, code, name, real_price, hist_df, weekly_df,
                          market, today, state, ai_client=None, cost_price=None) -> str:
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy(),
                         cost_price=cost_price)
        ctx = self._core_analysis(ctx)
        if ctx.error:
            return f"【{name} ({code})】{ctx.error}，无法分析。"

        final = ctx.final_score
        _, action_level = self.get_action(final, state.get("score_history", []))
        
        if cost_price is not None:
            _, final_level = self._apply_cost_based_overrides(_, action_level, ctx)
        else:
            final_level = action_level

        # ----- AI 评论（传入真实数据）-----
        ai_comment = None
        if ai_client:
            try:
                ai_comment = ai_client.comment_on_etf(
                    code=code,
                    name=name,
                    final_score=ctx.final_score,
                    action_level=final_level,
                    market_state=market["state"],
                    market_factor=market["factor"],
                    tmsv=ctx.tmsv,
                    atr_pct=ctx.atr_pct
                )
            except Exception as e:
                logger.warning(f"AI评论生成失败: {e}")
                ai_comment = "（AI评论生成失败）"

        return format_detailed_report(ctx, market, self.params, final_level, ai_comment)