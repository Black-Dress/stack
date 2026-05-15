#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""核心分析引擎：仅负责计算评分和技术指标，不生成信号或仓位建议"""
import datetime
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from .config import *
from .utils import weighted_sum, nonlinear_score_transform, safe_ratio
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
    code: str
    name: str
    real_price: Optional[float]
    hist_df: Optional[pd.DataFrame]
    weekly_df: Optional[pd.DataFrame]
    today: datetime.date
    market: Dict
    params: Dict
    cost_price: Optional[float] = None
    shares: int = 0

    change_pct: float = 0.0
    change_5d: float = 0.0
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


class DataAnalyzer:
    def __init__(self):
        self.market_info = {}
        self.buy_weights = {}
        self.sell_weights = {}
        self.params = {
            "RECENT_HIGH_WINDOW": 10,
            "RECENT_LOW_WINDOW": 14,
        }

    def set_environment(self, market_info, buy_w, sell_w):
        self.market_info = market_info
        self.buy_weights = buy_w
        self.sell_weights = sell_w

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

    def _core_analysis(self, ctx: ETFContext) -> ETFContext:
        real_price = ctx.real_price
        hist_df = ctx.hist_df
        if real_price is None or hist_df is None or len(hist_df) < 20:
            ctx.error = "数据不足" if hist_df is not None else "实时价格获取失败"
            return ctx

        ctx.change_pct = self.calc_change_pct(real_price, hist_df, ctx.today)
        if len(hist_df) >= 5:
            close_5d_ago = hist_df.iloc[-5]["close"]
            ctx.change_5d = (real_price - close_5d_ago) / close_5d_ago
        else:
            ctx.change_5d = 0.0

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
        transformed = nonlinear_score_transform(ctx.raw_score, market_state, NONLINEAR_SCALE_BULL, NONLINEAR_SCALE_RANGE)
        ctx.final_score = (transformed * env_factor) * 50 + 50
        ctx.final_score = max(1.0, min(99.0, ctx.final_score))

        if ctx.profit_pct_from_low > 0.12:
            ctx.final_score *= 0.92
        if ctx.rsi > 75:
            ctx.final_score *= 0.95
        ctx.final_score = max(1.0, min(99.0, ctx.final_score))
        return ctx

    def _evaluate_take_profit(self, ctx):
        if ctx.real_price is None or ctx.hist_df is None or ctx.hist_df.empty:
            return
        price = ctx.real_price
        d = ctx.hist_df.iloc[-1]
        recent_high = d.get(f"recent_high_{self.params['RECENT_HIGH_WINDOW']}",
                            ctx.hist_df["high"].rolling(self.params['RECENT_HIGH_WINDOW']).max().iloc[-1])
        recent_low = d.get(f"recent_low_{self.params['RECENT_LOW_WINDOW']}",
                           ctx.hist_df["low"].rolling(self.params['RECENT_LOW_WINDOW']).min().iloc[-1])
        ctx.recent_high_price = recent_high
        ctx.recent_low_price = recent_low
        ctx.max_drawdown_pct = (recent_high - price) / recent_high if recent_high > 0 else 0.0
        ctx.trailing_profit_level = get_trailing_profit_signals(price, recent_high, ctx.atr_pct)
        if recent_low > 0:
            ctx.profit_pct_from_low = (price - recent_low) / recent_low
            threshold = 0.15
            if ctx.profit_pct_from_low >= threshold * 1.6:
                ctx.profit_level = 'clear'
            elif ctx.profit_pct_from_low >= threshold:
                ctx.profit_level = 'half'
            elif ctx.profit_pct_from_low >= threshold * 0.7:
                ctx.profit_level = 'watch'
            else:
                ctx.profit_level = None
        else:
            ctx.profit_pct_from_low = 0.0
            ctx.profit_level = None

    def _compute_factors(self, ctx, d):
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

    def analyze_single_etf(self, code, name, real_price, hist_df, weekly_df,
                           market, today, state, cost_price=None, shares=0) -> Tuple[str, Optional[dict], dict, float, Optional[dict], Optional[dict]]:
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy(),
                         cost_price=cost_price, shares=shares)
        ctx = self._core_analysis(ctx)

        if ctx.error:
            return "", None, state, -100.0, None, None

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

        # 构建风险标签
        risk_str = self._build_risk_str(ctx, state, "")
        # 构建数据字典供事件检测使用
        # 注意：需要从 hist_df 中提取 ma5, ma10, ma20_trend 等
        d = ctx.hist_df.iloc[-1] if ctx.hist_df is not None else None
        ma5 = d["ma5"] if d is not None else None
        ma10 = d["ma10"] if d is not None else None
        ma20 = d["ma_short"] if d is not None else None
        # 趋势简单计算（今日与前日比较）
        ma10_trend = 0
        if d is not None and len(ctx.hist_df) >= 2:
            prev_ma10 = ctx.hist_df["ma10"].iloc[-2]
            ma10_trend = 1 if ma10 > prev_ma10 else -1 if ma10 < prev_ma10 else 0
        ma20_trend = 0
        if d is not None and len(ctx.hist_df) >= 2:
            prev_ma20 = ctx.hist_df["ma_short"].iloc[-2]
            ma20_trend = 1 if ma20 > prev_ma20 else -1 if ma20 < prev_ma20 else 0
        # 近期高点
        recent_high_10 = d.get(f"recent_high_10", ctx.recent_high_price) if d is not None else ctx.recent_high_price
        recent_high_20 = d.get(f"recent_high_20", ctx.recent_high_price) if d is not None else ctx.recent_high_price
        macd_hist = d["macd_hist"] if d is not None else 0.0

        scan_info = {
            "profit_pct_from_low": ctx.profit_pct_from_low or 0.0,
            "max_drawdown_pct": ctx.max_drawdown_pct or 0.0,
            "change_pct": ctx.change_pct / 100.0,
            "change_5d": ctx.change_5d,
            "final_score": final,
            "rsi": ctx.rsi,
            "tmsv": ctx.tmsv,
            "cost_profit_pct": ctx.cost_profit_pct,
            "cost_price": ctx.cost_price,
            "shares": shares,
            "above_ma": ctx.real_price > ma20 if ma20 is not None else False,
            "vol_ratio": d["volume"] / d["vol_ma"] if d is not None else 1.0,
            "atr_pct": ctx.atr_pct,
            "ma5": ma5,
            "ma10": ma10,
            "ma20": ma20,
            "ma10_trend": ma10_trend,
            "ma20_trend": ma20_trend,
            "recent_high_10": recent_high_10,
            "recent_high_20": recent_high_20,
            "macd_hist": macd_hist,
            "price": ctx.real_price,
            "display": {
                "name": name, "code": code, "price": ctx.real_price, "change_pct": ctx.change_pct,
                "final_score": final, "action_level": "中性", "risk_str": risk_str,
            }
        }
        return "", None, state, final, None, scan_info

    def _build_risk_str(self, ctx, state, final_level):
        labels = []
        if ctx.real_price is not None and ctx.hist_df is not None and not ctx.hist_df.empty:
            d = ctx.hist_df.iloc[-1]
            vol_ratio = safe_ratio(d["volume"], d["vol_ma"], default=None)
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
        if ctx.profit_pct_from_low is not None and ctx.profit_pct_from_low <= 0.03:
            labels.append("📉 低位")
        if ctx.is_weak_ma:
            labels.append("🔽 弱于均线")
        if len(state.get("score_history", [])) >= RISK_WARNING_DAYS:
            recent_scores = [s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]]
            if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
                labels.append("🛑 连续低分")
        return " ".join(labels)