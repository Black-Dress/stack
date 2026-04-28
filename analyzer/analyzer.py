#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心分析模块：深度融合AI，增加参数建议、个股权重微调、批量评论/止盈。
（批处理入口已迁移至 main.py）
"""
import datetime
import logging
import numpy as np
import pandas as pd
import math
import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from .config import *
from .utils import (
    calc_rsi,
    calc_macd,
    calculate_atr,
    calculate_adx,
    pad_display,
    validate_and_filter_weights,
    sigmoid_normalize,
    nonlinear_score_transform,
    clip_env_factor,
    cap,
    weighted_sum,
    get_dynamic_confirm_days,
    get_dynamic_history_days,
    fallback_market_state,
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
)
from .ai import AIClient

logger = logging.getLogger(__name__)

# 显示列宽（可能由外部（main.py）使用，但保留在 analyzer 内无妨）
DISPLAY_NAME_WIDTH = 14
DISPLAY_CODE_WIDTH = 12
DISPLAY_PRICE_WIDTH = 8
DISPLAY_CHANGE_WIDTH = 8
DISPLAY_SCORE_WIDTH = 6
DISPLAY_ACTION_WIDTH = 16

DETAIL_COL_NAME = 25
DETAIL_COL_STRENGTH = 8
DETAIL_COL_WEIGHT = 8
DETAIL_COL_CONTRIB = 8


@dataclass
class ETFContext:
    """单只 ETF 完整分析上下文"""
    code: str
    name: str
    real_price: Optional[float]
    hist_df: Optional[pd.DataFrame]
    weekly_df: Optional[pd.DataFrame]
    today: datetime.date
    market: Dict[str, Any]
    params: Dict[str, Any]

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
    effective_profit_threshold: float = TAKE_PROFIT_WARNING_THRESHOLD
    rsi: float = 50.0
    recent_high_price: float = 0.0
    trailing_profit_level: Optional[str] = None
    profit_level: Optional[str] = None


class DataAnalyzer:
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

    def __init__(self, buy_weights=None, sell_weights=None, params=None):
        self.buy_weights = buy_weights or DEFAULT_BUY_WEIGHTS.copy()
        self.sell_weights = sell_weights or DEFAULT_SELL_WEIGHTS.copy()
        self.params = params or DEFAULT_PARAMS.copy()
        self.market_info = {}
        self._indicator_cache = {}
        self.ai_params_advice = None

    def set_market_info(self, market_info):
        self.market_info = market_info

    def set_weights(self, buy_w, sell_w):
        self.buy_weights = buy_w
        self.sell_weights = sell_w

    def set_ai_params_advice(self, advice: dict):
        self.ai_params_advice = advice

    def _apply_ai_params_advice(self, base_params: dict) -> dict:
        if not self.ai_params_advice or not AI_PARAMS_ADVISE:
            return base_params
        try:
            trust = AI_PARAMS_ADVISE_TRUST
            params = base_params.copy()
            buy_shift = self.ai_params_advice.get("buy_threshold_shift", 0)
            params["BUY_THRESHOLD"] = max(0.35, min(0.65, params["BUY_THRESHOLD"] + buy_shift * trust))
            sell_shift = self.ai_params_advice.get("sell_threshold_shift", 0)
            params["SELL_THRESHOLD"] = max(-0.4, min(-0.1, params["SELL_THRESHOLD"] + sell_shift * trust))
            days_shift = self.ai_params_advice.get("confirm_days_shift", 0)
            params["CONFIRM_DAYS"] = int(max(MIN_CONFIRM_DAYS, min(MAX_CONFIRM_DAYS, params["CONFIRM_DAYS"] + days_shift * trust)))
            return params
        except Exception:
            return base_params

    @staticmethod
    def _nonlinear_score_transform(raw, market_status="震荡偏弱"):
        return nonlinear_score_transform(raw, market_status, NONLINEAR_SCALE_BULL, NONLINEAR_SCALE_RANGE)

    def calc_change_pct(self, real_price, hist_df, today):
        if real_price is None or hist_df is None or hist_df.empty:
            return 0.0
        finished_bars = hist_df[hist_df.index.date < today]
        if not finished_bars.empty:
            base_close = finished_bars.iloc[-1]["close"]
        else:
            base_close = hist_df.iloc[-1]["close"]
        return (real_price - base_close) / base_close * 100 if base_close > 0 else 0.0

    def _get_cache_key(self, code, start_date, end_date):
        today_str = datetime.date.today().strftime("%Y%m%d")
        raw = f"{code}_{start_date}_{end_date}_{today_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def calculate_indicators(self, df, need_amount_ma=True,
                             recent_high_window=10, recent_low_window=20,
                             use_cache=False, cache_key=None):
        if use_cache and cache_key and cache_key in self._indicator_cache:
            cached_df, cache_time = self._indicator_cache[cache_key]
            if (time.time() - cache_time) < CACHE_EXPIRE_SECONDS:
                return cached_df.copy()

        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=ETF_MA).mean()
        df["vol_ma"] = df["volume"].rolling(window=ETF_VOL_MA).mean()
        if need_amount_ma:
            df["amount_ma"] = df["amount"].rolling(window=ETF_VOL_MA).mean()
        df["ma30"] = df["close"].rolling(window=MA30_WINDOW).mean()

        df["macd_dif"], df["macd_dea"], _ = calc_macd(df["close"])

        low_n = df["low"].rolling(KDJ_N).min()
        high_n = df["high"].rolling(KDJ_N).max()
        rsv = (df["close"] - low_n) / (high_n - low_n) * 100
        df["kdj_k"] = rsv.ewm(alpha=1/3, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(alpha=1/3, adjust=False).mean()

        df["boll_mid"] = df["close"].rolling(BOLL_WINDOW).mean()
        df["boll_std"] = df["close"].rolling(BOLL_WINDOW).std()
        df["boll_up"] = df["boll_mid"] + BOLL_STD_MULT * df["boll_std"]
        df["boll_low"] = df["boll_mid"] - BOLL_STD_MULT * df["boll_std"]

        high_14 = df["high"].rolling(WILLIAMS_WINDOW).max()
        low_14 = df["low"].rolling(WILLIAMS_WINDOW).min()
        df["williams_r"] = (high_14 - df["close"]) / (high_14 - low_14) * -100

        df["rsi"] = calc_rsi(df["close"])
        df["atr"] = calculate_atr(df, ATR_PERIOD)
        adx_df = calculate_adx(df)
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]
        df["adx"] = adx_df["adx"]

        df["downside_momentum_raw"] = np.where(
            (df["close"] < df["ma_short"]) & (df["minus_di"] > df["plus_di"]),
            (df["ma_short"] - df["close"]) / df["ma_short"] * (df["volume"] / df["vol_ma"]).clip(0, 3), 0)

        df[f"recent_high_{recent_high_window}"] = df["high"].rolling(recent_high_window).max()
        df[f"recent_low_{recent_low_window}"] = df["low"].rolling(recent_low_window).min()

        if use_cache and cache_key:
            self._indicator_cache[cache_key] = (df.copy(), time.time())
        return df

    def compute_dynamic_trust(self, ai_weights, default_weights):
        base_trust = 0.75
        zero_count = sum(1 for v in ai_weights.values() if v < 0.01)
        if zero_count > len(ai_weights) * 0.3:
            base_trust = min(base_trust, 0.5)
        max_w = max(ai_weights.values())
        if max_w > 0.45:
            base_trust = min(base_trust, 0.6)
        dot = sum(ai_weights.get(k, 0) * default_weights.get(k, 0) for k in default_weights)
        norm_ai = math.sqrt(sum(v**2 for v in ai_weights.values()))
        norm_def = math.sqrt(sum(v**2 for v in default_weights.values()))
        if norm_ai > 0 and norm_def > 0 and (similarity := dot / (norm_ai * norm_def)) < 0.6:
            base_trust = min(base_trust, 0.55)
        return base_trust

    def blend_weights(self, ai_w, def_w, trust):
        blended = {k: ai_w.get(k, 0) * trust + def_w[k] * (1 - trust) for k in def_w}
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()} if total > 0 else blended

    def apply_correlation_penalty(self, weights, factor_names, corr_matrix, penalty_threshold=0.7):
        w = weights.copy()
        high_pairs = []
        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i+1:]:
                if corr_matrix.loc[f1, f2] > penalty_threshold and w.get(f1, 0) > 0.12 and w.get(f2, 0) > 0.12:
                    high_pairs.append((f1, f2))
        for f1, f2 in high_pairs:
            w[f1] *= 0.8
            w[f2] *= 0.8
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else w

    def compute_factor_correlation(self, df, factor_names):
        return pd.DataFrame(np.eye(len(factor_names)), index=factor_names, columns=factor_names)

    def generate_ai_weights(self, ai_client, market_state, sentiment,
                            market_above_ma20, market_above_ma60, market_amount_above_ma20, volatility):
        ai_buy, ai_sell = ai_client.generate_weights(market_state, sentiment,
                                                     market_above_ma20, market_above_ma60,
                                                     market_amount_above_ma20, volatility)
        trust = min(self.compute_dynamic_trust(ai_buy, DEFAULT_BUY_WEIGHTS),
                    self.compute_dynamic_trust(ai_sell, DEFAULT_SELL_WEIGHTS))
        corr_buy = self.compute_factor_correlation(None, BUY_FACTOR_NAMES)
        corr_sell = self.compute_factor_correlation(None, SELL_FACTOR_NAMES)
        ai_buy = self.apply_correlation_penalty(ai_buy, BUY_FACTOR_NAMES, corr_buy)
        ai_sell = self.apply_correlation_penalty(ai_sell, SELL_FACTOR_NAMES, corr_sell)
        buy_w = self.blend_weights(ai_buy, DEFAULT_BUY_WEIGHTS, trust)
        sell_w = self.blend_weights(ai_sell, DEFAULT_SELL_WEIGHTS, trust)
        return buy_w, sell_w

    def adjust_weights_for_etf(self, ai_client: AIClient, ctx: ETFContext) -> Tuple[Dict, Dict]:
        if not ai_client or not AI_PER_ETF_WEIGHT_ADJUST:
            return self.buy_weights, self.sell_weights
        features = {
            "tmsv": ctx.tmsv,
            "atr_pct": ctx.atr_pct,
            "rsi": ctx.rsi,
            "above_ma30": ctx.above_ma30,
            "change_5d": (ctx.real_price / ctx.hist_df.iloc[-5]["close"] - 1) if ctx.hist_df is not None and len(ctx.hist_df) >= 5 else 0,
        }
        try:
            new_buy, new_sell = ai_client.adjust_weights_per_etf(self.buy_weights, self.sell_weights, features)
            return new_buy, new_sell
        except Exception as e:
            logger.warning(f"个股权重微调失败，使用全局权重: {e}")
            return self.buy_weights, self.sell_weights

    def _compute_factors(self, ctx, d):
        price = ctx.real_price
        ma20 = d["ma_short"]
        volume = d["volume"]
        vol_ma = d["vol_ma"]
        rsi = d["rsi"]
        boll_up = d["boll_up"]
        boll_low = d["boll_low"]
        williams_r = d["williams_r"]

        macd_golden = kdj_golden = 0
        if len(ctx.hist_df) >= 2:
            prev = ctx.hist_df.iloc[-2]
            macd_golden = 1 if (d["macd_dif"] > d["macd_dea"] and prev["macd_dif"] <= prev["macd_dea"]) else 0
            kdj_golden = 1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0

        ret_etf_5d = (price / ctx.hist_df.iloc[-5]["close"] - 1) if len(ctx.hist_df) >= 5 else 0
        ret_market_5d = ctx.market.get("ret_market_5d", 0)
        mkt_ma20 = ctx.market.get("market_above_ma20", False)
        mkt_ma60 = ctx.market.get("market_above_ma60", False)
        mkt_amt = ctx.market.get("market_amount_above_ma20", False)

        buy_factors = {
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
        }

        sell_factors = {
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
        return buy_factors, sell_factors

    def get_action(self, score, score_history, params, atr_pct=None) -> Tuple[str, str]:
        hist_scores = [s["score"] for s in score_history]
        buy_thresh = params["BUY_THRESHOLD"]
        sell_thresh = params["SELL_THRESHOLD"]

        def _get_level(s):
            for th, level in zip(ACTION_LEVEL_THRESHOLDS, ACTION_LEVEL_NAMES):
                if s >= th:
                    return level
            return "强烈卖出"

        action_level = _get_level(score)

        if len(hist_scores) < 2:
            if score > buy_thresh: return "BUY", action_level
            elif score < sell_thresh: return "SELL", action_level
            else: return "HOLD", action_level

        confirm_days = get_dynamic_confirm_days(atr_pct, params["CONFIRM_DAYS"])
        window = get_dynamic_history_days(atr_pct) if atr_pct else 12
        window = min(window, len(hist_scores))
        recent = hist_scores[-window:]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        up_days = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i-1])
        down_days = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i-1])

        if score > buy_thresh:
            if score > params["QUICK_BUY_THRESHOLD"] and slope > SIGNAL_SLOPE_BUY_THRESH and avg > buy_thresh + SIGNAL_AVG_OFFSET:
                return "BUY", action_level
            if len(hist_scores) >= confirm_days and all(s > buy_thresh for s in hist_scores[-confirm_days:]) and slope >= 0:
                return "BUY", action_level
            if down_days >= 2 and slope > SIGNAL_SLOPE_WEAK and score > avg + SIGNAL_AVG_OFFSET:
                return "BUY", action_level
            return "PREP_BUY", action_level

        if score < sell_thresh:
            if score < sell_thresh - SIGNAL_AVG_OFFSET and slope < SIGNAL_SELL_SLOPE and avg < sell_thresh - SIGNAL_AVG_OFFSET:
                return "SELL", action_level
            if len(hist_scores) >= confirm_days and all(s < sell_thresh for s in hist_scores[-confirm_days:]) and slope <= 0:
                return "SELL", action_level
            if up_days >= 2 and slope < SIGNAL_SELL_WEAK_SLOPE and score < avg - SIGNAL_AVG_OFFSET:
                return "SELL", action_level
            return "PREP_SELL", action_level

        if atr_pct and atr_pct > VOL_HIGH_CONFIRM:
            if score > buy_thresh + 0.15 and slope > SIGNAL_HIGH_VOL_BUY_SLOPE and up_days >= SIGNAL_HIGH_VOL_DAYS:
                return "BUY", action_level
            if score < sell_thresh - 0.05 and slope < -0.08 and down_days >= 3:
                return "SELL", action_level
            return "HOLD", action_level
        if atr_pct and atr_pct > VOL_MID_CONFIRM:
            if score > buy_thresh + 0.1 and slope > SIGNAL_MID_VOL_BUY_SLOPE and up_days >= SIGNAL_MID_VOL_DAYS:
                return "BUY", action_level
            if score < sell_thresh - 0.1 and slope < -0.08 and down_days >= 3:
                return "SELL", action_level
        return "HOLD", action_level

    def adjust_params_based_on_history(self, params, score_history, volatility, market_factor):
        if len(score_history) < 10:
            return params
        window = min(get_dynamic_history_days(volatility), len(score_history))
        recent = [s["score"] for s in score_history[-window:]]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        short_recent = [s["score"] for s in score_history[-min(3, len(score_history)):]]
        short_slope = np.polyfit(range(len(short_recent)), short_recent, 1)[0] if len(short_recent) >= 2 else 0
        adjust_mult = ADJUST_MULT_BASE / market_factor
        adjusted = params.copy()

        if slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = min(ADJUST_BUY_DELTA_MAX * 0.5, abs(avg - params["BUY_THRESHOLD"]) * 0.1) * adjust_mult
            adjusted["BUY_THRESHOLD"] = max(0.45, params["BUY_THRESHOLD"] - delta)
        elif slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = min(ADJUST_BUY_DELTA_MAX * 0.5, abs(avg - params["BUY_THRESHOLD"]) * 0.1) * adjust_mult
            adjusted["BUY_THRESHOLD"] = min(0.55, params["BUY_THRESHOLD"] + delta)
        else:
            adjusted["BUY_THRESHOLD"] = params["BUY_THRESHOLD"] * 0.95 + DEFAULT_PARAMS["BUY_THRESHOLD"] * 0.05

        if slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = min(ADJUST_SELL_DELTA_MAX * 0.5, abs(avg - params["SELL_THRESHOLD"]) * 0.1) * adjust_mult
            adjusted["SELL_THRESHOLD"] = max(-0.35, params["SELL_THRESHOLD"] - delta)
        elif slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = min(ADJUST_SELL_DELTA_MAX * 0.5, abs(avg - params["SELL_THRESHOLD"]) * 0.1) * adjust_mult
            adjusted["SELL_THRESHOLD"] = min(-0.15, params["SELL_THRESHOLD"] + delta)
        else:
            adjusted["SELL_THRESHOLD"] = params["SELL_THRESHOLD"] * 0.95 + DEFAULT_PARAMS["SELL_THRESHOLD"] * 0.05

        adjusted["CONFIRM_DAYS"] = get_dynamic_confirm_days(volatility, params["CONFIRM_DAYS"])
        return adjusted

    def _evaluate_take_profit(self, ctx: ETFContext) -> None:
        d = ctx.hist_df.iloc[-1]
        recent_high = d.get(f"recent_high_{ctx.params['RECENT_HIGH_WINDOW']}",
                            ctx.hist_df["high"].rolling(ctx.params["RECENT_HIGH_WINDOW"]).max().iloc[-1])
        recent_low = d.get(f"recent_low_{ctx.params['RECENT_LOW_WINDOW']}",
                           ctx.hist_df["low"].rolling(ctx.params["RECENT_LOW_WINDOW"]).min().iloc[-1])

        ctx.recent_high_price = recent_high
        ctx.recent_low_price = recent_low
        ctx.max_drawdown_pct = (recent_high - ctx.real_price) / recent_high if recent_high > 0 else 0.0

        ctx.trailing_profit_level = get_trailing_profit_signals(ctx.real_price, recent_high, ctx.atr_pct)

        if recent_low > 0:
            ctx.profit_pct_from_low = (ctx.real_price - recent_low) / recent_low
            ctx.effective_profit_threshold = self._get_effective_take_profit_threshold(ctx)
            if ctx.profit_pct_from_low >= ctx.effective_profit_threshold:
                ctx.should_take_profit = True

            if ctx.profit_pct_from_low >= ctx.effective_profit_threshold * PROFIT_LOW_CLEAR_MULT:
                ctx.profit_level = 'clear'
            elif ctx.profit_pct_from_low >= ctx.effective_profit_threshold * PROFIT_LOW_HALF_MULT:
                ctx.profit_level = 'half'
            elif ctx.profit_pct_from_low >= ctx.effective_profit_threshold * PROFIT_LOW_WATCH_MULT:
                ctx.profit_level = 'watch'
            else:
                ctx.profit_level = None
        else:
            ctx.profit_pct_from_low = 0.0
            ctx.should_take_profit = False
            ctx.profit_level = None

    def _get_effective_take_profit_threshold(self, ctx):
        base = TAKE_PROFIT_WARNING_THRESHOLD
        market_status = ctx.market.get("macro_status", "")
        if "牛" in market_status:
            base *= TAKE_PROFIT_BULL_MULT
        elif "熊" in market_status:
            base *= TAKE_PROFIT_BEAR_MULT
        if ctx.atr_pct > 0.03:
            base *= TAKE_PROFIT_HIGHVOL_MULT
        return base

    def _build_risk_str(self, ctx: ETFContext, state: dict) -> str:
        risk_parts = []
        final = ctx.final_score
        if len(state.get("score_history", [])) >= RISK_WARNING_DAYS:
            recent_scores = [s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]]
            if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
                risk_parts.append(f"连续{RISK_WARNING_DAYS}日低评分")
            elif final < RISK_EXTREME_LOW or final > RISK_EXTREME_HIGH:
                risk_parts.append(f"极端评分 {final:.2f}")
            elif ctx.atr_pct > RISK_HIGH_VOL_THRESH:
                risk_parts.append(f"高波动 {ctx.atr_pct*100:.1f}%")
        tp_detail = []
        if ctx.trailing_profit_level:
            if ctx.recent_high_price > 0 and ctx.real_price:
                fall = (ctx.recent_high_price - ctx.real_price) / ctx.recent_high_price
                tp_detail.append(f"高点回落{fall:.1%}")
        if ctx.profit_level:
            tp_detail.append(f"低点涨{ctx.profit_pct_from_low:.1%}")
        if tp_detail:
            risk_parts.append("止盈风险:" + "，".join(tp_detail))
        if ctx.profit_pct_from_low is not None and ctx.profit_pct_from_low <= 0.03:
            risk_parts.append("低位区间")
        if not ctx.above_ma30:
            risk_parts.append("弱于中期均线")
        return " ".join(risk_parts) if risk_parts else ""

    @staticmethod
    def _adjust_action_level_for_profit(action_level: str, ctx: ETFContext) -> str:
        sell_or_neutral = {"中性偏空", "偏空持有", "谨慎卖出", "卖出", "强烈卖出"}
        if action_level in sell_or_neutral:
            return action_level
        if ctx.trailing_profit_level == 'clear' or ctx.profit_level == 'clear':
            return DataAnalyzer.PROFIT_DOWNGRADE_CLEAR.get(action_level, action_level)
        elif ctx.trailing_profit_level == 'half' or ctx.profit_level == 'half':
            return DataAnalyzer.PROFIT_DOWNGRADE_HALF.get(action_level, action_level)
        return action_level

    def _core_analysis(self, ctx, ai_client=None):
        if ctx.real_price is None:
            ctx.error = "实时价格获取失败"
            return ctx
        if ctx.hist_df is None or len(ctx.hist_df) < 20:
            ctx.error = "历史数据不足"
            return ctx

        ctx.change_pct = self.calc_change_pct(ctx.real_price, ctx.hist_df, ctx.today)
        d = ctx.hist_df.iloc[-1]
        atr_pct = d["atr"] / ctx.real_price if ctx.real_price > 0 else 0
        ctx.atr_pct = atr_pct
        ctx.rsi = d["rsi"]

        if "ma30" in d and not pd.isna(d["ma30"]):
            ctx.above_ma30 = ctx.real_price > d["ma30"]
        else:
            ctx.above_ma30 = True

        ctx.weekly_above = ctx.weekly_below = False
        if ctx.weekly_df is not None and not ctx.weekly_df.empty:
            w = ctx.weekly_df.iloc[-1]
            if "ma_short" in w.index and not pd.isna(w["ma_short"]):
                ctx.weekly_above = w["close"] > w["ma_short"]
                ctx.weekly_below = w["close"] < w["ma_short"]

        market_status = ctx.market.get("macro_status", "震荡偏弱")
        try:
            tmsv_series = compute_tmsv(ctx.hist_df, market_status, atr_pct)
            tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
            tmsv = 50.0 if np.isnan(tmsv) else tmsv
        except Exception:
            logger.error("TMSV 计算失败", exc_info=True)
            tmsv = 50.0
        ctx.tmsv = tmsv
        ctx.tmsv_strength = tmsv / 100.0

        ctx.downside_momentum = d.get("downside_momentum_raw", 0.0)

        self._evaluate_take_profit(ctx)

        if ai_client:
            etf_buy_w, etf_sell_w = self.adjust_weights_for_etf(ai_client, ctx)
        else:
            etf_buy_w, etf_sell_w = self.buy_weights, self.sell_weights

        ctx.buy_factors, ctx.sell_factors = self._compute_factors(ctx, d)
        ctx.buy_score = weighted_sum(ctx.buy_factors, etf_buy_w)
        ctx.sell_score = weighted_sum(ctx.sell_factors, etf_sell_w)

        self._apply_midterm_filter(ctx)
        self._apply_momentum_penalty(ctx)

        max_dd_active = ctx.sell_factors.get("max_drawdown_stop", 0) > 0
        ma_break_active = ctx.sell_factors.get("stop_loss_ma_break", 0) > 0
        ma20 = d.get("ma_short", 0)
        if ma_break_active or (max_dd_active and ctx.real_price < ma20):
            ctx.final_score = -1.0
            ctx.raw_score = ctx.buy_score - ctx.sell_score
            return ctx

        self._apply_sentiment_penalty(ctx)

        ctx.raw_score = ctx.buy_score - ctx.sell_score
        transformed_raw = self._nonlinear_score_transform(ctx.raw_score, market_status)
        env_factor = clip_env_factor(ctx.market["market_factor"], ctx.market["sentiment_factor"])
        ctx.final_score = max(-1.0, min(1.0, transformed_raw * env_factor))
        return ctx

    def _apply_midterm_filter(self, ctx):
        if not ctx.above_ma30:
            ctx.buy_score *= MA30_WEAKNESS_PENALTY

    def _apply_momentum_penalty(self, ctx):
        if ctx.buy_factors.get("macd_golden_cross", 0) == 0 and ctx.buy_factors.get("kdj_golden_cross", 0) == 0:
            ctx.buy_score *= MOMENTUM_MISSING_PENALTY

    def _apply_sentiment_penalty(self, ctx):
        sentiment = ctx.market.get("sentiment_factor", 1.0)
        if sentiment >= SENTIMENT_OVERHEAT_THRESHOLD:
            ctx.buy_score *= SENTIMENT_PENALTY_FACTOR

    def analyze_single_etf(self, code, name, real_price, hist_df, weekly_df,
                        market, today, state, ai_client=None):
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy())
        ctx = self._core_analysis(ctx, ai_client=ai_client)

        if ctx.error:
            if "实时价格" in ctx.error:
                out = (f"{pad_display(name, DISPLAY_NAME_WIDTH)} {pad_display(code, DISPLAY_CODE_WIDTH)} "
                    f"{pad_display('获取失败', DISPLAY_PRICE_WIDTH)} {pad_display('0.00%', DISPLAY_CHANGE_WIDTH, 'right')} "
                    f"{pad_display('0.00', DISPLAY_SCORE_WIDTH, 'right')}  {pad_display('价格缺失', DISPLAY_ACTION_WIDTH)}")
            else:
                price_str = f"{real_price:.3f}" if real_price else "N/A"
                change_str = f"{ctx.change_pct:+.2f}%"
                out = (f"{pad_display(name, DISPLAY_NAME_WIDTH)} {pad_display(code, DISPLAY_CODE_WIDTH)} "
                    f"{pad_display(price_str, DISPLAY_PRICE_WIDTH, 'right')} "
                    f"{pad_display(change_str, DISPLAY_CHANGE_WIDTH, 'right')} "
                    f"{pad_display('0.00', DISPLAY_SCORE_WIDTH, 'right')}  {pad_display('数据不足', DISPLAY_ACTION_WIDTH)}")
            return out, None, state, 0.0

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

        if len(state["score_history"]) >= 7:
            self.params = self.adjust_params_based_on_history(
                self.params, state["score_history"], ctx.atr_pct, market["market_factor"]
            )

        action, action_level = self.get_action(final, state["score_history"], self.params, ctx.atr_pct)
        risk_str = self._build_risk_str(ctx, state)
        adjusted_level = self._adjust_action_level_for_profit(action_level, ctx)

        output = format_etf_output_line(
            name=name,
            code=code,
            price=real_price,
            change_pct=ctx.change_pct,
            final_score=final,
            action_level=adjusted_level,
            atr_pct=ctx.atr_pct,
            recent_high_price=ctx.recent_high_price,
            risk_str=risk_str,
            signal_action=action if action in ("BUY", "SELL") else None,
        )

        signal = None
        if action in ("BUY", "SELL"):
            signal = {"action": action, "name": name, "code": code, "score": final}

        return output, signal, state, final

    def detailed_analysis(self, code, name, real_price, hist_df, weekly_df,
                        market, today, state, ai_client=None, ai_comment_override=None, ai_tp_override=None):
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy())
        ctx = self._core_analysis(ctx, ai_client=ai_client)
        if ctx.error:
            return f"【{name} ({code})】{ctx.error}，无法分析。"

        final = ctx.final_score
        _, action_level = self.get_action(final, state.get("score_history", []), self.params, ctx.atr_pct)

        # 获取 AI 评论（如果不为空）
        ai_comment = ai_comment_override
        if ai_comment is None and ai_client:
            try:
                ai_comment = ai_client.comment_on_etf(code, name, final, action_level,
                                                    market["macro_status"], market["market_factor"],
                                                    market["sentiment_factor"],
                                                    self.buy_weights, self.sell_weights,
                                                    ctx.buy_factors, ctx.sell_factors,
                                                    ctx.tmsv, ctx.atr_pct)
            except Exception as e:
                logger.error(f"AI评论生成失败: {e}")
                ai_comment = "（AI 评论生成失败）"

        ai_tp = ai_tp_override
        if ai_tp is None and ai_client and (ctx.trailing_profit_level or ctx.profit_level):
            try:
                ai_tp = ai_client.take_profit_advice(
                    code, name, ctx.profit_pct_from_low,
                    ctx.recent_low_price, real_price,
                    ctx.tmsv, ctx.rsi, ctx.atr_pct,
                    market["macro_status"], market["sentiment_factor"]
                )
            except Exception as e:
                logger.error(f"AI止盈建议生成失败: {e}")
                ai_tp = "（止盈建议生成失败）"

        # 使用 utils 中的格式化函数生成最终报告
        return format_detailed_report(ctx, market, self.params, ai_comment, ai_tp)
