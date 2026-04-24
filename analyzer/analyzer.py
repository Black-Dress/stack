#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心分析模块：负责技术指标计算、评分、信号确认及批量分析入口。
"""
import datetime
import logging
import re
import time
import numpy as np
import pandas as pd
import math
import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from .config import *
from .utils import (
    pad_display,
    validate_and_filter_weights,
    send_email,
    sigmoid_normalize,
    nonlinear_score_transform,
    apply_sentiment_adjustment,
    clip_env_factor,
)
from .ai import AIClient
from .fetcher import DataFetcher, AKSHARE_AVAILABLE

logger = logging.getLogger(__name__)


# ========================== 数据类 ==========================
@dataclass
class ETFContext:
    """单只 ETF 的完整分析上下文，承载所有中间计算结果。"""

    code: str
    name: str
    real_price: Optional[float]
    hist_df: Optional[pd.DataFrame]
    weekly_df: Optional[pd.DataFrame]
    today: datetime.date
    market: Dict[str, Any]  # 市场环境信息
    params: Dict[str, Any]  # 动态参数

    change_pct: float = 0.0  # 当日涨跌幅
    atr_pct: float = 0.0  # ATR/价格
    tmsv: float = 50.0  # TMSV复合强度值
    tmsv_strength: float = 0.5  # TMSV归一化强度 0-1
    downside_momentum: float = 0.0  # 下跌动量
    max_drawdown_pct: float = 0.0  # 近期最大回撤百分比
    weekly_above: bool = False  # 周线站上20均线
    weekly_below: bool = False  # 周线跌破20均线
    buy_factors: Dict = field(default_factory=dict)  # 买入因子强度
    sell_factors: Dict = field(default_factory=dict)  # 卖出因子强度
    buy_score: float = 0.0  # 加权买入总分
    sell_score: float = 0.0  # 加权卖出总分
    raw_score: float = 0.0  # 原始净分（买入-卖出）
    final_score: float = 0.0  # 最终评分（非线性变换+环境因子）
    error: Optional[str] = None  # 错误信息
    # 新增止盈相关字段
    recent_low_price: float = 0.0
    profit_pct_from_low: float = 0.0
    should_take_profit: bool = False


# ========================== 公共指标计算 ==========================
def calc_rsi(series: pd.Series, period: int = RSI_WINDOW) -> pd.Series:
    """计算 RSI 指标"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


def calc_macd(
    series: pd.Series,
    fast: int = MACD_FAST,
    slow: int = MACD_SLOW,
    signal: int = MACD_SIGNAL,
):
    """计算 MACD 相关序列，返回 (DIF, DEA, 柱)"""
    exp_fast = series.ewm(span=fast, adjust=False).mean()
    exp_slow = series.ewm(span=slow, adjust=False).mean()
    dif = exp_fast - exp_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


# ========================== 核心分析类 ==========================
class DataAnalyzer:
    """主要分析引擎，包含技术指标计算、因子评分、信号判定、权重管理等功能。"""

    def __init__(
        self, buy_weights: Dict = None, sell_weights: Dict = None, params: Dict = None
    ):
        self.buy_weights = buy_weights or DEFAULT_BUY_WEIGHTS.copy()
        self.sell_weights = sell_weights or DEFAULT_SELL_WEIGHTS.copy()
        self.params = params or DEFAULT_PARAMS.copy()
        self.market_info = {}
        self._indicator_cache = {}  # 指标缓存：{(code, start, end): (df, timestamp)}

    def set_market_info(self, market_info: Dict):
        """设置当前市场环境信息"""
        self.market_info = market_info

    def set_weights(self, buy_w: Dict, sell_w: Dict):
        """更新买卖权重"""
        self.buy_weights = buy_w
        self.sell_weights = sell_w

    # ---------- 非线性缩放 ----------
    def _nonlinear_score_transform(
        self, raw: float, market_status: str = "震荡偏弱"
    ) -> float:
        """对原始净分进行 tanh 非线性变换"""
        return nonlinear_score_transform(
            raw, market_status, NONLINEAR_SCALE_BULL, NONLINEAR_SCALE_RANGE
        )

    # ---------- TMSV 动态权重 ----------
    def _get_tmsv_weights(
        self, market_status: str, volatility: float
    ) -> Dict[str, float]:
        """
        根据市场状态和波动率动态调整 TMSV 的子项权重（趋势、动量、成交量）

        Args:
            market_status: 市场状态文本（含“牛”、“熊”或震荡）
            volatility: 当前波动率（ATR/价格）

        Returns:
            包含 trend/momentum/volume 权重的字典
        """
        if "牛" in market_status:
            w = {"trend": 0.40, "momentum": 0.25, "volume": 0.15}
        elif "熊" in market_status:
            w = {"trend": 0.20, "momentum": 0.30, "volume": 0.25}
        else:
            w = {"trend": 0.25, "momentum": 0.40, "volume": 0.15}

        if volatility > TMSV_HIGH_VOL_THRESH:
            w["trend"] = max(TMSV_MIN_TREND_WEIGHT, w["trend"] - TMSV_TREND_REDUCE)
            w["momentum"] += TMSV_TREND_REDUCE
        return w

    # ---------- 涨跌幅计算 ----------
    def calc_change_pct(
        self, real_price: float, hist_df: pd.DataFrame, today: datetime.date
    ) -> float:
        """计算当日涨跌幅（相对于前一交易日收盘价）"""
        if real_price is None or hist_df is None or len(hist_df) < 1:
            return 0.0
        last_date = hist_df.index[-1].date()
        # 若最后日期为今日，则用前一日收盘作为基准，否则用最后一日收盘
        if last_date == today and len(hist_df) >= 2:
            base_close = hist_df.iloc[-2]["close"]
        else:
            base_close = hist_df.iloc[-1]["close"]
        return (real_price - base_close) / base_close * 100 if base_close > 0 else 0.0

    # ---------- 技术指标计算 ----------
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算平均真实波幅 ATR"""
        tr = pd.concat(
            [
                df["high"] - df["low"],
                abs(df["high"] - df["close"].shift()),
                abs(df["low"] - df["close"].shift()),
            ],
            axis=1,
        ).max(1)
        return tr.rolling(period).mean()

    def calculate_adx(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """计算 ADX 及 +DI、-DI"""
        high, low, close = df["high"], df["low"], df["close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        tr = self.calculate_atr(df, 1)
        atr = tr.rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        return pd.DataFrame(
            {"plus_di": plus_di, "minus_di": minus_di, "adx": adx}, index=df.index
        )

    def _get_cache_key(self, code: str, start_date: str, end_date: str) -> str:
        """生成指标缓存的 key（基于代码和日期，并按天变化）"""
        today_str = datetime.date.today().strftime("%Y%m%d")
        raw = f"{code}_{start_date}_{end_date}_{today_str}"
        return hashlib.md5(raw.encode()).hexdigest()

    def calculate_indicators(
        self,
        df: pd.DataFrame,
        need_amount_ma: bool = True,
        recent_high_window: int = 10,
        recent_low_window: int = 20,
        use_cache: bool = False,
        cache_key: str = None,
    ) -> pd.DataFrame:
        """
        对原始 OHLCV 数据批量计算所有技术指标（MA、MACD、KDJ、布林、RSI、ATR、ADX 等）

        Args:
            df: 日线数据 DataFrame
            need_amount_ma: 是否计算成交额均线（大盘需要）
            recent_high_window: 近期高点的窗口
            recent_low_window: 近期低点的窗口
            use_cache: 是否使用缓存
            cache_key: 缓存键

        Returns:
            添加了所有技术指标列的 DataFrame
        """
        # 检查缓存
        if use_cache and cache_key and cache_key in self._indicator_cache:
            cached_df, cache_time = self._indicator_cache[cache_key]
            if (time.time() - cache_time) < CACHE_EXPIRE_SECONDS:
                return cached_df.copy()

        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=ETF_MA).mean()
        df["vol_ma"] = df["volume"].rolling(window=ETF_VOL_MA).mean()
        if need_amount_ma:
            df["amount_ma"] = df["amount"].rolling(window=ETF_VOL_MA).mean()
        df["macd_dif"], df["macd_dea"], _ = calc_macd(df["close"])
        # KDJ 计算
        low_n = df["low"].rolling(KDJ_N).min()
        high_n = df["high"].rolling(KDJ_N).max()
        rsv = (df["close"] - low_n) / (high_n - low_n) * 100
        df["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
        # 布林带
        df["boll_mid"] = df["close"].rolling(BOLL_WINDOW).mean()
        df["boll_std"] = df["close"].rolling(BOLL_WINDOW).std()
        df["boll_up"] = df["boll_mid"] + BOLL_STD_MULT * df["boll_std"]
        df["boll_low"] = df["boll_mid"] - BOLL_STD_MULT * df["boll_std"]
        # 威廉指标
        high_14 = df["high"].rolling(WILLIAMS_WINDOW).max()
        low_14 = df["low"].rolling(WILLIAMS_WINDOW).min()
        df["williams_r"] = (high_14 - df["close"]) / (high_14 - low_14) * -100
        df["rsi"] = calc_rsi(df["close"])
        df["atr"] = self.calculate_atr(df, ATR_PERIOD)
        adx_df = self.calculate_adx(df)
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]
        df["adx"] = adx_df["adx"]
        # 下跌动量（价格低于短期均线且 -DI > +DI 时放大）
        df["downside_momentum_raw"] = np.where(
            (df["close"] < df["ma_short"]) & (df["minus_di"] > df["plus_di"]),
            (df["ma_short"] - df["close"])
            / df["ma_short"]
            * (df["volume"] / df["vol_ma"]).clip(0, 3),
            0,
        )
        df[f"recent_high_{recent_high_window}"] = (
            df["high"].rolling(recent_high_window).max()
        )
        df[f"recent_low_{recent_low_window}"] = (
            df["low"].rolling(recent_low_window).min()
        )

        if use_cache and cache_key:
            self._indicator_cache[cache_key] = (df.copy(), time.time())
        return df

    def compute_tmsv(
        self,
        df: pd.DataFrame,
        market_status: str = "震荡偏弱",
        volatility: float = 0.02,
    ) -> pd.Series:
        """
        计算 TMSV 复合强度指标（趋势、动量、成交量加权合成，经波动率因子修正）

        Args:
            df: 日线数据（至少包含 OHLCV）
            market_status: 市场状态
            volatility: 波动率

        Returns:
            TMSV 序列 (0-100)
        """
        if df is None or len(df) < 20:
            return (
                pd.Series([50.0] * max(1, len(df)))
                if len(df) > 0
                else pd.Series([50.0])
            )

        df = df.copy()
        # 确保所需列存在
        if "ma20" not in df.columns:
            df["ma20"] = df["close"].rolling(TMSV_MA20_WINDOW).mean()
        if "ma60" not in df.columns:
            df["ma60"] = df["close"].rolling(TMSV_MA60_WINDOW).mean()
        if "rsi" not in df.columns:
            df["rsi"] = calc_rsi(df["close"])
        if "macd_hist" not in df.columns:
            _, _, df["macd_hist"] = calc_macd(df["close"])
        if "atr" not in df.columns:
            df["atr"] = self.calculate_atr(df, TMSV_ATR_WINDOW)
        if "vol_ma" not in df.columns:
            df["vol_ma"] = df["volume"].rolling(TMSV_VOL_MA_WINDOW).mean()

        # 趋势得分：价格与均线偏离度 + 均线斜率
        price_above_ma20 = (
            ((df["close"] - df["ma20"]) / (df["ma20"].replace(0, np.nan) * 0.1))
            .clip(0, 1)
            .fillna(0)
        )
        price_above_ma60 = (
            ((df["close"] - df["ma60"]) / (df["ma60"].replace(0, np.nan) * 0.1))
            .clip(0, 1)
            .fillna(0)
        )
        ma20_slope = df["ma20"].diff(5) / df["ma20"].shift(5).replace(0, np.nan)
        slope_score = (ma20_slope * 10).clip(0, 1).fillna(0)
        trend_score = (
            price_above_ma20 * TMSV_TREND_MA20_WEIGHT
            + price_above_ma60 * TMSV_TREND_MA60_WEIGHT
            + slope_score * TMSV_TREND_SLOPE_WEIGHT
        ) * 100

        # 动量得分：RSI 和 MACD 柱变化
        rsi_score = ((df["rsi"] - 50) * 3.33).clip(0, 100).fillna(50)
        macd_change = df["macd_hist"].diff() / (df["macd_hist"].shift(1).abs() + 0.001)
        macd_score = (macd_change * 100).clip(0, 100).fillna(50)
        mom_score = rsi_score * TMSV_MOM_RSI_WEIGHT + macd_score * TMSV_MOM_MACD_WEIGHT

        # 成交量得分：量比 + 价量一致性
        vol_ratio = df["volume"] / df["vol_ma"].replace(0, np.nan)
        vol_ratio_score = ((vol_ratio - 0.8) / 1.2 * 100).clip(0, 100).fillna(50)
        price_up = df["close"] > df["close"].shift(1)
        vol_up = df["volume"] > df["vol_ma"]
        consistency = np.where(price_up == vol_up, 100, 0)
        vol_score = (
            vol_ratio_score * TMSV_VOL_RATIO_WEIGHT
            + consistency * TMSV_VOL_CONSIST_WEIGHT
        )

        # 波动率修正因子
        atr_pct = df["atr"] / df["close"].replace(0, np.nan)
        vol_factor = np.select(
            [atr_pct < TMSV_VOL_LOW_THRESH, atr_pct > TMSV_VOL_HIGH_THRESH],
            [TMSV_VOL_LOW_FACTOR, TMSV_VOL_HIGH_FACTOR],
            default=TMSV_VOL_MID_FACTOR_BASE
            - (atr_pct - TMSV_VOL_LOW_THRESH)
            / TMSV_VOL_BAND_WIDTH
            * TMSV_VOL_MID_FACTOR_SLOPE,
        )
        vol_factor = np.nan_to_num(vol_factor, nan=1.0)

        w = self._get_tmsv_weights(market_status, volatility)
        tmsv = (
            trend_score * w["trend"]
            + mom_score * w["momentum"]
            + vol_score * w["volume"]
        ) * vol_factor
        return tmsv.clip(0, 100).fillna(50)

    # ---------- 权重处理 ----------
    def compute_dynamic_trust(self, ai_weights: Dict, default_weights: Dict) -> float:
        """
        计算 AI 权重的信任度（0~1），综合考量零权重比例、单因子集中度、与默认权重相似度

        Args:
            ai_weights: AI 生成的权重
            default_weights: 默认权重

        Returns:
            信任度系数
        """
        base_trust = 0.75
        zero_count = sum(1 for v in ai_weights.values() if v < 0.01)
        if zero_count > len(ai_weights) * 0.3:
            base_trust = min(base_trust, 0.5)
        max_w = max(ai_weights.values())
        if max_w > 0.45:
            base_trust = min(base_trust, 0.6)
        dot = sum(
            ai_weights.get(k, 0) * default_weights.get(k, 0) for k in default_weights
        )
        norm_ai = math.sqrt(sum(v**2 for v in ai_weights.values()))
        norm_def = math.sqrt(sum(v**2 for v in default_weights.values()))
        if (
            norm_ai > 0
            and norm_def > 0
            and (similarity := dot / (norm_ai * norm_def)) < 0.6
        ):
            base_trust = min(base_trust, 0.55)
        return base_trust

    def blend_weights(self, ai_w: Dict, def_w: Dict, trust: float) -> Dict:
        """
        按信任度混合 AI 权重和默认权重

        Args:
            ai_w: AI 权重
            def_w: 默认权重
            trust: 信任度

        Returns:
            混合并归一化后的权重
        """
        blended = {k: ai_w.get(k, 0) * trust + def_w[k] * (1 - trust) for k in def_w}
        total = sum(blended.values())
        return {k: v / total for k, v in blended.items()} if total > 0 else blended

    def apply_correlation_penalty(
        self,
        weights: Dict,
        factor_names: List[str],
        corr_matrix: pd.DataFrame,
        penalty_threshold: float = 0.7,
    ) -> Dict:
        """
        对高度相关的因子同时给高权重的情况施加惩罚（降低权重）

        Args:
            weights: 原始权重
            factor_names: 因子名列表
            corr_matrix: 因子相关性矩阵
            penalty_threshold: 相关性阈值

        Returns:
            惩罚后的权重
        """
        w = weights.copy()
        high_pairs = []
        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i + 1 :]:
                if (
                    corr_matrix.loc[f1, f2] > penalty_threshold
                    and w.get(f1, 0) > 0.12
                    and w.get(f2, 0) > 0.12
                ):
                    high_pairs.append((f1, f2))
        for f1, f2 in high_pairs:
            w[f1] *= 0.8
            w[f2] *= 0.8
        total = sum(w.values())
        return {k: v / total for k, v in w.items()} if total > 0 else w

    def compute_factor_correlation(
        self, df: pd.DataFrame, factor_names: List[str]
    ) -> pd.DataFrame:
        """计算因子之间的相关性矩阵（当前简化为单位矩阵）"""
        # 实际项目中可根据历史因子序列计算
        return pd.DataFrame(
            np.eye(len(factor_names)), index=factor_names, columns=factor_names
        )

    # ---------- AI 权重生成 ----------
    def generate_ai_weights(
        self,
        ai_client: AIClient,
        market_state: str,
        sentiment: float,
        market_above_ma20: bool,
        market_above_ma60: bool,
        market_amount_above_ma20: bool,
        volatility: float,
    ) -> Tuple[Dict, Dict]:
        """
        生成经过混合、惩罚及情绪调整的最终买卖权重

        Args:
            ai_client: AI 客户端
            market_state: 市场状态
            sentiment: 情绪因子
            market_above_ma20: 大盘在20日线上
            market_above_ma60: 大盘在60日线上
            market_amount_above_ma20: 成交额在20日均额上
            volatility: 波动率

        Returns:
            (最终买入权重, 最终卖出权重)
        """
        ai_buy, ai_sell = ai_client.generate_weights(
            market_state,
            sentiment,
            market_above_ma20,
            market_above_ma60,
            market_amount_above_ma20,
            volatility,
        )
        trust = min(
            self.compute_dynamic_trust(ai_buy, DEFAULT_BUY_WEIGHTS),
            self.compute_dynamic_trust(ai_sell, DEFAULT_SELL_WEIGHTS),
        )
        corr_buy = self.compute_factor_correlation(
            None, list(DEFAULT_BUY_WEIGHTS.keys())
        )
        corr_sell = self.compute_factor_correlation(
            None, list(DEFAULT_SELL_WEIGHTS.keys())
        )
        ai_buy = self.apply_correlation_penalty(
            ai_buy, list(DEFAULT_BUY_WEIGHTS.keys()), corr_buy
        )
        ai_sell = self.apply_correlation_penalty(
            ai_sell, list(DEFAULT_SELL_WEIGHTS.keys()), corr_sell
        )
        buy_w = self.blend_weights(ai_buy, DEFAULT_BUY_WEIGHTS, trust)
        sell_w = self.blend_weights(ai_sell, DEFAULT_SELL_WEIGHTS, trust)
            
        return buy_w, sell_w

    # ---------- 因子计算 ----------
    def _compute_factors(self, ctx: ETFContext, d: pd.Series) -> Tuple[Dict, Dict]:
        """
        根据 ETF 的实时状态和最后一根 K 线指标，计算所有买入因子和卖出因子的强度 (0~1)

        Args:
            ctx: ETF 上下文
            d: 最后一根日线数据（包含技术指标）

        Returns:
            (买入因子字典, 卖出因子字典)
        """
        price = ctx.real_price
        ma20 = d["ma_short"]
        volume = d["volume"]
        vol_ma = d["vol_ma"]
        rsi = d["rsi"]
        boll_up = d["boll_up"]
        boll_low = d["boll_low"]
        williams_r = d["williams_r"]

        # 金叉/死叉判断（需要前一交易日数据）
        macd_golden = kdj_golden = 0
        if len(ctx.hist_df) >= 2:
            prev = ctx.hist_df.iloc[-2]
            macd_golden = (
                1
                if (
                    d["macd_dif"] > d["macd_dea"]
                    and prev["macd_dif"] <= prev["macd_dea"]
                )
                else 0
            )
            kdj_golden = (
                1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0
            )

        # 近期收益比较
        ret_etf_5d = (
            (price / ctx.hist_df.iloc[-5]["close"] - 1) if len(ctx.hist_df) >= 5 else 0
        )
        ret_market_5d = ctx.market.get("ret_market_5d", 0)
        weekly_above = ctx.weekly_above
        weekly_below = ctx.weekly_below
        recent_high = d.get(
            f"recent_high_{ctx.params['RECENT_HIGH_WINDOW']}",
            ctx.hist_df["high"]
            .rolling(ctx.params["RECENT_HIGH_WINDOW"])
            .max()
            .iloc[-1],
        )
        recent_low = d.get(
            f"recent_low_{ctx.params['RECENT_LOW_WINDOW']}",
            ctx.hist_df["low"].rolling(ctx.params["RECENT_LOW_WINDOW"]).min().iloc[-1],
        )
        atr_pct = ctx.atr_pct
        tmsv_strength = ctx.tmsv_strength
        downside_momentum = ctx.downside_momentum
        max_drawdown_pct = ctx.max_drawdown_pct
        market_above_ma20 = ctx.market.get("market_above_ma20", False)
        market_above_ma60 = ctx.market.get("market_above_ma60", False)
        market_amount_above_ma20 = ctx.market.get("market_amount_above_ma20", False)

        def cap(x):
            return max(0.0, min(1.0, x))

        price_deviation = (
            (price - ma20) / (ma20 * PRICE_DEVIATION_MA_MULT) if ma20 > 0 else 0
        )

        buy_factors = {
            "price_above_ma20": (
                sigmoid_normalize(price_deviation, center=0.2) if price > ma20 else 0
            ),
            "volume_above_ma5": (
                sigmoid_normalize(
                    volume / vol_ma - 1.0,
                    center=VOLUME_RATIO_CENTER,
                    steepness=SIGMOID_STEEPNESS_VOLUME,
                )
                if volume > vol_ma
                else 0
            ),
            "macd_golden_cross": macd_golden,
            "kdj_golden_cross": kdj_golden,
            "bollinger_break_up": (
                sigmoid_normalize((price - boll_up) / boll_up, center=0.01)
                if price > boll_up
                else 0
            ),
            "williams_oversold": (
                sigmoid_normalize(
                    (WILLIAMS_OVERSOLD_THRESH - williams_r) / WILLIAMS_NORMALIZE_DIV,
                    center=0.5,
                )
                if williams_r < WILLIAMS_OVERSOLD_THRESH
                else 0
            ),
            "market_above_ma20": 1 if market_above_ma20 else 0,
            "market_above_ma60": 1 if market_above_ma60 else 0,
            "market_amount_above_ma20": 1 if market_amount_above_ma20 else 0,
            "outperform_market": (
                sigmoid_normalize(
                    (ret_etf_5d - ret_market_5d) / OUTPERFORM_MARKET_DIV, center=0.2
                )
                if ret_etf_5d > ret_market_5d
                else 0
            ),
            "weekly_above_ma20": 1 if weekly_above else 0,
            "tmsv_score": tmsv_strength,
        }

        sell_factors = {
            "price_below_ma20": (
                sigmoid_normalize(-price_deviation, center=0.2) if price < ma20 else 0
            ),
            "bollinger_break_down": (
                sigmoid_normalize((boll_low - price) / boll_low, center=0.01)
                if price < boll_low
                else 0
            ),
            "williams_overbought": (
                sigmoid_normalize(
                    (WILLIAMS_OVERBOUGHT_THRESH - williams_r) / WILLIAMS_NORMALIZE_DIV,
                    center=0.5,
                )
                if williams_r < WILLIAMS_OVERBOUGHT_THRESH
                else 0
            ),
            "rsi_overbought": (
                sigmoid_normalize(
                    (rsi - RSI_OVERBOUGHT_THRESH) / RSI_OVERBOUGHT_DIV, center=0.2
                )
                if rsi > RSI_OVERBOUGHT_THRESH
                else 0
            ),
            "underperform_market": (
                sigmoid_normalize(
                    (ret_market_5d - ret_etf_5d) / OUTPERFORM_MARKET_DIV, center=0.2
                )
                if ret_etf_5d < ret_market_5d
                else 0
            ),
            "stop_loss_ma_break": (
                cap((ma20 - price) / (ma20 * HARD_STOP_MA_BREAK_PCT))
                if price < ma20
                else 0
            ),
            "trailing_stop_clear": (
                cap((recent_high - price) / recent_high / (ATR_STOP_MULT * atr_pct))
                if recent_high > 0
                and atr_pct > 0
                and (recent_high - price) / recent_high >= ATR_STOP_MULT * atr_pct
                else 0
            ),
            "trailing_stop_half": (
                cap((recent_high - price) / recent_high / (ATR_TRAILING_MULT * atr_pct))
                if recent_high > 0
                and atr_pct > 0
                and (recent_high - price) / recent_high >= ATR_TRAILING_MULT * atr_pct
                else 0
            ),

            "weekly_below_ma20": 1 if weekly_below else 0,
            "downside_momentum": cap(downside_momentum),
            "max_drawdown_stop": (
                cap(max_drawdown_pct / MAX_DRAWDOWN_STOP_DIV)
                if max_drawdown_pct >= MAX_DRAWDOWN_STOP_DIV
                else 0
            ),
        }
        return buy_factors, sell_factors

    # ---------- 信号确认 ----------
    def get_dynamic_history_days(self, volatility: float) -> int:
        """根据波动率动态调整用于判断趋势的评分历史窗口天数"""
        if volatility > VOL_HIGH_CONFIRM:
            return 5
        if volatility > VOL_MID_CONFIRM:
            return 8
        return 20 if volatility <= 0.015 else 12

    def _get_dynamic_confirm_days(
        self, atr_pct: Optional[float], base_days: int
    ) -> int:
        """根据波动率动态调整确认信号的连续天数"""
        if atr_pct is None:
            return base_days
        if atr_pct > VOL_HIGH_CONFIRM:
            return max(MIN_CONFIRM_DAYS, base_days - 1)
        elif atr_pct > VOL_MID_CONFIRM:
            return base_days
        else:
            return min(MAX_CONFIRM_DAYS, base_days + 1)

    def get_action(
        self,
        score: float,
        score_history: List[Dict],
        params: Dict,
        atr_pct: float = None,
    ) -> str:
        """
        根据当前评分、历史评分序列和参数判断操作信号（BUY/SELL/PREP_BUY/PREP_SELL/HOLD）

        Args:
            score: 当期评分
            score_history: 历史评分记录列表 [{"date":..., "score":...}]
            params: 当前参数（包含阈值）
            atr_pct: 当前波动率（用于动态调整）

        Returns:
            操作信号字符串
        """
        hist_scores = [s["score"] for s in score_history]
        buy_thresh = params["BUY_THRESHOLD"]
        sell_thresh = params["SELL_THRESHOLD"]

        if len(hist_scores) < 2:
            if score > buy_thresh:
                return "BUY"
            elif score < sell_thresh:
                return "SELL"
            else:
                return "HOLD"

        confirm_days = self._get_dynamic_confirm_days(atr_pct, params["CONFIRM_DAYS"])
        window = self.get_dynamic_history_days(atr_pct) if atr_pct else 12
        window = min(window, len(hist_scores))
        recent = hist_scores[-window:]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        up_days = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
        down_days = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])

        # 买入信号判断
        if score > buy_thresh:
            if (
                score > params["QUICK_BUY_THRESHOLD"]
                and slope > SIGNAL_SLOPE_BUY_THRESH
                and avg > buy_thresh + SIGNAL_AVG_OFFSET
            ):
                return "BUY"
            if (
                len(hist_scores) >= confirm_days
                and all(s > buy_thresh for s in hist_scores[-confirm_days:])
                and slope >= 0
            ):
                return "BUY"
            if (
                down_days >= 2
                and slope > SIGNAL_SLOPE_WEAK
                and score > avg + SIGNAL_AVG_OFFSET
            ):
                return "BUY"
            return "PREP_BUY"

        # 卖出信号判断
        if score < sell_thresh:
            if (
                score < sell_thresh - SIGNAL_AVG_OFFSET
                and slope < SIGNAL_SELL_SLOPE
                and avg < sell_thresh - SIGNAL_AVG_OFFSET
            ):
                return "SELL"
            if (
                len(hist_scores) >= confirm_days
                and all(s < sell_thresh for s in hist_scores[-confirm_days:])
                and slope <= 0
            ):
                return "SELL"
            if (
                up_days >= 2
                and slope < SIGNAL_SELL_WEAK_SLOPE
                and score < avg - SIGNAL_AVG_OFFSET
            ):
                return "SELL"
            return "PREP_SELL"

        # 高波动下的额外规则
        if atr_pct and atr_pct > VOL_HIGH_CONFIRM:
            if (
                score > buy_thresh + 0.15
                and slope > SIGNAL_HIGH_VOL_BUY_SLOPE
                and up_days >= SIGNAL_HIGH_VOL_DAYS
            ):
                return "BUY"
            if score < sell_thresh - 0.05 and slope < -0.08 and down_days >= 3:
                return "SELL"
            return "HOLD"
        if atr_pct and atr_pct > VOL_MID_CONFIRM:
            if (
                score > buy_thresh + 0.1
                and slope > SIGNAL_MID_VOL_BUY_SLOPE
                and up_days >= SIGNAL_MID_VOL_DAYS
            ):
                return "BUY"
            if score < sell_thresh - 0.1 and slope < -0.08 and down_days >= 3:
                return "SELL"
        return "HOLD"

    def get_action_level(self, score: float) -> str:
        """将评分映射为操作等级文本（极度看好/强烈买入/.../卖出）"""
        for th, level in zip(ACTION_LEVEL_THRESHOLDS, ACTION_LEVEL_NAMES):
            if score >= th:
                return level
        return "强烈卖出"

    def adjust_params_based_on_history(
        self,
        params: Dict,
        score_history: List[Dict],
        volatility: float,
        market_factor: float,
    ) -> Dict:
        """
        根据近期评分趋势和波动率自适应调整买卖阈值和确认天数

        Args:
            params: 原始参数
            score_history: 评分历史
            volatility: 波动率
            market_factor: 市场因子

        Returns:
            调整后的参数字典
        """
        if len(score_history) < 10:
            return params
        window = min(self.get_dynamic_history_days(volatility), len(score_history))
        recent = [s["score"] for s in score_history[-window:]]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        short_recent = [
            s["score"] for s in score_history[-min(3, len(score_history)) :]
        ]
        short_slope = (
            np.polyfit(range(len(short_recent)), short_recent, 1)[0]
            if len(short_recent) >= 2
            else 0
        )
        adjust_mult = ADJUST_MULT_BASE / market_factor
        adjusted = params.copy()

        # 根据趋势调整买入阈值
        if slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = (
                min(ADJUST_BUY_DELTA_MAX, abs(avg - params["BUY_THRESHOLD"]) * 0.15)
                * adjust_mult
            )
            adjusted["BUY_THRESHOLD"] = max(0.35, params["BUY_THRESHOLD"] - delta)
        elif slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = (
                min(ADJUST_BUY_DELTA_MAX, abs(avg - params["BUY_THRESHOLD"]) * 0.15)
                * adjust_mult
            )
            adjusted["BUY_THRESHOLD"] = min(0.65, params["BUY_THRESHOLD"] + delta)
        else:
            adjusted["BUY_THRESHOLD"] = (
                params["BUY_THRESHOLD"] * 0.9 + DEFAULT_PARAMS["BUY_THRESHOLD"] * 0.1
            )

        # 调整卖出阈值
        if slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = (
                min(ADJUST_SELL_DELTA_MAX, abs(avg - params["SELL_THRESHOLD"]) * 0.15)
                * adjust_mult
            )
            adjusted["SELL_THRESHOLD"] = max(-0.45, params["SELL_THRESHOLD"] - delta)
        elif slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = (
                min(ADJUST_SELL_DELTA_MAX, abs(avg - params["SELL_THRESHOLD"]) * 0.15)
                * adjust_mult
            )
            adjusted["SELL_THRESHOLD"] = min(-0.15, params["SELL_THRESHOLD"] + delta)
        else:
            adjusted["SELL_THRESHOLD"] = (
                params["SELL_THRESHOLD"] * 0.9 + DEFAULT_PARAMS["SELL_THRESHOLD"] * 0.1
            )

        # 根据波动率调整确认天数
        if volatility > VOL_HIGH_CONFIRM:
            adjusted["CONFIRM_DAYS"] = min(5, int(round(params["CONFIRM_DAYS"] * 1.1)))
        elif volatility < 0.01:
            adjusted["CONFIRM_DAYS"] = max(2, int(round(params["CONFIRM_DAYS"] * 0.9)))
        else:
            adjusted["CONFIRM_DAYS"] = int(
                round(
                    params["CONFIRM_DAYS"] * 0.95
                    + DEFAULT_PARAMS["CONFIRM_DAYS"] * 0.05
                )
            )
        return adjusted

    # ---------- 核心分析 ----------
    def _core_analysis(self, ctx: ETFContext) -> ETFContext:
        """
        核心分析流水线：填充 ETFContext 中的计算结果，包括因子强度、评分、TMSV 等

        Args:
            ctx: ETF 上下文（部分字段预先填充）

        Returns:
            更新后的 ETFContext
        """
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

        # 周线状态
        weekly_above = weekly_below = False
        if ctx.weekly_df is not None and not ctx.weekly_df.empty:
            w = ctx.weekly_df.iloc[-1]
            if "ma_short" in w.index and not pd.isna(w["ma_short"]):
                weekly_above = w["close"] > w["ma_short"]
                weekly_below = w["close"] < w["ma_short"]
        ctx.weekly_above = weekly_above
        ctx.weekly_below = weekly_below

        market_status = ctx.market.get("macro_status", "震荡偏弱")
        try:
            tmsv_series = self.compute_tmsv(ctx.hist_df, market_status, atr_pct)
            tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
            tmsv = 50.0 if np.isnan(tmsv) else tmsv
        except Exception:
            logger.error("TMSV 计算失败", exc_info=True)
            tmsv = 50.0
        ctx.tmsv = tmsv
        ctx.tmsv_strength = tmsv / 100.0

        ctx.downside_momentum = d.get("downside_momentum_raw", 0.0)
        recent_high = d.get(
            f"recent_high_{ctx.params['RECENT_HIGH_WINDOW']}",
            ctx.hist_df["high"]
            .rolling(ctx.params["RECENT_HIGH_WINDOW"])
            .max()
            .iloc[-1],
        )
        ctx.max_drawdown_pct = (
            (recent_high - ctx.real_price) / recent_high if recent_high > 0 else 0.0
        )

        # 计算近期低点并判断止盈信号
        recent_low = d.get(f"recent_low_{ctx.params['RECENT_LOW_WINDOW']}",
                        ctx.hist_df["low"].rolling(ctx.params["RECENT_LOW_WINDOW"]).min().iloc[-1])
        ctx.recent_low_price = recent_low
        if recent_low > 0:
            ctx.profit_pct_from_low = (ctx.real_price - recent_low) / recent_low
            # 使用独立的提示阈值，例如 0.10
            if ctx.profit_pct_from_low >= TAKE_PROFIT_WARNING_THRESHOLD:   # 新配置项
                ctx.should_take_profit = True

        ctx.buy_factors, ctx.sell_factors = self._compute_factors(ctx, d)

        # 加权计算买入/卖出总分
        ctx.buy_score = sum(
            self.buy_weights.get(k, 0) * ctx.buy_factors[k] for k in ctx.buy_factors
        )
        ctx.sell_score = sum(
            self.sell_weights.get(k, 0) * ctx.sell_factors[k] for k in ctx.sell_factors
        )

        # 动量因子缺失惩罚
        if (
            ctx.buy_factors.get("macd_golden_cross", 0) == 0
            and ctx.buy_factors.get("kdj_golden_cross", 0) == 0
        ):
            ctx.buy_score *= MOMENTUM_MISSING_PENALTY

        # 强制止损信号（最大回撤或均线跌破）直接锁定最终评分
        if (
            ctx.sell_factors.get("max_drawdown_stop", 0) > 0
            or ctx.sell_factors.get("stop_loss_ma_break", 0) > 0
        ):
            ctx.final_score = -1.0
            ctx.raw_score = ctx.buy_score - ctx.sell_score
            return ctx

        sentiment = ctx.market.get("sentiment_factor", 1.0)
        if sentiment >= SENTIMENT_OVERHEAT_THRESHOLD:
            ctx.buy_score *= SENTIMENT_PENALTY_FACTOR

        ctx.raw_score = ctx.buy_score - ctx.sell_score
        transformed_raw = self._nonlinear_score_transform(ctx.raw_score, market_status)
        env_factor = clip_env_factor(ctx.market["market_factor"], sentiment)
        ctx.final_score = max(-1.0, min(1.0, transformed_raw * env_factor))
        return ctx

    # ---------- 单只 ETF 简要分析 ----------
    def analyze_single_etf(
        self,
        code: str,
        name: str,
        real_price: Optional[float],
        hist_df: Optional[pd.DataFrame],
        weekly_df: Optional[pd.DataFrame],
        market: Dict,
        today: datetime.date,
        state: Dict,
    ) -> Tuple[str, Optional[Dict], Dict, float]:
        """
        对单只 ETF 进行简要分析，返回格式化的输出行、交易信号、状态字典及最终评分

        Args:
            code: ETF 代码
            name: ETF 名称
            real_price: 实时价格
            hist_df: 日线指标数据
            weekly_df: 周线数据
            market: 市场环境字典
            today: 分析日期
            state: 该 ETF 的状态（含历史评分）

        Returns:
            (输出字符串, 信号字典或None, 更新后的状态, 最终评分)
        """
        ctx = ETFContext(
            code,
            name,
            real_price,
            hist_df,
            weekly_df,
            today,
            market,
            self.params.copy(),
        )
        ctx = self._core_analysis(ctx)

        if ctx.error:
            if "实时价格" in ctx.error:
                out = (
                    f"{pad_display(name, 16)} {pad_display(code, 12)} "
                    f"{pad_display('获取失败', 8)} {pad_display('0.00%', 8, 'right')} "
                    f"{pad_display('0.00', 6, 'right')}  {pad_display('价格缺失', 10)}"
                )
            else:
                price_str = f"{real_price:.3f}" if real_price else "N/A"
                change_str = f"{ctx.change_pct:+.2f}%"
                out = (
                    f"{pad_display(name, 16)} {pad_display(code, 12)} "
                    f"{pad_display(price_str, 8, 'right')} "
                    f"{pad_display(change_str, 8, 'right')} "
                    f"{pad_display('0.00', 6, 'right')}  {pad_display('数据不足', 10)}"
                )
            return out, None, state, 0.0

        final = ctx.final_score
        today_str = today.strftime("%Y-%m-%d")
        # 更新评分历史
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

        # 根据历史自适应调整参数
        if len(state["score_history"]) >= 7:
            self.params = self.adjust_params_based_on_history(
                self.params,
                state["score_history"],
                ctx.atr_pct,
                market["market_factor"],
            )

        action = self.get_action(
            final, state["score_history"], self.params, ctx.atr_pct
        )
        action_level = self.get_action_level(final)
        display_action = action_level
        if action == "PREP_BUY":
            display_action = "预备买入"
        elif action == "PREP_SELL":
            display_action = "预备卖出"

        # 风险提示
        risk_warning = ""
        if len(state["score_history"]) >= RISK_WARNING_DAYS:
            recent_scores = [
                s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]
            ]
            if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
                risk_warning = f"风险提示:连续{RISK_WARNING_DAYS}天评分低于{RISK_WARNING_THRESHOLD}"
            elif final < -0.5 or final > 0.8:
                risk_warning = f"风险提示:极端评分{final:.2f}"
            elif ctx.atr_pct > 0.03:
                risk_warning = f"风险提示:高波动{ctx.atr_pct:.3f}"

        price_str = f"{real_price:.3f}"
        change_str = f"{ctx.change_pct:+.2f}%"
        final_str = f"{final:.2f}"
        output = (
            f"{pad_display(name, 16)} {pad_display(code, 12)} "
            f"{pad_display(price_str, 8, 'right')} "
            f"{pad_display(change_str, 8, 'right')} "
            f"{pad_display(final_str, 6, 'right')}  "
            f"{pad_display(display_action, 10)}"
        )
        if risk_warning:
            output += f"  {risk_warning}"
        if ctx.should_take_profit:
            profit_pct = ctx.profit_pct_from_low
            if profit_pct < 0.12:
                tip = f"💡 止盈预警({ctx.params['RECENT_LOW_WINDOW']}日): +{profit_pct:.1%}"
            elif profit_pct < 0.15:
                tip = f"🔔 止盈提醒({ctx.params['RECENT_LOW_WINDOW']}日): +{profit_pct:.1%}"
            else:
                tip = f"🔔🔔 强烈止盈({ctx.params['RECENT_LOW_WINDOW']}日): +{profit_pct:.1%}"
            output += f"  {tip}"
        signal = (
            {"action": action, "name": name, "code": code, "score": final}
            if action in ("BUY", "SELL")
            else None
        )
        return output, signal, state, final

    # ---------- 详细分析报告 ----------
    def detailed_analysis(
        self,
        code: str,
        name: str,
        real_price: Optional[float],
        hist_df: Optional[pd.DataFrame],
        weekly_df: Optional[pd.DataFrame],
        market: Dict,
        today: datetime.date,
        state: Dict,
        ai_client: Optional[AIClient] = None,
    ) -> str:
        """
        生成单个 ETF 的详细分析报告（包含因子明细、AI 点评）

        Args:
            code: ETF 代码
            name: ETF 名称
            ... (同上)
            ai_client: AI 客户端用于生成点评

        Returns:
            格式化的多行文本报告
        """
        ctx = ETFContext(
            code,
            name,
            real_price,
            hist_df,
            weekly_df,
            today,
            market,
            self.params.copy(),
        )
        ctx = self._core_analysis(ctx)

        if ctx.error:
            return f"【{name} ({code})】{ctx.error}，无法分析。"

        final = ctx.final_score
        action_level = self.get_action_level(final)

        lines = [
            "=" * 70,
            f"ETF详细分析报告 - {name} ({code})",
            f"分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 70,
            f"实时价格：{real_price:.3f}",
            f"涨跌幅：{ctx.change_pct:+.2f}%",
            f"市场状态：{market['macro_status']}，市场因子：{market['market_factor']:.2f}，情绪因子：{market['sentiment_factor']:.2f}",
        ]
        if market.get("sentiment_risk_tip"):
            lines.append(f"情绪风险提示：{market['sentiment_risk_tip']}")
        lines += [
            f"波动率(ATR%)：{ctx.atr_pct*100:.2f}%",
            f"TMSV复合强度：{ctx.tmsv:.1f} (强度系数 {ctx.tmsv_strength:.3f})",
            f"最大回撤：{ctx.max_drawdown_pct*100:.2f}%",
            "",
        ]
        if ctx.should_take_profit:
            lines.append(
                f"🔔 止盈提示：当前价格较{ctx.params['RECENT_LOW_WINDOW']}日内最低点 "
                f"{ctx.recent_low_price:.3f} 上涨 {ctx.profit_pct_from_low:.1%}，已达到10%止盈线。"
            )
            lines.append("")

        col_name, col_strength, col_weight, col_contrib = 25, 8, 8, 8

        def row_line(items):
            return "".join(
                [
                    pad_display(items[0], col_name),
                    pad_display(items[1], col_strength, "right"),
                    pad_display(items[2], col_weight, "right"),
                    pad_display(items[3], col_contrib, "right"),
                ]
            )

        lines.append("【买入因子详情】")
        lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
        lines.append("-" * 50)
        buy_contribs = sorted(
            [
                (
                    k,
                    ctx.buy_factors[k],
                    self.buy_weights.get(k, 0),
                    self.buy_weights.get(k, 0) * ctx.buy_factors[k],
                )
                for k in ctx.buy_factors
            ],
            key=lambda x: x[3],
            reverse=True,
        )
        for name_f, s, w, contrib in buy_contribs:
            lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
        lines.append(row_line(["买入总分", "", "", f"{ctx.buy_score:.3f}"]))
        lines.append("")

        lines.append("【卖出因子详情】")
        lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
        lines.append("-" * 50)
        sell_contribs = sorted(
            [
                (
                    k,
                    ctx.sell_factors[k],
                    self.sell_weights.get(k, 0),
                    self.sell_weights.get(k, 0) * ctx.sell_factors[k],
                )
                for k in ctx.sell_factors
            ],
            key=lambda x: x[3],
            reverse=True,
        )
        for name_f, s, w, contrib in sell_contribs:
            lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
        lines.append(row_line(["卖出总分", "", "", f"{ctx.sell_score:.3f}"]))
        lines.append("")

        scale = (
            NONLINEAR_SCALE_BULL
            if ("牛" in market["macro_status"] or "熊" in market["macro_status"])
            else NONLINEAR_SCALE_RANGE
        )
        env_factor = clip_env_factor(
            market["market_factor"], market["sentiment_factor"]
        )
        lines += [
            "【评分合成】",
            f"原始净分 = 买入总分 - 卖出总分 = {ctx.buy_score:.3f} - {ctx.sell_score:.3f} = {ctx.raw_score:.3f}",
            f"非线性变换: tanh({scale} * raw) → 最终评分 = 变换后 × 环境因子",
            f"        = tanh({scale} × {ctx.raw_score:.3f}) × {env_factor:.2f} = {final:.3f}",
            f"操作等级：{action_level}",
        ]

        if ai_client:
            lines += ["", "【AI 专业点评】"]
            ai_comment = ai_client.comment_on_etf(
                code,
                name,
                final,
                action_level,
                market["macro_status"],
                market["market_factor"],
                market["sentiment_factor"],
                self.buy_weights,
                self.sell_weights,
                ctx.buy_factors,
                ctx.sell_factors,
                ctx.tmsv,
                ctx.atr_pct,
            )
            lines.append(ai_comment)
        else:
            lines += ["", "【AI 专业点评】未配置 API_KEY，无法生成。"]
        lines.append("=" * 70)
        return "\n".join(lines)


# ========================== 辅助函数 ==========================
def _prepare_etf_data(
    code: str,
    fetcher: DataFetcher,
    analyzer: DataAnalyzer,
    start: str,
    today_str: str,
    params: Dict,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[float]]:
    """准备单只 ETF 的日线（含指标）、周线数据和实时价格"""
    cache_key_hist = analyzer._get_cache_key(code, start, today_str)
    hist = fetcher.get_daily_data(code, start, today_str)
    if hist is not None:
        hist = analyzer.calculate_indicators(
            hist,
            need_amount_ma=False,
            recent_high_window=params["RECENT_HIGH_WINDOW"],
            recent_low_window=params["RECENT_LOW_WINDOW"],
            use_cache=True,
            cache_key=cache_key_hist,
        )
    weekly = fetcher.get_weekly_data(code, start, today_str)
    real_price = fetcher.get_realtime_price(code)
    return hist, weekly, real_price


def _get_or_create_environment(
    fetcher: DataFetcher,
    analyzer: DataAnalyzer,
    market_df: pd.DataFrame,
    macro_df: pd.DataFrame,
    volatility: float,
    market_info_basic: Dict[str, Any],
    api_key: Optional[str],
) -> Tuple[str, float, float, Dict, Dict, str]:
    """
    获取或创建市场环境缓存：市场状态、市场因子、情绪因子、最终权重、风险提示

    Returns:
        (市场状态, 市场因子, 情绪因子, 买入权重, 卖出权重, 风险提示)
    """
    cached_env = fetcher.get_cached_environment()
    if cached_env:
        return (
            cached_env["market_state"],
            cached_env["market_factor"],
            cached_env["sentiment"],
            cached_env["buy_weights"],
            cached_env["sell_weights"],
            fetcher.get_sentiment_risk_tip(cached_env["sentiment"]),
        )

    ai_client = AIClient(api_key) if api_key else None
    if ai_client:
        market_state, market_factor = fetcher.get_market_state(market_df, ai_client)
    else:
        last = market_df.iloc[-1]
        above_ma20 = last["close"] > last["ma_short"]
        above_ma60 = last["close"] > last.get("ma_long", last["ma_short"])
        if above_ma20 and above_ma60:
            market_state, market_factor = "正常牛市", 1.2
        elif not above_ma20 and not above_ma60:
            market_state, market_factor = "熊市下跌", 0.8
        else:
            market_state, market_factor = "震荡偏弱", 1.0

    sentiment, sentiment_raw = 1.0, 1.0
    if AKSHARE_AVAILABLE:
        try:
            ind = fetcher.fetch_sentiment_indicators()
            sentiment, sentiment_raw = fetcher.compute_sentiment_factor(ind)
        except Exception as e:
            logger.warning(f"获取情绪失败，使用后备: {e}")
            sentiment, sentiment_raw = fetcher.get_sentiment_factor_simple(macro_df)
    else:
        sentiment, sentiment_raw = fetcher.get_sentiment_factor_simple(macro_df)
    sentiment_risk_tip = fetcher.get_sentiment_risk_tip(sentiment)

    buy_w, sell_w = DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()
    if api_key:
        ai_temp = AIClient(api_key)
        buy_w, sell_w = analyzer.generate_ai_weights(
            ai_temp,
            market_state,
            sentiment,
            market_info_basic["market_above_ma20"],
            market_info_basic["market_above_ma60"],
            market_info_basic["market_amount_above_ma20"],
            volatility,
        )

    fetcher.save_environment_cache(
        market_state, market_factor, buy_w, sell_w, sentiment, sentiment_raw
    )
    return market_state, market_factor, sentiment, buy_w, sell_w, sentiment_risk_tip


def run_batch_analysis(
    api_key: Optional[str] = None, target_code: Optional[str] = None
):
    """
    批量分析入口函数：加载持仓、获取数据、生成环境、并行分析所有 ETF

    Args:
        api_key: DeepSeek API 密钥
        target_code: 若指定，仅分析该代码的 ETF
    """
    fetcher = DataFetcher()
    analyzer = DataAnalyzer()

    if not fetcher.login():
        return
    try:
        etf_list = fetcher.load_positions()
    except Exception as e:
        print(f"请准备 positions.csv (代码,名称)，错误: {e}")
        return

    today = datetime.date.today()
    start = (today - datetime.timedelta(days=200)).strftime("%Y-%m-%d")
    today_str = today.strftime("%Y-%m-%d")

    market_df = fetcher.get_daily_data(MARKET_INDEX, start, today_str)
    macro_df = fetcher.get_daily_data(MACRO_INDEX, start, today_str)
    if market_df is None or macro_df is None:
        print("获取宏观数据失败")
        return

    macro_df = analyzer.calculate_indicators(macro_df, need_amount_ma=False)
    macro_df["ma_long"] = macro_df["close"].rolling(MACRO_MA_LONG).mean()
    market_df = analyzer.calculate_indicators(market_df, need_amount_ma=True)
    market_df["atr"] = analyzer.calculate_atr(market_df, ATR_PERIOD)
    volatility = (market_df["atr"] / market_df["close"]).iloc[-20:].mean()

    mkt = market_df.iloc[-1]
    market_info_basic = {
        "market_above_ma20": mkt["close"] > mkt["ma_short"],
        "market_above_ma60": mkt["close"] > mkt.get("ma_long", mkt["ma_short"]),
        "market_amount_above_ma20": mkt["amount"] > mkt["amount_ma"],
        "ret_market_5d": (
            (mkt["close"] / market_df.iloc[-5]["close"] - 1)
            if len(market_df) >= 5
            else 0
        ),
    }

    market_state, market_factor, sentiment, buy_w, sell_w, risk_tip = (
        _get_or_create_environment(
            fetcher,
            analyzer,
            market_df,
            macro_df,
            volatility,
            market_info_basic,
            api_key,
        )
    )

    market_info = {
        "macro_status": market_state,
        "market_factor": market_factor,
        "sentiment_factor": sentiment,
        "sentiment_risk_tip": risk_tip,
        **market_info_basic,
    }
    analyzer.set_market_info(market_info)
    analyzer.set_weights(buy_w, sell_w)

    # 根据波动率初始化参数
    params = DEFAULT_PARAMS.copy()
    if volatility > 0.04:
        params.update(
            {
                "BUY_THRESHOLD": 0.65,
                "SELL_THRESHOLD": -0.35,
                "CONFIRM_DAYS": 5,
                "QUICK_BUY_THRESHOLD": 0.75,
            }
        )
    elif volatility > 0.02:
        params.update(
            {
                "BUY_THRESHOLD": 0.6,
                "SELL_THRESHOLD": -0.3,
                "CONFIRM_DAYS": 4,
                "QUICK_BUY_THRESHOLD": 0.7,
            }
        )
    elif volatility < 0.01:
        params.update(
            {
                "BUY_THRESHOLD": 0.4,
                "SELL_THRESHOLD": -0.1,
                "CONFIRM_DAYS": 2,
                "QUICK_BUY_THRESHOLD": 0.5,
            }
        )
    analyzer.params = params

    state = fetcher.load_state()
    ai_client = AIClient(api_key) if api_key else None

    # 单只详细分析模式
    if target_code:
        target = etf_list[etf_list["代码"] == target_code]
        if target.empty:
            print(f"未找到代码 {target_code}")
            fetcher.logout()
            return
        code, name = target.iloc[0]["代码"], target.iloc[0]["名称"]
        hist, weekly, real_price = _prepare_etf_data(
            code, fetcher, analyzer, start, today_str, params
        )
        etf_state = state.get(code, {})
        report = analyzer.detailed_analysis(
            code,
            name,
            real_price,
            hist,
            weekly,
            market_info,
            today,
            etf_state,
            ai_client,
        )
        print(report)
        fetcher.logout()
        return

    # 批量分析模式
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ETF 分析报告")
    print(f"市场状态: {market_state}, 市场因子: {market_factor:.2f}")
    if risk_tip:
        print(f"情绪因子: {sentiment:.3f} - {risk_tip}")
    else:
        print(f"情绪因子: {sentiment:.3f}")

    print(
        pad_display("名称", 16),
        pad_display("代码", 12),
        pad_display("价格", 8, "right"),
        pad_display("涨跌幅", 8, "right"),
        pad_display("评分", 6, "right"),
        "  " + pad_display("操作", 10),
    )
    print("-" * 80)

    output_lines, results = [], []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = []
        for _, row in etf_list.iterrows():
            code, name = row["代码"], row["名称"]
            hist, weekly, real_price = _prepare_etf_data(
                code, fetcher, analyzer, start, today_str, params
            )
            s = state.get(code, {})
            futures.append(
                ex.submit(
                    analyzer.analyze_single_etf,
                    code,
                    name,
                    real_price,
                    hist,
                    weekly,
                    market_info,
                    today,
                    s,
                )
            )
        for f in futures:
            out, _, new_state, score = f.result()
            results.append((out, score))
            m = re.search(r"【.*?\((.*?)\)】", out)
            if m:
                state[m.group(1)] = new_state

    results.sort(key=lambda x: x[1], reverse=True)
    for out, _ in results:
        print(out)
        output_lines.append(out)

    fetcher.save_state(state)
    fetcher.logout()

    email_cfg = get_email_config()
    if email_cfg["send_email"]:
        send_email(f"ETF分析报告 - {today_str}", "\n".join(output_lines))
