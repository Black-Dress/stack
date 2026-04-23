#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
核心分析模块：负责技术指标计算、评分、信号确认及批量分析入口。
依赖 fetcher.py 中的数据获取功能。
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

from .config import (
    ETF_MA,
    ETF_VOL_MA,
    ATR_PERIOD,
    ATR_STOP_MULT,
    ATR_TRAILING_MULT,
    PROFIT_TARGET,
    RISK_WARNING_DAYS,
    RISK_WARNING_THRESHOLD,
    DEFAULT_BUY_WEIGHTS,
    DEFAULT_SELL_WEIGHTS,
    DEFAULT_PARAMS,
    MACRO_INDEX,
    MARKET_INDEX,
    MACRO_MA_LONG,
    get_email_config,
)
from .utils import pad_display, validate_and_filter_weights, send_email
from .ai import AIClient
from .fetcher import DataFetcher, AKSHARE_AVAILABLE

logger = logging.getLogger(__name__)

# ========================== 硬编码常量化 ==========================
# ---------- 技术指标参数 ----------
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
KDJ_N = 9
BOLL_WINDOW = 20
BOLL_STD_MULT = 2
WILLIAMS_WINDOW = 14
RSI_WINDOW = 14
TMSV_MA20_WINDOW = 20
TMSV_MA60_WINDOW = 60
TMSV_ATR_WINDOW = 14
TMSV_VOL_MA_WINDOW = 20


NONLINEAR_SCALE_BULL = 2.5      # 趋势市缩放系数
NONLINEAR_SCALE_RANGE = 1.5     # 震荡市缩放系数

# ---------- 非线性评分变换参数 ----------
NONLINEAR_SCALE = 2.5          # tanh 缩放因子

# ---------- Sigmoid 归一化参数 ----------
SIGMOID_STEEPNESS_DEFAULT = 5.0        # 默认陡峭度
SIGMOID_STEEPNESS_VOLUME = 3.0         # 成交量因子专用陡峭度

# ---------- 硬止损阈值 ----------
HARD_STOP_DRAWDOWN = 0.08      # 最大回撤触发阈值
HARD_STOP_MA_BREAK_PCT = 0.05  # 均线跌破幅度阈值

# ---------- 情绪过热惩罚 ----------
SENTIMENT_OVERHEAT_THRESHOLD = 1.25
SENTIMENT_PENALTY_FACTOR = 0.8

# ---------- 缓存有效期 ----------
CACHE_EXPIRE_SECONDS = 600

# ---------- 因子计算通用参数 ----------
PRICE_DEVIATION_MA_MULT = 0.1           # 价格偏离均线的归一化除数（ma * 0.1）
VOLUME_RATIO_CENTER = 0.1               # 成交量比率 sigmoid 中心偏移
OUTPERFORM_MARKET_DIV = 0.05            # 超额收益归一化除数
WILLIAMS_OVERBOUGHT_THRESH = -20        # 威廉指标超买阈值（注意负值）
WILLIAMS_OVERSOLD_THRESH = -80          # 超卖阈值
RSI_OVERBOUGHT_THRESH = 70
RSI_OVERBOUGHT_DIV = 30
PROFIT_TARGET_DIV = PROFIT_TARGET       # 止盈目标（沿用 config）
MAX_DRAWDOWN_STOP_DIV = 0.08            # 最大回撤归一化除数（与硬止损一致）

# ---------- 动态确认天数波动率阈值 ----------
VOL_HIGH_CONFIRM = 0.04
VOL_MID_CONFIRM = 0.025

# ---------- 评分等级阈值 ----------
ACTION_LEVEL_THRESHOLDS = [0.8, 0.7, 0.6, 0.4, 0.2, 0.0, -0.2, -0.4, -0.6, -0.8]
ACTION_LEVEL_NAMES = [
    "极度看好", "强烈买入", "买入", "谨慎买入", "偏多持有",
    "中性偏多", "中性偏空", "偏空持有", "谨慎卖出", "卖出"
]

# ---------- TMSV 计算参数 ----------
TMSV_TREND_MA20_WEIGHT = 0.5
TMSV_TREND_MA60_WEIGHT = 0.3
TMSV_TREND_SLOPE_WEIGHT = 0.2
TMSV_MOM_RSI_WEIGHT = 0.6
TMSV_MOM_MACD_WEIGHT = 0.4
TMSV_VOL_RATIO_WEIGHT = 0.7
TMSV_VOL_CONSIST_WEIGHT = 0.3
TMSV_VOL_LOW_THRESH = 0.01
TMSV_VOL_HIGH_THRESH = 0.03
TMSV_VOL_LOW_FACTOR = 1.5
TMSV_VOL_HIGH_FACTOR = 0.6
TMSV_VOL_MID_FACTOR_BASE = 1.2
TMSV_VOL_MID_FACTOR_SLOPE = 0.6

# ---------- 参数动态调整 ----------
ADJUST_MULT_BASE = 1.2
ADJUST_BUY_DELTA_MAX = 0.03
ADJUST_SELL_DELTA_MAX = 0.03


# ========================== 数据类定义 ==========================
@dataclass
class ETFContext:
    """ETF 分析上下文数据容器，用于传递和存储分析过程中的所有状态"""
    code: str
    name: str
    real_price: Optional[float]
    hist_df: Optional[pd.DataFrame]
    weekly_df: Optional[pd.DataFrame]
    today: datetime.date
    market: Dict[str, Any]
    params: Dict[str, Any]

    # 以下字段在分析过程中填充
    change_pct: float = 0.0
    atr_pct: float = 0.0
    tmsv: float = 50.0
    tmsv_strength: float = 0.5
    downside_momentum: float = 0.0
    max_drawdown_pct: float = 0.0
    weekly_above: bool = False
    weekly_below: bool = False
    buy_factors: Dict = field(default_factory=dict)
    sell_factors: Dict = field(default_factory=dict)
    buy_score: float = 0.0
    sell_score: float = 0.0
    raw_score: float = 0.0
    final_score: float = 0.0
    error: Optional[str] = None


# ========================== 公共指标计算函数 ==========================
def calc_rsi(series: pd.Series, period: int = RSI_WINDOW) -> pd.Series:
    """计算 RSI 指标"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


def calc_macd(series: pd.Series, fast: int = MACD_FAST, slow: int = MACD_SLOW, signal: int = MACD_SIGNAL):
    """计算 MACD 指标，返回 (dif, dea, hist)"""
    exp_fast = series.ewm(span=fast, adjust=False).mean()
    exp_slow = series.ewm(span=slow, adjust=False).mean()
    dif = exp_fast - exp_slow
    dea = dif.ewm(span=signal, adjust=False).mean()
    hist = dif - dea
    return dif, dea, hist


# ========================== 核心分析类 ==========================
class DataAnalyzer:
    """负责所有技术指标计算、评分、信号确认、状态管理、缓存等"""

    def __init__(self, buy_weights: Dict = None, sell_weights: Dict = None, params: Dict = None):
        self.buy_weights = buy_weights or DEFAULT_BUY_WEIGHTS.copy()
        self.sell_weights = sell_weights or DEFAULT_SELL_WEIGHTS.copy()
        self.params = params or DEFAULT_PARAMS.copy()
        self.market_info = {}          # 市场环境信息
        self._indicator_cache = {}     # 技术指标缓存

    def set_market_info(self, market_info: Dict):
        """设置市场环境信息"""
        self.market_info = market_info

    def set_weights(self, buy_w: Dict, sell_w: Dict):
        """设置当前使用的买卖权重"""
        self.buy_weights = buy_w
        self.sell_weights = sell_w

    # ---------- 非线性辅助函数 ----------
    @staticmethod
    def _sigmoid_normalize(x: float, center: float = 0.0, steepness: float = SIGMOID_STEEPNESS_DEFAULT) -> float:
        """Sigmoid 归一化，将任意实数映射到 [0,1]"""
        return 1.0 / (1.0 + math.exp(-steepness * (x - center)))

    def _get_nonlinear_scale(self, market_status: str) -> float:
        status_lower = market_status.lower()
        if "牛" in status_lower or "熊" in status_lower:
            return NONLINEAR_SCALE_BULL
        else:
            return NONLINEAR_SCALE_RANGE
    
    def _nonlinear_score_transform(self, raw: float, market_status: str = "震荡偏弱") -> float:
        scale = self._get_nonlinear_scale(market_status)
        return math.tanh(scale * raw)

    # ---------- TMSV 动态权重 ----------
    def _get_tmsv_weights(self, market_status: str, volatility: float) -> Dict[str, float]:
        """根据市场状态和波动率返回 TMSV 子项权重"""
        if "牛" in market_status:
            w = {'trend': 0.40, 'momentum': 0.25, 'volume': 0.15}
        elif "熊" in market_status:
            w = {'trend': 0.20, 'momentum': 0.30, 'volume': 0.25}
        else:  # 震荡
            w = {'trend': 0.25, 'momentum': 0.35, 'volume': 0.20}

        # 高波动时略微降低趋势权重
        if volatility > 0.03:
            w['trend'] = max(0.15, w['trend'] - 0.05)
            w['momentum'] += 0.05
        return w

    # ---------- 涨跌幅计算 ----------
    def calc_change_pct(
        self, real_price: float, hist_df: pd.DataFrame, today: datetime.date
    ) -> float:
        """计算实时价格相对于上一个交易日收盘价的涨跌幅"""
        if real_price is None or hist_df is None or len(hist_df) < 1:
            return 0.0
        last_date = hist_df.index[-1].date()
        if last_date == today and len(hist_df) >= 2:
            base_close = hist_df.iloc[-2]["close"]
        else:
            base_close = hist_df.iloc[-1]["close"]
        if base_close > 0:
            return (real_price - base_close) / base_close * 100
        return 0.0

    # ---------- 技术指标计算 ----------
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
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
        """生成指标缓存键"""
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
        """计算全部技术指标，支持缓存"""
        if use_cache and cache_key and cache_key in self._indicator_cache:
            cached_df, cache_time = self._indicator_cache[cache_key]
            if (time.time() - cache_time) < CACHE_EXPIRE_SECONDS:
                return cached_df.copy()

        df = df.copy()
        df["ma_short"] = df["close"].rolling(window=ETF_MA).mean()
        df["vol_ma"] = df["volume"].rolling(window=ETF_VOL_MA).mean()
        if need_amount_ma:
            df["amount_ma"] = df["amount"].rolling(window=ETF_VOL_MA).mean()
        # MACD
        df["macd_dif"], df["macd_dea"], _ = calc_macd(df["close"])
        # KDJ
        low_n = df["low"].rolling(KDJ_N).min()
        high_n = df["high"].rolling(KDJ_N).max()
        rsv = (df["close"] - low_n) / (high_n - low_n) * 100
        df["kdj_k"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
        df["kdj_d"] = df["kdj_k"].ewm(alpha=1 / 3, adjust=False).mean()
        # 布林
        df["boll_mid"] = df["close"].rolling(BOLL_WINDOW).mean()
        df["boll_std"] = df["close"].rolling(BOLL_WINDOW).std()
        df["boll_up"] = df["boll_mid"] + BOLL_STD_MULT * df["boll_std"]
        df["boll_low"] = df["boll_mid"] - BOLL_STD_MULT * df["boll_std"]
        # 威廉
        high_14 = df["high"].rolling(WILLIAMS_WINDOW).max()
        low_14 = df["low"].rolling(WILLIAMS_WINDOW).min()
        df["williams_r"] = (high_14 - df["close"]) / (high_14 - low_14) * -100
        # RSI
        df["rsi"] = calc_rsi(df["close"])
        # ATR
        df["atr"] = self.calculate_atr(df, ATR_PERIOD)
        # ADX
        adx_df = self.calculate_adx(df)
        df["plus_di"] = adx_df["plus_di"]
        df["minus_di"] = adx_df["minus_di"]
        df["adx"] = adx_df["adx"]
        # 下跌动量
        df["downside_momentum_raw"] = np.where(
            (df["close"] < df["ma_short"]) & (df["minus_di"] > df["plus_di"]),
            (df["ma_short"] - df["close"])
            / df["ma_short"]
            * (df["volume"] / df["vol_ma"]).clip(0, 3),
            0,
        )
        # 近期高低点
        df[f"recent_high_{recent_high_window}"] = (
            df["high"].rolling(recent_high_window).max()
        )
        df[f"recent_low_{recent_low_window}"] = (
            df["low"].rolling(recent_low_window).min()
        )

        if use_cache and cache_key:
            self._indicator_cache[cache_key] = (df.copy(), time.time())
        return df

    def compute_tmsv(self, df: pd.DataFrame, market_status: str = "震荡偏弱", volatility: float = 0.02) -> pd.Series:
        """计算 TMSV 复合指标（支持动态权重）"""
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
            price_above_ma20 * TMSV_TREND_MA20_WEIGHT +
            price_above_ma60 * TMSV_TREND_MA60_WEIGHT +
            slope_score * TMSV_TREND_SLOPE_WEIGHT
        ) * 100

        rsi_score = ((df["rsi"] - 50) * 3.33).clip(0, 100).fillna(50)
        macd_change = df["macd_hist"].diff() / (df["macd_hist"].shift(1).abs() + 0.001)
        macd_score = (macd_change * 100).clip(0, 100).fillna(50)
        mom_score = rsi_score * TMSV_MOM_RSI_WEIGHT + macd_score * TMSV_MOM_MACD_WEIGHT

        vol_ratio = df["volume"] / df["vol_ma"].replace(0, np.nan)
        vol_ratio_score = ((vol_ratio - 0.8) / 1.2 * 100).clip(0, 100).fillna(50)
        price_up = df["close"] > df["close"].shift(1)
        vol_up = df["volume"] > df["vol_ma"]
        consistency = np.where(price_up == vol_up, 100, 0)
        vol_score = vol_ratio_score * TMSV_VOL_RATIO_WEIGHT + consistency * TMSV_VOL_CONSIST_WEIGHT

        atr_pct = df["atr"] / df["close"].replace(0, np.nan)
        vol_factor = np.select(
            [atr_pct < TMSV_VOL_LOW_THRESH, atr_pct > TMSV_VOL_HIGH_THRESH],
            [TMSV_VOL_LOW_FACTOR, TMSV_VOL_HIGH_FACTOR],
            default=TMSV_VOL_MID_FACTOR_BASE - (atr_pct - TMSV_VOL_LOW_THRESH) / 0.02 * TMSV_VOL_MID_FACTOR_SLOPE,
        )
        vol_factor = np.nan_to_num(vol_factor, nan=1.0)

        # 动态权重
        w = self._get_tmsv_weights(market_status, volatility)
        tmsv = (trend_score * w['trend'] + mom_score * w['momentum'] + vol_score * w['volume']) * vol_factor
        return tmsv.clip(0, 100).fillna(50)

    # ---------- 权重处理 ----------
    def compute_dynamic_trust(self, ai_weights: Dict, default_weights: Dict) -> float:
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
        if norm_ai > 0 and norm_def > 0:
            similarity = dot / (norm_ai * norm_def)
            if similarity < 0.6:
                base_trust = min(base_trust, 0.55)
        return base_trust

    def blend_weights(self, ai_w: Dict, def_w: Dict, trust: float) -> Dict:
        blended = {k: ai_w.get(k, 0) * trust + def_w[k] * (1 - trust) for k in def_w}
        total = sum(blended.values())
        if total > 0:
            blended = {k: v / total for k, v in blended.items()}
        return blended

    def apply_correlation_penalty(
        self,
        weights: Dict,
        factor_names: List[str],
        corr_matrix: pd.DataFrame,
        penalty_threshold: float = 0.7,
    ) -> Dict:
        w = weights.copy()
        high_corr_pairs = []
        for i, f1 in enumerate(factor_names):
            for j, f2 in enumerate(factor_names):
                if i >= j:
                    continue
                if corr_matrix.loc[f1, f2] > penalty_threshold:
                    if w.get(f1, 0) > 0.12 and w.get(f2, 0) > 0.12:
                        high_corr_pairs.append((f1, f2))
        if not high_corr_pairs:
            return w
        for f1, f2 in high_corr_pairs:
            penalty = 0.2
            w[f1] *= 1 - penalty
            w[f2] *= 1 - penalty
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        return w

    def compute_factor_correlation(
        self, df: pd.DataFrame, factor_names: List[str]
    ) -> pd.DataFrame:
        """简化版相关性矩阵"""
        return pd.DataFrame(
            np.eye(len(factor_names)), index=factor_names, columns=factor_names
        )

    # ---------- 环境因子裁剪 ----------
    @staticmethod
    def _clip_env_factor(market_factor: float, sentiment_factor: float) -> float:
        env_factor = market_factor * sentiment_factor
        return max(0.60, min(1.30, env_factor))

    # ---------- 因子计算（含非线性）----------
    def _compute_factors(
        self,
        price: float,
        ma20: float,
        volume: float,
        vol_ma: float,
        macd_golden: int,
        kdj_golden: int,
        rsi: float,
        boll_up: float,
        boll_low: float,
        williams_r: float,
        ret_etf_5d: float,
        ret_market_5d: float,
        weekly_above: bool,
        weekly_below: bool,
        recent_high: float,
        recent_low: float,
        atr_pct: float,
        market_above_ma20: bool,
        market_above_ma60: bool,
        market_amount_above_ma20: bool,
        tmsv_strength: float,
        downside_momentum: float,
        max_drawdown_pct: float,
    ) -> Tuple[Dict, Dict]:
        """返回买入因子字典和卖出因子字典（部分连续因子使用 sigmoid 非线性）"""

        def cap(x):
            return max(0.0, min(1.0, x))

        # 价格偏离均线比例（用于 sigmoid 输入）
        price_deviation = (price - ma20) / (ma20 * PRICE_DEVIATION_MA_MULT) if ma20 > 0 else 0
        buy_factors = {
            "price_above_ma20": self._sigmoid_normalize(price_deviation, center=0.2) if price > ma20 else 0,
            "volume_above_ma5": self._sigmoid_normalize(volume / vol_ma - 1.0, center=VOLUME_RATIO_CENTER, steepness=SIGMOID_STEEPNESS_VOLUME) if volume > vol_ma else 0,
            "macd_golden_cross": macd_golden,
            "kdj_golden_cross": kdj_golden,
            "bollinger_break_up": self._sigmoid_normalize((price - boll_up) / boll_up, center=0.01) if price > boll_up else 0,
            "williams_oversold": self._sigmoid_normalize((WILLIAMS_OVERSOLD_THRESH - williams_r) / 20, center=0.5) if williams_r < WILLIAMS_OVERSOLD_THRESH else 0,
            "market_above_ma20": 1 if market_above_ma20 else 0,
            "market_above_ma60": 1 if market_above_ma60 else 0,
            "market_amount_above_ma20": 1 if market_amount_above_ma20 else 0,
            "outperform_market": self._sigmoid_normalize((ret_etf_5d - ret_market_5d) / OUTPERFORM_MARKET_DIV, center=0.2) if ret_etf_5d > ret_market_5d else 0,
            "weekly_above_ma20": 1 if weekly_above else 0,
            "tmsv_score": tmsv_strength,
        }

        sell_factors = {
            "price_below_ma20": self._sigmoid_normalize(-price_deviation, center=0.2) if price < ma20 else 0,
            "bollinger_break_down": self._sigmoid_normalize((boll_low - price) / boll_low, center=0.01) if price < boll_low else 0,
            "williams_overbought": self._sigmoid_normalize((WILLIAMS_OVERBOUGHT_THRESH - williams_r) / 20, center=0.5) if williams_r < WILLIAMS_OVERBOUGHT_THRESH else 0,
            "rsi_overbought": self._sigmoid_normalize((rsi - RSI_OVERBOUGHT_THRESH) / RSI_OVERBOUGHT_DIV, center=0.2) if rsi > RSI_OVERBOUGHT_THRESH else 0,
            "underperform_market": self._sigmoid_normalize((ret_market_5d - ret_etf_5d) / OUTPERFORM_MARKET_DIV, center=0.2) if ret_etf_5d < ret_market_5d else 0,
            "stop_loss_ma_break": cap((ma20 - price) / (ma20 * HARD_STOP_MA_BREAK_PCT)) if price < ma20 else 0,
            "trailing_stop_clear": (
                cap((recent_high - price) / recent_high / (ATR_STOP_MULT * atr_pct))
                if recent_high > 0 and atr_pct > 0 and (recent_high - price) / recent_high >= ATR_STOP_MULT * atr_pct
                else 0
            ),
            "trailing_stop_half": (
                cap((recent_high - price) / recent_high / (ATR_TRAILING_MULT * atr_pct))
                if recent_high > 0 and atr_pct > 0 and (recent_high - price) / recent_high >= ATR_TRAILING_MULT * atr_pct
                else 0
            ),
            "profit_target_hit": (
                cap((price - recent_low) / recent_low / PROFIT_TARGET_DIV)
                if recent_low > 0 and (price - recent_low) / recent_low >= PROFIT_TARGET_DIV
                else 0
            ),
            "weekly_below_ma20": 1 if weekly_below else 0,
            "downside_momentum": cap(downside_momentum),
            "max_drawdown_stop": cap(max_drawdown_pct / MAX_DRAWDOWN_STOP_DIV) if max_drawdown_pct >= MAX_DRAWDOWN_STOP_DIV else 0,
        }
        return buy_factors, sell_factors

    # ---------- 信号确认（含动态确认天数）----------
    def get_dynamic_history_days(self, volatility: float) -> int:
        if volatility > VOL_HIGH_CONFIRM:
            return 5
        if volatility > VOL_MID_CONFIRM:
            return 8
        if volatility > 0.015:
            return 12
        return 20

    def _get_dynamic_confirm_days(self, atr_pct: Optional[float], base_days: int) -> int:
        """根据波动率动态调整确认天数：高波动缩短，低波动延长"""
        if atr_pct is None:
            return base_days
        if atr_pct > VOL_HIGH_CONFIRM:
            return max(2, base_days - 1)
        elif atr_pct > VOL_MID_CONFIRM:
            return base_days
        else:
            return min(5, base_days + 1)

    def get_action(
        self,
        score: float,
        score_history: List[Dict],
        params: Dict,
        atr_pct: float = None,
    ) -> str:
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

        # 买入确认逻辑
        if score > buy_thresh:
            if score > params["QUICK_BUY_THRESHOLD"] and slope > 0.05 and avg > buy_thresh + 0.1:
                return "BUY"
            if len(hist_scores) >= confirm_days:
                confirm = hist_scores[-confirm_days:]
                if all(s > buy_thresh for s in confirm) and slope >= 0:
                    return "BUY"
            if down_days >= 2 and slope > 0.02 and score > avg + 0.1:
                return "BUY"
            # 预备信号：突破但未确认
            return "PREP_BUY"

        # 卖出确认逻辑
        if score < sell_thresh:
            if score < sell_thresh - 0.1 and slope < -0.05 and avg < sell_thresh - 0.1:
                return "SELL"
            if len(hist_scores) >= confirm_days:
                confirm = hist_scores[-confirm_days:]
                if all(s < sell_thresh for s in confirm) and slope <= 0:
                    return "SELL"
            if up_days >= 2 and slope < -0.02 and score < avg - 0.1:
                return "SELL"
            return "PREP_SELL"

        # 高波动特殊处理
        if atr_pct and atr_pct > VOL_HIGH_CONFIRM:
            if score > buy_thresh + 0.15 and slope > 0.1 and up_days >= 4:
                return "BUY"
            if score < sell_thresh - 0.05 and slope < -0.08 and down_days >= 3:
                return "SELL"
            return "HOLD"
        if atr_pct and atr_pct > VOL_MID_CONFIRM:
            if score > buy_thresh + 0.1 and slope > 0.08 and up_days >= 3:
                return "BUY"
            if score < sell_thresh - 0.1 and slope < -0.08 and down_days >= 3:
                return "SELL"
        return "HOLD"

    def get_action_level(self, score: float) -> str:
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
        if len(score_history) < 10:
            return params
        window = min(self.get_dynamic_history_days(volatility), len(score_history))
        recent = [s["score"] for s in score_history[-window:]]
        avg = sum(recent) / len(recent)
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) >= 3 else 0
        short_recent = [s["score"] for s in score_history[-min(3, len(score_history)):]]
        short_slope = np.polyfit(range(len(short_recent)), short_recent, 1)[0] if len(short_recent) >= 2 else 0
        adjust_mult = ADJUST_MULT_BASE / market_factor
        adjusted = params.copy()

        if slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = min(ADJUST_BUY_DELTA_MAX, abs(avg - params["BUY_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["BUY_THRESHOLD"] = max(0.35, params["BUY_THRESHOLD"] - delta)
        elif slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = min(ADJUST_BUY_DELTA_MAX, abs(avg - params["BUY_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["BUY_THRESHOLD"] = min(0.65, params["BUY_THRESHOLD"] + delta)
        else:
            adjusted["BUY_THRESHOLD"] = params["BUY_THRESHOLD"] * 0.9 + DEFAULT_PARAMS["BUY_THRESHOLD"] * 0.1

        if slope < -0.02 and avg < -0.15 and short_slope < -0.03:
            delta = min(ADJUST_SELL_DELTA_MAX, abs(avg - params["SELL_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["SELL_THRESHOLD"] = max(-0.45, params["SELL_THRESHOLD"] - delta)
        elif slope > 0.02 and avg > 0.15 and short_slope > 0.03:
            delta = min(ADJUST_SELL_DELTA_MAX, abs(avg - params["SELL_THRESHOLD"]) * 0.15) * adjust_mult
            adjusted["SELL_THRESHOLD"] = min(-0.15, params["SELL_THRESHOLD"] + delta)
        else:
            adjusted["SELL_THRESHOLD"] = params["SELL_THRESHOLD"] * 0.9 + DEFAULT_PARAMS["SELL_THRESHOLD"] * 0.1

        if volatility > VOL_HIGH_CONFIRM:
            adjusted["CONFIRM_DAYS"] = min(5, int(round(params["CONFIRM_DAYS"] * 1.1)))
        elif volatility < 0.01:
            adjusted["CONFIRM_DAYS"] = max(2, int(round(params["CONFIRM_DAYS"] * 0.9)))
        else:
            adjusted["CONFIRM_DAYS"] = int(round(params["CONFIRM_DAYS"] * 0.95 + DEFAULT_PARAMS["CONFIRM_DAYS"] * 0.05))
        return adjusted

    # ---------- 核心分析计算 ----------
    def _core_analysis(self, ctx: ETFContext) -> ETFContext:
        """执行 ETF 分析的核心计算，填充 ctx 并返回"""
        if ctx.real_price is None:
            ctx.error = "实时价格获取失败"
            return ctx
        if ctx.hist_df is None or len(ctx.hist_df) < 20:
            ctx.error = "历史数据不足"
            return ctx

        ctx.change_pct = self.calc_change_pct(ctx.real_price, ctx.hist_df, ctx.today)

        d = ctx.hist_df.iloc[-1]
        ma20, vol_ma, volume = d["ma_short"], d["vol_ma"], d["volume"]
        rsi, boll_up, boll_low, williams_r = d["rsi"], d["boll_up"], d["boll_low"], d["williams_r"]
        atr_pct = d["atr"] / ctx.real_price if ctx.real_price > 0 else 0
        ctx.atr_pct = atr_pct

        recent_high = d.get(f"recent_high_{ctx.params['RECENT_HIGH_WINDOW']}",
                            ctx.hist_df["high"].rolling(ctx.params["RECENT_HIGH_WINDOW"]).max().iloc[-1])
        recent_low = d.get(f"recent_low_{ctx.params['RECENT_LOW_WINDOW']}",
                           ctx.hist_df["low"].rolling(ctx.params["RECENT_LOW_WINDOW"]).min().iloc[-1])

        macd_golden = kdj_golden = 0
        if len(ctx.hist_df) >= 2:
            prev = ctx.hist_df.iloc[-2]
            macd_golden = 1 if (d["macd_dif"] > d["macd_dea"] and prev["macd_dif"] <= prev["macd_dea"]) else 0
            kdj_golden = 1 if (d["kdj_k"] > d["kdj_d"] and prev["kdj_k"] <= prev["kdj_d"]) else 0

        ret_etf_5d = (ctx.real_price / ctx.hist_df.iloc[-5]["close"] - 1) if len(ctx.hist_df) >= 5 else 0

        weekly_above = weekly_below = False
        if ctx.weekly_df is not None and not ctx.weekly_df.empty:
            w = ctx.weekly_df.iloc[-1]
            if "ma_short" in w.index and not pd.isna(w["ma_short"]):
                weekly_above = w["close"] > w["ma_short"]
                weekly_below = w["close"] < w["ma_short"]
        ctx.weekly_above = weekly_above
        ctx.weekly_below = weekly_below

        # TMSV (传入市场状态和波动率以启用动态权重)
        market_status = ctx.market.get("macro_status", "震荡偏弱")
        try:
            tmsv_series = self.compute_tmsv(ctx.hist_df, market_status, atr_pct)
            tmsv = tmsv_series.iloc[-1] if not tmsv_series.empty else 50.0
            tmsv = 50.0 if np.isnan(tmsv) else tmsv
        except Exception:
            tmsv = 50.0
        ctx.tmsv = tmsv
        ctx.tmsv_strength = tmsv / 100.0

        ctx.downside_momentum = d.get("downside_momentum_raw", 0.0)
        ctx.max_drawdown_pct = (recent_high - ctx.real_price) / recent_high if recent_high > 0 else 0.0

        ctx.buy_factors, ctx.sell_factors = self._compute_factors(
            ctx.real_price, ma20, volume, vol_ma, macd_golden, kdj_golden,
            rsi, boll_up, boll_low, williams_r, ret_etf_5d, ctx.market["ret_market_5d"],
            weekly_above, weekly_below, recent_high, recent_low, atr_pct,
            ctx.market["market_above_ma20"], ctx.market["market_above_ma60"],
            ctx.market["market_amount_above_ma20"], ctx.tmsv_strength,
            ctx.downside_momentum, ctx.max_drawdown_pct
        )

        ctx.buy_score = sum(self.buy_weights.get(k, 0) * ctx.buy_factors[k] for k in ctx.buy_factors)
        ctx.sell_score = sum(self.sell_weights.get(k, 0) * ctx.sell_factors[k] for k in ctx.sell_factors)

        # 硬止损覆盖
        if ctx.sell_factors.get("max_drawdown_stop", 0) > 0 or ctx.sell_factors.get("stop_loss_ma_break", 0) > 0:
            ctx.final_score = -1.0
            ctx.raw_score = ctx.buy_score - ctx.sell_score
            return ctx

        sentiment = ctx.market.get("sentiment_factor", 1.0)
        if sentiment >= SENTIMENT_OVERHEAT_THRESHOLD:
            ctx.buy_score *= SENTIMENT_PENALTY_FACTOR
            ctx.raw_score = ctx.buy_score - ctx.sell_score
        else:
            ctx.raw_score = ctx.buy_score - ctx.sell_score

        # 非线性评分变换
        transformed_raw = self._nonlinear_score_transform(ctx.raw_score, market_status)
        env_factor = self._clip_env_factor(ctx.market["market_factor"], sentiment)
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
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy())
        ctx = self._core_analysis(ctx)

        if ctx.error:
            if "实时价格" in ctx.error:
                out = (f"{pad_display(name, 16)} {pad_display(code, 12)} "
                       f"{pad_display('获取失败', 8)} {pad_display('0.00%', 8, 'right')} "
                       f"{pad_display('0.00', 6, 'right')}  {pad_display('价格缺失', 10)}")
            else:
                price_str = f"{real_price:.3f}" if real_price else "N/A"
                change_str = f"{ctx.change_pct:+.2f}%"
                out = (f"{pad_display(name, 16)} {pad_display(code, 12)} "
                       f"{pad_display(price_str, 8, 'right')} "
                       f"{pad_display(change_str, 8, 'right')} "
                       f"{pad_display('0.00', 6, 'right')}  {pad_display('数据不足', 10)}")
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

        action = self.get_action(final, state["score_history"], self.params, ctx.atr_pct)
        action_level = self.get_action_level(final)

        # 预备信号展示调整
        display_action = action_level
        if action == "PREP_BUY":
            display_action = "预备买入"
        elif action == "PREP_SELL":
            display_action = "预备卖出"

        risk_warning = ""
        if len(state["score_history"]) >= RISK_WARNING_DAYS:
            recent_scores = [s["score"] for s in state["score_history"][-RISK_WARNING_DAYS:]]
            if all(s < RISK_WARNING_THRESHOLD for s in recent_scores):
                risk_warning = f"风险提示:连续{RISK_WARNING_DAYS}天评分低于{RISK_WARNING_THRESHOLD}"
            elif final < -0.5 or final > 0.8:
                risk_warning = f"风险提示:极端评分{final:.2f}"
            elif ctx.atr_pct > 0.03:
                risk_warning = f"风险提示:高波动{ctx.atr_pct:.3f}"

        price_str = f"{real_price:.3f}"
        change_str = f"{ctx.change_pct:+.2f}%"
        final_str = f"{final:.2f}"
        output = (f"{pad_display(name, 16)} {pad_display(code, 12)} "
                  f"{pad_display(price_str, 8, 'right')} "
                  f"{pad_display(change_str, 8, 'right')} "
                  f"{pad_display(final_str, 6, 'right')}  "
                  f"{pad_display(display_action, 10)}")
        if risk_warning:
            output += f"  {risk_warning}"
        signal = {"action": action, "name": name, "code": code, "score": final} if action in ("BUY", "SELL") else None
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
        ctx = ETFContext(code, name, real_price, hist_df, weekly_df, today, market, self.params.copy())
        ctx = self._core_analysis(ctx)

        if ctx.error:
            return f"【{name} ({code})】{ctx.error}，无法分析。"

        final = ctx.final_score
        action_level = self.get_action_level(final)

        lines = []
        lines.append("=" * 70)
        lines.append(f"ETF详细分析报告 - {name} ({code})")
        lines.append(f"分析时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        lines.append(f"实时价格：{real_price:.3f}")
        lines.append(f"涨跌幅：{ctx.change_pct:+.2f}%")
        lines.append(f"市场状态：{market['macro_status']}，市场因子：{market['market_factor']:.2f}，情绪因子：{market['sentiment_factor']:.2f}")
        if market.get("sentiment_risk_tip"):
            lines.append(f"情绪风险提示：{market['sentiment_risk_tip']}")
        lines.append(f"波动率(ATR%)：{ctx.atr_pct*100:.2f}%")
        lines.append(f"TMSV复合强度：{ctx.tmsv:.1f} (强度系数 {ctx.tmsv_strength:.3f})")
        lines.append(f"最大回撤：{ctx.max_drawdown_pct*100:.2f}%")
        lines.append("")

        col_name, col_strength, col_weight, col_contrib = 25, 8, 8, 8

        def row_line(items):
            return "".join([
                pad_display(items[0], col_name),
                pad_display(items[1], col_strength, "right"),
                pad_display(items[2], col_weight, "right"),
                pad_display(items[3], col_contrib, "right"),
            ])

        lines.append("【买入因子详情】")
        lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
        lines.append("-" * 50)
        buy_contribs = sorted(
            [(k, ctx.buy_factors[k], self.buy_weights.get(k, 0), self.buy_weights.get(k, 0) * ctx.buy_factors[k]) for k in ctx.buy_factors],
            key=lambda x: x[3], reverse=True
        )
        for name_f, s, w, contrib in buy_contribs:
            lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
        lines.append(row_line(["买入总分", "", "", f"{ctx.buy_score:.3f}"]))
        lines.append("")
        lines.append("【卖出因子详情】")
        lines.append(row_line(["因子名称", "强度", "权重", "贡献"]))
        lines.append("-" * 50)
        sell_contribs = sorted(
            [(k, ctx.sell_factors[k], self.sell_weights.get(k, 0), self.sell_weights.get(k, 0) * ctx.sell_factors[k]) for k in ctx.sell_factors],
            key=lambda x: x[3], reverse=True
        )
        for name_f, s, w, contrib in sell_contribs:
            lines.append(row_line([name_f, f"{s:.3f}", f"{w:.3f}", f"{contrib:.3f}"]))
        lines.append(row_line(["卖出总分", "", "", f"{ctx.sell_score:.3f}"]))
        lines.append("")
        lines.append("【评分合成】")
        lines.append(f"原始净分 = 买入总分 - 卖出总分 = {ctx.buy_score:.3f} - {ctx.sell_score:.3f} = {ctx.raw_score:.3f}")
        lines.append("非线性变换: tanh(2.5 * raw) → 最终评分 = 变换后 × 环境因子")
        env_factor = self._clip_env_factor(market["market_factor"], market["sentiment_factor"])
        lines.append(f"        = tanh(2.5×{ctx.raw_score:.3f}) × {env_factor:.2f} = {final:.3f}")
        lines.append(f"操作等级：{action_level}")

        if ai_client:
            lines.append("")
            lines.append("【AI 专业点评】")
            ai_comment = ai_client.comment_on_etf(
                code, name, final, action_level, market["macro_status"],
                market["market_factor"], market["sentiment_factor"],
                self.buy_weights, self.sell_weights, ctx.buy_factors, ctx.sell_factors,
                ctx.tmsv, ctx.atr_pct
            )
            lines.append(ai_comment)
        else:
            lines.append("")
            lines.append("【AI 专业点评】未配置 API_KEY，无法生成。")
        lines.append("=" * 70)
        return "\n".join(lines)


# ========================== 批量分析入口（略作调整以使用缓存） ==========================
def run_batch_analysis(api_key: Optional[str] = None, target_code: Optional[str] = None):
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

    ai_client = None
    if api_key:
        ai_client = AIClient(api_key)
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
        logger.info(f"简单规则市场状态: {market_state}, 因子: {market_factor}")

    sentiment = 1.0
    sentiment_risk_tip = ""
    if AKSHARE_AVAILABLE:
        try:
            sentiment_indicators = fetcher.fetch_sentiment_indicators()
            sentiment = fetcher.compute_sentiment_factor(sentiment_indicators)
            sentiment_risk_tip = fetcher.get_sentiment_risk_tip(sentiment)
            logger.info(f"综合情绪因子: {sentiment:.3f}")
        except Exception as e:
            logger.warning(f"获取情绪指标失败，使用后备RSI情绪: {e}")
            sentiment = fetcher.get_sentiment_factor_simple(macro_df)
    else:
        sentiment = fetcher.get_sentiment_factor_simple(macro_df)
        logger.info(f"使用后备RSI情绪因子: {sentiment:.3f}")

    mkt = market_df.iloc[-1]
    market_info = {
        "macro_status": market_state,
        "market_factor": market_factor,
        "sentiment_factor": sentiment,
        "sentiment_risk_tip": sentiment_risk_tip,
        "market_above_ma20": mkt["close"] > mkt["ma_short"],
        "market_above_ma60": mkt["close"] > mkt.get("ma_long", mkt["ma_short"]),
        "market_amount_above_ma20": mkt["amount"] > mkt["amount_ma"],
        "ret_market_5d": (mkt["close"] / market_df.iloc[-5]["close"] - 1) if len(market_df) >= 5 else 0,
    }

    analyzer.set_market_info(market_info)

    params = DEFAULT_PARAMS.copy()
    if volatility > 0.04:
        params.update({"BUY_THRESHOLD": 0.65, "SELL_THRESHOLD": -0.35, "CONFIRM_DAYS": 5, "QUICK_BUY_THRESHOLD": 0.75})
    elif volatility > 0.02:
        params.update({"BUY_THRESHOLD": 0.6, "SELL_THRESHOLD": -0.3, "CONFIRM_DAYS": 4, "QUICK_BUY_THRESHOLD": 0.7})
    elif volatility < 0.01:
        params.update({"BUY_THRESHOLD": 0.4, "SELL_THRESHOLD": -0.1, "CONFIRM_DAYS": 2, "QUICK_BUY_THRESHOLD": 0.5})
    analyzer.params = params

    state = fetcher.load_state()
    buy_w, sell_w = DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()

    if ai_client:
        def _generate_weights_from_ai():
            ai_buy, ai_sell = ai_client.generate_weights(
                market_state, sentiment, market_info["market_above_ma20"],
                market_info["market_above_ma60"], market_info["market_amount_above_ma20"], volatility
            )
            trust = min(analyzer.compute_dynamic_trust(ai_buy, DEFAULT_BUY_WEIGHTS),
                        analyzer.compute_dynamic_trust(ai_sell, DEFAULT_SELL_WEIGHTS))
            corr_buy = analyzer.compute_factor_correlation(None, list(DEFAULT_BUY_WEIGHTS.keys()))
            corr_sell = analyzer.compute_factor_correlation(None, list(DEFAULT_SELL_WEIGHTS.keys()))
            ai_buy = analyzer.apply_correlation_penalty(ai_buy, list(DEFAULT_BUY_WEIGHTS.keys()), corr_buy)
            ai_sell = analyzer.apply_correlation_penalty(ai_sell, list(DEFAULT_SELL_WEIGHTS.keys()), corr_sell)
            buy_w = analyzer.blend_weights(ai_buy, DEFAULT_BUY_WEIGHTS, trust)
            sell_w = analyzer.blend_weights(ai_sell, DEFAULT_SELL_WEIGHTS, trust)
            if sentiment >= 1.25:
                boost = 0.1
                sell_w["profit_target_hit"] = min(0.5, sell_w.get("profit_target_hit", 0) + boost)
                other_keys = [k for k in sell_w if k != "profit_target_hit"]
                if other_keys:
                    reduce_each = boost / len(other_keys)
                    for k in other_keys:
                        sell_w[k] = max(0.02, sell_w[k] - reduce_each)
                total = sum(sell_w.values())
                sell_w = {k: v / total for k, v in sell_w.items()}
            return buy_w, sell_w

        cache_key = fetcher._get_cache_key_fuzzy(
            market_state, sentiment, market_info["market_above_ma20"],
            market_info["market_above_ma60"], market_info["market_amount_above_ma20"], volatility
        )
        cache = fetcher._load_cache()
        if cache_key in cache and "buy" in cache[cache_key] and "sell" in cache[cache_key]:
            cached_buy = validate_and_filter_weights(cache[cache_key]["buy"], DEFAULT_BUY_WEIGHTS.keys(), "缓存买入")
            cached_sell = validate_and_filter_weights(cache[cache_key]["sell"], DEFAULT_SELL_WEIGHTS.keys(), "缓存卖出")
            if cached_buy and cached_sell:
                buy_w, sell_w = cached_buy, cached_sell
            else:
                buy_w, sell_w = _generate_weights_from_ai()
                cache[cache_key] = {"buy": buy_w, "sell": sell_w}
                fetcher._save_cache(cache)
        else:
            buy_w, sell_w = _generate_weights_from_ai()
            cache[cache_key] = {"buy": buy_w, "sell": sell_w}
            fetcher._save_cache(cache)

    analyzer.set_weights(buy_w, sell_w)

    if target_code:
        target = etf_list[etf_list["代码"] == target_code]
        if target.empty:
            print(f"未找到代码 {target_code}，请检查 positions.csv")
            fetcher.logout()
            return
        row = target.iloc[0]
        code, name = row["代码"], row["名称"]
        cache_key_hist = analyzer._get_cache_key(code, start, today_str)
        hist = fetcher.get_daily_data(code, start, today_str)
        if hist is not None:
            hist = analyzer.calculate_indicators(
                hist, need_amount_ma=False,
                recent_high_window=params["RECENT_HIGH_WINDOW"],
                recent_low_window=params["RECENT_LOW_WINDOW"],
                use_cache=True, cache_key=cache_key_hist
            )
        weekly = fetcher.get_weekly_data(code, start, today_str)
        etf_state = state.get(code, {})
        real_price = fetcher.get_realtime_price(code)
        report = analyzer.detailed_analysis(code, name, real_price, hist, weekly, market_info, today, etf_state, ai_client)
        print(report)
        fetcher.logout()
        return
    else:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ETF 分析报告")
        print(f"市场状态: {market_state}, 市场因子: {market_factor:.2f}")
        if sentiment_risk_tip:
            print(f"情绪因子: {sentiment:.3f} - {sentiment_risk_tip}")
        else:
            print(f"情绪因子: {sentiment:.3f}")

        print(pad_display("名称", 16), pad_display("代码", 12), pad_display("价格", 8, "right"),
              pad_display("涨跌幅", 8, "right"), pad_display("评分", 6, "right"), "  " + pad_display("操作", 10))
        print("-" * 80)

        output_lines = []
        results = []
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = []
            for _, row in etf_list.iterrows():
                code, name = row["代码"], row["名称"]
                cache_key_hist = analyzer._get_cache_key(code, start, today_str)
                hist = fetcher.get_daily_data(code, start, today_str)
                if hist is not None:
                    hist = analyzer.calculate_indicators(
                        hist, need_amount_ma=False,
                        recent_high_window=params["RECENT_HIGH_WINDOW"],
                        recent_low_window=params["RECENT_LOW_WINDOW"],
                        use_cache=True, cache_key=cache_key_hist
                    )
                weekly = fetcher.get_weekly_data(code, start, today_str)
                s = state.get(code, {})
                futures.append(ex.submit(
                    analyzer.analyze_single_etf, code, name, fetcher.get_realtime_price(code),
                    hist, weekly, market_info, today, s
                ))
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
        subject = f"ETF分析报告 - {today_str}"
        send_email(subject, "\n".join(output_lines))