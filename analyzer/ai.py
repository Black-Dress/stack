#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 服务模块：调用 DeepSeek API 生成权重、市场状态分析、ETF 评论。
"""
import json
import re
import logging
import openai
import pandas as pd
from typing import Dict, Tuple, Optional

from .config import DEFAULT_BUY_WEIGHTS, DEFAULT_SELL_WEIGHTS
from .utils import validate_and_filter_weights

logger = logging.getLogger(__name__)


class AIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # ---------- 权重生成 ----------
    def _build_weights_prompt(self, macro_status: str, sentiment_factor: float,
                              market_above_ma20: bool, market_above_ma60: bool,
                              market_amount_above_ma20: bool, volatility: float) -> str:
        buy_keys = list(DEFAULT_BUY_WEIGHTS.keys())
        sell_keys = list(DEFAULT_SELL_WEIGHTS.keys())
        return f"""
你是一个量化交易策略专家。当前市场环境如下：
- 宏观状态：{macro_status}
- 情绪系数：{sentiment_factor}
- 大盘站上20日均线：{"是" if market_above_ma20 else "否"}
- 大盘站上60日均线：{"是" if market_above_ma60 else "否"}
- 成交额高于20日均额：{"是" if market_amount_above_ma20 else "否"}
- 波动率(ATR/收盘价)：{volatility:.3f}

买入因子说明：tmsv_score 是复合指标(0-100)，强度为 tmsv/100。
卖出因子新增：downside_momentum（下跌动量）、max_drawdown_stop（最大回撤止损，通常权重应较低，仅在触发时有效）。

特别提示：以下指标存在较高相关性，请避免同时给予高权重以防止信号放大失真：
1. 威廉指标、RSI、KDJ 均为超买超卖类指标，震荡市可侧重一个，趋势市建议全部降权。
2. TMSV 复合指标已包含趋势、动量和量价信息，若给予 TMSV 较高权重（>0.15），应相应降低其子成分因子（如价格站上均线、MACD金叉等）的权重。
3. 布林带突破信号通常与威廉/RSI极端值相伴，请考虑在趋势明确时侧重布林带，震荡时侧重超买超卖。

严格约束：
1. 任何因子的权重不得低于 0.02（除非该因子在当前市场状态下完全无效）。
2. tmsv_score 是复合指标，其权重在 [0.25,0.40]。
3. 价格站上均线(price_above_ma20)和成交量(volume_above_ma5)是趋势确认的核心，合计权重不应低于 0.30。
4. 卖出因子中，止损类(stop_loss, trailing_stop)在震荡市应保持中等权重(0.05~0.15)，超买类(williams, rsi)在当前市场可适当提高至 0.12~0.18。
5. 权重分布应体现分散化原则，单个因子权重上限为 0.40（熊市止损除外）。
6. 新因子 downside_momentum 在下跌趋势明显时赋予较高权重（0.10~0.20），max_drawdown_stop 作为强制止损信号，平时权重可接近0。

请输出买入权重和卖出权重，JSON格式：{{"buy":{{...}},"sell":{{...}}}}，每个部分总和为1。禁止添加未列出的键。严格JSON，无解释。"""

    def generate_weights(self, macro_status: str, sentiment_factor: float,
                         market_above_ma20: bool, market_above_ma60: bool,
                         market_amount_above_ma20: bool, volatility: float) -> Tuple[Dict, Dict]:
        """调用 AI 生成原始权重（未混合、未惩罚）"""
        prompt = self._build_weights_prompt(
            macro_status, sentiment_factor, market_above_ma20,
            market_above_ma60, market_amount_above_ma20, volatility
        )
        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "输出严格JSON，禁止添加注释或额外文字。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=800,
                    temperature=0.0,
                    timeout=15,
                )
                content = resp.choices[0].message.content
                json_match = re.search(r"\{.*\}", content, re.DOTALL)
                if not json_match:
                    raise ValueError("未找到JSON")
                data = json.loads(json_match.group())
                if "buy" not in data or "sell" not in data:
                    raise ValueError("缺少buy/sell字段")
                ai_buy = validate_and_filter_weights(
                    data["buy"], DEFAULT_BUY_WEIGHTS.keys(), "AI买入权重"
                )
                ai_sell = validate_and_filter_weights(
                    data["sell"], DEFAULT_SELL_WEIGHTS.keys(), "AI卖出权重"
                )
                if ai_buy and ai_sell:
                    return ai_buy, ai_sell
            except Exception as e:
                logger.warning(f"AI权重生成失败(尝试{attempt+1}/3): {e}")
                import time
                time.sleep(2 ** attempt)
        logger.warning("AI权重生成失败，使用默认权重")
        return DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()

    # ---------- 市场状态分析 ----------
    def refine_market_state(self, market_df: pd.DataFrame) -> Tuple[str, float]:
        """使用 AI 分析市场状态，返回 (状态标签, 市场因子)"""
        recent = market_df.tail(20)
        close_pct = recent["close"].pct_change().mean()
        vol_pct = recent["volume"].pct_change().mean()
        volatility = (recent["close"].pct_change().std()) * 100
        above_ma20 = recent["close"].iloc[-1] > recent["ma_short"].iloc[-1]
        above_ma60 = recent["close"].iloc[-1] > recent.get("ma_long", recent["ma_short"]).iloc[-1]

        prompt = f"""市场分析专家。根据以下大盘最近20日数据判断市场状态并给出市场因子(0.6-1.4)：
- 平均日涨跌幅:{close_pct:.4f}
- 成交量变化率:{vol_pct:.4f}
- 日波动率:{volatility:.2f}%
- 站上20日线:{"是" if above_ma20 else "否"}
- 站上60日线:{"是" if above_ma60 else "否"}
- RSI:{recent['rsi'].iloc[-1]:.1f}
- MACD:{recent['macd_dif'].iloc[-1]:.3f}/{recent['macd_dea'].iloc[-1]:.3f}
- 布林位置:{(recent['close'].iloc[-1]-recent['boll_mid'].iloc[-1])/recent['boll_std'].iloc[-1]:.2f}σ
- 威廉:{recent['williams_r'].iloc[-1]:.1f}
输出JSON:{{"state":"市场状态标签","factor":因子系数}} 可选标签:强势牛市、正常牛市、震荡偏强、震荡偏弱、弱势反弹、熊市下跌中继、熊市加速下跌。"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "输出严格JSON。"}, {"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.0, timeout=10
            )
            data = json.loads(re.search(r"\{.*\}", resp.choices[0].message.content, re.DOTALL).group())
            state = data.get("state", "震荡偏弱")
            factor = max(0.6, min(1.4, float(data.get("factor", 1.0))))
            return state, factor
        except Exception as e:
            logger.error(f"市场状态AI分析失败: {e}")
            if above_ma20 and above_ma60:
                return "正常牛市", 1.2
            if not above_ma20 and not above_ma60:
                return "熊市下跌", 0.8
            return "震荡偏弱", 1.0

    # ---------- ETF 评论 ----------
    def comment_on_etf(self, code: str, name: str, final_score: float,
                       action_level: str, market_state: str, market_factor: float,
                       sentiment_factor: float, buy_weights: Dict, sell_weights: Dict,
                       buy_factors: Dict, sell_factors: Dict, tmsv: float, atr_pct: float) -> str:
        """生成 ETF 的 AI 点评"""
        prompt = f"""
你是一名资深 ETF 量化分析师。请根据以下数据，对该 ETF 给出 80~120 字的专业点评。

【基本信息】
ETF名称：{name}，代码：{code}
市场环境：{market_state}，市场因子：{market_factor:.2f}，情绪因子：{sentiment_factor:.2f}

【综合评分】
最终评分：{final_score:.2f}，操作等级：{action_level}

【核心指标】
TMSV复合强度：{tmsv:.1f}，ATR波动率：{atr_pct*100:.2f}%

【买入因子强度（部分）】
{', '.join([f"{k}: {v:.2f}" for k, v in list(buy_factors.items())[:6]])}

【卖出因子强度（部分）】
{', '.join([f"{k}: {v:.2f}" for k, v in list(sell_factors.items())[:6]])}

请用简洁专业的语言，从技术面、市场适配性和风险角度进行点评，不要复述数据。
"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.3,
                timeout=10,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"AI评论生成失败: {e}")
            return "（AI 评论生成失败）"
