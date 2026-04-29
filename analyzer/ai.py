#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI 服务模块：调用 DeepSeek API 生成权重、市场状态分析、ETF 评论及止盈建议。
深度融合后新增：参数建议、个股权重微调、批量评论/止盈。
"""
import json
import re
import logging
import openai
import pandas as pd
from typing import Dict, Tuple, Optional, List

from .config import (
    DEFAULT_BUY_WEIGHTS,
    DEFAULT_SELL_WEIGHTS,
    RSI_OVERSOLD_THRESH,
    AI_BATCH_COMMENT_SIZE,
    AI_BATCH_TAKE_PROFIT_SIZE,
    AI_PER_ETF_WEIGHT_MAX_DELTA,
)
from .utils import validate_and_filter_weights

logger = logging.getLogger(__name__)


class AIClient:
    """封装 DeepSeek API 调用，负责权重生成、市场状态分析、ETF 评论及止盈建议生成。"""

    def __init__(self, api_key: str):
        """
        初始化 AI 客户端

        Args:
            api_key: DeepSeek API 密钥
        """
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    # ---------- 通用 JSON 提取 ----------
    @staticmethod
    def _extract_json(content: Optional[str]) -> dict:
        """从响应内容中提取 JSON 对象"""
        if content is None :
            raise ValueError("content is None")
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if not json_match:
            raise ValueError("未找到JSON")
        return json.loads(json_match.group())

    # ---------- 权重生成 ----------
    @staticmethod
    def _build_weights_prompt(macro_status: str, sentiment_factor: float,
                              market_above_ma20: bool, market_above_ma60: bool,
                              market_amount_above_ma20: bool, volatility: float) -> str:
        """
        构建权重生成提示词（已移除 trailing_stop 相关约束）
        """
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

买入因子说明：
- tmsv_score 是复合指标(0-100)，强度为 tmsv/100。
- 新增 rsi_oversold：当 RSI 低于 {RSI_OVERSOLD_THRESH} 时激活，强度为 (30 - RSI)/30，用于捕捉超卖反弹。在恐慌情绪或震荡市中可给予 0.08~0.15 的较高权重，但在强趋势牛市中应保持较低权重（≤ 0.06），避免逆势。
卖出因子说明：downside_momentum（下跌动量）、max_drawdown_stop（最大回撤止损，通常权重应较低）。
注意：策略已去除固定止盈因子，卖出权重应聚焦于风险控制和趋势反转。

特别提示：以下指标存在较高相关性，请避免同时给予高权重以防止信号放大失真：
1. 威廉指标、RSI、KDJ 均为超买超卖类指标，震荡市可侧重一个，趋势市建议全部降权。
2. TMSV 复合指标已包含趋势、动量和量价信息，若给予 TMSV 较高权重（>0.15），应相应降低其子成分因子（如价格站上均线、MACD金叉等）的权重。
3. 布林带突破信号通常与威廉/RSI极端值相伴，请考虑在趋势明确时侧重布林带，震荡时侧重超买超卖/超卖因子。

严格约束：
1. 任何因子的权重不得低于 0.02（除非该因子在当前市场状态下完全无效）。
2. tmsv_score 是复合指标，其权重在 [0.25,0.40]。
3. 价格站上均线(price_above_ma20)和成交量(volume_above_ma5)是趋势确认的核心，合计权重不应低于 0.30。
4. 卖出因子中，止损类(stop_loss)在震荡市应保持中等权重(0.05~0.15)，超买类(williams, rsi)在当前市场可适当提高至 0.12~0.18。
5. 权重分布应体现分散化原则，单个因子权重上限为 0.40（熊市止损除外）。
6. 新因子 downside_momentum 在下跌趋势明显时赋予较高权重（0.10~0.20），max_drawdown_stop 平时权重可接近0。
7. 根据市场状态灵活调整 rsi_oversold 权重：恐慌/震荡时增大，强势牛市时减小。
请输出买入权重和卖出权重，JSON格式：{{"buy":{{...}},"sell":{{...}}}}，每个部分总和为1。禁止添加未列出的键。严格JSON，无解释。"""

    def generate_weights(self, macro_status: str, sentiment_factor: float,
                         market_above_ma20: bool, market_above_ma60: bool,
                         market_amount_above_ma20: bool, volatility: float) -> Tuple[Dict, Dict]:
        """
        调用 AI 生成原始权重（未混合、未惩罚）

        Args:
            macro_status: 市场状态标签
            sentiment_factor: 情绪因子
            market_above_ma20: 大盘站上20日均线
            market_above_ma60: 大盘站上60日均线
            market_amount_above_ma20: 成交额超20日均额
            volatility: 波动率

        Returns:
            (买入权重字典, 卖出权重字典) 失败时返回默认权重
        """
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
                if content is None: raise ValueError("AI return None")
                data = self._extract_json(content)
                if "buy" not in data or "sell" not in data:
                    raise ValueError("缺少buy/sell字段")
                # 验证并归一化权重
                ai_buy = validate_and_filter_weights(
                    data["buy"], list(DEFAULT_BUY_WEIGHTS.keys()), "AI买入权重"
                )
                ai_sell = validate_and_filter_weights(
                    data["sell"], list(DEFAULT_SELL_WEIGHTS.keys()), "AI卖出权重"
                )
                if ai_buy and ai_sell:
                    return ai_buy, ai_sell
            except Exception as e:
                logger.warning(f"AI权重生成失败(尝试{attempt+1}/3): {e}")
                import time
                time.sleep(2**attempt)
        logger.warning("AI权重生成失败，使用默认权重")
        return DEFAULT_BUY_WEIGHTS.copy(), DEFAULT_SELL_WEIGHTS.copy()

    # ---------- 参数建议（新增） ----------
    def generate_params_advice(
        self,
        macro_status: str,
        sentiment_factor: float,
        volatility: float,
        recent_score_slope: float = 0.0,
    ) -> Dict:
        """
        AI 根据市场环境给出交易参数的调整建议

        Args:
            macro_status: 市场状态
            sentiment_factor: 情绪因子
            volatility: 波动率
            recent_score_slope: 近期评分趋势斜率（可选）

        Returns:
            字典包含：buy_threshold_shift, sell_threshold_shift, confirm_days_shift
            范围在 [-0.1, 0.1] 或 [-2, 2] 天
        """
        prompt = f"""
你是一位量化交易参数优化专家。当前市场：
- 状态：{macro_status}
- 情绪因子：{sentiment_factor:.2f}（>1乐观，<1悲观）
- 波动率(ATR%)：{volatility*100:.2f}%
- 近期评分趋势斜率：{recent_score_slope:.3f}（正=改善，负=恶化）

请给出建议的交易参数偏移（相对于默认值）：
- buy_threshold_shift: 买入阈值偏移，范围[-0.08,0.08]，负值更容易触发买入。
- sell_threshold_shift: 卖出阈值偏移，范围[-0.08,0.08]，负值更容易触发卖出。
- confirm_days_shift: 确认天数偏移，范围[-1,2]天，整数，正数更保守。

原则：高波动或趋势不明朗时提高确认天数、降低买卖阈值灵敏度；恐慌时可略微降低买入阈值；过热时略微提高卖出阈值。
输出严格JSON：{{"buy_threshold_shift":..., "sell_threshold_shift":..., "confirm_days_shift":...}}。无解释。"""
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "输出严格JSON。"},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=120,
                    temperature=0.0,
                    timeout=8,
                )
                if resp.choices[0].message.content is None: raise ValueError("AI return None");
                data = self._extract_json(resp.choices[0].message.content)
                # 类型检查和边界限制
                buy_s = max(-0.08, min(0.08, float(data.get("buy_threshold_shift", 0))))
                sell_s = max(
                    -0.08, min(0.08, float(data.get("sell_threshold_shift", 0)))
                )
                days_s = int(max(-1, min(2, float(data.get("confirm_days_shift", 0)))))
                return {
                    "buy_threshold_shift": buy_s,
                    "sell_threshold_shift": sell_s,
                    "confirm_days_shift": days_s,
                }
            except Exception as e:
                logger.warning(f"AI参数建议生成失败(尝试{attempt+1}): {e}")
                import time

                time.sleep(1)
        return {
            "buy_threshold_shift": 0.0,
            "sell_threshold_shift": 0.0,
            "confirm_days_shift": 0,
        }

    # ---------- 个股权重微调（新增） ----------
    def adjust_weights_per_etf(
        self,
        global_buy: Dict[str, float],
        global_sell: Dict[str, float],
        etf_features: Dict,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        根据单只 ETF 的特征，微调全局权重

        Args:
            global_buy: 全局买入权重
            global_sell: 全局卖出权重
            etf_features: 包含 tmsv, volatility, rsi, above_ma30, change_5d 等

        Returns:
            (调整后的买入权重, 调整后的卖出权重) ，均归一化
        """
        if not etf_features:
            return global_buy, global_sell

        prompt = f"""
你是一位量化策略调优专家。当前ETF特征：
- TMSV: {etf_features.get('tmsv', 50):.1f}
- 波动率: {etf_features.get('atr_pct', 0.02)*100:.2f}%
- RSI: {etf_features.get('rsi', 50):.1f}
- 站上30日均线: {'是' if etf_features.get('above_ma30', True) else '否'}
- 近5日涨跌: {etf_features.get('change_5d', 0)*100:.2f}%

现有全局买入权重：{json.dumps(global_buy, ensure_ascii=False)}
现有全局卖出权重：{json.dumps(global_sell, ensure_ascii=False)}

请返回各因子的权重调整系数（delta），范围[-{AI_PER_ETF_WEIGHT_MAX_DELTA}, {AI_PER_ETF_WEIGHT_MAX_DELTA}]，表示在全局权重上增加或减少的量。
注意：最终权重 = 全局权重 + delta，后自行归一化；请保持总和为1；只输出调整系数JSON，键名与输入相同。
输出格式：{{"buy_deltas": {{...}}, "sell_deltas": {{...}}}}。无解释。"""
        for attempt in range(2):
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {
                            "role": "system",
                            "content": "输出严格JSON，键名与给定因子完全一致。",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=500,
                    temperature=0.0,
                    timeout=8,
                )
                data = self._extract_json(resp.choices[0].message.content)
                buy_deltas = data.get("buy_deltas", {})
                sell_deltas = data.get("sell_deltas", {})

                # 应用调整
                adj_buy = {}
                for k in global_buy:
                    delta = float(buy_deltas.get(k, 0))
                    delta = max(-AI_PER_ETF_WEIGHT_MAX_DELTA, min(AI_PER_ETF_WEIGHT_MAX_DELTA, delta))
                    adj_buy[k] = max(0.0, global_buy[k] + delta)  # ★ 关键：保证非负

                adj_sell = {}
                for k in global_sell:
                    delta = float(sell_deltas.get(k, 0))
                    delta = max(-AI_PER_ETF_WEIGHT_MAX_DELTA, min(AI_PER_ETF_WEIGHT_MAX_DELTA, delta))
                    adj_sell[k] = max(0.0, global_sell[k] + delta)

                # 归一化
                buy_sum = sum(adj_buy.values())
                if buy_sum > 0:
                    adj_buy = {k: v / buy_sum for k, v in adj_buy.items()}
                else:
                    adj_buy = global_buy.copy()  # 回退

                sell_sum = sum(adj_sell.values())
                if sell_sum > 0:
                    adj_sell = {k: v / sell_sum for k, v in adj_sell.items()}
                else:
                    adj_sell = global_sell.copy()

                return adj_buy, adj_sell
            except Exception as e:
                logger.warning(f"个股权重微调失败(尝试{attempt+1}): {e}")
                import time
                time.sleep(1)
        return global_buy, global_sell
    # ---------- 市场状态分析（增强输入） ----------
    def refine_market_state(
        self, market_df: pd.DataFrame, extra_sentiment: Optional[Dict] = None
    ) -> Tuple[str, float]:
        """
        使用 AI 分析市场状态，返回（状态标签, 市场因子）

        Args:
            market_df: 包含技术指标的日线数据（含近日20日数据）
            extra_sentiment: 可选的情绪指标字典，可包含 north_net, main_net_pct, zt_dt_ratio 等

        Returns:
            (市场状态描述字符串, 市场因子系数[0.6,1.4])
        """
        recent = market_df.tail(20)
        # 计算近期统计特征
        close_pct = recent["close"].pct_change().mean()
        vol_pct = recent["volume"].pct_change().mean()
        volatility = (recent["close"].pct_change().std()) * 100
        above_ma20 = recent["close"].iloc[-1] > recent["ma_short"].iloc[-1]
        above_ma60 = recent["close"].iloc[-1] > recent.get("ma_long", recent["ma_short"]).iloc[-1]

        extra_str = ""
        if extra_sentiment:
            parts = []
            if "north_net" in extra_sentiment:
                parts.append(f"北向净买额(亿):{extra_sentiment['north_net']:.2f}")
            if "main_net_pct" in extra_sentiment:
                parts.append(f"主力净流入占比:{extra_sentiment['main_net_pct']:.2f}%")
            if "zt_dt_ratio" in extra_sentiment:
                parts.append(f"涨跌停比:{extra_sentiment['zt_dt_ratio']:.2f}")
            if "up_down_ratio" in extra_sentiment:
                parts.append(f"上涨下跌家数比:{extra_sentiment['up_down_ratio']:.2f}")
            if parts:
                extra_str = "\n额外情绪指标：" + "，".join(parts)

        prompt = f"""市场分析专家。根据以下大盘最近20日数据判断市场状态并给出市场因子(0.6-1.4)：
- 平均日涨跌幅:{close_pct:.4f}
- 成交量变化率:{vol_pct:.4f}
- 日波动率:{volatility:.2f}%
- 站上20日线:{"是" if above_ma20 else "否"}
- 站上60日线:{"是" if above_ma60 else "否"}
- RSI:{recent['rsi'].iloc[-1]:.1f}
- MACD:{recent['macd_dif'].iloc[-1]:.3f}/{recent['macd_dea'].iloc[-1]:.3f}
- 布林位置:{(recent['close'].iloc[-1]-recent['boll_mid'].iloc[-1])/recent['boll_std'].iloc[-1]:.2f}σ
- 威廉:{recent['williams_r'].iloc[-1]:.1f}{extra_str}
输出JSON:{{"state":"市场状态标签","factor":因子系数}} 可选标签:强势牛市、正常牛市、震荡偏强、震荡偏弱、弱势反弹、熊市下跌中继、熊市加速下跌。"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": "输出严格JSON。"}, {"role": "user", "content": prompt}],
                max_tokens=200, temperature=0.0, timeout=10
            )
            data = self._extract_json(resp.choices[0].message.content)
            state = data.get("state", "震荡偏弱")
            factor = max(0.6, min(1.4, float(data.get("factor", 1.0))))
            return state, factor
        except Exception as e:
            logger.error(f"市场状态AI分析失败: {e}")
            # 回退规则：按均线简单判断
            if above_ma20 and above_ma60:
                return "正常牛市", 1.2
            if not above_ma20 and not above_ma60:
                return "熊市下跌", 0.8
            return "震荡偏弱", 1.0

    # ---------- 单个 ETF 评论（保留原接口，内部可被批量方法调用） ----------
    def comment_on_etf(self, code: str, name: str, final_score: float,
                       action_level: str, market_state: str, market_factor: float,
                       sentiment_factor: float, buy_weights: Dict, sell_weights: Dict,
                       buy_factors: Dict, sell_factors: Dict, tmsv: float, atr_pct: float) -> str:
        """单个ETF评论（向后兼容）"""
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
            content = resp.choices[0].message.content
            if content is None: raise ValueError("AI return None")
            return content.strip()
        except Exception as e:
            logger.error(f"AI评论生成失败: {e}")
            return "（AI 评论生成失败）"

    # ---------- 批量 ETF 评论（新增） ----------
    def batch_comment_on_etfs(self, etf_list: List[Dict]) -> List[str]:
        """
        批量生成多只 ETF 的点评，返回与输入顺序一致的评论列表

        Args:
            etf_list: 每个元素包含字段 code, name, final_score, action_level, market_state,
                      market_factor, sentiment_factor, buy_weights, sell_weights,
                      buy_factors, sell_factors, tmsv, atr_pct

        Returns:
            评论字符串列表
        """
        if not etf_list:
            return []

        # 分批处理
        results = [""] * len(etf_list)
        for i in range(0, len(etf_list), AI_BATCH_COMMENT_SIZE):
            batch = etf_list[i : i + AI_BATCH_COMMENT_SIZE]
            prompts = []
            for idx, etf in enumerate(batch):
                prompt = f"""
ETF {idx}: {etf['name']}({etf['code']})
市场: {etf['market_state']} 因子{etf['market_factor']:.2f} 情绪{etf['sentiment_factor']:.2f}
评分: {etf['final_score']:.2f} 等级: {etf['action_level']} TMSV: {etf['tmsv']:.1f} ATR: {etf['atr_pct']*100:.2f}%
买入: { {k: f"{v:.2f}" for k,v in list(etf['buy_factors'].items())[:4]} }
卖出: { {k: f"{v:.2f}" for k,v in list(etf['sell_factors'].items())[:4]} }
"""
                prompts.append(prompt)
            combined = (
                "为以下ETF分别生成80~120字专业点评，用JSON返回，键为序号(字符串)，值为点评文本。\n"
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
                    comment = data.get(str(j), "（批量评论缺失）")
                    results[i + j] = comment.strip()
            except Exception as e:
                logger.error(f"批量评论生成失败(批次{i}): {e}")
                for j in range(len(batch)):
                    comment = data.get(str(j), "（批量评论缺失）")
                    results[i + j] = comment.strip() if comment else "（空评）"
        return results

    # ---------- 止盈建议（保留原接口） ----------
    def take_profit_advice(
        self,
        code: str,
        name: str,
        profit_pct: float,
        recent_low: float,
        current_price: float,
        tmsv: float,
        rsi: float,
        atr_pct: float,
        market_state: str,
        sentiment_factor: float,
    ) -> str:
        """单只止盈建议（向后兼容）"""
        prompt = f"""
你是一名量化交易策略师。某ETF当前距近期低点已上涨{profit_pct:.1%}，请给出50~80字的止盈操作建议。
基本信息：
- ETF：{name} ({code})
- 近期低点：{recent_low:.3f}，现价：{current_price:.3f}
- TMSV强度：{tmsv:.1f}，RSI：{rsi:.1f}，ATR波动率：{atr_pct*100:.2f}%
- 市场状态：{market_state}，情绪因子：{sentiment_factor:.2f}
请从技术角度建议：1) 是否应分批止盈；2) 建议止盈比例；3) 移动止盈价位参考。简洁专业，不重复数据。
"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=120,
                temperature=0.3,
                timeout=10,
            )
            content = resp.choices[0].message.content
            if content is None: raise ValueError("AI return None")
            return content.strip()
        except Exception as e:
            logger.error(f"AI止盈建议生成失败: {e}")
            return ""

    # ---------- 批量止盈建议（新增） ----------
    def batch_take_profit_advice(self, tp_list: List[Dict]) -> List[str]:
        """
        批量生成止盈建议

        Args:
            tp_list: 每个元素包含 code, name, profit_pct, recent_low, current_price,
                     tmsv, rsi, atr_pct, market_state, sentiment_factor

        Returns:
            建议字符串列表
        """
        if not tp_list:
            return []
        results = [""] * len(tp_list)
        for i in range(0, len(tp_list), AI_BATCH_TAKE_PROFIT_SIZE):
            batch = tp_list[i : i + AI_BATCH_TAKE_PROFIT_SIZE]
            prompts = []
            for idx, etf in enumerate(batch):
                prompt = f"""
ETF {idx}: {etf['name']}({etf['code']}) 低点涨幅{etf['profit_pct']:.1%} 现价{etf['current_price']:.3f} 低点{etf['recent_low']:.3f}
TMSV:{etf['tmsv']:.1f} RSI:{etf['rsi']:.1f} ATR:{etf['atr_pct']*100:.2f}% 市场:{etf['market_state']} 情绪:{etf['sentiment_factor']:.2f}
"""
                prompts.append(prompt)
            combined = (
                "为以下ETF分别生成50~80字止盈操作建议，用JSON返回，键为序号(字符串)，值为建议文本。\n"
                + "\n".join(prompts)
                + '\n输出格式：{"0":"...","1":"..."}'
            )
            try:
                resp = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": combined}],
                    max_tokens=200 * len(batch),
                    temperature=0.3,
                    timeout=20,
                )
                data = self._extract_json(resp.choices[0].message.content)
                for j in range(len(batch)):
                    advice = data.get(str(j), "（批量止盈建议缺失）")
                    results[i + j] = advice.strip()
            except Exception as e:
                logger.error(f"批量止盈建议生成失败(批次{i}): {e}")
                for j in range(len(batch)):
                    advice = data.get(str(j), "（批量止盈建议缺失）")
                    results[i + j] = advice.strip() if advice else "（缺失）"
        return results
