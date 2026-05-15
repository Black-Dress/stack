#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI 功能集中模块：单只点评、批量建议（适配新方案）"""
import time
import logging
import json
import openai
from typing import Dict, Any
from .config import AI_ENABLE

logger = logging.getLogger(__name__)


class AIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    @staticmethod
    def _extract_content(resp: Any) -> str:
        content = getattr(resp.choices[0].message, "content", None)
        return content.strip() if isinstance(content, str) and content.strip() else ""

    def comment_on_etf(self, code: str, name: str, final_score: float,
                       action_level: str, market_state: str, market_factor: float,
                       tmsv: float, atr_pct: float,
                       cost_price=None, cost_profit_pct=None,
                       signal_action=None, risk_tags="",
                       rsi=None, macd_status="", vol_ratio=None) -> str:
        extra = []
        if cost_price and cost_price > 0:
            profit_str = f"{cost_profit_pct*100:+.2f}%" if cost_profit_pct is not None else "未知"
            extra.append(f"持仓成本：{cost_price:.3f}，浮动盈亏：{profit_str}")
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
            content = self._extract_content(resp)
            return content if content else "（AI评论生成失败）"
        except Exception as e:
            logger.error(f"AI评论生成失败: {e}")
            return "（AI评论生成失败）"

    def get_batch_recommendations(self, advice_map: Dict[str, str], market_state: str) -> Dict[str, str]:
        """
        接收原始建议字典 {code: advice_text}，返回润色后的建议字典。
        """
        if not AI_ENABLE:
            return advice_map

        items = []
        for code, advice in advice_map.items():
            items.append({
                "code": code,
                "advice": advice,
            })
        prompt = f"""你是ETF投资顾问。以下列表中是系统根据技术事件生成的原始操作建议，请为每个ETF输出一条最终建议，可以适当润色或补充理由（例如补充“RSI超买”等），但不要改变操作方向和比例。
市场状态：{market_state}
ETF列表：
{json.dumps(items, ensure_ascii=False, indent=2)[:3000]}
请输出JSON格式，键为代码，值为最终建议字符串。不要输出其他文字。"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2,
                timeout=15,
            )
            content = self._extract_content(resp)
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            result = json.loads(content)
            if isinstance(result, dict):
                return result
            else:
                return advice_map
        except Exception as e:
            logger.warning(f"AI润色失败，使用原始建议: {e}")
            return advice_map
        

		

