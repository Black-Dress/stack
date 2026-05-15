#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI 功能集中模块：单只点评、批量建议（无后备）"""
import time
import logging
import json
import numpy as np
import openai
from typing import Dict, Any
from .config import AI_CACHE_TTL, AI_ENABLE

logger = logging.getLogger(__name__)


class AIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self._trend_cache = {"result": None, "timestamp": 0}

    @staticmethod
    def _extract_content(resp: Any) -> str:
        content = getattr(resp.choices[0].message, "content", None)
        return content.strip() if isinstance(content, str) and content.strip() else ""

    # 单只ETF点评（保留不变）
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
            content = self._extract_content(resp)
            return content if content else "（AI评论生成失败）"
        except Exception as e:
            logger.error(f"AI评论生成失败: {e}")
            return "（AI评论生成失败）"






    def get_batch_recommendations(self, etf_dict: Dict[str, Dict], market_state: str) -> Dict[str, str]:
        """
        返回 {code: 建议文本}
        输出格式要求：
          - 若系统建议(sys_advice)包含“已加仓”或“已减仓”，输出格式为：“已操作文本 | 后续操作建议”
            其中后续操作建议必须具体到价位和比例，如“在 1.234 加仓 20%”或“在 1.184 减仓 30%”
          - 若系统建议不包含操作记录（未持仓或已持仓无操作）：
              - 未持仓：输出“买入 | 程度：强烈推荐/推荐/谨慎”
              - 已持仓无操作：输出“在 X 加仓 Y%”或“在 X 减仓 Y%”（X为具体价位，Y为比例）
        注意：对于高评分（>70）强烈建议加仓，低评分（<40）建议减仓，严禁高评分给出减仓建议。
        """
        if not AI_ENABLE:
            raise RuntimeError("AI未启用但调用了get_batch_recommendations")

        items = []
        for code, info in etf_dict.items():
            items.append({
                "code": code,
                "name": info["name"],
                "score": info["final_score"],
                "rsi": info["rsi"],
                "vol_ratio": info["vol_ratio"],
                "change": info["change_pct"],
                "above_ma": info["above_ma"],
                "low_rise": info["profit_pct_from_low"],
                "shares": info["shares"],
                "cost_price": info.get("cost_price"),
                "risk_str": info.get("risk_str", ""),
                "price": info["price"],
                "atr_pct": info.get("atr_pct", 0.02),
                "sys_advice": info.get("sys_advice", ""),
            })

        prompt = f"""你是ETF交易顾问。根据以下每个ETF的数据，为每个ETF生成一条操作建议。

输出格式严格要求（非常重要）：
1. **如果系统建议(sys_advice)中包含“已加仓”或“已减仓”**（表示今日已有操作），则输出格式为：
   "已加仓X股 | 在 Y 加仓 Z%" 或 "已减仓X股 | 在 Y 减仓 Z%"
   其中 Y 必须是具体的价位（根据当前价格和 ATR 合理推算），Z 是百分比（整数，不超过50）。
   示例： "已加仓1300股 | 在 1.120 加仓 15%"
         "已减仓1400股 | 在 1.170 减仓 20%"
   注意：竖线前后各有一个空格，竖线前面是已操作文本，竖线后面是后续操作建议（必须包含具体价位和百分比）。

2. **如果系统建议中不包含“已加仓”或“已减仓”**：
   - 当 shares == 0（未持仓）：输出 "买入 | 程度：强烈推荐" 或 "买入 | 程度：推荐" 或 "买入 | 程度：谨慎"。
   - 当 shares > 0（已持仓且无今日操作）：输出 "在 Y 加仓 Z%" 或 "在 Y 减仓 Z%"（Y 为具体价位，Z 为百分比）。
     例如： "在 1.120 加仓 15%"

3. 输出必须是严格的JSON对象，键为代码，值为建议字符串。不要有任何额外文字。

市场状态：{market_state}

ETF列表：
{json.dumps(items, ensure_ascii=False, indent=2)[:4000]}

请输出JSON："""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1500,
                temperature=0.2,
                timeout=20,
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
            if not isinstance(result, dict):
                raise ValueError("AI返回的不是字典")
            return result
        except Exception as e:
            logger.error(f"批量建议生成失败: {e}")
            raise RuntimeError(f"AI调用失败: {e}")

