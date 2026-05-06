#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI 服务模块：仅用于生成 ETF 评论，不参与任何策略计算"""
import json
import re
import logging
import openai
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


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
                       tmsv: float, atr_pct: float) -> str:
        prompt = f"""你是一名资深ETF量化分析师，请根据以下数据给出80~120字专业点评。
ETF：{name}（{code}），综合评分：{final_score:.1f}，等级：{action_level}
市场状态：{market_state}，环境因子：{market_factor:.2f}
TMSV复合强度：{tmsv:.1f}，ATR波动率：{atr_pct*100:.2f}%
请从技术面、市场适配性和风险角度点评，不重复数据。"""
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