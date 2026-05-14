#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI 功能集中模块：单只点评、仓位建议、买入力度、趋势扫描、历史分析"""
import time
import logging
import json
import numpy as np
import openai
from typing import List, Dict, Optional, Any
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

    # ========== 单只ETF点评 ==========
    def comment_on_etf(self, code: str, name: str, final_score: float,
                       action_level: str, market_state: str, market_factor: float,
                       tmsv: float, atr_pct: float,
                       cost_price: Optional[float] = None,
                       cost_profit_pct: Optional[float] = None,
                       signal_action: Optional[str] = None,
                       risk_tags: str = "",
                       rsi: Optional[float] = None,
                       macd_status: str = "",
                       vol_ratio: Optional[float] = None) -> str:
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

    # ========== 仓位管理建议（单个ETF） ==========
    def get_position_advice(self, ctx, final_score: float, risk_str: str) -> str:
        if ctx.shares <= 0 or ctx.cost_price is None or ctx.real_price is None:
            return ""
        profit_pct = (ctx.real_price - ctx.cost_price) / ctx.cost_price * 100
        d = ctx.hist_df.iloc[-1] if ctx.hist_df is not None else None
        vol_ratio = (d["volume"] / d["vol_ma"]) if d is not None and d.get("vol_ma", 0) > 0 else 1.0
        macd_status = "金叉" if d is not None and d.get("macd_dif", 0) > d.get("macd_dea", 0) else "死叉"
        level = ""
        for th, lvl in zip([80,70,60,40,20,0,-20,-40,-60,-999],
                           ["极强","强势","偏强","中性偏强","中性","中性偏弱","偏弱","弱势","极弱","极弱"]):
            if final_score >= th:
                level = lvl
                break
        prompt = f"""你是仓位管理专家。根据以下数据输出**具体操作建议**（20字内，必须含动作和比例或价格）：
ETF：{ctx.name}
持仓：{ctx.shares}份，成本{ctx.cost_price:.3f}，现价{ctx.real_price:.3f}，盈亏{profit_pct:.1f}%
综合评分：{final_score:.1f}（{level}）
技术信号：RSI {ctx.rsi}，量比{vol_ratio:.2f}，MACD{macd_status}
风险标签：{risk_str}
输出格式示例："加仓20%"或"减仓30%止盈"或"止损卖出"或"持有不动"。”"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=40,
                temperature=0.2,
                timeout=5,
            )
            return self._extract_content(resp)
        except Exception as e:
            logger.warning(f"AI仓位建议失败: {e}")
            return ""

    # ========== 买入力度建议（仅未持仓） ==========
    def get_buy_level(self, scan_info: dict) -> str:
        score = scan_info.get("final_score", 0)
        rsi = scan_info.get("rsi", 50)
        vol_ratio = scan_info.get("vol_ratio", 1.0)
        above_ma = not scan_info.get("has_weak_ma_text", False)
        change = scan_info.get("change_pct", 0)
        has_clear_stop = scan_info.get("has_clear_stop_text", False)
        has_strong_sell = scan_info.get("has_strong_sell_text", False)
        if has_clear_stop or has_strong_sell:
            return ""
        prompt = f"""你是ETF买入力度评估专家。根据以下数据，输出**唯一一个**买入等级，不要多余解释：
等级选项：🔥 大量买入、📈 适量买入、💡 少量买入
若无买入价值，输出空。
数据：
评分：{score:.0f}
RSI：{rsi:.0f}
成交量比：{vol_ratio:.2f}
站上均线：{"是" if above_ma else "否"}
涨跌幅：{change*100:+.2f}%
输出格式示例：🔥 大量买入"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.2,
                timeout=5,
            )
            advice = self._extract_content(resp)
            if advice in ["🔥 大量买入", "📈 适量买入", "💡 少量买入"]:
                return advice
            return ""
        except Exception as e:
            logger.warning(f"AI买入力度建议失败: {e}")
            return ""

    # ========== 趋势扫描综合推荐（增强版，输出买入/卖出程度） ==========
    def get_trend_recommendations(self, buy_candidates: List[Dict], sell_candidates: List[Dict],
                                  market_state: str) -> Dict[str, List]:
        """
        输入：buy_candidates（经过硬指标筛选的买入候选，已包含type字段），
              sell_candidates（经过硬指标筛选的卖出候选，仅含持仓ETF）
        输出：{"buy": [{"code":..., "direction":..., "advice_text":...}],
              "sell": [{"code":..., "advice_text":...}]}
        其中 advice_text 只包含买入/卖出程度，不涉及具体仓位百分比。
        """
        if not AI_ENABLE:
            return {"buy": [], "sell": []}
        now = time.time()
        if now - self._trend_cache["timestamp"] < AI_CACHE_TTL:
            return self._trend_cache["result"]
        
        # 辅助函数：安全转换为JSON可序列化类型
        def make_json_safe(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj

        buy_json = []
        for e in buy_candidates:
            buy_json.append({
                "code": e["code"],
                "name": e["name"],
                "score": make_json_safe(e["final_score"]),
                "rsi": make_json_safe(e["rsi"]),
                "vol_ratio": make_json_safe(e.get("vol_ratio", 1.0)),
                "change": make_json_safe(e["change_pct"]),
                "above_ma": make_json_safe(e["above_ma"]),
                "low_rise": make_json_safe(e.get("profit_pct_from_low", 0)),
                "type": e.get("type", "right")
            })
        sell_json = []
        for e in sell_candidates:
            sell_json.append({
                "code": e["code"],
                "name": e["name"],
                "score": make_json_safe(e["final_score"]),
                "rsi": make_json_safe(e["rsi"]),
                "change": make_json_safe(e["change_pct"]),
                "weak_ma": make_json_safe(e.get("weak_ma", False))
            })
        
        prompt = f"""你是量化交易策略师。以下候选列表已经过硬指标筛选，请从中选择并输出最终推荐。

市场状态：{market_state}

买入候选（已按右侧/左侧分类，需从每类中选**至少1个**，总数不超过4个）：
{json.dumps(buy_json, ensure_ascii=False)[:3000]}

卖出候选（已按风险指标筛选，最多选3个）：
{json.dumps(sell_json, ensure_ascii=False)[:2000]}

要求：
1. 对于买入，必须保留原候选中的 `type` 字段，并在 `direction` 中使用 `right_buy` 或 `left_buy`。
2. 对于卖出，直接输出 `sell`。
3. 同一只ETF不得同时出现在买入和卖出。
4. **输出格式要求**：
   - 买入建议的 `advice_text` 只包含买入力度等级，格式为："🔥 大量买入" 或 "📈 适量买入" 或 "💡 少量买入"，不要包含百分比仓位。
   - 卖出建议的 `advice_text` 包含卖出程度，格式为："❗ 强烈卖出" 或 "⚠️ 减仓" 或 "🔻 止盈" 等，不要包含百分比仓位。
5. 输出JSON格式，不要有其他文字。

输出示例：
{{
"buy": [
    {{"code": "sz.159611", "direction": "right_buy", "advice_text": "🔥 大量买入"}},
    {{"code": "sh.513210", "direction": "left_buy", "advice_text": "📈 适量买入"}}
],
"sell": [
    {{"code": "sh.512800", "advice_text": "❗ 强烈卖出"}}
]
}}"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.1,
                timeout=15,
            )
            content = self._extract_content(resp)
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            result = json.loads(content)
            if "buy" not in result:
                result["buy"] = []
            if "sell" not in result:
                result["sell"] = []
            self._trend_cache = {"result": result, "timestamp": now}
            return result
        except Exception as e:
            logger.error(f"AI趋势扫描失败: {e}")
            return {"buy": [], "sell": []}
    


    # ========== 趋势扫描综合推荐 持仓和推荐同时判断 ==========
    def get_batch_recommendations(self, etf_dict: Dict[str, Dict], market_state: str) -> Dict[str, str]:
        """
        批量生成ETF操作建议
        输入：etf_dict = {code: {"name":..., "final_score":..., "rsi":..., "vol_ratio":..., 
                                "change_pct":..., "above_ma":..., "profit_pct_from_low":...,
                                "shares":..., "cost_price":..., "risk_str":..., "price":...,
                                "change_pct_display":...}}
        输出：{code: advice_text}
        """
        if not AI_ENABLE:
            return {}
        
        # 构建请求数据
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
                "change_pct_display": info["change_pct_display"]
            })
        
        prompt = f"""你是ETF投资顾问。以下列表包含当前**持仓**或**值得关注**的ETF。请为每个ETF给出**唯一一个**操作建议，要求：

    **规则**：
    - 对于**未持仓**（shares=0）的ETF，建议从以下选择：  
    "🔥 大量买入"、"📈 适量买入"、"💡 少量买入"、"观望"  
    根据评分、RSI、量比、是否站上均线、低点涨幅综合判断。
    - 对于**已持仓**（shares>0）的ETF，建议从以下选择：  
    "加仓20%"、"减仓30%"、"持有不动"、"止损卖出"、"清仓止盈"  
    结合持仓盈亏（可算出）、风险标签、评分高低给出合理建议。
    - 如果风险标签中包含"止损"、"止盈"、"连续低分"等，应优先给出相应建议。

    市场状态：{market_state}

    ETF列表：
    {json.dumps(items, ensure_ascii=False, indent=2)[:4000]}

    输出格式：JSON对象，键为代码，值为建议文本。

    示例：
    {{
        "sz.159611": "🔥 大量买入",
        "sh.515790": "持有不动",
        "sh.512800": "止损卖出",
        "sz.159583": "📈 适量买入"
    }}
    """
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.2,
                timeout=20,
            )
            content = self._extract_content(resp)
            if content.startswith("```json"):
                content = content[7:]
            if content.endswith("```"):
                content = content[:-3]
            result = json.loads(content)
            # 确保返回字典
            if isinstance(result, dict):
                return result
            else:
                logger.warning(f"批量建议返回格式错误: {result}")
                return {}
        except Exception as e:
            logger.error(f"批量建议生成失败: {e}")
            return {}

    
    
    
    # ========== 历史持仓分析 ==========
    def analyze_position_history(self, code: str, name: str, history: List[Dict]) -> str:
        if not AI_ENABLE or not history:
            return ""
        prompt = f"""你是量化策略分析师。根据以下ETF的历史持仓变化（每日份额、成本），分析中长线逻辑是否合理，并给出优化建议。
ETF：{name}（{code}）
历史记录（日期、份额、成本）：
{str(history)[:2000]}
输出要求：50字内，指出调仓是否合理，建议改进方向。"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.2,
                timeout=8,
            )
            return self._extract_content(resp)
        except Exception as e:
            logger.warning(f"AI历史分析失败: {e}")
            return ""