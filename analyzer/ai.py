#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""AI 功能集中模块：单只点评、仓位建议、买入力度、趋势扫描、历史分析"""
import time
import logging
import openai
from typing import List, Dict, Optional, Any
from .config import AI_CACHE_TTL, AI_ENABLE

logger = logging.getLogger(__name__)

class AIClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        # 趋势扫描缓存
        self._trend_cache = {"result": None, "timestamp": 0}

    
    
    # ========== 单只ETF点评（保留用于详细报告） ==========
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
            content = resp.choices[0].message.content
            return content.strip() if content else "（AI评论生成失败）"
        except Exception as e:
            logger.error(f"AI评论生成失败: {e}")
            return "（AI评论生成失败）"

    
    
    # ========== 仓位管理建议（单个ETF） ==========
    def get_position_advice(self, ctx, final_score: float, risk_str: str) -> str:
        """根据持仓、技术指标生成仓位操作建议（加仓/减仓/持有/止盈/止损）"""
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
输出格式示例："加仓20%"或"减仓30%止盈"或"止损卖出"或"持有不动"。”
"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=40,
                temperature=0.2,
                timeout=5,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"AI仓位建议失败: {e}")
            return ""

    
    
    # ========== 买入力度建议（仅未持仓） ==========
    def get_buy_level(self, scan_info: dict) -> str:
        """根据技术指标、评分生成买入力度建议（大量/适量/少量）"""
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
            advice = resp.choices[0].message.content.strip()
            if advice in ["🔥 大量买入", "📈 适量买入", "💡 少量买入"]:
                return advice
            return ""
        except Exception as e:
            logger.warning(f"AI买入力度建议失败: {e}")
            return ""

    
    
    # ========== 趋势扫描综合推荐（带缓存） ==========
    def get_trend_recommendations(self, buy_candidates: List[Dict], sell_candidates: List[Dict],
                                market_state: str) -> Dict[str, List]:
        """
        输入：buy_candidates（未持仓且不弱于均线的ETF数据列表），sell_candidates（所有ETF列表）
        输出：{"buy": [{"code":..., "direction":..., "advice_text":...}],
            "sell": [{"code":..., "advice_text":...}]}
        """
        if not AI_ENABLE:
            return {"buy": [], "sell": []}
        # 检查缓存
        now = time.time()
        if now - self._trend_cache["timestamp"] < AI_CACHE_TTL:
            return self._trend_cache["result"]
        
        # 构建买入候选JSON（精简）
        buy_json = []
        for idx, etf in enumerate(buy_candidates):
            buy_json.append({
                "code": etf["code"],
                "name": etf["name"],
                "score": etf["final_score"],
                "rsi": etf["rsi"],
                "vol_ratio": etf.get("vol_ratio", 1.0),
                "change_pct": etf["change_pct"],
                "above_ma": etf["above_ma"],
                "low_rise": etf.get("profit_pct_from_low", 0),
                "pullback": etf.get("max_drawdown_pct", 0)
            })
        sell_json = []
        for etf in sell_candidates:
            sell_json.append({
                "code": etf["code"],
                "name": etf["name"],
                "score": etf["final_score"],
                "rsi": etf["rsi"],
                "change_pct": etf["change_pct"],
                "weak_ma": etf.get("has_weak_ma_text", False),
                "clear_stop": etf.get("has_clear_stop_text", False),
                "strong_sell": etf.get("has_strong_sell_text", False)
            })
        
        prompt = f"""你是量化交易策略师。根据以下所有ETF数据和市场状态，输出趋势扫描推荐。

    市场状态：{market_state}

    买入候选列表（仅未持仓且站上均线的ETF，用于推荐买入，最多4个，需区分左侧买入和右侧买入）：
    {str(buy_json)[:3000]}

    卖出候选列表（所有ETF，用于推荐卖出，最多3个）：
    {str(sell_json)[:2000]}

    任务：
    1. 推荐最多4个买入标的。每类（左侧/右侧）最多2个，若某类无合适标的可少给。
    - 右侧买入：价格站上MA20，近5日涨幅>0.5%，RSI>50，放量突破。
    - 左侧买入：价格低于MA20（但不得弱于均线），RSI<40，低点涨幅<8%，未形成明显上升趋势。
    - **重要：不要推荐任何 weak_ma 为 true 的ETF。买入候选列表已经排除了weak_ma的ETF，你只需从中选择。**
    2. 推荐最多3个卖出标的（趋势转弱、破位、连续低分等）。
    3. 每条输出必须包含ETF代码，方向（left_buy/right_buy/sell），以及20字内的建议文本（可含emoji，但不要使用⚠️符号）。
    输出格式（纯JSON）：
    {{
    "buy": [
        {{"code": "sz.159611", "direction": "right_buy", "advice_text": "🔥 右侧买入，放量突破"}},
        {{"code": "sh.513210", "direction": "left_buy", "advice_text": "📉 左侧低吸，RSI超卖"}}
    ],
    "sell": [
        {{"code": "sh.512800", "advice_text": "❗ 卖出，连续低分"}}
    ]
    }}
    注意：只输出JSON，不要其他文字。"""
        try:
            resp = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.2,
                timeout=15,
            )
            content = resp.choices[0].message.content
            import json
            result = json.loads(content)
            self._trend_cache = {"result": result, "timestamp": now}
            return result
        except Exception as e:
            logger.error(f"AI趋势扫描失败: {e}")
            return {"buy": [], "sell": []}   
        
    
    
    # ========== 历史持仓分析（供后续扩展） ==========
    def analyze_position_history(self, code: str, name: str, history: List[Dict]) -> str:
        """分析单个ETF的历史持仓变化，输出中长线评价及优化建议"""
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
            return resp.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"AI历史分析失败: {e}")
            return ""