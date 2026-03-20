# ai.py
# AI动态权重生成

import json
import re
import openai

def deepseek_generate_weights(macro_status, sentiment_factor, market_above_ma20, market_above_ma60, market_amount_above_ma20, api_key):
    """调用 DeepSeek API 生成动态权重，返回权重字典"""
    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    prompt = f"""
    你是一个量化交易策略专家。当前市场环境如下：
    - 宏观状态：{macro_status}（牛市、震荡市、熊市之一）
    - 情绪系数：{sentiment_factor}（0.6-1.2之间，越低越恐慌）
    - 大盘站上20日均线：{"是" if market_above_ma20 else "否"}
    - 大盘站上60日均线：{"是" if market_above_ma60 else "否"}
    - 市场成交额高于20日均额：{"是" if market_amount_above_ma20 else "否"}

    请根据以上市场环境，为以下ETF技术指标分配权重（所有权重之和应为1），以优化买入/卖出决策。输出格式必须为严格的JSON，不要包含其他文本。

    指标列表：
    - price_above_ma20 (价格站上20日线)
    - volume_above_ma5 (成交量高于5日均量)
    - macd_golden_cross (MACD金叉)
    - kdj_golden_cross (KDJ金叉)
    - bollinger_break_up (突破布林带上轨)
    - williams_oversold (威廉超卖)
    - price_below_ma20 (价格跌破20日线)
    - bollinger_break_down (跌破布林带下轨)
    - williams_overbought (威廉超买)
    - rsi_overbought (RSI超买)
    - market_above_ma20 (大盘站上20日线)
    - market_above_ma60 (大盘站上60日线)
    - market_amount_above_ma20 (市场成交额高于20日均额)
    - outperform_market (近5日跑赢大盘)
    - underperform_market (近5日跑输大盘)
    - stop_loss_ma_break (跌破20日均线)
    - trailing_stop_clear (从高点回撤超过8%)
    - trailing_stop_half (从高点回撤超过5%)
    - profit_target_hit (达到涨幅目标)

    请输出JSON，键为指标名称，值为权重（浮点数），确保总和为1。
    """

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.5
        )
        content = response.choices[0].message.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            weights = json.loads(json_match.group())
            total = sum(weights.values())
            if abs(total - 1.0) < 0.01:
                return weights
            else:
                print(f"权重总和为{total}，不等于1，使用默认权重")
                return None
        else:
            print("无法解析JSON，使用默认权重")
            return None
    except Exception as e:
        print(f"DeepSeek API调用失败: {e}，使用默认权重")
        return None