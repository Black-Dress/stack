# ai.py
# AI动态权重生成（优化版）

import json
import re
import hashlib
import logging
import time
from pathlib import Path
import openai

from config import STRATEGY_WEIGHTS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CACHE_FILE = "weight_cache.json"
DEFAULT_WEIGHTS = STRATEGY_WEIGHTS.copy()
API_TIMEOUT = 10

def _get_cache_key(macro_status, sentiment_factor, market_above_ma20, market_above_ma60, market_amount_above_ma20):
    data = f"{macro_status}_{sentiment_factor}_{market_above_ma20}_{market_above_ma60}_{market_amount_above_ma20}"
    return hashlib.md5(data.encode()).hexdigest()

def _load_cache():
    if Path(CACHE_FILE).exists():
        try:
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def _save_cache(cache):
    with open(CACHE_FILE, 'w', encoding='utf-8') as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

def _validate_weights(weights, expected_keys):
    if not isinstance(weights, dict):
        return False
    missing = set(expected_keys) - set(weights.keys())
    if missing:
        logger.warning(f"权重缺少键: {missing}，将补0")
        for k in missing:
            weights[k] = 0.0
    for k, v in weights.items():
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            logger.warning(f"权重键 {k} 值 {v} 不在[0,1]内，将裁剪")
            weights[k] = max(0.0, min(1.0, v))
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        logger.warning(f"权重总和 {total} 不为1，将归一化")
        for k in weights:
            weights[k] /= total
    return True

def _build_prompt(macro_status, sentiment_factor, market_above_ma20, market_above_ma60, market_amount_above_ma20, indicators):
    indicator_desc = "\n".join([f"- {k}: {v}" for k, v in indicators.items()])
    prompt = f"""
你是一个量化交易策略专家。当前市场环境如下：
- 宏观状态：{macro_status}（牛市、震荡市、熊市之一）
- 情绪系数：{sentiment_factor}（0.6-1.2之间，越低越恐慌）
- 大盘站上20日均线：{"是" if market_above_ma20 else "否"}
- 大盘站上60日均线：{"是" if market_above_ma60 else "否"}
- 市场成交额高于20日均额：{"是" if market_amount_above_ma20 else "否"}

请为以下指标分配权重（所有权重之和应为1），输出严格的JSON对象，键为指标名称，值为权重（浮点数，保留2位小数）。示例输出格式：
{{
    "price_above_ma20": 0.25,
    "volume_above_ma5": 0.20,
    ...
}}
指标列表：
{indicator_desc}
"""
    return prompt

def deepseek_generate_weights(macro_status, sentiment_factor, market_above_ma20, market_above_ma60, market_amount_above_ma20, api_key, model="deepseek-chat", use_cache=True):
    cache_key = _get_cache_key(macro_status, sentiment_factor, market_above_ma20, market_above_ma60, market_amount_above_ma20)

    if use_cache:
        cache = _load_cache()
        if cache_key in cache:
            logger.info("使用缓存权重")
            return cache[cache_key]

    indicator_names = list(STRATEGY_WEIGHTS.keys())
    indicator_desc = {k: "" for k in indicator_names}
    prompt = _build_prompt(macro_status, sentiment_factor, market_above_ma20, market_above_ma60, market_amount_above_ma20, indicator_desc)

    client = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "你是一个量化交易专家，输出严格的JSON格式。不要包含其他解释。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.0,
            timeout=API_TIMEOUT
        )
        elapsed = time.time() - start_time
        content = response.choices[0].message.content
        logger.info(f"API调用成功，耗时 {elapsed:.2f}s")

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            weights = json.loads(json_match.group())
            if _validate_weights(weights, indicator_names):
                if use_cache:
                    cache = _load_cache()
                    cache[cache_key] = weights
                    _save_cache(cache)
                return weights
            else:
                logger.error("权重验证失败")
                return None
        else:
            logger.error("无法解析JSON")
            return None
    except Exception as e:
        logger.error(f"DeepSeek API调用失败: {e}")
        if use_cache:
            cache = _load_cache()
            if cache_key in cache:
                logger.info("使用缓存中的权重（上次成功）")
                return cache[cache_key]
        logger.warning("使用默认权重")
        return DEFAULT_WEIGHTS.copy()