#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ETF 智能分析系统主入口
用法：
    python main.py                # 批量分析所有 ETF
    python main.py --code sh.512800  # 详细分析单只 ETF
"""
import argparse
import logging
import os
from .analyzer import run_batch_analysis

# 配置日志
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

for lib in ["openai", "httpx", "httpcore", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

def main():
    parser = argparse.ArgumentParser(description="ETF智能分析系统")
    parser.add_argument(
        "--code",
        type=str,
        help="指定分析某个ETF代码（例如 sh.512800），不指定则批量分析所有",
    )
    args = parser.parse_args()

    api_key = os.getenv("DEEPSEEK_API_KEY")
    run_batch_analysis(api_key=api_key, target_code=args.code)

if __name__ == "__main__":
    main()
