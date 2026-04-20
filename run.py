#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import os

# 将项目根目录添加到 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from etf_analyzer.main import main

if __name__ == "__main__":
    main()