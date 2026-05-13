# ETF 智能分析系统

一个面向中线波段（持有期 14~20 天）的 ETF 智能分析工具。  
系统实时获取行情数据，计算技术指标，融合 AI 动态权重，输出综合评分与操作等级（强烈买入 / 买入 / 偏多持有 / 中性 / 卖出 / 强烈卖出）。

> 即使不配置 AI 密钥，系统也能使用默认权重和简单市场规则正常运行。

---

## 核心特性

- **中线策略主导**：强制中线因子（价格站上均线、MACD、布林带、TMSV 等）权重 ≥65%，短线因子 ≤25%。
- **AI 动态权重**：调用 DeepSeek API，结合市场状态、情绪、波动率及实时因子强度，生成买卖权重。
- **历史趋势分析**：保存最近 7 个交易日评分，计算趋势斜率，动态调整买入/卖出阈值及确认天数。
- **情绪过热保护**：当情绪因子 ≥1.25 时，自动提高卖出超买因子权重。
- **高波动适应**：根据 ATR 波动率自动调整确认天数与快速买入阈值。
- **缓存机制**：AI 权重按市场状态缓存 7 天，减少 API 调用。
- **可选邮件报告**：支持 SMTP 发送每日分析报告。

---

## 项目结构

```bash
etf_project/
├── etf_analyzer/               # 主包
│   ├── __init__.py
│   ├── config.py               # 配置常量、默认权重、文件路径
│   ├── utils.py                # 通用工具（显示宽度、离散化、权重验证、邮件发送）
│   ├── ai.py                   # AI 客户端（DeepSeek）
│   ├── analyzer.py             # 核心分析（数据获取、指标计算、评分、信号确认）
│   └── main.py                 # 命令行入口
├── data/                       # 数据目录（需手动创建）
│   ├── positions.csv           # ETF 持仓列表（必须）
│   ├── etf_state.json          # 状态文件（自动生成）
│   └── weight_cache.json       # AI 权重缓存（自动生成）
└── README.md
```

---

## 安装与配置

### 1. 环境要求

- Python 3.8+
- 依赖库：`baostock`, `requests`, `numpy`, `pandas`, `openai`
- 可选：`akshare`（提供更丰富的情绪指标）

### 2. 安装依赖

```bash
pip install baostock requests numpy pandas openai
# 如需情绪指标
pip install akshare
```

### 3. 准备持仓列表

在 `data/` 目录下创建 `positions.csv`，至少包含两列：

```csv
代码,名称
sh.510300,沪深300ETF
sh.510500,中证500ETF
sz.159206,商业航天
```

### 4. 配置 AI 密钥（可选）

```bash
# Linux / macOS
export DEEPSEEK_API_KEY="your_api_key"

# Windows PowerShell
$env:DEEPSEEK_API_KEY="your_api_key"
```

不配置时系统使用默认权重，市场状态通过简单规则判断（大盘站上均线等）。

### 5. 配置邮件通知（可选）

```bash
export SMTP_SERVER="smtp.qq.com"
export SMTP_PORT="587"
export SENDER_EMAIL="your_email@qq.com"
export SENDER_PASSWORD="your_password"
export RECEIVER_EMAIL="receiver@example.com"
export SEND_EMAIL="true"
```

---

## 使用方法

进入项目根目录 `etf_project` 后：

### 批量分析所有 ETF

```bash
python -m etf_analyzer.main
```

### 详细分析单只 ETF

```bash
python -m etf_analyzer.main --code sh.510300
```

### 使用启动脚本（可选）

创建 `run.py` 并写入：

```python
from etf_analyzer.main import main
if __name__ == "__main__":
    main()
```

然后执行：

```bash
python run.py
```

---

## 输出示例

```
2026-04-20 17:05:20 - INFO - AI市场状态: 正常牛市, 因子: 1.15
2026-04-20 17:05:20 - INFO - 综合情绪因子: 1.300

名称             代码           价格   评分  操作
------------------------------------------------------------
商业航天        sz.159206    1.883   0.37  偏多持有

ETF详细分析报告 - 商业航天 (sz.159206)
======================================================================
实时价格：1.883
市场状态：正常牛市，市场因子：1.15，情绪因子：1.30
情绪风险提示：💸 市场情绪过热，短期回调风险较高
波动率(ATR%)：3.06%
TMSV复合强度：20.3 (强度系数 0.203)
最大回撤：-3.86%

【买入因子详情】...
【卖出因子详情】...
【评分合成】...
【AI 专业点评】...
```

---

## 算法说明

### 技术指标

- 均线系统（MA20 / MA60 / 周线 MA20）
- MACD、KDJ、布林带、威廉指标、RSI
- ATR 波动率、ADX 趋势强度
- 自定义下跌动量因子
- **TMSV 复合指标**：融合趋势、动量、量价，输出 0–100 强度分

### 评分公式

- 买入分 / 卖出分 = Σ(因子强度 × 因子权重)
- 原始净分 = 买入分 - 卖出分
- 最终评分 = 原始净分 × 市场因子 × 情绪因子

### 动态参数

- `CONFIRM_DAYS`：连续信号确认天数（波动率高时延长）
- `BUY_THRESHOLD` / `SELL_THRESHOLD`：根据历史评分趋势动态调整
- `QUICK_BUY_THRESHOLD`：强趋势时允许快速买入

### AI 权重生成流程

1. 收集市场状态、情绪因子、波动率及实时因子强度（TMSV、量比、MACD 状态等）
2. 调用 DeepSeek API 生成原始买卖权重 JSON
3. 软化处理：无零权重，最低 0.02
4. 计算动态信任度（基于零权重数、中线因子缺失、TMSV 强度）
5. 混合权重：`AI权重×信任度 + 默认权重×(1−信任度)`
6. 强制策略：中线总权重 ≥0.65，短线总权重 ≤0.25，情绪过热时提高卖出超买因子
7. 归一化后存入缓存（有效期 7 天）

### 强制策略（`enforce_weight_strategy`）

- 确保每个因子 ≥0.02
- 中线因子总权重 ≥0.65，短线 ≤0.25
- 情绪因子 ≥1.25 时，`williams_overbought` 和 `rsi_overbought` 强制升至 ≥0.10
- 重新归一化

---

## 状态与缓存

- `etf_state.json`：保存每只 ETF 最近 7 个交易日的评分历史，用于趋势计算与信号确认。
- `weight_cache.json`：缓存 AI 生成的权重，按市场状态、情绪、波动率等特征分组，有效期 7 天。

---

## 注意事项

- 首次运行需要网络畅通，`baostock` 会自动登录。
- 若未安装 `akshare`，情绪因子将使用基于 RSI 的简化版本。
- 若不配置 AI Key，系统仍可正常运行，仅使用默认权重和简单市场规则。
- 确保 `data/` 目录有写入权限，否则状态与缓存无法保存。
- 本系统不构成投资建议，仅供参考。

---

## 扩展建议

- 添加 `requirements.txt` 管理依赖。
- 增加更多技术指标（如 OBV、CCI）作为备选因子。
- 支持多周期共振分析（日线 + 周线 + 月线）。
- 对接模拟交易或实盘接口。

---

**版本**：2.0（中线策略优化版）  
**最后更新**：2026-04-20
