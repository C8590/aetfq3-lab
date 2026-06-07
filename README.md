# 日频右侧确认型 ETF 动量轮动策略

本仓库是 `aetfq3-lab / Lab`，属于 AETF Lab / 实验室 / 试车场，不属于 V2.1 Stable；Lab 输出只能作为只读研究建议，不直接触发正式交易。

Lab 规则入口：`docs/lab/aetfq3_lab_charter.md`、`docs/lab/aetfq3_lab_output_contract.md`、`docs/lab/aetfq3_lab_advisory_package_spec.md`、`docs/lab/aetfq3_lab_file_naming.md`、`docs/lab/aetfq3_lab_advisory_examples.md`、`docs/lab/aetfq3_lab_research_review_checklist.md`。

第三仓库 protocol reference 已验证为 `aetfq3-protocol v0.1.0-rc1`，用于通信合同 / schema / bundle 校验 / 升级门禁，不授权 Lab 扩权，也不授权 Lab 直接修改 Stable。

```text
Remote:
https://github.com/C8590/aetfq3-protocol

protocol_reference:
verified

protocol_repo:
C8590/aetfq3-protocol

Tag:
v0.1.0-rc1

Protocol commit:
9e15a78c43ec874441429ef14edad34b36ab83bf

Closeout ledger commit:
6c72df96a79aa66c2780692c64af7661da07213e
```

本项目当前只保留一个主策略：日频右侧确认型 ETF 动量轮动策略。

系统定位是人工观察和复盘工具，不是自动交易系统。它不预测涨跌，不推荐新闻，不保证收益，不自动下单，也不直接连接券商。策略只在 ETF 已经表现出相对强势、趋势形态和成交活跃度后，生成买入、持有、减仓、卖出或观察建议。

## 当前主链路

1. 数据更新与质量检查。
2. ETF 行业、主题和风险分组映射。
3. 日频动量排名和右侧确认。
4. `risk_warning` P0 上层风险刹车。
5. `trade_policy` 交易安全和持仓控制。
6. 前端展示当前信号、持仓、数据质量和历史对照。
7. learning 记录风险、买点、退出和复盘上下文。

优先级明确为：

```text
risk_warning > market_state > sector_rank > etf_rank > entry_signal
```

## 核心边界

- `risk_warning` 是 entry 前最高优先级风险门控，只负责降速、冻结或进入人工复核。
- `risk_warning` 不选 ETF，不预测涨跌，不把新闻变成买入信号。
- R3/R4 只冻结新开仓和加仓，不阻断卖出、减仓、止损和退出。
- `trade_policy` 只生成交易安全建议，不操作 QMT，不调用真实券商接口。
- 前端当前主入口只有日频动量策略；旧版结果仅作为历史对照 / 旧版参考。
- Lab 只负责研究、模型诊断、实验报告和只读 advisory。
- Lab advisory 必须是 `READ_ONLY`，且 `final_action_change_allowed: false`、`contains_live_order: false`、`contains_secret: false`。
- Lab 不直接生成 Stable 正式 `OrderIntent`，不绕过 Stable `RiskGate`，不直接修改 Stable。
- Lab 成果进入 Stable 必须走人工 promotion gate。
- 包含 secret、live order 或 `final_action_change_allowed=true` 的 bundle 一律禁止。

## 常用命令

Lab 表格 ML / PyTorch / baseline smoke 命令必须使用当前仓库 `.venv` Python：

```powershell
.\.venv\Scripts\python.exe
```

或先激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

不推荐裸 `python`，因为它可能指向系统 Python。若使用裸 `python`，必须先执行 `python -c "import sys; print(sys.executable)"`，确认输出为 `E:\aetfq3-lab\.venv\Scripts\python.exe`。

更新 ETF 日线数据：

```powershell
.\.venv\Scripts\python.exe main.py update-data
```

生成当前信号：

```powershell
.\.venv\Scripts\python.exe main.py generate-signal --use-cache
```

生成指定信号日的复盘信号：

```powershell
.\.venv\Scripts\python.exe main.py generate-signal --signal-date 2026-05-13 --use-cache
```

录入或查看风险事件：

```powershell
.\.venv\Scripts\python.exe main.py risk add
.\.venv\Scripts\python.exe main.py risk list
.\.venv\Scripts\python.exe main.py risk score
```

启动前端页面：

```powershell
.\.venv\Scripts\streamlit.exe run app.py
```

## 主要输出

- `output/compare_signal.csv`
- `output/compare_signal.txt`
- `output/risk_gate.json`
- `output/risk_warning_next_day.csv`
- `output/risk_learning_context.csv`

这些都是运行期输出，不应提交到 Git。

## 配置和本地状态

- `config/strategy.yaml`：唯一主策略配置。
- `config/risk_warning.yaml`：风险预警评分配置。
- `config/risk_events.example.yaml`：风险事件录入示例。
- `config/etf_universe.yaml`：ETF 池配置。
- `config/etf_sector_map.yaml`：ETF 名称、行业、主题、风险分组映射。
- `config/current_position.yaml`：本地模拟持仓状态，需人工同步。
- `config/live_observation.yaml`：本地观察资金和使用模式。

`data/risk_warning/`、`data/quote_cache/`、`data/universe/`、`output/`、`runtime/` 等目录属于运行期数据或缓存，不应提交到 Git。

## 安全说明

本项目仅用于人工观察、复盘和研究验证，不构成投资建议。任何真实交易都必须由人工确认数据、价格、份额、现金余额、交易单位、风险事件和执行窗口。
