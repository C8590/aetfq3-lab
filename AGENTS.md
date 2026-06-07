# AETF Q3 Lab Codex 项目宪法

## Lab 身份声明

每次给 Codex 派发本仓库任务前，必须先写：

```text
本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
```

`aetfq3-lab` 是 AETF Lab / 实验室 / 试车场。它不是 V2.1 Stable 的附属模块，而是研究实验主线。Lab 可以研究、验证、模拟、诊断和形成 advisory 建议包，但不得直接修改 Stable，不得绕过 Stable 的风控、总控和人工确认边界。

## Lab Advisory Protocol RC1 边界

每个 Codex 任务仍必须以以下声明开头：

```text
本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
```

Lab advisory 只能是 `READ_ONLY`，不得包含 live order，不得包含 secret，不得允许 final action change。默认边界必须保持：

```text
access_mode: READ_ONLY
final_action_change_allowed: false
contains_live_order: false
contains_secret: false
requires_human_review: true
promotion_gate_required: true
```

`aetfq3-protocol` is verified and available as `v0.1.0-rc1`.

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

Protocol 只定义通信合同 / schema / 校验 / promotion gate，不授权 Lab 扩权，不授权生成正式交易计划、正式 `OrderIntent`、绕过 `RiskGate` 或直接修改 Stable。

## Lab 负责范围

Lab 负责以下研究与实验方向：

- `historical_ml`
- sector map
- ML false downgrade
- 强主线保护
- 同板块 ETF 排序
- 第一板块切入位置
- 5分钟K 回测
- Intraday Watch Engine
- 盘口特征
- QMT mock / readonly / 模拟盘
- PyTorch / GRU / TCN 执行模型
- Q3 / Q4 / Q5 前沿策略原型
- Lab advisory 报告

## Lab 禁止事项

- 不直接改 Stable。
- 不修改 Stable entry。
- 不修改 `final_buy_action`。
- 不修改 `target_weight`。
- 不修改 BUY / PROBE 阈值。
- 不生成 Stable 正式 `OrderIntent`。
- 不绕过 `RiskGate`。
- 不自动下单。
- 不把研究输出当正式交易计划。
- 不把模型直接接入 Stable。
- 不提交 Stable 运行产物。
- 不写 Stable `runtime/`。
- 不写 Stable `output/`。
- 不把 QMT 实验回写 Stable。
- 不接正式 QMT；QMT 相关实验只允许 mock、readonly 或模拟盘边界。

## Stable 输出边界

Lab 给 Stable 的输出只能是只读建议包。允许的建议包文件名包括：

- `ml_advisory_summary.json`
- `sector_research_report.json`
- `intraday_watch_research.json`
- `qmt_readonly_report.json`
- `model_diagnostics.json`
- `research_notes.md`

这些建议包不得被视为正式交易计划，不得自动改变 Stable 参数，不得直接触发 `OrderIntent`、QMT 或真实下单。

## Lab 任务结束必答

每个 Lab 任务结束必须回答：

- 研究了什么。
- 数据来自哪里。
- 是否来自 Stable bundle。
- 是否有未来函数。
- 是否影响 Stable 正式交易。
- 是否只读 advisory。
- 是否建议进入 Stable。
- 如果建议进入 Stable，最小合并方案是什么。
- 不允许直接提交到 Stable。

详细任务模板见 `docs/lab/aetfq3_lab_codex_task_template.md`，Lab charter 见 `docs/lab/aetfq3_lab_charter.md`。Lab 输出规范入口见：

- `docs/lab/aetfq3_lab_output_contract.md`
- `docs/lab/aetfq3_lab_advisory_package_spec.md`
- `docs/lab/aetfq3_lab_file_naming.md`
- `docs/lab/aetfq3_lab_advisory_examples.md`
- `docs/lab/aetfq3_lab_advisory_examples.json`
- `docs/lab/aetfq3_lab_research_review_checklist.md`
- `docs/lab/aetfq3_lab_research_review_checklist.json`

## 与旧 AETFv2 规则的关系

## 项目定位

本仓库继承 AETFv2 的可审计、可回滚工程纪律，但当前仓库定位为 `aetfq3-lab / Lab`，不是 V2.1 Stable。以下 AETFv2 项目部规则仅作为研究边界和接口理解参考，不授权 Lab 直接修改 Stable 或生成正式交易动作。

本项目采用“8 项目部 + 总控规则”的 custom agents 架构。任何正式交易判断、执行动作、前端动作快照或 Action API 输出，都必须经过 `aetfv2_08_control_center` 统一汇总与裁决。

## 8 项目部职责边界

### 1. aetfv2_01_pre_selection：候选池 / 预选项目部

负责 ETF 池、样本过滤、数据质量、板块分类、候选池审查与改进建议。不负责买入、卖出、仓位、下单或交易执行。

### 2. aetfv2_02_entry：买入决策项目部

负责买入条件、买入等级、买入解释、建议仓位与 entry 侧证据审查。不得直接下单，不得绕过 `risk_warning`，不得在风险冻结状态下生成普通买入。

### 3. aetfv2_03_exit：卖出 / 退出项目部

负责持仓退出、减仓、止损、清仓建议与退出理由审查。不得直接下单，不得直接操作 QMT，不得越过总控输出正式交易动作。

### 4. aetfv2_04_learning：复盘学习项目部

负责模拟盘复盘、买后 / 卖后表现、失败归因、策略健康度与改进建议。只输出建议，不自动修改正式交易参数。

### 5. aetfv2_05_historical_ml：历史回放与机器学习项目部

负责历史回放、样本生产、自动标签、人工复核队列、entry 校准报告与过拟合检查。不得直接修改 entry 正式交易规则，不得将模型建议自动提升为交易参数。

### 6. aetfv2_06_risk_warning：P0 信息预警 / 风险门控项目部

负责 R0-R4 风险等级、P0 预警、风险事件、entry 冻结、仓位上限覆盖与人工接管判断。该项目部拥有最高否决权。

### 7. aetfv2_07_qmt_execution：QMT 交易执行项目部

负责 OrderIntent、QMT / miniQMT / XtQuant 适配、mock broker、下单前风控、订单日志、成交 / 持仓回读。第一阶段默认人工确认，禁止实盘全自动。

### 8. aetfv2_08_control_center：总控项目部

负责统一合同、调度 7 个项目部、处理冲突、生成最终决策、输出前端快照和 Action API。所有正式决策必须经过 `control_center`。

## 总控优先级

当项目部之间出现冲突时，按照以下顺序裁决：

1. `risk_warning` / P0 风险
2. 真实持仓风险
3. `market_state`
4. `pre_selection`
5. `entry` / `exit`
6. `qmt_execution`
7. `learning` / `historical_ml`

高优先级模块的否决、冻结、人工接管或风险降级要求，必须覆盖低优先级模块的建议。

## 禁止事项

- 不得绕过 `control_center` 直接交易。
- 不得让 `entry` / `exit` 直接调用 QMT。
- 不得在 R3 / R4 / P0 风险下生成普通买入。
- 不得把 `historical_ml` 建议自动改成正式交易参数。
- 不得把 `OrderIntent` 当成自动下单。
- 不得刷新行情、生成信号、改 `output`，除非任务明确要求。
- 不得提交 token、key、env、venv、大体积缓存。
- 不得提交 `data/cache`、行情缓存、策略输出、临时回测结果或大体积生成文件。
- 不得把子 agent 的建议直接视为最终决策。

## 文件与运行边界

- 默认只允许修改当前任务明确授权的文件。
- 未经明确授权，不运行策略、不刷新行情、不生成信号、不写入 `output`、不触碰 `data/cache`。
- Lab 表格 ML / PyTorch / baseline smoke 命令必须使用 `E:\aetfq3-lab\.venv\Scripts\python.exe`，或先执行 `.\.venv\Scripts\Activate.ps1`。不推荐裸 `python`；若使用裸 `python`，必须先执行 `python -c "import sys; print(sys.executable)"` 并确认输出为 `E:\aetfq3-lab\.venv\Scripts\python.exe`。系统 Python 缺少 ML 包不是 P0，只要 `.venv` 正常即可。
- 涉及交易、执行、真实持仓、风控冻结的任务，必须先经过 `aetfv2_08_control_center`。
- 涉及 QMT 的任务，第一阶段只允许人工确认流程、mock broker、适配层审查和日志审查，不允许实盘全自动。

## 所有项目部统一输出要求

每个项目部的输出必须包含以下字段：

## 审查范围

说明本次查看了哪些模块、文件、接口、数据合同或行为边界。

## 修改范围

说明本次是否修改文件、修改哪些文件、是否只限于本部门职责。

## P0 Blocker

列出必须立即阻断的风险；如无，写“无”。

## P1 高风险

列出可能导致交易错误、风控失效、数据污染或执行误导的高风险问题；如无，写“无”。

## P2 建议

列出优化建议、后续任务或低风险改进项；如无，写“无”。

## 证据

给出文件路径、函数名、配置项、命令输出摘要或其他可审计证据。

## 影响模块

说明可能影响的项目部、模块、接口、前端或运行流程。

## 测试命令

列出已运行或建议运行的验证命令。若因任务边界未运行，必须说明原因。

## 下一张 Codex 任务卡

给出一个清晰、可独立执行的下一步任务，方便继续派发。

## Commit 前检查清单

- `git diff --name-only` 只包含本任务授权文件。
- 未提交 token、key、env、venv、大体积缓存。
- 未刷新行情、未生成正式信号、未写入 `output`、未触碰 `data/cache`，除非任务明确要求。
- agent 名称、TOML 文件名、文档中的职责边界保持一致。
- `.codex/config.toml` 保持 `max_threads = 4` 与 `max_depth = 1`，第一阶段只允许主 agent 派一层子 agent。
