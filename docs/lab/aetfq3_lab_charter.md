# AETF Q3 Lab Charter

## 身份声明

每次给 Codex 派发本仓库任务前，必须先写：

```text
本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
```

`aetfq3-lab` 是 AETF Lab / 实验室 / 试车场。它不是 Stable 的附属模块，而是研究实验主线。Lab 的职责是验证想法、做只读研究、形成可审计 advisory 建议包，并把是否值得进入 Stable 的判断说清楚。

第三仓库 protocol reference 已正式发布为 `aetfq3-protocol v0.1.0-rc1`。它用于通信合同 / schema / bundle 校验 / 升级门禁，不是 Lab，也不是 Stable，不授权 Lab 扩权。

```text
protocol_reference: aetfq3-protocol v0.1.0-rc1
protocol_repo: https://github.com/C8590/aetfq3-protocol
protocol_version: v0.1.0-rc1
protocol_freeze_commit: 9e15a78c43ec874441429ef14edad34b36ab83bf
protocol_tag: v0.1.0-rc1
```

Protocol 只定义合同 / schema / 校验 / promotion gate，不授权 Lab 生成正式交易计划、正式 `OrderIntent`、绕过 `RiskGate` 或直接修改 Stable。

## Lab 定位

Lab 可以做研究、回测、模拟、诊断、实验模型和报告。Lab 不能把研究结果直接变成正式交易参数、正式订单或 Stable 运行产物。

Lab 与 Stable 的关系是：

- Lab 产生证据和建议。
- Stable 保持正式交易链路、风控链路和人工确认边界。
- Lab 不能绕过 Stable 的 RiskGate、entry、control center 或执行确认。
- protocol reference 只定义通信合同、schema、bundle 校验和升级门禁职责；不授权 Lab 扩权。

## Lab 负责范围

Lab 负责：

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
- 只读 advisory bundle 的研究侧说明

## Lab 禁止事项

Lab 禁止：

- 不直接改 Stable。
- 不改 Stable entry。
- 不改 `final_buy_action`。
- 不允许 Lab advisory 改变 Stable final action。
- 不改 `target_weight`。
- 不改 BUY / PROBE 阈值。
- 不生成 Stable 正式 `OrderIntent`。
- 不绕过 `RiskGate`。
- 不自动下单。
- 不把研究输出当正式交易计划。
- 不把模型直接接入 Stable。
- 不提交 Stable 运行产物。
- 不把 QMT 实验回写 Stable。
- 不接正式 QMT。
- 不写 Stable `runtime/` 或 runtime exchange。
- 不包含 secret。
- 不包含 live order。
- 不发布 `final_action_change_allowed=true` 的 bundle。

## Protocol Advisory 边界

Lab advisory 必须保持：

```text
access_mode: READ_ONLY
protocol_reference: aetfq3-protocol v0.1.0-rc1
protocol_repo: https://github.com/C8590/aetfq3-protocol
protocol_version: v0.1.0-rc1
protocol_freeze_commit: 9e15a78c43ec874441429ef14edad34b36ab83bf
protocol_tag: v0.1.0-rc1
final_action_change_allowed: false
contains_live_order: false
contains_secret: false
requires_human_review: true
promotion_gate_required: true
```

包含 secret、live order、`final_action_change_allowed=true` 的 bundle 一律禁止。Lab 成果进入 Stable 必须走人工 promotion gate；Lab 不能直接修改 Stable，也不能绕过 Stable `RiskGate`。

## QMT 边界

Lab 只能做 QMT mock、readonly 或模拟盘实验。任何 QMT 实验必须保持只读、可回放、可撤销、可审计，不得回写 Stable，不得生成正式 `OrderIntent`，不得自动下单。

## Stable 输出边界

Lab 给 Stable 的输出只能是只读建议包。允许的建议包文件名包括：

- `ml_advisory_summary.json`
- `sector_research_report.json`
- `intraday_watch_research.json`
- `qmt_readonly_report.json`
- `model_diagnostics.json`
- `research_notes.md`

建议包必须说明数据来源、样本窗口、是否使用 Stable bundle、是否可能有未来函数、风险和最小合并方案。建议包不得被当作正式交易计划，不得自动改变 Stable 参数。

## 进入 Stable 的条件

Lab 只允许提出“是否建议进入 Stable”的判断。若建议进入 Stable，必须提供最小合并方案：

- 只读输入文件或报告名称。
- Stable 需要读取的字段。
- 不改变 `final_buy_action`、`target_weight`、BUY / PROBE 阈值的接入方式。
- 风控前置和人工确认点。
- 回滚方式。

未经人工单独授权，Lab 不允许直接提交到 Stable。

## 任务结束必答

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
