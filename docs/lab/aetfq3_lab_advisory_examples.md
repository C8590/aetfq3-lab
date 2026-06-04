# AETF Q3 Lab Advisory Examples

## 文档定位

本文档只提供 `aetfq3-lab / Lab` advisory JSON 的 example / schema 示例。它不是运行产物，不创建 `lab_advisory/` 目录，不生成真实 advisory 包，不产生 Stable 可消费运行文件。

配套 JSON 示例见 `docs/lab/aetfq3_lab_advisory_examples.json`。该 JSON 同样只是 mock/example 文档，不包含真实交易建议、真实 `OrderIntent` 或真实账户信息。

## 通用边界

所有示例必须保持：

- `protocol_reference: pending human confirmation`
- `protocol_repo: pending`
- `protocol_version: pending`
- `protocol_freeze_commit: pending`
- `protocol_tag: pending`
- `access_mode: READ_ONLY`
- `affects_stable_trading: false`
- `advisory_only: true`
- `final_action_change_allowed: false`
- `contains_live_order: false`
- `contains_secret: false`
- `requires_human_review: true`
- `recommended_for_stable: false` 默认值
- `forbidden_actions` 必须包含全部禁止动作

如果某个示例展示 `recommended_for_stable: true`，必须填写 `stable_merge_minimal_plan`，并明确仍需 Stable 侧人工审批、`RiskGate` 检查和回滚方案。即使建议进入 Stable，也不允许 Lab 直接提交到 Stable。

包含 secret、live order 或 `final_action_change_allowed=true` 的 bundle 一律禁止，不得作为 Lab -> Stable advisory bundle。

在 protocol repo / tag / freeze commit 经人工确认前，示例不得写死未经确认的 commit 或 tag，也不得把任何 hardcoded protocol rc1 锚点作为 Stable adoption 依据。

## 通用字段说明

- `schema_version`: 示例 schema 版本。
- `example_only`: 必须为 `true`，表示这不是运行产物。
- `artifact_name`: 预期 advisory 文件名。
- `generated_at`: 示例时间戳，不代表真实生成时间。
- `lab_name`: 固定为 `aetfq3-lab`。
- `protocol_reference`: 固定为 `pending human confirmation`，直到 protocol repo / tag / commit 经人工确认。
- `protocol_repo`: 固定为 `pending`，直到人工确认。
- `protocol_version`: 固定为 `pending`，直到人工确认。
- `protocol_freeze_commit`: 固定为 `pending`，直到人工确认。
- `protocol_tag`: 固定为 `pending`，直到人工确认。
- `source_task`: 示例任务名或任务类型。
- `data_sources`: mock 数据来源说明，不得引用真实账户或真实订单。
- `uses_stable_bundle`: 是否使用 Stable bundle，示例必须显式填写。
- `has_future_leakage_check`: 是否完成未来函数检查，示例必须显式填写。
- `access_mode`: 必须为 `READ_ONLY`。
- `affects_stable_trading`: 必须为 `false`。
- `advisory_only`: 必须为 `true`。
- `final_action_change_allowed`: 必须为 `false`。
- `contains_live_order`: 必须为 `false`。
- `contains_secret`: 必须为 `false`。
- `recommended_for_stable`: 默认 `false`。
- `requires_human_review`: 必须为 `true`。
- `stable_merge_minimal_plan`: 仅在建议进入 Stable 时非空；否则写明 no merge proposed。
- `forbidden_actions`: 必须包含禁止修改交易动作、仓位、风控、下单、Stable runtime、Stable output 的动作。
- `summary`: mock 摘要，不得包含正式交易指令。
- `evidence_files`: 示例 evidence 路径，可指向文档摘要或本地 ignored 路径。
- `risk_notes`: 数据、模型、执行、Stable 接入风险说明。

## 示例一：ml_advisory_summary.json

用途：记录 ML false downgrade、样本质量、模型诊断和建议观察点。

边界：

- 不直接改变 Stable entry。
- 不修改 `final_buy_action`。
- 不修改 `target_weight`。
- 不把模型输出直接接入 Stable。

示例见 JSON 的 `examples.ml_advisory_summary`。

## 示例二：sector_research_report.json

用途：记录 sector map、同板块 ETF 排序、第一板块切入位置、强主线保护等研究结论。

边界：

- 不直接改变 BUY / PROBE 阈值。
- 不直接改变 Stable 板块排序逻辑。
- 若建议进入 Stable，必须提供最小合并方案和人工审批点。

示例见 JSON 的 `examples.sector_research_report`。该示例展示 `recommended_for_stable: true` 的写法，但仍然是 example / schema，不是运行产物。

## 示例三：intraday_watch_research.json

用途：记录 Intraday Watch Engine、5分钟K、盘中事件和观察信号研究。

边界：

- 不写 `output/`。
- 不生成正式买卖动作。
- 不把盘中研究当作正式交易计划。

示例见 JSON 的 `examples.intraday_watch_research`。

## 示例四：qmt_readonly_report.json

用途：记录 QMT readonly 或 mock 观察结果。

边界：

- 不接正式 QMT。
- 不自动下单。
- 不生成 `OrderIntent`。
- 不包含真实账户、真实持仓、真实成交或真实券商回报。

示例见 JSON 的 `examples.qmt_readonly_report`。

## 示例五：model_diagnostics.json

用途：记录 PyTorch / GRU / TCN 执行模型的诊断摘要。

边界：

- 不提交模型权重。
- 不提交训练样本。
- 不把模型诊断直接变成 Stable 参数。
- 不把模型输出直接用于正式交易。

示例见 JSON 的 `examples.model_diagnostics`。

## 禁止误用

这些示例不得被误用为：

- 真实 advisory 包。
- Stable 正式交易计划。
- 正式 `OrderIntent`。
- QMT 执行命令。
- 自动下单依据。
- 修改 `final_buy_action`、`target_weight`、BUY / PROBE 阈值的授权。
- 绕过 `RiskGate` 的授权。
- 直接提交到 Stable 的授权。

## Stable 审批边界

Lab 可以建议 Stable，但不能直接改 Stable。任何建议进入 Stable 的示例或实际 advisory 都必须包含：

1. 最小合并方案。
2. 风险点。
3. Stable 侧人工审批要求。
4. `RiskGate` 检查点。
5. 回滚方案。
6. 不允许直接提交到 Stable的声明。

Stable 侧必须独立审阅数据来源、未来函数检查、风险边界和最小合并方案后，才可以决定是否创建 Stable 侧任务。

## 任务结束检查清单

- 只写 schema/example 文档。
- 未创建 `lab_advisory/` 目录。
- 未生成真实 advisory 包。
- 未写 `output/`。
- 未提交 `data/`、`artifacts/`、`.local_research_outputs/`。
- 未包含真实交易建议、真实 `OrderIntent` 或真实账户信息。
- `affects_stable_trading` 均为 `false`。
- `advisory_only` 均为 `true`。
- `access_mode` 均为 `READ_ONLY`。
- `final_action_change_allowed` 均为 `false`。
- `contains_live_order` 均为 `false`。
- `contains_secret` 均为 `false`。
- `requires_human_review` 均为 `true`。
- `forbidden_actions` 包含全部必需禁止动作。
