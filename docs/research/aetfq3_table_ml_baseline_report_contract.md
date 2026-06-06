# AETF Q3 Lab Baseline Smoke Report Contract

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

Baseline smoke report 是 Lab-only 的代码路径验证报告。它只证明 no-save baseline smoke 工具能读取样本、执行 chronological split、运行极小型评估并输出可审计指标。它不是正式训练报告，不是 advisory 包，不是 Stable 输入，也不是交易建议。

## 必需边界

报告必须明确：

- `lab_only=true`
- `no_save=true`
- `no_tuning=true`
- `no_stable=true`
- `no_qmt=true`
- `no_order_intent=true`
- `no_output=true`
- `no_lab_advisory=true`
- `model_saved=false`
- `checkpoint_saved=false`

报告不得包含：

- `OrderIntent`
- `target_weight`
- `final_buy_action`
- `stable_action`
- `qmt_order`
- `live_order`
- `trade_instruction`

## 必需结构

`tools/lab/table_ml_baseline_report_reader.py` 至少校验以下字段：

- `report_type`
- `task_scope`
- `target_label`
- `feature_columns`
- `forbidden_columns`
- `train_count`
- `valid_count`
- `split_method`
- `group_leakage_check`
- `models`
- `metrics`
- `prediction_file`
- `review_checklist`

`split_method` 必须是 `chronological`，`group_leakage_check` 必须是 `passed`。

## Metrics 解释限制

Accuracy、ROC AUC、log loss、by-date summary、by-sector summary 只用于 smoke / code path validation。它们不代表模型有效性，不代表策略收益，不代表 Stable 可用性，也不得被解释为交易建议。

## Stable/QMT 边界

Stable 不得把 baseline smoke report 读取为正式交易输入。报告不得修改 `final_buy_action`、`target_weight`、BUY / PROBE 阈值，不得生成 OrderIntent，不得触发 QMT，不得写 `output/`、Stable runtime 或 `lab_advisory/`。

## Reader 行为

Reader 只读 baseline smoke report JSON，输出 summary JSON 到 stdout。若必需字段缺失、边界字段不符合要求，或发现交易相关字段，reader 必须返回非 0。

## Review Checklist 映射

1. 研究了什么：Lab-only baseline smoke report contract。
2. 数据来自哪里：baseline smoke report JSON。
3. 是否来自 Stable bundle：否。
4. 是否有未来函数：reader 校验 feature/forbidden contract，但不重新计算样本。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：否，不是 advisory 包。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议：如需要更强 contract，可在后续任务中把 report writer 输出改为扁平 contract 原生字段。
