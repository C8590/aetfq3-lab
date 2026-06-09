本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETFQ3 Intraday Group-Level No-Save Diagnostic Smoke

## 当前状态

`F_PUBLIC_GROUP_LEVEL_NO_SAVE_DIAGNOSTIC_SMOKE=LAB_ONLY_DIAGNOSTIC_READY_TO_RUN`

本文件定义 Lab-only group-level no-save diagnostic smoke 的工程入口和边界。该 smoke 只用于验证 group-level 样本上的最小 sklearn 监督流程能否跑通，并观察 majority-class collapse 是否缓解。

## 输入前置

- group-level sample dry-run readiness: `GROUP_LEVEL_SAMPLE_DRY_RUN_PASSED_CLASS_DIVERSE_REVIEW_REQUIRED`
- group-level supervised smoke readiness: `GROUP_LEVEL_SUPERVISED_SMOKE_READINESS_PASSED_WITH_LABEL_INCONSISTENCY_REVIEW_REQUIRED`
- P1 warning 必须保留: `P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED`
- target: `three_day_positive_label`
- group_label_policy: `anchor_close_last_bar`
- intraday_live_decision_ready: `false`

## 执行范围

- CPU-only sklearn diagnostic smoke
- no-save
- no checkpoint
- no torchrun
- no GPU
- no formal training
- no hyperparameter tuning
- no QMT
- no OrderIntent
- no Stable
- no `output/`
- no `lab_advisory/`
- not trading advice

## 模型范围

允许的 diagnostic probe:

- `dummy_most_frequent`
- `dummy_stratified`
- `logistic_regression`
- `logistic_regression_balanced_scaled`

`logistic_regression_balanced_scaled` 只作为 diagnostic probe，用于观察 class-balanced + scaled logistic 是否缓解 majority-class collapse；不得解释为调参、正式训练、交易建议或 Stable evidence。

## 输出

只允许写入 ignored 目录:

`.local_research_outputs/aetfq3_lab/intraday_group_level_no_save_diagnostic_smoke/`

生成:

- `intraday_group_level_no_save_diagnostic_smoke_report.md`
- `intraday_group_level_no_save_diagnostic_smoke_report.json`
- `readiness_decision.json`

## readiness decision

如果 smoke 跑通且 P1 warning 被保留:

`GROUP_LEVEL_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_WITH_P1_LABEL_INCONSISTENCY_REVIEW_REQUIRED`

阻断状态:

- `BLOCKED_GROUP_LEVEL_READINESS_NOT_PASSED`
- `BLOCKED_MANIFEST_LEAKAGE_P0`
- `BLOCKED_BOUNDARY_FLAG`
- `BLOCKED_MODEL_ARTIFACT_CREATED`
- `BLOCKED_SMOKE_RUNTIME_ERROR`

## Stable 边界

本任务不来自 Stable bundle，不影响 Stable 正式交易，不生成 OrderIntent，不连接 QMT，不允许 promotion。Smoke metrics 只代表流程诊断，不是模型有效性证据，不是交易建议，也不得直接提交到 Stable。
