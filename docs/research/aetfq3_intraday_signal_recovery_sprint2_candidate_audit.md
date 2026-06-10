本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Intraday Signal Recovery Sprint2 Candidate Audit

本文件定义 F public intraday signal recovery sprint2：candidate audit + no-save robustness diagnostic。它不是正式训练，不保存模型、scaler 或 checkpoint，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。

## Scope

- 读取 Sprint1 diagnostic candidates，对 label policy、feature set、model、transform policy、class balance、PR-AUC margin、collapse 和 dummy improvement 做审计。
- 按 `candidate_family = label_policy + feature_set + model_family + transform_policy` 聚合，避免只挑最高单点 metric。
- 选择最多 12 个 candidate families，执行 no-save robustness validation。
- robustness split 仅允许 time-based：`anchor_date_70_30`、`anchor_date_60_40`、`walk_forward_3fold_contiguous_anchor`。
- seeds 固定为 `7`、`13`、`42`、`101`、`2026`，只用于稳健性诊断，不用于调参。

## Boundaries

- `formal_model_evidence=false`
- `stable_promotion_ready=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`
- `requires_human_review=true`
- `promotion_gate_required=true`

即使出现 `ROBUST_DIAGNOSTIC_SIGNAL_CANDIDATE_REVIEW_REQUIRED`，也只能进入 human review；不得自动 promotion，不得修改 Stable 参数，不得生成 OrderIntent。

## Candidate Audit

Candidate audit 聚合以下维度：

- candidate count by label policy
- candidate count by feature set
- candidate count by model / model family
- candidate count by transform policy
- isolated label-policy candidates
- isolated model-family candidates
- weak PR-AUC margin candidates
- near-collapse prediction distribution candidates

## Robustness Gate

Robust diagnostic candidate 必须满足：

- 至少 2 个 available time-based split；
- 所有 available split 不 collapse；
- balanced accuracy 至少在 2 个 split 中超过 dummy most frequent 0.03；
- ROC-AUC 至少在 2 个 split 中达到 0.53；
- PR-AUC 至少在 2 个 split 中超过 prevalence 0.03；
- 无 model/scaler/checkpoint artifact；
- 无 leakage。

不满足则标记 `CANDIDATE_WEAK_OR_SPLIT_UNSTABLE`。

## Outputs

只允许写入 ignored 目录：

`.local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint2_candidate_audit/`

允许输出：

- `signal_recovery_sprint2_candidate_family_summary.csv`
- `signal_recovery_sprint2_candidate_audit_report.md`
- `signal_recovery_sprint2_candidate_audit_report.json`
- `signal_recovery_sprint2_robustness_report.md`
- `signal_recovery_sprint2_robustness_report.json`
- `signal_recovery_sprint2_decision.json`

## Stable Boundary

本 sprint 不修改 Stable，不写 `output/`，不写 Stable runtime/output，不创建 `lab_advisory/`，不生成 advisory 包，不接 QMT，不读取账户、资金、持仓、委托或成交，不生成真实交易计划。
