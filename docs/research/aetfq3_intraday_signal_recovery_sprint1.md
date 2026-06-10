本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Intraday Signal Recovery Sprint 1

本文件定义 F public intraday signal recovery sprint 1 的 Lab-only diagnostic 方案。它不是正式训练，不保存模型、scaler 或 checkpoint，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。

## Scope

- 生成 dry-run label policy variants，用于排查 `three_day_positive_label` 过粗导致的 label noise。
- 从 public daily source 恢复 past-only 日线 context feature。
- 构造 base、past daily、scale transform policy 组合的 feature set variants。
- 运行 no-save diagnostic model suite，只输出诊断指标。
- 即使发现 `DIAGNOSTIC_SIGNAL_CANDIDATE`，也只允许进入 human review，不允许自动 promotion。

## Boundaries

- `formal_model_evidence=false`
- `stable_promotion_ready=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `requires_human_review=true`
- `promotion_gate_required=true`

## Label Policy Variants

新增 label variants 均为 dry-run outcome labels，不是交易信号，且不得进入 `feature_columns`：

- `label_ret3d_gt_0bp`
- `label_ret3d_gt_20bp`
- `label_ret3d_gt_50bp`
- `label_ret3d_gt_100bp`
- `label_safe_positive_3d`
- `label_neutral_band_20bp`
- `label_neutral_band_50bp`

原始 `three_day_positive_label` 保留用于对照。

## Past Daily Feature Recovery

过去日线特征必须严格 past-only。对 anchor date `D`，只允许使用 `D` 或 `D` 之前的 daily rows，不允许使用 `D+1`、`D+3`、future outcome、或 full train+valid combined future statistics。若数据不足，报告 skipped，不伪造。

## No-Save Diagnostic Suite

允许模型只用于诊断拟合：

- `dummy_most_frequent`
- `dummy_stratified`
- `logistic_balanced_scaled`
- `logistic_log1p_scaled_balanced`
- `random_forest_shallow_no_save`
- `hist_gradient_boosting_no_save`

若 `.venv` 已安装 LightGBM、XGBoost、CatBoost，可选运行对应 CPU/no-save 小模型。所有模型不得保存 artifact，不得调参搜索，不得调用 GPU。

## Candidate Gate

`DIAGNOSTIC_SIGNAL_CANDIDATE` 只表示需要人工复核的诊断候选，不是 formal model evidence。候选条件包括：无 collapse、balanced accuracy 至少比 dummy most frequent 高 0.03、ROC-AUC 至少 0.53、PR-AUC 至少比 prevalence 高 0.03、valid prediction 同时包含两类、无 leakage、无 artifact。

## Outputs

只允许写入 ignored 目录：

`.local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint1/`

允许输出：

- `signal_recovery_sprint1_feature_samples.csv`
- `signal_recovery_sprint1_manifest.json`
- `signal_recovery_sprint1_label_policy_report.json`
- `signal_recovery_sprint1_feature_recovery_report.json`
- `signal_recovery_sprint1_diagnostic_smoke_report.json`
- `signal_recovery_sprint1_decision.json`
- 对应 `.md` 摘要

## Stable Boundary

本 sprint 不修改 Stable，不写 Stable runtime/output，不创建 `lab_advisory/`，不生成 advisory 包，不生成正式交易计划，不生成 OrderIntent，不接 QMT，不读取账户、资金、持仓、委托或成交。
