本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETFQ3 Intraday Group-Level Transform-Aware No-Save Smoke

## Purpose

This document defines a Lab-only transform-aware no-save diagnostic smoke for public intraday group-level past-only feature samples.

The smoke consumes the feature scale diagnostic transform policy recommendation and checks whether train-only scaling plus log1p treatment for raw amount / volume flow features changes the known majority-class collapse behavior. It is not formal model training, not hyperparameter tuning, not scaler persistence, not model persistence, not QMT, not OrderIntent, not Stable promotion, and not trading advice.

## Inputs

Read-only inputs:

- group-level past-only feature samples CSV
- group-level past-only feature manifest JSON
- supervised smoke readiness report JSON
- feature scale diagnostic transform policy recommendation JSON
- feature scale diagnostic report JSON
- prior group-level no-save diagnostic smoke report JSON

Feature columns must come from `manifest.feature_columns` and must exclude future, outcome, and label fields. Manifest leakage check must pass before any smoke model runs.

## Transform Rules

The transform policy is applied only inside this diagnostic run:

- `log1p` is applied only to policy-recommended raw amount / volume flow features.
- Features with negative, missing, or non-finite values are skipped for `log1p`.
- Ratio, rank, return, and relative fields are not treated as raw flow features for `log1p`.
- `StandardScaler` is fit on train rows only.
- Valid rows are transformed only with train-fitted scaler statistics.
- No scaler is saved.
- No clipping or winsorization is applied; clip/winsorize candidates are review-only.
- Valid metrics are not used to choose thresholds, transforms, or model settings.

## Smoke Models

Allowed diagnostic probes:

- `dummy_most_frequent`
- `dummy_stratified`
- `logistic_regression_raw`
- `logistic_regression_balanced_scaled`
- `logistic_regression_log1p_scaled_balanced`

The smoke intentionally excludes LightGBM, CatBoost, XGBoost, PyTorch, GPU, torchrun, model save, scaler save, and checkpoint creation.

## Required Report Fields

Outputs must include:

- `report_type=intraday_group_level_transform_aware_no_save_smoke`
- `smoke_scope=lab_only_transform_aware_no_save_diagnostic`
- `target=three_day_positive_label`
- `transform_policy_applied`
- `log1p_features_applied`
- `log1p_features_skipped`
- `standard_scaler_fit_scope=train_only`
- `scaler_saved=false`
- `models_run`
- `train_group_count`
- `valid_group_count`
- `train_label_distribution`
- `valid_label_distribution`
- `metrics`
- `prediction_distribution_by_model`
- `collapse_check`
- `comparison_to_baseline_group_smoke`
- `model_saved=false`
- `checkpoint_saved=false`
- `gpu_used=false`
- `torchrun_used=false`
- `qmt_used=false`
- `order_intent_generated=false`
- `stable_affected=false`
- `metrics_are_effectiveness_evidence=false`
- `automatic_promotion_ready=false`
- `not_trading_advice=true`

## Collapse Review

The smoke must compare:

- whether raw logistic reproduces the ordinary majority-class collapse;
- whether balanced scaled logistic reduces collapse;
- whether log1p scaled balanced logistic reduces collapse;
- whether diagnostic metrics remain below or near dummy baselines.

Expected review flags may include:

- `RAW_LOGISTIC_COLLAPSE_REPRODUCED`
- `BALANCED_SCALED_REDUCES_COLLAPSE`
- `LOG1P_SCALED_BALANCED_REDUCES_COLLAPSE`
- `NO_FORMAL_MODEL_EVIDENCE`
- `TRANSFORM_AWARE_SMOKE_REVIEW_REQUIRED`

## Decision

If the smoke completes with no forbidden artifacts, the readiness decision is:

`TRANSFORM_AWARE_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_REVIEW_REQUIRED`

Even when transform-aware variants reduce collapse, the result remains diagnostic-only and must retain:

- `NO_FORMAL_MODEL_EVIDENCE`
- `metrics_are_effectiveness_evidence=false`
- `automatic_promotion_ready=false`

## Boundary

- no formal training
- no hyperparameter search
- no scaler save
- no model save
- no checkpoint
- no GPU
- no torchrun
- no QMT
- no QMT API
- no account / cash / position / order / fill reads
- no OrderIntent
- no automatic order
- no `output/`
- no Stable runtime/output
- no Stable modification
- no `lab_advisory/`
- no advisory package
- not trading advice

Passing this smoke can only support human review / closeout. It does not authorize automatic promotion or direct Stable submission.
