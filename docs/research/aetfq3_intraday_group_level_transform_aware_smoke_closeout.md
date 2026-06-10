本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETFQ3 Intraday Group-Level Transform-Aware Smoke Closeout

## Purpose

This closeout records the completed F public intraday group-level transform-aware no-save diagnostic smoke and the human review decision:

`HUMAN_REVIEW_DECISION_AFTER_TRANSFORM_AWARE_SMOKE=APPROVE_CLOSEOUT_AND_LEDGER_UPDATE_ONLY`

It is a Lab research closeout and ledger update only. It is not Stable promotion, not formal model training, not QMT, not OrderIntent, not model deployment, not advisory generation, and not trading advice.

## Completed chain

- group-level sample dry-run
- group-level readiness precheck
- group-level no-save diagnostic smoke
- group label inconsistency diagnostic
- group-level past-only feature expansion design
- group-level past-only feature expansion dry-run
- feature scale diagnostic / transform policy
- transform-aware no-save diagnostic smoke

## Transform-aware smoke summary

- target: `three_day_positive_label`
- feature_count: `39`
- group_count: `296`
- data source: AKShare public minute / public daily OHLCV
- Stable bundle source: `false`
- readiness: `TRANSFORM_AWARE_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_REVIEW_REQUIRED`
- smoke commit: `8f4c528a4048fa5ed305a1b2ad087d98a13316d8`

Transform policy:

- log1p amount / volume raw-flow features
- train-only `StandardScaler`
- no scaler saved
- no model saved
- clip / winsorization review only; not applied
- valid metrics were not used to choose transforms

Models run:

- `dummy_most_frequent`
- `dummy_stratified`
- `logistic_regression_raw`
- `logistic_regression_balanced_scaled`
- `logistic_regression_log1p_scaled_balanced`

Train / valid split:

- split policy: `anchor_date_70_30`
- train groups: `200`
- valid groups: `96`
- train label distribution: `{0:60, 1:140}`
- valid label distribution: `{0:55, 1:41}`

Metrics, balanced accuracy:

- dummy_most_frequent: `0.5`
- dummy_stratified: `0.474723`
- logistic_raw: `0.5`
- logistic_regression_balanced_scaled: `0.617295`
- logistic_regression_log1p_scaled_balanced: `0.605100`

These metrics are diagnostic-only and are not model effectiveness evidence.

## Collapse review

- raw logistic collapse reproduced: `true`
- raw logistic prediction distribution: `{0:0, 1:96}`
- balanced_scaled prediction distribution: `{0:41, 1:55}`
- log1p_scaled_balanced prediction distribution: `{0:42, 1:54}`
- balanced_scaled reduces collapse: `true`
- log1p_scaled_balanced reduces collapse: `true`
- balanced_scaled balanced_accuracy: `0.617295`
- log1p_scaled_balanced balanced_accuracy: `0.605100`
- formal model evidence: `false`
- metrics_are_effectiveness_evidence: `false`

The transform-aware probes reduced majority-class collapse, but this remains a diagnostic smoke result only. It does not authorize formal training, model deployment, QMT, OrderIntent, advisory generation, automatic promotion, or Stable promotion.

## P1 warnings

- `P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED`
- `P1_EXTREME_FEATURE_SCALE_REVIEW_REQUIRED`
- `P1_TRAIN_VALID_FEATURE_SHIFT_REVIEW_REQUIRED`

## Boundary

- no Stable
- no QMT
- no OrderIntent
- no advisory
- no formal training
- no model save
- no scaler save
- no checkpoint
- no automatic promotion
- not trading advice

## Final review status

`F_PUBLIC_GROUP_LEVEL_TRANSFORM_AWARE_STATUS=TRANSFORM_AWARE_DIAGNOSTIC_SMOKE_COMPLETED_REVIEW_REQUIRED_NO_FORMAL_EVIDENCE`

- `stable_promotion_ready=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`

## Next allowed actions

Only the following are allowed:

1. human review of transform-aware diagnostic result
2. no-save repeatability check with transform-aware pipeline
3. past-daily context input recovery / dry-run design
4. feature/label diagnostic review

The following remain forbidden:

- Stable promotion
- formal training
- QMT
- OrderIntent
- model deployment
- advisory generation
- automatic next-stage model experiment
