本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab group-level diagnostic smoke model-signal review closeout，不是 Stable promotion，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Group-Level Model Signal Review Closeout

## Purpose

This document closes out the F public intraday group-level no-save diagnostic smoke model-signal review. It records Lab-only diagnostic findings after the group-level sample dry-run and supervised smoke readiness precheck. It is not Stable promotion, not formal training, not model deployment, not trading advice, and not Stable evidence.

## Completed chain

- majority-class collapse diagnostic
- group-level sample dry-run
- group-level readiness precheck
- group-level no-save diagnostic smoke

## Group-level sample summary

- group_count: `296`
- anchors: `37`
- ETFs: `8`
- group_label_policy: `anchor_close_last_bar`
- intraday_live_decision_ready: `false`
- P1 warning: `P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED`

The group-level sample came from the existing F public intraday group-level dry-run chain. It was not sourced from a Stable bundle.

## Diagnostic smoke summary

- readiness: `GROUP_LEVEL_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_WITH_P1_LABEL_INCONSISTENCY_REVIEW_REQUIRED`
- target: `three_day_positive_label`
- models run:
  - `dummy_most_frequent`
  - `dummy_stratified`
  - `logistic_regression`
  - `logistic_regression_balanced_scaled`
- train/valid split:
  - train groups: `200`
  - valid groups: `96`
  - train labels: `0=60`, `1=140`
  - valid labels: `0=55`, `1=41`
- metrics:
  - `dummy_most_frequent`: accuracy `0.427083`, balanced_accuracy `0.5`
  - `dummy_stratified`: accuracy `0.447917`, balanced_accuracy `0.474723`
  - `logistic_regression`: accuracy `0.427083`, balanced_accuracy `0.5`
  - `logistic_regression_balanced_scaled`: accuracy `0.46875`, balanced_accuracy `0.464967`
- no-save checks:
  - `model_saved=false`
  - `checkpoint_saved=false`
  - `gpu_used=false`
  - `torchrun_used=false`
  - `qmt_used=false`
  - `order_intent_generated=false`
  - `stable_affected=false`
  - `metrics_are_effectiveness_evidence=false`

Smoke metrics were generated only to confirm that the Lab-only CPU sklearn no-save diagnostic flow runs on the group-level sample. They do not prove model effectiveness and do not authorize Stable promotion.

## Collapse review

- ordinary logistic still collapsed.
- ordinary logistic prediction distribution: `0=0`, `1=96`
- `logistic_matches_dummy_most_frequent=true`
- balanced_scaled prediction distribution: `0=50`, `1=46`
- balanced_scaled reduces collapse.
- balanced_scaled balanced_accuracy: `0.464967`
- balanced/scaled does not prove model signal.
- balanced accuracy remains below `0.5`.
- no formal model evidence.

Required flags:

- `GROUP_LEVEL_LOGISTIC_MATCHES_DUMMY_MOST_FREQUENT`
- `GROUP_LEVEL_BALANCED_SCALED_PROBE_REDUCES_COLLAPSE`

## Boundary

- no Stable
- no QMT
- no OrderIntent
- no advisory
- no formal training
- no model save
- no checkpoint
- no automatic promotion
- not trading advice

## Final review status

`F_PUBLIC_GROUP_LEVEL_MODEL_SIGNAL_STATUS=GROUP_LEVEL_DIAGNOSTIC_SMOKE_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED_NO_FORMAL_EVIDENCE`

- `stable_promotion_ready=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`

## Next allowed actions

Only these actions are allowed:

1. human review of label inconsistency
2. past-only feature expansion design
3. group-level feature diagnostic design
4. optional no-save diagnostic smoke with separate task card

The following actions remain forbidden:

- Stable promotion
- formal training
- QMT
- OrderIntent
- model deployment
