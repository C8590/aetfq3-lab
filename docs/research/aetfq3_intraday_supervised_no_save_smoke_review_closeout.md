本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab no-save supervised smoke review closeout，不是 Stable promotion，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Supervised No-Save Smoke Review Closeout

## Purpose

This document closes out the F public intraday eligible-anchor no-save supervised smoke for human review. It is not Stable promotion, not model deployment, not formal training, not trading advice, and not Stable evidence.

## Completed Chain

- public 5m OHLCV
- eligible-anchor expansion
- `three_day_positive_label` dry-run
- class-diverse precheck
- supervised smoke readiness precheck
- no-save supervised smoke

## Smoke Result

- target: `three_day_positive_label`
- models run:
  - `dummy_most_frequent`
  - `dummy_stratified`
  - `logistic_regression`
- train anchors:
  - `2026-05-25`
  - `2026-05-26`
  - `2026-05-27`
  - `2026-05-28`
  - `2026-05-29`
- valid anchors:
  - `2026-06-01`
  - `2026-06-02`
  - `2026-06-03`
- train rows: `720`
- valid rows: `432`
- train label distribution: `0=357`, `1=363`
- valid label distribution: `0=338`, `1=94`
- metrics generated: accuracy, balanced accuracy, precision, recall, prediction distribution for each smoke model
- `metrics_are_effectiveness_evidence=false`

Smoke metrics were generated only to confirm the minimal supervised pipeline runs end to end. They do not prove model effectiveness, do not authorize parameter changes, and do not support Stable promotion.

## No-Save Boundary

- `model_saved=false`
- `checkpoint_saved=false`
- `gpu_used=false`
- `torchrun_used=false`
- `qmt_used=false`
- `order_intent_generated=false`
- `stable_affected=false`

## Review Status

`F_PUBLIC_SUPERVISED_SMOKE_STATUS=NO_SAVE_SUPERVISED_SMOKE_COMPLETED_REVIEW_REQUIRED`

## Not Allowed

- no Stable promotion
- no QMT
- no OrderIntent
- no advisory
- no formal training
- no model deployment
- no checkpoint
- no model save
- no automatic promotion
- no trading advice

## Next Allowed Action

- human review
- optional larger eligible-anchor data collection
- optional no-save repeatability check
- no Stable promotion unless separate promotion gate and manual approval exist

Any later task must restate the Lab boundary and must not treat this smoke closeout as permission to train, deploy, trade, or modify Stable.
