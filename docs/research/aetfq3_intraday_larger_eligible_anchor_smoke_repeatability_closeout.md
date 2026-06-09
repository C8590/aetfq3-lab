本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab larger eligible-anchor no-save smoke + repeatability review closeout，不是 Stable promotion，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Larger Eligible-Anchor Smoke + Repeatability Closeout

## Purpose

This document closes out the F public intraday larger eligible-anchor no-save supervised smoke and repeatability review. It records Lab-only engineering results and the required model-signal review. It is not Stable promotion, not formal training, not model deployment, not trading advice, and not Stable evidence.

## Completed chain

- larger public eligible-anchor data collection
- class-balance precheck
- supervised smoke readiness precheck
- larger no-save supervised smoke
- larger no-save repeatability check

## Larger data summary

- readiness: `LARGER_ELIGIBLE_ANCHOR_DATA_COLLECTION_PASSED_READINESS_REVIEW_REQUIRED`
- anchors: `37`
- ETF count: `8`
- rows: `14208`
- label_0: `5351`
- label_1: `8857`
- positive_rate: `0.6233811936936937`
- train/valid split: `anchor_date_70_30`
- train rows: `9600`
- valid rows: `4608`
- train label distribution: `0=2821`, `1=6779`
- valid label distribution: `0=2530`, `1=2078`

The larger eligible-anchor sample came from AKShare public minute and public daily OHLCV. It was not sourced from a Stable bundle.

## Smoke summary

- readiness: `LARGER_ELIGIBLE_ANCHOR_NO_SAVE_SUPERVISED_SMOKE_COMPLETED_REVIEW_REQUIRED`
- target: `three_day_positive_label`
- models run:
  - `dummy_most_frequent`
  - `dummy_stratified`
  - `logistic_regression`
- metrics generated: accuracy, balanced accuracy, precision, recall, and prediction distribution
- no-save checks passed:
  - `model_saved=false`
  - `checkpoint_saved=false`
  - `gpu_used=false`
  - `torchrun_used=false`
  - `qmt_used=false`
  - `order_intent_generated=false`
  - `stable_affected=false`
- `metrics_are_effectiveness_evidence=false`

Smoke metrics were generated only to confirm that the Lab-only CPU sklearn no-save pipeline runs on the larger eligible-anchor sample. They do not prove model effectiveness and do not authorize Stable promotion.

## Repeatability summary

- readiness: `LARGER_ELIGIBLE_ANCHOR_NO_SAVE_REPEATABILITY_COMPLETED_REVIEW_REQUIRED`
- seeds: `7`, `13`, `42`, `101`, `2026`
- `dummy_most_frequent` was stable across seeds.
- `dummy_stratified` varied with seed as expected.
- `logistic_regression` was stable across seeds but matched the majority-class dummy.
- metrics variability:
  - `dummy_most_frequent`: all tracked metric ranges were `0.0`
  - `logistic_regression`: all tracked metric ranges were `0.0`
  - `dummy_stratified`: accuracy range `0.014539930555555525`, balanced accuracy range `0.013971894532215923`
- no-save checks passed:
  - `model_saved=false`
  - `checkpoint_saved=false`
  - `gpu_used=false`
  - `torchrun_used=false`
  - `qmt_used=false`
  - `order_intent_generated=false`
  - `stable_affected=false`

## Majority-class collapse review

- observed signal review flag: `MODEL_SIGNAL_REVIEW_REQUIRED_MAJORITY_CLASS_COLLAPSE_OBSERVED`
- `logistic_matches_dummy_most_frequent=true`
- logistic valid prediction distribution: `0=0`, `1=4608`
- This is not a P0 blocker because the smoke and repeatability tasks ran successfully and no forbidden artifact or boundary violation was found.
- This must be reviewed by a human before any further supervised modeling claim is made.
- This must not be interpreted as an effective model, trading advice, Stable evidence, or promotion-ready signal.

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

`F_PUBLIC_LARGER_ELIGIBLE_ANCHOR_STATUS=LARGER_NO_SAVE_REPEATABILITY_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED`

- `stable_promotion_ready=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`

## Next allowed actions

Only these actions are allowed:

1. human review of majority-class collapse
2. feature/label diagnostic design
3. optional no-save diagnostic smoke with explicit task card

The following actions remain forbidden:

- Stable promotion
- formal training
- QMT
- OrderIntent
- model deployment

