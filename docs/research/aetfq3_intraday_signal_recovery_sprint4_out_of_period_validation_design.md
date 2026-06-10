本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Intraday Signal Recovery Sprint4 Out-of-Period Validation Design

Sprint4 is an out-of-period no-save validation design and public data readiness task. It does not run model validation, does not train, does not tune hyperparameters, does not save a model, scaler, or checkpoint, does not connect QMT, does not generate OrderIntent, and does not enter Stable.

## Scope

Sprint3 produced 3 robust diagnostic candidate shortlist families. Sprint4 asks whether these candidates currently have a strict out-of-period public data window suitable for a later no-save validation task.

## Shortlist Fixed For Any Future OOP Task

| Label policy | Feature set | Model family | Transform policy |
|---|---|---|---|
| `label_ret3d_gt_100bp` | `base_39_plus_scale_transform_policy` | `logistic_balanced_scaled_variants` | `scale_transform_policy` |
| `label_ret3d_gt_100bp` | `base_39_plus_past_daily_plus_scale_transform_policy` | `logistic_balanced_scaled_variants` | `scale_transform_policy` |
| `label_safe_positive_3d` | `base_39_plus_scale_transform_policy` | `logistic_balanced_scaled_variants` | `scale_transform_policy` |


The shortlist must not be changed by any future OOP result. Future OOP output must not be used to retrofit the shortlist.

## OOP Window Design

The Sprint1/Sprint2 anchor range is `2026-04-09` to `2026-06-03`. A valid OOP window must be strictly outside this range, must have public 5m bars, and must have T+1/T+3 public daily future coverage.

| Candidate window | Dates | Strict OOP | Sprint overlap | Eligible anchors now | Data readiness |
|---|---|---:|---:|---:|---|
| `primary_future_after_sprint` | `2026-06-04` to `2026-06-09` | `True` | 0 | 0 | blocked: insufficient current public data |
| `backup_pre_sprint` | `2026-04-01` to `2026-04-08` | `True` | 0 | 0 | blocked: insufficient current public data |
| `overlap_control_not_selectable` | `2026-04-09` to `2026-06-03` | `False` | 37 | 0 | not selectable: overlaps Sprint anchors |


## Data Readiness

- Public 5m coverage in the current source is `2026-04-09` to `2026-06-03` and therefore fully overlaps the Sprint1/Sprint2 anchor range.
- Later OOP candidates after `2026-06-03` currently have no public 5m bars in this source.
- Earlier OOP candidates before `2026-04-09` currently have daily future coverage but no public 5m bars in this source.
- Current estimated eligible strict OOP anchors: `0`.
- Minimum required eligible OOP anchors: `10`.
- Minimum required group count: `50`.
- Class labels are expected in a later validation task but are not generated in Sprint4.
- Source is public market data export, not a Stable bundle.

## Validation Gate

A later separately authorized no-save OOP validation task must keep the shortlist fixed and apply diagnostic-only gates:

- no collapse
- valid prediction contains both classes
- `balanced_accuracy >= dummy + 0.03`
- `ROC-AUC >= 0.53`
- `PR-AUC >= prevalence + 0.03`
- no leakage
- no artifact
- no Stable / QMT / OrderIntent

Even if a later OOP no-save validation passes, it must still keep `formal_model_evidence=false`, `stable_promotion_ready=false`, and `human_review_required=true`.

## Readiness Decision

`SPRINT4_OOP_BLOCKED_INSUFFICIENT_PUBLIC_DATA`

Blocker: No strict out-of-period window currently has both public 5m bars and required T+3 daily future coverage; eligible OOP anchors=0, below minimum 10.

## Boundary

- `formal_model_evidence=false`
- `stable_promotion_ready=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`
- no model run
- no training
- no hyperparameter tuning
- no model/scaler/checkpoint save
- no QMT
- no OrderIntent
- no Stable
- no `output/`
- no `lab_advisory/`
- not trading advice

## Next Allowed Action

Only a separate Lab task may refresh or provide public OOP data and then request a separate no-save OOP validation. Sprint4 itself does not authorize validation execution.
