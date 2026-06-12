本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# Intraday Lab Monitor Refresh Gate Closeout

## Purpose

This closeout records the Lab monitor refresh gate and evaluable-group accounting patch. It is not a validation rerun, not rolling-origin validation, not model training, not threshold tuning, and not Stable evidence. It does not authorize Stable promotion, QMT, OrderIntent, advisory output, formal training, model/scaler/checkpoint save, or automatic promotion.

## Completed refresh gate

- tool: `tools/lab/intraday_lab_monitor_refresh_gate.py`
- initial commit: `ff14cb8`
- accounting patch commit: `52585c3`
- latest tests: `366 passed, 5 warnings`
- decision: `LAB_MONITOR_REFRESH_NOT_DUE`
- monitor candidate: `label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy`
- monitor status: `LAB_MONITOR_CANDIDATE_REVIEW_READY`

## Accounting correction

The refresh gate now separates raw post-sprint groups from label/T+3 evaluable groups and gate-used groups.

- raw group count is recorded as `post_sprint_raw_group_count`
- evaluable group count is recorded as `post_sprint_evaluable_group_count`
- gate group count is recorded as `post_sprint_gate_group_count`
- `group_count_basis="evaluable_groups"`
- raw groups do not drive the rerun gate
- the gate no longer falls back to raw groups
- if evaluable group count is unavailable, the decision must be `LAB_MONITOR_REFRESH_REVIEW_REQUIRED`

## Current gate status

- raw groups: `56`
- evaluable groups: `32`
- gate groups: `32`
- post-sprint anchors: `7`
- post-sprint evaluable group threshold: `50`
- `t_plus_1_coverage_passed=true`
- `t_plus_3_coverage_passed=false`
- `rerun_gate_passed=false`
- reason: no new data and evaluable groups below threshold
- refresh decision: `LAB_MONITOR_REFRESH_NOT_DUE`

## Allowed next task

- current: `wait_for_new_data_or_manual_review`
- if new raw export appears: `run_broker_export_packager_and_manual_intake_validator`
- if new anchors and evaluable groups pass the gate: `rerun_fixed_shortlist_oop_no_save_validation_and_attribution`

## Forbidden next tasks

- Stable promotion
- QMT trading
- OrderIntent generation
- formal training
- threshold tuning
- `target_weight` change
- `final_buy_action` change

## Boundary

- `stable_promotion_ready=false`
- `stable_evidence=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `model_saved=false`
- `scaler_saved=false`
- `checkpoint_saved=false`
- not trading advice
