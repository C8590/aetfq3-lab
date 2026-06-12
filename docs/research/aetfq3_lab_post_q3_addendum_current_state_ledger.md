本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research current-state ledger，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Post-Q3 Addendum / Current State Ledger

## Purpose

This document is a post-Q3 addendum and current state ledger for `aetfq3-lab / Lab`. It does not replace the Q3 phase closeout. It only records Lab work completed after the Q3 closeout and reconciles the current E / A / F / Stable state so future tasks do not confuse post-Q3 F public label gates with the older Q3 closeout baseline.

## Baseline

Q3 phase closeout remains the baseline:

- `docs/research/aetfq3_lab_q3_phase_closeout.md`
- `docs/research/aetfq3_lab_q3_phase_closeout.json`
- closeout commit: `04b5049f5a2c7b878e8423370a21736127d9ff3d`

The Q3 closeout decision remains unchanged: no Stable promotion, no QMT, no OrderIntent, no model training, no trading advice.

## Post-Q3 Completed Work

- `c08f5f9` public no-label tensor validation: passed.
- `039f158` label/outcome design: completed.
- `1664daa` manifest leakage checker: completed.
- `2d1cfe3` label intake orchestrator: completed with initial `BLOCKED_MISSING_FUTURE_WINDOW_SOURCE`.
- `3130b1c` coverage gate fix: fixed and regression validated.
- `a6e721f` label pause closeout: `PAUSED_BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA`.
- `5f544d2` `three_day_positive_label` formula / eligible-anchor label dry-run: completed.
- `91eea9a` supervised smoke readiness precheck: `SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED`.
- `c6323f6` supervised no-save smoke: `NO_SAVE_SUPERVISED_SMOKE_COMPLETED_REVIEW_REQUIRED`.
- `5e9e254` no-save smoke repeatability checker: completed for repeatability tooling.
- larger eligible-anchor no-save smoke / repeatability ignored reports: completed and reviewed for closeout.
- `c449bd8` majority-class collapse diagnostic: completed.
- `0f096df` group-level sample dry-run: `GROUP_LEVEL_SAMPLE_DRY_RUN_PASSED_CLASS_DIVERSE_REVIEW_REQUIRED`.
- `0a6a8b6` group-level readiness precheck: `GROUP_LEVEL_SUPERVISED_SMOKE_READINESS_PASSED_WITH_LABEL_INCONSISTENCY_REVIEW_REQUIRED`.
- `d1b2ab6` group-level no-save diagnostic smoke: `GROUP_LEVEL_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_WITH_P1_LABEL_INCONSISTENCY_REVIEW_REQUIRED`.
- `6d5464a` feature scale diagnostic / transform policy: `FEATURE_SCALE_DIAGNOSTIC_COMPLETED_TRANSFORM_POLICY_RECOMMENDED`.
- `8f4c528` transform-aware no-save diagnostic smoke: `TRANSFORM_AWARE_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_REVIEW_REQUIRED`.
- `a5ae839` rolling OOP pool capture runner: `ROLLING_OOP_POOL_LIMITED_ACCUMULATING`.
- `ccf7342` manual/export historical 5m intake validator: `MANUAL_HISTORICAL_5M_PACKAGE_NOT_FOUND_WAITING_FOR_INPUT` by default when no package is present.
- this closeout commit: manual/export historical 5m intake validator closeout and ledger update.
- `eb9b9dd` rolling-origin walk-forward no-save validation: `ROLLING_ORIGIN_WALK_FORWARD_DIAGNOSTIC_STABILITY_OBSERVED_REVIEW_REQUIRED`.
- `cb966bc` rolling-origin stability attribution: `ROLLING_ORIGIN_STABILITY_ATTRIBUTION_LAB_MONITOR_CANDIDATE_REVIEW_READY`.
- this monitor candidate protocol commit: Lab-only intraday monitor candidate registered; Stable remains blocked.
- `ff14cb8` Lab monitor refresh gate: `LAB_MONITOR_REFRESH_NOT_DUE`.
- `52585c3` refresh gate evaluable-group accounting patch: raw groups, evaluable groups, and gate groups are now separated; gate count basis is `evaluable_groups`.
- this closeout commit: Lab monitor refresh gate closeout and post-Q3 ledger update.

## Current State Ledger

| Line | Status | Evidence docs | Blocker | Next trigger | Allowed next action | Forbidden action |
| --- | --- | --- | --- | --- | --- | --- |
| E | `E_STATUS=LAB_ONLY_SMOKE_REPLAY_COMPLETED` | Q3 closeout; sector internal ranking smoke/replay summaries referenced there | None for Lab-only engineering replay; not Stable evidence | New Lab-only E research task with explicit boundary | Lab-only read-only replay or documentation | Treating smoke/replay as trading advice or Stable promotion evidence |
| A | `A_STATUS=READY_RECONSTRUCTED_ONLY` | Q3 closeout; reconstructed v2 no-save smoke docs referenced there | True historical V2/ML_SIM still missing; reconstructed is not true historical and not true ML_SIM | True historical V2/ML_SIM complete package appears | Intake-only historical validation planning | Treating reconstructed A as true historical, ML_SIM proof, Stable evidence, or trading advice |
| F-public | `F_PUBLIC_STATUS=LAB_MONITOR_CANDIDATE_REVIEW_READY_REFRESH_NOT_DUE_STABLE_BLOCKED` | `aetfq3_intraday_public_no_label_tensor_validation.*`; `aetfq3_intraday_label_outcome_design.*`; `aetfq3_intraday_label_manifest_leakage_checker.*`; `aetfq3_intraday_label_generation_intake_orchestrator.*`; `aetfq3_intraday_label_generation_pause_closeout.*`; `aetfq3_intraday_supervised_smoke_readiness_precheck.*`; `aetfq3_intraday_supervised_no_save_smoke.*`; `aetfq3_intraday_supervised_no_save_repeatability_check.*`; `aetfq3_intraday_group_level_transform_aware_no_save_smoke.*`; `aetfq3_intraday_rolling_oop_pool_capture.*`; `aetfq3_intraday_historical_5m_manual_intake_closeout.*`; `aetfq3_intraday_fixed_shortlist_oop_no_save_validation.*`; `aetfq3_intraday_oop_post_sprint_reversal_attribution.*`; `aetfq3_intraday_rolling_origin_walk_forward_no_save_validation.*`; `aetfq3_intraday_rolling_origin_stability_attribution_review.*`; `aetfq3_intraday_lab_monitor_candidate_protocol.*`; `aetfq3_intraday_lab_monitor_candidate_closeout.*`; `aetfq3_intraday_lab_monitor_refresh_gate.*`; `aetfq3_intraday_lab_monitor_refresh_gate_closeout.*`; ignored Lab monitor refresh gate report | A unique Lab-only monitor candidate is active and review-ready after rolling-origin stability attribution, but refresh is not due. Raw post-sprint groups are `56`, evaluable groups are `32`, gate groups are `32`, group basis is `evaluable_groups`, post-sprint anchors are `7`, and the rerun gate did not pass. This is not Stable evidence. | New compliant raw/manual data, or post-sprint anchors >= 10 and evaluable groups >= 50 | `wait_for_new_data_or_manual_review`; if new raw export appears, run broker export packager and manual intake validator; if anchors/evaluable groups pass the gate, rerun fixed-shortlist OOP no-save validation and attribution as a separate Lab-only task | Stable promotion, QMT, OrderIntent, advisory package, formal training, threshold tuning, model deployment, checkpoint/model/scaler save, automatic promotion, BUY/PROBE threshold change, `target_weight` change, or `final_buy_action` change |
| F-real/QMT | `F_REAL_QMT_STATUS=BLOCKED_NO_SAFE_REAL_PROVIDER_OR_COMPLIANT_EXPORT` | Q3 closeout and provider blocker docs referenced there | No safe real provider or compliant export; no QMT connection; no account/position/order/trade access | Safe real provider or compliant export appears | Static review / intake-only provider validation | Connecting QMT, reading account data, placing orders, or generating OrderIntent |
| Stable | `STABLE_ALLOWED=false`; `QMT_ALLOWED=false`; `ORDER_INTENT_ALLOWED=false`; `ADVISORY_ALLOWED=false`; `TRAINING_ALLOWED=false`; `CHECKPOINT_ALLOWED=false` | Q3 closeout and post-Q3 closeout docs | No promotion-ready evidence | Formal promotion gate with true inputs and human review | None in this ledger | Any Stable runtime/output write, parameter change, `final_buy_action`, `target_weight`, BUY / PROBE threshold change, QMT, OrderIntent, advisory package |

## Blockers

- A true historical blocked: missing true historical V2/ML_SIM complete input.
- F public Lab monitor candidate review ready: rolling-origin stability attribution identified one Lab-only monitor candidate, but it is not Stable evidence and cannot trigger Stable/QMT/OrderIntent/formal training/advisory/automatic promotion.
- F public Lab monitor refresh gate is not due: no new raw/manual data was detected, post-sprint anchors remain `7`, and evaluable post-sprint groups remain `32` below the `50` group gate threshold.
- F public post-sprint forward-only evidence remains underpowered; rolling-origin stability does not erase that review item.
- F public transform-aware diagnostic smoke completed: raw logistic collapse was reproduced, balanced/scaled and log1p/scaled/balanced probes reduced collapse, and no formal model evidence exists. P1 label inconsistency, extreme feature scale, and train/valid shift reviews remain required.
- F real/QMT blocked: missing safe real provider or compliant export.

## Gate Semantics

- raw presence READY does not equal effective readiness.
- coverage gate is the formal label generation gate.
- coverage insufficient must block label generation dry-run.
- reconstructed A does not equal true historical.
- public-data validation does not equal Stable evidence.
- smoke, reconstructed, and public-data validation outputs are not trading advice and do not prove model effectiveness.
- no-save supervised smoke metrics are not effectiveness evidence and do not authorize automatic promotion.
- larger eligible-anchor repeatability metrics are not effectiveness evidence and do not authorize automatic promotion.
- majority-class collapse observed in logistic regression is a model-signal review item, not a Stable promotion signal.
- group-level balanced/scaled diagnostic probe reducing collapse is not formal model evidence and does not authorize Stable promotion, formal training, QMT, OrderIntent, advisory, or deployment.
- transform-aware diagnostic smoke metrics are not formal model evidence and do not authorize Stable promotion, formal training, QMT, OrderIntent, advisory, automatic promotion, or deployment.
- rolling OOP pool readiness is data-pool coverage readiness only, not model effectiveness evidence, not trading advice, and not authorization for fixed-shortlist OOP validation before the 10-anchor / 50-group threshold is reached.
- manual/export historical 5m intake readiness is intake-tool readiness only, not package validation, not fixed-shortlist OOP validation, not model effectiveness evidence, not trading advice, and not Stable promotion evidence.
- Lab monitor candidate readiness is rolling-origin diagnostic monitor status only, not Stable evidence, not trading advice, not formal training readiness, not QMT readiness, not OrderIntent readiness, and not automatic promotion readiness.
- Lab monitor refresh gate readiness is a read-only freshness/status check only, not Stable evidence, not trading advice, not formal training readiness, not QMT readiness, not OrderIntent readiness, not threshold tuning permission, and not automatic promotion readiness.
- Lab monitor refresh gate group count basis is `evaluable_groups`; raw groups must not drive rerun readiness, and fallback to raw groups is forbidden.

## Boundary

- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model training
- no torchrun
- no checkpoint
- no strategy parameter changes
- no `final_buy_action` changes
- no `target_weight` changes
- no BUY / PROBE threshold changes
- not trading advice

## Next Triggers

Only five trigger classes may reopen work:

1. A true historical V2/ML_SIM complete package appears.
2. F public rolling OOP pool accumulates strict OOP anchors >= 10 and group_count >= 50.
3. F safe real provider / compliant export appears.
4. A legal external historical ETF 5m manual/export package appears in the ignored inbox and passes manual intake validation.
5. A Lab-only periodic monitor refresh is explicitly requested with compliant data and read-only boundaries. Current refresh gate is not due until new raw/manual data appears or post-sprint anchors >= 10 and evaluable groups >= 50.

For the current F-public monitor candidate, periodic Lab-only monitor refresh is allowed only as read-only diagnostics. Stable promotion, formal training, QMT, OrderIntent, advisory, automatic next-stage model experiment, model deployment, BUY/PROBE threshold change, `target_weight` change, and `final_buy_action` change remain forbidden.

## Final Decision

- `LAB_POST_Q3_STATUS=PAUSED_WAITING_FOR_TRUE_INPUTS`
- `STABLE_PROMOTION_READY=false`
- `QMT_READY=false`
- `ORDER_INTENT_READY=false`
- `LABEL_GENERATION_READY=false`
- `TRAINING_READY=false`
- `F_PUBLIC_STATUS=LAB_MONITOR_CANDIDATE_REVIEW_READY_REFRESH_NOT_DUE_STABLE_BLOCKED`
- `F_PUBLIC_LAB_MONITOR_CANDIDATE_STATUS=LAB_MONITOR_CANDIDATE_STATUS_ACTIVE_REVIEW_READY`
- `F_PUBLIC_LAB_MONITOR_CANDIDATE_ID=label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy`
- `F_PUBLIC_LAB_MONITOR_STABLE_EVIDENCE=false`
- `F_PUBLIC_LAB_MONITOR_AUTOMATIC_PROMOTION_READY=false`
- `F_PUBLIC_LAB_MONITOR_REFRESH_GATE_STATUS=LAB_MONITOR_REFRESH_NOT_DUE`
- `F_PUBLIC_LAB_MONITOR_REFRESH_RAW_GROUP_COUNT=56`
- `F_PUBLIC_LAB_MONITOR_REFRESH_EVALUABLE_GROUP_COUNT=32`
- `F_PUBLIC_LAB_MONITOR_REFRESH_GATE_GROUP_COUNT=32`
- `F_PUBLIC_LAB_MONITOR_REFRESH_GROUP_COUNT_BASIS=evaluable_groups`
- `F_PUBLIC_LAB_MONITOR_REFRESH_RERUN_GATE_PASSED=false`
- `F_PUBLIC_ROLLING_OOP_POOL_STATUS=ROLLING_OOP_POOL_LIMITED_ACCUMULATING`
- `F_PUBLIC_FIXED_SHORTLIST_OOP_VALIDATION_READY=false`
- `F_PUBLIC_ROLLING_OOP_ELIGIBLE_ANCHORS=2`
- `F_PUBLIC_ROLLING_OOP_GROUP_COUNT=16`
- `F_PUBLIC_MANUAL_5M_INTAKE_VALIDATOR_READY=true`
- `F_PUBLIC_MANUAL_5M_PACKAGE_PRESENT=false`
- `F_PUBLIC_MANUAL_5M_PACKAGE_READINESS_DECISION=MANUAL_HISTORICAL_5M_PACKAGE_NOT_FOUND_WAITING_FOR_INPUT`
- `F_PUBLIC_LARGER_ELIGIBLE_ANCHOR_STATUS=LARGER_NO_SAVE_REPEATABILITY_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED`
- `F_PUBLIC_GROUP_LEVEL_MODEL_SIGNAL_STATUS=GROUP_LEVEL_DIAGNOSTIC_SMOKE_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED_NO_FORMAL_EVIDENCE`
- `F_PUBLIC_GROUP_LEVEL_TRANSFORM_AWARE_STATUS=TRANSFORM_AWARE_DIAGNOSTIC_SMOKE_COMPLETED_REVIEW_REQUIRED_NO_FORMAL_EVIDENCE`
- `F_PUBLIC_MODEL_SIGNAL_REVIEW_REQUIRED=true`
- `F_PUBLIC_MAJORITY_CLASS_COLLAPSE_OBSERVED=true`
- `F_PUBLIC_LOGISTIC_MATCHES_DUMMY_MOST_FREQUENT=true`
- `F_PUBLIC_BALANCED_SCALED_REDUCES_COLLAPSE=true`
- `F_PUBLIC_LOG1P_SCALED_BALANCED_REDUCES_COLLAPSE=true`
- `F_PUBLIC_FORMAL_MODEL_EVIDENCE=false`
- `F_PUBLIC_SUPERVISED_SMOKE_STATUS=LARGER_NO_SAVE_REPEATABILITY_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED`
