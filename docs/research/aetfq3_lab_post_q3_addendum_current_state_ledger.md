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
- this closeout commit: larger eligible-anchor smoke + repeatability review closeout and ledger update.

## Current State Ledger

| Line | Status | Evidence docs | Blocker | Next trigger | Allowed next action | Forbidden action |
| --- | --- | --- | --- | --- | --- | --- |
| E | `E_STATUS=LAB_ONLY_SMOKE_REPLAY_COMPLETED` | Q3 closeout; sector internal ranking smoke/replay summaries referenced there | None for Lab-only engineering replay; not Stable evidence | New Lab-only E research task with explicit boundary | Lab-only read-only replay or documentation | Treating smoke/replay as trading advice or Stable promotion evidence |
| A | `A_STATUS=READY_RECONSTRUCTED_ONLY` | Q3 closeout; reconstructed v2 no-save smoke docs referenced there | True historical V2/ML_SIM still missing; reconstructed is not true historical and not true ML_SIM | True historical V2/ML_SIM complete package appears | Intake-only historical validation planning | Treating reconstructed A as true historical, ML_SIM proof, Stable evidence, or trading advice |
| F-public | `F_PUBLIC_STATUS=LARGER_NO_SAVE_REPEATABILITY_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED` | `aetfq3_intraday_public_no_label_tensor_validation.*`; `aetfq3_intraday_label_outcome_design.*`; `aetfq3_intraday_label_manifest_leakage_checker.*`; `aetfq3_intraday_label_generation_intake_orchestrator.*`; `aetfq3_intraday_label_generation_pause_closeout.*`; `aetfq3_intraday_supervised_smoke_readiness_precheck.*`; `aetfq3_intraday_supervised_no_save_smoke.*`; `aetfq3_intraday_supervised_no_save_smoke_review_closeout.*`; `aetfq3_intraday_supervised_no_save_repeatability_check.*`; `aetfq3_intraday_larger_eligible_anchor_smoke_repeatability_closeout.*`; ignored larger eligible-anchor smoke / repeatability reports | Larger no-save smoke and repeatability are complete, but logistic regression collapsed to majority-class behavior and model-signal review is required; metrics are not effectiveness evidence and no automatic promotion is allowed | Human review of majority-class collapse, feature/label diagnostic design, or optional no-save diagnostic smoke with explicit task card | Review-only diagnostic follow-up within Lab boundary | Stable promotion, QMT, OrderIntent, advisory, formal training, model deployment, checkpoint/model save |
| F-real/QMT | `F_REAL_QMT_STATUS=BLOCKED_NO_SAFE_REAL_PROVIDER_OR_COMPLIANT_EXPORT` | Q3 closeout and provider blocker docs referenced there | No safe real provider or compliant export; no QMT connection; no account/position/order/trade access | Safe real provider or compliant export appears | Static review / intake-only provider validation | Connecting QMT, reading account data, placing orders, or generating OrderIntent |
| Stable | `STABLE_ALLOWED=false`; `QMT_ALLOWED=false`; `ORDER_INTENT_ALLOWED=false`; `ADVISORY_ALLOWED=false`; `TRAINING_ALLOWED=false`; `CHECKPOINT_ALLOWED=false` | Q3 closeout and post-Q3 closeout docs | No promotion-ready evidence | Formal promotion gate with true inputs and human review | None in this ledger | Any Stable runtime/output write, parameter change, `final_buy_action`, `target_weight`, BUY / PROBE threshold change, QMT, OrderIntent, advisory package |

## Blockers

- A true historical blocked: missing true historical V2/ML_SIM complete input.
- F public larger no-save repeatability completed: model-signal review is required because logistic regression matched the majority-class dummy across all repeatability seeds; metrics are not effectiveness evidence and no automatic promotion is allowed.
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

Only three trigger classes may reopen work:

1. A true historical V2/ML_SIM complete package appears.
2. F public daily OHLCV covers required future-window dates.
3. F safe real provider / compliant export appears.

For the completed F-public larger eligible-anchor no-save smoke and repeatability, only review follow-ups are allowed: human review of majority-class collapse, feature/label diagnostic design, or optional no-save diagnostic smoke with an explicit task card. Stable promotion, formal training, QMT, OrderIntent, advisory, and model deployment remain forbidden.

## Final Decision

- `LAB_POST_Q3_STATUS=PAUSED_WAITING_FOR_TRUE_INPUTS`
- `STABLE_PROMOTION_READY=false`
- `QMT_READY=false`
- `ORDER_INTENT_READY=false`
- `LABEL_GENERATION_READY=false`
- `TRAINING_READY=false`
- `F_PUBLIC_STATUS=LARGER_NO_SAVE_REPEATABILITY_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED`
- `F_PUBLIC_LARGER_ELIGIBLE_ANCHOR_STATUS=LARGER_NO_SAVE_REPEATABILITY_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED`
- `F_PUBLIC_MODEL_SIGNAL_REVIEW_REQUIRED=true`
- `F_PUBLIC_MAJORITY_CLASS_COLLAPSE_OBSERVED=true`
- `F_PUBLIC_LOGISTIC_MATCHES_DUMMY_MOST_FREQUENT=true`
- `F_PUBLIC_SUPERVISED_SMOKE_STATUS=LARGER_NO_SAVE_REPEATABILITY_COMPLETED_MODEL_SIGNAL_REVIEW_REQUIRED`
