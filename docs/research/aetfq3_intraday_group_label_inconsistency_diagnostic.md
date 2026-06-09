本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETFQ3 Intraday Group Label Inconsistency Diagnostic

## Purpose

This document defines the Lab-only group label inconsistency diagnostic for F public intraday group-level samples. It diagnoses why bar-level labels can differ inside one `(trade_date, etf_code)` group and reviews whether `anchor_close_last_bar` remains suitable as the group-level diagnostic label policy.

This is not training, not no-save smoke, not hyperparameter tuning, not Stable promotion, not QMT, not OrderIntent, and not trading advice.

## Diagnostic Scope

The diagnostic compares:

- bar-level public intraday samples
- group-level samples generated with `group_key=["trade_date", "etf_code"]`
- group-level dry-run report statistics

For each group it records:

- `bar_count`
- first bar label
- last bar label
- `label_0_count`
- `label_1_count`
- `label_switch_count`
- `label_unique_count`
- `future_return_3d` min/max
- close min/max
- `close_last`
- whether the last-bar label matches the group-level label

## Inconsistency Interpretation

Group-internal label inconsistency can be expected in a bar-level label set because each 5m bar can use a different anchor close as the outcome denominator. A group can therefore contain both class `0` and class `1` labels even when the end-of-anchor-day group label is stable.

The diagnostic may emit these drivers:

- `BAR_LEVEL_OUTCOME_DENOMINATOR_VARIATION`
- `THRESHOLD_NEAR_ZERO_LABEL_FLIP`
- `DATA_QUALITY_SUSPECT`
- `GROUP_POLICY_EXPECTED_DIAGNOSTIC_BEHAVIOR`

Near-zero flips are diagnostic evidence only. They do not prove model signal and do not permit Stable promotion.

## Policy Review

`anchor_close_last_bar` is an end-of-anchor-day diagnostic policy. It is suitable when the last-bar label matches the group-level label and no data quality suspect condition is found.

It is not an intraday live decision policy. The diagnostic must preserve:

`intraday_live_decision_ready=false`

## Outputs

Only ignored Lab output is allowed:

`.local_research_outputs/aetfq3_lab/intraday_group_label_inconsistency_diagnostic/`

Generated files:

- `intraday_group_label_inconsistency_report.md`
- `intraday_group_label_inconsistency_report.json`
- `policy_review_decision.json`

## Boundary

- no model training
- no no-save smoke
- no hyperparameter tuning
- no torchrun
- no GPU
- no model save
- no checkpoint
- no QMT
- no OrderIntent
- no Stable
- no `output/`
- no `lab_advisory/`
- not trading advice
- metrics are not effectiveness evidence

## Expected Decision

If no data quality anomaly is found and last-bar labels match group-level labels, the preferred decision is:

`GROUP_LABEL_POLICY_ANCHOR_CLOSE_LAST_BAR_ACCEPTED_FOR_END_OF_DAY_DIAGNOSTIC`

The decision does not authorize formal training, Stable promotion, QMT, OrderIntent, or model deployment.
