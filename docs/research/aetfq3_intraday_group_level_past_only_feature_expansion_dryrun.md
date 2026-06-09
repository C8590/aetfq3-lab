本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Lab Intraday Group-Level Past-Only Feature Expansion Dry Run

## Task Scope

This document defines the Lab-only dry-run that materializes the reviewed group-level past-only feature expansion design.

The dry-run:

- reads existing public intraday 5m bar samples
- groups rows by `trade_date + etf_code`
- generates past-only group-level features
- preserves existing group-level last-bar outcomes and labels as outcome/label fields only
- writes ignored Lab outputs
- runs manifest leakage, feature quality, class-balance, and readiness prechecks

It does not run no-save supervised smoke, does not train, does not tune hyperparameters, does not connect QMT, does not generate OrderIntent, does not create advisory packages, and does not enter Stable.

## Inputs

- `docs/research/aetfq3_intraday_group_level_past_only_feature_expansion_design.json`
- `docs/research/aetfq3_intraday_group_level_feature_leakage_checklist.json`
- `.local_research_outputs/aetfq3_lab/intraday_larger_eligible_anchor_readiness/larger_eligible_anchor_label_samples.csv`
- `.local_research_outputs/aetfq3_lab/intraday_group_level_sample_dryrun/intraday_group_level_samples.csv`

The group-level sample path is retained as an audit input, while feature generation uses the bar-level public sample so anchor-day structure can be recomputed.

## Time Semantics

Required semantics are preserved:

```text
feature_time_scope = anchor_day_only_or_prior
label_time_scope = after_anchor_day
group_label_policy = anchor_close_last_bar
intraday_live_decision_ready = false
```

Generated features use only anchor-day bars available up to the anchor-day close. Existing T+1/T+3 outputs remain outcome or label fields and are forbidden from `feature_columns`.

## Generated Outputs

Only ignored Lab output is allowed:

```text
.local_research_outputs/aetfq3_lab/intraday_group_level_past_only_feature_expansion_dryrun/
```

Expected files:

- `intraday_group_level_past_only_feature_samples.csv`
- `intraday_group_level_past_only_feature_manifest.json`
- `intraday_group_level_past_only_feature_report.md`
- `intraday_group_level_past_only_feature_report.json`
- `feature_quality_precheck.json`
- `class_balance_precheck.json`
- `supervised_smoke_readiness_report.json`
- `readiness_decision.json`

## Feature Generation Rules

The tool must generate the required core features:

- `open_first`
- `high_max`
- `low_min`
- `close_last`
- `volume_sum`
- `amount_sum`
- `vwap_day`
- `day_return`
- `high_low_range`
- `close_to_vwap`
- `intraday_return_mean`
- `intraday_return_std`
- `distance_to_vwap_mean`
- `distance_to_vwap_last`
- `volume_first_half_sum`
- `volume_second_half_sum`
- `amount_first_half_sum`
- `amount_second_half_sum`

When inputs are available, it also generates anchor-day optional features and same-date cross-sectional relative features. Past daily context features are skipped when only intraday 5m samples are available; they must not be fabricated.

## Prechecks

The dry-run records:

- manifest leakage status
- generated feature count
- skipped feature count
- per-feature missing and infinite counts
- zero-variance features
- extreme-scale features
- train/valid standardized mean-difference diagnostics when an anchor-date split is available
- group count, anchor count, ETF count, label null count, label distribution, class count, and minimum class count
- readiness decision

## Readiness Decisions

Possible decisions:

- `GROUP_LEVEL_PAST_ONLY_FEATURE_EXPANSION_DRY_RUN_PASSED_READINESS_REVIEW_REQUIRED`
- `GROUP_LEVEL_PAST_ONLY_FEATURE_EXPANSION_DRY_RUN_PASSED_WITH_FEATURE_QUALITY_WARNINGS`
- `BLOCKED_FEATURE_GENERATION_TOO_FEW_FEATURES`
- `BLOCKED_MANIFEST_LEAKAGE_P0`
- `BLOCKED_BOUNDARY_FLAG`
- `BLOCKED_GROUP_LEVEL_READINESS`

Even when passed, all boundary fields remain false:

- `training_allowed=false`
- `stable_allowed=false`
- `qmt_allowed=false`
- `order_intent_allowed=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`

## Boundary

This dry-run is not training, not no-save smoke, not model evidence, not trading advice, not QMT, not OrderIntent, and not Stable promotion.
