本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETFQ3 Intraday Group-Level Feature Scale Diagnostic

## Purpose

This document defines a Lab-only feature scale diagnostic for the public intraday group-level past-only feature expansion dry-run.

The diagnostic reviews feature scale, train/valid distribution shift, zero variance, missing/inf values, and transform policy design. It is not training, not no-save supervised smoke, not hyperparameter tuning, not QMT, not OrderIntent, not Stable promotion, and not trading advice.

## Diagnostic Scope

Inputs are read-only:

- group-level past-only feature samples CSV
- group-level past-only feature manifest JSON
- supervised smoke readiness report JSON

Feature columns must come from `manifest.feature_columns`. They must not contain future, outcome, or label columns.

The diagnostic splits rows by the readiness split. When explicit train/valid anchor dates are unavailable, it reproduces `anchor_date_70_30` from sorted anchor dates and checks the resulting train/valid group counts against readiness.

For each feature it reports:

- train min/max/mean/std
- valid min/max/mean/std
- missing count
- inf count
- zero variance flag
- absolute max
- scale order
- train-vs-valid standardized mean difference

Global scale review includes:

- cross-feature train std ratio
- cross-feature absolute max ratio
- log1p transform candidates
- train-only standardization candidates
- clipping / winsorization review candidates
- train/valid shift flags

## Transform Policy

The transform policy is diagnostic-only:

- `policy_scope=diagnostic_only`
- `train_only_fit_required=true`
- `save_scaler=false`
- `model_training_allowed=false`
- `stable_allowed=false`

Recommended transform groups:

- `log1p_recommended`: raw volume / amount features and extreme spike ratios
- `standardize_recommended`: continuous numeric features, fitted on train only and applied to valid with train statistics
- `clip_winsorize_review`: extreme outlier features, reviewed with train quantiles only
- `no_transform_or_bounded`: ratio / rank / return style features, while still eligible for train-only standardization

This diagnostic does not actually clip, winsorize, fit, save, or persist any scaler.

## Outputs

Only ignored Lab output is allowed:

`.local_research_outputs/aetfq3_lab/intraday_group_level_feature_scale_diagnostic/`

Generated files:

- `intraday_group_level_feature_scale_diagnostic_report.md`
- `intraday_group_level_feature_scale_diagnostic_report.json`
- `transform_policy_recommendation.json`

## Boundary

- no no-save supervised smoke
- no model training
- no hyperparameter tuning
- no torchrun
- no GPU
- no model save
- no scaler save
- no checkpoint
- no QMT
- no OrderIntent
- no Stable
- no `output/`
- no `lab_advisory/`
- not trading advice
- metrics are not effectiveness evidence

## Expected Decision

The successful diagnostic decision is:

`FEATURE_SCALE_DIAGNOSTIC_COMPLETED_TRANSFORM_POLICY_RECOMMENDED`

If no P0 is found and transform policy review is still needed, the result may be used only to request a transform-aware no-save diagnostic smoke. It does not authorize formal training, Stable promotion, QMT, OrderIntent, or model deployment.
