# AETFQ3 Lab Intraday Supervised No-Save Repeatability Check

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Scope

This document defines the Lab-only repeatability check for the completed eligible-anchor intraday supervised no-save smoke.

It is a repeatability check, not formal training. It does not expand the sample, tune parameters, save models, save checkpoints, connect QMT, generate OrderIntent, create advisory packages, or promote anything into Stable.

## Fixed Inputs

- sample: eligible-anchor expanded three-day label sample
- target: `three_day_positive_label`
- split: same anchor-date train/validation split as the baseline no-save smoke
- features: same `feature_columns` as the baseline no-save smoke report
- models: `dummy_most_frequent`, `dummy_stratified`, `logistic_regression`
- seeds: `7`, `13`, `42`, `101`, `2026`

LightGBM, CatBoost, XGBoost, PyTorch, GPU, torchrun, checkpointing, and model persistence are out of scope.

## Boundary

- no larger sample
- no market refresh
- no formal training
- no hyperparameter search
- no model save
- no checkpoint
- no GPU
- no torchrun
- no QMT
- no QMT API
- no account, fund, position, order, or trade read
- no OrderIntent
- no `output/`
- no Stable runtime/output
- no Stable modification
- no `lab_advisory/`
- no advisory package

## Metrics Use

Metrics variability is used only to observe engineering repeatability under the same sample, split, feature, and label boundaries. It is not model effectiveness evidence, not trading advice, and not Stable evidence.

## Allowed Outcome

If all seeds run, no model artifacts are created, and boundary checks pass, the readiness decision may be:

`NO_SAVE_SUPERVISED_SMOKE_REPEATABILITY_COMPLETED_REVIEW_REQUIRED`

This allows only human review or a separate application for larger eligible-anchor data collection. It does not authorize Stable promotion or formal training.
