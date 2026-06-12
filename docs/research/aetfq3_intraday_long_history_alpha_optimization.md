# AETF Q3 Lab Intraday Long-History Alpha Optimization

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Scope

This document defines the Lab-only long-history intraday data lake and bounded alpha optimizer v0. The workflow is research-only and may ingest user-provided 1m/5m historical ETF bars, normalize them into an ignored data lake, build time-censored intraday features, and run rolling-origin no-save candidate searches.

## Boundary

- access_mode: READ_ONLY
- final_action_change_allowed: false
- contains_live_order: false
- contains_secret: false
- requires_human_review: true
- promotion_gate_required: true
- formal_training: false
- model_saved: false
- scaler_saved: false
- checkpoint_saved: false
- stable_promotion_ready: false
- qmt_ready: false
- order_intent_generated: false

The optimizer must not search or modify Stable BUY/PROBE thresholds, `target_weight`, `final_buy_action`, OrderIntent, QMT execution, Stable runtime, Stable output, or formal model artifacts.

## Data Lake

The data lake CLI is:

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_long_history_data_lake.py --raw-dir .local_artifact_backup\aetfq3_lab_sources\intraday_long_history_raw_exports --fallback-manual-inbox .local_artifact_backup\aetfq3_lab_sources\intraday_historical_5m_manual_inbox --out-artifact-dir .local_artifact_backup\aetfq3_lab_sources\intraday_long_history_data_lake --out-report-dir .local_research_outputs\aetfq3_lab\intraday_long_history_alpha_optimization
```

It supports CSV, TXT, ZIP, parquet, English headers, Chinese headers, UTF-8/GBK/ANSI text, tabular delimiters, single-ETF files, multi-ETF files, and `SH#` / `SZ#` code patterns. If the long-history raw export folder is absent, v0 may fall back to the ignored manual 5m inbox for smoke validation.

## Features

Signal clocks are `10:00`, `10:30`, `11:00`, `11:30`, `13:30`, `14:00`, `14:30`, and `14:50`. Time-censored intraday features only use bars from the same trading day at or before the signal clock. Future labels and outcomes are diagnostic targets only and must not enter the feature column list.

## Optimizer

The optimizer CLI is:

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_long_history_alpha_optimizer.py --data-lake .local_artifact_backup\aetfq3_lab_sources\intraday_long_history_data_lake --out-dir .local_research_outputs\aetfq3_lab\intraday_long_history_alpha_optimization --mode bounded_search
```

The primary validation protocol is monthly rolling-origin walk-forward with expanding train windows, at least 60 train anchors, at least 10 validation anchors, at least 50 validation groups, and a 3-day embargo. Random split is not a primary conclusion mechanism.

## Candidate Gate

Candidates can only be marked `LAB_DIAGNOSTIC_ALPHA_CANDIDATE_REVIEW_REQUIRED`. A passing candidate must remain Lab-only, human-review-only, no-save, no-QMT, no-OrderIntent, and not Stable evidence. Suggested gate checks include positive net total return, controlled drawdown, positive Calmar-like ratio, win rate above 50%, profit factor above 1.05, monthly positive fraction at least 55%, no one-month or one-ETF domination, 10 bps per-side cost survival, no leakage, and no saved artifact.

## Outputs

Tracked documentation lives in `docs/research/`. Runtime reports and generated data must stay under:

- `.local_artifact_backup/aetfq3_lab_sources/intraday_long_history_data_lake/`
- `.local_research_outputs/aetfq3_lab/intraday_long_history_alpha_optimization/`

These ignored outputs are not Stable bundles, not formal training artifacts, and not promotion evidence.
