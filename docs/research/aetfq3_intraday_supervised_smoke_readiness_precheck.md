本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab supervised smoke readiness precheck，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Supervised Smoke Readiness Precheck

## 任务定位

本阶段新增 `intraday_supervised_smoke_readiness_precheck.py`，用于检查 eligible-anchor expanded three-day label dry-run 样本是否满足后续 no-save supervised smoke 的最低申请条件。

这是 readiness precheck，不是训练。它不运行模型，不运行 baseline，不调用 GPU，不运行 `torchrun`，不评估模型胜率，不保存 checkpoint，不保存模型，不接 QMT，不生成 `OrderIntent`，不进入 Stable。通过只代表可以申请下一阶段 no-save supervised smoke，不构成交易建议，也不是 Stable evidence。

## 输入

- labelled intraday 5m samples CSV
- label dry-run manifest JSON

当前 public ignored dry-run 输入为：

- `.local_research_outputs/aetfq3_lab/intraday_eligible_anchor_expanded_three_day_label_dryrun/eligible_anchor_expanded_three_day_label_samples.csv`
- `.local_research_outputs/aetfq3_lab/intraday_eligible_anchor_expanded_three_day_label_dryrun/eligible_anchor_expanded_three_day_label_manifest.json`

## CLI

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_supervised_smoke_readiness_precheck.py --samples .local_research_outputs\aetfq3_lab\intraday_eligible_anchor_expanded_three_day_label_dryrun\eligible_anchor_expanded_three_day_label_samples.csv --manifest .local_research_outputs\aetfq3_lab\intraday_eligible_anchor_expanded_three_day_label_dryrun\eligible_anchor_expanded_three_day_label_manifest.json --out-dir .local_research_outputs\aetfq3_lab\intraday_supervised_smoke_readiness_precheck\
```

输出 ignored 目录：

- `.local_research_outputs/aetfq3_lab/intraday_supervised_smoke_readiness_precheck/intraday_supervised_smoke_readiness_report.md`
- `.local_research_outputs/aetfq3_lab/intraday_supervised_smoke_readiness_precheck/intraday_supervised_smoke_readiness_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_supervised_smoke_readiness_precheck/readiness_decision.json`

## Required Checks

- Manifest leakage checker must pass.
- `three_day_positive_label` must exist in samples.
- `three_day_positive_label` must not be in `feature_columns`.
- `future_return_1d`, `future_return_3d`, and `max_drawdown_3d` must not be in `feature_columns`.
- Label must not be single-class.
- Label null count is reported.
- `anchor_count >= 5`.
- `etf_count >= 2`.
- `row_count >= 500`.
- each class must have at least 50 samples.
- time-based split by anchor date must be feasible.
- train and valid splits must both contain class 0 and class 1.
- boundary flags must remain false: `training_allowed`, `supervised_training_allowed`, `stable_effect_allowed`, `contains_order_intent`, `contains_live_order`, `contains_secret`.

## Split Policy

The precheck only tests split feasibility and never trains:

1. Sort anchor dates ascending.
2. Try `anchor_date_70_30`: first 70% anchors as train, remaining anchors as valid.
3. If valid is single-class, try `anchor_date_60_40`.
4. If train or valid remains single-class, block with `BLOCKED_SPLIT_NOT_CLASS_DIVERSE`.

If samples do not contain `anchor_date`, the checker uses `trade_date` as the anchor date.

## Readiness Decisions

- `SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED`
- `BLOCKED_SINGLE_CLASS_LABEL`
- `BLOCKED_INSUFFICIENT_ROWS`
- `BLOCKED_INSUFFICIENT_ANCHORS`
- `BLOCKED_SPLIT_NOT_CLASS_DIVERSE`
- `BLOCKED_MANIFEST_LEAKAGE_P0`
- `BLOCKED_BOUNDARY_FLAG`

Even when readiness passes, the report keeps:

- `training_allowed=false`
- `stable_allowed=false`
- `order_intent_allowed=false`
- `qmt_allowed=false`

## Boundary

- no training
- no supervised smoke run
- no baseline run
- no GPU
- no torchrun
- no checkpoint
- no model save
- no QMT
- no OrderIntent
- no output/
- no Stable runtime/output
- no lab_advisory/
- not trading advice
- not Stable evidence

## Next Steps

If the precheck returns `SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED`, the only allowed next step is to apply for a separate no-save supervised smoke task with explicit human review. That later task must still keep no QMT, no OrderIntent, no Stable, no checkpoint, and no trading-advice boundaries unless separately authorized by a promotion gate.
