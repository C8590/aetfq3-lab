本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab no-save supervised smoke，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Supervised No-Save Smoke

## 任务定位

本阶段新增 `intraday_supervised_no_save_smoke.py`，基于已通过 readiness precheck 的 eligible-anchor expanded three-day label dry-run 样本，执行一次 Lab-only no-save supervised smoke。

这是 no-save smoke，不是正式训练。它只验证最小监督学习流程能否跑通，不调参，不保存模型，不保存 checkpoint，不运行 `torchrun`，不调用 GPU，不接 QMT，不生成 `OrderIntent`，不进入 Stable。smoke metrics 不代表模型有效性，不构成交易建议，也不是 Stable evidence。

## 输入

- labelled intraday 5m samples CSV
- label dry-run manifest JSON
- supervised smoke readiness precheck report JSON

当前 public ignored smoke 输入为：

- `.local_research_outputs/aetfq3_lab/intraday_eligible_anchor_expanded_three_day_label_dryrun/eligible_anchor_expanded_three_day_label_samples.csv`
- `.local_research_outputs/aetfq3_lab/intraday_eligible_anchor_expanded_three_day_label_dryrun/eligible_anchor_expanded_three_day_label_manifest.json`
- `.local_research_outputs/aetfq3_lab/intraday_supervised_smoke_readiness_precheck/intraday_supervised_smoke_readiness_report.json`

## CLI

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_supervised_no_save_smoke.py --samples .local_research_outputs\aetfq3_lab\intraday_eligible_anchor_expanded_three_day_label_dryrun\eligible_anchor_expanded_three_day_label_samples.csv --manifest .local_research_outputs\aetfq3_lab\intraday_eligible_anchor_expanded_three_day_label_dryrun\eligible_anchor_expanded_three_day_label_manifest.json --readiness .local_research_outputs\aetfq3_lab\intraday_supervised_smoke_readiness_precheck\intraday_supervised_smoke_readiness_report.json --out-dir .local_research_outputs\aetfq3_lab\intraday_supervised_no_save_smoke\
```

输出 ignored 目录：

- `.local_research_outputs/aetfq3_lab/intraday_supervised_no_save_smoke/intraday_supervised_no_save_smoke_report.md`
- `.local_research_outputs/aetfq3_lab/intraday_supervised_no_save_smoke/intraday_supervised_no_save_smoke_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_supervised_no_save_smoke/readiness_decision.json`

## Scope

Allowed CPU / sklearn smoke models:

- `dummy_most_frequent`
- `dummy_stratified`
- `logistic_regression`

Forbidden:

- LightGBM
- CatBoost
- XGBoost
- PyTorch
- GPU
- `torchrun`
- checkpoint
- model save

## Required Gates

- Readiness report must be `SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED`.
- Manifest leakage checker must pass.
- Boundary flags must remain false: `training_allowed`, `supervised_training_allowed`, `stable_effect_allowed`, `contains_order_intent`, `contains_live_order`, `contains_secret`.
- Feature columns must come from manifest.
- Target must be `three_day_positive_label`.
- Outcome, label, and `future_*` fields must not enter features.
- Split must use readiness report train/valid anchor dates.
- Out-dir must not contain `.pkl`, `.joblib`, `.pt`, `.pth`, `.ckpt`, or `.onnx`.

## Readiness Decisions

- `NO_SAVE_SUPERVISED_SMOKE_COMPLETED_REVIEW_REQUIRED`
- `BLOCKED_READINESS_NOT_PASSED`
- `BLOCKED_MANIFEST_LEAKAGE_P0`
- `BLOCKED_BOUNDARY_FLAG`
- `BLOCKED_MODEL_ARTIFACT_CREATED`
- `BLOCKED_SMOKE_RUNTIME_ERROR`

## Boundary

- no formal training
- no tuning
- no model save
- no checkpoint
- no GPU
- no torchrun
- no QMT
- no OrderIntent
- no output/
- no Stable runtime/output
- no lab_advisory/
- not trading advice
- metrics are not effectiveness evidence

## Next Steps

If the smoke completes, the result may enter human review only. It does not allow automatic promotion, Stable entry, model deployment, QMT integration, OrderIntent generation, parameter changes, or trading action.
