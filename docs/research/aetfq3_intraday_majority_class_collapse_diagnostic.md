本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Lab Intraday Majority-Class Collapse Diagnostic

## 任务定位

本文件定义 F public intraday majority-class collapse feature / label / split / probability diagnostic。它只用于解释 no-save supervised smoke 中 `logistic_regression` 退化为 majority-class 行为的可能原因，不是正式训练，不是调参，不保存模型，不进入 Stable。

## 诊断目标

诊断工具：

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_majority_class_collapse_diagnostic.py --samples .local_research_outputs\aetfq3_lab\intraday_larger_eligible_anchor_readiness\larger_eligible_anchor_label_samples.csv --manifest .local_research_outputs\aetfq3_lab\intraday_larger_eligible_anchor_readiness\larger_eligible_anchor_label_manifest.json --readiness .local_research_outputs\aetfq3_lab\intraday_larger_eligible_anchor_readiness\supervised_smoke_readiness_report.json --repeatability .local_research_outputs\aetfq3_lab\intraday_larger_eligible_anchor_no_save_repeatability_check\intraday_larger_eligible_anchor_no_save_repeatability_report.json --out-dir .local_research_outputs\aetfq3_lab\intraday_majority_class_collapse_diagnostic\
```

检查范围：

- label distribution: overall / train / valid / per-anchor / per-ETF。
- sample granularity: row_count、anchor_count、ETF count、anchor x ETF group count、bars_per_anchor_etf、重复标签结构。
- feature leakage recheck: `future_*`、`*_label`、`future_return_1d`、`future_return_3d`、`max_drawdown_3d`、`three_day_positive_label`、execution/outcome 字段均不得进入 feature_columns。
- feature scale diagnostic: train / valid mean、std、min、max、standardized mean difference、zero variance、scale ratio、missing / inf。
- univariate signal diagnostic: class 0 mean、class 1 mean、absolute standardized difference、simple univariate ROC-AUC。
- logistic probability diagnostic: CPU sklearn no-save replay，只 fit train，只输出 valid probability summary 与 threshold 0.5 prediction distribution。
- balanced/scaled diagnostic probe: `StandardScaler` 只 fit train，`LogisticRegression(class_weight="balanced")` 只输出 diagnostic comparison。

## 输出边界

只允许写入：

```text
.local_research_outputs/aetfq3_lab/intraday_majority_class_collapse_diagnostic/
```

输出文件：

- `intraday_majority_class_collapse_diagnostic_report.json`
- `intraday_majority_class_collapse_diagnostic_report.md`
- `diagnostic_decision.json`

## 安全声明

- diagnostic 不是训练。
- balanced/scaled probe 只是诊断，不是调参结论。
- 不保存模型。
- 不保存 checkpoint。
- 不运行 torchrun。
- 不调用 GPU。
- 不接 QMT。
- 不生成 OrderIntent。
- 不进入 Stable。
- 不构成交易建议。
- metrics 不得解释为模型有效性、交易建议或 Stable evidence。

## 诊断结论

工具输出一个或多个 flags：

- `TRAIN_VALID_LABEL_SHIFT_OBSERVED`
- `FEATURE_SCALE_RISK_OBSERVED`
- `WEAK_UNIVARIATE_SIGNAL_OBSERVED`
- `GROUP_REPEATED_LABEL_STRUCTURE_OBSERVED`
- `LOGISTIC_THRESHOLD_COLLAPSE_OBSERVED`
- `BALANCED_SCALED_PROBE_REDUCES_COLLAPSE`
- `BALANCED_SCALED_PROBE_STILL_COLLAPSED`
- `NO_FORMAL_MODEL_EVIDENCE`

可输出的 `diagnostic_decision`：

- `DIAGNOSTIC_COMPLETED_FEATURE_LABEL_REVIEW_REQUIRED`
- `DIAGNOSTIC_COMPLETED_BALANCED_SCALED_PROBE_RECOMMENDED`
- `DIAGNOSTIC_COMPLETED_PAST_ONLY_FEATURE_EXPANSION_RECOMMENDED`
- `DIAGNOSTIC_COMPLETED_GROUP_LEVEL_SAMPLE_RECOMMENDED`
- `BLOCKED_MANIFEST_LEAKAGE_P0`
- `BLOCKED_BOUNDARY_FLAG`
- `BLOCKED_DIAGNOSTIC_RUNTIME_ERROR`

即使 diagnostic probe 看起来更好，也必须保持：

- `stable_promotion_ready=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`

## 后续允许方向

诊断后只能进入：

- feature / label review
- past-only feature expansion design
- group-level sample design
- no-save diagnostic smoke 申请

不得自动进入 formal training、Stable promotion、QMT、OrderIntent 或交易结论。
