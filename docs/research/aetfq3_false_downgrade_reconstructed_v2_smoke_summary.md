# AETF Q3 Lab A False Downgrade Reconstructed V2 Smoke Summary

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

本记录是 A false downgrade reconstructed v2 named no-save baseline smoke 的 Lab-only 小型总结文档。

它是 Lab-only reconstructed v2 no-save smoke，不是 true historical，不是 true ML_SIM，不是 Stable，不是 advisory，不是交易信号，不训练正式模型，不保存模型，不生成 checkpoint，不接 QMT，不生成 OrderIntent，不写 `output/`，不创建 `lab_advisory/`。

## Reconstructed V2 来源

- deterministic reconstructed ML action v2
- selected variant = `balanced_past_only`
- rulebook = replay-equivalent audit artifact
- not true historical
- not true ML_SIM
- true_historical_available=false

Rulebook recovery status: `reconstructed_from_csv_replay_equivalent`。Rulebook selfcheck passed，`balanced_past_only` replay mismatch count=0。该 rulebook 只能说明 CSV replay-equivalent 审计通过，不证明存在 true historical ML_SIM source code。

## 样本摘要

- rows=37
- date range=2026-01-06..2026-04-30
- ETF count=8
- sector count=3
- feature count=23
- train=28
- valid=9
- target=`false_downgrade_3d`

## 校验链路

- base dry validation passed
- feature dry validation passed
- rulebook recovery passed
- review recommendation = `allow_no_save_smoke`
- baseline smoke passed
- reader boundary_passed=true

## 四模型 Smoke 指标

| Model | Report Model | Status | Accuracy | ROC AUC | Log Loss |
| --- | --- | --- | ---: | ---: | ---: |
| numpy_logistic | numpy_logistic_regression_smoke | passed | 0.1111111111111111 | 0.0 | 1.6801540231371865 |
| lightgbm | lightgbm_smoke | passed | 0.6666666666666666 | 0.5 | 0.6555098120860163 |
| catboost | catboost_smoke | passed | 0.5555555555555556 | 0.4444444444444444 | 0.7014873814149075 |
| xgboost | xgboost_smoke | passed | 0.5555555555555556 | 0.3333333333333333 | 0.8087322612603506 |

Metrics only validate smoke/code path and do not prove model effectiveness. These metrics are not trading advice, not model acceptance evidence, and not a Stable promotion signal.

## 命名修正

- output_prefix=`false_downgrade_reconstructed_v2`
- 不再使用 `sector_internal_ranking` 前缀
- report metadata contains `task_name` / `output_prefix` / `sample_type` / `target_label`

Named smoke metadata:

- task_name=`false_downgrade_reconstructed_v2`
- output_prefix=`false_downgrade_reconstructed_v2`
- sample_type=`false_downgrade`
- target_label=`false_downgrade_3d`

## 边界声明

- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model save
- no checkpoint
- no formal training
- no trading advice
- not Stable input

## 当前裁决

- A reconstructed v2 no-save smoke completed
- 不进入 Stable
- 不生成 advisory
- 不做正式训练
- 若要继续，需要 true historical V2/ML_SIM 或更大 Lab reconstructed 样本

## 下一步建议

- 恢复 true historical V2/ML_SIM
- 扩大 reconstructed sample 但继续降级
- 做更多 Lab-only smoke
- 不进入 Stable

## 任务结束回答

- 研究了什么：A false downgrade reconstructed v2 named no-save baseline smoke 的样本、rulebook、校验链路、命名修正和四模型 smoke 指标。
- 数据来自哪里：允许读取的 `.local_research_outputs/aetfq3_lab/false_downgrade_reconstructed_v2/`、`baseline_smoke_named/`、`baseline_smoke/`、`rulebook_recovery/`、`review/` ignored 本地报告。
- 是否来自 Stable bundle：否。
- 是否有未来函数：未发现 future/label/id/group 字段进入 feature；future labels 仅作为标签或 outcome。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：否，本文件不是 advisory package；本任务只写 docs/research 总结。
- 是否建议进入 Stable：否。
- 如果建议进入 Stable，最小合并方案是什么：不建议进入 Stable，因此无合并方案。
- 不允许直接提交到 Stable：已遵守。
