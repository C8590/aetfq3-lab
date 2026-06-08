本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Label Manifest Leakage Checker

## 任务定位

本阶段新增 manifest-only leakage checker，用于真实 label 生成前检查 supervised intraday manifest 的 label / outcome / feature 边界。checker 只读取 manifest JSON，不读取行情，不生成 labels，不训练模型，不运行 `torchrun`，不保存 checkpoint，不接 QMT，不生成 `OrderIntent`，不写 `output/`，不创建 `lab_advisory/`，不进入 Stable。

它是生成真实 label 文件之前的门禁，不是交易建议，不是 supervised smoke，也不是模型训练。

## CLI

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_label_manifest_leakage_checker.py --manifest tests\fixtures\aetfq3_lab\mock_intraday_label_manifest_valid.json
```

stdout JSON 包含：

- `status`
- `manifest_path`
- `p0_blockers`
- `p1_warnings`
- `feature_count`
- `label_count`
- `outcome_count`
- `boundary_passed`

## Required Fields

- `sample_type`
- `feature_columns`
- `label_generated`
- `label_source_kind`
- `label_horizon`
- `label_generation_method`
- `label_columns`
- `outcome_columns`
- `label_status_column`
- `insufficient_future_window_policy`
- `feature_label_overlap_check`
- `label_generation_authorized`
- `supervised_training_allowed`
- `training_allowed`
- `stable_effect_allowed`
- `contains_order_intent`
- `contains_live_order`
- `contains_secret`

## P0 Rules

- `feature_columns` 与 `label_columns` 有交集。
- `feature_columns` 与 `outcome_columns` 有交集。
- `feature_columns` 包含 `future_*`。
- `feature_columns` 包含 `*_label`。
- `feature_columns` 包含 `max_drawdown_3d`。
- `supervised_training_allowed=true`。
- `training_allowed=true`。
- `stable_effect_allowed=true`。
- `contains_order_intent=true`。
- `contains_live_order=true`。
- `contains_secret=true`。
- `label_generation_authorized=false`。
- `insufficient_future_window_policy` 缺失。
- `feature_label_overlap_check` 不是 true。

## Test Fixtures

- `tests/fixtures/aetfq3_lab/mock_intraday_label_manifest_valid.json`
- `tests/fixtures/aetfq3_lab/mock_intraday_label_manifest_bad_feature_overlap.json`

## Boundary

- no real label file generation
- no more real market data read
- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model training
- no torchrun
- no checkpoint
- no model save
- no trading advice

## Next Steps

- 后续可将该 checker 接入 supervised label manifest intake-only 流程。
- 若要生成真实 labels，必须先提供 future-window 数据、hash、source note、manifest 和人工授权。
- 真实 label 文件生成后，仍需先通过 manifest-only leakage checker，再申请 supervised dry validation。
- 不进入 Stable。
