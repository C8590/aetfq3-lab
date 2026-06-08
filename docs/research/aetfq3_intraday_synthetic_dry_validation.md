本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Synthetic Dry Validation

## 任务定位

F-8 是 synthetic/mock intraday dry validation tooling。它只验证 Lab 内部 5分钟K dry validation 门禁链路，不读取真实 intraday / tick / 盘口数据，不训练模型，不接 QMT，不生成 OrderIntent，不写 `output/`，不生成 advisory。

目标链路：

1. synthetic intraday manifest
2. intake checker
3. schema validator
4. forbidden feature scan
5. tensor shape dry validation
6. Intraday Watch state-machine dry-run
7. report reader
8. ignored local report

## Tooling

- `tools/lab/intraday_dry_validation_intake_checker.py`
- `tools/lab/intraday_5m_schema_validator.py`
- `tools/lab/intraday_dry_validation_orchestrator.py`
- `tools/lab/intraday_dry_validation_report_reader.py`

Fixture:

- `tests/fixtures/aetfq3_lab/mock_intraday_5m_manifest.json`
- `tests/fixtures/aetfq3_lab/mock_intraday_5m_bad_future_feature_manifest.json`
- `tests/fixtures/aetfq3_lab/mock_intraday_5m_samples.csv`
- `tests/fixtures/aetfq3_lab/mock_intraday_watch_events.json`

## Manifest Contract

Synthetic manifest 必须保持：

- `sample_type=intraday_5m`
- `sample_path_type=repo_mock_fixture`
- `source_kind=synthetic_mock`
- `human_authorized=true`
- `training_allowed=false`
- `stable_effect_allowed=false`
- `advisory_only=true`
- `affects_stable_trading=false`
- `contains_secret=false`
- `contains_live_order=false`
- `contains_order_intent=false`
- `allowed_for=["dry_validation_only", "mock_validation_only"]`
- `has_future_leakage_check=true`
- `review_checklist_passed=true`

`feature_columns` 必须排除所有 `future_*`、所有 `*_label`、`max_drawdown_3d`、`execution_return_to_close`、`execution_return_to_next_open`、`execution_drawdown_after_entry`。

## Validation Steps

1. Intake checker 校验 manifest JSON、必需字段、权限边界、QMT mode 边界、人工授权和 future leakage checklist。
2. Schema validator 校验 mock CSV 必需字段、每个 ETF / trade_date 至少 12 根 bar、`bar_index` 单调和 OHLC 基本合理性。
3. Forbidden feature scan 再次确认 feature 不含 future label 或 post-hoc outcome。
4. Tensor shape dry validation 只构建 tensor shape，不训练、不保存模型、不生成 checkpoint。
5. Intraday Watch state-machine dry-run 只消费 mock events，输出 visited states，不产生交易动作。
6. Report reader 复核 summary report 的 Lab-only 边界字段和禁用 token。

## Report Contract

主报告写入 ignored 目录：

```text
.local_research_outputs/aetfq3_lab/intraday_synthetic_dry_validation/intraday_synthetic_dry_validation_report.json
```

报告必须包含：

- `report_type=intraday_synthetic_dry_validation`
- `task_scope=Lab-only synthetic intraday dry validation`
- `lab_only=true`
- `no_stable=true`
- `no_qmt=true`
- `no_order_intent=true`
- `no_output=true`
- `no_lab_advisory=true`
- `no_training=true`
- `checkpoint_saved=false`
- `model_saved=false`
- `order_intent_generated=false`
- `qmt_allowed=false`
- `stable_effect_allowed=false`
- intake / schema / forbidden feature / tensor shape / state-machine pass flags
- rows, batch, time steps, feature count, target count, visited states

## Boundary

- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model training
- no checkpoint
- no model save
- no real intraday data
- no trading advice

## Next Steps

- 真实数据仍需要 F-7 safe provider 或人工 export。
- 若人工提供独立 safe provider 文件，先重跑 F-7 静态审查。
- 若人工提供 export 文件，先补 manifest、hash、source note 和 intake-only 检查。
- 不进入 Stable。
