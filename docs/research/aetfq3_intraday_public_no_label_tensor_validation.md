本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Public No-Label Tensor Validation

## 任务定位

本阶段新增 public intraday no-label tensor validation，用于已经通过 hash / source / schema / data quality 的 public 5分钟K OHLCV export。它只检查无监督标签场景下的 `[batch, time_steps, features]` 序列形状、feature 边界和 NaN / Inf，不需要 mock labels，不做 supervised learning，不训练模型。

该工具不是 QMT 接入，不是真实交易系统，不生成 `OrderIntent`，不写 `output/`，不生成 advisory 包，不进入 Stable。

## 输入类型

- source: AKShare `stock_zh_a_minute` / Sina public minute data
- period: 5m
- input: public OHLCV export CSV
- labels_required: false
- target_count: 0

## Validation Steps

1. 读取 public OHLCV export。
2. 校验必需字段：`trade_date`、`datetime`、`etf_code`、`open`、`high`、`low`、`close`、`volume`、`amount`。
3. 若 `vwap` 不存在且存在 `amount` / `volume`，计算 `vwap = amount / volume`，并安全处理 `volume=0`。
4. 构造 feature columns：`open`、`high`、`low`、`close`、`volume`、`amount`、`vwap`、`intraday_return`、`return_from_open`、`distance_to_vwap`。
5. 拦截 `future_*`、`*_label`、`max_drawdown_3d`、`execution_return_to_close`、`execution_return_to_next_open`、`execution_drawdown_after_entry` 进入 feature。
6. 按 `etf_code + trade_date` 分组，构造 no-label tensor shape：`[batch, time_steps, features]`。
7. 要求每组最少 12 根 5分钟K，`feature_count > 0`。
8. 只做 tensor shape validation 和 NaN / Inf check，不运行 MLP / GRU / TCN backward。
9. 写入 ignored local report，不保存模型，不生成 checkpoint。

## Public Ignored Smoke Result

Public ignored smoke 针对 `.local_artifact_backup/aetfq3_lab_sources/intraday_5m_auto_export/intraday_5m_export.csv` 输出：

- rows_checked: 432
- etf_count: 3
- trade_date_count: 3
- batch_size: 9
- min_time_steps: 48
- max_time_steps: 48
- feature_count: 10
- labels_required: false
- target_count: 0
- tensor_shape_passed: true
- model_saved: false
- checkpoint_saved: false
- no QMT / no Stable / no OrderIntent

## Report Contract

Report 写入 ignored 目录：

```text
.local_research_outputs/aetfq3_lab/intraday_public_no_label_tensor_validation/intraday_public_no_label_tensor_report.json
```

报告必须包含：

- `report_type=intraday_public_no_label_tensor_validation`
- `lab_only=true`
- `no_training=true`
- `no_qmt=true`
- `no_order_intent=true`
- `no_stable=true`
- `no_lab_advisory=true`
- `model_saved=false`
- `checkpoint_saved=false`
- rows / ETF / trade_date counts
- batch / min_time_steps / max_time_steps / feature_count
- feature_columns
- nan_count / inf_count
- `tensor_shape_passed`
- `labels_required=false`
- `target_count=0`

## Boundary

- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model training
- no checkpoint
- no model save
- no fake labels
- no synthetic labels
- no trading advice

## Next Steps

- 后续如要 supervised training，需要真实 labels / future outcomes 另行 dry validation。
- 如要接真实 provider，仍需独立 safe provider 文件并先重跑 F-7 静态审查。
- 该 no-label validator 只能作为 public OHLCV 序列形状门禁，不进入 Stable。
