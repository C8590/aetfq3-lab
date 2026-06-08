本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Dry Validation Manifest

## 任务定位

本文件定义 F-2 / F-3 真实 intraday dry validation 的 manifest 规范。Manifest 只授权 Lab-only dry validation 的只读数据摄取审计，不授权训练正式模型，不授权接 QMT，不授权生成 `OrderIntent`，不授权进入 Stable。

## Manifest 必需字段

| 字段 | 默认值 / 类型 | 说明 |
| --- | --- | --- |
| `manifest_version` | string | Manifest 版本，例如 `intraday_dry_validation_manifest_v1`。 |
| `sample_type` | `intraday_5m` | 样本类型。 |
| `sample_path` | string | 样本路径，必须是 local ignored 或 external readonly。 |
| `sample_path_type` | `local_ignored` / `external_readonly` | 不允许 Stable output/runtime。 |
| `source_kind` | string | 例如 `intraday_5m_bar`、`qmt_export_5m_bar`。 |
| `source_description` | string | 数据源说明。 |
| `generated_at` | datetime | Manifest 生成时间。 |
| `generated_by` | string | 生成者，例如 `codex_lab`。 |
| `human_authorized` | false | 实际 dry validation 前必须为 true。 |
| `authorized_by` | null | 授权人。 |
| `authorization_scope` | null | 授权范围，只读 dry validation。 |
| `uses_stable_bundle` | false | 是否来自 Stable bundle。 |
| `stable_bundle_path` | null | Stable bundle 路径。 |
| `stable_bundle_commit` | null | Stable bundle commit。 |
| `stable_bundle_snapshot_date` | null | Stable bundle snapshot date。 |
| `uses_qmt_export` | false | 是否来自 QMT export。 |
| `qmt_export_path` | null | QMT export 路径。 |
| `qmt_mode` | `readonly` / `mock` / `export_only` | 若 QMT 相关，只允许这三类。 |
| `data_time_start` | null | 数据起始时间。 |
| `data_time_end` | null | 数据结束时间。 |
| `row_count` | null | 行数。 |
| `etf_count` | null | ETF 数量。 |
| `trade_date_count` | null | 交易日数量。 |
| `bar_count_per_etf_day` | null | 每 ETF 每日 bar 数，可为统计摘要。 |
| `contains_future_labels` | false | 是否包含 future label。 |
| `future_label_columns` | [] | future label 字段。 |
| `feature_columns` | [] | feature 字段。 |
| `forbidden_feature_columns` | [] | forbidden 字段命中。 |
| `has_future_leakage_check` | false | 是否执行 future leakage check。 |
| `allowed_for` | `["dry_validation_only"]` | 只允许 dry validation。 |
| `training_allowed` | false | 禁止训练正式模型。 |
| `stable_effect_allowed` | false | 禁止影响 Stable。 |
| `advisory_only` | true | 只读研究边界。 |
| `affects_stable_trading` | false | 不影响 Stable 正式交易。 |
| `contains_secret` | false | 不得含 secret。 |
| `contains_live_order` | false | 不得含 live order。 |
| `contains_order_intent` | false | 不得含 OrderIntent。 |
| `qmt_related` | false | 默认非 QMT，除非明确 QMT export。 |
| `review_checklist_passed` | false | 人工复核和工具检查通过后才可 true。 |
| `notes` | [] | 审计备注。 |

## 默认边界

- `sample_type="intraday_5m"`
- `sample_path_type="local_ignored"` 或 `external_readonly`
- `training_allowed=false`
- `stable_effect_allowed=false`
- `advisory_only=true`
- `affects_stable_trading=false`
- `contains_secret=false`
- `contains_live_order=false`
- `contains_order_intent=false`
- `qmt_related=false`，除非明确是 QMT export
- `qmt_mode` 只能是 `readonly` / `mock` / `export_only`
- `allowed_for=["dry_validation_only"]`

## P0 规则

任一命中即停止：

- `human_authorized=false`
- `training_allowed=true`
- `stable_effect_allowed=true`
- `affects_stable_trading=true`
- `contains_secret=true`
- `contains_live_order=true`
- `contains_order_intent=true`
- `qmt_related=true` 且 `qmt_mode` 不是 `readonly` / `mock` / `export_only`
- future label 出现在 `feature_columns`
- 使用未来 bar 作为 feature
- 使用 T+1 / T+3 outcome 作为当前 bar feature
- 写 Stable `output/` / `runtime/`
- 生成 `OrderIntent`
- 接真实 QMT
- 自动下单
- 修改 `final_buy_action` / `target_weight` / BUY / PROBE 阈值

## P1 规则

- 数据源来自 QMT export，需要人工复核。
- 日期范围太短。
- `bar_count_per_etf_day` 不稳定。
- 缺 `vwap` / `amount` / `volume`。
- 缺 sector / candidate context。
- 缺 future leakage check。
- `source_kind` 不明确。

## P2 规则

- 字段命名需标准化。
- 缺少部分可选上下文字段。
- 需要后续补数据质量报告。

## 是否可以进入真实 intraday dry validation

Manifest 规范本身不授权进入 dry validation。只有当真实数据源路径、人工授权、future leakage check、forbidden feature scan、QMT/Stable/OrderIntent 边界全部通过后，才能单独派发 F-3 dry validation 任务。

