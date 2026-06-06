# AETF Q3 Lab Table ML Sample Intake Manifest

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 文档定位

本文定义 A `ML false downgrade` 与 E `同板块 ETF 内部排序` 在真实小样本进入 dry validation 前必须通过的 Lab-only sample intake manifest 门禁。它不是训练器，不是 Stable 接口，不读取真实行情，不读取真实 CSV 行级内容，不生成 advisory 包，不生成 `OrderIntent`。

## 为什么需要 manifest

真实小样本即使只用于 dry validation，也必须先声明来源、授权范围、Stable bundle 边界、时间范围、样本规模、未来函数风险和只读限制。manifest 是进入 schema validator 之前的人工审计入口，用来防止未授权 CSV、Stable 输出、secret、live order 或带未来函数的字段被误用。

## manifest 字段说明

必需字段：

- `manifest_version`: manifest 合同版本。
- `sample_type`: 样本任务类型。
- `sample_path`: 样本路径；checker 只验证路径存在性，不读取 CSV 行级内容。
- `sample_path_type`: 样本路径边界类型。
- `source_kind`: 数据来源类别。
- `source_description`: 人工可读来源说明。
- `generated_at`: manifest 生成时间。
- `generated_by`: manifest 生成者或生成流程。
- `human_authorized`: 必须为 `true`。
- `authorized_by`: 授权人或授权记录说明。
- `authorization_scope`: 授权范围。
- `uses_stable_bundle`: 是否来自 Stable bundle。
- `stable_bundle_path`: Stable bundle 只读路径；未使用 Stable bundle 时可为空字符串。
- `stable_bundle_commit`: Stable bundle commit；未使用 Stable bundle 时可为空字符串。
- `stable_bundle_snapshot_date`: Stable bundle snapshot date；未使用 Stable bundle 时可为空字符串。
- `data_time_start`: 样本开始日期。
- `data_time_end`: 样本结束日期。
- `row_count`: 声明行数。
- `symbol_count`: 声明 ETF 数。
- `sector_count`: 声明板块数。
- `contains_future_labels`: 是否包含未来标签列。
- `future_label_columns`: 未来标签列数组。
- `feature_columns`: 本次拟用于 schema / dry validation 的 feature 列数组。
- `forbidden_feature_columns`: 禁止进入 feature 的列数组。
- `has_future_leakage_check`: 是否已完成人工未来函数检查。
- `allowed_for`: 允许用途数组。
- `training_allowed`: 必须为 `false`。
- `stable_effect_allowed`: 必须为 `false`。
- `advisory_only`: 必须为 `true`。
- `affects_stable_trading`: 必须为 `false`。
- `contains_secret`: 必须为 `false`。
- `contains_live_order`: 必须为 `false`。
- `contains_order_intent`: 必须为 `false`。
- `qmt_related`: 是否涉及 QMT；本阶段默认 P0。
- `review_checklist_passed`: 是否通过 Lab review checklist。
- `notes`: 其他人工说明。

可选字段：

- `path_may_not_exist_for_template`: 模板 manifest 可设为 `true`，真实接入应保持 `false`。
- `qmt_access_mode`: 仅当 `qmt_related=true` 时用于说明 `mock` 或 `readonly` 边界。

## 允许的 sample_type

- `false_downgrade`
- `sector_internal_ranking`

## 允许的 sample_path_type

- `local_ignored`: 路径必须位于 `.local_research_outputs/aetfq3_lab/`。
- `external_readonly`: 外部只读授权样本或人工 mock fixture。
- `stable_bundle_readonly`: Stable bundle 只读抽取，不允许回写 Stable。

## 允许的 source_kind

- `manual_small_sample`
- `lab_generated_small_sample`
- `stable_bundle_extract`
- `external_authorized_extract`

## allowed_for 取值

`allowed_for` 必须是数组，只允许：

- `schema_validation_only`
- `dry_validation_only`
- `mock_validation_only`

`allowed_for` 不得包含 `training` 或任何训练授权含义。

## Stable bundle 只读声明

如果 `uses_stable_bundle=true`，必须填写：

- `stable_bundle_path`
- `stable_bundle_commit` 或 `stable_bundle_snapshot_date`
- `authorization_scope`

Stable bundle 只能作为只读来源，不授权 Lab 修改 Stable，不授权读取 Stable 真实 output，不授权生成正式交易动作，不授权绕过 `RiskGate`。

## 未来函数检查要求

`feature_columns` 不得包含：

- `future_label_columns` 中任何列。
- `forbidden_feature_columns` 中任何列。
- `future_*`
- `max_drawdown_*`
- `best_in_sector_*`
- `top_quantile_*`
- `*_label`
- A false downgrade / E sector internal ranking 的标签字段。

若发现上述字段进入 feature，checker 必须 P0 fail。`has_future_leakage_check` 未为 `true` 时必须至少给 P1 warning，真实 dry validation 前必须人工复核。

## 训练禁止声明

sample intake manifest 不授权训练模型。默认边界：

```text
training_allowed: false
stable_effect_allowed: false
```

本阶段不跑 LightGBM / CatBoost / XGBoost 真实样本训练，不下载大模型，不生产模型权重。

## advisory-only 声明

sample intake 只允许为后续 dry validation 提供只读门禁。默认边界：

```text
advisory_only: true
affects_stable_trading: false
contains_secret: false
contains_live_order: false
contains_order_intent: false
```

该 manifest 不得被视为正式交易计划，不得自动改变 Stable 参数。

## QMT / OrderIntent 禁止声明

本阶段默认不接 QMT，不生成 `OrderIntent`，不读取真实账户、资金、成交或持仓。若 `qmt_related=true`，默认 P0；只有明确 `qmt_access_mode=mock` 或 `readonly` 且已人工授权时，才可降为 P1 warning，并且仍不得进入正式执行。

## 与 schema validator 的关系

`tools/lab/table_ml_sample_intake_checker.py` 位于 schema validator 之前，只检查 manifest 元数据和边界声明，不读取 CSV 行级内容。

`tools/lab/table_ml_schema_validator.py` 负责 CSV schema、主键、标签、feature leakage、时间切分和 group split 校验。真实 CSV 进入 dry validation 前必须先通过 intake checker，再进入 schema validator。

## CLI 用法

```powershell
python tools/lab/table_ml_sample_intake_checker.py --manifest tests\fixtures\aetfq3_lab\mock_sample_intake_manifest.json
```

成功时输出：

```text
OK sample_intake_manifest_valid=true
```

失败时输出 `FAILED` 和 `P0` 明细；高风险但不阻断项输出 `P1`。

## P0 阻断项

- manifest JSON 不可解析。
- 缺少必需字段。
- `sample_type`、`sample_path_type`、`source_kind` 不合法。
- `sample_path` 不存在，且未声明 `path_may_not_exist_for_template=true`。
- `sample_path_type=local_ignored` 但路径不在 `.local_research_outputs/aetfq3_lab/`。
- `human_authorized=false`。
- `training_allowed=true`。
- `stable_effect_allowed=true`。
- `advisory_only=false`。
- `affects_stable_trading=true`。
- `contains_secret=true`。
- `contains_live_order=true`。
- `contains_order_intent=true`。
- `allowed_for` 包含 `training` 或不允许的用途。
- `uses_stable_bundle=true` 但缺少 Stable bundle 元数据。
- `feature_columns` 包含 future label 或 forbidden feature。
- `qmt_related=true` 且不是 mock / readonly 人工授权边界。

## P1 高风险项

- `has_future_leakage_check` 未为 `true`。
- `qmt_related=true`，即使声明为 mock / readonly 且人工授权，也必须人工复核。
- 数据来源、授权范围、时间范围、样本数量不完整时，真实 dry validation 前必须补齐。

## 后续真实 dry validation 流程

1. 人工准备真实小样本 manifest，只声明路径和边界，不在 intake 阶段读取 CSV 行级内容。
2. 运行 sample intake checker。
3. 若 checker 通过，再运行 table ML schema validator。
4. 人工确认未来函数、Stable bundle 只读边界、样本时间范围和用途。
5. 仅在 Lab-only dry validation 范围内继续，不训练模型，不接 Stable，不接 QMT，不生成 advisory 包。

## Review Checklist 自检

1. 研究了什么：定义 Lab-only 真实小样本接入 manifest 与 checker 门禁。
2. 数据来自哪里：仅来自已提交的 Lab 文档和人工 mock fixture 路径声明。
3. 是否来自 Stable bundle：否；如未来来自 Stable bundle，只允许 readonly 且必须声明元数据。
4. 是否有未来函数：本 manifest checker 不读取样本内容，但阻断 future label / forbidden feature 进入 `feature_columns`。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：是。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用；本工具不建议进入 Stable。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议是什么：准备一个人工授权的 local ignored 真实小样本 manifest，先跑 intake checker，再跑 schema validator dry validation。
