# AETF Q3 Lab Table ML Real Small Sample Source Plan

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 文档定位

本文确认 A `ML false downgrade` 与 E `同板块 ETF 内部排序` 的真实小样本来源、最小字段、Stable bundle 边界、人工授权要求和 ignored 落盘路径。本文是 Lab-only 样本准备计划，不读取真实 CSV 行级内容，不训练模型，不接 Stable，不接 QMT，不生成 advisory 包。

## 总边界

默认边界：

```text
access_mode: READ_ONLY
final_action_change_allowed: false
contains_live_order: false
contains_secret: false
requires_human_review: true
promotion_gate_required: true
training_allowed: false
stable_effect_allowed: false
advisory_only: true
affects_stable_trading: false
contains_order_intent: false
qmt_related: false
```

真实小样本只能用于：

- `schema_validation_only`
- `dry_validation_only`

不得用于训练、调参、Stable 参数修改、正式交易判断或 QMT 执行。

## 样本来源确认

### false_downgrade

首选来源：

- 人工授权的 Lab local ignored 小样本。
- 或人工授权的 Stable bundle readonly extract。

不允许来源：

- 未授权真实行情 CSV。
- Stable 真实 `output/`。
- QMT 实盘日志、真实持仓、资金或成交回报。
- `data/etf_daily.csv` 直接读取。

来源确认要求：

- 必须有人确认 `human_authorized=true`。
- 必须填写 `authorized_by` 与 `authorization_scope`。
- 如来自 Stable bundle，必须填写 `uses_stable_bundle=true`、`stable_bundle_path`、`stable_bundle_commit` 或 `stable_bundle_snapshot_date`。
- 如不来自 Stable bundle，必须填写 `uses_stable_bundle=false` 并说明来源为 `manual_small_sample`、`lab_generated_small_sample` 或 `external_authorized_extract`。

### sector_internal_ranking

首选来源：

- 人工授权的 Lab local ignored 小样本。
- 或人工授权的 Stable bundle readonly extract，且只抽取同一 `trade_date + sector` 内 ETF 横截面。

不允许来源同 false_downgrade。

来源确认要求：

- 每个 `ranking_group_id` 必须可追溯到 `trade_date + sector`。
- 同一 group 内至少 2 个 ETF。
- 必须说明 sector mapping 来源、版本和人工授权范围。

## 最小样本规模

本计划只面向 dry validation，不面向训练。建议最小规模：

### false_downgrade

- 至少 8 行。
- 至少 2 个 `trade_date`。
- 至少 4 个 ETF。
- 至少 2 个 sector。
- 至少包含一个疑似 false downgrade 示例、一个 true downgrade 示例、一个 neutral 示例。

### sector_internal_ranking

- 至少 8 行。
- 至少 2 个 `trade_date`。
- 至少 2 个 sector。
- 每个 `ranking_group_id` 至少 2 个 ETF。
- 至少包含一个 best / top quantile / avoid 示例。

这些数量只证明门禁链路可跑通，不代表统计有效性，不授权训练模型。

## 字段要求

### 通用字段

- `trade_date`
- `etf_code`
- `sector`
- `model_version`
- `feature_version`
- `data_source_name`
- `data_source_version`
- `feature_cutoff_time`
- `label_window`
- `split_name`
- `split_cutoff_date`
- `uses_stable_bundle`
- `future_leakage_checked`

### false_downgrade 必需字段

- `etf_name`
- `v2_action`
- `ml_action`
- `v2_score`
- `ml_score`
- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `false_downgrade_1d`
- `false_downgrade_3d`
- `false_downgrade_lock3`
- `true_downgrade`
- `neutral_downgrade`

### sector_internal_ranking 必需字段

- `etf_name`
- `rank_scope`
- `ranking_group_id`
- `sector_member_count`
- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `best_in_sector_1d`
- `best_in_sector_3d`
- `top_quantile_in_sector_3d`
- `avoid_in_sector`
- `pairwise_outperform_label`

## feature_columns 初始建议

### false_downgrade

可先声明：

- `v2_score`
- `ml_score`
- `sector_rank`
- `etf_momentum`

### sector_internal_ranking

可先声明：

- `ml_score`
- `etf_momentum`
- `sector_relative_strength`

## 禁止进入 feature 的字段

以下字段不得进入 `feature_columns`：

- 所有 `future_*`
- 所有 `max_drawdown_*`
- `false_downgrade_1d`
- `false_downgrade_3d`
- `false_downgrade_lock3`
- `true_downgrade`
- `neutral_downgrade`
- `best_in_sector_1d`
- `best_in_sector_3d`
- `top_quantile_in_sector_3d`
- `avoid_in_sector`
- `pairwise_outperform_label`
- 任何 T+1 / T+3 收益、回撤、排名、价格或未来结果。

## ignored 路径规划

真实小样本 CSV 和 manifest 放在 ignored 目录：

```text
.local_research_outputs/aetfq3_lab/table_ml_real_small_samples/false_downgrade/
.local_research_outputs/aetfq3_lab/table_ml_real_small_samples/sector_internal_ranking/
```

建议文件名：

```text
false_downgrade_real_small_sample.csv
false_downgrade_real_small_sample_manifest.json
sector_internal_ranking_real_small_sample.csv
sector_internal_ranking_real_small_sample_manifest.json
```

dry validation 报告仍输出到：

```text
.local_research_outputs/aetfq3_lab/table_ml_dry_validation/
```

这些 ignored 产物不得提交 Git。

## manifest 准备要求

每个真实小样本必须先准备 manifest，并通过 `tools/lab/table_ml_sample_intake_checker.py`。

关键字段：

- `sample_type`
- `sample_path`
- `sample_path_type=local_ignored` 或 `stable_bundle_readonly`
- `source_kind`
- `source_description`
- `human_authorized=true`
- `authorized_by`
- `authorization_scope`
- `uses_stable_bundle`
- `stable_bundle_path`
- `stable_bundle_commit` 或 `stable_bundle_snapshot_date`
- `data_time_start`
- `data_time_end`
- `row_count`
- `symbol_count`
- `sector_count`
- `future_label_columns`
- `feature_columns`
- `forbidden_feature_columns`
- `has_future_leakage_check=true`
- `allowed_for=["schema_validation_only", "dry_validation_only"]`
- `training_allowed=false`
- `stable_effect_allowed=false`
- `advisory_only=true`
- `affects_stable_trading=false`
- `contains_secret=false`
- `contains_live_order=false`
- `contains_order_intent=false`
- `qmt_related=false`

## 执行顺序

1. 人工确认数据来源和授权范围。
2. 将真实小样本 CSV 放入 `.local_research_outputs/aetfq3_lab/table_ml_real_small_samples/...`。
3. 人工填写对应 manifest。
4. 运行 sample intake checker，只验证 manifest。
5. intake 通过后，运行 dry validation orchestrator。
6. 生成 ignored dry validation report。
7. 使用 report reader 摘要检查 `status`、P0 / P1 和边界字段。
8. 人工决定是否继续 baseline 讨论；不得直接进入 Stable。

## P0 阻断项

- 无人工授权。
- 来源不清楚。
- `uses_stable_bundle=true` 但缺少 bundle 元数据。
- `sample_path_type=local_ignored` 但样本不在 `.local_research_outputs/aetfq3_lab/`。
- `training_allowed=true`。
- `stable_effect_allowed=true`。
- `affects_stable_trading=true`。
- `advisory_only=false`。
- 包含 secret、live order、OrderIntent 或 QMT 实盘信息。
- `feature_columns` 包含 future label 或 forbidden feature。
- 试图提交 ignored 样本、data、artifacts 或 `.local_research_outputs/`。

## P1 高风险

- 样本数量、时间范围或 sector 覆盖不完整。
- Stable bundle 边界不清楚。
- sector mapping 来源不清楚。
- label 生成口径不清楚。
- 缺失值 / 异常值处理未说明。
- E ranking 的 group split 规则未确认。

## Review Checklist 自检

1. 研究了什么：A/E 表格 ML 真实小样本来源确认与最小样本准备计划。
2. 数据来自哪里：本任务未读取真实样本；计划中的数据只能来自人工授权 local ignored 小样本或 Stable bundle readonly extract。
3. 是否来自 Stable bundle：当前未使用；未来若使用，必须 readonly 且记录 bundle path、commit 或 snapshot date。
4. 是否有未来函数：本任务不读取样本内容；计划明确 future label / forbidden feature 不得进入 feature columns。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：是，且本任务不生成真实 advisory 包。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用；该计划不建议进入 Stable。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议是什么：人工准备两个 local ignored manifest 模板，不填真实行级内容，先跑 intake checker dry smoke。
