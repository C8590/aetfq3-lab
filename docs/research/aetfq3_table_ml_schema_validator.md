# AETF Q3 Lab Table ML Schema Validator

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 文档定位

`tools/lab/table_ml_schema_validator.py` 是 Lab-only 表格 ML 样本 schema validator。它只验证字段合同、主键、标签、feature leakage、时间切分和 group split 规则，不是训练器，不是 Stable 接口，不读取真实行情，不生成 advisory 包。

该工具面向 A `ML false downgrade` 与 E `同板块 ETF 内部排序` 的后续研究准备阶段，用于在真实数据接入前先固定数据合同防线。

## 支持的 sample_type

- `false_downgrade`
- `sector_internal_ranking`

## 必需字段

### false_downgrade

- `trade_date`
- `etf_code`
- `etf_name`
- `sector`
- `v2_action`
- `ml_action`
- `model_version`
- `feature_version`
- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `false_downgrade_1d`
- `false_downgrade_3d`
- `false_downgrade_lock3`
- `true_downgrade`
- `neutral_downgrade`

### sector_internal_ranking

- `trade_date`
- `sector`
- `etf_code`
- `etf_name`
- `ranking_group_id`
- `model_version`
- `feature_version`
- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `best_in_sector_1d`
- `best_in_sector_3d`
- `top_quantile_in_sector_3d`
- `avoid_in_sector`
- `pairwise_outperform_label`

## 主键规则

`false_downgrade` 主键必须唯一：

```text
trade_date + etf_code + model_version + feature_version
```

`sector_internal_ranking` 主键必须唯一：

```text
trade_date + sector + etf_code + model_version + feature_version
```

`sector_internal_ranking` 还要求：

```text
ranking_group_id = trade_date + sector
```

工具支持以下人工可读 group id 格式：

- `trade_date_sector`
- `trade_date|sector`
- `trade_date-sector`
- `trade_date:sector`

## 标签规则

`false_downgrade` 标签字段：

- `false_downgrade_1d`
- `false_downgrade_3d`
- `false_downgrade_lock3`
- `true_downgrade`
- `neutral_downgrade`

`sector_internal_ranking` 标签字段：

- `best_in_sector_1d`
- `best_in_sector_3d`
- `top_quantile_in_sector_3d`
- `avoid_in_sector`
- `pairwise_outperform_label`

标签必须是 `0` / `1` 或 boolean 文本。单行样本的标签不应全部缺失。

## Feature Leakage 检查

CLI 可通过 `--feature-columns` 传入本次计划用于训练的特征列：

```powershell
.\.venv\Scripts\python.exe tools/lab/table_ml_schema_validator.py `
  --sample-type false_downgrade `
  --input tests/fixtures/aetfq3_lab/mock_false_downgrade_samples.csv `
  --feature-columns v2_score,ml_score,sector_rank
```

以下字段不得出现在 feature columns 中：

- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- 所有 label 字段。
- `best_in_sector_*`
- `top_quantile_*`
- `pairwise_outperform_label`
- `true_future_*`
- `future_best_etf_code`
- `future_sector_rank`
- `future_etf_rank`

如果发现 forbidden feature，validator 必须输出 `P0` 并返回非 0 exit code。

## 时间切分 / Group Split 检查

validator 提供 helper：

- `validate_chronological_split(train_end_date, valid_start_date)`
- `validate_no_group_split_leakage(train_rows, validation_rows)`

规则：

- chronological split 不允许随机 shuffle 作为正式评估。
- `train_end_date < valid_start_date`。
- `sector_internal_ranking` 中同一个 `ranking_group_id` 不得同时出现在 train 和 validation。
- 同一个 `ranking_group_id` 至少应有 2 个 ETF；单成员 group 作为 P1 warning。

## CLI 用法

验证 A false downgrade mock：

```powershell
.\.venv\Scripts\python.exe tools/lab/table_ml_schema_validator.py `
  --sample-type false_downgrade `
  --input tests/fixtures/aetfq3_lab/mock_false_downgrade_samples.csv
```

验证 E sector internal ranking mock：

```powershell
.\.venv\Scripts\python.exe tools/lab/table_ml_schema_validator.py `
  --sample-type sector_internal_ranking `
  --input tests/fixtures/aetfq3_lab/mock_sector_internal_ranking_samples.csv
```

验证 feature leakage：

```powershell
.\.venv\Scripts\python.exe tools/lab/table_ml_schema_validator.py `
  --sample-type sector_internal_ranking `
  --input tests/fixtures/aetfq3_lab/mock_sector_internal_ranking_samples.csv `
  --feature-columns ml_score,etf_momentum,future_return_3d
```

该命令应失败并输出 P0，因为 `future_return_3d` 是 label / future outcome。

## Mock 数据说明

Mock fixture 位于：

- `tests/fixtures/aetfq3_lab/mock_false_downgrade_samples.csv`
- `tests/fixtures/aetfq3_lab/mock_sector_internal_ranking_samples.csv`

这些样本全部为人工构造的小样本，不来自真实行情，不来自 `data/etf_daily.csv`，不来自 Stable output。

`mock_false_downgrade_samples.csv` 包含：

- 至少 8 行。
- 至少 2 个日期。
- 至少 2 个 ETF。
- `PROBE` / `BUY` 被 `ML_DOWNGRADED` 后 `future_return_3d > 0` 的 false downgrade 示例。
- true downgrade 示例。
- neutral downgrade 示例。

`mock_sector_internal_ranking_samples.csv` 包含：

- 至少 8 行。
- 至少 2 个日期。
- 至少 2 个 sector。
- 每个 `ranking_group_id` 至少 2 个 ETF。
- `best_in_sector_3d` / `top_quantile_in_sector_3d` 示例。

## 不允许事项

- 不读取真实行情。
- 不读取 `data/etf_daily.csv`。
- 不训练模型。
- 不运行 LightGBM / CatBoost / XGBoost 真实样本训练。
- 不生成 `OrderIntent`。
- 不接 QMT。
- 不写 Stable。
- 不写 `output/`。
- 不创建 `lab_advisory/`。
- 不影响 `final_buy_action`。
- 不修改 `target_weight`。
- 不修改 BUY / PROBE 阈值。
- 不替代 Stable entry。

## 后续真实数据接入前置条件

接入真实数据前必须完成：

- 人工确认数据来源。
- 明确是否来自 Stable bundle。
- 明确日期范围。
- 明确样本数量。
- 明确缺失值和异常值处理。
- 完成未来函数检查。
- 确认 feature columns 不包含 future outcome 或 label。
- 确认 chronological split / walk-forward 规则。
- E ranking 必须确认 `ranking_group_id` group split 无泄漏。
- Review Checklist 通过。

## Review Checklist 自检

1. 研究了什么：实现 Lab-only 表格 ML schema validator，并用人工 mock 小样本验证字段合同、主键、时间切分、group split 和 future leakage 防线。
2. 数据来自哪里：仅来自人工构造 mock CSV 和已提交的 Lab 设计文档。
3. 是否来自 Stable bundle：否。
4. 是否有未来函数：否；validator 明确禁止 future outcome 和 label 进入 feature columns。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：是，工具只做 schema 校验，不输出正式交易建议。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用。本工具不建议进入 Stable。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议是什么：在人工确认数据来源后，做 Lab-only 真实样本抽样校验任务，仍不训练模型、不接 Stable、不接 QMT。
