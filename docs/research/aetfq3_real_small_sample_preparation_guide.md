# AETF Q3 Lab Real Small Sample Preparation Guide

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## 目标

本文指导人工准备 A `false_downgrade` 与 E `sector_internal_ranking` 的真实小样本 CSV，用于后续 Lab-only sample intake gate 和 schema validator gate。本文不是样本生成器，不读取真实行情，不读取 Stable output，不训练模型，不生成 advisory 包，不生成 `OrderIntent`，不接 QMT，不修改 Stable。

目标 CSV 放在 ignored 本地目录：

- `.local_research_outputs/aetfq3_lab/table_ml_dry_validation_inputs/false_downgrade_real_small_sample.csv`
- `.local_research_outputs/aetfq3_lab/table_ml_dry_validation_inputs/sector_internal_ranking_real_small_sample.csv`

## false_downgrade 小样本怎么做

`false_downgrade` 小样本用于观察 A 方向：V2 / Stable 侧原本可能给出 `BUY` 或 `PROBE`，但 ML 侧降级后，后续收益显示该降级可能是误杀的案例。人工制样时应选择少量可审计样本，覆盖：

- 被 ML 降级但 T+1 / T+3 表现为正的疑似 false downgrade。
- 被 ML 降级且后续表现确实较差的 true downgrade。
- 表现不强不弱、难以归因的 neutral downgrade。
- 至少两个交易日，尽量覆盖多个 ETF 和多个 sector。

CSV 行级内容应只来自人工已授权来源。每行保留当日可见字段、版本字段和后验标签字段；后验标签只能作为 label 或评估字段，不得进入 `feature_columns`。

## sector_internal_ranking 小样本怎么做

`sector_internal_ranking` 小样本用于观察 E 方向：同一交易日、同一 sector 内，多个 ETF 的相对表现和排序标签。人工制样时应按 `trade_date + sector` 形成 `ranking_group_id`，每个 group 建议至少包含 2 个 ETF，覆盖：

- 同板块内 T+1 / T+3 最强 ETF。
- 同板块内 top quantile ETF。
- 同板块内应避免 ETF。
- 可用于 pairwise 对比的同组 ETF。
- 至少两个交易日，尽量覆盖两个或以上 sector。

`ranking_group_id` 必须能由 `trade_date` 和 `sector` 人工复核，可使用 `trade_date_sector`、`trade_date|sector`、`trade_date-sector` 或 `trade_date:sector` 格式。

## 必须字段

两类样本通用必须字段：

- `trade_date`
- `etf_code`
- `etf_name`
- `sector`
- `model_version`
- `feature_version`
- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`

`false_downgrade` 额外必须字段：

- `v2_action`
- `ml_action`
- `false_downgrade_1d`
- `false_downgrade_3d`
- `false_downgrade_lock3`
- `true_downgrade`
- `neutral_downgrade`

`sector_internal_ranking` 额外必须字段：

- `ranking_group_id`
- `best_in_sector_1d`
- `best_in_sector_3d`
- `top_quantile_in_sector_3d`
- `avoid_in_sector`
- `pairwise_outperform_label`

建议同时保留数据合同中的审计字段：

- `data_source_name`
- `data_source_version`
- `feature_cutoff_time`
- `label_window`
- `split_name`
- `split_cutoff_date`
- `uses_stable_bundle`
- `future_leakage_checked`

## feature_columns 填写规则

`feature_columns` 写在 manifest 中，不要求 CSV 只包含 feature。人工填写时只列入 dry validation 计划使用的当日可见特征，例如：

- `v2_score`
- `ml_score`
- `sector_rank`
- `sector_momentum`
- `sector_breadth`
- `sector_overheat`
- `etf_momentum`
- `etf_acceleration`
- `etf_volatility`
- `etf_liquidity`
- `etf_amount`
- `previous_day_gap`
- `position_in_sector`
- `distance_to_sector_leader`
- `crowding_score`
- `concentration_score`
- `risk_flags`

特征必须满足：

- 只使用 `trade_date` 当日或之前可见信息。
- `feature_cutoff_time` 早于 label window。
- rolling / sector 聚合特征只用当日或历史数据。
- E ranking 的 group 特征不得混入未来同组表现。
- 列名含义可人工解释，并能对应 CSV 表头。

## future label 禁止进入 feature

以下字段只能作为 label、评估字段或报告字段，禁止进入 manifest 的 `feature_columns`：

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
- `future_best_etf_code`
- `future_sector_rank`
- `future_etf_rank`
- 任何 T+1 / T+3 价格、收益、排名、回撤或后验表现字段。

如果 `feature_columns` 包含这些字段，sample intake checker 或 schema validator 应按 P0 阻断。

## Stable bundle 来源怎么记录

如果样本来自 Stable bundle 的只读抽取，manifest 中 `uses_stable_bundle` 必须为 `true`，并且 `source_kind` 建议设为 `stable_bundle_extract`。这只表示数据来源，不授权 Lab 修改 Stable，不授权读取 Stable output，不授权生成正式交易动作，不授权绕过 `RiskGate`。

如果样本不来自 Stable bundle，manifest 中 `uses_stable_bundle` 填 `false`，`stable_bundle_path`、`stable_bundle_commit`、`stable_bundle_snapshot_date` 可填空字符串或 `null`，但仍需说明人工授权来源。

## uses_stable_bundle=true 时必须填什么

`uses_stable_bundle=true` 时，manifest 必须补齐：

- `stable_bundle_path`: Stable bundle 的只读路径或人工记录路径。
- `stable_bundle_commit`: bundle 对应 commit；或
- `stable_bundle_snapshot_date`: bundle snapshot date。
- `authorization_scope`: 明确只允许 `schema dry validation only; no training; no Stable effect` 或更窄范围。

`stable_bundle_commit` 与 `stable_bundle_snapshot_date` 至少填一个。不得把 Stable bundle 来源解释为可回写 Stable 或可改变 Stable 正式交易。

## 日期范围怎么填

manifest 的日期范围来自 CSV 的 `trade_date`：

- `data_time_start`: CSV 中最早 `trade_date`。
- `data_time_end`: CSV 中最晚 `trade_date`。

日期建议使用 `YYYY-MM-DD`。如果样本跨多个市场日，必须保持原始交易日顺序，后续 schema / dry validation 不应使用随机打乱作为正式评估说明。

## row_count / symbol_count / sector_count 怎么填

这些字段应由人工对 CSV 做轻量核对后填写：

- `row_count`: CSV 数据行数，不含表头。
- `symbol_count`: 去重后的 `etf_code` 数量。
- `sector_count`: 去重后的 `sector` 数量。

允许在 intake-only 任务中只为统计行数和表头读取 CSV，不得做统计建模、特征选择、训练或收益分析。

## 人工授权字段怎么填

manifest 中必须填写：

- `human_authorized`: `true`
- `authorized_by`: 授权人、操作者或人工审核记录标识。
- `authorization_scope`: `schema dry validation only; no training; no Stable effect`
- `has_future_leakage_check`: `true`
- `review_checklist_passed`: `true`
- `training_allowed`: `false`
- `stable_effect_allowed`: `false`
- `advisory_only`: `true`
- `affects_stable_trading`: `false`
- `contains_secret`: `false`
- `contains_live_order`: `false`
- `contains_order_intent`: `false`
- `qmt_related`: `false`

人工授权只允许样本进入 Lab-only dry validation 门禁，不允许训练，不允许接 Stable，不允许接 QMT，不允许生成 advisory 包。

## 制样后怎么跑 intake checker

先基于本地模板另存为真实 manifest：

- `.local_research_outputs/aetfq3_lab/table_ml_dry_validation_inputs/false_downgrade_real_sample_manifest.json`
- `.local_research_outputs/aetfq3_lab/table_ml_dry_validation_inputs/sector_internal_ranking_real_sample_manifest.json`

真实 manifest 中 `path_may_not_exist_for_template` 应删除或改为 `false`，并确认 `sample_path` 指向已存在 CSV。

运行 sample intake checker：

```powershell
python tools/lab/table_ml_sample_intake_checker.py --manifest .local_research_outputs\aetfq3_lab\table_ml_dry_validation_inputs\false_downgrade_real_sample_manifest.json
python tools/lab/table_ml_sample_intake_checker.py --manifest .local_research_outputs\aetfq3_lab\table_ml_dry_validation_inputs\sector_internal_ranking_real_sample_manifest.json
```

通过时应输出：

```text
OK sample_intake_manifest_valid=true
```

如果输出 `FAILED`、`P0` 或 `P1`，先修正 manifest 或人工复核记录，不要进入 schema validator。

## intake 通过后怎么跑 schema validator

只有 intake checker 通过后，才允许进入 schema validator。schema validator 只校验字段合同、主键、标签、feature leakage、时间切分和 group split，不训练模型。

示例命令：

```powershell
python tools/lab/table_ml_schema_validator.py --sample-type false_downgrade --input .local_research_outputs\aetfq3_lab\table_ml_dry_validation_inputs\false_downgrade_real_small_sample.csv --feature-columns v2_score,ml_score,sector_rank,etf_momentum
python tools/lab/table_ml_schema_validator.py --sample-type sector_internal_ranking --input .local_research_outputs\aetfq3_lab\table_ml_dry_validation_inputs\sector_internal_ranking_real_small_sample.csv --feature-columns ml_score,sector_rank,etf_momentum,sector_breadth
```

`--feature-columns` 必须与 manifest 中的 `feature_columns` 保持一致，并再次确认不包含 future label 或 forbidden feature。

## 禁止事项

- 不读取 `data/etf_daily.csv`。
- 不读取 Stable output。
- 不读取 ML_SIM。
- 不生成真实 CSV。
- 不训练模型。
- 不运行 LightGBM / CatBoost / XGBoost 真实训练。
- 不写 `output/`。
- 不创建 `lab_advisory/`。
- 不生成 advisory 包。
- 不接 Stable。
- 不修改 Stable。
- 不接 QMT。
- 不生成 `OrderIntent`。
- 不影响 `final_buy_action`。
- 不修改 `target_weight`。
- 不修改 BUY / PROBE 阈值。
- 不把研究输出当正式交易计划。
- 不把模型输出直接接入 Stable。

## 常见错误

- 把 `future_return_1d`、`future_return_3d`、`max_drawdown_3d` 填进 `feature_columns`。
- 把 `best_in_sector_3d`、`pairwise_outperform_label` 或 `false_downgrade_3d` 当成可用特征。
- `uses_stable_bundle=true` 但缺少 `stable_bundle_path`、`stable_bundle_commit` 或 `stable_bundle_snapshot_date`。
- `human_authorized=false` 或授权范围没有写明 `no training; no Stable effect`。
- `path_may_not_exist_for_template=true` 没有在真实 manifest 中关闭。
- `sample_path` 指向模板 CSV，而不是人工准备的真实小样本 CSV。
- `row_count` 把表头也算进去。
- `symbol_count` 或 `sector_count` 没有按去重统计。
- E ranking 的 `ranking_group_id` 与 `trade_date + sector` 不一致。
- 同一个 E ranking group 只有一个 ETF，导致排序样本不可比。
- 把 intake checker 通过误解为可以训练或可以影响 Stable。
- intake 未通过就运行 schema validator。
