# AETF Q3 Lab Table ML Data Contract

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 文档定位

本文定义 A `ML false downgrade` 与 E `同板块 ETF 内部排序` 共用的表格 ML 数据合同。它只定义 schema、字段、标签、时间切分和 advisory-only 边界，不读取真实行情，不训练模型，不生成 advisory 包。

## 通用主键

基础样本键：

- `trade_date`
- `etf_code`
- `sector`
- `model_version`
- `feature_version`

A false downgrade 额外键：

- `v2_action`
- `ml_action`

E sector internal ranking 额外键：

- `rank_scope = sector_internal`
- `ranking_group_id = trade_date + sector`

建议所有训练或评估样本都保留：

```text
sample_id = task_name + trade_date + sector + etf_code + model_version + feature_version
```

## 通用字段命名

命名规则：

- 日期字段使用 `_date` 后缀，例如 `trade_date`。
- 未来标签字段使用 `future_` 前缀。
- 回撤标签字段使用 `max_drawdown_` 前缀。
- 当日可见特征使用明确来源前缀，例如 `v2_`、`ml_`、`sector_`、`etf_`、`risk_`。
- 版本字段必须使用 `model_version`、`feature_version`。
- 排名任务 group 字段使用 `ranking_group_id`。
- shadow 输出字段使用 `shadow_` 前缀。

## 必须字段

所有任务必须包含：

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

A false downgrade 必须额外包含：

- `v2_action`
- `ml_action`
- `v2_score`
- `ml_score`

E sector internal ranking 必须额外包含：

- `rank_scope`
- `ranking_group_id`
- `sector_member_count`

## 可选字段

可选但建议记录：

- `risk_flags`
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
- `p_good`
- `p_bad`
- `review_queue_reason`

## 标签字段

A false downgrade 标签：

- `false_downgrade_1d`
- `false_downgrade_3d`
- `false_downgrade_lock3`
- `true_downgrade`
- `neutral_downgrade`
- `future_return_1d`
- `future_return_3d`
- `max_drawdown_1d`
- `max_drawdown_3d`

E sector internal ranking 标签：

- `best_in_sector_1d`
- `best_in_sector_3d`
- `top_quantile_in_sector_3d`
- `avoid_in_sector`
- `pairwise_outperform_label`
- `expected_3d_return`
- `future_return_1d`
- `future_return_3d`
- `max_drawdown_1d`
- `max_drawdown_3d`

## 禁止进入 feature 的字段

以下字段只能作为 label、评估指标或报告字段，不得进入当日 feature：

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
- 任何 T+1 / T+3 价格、收益、排名、回撤。

## 时间切分规则

正式评估必须使用时间序列切分：

- chronological split。
- walk-forward。
- `train <= date, validate > date`。
- 不允许随机打乱时间序列作为正式评估。
- 每次训练和评估必须记录 `split_cutoff_date`。
- 每个报告必须记录训练起止日期、验证起止日期、样本数量。

## Group Split 规则

A false downgrade：

- 同一 `trade_date` 样本建议留在同一个 split。
- 若使用 sector 级聚合特征，确保聚合只来自当日或过去。

E sector internal ranking：

- `ranking_group_id = trade_date + sector` 是不可拆分 group。
- 同一个 `ranking_group_id` 不得同时出现在 train 和 validate。
- Pairwise 样本必须继承原始 `ranking_group_id`。
- 不允许把同一天同板块内部分 ETF 放 train、部分 ETF 放 validate。

## Future Leakage 检查

每次研究任务必须回答：

- `future_return` 是否只作为 label。
- T+1 / T+3 收益、回撤、排名是否被排除出 feature。
- rolling 特征是否只使用 `trade_date` 当日或之前数据。
- `feature_cutoff_time` 是否早于 label window。
- `model_version` / `feature_version` 是否记录。
- E ranking 是否按 `ranking_group_id` 做 group split。
- 是否有人工确认无未来函数。

建议在样本生成代码中加入列名前缀检查：

```text
feature columns must not start with future_
feature columns must not start with max_drawdown_
feature columns must not include known label names
```

## 真实样本接入前置门禁

任何真实小样本 CSV 进入 dry validation 或 schema validator 前，必须先提供 Lab-only sample intake manifest，并通过 `tools/lab/table_ml_sample_intake_checker.py`。manifest 规范见 `docs/research/aetfq3_table_ml_sample_intake.md`。

sample intake checker 只验证数据来源、授权范围、Stable bundle readonly 边界、future leakage 声明、训练禁止和 advisory-only 边界，不读取真实 CSV 行级内容，不训练模型，不生成 advisory 包。

## 输出文件命名

未来 A false downgrade 可输出，但本任务不生成：

- `ml_false_downgrade_cases.csv`
- `ml_false_downgrade_pattern_report.md`
- `ml_false_downgrade_pattern_report.json`
- `false_downgrade_model_report.md`
- `false_downgrade_model_report.json`
- `false_downgrade_shadow_scores.csv`

未来 E sector internal ranking 可输出，但本任务不生成：

- `sector_internal_etf_ranking_research.csv`
- `sector_internal_etf_ranking_report.md`
- `sector_internal_etf_ranking_report.json`
- `sector_internal_shadow_ranking.csv`
- `sector_internal_model_diagnostics.json`

输出位置：

- 大 CSV、训练样本、模型日志、明细诊断：`.local_research_outputs/aetfq3_lab/`
- 小型方法摘要、schema、人工复核结论：`docs/research/`
- 禁止输出到：`output/`、Stable `output/`、Stable `runtime/`、Stable order intent 目录、`data/cache/`、QMT 实盘目录。

## Advisory-only 边界

所有 A / E 表格 ML 研究输出默认：

```text
affects_stable_trading: false
advisory_only: true
requires_human_review: true
recommended_for_stable: false
```

禁止动作：

- `do_not_modify_final_buy_action`
- `do_not_modify_target_weight`
- `do_not_generate_order_intent`
- `do_not_bypass_riskgate`
- `do_not_auto_trade`
- 不直接改 Stable。
- 不把模型输出直接接入 Stable。
- 不把研究输出当正式交易计划。

若未来建议进入 Stable，必须提供：

1. 最小合并方案。
2. 风险点。
3. Stable 侧人工审批。
4. `RiskGate` 检查点。
5. 回滚方案。
6. 不允许直接提交到 Stable 的明确声明。

## Review Checklist 映射

任务结束必须回答：

1. 研究了什么：A false downgrade / E sector internal ranking 的数据口径或模型实验。
2. 数据来自哪里：列明 V2 core、ML_SIM、ETF 日线、sector mapping、本地 ignored 数据或其他来源。
3. 是否来自 Stable bundle：必须明确 true / false；若 true，说明只读来源和 bundle 版本。
4. 是否有未来函数：必须说明 label / feature 隔离、时间切分和 group split。
5. 是否影响 Stable 正式交易：默认 false。
6. 是否只读 advisory：默认 true。
7. 是否建议进入 Stable：默认 false。
8. 如果建议进入 Stable，最小合并方案是什么：默认不适用；若 true，必须补齐最小合并方案、风险点、审批、RiskGate、回滚。
9. 不允许直接提交到 Stable：必须确认。
10. 下一步建议是什么：给出下一张 Lab-only 任务卡。

## 本设计任务结论等级

- A false downgrade 数据口径：保留结论。
- E sector internal ranking 数据口径：保留结论。
- 是否进入 Stable：待人工复核；当前不得进入 Stable。
- 是否可直接训练正式模型：阻塞结论；需要另行授权和数据来源确认。
