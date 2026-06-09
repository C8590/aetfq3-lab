本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Lab Intraday Group-Level Sample Dry Run

## 任务定位

本文件定义 F public intraday group-level sample dry-run。它基于 majority-class collapse diagnostic 的结论，将 bar-level 5m 样本按 anchor date 与 ETF 聚合，减少 48 根 bar 共享或近似共享同一 label 的重复标签结构风险。

这是 Lab-only group-level 样本设计、生成、leakage check、class-balance precheck 与 readiness precheck；不是训练，不运行 no-save supervised smoke，不调参，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。

## Group-Level Design

group key：

```text
trade_date + etf_code
```

如果输入字段存在 `anchor_date`，工具优先使用 `anchor_date`；否则使用 `trade_date`。每个 group 表示：

```text
one ETF on one eligible anchor date
```

报告必须记录：

- 原始 bar-level row_count
- group_count
- bars_per_group min / median / max
- label consistency per group
- inconsistent_label_group_count
- single_label_group_count

group_count 明显小于 bar-level row_count 是预期行为，不是样本丢失。

## Label Policy

默认 label 聚合策略：

```text
group_label_policy = anchor_close_last_bar
intraday_live_decision_ready = false
```

对每个 anchor date + ETF group，选择当天最后一根 5m bar，并使用该最后一根 bar 的：

- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `three_day_positive_label`

理由：

- features 使用 anchor 当天完整 5m 信息。
- label 使用 anchor close 后的 T+1 / T+3 outcome。
- 这是 end-of-anchor-day diagnostic 样本。
- 不是盘中实时交易样本。
- 不是交易建议。

不得使用随意 majority vote。

## Feature Aggregation

允许的 group-level features 仅来自 anchor 当天 5m bar 内可得信息：

- `open_first`
- `high_max`
- `low_min`
- `close_last`
- `volume_sum`
- `amount_sum`
- `vwap_day`
- `day_return`
- `high_low_range`
- `close_to_vwap`
- `intraday_return_mean`
- `intraday_return_std`
- `distance_to_vwap_mean`
- `distance_to_vwap_last`
- `volume_first_half_sum`
- `volume_second_half_sum`
- `amount_first_half_sum`
- `amount_second_half_sum`

禁止进入 `feature_columns`：

- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `three_day_positive_label`
- 任何 `future_*`
- 任何 `*_label`
- 任何 execution/outcome 字段
- 任何 QMT / account / order / fill 字段

## 输出

只允许写入：

```text
.local_research_outputs/aetfq3_lab/intraday_group_level_sample_dryrun/
```

输出文件：

- `intraday_group_level_samples.csv`
- `intraday_group_level_manifest.json`
- `intraday_group_level_report.md`
- `intraday_group_level_report.json`
- `class_balance_precheck.json`
- `readiness_decision.json`

## Readiness Decision

可能输出：

- `GROUP_LEVEL_SAMPLE_DRY_RUN_PASSED_REVIEW_REQUIRED`
- `GROUP_LEVEL_SAMPLE_DRY_RUN_PASSED_CLASS_DIVERSE_REVIEW_REQUIRED`
- `BLOCKED_GROUP_LEVEL_SINGLE_CLASS_LABEL`
- `BLOCKED_GROUP_LEVEL_INSUFFICIENT_GROUPS`
- `BLOCKED_GROUP_LABEL_INCONSISTENCY`
- `BLOCKED_MANIFEST_LEAKAGE_P0`
- `BLOCKED_BOUNDARY_FLAG`

即使 passed，也必须保留：

- `training_allowed=false`
- `stable_allowed=false`
- `qmt_allowed=false`
- `order_intent_allowed=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`

## 后续边界

通过后只能进入：

- human review
- group-level no-save diagnostic smoke 申请

不得自动进入 formal training、Stable promotion、QMT、OrderIntent、advisory 包或交易结论。
