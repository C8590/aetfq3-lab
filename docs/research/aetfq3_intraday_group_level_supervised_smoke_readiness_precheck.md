本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Lab Group-Level Supervised Smoke Readiness Precheck

## 任务定位

本文件定义 F public intraday group-level supervised smoke readiness precheck。它只检查 group-level 样本是否具备后续 no-save diagnostic smoke 的最低条件，不运行 no-save smoke，不训练模型，不调参，不保存模型，不进入 Stable。

group-level 样本用于缓解 bar-level 5m 样本中 48-bar repeated label structure 对 supervised smoke 的干扰。`anchor_close_last_bar` 是 end-of-anchor-day diagnostic policy，不是盘中实时交易决策。

## 输入

- `intraday_group_level_samples.csv`
- `intraday_group_level_manifest.json`
- `intraday_group_level_report.json`

## 检查项

- Manifest leakage checker passed。
- `group_level_sample=true`。
- `group_key=["trade_date","etf_code"]` 或 `["anchor_date","etf_code"]`。
- `group_label_policy=anchor_close_last_bar`。
- `intraday_live_decision_ready=false`。
- target `three_day_positive_label` 存在且不在 `feature_columns`。
- `future_return_1d`、`future_return_3d`、`max_drawdown_3d` 不在 `feature_columns`。
- 任何 `future_*` / `*_label` 不在 `feature_columns`。
- group_count >= 200。
- anchor_count >= 20。
- ETF count >= 3。
- class_count = 2。
- min_class_count >= 50。
- label_null_count = 0。
- time-based split feasible。
- train / valid split 均至少包含两个 class。
- boundary flags 全部 false。

## Split 规则

只做 split feasibility，不训练。

默认按 anchor date 排序：

- 前 70% anchor 作为 train，后 30% anchor 作为 valid。
- 如果 valid 单类别，尝试 60/40。
- 如果仍不满足，blocked。

报告必须包含：

- selected_split_policy
- train_anchor_dates
- valid_anchor_dates
- train_group_count
- valid_group_count
- train_label_0_count
- train_label_1_count
- valid_label_0_count
- valid_label_1_count
- split_feasible

## Inconsistent Label Review

必须记录：

- inconsistent_label_group_count
- inconsistent_label_group_rate
- group_label_policy=anchor_close_last_bar

若 inconsistent_label_group_rate 过高，输出 P1 warning：

```text
P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED
```

该 warning 不直接 P0，但不得把 group-level 样本解释为盘中实时决策样本。

## Readiness Decision

可能输出：

- `GROUP_LEVEL_SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED`
- `GROUP_LEVEL_SUPERVISED_SMOKE_READINESS_PASSED_WITH_LABEL_INCONSISTENCY_REVIEW_REQUIRED`
- `BLOCKED_GROUP_LEVEL_SINGLE_CLASS_LABEL`
- `BLOCKED_GROUP_LEVEL_INSUFFICIENT_GROUPS`
- `BLOCKED_GROUP_LEVEL_SPLIT_NOT_CLASS_DIVERSE`
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

通过后只允许申请 group-level no-save diagnostic smoke。不得自动训练、不得接 QMT、不得生成 OrderIntent、不得进入 Stable、不得构成交易建议。
