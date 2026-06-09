本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Label / Outcome Design

## 任务定位

本设计只定义 public intraday 5分钟K supervised label / outcome 的 Lab-only 生成规范、manifest 扩展和 leakage checker 要求。当前 public intraday 5分钟K 已完成 no-label tensor validation，但仍不能进入 supervised smoke，因为缺真实 labels / future outcomes。

本任务不生成真实 label 文件，不读取更多真实行情，不训练模型，不运行 `torchrun`，不保存 checkpoint，不接 QMT，不生成 `OrderIntent`，不写 `output/`，不创建 `lab_advisory/`，不进入 Stable。

## 当前前置

- public 5m export: 432 rows
- no-label tensor validation: passed
- batch_size: 9
- time_steps: 48
- feature_count: 10
- no QMT
- no Stable bundle
- no labels

## Labels

以下字段只能作为 supervised target / label 使用，永远不能进入 `feature_columns`：

- `buy_now_label`
- `wait_pullback_label`
- `cancel_buy_label`
- `three_day_positive_label`

建议语义仅用于后续 Lab dry validation 设计，不构成交易建议：

- `buy_now_label`: 基于当前 bar 之后的收益 / 回撤 outcome，标记当前 bar 是否满足进入监督目标的候选条件。
- `wait_pullback_label`: 基于当前 bar 之后窗口内是否出现更优回落确认场景，标记是否应等待回撤。
- `cancel_buy_label`: 基于当前 bar 之后窗口内是否触发不利 outcome，标记是否取消候选买入。
- `three_day_positive_label`: 基于 T+3 或等价未来窗口收益是否为正，标记三日正收益 outcome。

### `three_day_positive_label` dry-run 公式

本公式只用于 Lab-only eligible-anchor label dry-run，不改变既有 `future_return_3d` 计算口径，不构成真实 label 文件生成授权，不构成 supervised training 授权，不构成 Stable 信号，不代表买入建议。

```text
three_day_positive_label =
  1 if future_return_3d > 0
  0 if future_return_3d <= 0
  null if future_return_3d is null or outcome_status_3d != available
```

约束：

- `three_day_positive_label` 只能从 `future_return_3d` 派生。
- `three_day_positive_label` 不能进入 `feature_columns`。
- 缺 future window 时 `three_day_positive_label` 必须为 null。
- `three_day_positive_label` 不能作为 Stable 信号。
- `three_day_positive_label` 不代表买入建议。
- `three_day_positive_label` 不授权 supervised training。

## Outcomes

以下字段只能作为 label 生成依据、target、outcome 或离线审计字段，永远不能进入 `feature_columns`：

- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `execution_return_to_close`
- `execution_return_to_next_open`
- `execution_drawdown_after_entry`
- `expected_3d_return`
- `expected_3d_drawdown`

`expected_3d_return` 和 `expected_3d_drawdown` 只允许由未来窗口 outcome 汇总而来，必须标记为 outcome-derived 字段，不允许伪装成当前 bar feature。

## 标签生成原则

- labels / outcomes 只能由当前 bar 之后的数据生成。
- labels / outcomes 永远不能进入 `feature_columns`。
- 当前 bar 的 feature 只能使用当前 bar 或之前数据。
- 对每个 `bar_index` 必须记录 `label_horizon`。
- 缺 future window 时必须记录 `label_status=insufficient_future_window`。
- 不得补假 label。
- 不得生成 synthetic labels 并标记为 real。
- 不得使用 QMT 成交回报作为 feature。
- 不得使用真实交易结果作为 feature。
- 不得把 label 方案解释为交易建议。

## Manifest 扩展

后续任何 supervised intraday manifest 必须新增或显式声明：

| 字段 | 要求 |
| --- | --- |
| `label_generated` | boolean；当前设计阶段为 false；真实生成后才可为 true。 |
| `label_source_kind` | enum；如 `human_exported_future_window`、`public_future_window`、`synthetic_mock`；synthetic 不得标记为 real。 |
| `label_horizon` | 每个 bar 或批次的 horizon 定义；必须覆盖 bar_index。 |
| `label_generation_method` | 文字说明或版本化方法 id；必须可审计。 |
| `label_columns` | label 字段列表；不得与 `feature_columns` 重叠。 |
| `outcome_columns` | outcome 字段列表；不得与 `feature_columns` 重叠。 |
| `label_status_column` | 建议固定为 `label_status`。 |
| `insufficient_future_window_policy` | 缺 future window 时必须标记并排除 supervised target，不得填假值。 |
| `feature_label_overlap_check` | 必须为 true，并输出 overlap 明细。 |
| `label_generation_authorized` | 必须由人工授权；未授权不得生成 real labels。 |
| `supervised_training_allowed` | 本阶段必须为 false。 |

## Leakage Checker 要求

leakage checker 必须阻断：

- label column in `feature_columns`
- future outcome in `feature_columns`
- current bar feature uses future bar
- label generated without sufficient future window
- synthetic label marked as real
- QMT fill / real trade result used as feature
- `supervised_training_allowed=true` without authorization
- `OrderIntent` generated
- Stable affected

## Boundary

- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model training
- no torchrun
- no checkpoint
- no model save
- no fake labels
- no synthetic labels marked as real
- no trading advice

## Next Steps

- 先实现 manifest-only leakage checker，并用 mock manifest 覆盖 P0 checklist。
- 若后续要生成 real labels，必须先提供 future window 数据、hash、source note、manifest 和人工授权。
- 生成 label 文件后仍需 intake-only / leakage-only 检查通过，才能申请 supervised dry validation。
- 不进入 Stable。
