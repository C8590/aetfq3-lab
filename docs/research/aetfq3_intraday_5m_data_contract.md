本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday 5m Data Contract

## 任务定位

本合同定义 F 方向 5分钟K 执行研究的候选样本字段、feature 边界、label / outcome 边界和泄漏禁止项。它只用于 Lab-only 方案设计，不读取真实 5分钟K、tick 或盘口数据，不训练模型，不生成 advisory 包，不进入 Stable。

PyTorch 或其他模型在本方向中只研究盘中执行时点，不负责选择 ETF，不修改 `final_buy_action`、`target_weight` 或 BUY / PROBE 阈值。

## 样本粒度

- 一行代表一个 ETF 在一个交易日内的一个 5分钟 bar。
- 所有可用 feature 必须来自当前 bar 或当前 bar 之前的信息。
- T+1 / T+3 / 成交后收益回撤只能作为 label 或 outcome。
- 不允许用未来 5分钟K 信息推断当前 bar 决策。
- 不允许跨日泄漏；跨日上下文只能来自已完成交易日的显式字段。
- 不允许把真实成交回报、后验收益或后验回撤作为训练 feature。

## 字段分组

### 基础主键

| 字段 | 类型 | feature 允许 | 说明 |
| --- | --- | --- | --- |
| `trade_date` | date | 否 | 交易日，用于 split / group，不进 feature。 |
| `datetime` | datetime | 否 | 5分钟 bar 时间戳，用于排序和审计。 |
| `etf_code` | string | 否 | ETF 代码，不直接进 feature。 |
| `etf_name` | string | 否 | ETF 名称，不直接进 feature。 |
| `sector` | string | 否 | 板块名称，用于 group / context，不直接进 feature。 |
| `bar_index` | integer | 是 | 当日 bar 序号，只代表已发生时间位置。 |
| `session_phase` | enum | 是 | `open_range` / `midday` / `afternoon` / `close_window`。 |

### OHLCV

| 字段 | 类型 | feature 允许 | 说明 |
| --- | --- | --- | --- |
| `open` | number | 是 | 当前 5分钟 bar 开盘价。 |
| `high` | number | 是 | 当前 5分钟 bar 最高价。 |
| `low` | number | 是 | 当前 5分钟 bar 最低价。 |
| `close` | number | 是 | 当前 5分钟 bar 收盘价。 |
| `volume` | number | 是 | 当前 5分钟 bar 成交量。 |
| `amount` | number | 是 | 当前 5分钟 bar 成交额。 |
| `vwap` | number | 是 | 截至当前 bar 可计算的 VWAP。 |

### 前日 / 开盘上下文

| 字段 | 类型 | feature 允许 | 说明 |
| --- | --- | --- | --- |
| `prev_close` | number | 是 | 前一交易日收盘价。 |
| `open_price` | number | 是 | 当日开盘价。 |
| `open_gap_pct` | number | 是 | 相对前收的开盘跳空幅度。 |
| `open_range_high` | number | 是 | 已确认开盘区间高点。 |
| `open_range_low` | number | 是 | 已确认开盘区间低点。 |
| `open_range_return` | number | 是 | 已确认开盘区间收益。 |

### 盘中特征

| 字段 | 类型 | feature 允许 | 说明 |
| --- | --- | --- | --- |
| `intraday_return` | number | 是 | 当前 bar 相对上一可比价格的盘中收益。 |
| `return_from_open` | number | 是 | 当前 bar 相对开盘价收益。 |
| `drawdown_from_open` | number | 是 | 截至当前 bar 相对开盘后的回撤。 |
| `high_from_open` | number | 是 | 截至当前 bar 相对开盘后的最高涨幅。 |
| `pullback_to_vwap` | number | 是 | 当前 bar 回落接近 VWAP 的幅度。 |
| `distance_to_vwap` | number | 是 | 当前价格与 VWAP 的距离。 |
| `break_open_low` | boolean | 是 | 当前或此前 bar 是否跌破开盘区间低点。 |
| `reclaim_vwap` | boolean | 是 | 当前或此前 bar 是否重新站上 VWAP。 |
| `volume_ratio_5m` | number | 是 | 当前 5分钟成交量相对历史或当日已知基准。 |
| `amount_ratio_5m` | number | 是 | 当前 5分钟成交额相对历史或当日已知基准。 |
| `volatility_5m` | number | 是 | 当前 5分钟波动率。 |
| `volatility_15m` | number | 是 | 截至当前 bar 的 15分钟窗口波动率。 |

### 板块上下文

| 字段 | 类型 | feature 允许 | 说明 |
| --- | --- | --- | --- |
| `sector_top1_flag` | boolean | 是 | 当前候选所属板块是否为已知 top1。 |
| `sector_rank_at_open` | integer | 是 | 开盘阶段板块排名。 |
| `sector_momentum_lag1` | number | 是 | 板块动量滞后一档。 |
| `sector_acceleration_lag1` | number | 是 | 板块加速度滞后一档。 |
| `sector_breadth_lag1` | number | 是 | 板块广度滞后一档。 |
| `sector_overheat_lag1` | number | 是 | 板块过热滞后一档。 |
| `sector_crowding_lag1` | number | 是 | 板块拥挤度滞后一档。 |

### 候选上下文

| 字段 | 类型 | feature 允许 | 说明 |
| --- | --- | --- | --- |
| `source_candidate_type` | enum | 是 | 候选来源类型，例如 E ranking / A reconstructed。 |
| `v2_action_context` | enum | 是 | 只读历史上下文，不授权 Stable 行为改变。 |
| `lab_research_context` | string | 是 | Lab 研究批次或来源说明。 |
| `candidate_rank` | integer | 是 | 候选内排名。 |
| `ranking_group_id` | string | 否 | 分组审计字段，不进入 feature。 |

### 标签 / outcome 字段

以下字段只能用于 label、target、outcome 或离线审计，禁止进入 feature：

- `buy_now_label`
- `wait_pullback_label`
- `cancel_buy_label`
- `three_day_positive_label`
- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `execution_return_to_close`
- `execution_return_to_next_open`
- `execution_drawdown_after_entry`

## 禁止进入 feature 的字段

- 所有 `future_*` 字段。
- 所有 `*_label` 字段。
- 所有 `execution_return_*` 字段。
- `max_drawdown_3d`。
- 后验收益、后验回撤、真实成交结果字段。
- 分组审计字段：`trade_date`、`datetime`、`etf_code`、`etf_name`、`sector`、`ranking_group_id`。

## 切分与防泄漏原则

- 正式评估必须使用 chronological split 或 walk-forward。
- 同一交易日、同一 ETF、同一板块场景不得在 train / validation 间形成标签泄漏。
- 任何使用 T+1 / T+3 结果构造的字段必须被标记为 label/outcome。
- 第一阶段只保留合同；第二阶段只允许 mock tensor smoke；第三阶段才可申请真实 5分钟K dry validation。

