本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Watch Engine State Machine

## 任务定位

Intraday Watch Engine 是 F 方向的 Lab-only 盘中观察状态机方案。它只定义状态、事件、输入输出和审计边界，不读取真实 5分钟K 数据，不运行正式策略，不影响 Stable，不生成 advisory 包。

## 默认边界

- `does_not_generate_order_intent=true`
- `advisory_only=true`
- `stable_effect_allowed=false`
- `qmt_allowed=false`
- 不允许自动下单。
- 不允许修改 `final_buy_action`、`target_weight`、BUY / PROBE 阈值。

## 状态定义

| state_id | 中文解释 | 进入条件 | 退出条件 | 输入字段 | 输出字段 | 允许交易动作 | advisory only | 可能生成 OrderIntent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `WAIT_OPEN` | 等待开盘打印和开盘区间形成。 | 候选进入观察队列但尚未形成开盘信息。 | `OPEN_PRINTED` 或 `GAP_TOO_HIGH`。 | `trade_date`, `etf_code`, `open_price`, `prev_close` | `watch_state`, `open_gap_pct` | 否 | 是 | 否 |
| `OPEN_GAP_TOO_HIGH` | 开盘跳空过高，暂不追入。 | `open_gap_pct` 超过研究阈值或触发 `GAP_TOO_HIGH`。 | `VWAP_PULLBACK`, `RISK_CANCEL` 或收盘观察结束。 | `open_gap_pct`, `open_range_high`, `distance_to_vwap` | `watch_state`, `cancel_reason` | 否 | 是 | 否 |
| `INTRADAY_CONFIRMING` | 盘中确认板块和 ETF 强度。 | 已开盘且未触发取消，等待开盘区间与板块确认。 | `OPEN_RANGE_CONFIRMED`, `SECTOR_CONFIRM`, `SECTOR_FADE`, `BREAK_OPEN_LOW`。 | `open_range_high`, `open_range_low`, `sector_momentum_lag1`, `sector_breadth_lag1` | `watch_state`, `confirm_score` | 否 | 是 | 否 |
| `PROBE_READY` | 研究意义上的试探准备状态。 | 板块确认、未跌破开盘低点、VWAP 结构较好。 | `VWAP_PULLBACK`, `BREAK_OPEN_LOW`, `RISK_CANCEL`, `LOCK_HOLD_START`。 | `reclaim_vwap`, `distance_to_vwap`, `sector_top1_flag`, `candidate_rank` | `watch_state`, `research_signal` | 否 | 是 | 否 |
| `WAIT_PULLBACK` | 等待靠近 VWAP 或开盘区间的回落。 | 跳空较高或价格偏离 VWAP，需要等待更好的执行点。 | `VWAP_PULLBACK`, `VWAP_RECLAIM`, `BREAK_OPEN_LOW`, `RISK_CANCEL`。 | `pullback_to_vwap`, `distance_to_vwap`, `volume_ratio_5m` | `watch_state`, `pullback_status` | 否 | 是 | 否 |
| `CANCEL_BUY` | 研究视角取消买入观察。 | 跌破开盘低点、板块转弱、风险取消或结构失效。 | 观察日结束或人工复核后移出队列。 | `break_open_low`, `sector_fade`, `risk_flag` | `watch_state`, `cancel_reason` | 否 | 是 | 否 |
| `HOLD_LOCKED` | 已进入持有锁定观察，仅研究持有状态。 | 研究快照中出现锁定持有事件。 | `PROFIT_PROTECT_TRIGGER`, `EXIT_SIGNAL_RESEARCH_ONLY` 或观察结束。 | `intraday_return`, `drawdown_from_open`, `sector_momentum_lag1` | `watch_state`, `hold_lock_reason` | 否 | 是 | 否 |
| `PROFIT_PROTECT` | 盈利保护观察状态。 | 达到研究定义的盈利保护触发条件。 | `EXIT_SIGNAL_RESEARCH_ONLY` 或结构恢复。 | `high_from_open`, `drawdown_from_open`, `distance_to_vwap` | `watch_state`, `protect_reason` | 否 | 是 | 否 |
| `EXIT_READY` | 研究意义上的退出准备状态。 | 盈利保护或风险退出研究信号触发。 | 观察日结束或人工复核。 | `break_open_low`, `sector_fade`, `drawdown_from_open` | `watch_state`, `exit_research_reason` | 否 | 是 | 否 |

## 事件定义

- `OPEN_PRINTED`：当日开盘价已形成。
- `GAP_TOO_HIGH`：开盘跳空超过研究阈值。
- `OPEN_RANGE_CONFIRMED`：开盘区间高低点已确认。
- `VWAP_PULLBACK`：价格回落接近 VWAP。
- `VWAP_RECLAIM`：价格重新站上 VWAP。
- `BREAK_OPEN_LOW`：价格跌破开盘区间低点。
- `SECTOR_CONFIRM`：板块强度确认。
- `SECTOR_FADE`：板块强度转弱。
- `RISK_CANCEL`：研究风险取消。
- `LOCK_HOLD_START`：进入持有锁定观察。
- `PROFIT_PROTECT_TRIGGER`：触发盈利保护观察。
- `EXIT_SIGNAL_RESEARCH_ONLY`：仅研究用途的退出观察信号。

## 研究输出定义

未来实现时可定义以下运行产物，但本任务不实际生成这些文件：

- `intraday_watch_snapshot.json`
- `intraday_watch_events.csv`
- `intraday_watch_strategy_report.md`

这些文件如后续生成，也只能是 Lab research 输出，不得写入 Stable `output/` 或 `runtime/`，不得包含 live order，不得包含 `OrderIntent`。

