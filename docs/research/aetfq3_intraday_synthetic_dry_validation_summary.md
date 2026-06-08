本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Synthetic Dry Validation Summary

## 任务定位

本任务是 F-9 synthetic intraday dry validation 阶段性总结。它整理 F-8 synthetic/mock intraday dry validation 工具链结果，作为 F 方向在无真实 5分钟K provider 情况下的阶段性收口。

这是 Lab-only synthetic dry validation，不是真实行情，不是交易系统，不训练模型，不接 QMT，不生成 OrderIntent，不写 `output/`，不生成 advisory package。

## 已完成链路

- manifest intake checker
- 5m schema validator
- forbidden feature scan
- tensor shape dry validation
- Intraday Watch state machine dry-run
- dry validation orchestrator
- report reader

F-8 工具链入口文档：

- `docs/research/aetfq3_intraday_synthetic_dry_validation.md`
- `docs/research/aetfq3_intraday_synthetic_dry_validation.json`

F-8 ignored smoke 报告：

- `.local_research_outputs/aetfq3_lab/intraday_synthetic_dry_validation/intraday_synthetic_dry_validation_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_synthetic_dry_validation/intraday_synthetic_report_reader_check.json`
- `.local_research_outputs/aetfq3_lab/intraday_synthetic_dry_validation/intraday_synthetic_tensor_shape_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_synthetic_dry_validation/intraday_synthetic_watch_dryrun_report.json`

## 测试结果

- specified tests: 17 passed
- tests/lab: 118 passed, 2 warnings

测试覆盖：

- valid mock manifest passed
- future feature leakage fails
- `training_allowed=true` fails
- `contains_order_intent=true` fails
- invalid QMT mode fails
- valid mock CSV schema passed
- missing required column fails
- bad OHLC fails
- insufficient bars fails
- orchestrator end-to-end passed
- report boundary fields are present
- no OrderIntent generated
- no checkpoint generated
- report reader catches boundary violations

## Smoke 结果

- orchestrator: passed
- reader: passed
- no OrderIntent
- no QMT
- no Stable
- no checkpoint
- no model save

主报告摘要：

- rows_checked: 48
- trade_date_count: 2
- etf_count: 2
- min_bars_per_etf_day: 12
- batch_size: 4
- time_steps: 12
- feature_count: 20
- target_count: 4
- visited_states: `WAIT_OPEN`, `INTRADAY_CONFIRMING`, `PROBE_READY`, `WAIT_PULLBACK`, `HOLD_LOCKED`, `PROFIT_PROTECT`, `EXIT_READY`
- terminal_state: `EXIT_READY`
- p0_blockers: none
- p1_warnings: none

## 当前阻塞

- no safe real 5m provider
- no human-exported 5m data package

当前只能说明 synthetic/mock dry validation 门禁链路可运行，不能说明真实 5分钟K provider 可用，也不能进入真实 intraday dry validation。

## 后续恢复条件

- independent safe provider file
- or human exported 5m package
- manifest + hash + source_note
- intake-only before dry validation

若恢复真实数据方向，必须先通过 F-7 safe provider 静态审查，或由人工提供 export-only / readonly 的 5分钟K 数据包并补齐 manifest、hash、source note。恢复后仍必须先做 intake-only 检查，不得直接进入 Stable。

## 边界

- no Stable
- no QMT trading
- no OrderIntent
- no advisory package
- no model training
- no trading advice

## 下一步建议

- 保持 F-8 synthetic/mock 工具链作为 Lab 内部门禁工具。
- 人工提供 independent safe provider file 后重跑 F-7。
- 或人工提供 5分钟K export package 后先做 manifest + hash + source_note intake-only 检查。
- 不建议进入 Stable。

## 任务结束核对

- 研究了什么：F-8 synthetic/mock intraday dry validation 工具链结果。
- 数据来自哪里：repo mock fixture、F-8 docs/research 文档、ignored local smoke report。
- 是否来自 Stable bundle：否。
- 是否有未来函数：未发现用于 feature 的 future label；F-8 intake 和 forbidden feature scan 已覆盖。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：本总结为 Lab-only READ_ONLY research summary，不生成 advisory package。
- 是否建议进入 Stable：否。
- 如果建议进入 Stable，最小合并方案是什么：不建议进入 Stable，无最小合并方案。
- 不允许直接提交到 Stable：确认不提交 Stable。
