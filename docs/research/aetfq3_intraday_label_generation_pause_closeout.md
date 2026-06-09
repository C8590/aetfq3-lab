本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research closeout，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Label Generation Pause Closeout

## 当前状态

`F_PUBLIC_INTRADAY_LABEL_STATUS=PAUSED_BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA`

F public intraday label generation 线当前正式暂停。暂停原因不是代码 gate 回归，而是 public future-window daily OHLCV 覆盖不足。coverage gate regression 已证明：raw presence READY 只代表 source/hash/source-note 存在，effective readiness 必须继续由 coverage gate 阻断。

## 已完成链路

- public 5m OHLCV export 已完成。
- no-label tensor validation 已完成。
- label/outcome design 已完成。
- manifest leakage checker 已完成。
- label generation intake orchestrator 已完成。
- coverage gate fix 已完成。
- coverage gate regression 已通过。

## 当前 blocker

required future dates:

- `2026-06-09`
- `2026-06-10`
- `2026-06-11`

actual available dates:

- `2026-06-04`
- `2026-06-05`
- `2026-06-08`

missing dates by ETF:

- `159915`: `2026-06-09`, `2026-06-10`, `2026-06-11`
- `510050`: `2026-06-09`, `2026-06-10`, `2026-06-11`
- `510300`: `2026-06-09`, `2026-06-10`, `2026-06-11`

effective readiness decision:

- `BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA`

## 正确 gate 语义

- raw presence READY 只代表 future-window source、hash、source-note 存在。
- raw presence READY 不等于 effective readiness。
- effective readiness 必须以 coverage gate 为准。
- coverage 不足时必须阻断 label generation dry-run。
- 缺任一 ETF / 任一 required future date，均不得输出 `READY_FOR_LABEL_GENERATION_DRY_RUN`。

## 边界

- no QMT
- no OrderIntent
- no Stable
- no output/
- no lab_advisory/
- no training
- no torchrun
- no checkpoint
- no model save
- not trading advice

## 证据来源

- `docs/research/aetfq3_intraday_public_no_label_tensor_validation.md/json`
- `docs/research/aetfq3_intraday_label_outcome_design.md/json`
- `docs/research/aetfq3_intraday_label_leakage_checklist.json`
- `docs/research/aetfq3_intraday_label_manifest_leakage_checker.md/json`
- `docs/research/aetfq3_intraday_label_generation_intake_orchestrator.md/json`
- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake_gate_regression/coverage_gate_regression_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake_final_retry/readiness_decision.json`

## 下一步触发条件

只有当 public daily OHLCV 覆盖到 `2026-06-11`，并且 coverage gate 通过后，才允许申请下一张 `label generation dry-run` 任务卡。

即使 future-window coverage gate 通过，仍不授权训练、不进入 Stable、不生成 OrderIntent、不接 QMT；后续任务仍需人工 review 和 promotion gate。
