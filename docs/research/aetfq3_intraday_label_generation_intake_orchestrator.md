本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Label Generation Intake Orchestrator

## 任务定位

本阶段新增 label generation intake-only orchestrator，用于真实 label 生成前的只读门禁编排。它串联 manifest-only leakage checker、public no-label tensor report check、data quality report check、hash / source note presence check、future-window source presence check、future-window coverage audit 和边界检查。

该工具不只是 presence gate。future-window coverage 是正式 intake gate：raw presence READY 只表示 source/hash/source-note 存在，不等于 effective readiness。若 required ETF 缺少任一 required future date，effective readiness 必须阻断 label generation dry-run。

该工具不生成 labels，不读取更多真实行情，不训练模型，不运行 `torchrun`，不保存 checkpoint，不接 QMT，不生成 `OrderIntent`，不写 `output/`，不创建 `lab_advisory/`，不修改 Stable。readiness decision 只表示 intake 门禁状态，不授权 supervised training，也不是交易建议。

## CLI

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_label_generation_intake_orchestrator.py --manifest tests\fixtures\aetfq3_lab\mock_intraday_label_generation_intake_manifest.json --out-dir .local_research_outputs\aetfq3_lab\intraday_label_generation_intake\
```

输出 ignored 目录：

- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake/intraday_label_generation_intake_report.md`
- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake/intraday_label_generation_intake_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake/readiness_decision.json`

`--out-dir` 仅允许 `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake*` 子目录，例如：

- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake`
- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake_retry`
- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake_final_retry`

仍阻断 `output/`、Stable 路径、repo 外路径和非 `.local_research_outputs/aetfq3_lab/` 子树。

## Orchestration Steps

1. 运行 manifest-only leakage checker。
2. 检查 public artifact 目录是否包含 `intraday_5m_export.csv`、`EXPORT_MANIFEST.json`、`source_note.md`、`SHA256SUMS.txt`。
3. 读取既有 hash/source validation report，确认 status passed 且 hash matched。
4. 读取既有 public no-label tensor report，确认 tensor shape passed、`labels_required=false`、`target_count=0`。
5. 读取 data quality report，确认 schema / OHLC / volume / datetime / VWAP quality passed。
6. 检查 future-window source fields 是否齐备，形成 `presence_gate_passed` 和 `raw_presence_decision`。
7. 读取 future-window daily source，检查 `trade_date` / `etf_code` 字段。
8. 按 manifest 中 `future_window_required_dates`、`future_window_required_coverage_end` 和 required ETF 集合执行 coverage audit。
9. 记录 `coverage_gate_passed`、`coverage_sufficient`、`missing_future_dates_by_etf`。
10. 检查 no training、no QMT、no OrderIntent、no Stable 等边界。

## Readiness Decisions

- `READY_FOR_LABEL_GENERATION_DRY_RUN`
- `BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA`
- `BLOCKED_MISSING_FUTURE_WINDOW_SOURCE`
- `BLOCKED_MANIFEST_P0`
- `BLOCKED_HASH_OR_SOURCE_NOTE`
- `BLOCKED_BOUNDARY_VIOLATION`

只有 manifest leakage checker、hash/source-note、no-label tensor report、data quality report、future-window presence、future-window coverage 和边界检查全部通过，才允许 `READY_FOR_LABEL_GENERATION_DRY_RUN`。

当前 fixture 因缺少 future-window source，预期 decision 为 `BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA`。这不是错误；它说明当前仍未具备真实 label generation dry-run 的 future window 输入。

若 presence gate 返回 READY，但 coverage audit 发现缺少 required dates，例如 `2026-06-09`、`2026-06-10`、`2026-06-11`，effective readiness 必须保持 `BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA`，不得让 raw presence READY 覆盖 coverage blocker。

## Boundary

- intake-only
- no real label file generation
- no more real market data read
- no model training
- no torchrun
- no checkpoint
- no model save
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no Stable
- no trading advice

## Next Steps

- 若后续提供 future-window 数据，必须同时提供 hash、source note、manifest 和人工授权。
- 只有 intake report 达到 `READY_FOR_LABEL_GENERATION_DRY_RUN`，才可申请下一张真实 label generation dry-run 任务卡。
- `READY_FOR_LABEL_GENERATION_DRY_RUN` 仍不等于 supervised training authorization，不进入 Stable。
