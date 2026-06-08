本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Label Generation Intake Orchestrator

## 任务定位

本阶段新增 label generation intake-only orchestrator，用于真实 label 生成前的只读门禁编排。它串联 manifest-only leakage checker、public no-label tensor report check、data quality report check、hash / source note presence check、future-window readiness fields check 和边界检查。

该工具不生成 labels，不读取更多真实行情，不训练模型，不运行 `torchrun`，不保存 checkpoint，不接 QMT，不生成 `OrderIntent`，不写 `output/`，不创建 `lab_advisory/`，不修改 Stable。readiness decision 只表示 intake 门禁状态，不授权 supervised training，也不是交易建议。

## CLI

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_label_generation_intake_orchestrator.py --manifest tests\fixtures\aetfq3_lab\mock_intraday_label_generation_intake_manifest.json --out-dir .local_research_outputs\aetfq3_lab\intraday_label_generation_intake\
```

输出 ignored 目录：

- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake/intraday_label_generation_intake_report.md`
- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake/intraday_label_generation_intake_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_label_generation_intake/readiness_decision.json`

## Orchestration Steps

1. 运行 manifest-only leakage checker。
2. 检查 public artifact 目录是否包含 `intraday_5m_export.csv`、`EXPORT_MANIFEST.json`、`source_note.md`、`SHA256SUMS.txt`。
3. 读取既有 hash/source validation report，确认 status passed 且 hash matched。
4. 读取既有 public no-label tensor report，确认 tensor shape passed、`labels_required=false`、`target_count=0`。
5. 读取 data quality report，确认 schema / OHLC / volume / datetime / VWAP quality passed。
6. 检查 future-window source fields 是否齐备。
7. 检查 no training、no QMT、no OrderIntent、no Stable 等边界。

## Readiness Decisions

- `READY_FOR_LABEL_GENERATION_DRY_RUN`
- `BLOCKED_MISSING_FUTURE_WINDOW_SOURCE`
- `BLOCKED_MANIFEST_P0`
- `BLOCKED_HASH_OR_SOURCE_NOTE`
- `BLOCKED_BOUNDARY_VIOLATION`

当前 fixture 因缺少 future-window source，预期 decision 为 `BLOCKED_MISSING_FUTURE_WINDOW_SOURCE`。这不是错误；它说明当前仍未具备真实 label 生成 dry-run 的 future window 输入。

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
