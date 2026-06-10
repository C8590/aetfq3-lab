# AETF Q3 Lab Intraday Rolling OOP Pool Capture

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Scope

`tools/lab/intraday_rolling_oop_pool_capture.py` is a Lab-only repeatable runner for the public Sina rolling 5m ETF window exposed through AKShare. It appends the visible public 5m window into an ignored local pool, updates public daily OHLCV coverage, hashes the pool files, builds inventory, and computes strict OOP anchor readiness.

It does not run OOP validation, labels, models, training, torchrun, GPU, QMT, OrderIntent, Stable runtime/output, or Stable promotion.

## CLI

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_rolling_oop_pool_capture.py --out-artifact-dir .local_artifact_backup\aetfq3_lab_sources\intraday_signal_recovery_rolling_oop_pool --out-report-dir .local_research_outputs\aetfq3_lab\intraday_signal_recovery_rolling_oop_pool_readiness
```

Optional parameters:

```text
--etfs 159915,510050,510300,510500,512100,588000,159949,512880
--sprint-anchor-start 2026-04-09
--sprint-anchor-end 2026-06-03
--min-oop-anchors 10
--min-etfs 5
--min-groups 50
```

## Outputs

The artifact output directory must be under `.local_artifact_backup` and the report output directory must be under `.local_research_outputs`.

Pool artifacts:

- `rolling_oop_5m_pool.csv`
- `rolling_oop_daily_ohlcv_pool.csv`
- `POOL_INVENTORY.csv`
- `POOL_MANIFEST.json`
- `SHA256SUMS.txt`
- `source_note.md`

Readiness reports:

- `rolling_oop_capture_report.md`
- `rolling_oop_capture_report.json`
- `rolling_oop_anchor_readiness.json`
- `rolling_oop_readiness_decision.json`

## Append-Only Contract

The 5m key is `etf_code + datetime`. The daily key is `etf_code + trade_date`.

The runner never overwrites an existing row. If an incoming row has the same key and identical values, it is skipped as a duplicate. If an incoming row has the same key but different values, the existing row is kept and the conflict is recorded in manifest/report merge stats.

## Strict OOP Rule

Sprint1/2 anchor overlap range is `2026-04-09` to `2026-06-03`.

A strict OOP ETF anchor requires:

- anchor date is outside the Sprint1/2 overlap range
- complete public 5m bars for that ETF/date
- same-day public daily OHLCV exists
- T+1 daily exists
- T+3 daily exists
- no Stable bundle
- no QMT

An anchor date is eligible when its strict OOP ETF coverage reaches the configured minimum ETF count.

## Decisions

The runner emits one of:

- `ROLLING_OOP_POOL_READY_FOR_FIXED_SHORTLIST_VALIDATION`
- `ROLLING_OOP_POOL_LIMITED_ACCUMULATING`
- `ROLLING_OOP_POOL_NO_ELIGIBLE_ANCHORS_YET`
- `ROLLING_OOP_POOL_BLOCKED_DATA_QUALITY`
- `ROLLING_OOP_POOL_BLOCKED_SOURCE_UNAVAILABLE`

Even when ready, the report keeps `formal_model_evidence=false`, `stable_promotion_ready=false`, `formal_training_ready=false`, `qmt_ready=false`, `order_intent_ready=false`, and `automatic_promotion_ready=false`.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest tests\lab\test_intraday_rolling_oop_pool_capture.py
.\.venv\Scripts\python.exe -m pytest tests\lab
```

## Boundary

This tool is a public-data Lab capture runner only. It is not an advisory package, not model effectiveness evidence, not trading advice, and not a path to automatic Stable promotion.
