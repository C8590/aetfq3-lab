# AETF Q3 Lab Historical 5m Manual Intake Contract

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Purpose

This contract defines a Lab-only manual/export intake path for legally obtained historical ETF 5m OHLCV packages. It is an intake and readiness validator, not training, not fixed-shortlist OOP validation, not model evidence, not trading advice, and not Stable promotion.

The validator is:

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_historical_5m_manual_intake_validator.py --inbox .local_artifact_backup\aetfq3_lab_sources\intraday_historical_5m_manual_inbox --out-dir .local_research_outputs\aetfq3_lab\intraday_historical_5m_manual_intake
```

## Allowed Inbox

Only this ignored manual inbox may be used by default:

`.local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox/`

Allowed package files:

- `.csv`
- `.zip`
- `.parquet` when the local Python environment supports parquet
- `source_note.md`
- `SHA256SUMS.txt`
- `MANIFEST.json`

The validator writes reports only under:

`.local_research_outputs/aetfq3_lab/intraday_historical_5m_manual_intake/`

If the inbox is missing, the validator emits `MANUAL_HISTORICAL_5M_PACKAGE_NOT_FOUND_WAITING_FOR_INPUT` and writes a report instead of failing as P0.

## Data Schema

The import must map to:

- `trade_date`
- `datetime`
- `etf_code`
- `open`
- `high`
- `low`
- `close`
- `volume`

Recommended:

- `amount`
- `vwap`

If `amount` is missing, it remains empty and is recorded. If `vwap` is missing and `amount / volume` is available for `volume > 0`, the validator may compute `vwap`. It must not fabricate missing market data.

## Source Note

`source_note.md` must state:

- `source_name`
- `source_type`
- `export_method`
- `exported_at`
- `date_range`
- `etf_universe`
- `frequency: 5m`
- `whether_qmt_export`
- `whether_account_related`
- `whether_order_related`
- `whether_contains_trades_or_fills`
- `whether_contains_secret`
- `whether_stable_bundle`
- `human_authorized: true`

If `source_note.md` is missing, readiness is blocked as `BLOCKED_MISSING_SOURCE_NOTE`.

## Manifest

`MANIFEST.json` must match `docs/research/aetfq3_intraday_historical_5m_manual_manifest_template.json`.

Even if `qmt_related=true`, the package may only be `qmt_mode=export_only`. It must not contain account, position, order, trade, fill, OrderIntent, target_weight, final_buy_action, or secrets.

## Validation Gates

The validator checks:

- package inventory
- `source_note.md`
- `SHA256SUMS.txt`
- `MANIFEST.json`
- hash match
- schema mapping
- forbidden fields
- secret-like fields
- duplicate bars
- datetime monotonicity
- OHLC consistency
- nonnegative volume
- nonnegative amount when present
- bars per ETF/day
- strict OOP anchor readiness outside `2026-04-09` to `2026-06-03`

Readiness thresholds:

- strict OOP anchors >= `10`
- ETF count >= `5`
- group_count >= `50`
- data quality passed
- source authorized
- no forbidden fields
- no Stable bundle
- no account/order/trade/fill fields
- no OrderIntent

Ready only means a separate fixed-shortlist OOP no-save validation task may be created after human review. This task does not run validation.

## Boundary

- no QMT connection
- no account / funds / position / order / trade / fill data
- no OrderIntent
- no Stable runtime/output
- no `output/`
- no `lab_advisory/`
- no labels
- no model training
- no model / scaler / checkpoint save
- no torchrun
- no GPU
- no automatic promotion
- not trading advice
