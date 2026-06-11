# AETF Q3 Lab Historical 5m Manual Intake Closeout

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Purpose

This document closes out the manual/export historical 5m intake validator as ready for future legal package intake. It is a validator closeout only. It is not data validation for a real package, not fixed-shortlist OOP validation, not label generation, not model training, not model evidence, not trading advice, and not Stable promotion.

## Completed validator

- tool path: `tools/lab/intraday_historical_5m_manual_intake_validator.py`
- validator commit: `ccf7342`
- contract docs:
  - `docs/research/aetfq3_intraday_historical_5m_manual_intake_contract.md`
  - `docs/research/aetfq3_intraday_historical_5m_manual_intake_contract.json`
  - `docs/research/aetfq3_intraday_historical_5m_manual_manifest_template.json`
- tests:
  - specific validator test: `9 passed`
  - `tests/lab`: `271 passed, 2 warnings`
- supported ignored inbox: `.local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox/`
- output directory: `.local_research_outputs/aetfq3_lab/intraday_historical_5m_manual_intake/`
- required package files / notes:
  - `source_note.md`
  - `MANIFEST.json`
  - `SHA256SUMS.txt`
  - package data as legal `.csv`, `.zip`, or supported `.parquet`
- source note checks:
  - source identity, export method, export time, date range, ETF universe, `frequency: 5m`
  - QMT/export-only flags
  - account/order/trade/fill/secret/Stable-bundle flags
  - `human_authorized: true`
- manifest / hash checks:
  - manifest conforms to `docs/research/aetfq3_intraday_historical_5m_manual_manifest_template.json`
  - SHA256 inventory is required
  - hash mismatches block readiness
- forbidden field checks:
  - account
  - position
  - order
  - trade
  - fill
  - OrderIntent
  - target_weight
  - final_buy_action
  - secret-like fields
- schema mapping:
  - required: `trade_date`, `datetime`, `etf_code`, `open`, `high`, `low`, `close`, `volume`
  - recommended: `amount`, `vwap`
  - missing `amount` is recorded
  - `vwap` may only be computed when `amount` and positive `volume` are available
- data quality checks:
  - duplicate bars
  - datetime monotonicity
  - OHLC consistency
  - nonnegative volume
  - nonnegative amount when present
  - bars per ETF/day
- strict OOP readiness checks:
  - anchor dates outside `2026-04-09` to `2026-06-03`
  - ETF count >= 5
  - strict OOP anchors >= 10
  - group_count >= 50
  - T+1 / T+3 daily coverage must be available
- default ignored run decision: `MANUAL_HISTORICAL_5M_PACKAGE_NOT_FOUND_WAITING_FOR_INPUT`

## Current data status

- online historical 5m source acquisition is blocked as `HISTORICAL_5M_BACKFILL_BLOCKED_NO_USABLE_SOURCE`.
- rolling OOP pool is accumulating as `ROLLING_OOP_POOL_LIMITED_ACCUMULATING`.
- current rolling OOP pool has 2 eligible strict OOP anchors and group_count 16, below the required 10 anchors / 50 groups.
- manual/export intake validator is ready, but no real legal external 1y/3y ETF 5m package is present yet.
- fixed-shortlist OOP validation is still blocked.

## Manual package trigger

Only after a compliant historical 5m package appears in the ignored manual inbox may the validator be run:

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_historical_5m_manual_intake_validator.py --inbox .local_artifact_backup\aetfq3_lab_sources\intraday_historical_5m_manual_inbox --out-dir .local_research_outputs\aetfq3_lab\intraday_historical_5m_manual_intake
```

The package must be legally obtained, human-authorized, no-secret, no-account, no-position, no-order, no-trade, no-fill, no OrderIntent, and not from a Stable bundle.

## Readiness trigger

Only if the validator outputs:

`MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION`

may a separate fixed-shortlist OOP no-save validation task be opened after human review. This closeout does not authorize validation by itself.

## Boundary

- no Stable
- no QMT account connection
- no account / position / order / trade / fill access
- no OrderIntent
- no model training
- no model / scaler / checkpoint save
- no labels
- no OOP validation
- no advisory package
- no automatic promotion
- no trading advice

Default Lab advisory protocol boundary remains:

```text
access_mode: READ_ONLY
final_action_change_allowed: false
contains_live_order: false
contains_secret: false
requires_human_review: true
promotion_gate_required: true
```

## Final review status

- manual intake validator ready: true
- manual package present: false
- manual package readiness decision: `MANUAL_HISTORICAL_5M_PACKAGE_NOT_FOUND_WAITING_FOR_INPUT`
- fixed-shortlist OOP validation ready: false
- Stable promotion ready: false
- formal training ready: false
- QMT ready: false
- OrderIntent ready: false
- automatic promotion ready: false

不允许直接提交到 Stable。
