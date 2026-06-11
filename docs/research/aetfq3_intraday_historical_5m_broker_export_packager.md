# AETF Q3 Lab Historical 5m Broker Export Packager

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Purpose

This document defines the Lab-only broker-terminal export packager for user-supplied historical ETF 5m OHLCV files. It packages files that the user explicitly places in an ignored raw export directory, writes a standardized manual intake package, and optionally hands it to the manual intake validator.

This is packaging and readiness tooling only. It is not broker connectivity, not login automation, not QMT / xtdata access, not account / position / order / trade / fill access, not label generation, not model training, not OrderIntent generation, not advisory packaging, and not Stable promotion.

## CLI

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_historical_5m_broker_export_packager.py --raw-export-dir .local_artifact_backup\aetfq3_lab_sources\intraday_historical_5m_raw_exports --manual-inbox .local_artifact_backup\aetfq3_lab_sources\intraday_historical_5m_manual_inbox --out-dir .local_research_outputs\aetfq3_lab\intraday_historical_5m_broker_export_packager --run-manual-intake-validator
```

The tool rejects report output paths outside `.local_research_outputs` and artifact paths outside `.local_artifact_backup`.

## Inputs

Raw export directory:

`.local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports/`

Allowed file types:

- `.csv`
- `.zip`
- `.parquet` when supported by the local environment

Target ETF universe:

- `159915`
- `510050`
- `510300`
- `510500`
- `512100`
- `588000`
- `159949`
- `512880`

The files must contain historical ETF 5m OHLCV market bars only. They must not contain account, funds, position, order, trade, fill, password, token, secret, target weight, final buy action, or OrderIntent fields.

## Field Mapping

The packager maps common Chinese and English columns:

- `证券代码` / `代码` / `symbol` / `code` / `etf_code` -> `etf_code`
- `日期` / `交易日期` / `date` / `trade_date` -> `trade_date`
- `时间` / `日期时间` / `datetime` / `time` -> `datetime`
- `开盘` / `open` -> `open`
- `最高` / `high` -> `high`
- `最低` / `low` -> `low`
- `收盘` / `close` -> `close`
- `成交量` / `volume` / `vol` -> `volume`
- `成交额` / `amount` / `turnover` -> `amount`

If the required schema cannot be mapped, the decision is:

`BROKER_EXPORT_PACKAGE_BLOCKED_SCHEMA_UNMAPPABLE`

## Safety Checks

The packager blocks if raw file names, ZIP member names, or data fields contain forbidden account / trading / secret semantics. Blocking decision:

`BROKER_EXPORT_PACKAGE_BLOCKED_FORBIDDEN_FIELDS`

Forbidden tokens include:

- account / 账户
- 资金 / balance
- position / 持仓
- order / 委托
- trade / 成交
- fill
- password
- token
- secret
- target_weight
- final_buy_action
- OrderIntent

Market OHLCV fields such as `成交量` and `成交额` are allowed.

## Manual Inbox Package

On successful packaging, the tool writes:

`.local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox/`

Generated files:

- `historical_5m_manual_export.csv`
- `source_note.md`
- `MANIFEST.json`
- `SHA256SUMS.txt`

The generated `source_note.md` marks:

- `source_type: broker_terminal_manual_export`
- `acquisition_mode: user_manual_export`
- `human_authorized: true`
- `whether_account_related: false`
- `whether_order_related: false`
- `whether_contains_trades_or_fills: false`
- `whether_contains_secret: false`
- `whether_stable_bundle: false`
- `frequency: 5m`

The generated `MANIFEST.json` marks training, Stable effect, secret, OrderIntent, live order, account, position, order, and trade flags as false. `qmt_related=false` and `qmt_mode=not_qmt` by default.

## Reports

The tool writes reports under:

`.local_research_outputs/aetfq3_lab/intraday_historical_5m_broker_export_packager/`

Generated files:

- `broker_export_package_report.md`
- `broker_export_package_report.json`
- `broker_export_inventory.csv`
- `broker_export_package_decision.json`

## Validator Handoff

If `--run-manual-intake-validator` is set and packaging succeeds, the tool runs:

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_historical_5m_manual_intake_validator.py --inbox .local_artifact_backup\aetfq3_lab_sources\intraday_historical_5m_manual_inbox --out-dir .local_research_outputs\aetfq3_lab\intraday_historical_5m_manual_intake
```

Only if the validator outputs:

`MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION`

may a separate fixed-shortlist OOP no-save validation task be opened after human review. This packager does not run OOP validation.

## Decisions

Possible decisions:

- `BROKER_EXPORT_PACKAGE_READY_FOR_MANUAL_INTAKE_VALIDATOR`
- `BROKER_EXPORT_PACKAGE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION`
- `BROKER_EXPORT_PACKAGE_BLOCKED_WAITING_FOR_RAW_EXPORT`
- `BROKER_EXPORT_PACKAGE_BLOCKED_SCHEMA_UNMAPPABLE`
- `BROKER_EXPORT_PACKAGE_BLOCKED_FORBIDDEN_FIELDS`
- `BROKER_EXPORT_PACKAGE_BLOCKED_DATA_QUALITY`
- `BROKER_EXPORT_PACKAGE_BLOCKED_VALIDATOR_NOT_READY`

## Boundary

- no broker client connection
- no login
- no QMT
- no xtdata
- no account / funds / position / order / trade / fill access
- no secret / token / password / cookie read
- no full-disk scan
- no model training
- no labels
- no OrderIntent
- no automatic order
- no `output/`
- no Stable runtime/output
- no Stable modification
- no advisory package
- no automatic promotion
- not trading advice

Default Lab advisory protocol boundary remains:

```text
access_mode: READ_ONLY
final_action_change_allowed: false
contains_live_order: false
contains_secret: false
requires_human_review: true
promotion_gate_required: true
```

不允许直接提交到 Stable。
