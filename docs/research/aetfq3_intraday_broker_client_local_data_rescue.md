# AETFQ3 Intraday Broker Client Local Data Rescue

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Scope

This Lab-only rescue diagnoses empty broker-client ETF 5m exports and inspects only explicitly authorized local broker-client market-data paths. It does not log in to any broker client, control a GUI, connect to broker servers, call QMT or xtdata, or read account, position, order, trade, fill, secret, token, password, or cookie files.

## Inputs

- `E:\CGS\T0002\export\`
- `E:\CGS\T0002\`
- `E:\XGS\T0002\export\`
- `E:\XGS\T0002\`
- `.local_artifact_backup\aetfq3_lab_sources\intraday_historical_5m_raw_exports\`

## Outputs

The report output is restricted to `.local_research_outputs\aetfq3_lab\intraday_broker_client_local_data_rescue\`.

If bounded parsing finds target ETF 5m OHLCV data, rescued artifacts are written to `.local_artifact_backup\aetfq3_lab_sources\intraday_historical_5m_raw_exports_rescued\` and the rescued CSV is copied into the existing raw-export handoff directory for the existing packager.

## Decision Values

- `BROKER_CLIENT_LOCAL_5M_RESCUE_READY_FOR_MANUAL_INTAKE`
- `BROKER_CLIENT_LOCAL_5M_RESCUE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION`
- `BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_EMPTY_EXPORT_ONLY`
- `BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_NO_LOCAL_5M_FILES_FOUND`
- `BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_UNSUPPORTED_LOCAL_FORMAT`
- `BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_FORBIDDEN_ACCOUNT_OR_TRADE_FILES`
- `BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_DATA_QUALITY`

## Boundary

Readiness from this rescue is only a manual-intake data handoff signal. It is not model validity evidence, not trading advice, not Stable evidence, and not permission to train, promote, generate labels, generate OrderIntent, or place orders.
