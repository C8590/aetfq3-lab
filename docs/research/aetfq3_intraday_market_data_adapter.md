本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Market Data Adapter

## 任务定位

本任务新增 Lab-only market-data adapter wrapper 和静态安全扫描工具。它只服务 5分钟K mock export 与后续真实行情 adapter 的安全审查，不是 QMT 接入，不是交易执行，不是 Stable 参数变更。

默认边界：

```text
access_mode: READ_ONLY
final_action_change_allowed: false
contains_live_order: false
contains_secret: false
requires_human_review: true
promotion_gate_required: true
```

## 支持状态

- mock provider supported
- qmt provider not implemented
- xtdata provider not implemented
- real provider requires separate human authorization and safety review

当前 CLI 只允许 `--provider mock`。`qmt`、`xtdata`、`live` 必须失败并提示：

```text
real provider not implemented; requires separate human authorization and safety review
```

## 安全边界

- no account
- no positions
- no orders
- no trades
- no submit/cancel
- no OrderIntent
- no Stable
- no output/
- ignored out-dir only

允许输出目录只限：

- `.local_research_outputs/...`
- `.local_artifact_backup/...`

这些输出不得提交，不得被解释为真实行情，不得被解释为正式交易计划。

## 静态扫描

扫描工具：

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_market_data_safety_scan.py --path tests\fixtures\aetfq3_lab\mock_intraday_provider_safe.py
.\.venv\Scripts\python.exe tools\lab\intraday_market_data_safety_scan.py --path tests\fixtures\aetfq3_lab\mock_intraday_provider_unsafe.py
```

输出 JSON 到 stdout，包含：

- `safe`
- `forbidden_hits`
- `path`
- `scan_scope`
- `p0_blockers`

禁止关键词至少覆盖：

- submit/cancel/order APIs
- buy/sell
- account/asset/position/order/trade/fund/balance/cash
- OrderIntent / order_intent / live_order
- secret/token/password/api_key

任一命中默认 `safe=false`，并进入 P0 blocker。

## Mock Export Smoke

示例：

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_5m_market_data_adapter.py --provider mock --symbols 510300,159915 --start-date 2026-06-01 --end-date 2026-06-02 --out-dir .local_research_outputs\aetfq3_lab\intraday_market_data_adapter_smoke\
```

生成文件：

- `mock_intraday_5m_export.csv`
- `EXPORT_MANIFEST.json`
- `SHA256SUMS.txt`
- `source_note.md`

## 后续真实 Provider 接入门槛

真实 provider 接入必须满足：

- 单独任务卡
- 人工授权
- adapter safety scan
- readonly/export_only
- no secret
- no account/position/order/trade
- output only ignored directory

真实 provider 仍不得自动进入 Stable，不得生成正式交易计划，不得触发 QMT 或真实下单。

## P0 Blocker

- real provider 未经单独授权即启用
- 读取账户、资金、持仓、委托、成交
- 出现 submit/cancel、OrderIntent、live order、secret
- 写入 `output/`、Stable `runtime/` 或 Stable `output/`
- 将 mock 输出解释为真实行情或正式交易计划

## 后续任务卡

任务名：F-6 人工授权真实 5分钟K provider adapter 设计审查。

任务边界：只读行情 export-only；先做静态扫描和接口设计审查，再由人工确认是否允许接入真实行情源。禁止读取账户、持仓、委托、成交，禁止下单，禁止写 Stable。
