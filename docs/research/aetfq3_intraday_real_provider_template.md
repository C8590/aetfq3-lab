本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Real Provider Template

## 任务定位

这是 Lab-only provider 接入模板，不是真实 QMT 接入，不是交易执行。模板只定义 5分钟K market-data-only provider 的接口、输出 schema 校验和能力声明。

当前模板中的真实 provider 方法固定抛出：

```text
real provider requires separate human authorization and safety review
```

## 当前支持状态

- mock provider supported
- real provider template only
- qmt provider not implemented
- xtdata provider not implemented
- live provider not implemented

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

## Provider Capabilities

模板能力声明必须保持：

- `market_data_only=true`
- `supports_account=false`
- `supports_position=false`
- `supports_order=false`
- `supports_trade=false`
- `supports_submit_order=false`
- `supports_cancel_order=false`
- `supports_order_intent=false`
- `requires_secret=false`
- `requires_live_session=false`

这些字段只用于声明边界，不代表真实 provider 已接入。

## 静态扫描

扫描工具：

```powershell
.\.venv\Scripts\python.exe tools\lab\intraday_market_data_safety_scan.py --path tools\lab\intraday_provider_template.py
.\.venv\Scripts\python.exe tools\lab\intraday_market_data_safety_scan.py --path tests\fixtures\aetfq3_lab\mock_intraday_provider_template_safe.py
```

扫描输出包含：

- `safe`
- `severity`
- `forbidden_hits`
- `path`
- `scan_scope`
- `p0_blockers`

Forbidden keywords 命中时必须视为 P0 blocker。关键词覆盖 submit/cancel、buy/sell、account、position、order、trade、fund、balance、cash、OrderIntent、live order、secret、token、password、api key 等类别。

## 后续真实 Provider 接入门槛

1. 单独任务卡。
2. 人工授权。
3. provider 文件 safety scan。
4. readonly/export_only。
5. no secret。
6. no account/position/order/trade。
7. output only ignored directory。
8. no Stable。
9. no QMT trading。
10. no OrderIntent。

## P0 Blocker

- 未经单独授权实现真实 provider。
- 静态扫描命中 forbidden keyword。
- 读取账户、资金、持仓、委托、成交。
- 触发 submit/cancel 或交易执行。
- 生成 OrderIntent。
- 写入 `output/`、Stable runtime 或 Stable output。

## 下一步任务卡

任务名：F-7 人工授权真实行情源 provider 文件静态审查。

边界：只读行情 export-only；先审查 provider 文件和依赖导入，再决定是否允许在 ignored 目录做人工导出 smoke。禁止账户、持仓、委托、成交、交易执行、Stable 写入和 OrderIntent。
