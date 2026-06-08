本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab QMT Readonly 5m Export Plan

## 任务定位

本文件是 F-3 QMT readonly/export 5分钟K 数据获取方案。它只定义人工导出流程、目录规范、manifest 规则和安全边界，不是 QMT 接入，不是交易执行，不调用 QMT API，不读取真实 QMT 数据，不导出真实 5分钟K。

Codex 在本阶段不得连接真实 QMT，不得读取账户、资金、持仓、委托、成交，不得生成 `OrderIntent`，不得自动下单。

## 数据目标

人工导出的数据目标是 ETF 5分钟K 序列，至少包含：

- `trade_date`
- `datetime`
- `etf_code`
- `etf_name`
- `sector`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `amount`
- `vwap`，或可由 `amount / volume` 审计计算
- `prev_close`
- `open_price`

数据只用于 Lab intraday dry validation intake，不用于训练正式模型，不用于 Stable 交易规则。

## 推荐导出目录

只允许人工导出到 ignored 本地目录，例如：

```text
.local_artifact_backup/aetfq3_lab_sources/intraday_5m_qmt_export/
```

或：

```text
.local_research_outputs/aetfq3_lab/intraday_dry_validation_inputs/
```

禁止导出到：

- `output/`
- Stable `runtime/`
- Stable `output/`
- Git tracked 目录
- `data/` 或 `artifacts/`，除非后续任务明确授权且仍不得提交

## 文件格式

建议格式：

- CSV 或 parquet。
- 一文件一交易日，或一文件一 ETF。
- 每批导出必须包含 manifest。
- 每个数据文件必须有 hash，例如 SHA256。
- 每批导出必须有 source note，说明人工导出人、导出时间、行情源、QMT 模式和授权范围。

建议目录结构：

```text
intraday_5m_qmt_export/
  manifest.json
  source_note.md
  hashes.sha256
  by_trade_date/
    2026-06-01.csv
  by_etf/
    510300.csv
```

## QMT 模式边界

只允许：

- `readonly`
- `mock`
- `export_only`

禁止：

- `trade`
- `live_order`
- `submit_order`
- `cancel_order`
- account query with secrets
- position / fund export，除非单独授权并脱敏

`config/qmt_execution.example.yaml` 中的示例边界应保持：`enabled=false`、`read_only=true`、`qmt_submit_enabled=false`、`allow_place_order=false`、`allow_cancel_order=false`。本方案不要求也不允许 Codex 修改本地 QMT 配置。

## Manifest 字段

沿用 `aetfq3_intraday_dry_validation_manifest`，QMT export 场景必须设置：

- `sample_type="intraday_5m"`
- `sample_path_type="local_ignored"` 或 `external_readonly`
- `source_kind="qmt_export_5m_bar"`
- `uses_qmt_export=true`
- `qmt_related=true`
- `qmt_mode="readonly"` 或 `"export_only"`，mock 演练可用 `"mock"`
- `contains_secret=false`
- `contains_live_order=false`
- `contains_order_intent=false`
- `training_allowed=false`
- `stable_effect_allowed=false`
- `affects_stable_trading=false`
- `advisory_only=true`
- `allowed_for=["dry_validation_only"]`
- `human_authorized=true`，并填写 `authorized_by` 与 `authorization_scope`

Manifest 还必须记录 `data_time_start`、`data_time_end`、`row_count`、`etf_count`、`trade_date_count`、`bar_count_per_etf_day`、`feature_columns`、`forbidden_feature_columns`、`has_future_leakage_check` 和 `review_checklist_passed`。

## P0 阻断

任一命中即停止：

- `qmt_mode` 不是 `readonly` / `mock` / `export_only`
- 出现账户、资金、持仓、委托、成交、secret、live order
- 出现 `OrderIntent`
- 写入 `output/` 或 Stable `runtime/`
- 未人工授权
- feature 包含未来 bar 或 future label
- 连接真实 QMT
- 自动下单
- 修改 `final_buy_action`、`target_weight`、BUY / PROBE 阈值

## 人工流程

1. 人工打开 QMT 或行情源。
2. 手工导出 5分钟K 到 ignored 目录。
3. 生成人工可复核的 hash。
4. 填写 manifest 和 source note。
5. Codex 只做 intake / dry validation 检查。
6. 不允许 Codex 下单或连接交易 API。

## Codex intake 边界

后续 intake-only 任务只能：

- 读取人工授权的 manifest。
- 校验路径位于 ignored 或 external readonly 边界。
- 校验 hash 和 source note。
- 检查字段、bar count、日期范围和 forbidden feature。
- 生成 dry validation 报告。

后续 intake-only 任务仍不得：

- 连接 QMT。
- 查询账户、资金、持仓、委托、成交。
- 导出更多数据。
- 训练模型。
- 写 Stable output/runtime。
- 生成 advisory 包或 OrderIntent。

## 后续任务卡

任务名：人工导出 5分钟K 后的 manifest 填写 + intake-only 检查任务。

任务边界：

- 输入：人工导出的 ignored 目录、manifest、source note、hash 文件。
- 只做：manifest schema 校验、hash 校验、字段检查、future leakage precheck、QMT/Stable/OrderIntent 边界检查。
- 禁止：连接 QMT、读取账户、训练、生成 OrderIntent、写 Stable、写 `output/`。

