本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Real Data Source Inventory

## 任务定位

F-2 是 intraday 5分钟K dry validation 的前置门禁。本文件只记录真实 5分钟K / tick / 盘口 / QMT export 数据源的文件名级盘点结果，不读取真实行级内容，不复制真实数据，不生成真实样本，不训练模型。

## 搜索范围

本次只做文件名、大小、修改时间元数据盘点。不存在的路径只记录为不存在。

| 路径 | 是否存在 |
| --- | --- |
| `E:\aetfq3-lab\` | true |
| `E:\aetfq3-lab\data\` | true |
| `E:\aetfq3-lab.local_research_outputs\` | false |
| `E:\aetfq3-lab.local_artifact_backup\` | false |
| `E:\aetfq3-stable\` | false |
| `E:\aetfv2.1ml\` | true |
| `E:\AETF-ModelLab\` | false |

关键词包括：`5m`、`5min`、`intraday`、`minute`、`tick`、`orderbook`、`order_book`、`盘口`、`分钟`、`vwap`、`qmt`、`quote`、`xtdata`、`bar`、`kline`、`1m`、`5分钟`、`tick_data`、`depth`、`level2`。

为避免依赖包和代码文件误判，最终严格 inventory 排除了 `.git`、`.venv`、`__pycache__`、`docs`、`tools`、`tests`、`.pytest_cache`、`.local_research_outputs`，并只对相对搜索根的路径和文件名做关键词匹配。盘点未读取 CSV / parquet / JSON 行级内容。

Ignored 本地报告：

- `.local_research_outputs/aetfq3_lab/intraday_data_inventory/intraday_data_source_inventory.md`
- `.local_research_outputs/aetfq3_lab/intraday_data_inventory/intraday_data_source_inventory.json`
- `.local_research_outputs/aetfq3_lab/intraday_data_inventory/intraday_candidate_files.csv`

## 是否找到真实 5分钟K 数据源

未找到可直接判定为真实 5分钟K / 分钟线 / intraday bar 序列的数据源文件。

`usable_for_intraday_dry_validation` 数量为 0。当前不能进入真实 intraday dry validation。

## 是否找到 QMT export

未找到真实 QMT export 数据文件。仅发现 2 个 QMT example 配置文件：

| path | filename | size | modified_time | category | can_use_for_dry_validation |
| --- | --- | ---: | --- | --- | --- |
| `E:\aetfq3-lab\config` | `qmt_execution.example.yaml` | 447 | `2026-06-04T13:36:41.1391579+08:00` | `qmt_export_readonly_candidate` | false |
| `E:\aetfv2.1ml\config` | `qmt_execution.example.yaml` | 447 | `2026-06-03T22:36:04.7617590+08:00` | `qmt_export_readonly_candidate` | false |

它们只说明存在 QMT 配置模板，不是 QMT export 行情样本，不得用于 dry validation。

## 是否找到 tick/orderbook

未找到可用 tick / orderbook / depth / level2 数据源文件。

## 哪些数据可用于 F-2 dry validation

当前无。下一步必须由人工指定一个真实 intraday 5分钟K 数据源，并提供只读授权。授权前不得读取行级内容。

## 哪些需要人工授权

- 任意真实 5分钟K / 分钟线 / intraday bar 文件。
- 任意 QMT export 文件，即使只读，也必须确认 `qmt_mode=readonly`、`mock` 或 `export_only`。
- 任意 tick / orderbook / quote cache 文件。

## 哪些绝对不能用

- 可能包含账户、密钥、资金、持仓、委托、成交、live order 的文件。
- 能触发真实 QMT 或下单链路的文件 / 配置 / runtime。
- Stable `output/` 或 `runtime/`。
- 包含 `OrderIntent`、live order、secret 的文件。
- 未经人工授权的数据源。

## 数据源分类结果

| 分类 | 数量 | 结论 |
| --- | ---: | --- |
| `usable_for_intraday_dry_validation` | 0 | 未找到可用真实 intraday 数据源。 |
| `qmt_export_readonly_candidate` | 2 | 仅 example config，不是 export 数据，不可用于 dry validation。 |
| `quote_cache_or_realtime_snapshot` | 0 | 未发现。 |
| `unsafe_or_forbidden` | 0 | 严格候选中未发现。 |
| `not_relevant` | 0 | 严格候选中未保留。 |

## 是否可以进入真实 intraday dry validation

不可以。

原因：

- 未找到真实 5分钟K / 分钟线 / intraday bar 序列文件。
- 未获得 `human_authorized=true` 的数据源授权。
- 未建立具体 dry validation manifest 实例。

## 如果可以，下一步需要什么授权

若人工后续提供数据源，至少需要：

- 数据源路径和只读授权。
- `sample_path_type=local_ignored` 或 `external_readonly`。
- `source_kind=intraday_5m_bar` 或等价可审计类型。
- 明确是否来自 Stable bundle、是否来自 QMT export。
- 明确 `qmt_mode=readonly/mock/export_only`，且不得连接真实 QMT。
- 完整 forbidden feature scan 和 future leakage check。

## 如果不可以，缺什么

- 真实 intraday 5分钟K 数据源。
- 人工授权。
- dry validation manifest 实例。
- 数据质量报告和字段标准化检查。

## Manifest 字段摘要

F-2 dry validation manifest 必须记录来源、授权、数据范围、字段、forbidden feature、future leakage check、Stable/QMT/OrderIntent 边界和 review checklist。默认只允许 `allowed_for=["dry_validation_only"]`，`training_allowed=false`，`stable_effect_allowed=false`，`advisory_only=true`。

## P0/P1/P2 规则

P0：未授权、训练开启、Stable 影响开启、包含 secret/live order/OrderIntent、QMT 模式越界、future label 进入 feature、未来 bar 泄漏、写 Stable output/runtime、接真实 QMT、自动下单、修改 `final_buy_action` / `target_weight` / BUY / PROBE 阈值。

P1：QMT export 需人工复核、日期范围太短、bar 数不稳定、缺 `vwap` / `amount` / `volume`、缺 sector / candidate context、缺 future leakage check、source_kind 不明确。

P2：字段命名需标准化、缺可选上下文字段、需要后续数据质量报告。

