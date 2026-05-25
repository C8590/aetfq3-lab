# AETFv2 Codex 多 Agent 工作流

本文档说明 AETFv2 项目如何使用“8 项目部 + 总控规则”的 Codex custom agents 架构。该架构用于职责拆分、风险隔离、审查留痕和总控决策，不用于绕过交易风控。

## 组织结构

| Agent | 中文名 | 职责 |
| --- | --- | --- |
| `aetfv2_01_pre_selection` | 候选池 / 预选项目部 | ETF 池、样本过滤、数据质量、板块分类、候选池 |
| `aetfv2_02_entry` | 买入决策项目部 | 买入条件、买入等级、买入解释、建议仓位 |
| `aetfv2_03_exit` | 卖出 / 退出项目部 | 持仓退出、减仓、止损、清仓建议 |
| `aetfv2_04_learning` | 复盘学习项目部 | 模拟盘复盘、买后 / 卖后表现、失败归因、策略健康度 |
| `aetfv2_05_historical_ml` | 历史回放与机器学习项目部 | 历史回放、样本生产、自动标签、人工复核队列、entry 校准报告、过拟合检查 |
| `aetfv2_06_risk_warning` | P0 信息预警 / 风险门控项目部 | R0-R4 风险等级、P0 预警、风险事件、entry 冻结、仓位上限覆盖、人工接管 |
| `aetfv2_07_qmt_execution` | QMT 交易执行项目部 | OrderIntent、QMT / miniQMT / XtQuant 适配、mock broker、下单前风控、订单日志、成交 / 持仓回读 |
| `aetfv2_08_control_center` | 总控项目部 | 统一合同、调度 7 个项目部、处理冲突、生成最终决策、输出前端快照和 Action API |

第一阶段 `.codex/config.toml` 固定：

```toml
[agents]
max_threads = 4
max_depth = 1
```

这表示最多 4 个 agent 并发，并且只允许主 agent 派一层子 agent，避免 8 个项目部同时无约束运行。

## 什么时候调用哪个 Agent

| 任务类型 | 推荐 Agent |
| --- | --- |
| ETF 池、候选池、样本过滤、数据质量、板块分类 | `aetfv2_01_pre_selection` |
| 买入规则、买入等级、买入解释、建议仓位 | `aetfv2_02_entry` |
| 退出条件、减仓、止损、清仓建议 | `aetfv2_03_exit` |
| 模拟盘复盘、失败归因、策略健康度 | `aetfv2_04_learning` |
| 历史回放、样本生产、标签、ML 校准、过拟合检查 | `aetfv2_05_historical_ml` |
| P0、R0-R4、风险冻结、仓位上限、人工接管 | `aetfv2_06_risk_warning` |
| OrderIntent、QMT 适配、mock broker、订单日志、成交 / 持仓回读 | `aetfv2_07_qmt_execution` |
| 跨部门任务、正式决策、冲突裁决、前端快照、Action API | `aetfv2_08_control_center` |

当任务涉及两个以上部门，或者可能影响正式决策、交易执行、真实持仓、风控冻结、前端 Action API 时，优先交给 `aetfv2_08_control_center`。

## 不能派给子 Agent 的任务

以下任务不得直接派给普通子 agent，必须由主 agent 或 `aetfv2_08_control_center` 处理：

- 生成正式交易决策。
- 直接下单或开启实盘自动执行。
- 跳过 `risk_warning` 的买入、卖出、仓位或执行请求。
- 在 R3 / R4 / P0 风险下生成普通买入。
- 把 `OrderIntent` 当成自动下单。
- 把 `historical_ml` 或 `learning` 建议自动写入正式交易参数。
- 刷新行情、生成信号、修改 `output`、触碰 `data/cache`，除非任务明确授权。
- 提交 token、key、env、venv、大体积缓存或策略输出。

## 总控如何汇总

`aetfv2_08_control_center` 汇总 7 个项目部意见时，按照以下优先级裁决：

1. `risk_warning` / P0 风险
2. 真实持仓风险
3. `market_state`
4. `pre_selection`
5. `entry` / `exit`
6. `qmt_execution`
7. `learning` / `historical_ml`

总控输出必须明确：

- 是否存在 P0 Blocker。
- 是否需要人工接管。
- 哪些建议被采纳，哪些建议被覆盖。
- 是否允许进入 QMT 执行前检查。
- 是否只是研究建议，不进入正式交易链路。

## frontend_console 的定位

`frontend_console` 不属于 8 项目部。它只是总控操作台，用于展示 `control_center` 的前端快照、人工确认状态、Action API 输出与审计信息。

前端不得绕过 `control_center` 直接触发 entry、exit 或 QMT 执行。前端按钮、确认框、动作 API 都必须体现风险门控与人工确认要求。

## 任务交接格式

所有项目部交接给主 agent 或总控时，使用以下格式：

```markdown
## 审查范围

## 本部门判断

## P0 Blocker

## P1 高风险

## P2 建议

## 证据

## 是否影响总控

## 测试建议

## 下一张 Codex 任务卡
```

主 agent 或总控汇总时，使用以下格式：

```markdown
## 审查范围

## 修改范围

## P0 Blocker

## P1 高风险

## P2 建议

## 证据

## 影响模块

## 测试命令

## 下一张 Codex 任务卡
```

## Commit 前检查清单

提交前必须确认：

- `git diff --name-only` 只包含本次任务授权文件。
- `.codex/config.toml` 存在，且 `max_threads = 4`、`max_depth = 1`。
- 8 个 agent TOML 文件存在，且 `name` 与标准命名一致。
- 每个 TOML 都包含 `name`、`nickname_candidates`、`description`、`developer_instructions`。
- 每个 agent 的 `developer_instructions` 都包含固定回复标题、职责边界、禁止事项和统一输出格式。
- 未修改业务代码，除非任务明确授权。
- 未刷新行情、未生成信号、未修改 `output`、未触碰 `data/cache`。
- 未提交 token、key、env、venv、大体积缓存或策略输出。
- 若工作区存在本任务之外的业务代码改动，必须在最终报告中说明这些改动不是本次范围，且不得混入本次提交。
