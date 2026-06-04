# AETF Q3 Lab Research Review Checklist

## 使用场景

本 checklist 用于 `aetfq3-lab / Lab` 每个研究任务结束前的统一自检。它约束研究输出是否具备清楚的数据来源、Stable bundle 边界、未来函数检查、只读 advisory 属性、Stable 正式交易影响说明，以及进入 Stable 评估时的最小合并方案。

本 checklist 不授权直接修改 Stable，不授权生成 `OrderIntent`，不授权接正式 QMT，不授权写 `output/`，不授权创建 `lab_advisory/` 运行目录。

结构化版本见 `docs/lab/aetfq3_lab_research_review_checklist.json`。

## 快速检查表

### 一、任务定位检查

- 是否以“本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。”开头。
- 是否明确所属研究方向。
- 是否明确不是 Stable 正式策略任务。
- 是否明确不直接改 Stable。

### 二、数据来源检查

- 数据来自哪里。
- 是否来自 Stable bundle。
- 是否来自本地 ignored 数据。
- 是否来自历史 artifacts。
- 是否来自 QMT mock / readonly / 模拟盘。
- 是否使用真实账户或实盘数据。
- 是否有数据时间范围。
- 是否有样本数量。
- 是否有缺失值说明。
- 是否有异常值说明。

### 三、未来函数检查

- 是否使用未来收益作为标签。
- 是否将未来收益泄漏进特征。
- 是否使用了 T+1 / T+3 信息做当日决策。
- 是否存在日期错位。
- 是否有 walk-forward / chronological split 说明。
- 是否有人工确认无未来函数。

### 四、Stable 边界检查

- 是否影响 Stable 正式交易。
- 是否修改 `final_buy_action`。
- 是否修改 `target_weight`。
- 是否修改 BUY / PROBE 阈值。
- 是否生成 `OrderIntent`。
- 是否绕过 `RiskGate`。
- 是否写 Stable `output/`。
- 是否写 Stable `runtime/`。
- 是否直接提交到 Stable。

### 五、QMT 边界检查

- 是否连接真实 QMT。
- 是否只做 mock / readonly / 模拟盘。
- 是否读取真实持仓 / 资金。
- 是否生成委托。
- 是否撤单。
- 是否有成交回报。
- 是否可能被误解为正式执行。

### 六、输出文件检查

- 输出是否在 allowed 目录。
- 大 CSV 是否只在 `.local_research_outputs/`。
- 小型 Markdown / JSON 是否可进入 `docs/research/`。
- 是否创建了 `lab_advisory/` 运行目录。
- 是否生成真实 advisory 包。
- 是否写 `output/`。
- 是否提交 `data/`、`artifacts/`、model weights。

### 七、advisory-only 检查

- `affects_stable_trading` 是否为 `false`。
- `advisory_only` 是否为 `true`。
- `requires_human_review` 是否为 `true`。
- `recommended_for_stable` 是否为 `false`。
- 如 `recommended_for_stable` 为 `true`，是否给 `stable_merge_minimal_plan`。
- 是否包含 `forbidden_actions`。

### 八、结论等级检查

每个研究结论必须标注一个等级：

- 保留结论
- 降级结论
- 阻塞结论
- 待人工复核
- 可考虑进入 Stable 评估，但不得直接进入 Stable

### 九、任务结束必须回答

1. 研究了什么。
2. 数据来自哪里。
3. 是否来自 Stable bundle。
4. 是否有未来函数。
5. 是否影响 Stable 正式交易。
6. 是否只读 advisory。
7. 是否建议进入 Stable。
8. 如果建议进入 Stable，最小合并方案是什么。
9. 不允许直接提交到 Stable。
10. 下一步建议是什么。

## P0 阻断项

出现以下任一情况必须停止并报告：

- 任务未声明属于 `aetfq3-lab / Lab`，或被误派为 Stable 正式策略任务。
- 研究直接修改 Stable、`final_buy_action`、`target_weight`、BUY / PROBE 阈值。
- 生成正式 `OrderIntent`，绕过 `RiskGate`，或自动下单。
- 连接真实 QMT，生成委托，撤单，或读取真实账户/资金/成交回报且未获明确授权。
- 写入 Stable `output/`、Stable `runtime/`、Stable order intent 目录。
- 创建真实 `lab_advisory/` 运行目录或生成真实 advisory 包，但任务只授权 schema/example。
- 发现未来函数或日期错位且无法证明隔离。
- 试图直接提交到 Stable。

## P1 高风险项

出现以下情况必须显式披露，通常需要人工复核后才能继续：

- 数据来源、时间范围、样本数量不完整。
- 是否来自 Stable bundle 不明确。
- 缺失值、异常值处理说明不完整。
- 使用 T+1 / T+3 标签但未说明只用于训练或评估。
- QMT mock / readonly / 模拟盘边界不清楚，可能被误解为正式执行。
- 大 CSV、训练样本、模型权重、QMT 原始日志可能进入 Git。
- `recommended_for_stable: true` 但最小合并方案、风险点、人工审批、`RiskGate` 检查点或回滚方案不完整。

## P2 建议项

以下项不一定阻断，但建议补齐：

- 增加 walk-forward / chronological split 说明。
- 增加样本覆盖、行业分布、极端行情切片说明。
- 增加 evidence 文件路径和本地 ignored 产物位置说明。
- 将可入库小型摘要放入 `docs/research/`。
- 将大文件、训练日志、图表缓存放入 `.local_research_outputs/aetfq3_lab/`。
- 对每个结论补充等级和下一步任务卡。

## JSON checklist 字段说明

结构化 JSON 每个检查项包含：

- `check_id`: 稳定检查项编号。
- `question`: 需要回答的问题。
- `required_answer_type`: 期望回答类型，例如 boolean、text、enum、number、list。
- `pass_condition`: 通过条件。
- `fail_action`: 未通过时的动作。
- `severity`: `P0`、`P1` 或 `P2`。

JSON 顶层分组：

- `task_positioning`
- `data_sources`
- `future_leakage`
- `stable_boundary`
- `qmt_boundary`
- `output_files`
- `advisory_only`
- `conclusion_grading`
- `required_final_answers`

## 任务结束回答模板

```text
## 研究了什么
## 数据来自哪里
## 是否来自 Stable bundle
## 是否有未来函数
## 是否影响 Stable 正式交易
## 是否只读 advisory
## 是否建议进入 Stable
## 如果建议进入 Stable，最小合并方案是什么
## 不允许直接提交到 Stable
## 下一步建议是什么
```
