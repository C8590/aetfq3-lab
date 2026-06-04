# AETF Q3 Lab False Downgrade Data Design

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 文档定位

本文只定义 A 方向 `ML false downgrade` 的研究问题、样本口径、标签、特征候选、时间切分和未来函数检查。本文不读取真实行情，不训练模型，不生成 advisory 包，不修改 Stable，不影响 `final_buy_action`、`target_weight` 或 BUY / PROBE 阈值。

## 研究问题

`ML false downgrade` 研究 ML 是否误杀 V2 core 强主线：

- V2 core 给出 `PROBE` / `BUY`，但 ML 给出 `ML_DOWNGRADED` / `AVOID` / `NO_BUY` 后，后续仍上涨，是否构成 false downgrade。
- 哪些特征导致 ML 错杀强主线，例如高波动、短期过热、板块拥挤、成交异常或 ML score 断层。
- 如何保护 V2 core 强主线，但不让 ML 直接改 Stable；Lab 只输出 shadow / advisory 结论。

## 需要输入

未来研究任务可读取或整理以下输入，但本设计任务不实际读取：

- V2 core 多日输出：`pre_selection_result.csv`、`entry_signal.csv`
- ML_SIM 多日输出：`ml_sim_daily_comparison.csv`、`ml_sim_daily_comparison.json`、`ml_sim_summary.json`、`ml_sim_review_queue.csv`
- `ml_entry_scores.csv`
- ETF 日线数据
- sector mapping
- future return / drawdown 派生标签
- risk flags，可选

所有大 CSV、训练样本、明细输出应放入 `.local_research_outputs/aetfq3_lab/`，不得提交到 Git。

## 样本主键

每一行样本表示某交易日某 ETF 的 V2 core 与 ML_SIM 决策对照：

- `trade_date`: 信号日期，当日决策锚点。
- `etf_code`: ETF 代码。
- `sector`: sector mapping 后的板块。
- `v2_action`: V2 core 当日动作，重点关注 `PROBE` / `BUY`。
- `ml_action`: ML_SIM 当日动作，重点关注 `ML_DOWNGRADED` / `AVOID` / `NO_BUY`。
- `model_version`: ML 模型版本。
- `feature_version`: 特征口径版本。

建议唯一键：

```text
trade_date + etf_code + model_version + feature_version
```

## 标签定义

标签只允许由 `trade_date` 之后的未来窗口计算，不能进入特征。

### false_downgrade_1d

```text
v2_action in {PROBE, BUY}
and ml_action in {ML_DOWNGRADED, AVOID, NO_BUY}
and future_return_1d > threshold_1d
and max_drawdown_1d >= drawdown_floor_1d
```

### false_downgrade_3d

```text
v2_action in {PROBE, BUY}
and ml_action in {ML_DOWNGRADED, AVOID, NO_BUY}
and future_return_3d > threshold_3d
and max_drawdown_3d >= drawdown_floor_3d
```

### false_downgrade_lock3

```text
v2_action in {PROBE, BUY}
and ml_action in {ML_DOWNGRADED, AVOID, NO_BUY}
and future_return_3d > threshold_3d
and close_t_plus_3 >= close_t
and max_drawdown_3d >= drawdown_floor_3d
```

### true_downgrade

```text
v2_action in {PROBE, BUY}
and ml_action in {ML_DOWNGRADED, AVOID, NO_BUY}
and (
    future_return_3d <= 0
    or max_drawdown_3d < drawdown_floor_3d
)
```

### neutral_downgrade

```text
v2_action in {PROBE, BUY}
and ml_action in {ML_DOWNGRADED, AVOID, NO_BUY}
and false_downgrade_3d == false
and true_downgrade == false
```

## 阈值设计

建议保留多组可调阈值，不在首轮固定单一结论：

- `future_return_1d > 0`
- `future_return_3d > 0`
- `future_return_3d > 0.01`
- `max_drawdown_3d >= -0.02`
- `sector_rank <= 3`
- V2 core strength score 分层：`weak` / `medium` / `strong` / `top_line`

建议在报告中同时展示严格口径和宽松口径，避免结论只由一个阈值驱动。

## 特征候选

候选特征只能来自 `trade_date` 当日或之前：

- V2 score / action / rank。
- sector momentum / breadth / overheat。
- ETF momentum / volatility / liquidity。
- ML score / `p_good` / `p_bad`。
- crowding features。
- risk flags。
- previous day gap。
- position in sector。
- V2 core strength score 分层。
- recent false downgrade / true downgrade 历史统计，但只能使用过去已发生窗口。

## 禁止进入 feature 的字段

- `future_return_1d`
- `future_return_3d`
- `future_return_lock3`
- `max_drawdown_1d`
- `max_drawdown_3d`
- `false_downgrade_1d`
- `false_downgrade_3d`
- `false_downgrade_lock3`
- `true_downgrade`
- `neutral_downgrade`
- 任何 T+1 / T+3 价格、收益、回撤、排名或未来最优结论。

## 模型建议

首轮建议使用表格模型，不先上 PyTorch：

- LightGBM
- CatBoost
- XGBoost

建议输出：

- `false_downgrade_risk_score`: ML 错杀强主线概率或风险分。
- `protect_v2_core_candidate`: 是否建议进入强主线保护候选。
- `downgrade_reason_cluster`: 错杀原因聚类或规则解释。

这些输出只能作为 shadow / advisory，不得直接修改 Stable。

## 时间切分

正式评估必须使用时间序列切分：

- chronological split。
- walk-forward。
- `train <= date, validate > date`。
- 不允许随机打乱时间序列作为正式评估。
- 相同 `trade_date` 的样本应在同一个 split，避免同日信息泄漏。
- 每次评估记录 `model_version`、`feature_version`、训练起止日期、验证起止日期。

## 未来函数检查

- `future_return` 只能作为 label。
- `max_drawdown` 只能作为 label 或评估指标。
- 不得把未来收益、未来回撤、T+1 / T+3 价格或未来排名放入 feature。
- 不能用 T+1 / T+3 信息做当日判断。
- `model_version` / `feature_version` 必须记录。
- 所有 rolling 特征必须确认只使用 `trade_date` 当日或之前的数据。
- 如果使用 sector topN 历史，必须确认 topN 是当日可见结果，不包含未来窗口表现。

## 未来可输出文件

未来研究任务可生成以下文件，但本任务不实际生成这些运行产物：

- `ml_false_downgrade_cases.csv`
- `ml_false_downgrade_pattern_report.md`
- `ml_false_downgrade_pattern_report.json`
- `false_downgrade_model_report.md`
- `false_downgrade_model_report.json`
- `false_downgrade_shadow_scores.csv`

大 CSV、训练明细和模型诊断应写入 `.local_research_outputs/aetfq3_lab/`；小型方法摘要可进入 `docs/research/`。

## Stable 边界

- Lab 可以建议 Stable，但不能直接改 Stable。
- 不修改 `final_buy_action`。
- 不修改 `target_weight`。
- 不修改 BUY / PROBE 阈值。
- 不生成 Stable 正式 `OrderIntent`。
- 不绕过 `RiskGate`。
- 不把 false downgrade score 直接接入 Stable entry。
- 若未来建议进入 Stable，只能提交最小合并方案、风险点、Stable 侧人工审批要求、`RiskGate` 检查点和回滚方案。

## Review Checklist 映射

- 研究了什么：设计 ML false downgrade 数据口径。
- 数据来自哪里：本文只列未来可用输入，不读取真实数据。
- 是否来自 Stable bundle：本任务否；未来若使用必须显式标注。
- 是否有未来函数：本文定义标签与特征隔离要求。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：是。
- 是否建议进入 Stable：否，设计阶段不建议。
- 如果建议进入 Stable，最小合并方案是什么：不适用。
- 不允许直接提交到 Stable：确认不允许。
