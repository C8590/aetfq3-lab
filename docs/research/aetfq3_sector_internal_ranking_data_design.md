# AETF Q3 Lab Sector Internal ETF Ranking Data Design

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 文档定位

本文只定义 E 方向 `同板块 ETF 内部排序` 的研究问题、样本口径、标签、排序目标、特征候选、时间切分和未来函数检查。本文不读取真实行情，不训练模型，不生成 advisory 包，不替代 Stable entry。

## 研究问题

同板块 ETF 内部排序研究的目标是：当 V2 / Lab 已识别强板块后，研究是否能在同一板块内排序 ETF，而不是替代 V2 entry。

核心问题：

- 在同一 `sector` 内，哪个 ETF 后续收益 / 回撤更好。
- ML 是否能提升 sector 内 ETF 选择。
- 排序结果是否能作为 shadow / advisory，不直接替代 Stable entry。

## 需要输入

未来研究任务可读取或整理以下输入，但本设计任务不实际读取：

- sector mapping。
- ETF 日线数据。
- V2 原选 ETF 多日输出。
- sector top1 / topN 历史。
- `ml_entry_scores.csv`。
- `daily_ml_universe_samples.csv`。
- future return / drawdown 派生标签。

所有大 CSV、训练样本、排序明细应放入 `.local_research_outputs/aetfq3_lab/`，不得提交到 Git。

## 样本主键

每一行样本表示某交易日、某板块内的某 ETF：

- `trade_date`
- `sector`
- `etf_code`
- `rank_scope = sector_internal`
- `model_version`
- `feature_version`

建议唯一键：

```text
trade_date + sector + etf_code + rank_scope + model_version + feature_version
```

同一天同一个 `sector` 内的全部 ETF 构成一个 ranking group：

```text
ranking_group_id = trade_date + sector
```

## 标签定义

标签只能使用 `trade_date` 之后的未来窗口计算，不能进入特征。

### best_in_sector_1d

```text
future_return_1d == max(future_return_1d within trade_date + sector)
and max_drawdown_1d >= drawdown_floor_1d
```

### best_in_sector_3d

```text
future_return_3d == max(future_return_3d within trade_date + sector)
and max_drawdown_3d >= drawdown_floor_3d
```

### top_quantile_in_sector_3d

```text
future_return_3d percentile within trade_date + sector >= quantile_threshold
and max_drawdown_3d >= drawdown_floor_3d
```

默认可先试：

```text
quantile_threshold = 0.75
```

### avoid_in_sector

```text
future_return_3d <= sector_median_future_return_3d
or max_drawdown_3d < drawdown_floor_3d
```

### pairwise_outperform_label

对同一 `trade_date + sector` 内 ETF 两两构造：

```text
pairwise_outperform_label(A, B) = 1
if future_return_3d(A) > future_return_3d(B)
and max_drawdown_3d(A) >= drawdown_floor_3d
```

Pairwise 样本必须保留 group 信息，避免跨板块或跨日期比较。

## 排序目标

至少设计三类目标：

- regression: `expected_3d_return`
- classification: `top_quantile_in_sector_3d`
- pairwise ranking: ETF A 是否优于 ETF B
- 可选：LambdaRank / rank objective

建议第一阶段同时输出收益排序与风险约束排序，避免只追逐短期涨幅。

## 特征候选

候选特征只能来自 `trade_date` 当日或之前：

- ETF momentum / acceleration。
- ETF liquidity / amount。
- ETF volatility。
- sector-relative strength。
- `distance_to_sector_leader`。
- crowding / concentration。
- ML score。
- risk flags。
- gap / previous close features。
- V2 原选 ETF 标记。
- sector rank / sector breadth / sector momentum。
- ETF 在同板块内的当日可见排名、成交额分位、波动率分位。

## 禁止进入 feature 的字段

- `future_return_1d`
- `future_return_3d`
- `max_drawdown_1d`
- `max_drawdown_3d`
- `best_in_sector_1d`
- `best_in_sector_3d`
- `top_quantile_in_sector_3d`
- `avoid_in_sector`
- `pairwise_outperform_label`
- 未来窗口内最优 ETF、未来收益排名、未来 drawdown 排名。

## 模型建议

首轮建议使用表格 ranking / regression，不先上 PyTorch：

- LightGBM ranking / regression。
- CatBoost ranking / classification。
- XGBoost ranking。

输出只能作为 shadow / advisory：

- `sector_internal_shadow_rank`
- `sector_internal_rank_score`
- `sector_internal_pick_reason`
- `sector_internal_risk_note`

不得直接替代 Stable entry。

## 时间切分

正式评估必须按日期 walk-forward：

- `train_date <= cutoff_date`
- `validate_date > cutoff_date`
- 不允许随机打乱日期作为正式评估。
- 同一天同板块所有 ETF 必须留在同一个 split，避免 group leakage。
- Pairwise ranking 的 A/B 样本必须继承同一个 `ranking_group_id`。
- 所有模型报告必须记录 `model_version`、`feature_version`、训练区间、验证区间。

## Group Split 规则

排名研究的最小不可拆分单位是：

```text
ranking_group_id = trade_date + sector
```

规则：

- 同一个 `ranking_group_id` 不得同时出现在 train 和 validate。
- 同一天不同 sector 可以进入同一 split，但不能跨日期打乱验证顺序。
- 若使用 sector top1 / topN 历史，必须确认该 topN 是当日可见，不由未来收益决定。
- 若样本太少，优先扩大日期窗口，不拆 group。

## 未来函数检查

- `future_return` 只能作为 label。
- sector 当日排序不能用未来价格。
- group 内 ranking label 只能来自未来窗口。
- 不能把未来最优 ETF 特征泄漏进当日样本。
- 当日 feature 必须由 `trade_date` 当日或之前数据计算。
- 同板块内部分位特征必须使用当日可见值，不得使用 T+1 / T+3 表现。

## 未来可输出文件

未来研究任务可生成以下文件，但本任务不实际生成这些运行产物：

- `sector_internal_etf_ranking_research.csv`
- `sector_internal_etf_ranking_report.md`
- `sector_internal_etf_ranking_report.json`
- `sector_internal_shadow_ranking.csv`
- `sector_internal_model_diagnostics.json`

大 CSV、训练明细和模型诊断应写入 `.local_research_outputs/aetfq3_lab/`；小型方法摘要可进入 `docs/research/`。

## Stable 边界

- Lab 可以建议 Stable，但不能直接改 Stable。
- 排序模型不得替代 Stable entry。
- 不修改 `final_buy_action`。
- 不修改 `target_weight`。
- 不修改 BUY / PROBE 阈值。
- 不生成 Stable 正式 `OrderIntent`。
- 不绕过 `RiskGate`。
- 若未来建议进入 Stable，只能提交最小合并方案、风险点、Stable 侧人工审批要求、`RiskGate` 检查点和回滚方案。

## Review Checklist 映射

- 研究了什么：设计同板块 ETF 内部排序数据口径。
- 数据来自哪里：本文只列未来可用输入，不读取真实数据。
- 是否来自 Stable bundle：本任务否；未来若使用必须显式标注。
- 是否有未来函数：本文定义 group split 和 label / feature 隔离要求。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：是。
- 是否建议进入 Stable：否，设计阶段不建议。
- 如果建议进入 Stable，最小合并方案是什么：不适用。
- 不允许直接提交到 Stable：确认不允许。
