本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab PyTorch Intraday Execution Model Plan

## 任务定位

PyTorch 执行模型只研究候选 ETF 的盘中执行时点和取消观察，不负责选 ETF，不修改 Stable entry，不修改 `final_buy_action`、`target_weight` 或 BUY / PROBE 阈值。

本文件是第一阶段方案，不读取真实 5分钟K 数据，不训练模型，不运行 `torchrun`，不保存模型，不生成 checkpoint，不接 QMT，不生成 `OrderIntent`。

## 模型目标

- `p_buy_now`：当前 bar 作为研究执行点的概率。
- `p_wait_pullback`：继续等待回落的概率。
- `p_cancel_buy`：取消买入观察的概率。
- `p_three_day_positive`：T+3 为正的概率，仅作 outcome 学习目标。
- `expected_3d_return`：预期 3日收益回归目标。
- `expected_3d_drawdown`：预期 3日回撤回归目标。

这些目标是 Lab research target，不是交易动作，不是正式模型效果声明。

## 候选模型

- logistic / MLP baseline：作为 mock tensor smoke 的最小路径。
- GRU：建模 5分钟序列的短期记忆。
- TCN：建模固定窗口局部时序结构。
- small Transformer：用于观察注意力机制在盘中窗口上的可行性。
- temporal CNN：轻量卷积序列 baseline。
- tabular + sequence hybrid：结合候选上下文和 bar 序列。

## 输入张量建议

- shape: `[batch, time_steps, features]`
- time_steps: `6 / 12 / 24 / 48`
- bar interval: `5m`
- feature groups:
  - price
  - volume
  - vwap
  - open range
  - sector context
  - candidate context

禁止把 `future_*`、`*_label`、`execution_return_*`、`max_drawdown_3d` 或真实成交结果放入输入张量。

## 训练切分规则

- 使用 chronological split。
- 使用 walk-forward 作为后续更严格评估。
- formal evaluation 不允许 random shuffle。
- 同一 `trade_date` 不得在同一场景中同时进入 train / validation。
- 必要时按 `trade_date` / `etf_code` / `sector` 做 group split。

## Loss 设计

- `p_buy_now` / `p_cancel_buy` 使用 binary classification loss。
- `expected_3d_return` / `expected_3d_drawdown` 使用 regression loss。
- 可选 multi-task loss，但必须记录各任务权重。
- 可选 class imbalance handling，但不能以此解释为模型有效性证明。

## 评估指标

- AUC。
- accuracy。
- precision / recall。
- calibration。
- MAE / RMSE for expected return。
- drawdown error。
- decision bucket stability。

第一阶段指标只定义口径；第二阶段 mock smoke 指标只证明代码路径，不说明真实效果。

## 阶段门

1. 第一阶段：只做 plan、数据合同、状态机和 checklist，不训练。
2. 第二阶段：只做 mock tensor smoke，不读真实 5分钟K，不保存模型。
3. 第三阶段：单独授权后才允许真实 5分钟K dry validation。

任何阶段都不允许接 Stable、不允许接 QMT、不允许自动下单、不允许生成 OrderIntent、不允许保存正式模型。

