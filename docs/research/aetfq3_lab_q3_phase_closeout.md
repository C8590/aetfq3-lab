本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research closeout，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Phase Closeout

## 任务定位

本 closeout 是 `aetfq3-lab` 对 A / E / F 三条研究线的 Q3 Lab 阶段性总收口。目标不是继续研究，也不是形成交易建议，而是把已经完成的 Lab-only 工程链路、仍然阻塞的真实数据条件、以及不得进入 Stable 的边界一次性写清楚。

本 closeout 不读取真实账户、资金、持仓、委托、成交，不连接 QMT，不训练模型，不生成 advisory package，不生成 OrderIntent，不修改 Stable，不修改 `final_buy_action` / `target_weight` / BUY-PROBE 阈值。

## 总体结论

- E 方向：sector internal ranking 已完成 Lab-only 工程链路；小样本、expanded sample、多窗口 smoke、replay config 均已完成。结论是工程链路可用，但不是交易建议，不进 Stable。
- A 方向：false downgrade true historical 线尚未完成，因为缺真实 V2 / ML_SIM；reconstructed v2 线完成 no-save smoke。结论是 A-R 完成，A-H 阻塞，不能当作真实 ML 误杀证据。
- F 方向：intraday / 5分钟K / PyTorch 方向完成 F-0 到 F-9；mock/synthetic dry validation 工具链完成。真实 provider 线仍阻塞，因为没有 safe real 5m provider。结论是 F synthetic chain 完成，real intraday blocked。

## E 方向 Closeout

E 方向目标是 sector internal ranking 的 Lab-only 工程链路验证。

已完成：

- sector internal ranking sample generator
- table ML intake checker
- schema validator
- dry validation orchestrator
- baseline smoke
- report reader
- 小样本 smoke
- expanded sample smoke
- 20 / 40 / 60 / 90 多窗口 smoke
- replay config template

证据文档：

- `docs/research/aetfq3_sector_internal_ranking_baseline_smoke_summary.md/json`
- `docs/research/aetfq3_sector_internal_ranking_expanded_smoke_summary.md/json`
- `docs/research/aetfq3_sector_internal_ranking_window_smoke_summary.md/json`
- `docs/research/aetfq3_sector_internal_ranking_replay_config.md/json`

收口结论：

- 工程链路可用。
- smoke 指标只证明代码路径和门禁链路可运行，不证明模型有效性。
- 不是交易建议。
- 不是 advisory package。
- 不进入 Stable。

## A 方向 Closeout

A 方向目标是 false downgrade / ML 误杀方向的诊断链路。

已完成：

- reconstructed v2 Lab-only 样本链路
- reconstructed rulebook / replay-equivalent audit
- false downgrade no-save smoke
- report reader boundary check

仍阻塞：

- true historical V2 / ML_SIM 不可用。
- 缺真实 `entry_signal.csv` / `pre_selection_result.csv`、`ml_sim_daily_comparison.csv/json` 或等价 ML_SIM 历史证据。
- reconstructed v2 不是 true historical，不是 true ML_SIM。

证据文档：

- `docs/research/aetfq3_false_downgrade_reconstructed_v2_smoke_summary.md/json`
- `docs/research/aetfq3_false_downgrade_reconstructed_block_summary.md/json`

收口结论：

- A-R completed。
- A-H blocked。
- reconstructed v2 no-save smoke 只能证明降级后的 Lab-only reconstructed 链路可跑。
- 不能作为真实 ML 误杀证据。
- 不进入 Stable。

## F 方向 Closeout

F 方向目标是 intraday / 5分钟K / PyTorch 执行研究链路。

已完成：

- F-0 到 F-9 阶段性文档与工具链
- 5分钟K data contract
- dry validation manifest
- Intraday Watch state-machine
- PyTorch intraday execution model plan
- market data adapter safety gate
- real provider template
- provider static review blocker summary
- synthetic/mock dry validation tooling
- synthetic dry validation summary

mock/synthetic dry validation 完成链路：

- manifest intake checker
- 5m schema validator
- forbidden feature scan
- tensor shape dry validation
- Intraday Watch state machine dry-run
- dry validation orchestrator
- report reader

F-8 / F-9 结果：

- specified tests: 17 passed
- tests/lab: 118 passed, 2 warnings
- orchestrator smoke: passed
- reader smoke: passed
- no OrderIntent
- no QMT
- no Stable
- no checkpoint
- no model save

真实 provider 静态审查：

- candidates reviewed: 17
- safe real 5m provider candidates: 0
- unsafe candidates: 9
- not intraday provider: 8
- requires manual review: 0

证据文档：

- `docs/research/aetfq3_intraday_synthetic_dry_validation.md/json`
- `docs/research/aetfq3_intraday_synthetic_dry_validation_summary.md/json`
- `docs/research/aetfq3_intraday_provider_static_review_block_summary.md/json`

收口结论：

- F synthetic chain completed。
- real intraday blocked。
- 当前没有 safe real 5m provider。
- 当前没有 human-exported 5m data package。
- 不进入 Stable。

## 当前统一阻塞

- 缺真实数据。
- 缺 true historical V2 / ML_SIM。
- 缺 safe real 5m provider。
- 缺人工导出的 5分钟K 数据包。

这些阻塞未解除前，Lab 不能将 A/E/F 任一方向提升为 Stable 输入、交易建议、advisory package、正式模型证据或真实执行链路。

## 下一阶段恢复条件

下一阶段若继续推进，至少需要：

- true historical V2 / ML_SIM 证据包。
- 独立 safe provider 文件，且只包含 5分钟K market-data 方法。
- 或人工导出的 5分钟K package。
- 对人工 export package 补齐 manifest + hash + source_note。
- 对真实数据先执行 intake-only 检查，再考虑 dry validation。
- 对任何 provider 文件先重跑 F-7 static review。

## 永久边界

- 不进入 Stable。
- 不生成 advisory。
- 不接 QMT。
- 不生成 OrderIntent。
- 不自动下单。
- 不修改 Stable。
- 不修改 `final_buy_action`。
- 不修改 `target_weight`。
- 不修改 BUY / PROBE 阈值。
- 不写 Stable runtime/output。
- 不写 `output/`。
- 不创建 `lab_advisory/`。
- 不提交 `.local_research_outputs/`。
- 不把 Lab smoke 或 mock/synthetic 结果解释为交易建议。

## 任务结束核对

- 研究了什么：A / E / F 三条 Lab 研究线的 Q3 阶段性工程 closeout。
- 数据来自哪里：已提交 `docs/research` summary/block 文档与 ignored F-8 smoke report 摘要。
- 是否来自 Stable bundle：否。
- 是否有未来函数：本 closeout 不做特征计算；既有 A/E/F 文档均要求 future label / forbidden feature scan，不把 future label 当 feature。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：这是 Lab-only READ_ONLY closeout，不生成 advisory package。
- 是否建议进入 Stable：否。
- 如果建议进入 Stable，最小合并方案是什么：不建议进入 Stable，无最小合并方案。
- 不允许直接提交到 Stable：确认不提交 Stable。
