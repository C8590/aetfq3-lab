# AETF Q3 Lab A False Downgrade Reconstructed Block Summary

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

本记录是 A false downgrade reconstructed 线的 Lab-only reconstructed diagnostic 阻塞总结。

它不是 true historical，不是 Stable，不是 advisory，不是交易信号，不训练模型，不保存模型，不生成 checkpoint，不接 QMT，不生成 OrderIntent，不写 `output/`，不创建 `lab_advisory/`。

## 数据源

- AKShare ETF daily OHLCV via `data.downloader.download_etf_history()`
- reconstructed V2 core actions: `.local_research_outputs/aetfq3_lab/false_downgrade_reconstructed/reconstructed_v2_core_actions.csv`
- reconstructed ML actions: `.local_research_outputs/aetfq3_lab/false_downgrade_reconstructed/reconstructed_ml_actions.csv`
- reconstructed false downgrade samples: `.local_research_outputs/aetfq3_lab/false_downgrade_reconstructed/false_downgrade_reconstructed_samples.csv`
- feature expand diagnostic: `.local_research_outputs/aetfq3_lab/false_downgrade_reconstructed/feature_expand/`
- action intersection diagnostic: `.local_research_outputs/aetfq3_lab/false_downgrade_reconstructed/action_intersection_diagnostic/`
- `uses_stable_bundle=false`

## 已完成

- reconstructed V2 action rows=1600
- reconstructed ML action rows=1600
- reconstructed false downgrade sample rows=12
- intake/schema/dry validation passed
- past-only features=23
- baseline precheck blocked
- action intersection diagnostic completed

## Action Intersection 结论

Existing exact cross table:

| V2 action | ML action | Rows |
| --- | --- | ---: |
| BUY | KEEP_ORIGINAL | 46 |
| BUY | NO_BUY | 3 |
| PROBE | KEEP_ORIGINAL | 42 |
| PROBE | NO_BUY | 9 |
| OBSERVE | NO_BUY | 1127 |
| OBSERVE | UPGRADE_PROBE | 17 |
| NO_BUY | NO_BUY | 334 |
| NO_BUY | UPGRADE_PROBE | 22 |

A_existing_exact_20etf_80dates has 1600 joined action rows, but only 100 V2 BUY/PROBE rows. Of those 100 V2 BUY/PROBE rows, 88 are ML KEEP_ORIGINAL and only 12 are ML NO_BUY. The false downgrade definition requires both V2 BUY/PROBE and ML downgrade/no-buy, so OBSERVE|NO_BUY and NO_BUY|NO_BUY rows are not legal candidates.

## 阻塞原因

- legal candidates = 12
- row_count < 30
- V2 BUY/PROBE 本身少
- ML reconstructed rule 对 V2 BUY/PROBE 大多 KEEP_ORIGINAL
- expanded diagnostic 未可靠突破 30
- not true historical
- not true ML_SIM

## 当前裁决

- A reconstructed no-save smoke 不允许
- A reconstructed baseline 不继续
- A 方向封存，等待 true historical V2/ML_SIM

## 恢复条件

- `entry_signal.csv` 或 `pre_selection_result.csv`
- `ml_sim_daily_comparison.csv/json` 或 `ml_sim_review_queue.csv`
- `ml_entry_scores.csv`
- V2/ML action 对照
- future label source

## 边界

- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model training
- no trading advice

## 下一步建议

- 回到 E 扩样 / 多窗口 / 固定输入复验
- 或转 F intraday 方案
- 不进入 Stable

## 任务结束回答

- 研究了什么：A false downgrade reconstructed 线的样本量、action intersection 和 baseline 阻塞原因。
- 数据来自哪里：允许读取的 `.local_research_outputs/aetfq3_lab/false_downgrade_reconstructed/`、`feature_expand/`、`action_intersection_diagnostic/` ignored 报告，以及其中记录的 AKShare ETF daily OHLCV 来源。
- 是否来自 Stable bundle：否，`uses_stable_bundle=false`。
- 是否有未来函数：action reconstruction 和 past-only features 不使用未来字段；future labels 只作为后验标签来源，不进入 action/feature 输入。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：本文件不是 advisory package；本任务只读诊断报告并写 docs/research 总结。
- 是否建议进入 Stable：否。
- 如果建议进入 Stable，最小合并方案是什么：不建议进入 Stable，因此无合并方案。
- 不允许直接提交到 Stable：已遵守。
