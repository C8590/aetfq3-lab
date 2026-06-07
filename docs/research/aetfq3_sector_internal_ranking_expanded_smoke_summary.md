# AETF Q3 Lab E Sector Internal Ranking Expanded Smoke Summary

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

本记录是 E sector internal ranking expanded sample 的 Lab-only smoke summary，作为 E 方向第二阶段基线记录。

它不是 Stable，不是 advisory，不是交易信号，不训练正式模型，不保存模型，不生成 checkpoint，不接 QMT，不生成 OrderIntent，不写 `output/`，不创建 `lab_advisory/`。

## 数据源

- AKShare ETF daily OHLCV
- `config/etf_sector_map.yaml`
- `uses_stable_bundle=false`
- 输入样本目录：`.local_research_outputs/aetfq3_lab/sector_internal_ranking_expanded/akshare_tool_smoke/`
- 完整 smoke 目录：`.local_research_outputs/aetfq3_lab/sector_internal_ranking_expanded/akshare_full_smoke/`

## 样本摘要

- rows=1200
- date_count=60
- ETF count=20
- sector count=5
- group_count=300
- feature_count=25
- missing_rate=0.0
- train_count=840
- valid_count=360
- split_method=`chronological`
- group_leakage_check=`passed`

## 校验链路

- intake passed
- schema passed
- dry validation passed
- reader passed

Dry validation summary:

- status=`passed`
- rows_checked=1200
- intake_passed=true
- schema_passed=true
- p0_blockers=[]
- p1_warnings=[]

Reader summary:

- status=`OK`
- boundary_passed=true
- model_count=4

## 四模型 smoke

| Model | Status | Accuracy | ROC AUC | Log Loss |
| --- | --- | ---: | ---: | ---: |
| numpy_logistic | passed | 0.4639 | 0.4819 | 0.7041 |
| lightgbm | passed | 0.5222 | 0.5437 | 0.6944 |
| catboost | passed | 0.4972 | 0.5220 | 0.6959 |
| xgboost | passed | 0.5167 | 0.5030 | 0.7046 |

Metrics only validate smoke/code path and do not prove model effectiveness. These metrics are not trading advice, not model acceptance evidence, and not a Stable promotion signal.

## 边界声明

- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model save
- no checkpoint
- no trading advice

## 下一步建议

- 扩大 Lab-only sample
- 固定输入清单和依赖版本
- 做多窗口对比
- 不进入 Stable

## Review Checklist 自检

- 研究了什么：AKShare expanded E sector internal ranking sample 的 dry validation、四模型 no-save smoke、reader contract。
- 数据来自哪里：AKShare ETF daily OHLCV 与 `config/etf_sector_map.yaml`。
- 是否来自 Stable bundle：否，`uses_stable_bundle=false`。
- 是否有未来函数：future return / drawdown 只作为 label/outcome，不进入 feature。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：否，本文件只是 research summary，不是 advisory 包。
- 是否建议进入 Stable：否。
- 如果建议进入 Stable，最小合并方案是什么：无，本记录不建议进入 Stable。
- 不允许直接提交到 Stable：是。
- 人工复核要求：继续扩大样本或形成任何 promotion 讨论前必须人工复核。
