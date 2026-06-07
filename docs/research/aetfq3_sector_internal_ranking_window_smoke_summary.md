# AETF Q3 Lab E Sector Internal Ranking Window Smoke Summary

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

本记录是 E sector internal ranking 的 20 / 40 / 60 / 90 trading days Lab-only 多窗口 smoke 稳定性验证，作为 E 方向第三阶段工程稳定性记录。

它不是 Stable，不是 advisory，不是交易信号，不训练正式模型，不保存模型，不生成 checkpoint，不接 QMT，不生成 OrderIntent，不写 `output/`，不创建 `lab_advisory/`。

## 数据源

- AKShare ETF daily OHLCV
- `config/etf_sector_map.yaml`
- `uses_stable_bundle=false`
- no Stable output
- no QMT
- no OrderIntent
- 输入报告目录：`.local_research_outputs/aetfq3_lab/sector_internal_ranking_window_compare/`

## 窗口结果表

| window | rows | dates | ETF | sectors | groups | features | missing | intake | schema | dry | reader | models | leakage |
| ------ | ---: | ----: | --: | ------: | -----: | -------: | ------: | ------ | ------ | ------ | ------ | ------ | ------- |
| 20 | 400 | 20 | 20 | 5 | 100 | 25 | 0.0 | passed | passed | passed | OK | passed | passed |
| 40 | 800 | 40 | 20 | 5 | 200 | 25 | 0.0 | passed | passed | passed | OK | passed | passed |
| 60 | 1200 | 60 | 20 | 5 | 300 | 25 | 0.0 | passed | passed | passed | OK | passed | passed |
| 90 | 1800 | 90 | 20 | 5 | 450 | 25 | 0.0 | passed | passed | passed | OK | passed | passed |

## 四模型 smoke 汇总

20 trading days:

- numpy: acc 0.517, auc 0.494, logloss 0.708
- lightgbm: acc 0.550, auc 0.526, logloss 0.697
- catboost: acc 0.533, auc 0.523, logloss 0.695
- xgboost: acc 0.583, auc 0.597, logloss 0.680

40 trading days:

- numpy: acc 0.496, auc 0.472, logloss 0.726
- lightgbm: acc 0.483, auc 0.481, logloss 0.717
- catboost: acc 0.496, auc 0.495, logloss 0.701
- xgboost: acc 0.492, auc 0.514, logloss 0.706

60 trading days:

- numpy: acc 0.464, auc 0.482, logloss 0.704
- lightgbm: acc 0.522, auc 0.544, logloss 0.694
- catboost: acc 0.497, auc 0.522, logloss 0.696
- xgboost: acc 0.517, auc 0.503, logloss 0.705

90 trading days:

- numpy: acc 0.507, auc 0.509, logloss 0.697
- lightgbm: acc 0.518, auc 0.517, logloss 0.708
- catboost: acc 0.491, auc 0.485, logloss 0.701
- xgboost: acc 0.521, auc 0.527, logloss 0.704

Metrics only validate smoke/code path and do not prove model effectiveness. These metrics are not model acceptance evidence, not trading advice, and not a Stable promotion signal.

## 稳定性结论

- 20 / 40 / 60 / 90 全窗口工程链路通过。
- missing rate 全窗口为 0。
- reader contract 全窗口通过。
- group leakage 全窗口 passed。
- 四模型全窗口 no-save passed。
- 未发现 `catboost_info` 残留。

## 边界声明

- not a model acceptance
- not a trading signal
- not Stable input
- not advisory package
- no model saved
- no checkpoint
- no OrderIntent
- no QMT
- no output/
- no lab_advisory/

## 下一步建议

- Lab-only 扩大 ETF / sector 覆盖。
- 固定输入清单与依赖版本。
- 做多窗口复验。
- 不进入 Stable。
- 不生成 advisory 包。

## Review Checklist 自检

- 研究了什么：E sector internal ranking 20 / 40 / 60 / 90 trading days 多窗口 smoke 工程稳定性。
- 数据来自哪里：AKShare ETF daily OHLCV 与 `config/etf_sector_map.yaml`，读取本地 ignored smoke summary。
- 是否来自 Stable bundle：否，`uses_stable_bundle=false`。
- 是否有未来函数：本记录未新增样本或特征；沿用 smoke 报告中的 feature leakage check，group leakage 全窗口 passed。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：否，本文件是 research summary，不是 advisory 包。
- 是否建议进入 Stable：否。
- 如果建议进入 Stable，最小合并方案是什么：无，本记录不建议进入 Stable。
- 不允许直接提交到 Stable：确认不允许。
