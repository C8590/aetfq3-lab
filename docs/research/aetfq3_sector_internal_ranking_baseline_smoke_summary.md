# AETF Q3 Lab Sector Internal Ranking Baseline Smoke Summary

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

本总结记录 E sector internal ranking 从真实小样本到四模型 no-save smoke 的 Lab-only 链路。它只总结研究和工程验证状态，不是正式训练，不是模型验收，不是 advisory 包，不是 Stable 输入，不是交易信号，也不连接 QMT 或生成 OrderIntent。

## 完成链路

- sample intake：真实小样本 manifest 通过 Lab sample intake checker，边界为 `training_allowed=false`、`stable_effect_allowed=false`、`advisory_only=true`、`affects_stable_trading=false`。
- schema validator：E sector internal ranking 样本 schema 校验通过，`rows_checked=60`。
- dry validation：真实 feature sample dry validation 通过，`status=passed`、`intake_passed=true`、`schema_passed=true`、`p0_blockers=[]`、`p1_warnings=[]`。
- past-only feature sample：生成 25 个 lag1 / past-only 数值特征，禁止 future label、id/group、ETF 身份和原始 sector 字符串进入 feature。
- baseline precheck：baseline precheck 通过，chronological split 与 group split 条件可满足，`baseline_smoke_allowed=true`。
- no-save baseline smoke：baseline smoke 工具以 chronological split 运行 no-save code path validation，不保存模型，不调参，不生成 checkpoint。
- report reader contract：baseline report reader 校验 Lab-only / no-save / no Stable / no QMT / no OrderIntent / no output / no lab_advisory 合同字段，禁止交易相关字段。
- four-model `.venv` smoke：使用 Lab `.venv` 运行 `numpy_logistic`、`lightgbm`、`catboost`、`xgboost` 四个极小型 no-save smoke，四模型均 `passed`。

## 数据口径

- 数据来自 AKShare ETF 日线和本地 sector map，用于 Lab-only 小样本与 past-only feature 生成。
- `uses_stable_bundle=false`。
- 不读取 Stable output。
- 不接 QMT。
- 不生成 OrderIntent。
- 不写 `output/`。
- 不生成 advisory 包。

## 样本摘要

- rows: 60
- date range: 2026-05-20 到 2026-06-02
- ETF count: 6
- sector count: 3
- group count: 30
- feature count: 25
- missing rate: 0
- target label: `top_quantile_in_sector_3d`
- train / validation split: 前 70% 日期训练，后 30% 日期验证；no shuffle；group leakage check passed

## 模型 Smoke 摘要

四模型 no-save smoke 均 passed。Metrics only validate code path, not model effectiveness. 这些指标只说明读取样本、feature contract、chronological split、模型内存训练和评估输出链路可运行，不代表模型有效性、策略收益或 Stable 可用性。

## 四模型指标

| model | status | accuracy | roc_auc | log_loss |
| --- | --- | ---: | ---: | ---: |
| numpy_logistic | passed | 0.3333 | 0.2716 | 1.6458 |
| lightgbm | passed | 0.3333 | 0.3889 | 0.9611 |
| catboost | passed | 0.3333 | 0.3086 | 1.0178 |
| xgboost | passed | 0.3333 | 0.2963 | 1.0364 |

## 边界

- not a model acceptance
- not a trading signal
- not Stable input
- not advisory package
- no model saved
- no checkpoint
- no OrderIntent
- no Stable
- no QMT
- no output/
- no lab_advisory/
- metrics 不得解释为模型有效性或交易建议

## Review Checklist 自检

1. 研究了什么：E sector internal ranking 从真实小样本、dry validation、feature contract、baseline precheck 到四模型 no-save smoke 的 Lab-only 工程链路。
2. 数据来自哪里：AKShare ETF 日线、本地 sector map、Lab ignored 小样本与报告。
3. 是否来自 Stable bundle：否，`uses_stable_bundle=false`。
4. 是否有未来函数：feature contract 禁止 future return、drawdown、label、id/group、ETF 身份和原始 sector 字符串进入 feature；当前 25 个 feature 为 lag1 / past-only。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：否，本总结不是 advisory 包，也不生成 advisory。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用；当前阶段不进入 Stable。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议：只在 Lab 内扩大样本日期与 sector 覆盖，继续保留 intake / schema / dry validation / report reader gate。

## 下一步建议

- 扩大 Lab-only sample。
- 增加更多日期和 sector。
- 保持 intake、schema、dry validation、report reader gate。
- 不进入 Stable。
