本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。

# AETF Q3 Lab Intraday Provider Static Review Block Summary

## 任务定位

本任务是 Lab-only provider static review 阻塞总结文档。它只整理 F-7 真实行情源 provider 静态审查结果，不是真实 QMT 接入，不是交易执行，不实现 real provider wrapper，不导出真实 5分钟K。

默认边界：

```text
access_mode: READ_ONLY
final_action_change_allowed: false
contains_live_order: false
contains_secret: false
requires_human_review: true
promotion_gate_required: true
```

本总结的证据来自 ignored 静态审查报告：

- `.local_research_outputs/aetfq3_lab/intraday_provider_static_review/provider_static_review.md`
- `.local_research_outputs/aetfq3_lab/intraday_provider_static_review/provider_static_review.json`
- `.local_research_outputs/aetfq3_lab/intraday_provider_static_review/provider_candidate_inventory.csv`

同时对照已提交的 F 方向 research 文档：

- `docs/research/aetfq3_intraday_market_data_adapter.md`
- `docs/research/aetfq3_intraday_real_provider_template.md`
- `docs/research/aetfq3_qmt_readonly_5m_export_plan.md`
- `docs/research/aetfq3_intraday_real_data_source_inventory.md`

## 当前 F 方向状态

- F-0 completed
- F-1 completed
- F-2 completed
- F-3 completed
- F-4 blocked because no safe adapter
- F-5 completed
- F-6 completed
- F-7 blocked because no safe real provider

## 静态审查结论

- candidates reviewed: 17
- safe real 5m provider candidates: 0
- unsafe candidates: 9
- not intraday provider: 8
- requires manual review: 0

当前没有静态安全且可证明提供真实 5分钟K readonly/export_only 接口的 provider 文件。F-7 不能向 F-8 real provider wrapper 实现推进。

## 不安全原因

- QMT adapter / readonly smoke / mock broker contain account / positions / orders / trades or order paths
- `data/quotes.py` is quote snapshot, not 5m sequence
- `data/daily_export.py` is daily, not 5m

静态审查还显示，部分候选虽然与 intraday、5m、quote 或 market data 相关，但属于 mock adapter、template only、执行占位、broker contract、readonly account/order/trade smoke，不能作为 real 5分钟K provider 输入。

## 当前裁决

- F-8 real provider wrapper not allowed
- do not call existing QMT adapter
- do not use qmt_readonly_smoke
- do not use mock_broker
- do not proceed to real intraday dry validation

本裁决不改变 Stable，不生成 advisory package，不触发任何 QMT、OrderIntent、账户、持仓、委托、成交或真实 5分钟K 导出流程。

## 恢复条件

恢复 F-7 / F-8 前，必须同时满足：

- independent provider file
- only 5m market data methods
- no account / position / order / trade
- no submit / cancel
- no secret
- safety scan safe=true
- human authorization
- output only ignored directory

恢复后仍只能进入 Lab-only readonly/export_only 审查流程，不得自动进入 Stable。

## 边界

- no Stable
- no QMT trading
- no OrderIntent
- no output/
- no lab_advisory/
- no model training
- no trading advice

## 下一步建议

- 人工提供独立 provider 文件后重跑 F-7
- 或继续 synthetic/mock intraday dry validation tooling
- 不进入 Stable

## 任务结束核对

- 研究了什么：F-7 真实行情源 provider 静态审查阻塞结论。
- 数据来自哪里：ignored 静态审查报告和已提交 `docs/research` F 方向文档。
- 是否来自 Stable bundle：否。
- 是否有未来函数：否；本任务不读取行情序列，不做特征或标签计算。
- 是否影响 Stable 正式交易：否。
- 是否只读 advisory：是，仅为 Lab-only READ_ONLY 静态总结。
- 是否建议进入 Stable：否。
- 如果建议进入 Stable，最小合并方案是什么：不建议进入 Stable，无最小合并方案。
- 不允许直接提交到 Stable：确认不提交 Stable。
