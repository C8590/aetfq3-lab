- 本文属于 aetfv2ml / V2.1 ML Lab。
- 本文是日线重建版研究摘要。
- 本文不属于 V2.1 Stable。
- 本文不得修改 Stable entry。
- 本文不得影响 final_buy_action。
- 本文不得修改 BUY / PROBE 阈值。
- 本文不得接入 QMT。
- 本文结论只能作为 shadow/advisory 和人工审阅材料。

# Clean B/C/D Shadow Advisory Rules

适用范围：日线重建版、ML Lab、shadow/advisory。不得进入 Stable；不得影响 final_buy_action；不得修改 BUY/PROBE 阈值；不得接入 QMT。

## 允许表达

- `breadth_adjusted_top1`：`sector_etf_count >= 3` 后的主研究候选，仅限 shadow/advisory。
- `mom_3d_top1`：combined conservative filter 下的稳健性对照，仅限 shadow/advisory。
- `mom_5d_top1`：低漂移 fallback / tie-breaker 研究参考。
- `acceleration_top1`：多数去极值口径下仍有研究价值，但在 sector 稀疏过滤和强保守过滤下需降级为观察项。
- `breadth_adjusted_top1` / `mom_3d_top1`：作为清洗后挑战者继续比较。
- `lock3` / `lock5`：只作为持有期尾部风险观察，不生成动作。
- `crowded_late_stage`：只作为人工复核提示，重点提示回撤风险。
- `sector_etf_count >= 3`：只作为 ML Lab 数据质量过滤口径。

## 禁止升级

- 不把任何 B/C/D 指标写入 Stable。
- 不让任何 B/C/D 结论影响 `final_buy_action`。
- 不用本报告调整 BUY/PROBE 阈值。
- 不把 `crowded_late_stage` 当成交易否决。
- 不把 `lock3` / `lock5` 当成动作规则。
- 不接入 QMT、miniQMT、XtQuant 或 OrderIntent。
