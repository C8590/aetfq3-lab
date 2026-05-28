- 本文属于 aetfv2ml / V2.1 ML Lab。
- 本文是日线重建版研究摘要。
- 本文不属于 V2.1 Stable。
- 本文不得修改 Stable entry。
- 本文不得影响 final_buy_action。
- 本文不得修改 BUY / PROBE 阈值。
- 本文不得接入 QMT。
- 本文结论只能作为 shadow/advisory 和人工审阅材料。

# AETFv2 ML Lab B/C/D 清洗后研究摘要

生成时间：2026-05-28T12:04:07  
主控 Agent：aetfv2_08_control_center  
来源：`.local_research_outputs/aetfv2ml/clean_bcd_summary/clean_bcd_summary_report.md` 与同目录 JSON 摘要。  
迁移范围：仅迁移 Markdown / JSON 摘要；不迁移 CSV、data、artifacts、output 或本地研究目录。

## 研究范围

本摘要迁移的是已经完成并准备人工审阅的 B/C/D 小报告，定位为 aetfv2ml / V2.1 ML Lab 日线重建版研究材料。研究对象包括 B 板块 top1 定义、C lock period 观察、D crowded_late_stage 风险提示，以及相关数据质量风险。

本摘要不覆盖 A false downgrade、E ML 排序、F 5分钟K 执行的正式结论；这些方向仍因数据和 artifacts 缺失保持阻塞。

## 数据口径

- 使用 `data/etf_daily.csv` 的既有研究口径，但本次迁移不读取或提交原始 data 文件。
- 使用 B/C/D 日线重建、稳健性复核、异常复核后的摘要结论。
- 不等于 V2.1 Stable 口径。
- 不等于 historical_ml 原始 artifacts 口径。
- 未使用 ML_SIM。
- 未使用 ml_entry_scores。
- 未使用 V2 core 多日 entry_signal。
- 未修改原始行情。
- `sector_etf_count >= 3` 是必要研究门槛。
- `行业未录入` 应单独分层，不应混入普通 sector 结论。

## 保留结论

- B/C/D 未被 156 条单 ETF 单日 `abs(return_1d)>20%` 异常完全推翻，可继续作为 ML Lab shadow/advisory 研究线。
- `breadth_adjusted_top1` 是清洗后第一优先级研究候选。
- `mom_3d_top1` 是第二优先级，在 combined conservative filter 下具备稳健性对照价值。
- `mom_5d_top1` 是第三优先级，适合作为低漂移 fallback 或 tie-breaker 研究参考。
- `crowded_late_stage` 可保留为人工复核提示，重点提示回撤风险。
- `lock3` / `lock5` 的尾部风险观察可保留，但只能用于风险研究。

## 降级结论

- `acceleration_top1` 降级为观察项，不能再表述为稳定最优。
- `acceleration_top1` 在 raw、exclude_abs_ret_gt_20、winsor_abs_ret_20、exclude_first_20_trading_days 下仍有研究价值，但在 `sector_etf_count >= 3` 与强保守过滤后不再是第一。
- `overheat_penalized_top1` 只作为风险侧 overlay，不适合作为主 top1 定义。
- top1 定义对 sector 稀疏样本敏感，必须加入 `sector_etf_count >= 3` 或分层说明。
- lock3 / lock5 的尾部回撤解释受异常点污染，需要随数据质量说明一起阅读。

## 阻塞结论

- A false downgrade 仍阻塞：缺 ML_SIM / V2 core / ml_entry_scores。
- E ML 排序仍阻塞：缺 ML score 和 V2 原选多日输出。
- F 5分钟K 执行仍阻塞：缺 5分钟K / VWAP / 盘口历史。
- 当前数据不能回答任何需要 V2 core、ML score、盘口或分钟级执行的问题。

## B 板块 top1 结论

清洗后 shadow/advisory 优先级：

| priority | definition | status | reason |
|---:|---|---|---|
| 1 | `breadth_adjusted_top1` | 清洗后主候选 | `sector_etf_count >= 3` 后 T+3 均值最高，说明广度 / 板块厚度对 B 结论重要；仍只限 shadow/advisory。 |
| 2 | `mom_3d_top1` | 强保守样本稳健对照 | combined conservative filter 后变优，切换率低于 breadth / acceleration，需要继续分年和 regime 复核。 |
| 3 | `mom_5d_top1` | 低漂移 fallback | 收益不最高，但切换率较低，适合作为低漂移对照或 tie-breaker 研究。 |
| 4 | `acceleration_top1` | 降级观察项 | 多数非稀疏去极值口径下仍领先，但高漂移，且稀疏过滤后不再第一。 |
| 5 | `overheat_penalized_top1` | 风险侧 overlay | 回撤较轻但样本数和板块偏向风险较大，不适合作为主 top1 定义。 |

按过滤口径的最优定义：

| filter_policy | best top1 | conclusion |
|---|---|---|
| raw | `acceleration_top1` | no_core_conclusion_change |
| exclude_abs_ret_gt_20 | `acceleration_top1` | no_core_conclusion_change |
| winsor_abs_ret_20 | `acceleration_top1` | no_core_conclusion_change |
| exclude_first_20_trading_days | `acceleration_top1` | no_core_conclusion_change |
| exclude_sector_single_member_days | `breadth_adjusted_top1` | acceleration_not_leading |
| combined_conservative_filter | `mom_3d_top1` | acceleration_not_leading |

结论：`acceleration_top1` 仍可保留，但只能降级为观察项；`breadth_adjusted_top1` / `mom_3d_top1` 在稀疏和强保守过滤下变优，说明 B 结论依赖板块厚度和样本质量。

## C lock period 结论

- lock1：风险收益更平衡，是最短暴露基准。
- lock2：可作为中间观察窗口。
- lock3：收益观察可保留，但回撤和尾部风险更高。
- lock5：均值可能更高，但回撤暴露最重，对异常点更敏感。
- 未看到“第三天稳定下跌”规律。
- lock3 / lock5 不能形成正式持有或退出规则，只能作为 ML Lab shadow/advisory 风险提示。

## D crowded_late_stage 结论

- `crowded_late_stage` 是回撤风险提示，不是稳定负收益信号。
- 去极值后结论保留：仍可作为人工复核提示。
- 不应作为交易否决。
- 避免误杀 V2 core 强主线：必须同时查看趋势持续、广度、回撤扩散、sector 厚度和 V2 core 缺失数据。
- 缺 V2 core 前只能保持提示级别。
- 推荐表述：`crowded_late_stage is a ML Lab shadow/advisory drawdown-risk prompt, not a veto signal`。

## 数据质量风险

- 156 条单 ETF 单日 `abs(return_1d)>20%` 异常不足以推翻 B/C/D，但足以要求尾部指标降级。
- 1506 条疑似价格断点 / 复权 / 高低价异常事件要求 max drawdown / tail 指标附带质量风险说明。
- 7009 行 sector 稀疏样本应通过 `sector_etf_count >= 3` 或分层报告处理。
- `行业未录入` 在异常样本中占比高，需要人工复查分类并单独分层。
- 后续应做行情复权 / 价格断点修复或隔离标记；本迁移任务未修改原始行情。

## shadow/advisory 边界

- 仅用于人工审阅、离线研究排序、数据质量复查和后续实验设计。
- 不能进入 V2.1 Stable。
- 不能修改 Stable entry。
- 不能影响 `final_buy_action`。
- 不能修改 BUY / PROBE 阈值。
- 不能接入 QMT、miniQMT、XtQuant 或 OrderIntent。
- 不能进入真实交易。

## A/E/F 仍阻塞

- A false downgrade：缺 ML_SIM / V2 core / ml_entry_scores。
- E ML 排序：缺 ML score 和 V2 原选多日输出。
- F 5分钟K 执行：缺 5分钟K / VWAP / 盘口历史。

## 不允许事项

- 不提交 CSV。
- 不提交 `data/etf_daily.csv`。
- 不提交 `.local_research_outputs/`。
- 不提交 `artifacts/`。
- 不写 `output/`。
- 不修改 Stable。
- 不修改 Stable entry。
- 不修改 `final_buy_action`。
- 不修改 BUY / PROBE 阈值。
- 不接 QMT。
- 不生成 OrderIntent。
- 不进入真实交易。
- 不把 shadow/advisory 写成正式规则。

## 下一步建议

- 人工审阅本次迁移的四个 Markdown / JSON 小文件。
- 对 `行业未录入` 做独立分层复查。
- 继续验证 `sector_etf_count >= 3` 或更高门槛是否应作为研究默认过滤。
- 对 >20% 单日异常、疑似价格断点 / 复权问题做隔离标记方案。
- 继续寻找 A/E/F 所缺的 ML_SIM、V2 core、ml_entry_scores、5分钟K / VWAP / 盘口历史数据。
