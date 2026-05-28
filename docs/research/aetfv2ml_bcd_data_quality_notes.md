- 本文属于 aetfv2ml / V2.1 ML Lab。
- 本文是日线重建版研究摘要。
- 本文不属于 V2.1 Stable。
- 本文不得修改 Stable entry。
- 本文不得影响 final_buy_action。
- 本文不得修改 BUY / PROBE 阈值。
- 本文不得接入 QMT。
- 本文结论只能作为 shadow/advisory 和人工审阅材料。

# Clean B/C/D Data Quality Notes

本说明服务于日线重建版 ML Lab shadow/advisory 小报告。

## 已确认风险

- 单 ETF 单日 `abs(return_1d)>20%`：156 条。
- 疑似价格断点 / 复权 / 高低价异常事件：1506 条。
- 单成员或少于 3 只 ETF 的 sector-date：7009 行。
- `行业未录入` 在异常样本中占比高，需要人工复查分类。

## 对结论的影响

- 156 条 >20% 异常不足以单独推翻 B/C/D，但足以要求尾部回撤降级解读。
- 1506 条疑似异常事件要求所有 max drawdown / tail 指标附带数据质量风险说明。
- 7009 行稀疏 sector 样本是 top1 定义切换的主要风险源。
- 后续建议做行情复权 / 价格断点修复或隔离标记，但本轮没有修改原始行情。

## 后续人工复查重点

- `行业未录入` 是否应从 sector 排名池中单独分层。
- `sector_etf_count >= 3` 或 `>= 5` 的样本门槛是否作为研究默认过滤。
- 成交额过低、volume/amount 缺失样本是否应单独标记。
