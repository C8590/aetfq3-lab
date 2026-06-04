# AETF Q3 Lab Codex Task Template

## 任务头

每次给 Codex 派发本仓库任务前，必须先写：

```text
本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。
```

## 任务定位

```text
【任务类型】

【目标】

【允许修改】

【禁止事项】
1. 不修改 V2.1 Stable。
2. 不修改 Stable entry。
3. 不修改 final_buy_action。
4. 不修改 target_weight。
5. 不修改 BUY / PROBE 阈值。
6. 不接正式 QMT。
7. 不生成 Stable 正式 OrderIntent。
8. 不绕过 RiskGate。
9. 不自动下单。
10. 不运行正式策略，除非任务明确授权研究或回测。
11. 不写 output/，除非任务明确授权本地研究输出且该输出不提交。
12. 不提交 data/、artifacts/、.local_research_outputs/。
13. 不 force push。
14. 不把研究输出当正式交易计划。
15. 不把模型直接接入 Stable。
16. 不把 QMT 实验回写 Stable。
```

## Lab 可负责方向

任务可以落在以下 Lab 方向：

- `historical_ml`
- sector map
- ML false downgrade
- 强主线保护
- 同板块 ETF 排序
- 第一板块切入位置
- 5分钟K 回测
- Intraday Watch Engine
- 盘口特征
- QMT mock / readonly / 模拟盘
- PyTorch / GRU / TCN 执行模型
- Q3 / Q4 / Q5 前沿策略原型
- Lab advisory 报告

## Stable 输出边界

如需形成给 Stable 的输出，只能使用只读建议包：

- `ml_advisory_summary.json`
- `sector_research_report.json`
- `intraday_watch_research.json`
- `qmt_readonly_report.json`
- `model_diagnostics.json`
- `research_notes.md`

建议包不得直接修改 Stable 参数，不得触发正式交易，不得生成正式 `OrderIntent`。

## 建议检查命令

开始前：

```powershell
git status --short --branch
git diff --name-only
```

结束后：

```powershell
git status --short --branch
git diff --name-only
git diff --stat
```

## 结束必答

每个任务结束必须回答：

```text
## 修改范围
## 研究了什么
## 数据来自哪里
## 是否来自 Stable bundle
## 是否有未来函数
## 是否影响 Stable 正式交易
## 是否只读 advisory
## 是否建议进入 Stable
## 如果建议进入 Stable，最小合并方案是什么
## 不允许直接提交到 Stable
## P0 Blocker
## P1 高风险
## P2 建议
## 下一步任务卡
```

## Commit 前检查

提交前必须确认：

- diff 只包含任务授权文件。
- 未修改 Stable、策略代码、`final_buy_action`、`target_weight`、BUY / PROBE 阈值。
- 未生成正式 `OrderIntent`。
- 未接正式 QMT，未自动下单。
- 未写入或提交 `output/`。
- 未提交 `data/`、`artifacts/`、`.local_research_outputs/`。
- 未提交 `.env`、密钥、模型权重、大体积 csv 或 zip。
- 未 force push。
