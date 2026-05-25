# V2.1 Legacy Baseline Report

生成日期：2026-05-24  
主责：aetfv2_08_control_center  
协作：aetfv2_01_pre_selection、aetfv2_02_entry、aetfv2_05_historical_ml

## 审查范围

本次只读审查当前快照和代码边界：

- `output/pre_selection_result.csv`
- `output/entry_signal.csv`
- `output/historical_ml_summary.json`
- `output/historical_ml_summary.csv`：用于交叉验证 JSON 摘要行数
- `output/etf_universe_snapshot.csv`：仅用于统计全市场 ETF 数
- `signal/pre_selection/engine.py`
- `signal/entry/engine.py`
- `signal/v21_orchestrator.py`
- `.codex/config.toml`

未运行策略，未刷新行情，未触发 QMT，未修改 BUY/PROBE 阈值，未改变任何交易逻辑或 output 快照。

## 修改范围

只新增本基准报告：

- `baseline_v21_legacy_report.md`
- `baseline_v21_legacy_report.json`

## 基准指标

| 指标 | 当前值 | 证据 |
| --- | ---: | --- |
| 全市场 ETF 数 | 1458 | `output/etf_universe_snapshot.csv` 行数 |
| 有效样本数 | 1160 | `output/pre_selection_result.csv` 行数；`output/entry_signal.csv` 行数一致 |
| pre_selection 入选数 | 5 | `selected=True` |
| entry PROBE 数 | 5 | `raw_entry_action=PROBE` |
| entry OBSERVE 数 | 1155 | `raw_entry_action=OBSERVE` |
| final PROBE 数 | 5 | `final_buy_action=PROBE` |
| final BLOCKED 数 | 0 | `final_buy_action=BLOCKED` |
| NO_ML 数 | 0 | `ml_action_suggestion=NO_ML` |
| historical_ml 摘要数 | 9 | `output/historical_ml_summary.json` UTF-8 解析成功 |
| historical_ml KEEP_ORIGINAL 数 | 9 | `historical_ml_summary.json` 中 `ml_action_suggestion=KEEP_ORIGINAL` |

当前 5 个 final PROBE：

| symbol | name | pre rank | score | final action | weight |
| --- | --- | ---: | ---: | --- | ---: |
| 560780 | 半导体设备ETF广发 | 1 | 41.9116 | PROBE | 0.3 |
| 562590 | 半导体设备ETF华夏 | 2 | 41.8598 | PROBE | 0.3 |
| 159558 | 半导体设备ETF易方达 | 3 | 41.6865 | PROBE | 0.3 |
| 588810 | 科创芯片ETF富国 | 8 | 36.6527 | PROBE | 0.3 |
| 588990 | 科创芯片ETF博时 | 9 | 36.4486 | PROBE | 0.3 |

## 为什么当前只有 5 个 PROBE

结论：当前只有 5 个 PROBE，直接原因是 pre_selection 的候选上限，而不是 entry、historical_ml 或 frontend。

证据链：

1. `signal/pre_selection/engine.py` 的 `PreSelectionConfig.max_candidates = 5`。
2. 当前市场状态为进攻；`_candidate_limit()` 在进攻状态返回 `max_candidates`，因此 candidate_limit=5。
3. `pre_selection_result.csv` 只有 5 行 `selected=True`。
4. entry 对未入选 pre_selection 的样本直接输出 OBSERVE；本次 1155 行未入选全部是 OBSERVE。
5. 5 行 pre_selection 入选样本全部进入 entry PROBE，且 `final_buy_action=PROBE`。
6. 当前还有 131 行原因是“满足右侧条件，但综合排名未进入前5”，证明 selected=5 是候选池 cap 截断，而不是市场没有更多右侧候选。

因此验收项 1 结论为：`selected=5` 可以明确证明由 pre_selection cap 导致。

## historical_ml 接入状态

historical_ml 当前是代码级接入，但只读旁路，不参与当日交易裁决。

证据：

- `signal/entry/engine.py` 会读取 `artifacts/historical_ml_61/generated/entry_calibration_suggestions.csv` 或等价路径。
- 参数级 historical_ml 建议被归并为 `KEEP_ORIGINAL`。
- entry 输出中的 1160 行全部为 `ml_action_suggestion=KEEP_ORIGINAL`，`NO_ML=0`。
- entry 的 `ml_reason` 明确写明：historical_ml 建议是 parameter-level、没有 ETF code/symbol 绑定、只读展示，并且不改变 `buy_action`、`final_buy_action` 或 `target_weight`。
- `signal/v21_orchestrator.py` 把 historical_ml 汇总写入总控快照，但说明 learning/historical_ml 只给建议，不自动修改交易参数。

因此验收项 2 结论为：historical_ml 已代码级接入，但当前只读旁路；它不是本次只有 5 个 PROBE 的瓶颈。

## 瓶颈归属

当前瓶颈属于：pre_selection。

- pre_selection：是瓶颈。`max_candidates=5` 在进攻状态下直接限制 selected 数。
- entry：不是瓶颈。5 个 selected 全部转为 PROBE，未 selected 的 1155 行按职责观察。
- historical_ml：不是瓶颈。全部为 KEEP_ORIGINAL，只读旁路，不改 final action。
- frontend：不是瓶颈。前端只读取 V2.1 总控快照，不生成或裁决 PROBE。

## P0 Blocker

无。

## P1 高风险

无。当前报告仅冻结和解释 baseline，未改变交易逻辑、阈值、行情、QMT 或 output。

## P2 建议

若后续要研究“超过 5 个 PROBE 是否合理”，应另开任务，在 historical_ml/回放证据支持下评估 pre_selection cap，而不是在本 baseline 任务中调整阈值或候选上限。

## 证据

- `signal/pre_selection/engine.py:42`：`max_candidates = 5`
- `signal/pre_selection/engine.py:423-428`：selected 只取 eligible 且受 candidate_limit 截断
- `signal/pre_selection/engine.py:440-445`：进攻状态 candidate_limit 返回 max_candidates
- `signal/entry/engine.py:193-202`：未进入预选候选池直接观察
- `signal/entry/engine.py:237-243`：进攻状态、selected 且 score>=35 可 PROBE
- `signal/entry/engine.py:435-438`：参数级 ML 建议 observe only，KEEP_ORIGINAL
- `signal/v21_orchestrator.py:137-140`：总控基于 selected_symbols 构建 entry actions 和 ML 观察摘要
- `signal/v21_orchestrator.py:426-457`：只有 selected 且 raw BUY/PROBE 才成为 intended/final buy

输入快照哈希：

- `output/pre_selection_result.csv`: `F2982BA36C013D9F7B403F523F989E88FA5583EFB0E0C00E15B713767A12B44A`
- `output/entry_signal.csv`: `DD7CD3A52D9DDD5451A32A01306512990888B5F00C50C7B7DEEFD85BF92090E3`
- `output/historical_ml_summary.json`: `FEEBA8FE364BA113E1A27D8E1D03673C20C86A2BBB42814810E648DDB349B2E5`

## 影响模块

- aetfv2_01_pre_selection：确认当前 selected=5 来自候选上限。
- aetfv2_02_entry：确认 entry 只是消费 selected 并输出 5 个 PROBE、1155 个 OBSERVE。
- aetfv2_05_historical_ml：确认已接入但只读旁路。
- aetfv2_08_control_center：冻结当前 V2.1 legacy baseline 解释口径。
- frontend：只读展示，不是本次瓶颈。

## 测试命令

已运行只读验证命令：

- `Import-Csv output\pre_selection_result.csv`
- `Import-Csv output\entry_signal.csv`
- `Get-Content output\historical_ml_summary.json -Raw -Encoding UTF8 | ConvertFrom-Json`
- `Import-Csv output\historical_ml_summary.csv`
- `Import-Csv output\etf_universe_snapshot.csv`
- `Get-FileHash output\pre_selection_result.csv, output\entry_signal.csv, output\historical_ml_summary.json -Algorithm SHA256`
- `Get-Content .codex\config.toml`

未运行策略、未刷新行情、未运行 QMT、未运行生成信号命令，因为本任务边界要求冻结当前 baseline。

## 下一张 Codex 任务卡

在不改当前 baseline 的前提下，新增一个离线诊断报告：对 131 个“满足右侧条件但未进入前5”的 ETF 做 sector、score、后验收益、historical_ml 失败类型聚合，评估 pre_selection cap 是否需要作为下一阶段研究参数进入回放实验。

## Commit 前检查清单

- `git diff --name-only` 应只新增 `baseline_v21_legacy_report.md` 和 `baseline_v21_legacy_report.json`，其余既有脏改不属于本任务。
- 未提交 token、key、env、venv、大体积缓存。
- 未刷新行情、未生成正式信号、未写入 `output`、未触碰 `data/cache`。
- agent 名称、职责边界保持一致。
- `.codex/config.toml` 已确认保持 `max_threads = 4` 与 `max_depth = 1`。
- `final_buy_action` 未被修改；`output/entry_signal.csv` 哈希保持 `DD7CD3A52D9DDD5451A32A01306512990888B5F00C50C7B7DEEFD85BF92090E3`。
