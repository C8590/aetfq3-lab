# AETF Q3 Lab Dry Validation Report Reader

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 文档定位

本文定义 Lab-only dry validation report reader。它不是训练器，不是 advisory 生成器，不是 Stable 接口，只读取 dry validation JSON report / summary 并打印状态摘要。

## 读取范围

reader 只读取 dry validation JSON report 或 smoke summary JSON，例如：

- `.local_research_outputs/aetfq3_lab/table_ml_dry_validation/false_downgrade_mock_dry_validation_report.json`
- `.local_research_outputs/aetfq3_lab/table_ml_dry_validation/sector_internal_ranking_mock_dry_validation_report.json`
- `.local_research_outputs/aetfq3_lab/table_ml_dry_validation/dry_validation_smoke_summary.json`

## 不读取 sample_path

单个 report JSON 可能包含 `sample_path` 字段。reader 只把它当作普通字符串处理，不打开该路径，不读取 CSV 样本行级内容，不做 schema validation，不训练模型。

## CLI 用法

读取单个 report，默认 text 输出：

```powershell
python tools/lab/table_ml_dry_validation_report_reader.py --report .local_research_outputs/aetfq3_lab/table_ml_dry_validation/false_downgrade_mock_dry_validation_report.json
```

读取 smoke summary：

```powershell
python tools/lab/table_ml_dry_validation_report_reader.py --summary .local_research_outputs/aetfq3_lab/table_ml_dry_validation/dry_validation_smoke_summary.json
```

输出 JSON：

```powershell
python tools/lab/table_ml_dry_validation_report_reader.py --report .local_research_outputs/aetfq3_lab/table_ml_dry_validation/false_downgrade_mock_dry_validation_report.json --format json
```

## 输出字段

单个 report 摘要至少包含：

- `review_status`
- `status`
- `sample_type`
- `rows_checked`
- `intake_passed`
- `schema_passed`
- `p0_blockers count`
- `p1_warnings count`
- `advisory_only`
- `training_allowed`
- `affects_stable_trading`

summary 摘要至少包含：

- `included report count`
- 每个 report 的 `status`
- `all_passed`
- `has_p0`
- `has_p1`
- `advisory_only`
- `training_allowed=false`
- `affects_stable_trading=false`

## P0_REVIEW_REQUIRED 触发条件

出现以下任一情况，reader 必须标记 `P0_REVIEW_REQUIRED`：

- `p0_blockers` 非空。
- `training_allowed=true`。
- `affects_stable_trading=true`。
- `advisory_only=false`。
- summary 显示存在 P0 或违反训练 / Stable / advisory-only 边界。

若 `status != passed` 或存在 P1 warning，reader 至少标记 `NEEDS_REVIEW`。

## 不允许事项

- 不读取真实行情。
- 不读取 CSV 样本行级内容。
- 不读取 `data/etf_daily.csv`。
- 不读取 Stable 真实 output。
- 不读取未授权真实样本。
- 不训练模型。
- 不跑 LightGBM / CatBoost / XGBoost 训练。
- 不生成 advisory 包。
- 不创建 `lab_advisory/`。
- 不接 Stable。
- 不修改 Stable。
- 不接 QMT。
- 不生成 `OrderIntent`。
- 不影响 `final_buy_action`。
- 不修改 `target_weight`。
- 不修改 BUY / PROBE 阈值。
- 不写 `output/`。

## 后续真实 dry validation 报告审阅流程

1. dry validation orchestrator 在 ignored 本地目录生成 JSON report。
2. report reader 只读取 JSON report / summary。
3. 人工查看 `review_status`、P0 / P1 计数和边界字段。
4. 若出现 `P0_REVIEW_REQUIRED`，停止后续 baseline 讨论并人工复核。
5. 若全部 `OK`，也只表示 dry validation report 可读，不表示可进入 Stable。

## Review Checklist 自检

1. 研究了什么：实现 Lab-only dry validation report reader，只读取 JSON report / summary 并输出状态摘要。
2. 数据来自哪里：来自 ignored dry validation JSON report / summary；测试使用人工构造 JSON。
3. 是否来自 Stable bundle：否。
4. 是否有未来函数：否；reader 不读取样本行级内容，也不计算特征或标签。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：是；reader 只读 JSON，不生成 advisory 包。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用；本工具不建议进入 Stable。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议是什么：如需继续，可新增本地 ignored reader smoke 输出，记录 reader 对 mock report 的摘要结果。
