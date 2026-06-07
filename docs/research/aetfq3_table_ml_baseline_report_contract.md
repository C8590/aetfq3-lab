# AETF Q3 Lab Baseline Smoke Report Contract

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

Baseline smoke report 是 Lab-only 的代码路径验证报告。它只证明 no-save baseline smoke 工具能读取样本、执行 chronological split、运行极小型评估并输出可审计指标。它不是正式训练报告，不是 advisory 包，不是 Stable 输入，也不是交易建议。

`tools/lab/table_ml_baseline_smoke.py` 必须原生输出扁平 JSON contract 字段；旧的嵌套字段可作为兼容信息保留，但不得作为新 contract 的唯一来源。

## 必需边界

报告必须明确：

- `lab_only=true`
- `no_save=true`
- `no_tuning=true`
- `no_stable=true`
- `no_qmt=true`
- `no_order_intent=true`
- `no_output=true`
- `no_lab_advisory=true`
- `model_saved=false`
- `checkpoint_saved=false`

报告不得包含：

- `OrderIntent`
- `target_weight`
- `final_buy_action`
- `stable_action`
- `qmt_order`
- `live_order`
- `trade_instruction`

## 必需结构

`tools/lab/table_ml_baseline_report_reader.py` 校验以下顶层扁平字段：

- `report_type`
- `task_scope`
- `lab_only`
- `no_save`
- `no_tuning`
- `no_stable`
- `no_qmt`
- `no_order_intent`
- `no_output`
- `no_lab_advisory`
- `model_saved`
- `checkpoint_saved`
- `target_label`
- `feature_columns`
- `forbidden_columns`
- `train_count`
- `valid_count`
- `split_method`
- `group_leakage_check`
- `models`
- `metrics`
- `prediction_file`
- `review_checklist`

固定值要求：

- `report_type="table_ml_baseline_smoke"`
- `task_scope="Lab-only no-save baseline smoke"`
- `lab_only=true`
- `no_save=true`
- `no_tuning=true`
- `no_stable=true`
- `no_qmt=true`
- `no_order_intent=true`
- `no_output=true`
- `no_lab_advisory=true`
- `model_saved=false`
- `checkpoint_saved=false`
- `split_method="chronological"`
- `group_leakage_check="passed"`

`split_method` 必须是 `chronological`，`group_leakage_check` 必须是 `passed`。

## 多模型字段

`models` 可包含一个或多个 no-save smoke 模型：

- `numpy_logistic_regression_smoke`
- `lightgbm_smoke`
- `catboost_smoke`
- `xgboost_smoke`

每个 `models` 条目应包含：

- `model_name`
- `status`
- `train_count`
- `valid_count`
- `accuracy`
- `roc_auc`
- `log_loss`
- `notes`
- `no_save=true`
- `no_tuning=true`
- `model_saved=false`
- `checkpoint_saved=false`

可选依赖缺失时，模型可标记为 `status="skipped"`；已运行模型使用 `status="passed"`。无论 passed 还是 skipped，都不得保存模型、checkpoint 或 feature importance 文件。

`metrics` 条目也必须带有 no-save / no-tuning / model_saved / checkpoint_saved 边界字段，避免指标被误读为正式训练结果。

## Metrics 解释限制

Accuracy、ROC AUC、log loss、by-date summary、by-sector summary 只用于 smoke / code path validation。它们不代表模型有效性，不代表策略收益，不代表 Stable 可用性，也不得被解释为交易建议。

## Stable/QMT 边界

Stable 不得把 baseline smoke report 读取为正式交易输入。报告不得修改 `final_buy_action`、`target_weight`、BUY / PROBE 阈值，不得生成 OrderIntent，不得触发 QMT，不得写 `output/`、Stable runtime 或 `lab_advisory/`。

## Reader 行为

Reader 只读 baseline smoke report JSON，输出 summary JSON 到 stdout。若必需字段缺失、边界字段不符合要求，或发现交易相关字段，reader 必须返回非 0。

新 writer 输出的 flat contract 应直接通过 reader 校验。Reader 仍支持旧嵌套 smoke report：当报告没有完整 flat 字段但包含旧 `boundary`、`feature_leakage_check` 或 `split` 结构时，reader 可将旧结构归一化为上述 flat contract 后再校验。该兼容路径只用于历史报告读取，不作为新 writer 的输出目标。

Reader 命令必须使用 Lab `.venv` Python，或先激活 `.venv` 后再运行。推荐入口：

```powershell
.\.venv\Scripts\python.exe tools\lab\table_ml_baseline_report_reader.py `
  --report .local_research_outputs\aetfq3_lab\table_ml_baseline_smoke\sector_internal_ranking_baseline_smoke_report.json
```

不推荐裸 `python`，因为它可能指向系统 Python。若使用裸 `python`，必须先确认：

```powershell
python -c "import sys; print(sys.executable)"
```

输出应为：

```text
E:\aetfq3-lab\.venv\Scripts\python.exe
```

系统 Python 缺少 `lightgbm` / `catboost` / `xgboost` / `torch` 不是 P0；只要 `.venv` 正常并使用 `.venv` Python 运行 Lab 表格 ML 命令即可。

## Review Checklist 映射

1. 研究了什么：Lab-only baseline smoke report contract。
2. 数据来自哪里：baseline smoke report JSON。
3. 是否来自 Stable bundle：否。
4. 是否有未来函数：reader 校验 feature/forbidden contract，但不重新计算样本。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：否，不是 advisory 包。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议：如需要更强 contract，可在后续任务中增加 schema 文件或更多只读 fixture，不改变 Stable/QMT/advisory/output 边界。
