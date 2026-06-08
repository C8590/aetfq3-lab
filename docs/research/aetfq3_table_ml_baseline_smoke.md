# AETF Q3 Lab Table ML Baseline Smoke

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

`tools/lab/table_ml_baseline_smoke.py` 是 Lab-only 表格 ML smoke 工具。它只验证 E sector internal ranking 小样本的 baseline 训练 / 评估代码路径可以运行，不是正式训练，不评估真实模型效果，不生成交易建议，不生成 advisory 包。

## 使用边界

- no-save：不保存模型文件，不生成 checkpoint。
- no tuning：不做超参数搜索。
- no Stable：不修改 Stable，不接 Stable entry，不修改 `final_buy_action`、`target_weight`、BUY / PROBE 阈值。
- no QMT：不接 QMT，不读取真实账户，不生成委托。
- no OrderIntent：不生成正式或模拟 `OrderIntent`。
- no output/：不写 `output/` 或 Stable runtime/output。
- no advisory：不创建 `lab_advisory/`，不生成 advisory 包。
- not trading advice：metrics 不能解释为交易建议或模型有效性证据。

## CLI

Lab 表格 ML 命令必须使用当前仓库 `.venv` Python：

```powershell
.\.venv\Scripts\python.exe
```

也可以先激活：

```powershell
.\.venv\Scripts\Activate.ps1
```

不推荐裸 `python`，因为它可能指向系统 Python。若使用裸 `python`，必须先执行：

```powershell
python -c "import sys; print(sys.executable)"
```

确认输出为：

```text
E:\aetfq3-lab\.venv\Scripts\python.exe
```

当前系统 Python `E:\Program Files (x86)\Python\Python312\python.exe` 缺少 `lightgbm` / `catboost` / `xgboost` / `torch`。这不是 P0；只要 `.venv` 正常并使用 `.venv` Python 运行 Lab 命令即可。

```powershell
.\.venv\Scripts\python.exe tools/lab/table_ml_baseline_smoke.py `
  --sample .local_research_outputs/aetfq3_lab/table_ml_dry_validation_inputs/sector_internal_ranking_real_feature_sample.csv `
  --manifest .local_research_outputs/aetfq3_lab/table_ml_dry_validation_inputs/sector_internal_ranking_real_feature_sample_manifest.json `
  --feature-contract .local_research_outputs/aetfq3_lab/table_ml_baseline_precheck/sector_internal_ranking_feature_contract.json `
  --target top_quantile_in_sector_3d `
  --models numpy_logistic,lightgbm,catboost,xgboost `
  --task-name sector_internal_ranking `
  --output-prefix sector_internal_ranking `
  --out-dir .local_research_outputs/aetfq3_lab/table_ml_baseline_smoke/
```

`--task-name` 与 `--output-prefix` 为可选参数。旧命令不传时默认保持：

```text
task_name=sector_internal_ranking
output_prefix=sector_internal_ranking
```

A false downgrade reconstructed v2 no-save smoke 示例：

```powershell
.\.venv\Scripts\python.exe tools/lab/table_ml_baseline_smoke.py `
  --sample .local_research_outputs\aetfq3_lab\false_downgrade_reconstructed_v2\false_downgrade_reconstructed_v2_feature_samples.csv `
  --manifest .local_research_outputs\aetfq3_lab\false_downgrade_reconstructed_v2\false_downgrade_reconstructed_v2_feature_manifest.json `
  --feature-contract .local_research_outputs\aetfq3_lab\false_downgrade_reconstructed_v2\false_downgrade_reconstructed_v2_feature_contract.json `
  --target false_downgrade_3d `
  --models numpy_logistic,lightgbm,catboost,xgboost `
  --task-name false_downgrade_reconstructed_v2 `
  --output-prefix false_downgrade_reconstructed_v2 `
  --out-dir .local_research_outputs\aetfq3_lab\false_downgrade_reconstructed_v2\baseline_smoke_named\
```

## 输入检查

工具会读取 manifest 并要求：

- `training_allowed=false`
- `stable_effect_allowed=false`
- `advisory_only=true`
- `affects_stable_trading=false`
- `contains_secret=false`
- `contains_live_order=false`
- `contains_order_intent=false`

工具会读取 feature contract，只使用其中的 `candidate_features`。以下字段禁止进入 feature：

- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `best_in_sector_1d`
- `best_in_sector_3d`
- `top_quantile_in_sector_3d`
- `avoid_in_sector`
- `pairwise_outperform_label`
- `trade_date`
- `sector`
- `etf_code`
- `etf_name`
- `ranking_group_id`
- `model_version`
- `feature_version`

## Split

- chronological split
- 按 `trade_date` 排序
- 前 70% 日期训练，后 30% 日期验证
- no shuffle
- `ranking_group_id` 不得同时出现在 train 和 validation

## 模型

默认只运行 numpy logistic regression smoke。可通过 `--models` 选择一个或多个 no-save smoke 模型：

- `numpy_logistic`
- `lightgbm`
- `catboost`
- `xgboost`

所有模型都必须保持：

- 模型只存在内存中
- 不保存模型
- 不生成 checkpoint
- 不做超参数搜索
- 使用 tiny fixed params
- `n_estimators` / `iterations` 不超过 30
- `max_depth` / `depth` 不超过 3

可选依赖缺失时，该模型在 report 中标记为 `status="skipped"`，不影响其它已选择模型的 no-save smoke 路径。CatBoost 必须设置 `allow_writing_files=false`，不得留下 `catboost_info`。

## 输出

输出只允许写入 ignored 本地目录，例如：

```text
.local_research_outputs/aetfq3_lab/table_ml_baseline_smoke/
```

生成：

- `<output_prefix>_baseline_smoke_report.md`
- `<output_prefix>_baseline_smoke_report.json`
- `<output_prefix>_baseline_predictions.csv`

JSON report 原生输出 flat contract 顶层字段，包括：

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
- `task_name`
- `output_prefix`
- `sample_type`
- `target_label`
- `feature_columns`
- `forbidden_columns`
- `train_count`
- `valid_count`
- `split_method="chronological"`
- `group_leakage_check="passed"`
- `models`
- `metrics`
- `prediction_file`
- `review_checklist`

为兼容历史读取，JSON report 仍保留 `data`、`feature_leakage_check`、`split`、`boundary` 等嵌套对象；新消费者应优先读取 flat contract 字段。

`models` 中每个模型条目包含：

- `model_name`
- `status`
- `train_count`
- `valid_count`
- `accuracy`
- `roc_auc`
- `log_loss`
- `notes`
- no-save / no-tuning / model_saved / checkpoint_saved 边界字段

`metrics` 中同样保留 no-save / no-tuning / model_saved / checkpoint_saved 边界字段。Metrics 只说明 smoke 代码路径，不说明模型有效性。

预测 CSV 只包含 validation smoke 字段，不包含交易动作、仓位、`OrderIntent` 或 Stable 参数。

## Report Contract Reader

`tools/lab/table_ml_baseline_report_reader.py` 可只读校验 baseline smoke report JSON 的稳定合同。Reader 会检查 Lab-only / no-save / no-tuning / no Stable / no QMT / no OrderIntent / no output / no lab_advisory 边界，并拒绝包含 `order_intent`、`target_weight`、`final_buy_action`、`stable_action`、`qmt_order`、`live_order` 或 `trade_instruction` 的报告。

新 writer 输出的 flat contract 可直接通过 reader；reader 仍保留旧嵌套 smoke report 的兼容读取路径。Reader 输出 summary JSON 到 stdout；若边界字段缺失或失败，返回非 0。报告合同细节见 `docs/research/aetfq3_table_ml_baseline_report_contract.md`。

Reader 示例同样必须使用 `.venv` Python：

```powershell
.\.venv\Scripts\python.exe tools\lab\table_ml_baseline_report_reader.py `
  --report .local_research_outputs\aetfq3_lab\table_ml_baseline_smoke\sector_internal_ranking_baseline_smoke_report.json
```

## Metrics 解释限制

报告中的 accuracy、ROC AUC、log loss、by-date summary、by-sector summary 只证明代码路径可运行。小样本 smoke 的 metrics 不代表模型有效性，不代表策略收益，不代表 Stable 可用性，也不得作为交易建议。

## Review Checklist 映射

1. 研究了什么：Lab-only table ML baseline smoke 代码路径。
2. 数据来自哪里：本地 ignored feature sample、manifest 和 feature contract。
3. 是否来自 Stable bundle：默认否；若未来使用必须显式标注且只读。
4. 是否有未来函数：工具禁止 future / label / id / group 字段进入 feature。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：否，本工具不生成 advisory。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议：仅在人工确认后扩展更多 no-save smoke 模型或更大样本。
