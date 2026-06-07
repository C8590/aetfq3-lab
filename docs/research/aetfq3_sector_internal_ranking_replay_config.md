# AETF Q3 Lab E Sector Internal Ranking Replay Config

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 任务定位

本文件固化 E sector internal ranking 多窗口 Lab-only smoke 的输入清单、依赖、数据源、边界和复验命令模板。它只用于工程复验准备，不是 Stable，不是 advisory，不是交易信号，不训练模型，不保存模型，不生成 checkpoint，不接 QMT，不生成 OrderIntent，不写 `output/`，不创建 `lab_advisory/`。

## 数据源

- AKShare ETF daily OHLCV
- `config/etf_sector_map.yaml`
- `uses_stable_bundle=false`

## 窗口参数

| parameter | value |
| --- | --- |
| windows | 20 / 40 / 60 / 90 trading days |
| max_etfs | 20 |
| min_etfs_per_sector | 4 |
| feature_count | 25 |
| target | `top_quantile_in_sector_3d` |
| models | `numpy_logistic,lightgbm,catboost,xgboost` |

## 工具链

- `tools/lab/sector_internal_ranking_sample_generator.py`
- `tools/lab/table_ml_sample_intake_checker.py`
- `tools/lab/table_ml_schema_validator.py`
- `tools/lab/table_ml_dry_validation_orchestrator.py`
- `tools/lab/table_ml_baseline_smoke.py`
- `tools/lab/table_ml_baseline_report_reader.py`

## Python 入口

推荐入口：

```powershell
.\.venv\Scripts\python.exe
```

## 边界

- no Stable
- no QMT
- no OrderIntent
- no output/
- no lab_advisory/
- no model save
- no checkpoint
- no trading advice

## 复验命令模板

以下命令是推荐模板，不应在本配置文档任务中执行。复验时只允许写入 ignored 本地目录，例如 `.local_research_outputs/aetfq3_lab/sector_internal_ranking_replay/window_<N>/`。

### Window 20

```powershell
$py = ".\.venv\Scripts\python.exe"
$out = ".local_research_outputs/aetfq3_lab/sector_internal_ranking_replay/window_20"
& $py tools/lab/sector_internal_ranking_sample_generator.py --source akshare --max-trading-days 20 --max-etfs 20 --min-etfs-per-sector 4 --out-dir $out
& $py tools/lab/table_ml_sample_intake_checker.py --manifest "$out/sector_internal_ranking_expanded_manifest.json"
& $py tools/lab/table_ml_schema_validator.py --sample-type sector_internal_ranking --input "$out/sector_internal_ranking_expanded_feature_sample.csv" --feature-columns "<comma-separated-feature-columns-from-contract>"
& $py tools/lab/table_ml_dry_validation_orchestrator.py --manifest "$out/sector_internal_ranking_expanded_manifest.json"
& $py tools/lab/table_ml_baseline_smoke.py --sample "$out/sector_internal_ranking_expanded_feature_sample.csv" --manifest "$out/sector_internal_ranking_expanded_manifest.json" --feature-contract "$out/sector_internal_ranking_expanded_feature_contract.json" --models numpy_logistic,lightgbm,catboost,xgboost --out-dir $out
& $py tools/lab/table_ml_baseline_report_reader.py --report "$out/sector_internal_ranking_baseline_smoke_report.json"
```

### Window 40

```powershell
$py = ".\.venv\Scripts\python.exe"
$out = ".local_research_outputs/aetfq3_lab/sector_internal_ranking_replay/window_40"
& $py tools/lab/sector_internal_ranking_sample_generator.py --source akshare --max-trading-days 40 --max-etfs 20 --min-etfs-per-sector 4 --out-dir $out
& $py tools/lab/table_ml_sample_intake_checker.py --manifest "$out/sector_internal_ranking_expanded_manifest.json"
& $py tools/lab/table_ml_schema_validator.py --sample-type sector_internal_ranking --input "$out/sector_internal_ranking_expanded_feature_sample.csv" --feature-columns "<comma-separated-feature-columns-from-contract>"
& $py tools/lab/table_ml_dry_validation_orchestrator.py --manifest "$out/sector_internal_ranking_expanded_manifest.json"
& $py tools/lab/table_ml_baseline_smoke.py --sample "$out/sector_internal_ranking_expanded_feature_sample.csv" --manifest "$out/sector_internal_ranking_expanded_manifest.json" --feature-contract "$out/sector_internal_ranking_expanded_feature_contract.json" --models numpy_logistic,lightgbm,catboost,xgboost --out-dir $out
& $py tools/lab/table_ml_baseline_report_reader.py --report "$out/sector_internal_ranking_baseline_smoke_report.json"
```

### Window 60

```powershell
$py = ".\.venv\Scripts\python.exe"
$out = ".local_research_outputs/aetfq3_lab/sector_internal_ranking_replay/window_60"
& $py tools/lab/sector_internal_ranking_sample_generator.py --source akshare --max-trading-days 60 --max-etfs 20 --min-etfs-per-sector 4 --out-dir $out
& $py tools/lab/table_ml_sample_intake_checker.py --manifest "$out/sector_internal_ranking_expanded_manifest.json"
& $py tools/lab/table_ml_schema_validator.py --sample-type sector_internal_ranking --input "$out/sector_internal_ranking_expanded_feature_sample.csv" --feature-columns "<comma-separated-feature-columns-from-contract>"
& $py tools/lab/table_ml_dry_validation_orchestrator.py --manifest "$out/sector_internal_ranking_expanded_manifest.json"
& $py tools/lab/table_ml_baseline_smoke.py --sample "$out/sector_internal_ranking_expanded_feature_sample.csv" --manifest "$out/sector_internal_ranking_expanded_manifest.json" --feature-contract "$out/sector_internal_ranking_expanded_feature_contract.json" --models numpy_logistic,lightgbm,catboost,xgboost --out-dir $out
& $py tools/lab/table_ml_baseline_report_reader.py --report "$out/sector_internal_ranking_baseline_smoke_report.json"
```

### Window 90

```powershell
$py = ".\.venv\Scripts\python.exe"
$out = ".local_research_outputs/aetfq3_lab/sector_internal_ranking_replay/window_90"
& $py tools/lab/sector_internal_ranking_sample_generator.py --source akshare --max-trading-days 90 --max-etfs 20 --min-etfs-per-sector 4 --out-dir $out
& $py tools/lab/table_ml_sample_intake_checker.py --manifest "$out/sector_internal_ranking_expanded_manifest.json"
& $py tools/lab/table_ml_schema_validator.py --sample-type sector_internal_ranking --input "$out/sector_internal_ranking_expanded_feature_sample.csv" --feature-columns "<comma-separated-feature-columns-from-contract>"
& $py tools/lab/table_ml_dry_validation_orchestrator.py --manifest "$out/sector_internal_ranking_expanded_manifest.json"
& $py tools/lab/table_ml_baseline_smoke.py --sample "$out/sector_internal_ranking_expanded_feature_sample.csv" --manifest "$out/sector_internal_ranking_expanded_manifest.json" --feature-contract "$out/sector_internal_ranking_expanded_feature_contract.json" --models numpy_logistic,lightgbm,catboost,xgboost --out-dir $out
& $py tools/lab/table_ml_baseline_report_reader.py --report "$out/sector_internal_ranking_baseline_smoke_report.json"
```

## 结果验收

- intake passed
- schema passed
- dry validation passed
- reader OK
- group leakage passed
- missing_rate=0
- four-model no-save passed

## 禁止事项

- 不修改策略代码。
- 不修改 Stable。
- 不运行策略。
- 不刷新行情。
- 不写 `output/`。
- 不接 QMT。
- 不生成 OrderIntent。
- 不训练模型。
- 不保存模型。
- 不生成 checkpoint。
- 不提交 `.local_research_outputs/`。

## 下一步建议

- 在 Lab-only 边界内用该清单做人工触发复验。
- 固定 ETF universe、sector map commit 与依赖版本后再扩大覆盖。
- 复验结果仍不得进入 Stable，不得生成 advisory 包。
