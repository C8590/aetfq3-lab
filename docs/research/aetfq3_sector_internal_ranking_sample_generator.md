# AETF Q3 Lab Sector Internal Ranking Sample Generator

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 目标

`tools/lab/sector_internal_ranking_sample_generator.py` 将 E sector internal ranking expanded sample 的一次性制样流程沉淀为 Lab-only 可复用 CLI。工具只生成 sample、feature sample、manifest、feature contract 和 generation report，用于 intake、schema、dry validation 或 no-save smoke 的前置检查。

## 边界

- Lab-only，不接 Stable。
- 不接 QMT，不读取真实账户。
- 不生成 `OrderIntent`、交易动作、`target_weight` 或 `final_buy_action`。
- 不训练模型，不保存模型，不生成 checkpoint，不做超参数搜索。
- 不创建 `lab_advisory/`，不生成 advisory 包。
- 不写 `output/`，输出只允许进入 ignored `.local_research_outputs/aetfq3_lab/` 目录。
- AKShare 数据只用于 Lab dry validation / smoke 检查，不是交易依据，不代表模型效果。

## CLI

Lab 表格 ML 命令必须使用仓库 `.venv` Python：

```powershell
.\.venv\Scripts\python.exe tools\lab\sector_internal_ranking_sample_generator.py `
  --max-trading-days 60 `
  --max-etfs 32 `
  --min-etfs-per-sector 4 `
  --out-dir .local_research_outputs\aetfq3_lab\sector_internal_ranking_expanded\
```

可选参数：

- `--max-trading-days`
- `--max-etfs`
- `--min-etfs-per-sector`
- `--max-rows`
- `--out-dir`
- `--source akshare` 或 `--source mock`
- `--mock-daily-csv`
- `--mock-sector-map`
- `--skip-baseline-smoke`

Mock 示例不访问网络：

```powershell
.\.venv\Scripts\python.exe tools\lab\sector_internal_ranking_sample_generator.py `
  --source mock `
  --mock-daily-csv tests\fixtures\aetfq3_lab\mock_etf_daily_for_sector_ranking.csv `
  --mock-sector-map tests\fixtures\aetfq3_lab\mock_sector_map_for_sector_ranking.yaml `
  --max-trading-days 6 `
  --max-etfs 4 `
  --min-etfs-per-sector 2 `
  --out-dir .local_research_outputs\aetfq3_lab\sector_internal_ranking_expanded\mock\
```

## 输出

工具生成：

- `sector_internal_ranking_expanded_sample.csv`
- `sector_internal_ranking_expanded_manifest.json`
- `sector_internal_ranking_expanded_feature_sample.csv`
- `sector_internal_ranking_expanded_feature_contract.json`
- `sector_internal_ranking_expanded_generation_report.md`
- `sector_internal_ranking_expanded_generation_report.json`

第一版只工具化 sample + feature + manifest + report。Baseline smoke 可由后续命令显式运行，不在 generator 内自动训练或保存模型。

## 样本规则

- `ranking_group_id = trade_date + "_" + sector`
- 每个 ranking group 至少满足 `--min-etfs-per-sector`
- `future_return_1d`、`future_return_3d`、`max_drawdown_3d` 只作为 label/outcome
- 可生成 `best_in_sector_1d`、`best_in_sector_3d`、`top_quantile_in_sector_3d`、`avoid_in_sector`、`pairwise_outperform_label`
- 所有 feature 均为 lag1 / past-only，或明确允许的当日组规模字段 `sector_etf_count`
- `feature_columns` 排除 future、label、id、group、原始 sector string、ETF identity 字段

## Manifest 边界

Manifest 固定写明：

- `uses_stable_bundle=false`
- `training_allowed=false`
- `stable_effect_allowed=false`
- `advisory_only=true`
- `affects_stable_trading=false`
- `contains_secret=false`
- `contains_live_order=false`
- `contains_order_intent=false`
- `qmt_related=false`
- `has_future_leakage_check=true`

## 校验

生成后可运行 intake checker：

```powershell
.\.venv\Scripts\python.exe tools\lab\table_ml_sample_intake_checker.py `
  --manifest .local_research_outputs\aetfq3_lab\sector_internal_ranking_expanded\sector_internal_ranking_expanded_manifest.json
```

再运行 schema validator：

```powershell
.\.venv\Scripts\python.exe tools\lab\table_ml_schema_validator.py `
  --sample-type sector_internal_ranking `
  --input .local_research_outputs\aetfq3_lab\sector_internal_ranking_expanded\sector_internal_ranking_expanded_feature_sample.csv `
  --feature-columns "etf_ret_1d_lag1,etf_ret_3d_lag1,etf_ret_5d_lag1,etf_ret_10d_lag1,etf_volatility_5d_lag1,etf_volatility_10d_lag1,etf_amount_5d_mean_lag1,etf_amount_10d_mean_lag1,etf_amount_change_5d_lag1,etf_drawdown_5d_lag1,etf_drawdown_10d_lag1,sector_ret_1d_mean_lag1,sector_ret_3d_mean_lag1,sector_ret_5d_mean_lag1,sector_breadth_1d_lag1,sector_breadth_3d_lag1,sector_amount_5d_mean_lag1,sector_etf_count,etf_vs_sector_ret_3d_lag1,etf_vs_sector_ret_5d_lag1,etf_amount_share_in_sector_lag1,etf_rank_ret_3d_in_sector_lag1,etf_rank_ret_5d_in_sector_lag1,etf_rank_amount_5d_in_sector_lag1,etf_rank_volatility_5d_in_sector_lag1"
```

## 常见错误

- 使用裸 `python` 导致依赖环境不一致；应使用 `.\.venv\Scripts\python.exe`。
- 将 `future_return_*`、`max_drawdown_3d`、`best_in_sector_*`、`top_quantile_in_sector_3d` 或 `pairwise_outperform_label` 放入 feature。
- 输出目录不在 `.local_research_outputs/aetfq3_lab/`。
- ETF 数不足导致每组小于 `--min-etfs-per-sector`。
- 把 smoke metrics 解释为模型有效性或交易建议。
