# AETF Q3 Lab Table ML Dry Validation Orchestrator

本任务属于 `aetfq3-lab / Lab`，不属于 V2.1 Stable。

## 文档定位

本文定义 A `ML false downgrade` 与 E `同板块 ETF 内部排序` 的 Lab-only dry validation orchestration。它不是训练器，不是 advisory 生成器，不是 Stable 接口，不读取真实行情，不运行策略，不接 QMT，不生成 `OrderIntent`。

## 两道门

- sample intake gate：`tools/lab/table_ml_sample_intake_checker.py` 先校验 manifest 的来源、授权、Stable bundle readonly 边界、未来函数声明、训练禁止和 advisory-only 边界。
- schema validator gate：`tools/lab/table_ml_schema_validator.py` 只在 intake 通过后校验 mock / local ignored 小样本 CSV 的字段合同、主键、标签和 feature leakage。

只有 sample intake gate 通过后，orchestrator 才允许调用 schema validator gate。

## 执行流程

1. 读取 dry validation manifest JSON。
2. 复用 sample intake checker 逻辑。
3. intake 失败时立即停止，返回非 0 exit code，并输出 summary JSON。
4. intake 通过后，从 manifest 读取 `sample_type`、`sample_path` 和 `feature_columns`。
5. 调用 table ML schema validator。
6. schema validation 失败时返回非 0 exit code，并输出 summary JSON。
7. 全部通过后输出 `status=passed` 的 summary JSON。

## CLI 用法

验证 A false downgrade mock：

```powershell
python tools/lab/table_ml_dry_validation_orchestrator.py --manifest tests\fixtures\aetfq3_lab\mock_dry_validation_manifest_false_downgrade.json
```

验证 E sector internal ranking mock：

```powershell
python tools/lab/table_ml_dry_validation_orchestrator.py --manifest tests\fixtures\aetfq3_lab\mock_dry_validation_manifest_sector_internal_ranking.json
```

## manifest 字段要求

dry validation manifest 继承 sample intake manifest 要求，必须包含：

- `sample_type`
- `sample_path`
- `feature_columns`
- `human_authorized=true`
- `training_allowed=false`
- `stable_effect_allowed=false`
- `advisory_only=true`
- `affects_stable_trading=false`
- `contains_secret=false`
- `contains_live_order=false`
- `contains_order_intent=false`
- `uses_stable_bundle=false`，或在 Stable bundle readonly 场景下补齐 bundle 元数据。
- `allowed_for` 包含 `dry_validation_only`。

`feature_columns` 不得包含 future label、forbidden feature、`future_*`、`max_drawdown_*`、`best_in_sector_*`、`top_quantile_*` 或 `*_label`。

## 失败类型

- `intake_failed`: manifest JSON、必需字段、枚举或边界校验失败。
- `schema_failed`: intake 已通过，但 CSV schema、主键、标签或 group 规则失败。
- `forbidden_future_feature`: `feature_columns` 包含 future label 或 forbidden feature。
- `unauthorized_input`: manifest 试图授权训练、影响 Stable、缺少人工授权或突破只读边界。
- `missing_sample_file`: `sample_path` 不存在，且不是允许缺失的模板路径。

## 输出 summary 字段

orchestrator 输出 JSON 到 stdout，至少包含：

- `status`
- `sample_type`
- `sample_path`
- `rows_checked`
- `intake_passed`
- `schema_passed`
- `warnings`
- `p0_blockers`
- `p1_warnings`
- `advisory_only`
- `training_allowed`
- `affects_stable_trading`

## 不允许事项

- 不读取真实行情。
- 不读取 `data/etf_daily.csv`。
- 不读取 Stable 真实 output。
- 不读取未授权真实样本。
- 不训练模型。
- 不跑 LightGBM / CatBoost / XGBoost 真实训练。
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
- 不提交 data、artifacts、`.local_research_outputs/` 或模型权重。

## 后续真实小样本 dry validation 流程

1. 人工准备 ignored 小样本，放入 `.local_research_outputs/aetfq3_lab/`。
2. 人工准备 manifest，声明来源、授权、Stable bundle 边界、时间范围、样本规模和 feature columns。
3. intake checker 通过。
4. schema validator 通过。
5. 仅生成本地 ignored dry validation report。
6. 再人工决定是否进入 baseline；进入 baseline 也不等于进入 Stable。

## Review Checklist 自检

1. 研究了什么：实现 Lab-only dry validation orchestration，将 sample intake gate 与 schema validator gate 串联。
2. 数据来自哪里：仅来自人工 mock fixture 和 manifest，不来自真实行情或 Stable output。
3. 是否来自 Stable bundle：否；未来若来自 Stable bundle，必须 readonly 且补齐 bundle 元数据。
4. 是否有未来函数：mock manifest 的 `feature_columns` 不包含 future label；checker 与 schema validator 均阻断 forbidden future feature。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：是；本工具只做 dry validation 门禁，不生成 advisory 包。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用；本工具不建议进入 Stable。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议是什么：使用人工授权的 local ignored 小样本生成 dry validation smoke report，仍不训练、不接 QMT、不接 Stable。
