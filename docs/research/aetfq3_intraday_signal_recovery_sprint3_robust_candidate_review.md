本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Intraday Signal Recovery Sprint3 Robust Candidate Review

本文件收口 F public intraday signal recovery sprint3：robust candidate family deep review / champion shortlist。Sprint3 只读取 Sprint2 ignored 输出，对 10 个 robust diagnostic candidate families 做人工复核式审计；本任务不是训练，不运行新模型，不调参，不保存模型、scaler 或 checkpoint，不接 QMT，不生成 OrderIntent，不进入 Stable。

## Scope

- 输入来自 Sprint2 ignored 输出与 Sprint2 research 文档。
- 复核对象为 Sprint2 产生的 10 个 robust diagnostic candidate families。
- 输出 candidate family review table、review report、shortlist 与 Sprint3 decision。
- Shortlist 只允许进入下一阶段 human review / out-of-period no-save validation design。
- Robust diagnostic candidate 仍不是 formal model evidence，不是交易建议，也不是 Stable evidence。

## Input Sources

- `.local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint2_candidate_audit/signal_recovery_sprint2_candidate_family_summary.csv`
- `.local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint2_candidate_audit/signal_recovery_sprint2_candidate_audit_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint2_candidate_audit/signal_recovery_sprint2_robustness_report.json`
- `.local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint2_candidate_audit/signal_recovery_sprint2_decision.json`
- `.local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint1/signal_recovery_sprint1_diagnostic_smoke_report.json`
- `docs/research/aetfq3_intraday_signal_recovery_sprint2_candidate_audit.md/json`

## Candidate Family Review

- Reviewed robust families: `10`
- Sprint2 selected families: `12`
- Robustness decision source: `SIGNAL_RECOVERY_SPRINT2_ROBUST_DIAGNOSTIC_CANDIDATE_FOUND_REVIEW_REQUIRED`
- Sprint3 decision: `SPRINT3_ROBUST_CANDIDATE_SHORTLIST_READY_FOR_HUMAN_REVIEW`

| Label policy | Count |
|---|---:|
| `label_ret3d_gt_100bp` | 4 |
| `label_safe_positive_3d` | 2 |
| `label_ret3d_gt_50bp` | 2 |
| `label_ret3d_gt_20bp` | 2 |


| Model family | Count |
|---|---:|
| `logistic_balanced_scaled_variants` | 10 |


## Label Policy Review

Robust candidates are not concentrated in a single narrow label policy. They are distributed across return-threshold labels and `label_safe_positive_3d`. The original `three_day_positive_label` and neutral-band labels do not appear in the robust family set, so future review should still check whether threshold-label selection is carrying too much of the apparent signal.

Label policy concentration flag: `False`.

## Model Family Review

All 10 robust diagnostic candidate families are concentrated in `logistic_balanced_scaled_variants`; no random forest, hist gradient boosting, LightGBM, XGBoost, or CatBoost family survived Sprint2 robustness as a robust family. This triggers `P1_MODEL_FAMILY_CONCENTRATION_REVIEW_REQUIRED`.

Model family concentration flag: `True`.

## Walk-Forward Review

Every robust family has walk-forward split rows available, and every family keeps non-collapse behavior across available split rows. However, multiple walk-forward folds show weak or negative dummy-margin, ROC-AUC, or PR-AUC margin behavior, so Sprint3 keeps walk-forward weakness under review and does not treat these metrics as effectiveness evidence.

## Economic Plausibility Review

- `label_ret3d_gt_100bp` families are plausible diagnostic signal hypotheses because the threshold is cleaner than raw positive-return noise, but they still require out-of-period no-save validation design.
- `label_safe_positive_3d` families are weakly plausible because a stricter outcome may reduce noisy positives, while label-policy filtering risk remains material.
- `label_ret3d_gt_50bp` and `label_ret3d_gt_20bp` families are weakly plausible with mixed walk-forward folds.
- Past daily context appears in 2 robust families and is economically interpretable as past-only continuation/regime context.
- Scale-transform policy remains necessary to review because earlier diagnostics retained extreme scale and train/valid shift warnings.

No statement in this review says any candidate can trade.

## Shortlist

Shortlist count: `3`.

- `label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy`: available_splits=5, walk_forward_folds=3, no_collapse_rate=1.0, PR-AUC margin positive in 5 splits, dummy margin positive in 4 splits; plausible diagnostic signal hypothesis Remaining risks stay review-only.
- `label_ret3d_gt_100bp|base_39_plus_past_daily_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy`: available_splits=5, walk_forward_folds=3, no_collapse_rate=1.0, PR-AUC margin positive in 5 splits, dummy margin positive in 4 splits; plausible diagnostic signal hypothesis Remaining risks stay review-only.
- `label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy`: available_splits=5, walk_forward_folds=3, no_collapse_rate=1.0, PR-AUC margin positive in 4 splits, dummy margin positive in 4 splits; weakly plausible Remaining risks stay review-only.


## Boundary

- `formal_model_evidence=false`
- `stable_promotion_ready=false`
- `formal_training_ready=false`
- `qmt_ready=false`
- `order_intent_ready=false`
- `automatic_promotion_ready=false`
- `metrics_are_effectiveness_evidence=false`
- no new model run
- no training
- no hyperparameter tuning
- no model/scaler/checkpoint save
- no QMT
- no OrderIntent
- no Stable
- no `output/`
- no `lab_advisory/`
- not trading advice

## Next Allowed Action

Only human review and out-of-period no-save validation design are allowed. No formal training, Stable promotion, QMT connection, OrderIntent generation, advisory package generation, or automatic promotion is allowed from this Sprint3 result.
