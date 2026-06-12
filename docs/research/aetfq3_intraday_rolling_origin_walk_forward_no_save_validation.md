本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# Intraday Rolling-Origin Walk-Forward No-Save Validation

Lab-only rolling-origin diagnostic validation. It uses historical cutoffs, train-only scaler fit, fixed Sprint3 shortlist only, and writes no model/scaler/checkpoint.

- readiness_decision: ROLLING_ORIGIN_WALK_FORWARD_DIAGNOSTIC_STABILITY_OBSERVED_REVIEW_REQUIRED
- fold_count: 12
- evaluated_fold_count: 11
- no_leakage_assertion_passed: true
- model_saved: false
- scaler_saved: false
- stable_promotion_ready: false

## Aggregate Stability
- label_ret3d_gt_100bp|base_39_plus_past_daily_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy: folds=11, ba_mean=0.523698283825238, ba_above_0_5=0.5454545454545454, roc_above_0_5=0.5454545454545454, pr_not_below_prev=0.5454545454545454, stability=false
- label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy: folds=11, ba_mean=0.5250427062607832, ba_above_0_5=0.7272727272727273, roc_above_0_5=0.7272727272727273, pr_not_below_prev=0.7272727272727273, stability=true
- label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy: folds=11, ba_mean=0.49228537299320224, ba_above_0_5=0.36363636363636365, roc_above_0_5=0.36363636363636365, pr_not_below_prev=0.5454545454545454, stability=false
