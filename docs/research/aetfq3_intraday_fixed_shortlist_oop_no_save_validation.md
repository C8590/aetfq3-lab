本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# Intraday Fixed Shortlist OOP No-Save Validation

Lab-only strict OOP diagnostic validation. It does not save models/scalers, does not write output/, does not connect QMT, does not generate OrderIntent, and is not Stable evidence.

- readiness_decision: FIXED_SHORTLIST_OOP_NO_SAVE_VALIDATION_COMPLETED_REVIEW_REQUIRED
- train_anchor_count: 37
- pre_sprint_oop_anchor_count: 268
- post_sprint_oop_anchor_count: 7
- combined_strict_oop_anchor_count: 275
- candidate_count: 3
- survived_candidate_count: 0
- row_level_predictions_emitted: true
- row_level_prediction_row_count: 41352
- model_saved: false
- scaler_saved: false
- stable_promotion_ready: false

## Candidate Summary
- label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy: balanced_accuracy=0.4777702998407567, roc_auc=0.4699865029366227, pr_auc=0.3380017032654373, survives=false
- label_ret3d_gt_100bp|base_39_plus_past_daily_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy: balanced_accuracy=0.5006768405085062, roc_auc=0.4985228951255539, pr_auc=0.3669226880607933, survives=false
- label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy: balanced_accuracy=0.5110493757809123, roc_auc=0.5204288335767566, pr_auc=0.5578311158343388, survives=false
