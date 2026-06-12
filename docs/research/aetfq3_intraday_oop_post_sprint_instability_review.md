本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# Intraday OOP Post-Sprint Instability Review

Lab-only forensic review. It does not run a new model, train, tune, save model/scaler, connect QMT, generate OrderIntent, or create Stable evidence.

- readiness_decision: POST_SPRINT_INSTABILITY_REVIEW_COMPLETED_SAMPLE_TOO_SMALL_REVIEW_REQUIRED
- status: completed
- post_sprint_anchor_count: 7
- post_sprint_group_count: 32
- post_sprint_oop_underpowered: true
- missing_row_level_predictions: false
- stable_promotion_ready: false

## Focus Candidate

- family_id: label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy
- pre_group_share_of_combined: 0.9852941176470589
- post_reversal_observed: true
- threshold_sensitivity: {'available': True, 'current_threshold_0_5': {'split_name': 'post_sprint_oop', 'threshold': 0.5, 'row_count': 32, 'balanced_accuracy': 0.487012987012987, 'error_rate': 0.53125, 'false_positive': 12, 'false_negative': 5, 'prediction_positive_rate': 0.5625}, 'best_threshold_by_balanced_accuracy': {'split_name': 'post_sprint_oop', 'threshold': 0.4, 'row_count': 32, 'balanced_accuracy': 0.5974025974025974, 'error_rate': 0.5, 'false_positive': 15, 'false_negative': 1, 'prediction_positive_rate': 0.78125}, 'false_positive_count': 12, 'false_negative_count': 5}
