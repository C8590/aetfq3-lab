本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# Intraday Lab Monitor Refresh Gate

Read-only freshness gate for the registered Lab monitor candidate. It checks manual package/raw export freshness, existing rolling-origin outputs, and post-sprint anchor thresholds without rerunning validation, fitting models, changing thresholds, or producing Stable evidence.

- readiness_decision: LAB_MONITOR_REFRESH_NOT_DUE
- candidate_id: `label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy`
- current_monitor_status: `LAB_MONITOR_CANDIDATE_REVIEW_READY`
- manual_package_latest_date: 2026-06-12
- last_rolling_origin_latest_date: 2026-06-12
- post_sprint_anchor_count: 7
- post_sprint_group_count: 56
- refresh_due: false
- next_allowed_task: `wait_for_new_data_or_manual_review`
- stable_promotion_ready: false
- stable_evidence: false
- qmt_ready: false
- order_intent_ready: false
