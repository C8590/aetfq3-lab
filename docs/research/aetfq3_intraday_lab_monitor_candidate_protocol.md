本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# Intraday Lab Monitor Candidate Protocol

This protocol registers one Lab-only monitor candidate. It is read-only research status, not Stable evidence, not trading advice, not QMT-ready, and not an OrderIntent source.

## Candidate Identity

- candidate_id: `label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy`
- label policy: `label_ret3d_gt_100bp`
- feature set: `base_39_plus_scale_transform_policy`
- model family: `logistic_balanced_scaled_variants`
- transform policy: `scale_transform_policy`
- automatic replacement of label / feature / model / threshold: false

## Monitor Status

- status: `LAB_MONITOR_CANDIDATE_REVIEW_READY`
- not a Stable candidate, trading strategy, advisory signal, OrderIntent source, QMT-ready model, or formal trained model

## Continuation Gates

- evaluable folds >= 6
- >= 60% folds BA > 0.5
- >= 60% folds ROC-AUC > 0.5
- PR-AUC not below label prevalence in >= 60% folds
- prediction non-collapse
- no month concentration, no ETF concentration, no leakage assertion failed, no model/scaler saved

## Promotion Boundary

Even sustained monitor stability cannot automatically enter Stable. Any Stable direction requires a separate human-review promotion gate and remains blocked from BUY/PROBE threshold changes, target_weight changes, final_buy_action changes, OrderIntent, QMT, and automatic promotion.
