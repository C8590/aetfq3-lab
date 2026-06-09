本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

# AETF Q3 Lab Intraday Group-Level Past-Only Feature Expansion Design

## Task Scope

This document defines a Lab-only past-only feature expansion design for the F public intraday group-level diagnostic chain.

It is feature design, contract definition, and leakage checklist only. It does not generate a new sample, does not run no-save smoke, does not train a model, does not tune parameters, does not connect QMT, does not generate OrderIntent, and does not enter Stable.

## Prior Diagnostic Context

The design follows these completed Lab diagnostics:

- majority-class collapse diagnostic: `c449bd8`
- group-level sample dry-run: `0f096df`
- group-level readiness precheck: `0a6a8b6`
- group-level no-save diagnostic smoke: `d1b2ab6`
- group-level model signal closeout: `2b494e5`
- group label inconsistency diagnostic: `253f447`

Current reviewed state:

- group_count: `296`
- anchors: `37`
- ETFs: `8`
- group_label_policy: `anchor_close_last_bar`
- policy decision: `GROUP_LABEL_POLICY_ANCHOR_CLOSE_LAST_BAR_ACCEPTED_FOR_END_OF_DAY_DIAGNOSTIC`
- intraday_live_decision_ready: `false`
- ordinary logistic collapsed
- balanced/scaled reduces collapse but provides no formal evidence
- retained P1: `P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED`

## Time Semantics

Required semantics:

```text
feature_time_scope = anchor_day_only_or_prior
label_time_scope = after_anchor_day
group_label_policy = anchor_close_last_bar
intraday_live_decision_ready = false
```

Allowed feature sources:

- anchor day bars up to and including the last anchor-day bar
- daily or intraday history dated on or before the anchor day
- same-anchor-date cross-sectional statistics computed only from the eligible ETF universe visible for that anchor date

Forbidden feature sources:

- T+1, T+3, or any post-anchor-date outcome window
- future daily close after the anchor date
- future ETF universe membership
- QMT fills, account, fund, position, order, or trade fields
- Stable action fields, target weights, final action fields, or OrderIntent fields

## Design Purpose

The feature set is intended to make the next dry-run candidate richer while preserving strict past-only boundaries. It targets known diagnostic risks:

- weak univariate signal
- feature scale risk
- logistic majority-class collapse
- group repeated label structure

This design does not verify model effectiveness. Any later metric is diagnostic only unless a separate approved formal review says otherwise.

## Feature Categories

### A. Anchor-Day Intraday Structure

These features summarize the anchor day price path and bar-level return shape:

- `open_first`
- `high_max`
- `low_min`
- `close_last`
- `vwap_day`
- `day_return`
- `high_low_range`
- `close_to_vwap`
- `intraday_return_mean`
- `intraday_return_std`
- `intraday_return_skew`
- `intraday_return_min`
- `intraday_return_max`

### B. Volume / Amount Structure

These features summarize anchor-day activity, concentration, and late-session intensity:

- `volume_sum`
- `amount_sum`
- `volume_first_half_sum`
- `volume_second_half_sum`
- `amount_first_half_sum`
- `amount_second_half_sum`
- `volume_second_half_ratio`
- `amount_second_half_ratio`
- `volume_spike_ratio`
- `amount_spike_ratio`

### C. Intraday Trend / Reversal

These features summarize the direction and reversal profile inside the anchor day:

- `morning_return`
- `afternoon_return`
- `last_hour_return`
- `close_vs_morning_high`
- `close_vs_intraday_high`
- `close_vs_intraday_low`
- `vwap_slope_proxy`
- `price_above_vwap_bar_ratio`

### D. Past Daily Context

These features use only daily data dated on or before the anchor day:

- `prev_1d_return`
- `prev_3d_return`
- `prev_5d_return`
- `prev_10d_return`
- `prev_5d_volatility`
- `prev_10d_volatility`
- `prev_5d_volume_zscore`
- `prev_close_to_5d_ma`
- `prev_close_to_10d_ma`

### E. Cross-Sectional Relative Features

These features are computed within the same anchor date and the same visible ETF universe:

- `rank_day_return`
- `rank_volume_sum`
- `rank_amount_sum`
- `rank_close_to_vwap`
- `relative_return_to_universe_mean`
- `relative_volume_to_universe_mean`

## Feature Contract

Every candidate feature must carry these fields in any future implementation contract:

- `name`
- `category`
- `formula`
- `required_input_fields`
- `lookback_window`
- `uses_anchor_day_data`
- `uses_prior_day_data`
- `uses_future_data=false`
- `allowed_for_feature_columns`
- `known_risk`

All allowed features in this design set `uses_future_data=false` and `allowed_for_feature_columns=true`. Derived ratios must handle zero or missing denominators explicitly. Scaling for any later supervised diagnostic must fit transform parameters on train only where train-only statistics are required.

## Forbidden Features

The following fields and patterns must not enter `feature_columns`:

- `future_return_1d`
- `future_return_3d`
- `max_drawdown_3d`
- `three_day_positive_label`
- `buy_now_label`
- `wait_pullback_label`
- `cancel_buy_label`
- any `future_*`
- any `*_label`
- any `outcome_*`
- QMT fills, account, fund, position, order, or trade fields
- realized trading result
- Stable action fields
- OrderIntent fields

## Leakage Controls

P0 blockers for the next dry-run implementation:

- feature uses future daily close after anchor date
- feature uses T+1/T+3 outcome
- feature uses `three_day_positive_label`
- feature uses any `future_*`
- feature uses any `*_label`
- feature uses QMT fill / order / account data
- feature uses Stable action / `target_weight` / `final_buy_action`
- feature uses OrderIntent
- feature calculated with train+valid combined statistics where train-only is required
- cross-sectional rank computed using future universe membership

## Boundary

- stable_allowed: `false`
- qmt_allowed: `false`
- order_intent_allowed: `false`
- training_allowed: `false`
- model_save_allowed: `false`
- checkpoint_allowed: `false`
- automatic_promotion_ready: `false`
- metrics_are_effectiveness_evidence: `false`

## Next Allowed Action

`GROUP_LEVEL_PAST_ONLY_FEATURE_EXPANSION_DRY_RUN_ALLOWED_REVIEW_REQUIRED`

This only allows a later, separately approved dry-run task to materialize the feature contract and run intake-style leakage checks. It does not allow training, Stable promotion, QMT, OrderIntent, model deployment, or trading conclusions.
