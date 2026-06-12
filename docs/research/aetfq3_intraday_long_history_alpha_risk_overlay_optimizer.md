# AETF Q3 Lab Intraday Long-History Risk Overlay Optimizer

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Scope

This document defines the v1 Lab-only risk overlay optimizer for long-history intraday alpha diagnostics. It starts from v0 rejected candidates, attributes drawdown by month, ETF, signal clock, and holding-period contribution, then runs bounded paper-only risk overlays.

## Boundary

- access_mode: READ_ONLY
- final_action_change_allowed: false
- contains_live_order: false
- contains_secret: false
- requires_human_review: true
- promotion_gate_required: true
- formal_training: false
- model_saved: false
- scaler_saved: false
- checkpoint_saved: false
- stable_promotion_ready: false
- stable_evidence: false
- qmt_ready: false
- order_intent_generated: false

The optimizer does not modify Stable thresholds, `target_weight`, `final_buy_action`, QMT execution, OrderIntent, Stable runtime, Stable output, or formal model artifacts.

## Attribution

Top rejected candidates are selected from v0 leaderboard by:

- top 10 `net_total_return`
- top 10 `calmar_like_ratio`
- top 10 `profit_factor`

The de-duplicated set is re-scored under rolling-origin validation and attributed by return, drawdown, win rate, profit factor, monthly win rate, worst/best month, drawdown period, ETF contribution, month contribution, signal-clock contribution, holding-period contribution, cost impact, and rejection reason.

## Risk Overlays

Allowed v1 overlays remain bounded and diagnostic-only:

- top-k per day selected by probability only
- min probability filter marked `threshold_search_lab_only=true`
- exposure caps and per-ETF sleeve caps
- holding period variants of 1d, 2d, and 3d
- predefined stop-loss and take-profit overlays
- train-only volatility and liquidity regime filters
- paper NAV drawdown throttle using past realized NAV only

No overlay may use future return to select entries.

## Candidate Gate

Candidates can only be marked `LAB_DIAGNOSTIC_ALPHA_CANDIDATE_REVIEW_REQUIRED`. Promising rows with uncontrolled drawdown can only be marked `PROMISING_BUT_DRAWDOWN_UNCONTROLLED_REVIEW_REQUIRED`. Neither status is Stable evidence.

The v1 gate requires positive net return, drawdown improved versus the v0 top raw candidate by at least 25% or drawdown no worse than -25%, win rate above 50%, profit factor above 1.05, monthly win rate and positive-month fraction at least 55%, no month/ETF domination, 10 bps per-side cost survival, no leakage, and no saved artifacts.

## Outputs

Runtime outputs are ignored and must stay under:

`.local_research_outputs/aetfq3_lab/intraday_long_history_alpha_risk_overlay_optimizer/`

The outputs are research diagnostics only and must not be treated as Stable plans, QMT inputs, or trading advice.
