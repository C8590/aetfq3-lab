# AETFQ3 Intraday Lab Monitor Candidate Paper PnL Replay

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Purpose

This document defines the Lab-only paper PnL replay for the registered intraday monitor candidate:

`label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy`

The replay is a simulated account diagnostic. It is not OOP validation, not model training, not threshold tuning, not QMT integration, not OrderIntent generation, not Stable evidence, and not promotion.

## Signal Source

The runner may only read existing rolling-origin row-level predictions:

`.local_research_outputs/aetfq3_lab/intraday_rolling_origin_walk_forward_no_save_validation/rolling_origin_row_level_predictions.csv`

It filters to the fixed monitor candidate and the existing `logistic_balanced_scaled` model rows. It must not train, refit, regenerate predictions, add candidates, or compare thresholds for selection.

## Execution Protocol

- `prediction=1` means a paper long signal.
- `prediction=0` means no position.
- The threshold remains the validator default `0.5`.
- Entry uses the next trading day's first available 5m open.
- Exit uses the T+3 trading day's last available 5m close.
- Missing entry or exit bars skip that paper trade and record the reason.

## Portfolio Model

- Initial paper cash: `1,000,000`.
- No leverage.
- No shorting.
- Cash earns zero.
- Each anchor date creates one 3-day sleeve.
- Each sleeve uses one third of current paper equity notional.
- Positive ETFs inside a sleeve are equal-weighted.
- Overlapping sleeves are allowed, with risk budget capped at 100% paper equity.

## Costs

The base replay applies `8 bps` per side:

- commission: `3 bps`
- slippage: `5 bps`

The runner also emits read-only cost sensitivity at `0`, `5`, `8`, `10`, and `20` bps per side. Cost sensitivity must not be used to select parameters.

## Outputs

The runner writes only ignored Lab research outputs under:

`.local_research_outputs/aetfq3_lab/intraday_lab_monitor_candidate_paper_pnl_replay/`

Expected files:

- `paper_pnl_replay_report.md`
- `paper_pnl_replay_report.json`
- `paper_pnl_nav.csv`
- `paper_pnl_sleeves.csv`
- `paper_pnl_simulated_trades.csv`
- `paper_pnl_monthly_returns.csv`
- `paper_pnl_etf_contribution.csv`
- `paper_pnl_cost_sensitivity.csv`
- `paper_pnl_benchmark_comparison.csv`
- `paper_pnl_decision.json`

## Decision Boundary

Possible decisions:

- `PAPER_PNL_REPLAY_COMPLETED_REVIEW_REQUIRED`
- `PAPER_PNL_REPLAY_PROFITABILITY_OBSERVED_REVIEW_REQUIRED`
- `PAPER_PNL_REPLAY_NO_PROFITABILITY_OBSERVED_REVIEW_REQUIRED`
- `PAPER_PNL_REPLAY_BLOCKED_MISSING_ROW_LEVEL_PREDICTIONS`
- `PAPER_PNL_REPLAY_BLOCKED_PRICE_DATA`
- `PAPER_PNL_REPLAY_BLOCKED_DATA_QUALITY`
- `PAPER_PNL_REPLAY_BLOCKED_SIGNAL_EMPTY`

Profitability observed remains review-only and must keep all Stable, QMT, OrderIntent, formal training, and automatic promotion flags false.

## Boundary

- No Stable modification.
- No QMT, xtdata, account, order, position, trade, or fill API.
- No OrderIntent.
- No real order.
- No training, refit, tuning, model save, scaler save, checkpoint save, GPU, or torchrun.
- No advisory package.
- No automatic promotion.
