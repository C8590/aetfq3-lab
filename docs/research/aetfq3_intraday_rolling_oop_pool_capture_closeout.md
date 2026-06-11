# AETF Q3 Lab Intraday Rolling OOP Pool Capture Closeout

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## Purpose

This is the closeout for the rolling OOP pool capture runner. It records that the repeatable Lab-only capture mechanism is complete and that the current OOP pool remains in accumulating status.

This is not OOP validation. It does not run fixed-shortlist validation, labels, models, training, QMT, OrderIntent, Stable runtime/output, or Stable promotion.

## Completed Runner

- Tool path: `tools/lab/intraday_rolling_oop_pool_capture.py`
- Commit: `a5ae839`
- Tests: `tests/lab` passed with `262 passed, 2 warnings`
- Append-only pool behavior: existing rows are preserved; identical duplicate keys are skipped
- Conflict detection: same-key different-value rows are recorded as conflicts while the existing row is kept
- Strict OOP anchor calculation: anchor date must be outside `2026-04-09` to `2026-06-03`, have complete public 5m bars, and have T+1/T+3 daily coverage
- Output boundary: tool writes only ignored `.local_artifact_backup` and `.local_research_outputs` pool/readiness artifacts

## Current OOP Pool Status

- 5m rows: `15760`
- daily rows: `336`
- 5m coverage range: `2026-04-09` to `2026-06-10`
- eligible strict OOP anchors: `2026-06-04`, `2026-06-05`
- eligible strict OOP anchor count: `2`
- ETF count: `8`
- group_count: `16`
- readiness decision: `ROLLING_OOP_POOL_LIMITED_ACCUMULATING`

## Thresholds

- eligible OOP anchors >= `10`
- ETF count >= `5`
- group_count >= `50`

## Current Blocker

The pool currently has only `2` eligible strict OOP anchors and `16` eligible groups. Fixed-shortlist OOP validation remains blocked until the threshold is met.

## Next Operation

- Run `tools/lab/intraday_rolling_oop_pool_capture.py` periodically to append future public rolling 5m windows.
- Do not run OOP validation while eligible strict OOP anchors remain below `10` or group_count remains below `50`.
- If the threshold is reached, create a separate fixed-shortlist OOP no-save validation task with explicit human review and Lab-only boundaries.

## Boundary

- no Stable
- no QMT
- no OrderIntent
- no training
- no model save
- no scaler save
- no checkpoint save
- no advisory
- no promotion
- no `output/`
- no Stable runtime/output
- no `lab_advisory/`
- no trading advice

## Final Review Status

The capture runner is complete and ready for periodic Lab-only use. The OOP pool is not ready for fixed-shortlist OOP validation. The pool readiness is not model effectiveness evidence, not trading advice, not a Stable advisory package, and not permission for automatic promotion.
