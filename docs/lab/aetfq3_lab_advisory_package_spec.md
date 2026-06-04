# AETF Q3 Lab Advisory Package Spec

## Purpose

Lab advisory packages are read-only handoff artifacts from `aetfq3-lab / Lab` to Stable reviewers. They summarize research evidence, risk notes, and possible minimal Stable adoption plans. They are not formal trading outputs.

This document defines schema only. It does not require generating or committing `lab_advisory/` files.

## Allowed Files

Allowed advisory package files:

- `ml_advisory_summary.json`
- `sector_research_report.json`
- `intraday_watch_research.json`
- `qmt_readonly_report.json`
- `model_diagnostics.json`
- `research_notes.md`

## Required JSON Fields

Every advisory JSON must contain:

```json
{
  "schema_version": "1.0",
  "generated_at": "ISO-8601 timestamp",
  "lab_name": "aetfq3-lab",
  "source_task": "task title or task id",
  "data_sources": [],
  "uses_stable_bundle": false,
  "has_future_leakage_check": false,
  "affects_stable_trading": false,
  "advisory_only": true,
  "recommended_for_stable": false,
  "requires_human_review": true,
  "stable_merge_minimal_plan": "",
  "forbidden_actions": [
    "do_not_modify_final_buy_action",
    "do_not_modify_target_weight",
    "do_not_generate_order_intent",
    "do_not_bypass_riskgate",
    "do_not_auto_trade"
  ],
  "summary": "",
  "evidence_files": [],
  "risk_notes": []
}
```

## Field Rules

- `schema_version`: Advisory schema version.
- `generated_at`: Creation timestamp.
- `lab_name`: Must identify `aetfq3-lab`.
- `source_task`: Must identify the Lab task that produced the advisory.
- `data_sources`: Must list data origin, sample window, and whether data is local, external, or from a Stable bundle.
- `uses_stable_bundle`: Must be explicit.
- `has_future_leakage_check`: Must say whether a future leakage check was performed.
- `affects_stable_trading`: Must default to `false`.
- `advisory_only`: Must default to `true`.
- `recommended_for_stable`: If `true`, `stable_merge_minimal_plan` must be non-empty.
- `requires_human_review`: Must default to `true`.
- `stable_merge_minimal_plan`: Must describe the smallest Stable-side change needed, if any.
- `forbidden_actions`: Must include all required forbidden actions.
- `summary`: Must be concise and evidence-backed.
- `evidence_files`: Must list small committed summaries or local ignored evidence paths.
- `risk_notes`: Must list data, model, market, execution, and Stable integration risks.

## Stable Merge Requirements

If `recommended_for_stable` is `true`, the advisory must include:

1. Minimal merge plan.
2. Risk points.
3. Stable-side human approval requirement.
4. `RiskGate` checkpoint.
5. Rollback plan.
6. Statement that Lab cannot directly commit to Stable.

## Forbidden Interpretations

An advisory package must not be interpreted as:

- A Stable trading plan.
- A formal `OrderIntent`.
- A QMT execution command.
- A direct update to `final_buy_action`.
- A direct update to `target_weight`.
- A direct update to BUY / PROBE thresholds.
- Permission to bypass `RiskGate`.
- Permission to auto trade.
