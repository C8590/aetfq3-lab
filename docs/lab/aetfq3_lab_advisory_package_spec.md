# AETF Q3 Lab Advisory Package Spec

## Purpose

Lab advisory packages are read-only handoff artifacts from `aetfq3-lab / Lab` to Stable reviewers. They summarize research evidence, risk notes, and possible minimal Stable adoption plans. They are not formal trading outputs.

This document defines schema only. It does not require generating or committing `lab_advisory/` files.

The third-repository protocol reference is reserved for the communication contract, protocol, bundle validation, and upgrade gate repository. The concrete protocol repo, tag, and freeze commit are not yet human-confirmed and must be recorded as pending:

```text
protocol_reference: pending human confirmation
protocol_repo: pending
protocol_version: pending
protocol_freeze_commit: pending
protocol_tag: pending
```

Before the protocol repo, tag, and freeze commit are human-confirmed, this repository must not rely on any hardcoded protocol rc1 anchor as a Stable adoption basis. Any task that cites protocol rc1 must verify the protocol repo, tag, and commit first.

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
  "protocol_reference": "pending human confirmation",
  "protocol_repo": "pending",
  "protocol_version": "pending",
  "protocol_freeze_commit": "pending",
  "protocol_tag": "pending",
  "source_task": "task title or task id",
  "data_sources": [],
  "uses_stable_bundle": false,
  "has_future_leakage_check": false,
  "access_mode": "READ_ONLY",
  "affects_stable_trading": false,
  "advisory_only": true,
  "final_action_change_allowed": false,
  "contains_live_order": false,
  "contains_secret": false,
  "recommended_for_stable": false,
  "requires_human_review": true,
  "stable_merge_minimal_plan": "",
  "forbidden_actions": [
    "do_not_modify_final_buy_action",
    "do_not_modify_target_weight",
    "do_not_generate_order_intent",
    "do_not_bypass_riskgate",
    "do_not_auto_trade",
    "do_not_modify_stable_directly",
    "do_not_write_stable_runtime",
    "do_not_write_stable_output",
    "do_not_skip_human_promotion_gate"
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
- `protocol_reference`: Must be `pending human confirmation` until the protocol repo, tag, and freeze commit are human-confirmed.
- `protocol_repo`: Must be `pending` until human confirmation.
- `protocol_version`: Must be `pending` until human confirmation.
- `protocol_freeze_commit`: Must be `pending` until human confirmation.
- `protocol_tag`: Must be `pending` until human confirmation.
- `source_task`: Must identify the Lab task that produced the advisory.
- `data_sources`: Must list data origin, sample window, and whether data is local, external, or from a Stable bundle.
- `uses_stable_bundle`: Must be explicit.
- `has_future_leakage_check`: Must say whether a future leakage check was performed.
- `access_mode`: Must be `READ_ONLY`.
- `affects_stable_trading`: Must default to `false`.
- `advisory_only`: Must default to `true`.
- `final_action_change_allowed`: Must be `false`.
- `contains_live_order`: Must be `false`.
- `contains_secret`: Must be `false`.
- `recommended_for_stable`: If `true`, `stable_merge_minimal_plan` must be non-empty.
- `requires_human_review`: Must default to `true`.
- `stable_merge_minimal_plan`: Must describe the smallest Stable-side change needed, if any.
- `forbidden_actions`: Must include all required forbidden actions, including `do_not_write_stable_runtime` and `do_not_write_stable_output`.
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
7. Statement that the proposal must pass a human promotion gate before Stable adoption.

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
- Permission for Lab to directly modify Stable.
- Permission to skip the human promotion gate.

## Forbidden Bundles

The following bundles are always forbidden:

- Any bundle with `contains_secret: true`.
- Any bundle with `contains_live_order: true`.
- Any bundle with `final_action_change_allowed: true`.
