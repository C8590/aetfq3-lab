# AETF Q3 Lab Output Contract

## Scope

This contract defines where Lab research outputs may live, which files may enter Git, and how Stable advisory packages must remain read-only. It applies to `aetfq3-lab / Lab`, not V2.1 Stable.

Lab can suggest Stable changes, but it cannot directly modify Stable, `final_buy_action`, `target_weight`, BUY / PROBE thresholds, Stable runtime files, QMT production settings, or formal `OrderIntent` outputs.

The third-repository protocol reference is reserved for the communication contract, protocol, bundle validation, and upgrade gate repository. The concrete protocol repo, tag, and freeze commit are not yet human-confirmed and must be recorded as pending:

```text
protocol_reference: pending human confirmation
protocol_repo: pending
protocol_version: pending
protocol_freeze_commit: pending
protocol_tag: pending
```

Before the protocol repo, tag, and freeze commit are human-confirmed, this repository must not rely on any hardcoded protocol rc1 anchor as a Stable adoption basis. Any task that cites protocol rc1 must verify the protocol repo, tag, and commit first.

## Layer 1: Local Ignored Research Outputs

Default directory:

```text
.local_research_outputs/aetfq3_lab/
```

Used for:

- Large CSV files.
- Backtest detail rows.
- Intermediate samples.
- Model training logs.
- Local experiment results.
- Chart caches.
- Temporary JSON files.
- PyTorch training outputs.
- QMT mock / readonly raw logs.

Rules:

- Default to ignored and out of Git.
- For local research and human review only.
- Must not be read directly by Stable.
- Must not be placed under `output/`.
- Must not be treated as a formal trading plan.

## Layer 2: Small Research Summaries That May Enter Git

Default directory:

```text
docs/research/
```

Allowed in Git:

- Small Markdown reports.
- Small JSON summaries.
- Method notes.
- Human review conclusions.
- Data scope notes.
- Risk boundary notes.

Forbidden in Git:

- Large CSV files.
- Raw market data.
- Training samples.
- Model weights.
- QMT raw responses.
- `output/` runtime products.

## Layer 3: Lab Advisory Packages

Default suggested directory:

```text
lab_advisory/
```

This repository currently defines the schema only and does not require generating or committing `lab_advisory/` files. If future tasks create advisory packages, they must remain read-only and must not directly change Stable formal results.

Allowed advisory package names:

- `ml_advisory_summary.json`
- `sector_research_report.json`
- `intraday_watch_research.json`
- `qmt_readonly_report.json`
- `model_diagnostics.json`
- `research_notes.md`

Rules:

- Advisory packages are read-only advice.
- Advisory package access mode must be `READ_ONLY`.
- `final_action_change_allowed` must be `false`.
- `contains_live_order` must be `false`.
- `contains_secret` must be `false`.
- They cannot directly change Stable `final_buy_action`, `target_weight`, BUY / PROBE thresholds, or risk gates.
- They cannot generate formal `OrderIntent`.
- They cannot trigger QMT or auto trading.
- They require Stable-side human review before any adoption.

## Forbidden Output Targets

Lab outputs must not target:

- `output/`
- Stable repository `output/`
- Stable `runtime/`
- Stable order intent directories
- `data/cache/`
- `artifacts/`, unless explicitly local ignored artifacts
- QMT production directories

中文边界原文：

- `output/`
- Stable 仓库 `output/`
- Stable `runtime/`
- Stable order intent 目录
- `data/cache/`
- `artifacts/`，除非明确是本地 ignored artifacts
- QMT 实盘目录

## Stable Boundary

If Lab recommends entering Stable, the recommendation must include:

1. Minimal merge plan.
2. Risk points.
3. Stable-side human approval requirement.
4. `RiskGate` checkpoints.
5. Rollback plan.
6. A clear statement that Lab is not allowed to commit directly to Stable.
7. A clear statement that adoption must pass a human promotion gate.

The default Stable effect of any Lab output is:

```text
affects_stable_trading: false
advisory_only: true
requires_human_review: true
access_mode: READ_ONLY
final_action_change_allowed: false
contains_live_order: false
contains_secret: false
```

Bundles that contain secrets, contain live orders, or set `final_action_change_allowed=true` are forbidden and must not be promoted.
