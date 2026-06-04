# AETF Q3 Lab File Naming

## Purpose

This document defines file naming rules for `aetfq3-lab / Lab` research outputs. Names must make it obvious whether a file is local ignored research material, a small Git-trackable research summary, or a read-only Stable advisory package.

## Directory Rules

Use these default locations:

- Local ignored research outputs: `.local_research_outputs/aetfq3_lab/`
- Small Git-trackable summaries: `docs/research/`
- Advisory package schema target: `lab_advisory/`

`lab_advisory/` is a schema target only until a future task explicitly authorizes actual advisory package generation.

Forbidden output targets:

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

## Sector Research

Allowed names:

- `sector_rank_definition_comparison.md`
- `sector_rank_definition_comparison.json`
- `top1_sector_tplus_return_study.csv`
- `top1_sector_lock3_risk_report.md`
- `top1_sector_lock3_risk_report.json`

Git boundary:

- Markdown and small JSON summaries may live in `docs/research/`.
- CSV studies default to `.local_research_outputs/aetfq3_lab/` unless explicitly proven small, summarized, and safe to commit.

## Intraday Execution Research

Allowed names:

- `intraday_5m_execution_samples.csv`
- `top1_sector_intraday_execution_report.md`
- `top1_sector_intraday_execution_report.json`

Git boundary:

- 5-minute sample CSV files default to `.local_research_outputs/aetfq3_lab/`.
- Small Markdown or JSON reports may live in `docs/research/`.

## Intraday Watch Engine

Allowed names:

- `intraday_watch_snapshot.json`
- `intraday_watch_events.csv`
- `intraday_watch_strategy_report.md`

Git boundary:

- Snapshots and event CSV files default to `.local_research_outputs/aetfq3_lab/` unless they are small, anonymized, and explicitly approved for summary use.
- Strategy reports may live in `docs/research/`.

## QMT Lab

Allowed names:

- `qmt_readonly_report.json`
- `qmt_mock_execution_report.json`
- `qmt_execution_risk_report.md`

Git boundary:

- QMT raw logs and readonly responses default to `.local_research_outputs/aetfq3_lab/`.
- Small readonly summaries may become advisory files only after human review.
- No Lab file may target QMT production directories.

## PyTorch Execution Model

Allowed names:

- `pytorch_intraday_execution_model_plan.md`
- `intraday_execution_model_baseline_report.md`
- `model_diagnostics.json`

Git boundary:

- Model plans and small diagnostics may live in `docs/research/` or advisory schema output if approved.
- Training logs, checkpoints, tensor dumps, sample arrays, and model weights default to `.local_research_outputs/aetfq3_lab/` and must not enter Git.

## Naming Principles

- Prefer explicit domain prefixes: `sector_`, `intraday_`, `qmt_`, `pytorch_`, `model_`.
- Use `_report` for human-readable conclusions.
- Use `_diagnostics` for model health and validation notes.
- Use `_samples` or `_events` for row-level data that should usually stay local ignored.
- Do not use names that look like Stable runtime outputs.
- Do not use names that imply a formal trading instruction.

## Stable Boundary

Lab file names must not imply direct Stable action. If a file recommends Stable adoption, it must include:

1. Minimal merge plan.
2. Risk points.
3. Stable-side human approval requirement.
4. `RiskGate` checkpoint.
5. Rollback plan.
6. Statement that Lab cannot directly commit to Stable.
