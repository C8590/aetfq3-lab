from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts, load_csv_rows, load_json


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_sample_dryrun")
REPORT_TYPE = "intraday_group_level_sample_dryrun"
SAMPLE_NAME = "intraday_group_level_samples.csv"
MANIFEST_NAME = "intraday_group_level_manifest.json"
TARGET_COLUMN = "three_day_positive_label"
GROUP_LABEL_POLICY = "anchor_close_last_bar"
READY = "GROUP_LEVEL_SAMPLE_DRY_RUN_PASSED_REVIEW_REQUIRED"
READY_CLASS_DIVERSE = "GROUP_LEVEL_SAMPLE_DRY_RUN_PASSED_CLASS_DIVERSE_REVIEW_REQUIRED"
BLOCKED_SINGLE_CLASS = "BLOCKED_GROUP_LEVEL_SINGLE_CLASS_LABEL"
BLOCKED_INSUFFICIENT_GROUPS = "BLOCKED_GROUP_LEVEL_INSUFFICIENT_GROUPS"
BLOCKED_LABEL_INCONSISTENCY = "BLOCKED_GROUP_LABEL_INCONSISTENCY"
BLOCKED_MANIFEST_LEAKAGE_P0 = "BLOCKED_MANIFEST_LEAKAGE_P0"
BLOCKED_BOUNDARY_FLAG = "BLOCKED_BOUNDARY_FLAG"
MIN_GROUPS = 30
MIN_ANCHORS = 5
MIN_ETFS = 2
ANCHOR_COLUMN_CANDIDATES = ("anchor_date", "trade_date")
BOUNDARY_FALSE_FIELDS = [
    "training_allowed",
    "supervised_training_allowed",
    "stable_effect_allowed",
    "contains_order_intent",
    "contains_live_order",
    "contains_secret",
]
BASE_GROUP_FEATURE_COLUMNS = [
    "open_first",
    "high_max",
    "low_min",
    "close_last",
    "volume_sum",
    "amount_sum",
    "vwap_day",
    "day_return",
    "high_low_range",
    "close_to_vwap",
    "intraday_return_mean",
    "intraday_return_std",
    "distance_to_vwap_mean",
    "distance_to_vwap_last",
    "volume_first_half_sum",
    "volume_second_half_sum",
    "amount_first_half_sum",
    "amount_second_half_sum",
]
EXPLICIT_FORBIDDEN_FEATURES = {
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
    "three_day_positive_label",
}
OUTCOME_PATTERNS = ("execution", "outcome")


class GroupLevelDryRunError(RuntimeError):
    pass


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def resolve_output_dir(out_dir: Path, repo_root: Path = REPO_ROOT) -> Path:
    repo_root = repo_root.resolve()
    resolved = resolve_repo_path(out_dir, repo_root).resolve()
    allowed = (repo_root / ALLOWED_OUTPUT_DIR).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise GroupLevelDryRunError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def run_group_level_dryrun(
    samples_path: Path,
    manifest_path: Path,
    diagnostic_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
    feature_columns_override: Sequence[str] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(samples_path, repo_root)
    resolved_manifest = resolve_repo_path(manifest_path, repo_root)
    resolved_diagnostic = resolve_repo_path(diagnostic_path, repo_root)
    for required_path, label in (
        (resolved_samples, "samples"),
        (resolved_manifest, "manifest"),
        (resolved_diagnostic, "diagnostic"),
    ):
        if not required_path.exists():
            raise GroupLevelDryRunError(f"{label} path does not exist: {required_path}")

    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    source_manifest = load_json(resolved_manifest)
    diagnostic = load_json(resolved_diagnostic)
    rows, columns = load_csv_rows(resolved_samples)
    anchor_column = next((column for column in ANCHOR_COLUMN_CANDIDATES if column in columns), None)
    if anchor_column is None:
        raise GroupLevelDryRunError("samples must contain anchor_date or trade_date")
    if "etf_code" not in columns:
        raise GroupLevelDryRunError("samples must contain etf_code")

    feature_columns = list(feature_columns_override or BASE_GROUP_FEATURE_COLUMNS)
    feature_leakage_check = run_feature_leakage_check(feature_columns)
    source_boundary_check = run_boundary_check(source_manifest)
    groups = build_groups(rows, anchor_column)
    group_rows, group_stats = build_group_rows(groups, anchor_column)
    class_balance_precheck = build_class_balance_precheck(group_rows, anchor_column)
    generated_manifest = build_group_manifest(source_manifest, feature_columns, anchor_column)
    manifest_path_out = resolved_out_dir / MANIFEST_NAME
    sample_path_out = resolved_out_dir / SAMPLE_NAME

    write_group_samples(sample_path_out, group_rows)
    manifest_path_out.write_text(json.dumps(generated_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_leakage_check = check_label_manifest(manifest_path_out)
    artifact_check = check_model_artifacts(resolved_out_dir)

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(source_boundary_check["p0_blockers"])
    p0_blockers.extend(feature_leakage_check["p0_blockers"])
    p0_blockers.extend(manifest_leakage_check.p0_blockers)
    p1_warnings.extend(manifest_leakage_check.p1_warnings)
    p0_blockers.extend(class_balance_precheck["p0_blockers"])
    p1_warnings.extend(class_balance_precheck["p1_warnings"])
    p0_blockers.extend(artifact_check["p0_blockers"])

    readiness_decision = decide_readiness(
        source_boundary_check,
        feature_leakage_check,
        manifest_leakage_check.ok,
        class_balance_precheck,
        artifact_check,
    )
    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": REPORT_TYPE,
        "status": "blocked" if readiness_decision.startswith("BLOCKED_") else "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "diagnostic_path": str(diagnostic_path),
        "diagnostic_decision": diagnostic.get("diagnostic_decision"),
        "diagnostic_flags": diagnostic.get("diagnostic_flags", []),
        "group_label_policy": GROUP_LABEL_POLICY,
        "intraday_live_decision_ready": False,
        "raw_bar_row_count": len(rows),
        "group_count": len(group_rows),
        "group_count_smaller_than_bar_rows_expected": len(group_rows) < len(rows),
        "group_count_reduction_note": "group_count is expected to be much smaller than raw bar-level rows because one group represents one ETF on one eligible anchor date.",
        "group_key": [anchor_column, "etf_code"],
        "group_statistics": group_stats,
        "feature_columns": feature_columns,
        "feature_aggregation": describe_feature_aggregation(),
        "feature_leakage_check": feature_leakage_check,
        "manifest_leakage_check": manifest_leakage_check.to_summary(),
        "source_boundary_check": source_boundary_check,
        "artifact_check": artifact_check,
        "class_balance_precheck": class_balance_precheck,
        "readiness_decision": readiness_decision,
        "training_allowed": False,
        "supervised_training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "model_training_performed": False,
        "no_save_supervised_smoke_run": False,
        "hyperparameter_tuning": False,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "not_trading_advice": True,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }
    write_reports(report, resolved_out_dir)
    return report


def build_groups(rows: list[dict[str, str]], anchor_column: str) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        anchor = str(row.get(anchor_column, "")).strip()
        etf_code = str(row.get("etf_code", "")).strip()
        groups.setdefault((anchor, etf_code), []).append(row)
    return groups


def build_group_rows(
    groups: dict[tuple[str, str], list[dict[str, str]]],
    anchor_column: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    bars_per_group: list[int] = []
    inconsistent_examples: list[dict[str, Any]] = []
    single_label_group_count = 0
    inconsistent_label_group_count = 0
    null_label_group_count = 0
    for (anchor, etf_code), group_rows in sorted(groups.items()):
        sorted_rows = sorted(group_rows, key=sort_key_for_bar)
        last_bar = sorted_rows[-1]
        labels = {normalize_label(row.get(TARGET_COLUMN, "")) for row in sorted_rows}
        labels_no_null = {label for label in labels if label is not None}
        if not labels_no_null:
            null_label_group_count += 1
        if len(labels_no_null) <= 1:
            single_label_group_count += 1
        else:
            inconsistent_label_group_count += 1
            if len(inconsistent_examples) < 10:
                inconsistent_examples.append(
                    {
                        "anchor": anchor,
                        "etf_code": etf_code,
                        "labels": sorted(labels_no_null),
                        "rows": len(sorted_rows),
                    }
                )
        bars_per_group.append(len(sorted_rows))
        output_rows.append(build_one_group_row(anchor, etf_code, sorted_rows, last_bar, anchor_column))
    return output_rows, {
        "bars_per_group": summarize_numbers(bars_per_group),
        "label_consistency_policy": "record all labels within group; group label still uses anchor_close_last_bar",
        "single_label_group_count": single_label_group_count,
        "inconsistent_label_group_count": inconsistent_label_group_count,
        "null_label_group_count": null_label_group_count,
        "inconsistent_label_group_examples": inconsistent_examples,
    }


def build_one_group_row(
    anchor: str,
    etf_code: str,
    rows: list[dict[str, str]],
    last_bar: dict[str, str],
    anchor_column: str,
) -> dict[str, Any]:
    first_bar = rows[0]
    high_values = floats(rows, "high")
    low_values = floats(rows, "low")
    close_last = to_float(last_bar.get("close"))
    open_first = to_float(first_bar.get("open"))
    volume_values = floats(rows, "volume")
    amount_values = floats(rows, "amount")
    intraday_returns = floats(rows, "intraday_return")
    distance_to_vwap_values = floats(rows, "distance_to_vwap")
    midpoint = max(1, len(rows) // 2)
    volume_sum = sum(volume_values)
    amount_sum = sum(amount_values)
    vwap_day = amount_sum / volume_sum if volume_sum else math.nan
    row = {
        "trade_date": anchor,
        "anchor_date": anchor if anchor_column == "anchor_date" else "",
        "etf_code": etf_code,
        "etf_name": last_bar.get("etf_name", ""),
        "bar_count": len(rows),
        "last_bar_datetime": last_bar.get("datetime", ""),
        "group_label_policy": GROUP_LABEL_POLICY,
        "open_first": open_first,
        "high_max": max(high_values) if high_values else math.nan,
        "low_min": min(low_values) if low_values else math.nan,
        "close_last": close_last,
        "volume_sum": volume_sum,
        "amount_sum": amount_sum,
        "vwap_day": vwap_day,
        "day_return": (close_last / open_first - 1.0) if open_first else math.nan,
        "high_low_range": (max(high_values) / min(low_values) - 1.0) if high_values and low_values and min(low_values) else math.nan,
        "close_to_vwap": (close_last / vwap_day - 1.0) if vwap_day else math.nan,
        "intraday_return_mean": safe_mean(intraday_returns),
        "intraday_return_std": safe_std(intraday_returns),
        "distance_to_vwap_mean": safe_mean(distance_to_vwap_values),
        "distance_to_vwap_last": to_float(last_bar.get("distance_to_vwap")),
        "volume_first_half_sum": sum(floats(rows[:midpoint], "volume")),
        "volume_second_half_sum": sum(floats(rows[midpoint:], "volume")),
        "amount_first_half_sum": sum(floats(rows[:midpoint], "amount")),
        "amount_second_half_sum": sum(floats(rows[midpoint:], "amount")),
        "future_return_1d": to_float(last_bar.get("future_return_1d")),
        "future_return_3d": to_float(last_bar.get("future_return_3d")),
        "max_drawdown_3d": to_float(last_bar.get("max_drawdown_3d")),
        "buy_now_label": "",
        "wait_pullback_label": "",
        "cancel_buy_label": "",
        TARGET_COLUMN: normalize_label(last_bar.get(TARGET_COLUMN, "")),
        "label_status": last_bar.get("label_status", ""),
        "label_horizon": last_bar.get("label_horizon", ""),
    }
    return row


def build_group_manifest(source_manifest: dict[str, Any], feature_columns: list[str], anchor_column: str) -> dict[str, Any]:
    eligible_anchors = source_manifest.get("eligible_anchor_dates")
    if not isinstance(eligible_anchors, list):
        eligible_anchors = []
    return {
        "manifest_version": "intraday_group_level_three_day_label_dryrun_v1",
        "sample_type": "intraday_5m",
        "sample_subtype": "intraday_group_level_three_day_label_dryrun",
        "group_level_sample": True,
        "group_key": [anchor_column, "etf_code"],
        "group_label_policy": GROUP_LABEL_POLICY,
        "intraday_live_decision_ready": False,
        "eligible_anchor_subset_only": True,
        "eligible_anchor_dates": eligible_anchors,
        "label_generation_scope": "dry_run_only",
        "label_generation_dryrun_allowed": True,
        "label_generation_performed": True,
        "generated_outcomes": ["future_return_1d", "future_return_3d", "max_drawdown_3d"],
        "generated_labels": ["three_day_positive_label"],
        "blocked_labels": ["buy_now_label", "wait_pullback_label", "cancel_buy_label"],
        "feature_columns": feature_columns,
        "label_generated": True,
        "label_source_kind": "public_future_window_anchor_close_last_bar",
        "label_horizon": source_manifest.get("label_horizon", {"unit": "trading_day", "required_horizons": ["T+1", "T+3"]}),
        "label_generation_method": "anchor_close_last_bar_group_level_dryrun_v1",
        "label_columns": ["buy_now_label", "wait_pullback_label", "cancel_buy_label", "three_day_positive_label"],
        "outcome_columns": [
            "future_return_1d",
            "future_return_3d",
            "max_drawdown_3d",
            "execution_return_to_close",
            "execution_return_to_next_open",
            "execution_drawdown_after_entry",
            "expected_3d_return",
            "expected_3d_drawdown",
        ],
        "label_status_column": "label_status",
        "insufficient_future_window_policy": "set label null when future window is unavailable",
        "feature_label_overlap_check": True,
        "label_generation_authorized": True,
        "supervised_training_allowed": False,
        "training_allowed": False,
        "stable_effect_allowed": False,
        "contains_order_intent": False,
        "contains_live_order": False,
        "contains_secret": False,
        "model_training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "not_trading_advice": True,
    }


def run_boundary_check(manifest: dict[str, Any]) -> dict[str, Any]:
    p0_blockers = [
        f"{field_name} must be false"
        for field_name in BOUNDARY_FALSE_FIELDS
        if manifest.get(field_name) is not False
    ]
    return {
        "passed": not p0_blockers,
        "checked_fields": BOUNDARY_FALSE_FIELDS,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_feature_leakage_check(feature_columns: Sequence[str]) -> dict[str, Any]:
    feature_set = set(feature_columns)
    forbidden = sorted(feature_set & EXPLICIT_FORBIDDEN_FEATURES)
    future_columns = sorted(column for column in feature_set if column.startswith("future_"))
    label_columns = sorted(column for column in feature_set if column.endswith("_label"))
    outcome_columns = sorted(
        column for column in feature_set if any(pattern in column for pattern in OUTCOME_PATTERNS)
    )
    p0_blockers: list[str] = []
    if forbidden:
        p0_blockers.append("feature_columns contains explicitly forbidden fields: " + ", ".join(forbidden))
    if future_columns:
        p0_blockers.append("feature_columns contains future_* fields: " + ", ".join(future_columns))
    if label_columns:
        p0_blockers.append("feature_columns contains *_label fields: " + ", ".join(label_columns))
    if outcome_columns:
        p0_blockers.append("feature_columns contains execution/outcome fields: " + ", ".join(outcome_columns))
    return {
        "passed": not p0_blockers,
        "feature_columns": list(feature_columns),
        "explicit_forbidden_features": sorted(EXPLICIT_FORBIDDEN_FEATURES),
        "outcome_patterns": list(OUTCOME_PATTERNS),
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def build_class_balance_precheck(group_rows: list[dict[str, Any]], anchor_column: str) -> dict[str, Any]:
    labels = [normalize_label(row.get(TARGET_COLUMN, "")) for row in group_rows]
    label_null_count = sum(1 for label in labels if label is None)
    label_0_count = sum(1 for label in labels if label == 0)
    label_1_count = sum(1 for label in labels if label == 1)
    class_count = int(label_0_count > 0) + int(label_1_count > 0)
    min_class_count = min((count for count in (label_0_count, label_1_count) if count > 0), default=0)
    anchors = sorted({str(row.get("trade_date", "")).strip() for row in group_rows if str(row.get("trade_date", "")).strip()})
    etfs = sorted({str(row.get("etf_code", "")).strip() for row in group_rows if str(row.get("etf_code", "")).strip()})
    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    if label_null_count:
        p1_warnings.append(f"{TARGET_COLUMN} null count is {label_null_count}")
    if class_count < 2:
        p0_blockers.append("group-level label must contain both class 0 and class 1")
    if len(group_rows) < MIN_GROUPS:
        p0_blockers.append(f"group_count must be >= {MIN_GROUPS}")
    if len(anchors) < MIN_ANCHORS:
        p0_blockers.append(f"anchor_count must be >= {MIN_ANCHORS}")
    if len(etfs) < MIN_ETFS:
        p0_blockers.append(f"etf_count must be >= {MIN_ETFS}")
    return {
        "passed": not p0_blockers,
        "group_count": len(group_rows),
        "anchor_count": len(anchors),
        "anchor_column": anchor_column,
        "etf_count": len(etfs),
        "label_null_count": label_null_count,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "positive_rate": label_1_count / (label_0_count + label_1_count) if (label_0_count + label_1_count) else None,
        "class_count": class_count,
        "min_class_count": min_class_count,
        "training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }


def decide_readiness(
    boundary_check: dict[str, Any],
    feature_leakage_check: dict[str, Any],
    manifest_ok: bool,
    class_balance_precheck: dict[str, Any],
    artifact_check: dict[str, Any],
) -> str:
    if not boundary_check["passed"]:
        return BLOCKED_BOUNDARY_FLAG
    if not feature_leakage_check["passed"] or not manifest_ok:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    if not artifact_check["passed"]:
        return BLOCKED_BOUNDARY_FLAG
    if class_balance_precheck["class_count"] < 2:
        return BLOCKED_SINGLE_CLASS
    if class_balance_precheck["group_count"] < MIN_GROUPS:
        return BLOCKED_INSUFFICIENT_GROUPS
    if class_balance_precheck["p0_blockers"]:
        return BLOCKED_INSUFFICIENT_GROUPS
    if class_balance_precheck["min_class_count"] >= 2:
        return READY_CLASS_DIVERSE
    return READY


def write_group_samples(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "trade_date",
        "anchor_date",
        "etf_code",
        "etf_name",
        "bar_count",
        "last_bar_datetime",
        "group_label_policy",
        *BASE_GROUP_FEATURE_COLUMNS,
        "future_return_1d",
        "future_return_3d",
        "max_drawdown_3d",
        "buy_now_label",
        "wait_pullback_label",
        "cancel_buy_label",
        TARGET_COLUMN,
        "label_status",
        "label_horizon",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "intraday_group_level_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "class_balance_precheck.json").write_text(
        json.dumps(report["class_balance_precheck"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "readiness_decision": report["readiness_decision"],
        "status": report["status"],
        "group_label_policy": GROUP_LABEL_POLICY,
        "intraday_live_decision_ready": False,
        "group_count": report["group_count"],
        "class_count": report["class_balance_precheck"]["class_count"],
        "training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }
    (out_dir / "readiness_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "",
        "# Intraday Group-Level Sample Dry Run",
        "",
        "本文件只用于 Lab group-level dry-run 样本设计与 readiness precheck，不训练模型，不运行 no-save supervised smoke，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。",
        "",
        f"- report_type: {report['report_type']}",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- group_label_policy: {report['group_label_policy']}",
        f"- intraday_live_decision_ready: {str(report['intraday_live_decision_ready']).lower()}",
        f"- raw_bar_row_count: {report['raw_bar_row_count']}",
        f"- group_count: {report['group_count']}",
        f"- group_count_reduction_note: {report['group_count_reduction_note']}",
        f"- bars_per_group: {json.dumps(report['group_statistics']['bars_per_group'], ensure_ascii=False, sort_keys=True)}",
        f"- single_label_group_count: {report['group_statistics']['single_label_group_count']}",
        f"- inconsistent_label_group_count: {report['group_statistics']['inconsistent_label_group_count']}",
        f"- label_0_count: {report['class_balance_precheck']['label_0_count']}",
        f"- label_1_count: {report['class_balance_precheck']['label_1_count']}",
        f"- positive_rate: {report['class_balance_precheck']['positive_rate']}",
        f"- training_allowed: {str(report['training_allowed']).lower()}",
        f"- stable_allowed: {str(report['stable_allowed']).lower()}",
        f"- qmt_allowed: {str(report['qmt_allowed']).lower()}",
        f"- order_intent_allowed: {str(report['order_intent_allowed']).lower()}",
        f"- metrics_are_effectiveness_evidence: {str(report['metrics_are_effectiveness_evidence']).lower()}",
    ]
    (out_dir / "intraday_group_level_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def describe_feature_aggregation() -> dict[str, str]:
    return {
        "open_first": "first 5m bar open on anchor day",
        "high_max": "max high over anchor-day bars",
        "low_min": "min low over anchor-day bars",
        "close_last": "last 5m bar close on anchor day",
        "volume_sum": "sum volume over anchor-day bars",
        "amount_sum": "sum amount over anchor-day bars",
        "vwap_day": "amount_sum / volume_sum",
        "day_return": "close_last / open_first - 1",
        "high_low_range": "high_max / low_min - 1",
        "close_to_vwap": "close_last / vwap_day - 1",
        "intraday_return_mean": "mean intraday_return over anchor-day bars",
        "intraday_return_std": "population std intraday_return over anchor-day bars",
        "distance_to_vwap_mean": "mean distance_to_vwap over anchor-day bars",
        "distance_to_vwap_last": "last 5m bar distance_to_vwap",
        "volume_first_half_sum": "sum volume over first half of anchor-day bars",
        "volume_second_half_sum": "sum volume over second half of anchor-day bars",
        "amount_first_half_sum": "sum amount over first half of anchor-day bars",
        "amount_second_half_sum": "sum amount over second half of anchor-day bars",
    }


def sort_key_for_bar(row: dict[str, str]) -> tuple[str, float]:
    return (str(row.get("datetime", "")), to_float(row.get("bar_index")))


def floats(rows: Sequence[dict[str, str]], column: str) -> list[float]:
    values = [to_float(row.get(column)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else math.nan


def safe_std(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return pstdev(finite) if len(finite) > 1 else 0.0


def normalize_label(value: Any) -> int | None:
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    return None


def summarize_numbers(values: Sequence[int | float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None, "mean": None}
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "median": median(numeric),
        "max": max(numeric),
        "mean": mean(numeric),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday group-level sample dry run.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_group_level_dryrun(args.samples, args.manifest, args.diagnostic, args.out_dir)
    except GroupLevelDryRunError as exc:
        print(
            json.dumps(
                {"status": "failed", "readiness_decision": BLOCKED_MANIFEST_LEAKAGE_P0, "p0_blockers": [str(exc)]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "readiness_decision": report["readiness_decision"],
                "group_count": report["group_count"],
                "label_0_count": report["class_balance_precheck"]["label_0_count"],
                "label_1_count": report["class_balance_precheck"]["label_1_count"],
                "training_allowed": report["training_allowed"],
                "stable_allowed": report["stable_allowed"],
                "qmt_allowed": report["qmt_allowed"],
                "order_intent_allowed": report["order_intent_allowed"],
                "metrics_are_effectiveness_evidence": report["metrics_are_effectiveness_evidence"],
                "p0_blockers": report["p0_blockers"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
