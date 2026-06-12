from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_fixed_shortlist_oop_no_save_validation import load_json, write_csv, write_json  # noqa: E402
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts  # noqa: E402


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_lab_monitor_refresh_gate"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_lab_monitor_refresh_gate")
DEFAULT_MANUAL_INBOX = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox")
DEFAULT_RAW_EXPORT_DIR = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports")
DEFAULT_CANDIDATE_STATUS_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_lab_monitor_candidate_status")
DEFAULT_ROLLING_ORIGIN_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_rolling_origin_walk_forward_no_save_validation")
DEFAULT_ATTRIBUTION_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_rolling_origin_stability_attribution_review")
DEFAULT_FIXED_OOP_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_fixed_shortlist_oop_no_save_validation")
DEFAULT_REVERSAL_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_oop_post_sprint_reversal_attribution")
DEFAULT_OUT_DIR = ALLOWED_OUTPUT_DIR

FOCUS_CANDIDATE_ID = "label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy"
POST_SPRINT_ANCHOR_THRESHOLD = 10
POST_SPRINT_GROUP_THRESHOLD = 50

DECISION_NOT_DUE = "LAB_MONITOR_REFRESH_NOT_DUE"
DECISION_DUE_NEW_RAW = "LAB_MONITOR_REFRESH_DUE_NEW_RAW_EXPORT"
DECISION_DUE_NEW_MANUAL = "LAB_MONITOR_REFRESH_DUE_NEW_MANUAL_PACKAGE"
DECISION_DUE_NEW_ANCHORS = "LAB_MONITOR_REFRESH_DUE_NEW_ANCHORS"
DECISION_DUE_STALE = "LAB_MONITOR_REFRESH_DUE_STALE_OUTPUTS"
DECISION_DUE_MISSING = "LAB_MONITOR_REFRESH_DUE_MISSING_OUTPUTS"
DECISION_BLOCKED_MISSING_MANUAL = "LAB_MONITOR_REFRESH_BLOCKED_MISSING_MANUAL_PACKAGE"
DECISION_BLOCKED_DATA = "LAB_MONITOR_REFRESH_BLOCKED_DATA_QUALITY"
DECISION_BLOCKED_RETIRED = "LAB_MONITOR_REFRESH_BLOCKED_STATUS_RETIRED"
DECISION_REVIEW = "LAB_MONITOR_REFRESH_REVIEW_REQUIRED"

FORBIDDEN_NEXT_TASKS = [
    "stable_promotion",
    "qmt_trading",
    "order_intent_generation",
    "formal_training",
    "threshold_tuning",
    "target_weight_change",
    "final_buy_action_change",
]


class LabMonitorRefreshGateError(RuntimeError):
    pass


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def resolve_output_dir(out_dir: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    repo_root = repo_root.resolve()
    resolved = resolve_repo_path(out_dir, repo_root).resolve()
    if enforce:
        allowed = (repo_root / ALLOWED_OUTPUT_DIR).resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError as exc:
            raise LabMonitorRefreshGateError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def mtime(path: Path) -> float | None:
    return path.stat().st_mtime if path.exists() else None


def mtime_iso(path: Path) -> str | None:
    value = mtime(path)
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def latest_mtime(paths: Sequence[Path]) -> float | None:
    values = [value for value in (mtime(path) for path in paths) if value is not None]
    return max(values) if values else None


def latest_mtime_iso(paths: Sequence[Path]) -> str | None:
    value = latest_mtime(paths)
    if value is None:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def list_data_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(path for path in directory.iterdir() if path.is_file() and not path.name.startswith("."))


def load_optional_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return load_json(path)


def manual_package_paths(manual_inbox: Path) -> dict[str, Path]:
    return {
        "manifest": manual_inbox / "MANIFEST.json",
        "sha256": manual_inbox / "SHA256SUMS.txt",
        "csv": manual_inbox / "historical_5m_manual_export.csv",
    }


def manual_inbox_summary(manual_inbox: Path) -> dict[str, Any]:
    paths = manual_package_paths(manual_inbox)
    present = manual_inbox.exists() and all(path.exists() for path in paths.values())
    latest_date: str | None = None
    anchor_count = 0
    group_count = 0
    etf_count = 0
    date_to_etfs: dict[str, set[str]] = {}
    if paths["csv"].exists():
        with paths["csv"].open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                trade_date = str(row.get("trade_date") or row.get("datetime", "")[:10])
                etf_code = str(row.get("etf_code") or "")
                if not trade_date:
                    continue
                date_to_etfs.setdefault(trade_date, set())
                if etf_code:
                    date_to_etfs[trade_date].add(etf_code)
        if date_to_etfs:
            latest_date = max(date_to_etfs)
            anchor_count = len(date_to_etfs)
            group_count = sum(len(values) for values in date_to_etfs.values())
            etf_count = len({etf for values in date_to_etfs.values() for etf in values})
    return {
        "manual_inbox_present": present,
        "manual_package_latest_date": latest_date,
        "manual_anchor_count": anchor_count,
        "manual_group_count": group_count,
        "manual_etf_count": etf_count,
        "manual_package_files": {name: str(path) for name, path in paths.items()},
        "manual_package_latest_mtime_utc": latest_mtime_iso(list(paths.values())),
    }


def raw_export_summary(raw_export_dir: Path) -> dict[str, Any]:
    files = list_data_files(raw_export_dir)
    return {
        "raw_export_present": raw_export_dir.exists() and bool(files),
        "raw_export_file_count": len(files),
        "raw_export_latest_mtime_utc": latest_mtime_iso(files),
        "raw_export_files": [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "last_write_time_utc": mtime_iso(path),
            }
            for path in files
        ],
    }


def latest_rolling_date(rolling_report: dict[str, Any]) -> str | None:
    dates: list[str] = []
    for fold in rolling_report.get("fold_manifest", []):
        dates.extend(str(date) for date in fold.get("validation_anchor_dates", []) if date)
    return max(dates) if dates else None


def latest_evaluated_rolling_date(rolling_report: dict[str, Any]) -> str | None:
    dates: list[str] = []
    for fold in rolling_report.get("fold_manifest", []):
        if fold.get("skipped"):
            continue
        dates.extend(str(date) for date in fold.get("validation_anchor_dates", []) if date)
    return max(dates) if dates else None


def post_sprint_counts_from_rolling(rolling_report: dict[str, Any]) -> dict[str, Any]:
    folds = rolling_report.get("fold_manifest", [])
    if not folds:
        return {
            "post_sprint_anchor_count": 0,
            "post_sprint_raw_group_count": 0,
            "post_sprint_etf_count": 0,
            "post_sprint_fold_id": None,
            "post_sprint_fold_skipped": None,
            "post_sprint_skip_reasons": [],
        }
    fold = max(folds, key=lambda item: str(item.get("validation_month", "")))
    return {
        "post_sprint_anchor_count": int(fold.get("validation_anchor_count") or 0),
        "post_sprint_raw_group_count": int(fold.get("validation_group_count") or 0),
        "post_sprint_etf_count": int(fold.get("validation_etf_count") or 0),
        "post_sprint_fold_id": fold.get("fold_id"),
        "post_sprint_fold_skipped": bool(fold.get("skipped")),
        "post_sprint_skip_reasons": fold.get("skip_reasons", []),
    }


def load_fixed_oop_split_counts(fixed_oop_dir: Path) -> dict[str, Any]:
    path = fixed_oop_dir / "fixed_shortlist_oop_split_manifest.json"
    if not path.exists():
        return {
            "available": False,
            "source": "fixed_shortlist_oop_split_manifest_missing",
        }
    manifest = load_json(path)
    post = manifest.get("post_sprint_oop", {})
    evaluable = post.get("t_plus_3_covered_group_count")
    raw = post.get("group_count")
    labels = post.get("label_distribution", {})
    label_counts = [
        sum(int(count) for count in values.values())
        for values in labels.values()
        if isinstance(values, dict) and values
    ]
    if evaluable is None and label_counts:
        evaluable = min(label_counts)
    if evaluable is None:
        return {
            "available": False,
            "source": "fixed_shortlist_oop_split_manifest_without_evaluable_group_count",
        }
    return {
        "available": True,
        "source": "fixed_shortlist_oop_split_manifest.post_sprint_oop.t_plus_3_covered_group_count",
        "post_sprint_anchor_count": int(post.get("anchor_count") or 0),
        "post_sprint_raw_group_count": int(raw or 0),
        "post_sprint_evaluable_group_count": int(evaluable),
        "post_sprint_etf_count": int(post.get("etf_count") or 0),
    }


def load_fixed_oop_row_level_counts(fixed_oop_dir: Path) -> dict[str, Any]:
    path = fixed_oop_dir / "fixed_shortlist_oop_row_level_predictions.csv"
    if not path.exists():
        return {"available": False, "source": "fixed_shortlist_oop_row_level_predictions_missing"}
    groups: set[tuple[str, str]] = set()
    dates: set[str] = set()
    etfs: set[str] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("is_post_sprint_oop") != "True":
                continue
            if row.get("candidate_id") != FOCUS_CANDIDATE_ID:
                continue
            if row.get("model") != "logistic_balanced_scaled":
                continue
            if row.get("label") in {"", None}:
                continue
            if row.get("future_return_3d") in {"", None}:
                continue
            anchor_date = str(row.get("anchor_date") or "")
            etf_code = str(row.get("etf_code") or "")
            if not anchor_date or not etf_code:
                continue
            groups.add((anchor_date, etf_code))
            dates.add(anchor_date)
            etfs.add(etf_code)
    if not groups:
        return {"available": False, "source": "fixed_shortlist_oop_row_level_predictions_without_post_sprint_evaluable_rows"}
    return {
        "available": True,
        "source": "fixed_shortlist_oop_row_level_predictions.distinct_post_sprint_label_tplus3_groups",
        "post_sprint_anchor_count": len(dates),
        "post_sprint_evaluable_group_count": len(groups),
        "post_sprint_etf_count": len(etfs),
    }


def load_reversal_sample_power_counts(reversal_dir: Path) -> dict[str, Any]:
    path = reversal_dir / "post_sprint_reversal_attribution_report.json"
    if not path.exists():
        return {"available": False, "source": "post_sprint_reversal_attribution_report_missing"}
    report = load_json(path)
    sample_power = report.get("sample_power", {})
    group_count = sample_power.get("post_sprint_group_count")
    if group_count is None:
        return {"available": False, "source": "post_sprint_reversal_attribution_report_without_sample_power_group_count"}
    return {
        "available": True,
        "source": "post_sprint_reversal_attribution_report.sample_power.post_sprint_group_count",
        "post_sprint_anchor_count": int(sample_power.get("post_sprint_anchor_count") or 0),
        "post_sprint_evaluable_group_count": int(group_count),
    }


def post_sprint_evaluable_group_summary(
    fixed_oop_dir: Path,
    reversal_dir: Path,
    rolling_counts: dict[str, Any],
) -> dict[str, Any]:
    split_counts = load_fixed_oop_split_counts(fixed_oop_dir)
    row_counts = load_fixed_oop_row_level_counts(fixed_oop_dir)
    reversal_counts = load_reversal_sample_power_counts(reversal_dir)
    selected = split_counts if split_counts.get("available") else row_counts if row_counts.get("available") else reversal_counts
    if not selected.get("available"):
        return {
            "evaluable_group_count_available": False,
            "evaluable_group_count_source": selected.get("source"),
            "post_sprint_raw_group_count": rolling_counts["post_sprint_raw_group_count"],
            "post_sprint_evaluable_group_count": None,
            "post_sprint_gate_group_count": None,
            "group_count_basis": "evaluable_groups",
            "t_plus_1_coverage_passed": False,
            "t_plus_3_coverage_passed": False,
        }

    raw_group_count = int(selected.get("post_sprint_raw_group_count") or rolling_counts["post_sprint_raw_group_count"])
    evaluable_group_count = int(selected["post_sprint_evaluable_group_count"])
    return {
        "evaluable_group_count_available": True,
        "evaluable_group_count_source": selected["source"],
        "post_sprint_anchor_count": int(selected.get("post_sprint_anchor_count") or rolling_counts["post_sprint_anchor_count"]),
        "post_sprint_raw_group_count": raw_group_count,
        "post_sprint_evaluable_group_count": evaluable_group_count,
        "post_sprint_gate_group_count": evaluable_group_count,
        "post_sprint_etf_count": int(selected.get("post_sprint_etf_count") or rolling_counts["post_sprint_etf_count"]),
        "group_count_basis": "evaluable_groups",
        "t_plus_1_coverage_passed": evaluable_group_count > 0,
        "t_plus_3_coverage_passed": raw_group_count == evaluable_group_count,
        "cross_check_sources": {
            "fixed_oop_split_manifest": split_counts,
            "fixed_oop_row_level_predictions": row_counts,
            "post_sprint_reversal_attribution": reversal_counts,
        },
    }


def count_manual_dates_after(manual_csv: Path, date_exclusive: str | None) -> dict[str, int]:
    if date_exclusive is None or not manual_csv.exists():
        return {"new_anchor_count": 0, "new_group_count": 0}
    date_to_etfs: dict[str, set[str]] = {}
    with manual_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            trade_date = str(row.get("trade_date") or row.get("datetime", "")[:10])
            etf_code = str(row.get("etf_code") or "")
            if trade_date and trade_date > date_exclusive:
                date_to_etfs.setdefault(trade_date, set())
                if etf_code:
                    date_to_etfs[trade_date].add(etf_code)
    return {
        "new_anchor_count": len(date_to_etfs),
        "new_group_count": sum(len(values) for values in date_to_etfs.values()),
    }


def required_output_paths(
    candidate_status_dir: Path,
    rolling_origin_dir: Path,
    attribution_dir: Path,
) -> dict[str, Path]:
    return {
        "candidate_status_report": candidate_status_dir / "lab_monitor_candidate_status_report.json",
        "candidate_status_decision": candidate_status_dir / "lab_monitor_candidate_protocol_decision.json",
        "rolling_origin_report": rolling_origin_dir / "rolling_origin_walk_forward_report.json",
        "rolling_origin_decision": rolling_origin_dir / "rolling_origin_decision.json",
        "rolling_origin_fold_manifest": rolling_origin_dir / "rolling_origin_fold_manifest.json",
        "attribution_report": attribution_dir / "rolling_origin_stability_attribution_report.json",
        "attribution_decision": attribution_dir / "rolling_origin_stability_attribution_decision.json",
    }


def current_status(candidate_report: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    candidate_status = candidate_report.get("candidate_status", {})
    return (
        candidate_report.get("status_decision"),
        candidate_status.get("monitor_status"),
        candidate_status.get("candidate_id") or FOCUS_CANDIDATE_ID,
    )


def decide_refresh(
    *,
    manual_present: bool,
    missing_outputs: Sequence[str],
    status_decision: str | None,
    new_raw_export: bool,
    new_manual_package: bool,
    new_anchor_due: bool,
    evaluable_group_count_available: bool,
    stale_outputs: bool,
    p0_blockers: Sequence[str],
) -> tuple[str, str, str]:
    if not manual_present:
        return (
            DECISION_BLOCKED_MISSING_MANUAL,
            "manual inbox package is missing MANIFEST.json, SHA256SUMS.txt, or historical_5m_manual_export.csv",
            "restore_or_build_manual_package",
        )
    if status_decision and "RETIRED" in status_decision:
        return (
            DECISION_BLOCKED_RETIRED,
            "current monitor candidate status is retired",
            "manual_review_retired_candidate",
        )
    if missing_outputs:
        return (
            DECISION_DUE_MISSING,
            "required monitor, rolling-origin, or attribution outputs are missing",
            "rerun_missing_lab_monitor_outputs",
        )
    if p0_blockers:
        return (DECISION_BLOCKED_DATA, "data quality or artifact blocker detected", "manual_data_quality_review")
    if not evaluable_group_count_available:
        return (DECISION_REVIEW, "evaluable_group_count_unavailable", "manual_monitor_refresh_gate_review")
    if new_raw_export:
        return (
            DECISION_DUE_NEW_RAW,
            "raw export files are newer than the last monitor status",
            "run_broker_export_packager_and_manual_intake_validator",
        )
    if new_manual_package:
        return (
            DECISION_DUE_NEW_MANUAL,
            "manual inbox MANIFEST.json, SHA256SUMS.txt, or CSV is newer than the last monitor status",
            "run_fixed_shortlist_oop_and_rolling_origin_refresh",
        )
    if new_anchor_due:
        return (
            DECISION_DUE_NEW_ANCHORS,
            "post-sprint anchor/group thresholds are met for a bounded refresh",
            "rerun_fixed_shortlist_oop_no_save_validation_and_attribution",
        )
    if stale_outputs:
        return (
            DECISION_DUE_STALE,
            "rolling-origin or attribution outputs are older than their upstream inputs",
            "rerun_stale_lab_monitor_outputs",
        )
    if status_decision is None:
        return (DECISION_REVIEW, "monitor candidate status could not be read", "manual_monitor_status_review")
    return (DECISION_NOT_DUE, "no new raw/manual package and no post-sprint threshold refresh trigger", "wait_for_new_data_or_manual_review")


def status_csv_columns() -> list[str]:
    return [
        "candidate_id",
        "current_monitor_status",
        "manual_inbox_present",
        "raw_export_present",
        "manual_package_latest_date",
        "last_rolling_origin_latest_date",
        "new_data_detected",
        "new_raw_export_detected",
        "new_manual_package_detected",
        "new_anchor_count",
        "post_sprint_anchor_count",
        "post_sprint_group_count",
        "post_sprint_raw_group_count",
        "post_sprint_evaluable_group_count",
        "post_sprint_gate_group_count",
        "group_count_basis",
        "t_plus_1_coverage_passed",
        "t_plus_3_coverage_passed",
        "evaluable_group_count_source",
        "post_sprint_anchor_threshold",
        "post_sprint_group_threshold",
        "refresh_due",
        "refresh_reason",
        "readiness_decision",
        "next_allowed_task",
        "forbidden_next_tasks",
    ]


def build_docs(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    docs_json = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_lab_monitor_refresh_gate",
        "readiness_decision": report["readiness_decision"],
        "candidate_id": report["candidate_id"],
        "current_monitor_status": report["current_monitor_status"],
        "refresh_due": report["refresh_due"],
        "refresh_reason": report["refresh_reason"],
        "next_allowed_task": report["next_allowed_task"],
        "forbidden_next_tasks": FORBIDDEN_NEXT_TASKS,
        "post_sprint_raw_group_count": report["post_sprint_raw_group_count"],
        "post_sprint_evaluable_group_count": report["post_sprint_evaluable_group_count"],
        "post_sprint_gate_group_count": report["post_sprint_gate_group_count"],
        "group_count_basis": report["group_count_basis"],
        "t_plus_1_coverage_passed": report["t_plus_1_coverage_passed"],
        "t_plus_3_coverage_passed": report["t_plus_3_coverage_passed"],
        "evaluable_group_count_source": report["evaluable_group_count_source"],
        "stable_promotion_ready": False,
        "stable_evidence": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
    }
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday Lab Monitor Refresh Gate",
        "",
        "Read-only freshness gate for the registered Lab monitor candidate. It checks manual package/raw export freshness, existing rolling-origin outputs, and post-sprint anchor thresholds without rerunning validation, fitting models, changing thresholds, or producing Stable evidence.",
        "",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- candidate_id: `{report['candidate_id']}`",
        f"- current_monitor_status: `{report['current_monitor_status']}`",
        f"- manual_package_latest_date: {report['manual_package_latest_date']}",
        f"- last_rolling_origin_latest_date: {report['last_rolling_origin_latest_date']}",
        f"- post_sprint_anchor_count: {report['post_sprint_anchor_count']}",
        f"- post_sprint_raw_group_count: {report['post_sprint_raw_group_count']}",
        f"- post_sprint_evaluable_group_count: {report['post_sprint_evaluable_group_count']}",
        f"- post_sprint_gate_group_count: {report['post_sprint_gate_group_count']}",
        f"- group_count_basis: {report['group_count_basis']}",
        f"- refresh_due: {str(report['refresh_due']).lower()}",
        f"- next_allowed_task: `{report['next_allowed_task']}`",
        "- stable_promotion_ready: false",
        "- stable_evidence: false",
        "- qmt_ready: false",
        "- order_intent_ready: false",
    ]
    return "\n".join(lines) + "\n", docs_json


def run_refresh_gate(
    manual_inbox: Path = DEFAULT_MANUAL_INBOX,
    raw_export_dir: Path = DEFAULT_RAW_EXPORT_DIR,
    candidate_status_dir: Path = DEFAULT_CANDIDATE_STATUS_DIR,
    rolling_origin_dir: Path = DEFAULT_ROLLING_ORIGIN_DIR,
    attribution_dir: Path = DEFAULT_ATTRIBUTION_DIR,
    fixed_oop_dir: Path = DEFAULT_FIXED_OOP_DIR,
    reversal_dir: Path = DEFAULT_REVERSAL_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    enforce_output_dir: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_out_dir = resolve_output_dir(out_dir, repo_root, enforce=enforce_output_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    manual_dir = resolve_repo_path(manual_inbox, repo_root)
    raw_dir = resolve_repo_path(raw_export_dir, repo_root)
    status_dir = resolve_repo_path(candidate_status_dir, repo_root)
    rolling_dir = resolve_repo_path(rolling_origin_dir, repo_root)
    attrib_dir = resolve_repo_path(attribution_dir, repo_root)
    fixed_oop_resolved_dir = resolve_repo_path(fixed_oop_dir, repo_root)
    reversal_resolved_dir = resolve_repo_path(reversal_dir, repo_root)

    outputs = required_output_paths(status_dir, rolling_dir, attrib_dir)
    missing_outputs = [name for name, path in outputs.items() if not path.exists()]
    candidate_report = load_optional_json(outputs["candidate_status_report"])
    rolling_report = load_optional_json(outputs["rolling_origin_report"])
    status_decision, monitor_status, candidate_id = current_status(candidate_report)
    manual_summary = manual_inbox_summary(manual_dir)
    raw_summary = raw_export_summary(raw_dir)

    manual_paths = manual_package_paths(manual_dir)
    status_mtime = mtime(outputs["candidate_status_report"])
    raw_latest = latest_mtime(list_data_files(raw_dir))
    manual_latest = latest_mtime(list(manual_paths.values()))
    rolling_latest_mtime = latest_mtime([outputs["rolling_origin_report"], outputs["rolling_origin_decision"], outputs["rolling_origin_fold_manifest"]])
    attribution_latest_mtime = latest_mtime([outputs["attribution_report"], outputs["attribution_decision"]])

    last_rolling_latest_date = latest_rolling_date(rolling_report)
    last_rolling_evaluated_latest_date = latest_evaluated_rolling_date(rolling_report)
    manual_date_counts_after_rolling = count_manual_dates_after(manual_paths["csv"], last_rolling_latest_date)
    post_sprint_counts = post_sprint_counts_from_rolling(rolling_report)
    evaluable_summary = post_sprint_evaluable_group_summary(fixed_oop_resolved_dir, reversal_resolved_dir, post_sprint_counts)
    new_data_detected = bool(
        manual_summary["manual_package_latest_date"]
        and last_rolling_latest_date
        and manual_summary["manual_package_latest_date"] > last_rolling_latest_date
    )
    new_raw_export = bool(status_mtime is not None and raw_latest is not None and raw_latest > status_mtime)
    new_manual_package = bool(status_mtime is not None and manual_latest is not None and manual_latest > status_mtime)
    stale_outputs = bool(
        manual_latest is not None
        and rolling_latest_mtime is not None
        and rolling_latest_mtime < manual_latest
        or rolling_latest_mtime is not None
        and attribution_latest_mtime is not None
        and attribution_latest_mtime < rolling_latest_mtime
    )
    post_sprint_anchor_count = int(evaluable_summary.get("post_sprint_anchor_count") or post_sprint_counts["post_sprint_anchor_count"])
    post_sprint_gate_group_count = evaluable_summary.get("post_sprint_gate_group_count")
    post_sprint_etf_count = int(evaluable_summary.get("post_sprint_etf_count") or post_sprint_counts["post_sprint_etf_count"])
    new_anchor_due = (
        post_sprint_anchor_count >= POST_SPRINT_ANCHOR_THRESHOLD
        and post_sprint_gate_group_count is not None
        and int(post_sprint_gate_group_count) >= POST_SPRINT_GROUP_THRESHOLD
        and post_sprint_etf_count >= 5
        and bool(evaluable_summary["t_plus_1_coverage_passed"])
        and bool(evaluable_summary["t_plus_3_coverage_passed"])
    )

    artifact_before = check_model_artifacts(resolved_out_dir)
    p0_blockers = list(artifact_before["p0_blockers"])
    readiness_decision, refresh_reason, next_allowed_task = decide_refresh(
        manual_present=bool(manual_summary["manual_inbox_present"]),
        missing_outputs=missing_outputs,
        status_decision=status_decision,
        new_raw_export=new_raw_export,
        new_manual_package=new_manual_package,
        new_anchor_due=new_anchor_due,
        evaluable_group_count_available=bool(evaluable_summary["evaluable_group_count_available"]),
        stale_outputs=stale_outputs,
        p0_blockers=p0_blockers,
    )
    refresh_due = readiness_decision.startswith("LAB_MONITOR_REFRESH_DUE")

    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": "blocked" if readiness_decision.startswith("LAB_MONITOR_REFRESH_BLOCKED") else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "manual_inbox": str(manual_inbox),
            "raw_export_dir": str(raw_export_dir),
            "candidate_status_dir": str(candidate_status_dir),
            "rolling_origin_dir": str(rolling_origin_dir),
            "attribution_dir": str(attribution_dir),
            "fixed_oop_dir": str(fixed_oop_dir),
            "reversal_dir": str(reversal_dir),
            "stable_bundle": False,
        },
        "candidate_id": candidate_id or FOCUS_CANDIDATE_ID,
        "current_status_decision": status_decision,
        "current_monitor_status": monitor_status,
        "manual_inbox_present": manual_summary["manual_inbox_present"],
        "raw_export_present": raw_summary["raw_export_present"],
        "raw_export_file_count": raw_summary["raw_export_file_count"],
        "manual_package_latest_date": manual_summary["manual_package_latest_date"],
        "last_rolling_origin_latest_date": last_rolling_latest_date,
        "last_rolling_origin_evaluated_latest_date": last_rolling_evaluated_latest_date,
        "new_data_detected": new_data_detected,
        "new_raw_export_detected": new_raw_export,
        "new_manual_package_detected": new_manual_package,
        "new_anchor_count": manual_date_counts_after_rolling["new_anchor_count"],
        "new_group_count": manual_date_counts_after_rolling["new_group_count"],
        "post_sprint_anchor_count": post_sprint_anchor_count,
        "post_sprint_group_count": post_sprint_gate_group_count,
        "post_sprint_raw_group_count": evaluable_summary["post_sprint_raw_group_count"],
        "post_sprint_evaluable_group_count": evaluable_summary["post_sprint_evaluable_group_count"],
        "post_sprint_gate_group_count": post_sprint_gate_group_count,
        "post_sprint_etf_count": post_sprint_etf_count,
        "group_count_basis": evaluable_summary["group_count_basis"],
        "t_plus_1_coverage_passed": evaluable_summary["t_plus_1_coverage_passed"],
        "t_plus_3_coverage_passed": evaluable_summary["t_plus_3_coverage_passed"],
        "evaluable_group_count_available": evaluable_summary["evaluable_group_count_available"],
        "evaluable_group_count_source": evaluable_summary["evaluable_group_count_source"],
        "evaluable_group_count_cross_check": evaluable_summary.get("cross_check_sources", {}),
        "rerun_gate_passed": new_anchor_due,
        "post_sprint_fold_id": post_sprint_counts["post_sprint_fold_id"],
        "post_sprint_fold_skipped": post_sprint_counts["post_sprint_fold_skipped"],
        "post_sprint_skip_reasons": post_sprint_counts["post_sprint_skip_reasons"],
        "post_sprint_anchor_threshold": POST_SPRINT_ANCHOR_THRESHOLD,
        "post_sprint_group_threshold": POST_SPRINT_GROUP_THRESHOLD,
        "refresh_due": refresh_due,
        "refresh_reason": refresh_reason,
        "readiness_decision": readiness_decision,
        "next_allowed_task": next_allowed_task,
        "forbidden_next_tasks": FORBIDDEN_NEXT_TASKS,
        "missing_outputs": missing_outputs,
        "output_freshness": {
            "status_report_mtime_utc": mtime_iso(outputs["candidate_status_report"]),
            "manual_package_latest_mtime_utc": manual_summary["manual_package_latest_mtime_utc"],
            "raw_export_latest_mtime_utc": raw_summary["raw_export_latest_mtime_utc"],
            "rolling_origin_latest_mtime_utc": latest_mtime_iso(
                [outputs["rolling_origin_report"], outputs["rolling_origin_decision"], outputs["rolling_origin_fold_manifest"]]
            ),
            "attribution_latest_mtime_utc": latest_mtime_iso([outputs["attribution_report"], outputs["attribution_decision"]]),
            "stale_outputs_detected": stale_outputs,
        },
        "raw_export_files": raw_summary["raw_export_files"],
        "artifact_check_before": artifact_before,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings(readiness_decision),
        "access_mode": "READ_ONLY",
        "final_action_change_allowed": False,
        "contains_live_order": False,
        "contains_secret": False,
        "requires_human_review": True,
        "promotion_gate_required": True,
        "formal_training": False,
        "formal_training_ready": False,
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "stable_promotion_ready": False,
        "stable_evidence": False,
        "qmt_ready": False,
        "qmt_used": False,
        "order_intent_ready": False,
        "order_intent_generated": False,
        "automatic_promotion_ready": False,
        "stable_affected": False,
        "advisory_package_created": False,
        "not_trading_advice": True,
    }
    artifact_after = check_model_artifacts(resolved_out_dir)
    report["artifact_check_after"] = artifact_after
    if artifact_after["p0_blockers"]:
        report["p0_blockers"] = list(dict.fromkeys([*report["p0_blockers"], *artifact_after["p0_blockers"]]))
        report["readiness_decision"] = DECISION_BLOCKED_DATA
        report["refresh_due"] = False
        report["refresh_reason"] = "model/scaler/checkpoint artifact blocker detected"
        report["next_allowed_task"] = "manual_data_quality_review"
        report["status"] = "blocked"

    emit_outputs(repo_root, resolved_out_dir, report)
    return report


def p1_warnings(readiness_decision: str) -> list[str]:
    warnings = [
        "P1_READ_ONLY_REFRESH_GATE_NOT_STABLE_EVIDENCE",
        "P1_NO_STABLE_PROMOTION_WITHOUT_PROMOTION_GATE",
        "P1_NO_QMT_ORDERINTENT_OR_FORMAL_TRAINING",
    ]
    if readiness_decision == DECISION_NOT_DUE:
        warnings.append("P1_WAIT_FOR_NEW_DATA_OR_MANUAL_REVIEW")
    if readiness_decision.startswith("LAB_MONITOR_REFRESH_DUE"):
        warnings.append("P1_REFRESH_REQUIRES_SEPARATE_TASK")
    return warnings


def emit_outputs(repo_root: Path, out_dir: Path, report: dict[str, Any]) -> None:
    row = {column: report.get(column) for column in status_csv_columns()}
    row["forbidden_next_tasks"] = ";".join(FORBIDDEN_NEXT_TASKS)
    docs_md, docs_json = build_docs(report)
    write_json(out_dir / "lab_monitor_refresh_gate_report.json", report)
    write_json(
        out_dir / "lab_monitor_refresh_gate_decision.json",
        {
            "lab_declaration": LAB_DECLARATION,
            "readiness_decision": report["readiness_decision"],
            "refresh_due": report["refresh_due"],
            "refresh_reason": report["refresh_reason"],
            "next_allowed_task": report["next_allowed_task"],
            "forbidden_next_tasks": FORBIDDEN_NEXT_TASKS,
            "stable_promotion_ready": False,
            "stable_evidence": False,
            "formal_training_ready": False,
            "qmt_ready": False,
            "order_intent_ready": False,
            "automatic_promotion_ready": False,
            "p0_blockers": report["p0_blockers"],
            "p1_warnings": report["p1_warnings"],
        },
    )
    write_csv(out_dir / "lab_monitor_refresh_gate_status.csv", [row], status_csv_columns())
    (out_dir / "lab_monitor_refresh_gate_report.md").write_text(docs_md, encoding="utf-8")
    write_json(repo_root / "docs/research/aetfq3_intraday_lab_monitor_refresh_gate.json", docs_json)
    (repo_root / "docs/research/aetfq3_intraday_lab_monitor_refresh_gate.md").write_text(docs_md, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--manual-inbox", type=Path, default=DEFAULT_MANUAL_INBOX)
    parser.add_argument("--raw-export-dir", type=Path, default=DEFAULT_RAW_EXPORT_DIR)
    parser.add_argument("--candidate-status-dir", type=Path, default=DEFAULT_CANDIDATE_STATUS_DIR)
    parser.add_argument("--rolling-origin-dir", type=Path, default=DEFAULT_ROLLING_ORIGIN_DIR)
    parser.add_argument("--attribution-dir", type=Path, default=DEFAULT_ATTRIBUTION_DIR)
    parser.add_argument("--fixed-oop-dir", type=Path, default=DEFAULT_FIXED_OOP_DIR)
    parser.add_argument("--reversal-dir", type=Path, default=DEFAULT_REVERSAL_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_refresh_gate(
            args.manual_inbox,
            args.raw_export_dir,
            args.candidate_status_dir,
            args.rolling_origin_dir,
            args.attribution_dir,
            args.fixed_oop_dir,
            args.reversal_dir,
            args.out_dir,
        )
    except Exception as exc:  # noqa: BLE001 - CLI emits auditable Lab blocker.
        print(
            json.dumps(
                {
                    "lab_declaration": LAB_DECLARATION,
                    "status": "failed",
                    "readiness_decision": DECISION_BLOCKED_DATA,
                    "p0_blockers": [str(exc)],
                    "stable_promotion_ready": False,
                    "stable_evidence": False,
                    "formal_training_ready": False,
                    "qmt_ready": False,
                    "order_intent_ready": False,
                    "automatic_promotion_ready": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "lab_declaration": LAB_DECLARATION,
                "status": report["status"],
                "readiness_decision": report["readiness_decision"],
                "candidate_id": report["candidate_id"],
                "current_monitor_status": report["current_monitor_status"],
                "refresh_due": report["refresh_due"],
                "refresh_reason": report["refresh_reason"],
                "next_allowed_task": report["next_allowed_task"],
                "stable_promotion_ready": False,
                "stable_evidence": False,
                "formal_training_ready": False,
                "qmt_ready": False,
                "order_intent_ready": False,
                "automatic_promotion_ready": False,
                "p0_blockers": report["p0_blockers"],
                "p1_warnings": report["p1_warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
