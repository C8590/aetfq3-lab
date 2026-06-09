from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts, load_json


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_group_label_inconsistency_diagnostic")
REPORT_TYPE = "intraday_group_label_inconsistency_diagnostic"
TARGET_COLUMN = "three_day_positive_label"
GROUP_KEY = ["trade_date", "etf_code"]
GROUP_LABEL_POLICY = "anchor_close_last_bar"
ACCEPTED = "GROUP_LABEL_POLICY_ANCHOR_CLOSE_LAST_BAR_ACCEPTED_FOR_END_OF_DAY_DIAGNOSTIC"
REVIEW_THRESHOLD_FLIPS = "GROUP_LABEL_POLICY_REVIEW_REQUIRED_THRESHOLD_FLIPS_HIGH"
BLOCKED_DATA_QUALITY = "GROUP_LABEL_POLICY_BLOCKED_DATA_QUALITY_SUSPECT"
NOT_INTRADAY_LIVE_READY = "GROUP_LABEL_POLICY_NOT_INTRADAY_LIVE_READY"
BAR_LEVEL_OUTCOME_DENOMINATOR_VARIATION = "BAR_LEVEL_OUTCOME_DENOMINATOR_VARIATION"
THRESHOLD_NEAR_ZERO_LABEL_FLIP = "THRESHOLD_NEAR_ZERO_LABEL_FLIP"
DATA_QUALITY_SUSPECT = "DATA_QUALITY_SUSPECT"
GROUP_POLICY_EXPECTED_DIAGNOSTIC_BEHAVIOR = "GROUP_POLICY_EXPECTED_DIAGNOSTIC_BEHAVIOR"
NEAR_ZERO_ABS_THRESHOLD = 0.002
THRESHOLD_FLIP_REVIEW_RATE = 0.50
CLOSE_TOLERANCE = 1e-9


class GroupLabelDiagnosticError(RuntimeError):
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
        raise GroupLabelDiagnosticError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise GroupLabelDiagnosticError(f"CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise GroupLabelDiagnosticError(f"CSV has no header: {path}")
    return rows, columns


def run_diagnostic(
    bar_samples_path: Path,
    group_samples_path: Path,
    group_report_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_bar_samples = resolve_repo_path(bar_samples_path, repo_root)
    resolved_group_samples = resolve_repo_path(group_samples_path, repo_root)
    resolved_group_report = resolve_repo_path(group_report_path, repo_root)
    for path, label in (
        (resolved_bar_samples, "bar-samples"),
        (resolved_group_samples, "group-samples"),
        (resolved_group_report, "group-report"),
    ):
        if not path.exists():
            raise GroupLabelDiagnosticError(f"{label} path does not exist: {path}")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    bar_rows, bar_columns = load_csv_rows(resolved_bar_samples)
    group_rows, group_columns = load_csv_rows(resolved_group_samples)
    group_report = load_json(resolved_group_report)
    validate_columns(bar_columns, group_columns)

    grouped_bars = group_bar_rows(bar_rows)
    grouped_samples = index_group_rows(group_rows)
    group_diagnostics = [
        diagnose_group(key, rows, grouped_samples.get(key)) for key, rows in sorted(grouped_bars.items())
    ]
    missing_bar_groups = sorted(key for key in grouped_samples if key not in grouped_bars)
    summary = summarize_diagnostics(group_diagnostics, missing_bar_groups)
    drivers = determine_drivers(summary)
    last_bar_policy_check = build_last_bar_policy_check(group_diagnostics, missing_bar_groups)
    policy_review_decision = decide_policy(summary, drivers, last_bar_policy_check)
    artifact_check = check_model_artifacts(resolved_out_dir)

    p0_blockers: list[str] = []
    if summary["data_quality_suspect_group_count"] > 0 or missing_bar_groups:
        p0_blockers.append("data quality suspect groups found")
    p0_blockers.extend(artifact_check["p0_blockers"])

    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": REPORT_TYPE,
        "status": "blocked" if policy_review_decision == BLOCKED_DATA_QUALITY else "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bar_samples_path": str(bar_samples_path),
        "group_samples_path": str(group_samples_path),
        "group_report_path": str(group_report_path),
        "source_group_report_inconsistent_label_group_count": group_report.get("group_statistics", {}).get(
            "inconsistent_label_group_count"
        ),
        "group_key": GROUP_KEY,
        "total_group_count": summary["total_group_count"],
        "consistent_group_count": summary["consistent_group_count"],
        "inconsistent_group_count": summary["inconsistent_group_count"],
        "inconsistent_group_rate": summary["inconsistent_group_rate"],
        "inconsistent_by_anchor_date": summary["inconsistent_by_anchor_date"],
        "inconsistent_by_etf": summary["inconsistent_by_etf"],
        "first_last_label_mismatch_group_count": summary["first_last_label_mismatch_group_count"],
        "near_zero_flip_group_count": summary["near_zero_flip_group_count"],
        "data_quality_suspect_group_count": summary["data_quality_suspect_group_count"],
        "inconsistency_drivers": drivers,
        "last_bar_policy_check": last_bar_policy_check,
        "policy_review_decision": policy_review_decision,
        "intraday_live_decision_ready": False,
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "not_trading_advice": True,
        "model_training_performed": False,
        "no_save_smoke_run": False,
        "hyperparameter_tuning": False,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "artifact_check": artifact_check,
        "p0_blockers": p0_blockers,
        "p1_warnings": ["P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED"] if summary["inconsistent_group_count"] else [],
        "group_diagnostics": group_diagnostics,
    }
    write_reports(report, resolved_out_dir)
    return report


def validate_columns(bar_columns: list[str], group_columns: list[str]) -> None:
    missing_bar = [column for column in [*GROUP_KEY, "datetime", "close", "future_return_3d", TARGET_COLUMN] if column not in bar_columns]
    missing_group = [column for column in [*GROUP_KEY, "bar_count", "close_last", TARGET_COLUMN] if column not in group_columns]
    if missing_bar:
        raise GroupLabelDiagnosticError("bar samples missing columns: " + ", ".join(missing_bar))
    if missing_group:
        raise GroupLabelDiagnosticError("group samples missing columns: " + ", ".join(missing_group))


def group_bar_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (str(row.get("trade_date", "")).strip(), str(row.get("etf_code", "")).strip())
        groups[key].append(row)
    return dict(groups)


def index_group_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    indexed: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        key = (str(row.get("trade_date", "")).strip(), str(row.get("etf_code", "")).strip())
        indexed[key] = row
    return indexed


def diagnose_group(
    key: tuple[str, str],
    rows: list[dict[str, str]],
    group_row: dict[str, str] | None,
) -> dict[str, Any]:
    sorted_rows = sorted(rows, key=sort_key_for_bar)
    first_bar = sorted_rows[0]
    last_bar = sorted_rows[-1]
    labels = [normalize_label(row.get(TARGET_COLUMN, "")) for row in sorted_rows]
    label_0_count = sum(1 for label in labels if label == 0)
    label_1_count = sum(1 for label in labels if label == 1)
    label_unique_count = len({label for label in labels if label is not None})
    label_switch_count = count_label_switches(labels)
    future_returns = finite_values(sorted_rows, "future_return_3d")
    closes = finite_values(sorted_rows, "close")
    group_label = normalize_label(group_row.get(TARGET_COLUMN, "")) if group_row else None
    last_bar_label = normalize_label(last_bar.get(TARGET_COLUMN, ""))
    first_bar_label = normalize_label(first_bar.get(TARGET_COLUMN, ""))
    close_last = to_float(last_bar.get("close"))
    group_close_last = to_float(group_row.get("close_last")) if group_row else math.nan
    bar_count_matches = group_row is not None and len(sorted_rows) == int(to_float(group_row.get("bar_count")))
    close_last_matches = group_row is not None and numeric_equal(close_last, group_close_last)
    last_bar_label_matches_group = group_row is not None and last_bar_label == group_label
    near_zero_flip = (
        label_unique_count > 1
        and future_returns
        and min(future_returns) <= 0.0 <= max(future_returns)
        and min(abs(value) for value in future_returns) <= NEAR_ZERO_ABS_THRESHOLD
    )
    data_quality_suspect = not (group_row is not None and bar_count_matches and close_last_matches and last_bar_label_matches_group)
    return {
        "trade_date": key[0],
        "etf_code": key[1],
        "bar_count": len(sorted_rows),
        "first_bar_datetime": first_bar.get("datetime", ""),
        "last_bar_datetime": last_bar.get("datetime", ""),
        "first_bar_label": first_bar_label,
        "last_bar_label": last_bar_label,
        "group_level_label": group_label,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "label_switch_count": label_switch_count,
        "label_unique_count": label_unique_count,
        "future_return_3d_min": min(future_returns) if future_returns else None,
        "future_return_3d_max": max(future_returns) if future_returns else None,
        "close_min": min(closes) if closes else None,
        "close_max": max(closes) if closes else None,
        "close_last": close_last if math.isfinite(close_last) else None,
        "group_close_last": group_close_last if math.isfinite(group_close_last) else None,
        "bar_count_matches_group": bar_count_matches,
        "close_last_matches_group": close_last_matches,
        "last_bar_label_matches_group_level_label": last_bar_label_matches_group,
        "first_bar_label_differs_from_last_bar_label": first_bar_label != last_bar_label,
        "inconsistent_group": label_unique_count > 1,
        "near_zero_label_flip": near_zero_flip,
        "data_quality_suspect": data_quality_suspect,
    }


def summarize_diagnostics(
    group_diagnostics: list[dict[str, Any]],
    missing_bar_groups: list[tuple[str, str]],
) -> dict[str, Any]:
    total = len(group_diagnostics)
    inconsistent = [item for item in group_diagnostics if item["inconsistent_group"]]
    consistent = [item for item in group_diagnostics if not item["inconsistent_group"]]
    first_last_mismatches = [item for item in group_diagnostics if item["first_bar_label_differs_from_last_bar_label"]]
    near_zero = [item for item in group_diagnostics if item["near_zero_label_flip"]]
    data_quality = [item for item in group_diagnostics if item["data_quality_suspect"]]
    return {
        "total_group_count": total,
        "consistent_group_count": len(consistent),
        "inconsistent_group_count": len(inconsistent),
        "inconsistent_group_rate": len(inconsistent) / total if total else 0.0,
        "inconsistent_by_anchor_date": dict(sorted(Counter(item["trade_date"] for item in inconsistent).items())),
        "inconsistent_by_etf": dict(sorted(Counter(item["etf_code"] for item in inconsistent).items())),
        "first_last_label_mismatch_group_count": len(first_last_mismatches),
        "first_last_label_mismatch_examples": first_last_mismatches[:10],
        "near_zero_flip_group_count": len(near_zero),
        "near_zero_flip_group_rate": len(near_zero) / len(inconsistent) if inconsistent else 0.0,
        "near_zero_flip_examples": near_zero[:10],
        "data_quality_suspect_group_count": len(data_quality) + len(missing_bar_groups),
        "data_quality_suspect_examples": data_quality[:10],
        "missing_bar_groups": [{"trade_date": key[0], "etf_code": key[1]} for key in missing_bar_groups],
    }


def determine_drivers(summary: dict[str, Any]) -> list[str]:
    drivers: list[str] = []
    if summary["inconsistent_group_count"] > 0:
        drivers.extend([BAR_LEVEL_OUTCOME_DENOMINATOR_VARIATION, GROUP_POLICY_EXPECTED_DIAGNOSTIC_BEHAVIOR])
    if summary["near_zero_flip_group_count"] > 0:
        drivers.append(THRESHOLD_NEAR_ZERO_LABEL_FLIP)
    if summary["data_quality_suspect_group_count"] > 0:
        drivers.append(DATA_QUALITY_SUSPECT)
    return drivers


def build_last_bar_policy_check(
    group_diagnostics: list[dict[str, Any]],
    missing_bar_groups: list[tuple[str, str]],
) -> dict[str, Any]:
    matched = [item for item in group_diagnostics if item["last_bar_label_matches_group_level_label"]]
    mismatched = [item for item in group_diagnostics if not item["last_bar_label_matches_group_level_label"]]
    return {
        "group_label_policy": GROUP_LABEL_POLICY,
        "checked_group_count": len(group_diagnostics),
        "last_bar_label_matches_group_level_label_count": len(matched),
        "last_bar_label_mismatch_count": len(mismatched),
        "missing_bar_group_count": len(missing_bar_groups),
        "all_last_bar_labels_match_group_level_labels": not mismatched and not missing_bar_groups,
        "accepted_for_end_of_day_diagnostic": not mismatched and not missing_bar_groups,
        "intraday_live_decision_ready": False,
        "not_intraday_live_decision_policy": True,
        "mismatch_examples": mismatched[:10],
    }


def decide_policy(
    summary: dict[str, Any],
    drivers: Sequence[str],
    last_bar_policy_check: dict[str, Any],
) -> str:
    if DATA_QUALITY_SUSPECT in drivers:
        return BLOCKED_DATA_QUALITY
    if last_bar_policy_check["all_last_bar_labels_match_group_level_labels"]:
        return ACCEPTED
    if summary["near_zero_flip_group_rate"] > THRESHOLD_FLIP_REVIEW_RATE:
        return REVIEW_THRESHOLD_FLIPS
    return NOT_INTRADAY_LIVE_READY


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "intraday_group_label_inconsistency_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "policy_review_decision": report["policy_review_decision"],
        "status": report["status"],
        "group_label_policy": GROUP_LABEL_POLICY,
        "intraday_live_decision_ready": False,
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "not_trading_advice": True,
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }
    (out_dir / "policy_review_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "",
        "# Intraday Group Label Inconsistency Diagnostic",
        "",
        "本文件只用于 Lab label inconsistency diagnostic，不训练模型，不运行 no-save smoke，不调参，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。",
        "",
        f"- report_type: {report['report_type']}",
        f"- status: {report['status']}",
        f"- group_key: {json.dumps(report['group_key'], ensure_ascii=False)}",
        f"- total_group_count: {report['total_group_count']}",
        f"- consistent_group_count: {report['consistent_group_count']}",
        f"- inconsistent_group_count: {report['inconsistent_group_count']}",
        f"- inconsistent_group_rate: {report['inconsistent_group_rate']}",
        f"- inconsistency_drivers: {json.dumps(report['inconsistency_drivers'], ensure_ascii=False)}",
        f"- policy_review_decision: {report['policy_review_decision']}",
        f"- intraday_live_decision_ready: {str(report['intraday_live_decision_ready']).lower()}",
        f"- stable_promotion_ready: {str(report['stable_promotion_ready']).lower()}",
        f"- formal_training_ready: {str(report['formal_training_ready']).lower()}",
        f"- qmt_ready: {str(report['qmt_ready']).lower()}",
        f"- order_intent_ready: {str(report['order_intent_ready']).lower()}",
        f"- metrics_are_effectiveness_evidence: {str(report['metrics_are_effectiveness_evidence']).lower()}",
        f"- not_trading_advice: {str(report['not_trading_advice']).lower()}",
    ]
    (out_dir / "intraday_group_label_inconsistency_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def sort_key_for_bar(row: dict[str, str]) -> tuple[str, float]:
    return (str(row.get("datetime", "")), to_float(row.get("bar_index")))


def count_label_switches(labels: Sequence[int | None]) -> int:
    clean = [label for label in labels if label is not None]
    return sum(1 for left, right in zip(clean, clean[1:]) if left != right)


def finite_values(rows: Sequence[dict[str, str]], column: str) -> list[float]:
    values = [to_float(row.get(column)) for row in rows]
    return [value for value in values if math.isfinite(value)]


def to_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def normalize_label(value: Any) -> int | None:
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    return None


def numeric_equal(left: float, right: float) -> bool:
    return math.isfinite(left) and math.isfinite(right) and abs(left - right) <= CLOSE_TOLERANCE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only group label inconsistency diagnostic.")
    parser.add_argument("--bar-samples", required=True, type=Path)
    parser.add_argument("--group-samples", required=True, type=Path)
    parser.add_argument("--group-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_diagnostic(args.bar_samples, args.group_samples, args.group_report, args.out_dir)
    except GroupLabelDiagnosticError as exc:
        print(
            json.dumps(
                {"status": "failed", "policy_review_decision": BLOCKED_DATA_QUALITY, "p0_blockers": [str(exc)]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "policy_review_decision": report["policy_review_decision"],
                "total_group_count": report["total_group_count"],
                "inconsistent_group_count": report["inconsistent_group_count"],
                "inconsistent_group_rate": report["inconsistent_group_rate"],
                "inconsistency_drivers": report["inconsistency_drivers"],
                "intraday_live_decision_ready": False,
                "stable_promotion_ready": False,
                "formal_training_ready": False,
                "qmt_ready": False,
                "order_intent_ready": False,
                "metrics_are_effectiveness_evidence": False,
                "not_trading_advice": True,
                "p0_blockers": report["p0_blockers"],
                "p1_warnings": report["p1_warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
