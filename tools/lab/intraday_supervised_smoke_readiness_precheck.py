from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_supervised_smoke_readiness_precheck")
LABEL_COLUMN = "three_day_positive_label"
ANCHOR_COLUMN_CANDIDATES = ("anchor_date", "trade_date")
FORBIDDEN_OUTCOME_FEATURES = {"future_return_1d", "future_return_3d", "max_drawdown_3d"}
BOUNDARY_FALSE_FIELDS = [
    "training_allowed",
    "supervised_training_allowed",
    "stable_effect_allowed",
    "contains_order_intent",
    "contains_live_order",
    "contains_secret",
]
READY = "SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED"
BLOCKED_SINGLE_CLASS_LABEL = "BLOCKED_SINGLE_CLASS_LABEL"
BLOCKED_INSUFFICIENT_ROWS = "BLOCKED_INSUFFICIENT_ROWS"
BLOCKED_INSUFFICIENT_ANCHORS = "BLOCKED_INSUFFICIENT_ANCHORS"
BLOCKED_SPLIT_NOT_CLASS_DIVERSE = "BLOCKED_SPLIT_NOT_CLASS_DIVERSE"
BLOCKED_MANIFEST_LEAKAGE_P0 = "BLOCKED_MANIFEST_LEAKAGE_P0"
BLOCKED_BOUNDARY_FLAG = "BLOCKED_BOUNDARY_FLAG"
MIN_ROWS = 500
MIN_ANCHORS = 5
MIN_ETFS = 2
MIN_CLASS_COUNT = 50
SPLIT_POLICIES = [
    ("anchor_date_70_30", 0.7),
    ("anchor_date_60_40", 0.6),
]


class ReadinessPrecheckError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ReadinessPrecheckError(f"JSON parse failed for {path}: {exc}") from exc
    except OSError as exc:
        raise ReadinessPrecheckError(f"JSON cannot be read: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReadinessPrecheckError(f"JSON root must be object: {path}")
    return payload


def resolve_repo_path(raw_path: str | None, repo_root: Path = REPO_ROOT) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def resolve_output_dir(out_dir: Path, repo_root: Path = REPO_ROOT) -> Path:
    repo_root = repo_root.resolve()
    resolved = (repo_root / out_dir if not out_dir.is_absolute() else out_dir).resolve()
    allowed = (repo_root / ALLOWED_OUTPUT_DIR).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ReadinessPrecheckError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise ReadinessPrecheckError(f"samples CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise ReadinessPrecheckError(f"samples CSV has no header: {path}")
    return rows, columns


def run_precheck(samples_path: Path, manifest_path: Path, out_dir: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(str(samples_path), repo_root)
    resolved_manifest = resolve_repo_path(str(manifest_path), repo_root)
    if resolved_samples is None or not resolved_samples.exists():
        raise ReadinessPrecheckError("samples path missing or does not exist")
    if resolved_manifest is None or not resolved_manifest.exists():
        raise ReadinessPrecheckError("manifest path missing or does not exist")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)

    manifest = load_json(resolved_manifest)
    rows, columns = load_csv_rows(resolved_samples)
    manifest_check = check_label_manifest(resolved_manifest)
    feature_columns = string_list(manifest.get("feature_columns"))
    boundary_check = run_boundary_check(manifest)
    sample_check = run_sample_check(rows, columns, feature_columns)
    split_check = run_split_check(rows, sample_check["anchor_column"])

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    p0_blockers.extend(boundary_check["p0_blockers"])
    p0_blockers.extend(sample_check["p0_blockers"])
    p1_warnings.extend(sample_check["p1_warnings"])
    p0_blockers.extend(split_check["p0_blockers"])

    readiness_decision = decide_readiness(
        manifest_check_ok=manifest_check.ok,
        boundary_ok=boundary_check["passed"],
        sample_check=sample_check,
        split_check=split_check,
    )
    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": "intraday_supervised_smoke_readiness_precheck",
        "status": "passed" if readiness_decision == READY else "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "readiness_decision": readiness_decision,
        "manifest_leakage_check": manifest_check.to_summary(),
        "boundary_check": boundary_check,
        "sample_check": sample_check,
        "split_check": split_check,
        "selected_split_policy": split_check["selected_split_policy"],
        "train_anchor_dates": split_check["train_anchor_dates"],
        "valid_anchor_dates": split_check["valid_anchor_dates"],
        "train_row_count": split_check["train_row_count"],
        "valid_row_count": split_check["valid_row_count"],
        "train_label_0_count": split_check["train_label_0_count"],
        "train_label_1_count": split_check["train_label_1_count"],
        "valid_label_0_count": split_check["valid_label_0_count"],
        "valid_label_1_count": split_check["valid_label_1_count"],
        "split_feasible": split_check["split_feasible"],
        "row_count": sample_check["row_count"],
        "anchor_count": sample_check["anchor_count"],
        "etf_count": sample_check["etf_count"],
        "label_null_count": sample_check["label_null_count"],
        "label_0_count": sample_check["label_0_count"],
        "label_1_count": sample_check["label_1_count"],
        "min_class_count": sample_check["min_class_count"],
        "positive_rate": sample_check["positive_rate"],
        "training_allowed": False,
        "supervised_training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "model_training_performed": False,
        "supervised_smoke_run": False,
        "gpu_used": False,
        "torchrun_used": False,
        "checkpoint_saved": False,
        "model_saved": False,
        "not_trading_advice": True,
        "not_stable_evidence": True,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }
    write_reports(report, resolved_out_dir)
    return report


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


def run_sample_check(rows: list[dict[str, str]], columns: list[str], feature_columns: list[str]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    feature_set = set(feature_columns)
    row_count = len(rows)
    anchor_column = next((column for column in ANCHOR_COLUMN_CANDIDATES if column in columns), None)

    if LABEL_COLUMN not in columns:
        p0_blockers.append(f"{LABEL_COLUMN} missing from samples")
    if LABEL_COLUMN in feature_set:
        p0_blockers.append(f"{LABEL_COLUMN} must not be in feature_columns")
    forbidden_present = sorted(FORBIDDEN_OUTCOME_FEATURES & feature_set)
    if forbidden_present:
        p0_blockers.append("outcome columns must not be in feature_columns: " + ", ".join(forbidden_present))
    if anchor_column is None:
        p0_blockers.append("samples must contain anchor_date or trade_date for time-based split")

    label_values = [normalize_label(row.get(LABEL_COLUMN, "")) for row in rows]
    label_null_count = sum(1 for value in label_values if value is None)
    label_0_count = sum(1 for value in label_values if value == 0)
    label_1_count = sum(1 for value in label_values if value == 1)
    class_counts = {0: label_0_count, 1: label_1_count}
    present_classes = [label for label, count in class_counts.items() if count > 0]
    min_class_count = min((count for count in class_counts.values() if count > 0), default=0)
    positive_rate = label_1_count / (label_0_count + label_1_count) if (label_0_count + label_1_count) else None

    if label_null_count:
        p1_warnings.append(f"{LABEL_COLUMN} null count is {label_null_count}")
    if len(present_classes) < 2:
        p0_blockers.append("label must contain both class 0 and class 1")
    if row_count < MIN_ROWS:
        p0_blockers.append(f"row_count must be >= {MIN_ROWS}")
    if min_class_count < MIN_CLASS_COUNT:
        p0_blockers.append(f"each class must have at least {MIN_CLASS_COUNT} samples")

    anchors = sorted({str(row.get(anchor_column, "")).strip() for row in rows if anchor_column and str(row.get(anchor_column, "")).strip()})
    etfs = sorted({str(row.get("etf_code", "")).strip() for row in rows if str(row.get("etf_code", "")).strip()})
    if len(anchors) < MIN_ANCHORS:
        p0_blockers.append(f"anchor_count must be >= {MIN_ANCHORS}")
    if len(etfs) < MIN_ETFS:
        p0_blockers.append(f"etf_count must be >= {MIN_ETFS}")

    return {
        "passed": not p0_blockers,
        "columns": columns,
        "feature_columns": feature_columns,
        "anchor_column": anchor_column,
        "row_count": row_count,
        "anchor_count": len(anchors),
        "anchor_dates": anchors,
        "etf_count": len(etfs),
        "etf_codes": etfs,
        "label_null_count": label_null_count,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "class_count": len(present_classes),
        "min_class_count": min_class_count,
        "positive_rate": positive_rate,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }


def run_split_check(rows: list[dict[str, str]], anchor_column: str | None) -> dict[str, Any]:
    if not anchor_column:
        return empty_split_check(["samples must contain anchor_date or trade_date for time-based split"])
    anchor_dates = sorted({str(row.get(anchor_column, "")).strip() for row in rows if str(row.get(anchor_column, "")).strip()})
    if len(anchor_dates) < 2:
        return empty_split_check(["at least two anchor dates are required for split feasibility"])

    attempts: list[dict[str, Any]] = []
    for policy_name, train_ratio in SPLIT_POLICIES:
        train_anchor_count = max(1, min(len(anchor_dates) - 1, math.floor(len(anchor_dates) * train_ratio)))
        train_anchor_dates = anchor_dates[:train_anchor_count]
        valid_anchor_dates = anchor_dates[train_anchor_count:]
        result = summarize_split(rows, anchor_column, train_anchor_dates, valid_anchor_dates, policy_name)
        attempts.append(result)
        if result["split_feasible"]:
            return {**result, "attempts": attempts, "p0_blockers": []}

    selected = attempts[-1] if attempts else empty_split_check([])
    return {
        **selected,
        "attempts": attempts,
        "split_feasible": False,
        "p0_blockers": ["time-based split train/valid must both contain class 0 and class 1"],
    }


def summarize_split(
    rows: list[dict[str, str]],
    anchor_column: str,
    train_anchor_dates: list[str],
    valid_anchor_dates: list[str],
    policy_name: str,
) -> dict[str, Any]:
    train_set = set(train_anchor_dates)
    valid_set = set(valid_anchor_dates)
    train_rows = [row for row in rows if str(row.get(anchor_column, "")).strip() in train_set]
    valid_rows = [row for row in rows if str(row.get(anchor_column, "")).strip() in valid_set]
    train_counts = label_counts(train_rows)
    valid_counts = label_counts(valid_rows)
    split_feasible = train_counts[0] > 0 and train_counts[1] > 0 and valid_counts[0] > 0 and valid_counts[1] > 0
    return {
        "selected_split_policy": policy_name,
        "train_anchor_dates": train_anchor_dates,
        "valid_anchor_dates": valid_anchor_dates,
        "train_row_count": len(train_rows),
        "valid_row_count": len(valid_rows),
        "train_label_0_count": train_counts[0],
        "train_label_1_count": train_counts[1],
        "valid_label_0_count": valid_counts[0],
        "valid_label_1_count": valid_counts[1],
        "split_feasible": split_feasible,
    }


def empty_split_check(p0_blockers: list[str]) -> dict[str, Any]:
    return {
        "selected_split_policy": None,
        "train_anchor_dates": [],
        "valid_anchor_dates": [],
        "train_row_count": 0,
        "valid_row_count": 0,
        "train_label_0_count": 0,
        "train_label_1_count": 0,
        "valid_label_0_count": 0,
        "valid_label_1_count": 0,
        "split_feasible": False,
        "attempts": [],
        "p0_blockers": p0_blockers,
    }


def label_counts(rows: list[dict[str, str]]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for row in rows:
        value = normalize_label(row.get(LABEL_COLUMN, ""))
        if value in counts:
            counts[value] += 1
    return counts


def normalize_label(value: Any) -> int | None:
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    return None


def decide_readiness(
    manifest_check_ok: bool,
    boundary_ok: bool,
    sample_check: dict[str, Any],
    split_check: dict[str, Any],
) -> str:
    if not boundary_ok:
        return BLOCKED_BOUNDARY_FLAG
    if not manifest_check_ok:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    if sample_check["class_count"] < 2:
        return BLOCKED_SINGLE_CLASS_LABEL
    if sample_check["row_count"] < MIN_ROWS or sample_check["etf_count"] < MIN_ETFS or sample_check["min_class_count"] < MIN_CLASS_COUNT:
        return BLOCKED_INSUFFICIENT_ROWS
    if sample_check["anchor_count"] < MIN_ANCHORS:
        return BLOCKED_INSUFFICIENT_ANCHORS
    if not split_check["split_feasible"]:
        return BLOCKED_SPLIT_NOT_CLASS_DIVERSE
    if not sample_check["passed"]:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    return READY


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intraday_supervised_smoke_readiness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "readiness_decision": report["readiness_decision"],
        "status": report["status"],
        "selected_split_policy": report["selected_split_policy"],
        "train_anchor_dates": report["train_anchor_dates"],
        "valid_anchor_dates": report["valid_anchor_dates"],
        "train_row_count": report["train_row_count"],
        "valid_row_count": report["valid_row_count"],
        "train_label_0_count": report["train_label_0_count"],
        "train_label_1_count": report["train_label_1_count"],
        "valid_label_0_count": report["valid_label_0_count"],
        "valid_label_1_count": report["valid_label_1_count"],
        "split_feasible": report["split_feasible"],
        "training_allowed": False,
        "stable_allowed": False,
        "order_intent_allowed": False,
        "qmt_allowed": False,
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }
    (out_dir / "readiness_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab supervised smoke readiness precheck，不训练模型，不接 QMT，不生成 OrderIntent，不进入 Stable。",
        "",
        "# Intraday Supervised Smoke Readiness Precheck",
        "",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- row_count: {report['row_count']}",
        f"- anchor_count: {report['anchor_count']}",
        f"- etf_count: {report['etf_count']}",
        f"- label_null_count: {report['label_null_count']}",
        f"- label_0_count: {report['label_0_count']}",
        f"- label_1_count: {report['label_1_count']}",
        f"- selected_split_policy: {report['selected_split_policy']}",
        f"- train_anchor_dates: {', '.join(report['train_anchor_dates'])}",
        f"- valid_anchor_dates: {', '.join(report['valid_anchor_dates'])}",
        f"- train_row_count: {report['train_row_count']}",
        f"- valid_row_count: {report['valid_row_count']}",
        f"- train_label_0_count: {report['train_label_0_count']}",
        f"- train_label_1_count: {report['train_label_1_count']}",
        f"- valid_label_0_count: {report['valid_label_0_count']}",
        f"- valid_label_1_count: {report['valid_label_1_count']}",
        f"- split_feasible: {str(report['split_feasible']).lower()}",
        "- boundary: no training, no supervised smoke run, no GPU, no torchrun, no checkpoint, no QMT, no OrderIntent, no Stable, no output/, no lab_advisory, not trading advice.",
    ]
    (out_dir / "intraday_supervised_smoke_readiness_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday supervised smoke readiness precheck.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_precheck(args.samples, args.manifest, args.out_dir)
    except ReadinessPrecheckError as exc:
        print(json.dumps({"status": "failed", "readiness_decision": BLOCKED_MANIFEST_LEAKAGE_P0, "p0_blockers": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": report["status"],
        "readiness_decision": report["readiness_decision"],
        "selected_split_policy": report["selected_split_policy"],
        "split_feasible": report["split_feasible"],
        "training_allowed": False,
        "stable_allowed": False,
        "order_intent_allowed": False,
        "qmt_allowed": False,
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
