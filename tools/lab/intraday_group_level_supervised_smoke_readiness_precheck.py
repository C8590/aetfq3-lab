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

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts, load_json


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_supervised_smoke_readiness_precheck")
REPORT_TYPE = "intraday_group_level_supervised_smoke_readiness_precheck"
TARGET_COLUMN = "three_day_positive_label"
GROUP_LABEL_POLICY = "anchor_close_last_bar"
READY = "GROUP_LEVEL_SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED"
READY_WITH_INCONSISTENCY = "GROUP_LEVEL_SUPERVISED_SMOKE_READINESS_PASSED_WITH_LABEL_INCONSISTENCY_REVIEW_REQUIRED"
BLOCKED_SINGLE_CLASS = "BLOCKED_GROUP_LEVEL_SINGLE_CLASS_LABEL"
BLOCKED_INSUFFICIENT_GROUPS = "BLOCKED_GROUP_LEVEL_INSUFFICIENT_GROUPS"
BLOCKED_SPLIT_NOT_CLASS_DIVERSE = "BLOCKED_GROUP_LEVEL_SPLIT_NOT_CLASS_DIVERSE"
BLOCKED_MANIFEST_LEAKAGE_P0 = "BLOCKED_MANIFEST_LEAKAGE_P0"
BLOCKED_BOUNDARY_FLAG = "BLOCKED_BOUNDARY_FLAG"
P1_INCONSISTENCY = "P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED"
MIN_GROUPS = 200
MIN_ANCHORS = 20
MIN_ETFS = 3
MIN_CLASS_COUNT = 50
INCONSISTENCY_RATE_P1_THRESHOLD = 0.10
SPLIT_POLICIES = [("anchor_date_70_30", 0.7), ("anchor_date_60_40", 0.6)]
BOUNDARY_FALSE_FIELDS = [
    "training_allowed",
    "supervised_training_allowed",
    "stable_effect_allowed",
    "contains_order_intent",
    "contains_live_order",
    "contains_secret",
]
FORBIDDEN_OUTCOME_FEATURES = {
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
}
EXTRA_FORBIDDEN_FEATURES = {
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
    TARGET_COLUMN,
}


class GroupLevelReadinessError(RuntimeError):
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
        raise GroupLevelReadinessError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise GroupLevelReadinessError(f"samples CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise GroupLevelReadinessError(f"samples CSV has no header: {path}")
    return rows, columns


def run_precheck(
    samples_path: Path,
    manifest_path: Path,
    group_report_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(samples_path, repo_root)
    resolved_manifest = resolve_repo_path(manifest_path, repo_root)
    resolved_group_report = resolve_repo_path(group_report_path, repo_root)
    for required_path, label in (
        (resolved_samples, "samples"),
        (resolved_manifest, "manifest"),
        (resolved_group_report, "group-report"),
    ):
        if not required_path.exists():
            raise GroupLevelReadinessError(f"{label} path does not exist: {required_path}")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(resolved_manifest)
    group_report = load_json(resolved_group_report)
    rows, columns = load_csv_rows(resolved_samples)
    manifest_check = check_label_manifest(resolved_manifest)
    group_contract_check = run_group_contract_check(manifest)
    boundary_check = run_boundary_check(manifest)
    feature_check = run_feature_check(manifest, columns)
    sample_check = run_sample_check(rows, columns)
    split_check = run_split_check(rows, sample_check["anchor_column"])
    inconsistent_label_review = run_inconsistent_label_review(group_report, sample_check["group_count"])
    artifact_check = check_model_artifacts(resolved_out_dir)

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    p0_blockers.extend(group_contract_check["p0_blockers"])
    p0_blockers.extend(boundary_check["p0_blockers"])
    p0_blockers.extend(feature_check["p0_blockers"])
    p0_blockers.extend(sample_check["p0_blockers"])
    p0_blockers.extend(split_check["p0_blockers"])
    p1_warnings.extend(inconsistent_label_review["p1_warnings"])
    p0_blockers.extend(artifact_check["p0_blockers"])

    readiness_decision = decide_readiness(
        manifest_ok=manifest_check.ok,
        group_contract_ok=group_contract_check["passed"],
        boundary_ok=boundary_check["passed"],
        feature_ok=feature_check["passed"],
        sample_check=sample_check,
        split_check=split_check,
        artifact_ok=artifact_check["passed"],
        p1_warnings=p1_warnings,
    )
    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": REPORT_TYPE,
        "status": "blocked" if readiness_decision.startswith("BLOCKED_") else "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "group_report_path": str(group_report_path),
        "readiness_decision": readiness_decision,
        "target": TARGET_COLUMN,
        "group_level_sample": manifest.get("group_level_sample"),
        "group_key": manifest.get("group_key"),
        "group_label_policy": manifest.get("group_label_policy"),
        "intraday_live_decision_ready": manifest.get("intraday_live_decision_ready"),
        "feature_columns": string_list(manifest.get("feature_columns")),
        "manifest_leakage_check": manifest_check.to_summary(),
        "group_contract_check": group_contract_check,
        "boundary_check": boundary_check,
        "feature_check": feature_check,
        "sample_check": sample_check,
        "inconsistent_label_review": inconsistent_label_review,
        "selected_split_policy": split_check["selected_split_policy"],
        "train_anchor_dates": split_check["train_anchor_dates"],
        "valid_anchor_dates": split_check["valid_anchor_dates"],
        "train_group_count": split_check["train_group_count"],
        "valid_group_count": split_check["valid_group_count"],
        "train_label_0_count": split_check["train_label_0_count"],
        "train_label_1_count": split_check["train_label_1_count"],
        "valid_label_0_count": split_check["valid_label_0_count"],
        "valid_label_1_count": split_check["valid_label_1_count"],
        "split_feasible": split_check["split_feasible"],
        "split_check": split_check,
        "artifact_check": artifact_check,
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


def run_group_contract_check(manifest: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    group_key = string_list(manifest.get("group_key"))
    if manifest.get("group_level_sample") is not True:
        p0_blockers.append("group_level_sample must be true")
    if not (group_key == ["trade_date", "etf_code"] or group_key == ["anchor_date", "etf_code"]):
        p0_blockers.append("group_key must be ['trade_date','etf_code'] or ['anchor_date','etf_code']")
    if manifest.get("group_label_policy") != GROUP_LABEL_POLICY:
        p0_blockers.append(f"group_label_policy must be {GROUP_LABEL_POLICY}")
    if manifest.get("intraday_live_decision_ready") is not False:
        p0_blockers.append("intraday_live_decision_ready must be false")
    return {
        "passed": not p0_blockers,
        "expected_group_label_policy": GROUP_LABEL_POLICY,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
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


def run_feature_check(manifest: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    feature_columns = string_list(manifest.get("feature_columns"))
    feature_set = set(feature_columns)
    if TARGET_COLUMN not in columns:
        p0_blockers.append(f"{TARGET_COLUMN} missing from samples")
    if TARGET_COLUMN in feature_set:
        p0_blockers.append(f"{TARGET_COLUMN} must not be in feature_columns")
    missing_features = [column for column in feature_columns if column not in columns]
    if missing_features:
        p0_blockers.append("feature columns missing from samples: " + ", ".join(missing_features))
    outcome_overlap = sorted(feature_set & FORBIDDEN_OUTCOME_FEATURES)
    if outcome_overlap:
        p0_blockers.append("outcomes must not be in feature_columns: " + ", ".join(outcome_overlap))
    explicit_forbidden = sorted(feature_set & EXTRA_FORBIDDEN_FEATURES)
    if explicit_forbidden:
        p0_blockers.append("feature_columns contains explicitly forbidden fields: " + ", ".join(explicit_forbidden))
    future_features = sorted(column for column in feature_set if column.startswith("future_"))
    if future_features:
        p0_blockers.append("feature_columns contains future_* fields: " + ", ".join(future_features))
    label_features = sorted(column for column in feature_set if column.endswith("_label"))
    if label_features:
        p0_blockers.append("feature_columns contains *_label fields: " + ", ".join(label_features))
    return {
        "passed": not p0_blockers,
        "feature_columns": feature_columns,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_sample_check(rows: list[dict[str, str]], columns: list[str]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    anchor_column = "anchor_date" if "anchor_date" in columns and any(row.get("anchor_date", "").strip() for row in rows) else "trade_date"
    if anchor_column not in columns:
        p0_blockers.append("samples must contain trade_date or anchor_date")
    if "etf_code" not in columns:
        p0_blockers.append("samples must contain etf_code")
    labels = [normalize_label(row.get(TARGET_COLUMN, "")) for row in rows]
    label_null_count = sum(1 for label in labels if label is None)
    label_0_count = sum(1 for label in labels if label == 0)
    label_1_count = sum(1 for label in labels if label == 1)
    class_count = int(label_0_count > 0) + int(label_1_count > 0)
    min_class_count = min((count for count in (label_0_count, label_1_count) if count > 0), default=0)
    anchors = sorted({str(row.get(anchor_column, "")).strip() for row in rows if str(row.get(anchor_column, "")).strip()})
    etfs = sorted({str(row.get("etf_code", "")).strip() for row in rows if str(row.get("etf_code", "")).strip()})
    if len(rows) < MIN_GROUPS:
        p0_blockers.append(f"group_count must be >= {MIN_GROUPS}")
    if len(anchors) < MIN_ANCHORS:
        p0_blockers.append(f"anchor_count must be >= {MIN_ANCHORS}")
    if len(etfs) < MIN_ETFS:
        p0_blockers.append(f"etf_count must be >= {MIN_ETFS}")
    if class_count < 2:
        p0_blockers.append("group-level label must contain both class 0 and class 1")
    if min_class_count < MIN_CLASS_COUNT:
        p0_blockers.append(f"min_class_count must be >= {MIN_CLASS_COUNT}")
    if label_null_count != 0:
        p0_blockers.append("label_null_count must be 0")
    return {
        "passed": not p0_blockers,
        "anchor_column": anchor_column,
        "group_count": len(rows),
        "anchor_count": len(anchors),
        "etf_count": len(etfs),
        "label_null_count": label_null_count,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "positive_rate": label_1_count / (label_0_count + label_1_count) if (label_0_count + label_1_count) else None,
        "class_count": class_count,
        "min_class_count": min_class_count,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_split_check(rows: list[dict[str, str]], anchor_column: str) -> dict[str, Any]:
    if not rows or not anchor_column:
        return empty_split_check(["samples must contain rows and anchor column"])
    anchors = sorted({str(row.get(anchor_column, "")).strip() for row in rows if str(row.get(anchor_column, "")).strip()})
    attempts: list[dict[str, Any]] = []
    for policy_name, ratio in SPLIT_POLICIES:
        if len(anchors) < 2:
            break
        train_count = max(1, min(len(anchors) - 1, int(len(anchors) * ratio)))
        train_anchors = anchors[:train_count]
        valid_anchors = anchors[train_count:]
        attempt = summarize_split(rows, anchor_column, train_anchors, valid_anchors, policy_name)
        attempts.append(attempt)
        if attempt["split_feasible"]:
            return {**attempt, "attempts": attempts, "p0_blockers": []}
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
    return {
        "selected_split_policy": policy_name,
        "train_anchor_dates": train_anchor_dates,
        "valid_anchor_dates": valid_anchor_dates,
        "train_group_count": len(train_rows),
        "valid_group_count": len(valid_rows),
        "train_label_0_count": train_counts[0],
        "train_label_1_count": train_counts[1],
        "valid_label_0_count": valid_counts[0],
        "valid_label_1_count": valid_counts[1],
        "split_feasible": train_counts[0] > 0 and train_counts[1] > 0 and valid_counts[0] > 0 and valid_counts[1] > 0,
    }


def empty_split_check(p0_blockers: list[str]) -> dict[str, Any]:
    return {
        "selected_split_policy": None,
        "train_anchor_dates": [],
        "valid_anchor_dates": [],
        "train_group_count": 0,
        "valid_group_count": 0,
        "train_label_0_count": 0,
        "train_label_1_count": 0,
        "valid_label_0_count": 0,
        "valid_label_1_count": 0,
        "split_feasible": False,
        "attempts": [],
        "p0_blockers": p0_blockers,
    }


def run_inconsistent_label_review(group_report: dict[str, Any], group_count: int) -> dict[str, Any]:
    stats = group_report.get("group_statistics", {}) if isinstance(group_report, dict) else {}
    inconsistent_count = int(stats.get("inconsistent_label_group_count") or 0)
    rate = inconsistent_count / group_count if group_count else 0.0
    p1_warnings = [P1_INCONSISTENCY] if rate > INCONSISTENCY_RATE_P1_THRESHOLD else []
    return {
        "inconsistent_label_group_count": inconsistent_count,
        "inconsistent_label_group_rate": rate,
        "group_label_policy": GROUP_LABEL_POLICY,
        "intraday_live_decision_ready": False,
        "p1_threshold": INCONSISTENCY_RATE_P1_THRESHOLD,
        "p1_warnings": p1_warnings,
    }


def decide_readiness(
    manifest_ok: bool,
    group_contract_ok: bool,
    boundary_ok: bool,
    feature_ok: bool,
    sample_check: dict[str, Any],
    split_check: dict[str, Any],
    artifact_ok: bool,
    p1_warnings: Sequence[str],
) -> str:
    if not boundary_ok or not artifact_ok:
        return BLOCKED_BOUNDARY_FLAG
    if not manifest_ok or not group_contract_ok or not feature_ok:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    if sample_check["class_count"] < 2:
        return BLOCKED_SINGLE_CLASS
    if (
        sample_check["group_count"] < MIN_GROUPS
        or sample_check["anchor_count"] < MIN_ANCHORS
        or sample_check["etf_count"] < MIN_ETFS
        or sample_check["min_class_count"] < MIN_CLASS_COUNT
        or sample_check["label_null_count"] != 0
    ):
        return BLOCKED_INSUFFICIENT_GROUPS
    if not split_check["split_feasible"]:
        return BLOCKED_SPLIT_NOT_CLASS_DIVERSE
    if P1_INCONSISTENCY in p1_warnings:
        return READY_WITH_INCONSISTENCY
    return READY


def label_counts(rows: list[dict[str, str]]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for row in rows:
        label = normalize_label(row.get(TARGET_COLUMN, ""))
        if label in counts:
            counts[label] += 1
    return counts


def normalize_label(value: Any) -> int | None:
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    return None


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "intraday_group_level_supervised_smoke_readiness_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "readiness_decision": report["readiness_decision"],
        "status": report["status"],
        "selected_split_policy": report["selected_split_policy"],
        "train_anchor_dates": report["train_anchor_dates"],
        "valid_anchor_dates": report["valid_anchor_dates"],
        "train_group_count": report["train_group_count"],
        "valid_group_count": report["valid_group_count"],
        "split_feasible": report["split_feasible"],
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
        "# Intraday Group-Level Supervised Smoke Readiness Precheck",
        "",
        "本文件只用于 group-level supervised smoke readiness precheck，不运行 no-save smoke，不训练模型，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。",
        "",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- group_level_sample: {str(report['group_level_sample']).lower()}",
        f"- group_key: {json.dumps(report['group_key'], ensure_ascii=False)}",
        f"- group_label_policy: {report['group_label_policy']}",
        f"- intraday_live_decision_ready: {str(report['intraday_live_decision_ready']).lower()}",
        f"- group_count: {report['sample_check']['group_count']}",
        f"- anchor_count: {report['sample_check']['anchor_count']}",
        f"- etf_count: {report['sample_check']['etf_count']}",
        f"- label_0_count: {report['sample_check']['label_0_count']}",
        f"- label_1_count: {report['sample_check']['label_1_count']}",
        f"- inconsistent_label_group_count: {report['inconsistent_label_review']['inconsistent_label_group_count']}",
        f"- inconsistent_label_group_rate: {report['inconsistent_label_review']['inconsistent_label_group_rate']}",
        f"- selected_split_policy: {report['selected_split_policy']}",
        f"- split_feasible: {str(report['split_feasible']).lower()}",
        f"- training_allowed: {str(report['training_allowed']).lower()}",
        f"- stable_allowed: {str(report['stable_allowed']).lower()}",
        f"- qmt_allowed: {str(report['qmt_allowed']).lower()}",
        f"- order_intent_allowed: {str(report['order_intent_allowed']).lower()}",
        f"- metrics_are_effectiveness_evidence: {str(report['metrics_are_effectiveness_evidence']).lower()}",
    ]
    (out_dir / "intraday_group_level_supervised_smoke_readiness_report.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only group-level supervised smoke readiness precheck.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--group-report", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_precheck(args.samples, args.manifest, args.group_report, args.out_dir)
    except GroupLevelReadinessError as exc:
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
                "selected_split_policy": report["selected_split_policy"],
                "split_feasible": report["split_feasible"],
                "train_group_count": report["train_group_count"],
                "valid_group_count": report["valid_group_count"],
                "training_allowed": False,
                "stable_allowed": False,
                "qmt_allowed": False,
                "order_intent_allowed": False,
                "metrics_are_effectiveness_evidence": False,
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
