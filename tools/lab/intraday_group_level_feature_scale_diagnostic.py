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

from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts, load_json


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_feature_scale_diagnostic")
REPORT_TYPE = "intraday_group_level_feature_scale_diagnostic"
TARGET_COLUMN = "three_day_positive_label"
ANCHOR_COLUMN_CANDIDATES = ("anchor_date", "trade_date")
BOUNDARY_FALSE_FIELDS = [
    "training_allowed",
    "supervised_training_allowed",
    "stable_effect_allowed",
    "contains_order_intent",
    "contains_live_order",
    "contains_secret",
    "model_training_allowed",
    "stable_allowed",
    "qmt_allowed",
    "order_intent_allowed",
    "automatic_promotion_ready",
    "metrics_are_effectiveness_evidence",
]
EXPECTED_SPLIT_POLICY = "anchor_date_70_30"
COMPLETED_TRANSFORM_RECOMMENDED = "FEATURE_SCALE_DIAGNOSTIC_COMPLETED_TRANSFORM_POLICY_RECOMMENDED"
COMPLETED_NO_P0 = "FEATURE_SCALE_DIAGNOSTIC_COMPLETED_NO_P0_REVIEW_REQUIRED"
BLOCKED_MANIFEST_LEAKAGE_P0 = "BLOCKED_MANIFEST_LEAKAGE_P0"
BLOCKED_BOUNDARY_FLAG = "BLOCKED_BOUNDARY_FLAG"
BLOCKED_DIAGNOSTIC_RUNTIME_ERROR = "BLOCKED_DIAGNOSTIC_RUNTIME_ERROR"
SHIFT_THRESHOLD = 0.5
EXTREME_SCALE_STD_RATIO = 1_000.0
EXTREME_ABS_MAX_RATIO = 1_000_000.0
OUTLIER_Z_REVIEW_THRESHOLD = 8.0
SPIKE_RATIO_LOG1P_THRESHOLD = 10.0


class FeatureScaleDiagnosticError(RuntimeError):
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
        raise FeatureScaleDiagnosticError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise FeatureScaleDiagnosticError(f"samples CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise FeatureScaleDiagnosticError(f"samples CSV has no header: {path}")
    return rows, columns


def run_diagnostic(
    samples_path: Path,
    manifest_path: Path,
    readiness_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(samples_path, repo_root)
    resolved_manifest = resolve_repo_path(manifest_path, repo_root)
    resolved_readiness = resolve_repo_path(readiness_path, repo_root)
    for path, label in (
        (resolved_samples, "samples"),
        (resolved_manifest, "manifest"),
        (resolved_readiness, "readiness"),
    ):
        if not path.exists():
            raise FeatureScaleDiagnosticError(f"{label} path does not exist: {path}")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    rows, columns = load_csv_rows(resolved_samples)
    manifest = load_json(resolved_manifest)
    readiness = load_json(resolved_readiness)

    feature_check = run_feature_check(manifest, columns)
    boundary_check = run_boundary_check(manifest, readiness)
    readiness_check = run_readiness_check(readiness)
    artifact_check_before = check_model_artifacts(resolved_out_dir)

    p0_blockers: list[str] = []
    p0_blockers.extend(feature_check["p0_blockers"])
    p0_blockers.extend(boundary_check["p0_blockers"])
    p0_blockers.extend(readiness_check["p0_blockers"])
    p0_blockers.extend(artifact_check_before["p0_blockers"])

    feature_columns = feature_check["feature_columns"]
    split_payload = build_split_payload(rows, readiness) if not p0_blockers else empty_split_payload()
    p0_blockers.extend(split_payload["p0_blockers"])

    if p0_blockers:
        feature_statistics: dict[str, Any] = {}
        feature_scale_summary = empty_scale_summary()
        feature_shift_summary = empty_shift_summary()
        recommended_transforms = build_recommended_transforms([], {}, feature_scale_summary)
        diagnostic_flags = ["NO_FORMAL_MODEL_EVIDENCE"]
        decision = BLOCKED_BOUNDARY_FLAG if boundary_check["p0_blockers"] else BLOCKED_MANIFEST_LEAKAGE_P0
    else:
        feature_statistics = {
            feature: diagnose_feature(feature, split_payload["train_rows"], split_payload["valid_rows"], rows)
            for feature in feature_columns
        }
        feature_scale_summary = summarize_feature_scales(feature_statistics)
        feature_shift_summary = summarize_feature_shift(feature_statistics)
        recommended_transforms = build_recommended_transforms(feature_columns, feature_statistics, feature_scale_summary)
        diagnostic_flags = determine_flags(feature_scale_summary, feature_shift_summary, recommended_transforms)
        decision = (
            COMPLETED_TRANSFORM_RECOMMENDED
            if recommended_transforms["log1p_recommended"]
            or recommended_transforms["standardize_recommended"]
            or recommended_transforms["clip_winsorize_review"]
            else COMPLETED_NO_P0
        )

    artifact_check_after = check_model_artifacts(resolved_out_dir)
    if artifact_check_after["p0_blockers"]:
        p0_blockers.extend(artifact_check_after["p0_blockers"])
        decision = BLOCKED_BOUNDARY_FLAG
    status = "blocked" if decision.startswith("BLOCKED_") else "passed"

    policy = build_transform_policy(recommended_transforms)
    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": REPORT_TYPE,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "readiness_path": str(readiness_path),
        "feature_count": len(feature_columns),
        "feature_columns": feature_columns,
        "target": TARGET_COLUMN,
        "selected_split_policy": split_payload["selected_split_policy"],
        "anchor_column": split_payload["anchor_column"],
        "train_anchor_dates": split_payload["train_anchor_dates"],
        "valid_anchor_dates": split_payload["valid_anchor_dates"],
        "train_group_count": split_payload["train_group_count"],
        "valid_group_count": split_payload["valid_group_count"],
        "feature_scale_summary": feature_scale_summary,
        "feature_shift_summary": feature_shift_summary,
        "feature_statistics": feature_statistics,
        "recommended_transforms": recommended_transforms,
        "transform_policy_recommendation": policy,
        "diagnostic_flags": diagnostic_flags,
        "decision": decision,
        "training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "model_training_allowed": False,
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "not_trading_advice": True,
        "feature_check": feature_check,
        "boundary_check": boundary_check,
        "readiness_check": readiness_check,
        "split_check": split_payload["split_check"],
        "artifact_check_before": artifact_check_before,
        "artifact_check_after": artifact_check_after,
        "p0_blockers": dedupe(p0_blockers),
        "p1_warnings": build_p1_warnings(diagnostic_flags),
    }
    write_reports(report, policy, resolved_out_dir)
    return report


def run_feature_check(manifest: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    feature_columns = string_list(manifest.get("feature_columns"))
    feature_set = set(feature_columns)
    label_columns = set(string_list(manifest.get("label_columns")))
    outcome_columns = set(string_list(manifest.get("outcome_columns")))
    if not feature_columns:
        p0_blockers.append("manifest.feature_columns must be non-empty")
    missing_features = [column for column in feature_columns if column not in columns]
    if missing_features:
        p0_blockers.append("feature columns missing from samples: " + ", ".join(missing_features))
    label_overlap = sorted(feature_set & label_columns)
    if label_overlap:
        p0_blockers.append("label columns must not be in feature_columns: " + ", ".join(label_overlap))
    outcome_overlap = sorted(feature_set & outcome_columns)
    if outcome_overlap:
        p0_blockers.append("outcome columns must not be in feature_columns: " + ", ".join(outcome_overlap))
    future_features = sorted(column for column in feature_set if "future" in column.lower())
    if future_features:
        p0_blockers.append("future columns must not be in feature_columns: " + ", ".join(future_features))
    label_pattern_columns = sorted(column for column in feature_set if "label" in column.lower())
    if label_pattern_columns:
        p0_blockers.append("label-pattern columns must not be in feature_columns: " + ", ".join(label_pattern_columns))
    outcome_pattern_columns = sorted(column for column in feature_set if "outcome" in column.lower())
    if outcome_pattern_columns:
        p0_blockers.append("outcome-pattern columns must not be in feature_columns: " + ", ".join(outcome_pattern_columns))
    return {
        "passed": not p0_blockers,
        "source": "manifest.feature_columns",
        "feature_columns": feature_columns,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_boundary_check(manifest: dict[str, Any], readiness: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    for source_name, payload in (("manifest", manifest), ("readiness", readiness)):
        for field_name in BOUNDARY_FALSE_FIELDS:
            if field_name in payload and payload.get(field_name) is not False:
                p0_blockers.append(f"{source_name}.{field_name} must be false")
    return {
        "passed": not p0_blockers,
        "checked_fields": BOUNDARY_FALSE_FIELDS,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_readiness_check(readiness: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    if readiness.get("status") not in (None, "passed"):
        p0_blockers.append("readiness.status must be passed when present")
    if readiness.get("selected_split_policy") not in (None, EXPECTED_SPLIT_POLICY):
        p0_blockers.append(f"readiness.selected_split_policy must be {EXPECTED_SPLIT_POLICY}")
    if int(readiness.get("train_group_count") or 0) <= 0 or int(readiness.get("valid_group_count") or 0) <= 0:
        p0_blockers.append("readiness train_group_count and valid_group_count must be positive")
    return {
        "passed": not p0_blockers,
        "expected_split_policy": EXPECTED_SPLIT_POLICY,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def build_split_payload(rows: list[dict[str, str]], readiness: dict[str, Any]) -> dict[str, Any]:
    anchor_column = next((column for column in ANCHOR_COLUMN_CANDIDATES if rows and column in rows[0]), "trade_date")
    if anchor_column == "anchor_date" and not any(str(row.get(anchor_column, "")).strip() for row in rows):
        anchor_column = "trade_date"
    train_anchor_dates = string_list(readiness.get("train_anchor_dates"))
    valid_anchor_dates = string_list(readiness.get("valid_anchor_dates"))
    if not train_anchor_dates or not valid_anchor_dates:
        train_anchor_dates, valid_anchor_dates = derive_anchor_date_split(rows, anchor_column)
    train_set = set(train_anchor_dates)
    valid_set = set(valid_anchor_dates)
    train_rows = [row for row in rows if str(row.get(anchor_column, "")).strip() in train_set]
    valid_rows = [row for row in rows if str(row.get(anchor_column, "")).strip() in valid_set]

    p0_blockers: list[str] = []
    if not train_rows or not valid_rows:
        p0_blockers.append("split must select non-empty train and valid groups")
    expected_train = readiness.get("train_group_count")
    expected_valid = readiness.get("valid_group_count")
    if expected_train is not None and len(train_rows) != int(expected_train):
        p0_blockers.append(f"train_group_count mismatch: expected {expected_train}, got {len(train_rows)}")
    if expected_valid is not None and len(valid_rows) != int(expected_valid):
        p0_blockers.append(f"valid_group_count mismatch: expected {expected_valid}, got {len(valid_rows)}")
    return {
        "selected_split_policy": readiness.get("selected_split_policy") or EXPECTED_SPLIT_POLICY,
        "anchor_column": anchor_column,
        "train_anchor_dates": train_anchor_dates,
        "valid_anchor_dates": valid_anchor_dates,
        "train_group_count": len(train_rows),
        "valid_group_count": len(valid_rows),
        "train_rows": train_rows,
        "valid_rows": valid_rows,
        "split_check": {
            "selected_split_policy": readiness.get("selected_split_policy") or EXPECTED_SPLIT_POLICY,
            "anchor_column": anchor_column,
            "train_anchor_dates": train_anchor_dates,
            "valid_anchor_dates": valid_anchor_dates,
            "p0_blockers": p0_blockers,
        },
        "p0_blockers": p0_blockers,
    }


def derive_anchor_date_split(rows: list[dict[str, str]], anchor_column: str) -> tuple[list[str], list[str]]:
    anchor_dates = sorted({str(row.get(anchor_column, "")).strip() for row in rows if str(row.get(anchor_column, "")).strip()})
    if len(anchor_dates) < 2:
        return anchor_dates, []
    train_count = max(1, int(len(anchor_dates) * 0.70))
    if train_count >= len(anchor_dates):
        train_count = len(anchor_dates) - 1
    return anchor_dates[:train_count], anchor_dates[train_count:]


def empty_split_payload() -> dict[str, Any]:
    return {
        "selected_split_policy": EXPECTED_SPLIT_POLICY,
        "anchor_column": "",
        "train_anchor_dates": [],
        "valid_anchor_dates": [],
        "train_group_count": 0,
        "valid_group_count": 0,
        "train_rows": [],
        "valid_rows": [],
        "split_check": {"p0_blockers": []},
        "p0_blockers": [],
    }


def diagnose_feature(
    feature: str,
    train_rows: list[dict[str, str]],
    valid_rows: list[dict[str, str]],
    all_rows: list[dict[str, str]],
) -> dict[str, Any]:
    train_profile = numeric_profile(train_rows, feature)
    valid_profile = numeric_profile(valid_rows, feature)
    all_profile = numeric_profile(all_rows, feature)
    train_std = train_profile["std"] or 0.0
    smd = None
    if train_std > 0 and train_profile["mean"] is not None and valid_profile["mean"] is not None:
        smd = (train_profile["mean"] - valid_profile["mean"]) / train_std
    absolute_max = all_profile["absolute_max"]
    scale_order = math.floor(math.log10(absolute_max)) if absolute_max and absolute_max > 0 else None
    max_train_abs_z = max_abs_train_zscore(train_rows, feature, train_profile)
    return {
        "train": train_profile,
        "valid": valid_profile,
        "missing_count": all_profile["missing_count"],
        "inf_count": all_profile["inf_count"],
        "zero_variance": train_std == 0,
        "absolute_max": absolute_max,
        "scale_order": scale_order,
        "train_vs_valid_standardized_mean_difference": smd,
        "max_train_abs_zscore": max_train_abs_z,
    }


def numeric_profile(rows: list[dict[str, str]], feature: str) -> dict[str, Any]:
    values: list[float] = []
    missing_count = 0
    inf_count = 0
    for row in rows:
        parsed = parse_float(row.get(feature, ""))
        if parsed is None:
            missing_count += 1
        elif math.isinf(parsed):
            inf_count += 1
        else:
            values.append(parsed)
    return {
        "count": len(values),
        "missing_count": missing_count,
        "inf_count": inf_count,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "mean": mean(values) if values else None,
        "std": population_std(values),
        "absolute_max": max((abs(value) for value in values), default=0.0),
    }


def parse_float(value: Any) -> float | None:
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def population_std(values: list[float]) -> float | None:
    if not values:
        return None
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def max_abs_train_zscore(rows: list[dict[str, str]], feature: str, profile: dict[str, Any]) -> float | None:
    std = profile["std"] or 0.0
    avg = profile["mean"]
    if std <= 0 or avg is None:
        return None
    zscores = []
    for row in rows:
        value = parse_float(row.get(feature, ""))
        if value is not None and math.isfinite(value):
            zscores.append(abs((value - avg) / std))
    return max(zscores, default=None)


def summarize_feature_scales(feature_statistics: dict[str, Any]) -> dict[str, Any]:
    stds = {
        feature: stats["train"]["std"]
        for feature, stats in feature_statistics.items()
        if stats["train"]["std"] is not None and stats["train"]["std"] > 0
    }
    absmax = {
        feature: stats["absolute_max"]
        for feature, stats in feature_statistics.items()
        if stats["absolute_max"] is not None and stats["absolute_max"] > 0
    }
    zero_variance = sorted(feature for feature, stats in feature_statistics.items() if stats["zero_variance"])
    return {
        "cross_feature_std_ratio": ratio(max(stds.values(), default=0.0), min(stds.values(), default=0.0)),
        "cross_feature_absolute_max_ratio": ratio(max(absmax.values(), default=0.0), min(absmax.values(), default=0.0)),
        "largest_train_std_feature": max(stds, key=stds.get) if stds else None,
        "smallest_positive_train_std_feature": min(stds, key=stds.get) if stds else None,
        "largest_absolute_max_feature": max(absmax, key=absmax.get) if absmax else None,
        "smallest_positive_absolute_max_feature": min(absmax, key=absmax.get) if absmax else None,
        "zero_variance_features": zero_variance,
    }


def summarize_feature_shift(feature_statistics: dict[str, Any]) -> dict[str, Any]:
    shifted = sorted(
        feature
        for feature, stats in feature_statistics.items()
        if stats["train_vs_valid_standardized_mean_difference"] is not None
        and abs(stats["train_vs_valid_standardized_mean_difference"]) >= SHIFT_THRESHOLD
    )
    return {
        "shift_threshold_abs_smd": SHIFT_THRESHOLD,
        "shifted_features": shifted,
        "shifted_feature_count": len(shifted),
        "max_abs_standardized_mean_difference": max(
            (
                abs(stats["train_vs_valid_standardized_mean_difference"])
                for stats in feature_statistics.values()
                if stats["train_vs_valid_standardized_mean_difference"] is not None
            ),
            default=0.0,
        ),
    }


def empty_scale_summary() -> dict[str, Any]:
    return {
        "cross_feature_std_ratio": None,
        "cross_feature_absolute_max_ratio": None,
        "largest_train_std_feature": None,
        "smallest_positive_train_std_feature": None,
        "largest_absolute_max_feature": None,
        "smallest_positive_absolute_max_feature": None,
        "zero_variance_features": [],
    }


def empty_shift_summary() -> dict[str, Any]:
    return {
        "shift_threshold_abs_smd": SHIFT_THRESHOLD,
        "shifted_features": [],
        "shifted_feature_count": 0,
        "max_abs_standardized_mean_difference": 0.0,
    }


def ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def build_recommended_transforms(
    feature_columns: list[str],
    feature_statistics: dict[str, Any],
    feature_scale_summary: dict[str, Any],
) -> dict[str, Any]:
    log1p = sorted(feature for feature in feature_columns if is_log1p_candidate(feature, feature_statistics.get(feature, {})))
    standardize = sorted(feature for feature in feature_columns if not feature_statistics.get(feature, {}).get("zero_variance"))
    clipping_review = sorted(
        feature
        for feature, stats in feature_statistics.items()
        if requires_clipping_review(feature, stats, feature_scale_summary)
    )
    bounded = sorted(feature for feature in feature_columns if is_bounded_or_ratio_feature(feature))
    return {
        "log1p_recommended": log1p,
        "standardize_recommended": standardize,
        "clip_winsorize_review": clipping_review,
        "no_transform_or_bounded": bounded,
        "notes": [
            "log1p is recommended only as a train/valid preprocessing design choice, not fitted or saved by this diagnostic.",
            "standardization must fit train statistics only, then transform valid with those train statistics.",
            "clipping or winsorization review must use train quantiles only; this diagnostic does not clip data.",
        ],
    }


def is_log1p_candidate(feature: str, stats: dict[str, Any]) -> bool:
    name = feature.lower()
    is_raw_flow = ("volume" in name or "amount" in name) and not any(
        token in name for token in ("ratio", "rank", "relative")
    )
    if is_raw_flow:
        return True
    if "spike_ratio" in name:
        return (stats.get("absolute_max") or 0.0) >= SPIKE_RATIO_LOG1P_THRESHOLD
    return False


def is_bounded_or_ratio_feature(feature: str) -> bool:
    name = feature.lower()
    return any(token in name for token in ("ratio", "rank", "return")) or name.startswith("close_vs_")


def requires_clipping_review(feature: str, stats: dict[str, Any], feature_scale_summary: dict[str, Any]) -> bool:
    if stats.get("max_train_abs_zscore") is not None and stats["max_train_abs_zscore"] >= OUTLIER_Z_REVIEW_THRESHOLD:
        return True
    if stats.get("absolute_max") is not None and stats["absolute_max"] >= 1_000_000_000:
        return True
    if is_log1p_candidate(feature, stats) and (feature_scale_summary.get("cross_feature_absolute_max_ratio") or 0) >= EXTREME_ABS_MAX_RATIO:
        return True
    return False


def determine_flags(
    feature_scale_summary: dict[str, Any],
    feature_shift_summary: dict[str, Any],
    recommended_transforms: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if (feature_scale_summary.get("cross_feature_std_ratio") or 0) >= EXTREME_SCALE_STD_RATIO or (
        feature_scale_summary.get("cross_feature_absolute_max_ratio") or 0
    ) >= EXTREME_ABS_MAX_RATIO:
        flags.append("FEATURE_SCALE_RISK_CONFIRMED")
    if recommended_transforms["log1p_recommended"]:
        flags.append("LOG1P_TRANSFORM_RECOMMENDED")
    if recommended_transforms["standardize_recommended"]:
        flags.append("TRAIN_ONLY_STANDARDIZATION_RECOMMENDED")
    if recommended_transforms["clip_winsorize_review"]:
        flags.append("CLIPPING_REVIEW_RECOMMENDED")
    if feature_scale_summary["zero_variance_features"]:
        flags.append("ZERO_VARIANCE_FEATURE_FOUND")
    if feature_shift_summary["shifted_features"]:
        flags.append("TRAIN_VALID_FEATURE_SHIFT_OBSERVED")
    flags.extend(["NO_FEATURE_SCALE_P0", "NO_FORMAL_MODEL_EVIDENCE"])
    return flags


def build_transform_policy(recommended_transforms: dict[str, Any]) -> dict[str, Any]:
    return {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "policy_scope": "diagnostic_only",
        "train_only_fit_required": True,
        "save_scaler": False,
        "model_training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "not_trading_advice": True,
        "recommended_transforms": recommended_transforms,
        "fit_policy": {
            "standardization": "fit mean/std on train only; transform valid with train statistics",
            "log1p": "apply deterministic log1p after non-negative review; no fitted artifact",
            "clip_winsorize": "review train quantiles only; this diagnostic does not clip",
        },
    }


def build_p1_warnings(flags: list[str]) -> list[str]:
    warnings: list[str] = []
    if "FEATURE_SCALE_RISK_CONFIRMED" in flags:
        warnings.append("P1_EXTREME_FEATURE_SCALE_REVIEW_REQUIRED")
    if "TRAIN_VALID_FEATURE_SHIFT_OBSERVED" in flags:
        warnings.append("P1_TRAIN_VALID_FEATURE_SHIFT_REVIEW_REQUIRED")
    return warnings


def write_reports(report: dict[str, Any], policy: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "intraday_group_level_feature_scale_diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "transform_policy_recommendation.json").write_text(
        json.dumps(policy, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "",
        "# Intraday Group-Level Feature Scale Diagnostic",
        "",
        "本文件只用于 Lab feature scale diagnostic / transform design，不训练模型，不保存 scaler，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。",
        "",
        f"- report_type: {report['report_type']}",
        f"- status: {report['status']}",
        f"- decision: {report['decision']}",
        f"- feature_count: {report['feature_count']}",
        f"- train_group_count: {report['train_group_count']}",
        f"- valid_group_count: {report['valid_group_count']}",
        f"- cross_feature_std_ratio: {report['feature_scale_summary']['cross_feature_std_ratio']}",
        f"- cross_feature_absolute_max_ratio: {report['feature_scale_summary']['cross_feature_absolute_max_ratio']}",
        f"- shifted_feature_count: {report['feature_shift_summary']['shifted_feature_count']}",
        f"- diagnostic_flags: {', '.join(report['diagnostic_flags'])}",
        "",
        "## Recommended Transforms",
        "",
        f"- log1p_recommended: {', '.join(report['recommended_transforms']['log1p_recommended']) or 'none'}",
        f"- standardize_recommended: {', '.join(report['recommended_transforms']['standardize_recommended']) or 'none'}",
        f"- clip_winsorize_review: {', '.join(report['recommended_transforms']['clip_winsorize_review']) or 'none'}",
        f"- no_transform_or_bounded: {', '.join(report['recommended_transforms']['no_transform_or_bounded']) or 'none'}",
        "",
        "## Boundary",
        "",
        "- policy_scope: diagnostic_only",
        "- train_only_fit_required: true",
        "- model_training_allowed: false",
        "- scaler_saved: false",
        "- checkpoint_saved: false",
        "- gpu_used: false",
        "- torchrun_used: false",
        "- qmt_used: false",
        "- order_intent_generated: false",
        "- stable_affected: false",
        "- metrics_are_effectiveness_evidence: false",
        "- not_trading_advice: true",
    ]
    (out_dir / "intraday_group_level_feature_scale_diagnostic_report.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        report = run_diagnostic(args.samples, args.manifest, args.readiness, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI must return an auditable blocked payload.
        payload = {
            "report_type": REPORT_TYPE,
            "status": "blocked",
            "decision": BLOCKED_DIAGNOSTIC_RUNTIME_ERROR,
            "error": str(exc),
            "training_allowed": False,
            "stable_allowed": False,
            "qmt_allowed": False,
            "order_intent_allowed": False,
            "model_saved": False,
            "scaler_saved": False,
            "checkpoint_saved": False,
            "gpu_used": False,
            "torchrun_used": False,
            "qmt_used": False,
            "order_intent_generated": False,
            "stable_affected": False,
            "not_trading_advice": True,
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({"status": report["status"], "decision": report["decision"], "diagnostic_flags": report["diagnostic_flags"]}, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
