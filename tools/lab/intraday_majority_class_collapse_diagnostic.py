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

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest
from tools.lab.intraday_supervised_no_save_smoke import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    BLOCKED_MODEL_ARTIFACT_CREATED,
    TARGET_COLUMN,
    check_model_artifacts,
    load_csv_rows,
    load_json,
    run_boundary_check,
    run_feature_check,
)


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_majority_class_collapse_diagnostic")
REPORT_TYPE = "intraday_majority_class_collapse_diagnostic"
READY_FEATURE_LABEL_REVIEW = "DIAGNOSTIC_COMPLETED_FEATURE_LABEL_REVIEW_REQUIRED"
READY_BALANCED_PROBE = "DIAGNOSTIC_COMPLETED_BALANCED_SCALED_PROBE_RECOMMENDED"
READY_PAST_ONLY_FEATURES = "DIAGNOSTIC_COMPLETED_PAST_ONLY_FEATURE_EXPANSION_RECOMMENDED"
READY_GROUP_LEVEL_SAMPLE = "DIAGNOSTIC_COMPLETED_GROUP_LEVEL_SAMPLE_RECOMMENDED"
BLOCKED_DIAGNOSTIC_RUNTIME_ERROR = "BLOCKED_DIAGNOSTIC_RUNTIME_ERROR"
ANCHOR_COLUMN_CANDIDATES = ("anchor_date", "trade_date")
ETF_COLUMN_CANDIDATES = ("etf_code", "symbol")
EXTRA_FORBIDDEN_FEATURES = {
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
    "three_day_positive_label",
}
OUTCOME_PATTERNS = ("execution", "outcome")


class MajorityCollapseDiagnosticError(RuntimeError):
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
        raise MajorityCollapseDiagnosticError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def run_diagnostic(
    samples_path: Path,
    manifest_path: Path,
    readiness_path: Path,
    repeatability_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(samples_path, repo_root)
    resolved_manifest = resolve_repo_path(manifest_path, repo_root)
    resolved_readiness = resolve_repo_path(readiness_path, repo_root)
    resolved_repeatability = resolve_repo_path(repeatability_path, repo_root)
    for required_path, label in (
        (resolved_samples, "samples"),
        (resolved_manifest, "manifest"),
        (resolved_readiness, "readiness"),
        (resolved_repeatability, "repeatability"),
    ):
        if not required_path.exists():
            raise MajorityCollapseDiagnosticError(f"{label} path does not exist: {required_path}")

    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    manifest = load_json(resolved_manifest)
    readiness = load_json(resolved_readiness)
    repeatability = load_json(resolved_repeatability)
    rows, columns = load_csv_rows(resolved_samples)
    feature_columns = choose_feature_columns(repeatability, readiness, manifest)

    manifest_check = check_label_manifest(resolved_manifest)
    boundary_check = run_boundary_check(manifest)
    feature_check = run_feature_check({**manifest, "feature_columns": feature_columns}, columns)
    explicit_feature_check = run_explicit_feature_leakage_check(feature_columns)
    artifact_check_before = check_model_artifacts(resolved_out_dir)
    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    p0_blockers.extend(boundary_check["p0_blockers"])
    p0_blockers.extend(feature_check["p0_blockers"])
    p0_blockers.extend(explicit_feature_check["p0_blockers"])
    p0_blockers.extend(artifact_check_before["p0_blockers"])

    split = build_diagnostic_split(rows, readiness, repeatability, feature_columns)
    label_distribution = build_label_distribution(rows, split)
    sample_granularity = build_sample_granularity(rows, split["anchor_column"], split["etf_column"])
    feature_scale_diagnostic = build_feature_scale_diagnostic(split, feature_columns)
    univariate_signal_diagnostic = build_univariate_signal_diagnostic(split, feature_columns)

    logistic_probability_diagnostic: dict[str, Any] = {"status": "not_run"}
    balanced_scaled_probe: dict[str, Any] = {"status": "not_run"}
    decision = decide_blocked(p0_blockers, boundary_check, manifest_check.ok, feature_check, explicit_feature_check)
    if decision is None:
        try:
            logistic_probability_diagnostic = run_logistic_probability_diagnostic(split)
            balanced_scaled_probe = run_balanced_scaled_probe(split)
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            if artifact_check_after["p0_blockers"]:
                decision = BLOCKED_MODEL_ARTIFACT_CREATED
        except Exception as exc:  # noqa: BLE001 - diagnostic must report runtime errors.
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.append(f"diagnostic runtime error: {exc}")
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            decision = BLOCKED_DIAGNOSTIC_RUNTIME_ERROR
    else:
        artifact_check_after = artifact_check_before

    diagnostic_flags = build_diagnostic_flags(
        label_distribution,
        sample_granularity,
        feature_scale_diagnostic,
        univariate_signal_diagnostic,
        logistic_probability_diagnostic,
        balanced_scaled_probe,
    )
    if decision is None:
        decision = choose_ready_decision(diagnostic_flags)

    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": REPORT_TYPE,
        "status": "blocked" if decision.startswith("BLOCKED_") else "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sample": str(samples_path),
        "manifest_path": str(manifest_path),
        "readiness_path": str(readiness_path),
        "repeatability_path": str(repeatability_path),
        "target": TARGET_COLUMN,
        "feature_columns": feature_columns,
        "label_distribution": label_distribution,
        "train_valid_label_shift": label_distribution["train_valid_label_shift"],
        "sample_granularity": sample_granularity,
        "feature_scale_diagnostic": feature_scale_diagnostic,
        "univariate_signal_diagnostic": univariate_signal_diagnostic,
        "logistic_probability_diagnostic": logistic_probability_diagnostic,
        "balanced_scaled_probe": balanced_scaled_probe,
        "diagnostic_flags": diagnostic_flags,
        "diagnostic_decision": decision,
        "manifest_leakage_check": manifest_check.to_summary(),
        "boundary_check": boundary_check,
        "feature_check": feature_check,
        "explicit_feature_leakage_check": explicit_feature_check,
        "artifact_check_before": artifact_check_before,
        "artifact_check_after": artifact_check_after,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "not_trading_advice": True,
        "formal_training": False,
        "hyperparameter_tuning": False,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }
    write_reports(report, resolved_out_dir)
    return report


def choose_feature_columns(
    repeatability: dict[str, Any],
    readiness: dict[str, Any],
    manifest: dict[str, Any],
) -> list[str]:
    for payload in (repeatability, readiness.get("sample_check", {}), readiness, manifest):
        value = payload.get("feature_columns") if isinstance(payload, dict) else None
        if isinstance(value, list) and value:
            return [str(item) for item in value if str(item)]
    raise MajorityCollapseDiagnosticError("feature_columns not found")


def run_explicit_feature_leakage_check(feature_columns: Sequence[str]) -> dict[str, Any]:
    feature_set = set(feature_columns)
    forbidden = sorted(feature_set & EXTRA_FORBIDDEN_FEATURES)
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
        "explicit_forbidden_features": sorted(EXTRA_FORBIDDEN_FEATURES),
        "outcome_patterns": list(OUTCOME_PATTERNS),
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def build_diagnostic_split(
    rows: list[dict[str, str]],
    readiness: dict[str, Any],
    repeatability: dict[str, Any],
    feature_columns: list[str],
) -> dict[str, Any]:
    if not rows:
        raise MajorityCollapseDiagnosticError("samples CSV has no rows")
    anchor_column = next((column for column in ANCHOR_COLUMN_CANDIDATES if column in rows[0]), "trade_date")
    etf_column = next((column for column in ETF_COLUMN_CANDIDATES if column in rows[0]), "etf_code")
    train_anchor_dates = first_string_list(
        repeatability.get("train_anchor_dates"),
        readiness.get("train_anchor_dates"),
        readiness.get("split_check", {}).get("train_anchor_dates") if isinstance(readiness.get("split_check"), dict) else None,
    )
    valid_anchor_dates = first_string_list(
        repeatability.get("valid_anchor_dates"),
        readiness.get("valid_anchor_dates"),
        readiness.get("split_check", {}).get("valid_anchor_dates") if isinstance(readiness.get("split_check"), dict) else None,
    )
    if not train_anchor_dates or not valid_anchor_dates:
        raise MajorityCollapseDiagnosticError("train/valid anchor dates not found")
    train_set = set(train_anchor_dates)
    valid_set = set(valid_anchor_dates)
    train_rows = [row for row in rows if str(row.get(anchor_column, "")).strip() in train_set]
    valid_rows = [row for row in rows if str(row.get(anchor_column, "")).strip() in valid_set]
    if not train_rows or not valid_rows:
        raise MajorityCollapseDiagnosticError("train and valid splits must both be non-empty")
    return {
        "anchor_column": anchor_column,
        "etf_column": etf_column,
        "train_anchor_dates": train_anchor_dates,
        "valid_anchor_dates": valid_anchor_dates,
        "train_rows_raw": train_rows,
        "valid_rows_raw": valid_rows,
        "x_train": rows_to_float_matrix(train_rows, feature_columns),
        "y_train": rows_to_labels(train_rows),
        "x_valid": rows_to_float_matrix(valid_rows, feature_columns),
        "y_valid": rows_to_labels(valid_rows),
    }


def first_string_list(*values: Any) -> list[str]:
    for value in values:
        if isinstance(value, list) and value:
            return [str(item) for item in value if str(item)]
    return []


def build_label_distribution(rows: list[dict[str, str]], split: dict[str, Any]) -> dict[str, Any]:
    anchor_column = split["anchor_column"]
    etf_column = split["etf_column"]
    train_rows = split["train_rows_raw"]
    valid_rows = split["valid_rows_raw"]
    train_dist = label_distribution(train_rows)
    valid_dist = label_distribution(valid_rows)
    train_rate = positive_rate(train_dist)
    valid_rate = positive_rate(valid_dist)
    return {
        "overall": label_distribution(rows),
        "train": train_dist,
        "valid": valid_dist,
        "per_anchor": grouped_label_distribution(rows, anchor_column),
        "per_etf": grouped_label_distribution(rows, etf_column),
        "train_positive_rate": train_rate,
        "valid_positive_rate": valid_rate,
        "train_valid_label_shift": {
            "train_positive_rate": train_rate,
            "valid_positive_rate": valid_rate,
            "absolute_positive_rate_delta": abs(train_rate - valid_rate),
            "observed": abs(train_rate - valid_rate) >= 0.10,
            "diagnostic_only": True,
        },
    }


def build_sample_granularity(rows: list[dict[str, str]], anchor_column: str, etf_column: str) -> dict[str, Any]:
    anchors = sorted({str(row.get(anchor_column, "")).strip() for row in rows if str(row.get(anchor_column, "")).strip()})
    etfs = sorted({str(row.get(etf_column, "")).strip() for row in rows if str(row.get(etf_column, "")).strip()})
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        groups.setdefault((str(row.get(anchor_column, "")).strip(), str(row.get(etf_column, "")).strip()), []).append(row)
    bars_per_group = [len(group_rows) for group_rows in groups.values()]
    repeated_groups = []
    for (anchor, etf), group_rows in sorted(groups.items()):
        labels = {label_from_row(row) for row in group_rows}
        if len(group_rows) > 1 and len(labels) == 1:
            repeated_groups.append({"anchor": anchor, "etf_code": etf, "rows": len(group_rows), "label": next(iter(labels))})
    return {
        "row_count": len(rows),
        "anchor_column": anchor_column,
        "anchor_count": len(anchors),
        "anchors": anchors,
        "etf_column": etf_column,
        "etf_count": len(etfs),
        "etf_codes": etfs,
        "estimated_independent_group_count": len(anchors) * len(etfs),
        "observed_anchor_etf_group_count": len(groups),
        "bars_per_anchor_etf": summarize_numbers(bars_per_group),
        "groups_with_single_repeated_label_count": len(repeated_groups),
        "all_groups_have_single_repeated_label": len(repeated_groups) == len(groups) if groups else False,
        "has_48_bar_repeated_label_groups": any(group["rows"] == 48 for group in repeated_groups),
        "repeated_label_group_structure_observed": bool(repeated_groups),
        "repeated_label_group_examples": repeated_groups[:10],
        "diagnostic_only": True,
    }


def build_feature_scale_diagnostic(split: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    x_train = split["x_train"]
    x_valid = split["x_valid"]
    per_feature: dict[str, Any] = {}
    zero_variance: list[str] = []
    extreme_scale_ratio: list[str] = []
    train_stds: list[float] = []
    for index, feature in enumerate(feature_columns):
        train_values = [row[index] for row in x_train]
        valid_values = [row[index] for row in x_valid]
        train_stats = numeric_stats(train_values)
        valid_stats = numeric_stats(valid_values)
        smd = standardized_mean_difference(train_values, valid_values)
        train_std = train_stats["std"]
        valid_std = valid_stats["std"]
        if train_std == 0:
            zero_variance.append(feature)
        elif train_std is not None:
            train_stds.append(train_std)
        scale_ratio = safe_ratio(abs_max(train_stats), abs_max(valid_stats))
        std_ratio = safe_ratio(train_std, valid_std)
        if scale_ratio is not None and (scale_ratio >= 100.0 or scale_ratio <= 0.01):
            extreme_scale_ratio.append(feature)
        per_feature[feature] = {
            "train": train_stats,
            "valid": valid_stats,
            "train_vs_valid_standardized_mean_difference": smd,
            "train_valid_abs_scale_ratio": scale_ratio,
            "train_valid_std_ratio": std_ratio,
            "zero_variance_train": train_std == 0,
            "missing_or_inf_count": count_missing_or_inf(train_values) + count_missing_or_inf(valid_values),
        }
    cross_feature_std_ratio = None
    if train_stds:
        min_std = min(train_stds)
        max_std = max(train_stds)
        if min_std > 0:
            cross_feature_std_ratio = max_std / min_std
    feature_scale_risk = bool(zero_variance or extreme_scale_ratio or (cross_feature_std_ratio is not None and cross_feature_std_ratio >= 1_000_000.0))
    return {
        "per_feature": per_feature,
        "zero_variance_features": zero_variance,
        "extreme_scale_ratio_features": extreme_scale_ratio,
        "cross_feature_train_std_ratio": cross_feature_std_ratio,
        "feature_scale_risk_observed": feature_scale_risk,
        "diagnostic_only": True,
    }


def build_univariate_signal_diagnostic(split: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    x_train = split["x_train"]
    y_train = split["y_train"]
    per_feature: dict[str, Any] = {}
    abs_smd_values: list[float] = []
    for index, feature in enumerate(feature_columns):
        values = [row[index] for row in x_train]
        class_0 = [value for value, label in zip(values, y_train) if label == 0]
        class_1 = [value for value, label in zip(values, y_train) if label == 1]
        abs_smd = abs(standardized_mean_difference(class_0, class_1))
        if math.isfinite(abs_smd):
            abs_smd_values.append(abs_smd)
        per_feature[feature] = {
            "class_0_mean": safe_mean(class_0),
            "class_1_mean": safe_mean(class_1),
            "absolute_standardized_difference": abs_smd,
            "univariate_roc_auc": safe_auc(y_train, values),
            "diagnostic_only": True,
        }
    max_abs_smd = max(abs_smd_values) if abs_smd_values else None
    weak_signal = max_abs_smd is None or max_abs_smd < 0.20
    return {
        "per_feature": per_feature,
        "max_absolute_standardized_difference": max_abs_smd,
        "weak_univariate_signal_observed": weak_signal,
        "diagnostic_only": True,
        "metrics_are_effectiveness_evidence": False,
    }


def run_logistic_probability_diagnostic(split: dict[str, Any]) -> dict[str, Any]:
    x_train = np.asarray(split["x_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=int)
    x_valid = np.asarray(split["x_valid"], dtype=float)
    y_valid = np.asarray(split["y_valid"], dtype=int)
    model = LogisticRegression(max_iter=200, solver="liblinear", random_state=42)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_valid)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    distribution = distribution_from_values([int(value) for value in predictions])
    collapsed = distribution["0"] == 0 or distribution["1"] == 0
    return {
        "status": "completed",
        "model_scope": "cpu_sklearn_no_save_diagnostic_replay",
        "probability_summary": probability_summary(probabilities.tolist()),
        "fraction_above_0_5": float(np.mean(probabilities >= 0.5)),
        "prediction_distribution_at_threshold_0_5": distribution,
        "all_probability_above_0_5": bool(np.all(probabilities >= 0.5)),
        "all_probability_below_0_5": bool(np.all(probabilities < 0.5)),
        "threshold_collapse_observed": collapsed,
        "roc_auc": safe_auc(y_valid.tolist(), probabilities.tolist()),
        "pr_auc": safe_pr_auc(y_valid.tolist(), probabilities.tolist()),
        "model_saved": False,
        "diagnostic_only": True,
        "metrics_are_effectiveness_evidence": False,
    }


def run_balanced_scaled_probe(split: dict[str, Any]) -> dict[str, Any]:
    x_train = np.asarray(split["x_train"], dtype=float)
    y_train = np.asarray(split["y_train"], dtype=int)
    x_valid = np.asarray(split["x_valid"], dtype=float)
    y_valid = np.asarray(split["y_valid"], dtype=int)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_valid_scaled = scaler.transform(x_valid)
    model = LogisticRegression(max_iter=500, solver="liblinear", class_weight="balanced", random_state=42)
    model.fit(x_train_scaled, y_train)
    probabilities = model.predict_proba(x_valid_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    distribution = distribution_from_values([int(value) for value in predictions])
    collapsed = distribution["0"] == 0 or distribution["1"] == 0
    return {
        "status": "completed",
        "probe_scope": "standard_scaler_fit_train_only_plus_balanced_logistic_no_save_diagnostic",
        "prediction_distribution": distribution,
        "threshold_collapse_observed": collapsed,
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
        "roc_auc": safe_auc(y_valid.tolist(), probabilities.tolist()),
        "pr_auc": safe_pr_auc(y_valid.tolist(), probabilities.tolist()),
        "probability_summary": probability_summary(probabilities.tolist()),
        "model_saved": False,
        "diagnostic_only": True,
        "metrics_are_effectiveness_evidence": False,
    }


def build_diagnostic_flags(
    label_distribution_payload: dict[str, Any],
    sample_granularity: dict[str, Any],
    feature_scale_diagnostic: dict[str, Any],
    univariate_signal_diagnostic: dict[str, Any],
    logistic_probability_diagnostic: dict[str, Any],
    balanced_scaled_probe: dict[str, Any],
) -> list[str]:
    flags: list[str] = []
    if label_distribution_payload["train_valid_label_shift"]["observed"]:
        flags.append("TRAIN_VALID_LABEL_SHIFT_OBSERVED")
    if feature_scale_diagnostic["feature_scale_risk_observed"]:
        flags.append("FEATURE_SCALE_RISK_OBSERVED")
    if univariate_signal_diagnostic["weak_univariate_signal_observed"]:
        flags.append("WEAK_UNIVARIATE_SIGNAL_OBSERVED")
    if sample_granularity["repeated_label_group_structure_observed"]:
        flags.append("GROUP_REPEATED_LABEL_STRUCTURE_OBSERVED")
    logistic_collapsed = bool(logistic_probability_diagnostic.get("threshold_collapse_observed"))
    balanced_collapsed = bool(balanced_scaled_probe.get("threshold_collapse_observed"))
    if logistic_collapsed:
        flags.append("LOGISTIC_THRESHOLD_COLLAPSE_OBSERVED")
        flags.append(
            "BALANCED_SCALED_PROBE_STILL_COLLAPSED"
            if balanced_collapsed
            else "BALANCED_SCALED_PROBE_REDUCES_COLLAPSE"
        )
    flags.append("NO_FORMAL_MODEL_EVIDENCE")
    return flags


def choose_ready_decision(flags: Sequence[str]) -> str:
    if "GROUP_REPEATED_LABEL_STRUCTURE_OBSERVED" in flags:
        return READY_GROUP_LEVEL_SAMPLE
    if "BALANCED_SCALED_PROBE_REDUCES_COLLAPSE" in flags:
        return READY_BALANCED_PROBE
    if "FEATURE_SCALE_RISK_OBSERVED" in flags or "WEAK_UNIVARIATE_SIGNAL_OBSERVED" in flags:
        return READY_PAST_ONLY_FEATURES
    return READY_FEATURE_LABEL_REVIEW


def decide_blocked(
    p0_blockers: Sequence[str],
    boundary_check: dict[str, Any],
    manifest_ok: bool,
    feature_check: dict[str, Any],
    explicit_feature_check: dict[str, Any],
) -> str | None:
    if not p0_blockers:
        return None
    if not boundary_check["passed"]:
        return BLOCKED_BOUNDARY_FLAG
    if not manifest_ok or not feature_check["passed"] or not explicit_feature_check["passed"]:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    return BLOCKED_MODEL_ARTIFACT_CREATED


def rows_to_float_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for row in rows:
        values: list[float] = []
        for column in feature_columns:
            raw_value = row.get(column, "")
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                value = math.nan
            values.append(value)
        matrix.append(values)
    return matrix


def rows_to_labels(rows: list[dict[str, str]]) -> list[int]:
    return [label_from_row(row) for row in rows]


def label_from_row(row: dict[str, str]) -> int:
    return int(float(row[TARGET_COLUMN]))


def label_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return distribution_from_values(rows_to_labels(rows))


def grouped_label_distribution(rows: list[dict[str, str]], column: str) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(column, "")).strip(), []).append(row)
    return {
        key: {**label_distribution(group_rows), "row_count": len(group_rows), "positive_rate": positive_rate(label_distribution(group_rows))}
        for key, group_rows in sorted(grouped.items())
    }


def distribution_from_values(values: list[int]) -> dict[str, int]:
    return {"0": sum(1 for value in values if value == 0), "1": sum(1 for value in values if value == 1)}


def positive_rate(distribution: dict[str, int]) -> float:
    total = distribution["0"] + distribution["1"]
    return float(distribution["1"] / total) if total else 0.0


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


def numeric_stats(values: Sequence[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": len(values), "finite_count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(values),
        "finite_count": len(finite),
        "mean": mean(finite),
        "std": pstdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def standardized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float:
    left_finite = [float(value) for value in left if math.isfinite(float(value))]
    right_finite = [float(value) for value in right if math.isfinite(float(value))]
    if not left_finite or not right_finite:
        return math.nan
    left_std = pstdev(left_finite) if len(left_finite) > 1 else 0.0
    right_std = pstdev(right_finite) if len(right_finite) > 1 else 0.0
    pooled = math.sqrt((left_std**2 + right_std**2) / 2.0)
    if pooled == 0:
        return 0.0 if mean(left_finite) == mean(right_finite) else math.inf
    return float((mean(left_finite) - mean(right_finite)) / pooled)


def count_missing_or_inf(values: Sequence[float]) -> int:
    return sum(1 for value in values if not math.isfinite(float(value)))


def safe_ratio(left: float | None, right: float | None) -> float | None:
    if left is None or right is None or not math.isfinite(left) or not math.isfinite(right) or right == 0:
        return None
    return float(left / right)


def abs_max(stats: dict[str, Any]) -> float | None:
    min_value = stats.get("min")
    max_value = stats.get("max")
    if min_value is None or max_value is None:
        return None
    return max(abs(float(min_value)), abs(float(max_value)))


def safe_mean(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(mean(finite)) if finite else None


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(set(int(label) for label in labels)) < 2:
        return None
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return None


def safe_pr_auc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
    if len(set(int(label) for label in labels)) < 2:
        return None
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return None


def probability_summary(probabilities: Sequence[float]) -> dict[str, float | None]:
    if not probabilities:
        return {"min": None, "p05": None, "p25": None, "median": None, "p75": None, "p95": None, "max": None}
    values = np.asarray(probabilities, dtype=float)
    return {
        "min": float(np.min(values)),
        "p05": float(np.percentile(values, 5)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intraday_majority_class_collapse_diagnostic_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "diagnostic_decision": report["diagnostic_decision"],
        "status": report["status"],
        "diagnostic_flags": report["diagnostic_flags"],
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "not_trading_advice": True,
        "p0_blockers": report["p0_blockers"],
    }
    (out_dir / "diagnostic_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "",
        "# Intraday Majority-Class Collapse Diagnostic",
        "",
        "本报告只用于 Lab feature / label / split / probability diagnostic。balanced/scaled probe 不是正式训练，不保存模型，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。",
        "",
        f"- report_type: {report['report_type']}",
        f"- status: {report['status']}",
        f"- diagnostic_decision: {report['diagnostic_decision']}",
        f"- target: {report['target']}",
        f"- feature_count: {len(report['feature_columns'])}",
        f"- train_label_distribution: {json.dumps(report['label_distribution']['train'], ensure_ascii=False, sort_keys=True)}",
        f"- valid_label_distribution: {json.dumps(report['label_distribution']['valid'], ensure_ascii=False, sort_keys=True)}",
        f"- train_valid_positive_rate_delta: {report['train_valid_label_shift']['absolute_positive_rate_delta']}",
        f"- row_count: {report['sample_granularity']['row_count']}",
        f"- anchor_count: {report['sample_granularity']['anchor_count']}",
        f"- etf_count: {report['sample_granularity']['etf_count']}",
        f"- repeated_label_group_structure_observed: {str(report['sample_granularity']['repeated_label_group_structure_observed']).lower()}",
        f"- feature_scale_risk_observed: {str(report['feature_scale_diagnostic']['feature_scale_risk_observed']).lower()}",
        f"- weak_univariate_signal_observed: {str(report['univariate_signal_diagnostic']['weak_univariate_signal_observed']).lower()}",
        f"- logistic_threshold_collapse_observed: {str(report['logistic_probability_diagnostic'].get('threshold_collapse_observed')).lower()}",
        f"- balanced_scaled_probe_threshold_collapse_observed: {str(report['balanced_scaled_probe'].get('threshold_collapse_observed')).lower()}",
        f"- diagnostic_flags: {', '.join(report['diagnostic_flags'])}",
        f"- model_saved: {str(report['model_saved']).lower()}",
        f"- checkpoint_saved: {str(report['checkpoint_saved']).lower()}",
        f"- gpu_used: {str(report['gpu_used']).lower()}",
        f"- qmt_used: {str(report['qmt_used']).lower()}",
        f"- order_intent_generated: {str(report['order_intent_generated']).lower()}",
        f"- stable_affected: {str(report['stable_affected']).lower()}",
        f"- metrics_are_effectiveness_evidence: {str(report['metrics_are_effectiveness_evidence']).lower()}",
        f"- not_trading_advice: {str(report['not_trading_advice']).lower()}",
        "",
        "Allowed next steps: feature/label review, past-only feature expansion design, group-level sample design, or an explicitly approved no-save diagnostic smoke request.",
    ]
    (out_dir / "intraday_majority_class_collapse_diagnostic_report.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday majority-class collapse diagnostic.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--repeatability", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_diagnostic(
            args.samples,
            args.manifest,
            args.readiness,
            args.repeatability,
            args.out_dir,
        )
    except MajorityCollapseDiagnosticError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "diagnostic_decision": BLOCKED_DIAGNOSTIC_RUNTIME_ERROR,
                    "p0_blockers": [str(exc)],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "diagnostic_decision": report["diagnostic_decision"],
                "diagnostic_flags": report["diagnostic_flags"],
                "model_saved": report["model_saved"],
                "checkpoint_saved": report["checkpoint_saved"],
                "metrics_are_effectiveness_evidence": report["metrics_are_effectiveness_evidence"],
                "automatic_promotion_ready": report["automatic_promotion_ready"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
