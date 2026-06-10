from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts, load_json


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_transform_aware_no_save_smoke")
REPORT_TYPE = "intraday_group_level_transform_aware_no_save_smoke"
SMOKE_SCOPE = "lab_only_transform_aware_no_save_diagnostic"
READY = "TRANSFORM_AWARE_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_REVIEW_REQUIRED"
BLOCKED_MANIFEST_LEAKAGE_P0 = "BLOCKED_MANIFEST_LEAKAGE_P0"
BLOCKED_BOUNDARY_FLAG = "BLOCKED_BOUNDARY_FLAG"
BLOCKED_MODEL_OR_SCALER_ARTIFACT_CREATED = "BLOCKED_MODEL_OR_SCALER_ARTIFACT_CREATED"
BLOCKED_TRANSFORM_RUNTIME_ERROR = "BLOCKED_TRANSFORM_RUNTIME_ERROR"
BLOCKED_SMOKE_RUNTIME_ERROR = "BLOCKED_SMOKE_RUNTIME_ERROR"
TARGET_COLUMN = "three_day_positive_label"
GROUP_LABEL_POLICY = "anchor_close_last_bar"
EXPECTED_SPLIT_POLICY = "anchor_date_70_30"
ANCHOR_COLUMN_CANDIDATES = ("anchor_date", "trade_date")
P1_INCONSISTENCY = "P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED"
P1_SCALE = "P1_EXTREME_FEATURE_SCALE_REVIEW_REQUIRED"
P1_SHIFT = "P1_TRAIN_VALID_FEATURE_SHIFT_REVIEW_REQUIRED"
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
POLICY_FALSE_FIELDS = [
    "save_scaler",
    "model_training_allowed",
    "stable_allowed",
    "qmt_allowed",
    "order_intent_allowed",
    "automatic_promotion_ready",
]
MODEL_NAMES = [
    "dummy_most_frequent",
    "dummy_stratified",
    "logistic_regression_raw",
    "logistic_regression_balanced_scaled",
    "logistic_regression_log1p_scaled_balanced",
]


class TransformAwareSmokeError(RuntimeError):
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
        raise TransformAwareSmokeError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise TransformAwareSmokeError(f"samples CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise TransformAwareSmokeError(f"samples CSV has no header: {path}")
    return rows, columns


def run_smoke(
    samples_path: Path,
    manifest_path: Path,
    readiness_path: Path,
    transform_policy_path: Path,
    baseline_smoke_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(samples_path, repo_root)
    resolved_manifest = resolve_repo_path(manifest_path, repo_root)
    resolved_readiness = resolve_repo_path(readiness_path, repo_root)
    resolved_transform_policy = resolve_repo_path(transform_policy_path, repo_root)
    resolved_baseline_smoke = resolve_repo_path(baseline_smoke_path, repo_root)
    for path, label in (
        (resolved_samples, "samples"),
        (resolved_manifest, "manifest"),
        (resolved_readiness, "readiness"),
        (resolved_transform_policy, "transform-policy"),
        (resolved_baseline_smoke, "baseline-smoke"),
    ):
        if not path.exists():
            raise TransformAwareSmokeError(f"{label} path does not exist: {path}")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    rows, columns = load_csv_rows(resolved_samples)
    manifest = load_json(resolved_manifest)
    readiness = load_json(resolved_readiness)
    transform_policy = load_json(resolved_transform_policy)
    baseline_smoke = load_json(resolved_baseline_smoke)

    manifest_check = check_label_manifest(resolved_manifest)
    group_contract_check = run_group_contract_check(manifest)
    feature_check = run_feature_check(manifest, columns)
    readiness_check = run_readiness_check(readiness)
    boundary_check = run_boundary_check(manifest, readiness, transform_policy)
    transform_policy_check = run_transform_policy_check(transform_policy)
    artifact_check_before = check_model_artifacts(resolved_out_dir)

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    p0_blockers.extend(group_contract_check["p0_blockers"])
    p0_blockers.extend(feature_check["p0_blockers"])
    p0_blockers.extend(readiness_check["p0_blockers"])
    p1_warnings.extend(readiness_check["p1_warnings"])
    p0_blockers.extend(boundary_check["p0_blockers"])
    p0_blockers.extend(transform_policy_check["p0_blockers"])
    p0_blockers.extend(artifact_check_before["p0_blockers"])

    split_payload = build_split_payload(rows, readiness, feature_check["feature_columns"]) if not p0_blockers else empty_split_payload()
    p0_blockers.extend(split_payload["p0_blockers"])

    metrics: dict[str, Any] = {}
    prediction_distribution_by_model: dict[str, dict[str, int]] = {}
    models_run: list[str] = []
    transform_policy_applied = empty_transform_application()
    collapse_check = empty_collapse_check()
    comparison_to_baseline = compare_to_baseline({}, baseline_smoke, {})
    scaler_audit: dict[str, Any] = empty_scaler_audit()
    decision = decide_pre_runtime(
        manifest_ok=manifest_check.ok and group_contract_check["passed"] and feature_check["passed"],
        readiness_ok=readiness_check["passed"],
        boundary_ok=boundary_check["passed"] and transform_policy_check["passed"],
        split_ok=split_payload["passed"],
        artifact_ok=artifact_check_before["passed"],
    )

    if decision is None:
        try:
            (
                metrics,
                prediction_distribution_by_model,
                models_run,
                transform_policy_applied,
                collapse_check,
                scaler_audit,
            ) = run_models(split_payload, transform_policy, feature_check["feature_columns"])
            comparison_to_baseline = compare_to_baseline(metrics, baseline_smoke, collapse_check)
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            if artifact_check_after["passed"]:
                decision = READY
            else:
                decision = BLOCKED_MODEL_OR_SCALER_ARTIFACT_CREATED
        except TransformAwareSmokeError as exc:
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.append(f"transform runtime error: {exc}")
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            decision = BLOCKED_TRANSFORM_RUNTIME_ERROR
        except Exception as exc:  # noqa: BLE001 - smoke must report runtime blockers.
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.append(f"smoke runtime error: {exc}")
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            decision = BLOCKED_SMOKE_RUNTIME_ERROR
    else:
        artifact_check_after = artifact_check_before

    if artifact_check_after["p0_blockers"]:
        decision = BLOCKED_MODEL_OR_SCALER_ARTIFACT_CREATED

    diagnostic_flags = build_diagnostic_flags(collapse_check, decision)
    p1_warnings.extend([P1_INCONSISTENCY, P1_SCALE, P1_SHIFT])
    status = "blocked" if decision.startswith("BLOCKED_") else "passed"
    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": REPORT_TYPE,
        "smoke_scope": SMOKE_SCOPE,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "readiness_path": str(readiness_path),
        "transform_policy_path": str(transform_policy_path),
        "baseline_smoke_path": str(baseline_smoke_path),
        "input_readiness_decision": readiness.get("readiness_decision"),
        "readiness_decision": decision,
        "target": TARGET_COLUMN,
        "feature_columns": feature_check["feature_columns"],
        "feature_count": len(feature_check["feature_columns"]),
        "transform_policy_applied": transform_policy_applied,
        "log1p_features_applied": transform_policy_applied["log1p_features_applied"],
        "log1p_features_skipped": transform_policy_applied["log1p_features_skipped"],
        "standard_scaler_fit_scope": "train_only",
        "scaler_saved": False,
        "models_run": models_run,
        "train_anchor_dates": split_payload["train_anchor_dates"],
        "valid_anchor_dates": split_payload["valid_anchor_dates"],
        "train_group_count": split_payload["train_group_count"],
        "valid_group_count": split_payload["valid_group_count"],
        "train_label_distribution": split_payload["train_label_distribution"],
        "valid_label_distribution": split_payload["valid_label_distribution"],
        "metrics": metrics,
        "prediction_distribution_by_model": prediction_distribution_by_model,
        "collapse_check": collapse_check,
        "comparison_to_baseline_group_smoke": comparison_to_baseline,
        "scaler_audit": scaler_audit,
        "diagnostic_flags": diagnostic_flags,
        "p0_blockers": dedupe(p0_blockers),
        "p1_warnings": dedupe(p1_warnings),
        "manifest_leakage_check": manifest_check.to_summary(),
        "group_contract_check": group_contract_check,
        "feature_check": feature_check,
        "readiness_check": readiness_check,
        "boundary_check": boundary_check,
        "transform_policy_check": transform_policy_check,
        "split_check": split_payload["split_check"],
        "artifact_check_before": artifact_check_before,
        "artifact_check_after": artifact_check_after,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "metrics_are_effectiveness_evidence": False,
        "automatic_promotion_ready": False,
        "not_trading_advice": True,
        "formal_training": False,
        "hyperparameter_tuning": False,
    }
    write_reports(report, resolved_out_dir)
    return report


def run_group_contract_check(manifest: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    if manifest.get("group_level_sample") is not True:
        p0_blockers.append("group_level_sample must be true")
    if string_list(manifest.get("group_key")) != ["trade_date", "etf_code"]:
        p0_blockers.append("group_key must be ['trade_date', 'etf_code']")
    if manifest.get("group_label_policy") != GROUP_LABEL_POLICY:
        p0_blockers.append(f"group_label_policy must be {GROUP_LABEL_POLICY}")
    if manifest.get("intraday_live_decision_ready") is not False:
        p0_blockers.append("intraday_live_decision_ready must be false")
    return {"passed": not p0_blockers, "p0_blockers": p0_blockers, "p1_warnings": []}


def run_feature_check(manifest: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    feature_columns = string_list(manifest.get("feature_columns"))
    feature_set = set(feature_columns)
    label_columns = set(string_list(manifest.get("label_columns")))
    outcome_columns = set(string_list(manifest.get("outcome_columns")))
    if TARGET_COLUMN not in columns:
        p0_blockers.append(f"{TARGET_COLUMN} missing from samples")
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


def run_readiness_check(readiness: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    p1_warnings = [warning for warning in string_list(readiness.get("p1_warnings")) if warning in {P1_INCONSISTENCY, P1_SCALE, P1_SHIFT}]
    if readiness.get("status") not in (None, "passed"):
        p0_blockers.append("readiness.status must be passed when present")
    decision = str(readiness.get("readiness_decision", ""))
    if not (decision.startswith("GROUP_LEVEL_") and ("PASSED" in decision or "COMPLETED" in decision)):
        p0_blockers.append("readiness_decision must be group-level passed or completed")
    if readiness.get("selected_split_policy") not in (None, EXPECTED_SPLIT_POLICY):
        p0_blockers.append(f"readiness.selected_split_policy must be {EXPECTED_SPLIT_POLICY}")
    if int(readiness.get("train_group_count") or 0) <= 0 or int(readiness.get("valid_group_count") or 0) <= 0:
        p0_blockers.append("readiness train_group_count and valid_group_count must be positive")
    return {
        "passed": not p0_blockers,
        "expected_split_policy": EXPECTED_SPLIT_POLICY,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }


def run_boundary_check(manifest: dict[str, Any], readiness: dict[str, Any], transform_policy: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    for source_name, payload in (("manifest", manifest), ("readiness", readiness)):
        for field_name in BOUNDARY_FALSE_FIELDS:
            if field_name in payload and payload.get(field_name) is not False:
                p0_blockers.append(f"{source_name}.{field_name} must be false")
    for field_name in POLICY_FALSE_FIELDS:
        if field_name in transform_policy and transform_policy.get(field_name) is not False:
            p0_blockers.append(f"transform_policy.{field_name} must be false")
    if transform_policy.get("train_only_fit_required") is not True:
        p0_blockers.append("transform_policy.train_only_fit_required must be true")
    if transform_policy.get("policy_scope") != "diagnostic_only":
        p0_blockers.append("transform_policy.policy_scope must be diagnostic_only")
    return {
        "passed": not p0_blockers,
        "checked_fields": BOUNDARY_FALSE_FIELDS,
        "policy_checked_fields": POLICY_FALSE_FIELDS,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_transform_policy_check(transform_policy: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    transforms = transform_policy.get("recommended_transforms")
    if not isinstance(transforms, dict):
        p0_blockers.append("transform_policy.recommended_transforms must be an object")
        transforms = {}
    if not isinstance(transforms.get("log1p_recommended", []), list):
        p0_blockers.append("recommended_transforms.log1p_recommended must be an array")
    if not isinstance(transforms.get("standardize_recommended", []), list):
        p0_blockers.append("recommended_transforms.standardize_recommended must be an array")
    return {
        "passed": not p0_blockers,
        "log1p_recommended": string_list(transforms.get("log1p_recommended", [])),
        "standardize_recommended": string_list(transforms.get("standardize_recommended", [])),
        "clip_winsorize_review_only": string_list(transforms.get("clip_winsorize_review", [])),
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def build_split_payload(rows: list[dict[str, str]], readiness: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
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
    try:
        x_train = rows_to_matrix(train_rows, feature_columns)
        y_train = rows_to_labels(train_rows)
        x_valid = rows_to_matrix(valid_rows, feature_columns)
        y_valid = rows_to_labels(valid_rows)
    except ValueError as exc:
        p0_blockers.append(f"numeric conversion failed: {exc}")
        x_train, y_train, x_valid, y_valid = [], [], [], []
    return {
        "passed": not p0_blockers,
        "anchor_column": anchor_column,
        "train_anchor_dates": train_anchor_dates,
        "valid_anchor_dates": valid_anchor_dates,
        "train_group_count": len(train_rows),
        "valid_group_count": len(valid_rows),
        "train_label_distribution": distribution_from_values(y_train),
        "valid_label_distribution": distribution_from_values(y_valid),
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
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
        "passed": False,
        "anchor_column": "",
        "train_anchor_dates": [],
        "valid_anchor_dates": [],
        "train_group_count": 0,
        "valid_group_count": 0,
        "train_label_distribution": {"0": 0, "1": 0},
        "valid_label_distribution": {"0": 0, "1": 0},
        "x_train": [],
        "y_train": [],
        "x_valid": [],
        "y_valid": [],
        "train_rows": [],
        "valid_rows": [],
        "split_check": {"p0_blockers": []},
        "p0_blockers": [],
    }


def run_models(
    split_payload: dict[str, Any],
    transform_policy: dict[str, Any],
    feature_columns: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, int]], list[str], dict[str, Any], dict[str, Any], dict[str, Any]]:
    x_train = split_payload["x_train"]
    y_train = split_payload["y_train"]
    x_valid = split_payload["x_valid"]
    y_valid = split_payload["y_valid"]
    if not x_train or not x_valid:
        raise TransformAwareSmokeError("train and valid splits must both be non-empty")

    log1p_application = build_log1p_application(transform_policy, feature_columns, split_payload["train_rows"], split_payload["valid_rows"])
    scaled_raw = train_only_standardize(x_train, x_valid)
    x_train_log1p, x_valid_log1p = apply_log1p_by_index(
        x_train,
        x_valid,
        [feature_columns.index(feature) for feature in log1p_application["log1p_features_applied"]],
    )
    scaled_log1p = train_only_standardize(x_train_log1p, x_valid_log1p)

    models: dict[str, tuple[Any, list[list[float]], list[list[float]]]] = {
        "dummy_most_frequent": (DummyClassifier(strategy="most_frequent"), x_train, x_valid),
        "dummy_stratified": (DummyClassifier(strategy="stratified", random_state=42), x_train, x_valid),
        "logistic_regression_raw": (
            LogisticRegression(max_iter=200, solver="liblinear", random_state=42),
            x_train,
            x_valid,
        ),
        "logistic_regression_balanced_scaled": (
            LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced", random_state=42),
            scaled_raw["x_train"],
            scaled_raw["x_valid"],
        ),
        "logistic_regression_log1p_scaled_balanced": (
            LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced", random_state=42),
            scaled_log1p["x_train"],
            scaled_log1p["x_valid"],
        ),
    }
    metrics: dict[str, Any] = {}
    predictions_by_model: dict[str, list[int]] = {}
    prediction_distribution_by_model: dict[str, dict[str, int]] = {}
    models_run: list[str] = []
    for model_name in MODEL_NAMES:
        model, train_matrix, valid_matrix = models[model_name]
        model.fit(train_matrix, y_train)
        predictions = [int(value) for value in model.predict(valid_matrix)]
        predictions_by_model[model_name] = predictions
        prediction_distribution_by_model[model_name] = distribution_from_values(predictions)
        metrics[model_name] = score_predictions(y_valid, predictions)
        models_run.append(model_name)

    transform_policy_applied = {
        "policy_scope": "diagnostic_only",
        "train_only_fit_required": True,
        "save_scaler": False,
        "clip_winsorization_applied": False,
        "clip_winsorization_review_only": string_list(
            transform_policy.get("recommended_transforms", {}).get("clip_winsorize_review", [])
            if isinstance(transform_policy.get("recommended_transforms"), dict)
            else []
        ),
        **log1p_application,
    }
    scaler_audit = {
        "standard_scaler_fit_scope": "train_only",
        "scaler_saved": False,
        "raw_scaled": scaled_raw["audit"],
        "log1p_scaled": scaled_log1p["audit"],
    }
    return (
        metrics,
        prediction_distribution_by_model,
        models_run,
        transform_policy_applied,
        build_collapse_check(predictions_by_model, metrics),
        scaler_audit,
    )


def build_log1p_application(
    transform_policy: dict[str, Any],
    feature_columns: list[str],
    train_rows: list[dict[str, str]],
    valid_rows: list[dict[str, str]],
) -> dict[str, Any]:
    transforms = transform_policy.get("recommended_transforms") if isinstance(transform_policy.get("recommended_transforms"), dict) else {}
    policy_features = [feature for feature in string_list(transforms.get("log1p_recommended", [])) if feature in feature_columns]
    applied: list[str] = []
    skipped: dict[str, str] = {}
    for feature in policy_features:
        if not is_raw_flow_feature(feature):
            skipped[feature] = "not_amount_volume_raw_flow_feature"
            continue
        values = [parse_float(row.get(feature)) for row in [*train_rows, *valid_rows]]
        finite_values = [value for value in values if value is not None and math.isfinite(value)]
        if len(finite_values) != len(values):
            skipped[feature] = "missing_or_non_finite_value"
            continue
        if any(value < 0 for value in finite_values):
            skipped[feature] = "negative_value_present"
            continue
        applied.append(feature)
    return {
        "log1p_policy_features": policy_features,
        "log1p_features_applied": sorted(applied),
        "log1p_features_skipped": skipped,
        "log1p_nonnegative_check_scope": "train_and_valid",
    }


def is_raw_flow_feature(feature: str) -> bool:
    name = feature.lower()
    return ("volume" in name or "amount" in name) and not any(token in name for token in ("ratio", "rank", "relative"))


def apply_log1p_by_index(
    x_train: list[list[float]],
    x_valid: list[list[float]],
    indices: list[int],
) -> tuple[list[list[float]], list[list[float]]]:
    index_set = set(indices)

    def transform_matrix(matrix: list[list[float]]) -> list[list[float]]:
        transformed: list[list[float]] = []
        for row in matrix:
            transformed.append([math.log1p(value) if index in index_set else value for index, value in enumerate(row)])
        return transformed

    return transform_matrix(x_train), transform_matrix(x_valid)


def train_only_standardize(x_train: list[list[float]], x_valid: list[list[float]]) -> dict[str, Any]:
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(x_train)
    valid_scaled = scaler.transform(x_valid)
    train_list = train_scaled.tolist()
    valid_list = valid_scaled.tolist()
    return {
        "x_train": train_list,
        "x_valid": valid_list,
        "audit": {
            "fit_scope": "train_only",
            "fit_row_count": len(x_train),
            "transform_train_row_count": len(x_train),
            "transform_valid_row_count": len(x_valid),
            "fit_feature_count": len(x_train[0]) if x_train else 0,
            "valid_fit_performed": False,
            "train_scaled_abs_mean_max": max_abs_column_mean(train_list),
            "valid_scaled_abs_mean_max": max_abs_column_mean(valid_list),
        },
    }


def max_abs_column_mean(matrix: list[list[float]]) -> float | None:
    if not matrix or not matrix[0]:
        return None
    width = len(matrix[0])
    means = [sum(row[index] for row in matrix) / len(matrix) for index in range(width)]
    return max(abs(value) for value in means)


def score_predictions(y_valid: list[int], predictions: list[int]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "prediction_distribution": distribution_from_values(predictions),
    }


def build_collapse_check(predictions_by_model: dict[str, list[int]], metrics: dict[str, Any]) -> dict[str, Any]:
    most_frequent = predictions_by_model.get("dummy_most_frequent", [])
    raw = predictions_by_model.get("logistic_regression_raw", [])
    balanced = predictions_by_model.get("logistic_regression_balanced_scaled", [])
    log1p = predictions_by_model.get("logistic_regression_log1p_scaled_balanced", [])
    raw_matches_dummy = raw == most_frequent and bool(raw)
    balanced_matches_dummy = balanced == most_frequent and bool(balanced)
    log1p_matches_dummy = log1p == most_frequent and bool(log1p)
    flags: list[str] = []
    if raw_matches_dummy:
        flags.append("RAW_LOGISTIC_COLLAPSE_REPRODUCED")
    if not balanced_matches_dummy and balanced:
        flags.append("BALANCED_SCALED_REDUCES_COLLAPSE")
    if not log1p_matches_dummy and log1p:
        flags.append("LOG1P_SCALED_BALANCED_REDUCES_COLLAPSE")
    flags.extend(["NO_FORMAL_MODEL_EVIDENCE", "TRANSFORM_AWARE_SMOKE_REVIEW_REQUIRED"])
    dummy_balanced_accuracy = metrics.get("dummy_most_frequent", {}).get("balanced_accuracy")
    return {
        "dummy_most_frequent_prediction_distribution": distribution_from_values(most_frequent),
        "raw_logistic_prediction_distribution": distribution_from_values(raw),
        "balanced_scaled_prediction_distribution": distribution_from_values(balanced),
        "log1p_scaled_balanced_prediction_distribution": distribution_from_values(log1p),
        "raw_logistic_matches_dummy_most_frequent": raw_matches_dummy,
        "balanced_scaled_reduces_collapse": not balanced_matches_dummy and bool(balanced),
        "log1p_scaled_balanced_reduces_collapse": not log1p_matches_dummy and bool(log1p),
        "balanced_scaled_balanced_accuracy": metrics.get("logistic_regression_balanced_scaled", {}).get("balanced_accuracy"),
        "log1p_scaled_balanced_balanced_accuracy": metrics.get("logistic_regression_log1p_scaled_balanced", {}).get("balanced_accuracy"),
        "dummy_most_frequent_balanced_accuracy": dummy_balanced_accuracy,
        "below_or_near_dummy_baselines": {
            "balanced_scaled": near_or_below_dummy(metrics.get("logistic_regression_balanced_scaled", {}).get("balanced_accuracy"), dummy_balanced_accuracy),
            "log1p_scaled_balanced": near_or_below_dummy(
                metrics.get("logistic_regression_log1p_scaled_balanced", {}).get("balanced_accuracy"),
                dummy_balanced_accuracy,
            ),
        },
        "flags": flags,
    }


def near_or_below_dummy(value: Any, dummy_value: Any, tolerance: float = 0.03) -> bool | None:
    if value is None or dummy_value is None:
        return None
    return float(value) <= float(dummy_value) + tolerance


def compare_to_baseline(metrics: dict[str, Any], baseline_smoke: dict[str, Any], collapse_check: dict[str, Any]) -> dict[str, Any]:
    baseline_metrics = baseline_smoke.get("metrics", {}) if isinstance(baseline_smoke.get("metrics"), dict) else {}
    baseline_collapse = baseline_smoke.get("majority_class_collapse_check", {})
    return {
        "baseline_report_type": baseline_smoke.get("report_type"),
        "baseline_readiness_decision": baseline_smoke.get("readiness_decision"),
        "baseline_raw_logistic_prediction_distribution": baseline_metrics.get("logistic_regression", {}).get("prediction_distribution"),
        "baseline_logistic_matches_dummy_most_frequent": baseline_collapse.get("logistic_matches_dummy_most_frequent"),
        "baseline_balanced_scaled_balanced_accuracy": baseline_metrics.get("logistic_regression_balanced_scaled", {}).get("balanced_accuracy"),
        "current_raw_logistic_prediction_distribution": metrics.get("logistic_regression_raw", {}).get("prediction_distribution"),
        "current_balanced_scaled_balanced_accuracy": metrics.get("logistic_regression_balanced_scaled", {}).get("balanced_accuracy"),
        "current_log1p_scaled_balanced_balanced_accuracy": metrics.get("logistic_regression_log1p_scaled_balanced", {}).get("balanced_accuracy"),
        "current_flags": collapse_check.get("flags", []),
        "formal_model_evidence": False,
    }


def build_diagnostic_flags(collapse_check: dict[str, Any], decision: str) -> list[str]:
    if decision.startswith("BLOCKED_"):
        return ["NO_FORMAL_MODEL_EVIDENCE"]
    return dedupe([*collapse_check.get("flags", []), "NO_FORMAL_MODEL_EVIDENCE", "TRANSFORM_AWARE_SMOKE_REVIEW_REQUIRED"])


def empty_transform_application() -> dict[str, Any]:
    return {
        "policy_scope": "diagnostic_only",
        "train_only_fit_required": True,
        "save_scaler": False,
        "clip_winsorization_applied": False,
        "clip_winsorization_review_only": [],
        "log1p_policy_features": [],
        "log1p_features_applied": [],
        "log1p_features_skipped": {},
        "log1p_nonnegative_check_scope": "train_and_valid",
    }


def empty_collapse_check() -> dict[str, Any]:
    return {
        "dummy_most_frequent_prediction_distribution": {"0": 0, "1": 0},
        "raw_logistic_prediction_distribution": {"0": 0, "1": 0},
        "balanced_scaled_prediction_distribution": {"0": 0, "1": 0},
        "log1p_scaled_balanced_prediction_distribution": {"0": 0, "1": 0},
        "raw_logistic_matches_dummy_most_frequent": None,
        "balanced_scaled_reduces_collapse": None,
        "log1p_scaled_balanced_reduces_collapse": None,
        "below_or_near_dummy_baselines": {},
        "flags": [],
    }


def empty_scaler_audit() -> dict[str, Any]:
    return {
        "standard_scaler_fit_scope": "train_only",
        "scaler_saved": False,
        "raw_scaled": {},
        "log1p_scaled": {},
    }


def decide_pre_runtime(
    manifest_ok: bool,
    readiness_ok: bool,
    boundary_ok: bool,
    split_ok: bool,
    artifact_ok: bool,
) -> str | None:
    if not boundary_ok:
        return BLOCKED_BOUNDARY_FLAG
    if not manifest_ok or not readiness_ok or not split_ok:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    if not artifact_ok:
        return BLOCKED_MODEL_OR_SCALER_ARTIFACT_CREATED
    return None


def rows_to_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> list[list[float]]:
    return [[float(row[column]) for column in feature_columns] for row in rows]


def rows_to_labels(rows: list[dict[str, str]]) -> list[int]:
    return [int(float(row[TARGET_COLUMN])) for row in rows]


def distribution_from_values(values: list[int]) -> dict[str, int]:
    return {"0": sum(1 for value in values if value == 0), "1": sum(1 for value in values if value == 1)}


def parse_float(value: Any) -> float | None:
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "intraday_group_level_transform_aware_no_save_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "lab_declaration": report["lab_declaration"],
        "readiness_decision": report["readiness_decision"],
        "status": report["status"],
        "report_type": report["report_type"],
        "smoke_scope": report["smoke_scope"],
        "target": report["target"],
        "models_run": report["models_run"],
        "diagnostic_flags": report["diagnostic_flags"],
        "collapse_check": report["collapse_check"],
        "transform_policy_applied": report["transform_policy_applied"],
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "metrics_are_effectiveness_evidence": False,
        "automatic_promotion_ready": False,
        "not_trading_advice": True,
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
        "# Intraday Group-Level Transform-Aware No-Save Smoke",
        "",
        "本文件只用于 Lab transform-aware no-save diagnostic smoke，不是正式训练，不调参，不保存 scaler，不保存模型，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。",
        "",
        f"- report_type: {report['report_type']}",
        f"- smoke_scope: {report['smoke_scope']}",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- target: {report['target']}",
        f"- models_run: {', '.join(report['models_run']) or 'none'}",
        f"- train_group_count: {report['train_group_count']}",
        f"- valid_group_count: {report['valid_group_count']}",
        f"- train_label_distribution: {json.dumps(report['train_label_distribution'], ensure_ascii=False, sort_keys=True)}",
        f"- valid_label_distribution: {json.dumps(report['valid_label_distribution'], ensure_ascii=False, sort_keys=True)}",
        f"- log1p_features_applied: {', '.join(report['log1p_features_applied']) or 'none'}",
        f"- log1p_features_skipped: {json.dumps(report['log1p_features_skipped'], ensure_ascii=False, sort_keys=True)}",
        f"- standard_scaler_fit_scope: {report['standard_scaler_fit_scope']}",
        f"- scaler_saved: {str(report['scaler_saved']).lower()}",
        f"- prediction_distribution_by_model: {json.dumps(report['prediction_distribution_by_model'], ensure_ascii=False, sort_keys=True)}",
        f"- collapse_flags: {json.dumps(report['collapse_check']['flags'], ensure_ascii=False)}",
        f"- diagnostic_flags: {json.dumps(report['diagnostic_flags'], ensure_ascii=False)}",
        f"- p1_warnings: {json.dumps(report['p1_warnings'], ensure_ascii=False)}",
        f"- model_saved: {str(report['model_saved']).lower()}",
        f"- checkpoint_saved: {str(report['checkpoint_saved']).lower()}",
        f"- gpu_used: {str(report['gpu_used']).lower()}",
        f"- torchrun_used: {str(report['torchrun_used']).lower()}",
        f"- qmt_used: {str(report['qmt_used']).lower()}",
        f"- order_intent_generated: {str(report['order_intent_generated']).lower()}",
        f"- stable_affected: {str(report['stable_affected']).lower()}",
        f"- metrics_are_effectiveness_evidence: {str(report['metrics_are_effectiveness_evidence']).lower()}",
        f"- automatic_promotion_ready: {str(report['automatic_promotion_ready']).lower()}",
        f"- not_trading_advice: {str(report['not_trading_advice']).lower()}",
        "",
        "## Boundary",
        "",
        "- no formal training",
        "- no hyperparameter search",
        "- no scaler/model/checkpoint save",
        "- no GPU or torchrun",
        "- no QMT, no OrderIntent, no Stable",
        "- metrics remain diagnostic only and are not effectiveness evidence",
    ]
    (out_dir / "intraday_group_level_transform_aware_no_save_smoke_report.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--transform-policy", required=True, type=Path)
    parser.add_argument("--baseline-smoke", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_smoke(
            args.samples,
            args.manifest,
            args.readiness,
            args.transform_policy,
            args.baseline_smoke,
            args.out_dir,
        )
    except TransformAwareSmokeError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "readiness_decision": BLOCKED_SMOKE_RUNTIME_ERROR,
                    "p0_blockers": [str(exc)],
                    "model_saved": False,
                    "scaler_saved": False,
                    "checkpoint_saved": False,
                    "gpu_used": False,
                    "torchrun_used": False,
                    "qmt_used": False,
                    "order_intent_generated": False,
                    "stable_affected": False,
                    "metrics_are_effectiveness_evidence": False,
                    "automatic_promotion_ready": False,
                    "not_trading_advice": True,
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
                "readiness_decision": report["readiness_decision"],
                "models_run": report["models_run"],
                "diagnostic_flags": report["diagnostic_flags"],
                "collapse_check": report["collapse_check"],
                "model_saved": False,
                "scaler_saved": False,
                "checkpoint_saved": False,
                "gpu_used": False,
                "torchrun_used": False,
                "qmt_used": False,
                "order_intent_generated": False,
                "stable_affected": False,
                "metrics_are_effectiveness_evidence": False,
                "automatic_promotion_ready": False,
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
