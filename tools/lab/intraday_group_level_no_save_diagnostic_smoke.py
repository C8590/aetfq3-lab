from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts, load_json


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_no_save_diagnostic_smoke")
REPORT_TYPE = "intraday_group_level_no_save_diagnostic_smoke"
SMOKE_SCOPE = "lab_only_group_level_no_save_diagnostic"
EXPECTED_READINESS = "GROUP_LEVEL_SUPERVISED_SMOKE_READINESS_PASSED_WITH_LABEL_INCONSISTENCY_REVIEW_REQUIRED"
READY = "GROUP_LEVEL_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_REVIEW_REQUIRED"
READY_WITH_P1 = "GROUP_LEVEL_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_WITH_P1_LABEL_INCONSISTENCY_REVIEW_REQUIRED"
BLOCKED_GROUP_LEVEL_READINESS_NOT_PASSED = "BLOCKED_GROUP_LEVEL_READINESS_NOT_PASSED"
BLOCKED_MANIFEST_LEAKAGE_P0 = "BLOCKED_MANIFEST_LEAKAGE_P0"
BLOCKED_BOUNDARY_FLAG = "BLOCKED_BOUNDARY_FLAG"
BLOCKED_MODEL_ARTIFACT_CREATED = "BLOCKED_MODEL_ARTIFACT_CREATED"
BLOCKED_SMOKE_RUNTIME_ERROR = "BLOCKED_SMOKE_RUNTIME_ERROR"
P1_INCONSISTENCY = "P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED"
TARGET_COLUMN = "three_day_positive_label"
GROUP_LABEL_POLICY = "anchor_close_last_bar"
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
MODEL_NAMES = [
    "dummy_most_frequent",
    "dummy_stratified",
    "logistic_regression",
    "logistic_regression_balanced_scaled",
]


class GroupLevelNoSaveSmokeError(RuntimeError):
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
        raise GroupLevelNoSaveSmokeError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise GroupLevelNoSaveSmokeError(f"samples CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise GroupLevelNoSaveSmokeError(f"samples CSV has no header: {path}")
    return rows, columns


def run_smoke(
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
            raise GroupLevelNoSaveSmokeError(f"{label} path does not exist: {path}")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    rows, columns = load_csv_rows(resolved_samples)
    manifest = load_json(resolved_manifest)
    readiness = load_json(resolved_readiness)
    manifest_check = check_label_manifest(resolved_manifest)
    group_contract_check = run_group_contract_check(manifest)
    readiness_check = run_readiness_check(readiness)
    boundary_check = run_boundary_check(manifest, readiness)
    feature_check = run_feature_check(manifest, columns)
    artifact_check_before = check_model_artifacts(resolved_out_dir)
    split_payload = build_split_payload(rows, readiness, feature_check["feature_columns"])

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    p0_blockers.extend(group_contract_check["p0_blockers"])
    p0_blockers.extend(readiness_check["p0_blockers"])
    p1_warnings.extend(readiness_check["p1_warnings"])
    p0_blockers.extend(boundary_check["p0_blockers"])
    p0_blockers.extend(feature_check["p0_blockers"])
    p0_blockers.extend(split_payload["p0_blockers"])
    p0_blockers.extend(artifact_check_before["p0_blockers"])

    metrics: dict[str, Any] = {}
    models_run: list[str] = []
    majority_class_collapse_check: dict[str, Any] = empty_collapse_check()
    decision = decide_pre_runtime(
        readiness_ok=readiness_check["passed"],
        manifest_ok=manifest_check.ok and group_contract_check["passed"] and feature_check["passed"],
        boundary_ok=boundary_check["passed"],
        split_ok=split_payload["passed"],
        artifact_ok=artifact_check_before["passed"],
    )

    if decision is None:
        try:
            metrics, models_run, majority_class_collapse_check = run_models(split_payload)
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            if artifact_check_after["passed"]:
                decision = READY_WITH_P1 if P1_INCONSISTENCY in p1_warnings else READY
            else:
                decision = BLOCKED_MODEL_ARTIFACT_CREATED
        except Exception as exc:  # noqa: BLE001 - diagnostic smoke must report runtime blockers.
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.append(f"smoke runtime error: {exc}")
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            decision = BLOCKED_SMOKE_RUNTIME_ERROR
    else:
        artifact_check_after = artifact_check_before

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
        "input_readiness_decision": readiness.get("readiness_decision"),
        "readiness_decision": decision,
        "target": TARGET_COLUMN,
        "models_run": models_run,
        "group_level_sample": True,
        "group_label_policy": GROUP_LABEL_POLICY,
        "intraday_live_decision_ready": False,
        "p1_warnings": dedupe(p1_warnings),
        "train_anchor_dates": split_payload["train_anchor_dates"],
        "valid_anchor_dates": split_payload["valid_anchor_dates"],
        "train_group_count": split_payload["train_group_count"],
        "valid_group_count": split_payload["valid_group_count"],
        "train_label_distribution": split_payload["train_label_distribution"],
        "valid_label_distribution": split_payload["valid_label_distribution"],
        "metrics": metrics,
        "majority_class_collapse_check": majority_class_collapse_check,
        "manifest_leakage_check": manifest_check.to_summary(),
        "group_contract_check": group_contract_check,
        "readiness_check": readiness_check,
        "boundary_check": boundary_check,
        "feature_check": feature_check,
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
        "not_trading_advice": True,
        "metrics_are_effectiveness_evidence": False,
        "automatic_promotion_ready": False,
        "formal_training": False,
        "hyperparameter_tuning": False,
        "p0_blockers": p0_blockers,
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
    return {
        "passed": not p0_blockers,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_readiness_check(readiness: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    p1_warnings = [warning for warning in string_list(readiness.get("p1_warnings")) if warning == P1_INCONSISTENCY]
    if readiness.get("readiness_decision") != EXPECTED_READINESS:
        p0_blockers.append(f"readiness_decision must be {EXPECTED_READINESS}")
    if P1_INCONSISTENCY not in p1_warnings:
        p0_blockers.append(f"readiness p1_warnings must include {P1_INCONSISTENCY}")
    if int(readiness.get("train_group_count") or 0) <= 0 or int(readiness.get("valid_group_count") or 0) <= 0:
        p0_blockers.append("readiness train_group_count and valid_group_count must be positive")
    if not string_list(readiness.get("train_anchor_dates")) or not string_list(readiness.get("valid_anchor_dates")):
        p0_blockers.append("readiness train_anchor_dates and valid_anchor_dates must be non-empty")
    return {
        "passed": not p0_blockers,
        "expected_readiness_decision": EXPECTED_READINESS,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
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


def run_feature_check(manifest: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    feature_columns = string_list(manifest.get("feature_columns"))
    feature_set = set(feature_columns)
    label_columns = set(string_list(manifest.get("label_columns")))
    outcome_columns = set(string_list(manifest.get("outcome_columns")))
    if TARGET_COLUMN not in columns:
        p0_blockers.append(f"{TARGET_COLUMN} missing from samples")
    missing_features = [column for column in feature_columns if column not in columns]
    if missing_features:
        p0_blockers.append("feature columns missing from samples: " + ", ".join(missing_features))
    label_overlap = sorted(feature_set & label_columns)
    if label_overlap:
        p0_blockers.append("label columns must not be in feature_columns: " + ", ".join(label_overlap))
    outcome_overlap = sorted(feature_set & outcome_columns)
    if outcome_overlap:
        p0_blockers.append("outcome columns must not be in feature_columns: " + ", ".join(outcome_overlap))
    future_features = sorted(column for column in feature_set if column.startswith("future_"))
    if future_features:
        p0_blockers.append("future_* columns must not be in feature_columns: " + ", ".join(future_features))
    label_pattern_columns = sorted(column for column in feature_set if column.endswith("_label"))
    if label_pattern_columns:
        p0_blockers.append("*_label columns must not be in feature_columns: " + ", ".join(label_pattern_columns))
    return {
        "passed": not p0_blockers,
        "feature_columns": feature_columns,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def build_split_payload(rows: list[dict[str, str]], readiness: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    anchor_column = next((column for column in ANCHOR_COLUMN_CANDIDATES if rows and column in rows[0]), "trade_date")
    if anchor_column == "anchor_date" and not any(str(row.get(anchor_column, "")).strip() for row in rows):
        anchor_column = "trade_date"
    train_anchor_dates = string_list(readiness.get("train_anchor_dates"))
    valid_anchor_dates = string_list(readiness.get("valid_anchor_dates"))
    train_set = set(train_anchor_dates)
    valid_set = set(valid_anchor_dates)
    train_rows = [row for row in rows if str(row.get(anchor_column, "")).strip() in train_set]
    valid_rows = [row for row in rows if str(row.get(anchor_column, "")).strip() in valid_set]
    p0_blockers: list[str] = []
    if not train_rows or not valid_rows:
        p0_blockers.append("readiness split must select non-empty train and valid groups")
    expected_train = readiness.get("train_group_count")
    expected_valid = readiness.get("valid_group_count")
    if expected_train is not None and len(train_rows) != int(expected_train):
        p0_blockers.append(f"train_group_count mismatch: expected {expected_train}, got {len(train_rows)}")
    if expected_valid is not None and len(valid_rows) != int(expected_valid):
        p0_blockers.append(f"valid_group_count mismatch: expected {expected_valid}, got {len(valid_rows)}")
    return {
        "passed": not p0_blockers,
        "split_check": {
            "anchor_column": anchor_column,
            "train_anchor_dates": train_anchor_dates,
            "valid_anchor_dates": valid_anchor_dates,
            "p0_blockers": p0_blockers,
        },
        "train_anchor_dates": train_anchor_dates,
        "valid_anchor_dates": valid_anchor_dates,
        "train_group_count": len(train_rows),
        "valid_group_count": len(valid_rows),
        "train_label_distribution": label_distribution(train_rows),
        "valid_label_distribution": label_distribution(valid_rows),
        "x_train": rows_to_matrix(train_rows, feature_columns),
        "y_train": rows_to_labels(train_rows),
        "x_valid": rows_to_matrix(valid_rows, feature_columns),
        "y_valid": rows_to_labels(valid_rows),
        "p0_blockers": p0_blockers,
    }


def run_models(split_payload: dict[str, Any]) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    x_train = split_payload["x_train"]
    y_train = split_payload["y_train"]
    x_valid = split_payload["x_valid"]
    y_valid = split_payload["y_valid"]
    if not x_train or not x_valid:
        raise GroupLevelNoSaveSmokeError("train and valid splits must both be non-empty")
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=42),
        "logistic_regression": LogisticRegression(max_iter=200, solver="liblinear", random_state=42),
        "logistic_regression_balanced_scaled": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced", random_state=42),
        ),
    }
    metrics: dict[str, Any] = {}
    predictions_by_model: dict[str, list[int]] = {}
    models_run: list[str] = []
    for model_name in MODEL_NAMES:
        model = models[model_name]
        model.fit(x_train, y_train)
        predictions = [int(value) for value in model.predict(x_valid)]
        predictions_by_model[model_name] = predictions
        metrics[model_name] = score_predictions(y_valid, predictions)
        models_run.append(model_name)
    return metrics, models_run, build_collapse_check(predictions_by_model)


def score_predictions(y_valid: list[int], predictions: list[int]) -> dict[str, Any]:
    return {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "prediction_distribution": distribution_from_values(predictions),
    }


def build_collapse_check(predictions_by_model: dict[str, list[int]]) -> dict[str, Any]:
    most_frequent = predictions_by_model.get("dummy_most_frequent", [])
    logistic = predictions_by_model.get("logistic_regression", [])
    balanced = predictions_by_model.get("logistic_regression_balanced_scaled", [])
    logistic_matches_dummy = logistic == most_frequent and bool(logistic)
    balanced_matches_dummy = balanced == most_frequent and bool(balanced)
    flags: list[str] = []
    flags.append("GROUP_LEVEL_LOGISTIC_MATCHES_DUMMY_MOST_FREQUENT" if logistic_matches_dummy else "GROUP_LEVEL_LOGISTIC_NOT_COLLAPSED")
    flags.append(
        "GROUP_LEVEL_BALANCED_SCALED_PROBE_STILL_COLLAPSED"
        if balanced_matches_dummy
        else "GROUP_LEVEL_BALANCED_SCALED_PROBE_REDUCES_COLLAPSE"
    )
    return {
        "dummy_most_frequent_prediction_distribution": distribution_from_values(most_frequent),
        "logistic_prediction_distribution_on_valid": distribution_from_values(logistic),
        "balanced_scaled_prediction_distribution_on_valid": distribution_from_values(balanced),
        "logistic_matches_dummy_most_frequent": logistic_matches_dummy,
        "balanced_scaled_probe_reduces_collapse": not balanced_matches_dummy,
        "flags": flags,
    }


def empty_collapse_check() -> dict[str, Any]:
    return {
        "dummy_most_frequent_prediction_distribution": {"0": 0, "1": 0},
        "logistic_prediction_distribution_on_valid": {"0": 0, "1": 0},
        "balanced_scaled_prediction_distribution_on_valid": {"0": 0, "1": 0},
        "logistic_matches_dummy_most_frequent": None,
        "balanced_scaled_probe_reduces_collapse": None,
        "flags": [],
    }


def decide_pre_runtime(
    readiness_ok: bool,
    manifest_ok: bool,
    boundary_ok: bool,
    split_ok: bool,
    artifact_ok: bool,
) -> str | None:
    if not readiness_ok:
        return BLOCKED_GROUP_LEVEL_READINESS_NOT_PASSED
    if not boundary_ok:
        return BLOCKED_BOUNDARY_FLAG
    if not manifest_ok or not split_ok:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    if not artifact_ok:
        return BLOCKED_MODEL_ARTIFACT_CREATED
    return None


def rows_to_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> list[list[float]]:
    return [[float(row[column]) for column in feature_columns] for row in rows]


def rows_to_labels(rows: list[dict[str, str]]) -> list[int]:
    return [int(float(row[TARGET_COLUMN])) for row in rows]


def label_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return distribution_from_values(rows_to_labels(rows))


def distribution_from_values(values: list[int]) -> dict[str, int]:
    return {"0": sum(1 for value in values if value == 0), "1": sum(1 for value in values if value == 1)}


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    (out_dir / "intraday_group_level_no_save_diagnostic_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "readiness_decision": report["readiness_decision"],
        "status": report["status"],
        "report_type": report["report_type"],
        "smoke_scope": report["smoke_scope"],
        "models_run": report["models_run"],
        "majority_class_collapse_check": report["majority_class_collapse_check"],
        "model_saved": False,
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
    }
    (out_dir / "readiness_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "",
        "# Intraday Group-Level No-Save Diagnostic Smoke",
        "",
        "本文件只用于 Lab-only group-level CPU no-save diagnostic smoke，不是正式训练，不调参，不保存模型，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。",
        "",
        f"- report_type: {report['report_type']}",
        f"- smoke_scope: {report['smoke_scope']}",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- target: {report['target']}",
        f"- group_level_sample: {str(report['group_level_sample']).lower()}",
        f"- group_label_policy: {report['group_label_policy']}",
        f"- intraday_live_decision_ready: {str(report['intraday_live_decision_ready']).lower()}",
        f"- p1_warnings: {json.dumps(report['p1_warnings'], ensure_ascii=False)}",
        f"- models_run: {', '.join(report['models_run'])}",
        f"- train_group_count: {report['train_group_count']}",
        f"- valid_group_count: {report['valid_group_count']}",
        f"- train_label_distribution: {json.dumps(report['train_label_distribution'], ensure_ascii=False, sort_keys=True)}",
        f"- valid_label_distribution: {json.dumps(report['valid_label_distribution'], ensure_ascii=False, sort_keys=True)}",
        f"- logistic_prediction_distribution_on_valid: {json.dumps(report['majority_class_collapse_check']['logistic_prediction_distribution_on_valid'], ensure_ascii=False, sort_keys=True)}",
        f"- logistic_matches_dummy_most_frequent: {str(report['majority_class_collapse_check']['logistic_matches_dummy_most_frequent']).lower()}",
        f"- balanced_scaled_probe_reduces_collapse: {str(report['majority_class_collapse_check']['balanced_scaled_probe_reduces_collapse']).lower()}",
        f"- majority_class_collapse_flags: {json.dumps(report['majority_class_collapse_check']['flags'], ensure_ascii=False)}",
        f"- model_saved: {str(report['model_saved']).lower()}",
        f"- checkpoint_saved: {str(report['checkpoint_saved']).lower()}",
        f"- gpu_used: {str(report['gpu_used']).lower()}",
        f"- torchrun_used: {str(report['torchrun_used']).lower()}",
        f"- qmt_used: {str(report['qmt_used']).lower()}",
        f"- order_intent_generated: {str(report['order_intent_generated']).lower()}",
        f"- stable_affected: {str(report['stable_affected']).lower()}",
        f"- metrics_are_effectiveness_evidence: {str(report['metrics_are_effectiveness_evidence']).lower()}",
        f"- automatic_promotion_ready: {str(report['automatic_promotion_ready']).lower()}",
        "- boundary: no formal training, no tuning, no model save, no checkpoint, no GPU, no torchrun, no QMT, no OrderIntent, no Stable, no output/, no lab_advisory, not trading advice.",
    ]
    (out_dir / "intraday_group_level_no_save_diagnostic_smoke_report.md").write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only group-level no-save diagnostic smoke.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_smoke(args.samples, args.manifest, args.readiness, args.out_dir)
    except GroupLevelNoSaveSmokeError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "readiness_decision": BLOCKED_SMOKE_RUNTIME_ERROR,
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
                "readiness_decision": report["readiness_decision"],
                "models_run": report["models_run"],
                "majority_class_collapse_check": report["majority_class_collapse_check"],
                "model_saved": False,
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
