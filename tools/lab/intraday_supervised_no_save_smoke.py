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


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_supervised_no_save_smoke")
EXPECTED_READINESS = "SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED"
READY = "NO_SAVE_SUPERVISED_SMOKE_COMPLETED_REVIEW_REQUIRED"
BLOCKED_READINESS_NOT_PASSED = "BLOCKED_READINESS_NOT_PASSED"
BLOCKED_MANIFEST_LEAKAGE_P0 = "BLOCKED_MANIFEST_LEAKAGE_P0"
BLOCKED_BOUNDARY_FLAG = "BLOCKED_BOUNDARY_FLAG"
BLOCKED_MODEL_ARTIFACT_CREATED = "BLOCKED_MODEL_ARTIFACT_CREATED"
BLOCKED_SMOKE_RUNTIME_ERROR = "BLOCKED_SMOKE_RUNTIME_ERROR"
TARGET_COLUMN = "three_day_positive_label"
ANCHOR_COLUMN_CANDIDATES = ("anchor_date", "trade_date")
FORBIDDEN_MODEL_ARTIFACT_EXTENSIONS = {".pkl", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"}
BOUNDARY_FALSE_FIELDS = [
    "training_allowed",
    "supervised_training_allowed",
    "stable_effect_allowed",
    "contains_order_intent",
    "contains_live_order",
    "contains_secret",
]
MODEL_NAMES = ["dummy_most_frequent", "dummy_stratified", "logistic_regression"]


class NoSaveSmokeError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise NoSaveSmokeError(f"JSON parse failed for {path}: {exc}") from exc
    except OSError as exc:
        raise NoSaveSmokeError(f"JSON cannot be read: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise NoSaveSmokeError(f"JSON root must be object: {path}")
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
        raise NoSaveSmokeError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise NoSaveSmokeError(f"samples CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise NoSaveSmokeError(f"samples CSV has no header: {path}")
    return rows, columns


def run_smoke(
    samples_path: Path,
    manifest_path: Path,
    readiness_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(str(samples_path), repo_root)
    resolved_manifest = resolve_repo_path(str(manifest_path), repo_root)
    resolved_readiness = resolve_repo_path(str(readiness_path), repo_root)
    if resolved_samples is None or not resolved_samples.exists():
        raise NoSaveSmokeError("samples path missing or does not exist")
    if resolved_manifest is None or not resolved_manifest.exists():
        raise NoSaveSmokeError("manifest path missing or does not exist")
    if resolved_readiness is None or not resolved_readiness.exists():
        raise NoSaveSmokeError("readiness path missing or does not exist")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    readiness = load_json(resolved_readiness)
    manifest = load_json(resolved_manifest)
    rows, columns = load_csv_rows(resolved_samples)
    manifest_check = check_label_manifest(resolved_manifest)
    boundary_check = run_boundary_check(manifest)
    feature_check = run_feature_check(manifest, columns)
    artifact_check_before = check_model_artifacts(resolved_out_dir)

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    p0_blockers.extend(boundary_check["p0_blockers"])
    p0_blockers.extend(feature_check["p0_blockers"])
    p0_blockers.extend(artifact_check_before["p0_blockers"])

    readiness_ok = readiness.get("readiness_decision") == EXPECTED_READINESS
    if not readiness_ok:
        p0_blockers.append(f"readiness_decision must be {EXPECTED_READINESS}")

    metrics: dict[str, Any] = {}
    models_run: list[str] = []
    split_payload = build_split_payload(rows, readiness, feature_check["feature_columns"])
    decision = decide_pre_runtime(
        readiness_ok=readiness_ok,
        boundary_ok=boundary_check["passed"],
        manifest_ok=manifest_check.ok and feature_check["passed"],
        artifact_ok=artifact_check_before["passed"],
    )

    if decision is None:
        try:
            metrics, models_run = run_models(split_payload)
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            decision = READY if artifact_check_after["passed"] else BLOCKED_MODEL_ARTIFACT_CREATED
        except Exception as exc:  # noqa: BLE001 - smoke must report runtime failures instead of hiding them.
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.append(f"smoke runtime error: {exc}")
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            decision = BLOCKED_SMOKE_RUNTIME_ERROR
    else:
        artifact_check_after = artifact_check_before

    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": "intraday_supervised_no_save_smoke",
        "smoke_scope": "lab_only_no_save",
        "status": "passed" if decision == READY else "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "readiness_path": str(readiness_path),
        "readiness_decision": decision,
        "input_readiness_decision": readiness.get("readiness_decision"),
        "target": TARGET_COLUMN,
        "feature_columns": feature_check["feature_columns"],
        "models_run": models_run,
        "train_anchor_dates": split_payload["train_anchor_dates"],
        "valid_anchor_dates": split_payload["valid_anchor_dates"],
        "train_rows": split_payload["train_rows"],
        "valid_rows": split_payload["valid_rows"],
        "train_label_distribution": split_payload["train_label_distribution"],
        "valid_label_distribution": split_payload["valid_label_distribution"],
        "metrics": metrics,
        "manifest_leakage_check": manifest_check.to_summary(),
        "boundary_check": boundary_check,
        "feature_check": feature_check,
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
        "formal_training": False,
        "hyperparameter_tuning": False,
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


def run_feature_check(manifest: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    feature_columns = string_list(manifest.get("feature_columns"))
    label_columns = set(string_list(manifest.get("label_columns")))
    outcome_columns = set(string_list(manifest.get("outcome_columns")))
    feature_set = set(feature_columns)
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
    return {
        "passed": not p0_blockers,
        "feature_columns": feature_columns,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def check_model_artifacts(out_dir: Path) -> dict[str, Any]:
    artifacts = sorted(
        str(path.relative_to(out_dir))
        for path in out_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_MODEL_ARTIFACT_EXTENSIONS
    )
    return {
        "passed": not artifacts,
        "forbidden_extensions": sorted(FORBIDDEN_MODEL_ARTIFACT_EXTENSIONS),
        "found_model_artifacts": artifacts,
        "p0_blockers": ["forbidden model artifact created or present: " + ", ".join(artifacts)] if artifacts else [],
        "p1_warnings": [],
    }


def build_split_payload(rows: list[dict[str, str]], readiness: dict[str, Any], feature_columns: list[str]) -> dict[str, Any]:
    anchor_column = next((column for column in ANCHOR_COLUMN_CANDIDATES if rows and column in rows[0]), "trade_date")
    train_anchor_dates = string_list(readiness.get("train_anchor_dates"))
    valid_anchor_dates = string_list(readiness.get("valid_anchor_dates"))
    train_set = set(train_anchor_dates)
    valid_set = set(valid_anchor_dates)
    train_rows_raw = [row for row in rows if str(row.get(anchor_column, "")).strip() in train_set]
    valid_rows_raw = [row for row in rows if str(row.get(anchor_column, "")).strip() in valid_set]
    return {
        "anchor_column": anchor_column,
        "train_anchor_dates": train_anchor_dates,
        "valid_anchor_dates": valid_anchor_dates,
        "train_rows": len(train_rows_raw),
        "valid_rows": len(valid_rows_raw),
        "train_label_distribution": label_distribution(train_rows_raw),
        "valid_label_distribution": label_distribution(valid_rows_raw),
        "x_train": rows_to_matrix(train_rows_raw, feature_columns),
        "y_train": rows_to_labels(train_rows_raw),
        "x_valid": rows_to_matrix(valid_rows_raw, feature_columns),
        "y_valid": rows_to_labels(valid_rows_raw),
    }


def run_models(split_payload: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    x_train = split_payload["x_train"]
    y_train = split_payload["y_train"]
    x_valid = split_payload["x_valid"]
    y_valid = split_payload["y_valid"]
    if not x_train or not x_valid:
        raise NoSaveSmokeError("train and valid splits must both be non-empty")
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=42),
        "logistic_regression": LogisticRegression(max_iter=200, solver="liblinear", random_state=42),
    }
    metrics: dict[str, Any] = {}
    models_run: list[str] = []
    for model_name in MODEL_NAMES:
        model = models[model_name]
        model.fit(x_train, y_train)
        predictions = model.predict(x_valid)
        metrics[model_name] = {
            "accuracy": float(accuracy_score(y_valid, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
            "precision": float(precision_score(y_valid, predictions, zero_division=0)),
            "recall": float(recall_score(y_valid, predictions, zero_division=0)),
            "prediction_distribution": distribution_from_values([int(value) for value in predictions]),
        }
        models_run.append(model_name)
    return metrics, models_run


def rows_to_matrix(rows: list[dict[str, str]], feature_columns: list[str]) -> list[list[float]]:
    return [[float(row[column]) for column in feature_columns] for row in rows]


def rows_to_labels(rows: list[dict[str, str]]) -> list[int]:
    return [int(float(row[TARGET_COLUMN])) for row in rows]


def label_distribution(rows: list[dict[str, str]]) -> dict[str, int]:
    return distribution_from_values(rows_to_labels(rows))


def distribution_from_values(values: list[int]) -> dict[str, int]:
    return {"0": sum(1 for value in values if value == 0), "1": sum(1 for value in values if value == 1)}


def decide_pre_runtime(
    readiness_ok: bool,
    boundary_ok: bool,
    manifest_ok: bool,
    artifact_ok: bool,
) -> str | None:
    if not readiness_ok:
        return BLOCKED_READINESS_NOT_PASSED
    if not boundary_ok:
        return BLOCKED_BOUNDARY_FLAG
    if not manifest_ok:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    if not artifact_ok:
        return BLOCKED_MODEL_ARTIFACT_CREATED
    return None


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intraday_supervised_no_save_smoke_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "readiness_decision": report["readiness_decision"],
        "status": report["status"],
        "models_run": report["models_run"],
        "train_anchor_dates": report["train_anchor_dates"],
        "valid_anchor_dates": report["valid_anchor_dates"],
        "train_rows": report["train_rows"],
        "valid_rows": report["valid_rows"],
        "train_label_distribution": report["train_label_distribution"],
        "valid_label_distribution": report["valid_label_distribution"],
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "metrics_are_effectiveness_evidence": False,
        "p0_blockers": report["p0_blockers"],
    }
    (out_dir / "readiness_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab no-save supervised smoke，不是正式训练，不接 QMT，不生成 OrderIntent，不进入 Stable。",
        "",
        "# Intraday Supervised No-Save Smoke Report",
        "",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- target: {report['target']}",
        f"- models_run: {', '.join(report['models_run'])}",
        f"- train_anchor_dates: {', '.join(report['train_anchor_dates'])}",
        f"- valid_anchor_dates: {', '.join(report['valid_anchor_dates'])}",
        f"- train_rows: {report['train_rows']}",
        f"- valid_rows: {report['valid_rows']}",
        f"- train_label_distribution: {json.dumps(report['train_label_distribution'], ensure_ascii=False, sort_keys=True)}",
        f"- valid_label_distribution: {json.dumps(report['valid_label_distribution'], ensure_ascii=False, sort_keys=True)}",
        f"- model_saved: {str(report['model_saved']).lower()}",
        f"- checkpoint_saved: {str(report['checkpoint_saved']).lower()}",
        f"- metrics_are_effectiveness_evidence: {str(report['metrics_are_effectiveness_evidence']).lower()}",
        "- boundary: no formal training, no tuning, no model save, no checkpoint, no GPU, no torchrun, no QMT, no OrderIntent, no Stable, no output/, no lab_advisory, not trading advice.",
    ]
    (out_dir / "intraday_supervised_no_save_smoke_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday supervised no-save smoke.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_smoke(args.samples, args.manifest, args.readiness, args.out_dir)
    except NoSaveSmokeError as exc:
        print(json.dumps({"status": "failed", "readiness_decision": BLOCKED_SMOKE_RUNTIME_ERROR, "p0_blockers": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": report["status"],
        "readiness_decision": report["readiness_decision"],
        "models_run": report["models_run"],
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "metrics_are_effectiveness_evidence": False,
        "p0_blockers": report["p0_blockers"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
