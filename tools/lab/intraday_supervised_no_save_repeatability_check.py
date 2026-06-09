from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest
from tools.lab.intraday_supervised_no_save_smoke import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    BLOCKED_MODEL_ARTIFACT_CREATED,
    BLOCKED_READINESS_NOT_PASSED,
    FORBIDDEN_MODEL_ARTIFACT_EXTENSIONS,
    MODEL_NAMES,
    TARGET_COLUMN,
    build_split_payload,
    check_model_artifacts,
    load_csv_rows,
    load_json,
    run_boundary_check,
    run_feature_check,
)


ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_supervised_no_save_repeatability_check")
EXPECTED_INPUT_READINESS = "SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED"
EXPECTED_BASELINE_DECISION = "NO_SAVE_SUPERVISED_SMOKE_COMPLETED_REVIEW_REQUIRED"
READY = "NO_SAVE_SUPERVISED_SMOKE_REPEATABILITY_COMPLETED_REVIEW_REQUIRED"
BLOCKED_REPEATABILITY_RUNTIME_ERROR = "BLOCKED_REPEATABILITY_RUNTIME_ERROR"
SEEDS = [7, 13, 42, 101, 2026]


class RepeatabilityCheckError(RuntimeError):
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
        raise RepeatabilityCheckError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def run_repeatability_check(
    samples_path: Path,
    manifest_path: Path,
    readiness_path: Path,
    baseline_smoke_path: Path,
    out_dir: Path,
    seeds: Sequence[int] = SEEDS,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(samples_path, repo_root)
    resolved_manifest = resolve_repo_path(manifest_path, repo_root)
    resolved_readiness = resolve_repo_path(readiness_path, repo_root)
    resolved_baseline = resolve_repo_path(baseline_smoke_path, repo_root)
    for required_path, label in (
        (resolved_samples, "samples"),
        (resolved_manifest, "manifest"),
        (resolved_readiness, "readiness"),
        (resolved_baseline, "baseline smoke"),
    ):
        if not required_path.exists():
            raise RepeatabilityCheckError(f"{label} path does not exist: {required_path}")

    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    readiness = load_json(resolved_readiness)
    manifest = load_json(resolved_manifest)
    baseline = load_json(resolved_baseline)
    rows, columns = load_csv_rows(resolved_samples)

    manifest_check = check_label_manifest(resolved_manifest)
    boundary_check = run_boundary_check(manifest)
    feature_columns = list_from_baseline(baseline)
    feature_check = run_feature_check({**manifest, "feature_columns": feature_columns}, columns)
    baseline_check = run_baseline_report_check(baseline)
    artifact_check_before = check_model_artifacts(resolved_out_dir)

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    p0_blockers.extend(boundary_check["p0_blockers"])
    p0_blockers.extend(feature_check["p0_blockers"])
    p0_blockers.extend(baseline_check["p0_blockers"])
    p0_blockers.extend(artifact_check_before["p0_blockers"])

    readiness_ok = readiness.get("readiness_decision") == EXPECTED_INPUT_READINESS
    if not readiness_ok:
        p0_blockers.append(f"readiness_decision must be {EXPECTED_INPUT_READINESS}")

    split_payload = build_split_payload(rows, baseline, feature_columns)
    decision = decide_pre_runtime(
        readiness_ok=readiness_ok,
        boundary_ok=boundary_check["passed"],
        manifest_ok=manifest_check.ok and feature_check["passed"],
        artifact_ok=artifact_check_before["passed"],
        baseline_ok=baseline_check["passed"],
    )

    metrics_by_seed: list[dict[str, Any]] = []
    models_run: list[str] = []
    if decision is None:
        try:
            metrics_by_seed, models_run = run_models_by_seed(split_payload, seeds)
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            decision = READY if artifact_check_after["passed"] else BLOCKED_MODEL_ARTIFACT_CREATED
        except Exception as exc:  # noqa: BLE001 - repeatability must surface runtime errors.
            artifact_check_after = check_model_artifacts(resolved_out_dir)
            p0_blockers.append(f"repeatability runtime error: {exc}")
            p0_blockers.extend(artifact_check_after["p0_blockers"])
            decision = BLOCKED_REPEATABILITY_RUNTIME_ERROR
    else:
        artifact_check_after = artifact_check_before

    report = {
        "lab_declaration": "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "report_type": "intraday_supervised_no_save_repeatability_check",
        "smoke_scope": "lab_only_no_save_repeatability",
        "status": "passed" if decision == READY else "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "readiness_path": str(readiness_path),
        "baseline_smoke_path": str(baseline_smoke_path),
        "readiness_decision": decision,
        "input_readiness_decision": readiness.get("readiness_decision"),
        "baseline_readiness_decision": baseline.get("readiness_decision"),
        "target": TARGET_COLUMN,
        "seeds": list(seeds),
        "feature_columns": feature_columns,
        "models_run": models_run,
        "train_anchor_dates": split_payload["train_anchor_dates"],
        "valid_anchor_dates": split_payload["valid_anchor_dates"],
        "train_rows": split_payload["train_rows"],
        "valid_rows": split_payload["valid_rows"],
        "train_label_distribution": split_payload["train_label_distribution"],
        "valid_label_distribution": split_payload["valid_label_distribution"],
        "metrics_by_seed": metrics_by_seed,
        "metric_variability_summary": summarize_metric_variability(metrics_by_seed),
        "manifest_leakage_check": manifest_check.to_summary(),
        "boundary_check": boundary_check,
        "feature_check": feature_check,
        "baseline_check": baseline_check,
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
        "p1_warnings": p1_warnings,
    }
    write_reports(report, resolved_out_dir)
    return report


def list_from_baseline(baseline: dict[str, Any]) -> list[str]:
    value = baseline.get("feature_columns")
    if not isinstance(value, list) or not value:
        raise RepeatabilityCheckError("baseline smoke report must contain non-empty feature_columns")
    return [str(item) for item in value if str(item)]


def run_baseline_report_check(baseline: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    if baseline.get("readiness_decision") != EXPECTED_BASELINE_DECISION:
        p0_blockers.append(f"baseline readiness_decision must be {EXPECTED_BASELINE_DECISION}")
    if baseline.get("target") != TARGET_COLUMN:
        p0_blockers.append(f"baseline target must be {TARGET_COLUMN}")
    if baseline.get("models_run") != MODEL_NAMES:
        p0_blockers.append("baseline models_run must match repeatability model scope")
    for field_name, expected in (
        ("model_saved", False),
        ("checkpoint_saved", False),
        ("gpu_used", False),
        ("torchrun_used", False),
        ("qmt_used", False),
        ("order_intent_generated", False),
        ("stable_affected", False),
        ("metrics_are_effectiveness_evidence", False),
    ):
        if baseline.get(field_name) is not expected:
            p0_blockers.append(f"baseline {field_name} must be {str(expected).lower()}")
    return {
        "passed": not p0_blockers,
        "expected_readiness_decision": EXPECTED_BASELINE_DECISION,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def decide_pre_runtime(
    readiness_ok: bool,
    boundary_ok: bool,
    manifest_ok: bool,
    artifact_ok: bool,
    baseline_ok: bool,
) -> str | None:
    if not readiness_ok:
        return BLOCKED_READINESS_NOT_PASSED
    if not boundary_ok:
        return BLOCKED_BOUNDARY_FLAG
    if not manifest_ok:
        return BLOCKED_MANIFEST_LEAKAGE_P0
    if not artifact_ok:
        return BLOCKED_MODEL_ARTIFACT_CREATED
    if not baseline_ok:
        return BLOCKED_BOUNDARY_FLAG
    return None


def run_models_by_seed(split_payload: dict[str, Any], seeds: Sequence[int]) -> tuple[list[dict[str, Any]], list[str]]:
    x_train = split_payload["x_train"]
    y_train = split_payload["y_train"]
    x_valid = split_payload["x_valid"]
    y_valid = split_payload["y_valid"]
    if not x_train or not x_valid:
        raise RepeatabilityCheckError("train and valid splits must both be non-empty")

    metrics_by_seed: list[dict[str, Any]] = []
    for seed in seeds:
        models = {
            "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
            "dummy_stratified": DummyClassifier(strategy="stratified", random_state=int(seed)),
            "logistic_regression": LogisticRegression(max_iter=200, solver="liblinear", random_state=int(seed)),
        }
        seed_metrics: dict[str, Any] = {}
        for model_name in MODEL_NAMES:
            model = models[model_name]
            model.fit(x_train, y_train)
            predictions = model.predict(x_valid)
            seed_metrics[model_name] = {
                "accuracy": float(accuracy_score(y_valid, predictions)),
                "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
                "precision": float(precision_score(y_valid, predictions, zero_division=0)),
                "recall": float(recall_score(y_valid, predictions, zero_division=0)),
                "prediction_distribution": distribution_from_values([int(value) for value in predictions]),
            }
        metrics_by_seed.append({"seed": int(seed), "metrics": seed_metrics})
    return metrics_by_seed, list(MODEL_NAMES)


def summarize_metric_variability(metrics_by_seed: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metric_names = ["accuracy", "balanced_accuracy", "precision", "recall"]
    for model_name in MODEL_NAMES:
        summary[model_name] = {}
        for metric_name in metric_names:
            values = [
                float(seed_payload["metrics"][model_name][metric_name])
                for seed_payload in metrics_by_seed
                if model_name in seed_payload.get("metrics", {})
            ]
            if not values:
                summary[model_name][metric_name] = None
                continue
            summary[model_name][metric_name] = {
                "min": min(values),
                "max": max(values),
                "mean": mean(values),
                "std": pstdev(values) if len(values) > 1 else 0.0,
                "range": max(values) - min(values),
                "all_finite": all(math.isfinite(value) for value in values),
            }
    return summary


def distribution_from_values(values: list[int]) -> dict[str, int]:
    return {"0": sum(1 for value in values if value == 0), "1": sum(1 for value in values if value == 1)}


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intraday_supervised_no_save_repeatability_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "readiness_decision": report["readiness_decision"],
        "status": report["status"],
        "seeds": report["seeds"],
        "models_run": report["models_run"],
        "train_anchor_dates": report["train_anchor_dates"],
        "valid_anchor_dates": report["valid_anchor_dates"],
        "train_rows": report["train_rows"],
        "valid_rows": report["valid_rows"],
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
    }
    (out_dir / "readiness_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab no-save supervised smoke repeatability check，不是正式训练，不接 QMT，不生成 OrderIntent，不进入 Stable。",
        "",
        "# Intraday Supervised No-Save Repeatability Report",
        "",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- target: {report['target']}",
        f"- seeds: {', '.join(map(str, report['seeds']))}",
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
        f"- automatic_promotion_ready: {str(report['automatic_promotion_ready']).lower()}",
        "- boundary: no larger sample, no formal training, no tuning, no model save, no checkpoint, no GPU, no torchrun, no QMT, no OrderIntent, no Stable, no output/, no lab_advisory, not trading advice.",
    ]
    (out_dir / "intraday_supervised_no_save_repeatability_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday supervised no-save repeatability check.")
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readiness", required=True, type=Path)
    parser.add_argument("--baseline-smoke", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_repeatability_check(
            args.samples,
            args.manifest,
            args.readiness,
            args.baseline_smoke,
            args.out_dir,
        )
    except RepeatabilityCheckError as exc:
        print(
            json.dumps(
                {"status": "failed", "readiness_decision": BLOCKED_REPEATABILITY_RUNTIME_ERROR, "p0_blockers": [str(exc)]},
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
                "seeds": report["seeds"],
                "models_run": report["models_run"],
                "train_rows": report["train_rows"],
                "valid_rows": report["valid_rows"],
                "model_saved": report["model_saved"],
                "checkpoint_saved": report["checkpoint_saved"],
                "metrics_are_effectiveness_evidence": report["metrics_are_effectiveness_evidence"],
                "automatic_promotion_ready": report["automatic_promotion_ready"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["readiness_decision"] == READY else 1


if __name__ == "__main__":
    raise SystemExit(main())
