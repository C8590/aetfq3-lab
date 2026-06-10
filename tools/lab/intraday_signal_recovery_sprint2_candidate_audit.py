from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_signal_recovery_sprint2_candidate_audit"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint2_candidate_audit")
SPRINT1_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint1")
DEFAULT_SAMPLES = SPRINT1_DIR / "signal_recovery_sprint1_feature_samples.csv"
DEFAULT_MANIFEST = SPRINT1_DIR / "signal_recovery_sprint1_manifest.json"
DEFAULT_LABEL_REPORT = SPRINT1_DIR / "signal_recovery_sprint1_label_policy_report.json"
DEFAULT_FEATURE_REPORT = SPRINT1_DIR / "signal_recovery_sprint1_feature_recovery_report.json"
DEFAULT_DIAGNOSTIC_REPORT = SPRINT1_DIR / "signal_recovery_sprint1_diagnostic_smoke_report.json"
DEFAULT_DECISION = SPRINT1_DIR / "signal_recovery_sprint1_decision.json"

SEEDS = [7, 13, 42, 101, 2026]
ALLOWED_SPLIT_POLICIES = {
    "anchor_date_70_30",
    "anchor_date_60_40",
    "walk_forward_3fold_contiguous_anchor",
}
MODEL_FAMILY_BY_MODEL = {
    "logistic_balanced_scaled": "logistic_balanced_scaled_variants",
    "logistic_log1p_scaled_balanced": "logistic_balanced_scaled_variants",
    "random_forest_shallow_no_save": "random_forest_shallow_no_save",
    "hist_gradient_boosting_no_save": "hist_gradient_boosting_no_save",
    "lightgbm_no_save": "lightgbm_no_save",
    "xgboost_no_save": "xgboost_no_save",
    "catboost_no_save": "catboost_no_save",
}
MODEL_PRIORITY_BY_FAMILY = {
    "logistic_balanced_scaled_variants": [
        "logistic_log1p_scaled_balanced",
        "logistic_balanced_scaled",
    ],
    "random_forest_shallow_no_save": ["random_forest_shallow_no_save"],
    "hist_gradient_boosting_no_save": ["hist_gradient_boosting_no_save"],
    "lightgbm_no_save": ["lightgbm_no_save"],
    "xgboost_no_save": ["xgboost_no_save"],
    "catboost_no_save": ["catboost_no_save"],
}
BOUNDARY_FALSE_FIELDS = {
    "formal_model_evidence": False,
    "stable_promotion_ready": False,
    "formal_training_ready": False,
    "qmt_ready": False,
    "order_intent_ready": False,
    "automatic_promotion_ready": False,
    "metrics_are_effectiveness_evidence": False,
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
DECISION_ROBUST = "SIGNAL_RECOVERY_SPRINT2_ROBUST_DIAGNOSTIC_CANDIDATE_FOUND_REVIEW_REQUIRED"
DECISION_WEAK = "SIGNAL_RECOVERY_SPRINT2_CANDIDATES_WEAK_OR_UNSTABLE"
DECISION_LEAKAGE = "SIGNAL_RECOVERY_SPRINT2_BLOCKED_LEAKAGE_P0"
DECISION_RUNTIME = "SIGNAL_RECOVERY_SPRINT2_BLOCKED_RUNTIME_ERROR"
ROBUST_CANDIDATE = "ROBUST_DIAGNOSTIC_SIGNAL_CANDIDATE_REVIEW_REQUIRED"
WEAK_CANDIDATE = "CANDIDATE_WEAK_OR_SPLIT_UNSTABLE"


class Sprint2AuditError(RuntimeError):
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
        raise Sprint2AuditError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise Sprint2AuditError(f"JSON cannot be read: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise Sprint2AuditError(f"JSON parse failed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise Sprint2AuditError(f"JSON root must be object: {path}")
    return payload


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise Sprint2AuditError(f"CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise Sprint2AuditError(f"CSV has no header: {path}")
    return rows, columns


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def audit_candidates(
    diagnostic_report: dict[str, Any],
    label_report: dict[str, Any],
) -> dict[str, Any]:
    candidates = [
        item
        for item in diagnostic_report.get("diagnostic_candidates", [])
        if isinstance(item, dict) and item.get("candidate_gate", {}).get("diagnostic_signal_candidate") is True
    ]
    enriched = [enrich_candidate(item, label_report) for item in candidates]
    labels_by_non_label = defaultdict(set)
    models_by_non_model = defaultdict(set)
    for item in enriched:
        labels_by_non_label[
            (item["feature_set"], item["model_family"], item["transform_policy"])
        ].add(item["label_policy"])
        models_by_non_model[
            (item["label_policy"], item["feature_set"], item["transform_policy"])
        ].add(item["model_family"])
    weak_pr_auc = [item for item in enriched if item["pr_auc_margin_over_prevalence"] is not None and item["pr_auc_margin_over_prevalence"] < 0.05]
    near_collapse = [item for item in enriched if is_near_collapse(item["prediction_distribution"])]
    return {
        "candidate_count": len(enriched),
        "candidate_count_by_label_policy": dict(Counter(item["label_policy"] for item in enriched)),
        "candidate_count_by_feature_set": dict(Counter(item["feature_set"] for item in enriched)),
        "candidate_count_by_model": dict(Counter(item["model"] for item in enriched)),
        "candidate_count_by_transform_policy": dict(Counter(item["transform_policy"] for item in enriched)),
        "candidate_count_by_model_family": dict(Counter(item["model_family"] for item in enriched)),
        "candidates": enriched,
        "candidates_isolated_to_one_label_policy": [
            item for item in enriched if len(labels_by_non_label[(item["feature_set"], item["model_family"], item["transform_policy"])]) == 1
        ],
        "candidates_isolated_to_one_model_family": [
            item for item in enriched if len(models_by_non_model[(item["label_policy"], item["feature_set"], item["transform_policy"])]) == 1
        ],
        "candidates_with_weak_pr_auc_margin": weak_pr_auc,
        "candidates_with_near_collapse_prediction_distribution": near_collapse,
    }


def enrich_candidate(candidate: dict[str, Any], label_report: dict[str, Any]) -> dict[str, Any]:
    label_policy = str(candidate.get("label_policy", ""))
    label_summary = label_report.get("label_policies", {}).get(label_policy, {})
    train_balance = candidate.get("train_label_distribution", {})
    valid_balance = candidate.get("valid_label_distribution", {})
    row_count = int(candidate.get("train_rows_used", 0)) + int(candidate.get("valid_rows_used", 0))
    null_count = int(label_summary.get("null_count", 0))
    prevalence = safe_div(float(valid_balance.get("1", 0)), float(sum_int_values(valid_balance)))
    pr_auc = to_float(candidate.get("pr_auc"))
    return {
        **candidate,
        "model_family": model_family(candidate.get("model")),
        "transform_policy": transform_policy_name(candidate.get("feature_set")),
        "candidate_family": candidate_family_key(candidate),
        "row_count": row_count,
        "null_count": null_count,
        "class_balance": {
            "train": train_balance,
            "valid": valid_balance,
        },
        "prevalence": prevalence,
        "pr_auc_margin_over_prevalence": None if pr_auc is None or prevalence is None else pr_auc - prevalence,
        "improvement_vs_dummy_most_frequent": candidate.get("compared_to_dummy_most_frequent", {}),
        "improvement_vs_dummy_stratified": candidate.get("compared_to_dummy_stratified", {}),
        "not_tiny_neutral_band_sample": not_tiny_neutral_band(label_policy, row_count, label_summary),
    }


def candidate_family_key(candidate: dict[str, Any]) -> str:
    return "|".join(
        [
            str(candidate.get("label_policy", "")),
            str(candidate.get("feature_set", "")),
            model_family(candidate.get("model")),
            transform_policy_name(candidate.get("feature_set")),
        ]
    )


def model_family(model: Any) -> str:
    return MODEL_FAMILY_BY_MODEL.get(str(model), str(model))


def transform_policy_name(feature_set: Any) -> str:
    text = str(feature_set)
    return "scale_transform_policy" if "scale_transform_policy" in text else "no_scale_transform_policy"


def not_tiny_neutral_band(label_policy: str, row_count: int, label_summary: dict[str, Any]) -> bool:
    if "neutral_band" not in label_policy:
        return True
    min_class_count = int(label_summary.get("min_class_count", 0) or 0)
    return row_count >= 100 and min_class_count >= 20


def is_near_collapse(distribution: dict[str, Any], threshold: float = 0.10) -> bool:
    total = sum_int_values(distribution)
    if total == 0:
        return True
    return min(int(distribution.get("0", 0) or 0), int(distribution.get("1", 0) or 0)) / total < threshold


def select_candidate_families(audit: dict[str, Any], max_families: int = 12) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in audit["candidates"]:
        if candidate_is_selectable(candidate):
            grouped[candidate["candidate_family"]].append(candidate)
    families = [build_family_record(key, values) for key, values in grouped.items()]
    families.sort(key=family_sort_key, reverse=True)
    return families[:max_families]


def candidate_is_selectable(candidate: dict[str, Any]) -> bool:
    gate = candidate.get("candidate_gate", {}).get("checks", {})
    return (
        gate.get("no_collapse") is True
        and gate.get("balanced_accuracy_beats_dummy_by_0_03") is True
        and gate.get("roc_auc_at_least_0_53") is True
        and gate.get("pr_auc_beats_prevalence_by_0_03") is True
        and gate.get("valid_prediction_contains_both_classes") is True
        and gate.get("no_leakage") is True
        and gate.get("no_artifact") is True
        and candidate.get("not_tiny_neutral_band_sample") is True
    )


def build_family_record(key: str, candidates: Sequence[dict[str, Any]]) -> dict[str, Any]:
    best = sorted(candidates, key=candidate_sort_key, reverse=True)[0]
    return {
        "candidate_family": key,
        "label_policy": best["label_policy"],
        "feature_set": best["feature_set"],
        "model_family": best["model_family"],
        "transform_policy": best["transform_policy"],
        "representative_model": choose_representative_model(best["model_family"], candidates),
        "candidate_count": len(candidates),
        "best_pr_auc_margin_over_prevalence": best.get("pr_auc_margin_over_prevalence"),
        "best_balanced_accuracy_margin_over_dummy": best.get("compared_to_dummy_most_frequent", {}).get("balanced_accuracy_delta"),
        "best_roc_auc": best.get("roc_auc"),
        "max_sample_size": max(int(item.get("row_count", 0)) for item in candidates),
        "source_candidates": list(candidates),
    }


def choose_representative_model(model_family_name: str, candidates: Sequence[dict[str, Any]]) -> str:
    candidate_models = {str(item.get("model")) for item in candidates}
    for model in MODEL_PRIORITY_BY_FAMILY.get(model_family_name, []):
        if model in candidate_models:
            return model
    return sorted(candidate_models)[0]


def candidate_sort_key(candidate: dict[str, Any]) -> tuple[float, float, float, int]:
    return (
        none_to_neg(candidate.get("pr_auc_margin_over_prevalence")),
        none_to_neg(candidate.get("compared_to_dummy_most_frequent", {}).get("balanced_accuracy_delta")),
        none_to_neg(candidate.get("roc_auc")),
        int(candidate.get("row_count", 0)),
    )


def family_sort_key(family: dict[str, Any]) -> tuple[float, float, float, int, int]:
    return (
        none_to_neg(family.get("best_pr_auc_margin_over_prevalence")),
        none_to_neg(family.get("best_balanced_accuracy_margin_over_dummy")),
        none_to_neg(family.get("best_roc_auc")),
        int(family.get("max_sample_size", 0)),
        1 if family.get("label_policy") not in {"three_day_positive_label", "label_ret3d_gt_0bp"} else 0,
    )


def build_time_splits(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    dates = sorted({str(row.get("trade_date", "")).strip() for row in rows if str(row.get("trade_date", "")).strip()})
    splits: list[dict[str, Any]] = []
    splits.append(ratio_split(dates, "anchor_date_70_30", 0.70))
    splits.append(ratio_split(dates, "anchor_date_60_40", 0.60))
    if len(dates) >= 4:
        chunk = max(1, len(dates) // 4)
        for fold in range(3):
            train_end = chunk * (fold + 1)
            valid_start = train_end
            valid_end = chunk * (fold + 2) if fold < 2 else len(dates)
            if valid_start < len(dates):
                splits.append(
                    {
                        "split_policy": "walk_forward_3fold_contiguous_anchor",
                        "split_id": f"walk_forward_3fold_contiguous_anchor_fold{fold + 1}",
                        "train_anchor_dates": dates[:train_end],
                        "valid_anchor_dates": dates[valid_start:valid_end],
                    }
                )
    return splits


def ratio_split(dates: Sequence[str], name: str, ratio: float) -> dict[str, Any]:
    if len(dates) < 2:
        train_count = len(dates)
    else:
        train_count = max(1, int(len(dates) * ratio))
        if train_count >= len(dates):
            train_count = len(dates) - 1
    return {
        "split_policy": name,
        "split_id": name,
        "train_anchor_dates": list(dates[:train_count]),
        "valid_anchor_dates": list(dates[train_count:]),
    }


def validate_split_policy(split_policy: str) -> None:
    if split_policy not in ALLOWED_SPLIT_POLICIES:
        raise Sprint2AuditError(f"random or unsupported split policy is not allowed: {split_policy}")


def run_robustness_validation(
    rows: Sequence[dict[str, Any]],
    feature_report: dict[str, Any],
    selected_families: Sequence[dict[str, Any]],
    out_dir: Path,
) -> dict[str, Any]:
    artifact_check_before = check_model_artifacts(out_dir)
    splits = build_time_splits(rows)
    feature_sets = feature_report.get("feature_set_variants", {})
    family_results: list[dict[str, Any]] = []
    p0_blockers: list[str] = []
    p0_blockers.extend(artifact_check_before["p0_blockers"])
    for split in splits:
        validate_split_policy(split["split_policy"])
    for family in selected_families:
        family_results.append(run_family_robustness(rows, feature_sets, family, splits))
    artifact_check_after = check_model_artifacts(out_dir)
    p0_blockers.extend(artifact_check_after["p0_blockers"])
    robust_families = [
        family for family in family_results if family["robustness_decision"] == ROBUST_CANDIDATE
    ]
    return {
        "report_scope": "no_save_candidate_family_robustness_diagnostic",
        "split_policies": ["anchor_date_70_30", "anchor_date_60_40", "walk_forward_3fold_contiguous_anchor"],
        "seed_values": SEEDS,
        "selected_family_count": len(selected_families),
        "family_results": family_results,
        "robust_family_count": len(robust_families),
        "robust_diagnostic_candidates": robust_families,
        "artifact_check_before": artifact_check_before,
        "artifact_check_after": artifact_check_after,
        "p0_blockers": dedupe(p0_blockers),
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
    }


def run_family_robustness(
    rows: Sequence[dict[str, Any]],
    feature_sets: dict[str, Any],
    family: dict[str, Any],
    splits: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    feature_set_payload = feature_sets.get(family["feature_set"], {})
    feature_columns = string_list(feature_set_payload.get("feature_columns"))
    if not feature_columns:
        feature_columns = string_list(feature_set_payload.get("feature_columns", []))
    model_name = family["representative_model"]
    split_results: list[dict[str, Any]] = []
    seed_results: list[dict[str, Any]] = []
    for split in splits:
        split_payload = build_split_payload(
            rows,
            family["label_policy"],
            feature_columns,
            split["train_anchor_dates"],
            split["valid_anchor_dates"],
        )
        if not split_payload["available"]:
            split_results.append({**split, "available": False, "reason": split_payload["reason"]})
            continue
        current_seed_results = [
            fit_family_on_split(model_name, split_payload, feature_columns, seed)
            for seed in SEEDS
        ]
        seed_results.extend([{**result, "split_id": split["split_id"], "split_policy": split["split_policy"]} for result in current_seed_results])
        split_results.append(
            summarize_split_result(split, current_seed_results, split_payload)
        )
    gate = robustness_gate(split_results, seed_results)
    return {
        **{key: family[key] for key in ("candidate_family", "label_policy", "feature_set", "model_family", "transform_policy", "representative_model")},
        "split_count_attempted": len(splits),
        "split_count_available": sum(1 for item in split_results if item.get("available") is True),
        "seed_count": len(SEEDS),
        "split_results": split_results,
        "seed_results": seed_results,
        "summary": summarize_seed_results(seed_results),
        "robustness_gate": gate,
        "robustness_decision": ROBUST_CANDIDATE if gate["robust_diagnostic_signal_candidate"] else WEAK_CANDIDATE,
        "artifact_check": {"passed": True, "model_artifact_created": False, "scaler_artifact_created": False},
        "boundary_check": build_boundary_check(),
    }


def build_split_payload(
    rows: Sequence[dict[str, Any]],
    label_policy: str,
    feature_columns: Sequence[str],
    train_dates: Sequence[str],
    valid_dates: Sequence[str],
) -> dict[str, Any]:
    train_set = set(train_dates)
    valid_set = set(valid_dates)
    train_rows = [row for row in rows if str(row.get("trade_date", "")).strip() in train_set]
    valid_rows = [row for row in rows if str(row.get("trade_date", "")).strip() in valid_set]
    x_train, y_train, train_dropped = rows_to_matrix_and_labels(train_rows, feature_columns, label_policy)
    x_valid, y_valid, valid_dropped = rows_to_matrix_and_labels(valid_rows, feature_columns, label_policy)
    reasons = []
    if not x_train or not x_valid:
        reasons.append("empty train or valid after null filtering")
    if len(set(y_train)) < 2:
        reasons.append("train split lacks two classes")
    if len(set(y_valid)) < 2:
        reasons.append("valid split lacks two classes")
    return {
        "available": not reasons,
        "reason": "; ".join(reasons),
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "feature_columns": list(feature_columns),
        "train_rows_used": len(x_train),
        "valid_rows_used": len(x_valid),
        "train_rows_dropped": train_dropped,
        "valid_rows_dropped": valid_dropped,
        "train_label_distribution": distribution_from_values(y_train),
        "valid_label_distribution": distribution_from_values(y_valid),
        "prevalence": safe_div(float(sum(y_valid)), float(len(y_valid))) if y_valid else None,
    }


def rows_to_matrix_and_labels(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    label_policy: str,
) -> tuple[list[list[float]], list[int], int]:
    matrix: list[list[float]] = []
    labels: list[int] = []
    dropped = 0
    for row in rows:
        label = label_value(row.get(label_policy))
        if label is None:
            dropped += 1
            continue
        values: list[float] = []
        missing = False
        for feature in feature_columns:
            value = to_float(row.get(feature))
            if value is None:
                missing = True
                break
            values.append(value)
        if missing:
            dropped += 1
            continue
        matrix.append(values)
        labels.append(label)
    return matrix, labels, dropped


def fit_family_on_split(model_name: str, split: dict[str, Any], feature_columns: Sequence[str], seed: int) -> dict[str, Any]:
    x_train = split["x_train"]
    y_train = split["y_train"]
    x_valid = split["x_valid"]
    y_valid = split["y_valid"]
    dummy_most = fit_predict("dummy_most_frequent", x_train, y_train, x_valid, seed)
    dummy_strat = fit_predict("dummy_stratified", x_train, y_train, x_valid, seed)
    model_result = fit_predict(model_name, x_train, y_train, x_valid, seed, feature_columns)
    dummy_most_metrics = score_predictions(y_valid, dummy_most["predictions"], dummy_most["scores"])
    dummy_strat_metrics = score_predictions(y_valid, dummy_strat["predictions"], dummy_strat["scores"])
    metrics = score_predictions(y_valid, model_result["predictions"], model_result["scores"])
    collapse = detect_collapse(model_result["predictions"], dummy_most["predictions"])
    prevalence = split["prevalence"]
    return {
        "seed": seed,
        "metrics": metrics,
        "prediction_distribution": metrics["prediction_distribution"],
        "collapse_check": collapse,
        "prevalence": prevalence,
        "pr_auc_margin_over_prevalence": None if metrics["pr_auc"] is None or prevalence is None else metrics["pr_auc"] - prevalence,
        "balanced_accuracy_margin_over_dummy_most_frequent": metrics["balanced_accuracy"] - dummy_most_metrics["balanced_accuracy"],
        "balanced_accuracy_margin_over_dummy_stratified": metrics["balanced_accuracy"] - dummy_strat_metrics["balanced_accuracy"],
        "beats_dummy_most_frequent_by_0_03": metrics["balanced_accuracy"] >= dummy_most_metrics["balanced_accuracy"] + 0.03,
        "roc_auc_at_least_0_53": metrics["roc_auc"] is not None and metrics["roc_auc"] >= 0.53,
        "pr_auc_beats_prevalence_by_0_03": prevalence is not None and metrics["pr_auc"] is not None and metrics["pr_auc"] >= prevalence + 0.03,
    }


def fit_predict(
    model_name: str,
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    x_valid: Sequence[Sequence[float]],
    seed: int,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    x_train_model = [list(row) for row in x_train]
    x_valid_model = [list(row) for row in x_valid]
    if model_name == "dummy_most_frequent":
        model = DummyClassifier(strategy="most_frequent")
    elif model_name == "dummy_stratified":
        model = DummyClassifier(strategy="stratified", random_state=seed)
    elif model_name in {"logistic_balanced_scaled", "logistic_log1p_scaled_balanced"}:
        if model_name == "logistic_log1p_scaled_balanced":
            x_train_model, x_valid_model = apply_log1p_to_flow_features(x_train_model, x_valid_model, feature_columns or [])
        scaler = StandardScaler()
        x_train_model = scaler.fit_transform(x_train_model).tolist()
        x_valid_model = scaler.transform(x_valid_model).tolist()
        model = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced", random_state=seed)
    elif model_name == "random_forest_shallow_no_save":
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed,
            n_jobs=1,
        )
    elif model_name == "hist_gradient_boosting_no_save":
        model = HistGradientBoostingClassifier(
            max_iter=50,
            max_leaf_nodes=7,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
        )
    elif model_name == "lightgbm_no_save":
        ensure_optional_model("lightgbm_no_save")
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
            device_type="cpu",
        )
    elif model_name == "xgboost_no_save":
        ensure_optional_model("xgboost_no_save")
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
        )
    elif model_name == "catboost_no_save":
        ensure_optional_model("catboost_no_save")
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=30,
            depth=3,
            learning_rate=0.05,
            random_seed=seed,
            verbose=False,
            task_type="CPU",
            allow_writing_files=False,
        )
    else:
        raise Sprint2AuditError(f"unknown model: {model_name}")
    model.fit(x_train_model, y_train)
    predictions = [int(value) for value in model.predict(x_valid_model)]
    scores = positive_scores(model, x_valid_model, predictions)
    return {"predictions": predictions, "scores": scores}


def apply_log1p_to_flow_features(
    x_train: Sequence[Sequence[float]],
    x_valid: Sequence[Sequence[float]],
    feature_columns: Sequence[str],
) -> tuple[list[list[float]], list[list[float]]]:
    indices = {
        index
        for index, feature in enumerate(feature_columns)
        if is_raw_flow_feature(feature)
    }

    def transform(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
        output: list[list[float]] = []
        for row in matrix:
            output.append(
                [
                    math.log1p(value) if index in indices and value >= 0 else value
                    for index, value in enumerate(row)
                ]
            )
        return output

    return transform(x_train), transform(x_valid)


def is_raw_flow_feature(feature: str) -> bool:
    name = feature.lower()
    return ("volume" in name or "amount" in name) and not any(token in name for token in ("ratio", "rank", "relative"))


def ensure_optional_model(model_name: str) -> None:
    module_name = {
        "lightgbm_no_save": "lightgbm",
        "xgboost_no_save": "xgboost",
        "catboost_no_save": "catboost",
    }.get(model_name)
    if module_name and importlib.util.find_spec(module_name) is None:
        raise Sprint2AuditError(f"optional model unavailable: {model_name}")


def positive_scores(model: Any, x_valid: Sequence[Sequence[float]], predictions: Sequence[int]) -> list[float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_valid)
        return [float(row[1]) for row in probabilities]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_valid)
        return [float(value) for value in scores]
    return [float(value) for value in predictions]


def score_predictions(y_valid: Sequence[int], predictions: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    roc_auc = None
    pr_auc = None
    if len(set(y_valid)) == 2:
        roc_auc = float(roc_auc_score(y_valid, scores))
        pr_auc = float(average_precision_score(y_valid, scores))
    return {
        "accuracy": float(accuracy_score(y_valid, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_valid, predictions)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": float(precision_score(y_valid, predictions, zero_division=0)),
        "recall": float(recall_score(y_valid, predictions, zero_division=0)),
        "prediction_distribution": distribution_from_values(predictions),
    }


def detect_collapse(predictions: Sequence[int], dummy_most_frequent_predictions: Sequence[int] | None = None) -> dict[str, Any]:
    distribution = distribution_from_values(predictions)
    contains_both = distribution["0"] > 0 and distribution["1"] > 0
    matches_dummy = bool(dummy_most_frequent_predictions) and list(predictions) == list(dummy_most_frequent_predictions)
    return {
        "collapse_flag": (not contains_both) or matches_dummy,
        "prediction_distribution": distribution,
        "valid_prediction_contains_both_classes": contains_both,
        "matches_dummy_most_frequent_predictions": matches_dummy,
    }


def summarize_split_result(split: dict[str, Any], seed_results: Sequence[dict[str, Any]], split_payload: dict[str, Any]) -> dict[str, Any]:
    return {
        **split,
        "available": True,
        "train_rows_used": split_payload["train_rows_used"],
        "valid_rows_used": split_payload["valid_rows_used"],
        "train_label_distribution": split_payload["train_label_distribution"],
        "valid_label_distribution": split_payload["valid_label_distribution"],
        "no_collapse_all_seeds": all(not item["collapse_check"]["collapse_flag"] for item in seed_results),
        "balanced_accuracy_beats_dummy_seed_count": sum(1 for item in seed_results if item["beats_dummy_most_frequent_by_0_03"]),
        "roc_auc_pass_seed_count": sum(1 for item in seed_results if item["roc_auc_at_least_0_53"]),
        "pr_auc_margin_pass_seed_count": sum(1 for item in seed_results if item["pr_auc_beats_prevalence_by_0_03"]),
        "metrics": summarize_seed_results(seed_results),
    }


def summarize_seed_results(seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not seed_results:
        return empty_metric_summary()
    return {
        "no_collapse_rate": mean([0.0 if item["collapse_check"]["collapse_flag"] else 1.0 for item in seed_results]),
        "balanced_accuracy": metric_stats([item["metrics"]["balanced_accuracy"] for item in seed_results]),
        "roc_auc": metric_stats([item["metrics"]["roc_auc"] for item in seed_results]),
        "pr_auc": metric_stats([item["metrics"]["pr_auc"] for item in seed_results]),
        "pr_auc_margin_over_prevalence": metric_stats([item["pr_auc_margin_over_prevalence"] for item in seed_results]),
        "balanced_accuracy_margin_over_dummy_most_frequent": metric_stats(
            [item["balanced_accuracy_margin_over_dummy_most_frequent"] for item in seed_results]
        ),
        "valid_class_diversity": {
            "all_seed_predictions_contain_both_classes": all(
                item["collapse_check"]["valid_prediction_contains_both_classes"] for item in seed_results
            )
        },
    }


def metric_stats(values: Sequence[Any]) -> dict[str, float | None]:
    numeric = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not numeric:
        return {"mean": None, "min": None, "max": None}
    return {"mean": mean(numeric), "min": min(numeric), "max": max(numeric)}


def empty_metric_summary() -> dict[str, Any]:
    return {
        "no_collapse_rate": 0.0,
        "balanced_accuracy": {"mean": None, "min": None, "max": None},
        "roc_auc": {"mean": None, "min": None, "max": None},
        "pr_auc": {"mean": None, "min": None, "max": None},
        "pr_auc_margin_over_prevalence": {"mean": None, "min": None, "max": None},
        "balanced_accuracy_margin_over_dummy_most_frequent": {"mean": None, "min": None, "max": None},
        "valid_class_diversity": {"all_seed_predictions_contain_both_classes": False},
    }


def robustness_gate(split_results: Sequence[dict[str, Any]], seed_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    available = [item for item in split_results if item.get("available") is True]
    split_ids_with_ba = {
        item["split_id"] for item in seed_results if item["beats_dummy_most_frequent_by_0_03"]
    }
    split_ids_with_roc = {
        item["split_id"] for item in seed_results if item["roc_auc_at_least_0_53"]
    }
    split_ids_with_pr = {
        item["split_id"] for item in seed_results if item["pr_auc_beats_prevalence_by_0_03"]
    }
    checks = {
        "at_least_2_available_time_splits": len(available) >= 2,
        "no_collapse_in_all_available_splits": bool(available)
        and all(item.get("no_collapse_all_seeds") is True for item in available),
        "balanced_accuracy_passes_at_least_2_splits": len(split_ids_with_ba) >= 2,
        "roc_auc_passes_at_least_2_splits": len(split_ids_with_roc) >= 2,
        "pr_auc_margin_passes_at_least_2_splits": len(split_ids_with_pr) >= 2,
        "no_model_or_scaler_artifacts": True,
        "no_leakage": True,
    }
    return {
        "robust_diagnostic_signal_candidate": all(checks.values()),
        "checks": checks,
        "available_split_ids": [item["split_id"] for item in available],
        "formal_model_evidence": False,
    }


def build_boundary_check() -> dict[str, Any]:
    return {
        "passed": True,
        "access_mode": "READ_ONLY",
        "final_action_change_allowed": False,
        "contains_live_order": False,
        "contains_secret": False,
        "requires_human_review": True,
        "promotion_gate_required": True,
        **BOUNDARY_FALSE_FIELDS,
    }


def build_decision(robustness_report: dict[str, Any], manifest_check: dict[str, Any]) -> dict[str, Any]:
    p0_blockers = dedupe([
        *manifest_check.get("p0_blockers", []),
        *robustness_report.get("p0_blockers", []),
    ])
    if p0_blockers:
        decision = DECISION_LEAKAGE if any("feature_columns" in blocker or "manifest" in blocker for blocker in p0_blockers) else DECISION_RUNTIME
    elif robustness_report.get("robust_family_count", 0) > 0:
        decision = DECISION_ROBUST
    else:
        decision = DECISION_WEAK
    return {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "sprint2_decision": decision,
        "robust_family_count": robustness_report.get("robust_family_count", 0),
        "selected_family_count": robustness_report.get("selected_family_count", 0),
        "robust_diagnostic_candidates": robustness_report.get("robust_diagnostic_candidates", []),
        "p0_blockers": p0_blockers,
        "p1_warnings": [
            "P1_DIAGNOSTIC_CANDIDATES_NOT_FORMAL_MODEL_EVIDENCE",
            "P1_SPLIT_ROBUSTNESS_REQUIRES_HUMAN_REVIEW",
            "P1_NO_STABLE_PROMOTION_WITHOUT_PROMOTION_GATE",
        ],
        "access_mode": "READ_ONLY",
        "final_action_change_allowed": False,
        "contains_live_order": False,
        "contains_secret": False,
        "requires_human_review": True,
        "promotion_gate_required": True,
        **BOUNDARY_FALSE_FIELDS,
    }


def run_sprint2(
    samples_path: Path,
    manifest_path: Path,
    label_report_path: Path,
    feature_report_path: Path,
    diagnostic_report_path: Path,
    decision_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_paths = {
        "samples": resolve_repo_path(samples_path, repo_root),
        "manifest": resolve_repo_path(manifest_path, repo_root),
        "label_report": resolve_repo_path(label_report_path, repo_root),
        "feature_report": resolve_repo_path(feature_report_path, repo_root),
        "diagnostic_report": resolve_repo_path(diagnostic_report_path, repo_root),
        "decision": resolve_repo_path(decision_path, repo_root),
    }
    for label, path in resolved_paths.items():
        if not path.exists():
            raise Sprint2AuditError(f"{label} path does not exist: {path}")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    rows, _ = load_csv_rows(resolved_paths["samples"])
    label_report = load_json(resolved_paths["label_report"])
    feature_report = load_json(resolved_paths["feature_report"])
    diagnostic_report = load_json(resolved_paths["diagnostic_report"])
    sprint1_decision = load_json(resolved_paths["decision"])
    manifest_check = check_label_manifest(resolved_paths["manifest"]).to_summary()

    audit = audit_candidates(diagnostic_report, label_report)
    selected_families = select_candidate_families(audit)
    robustness = {
        "report_scope": "blocked_before_runtime",
        "selected_family_count": len(selected_families),
        "family_results": [],
        "robust_family_count": 0,
        "robust_diagnostic_candidates": [],
        "p0_blockers": manifest_check.get("p0_blockers", []),
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
    }
    if not manifest_check.get("p0_blockers"):
        robustness = run_robustness_validation(rows, feature_report, selected_families, resolved_out_dir)

    audit_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "signal_recovery_sprint2_candidate_audit_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sprint1_decision": sprint1_decision.get("sprint_decision"),
        "sprint1_candidate_count": sprint1_decision.get("diagnostic_candidate_count"),
        "audit": audit,
        "selected_candidate_families": selected_families,
        "manifest_leakage_check": manifest_check,
        "formal_model_evidence": False,
        "stable_promotion_ready": False,
        "not_trading_advice": True,
    }
    robustness_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "signal_recovery_sprint2_robustness_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_model_evidence": False,
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "not_trading_advice": True,
        **robustness,
    }
    decision = build_decision(robustness_report, manifest_check)

    write_json(resolved_out_dir / "signal_recovery_sprint2_candidate_audit_report.json", audit_report)
    write_json(resolved_out_dir / "signal_recovery_sprint2_robustness_report.json", robustness_report)
    write_json(resolved_out_dir / "signal_recovery_sprint2_decision.json", decision)
    write_csv(
        resolved_out_dir / "signal_recovery_sprint2_candidate_family_summary.csv",
        family_summary_rows(selected_families, robustness_report),
        [
            "candidate_family",
            "label_policy",
            "feature_set",
            "model_family",
            "transform_policy",
            "representative_model",
            "candidate_count",
            "best_pr_auc_margin_over_prevalence",
            "best_balanced_accuracy_margin_over_dummy",
            "best_roc_auc",
            "max_sample_size",
            "robustness_decision",
            "split_count_available",
        ],
    )
    write_markdown_reports(resolved_out_dir, audit_report, robustness_report, decision)
    return {"audit_report": audit_report, "robustness_report": robustness_report, "decision": decision}


def family_summary_rows(selected_families: Sequence[dict[str, Any]], robustness_report: dict[str, Any]) -> list[dict[str, Any]]:
    robustness_by_family = {
        item["candidate_family"]: item for item in robustness_report.get("family_results", [])
    }
    rows: list[dict[str, Any]] = []
    for family in selected_families:
        robust = robustness_by_family.get(family["candidate_family"], {})
        rows.append(
            {
                "candidate_family": family["candidate_family"],
                "label_policy": family["label_policy"],
                "feature_set": family["feature_set"],
                "model_family": family["model_family"],
                "transform_policy": family["transform_policy"],
                "representative_model": family["representative_model"],
                "candidate_count": family["candidate_count"],
                "best_pr_auc_margin_over_prevalence": family["best_pr_auc_margin_over_prevalence"],
                "best_balanced_accuracy_margin_over_dummy": family["best_balanced_accuracy_margin_over_dummy"],
                "best_roc_auc": family["best_roc_auc"],
                "max_sample_size": family["max_sample_size"],
                "robustness_decision": robust.get("robustness_decision", ""),
                "split_count_available": robust.get("split_count_available", ""),
            }
        )
    return rows


def write_markdown_reports(out_dir: Path, audit_report: dict[str, Any], robustness_report: dict[str, Any], decision: dict[str, Any]) -> None:
    audit = audit_report["audit"]
    audit_md = [
        LAB_DECLARATION,
        "",
        "# Signal Recovery Sprint2 Candidate Audit",
        "",
        "This is a Lab-only candidate audit, not formal training, not trading advice, and not Stable evidence.",
        "",
        f"- sprint1_candidate_count: {audit_report.get('sprint1_candidate_count')}",
        f"- audited_candidate_count: {audit['candidate_count']}",
        f"- selected_candidate_families: {len(audit_report['selected_candidate_families'])}",
        f"- weak_pr_auc_margin_count: {len(audit['candidates_with_weak_pr_auc_margin'])}",
        f"- near_collapse_count: {len(audit['candidates_with_near_collapse_prediction_distribution'])}",
        f"- formal_model_evidence: {str(audit_report['formal_model_evidence']).lower()}",
        f"- stable_promotion_ready: {str(audit_report['stable_promotion_ready']).lower()}",
    ]
    (out_dir / "signal_recovery_sprint2_candidate_audit_report.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")
    robustness_md = [
        LAB_DECLARATION,
        "",
        "# Signal Recovery Sprint2 Robustness Report",
        "",
        "No-save robustness diagnostic only; models are not persisted and metrics are not formal evidence.",
        "",
        f"- selected_family_count: {robustness_report.get('selected_family_count')}",
        f"- robust_family_count: {robustness_report.get('robust_family_count')}",
        f"- decision: {decision['sprint2_decision']}",
        f"- model_saved: {str(robustness_report.get('model_saved', False)).lower()}",
        f"- scaler_saved: {str(robustness_report.get('scaler_saved', False)).lower()}",
        f"- checkpoint_saved: {str(robustness_report.get('checkpoint_saved', False)).lower()}",
        f"- qmt_used: {str(robustness_report.get('qmt_used', False)).lower()}",
        f"- order_intent_generated: {str(robustness_report.get('order_intent_generated', False)).lower()}",
        f"- stable_affected: {str(robustness_report.get('stable_affected', False)).lower()}",
    ]
    (out_dir / "signal_recovery_sprint2_robustness_report.md").write_text("\n".join(robustness_md) + "\n", encoding="utf-8")


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def label_value(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    label = int(number)
    return label if label in (0, 1) else None


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def sum_int_values(payload: dict[str, Any]) -> int:
    return sum(int(value or 0) for value in payload.values())


def distribution_from_values(values: Sequence[int]) -> dict[str, int]:
    return {"0": sum(1 for value in values if int(value) == 0), "1": sum(1 for value in values if int(value) == 1)}


def none_to_neg(value: Any) -> float:
    number = to_float(value)
    return -1e9 if number is None else number


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLES)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--label-report", type=Path, default=DEFAULT_LABEL_REPORT)
    parser.add_argument("--feature-report", type=Path, default=DEFAULT_FEATURE_REPORT)
    parser.add_argument("--diagnostic-report", type=Path, default=DEFAULT_DIAGNOSTIC_REPORT)
    parser.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    parser.add_argument("--out-dir", type=Path, default=ALLOWED_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_sprint2(
            args.samples,
            args.manifest,
            args.label_report,
            args.feature_report,
            args.diagnostic_report,
            args.decision,
            args.out_dir,
        )
    except Exception as exc:  # noqa: BLE001 - CLI must emit auditable blocker.
        print(
            json.dumps(
                {
                    "lab_declaration": LAB_DECLARATION,
                    "status": "failed",
                    "sprint2_decision": DECISION_RUNTIME,
                    "p0_blockers": [str(exc)],
                    "formal_model_evidence": False,
                    "stable_promotion_ready": False,
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
                "status": "completed",
                "sprint2_decision": payload["decision"]["sprint2_decision"],
                "selected_family_count": payload["robustness_report"]["selected_family_count"],
                "robust_family_count": payload["robustness_report"]["robust_family_count"],
                "formal_model_evidence": False,
                "stable_promotion_ready": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
