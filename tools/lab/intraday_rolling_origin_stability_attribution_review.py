from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_fixed_shortlist_oop_no_save_validation import (  # noqa: E402
    load_json,
    to_float,
    write_csv,
    write_json,
)
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts  # noqa: E402


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_rolling_origin_stability_attribution_review"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_rolling_origin_stability_attribution_review")
DEFAULT_ROLLING_ORIGIN_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_rolling_origin_walk_forward_no_save_validation")
DEFAULT_FIXED_OOP_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_fixed_shortlist_oop_no_save_validation")
DEFAULT_OUT_DIR = ALLOWED_OUTPUT_DIR
FOCUS_FAMILY_ID = "label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy"
FOCUS_MODEL = "logistic_balanced_scaled"
THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]

DECISION_MONITOR_READY = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_LAB_MONITOR_CANDIDATE_REVIEW_READY"
DECISION_WEAK = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_WEAK_STABILITY_REVIEW_REQUIRED"
DECISION_MONTH = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_MONTH_CONCENTRATION_OBSERVED"
DECISION_ETF = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_ETF_CONCENTRATION_OBSERVED"
DECISION_THRESHOLD = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_THRESHOLD_SENSITIVITY_OBSERVED"
DECISION_PROTOCOL = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_PROTOCOL_CONFLICT_REVIEW_REQUIRED"
DECISION_NO_STABILITY = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_NO_STABILITY_AFTER_REVIEW"
DECISION_BLOCKED_MISSING = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_BLOCKED_MISSING_OUTPUTS"
DECISION_BLOCKED_DATA = "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_BLOCKED_DATA_QUALITY"

CLASS_MONITOR_READY = "LAB_MONITOR_CANDIDATE_REVIEW_READY"
CLASS_WEAK = "WEAK_DIAGNOSTIC_STABILITY_REVIEW_REQUIRED"
CLASS_MONTH = "MONTH_CONCENTRATION_OBSERVED_REVIEW_REQUIRED"
CLASS_ETF = "ETF_CONCENTRATION_OBSERVED_REVIEW_REQUIRED"
CLASS_THRESHOLD = "THRESHOLD_SENSITIVITY_OBSERVED_REVIEW_REQUIRED"
CLASS_PROTOCOL = "PROTOCOL_CONFLICT_REVIEW_REQUIRED"
CLASS_NO_STABILITY = "NO_ROLLING_ORIGIN_STABILITY_AFTER_ATTRIBUTION"
CLASS_BLOCKED = "BLOCKED_MISSING_ROLLING_ORIGIN_OUTPUTS"


class StabilityAttributionError(RuntimeError):
    pass


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def resolve_output_dir(out_dir: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    repo_root = repo_root.resolve()
    resolved = resolve_repo_path(out_dir, repo_root).resolve()
    if enforce:
        allowed = (repo_root / ALLOWED_OUTPUT_DIR).resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError as exc:
            raise StabilityAttributionError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_dicts(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader), list(reader.fieldnames or [])
    except OSError as exc:
        raise StabilityAttributionError(f"CSV cannot be read: {path}: {exc}") from exc


def require_files(paths: Sequence[Path]) -> list[str]:
    return [str(path) for path in paths if not path.exists()]


def numeric(value: Any) -> float | None:
    return to_float(value)


def bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def label_int(value: Any) -> int | None:
    number = numeric(value)
    if number is None:
        return None
    label = int(number)
    return label if label in (0, 1) else None


def extract_winning_candidate(report: dict[str, Any]) -> dict[str, Any]:
    aggregates = report.get("aggregate_stability") or []
    for item in aggregates:
        if item.get("diagnostic_stability_observed") is True:
            return item
    for item in aggregates:
        if item.get("family_id") == FOCUS_FAMILY_ID:
            return item
    raise StabilityAttributionError("winning candidate not found in rolling-origin report")


def fold_robustness_rows(
    metrics_rows: Sequence[dict[str, str]],
    fold_manifest: Sequence[dict[str, Any]],
    family_id: str = FOCUS_FAMILY_ID,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest_by_fold = {str(row.get("fold_id")): row for row in fold_manifest}
    by_fold: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in metrics_rows:
        if row.get("family_id") == family_id:
            by_fold[str(row.get("fold_id"))][str(row.get("model"))] = row
    output: list[dict[str, Any]] = []
    for fold_id, models in sorted(by_fold.items()):
        if FOCUS_MODEL not in models:
            continue
        logistic = models[FOCUS_MODEL]
        dummy_most = models.get("dummy_most_frequent", {})
        dummy_strat = models.get("dummy_stratified", {})
        dummy_most_ba = numeric(dummy_most.get("balanced_accuracy"))
        dummy_strat_ba = numeric(dummy_strat.get("balanced_accuracy"))
        dummy_values = [value for value in [dummy_most_ba, dummy_strat_ba] if value is not None]
        best_dummy_ba = max(dummy_values) if dummy_values else None
        row_count = int(numeric(logistic.get("row_count")) or 0)
        prediction_1 = int(numeric(logistic.get("prediction_1")) or 0)
        ba = numeric(logistic.get("balanced_accuracy"))
        advantage = ba - best_dummy_ba if ba is not None and best_dummy_ba is not None else None
        fold_manifest_row = manifest_by_fold.get(fold_id, {})
        output.append(
            {
                "fold_id": fold_id,
                "cutoff": logistic.get("cutoff", ""),
                "validation_window": logistic.get("validation_month", ""),
                "validation_anchor_count": fold_manifest_row.get("validation_anchor_count"),
                "validation_group_count": row_count,
                "label_prevalence": numeric(logistic.get("label_prevalence")),
                "balanced_accuracy": ba,
                "roc_auc": numeric(logistic.get("roc_auc")),
                "pr_auc": numeric(logistic.get("pr_auc")),
                "accuracy": numeric(logistic.get("accuracy")),
                "dummy_most_frequent_balanced_accuracy": dummy_most_ba,
                "dummy_stratified_balanced_accuracy": dummy_strat_ba,
                "best_dummy_balanced_accuracy": best_dummy_ba,
                "advantage_over_best_dummy": advantage,
                "above_dummy": advantage is not None and advantage > 0,
                "prediction_positive_rate": prediction_1 / row_count if row_count else None,
                "probability_mean": numeric(logistic.get("probability_mean")),
                "probability_min": numeric(logistic.get("probability_min")),
                "probability_max": numeric(logistic.get("probability_max")),
                "collapse_flag": bool_value(logistic.get("single_class_prediction_collapse")),
                "tn": int(numeric(logistic.get("tn")) or 0),
                "fp": int(numeric(logistic.get("fp")) or 0),
                "fn": int(numeric(logistic.get("fn")) or 0),
                "tp": int(numeric(logistic.get("tp")) or 0),
            }
        )
    positive_advantages = [max(0.0, row["advantage_over_best_dummy"] or 0.0) for row in output]
    total_positive_advantage = sum(positive_advantages)
    sorted_advantages = sorted(positive_advantages, reverse=True)
    top1_share = sorted_advantages[0] / total_positive_advantage if total_positive_advantage else 0.0
    top2_share = sum(sorted_advantages[:2]) / total_positive_advantage if total_positive_advantage else 0.0
    concentration = top1_share > 0.35 or top2_share > 0.60
    for row in output:
        contribution = max(0.0, row["advantage_over_best_dummy"] or 0.0)
        row["positive_advantage_share"] = contribution / total_positive_advantage if total_positive_advantage else 0.0
    best = max(output, key=lambda row: row["balanced_accuracy"] or -1) if output else None
    worst = min(output, key=lambda row: row["balanced_accuracy"] or 1e9) if output else None
    return output, {
        "fold_count": len(output),
        "positive_fold_count": sum(1 for row in output if (row["balanced_accuracy"] or 0) > 0.5),
        "below_baseline_fold_count": sum(1 for row in output if not row["above_dummy"]),
        "best_fold": best,
        "worst_fold": worst,
        "month_concentration_observed": concentration,
        "top1_positive_advantage_share": top1_share,
        "top2_positive_advantage_share": top2_share,
    }


def month_degradation_rows(fold_rows: Sequence[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in fold_rows:
        errors = int(row["fp"]) + int(row["fn"])
        group_count = int(row["validation_group_count"] or 0)
        degrade = (row["balanced_accuracy"] or 0) <= 0.5 or not row["above_dummy"]
        dominates = row.get("positive_advantage_share", 0.0) > 0.35
        output.append(
            {
                "month": row["validation_window"],
                "cutoff": row["cutoff"],
                "validation_group_count": group_count,
                "label_prevalence": row["label_prevalence"],
                "balanced_accuracy": row["balanced_accuracy"],
                "roc_auc": row["roc_auc"],
                "pr_auc": row["pr_auc"],
                "accuracy": row["accuracy"],
                "prediction_positive_rate": row["prediction_positive_rate"],
                "error_rate": errors / group_count if group_count else None,
                "advantage_over_best_dummy": row["advantage_over_best_dummy"],
                "positive_advantage_share": row.get("positive_advantage_share", 0.0),
                "whether_degradation_month": degrade,
                "whether_month_dominates_decision": dominates,
            }
        )
    return output, {
        "degradation_month_count": sum(1 for row in output if row["whether_degradation_month"]),
        "month_dominates_decision": any(row["whether_month_dominates_decision"] for row in output),
    }


def confusion_from_rows(rows: Sequence[dict[str, Any]], *, threshold: float | None = None) -> dict[str, int]:
    counts = {"TP": 0, "TN": 0, "FP": 0, "FN": 0, "NA": 0}
    for row in rows:
        label = label_int(row.get("label"))
        if threshold is None:
            prediction = label_int(row.get("prediction"))
        else:
            probability = numeric(row.get("probability"))
            prediction = None if probability is None else int(probability >= threshold)
        if label is None or prediction is None:
            counts["NA"] += 1
        elif label == 1 and prediction == 1:
            counts["TP"] += 1
        elif label == 0 and prediction == 0:
            counts["TN"] += 1
        elif label == 0 and prediction == 1:
            counts["FP"] += 1
        elif label == 1 and prediction == 0:
            counts["FN"] += 1
    return counts


def balanced_accuracy(counts: dict[str, int]) -> float | None:
    pos_total = counts["TP"] + counts["FN"]
    neg_total = counts["TN"] + counts["FP"]
    if not pos_total or not neg_total:
        return None
    return ((counts["TP"] / pos_total) + (counts["TN"] / neg_total)) / 2


def etf_dispersion_rows(row_level_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    focus_rows = [row for row in row_level_rows if row.get("family_id") == FOCUS_FAMILY_ID and row.get("model") == FOCUS_MODEL]
    total_errors = sum(1 for row in focus_rows if row.get("error_type") in {"FP", "FN"})
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in focus_rows:
        grouped[str(row.get("etf_code", ""))].append(row)
    output: list[dict[str, Any]] = []
    for etf_code, rows in sorted(grouped.items()):
        counts = confusion_from_rows(rows)
        group_count = len(rows)
        errors = counts["FP"] + counts["FN"]
        labels = [label_int(row.get("label")) for row in rows]
        predictions = [label_int(row.get("prediction")) for row in rows]
        output.append(
            {
                "etf_code": etf_code,
                "group_count": group_count,
                "balanced_accuracy": balanced_accuracy(counts),
                "error_rate": errors / group_count if group_count else None,
                "positive_label_rate": sum(value for value in labels if value is not None) / group_count if group_count else None,
                "positive_prediction_rate": sum(value for value in predictions if value is not None) / group_count if group_count else None,
                "fp": counts["FP"],
                "fn": counts["FN"],
                "tp": counts["TP"],
                "tn": counts["TN"],
                "error_share": errors / total_errors if total_errors else 0.0,
                "whether_etf_concentration_observed": False,
            }
        )
    sorted_error_share = sorted((row["error_share"] for row in output), reverse=True)
    top1 = sorted_error_share[0] if sorted_error_share else 0.0
    top2 = sum(sorted_error_share[:2]) if sorted_error_share else 0.0
    concentration = top1 > 0.35 or top2 > 0.60
    if concentration:
        max_share = top1
        for row in output:
            row["whether_etf_concentration_observed"] = row["error_share"] == max_share
    return output, {
        "row_level_available": bool(focus_rows),
        "etf_count": len(output),
        "total_focus_rows": len(focus_rows),
        "total_errors": total_errors,
        "top1_error_share": top1,
        "top2_error_share": top2,
        "etf_concentration_observed": concentration,
    }


def label_prevalence_rows(fold_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "month": row["validation_window"],
            "cutoff": row["cutoff"],
            "validation_group_count": row["validation_group_count"],
            "label_prevalence": row["label_prevalence"],
            "prediction_positive_rate": row["prediction_positive_rate"],
        }
        for row in fold_rows
    ]


def threshold_sensitivity_rows(row_level_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    focus_rows = [row for row in row_level_rows if row.get("family_id") == FOCUS_FAMILY_ID and row.get("model") == FOCUS_MODEL]
    required = {"fold_id", "probability", "label"}
    available = bool(focus_rows) and required <= set(focus_rows[0])
    if not available:
        return [], {
            "threshold_selection_allowed": False,
            "threshold_tuned_on_walk_forward": False,
            "threshold_sensitivity_is_diagnostic_only": True,
            "threshold_sensitivity_limited_by_missing_row_level_probability": True,
            "threshold_sensitivity_observed": False,
        }
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in focus_rows:
        grouped[str(row.get("fold_id"))].append(row)
    output: list[dict[str, Any]] = []
    for threshold in THRESHOLDS:
        fold_bas: list[float] = []
        for fold_id, rows in sorted(grouped.items()):
            counts = confusion_from_rows(rows, threshold=threshold)
            ba = balanced_accuracy(counts)
            if ba is not None:
                fold_bas.append(ba)
            output.append(
                {
                    "threshold": threshold,
                    "fold_id": fold_id,
                    "validation_month": rows[0].get("validation_month", ""),
                    "balanced_accuracy": ba,
                    "fp": counts["FP"],
                    "fn": counts["FN"],
                    "prediction_positive_rate": (counts["TP"] + counts["FP"]) / len(rows) if rows else None,
                }
            )
        output.append(
            {
                "threshold": threshold,
                "fold_id": "AGGREGATE",
                "validation_month": "ALL",
                "balanced_accuracy": mean(fold_bas) if fold_bas else None,
                "fp": "",
                "fn": "",
                "prediction_positive_rate": "",
                "stability_fraction": sum(1 for value in fold_bas if value > 0.5) / len(fold_bas) if fold_bas else None,
            }
        )
    aggregate_rows = [row for row in output if row["fold_id"] == "AGGREGATE"]
    baseline = next((row for row in aggregate_rows if row["threshold"] == 0.5), None)
    best = max(aggregate_rows, key=lambda row: row["balanced_accuracy"] or -1) if aggregate_rows else None
    baseline_ba = baseline["balanced_accuracy"] if baseline else None
    best_ba = best["balanced_accuracy"] if best else None
    sensitivity_observed = (
        baseline_ba is not None
        and best_ba is not None
        and best["threshold"] != 0.5
        and best_ba - baseline_ba >= 0.03
    )
    return output, {
        "threshold_selection_allowed": False,
        "threshold_tuned_on_walk_forward": False,
        "threshold_sensitivity_is_diagnostic_only": True,
        "threshold_sensitivity_limited_by_missing_row_level_probability": False,
        "threshold_sensitivity_observed": sensitivity_observed,
        "baseline_threshold": 0.5,
        "baseline_mean_balanced_accuracy": baseline_ba,
        "best_threshold_by_mean_balanced_accuracy": best["threshold"] if best else None,
        "best_mean_balanced_accuracy": best_ba,
    }


def protocol_reconciliation_rows(fixed_metric: dict[str, Any] | None, rolling_metric: dict[str, Any] | None) -> list[dict[str, Any]]:
    fixed_key_metric = fixed_metric.get("balanced_accuracy") if fixed_metric else None
    rolling_key_metric = rolling_metric.get("balanced_accuracy_mean") if rolling_metric else None
    return [
        {
            "protocol": "fixed-shortlist strict OOP",
            "train_direction": "Sprint3 discovery window fixed in calendar middle",
            "validation_direction": "non-overlap but includes backward-in-calendar pre-sprint plus small post-sprint",
            "leakage_status": "no anchor overlap, but pre-sprint is backward-in-calendar relative to discovery train window",
            "main_question_answered": "Does the frozen shortlist survive strict non-overlap around Sprint3 discovery?",
            "key_metric": fixed_key_metric,
            "risk": "pre-sprint combined OOP can answer a different question than deployment-style forward validation; post-sprint remains underpowered",
            "allowed_interpretation": "review-only strict OOP diagnostic context",
            "forbidden_interpretation": "cannot be used as automatic Stable evidence or to erase post-sprint underpowered blocker",
        },
        {
            "protocol": "rolling-origin walk-forward",
            "train_direction": "past-to-future expanding window",
            "validation_direction": "next calendar month after each cutoff",
            "leakage_status": "train labels known by cutoff and validation strictly after cutoff",
            "main_question_answered": "Would the fixed shortlist have shown deployment-like historical monthly stability?",
            "key_metric": rolling_key_metric,
            "risk": "still historical diagnostic; candidate was selected after Sprint3 research and cannot become Stable evidence without promotion gate",
            "allowed_interpretation": "Lab diagnostic monitor candidate if no concentration is observed",
            "forbidden_interpretation": "cannot directly override fixed strict OOP or trigger Stable promotion, QMT, or OrderIntent",
        },
    ]


def find_fixed_focus_metric(metrics_rows: Sequence[dict[str, str]]) -> dict[str, Any] | None:
    for row in metrics_rows:
        if row.get("family_id") == FOCUS_FAMILY_ID and row.get("model") == FOCUS_MODEL and row.get("split") == "combined_strict_oop":
            return {"balanced_accuracy": numeric(row.get("balanced_accuracy")), "roc_auc": numeric(row.get("roc_auc")), "pr_auc": numeric(row.get("pr_auc"))}
    return None


def classify_and_decide(
    fold_summary: dict[str, Any],
    month_summary: dict[str, Any],
    etf_summary: dict[str, Any],
    threshold_summary: dict[str, Any],
    rolling_candidate: dict[str, Any],
    blockers: Sequence[str],
) -> tuple[str, str]:
    if blockers:
        return CLASS_BLOCKED, DECISION_BLOCKED_MISSING if any("missing" in blocker.lower() for blocker in blockers) else DECISION_BLOCKED_DATA
    fold_count = int(rolling_candidate.get("evaluated_fold_count") or fold_summary.get("fold_count") or 0)
    if fold_count < 6:
        return CLASS_WEAK, DECISION_WEAK
    if fold_summary.get("month_concentration_observed") or month_summary.get("month_dominates_decision"):
        return CLASS_MONTH, DECISION_MONTH
    if etf_summary.get("etf_concentration_observed"):
        return CLASS_ETF, DECISION_ETF
    if threshold_summary.get("threshold_sensitivity_observed"):
        return CLASS_THRESHOLD, DECISION_THRESHOLD
    stability_fraction = rolling_candidate.get("fraction_folds_balanced_accuracy_above_0_5")
    if stability_fraction is not None and float(stability_fraction) < 0.6:
        return CLASS_NO_STABILITY, DECISION_NO_STABILITY
    return CLASS_MONITOR_READY, DECISION_MONITOR_READY


def build_docs_report(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    summary = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_rolling_origin_stability_attribution_review",
        "status": report["status"],
        "candidate_classification": report["candidate_classification"],
        "readiness_decision": report["readiness_decision"],
        "winning_candidate": report["winning_candidate"]["family_id"],
        "fold_count": report["fold_robustness_summary"]["fold_count"],
        "positive_fold_count": report["fold_robustness_summary"]["positive_fold_count"],
        "below_baseline_fold_count": report["fold_robustness_summary"]["below_baseline_fold_count"],
        "month_concentration_observed": report["fold_robustness_summary"]["month_concentration_observed"],
        "etf_concentration_observed": report["etf_dispersion_summary"]["etf_concentration_observed"],
        "threshold_sensitivity_observed": report["threshold_sensitivity_summary"]["threshold_sensitivity_observed"],
        "stable_promotion_ready": False,
        "stable_evidence": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday Rolling-Origin Stability Attribution Review",
        "",
        "Lab-only read-only attribution review. It reads existing rolling-origin outputs and fixed OOP summaries; it does not fit, tune, save model/scaler, connect QMT, generate OrderIntent, or create Stable evidence.",
        "",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- candidate_classification: {report['candidate_classification']}",
        f"- winning_candidate: {report['winning_candidate']['family_id']}",
        f"- fold_count: {summary['fold_count']}",
        f"- positive_fold_count: {summary['positive_fold_count']}",
        f"- below_baseline_fold_count: {summary['below_baseline_fold_count']}",
        f"- month_concentration_observed: {str(summary['month_concentration_observed']).lower()}",
        f"- etf_concentration_observed: {str(summary['etf_concentration_observed']).lower()}",
        f"- threshold_sensitivity_observed: {str(summary['threshold_sensitivity_observed']).lower()}",
        f"- stable_promotion_ready: false",
        f"- stable_evidence: false",
        "",
        "## Protocol Reconciliation",
        "",
        "Fixed strict OOP and rolling-origin walk-forward answer different diagnostic questions. Rolling-origin monitor readiness cannot override fixed post-sprint underpowered risk and cannot become Stable evidence without human promotion gate.",
    ]
    return summary, "\n".join(lines) + "\n"


def run_review(
    rolling_origin_dir: Path = DEFAULT_ROLLING_ORIGIN_DIR,
    fixed_oop_dir: Path = DEFAULT_FIXED_OOP_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    enforce_output_dir: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    rolling_dir = resolve_repo_path(rolling_origin_dir, repo_root)
    fixed_dir = resolve_repo_path(fixed_oop_dir, repo_root)
    resolved_out_dir = resolve_output_dir(out_dir, repo_root, enforce=enforce_output_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    rolling_report_path = rolling_dir / "rolling_origin_walk_forward_report.json"
    rolling_metrics_path = rolling_dir / "rolling_origin_fold_metrics.csv"
    rolling_rows_path = rolling_dir / "rolling_origin_row_level_predictions.csv"
    rolling_manifest_path = rolling_dir / "rolling_origin_fold_manifest.json"
    fixed_metrics_path = fixed_dir / "fixed_shortlist_oop_metrics.csv"
    missing = require_files([rolling_report_path, rolling_metrics_path, rolling_rows_path, rolling_manifest_path, fixed_metrics_path])
    blockers = [f"missing required rolling-origin/fixed OOP output: {path}" for path in missing]

    artifact_before = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_before["p0_blockers"])
    if blockers:
        report = blocked_report(rolling_origin_dir, fixed_oop_dir, blockers, artifact_before)
        emit_outputs(repo_root, resolved_out_dir, report, [], [], [], [], [], [])
        return report

    rolling_report = load_json(rolling_report_path)
    rolling_manifest_payload = load_json(rolling_manifest_path)
    rolling_metrics_rows, _ = load_csv_dicts(rolling_metrics_path)
    row_level_rows, row_level_columns = load_csv_dicts(rolling_rows_path)
    fixed_metrics_rows, _ = load_csv_dicts(fixed_metrics_path)
    winning_candidate = extract_winning_candidate(rolling_report)
    if winning_candidate.get("family_id") != FOCUS_FAMILY_ID:
        blockers.append("winning candidate differs from expected fixed shortlist focus candidate")

    fold_rows, fold_summary = fold_robustness_rows(rolling_metrics_rows, rolling_manifest_payload.get("fold_manifest", []), FOCUS_FAMILY_ID)
    month_rows, month_summary = month_degradation_rows(fold_rows)
    etf_rows, etf_summary = etf_dispersion_rows(row_level_rows)
    prevalence_rows = label_prevalence_rows(fold_rows)
    threshold_rows, threshold_summary = threshold_sensitivity_rows(row_level_rows)
    fixed_focus_metric = find_fixed_focus_metric(fixed_metrics_rows)
    protocol_rows = protocol_reconciliation_rows(fixed_focus_metric, winning_candidate)
    classification, readiness_decision = classify_and_decide(fold_summary, month_summary, etf_summary, threshold_summary, winning_candidate, blockers)
    artifact_after = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_after["p0_blockers"])
    if blockers and readiness_decision == DECISION_MONITOR_READY:
        classification, readiness_decision = CLASS_BLOCKED, DECISION_BLOCKED_DATA

    p1_warnings = [
        "P1_DIAGNOSTIC_ONLY_NOT_STABLE_EVIDENCE",
        "P1_REQUIRES_HUMAN_REVIEW",
        "P1_NO_STABLE_PROMOTION_WITHOUT_PROMOTION_GATE",
        "P1_ROLLING_ORIGIN_MONITOR_ONLY_IF_ACCEPTED_BY_HUMAN_REVIEW",
    ]
    if classification == CLASS_MONITOR_READY:
        p1_warnings.append("P1_LAB_MONITOR_CANDIDATE_REVIEW_READY_NOT_TRADING_ADVICE")
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": "blocked" if readiness_decision in {DECISION_BLOCKED_MISSING, DECISION_BLOCKED_DATA} else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "rolling_origin_dir": str(rolling_origin_dir),
            "fixed_oop_dir": str(fixed_oop_dir),
            "stable_bundle": False,
            "source": "existing Lab ignored rolling-origin and fixed OOP diagnostics",
        },
        "winning_candidate": winning_candidate,
        "fold_robustness_summary": fold_summary,
        "month_degradation_summary": month_summary,
        "etf_dispersion_summary": etf_summary,
        "threshold_sensitivity_summary": threshold_summary,
        "protocol_reconciliation_summary": {
            "fixed_combined_strict_oop_balanced_accuracy": fixed_focus_metric.get("balanced_accuracy") if fixed_focus_metric else None,
            "rolling_origin_balanced_accuracy_mean": winning_candidate.get("balanced_accuracy_mean"),
            "protocols_answer_different_questions": True,
            "rolling_origin_does_not_override_post_sprint_underpowered": True,
            "allowed_state": "Lab diagnostic monitor candidate" if classification == CLASS_MONITOR_READY else "review required",
        },
        "row_level_columns": row_level_columns,
        "candidate_classification": classification,
        "readiness_decision": readiness_decision,
        "p0_blockers": list(dict.fromkeys(blockers)),
        "p1_warnings": list(dict.fromkeys(p1_warnings)),
        "artifact_check_before": artifact_before,
        "artifact_check_after": artifact_after,
        "access_mode": "READ_ONLY",
        "final_action_change_allowed": False,
        "contains_live_order": False,
        "contains_secret": False,
        "requires_human_review": True,
        "promotion_gate_required": True,
        "formal_training": False,
        "formal_training_ready": False,
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "stable_promotion_ready": False,
        "automatic_promotion_ready": False,
        "qmt_ready": False,
        "qmt_used": False,
        "order_intent_ready": False,
        "order_intent_generated": False,
        "stable_evidence": False,
        "stable_affected": False,
        "advisory_package_created": False,
        "not_trading_advice": True,
        "threshold_selection_allowed": False,
        "threshold_tuned_on_walk_forward": False,
        "threshold_sensitivity_is_diagnostic_only": True,
    }
    emit_outputs(repo_root, resolved_out_dir, report, fold_rows, month_rows, etf_rows, prevalence_rows, threshold_rows, protocol_rows)
    return report


def blocked_report(
    rolling_origin_dir: Path,
    fixed_oop_dir: Path,
    blockers: Sequence[str],
    artifact_before: dict[str, Any],
) -> dict[str, Any]:
    return {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "rolling_origin_dir": str(rolling_origin_dir),
            "fixed_oop_dir": str(fixed_oop_dir),
            "stable_bundle": False,
        },
        "winning_candidate": {"family_id": FOCUS_FAMILY_ID},
        "fold_robustness_summary": {"fold_count": 0, "positive_fold_count": 0, "below_baseline_fold_count": 0, "month_concentration_observed": False},
        "month_degradation_summary": {"month_dominates_decision": False},
        "etf_dispersion_summary": {"etf_concentration_observed": False, "row_level_available": False},
        "threshold_sensitivity_summary": {
            "threshold_selection_allowed": False,
            "threshold_tuned_on_walk_forward": False,
            "threshold_sensitivity_is_diagnostic_only": True,
            "threshold_sensitivity_limited_by_missing_row_level_probability": True,
            "threshold_sensitivity_observed": False,
        },
        "candidate_classification": CLASS_BLOCKED,
        "readiness_decision": DECISION_BLOCKED_MISSING,
        "p0_blockers": list(dict.fromkeys(blockers)),
        "p1_warnings": ["P1_MISSING_ROLLING_ORIGIN_OUTPUTS_REVIEW_REQUIRED"],
        "artifact_check_before": artifact_before,
        "access_mode": "READ_ONLY",
        "final_action_change_allowed": False,
        "contains_live_order": False,
        "contains_secret": False,
        "requires_human_review": True,
        "promotion_gate_required": True,
        "formal_training": False,
        "formal_training_ready": False,
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "stable_promotion_ready": False,
        "automatic_promotion_ready": False,
        "qmt_ready": False,
        "qmt_used": False,
        "order_intent_ready": False,
        "order_intent_generated": False,
        "stable_evidence": False,
        "stable_affected": False,
        "advisory_package_created": False,
        "not_trading_advice": True,
    }


def emit_outputs(
    repo_root: Path,
    out_dir: Path,
    report: dict[str, Any],
    fold_rows: Sequence[dict[str, Any]],
    month_rows: Sequence[dict[str, Any]],
    etf_rows: Sequence[dict[str, Any]],
    prevalence_rows: Sequence[dict[str, Any]],
    threshold_rows: Sequence[dict[str, Any]],
    protocol_rows: Sequence[dict[str, Any]],
) -> None:
    docs_json, docs_md = build_docs_report(report)
    write_json(out_dir / "rolling_origin_stability_attribution_report.json", report)
    write_json(
        out_dir / "rolling_origin_stability_attribution_decision.json",
        {
            "lab_declaration": LAB_DECLARATION,
            "readiness_decision": report["readiness_decision"],
            "candidate_classification": report["candidate_classification"],
            "status": report["status"],
            "p0_blockers": report["p0_blockers"],
            "p1_warnings": report["p1_warnings"],
            "stable_promotion_ready": False,
            "stable_evidence": False,
            "formal_training_ready": False,
            "qmt_ready": False,
            "order_intent_ready": False,
            "automatic_promotion_ready": False,
        },
    )
    write_csv(out_dir / "rolling_origin_candidate_fold_breakdown.csv", list(fold_rows), fold_breakdown_columns())
    write_csv(out_dir / "rolling_origin_month_degradation_table.csv", list(month_rows), month_columns())
    write_csv(out_dir / "rolling_origin_etf_dispersion_table.csv", list(etf_rows), etf_columns())
    write_csv(out_dir / "rolling_origin_label_prevalence_table.csv", list(prevalence_rows), prevalence_columns())
    write_csv(out_dir / "rolling_origin_threshold_sensitivity_table.csv", list(threshold_rows), threshold_columns())
    write_csv(out_dir / "rolling_origin_protocol_reconciliation.csv", list(protocol_rows), protocol_columns())
    (out_dir / "rolling_origin_stability_attribution_report.md").write_text(docs_md, encoding="utf-8")
    write_json(repo_root / "docs/research/aetfq3_intraday_rolling_origin_stability_attribution_review.json", docs_json)
    (repo_root / "docs/research/aetfq3_intraday_rolling_origin_stability_attribution_review.md").write_text(docs_md, encoding="utf-8")


def fold_breakdown_columns() -> list[str]:
    return [
        "fold_id",
        "cutoff",
        "validation_window",
        "validation_anchor_count",
        "validation_group_count",
        "label_prevalence",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "dummy_most_frequent_balanced_accuracy",
        "dummy_stratified_balanced_accuracy",
        "best_dummy_balanced_accuracy",
        "advantage_over_best_dummy",
        "above_dummy",
        "prediction_positive_rate",
        "probability_mean",
        "probability_min",
        "probability_max",
        "collapse_flag",
        "positive_advantage_share",
        "tn",
        "fp",
        "fn",
        "tp",
    ]


def month_columns() -> list[str]:
    return [
        "month",
        "cutoff",
        "validation_group_count",
        "label_prevalence",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "prediction_positive_rate",
        "error_rate",
        "advantage_over_best_dummy",
        "positive_advantage_share",
        "whether_degradation_month",
        "whether_month_dominates_decision",
    ]


def etf_columns() -> list[str]:
    return [
        "etf_code",
        "group_count",
        "balanced_accuracy",
        "error_rate",
        "positive_label_rate",
        "positive_prediction_rate",
        "fp",
        "fn",
        "tp",
        "tn",
        "error_share",
        "whether_etf_concentration_observed",
    ]


def prevalence_columns() -> list[str]:
    return ["month", "cutoff", "validation_group_count", "label_prevalence", "prediction_positive_rate"]


def threshold_columns() -> list[str]:
    return ["threshold", "fold_id", "validation_month", "balanced_accuracy", "fp", "fn", "prediction_positive_rate", "stability_fraction"]


def protocol_columns() -> list[str]:
    return [
        "protocol",
        "train_direction",
        "validation_direction",
        "leakage_status",
        "main_question_answered",
        "key_metric",
        "risk",
        "allowed_interpretation",
        "forbidden_interpretation",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--rolling-origin-dir", type=Path, default=DEFAULT_ROLLING_ORIGIN_DIR)
    parser.add_argument("--fixed-oop-dir", type=Path, default=DEFAULT_FIXED_OOP_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_review(args.rolling_origin_dir, args.fixed_oop_dir, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI emits auditable Lab blocker.
        print(
            json.dumps(
                {
                    "lab_declaration": LAB_DECLARATION,
                    "status": "failed",
                    "readiness_decision": DECISION_BLOCKED_DATA,
                    "p0_blockers": [str(exc)],
                    "stable_promotion_ready": False,
                    "stable_evidence": False,
                    "qmt_ready": False,
                    "order_intent_ready": False,
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
                "status": report["status"],
                "readiness_decision": report["readiness_decision"],
                "candidate_classification": report["candidate_classification"],
                "stable_promotion_ready": False,
                "stable_evidence": False,
                "qmt_ready": False,
                "order_intent_ready": False,
                "p0_blockers": report["p0_blockers"],
                "p1_warnings": report["p1_warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
