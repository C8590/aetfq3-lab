from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_fixed_shortlist_oop_no_save_validation import (  # noqa: E402
    DEFAULT_MANUAL_INBOX,
    MANUAL_CSV_NAME,
    SHORTLIST,
    build_feature_rows,
    collapse_check,
    distribution_from_values,
    feature_columns_for_set,
    flat_metric_row,
    label_value,
    load_csv_rows,
    manual_csv_path,
    metric_columns,
    probability_scores,
    resolve_repo_path,
    rows_to_labels,
    rows_to_matrix,
    score_predictions,
    validate_manual_manifest,
    write_csv,
    write_json,
)
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts  # noqa: E402


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_rolling_origin_walk_forward_no_save_validation"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_rolling_origin_walk_forward_no_save_validation")
DEFAULT_OUT_DIR = ALLOWED_OUTPUT_DIR
MODEL_NAMES = ["dummy_most_frequent", "dummy_stratified", "logistic_balanced_scaled"]
DEFAULT_CUTOFFS = [
    "2025-06-30",
    "2025-07-31",
    "2025-08-29",
    "2025-09-30",
    "2025-10-31",
    "2025-11-28",
    "2025-12-31",
    "2026-01-30",
    "2026-02-27",
    "2026-03-31",
    "2026-04-30",
    "2026-05-29",
]
MIN_TRAIN_ANCHORS = 40
MIN_VALIDATION_ANCHORS = 10
MIN_VALIDATION_GROUPS = 50
MIN_ETF_COUNT = 5

DECISION_COMPLETED = "ROLLING_ORIGIN_WALK_FORWARD_COMPLETED_REVIEW_REQUIRED"
DECISION_STABLE = "ROLLING_ORIGIN_WALK_FORWARD_DIAGNOSTIC_STABILITY_OBSERVED_REVIEW_REQUIRED"
DECISION_NO_STABILITY = "ROLLING_ORIGIN_WALK_FORWARD_NO_STABILITY_OBSERVED_REVIEW_REQUIRED"
DECISION_BLOCKED_FOLDS = "ROLLING_ORIGIN_BLOCKED_INSUFFICIENT_FOLDS"
DECISION_BLOCKED_DATA = "ROLLING_ORIGIN_BLOCKED_DATA_QUALITY"
DECISION_BLOCKED_LABEL = "ROLLING_ORIGIN_BLOCKED_LABEL_DEFINITION_MISMATCH"
DECISION_BLOCKED_LEAKAGE = "ROLLING_ORIGIN_BLOCKED_LEAKAGE_RISK"


class RollingOriginError(RuntimeError):
    pass


def resolve_output_dir(out_dir: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    repo_root = repo_root.resolve()
    resolved = resolve_repo_path(out_dir, repo_root).resolve()
    if enforce:
        allowed = (repo_root / ALLOWED_OUTPUT_DIR).resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError as exc:
            raise RollingOriginError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def next_month_prefix(cutoff: str) -> str:
    year, month, _day = [int(part) for part in cutoff.split("-")]
    month += 1
    if month == 13:
        year += 1
        month = 1
    return f"{year:04d}-{month:02d}"


def covered_for_train(row: dict[str, Any], cutoff: str) -> bool:
    t_plus_3 = str(row.get("t_plus_3_date") or "")
    return bool(t_plus_3) and t_plus_3 <= cutoff


def finite_rows_for_dates(
    rows: Sequence[dict[str, Any]],
    dates: Sequence[str],
    label_policy: str,
    feature_columns: Sequence[str],
    *,
    cutoff: str | None = None,
    require_train_label_known_by_cutoff: bool = False,
) -> list[dict[str, Any]]:
    date_set = set(dates)
    selected: list[dict[str, Any]] = []
    for row in rows:
        trade_date = str(row.get("trade_date", ""))
        if trade_date not in date_set:
            continue
        if require_train_label_known_by_cutoff and cutoff is not None and not covered_for_train(row, cutoff):
            continue
        if label_value(row.get(label_policy)) is None:
            continue
        if all(row.get(feature) != "" and row.get(feature) is not None for feature in feature_columns):
            selected.append(row)
    return selected


def make_fold_manifest(
    rows: Sequence[dict[str, Any]],
    cutoffs: Sequence[str] = DEFAULT_CUTOFFS,
    *,
    min_train_anchors: int = MIN_TRAIN_ANCHORS,
    min_validation_anchors: int = MIN_VALIDATION_ANCHORS,
    min_validation_groups: int = MIN_VALIDATION_GROUPS,
    min_etf_count: int = MIN_ETF_COUNT,
) -> list[dict[str, Any]]:
    all_dates = sorted({str(row.get("trade_date", "")) for row in rows})
    manifests: list[dict[str, Any]] = []
    for cutoff in cutoffs:
        validation_prefix = next_month_prefix(cutoff)
        train_dates = [date for date in all_dates if date <= cutoff]
        validation_dates = [date for date in all_dates if date > cutoff and date.startswith(validation_prefix)]
        train_rows_known = [row for row in rows if str(row.get("trade_date", "")) in set(train_dates) and covered_for_train(row, cutoff)]
        validation_rows_covered = [
            row
            for row in rows
            if str(row.get("trade_date", "")) in set(validation_dates) and label_value(row.get("label_ret3d_gt_100bp")) is not None
        ]
        validation_etfs = {str(row.get("etf_code", "")) for row in validation_rows_covered}
        skip_reasons: list[str] = []
        if len({str(row.get("trade_date", "")) for row in train_rows_known}) < min_train_anchors:
            skip_reasons.append("min_train_anchors_not_met")
        if len({str(row.get("trade_date", "")) for row in validation_rows_covered}) < min_validation_anchors:
            skip_reasons.append("min_validation_anchors_not_met")
        if len(validation_rows_covered) < min_validation_groups:
            skip_reasons.append("min_validation_groups_not_met")
        if len(validation_etfs) < min_etf_count:
            skip_reasons.append("min_etf_count_not_met")
        no_overlap = not (set(train_dates) & set(validation_dates))
        strictly_late = all(date > cutoff for date in validation_dates)
        if not no_overlap or not strictly_late:
            skip_reasons.append("leakage_risk_train_validation_overlap_or_not_late")
        manifests.append(
            {
                "fold_id": f"{cutoff}_to_{validation_prefix}",
                "cutoff": cutoff,
                "train_window_type": "expanding",
                "validation_month": validation_prefix,
                "train_anchor_dates": train_dates,
                "validation_anchor_dates": validation_dates,
                "train_anchor_count": len({str(row.get("trade_date", "")) for row in train_rows_known}),
                "validation_anchor_count": len({str(row.get("trade_date", "")) for row in validation_rows_covered}),
                "train_group_count_label_known_by_cutoff": len(train_rows_known),
                "validation_group_count": len(validation_rows_covered),
                "validation_etf_count": len(validation_etfs),
                "train_validation_no_overlap": no_overlap,
                "validation_strictly_after_cutoff": strictly_late,
                "train_label_t_plus_3_known_by_cutoff": all(covered_for_train(row, cutoff) for row in train_rows_known),
                "skipped": bool(skip_reasons),
                "skip_reasons": skip_reasons,
            }
        )
    return manifests


def train_only_scale(
    train_rows: Sequence[dict[str, Any]],
    validation_rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    from sklearn.preprocessing import StandardScaler

    x_train = rows_to_matrix(train_rows, feature_columns)
    x_validation = rows_to_matrix(validation_rows, feature_columns)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_validation_scaled = scaler.transform(x_validation) if validation_rows else []
    return {
        "x_train": x_train_scaled.tolist(),
        "x_validation": x_validation_scaled.tolist() if hasattr(x_validation_scaled, "tolist") else [],
        "audit": {
            "fit_scope": "train_only",
            "fit_row_count": len(train_rows),
            "validation_row_count": len(validation_rows),
            "fit_feature_count": len(feature_columns),
            "validation_fit_performed": False,
            "train_means": [float(value) for value in scaler.mean_.tolist()],
            "train_vars": [float(value) for value in scaler.var_.tolist()],
        },
    }


def run_fold_candidate(
    feature_rows: Sequence[dict[str, Any]],
    fold: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    feature_columns = feature_columns_for_set(candidate["feature_set"])
    train_rows = finite_rows_for_dates(
        feature_rows,
        fold["train_anchor_dates"],
        candidate["label_policy"],
        feature_columns,
        cutoff=fold["cutoff"],
        require_train_label_known_by_cutoff=True,
    )
    validation_rows = finite_rows_for_dates(feature_rows, fold["validation_anchor_dates"], candidate["label_policy"], feature_columns)
    blockers: list[str] = []
    if len({str(row.get("trade_date", "")) for row in train_rows}) < MIN_TRAIN_ANCHORS:
        blockers.append("candidate_min_train_anchors_not_met")
    if len({str(row.get("trade_date", "")) for row in validation_rows}) < MIN_VALIDATION_ANCHORS:
        blockers.append("candidate_min_validation_anchors_not_met")
    if len(validation_rows) < MIN_VALIDATION_GROUPS:
        blockers.append("candidate_min_validation_groups_not_met")
    if len({str(row.get("etf_code", "")) for row in validation_rows}) < MIN_ETF_COUNT:
        blockers.append("candidate_min_etf_count_not_met")
    if len(set(rows_to_labels(train_rows, candidate["label_policy"]))) < 2:
        blockers.append("candidate_train_single_class")
    if set(fold["train_anchor_dates"]) & set(fold["validation_anchor_dates"]):
        blockers.append("candidate_train_validation_overlap")
    if not all(str(row.get("trade_date", "")) > fold["cutoff"] for row in validation_rows):
        blockers.append("candidate_validation_not_strictly_after_cutoff")
    if blockers:
        return {
            **candidate,
            "fold_id": fold["fold_id"],
            "cutoff": fold["cutoff"],
            "validation_month": fold["validation_month"],
            "skipped": True,
            "skip_reasons": blockers,
            "train_group_count": len(train_rows),
            "validation_group_count": len(validation_rows),
        }

    scaled = train_only_scale(train_rows, validation_rows, feature_columns)
    y_train = rows_to_labels(train_rows, candidate["label_policy"])
    y_validation = rows_to_labels(validation_rows, candidate["label_policy"])
    models = {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "dummy_stratified": DummyClassifier(strategy="stratified", random_state=42),
        "logistic_balanced_scaled": LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            solver="liblinear",
            random_state=42,
        ),
    }
    metrics: dict[str, Any] = {}
    prediction_summary_rows: list[dict[str, Any]] = []
    row_level_rows: list[dict[str, Any]] = []
    for model_name, model in models.items():
        x_train = scaled["x_train"] if model_name == "logistic_balanced_scaled" else rows_to_matrix(train_rows, feature_columns)
        x_validation = scaled["x_validation"] if model_name == "logistic_balanced_scaled" else rows_to_matrix(validation_rows, feature_columns)
        model.fit(x_train, y_train)
        predictions = [int(value) for value in model.predict(x_validation)]
        probabilities = probability_scores(model, x_validation, predictions)
        score = score_predictions(y_validation, predictions, probabilities)
        metrics[model_name] = {"validation": score}
        prediction_summary_rows.append(
            {
                "fold_id": fold["fold_id"],
                "cutoff": fold["cutoff"],
                "validation_month": fold["validation_month"],
                "family_id": candidate["family_id"],
                "model": model_name,
                "row_count": len(validation_rows),
                "prediction_distribution": json.dumps(score["prediction_distribution"], sort_keys=True),
                "probability_min": score["probability_summary"]["min"],
                "probability_max": score["probability_summary"]["max"],
                "probability_mean": score["probability_summary"]["mean"],
            }
        )
        row_level_rows.extend(
            build_row_level_rows(fold, candidate, model_name, validation_rows, predictions, probabilities)
        )
    logistic = metrics["logistic_balanced_scaled"]["validation"]
    dummy_most = metrics["dummy_most_frequent"]["validation"]
    dummy_strat = metrics["dummy_stratified"]["validation"]
    baseline_best_ba = max(value for value in [dummy_most["balanced_accuracy"], dummy_strat["balanced_accuracy"]] if value is not None)
    pr_auc = logistic["pr_auc"]
    prevalence = logistic["label_prevalence"]
    single_class = logistic["prediction_distribution"]["0"] == 0 or logistic["prediction_distribution"]["1"] == 0
    return {
        **candidate,
        "fold_id": fold["fold_id"],
        "cutoff": fold["cutoff"],
        "validation_month": fold["validation_month"],
        "skipped": False,
        "skip_reasons": [],
        "feature_count": len(feature_columns),
        "train_group_count": len(train_rows),
        "validation_group_count": len(validation_rows),
        "validation_anchor_count": len({str(row.get("trade_date", "")) for row in validation_rows}),
        "validation_etf_count": len({str(row.get("etf_code", "")) for row in validation_rows}),
        "train_label_distribution": distribution_from_values(rows_to_labels(train_rows, candidate["label_policy"])),
        "validation_label_distribution": distribution_from_values(y_validation),
        "metrics": metrics,
        "scaler_audit": scaled["audit"],
        "collapse_check": collapse_check({"logistic_balanced_scaled": {"validation": logistic}}),
        "etf_level_dispersion": label_dispersion(validation_rows, candidate["label_policy"], "etf_code"),
        "date_level_instability": label_dispersion(validation_rows, candidate["label_policy"], "trade_date"),
        "better_than_dummy_baseline": logistic["balanced_accuracy"] is not None and logistic["balanced_accuracy"] > baseline_best_ba,
        "below_label_prevalence_pr_auc": pr_auc is None or prevalence is None or pr_auc < prevalence,
        "single_class_prediction_collapse": single_class,
        "prediction_summary_rows": prediction_summary_rows,
        "row_level_prediction_rows": row_level_rows,
    }


def build_row_level_rows(
    fold: dict[str, Any],
    candidate: dict[str, Any],
    model_name: str,
    validation_rows: Sequence[dict[str, Any]],
    predictions: Sequence[int],
    probabilities: Sequence[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, prediction, probability in zip(validation_rows, predictions, probabilities, strict=True):
        label = label_value(row.get(candidate["label_policy"]))
        output.append(
            {
                "fold_id": fold["fold_id"],
                "cutoff": fold["cutoff"],
                "validation_month": fold["validation_month"],
                "candidate_id": candidate["family_id"],
                "family_id": candidate["family_id"],
                "label_policy": candidate["label_policy"],
                "feature_set": candidate["feature_set"],
                "model_family": candidate["model_family"],
                "model": model_name,
                "anchor_date": row.get("anchor_date") or row.get("trade_date"),
                "etf_code": row.get("etf_code"),
                "label": label,
                "prediction": prediction,
                "probability": probability,
                "is_correct": label == prediction if label is not None else False,
                "error_type": prediction_error_type(label, prediction),
                "future_return_3d": row.get("future_return_3d"),
                "t_plus_3_date": row.get("t_plus_3_date"),
                "train_or_oop": "validation",
            }
        )
    return output


def prediction_error_type(label: int | None, prediction: int | None) -> str:
    if label is None or prediction is None:
        return "NA"
    if label == 1 and prediction == 1:
        return "TP"
    if label == 0 and prediction == 0:
        return "TN"
    if label == 0 and prediction == 1:
        return "FP"
    if label == 1 and prediction == 0:
        return "FN"
    return "NA"


def label_dispersion(rows: Sequence[dict[str, Any]], label_policy: str, key: str) -> dict[str, Any]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        value = label_value(row.get(label_policy))
        if value is not None:
            grouped[str(row.get(key, ""))].append(value)
    rates = {name: sum(values) / len(values) for name, values in grouped.items() if values}
    return {
        "group_count": len(rates),
        "min_positive_rate": min(rates.values()) if rates else None,
        "max_positive_rate": max(rates.values()) if rates else None,
        "std_positive_rate": pstdev(rates.values()) if len(rates) > 1 else 0.0 if rates else None,
        "positive_rate_by_group": rates,
    }


def flat_fold_metric_rows(candidate_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result in candidate_results:
        if result.get("skipped"):
            continue
        for model_name, by_split in result["metrics"].items():
            metric = by_split["validation"]
            rows.append(
                {
                    **flat_metric_row(result, model_name, "validation", metric),
                    "fold_id": result["fold_id"],
                    "cutoff": result["cutoff"],
                    "validation_month": result["validation_month"],
                    "better_than_dummy_baseline": result["better_than_dummy_baseline"] if model_name == "logistic_balanced_scaled" else "",
                    "below_label_prevalence_pr_auc": result["below_label_prevalence_pr_auc"] if model_name == "logistic_balanced_scaled" else "",
                    "single_class_prediction_collapse": result["single_class_prediction_collapse"] if model_name == "logistic_balanced_scaled" else "",
                }
            )
    return rows


def fold_metric_columns() -> list[str]:
    return [
        "fold_id",
        "cutoff",
        "validation_month",
        *metric_columns(),
        "better_than_dummy_baseline",
        "below_label_prevalence_pr_auc",
        "single_class_prediction_collapse",
    ]


def aggregate_candidate_results(candidate_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregates: list[dict[str, Any]] = []
    by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in candidate_results:
        if not result.get("skipped"):
            by_candidate[result["family_id"]].append(result)
    for family_id, results in sorted(by_candidate.items()):
        metrics = [result["metrics"]["logistic_balanced_scaled"]["validation"] for result in results]
        ba_values = [metric["balanced_accuracy"] for metric in metrics if metric["balanced_accuracy"] is not None]
        roc_values = [metric["roc_auc"] for metric in metrics if metric["roc_auc"] is not None]
        pr_values = [metric["pr_auc"] for metric in metrics if metric["pr_auc"] is not None]
        ba_above = sum(1 for value in ba_values if value > 0.5)
        roc_above = sum(1 for value in roc_values if value > 0.5)
        pr_not_below_prev = sum(
            1
            for metric in metrics
            if metric["pr_auc"] is not None
            and metric["label_prevalence"] is not None
            and metric["pr_auc"] >= metric["label_prevalence"]
        )
        non_collapse = sum(1 for result in results if not result["single_class_prediction_collapse"])
        above_dummy = sum(1 for result in results if result["better_than_dummy_baseline"])
        worst = min(results, key=lambda item: item["metrics"]["logistic_balanced_scaled"]["validation"]["balanced_accuracy"] or -1)
        best = max(results, key=lambda item: item["metrics"]["logistic_balanced_scaled"]["validation"]["balanced_accuracy"] or -1)
        fold_count = len(results)
        aggregates.append(
            {
                "family_id": family_id,
                "label_policy": results[0]["label_policy"],
                "feature_set": results[0]["feature_set"],
                "evaluated_fold_count": fold_count,
                "balanced_accuracy_mean": summarize_numeric(ba_values, "mean"),
                "balanced_accuracy_median": summarize_numeric(ba_values, "median"),
                "balanced_accuracy_std": summarize_numeric(ba_values, "std"),
                "roc_auc_mean": summarize_numeric(roc_values, "mean"),
                "roc_auc_median": summarize_numeric(roc_values, "median"),
                "roc_auc_std": summarize_numeric(roc_values, "std"),
                "pr_auc_mean": summarize_numeric(pr_values, "mean"),
                "pr_auc_median": summarize_numeric(pr_values, "median"),
                "pr_auc_std": summarize_numeric(pr_values, "std"),
                "positive_fold_count": ba_above,
                "negative_fold_count": fold_count - ba_above,
                "fraction_folds_balanced_accuracy_above_0_5": ba_above / fold_count if fold_count else 0,
                "fraction_folds_roc_auc_above_0_5": roc_above / fold_count if fold_count else 0,
                "fraction_folds_pr_auc_not_below_prevalence": pr_not_below_prev / fold_count if fold_count else 0,
                "fraction_folds_above_dummy": above_dummy / fold_count if fold_count else 0,
                "fraction_folds_non_collapse": non_collapse / fold_count if fold_count else 0,
                "worst_fold": {
                    "fold_id": worst["fold_id"],
                    "balanced_accuracy": worst["metrics"]["logistic_balanced_scaled"]["validation"]["balanced_accuracy"],
                },
                "best_fold": {
                    "fold_id": best["fold_id"],
                    "balanced_accuracy": best["metrics"]["logistic_balanced_scaled"]["validation"]["balanced_accuracy"],
                },
                "monthly_degradation_pattern": [
                    {
                        "validation_month": result["validation_month"],
                        "balanced_accuracy": result["metrics"]["logistic_balanced_scaled"]["validation"]["balanced_accuracy"],
                        "roc_auc": result["metrics"]["logistic_balanced_scaled"]["validation"]["roc_auc"],
                        "pr_auc": result["metrics"]["logistic_balanced_scaled"]["validation"]["pr_auc"],
                        "label_prevalence": result["metrics"]["logistic_balanced_scaled"]["validation"]["label_prevalence"],
                        "below_0_5_balanced_accuracy": (result["metrics"]["logistic_balanced_scaled"]["validation"]["balanced_accuracy"] or 0) <= 0.5,
                    }
                    for result in results
                ],
            }
        )
    return aggregates


def summarize_numeric(values: Sequence[float], kind: str) -> float | None:
    if not values:
        return None
    if kind == "mean":
        return float(mean(values))
    if kind == "median":
        return float(median(values))
    if kind == "std":
        return float(pstdev(values)) if len(values) > 1 else 0.0
    raise RollingOriginError(f"unknown numeric summary kind: {kind}")


def stability_observed(aggregate: dict[str, Any], no_leakage: bool, no_artifacts: bool) -> bool:
    return all(
        [
            aggregate["evaluated_fold_count"] >= 6,
            aggregate["fraction_folds_balanced_accuracy_above_0_5"] >= 0.6,
            aggregate["fraction_folds_roc_auc_above_0_5"] >= 0.6,
            aggregate["fraction_folds_pr_auc_not_below_prevalence"] >= 0.6,
            aggregate["fraction_folds_non_collapse"] >= 1.0,
            no_leakage,
            no_artifacts,
        ]
    )


def decide(aggregates: Sequence[dict[str, Any]], blockers: Sequence[str], no_leakage: bool, no_artifacts: bool) -> str:
    joined = " ".join(blockers).lower()
    if "label" in joined and "mismatch" in joined:
        return DECISION_BLOCKED_LABEL
    if "leakage" in joined or "overlap" in joined or not no_leakage:
        return DECISION_BLOCKED_LEAKAGE
    if blockers:
        return DECISION_BLOCKED_DATA
    if not aggregates or max((item["evaluated_fold_count"] for item in aggregates), default=0) < 6:
        return DECISION_BLOCKED_FOLDS
    if any(stability_observed(item, no_leakage, no_artifacts) for item in aggregates):
        return DECISION_STABLE
    return DECISION_NO_STABILITY


def build_docs_report(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    stability_count = sum(item["diagnostic_stability_observed"] for item in report["aggregate_stability"])
    summary = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_rolling_origin_walk_forward_no_save_validation",
        "status": report["status"],
        "readiness_decision": report["readiness_decision"],
        "source": report["input_sources"],
        "fold_count": len(report["fold_manifest"]),
        "evaluated_fold_count": report["evaluated_fold_count"],
        "candidate_count": len(SHORTLIST),
        "diagnostic_stability_observed_candidate_count": stability_count,
        "formal_training": False,
        "model_saved": False,
        "scaler_saved": False,
        "stable_promotion_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "stable_evidence": False,
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday Rolling-Origin Walk-Forward No-Save Validation",
        "",
        "Lab-only rolling-origin diagnostic validation. It uses historical cutoffs, train-only scaler fit, fixed Sprint3 shortlist only, and writes no model/scaler/checkpoint.",
        "",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- fold_count: {len(report['fold_manifest'])}",
        f"- evaluated_fold_count: {report['evaluated_fold_count']}",
        f"- no_leakage_assertion_passed: {str(report['no_leakage_assertion_passed']).lower()}",
        f"- model_saved: {str(report['model_saved']).lower()}",
        f"- scaler_saved: {str(report['scaler_saved']).lower()}",
        f"- stable_promotion_ready: {str(report['stable_promotion_ready']).lower()}",
        "",
        "## Aggregate Stability",
    ]
    for item in report["aggregate_stability"]:
        lines.append(
            f"- {item['family_id']}: folds={item['evaluated_fold_count']}, ba_mean={item['balanced_accuracy_mean']}, ba_above_0_5={item['fraction_folds_balanced_accuracy_above_0_5']}, roc_above_0_5={item['fraction_folds_roc_auc_above_0_5']}, pr_not_below_prev={item['fraction_folds_pr_auc_not_below_prevalence']}, stability={str(item['diagnostic_stability_observed']).lower()}"
        )
    return summary, "\n".join(lines) + "\n"


def run_validation(
    manual_inbox: Path = DEFAULT_MANUAL_INBOX,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    enforce_output_dir: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_out_dir = resolve_output_dir(out_dir, repo_root, enforce=enforce_output_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    manifest_check = validate_manual_manifest(manual_inbox, repo_root)
    csv_path = manual_csv_path(manual_inbox, repo_root)
    bar_rows, columns = load_csv_rows(csv_path)
    blockers = list(manifest_check["p0_blockers"])
    missing = sorted({"trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume"} - set(columns))
    if missing:
        blockers.append("manual CSV missing required columns: " + ", ".join(missing))

    feature_rows, data_build_report = build_feature_rows(bar_rows)
    fold_manifest = make_fold_manifest(feature_rows)
    no_leakage = all(
        fold["train_validation_no_overlap"]
        and fold["validation_strictly_after_cutoff"]
        and fold["train_label_t_plus_3_known_by_cutoff"]
        for fold in fold_manifest
    )
    if not no_leakage:
        blockers.append("rolling-origin leakage assertion failed")

    artifact_before = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_before["p0_blockers"])

    candidate_results: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    prediction_summary_rows: list[dict[str, Any]] = []
    row_level_prediction_rows: list[dict[str, Any]] = []
    if not blockers:
        for fold in fold_manifest:
            if fold["skipped"]:
                continue
            for candidate in SHORTLIST:
                result = run_fold_candidate(feature_rows, fold, candidate)
                candidate_results.append(result)
                if result.get("skipped"):
                    continue
                prediction_summary_rows.extend(result["prediction_summary_rows"])
                row_level_prediction_rows.extend(result["row_level_prediction_rows"])
        metric_rows = flat_fold_metric_rows(candidate_results)

    aggregate_stability = aggregate_candidate_results(candidate_results)
    artifact_after = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_after["p0_blockers"])
    no_artifacts = not artifact_after["p0_blockers"]
    for aggregate in aggregate_stability:
        aggregate["diagnostic_stability_observed"] = stability_observed(aggregate, no_leakage, no_artifacts)
    readiness_decision = decide(aggregate_stability, dedupe(blockers), no_leakage, no_artifacts)
    p1_warnings = [
        "P1_DIAGNOSTIC_ONLY_NOT_FORMAL_MODEL_EVIDENCE",
        "P1_REQUIRES_HUMAN_REVIEW",
        "P1_NO_STABLE_PROMOTION_WITHOUT_PROMOTION_GATE",
        "P1_ROLLING_ORIGIN_FIXED_SHORTLIST_ONLY_NOT_STABLE_EVIDENCE",
    ]
    if readiness_decision == DECISION_NO_STABILITY:
        p1_warnings.append("P1_ROLLING_ORIGIN_NO_STABILITY_OBSERVED")
    if readiness_decision == DECISION_STABLE:
        p1_warnings.append("P1_STABILITY_OBSERVED_BUT_STILL_REVIEW_ONLY")

    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": "blocked" if readiness_decision.startswith("ROLLING_ORIGIN_BLOCKED") else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "manual_inbox": str(manual_inbox),
            "manual_csv": str(csv_path),
            "manual_manifest": manifest_check["path"],
            "source": "broker terminal manual export",
            "stable_bundle": False,
        },
        "rolling_origin_protocol": {
            "train_window_type": "expanding",
            "validation_window": "next calendar month after cutoff",
            "train_label_rule": "training labels require t_plus_3_date <= cutoff",
            "validation_rule": "anchor_date > cutoff and no validation rows used for scaler fit",
            "fixed_shortlist_only": True,
            "threshold_changed": False,
            "hyperparameter_search": False,
        },
        "readiness_decision": readiness_decision,
        "shortlist": SHORTLIST,
        "manual_manifest_check": {key: value for key, value in manifest_check.items() if key != "manifest"},
        "data_build_report": data_build_report,
        "fold_manifest": fold_manifest,
        "evaluated_fold_count": len({result["fold_id"] for result in candidate_results if not result.get("skipped")}),
        "candidate_fold_results": strip_heavy_rows(candidate_results),
        "aggregate_stability": aggregate_stability,
        "no_leakage_assertion_passed": no_leakage,
        "artifact_check_before": artifact_before,
        "artifact_check_after": artifact_after,
        "row_level_predictions_emitted": True,
        "row_level_prediction_file": "rolling_origin_row_level_predictions.csv",
        "row_level_prediction_row_count": len(row_level_prediction_rows),
        "p0_blockers": dedupe(blockers),
        "p1_warnings": dedupe(p1_warnings),
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
        "metrics_are_effectiveness_evidence": False,
        "not_trading_advice": True,
        "advisory_package_created": False,
    }
    docs_json, docs_md = build_docs_report(report)
    write_json(resolved_out_dir / "rolling_origin_walk_forward_report.json", report)
    write_json(resolved_out_dir / "rolling_origin_fold_manifest.json", {"lab_declaration": LAB_DECLARATION, "fold_manifest": fold_manifest})
    write_json(
        resolved_out_dir / "rolling_origin_decision.json",
        {
            "lab_declaration": LAB_DECLARATION,
            "readiness_decision": readiness_decision,
            "status": report["status"],
            "evaluated_fold_count": report["evaluated_fold_count"],
            "no_leakage_assertion_passed": no_leakage,
            "p0_blockers": report["p0_blockers"],
            "p1_warnings": report["p1_warnings"],
            "formal_training": False,
            "model_saved": False,
            "scaler_saved": False,
            "stable_promotion_ready": False,
            "qmt_ready": False,
            "order_intent_ready": False,
            "stable_evidence": False,
        },
    )
    write_csv(resolved_out_dir / "rolling_origin_fold_metrics.csv", metric_rows, fold_metric_columns())
    write_csv(
        resolved_out_dir / "rolling_origin_fold_predictions_summary.csv",
        prediction_summary_rows,
        [
            "fold_id",
            "cutoff",
            "validation_month",
            "family_id",
            "model",
            "row_count",
            "prediction_distribution",
            "probability_min",
            "probability_max",
            "probability_mean",
        ],
    )
    write_csv(resolved_out_dir / "rolling_origin_row_level_predictions.csv", row_level_prediction_rows, row_level_prediction_columns())
    (resolved_out_dir / "rolling_origin_walk_forward_report.md").write_text(docs_md, encoding="utf-8")
    write_json(repo_root / "docs/research/aetfq3_intraday_rolling_origin_walk_forward_no_save_validation.json", docs_json)
    (repo_root / "docs/research/aetfq3_intraday_rolling_origin_walk_forward_no_save_validation.md").write_text(docs_md, encoding="utf-8")
    return report


def strip_heavy_rows(candidate_results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for result in candidate_results:
        stripped.append({key: value for key, value in result.items() if key not in {"prediction_summary_rows", "row_level_prediction_rows"}})
    return stripped


def row_level_prediction_columns() -> list[str]:
    return [
        "fold_id",
        "cutoff",
        "validation_month",
        "candidate_id",
        "family_id",
        "label_policy",
        "feature_set",
        "model_family",
        "model",
        "anchor_date",
        "etf_code",
        "label",
        "prediction",
        "probability",
        "is_correct",
        "error_type",
        "future_return_3d",
        "t_plus_3_date",
        "train_or_oop",
    ]


def dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--manual-inbox", type=Path, default=DEFAULT_MANUAL_INBOX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_validation(args.manual_inbox, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI emits auditable Lab blocker.
        print(
            json.dumps(
                {
                    "lab_declaration": LAB_DECLARATION,
                    "status": "failed",
                    "readiness_decision": DECISION_BLOCKED_DATA,
                    "p0_blockers": [str(exc)],
                    "formal_training": False,
                    "model_saved": False,
                    "scaler_saved": False,
                    "stable_promotion_ready": False,
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
                "evaluated_fold_count": report["evaluated_fold_count"],
                "diagnostic_stability_observed_candidate_count": sum(
                    item["diagnostic_stability_observed"] for item in report["aggregate_stability"]
                ),
                "model_saved": False,
                "scaler_saved": False,
                "stable_promotion_ready": False,
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
