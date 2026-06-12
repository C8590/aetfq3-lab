from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_fixed_shortlist_oop_no_save_validation import (  # noqa: E402
    BASE_39_FEATURES,
    DEFAULT_MANUAL_INBOX,
    MANUAL_CSV_NAME,
    build_feature_rows,
    load_csv_rows,
    to_float,
)
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts  # noqa: E402


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_oop_post_sprint_instability_review"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_oop_post_sprint_instability_review")
DEFAULT_OOP_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_fixed_shortlist_oop_no_save_validation")
DEFAULT_OUT_DIR = ALLOWED_OUTPUT_DIR
FOCUS_FAMILY_ID = "label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy"
FOCUS_LABEL = "label_safe_positive_3d"
FOCUS_FEATURE_SET = "base_39_plus_scale_transform_policy"
DECISION_SAMPLE_TOO_SMALL = "POST_SPRINT_INSTABILITY_REVIEW_COMPLETED_SAMPLE_TOO_SMALL_REVIEW_REQUIRED"
DECISION_LABEL_SHIFT = "POST_SPRINT_INSTABILITY_REVIEW_LABEL_SHIFT_OBSERVED"
DECISION_ETF_CONCENTRATION = "POST_SPRINT_INSTABILITY_REVIEW_ETF_CONCENTRATION_OBSERVED"
DECISION_DATE_CONCENTRATION = "POST_SPRINT_INSTABILITY_REVIEW_DATE_CONCENTRATION_OBSERVED"
DECISION_FEATURE_SHIFT = "POST_SPRINT_INSTABILITY_REVIEW_FEATURE_SHIFT_OBSERVED"
DECISION_CONTINUE = "POST_SPRINT_INSTABILITY_REVIEW_NO_CLEAR_SINGLE_CAUSE_CONTINUE_OOP_ACCUMULATION"
DECISION_BLOCKED_MISSING = "POST_SPRINT_INSTABILITY_REVIEW_BLOCKED_MISSING_OOP_OUTPUTS"
DECISION_BLOCKED_DATA = "POST_SPRINT_INSTABILITY_REVIEW_BLOCKED_DATA_QUALITY"
REQUIRED_OOP_FILES = [
    "fixed_shortlist_oop_validation_report.json",
    "fixed_shortlist_oop_split_manifest.json",
    "fixed_shortlist_oop_metrics.csv",
    "fixed_shortlist_oop_predictions_summary.csv",
    "fixed_shortlist_oop_decision.json",
]


class InstabilityReviewError(RuntimeError):
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
            raise InstabilityReviewError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise InstabilityReviewError(f"JSON cannot be read: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise InstabilityReviewError(f"JSON parse failed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InstabilityReviewError(f"JSON root must be object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_csv_dicts(path: Path) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f))
    except OSError as exc:
        raise InstabilityReviewError(f"CSV cannot be read: {path}: {exc}") from exc


def validate_oop_outputs(oop_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_OOP_FILES if not (oop_dir / name).exists()]
    prediction_summary = oop_dir / "fixed_shortlist_oop_predictions_summary.csv"
    row_prediction_files = [
        oop_dir / "fixed_shortlist_oop_row_predictions.csv",
        oop_dir / "fixed_shortlist_oop_predictions.csv",
    ]
    row_level_file = next((path for path in row_prediction_files if path.exists()), None)
    row_level_prediction_available = False
    row_level_reason = "row-level prediction file not found"
    if row_level_file is not None:
        rows = load_csv_dicts(row_level_file)
        required = {"anchor_date", "etf_code", "label", "prediction", "probability"}
        columns = set(rows[0].keys()) if rows else set()
        row_level_prediction_available = required.issubset(columns)
        row_level_reason = "row-level prediction columns found" if row_level_prediction_available else "row-level prediction file lacks required columns"
    elif prediction_summary.exists():
        rows = load_csv_dicts(prediction_summary)
        columns = set(rows[0].keys()) if rows else set()
        if not {"anchor_date", "etf_code", "label", "prediction", "probability"}.issubset(columns):
            row_level_reason = "prediction summary is split-level only; anchor/ETF FP/FN cannot be audited"
    return {
        "passed": not missing,
        "missing_files": missing,
        "row_level_prediction_available": row_level_prediction_available,
        "row_level_prediction_reason": row_level_reason,
        "p0_blockers": [f"missing OOP output file: {name}" for name in missing],
    }


def split_rows(feature_rows: Sequence[dict[str, Any]], split_manifest: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    dates_by_split = {
        "train": set(split_manifest.get("train_anchor_dates", [])),
        "pre_sprint_oop": set(split_manifest.get("pre_sprint_oop_anchor_dates", [])),
        "post_sprint_oop": set(split_manifest.get("post_sprint_oop_anchor_dates", [])),
        "combined_strict_oop": set(split_manifest.get("combined_strict_oop_anchor_dates", [])),
    }
    return {
        name: [row for row in feature_rows if str(row.get("trade_date", "")) in dates and row.get("t_plus_3_covered") is True]
        for name, dates in dates_by_split.items()
    }


def label_distribution(rows: Sequence[dict[str, Any]], label: str) -> dict[str, Any]:
    labels = [label_value(row.get(label)) for row in rows]
    valid = [value for value in labels if value is not None]
    zeros = sum(value == 0 for value in valid)
    ones = sum(value == 1 for value in valid)
    total = len(valid)
    prevalence = ones / total if total else None
    return {
        "group_count": total,
        "label_0_count": zeros,
        "label_1_count": ones,
        "positive_rate": prevalence,
        "class_imbalance": abs((ones / total) - 0.5) if total else None,
    }


def label_shift_rows(rows_by_split: dict[str, list[dict[str, Any]]], labels: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    flags: list[str] = []
    for label in labels:
        train = label_distribution(rows_by_split["train"], label)
        for split_name in ("train", "pre_sprint_oop", "post_sprint_oop", "combined_strict_oop"):
            current = label_distribution(rows_by_split[split_name], label)
            delta = abs_delta(current["positive_rate"], train["positive_rate"])
            kl = bernoulli_kl(current["positive_rate"], train["positive_rate"])
            shifted = (delta is not None and delta >= 0.15) or (kl is not None and kl >= 0.05)
            if split_name == "post_sprint_oop" and shifted:
                flags.append(f"{label}:post_sprint_label_shift")
            output.append(
                {
                    "label_policy": label,
                    "split": split_name,
                    **current,
                    "positive_rate_delta_vs_train": delta,
                    "kl_like_vs_train": kl,
                    "label_shift_observed": shifted,
                }
            )
    return output, {"label_shift_observed": bool(flags), "flags": flags}


def feature_shift_rows(rows_by_split: dict[str, list[dict[str, Any]]], features: Sequence[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    top_candidates: list[dict[str, Any]] = []
    for feature in features:
        train_values = numeric_values(rows_by_split["train"], feature)
        post_values = numeric_values(rows_by_split["post_sprint_oop"], feature)
        train_mean, train_std = safe_mean_std(train_values)
        post_mean, post_std = safe_mean_std(post_values)
        smd = standardized_mean_difference(train_values, post_values)
        kind = feature_kind(feature)
        shifted = abs(smd) >= 0.8 if smd is not None else False
        output.append(
            {
                "feature": feature,
                "feature_kind": kind,
                "train_mean": train_mean,
                "train_std": train_std,
                "pre_mean": safe_mean_std(numeric_values(rows_by_split["pre_sprint_oop"], feature))[0],
                "pre_std": safe_mean_std(numeric_values(rows_by_split["pre_sprint_oop"], feature))[1],
                "post_mean": post_mean,
                "post_std": post_std,
                "combined_oop_mean": safe_mean_std(numeric_values(rows_by_split["combined_strict_oop"], feature))[0],
                "combined_oop_std": safe_mean_std(numeric_values(rows_by_split["combined_strict_oop"], feature))[1],
                "post_vs_train_smd": smd,
                "feature_shift_observed": shifted,
            }
        )
        top_candidates.append({"feature": feature, "feature_kind": kind, "post_vs_train_smd": smd, "abs_smd": abs(smd) if smd is not None else -1.0})
    top_shifted = sorted(top_candidates, key=lambda item: item["abs_smd"], reverse=True)[:10]
    return output, {
        "feature_shift_observed": any(item["feature_shift_observed"] for item in output),
        "top_shifted_features": top_shifted,
        "amount_volume_scale_shift": any(item["feature_kind"] == "amount_volume" and item["abs_smd"] >= 0.8 for item in top_shifted),
        "intraday_return_volatility_shift": any(item["feature_kind"] == "intraday_return_volatility" and item["abs_smd"] >= 0.8 for item in top_shifted),
    }


def anchor_breakdown_rows(rows_by_split: dict[str, list[dict[str, Any]]], metrics_rows: Sequence[dict[str, str]], prediction_rows: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    post_rows = rows_by_split["post_sprint_oop"]
    grouped = group_by(post_rows, "trade_date")
    post_summary = find_prediction_summary(prediction_rows, FOCUS_FAMILY_ID, "logistic_balanced_scaled", "post_sprint_oop")
    metric_summary = find_metric(metrics_rows, FOCUS_FAMILY_ID, "logistic_balanced_scaled", "post_sprint_oop")
    total_groups = len(post_rows)
    rows: list[dict[str, Any]] = []
    for anchor_date, selected in sorted(grouped.items()):
        labels = label_distribution(selected, FOCUS_LABEL)
        returns = numeric_values(selected, "future_return_3d")
        rows.append(
            {
                "anchor_date": anchor_date,
                "etf_count": len({row.get("etf_code") for row in selected}),
                "group_count": len(selected),
                "group_share_of_post": len(selected) / total_groups if total_groups else None,
                "label_0_count": labels["label_0_count"],
                "label_1_count": labels["label_1_count"],
                "positive_rate": labels["positive_rate"],
                "prediction_distribution": post_summary.get("prediction_distribution", "unavailable_split_level_only"),
                "candidate_balanced_accuracy": metric_summary.get("balanced_accuracy"),
                "candidate_roc_auc": metric_summary.get("roc_auc"),
                "candidate_pr_auc": metric_summary.get("pr_auc"),
                "error_count": "unavailable_missing_row_level_predictions",
                "false_positive": "unavailable_missing_row_level_predictions",
                "false_negative": "unavailable_missing_row_level_predictions",
                "probability_min": post_summary.get("probability_min"),
                "probability_max": post_summary.get("probability_max"),
                "probability_mean": post_summary.get("probability_mean"),
                "daily_return_mean": mean(returns) if returns else None,
                "daily_return_min": min(returns) if returns else None,
                "daily_return_max": max(returns) if returns else None,
                "single_day_dominates_result": len(selected) / total_groups >= 0.35 if total_groups else False,
            }
        )
    return rows


def etf_breakdown_rows(rows_by_split: dict[str, list[dict[str, Any]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    post_rows = rows_by_split["post_sprint_oop"]
    grouped = group_by(post_rows, "etf_code")
    total = len(post_rows)
    output: list[dict[str, Any]] = []
    for etf_code, selected in sorted(grouped.items()):
        labels = label_distribution(selected, FOCUS_LABEL)
        output.append(
            {
                "etf_code": etf_code,
                "group_count": len(selected),
                "group_share_of_post": len(selected) / total if total else None,
                "label_0_count": labels["label_0_count"],
                "label_1_count": labels["label_1_count"],
                "positive_rate": labels["positive_rate"],
                "prediction_positive_rate": "unavailable_missing_row_level_predictions",
                "balanced_accuracy": "unavailable_missing_row_level_predictions",
                "error_rate": "unavailable_missing_row_level_predictions",
            }
        )
    max_share = max((row["group_share_of_post"] or 0 for row in output), default=0)
    concentration = max_share >= 0.4
    return output, {
        "etf_concentration_observed": concentration,
        "max_group_share": max_share,
        "worst_etf": "unavailable_missing_row_level_predictions",
        "best_etf": "unavailable_missing_row_level_predictions",
        "note": "ETF-level error leadership requires row-level prediction output.",
    }


def date_concentration_check(anchor_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    max_share = max((float(row["group_share_of_post"] or 0) for row in anchor_rows), default=0.0)
    high_positive_rate_dates = [
        row["anchor_date"]
        for row in anchor_rows
        if row["positive_rate"] is not None and (float(row["positive_rate"]) <= 0.2 or float(row["positive_rate"]) >= 0.8)
    ]
    return {
        "date_concentration_observed": max_share >= 0.35 or bool(high_positive_rate_dates),
        "max_group_share": max_share,
        "high_label_skew_dates": high_positive_rate_dates,
    }


def sample_power_check(split_manifest: dict[str, Any], post_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    post_anchor_count = len(split_manifest.get("post_sprint_oop_anchor_dates", []))
    post_group_count = len(post_rows)
    return {
        "post_sprint_anchor_count": post_anchor_count,
        "post_sprint_group_count": post_group_count,
        "post_sprint_oop_underpowered": post_anchor_count < 10 or post_group_count < 50,
        "minimum_anchor_count": 10,
        "minimum_group_count": 50,
    }


def candidate_specific_diagnosis(oop_report: dict[str, Any], metrics_rows: Sequence[dict[str, str]], prediction_rows: Sequence[dict[str, str]], sample_power: dict[str, Any]) -> dict[str, Any]:
    candidate = next((item for item in oop_report.get("candidate_results", []) if item.get("family_id") == FOCUS_FAMILY_ID), {})
    pre_metric = find_metric(metrics_rows, FOCUS_FAMILY_ID, "logistic_balanced_scaled", "pre_sprint_oop")
    post_metric = find_metric(metrics_rows, FOCUS_FAMILY_ID, "logistic_balanced_scaled", "post_sprint_oop")
    combined_metric = find_metric(metrics_rows, FOCUS_FAMILY_ID, "logistic_balanced_scaled", "combined_strict_oop")
    post_summary = find_prediction_summary(prediction_rows, FOCUS_FAMILY_ID, "logistic_balanced_scaled", "post_sprint_oop")
    pre_count = to_float(pre_metric.get("row_count")) or 0
    post_count = to_float(post_metric.get("row_count")) or 0
    combined_from_pre_share = pre_count / (pre_count + post_count) if pre_count + post_count else None
    return {
        "family_id": FOCUS_FAMILY_ID,
        "combined_strict_oop_minimum_metrics_pass": candidate.get("combined_strict_oop_minimum_metrics_pass"),
        "diagnostic_signal_survives_minimum_standard": candidate.get("diagnostic_signal_survives_minimum_standard"),
        "combined_better_mainly_from_pre_sprint": combined_from_pre_share is not None and combined_from_pre_share >= 0.9,
        "pre_group_share_of_combined": combined_from_pre_share,
        "pre_balanced_accuracy": pre_metric.get("balanced_accuracy"),
        "post_balanced_accuracy": post_metric.get("balanced_accuracy"),
        "combined_balanced_accuracy": combined_metric.get("balanced_accuracy"),
        "post_reversal_observed": to_float(post_metric.get("balanced_accuracy")) is not None and float(post_metric["balanced_accuracy"]) < 0.5,
        "probability_collapse": post_summary.get("probability_min") == post_summary.get("probability_max"),
        "threshold_sensitivity": "unavailable_missing_row_level_probabilities",
        "label_definition_degradation_risk": "review_required: label_safe_positive_3d depends on both positive ret3d and max_drawdown_3d; post window is underpowered",
        "post_sprint_oop_underpowered": sample_power["post_sprint_oop_underpowered"],
    }


def decide(flags: dict[str, Any], missing_outputs: bool) -> str:
    if flags.get("data_quality_blocked"):
        return DECISION_BLOCKED_DATA
    if missing_outputs:
        return DECISION_BLOCKED_MISSING
    if flags.get("post_sprint_oop_underpowered"):
        return DECISION_SAMPLE_TOO_SMALL
    if flags.get("label_shift_observed"):
        return DECISION_LABEL_SHIFT
    if flags.get("etf_concentration_observed"):
        return DECISION_ETF_CONCENTRATION
    if flags.get("date_concentration_observed"):
        return DECISION_DATE_CONCENTRATION
    if flags.get("feature_shift_observed"):
        return DECISION_FEATURE_SHIFT
    return DECISION_CONTINUE


def run_review(
    oop_dir: Path = DEFAULT_OOP_DIR,
    manual_inbox: Path = DEFAULT_MANUAL_INBOX,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    enforce_output_dir: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_oop_dir = resolve_repo_path(oop_dir, repo_root)
    resolved_manual_inbox = resolve_repo_path(manual_inbox, repo_root)
    resolved_out_dir = resolve_output_dir(out_dir, repo_root, enforce=enforce_output_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    oop_check = validate_oop_outputs(resolved_oop_dir)
    p0_blockers = list(oop_check["p0_blockers"])
    if not (resolved_manual_inbox / MANUAL_CSV_NAME).exists():
        p0_blockers.append(f"manual CSV not found: {resolved_manual_inbox / MANUAL_CSV_NAME}")

    oop_report = load_json(resolved_oop_dir / "fixed_shortlist_oop_validation_report.json") if not oop_check["missing_files"] else {}
    split_manifest = load_json(resolved_oop_dir / "fixed_shortlist_oop_split_manifest.json") if not oop_check["missing_files"] else {}
    metrics_rows = load_csv_dicts(resolved_oop_dir / "fixed_shortlist_oop_metrics.csv") if (resolved_oop_dir / "fixed_shortlist_oop_metrics.csv").exists() else []
    prediction_rows = load_csv_dicts(resolved_oop_dir / "fixed_shortlist_oop_predictions_summary.csv") if (resolved_oop_dir / "fixed_shortlist_oop_predictions_summary.csv").exists() else []
    bar_rows, _columns = load_csv_rows(resolved_manual_inbox / MANUAL_CSV_NAME) if (resolved_manual_inbox / MANUAL_CSV_NAME).exists() else ([], [])
    feature_rows, feature_build_report = build_feature_rows(bar_rows) if bar_rows else ([], {})
    rows_by_split = split_rows(feature_rows, split_manifest) if feature_rows and split_manifest else {"train": [], "pre_sprint_oop": [], "post_sprint_oop": [], "combined_strict_oop": []}

    sample_power = sample_power_check(split_manifest, rows_by_split["post_sprint_oop"])
    label_rows, label_shift = label_shift_rows(rows_by_split, ["label_ret3d_gt_100bp", FOCUS_LABEL])
    feature_rows_table, feature_shift = feature_shift_rows(rows_by_split, BASE_39_FEATURES)
    anchor_rows = anchor_breakdown_rows(rows_by_split, metrics_rows, prediction_rows)
    etf_rows, etf_check = etf_breakdown_rows(rows_by_split)
    date_check = date_concentration_check(anchor_rows)
    candidate_diagnosis = candidate_specific_diagnosis(oop_report, metrics_rows, prediction_rows, sample_power)
    artifact_check = check_model_artifacts(resolved_out_dir)
    p0_blockers.extend(artifact_check["p0_blockers"])
    missing_row_predictions = not oop_check["row_level_prediction_available"]
    flags = {
        "post_sprint_oop_underpowered": sample_power["post_sprint_oop_underpowered"],
        "label_shift_observed": label_shift["label_shift_observed"],
        "etf_concentration_observed": etf_check["etf_concentration_observed"],
        "date_concentration_observed": date_check["date_concentration_observed"],
        "feature_shift_observed": feature_shift["feature_shift_observed"],
        "missing_row_level_predictions": missing_row_predictions,
        "data_quality_blocked": bool(p0_blockers),
    }
    readiness_decision = decide(flags, missing_row_predictions)
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": "blocked" if readiness_decision.startswith("POST_SPRINT_INSTABILITY_REVIEW_BLOCKED") else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "oop_dir": str(oop_dir),
            "manual_inbox": str(manual_inbox),
            "stable_bundle": False,
        },
        "readiness_decision": readiness_decision,
        "oop_output_check": oop_check,
        "sample_power": sample_power,
        "post_sprint_oop_underpowered": sample_power["post_sprint_oop_underpowered"],
        "label_distribution_shift": label_shift,
        "feature_distribution_shift": feature_shift,
        "feature_shift_table": feature_rows_table,
        "etf_level_dispersion": etf_check,
        "date_level_instability": date_check,
        "candidate_specific_diagnosis": candidate_diagnosis,
        "feature_build_report": feature_build_report,
        "flags": flags,
        "artifact_check": artifact_check,
        "p0_blockers": dedupe(p0_blockers),
        "p1_warnings": build_p1_warnings(flags),
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
    }
    write_outputs(resolved_out_dir, repo_root, report, anchor_rows, etf_rows, label_rows, feature_rows_table)
    return report


def write_outputs(
    out_dir: Path,
    repo_root: Path,
    report: dict[str, Any],
    anchor_rows: Sequence[dict[str, Any]],
    etf_rows: Sequence[dict[str, Any]],
    label_rows: Sequence[dict[str, Any]],
    feature_rows_table: Sequence[dict[str, Any]],
) -> None:
    write_json(out_dir / "post_sprint_instability_review_report.json", report)
    write_json(
        out_dir / "post_sprint_instability_decision.json",
        {
            "lab_declaration": LAB_DECLARATION,
            "readiness_decision": report["readiness_decision"],
            "status": report["status"],
            "flags": report["flags"],
            "post_sprint_oop_underpowered": report["post_sprint_oop_underpowered"],
            "stable_promotion_ready": False,
            "formal_training_ready": False,
            "qmt_ready": False,
            "order_intent_ready": False,
            "automatic_promotion_ready": False,
            "p0_blockers": report["p0_blockers"],
            "p1_warnings": report["p1_warnings"],
        },
    )
    write_csv(out_dir / "post_sprint_anchor_breakdown.csv", anchor_rows, anchor_columns())
    write_csv(out_dir / "post_sprint_etf_breakdown.csv", etf_rows, etf_columns())
    write_csv(out_dir / "post_sprint_label_distribution.csv", label_rows, label_columns())
    write_csv(out_dir / "post_sprint_prediction_error_breakdown.csv", anchor_rows, anchor_columns())
    docs_json, docs_md = docs_report(report)
    write_json(repo_root / "docs/research/aetfq3_intraday_oop_post_sprint_instability_review.json", docs_json)
    (repo_root / "docs/research/aetfq3_intraday_oop_post_sprint_instability_review.md").write_text(docs_md, encoding="utf-8")
    (out_dir / "post_sprint_instability_review_report.md").write_text(docs_md, encoding="utf-8")


def docs_report(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    docs = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_oop_post_sprint_instability_review",
        "status": report["status"],
        "readiness_decision": report["readiness_decision"],
        "post_sprint_oop_underpowered": report["post_sprint_oop_underpowered"],
        "flags": report["flags"],
        "sample_power": report["sample_power"],
        "candidate_specific_diagnosis": report["candidate_specific_diagnosis"],
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "stable_evidence": False,
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday OOP Post-Sprint Instability Review",
        "",
        "Lab-only forensic review. It does not run a new model, train, tune, save model/scaler, connect QMT, generate OrderIntent, or create Stable evidence.",
        "",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- status: {report['status']}",
        f"- post_sprint_anchor_count: {report['sample_power']['post_sprint_anchor_count']}",
        f"- post_sprint_group_count: {report['sample_power']['post_sprint_group_count']}",
        f"- post_sprint_oop_underpowered: {str(report['post_sprint_oop_underpowered']).lower()}",
        f"- missing_row_level_predictions: {str(report['flags']['missing_row_level_predictions']).lower()}",
        f"- stable_promotion_ready: {str(report['stable_promotion_ready']).lower()}",
        "",
        "## Focus Candidate",
        "",
        f"- family_id: {FOCUS_FAMILY_ID}",
        f"- pre_group_share_of_combined: {report['candidate_specific_diagnosis']['pre_group_share_of_combined']}",
        f"- post_reversal_observed: {str(report['candidate_specific_diagnosis']['post_reversal_observed']).lower()}",
        f"- threshold_sensitivity: {report['candidate_specific_diagnosis']['threshold_sensitivity']}",
    ]
    return docs, "\n".join(lines) + "\n"


def build_p1_warnings(flags: dict[str, Any]) -> list[str]:
    warnings = [
        "P1_POST_SPRINT_OOP_UNDERPOWERED_REVIEW_REQUIRED" if flags.get("post_sprint_oop_underpowered") else "",
        "P1_ROW_LEVEL_PREDICTIONS_MISSING_FP_FN_UNAVAILABLE" if flags.get("missing_row_level_predictions") else "",
        "P1_NO_STABLE_PROMOTION_WITHOUT_PROMOTION_GATE",
        "P1_REVIEW_ONLY_NOT_TRADING_ADVICE",
    ]
    return [warning for warning in warnings if warning]


def anchor_columns() -> list[str]:
    return [
        "anchor_date",
        "etf_count",
        "group_count",
        "group_share_of_post",
        "label_0_count",
        "label_1_count",
        "positive_rate",
        "prediction_distribution",
        "candidate_balanced_accuracy",
        "candidate_roc_auc",
        "candidate_pr_auc",
        "error_count",
        "false_positive",
        "false_negative",
        "probability_min",
        "probability_max",
        "probability_mean",
        "daily_return_mean",
        "daily_return_min",
        "daily_return_max",
        "single_day_dominates_result",
    ]


def etf_columns() -> list[str]:
    return [
        "etf_code",
        "group_count",
        "group_share_of_post",
        "label_0_count",
        "label_1_count",
        "positive_rate",
        "prediction_positive_rate",
        "balanced_accuracy",
        "error_rate",
    ]


def label_columns() -> list[str]:
    return [
        "label_policy",
        "split",
        "group_count",
        "label_0_count",
        "label_1_count",
        "positive_rate",
        "class_imbalance",
        "positive_rate_delta_vs_train",
        "kl_like_vs_train",
        "label_shift_observed",
    ]


def feature_columns() -> list[str]:
    return [
        "feature",
        "feature_kind",
        "train_mean",
        "train_std",
        "pre_mean",
        "pre_std",
        "post_mean",
        "post_std",
        "combined_oop_mean",
        "combined_oop_std",
        "post_vs_train_smd",
        "feature_shift_observed",
    ]


def find_metric(rows: Sequence[dict[str, str]], family_id: str, model: str, split: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("family_id") == family_id and row.get("model") == model and row.get("split") == split), {})


def find_prediction_summary(rows: Sequence[dict[str, str]], family_id: str, model: str, split: str) -> dict[str, Any]:
    return next((row for row in rows if row.get("family_id") == family_id and row.get("model") == model and row.get("split") == split), {})


def numeric_values(rows: Sequence[dict[str, Any]], field: str) -> list[float]:
    return [value for row in rows if (value := to_float(row.get(field))) is not None]


def safe_mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(mean(values)), float(pstdev(values)) if len(values) > 1 else 0.0


def standardized_mean_difference(train: Sequence[float], post: Sequence[float]) -> float | None:
    if not train or not post:
        return None
    train_mean, train_std = safe_mean_std(train)
    post_mean, post_std = safe_mean_std(post)
    if train_mean is None or post_mean is None or train_std is None or post_std is None:
        return None
    pooled = math.sqrt((train_std**2 + post_std**2) / 2)
    if pooled == 0:
        return 0.0 if train_mean == post_mean else None
    return (post_mean - train_mean) / pooled


def feature_kind(feature: str) -> str:
    name = feature.lower()
    if "volume" in name or "amount" in name:
        return "amount_volume"
    if "return" in name or "volatility" in name or "std" in name:
        return "intraday_return_volatility"
    return "other"


def label_value(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    label = int(number)
    return label if label in (0, 1) else None


def abs_delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return abs(a - b)


def bernoulli_kl(p: float | None, q: float | None) -> float | None:
    if p is None or q is None:
        return None
    eps = 1e-9
    p = min(max(p, eps), 1 - eps)
    q = min(max(q, eps), 1 - eps)
    return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))


def group_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--oop-dir", type=Path, default=DEFAULT_OOP_DIR)
    parser.add_argument("--manual-inbox", type=Path, default=DEFAULT_MANUAL_INBOX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_review(args.oop_dir, args.manual_inbox, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI must emit auditable blocker.
        print(
            json.dumps(
                {
                    "lab_declaration": LAB_DECLARATION,
                    "status": "failed",
                    "readiness_decision": DECISION_BLOCKED_DATA,
                    "p0_blockers": [str(exc)],
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
                "post_sprint_oop_underpowered": report["post_sprint_oop_underpowered"],
                "flags": report["flags"],
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
