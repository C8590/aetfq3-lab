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
    DEFAULT_MANUAL_INBOX,
    MANUAL_CSV_NAME,
    ROW_LEVEL_PREDICTION_FILE,
    build_feature_rows,
    feature_columns_for_set,
    load_csv_rows,
    row_level_prediction_columns,
    to_float,
)
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts  # noqa: E402


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_oop_post_sprint_reversal_attribution"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_oop_post_sprint_reversal_attribution")
DEFAULT_OOP_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_fixed_shortlist_oop_no_save_validation")
DEFAULT_INSTABILITY_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_oop_post_sprint_instability_review")
DEFAULT_OUT_DIR = ALLOWED_OUTPUT_DIR
FOCUS_FAMILY_ID = "label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy"
FOCUS_LABEL = "label_safe_positive_3d"
FOCUS_FEATURE_SET = "base_39_plus_scale_transform_policy"
FOCUS_MODEL = "logistic_balanced_scaled"
THRESHOLD_GRID = [value / 100 for value in range(30, 71, 5)]

DECISION_SAMPLE_TOO_SMALL = "POST_SPRINT_REVERSAL_ATTRIBUTION_SAMPLE_TOO_SMALL_PRIMARY"
DECISION_DATE_REGIME = "POST_SPRINT_REVERSAL_ATTRIBUTION_DATE_REGIME_SHIFT_PRIMARY"
DECISION_LABEL_SHIFT = "POST_SPRINT_REVERSAL_ATTRIBUTION_LABEL_SHIFT_PRIMARY"
DECISION_THRESHOLD = "POST_SPRINT_REVERSAL_ATTRIBUTION_THRESHOLD_SENSITIVITY_PRIMARY"
DECISION_FEATURE_SHIFT = "POST_SPRINT_REVERSAL_ATTRIBUTION_FEATURE_SHIFT_PRIMARY"
DECISION_ETF = "POST_SPRINT_REVERSAL_ATTRIBUTION_ETF_CONCENTRATION_PRIMARY"
DECISION_CONTINUE = "POST_SPRINT_REVERSAL_ATTRIBUTION_NO_SINGLE_CAUSE_CONTINUE_OOP_ACCUMULATION"
DECISION_BLOCKED_MISSING_ROW = "POST_SPRINT_REVERSAL_ATTRIBUTION_BLOCKED_MISSING_ROW_LEVEL_DIAGNOSTICS"
DECISION_BLOCKED_DATA = "POST_SPRINT_REVERSAL_ATTRIBUTION_BLOCKED_DATA_QUALITY"


class ReversalAttributionError(RuntimeError):
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
            raise ReversalAttributionError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ReversalAttributionError(f"JSON cannot be read: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ReversalAttributionError(f"JSON parse failed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReversalAttributionError(f"JSON root must be object: {path}")
    return payload


def load_csv_dicts(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise ReversalAttributionError(f"CSV cannot be read: {path}: {exc}") from exc
    return rows, columns


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def focus_rows(rows: Sequence[dict[str, str]], split_name: str | None = None) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if row.get("candidate_id") == FOCUS_FAMILY_ID
        and row.get("model") == FOCUS_MODEL
        and (split_name is None or row.get("split_name") == split_name)
    ]


def label_value(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    label = int(number)
    return label if label in (0, 1) else None


def error_type(label: int | None, prediction: int | None) -> str:
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


def confusion_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    return {name: sum(1 for row in rows if row.get("error_type") == name) for name in ["TP", "TN", "FP", "FN", "NA"]}


def balanced_accuracy_from_counts(counts: dict[str, int]) -> float | None:
    pos_total = counts["TP"] + counts["FN"]
    neg_total = counts["TN"] + counts["FP"]
    if not pos_total or not neg_total:
        return None
    return ((counts["TP"] / pos_total) + (counts["TN"] / neg_total)) / 2


def positive_rate(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    values = [label_value(row.get(field)) for row in rows]
    valid = [value for value in values if value is not None]
    return sum(valid) / len(valid) if valid else None


def numeric_values(rows: Sequence[dict[str, Any]], field: str) -> list[float]:
    return [value for row in rows if (value := to_float(row.get(field))) is not None]


def group_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def date_attribution_rows(row_level_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    post_rows = focus_rows(row_level_rows, "post_sprint_oop")
    total_error = sum(1 for row in post_rows if row.get("error_type") in {"FP", "FN"})
    total_rows = len(post_rows)
    output: list[dict[str, Any]] = []
    for anchor_date, selected in sorted(group_by(post_rows, "anchor_date").items()):
        counts = confusion_counts(selected)
        errors = counts["FP"] + counts["FN"]
        probabilities = numeric_values(selected, "probability")
        output.append(
            {
                "anchor_date": anchor_date,
                "group_count": len(selected),
                "group_share": len(selected) / total_rows if total_rows else None,
                "label_positive_rate": positive_rate(selected, "label"),
                "prediction_positive_rate": positive_rate(selected, "prediction"),
                "fp": counts["FP"],
                "fn": counts["FN"],
                "tp": counts["TP"],
                "tn": counts["TN"],
                "error_rate": errors / len(selected) if selected else None,
                "balanced_accuracy": balanced_accuracy_from_counts(counts),
                "probability_mean": mean(probabilities) if probabilities else None,
                "probability_min": min(probabilities) if probabilities else None,
                "probability_max": max(probabilities) if probabilities else None,
                "date_contribution_to_total_error": errors / total_error if total_error else None,
                "whether_date_dominates_reversal": (errors / total_error >= 0.35) if total_error else False,
                "dominant_error_type": "FP" if counts["FP"] > counts["FN"] else "FN" if counts["FN"] > counts["FP"] else "balanced",
            }
        )
    first_half = [row for row in output if row["anchor_date"] <= "2026-06-05"]
    second_half = [row for row in output if row["anchor_date"] > "2026-06-05"]
    regime_reversal = bool(first_half and second_half and sum(row["fp"] for row in first_half) > sum(row["fn"] for row in first_half) and sum(row["fn"] for row in second_half) > sum(row["fp"] for row in second_half))
    return output, {
        "post_sprint_row_count": total_rows,
        "post_sprint_total_error": total_error,
        "date_dominates_reversal": any(row["whether_date_dominates_reversal"] for row in output),
        "front_back_regime_reversal_observed": regime_reversal,
        "front_half_fp": sum(row["fp"] for row in first_half),
        "front_half_fn": sum(row["fn"] for row in first_half),
        "back_half_fp": sum(row["fp"] for row in second_half),
        "back_half_fn": sum(row["fn"] for row in second_half),
    }


def etf_attribution_rows(row_level_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    post_rows = focus_rows(row_level_rows, "post_sprint_oop")
    total_error = sum(1 for row in post_rows if row.get("error_type") in {"FP", "FN"})
    total_rows = len(post_rows)
    output: list[dict[str, Any]] = []
    for etf_code, selected in sorted(group_by(post_rows, "etf_code").items()):
        counts = confusion_counts(selected)
        errors = counts["FP"] + counts["FN"]
        probabilities = numeric_values(selected, "probability")
        group_share = len(selected) / total_rows if total_rows else None
        error_share = errors / total_error if total_error else None
        output.append(
            {
                "etf_code": etf_code,
                "group_count": len(selected),
                "group_share": group_share,
                "label_positive_rate": positive_rate(selected, "label"),
                "prediction_positive_rate": positive_rate(selected, "prediction"),
                "fp": counts["FP"],
                "fn": counts["FN"],
                "tp": counts["TP"],
                "tn": counts["TN"],
                "error_rate": errors / len(selected) if selected else None,
                "error_share": error_share,
                "balanced_accuracy": balanced_accuracy_from_counts(counts),
                "probability_mean": mean(probabilities) if probabilities else None,
                "whether_etf_dominates_reversal": bool(group_share is not None and error_share is not None and group_share >= 0.4 and error_share >= 0.4),
            }
        )
    max_error_share = max((row["error_share"] or 0 for row in output), default=0)
    max_group_share = max((row["group_share"] or 0 for row in output), default=0)
    return output, {
        "etf_concentration_primary": any(row["whether_etf_dominates_reversal"] for row in output),
        "max_error_share": max_error_share,
        "max_group_share": max_group_share,
        "note": "single ETF error_rate alone is not treated as primary unless group share and error share are both concentrated",
    }


def threshold_sensitivity_rows(row_level_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    by_split: dict[str, list[dict[str, Any]]] = {}
    for split_name in ["train", "pre_sprint_oop", "post_sprint_oop", "combined_strict_oop"]:
        rows = focus_rows(row_level_rows, split_name)
        by_split[split_name] = []
        for threshold in THRESHOLD_GRID:
            evaluated: list[dict[str, Any]] = []
            for row in rows:
                label = label_value(row.get("label"))
                probability = to_float(row.get("probability"))
                if label is None or probability is None:
                    continue
                prediction = int(probability >= threshold)
                evaluated.append({"label": label, "prediction": prediction, "error_type": error_type(label, prediction)})
            counts = confusion_counts(evaluated)
            errors = counts["FP"] + counts["FN"]
            current = {
                "split_name": split_name,
                "threshold": threshold,
                "row_count": len(evaluated),
                "balanced_accuracy": balanced_accuracy_from_counts(counts),
                "fp": counts["FP"],
                "fn": counts["FN"],
                "tp": counts["TP"],
                "tn": counts["TN"],
                "error_rate": errors / len(evaluated) if evaluated else None,
                "prediction_positive_rate": positive_rate(evaluated, "prediction"),
                "threshold_selection_allowed": False,
                "threshold_tuned_on_post_sprint": False,
                "threshold_sensitivity_is_diagnostic_only": True,
            }
            output.append(current)
            by_split[split_name].append(current)
    post = by_split["post_sprint_oop"]
    at_05 = next((row for row in post if row["threshold"] == 0.5), {})
    best_post = max(post, key=lambda row: to_float(row.get("balanced_accuracy")) or -1.0, default={})
    pre_at_best = next((row for row in by_split["pre_sprint_oop"] if row["threshold"] == best_post.get("threshold")), {})
    combined_at_best = next((row for row in by_split["combined_strict_oop"] if row["threshold"] == best_post.get("threshold")), {})
    improvement = (to_float(best_post.get("balanced_accuracy")) or 0) - (to_float(at_05.get("balanced_accuracy")) or 0)
    inconsistent = (
        to_float(best_post.get("threshold")) is not None
        and to_float(best_post.get("threshold")) != 0.5
        and (
            to_float(pre_at_best.get("balanced_accuracy")) is None
            or to_float(combined_at_best.get("balanced_accuracy")) is None
            or abs((to_float(pre_at_best.get("balanced_accuracy")) or 0) - (to_float(best_post.get("balanced_accuracy")) or 0)) >= 0.05
        )
    )
    return output, {
        "threshold_selection_allowed": False,
        "threshold_tuned_on_post_sprint": False,
        "threshold_sensitivity_is_diagnostic_only": True,
        "post_threshold_0_5": at_05,
        "post_best_threshold_by_balanced_accuracy": best_post,
        "post_best_minus_0_5_balanced_accuracy": improvement,
        "pre_and_combined_inconsistent_with_post_best": inconsistent,
        "threshold_sensitivity_primary": improvement >= 0.08,
    }


def label_regime_rows(row_level_rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_names = ["train", "pre_sprint_oop", "post_sprint_oop", "combined_strict_oop"]
    rates = {split: positive_rate(focus_rows(row_level_rows, split), "label") for split in split_names}
    train_rate = rates["train"]
    pre_rate = rates["pre_sprint_oop"]
    output: list[dict[str, Any]] = []
    for split_name in split_names:
        rows = focus_rows(row_level_rows, split_name)
        rate = rates[split_name]
        delta_train = abs(rate - train_rate) if rate is not None and train_rate is not None else None
        delta_pre = abs(rate - pre_rate) if rate is not None and pre_rate is not None else None
        output.append(
            {
                "split_name": split_name,
                "group_count": len(rows),
                "positive_rate": rate,
                "absolute_delta_vs_train": delta_train,
                "absolute_delta_vs_pre": delta_pre,
                "whether_label_shift_observed": bool(delta_train is not None and delta_train >= 0.15),
                "whether_post_sprint_label_regime_is_outlier": split_name == "post_sprint_oop" and bool(delta_pre is not None and delta_pre >= 0.15),
            }
        )
    post_row = next(row for row in output if row["split_name"] == "post_sprint_oop")
    return output, {
        "label_shift_primary": bool(post_row["whether_label_shift_observed"] or post_row["whether_post_sprint_label_regime_is_outlier"]),
        "post_positive_rate": post_row["positive_rate"],
        "post_delta_vs_train": post_row["absolute_delta_vs_train"],
        "post_delta_vs_pre": post_row["absolute_delta_vs_pre"],
    }


def feature_shift_rows(manual_inbox: Path, split_manifest: dict[str, Any], repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    csv_path = resolve_repo_path(manual_inbox, repo_root) / MANUAL_CSV_NAME
    bar_rows, _columns = load_csv_rows(csv_path)
    feature_rows_all, _build_report = build_feature_rows(bar_rows)
    features = feature_columns_for_set(FOCUS_FEATURE_SET)
    dates = {
        "train": set(split_manifest.get("train_anchor_dates", [])),
        "pre_sprint_oop": set(split_manifest.get("pre_sprint_oop_anchor_dates", [])),
        "post_sprint_oop": set(split_manifest.get("post_sprint_oop_anchor_dates", [])),
    }
    rows_by_split = {
        split: [row for row in feature_rows_all if str(row.get("trade_date", "")) in date_set and row.get("t_plus_3_covered") is True]
        for split, date_set in dates.items()
    }
    output: list[dict[str, Any]] = []
    for feature in features:
        train_values = numeric_values(rows_by_split["train"], feature)
        pre_values = numeric_values(rows_by_split["pre_sprint_oop"], feature)
        post_values = numeric_values(rows_by_split["post_sprint_oop"], feature)
        train_post_smd = standardized_mean_difference(train_values, post_values)
        pre_post_smd = standardized_mean_difference(pre_values, post_values)
        output.append(
            {
                "feature": feature,
                "feature_group": feature_group(feature),
                "train_mean": safe_mean(train_values),
                "pre_mean": safe_mean(pre_values),
                "post_mean": safe_mean(post_values),
                "train_vs_post_smd": train_post_smd,
                "pre_vs_post_smd": pre_post_smd,
                "abs_train_vs_post_smd": abs(train_post_smd) if train_post_smd is not None else None,
                "abs_pre_vs_post_smd": abs(pre_post_smd) if pre_post_smd is not None else None,
                "whether_feature_shift_likely_contributes_to_reversal": bool((abs(train_post_smd) if train_post_smd is not None else 0) >= 0.8 or (abs(pre_post_smd) if pre_post_smd is not None else 0) >= 0.8),
            }
        )
    output.sort(key=lambda row: row["abs_train_vs_post_smd"] if row["abs_train_vs_post_smd"] is not None else -1, reverse=True)
    return output, {
        "feature_shift_primary": any(row["whether_feature_shift_likely_contributes_to_reversal"] for row in output[:10]),
        "top_shifted_features": output[:10],
        "feature_shift_is_not_trading_rule": True,
    }


def safe_mean(values: Sequence[float]) -> float | None:
    return float(mean(values)) if values else None


def safe_mean_std(values: Sequence[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return float(mean(values)), float(pstdev(values)) if len(values) > 1 else 0.0


def standardized_mean_difference(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right:
        return None
    left_mean, left_std = safe_mean_std(left)
    right_mean, right_std = safe_mean_std(right)
    if left_mean is None or right_mean is None or left_std is None or right_std is None:
        return None
    pooled = math.sqrt((left_std**2 + right_std**2) / 2)
    if pooled == 0:
        return 0.0 if left_mean == right_mean else None
    return (right_mean - left_mean) / pooled


def feature_group(feature: str) -> str:
    name = feature.lower()
    if "volume" in name:
        return "volume"
    if "amount" in name:
        return "amount"
    if "volatility" in name or "std" in name or "spike" in name:
        return "volatility"
    if "return" in name:
        return "intraday_return"
    if "range" in name or "high" in name or "low" in name or "vwap" in name or "close" in name or "open" in name:
        return "position/range"
    return "other"


def decide(sample_power: dict[str, Any], date_summary: dict[str, Any], etf_summary: dict[str, Any], threshold_summary: dict[str, Any], label_summary: dict[str, Any], feature_summary: dict[str, Any], blockers: Sequence[str]) -> str:
    if any("row-level" in blocker.lower() for blocker in blockers):
        return DECISION_BLOCKED_MISSING_ROW
    if blockers:
        return DECISION_BLOCKED_DATA
    if sample_power.get("post_sprint_group_count", 0) < 50:
        return DECISION_SAMPLE_TOO_SMALL
    if date_summary["date_dominates_reversal"] or date_summary["front_back_regime_reversal_observed"]:
        return DECISION_DATE_REGIME
    if etf_summary["etf_concentration_primary"]:
        return DECISION_ETF
    if label_summary["label_shift_primary"]:
        return DECISION_LABEL_SHIFT
    if threshold_summary["threshold_sensitivity_primary"]:
        return DECISION_THRESHOLD
    if feature_summary["feature_shift_primary"]:
        return DECISION_FEATURE_SHIFT
    return DECISION_CONTINUE


def run_attribution(
    oop_dir: Path = DEFAULT_OOP_DIR,
    instability_dir: Path = DEFAULT_INSTABILITY_DIR,
    manual_inbox: Path = DEFAULT_MANUAL_INBOX,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    enforce_output_dir: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_oop_dir = resolve_repo_path(oop_dir, repo_root)
    resolved_instability_dir = resolve_repo_path(instability_dir, repo_root)
    resolved_out_dir = resolve_output_dir(out_dir, repo_root, enforce=enforce_output_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    blockers: list[str] = []

    row_path = resolved_oop_dir / ROW_LEVEL_PREDICTION_FILE
    if not row_path.exists():
        blockers.append(f"row-level diagnostics missing: {row_path}")
        row_level_rows: list[dict[str, str]] = []
        row_columns: list[str] = []
    else:
        row_level_rows, row_columns = load_csv_dicts(row_path)
        missing_columns = sorted(set(row_level_prediction_columns()) - set(row_columns))
        if missing_columns:
            blockers.append("row-level diagnostics schema mismatch: " + ", ".join(missing_columns))

    oop_report = load_json(resolved_oop_dir / "fixed_shortlist_oop_validation_report.json") if (resolved_oop_dir / "fixed_shortlist_oop_validation_report.json").exists() else {}
    split_manifest = oop_report.get("split_manifest", {})
    instability_report = load_json(resolved_instability_dir / "post_sprint_instability_review_report.json") if (resolved_instability_dir / "post_sprint_instability_review_report.json").exists() else {}
    sample_power = instability_report.get("sample_power", {})
    if not sample_power:
        sample_power = {
            "post_sprint_anchor_count": len(split_manifest.get("post_sprint_oop_anchor_dates", [])),
            "post_sprint_group_count": len(focus_rows(row_level_rows, "post_sprint_oop")),
            "post_sprint_oop_underpowered": len(focus_rows(row_level_rows, "post_sprint_oop")) < 50,
            "minimum_anchor_count": 10,
            "minimum_group_count": 50,
        }

    artifact_check = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_check["p0_blockers"])

    if blockers:
        date_rows: list[dict[str, Any]] = []
        date_summary: dict[str, Any] = {}
        etf_rows: list[dict[str, Any]] = []
        etf_summary: dict[str, Any] = {}
        threshold_rows: list[dict[str, Any]] = []
        threshold_summary: dict[str, Any] = {}
        label_rows: list[dict[str, Any]] = []
        label_summary: dict[str, Any] = {}
        feature_rows_table: list[dict[str, Any]] = []
        feature_summary: dict[str, Any] = {}
    else:
        date_rows, date_summary = date_attribution_rows(row_level_rows)
        etf_rows, etf_summary = etf_attribution_rows(row_level_rows)
        threshold_rows, threshold_summary = threshold_sensitivity_rows(row_level_rows)
        label_rows, label_summary = label_regime_rows(row_level_rows)
        feature_rows_table, feature_summary = feature_shift_rows(manual_inbox, split_manifest, repo_root)

    readiness_decision = decide(sample_power, date_summary, etf_summary, threshold_summary, label_summary, feature_summary, blockers)
    post_underpowered = bool(sample_power.get("post_sprint_oop_underpowered") or sample_power.get("post_sprint_group_count", 0) < 50)
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": "blocked" if readiness_decision.startswith("POST_SPRINT_REVERSAL_ATTRIBUTION_BLOCKED") else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "oop_dir": str(oop_dir),
            "instability_dir": str(instability_dir),
            "manual_inbox": str(manual_inbox),
            "row_level_prediction_file": str(row_path),
            "stable_bundle": False,
        },
        "focus_candidate": {
            "candidate_id": FOCUS_FAMILY_ID,
            "label_policy": FOCUS_LABEL,
            "feature_set": FOCUS_FEATURE_SET,
            "model": FOCUS_MODEL,
        },
        "readiness_decision": readiness_decision,
        "sample_power": sample_power,
        "post_sprint_oop_underpowered": post_underpowered,
        "date_level_attribution": date_summary,
        "etf_level_attribution": etf_summary,
        "threshold_sensitivity_attribution": threshold_summary,
        "label_regime_attribution": label_summary,
        "feature_shift_attribution": feature_summary,
        "artifact_check": artifact_check,
        "p0_blockers": dedupe(blockers),
        "p1_warnings": build_p1_warnings(post_underpowered),
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
        "fixed_shortlist_oop_validation_ready_for_stable": False,
        "stable_evidence": False,
        "stable_affected": False,
        "threshold_selection_allowed": False,
        "threshold_tuned_on_post_sprint": False,
        "threshold_sensitivity_is_diagnostic_only": True,
        "not_trading_advice": True,
    }
    write_outputs(resolved_out_dir, repo_root, report, date_rows, etf_rows, threshold_rows, label_rows, feature_rows_table)
    return report


def build_p1_warnings(post_underpowered: bool) -> list[str]:
    warnings = [
        "P1_POST_SPRINT_OOP_UNDERPOWERED_REVIEW_REQUIRED" if post_underpowered else "",
        "P1_THRESHOLD_SENSITIVITY_DIAGNOSTIC_ONLY_NO_PARAMETER_UPDATE",
        "P1_NO_STABLE_PROMOTION_WITHOUT_PROMOTION_GATE",
        "P1_REVIEW_ONLY_NOT_TRADING_ADVICE",
    ]
    return [warning for warning in warnings if warning]


def write_outputs(
    out_dir: Path,
    repo_root: Path,
    report: dict[str, Any],
    date_rows: Sequence[dict[str, Any]],
    etf_rows: Sequence[dict[str, Any]],
    threshold_rows: Sequence[dict[str, Any]],
    label_rows: Sequence[dict[str, Any]],
    feature_rows_table: Sequence[dict[str, Any]],
) -> None:
    write_json(out_dir / "post_sprint_reversal_attribution_report.json", report)
    write_json(
        out_dir / "post_sprint_reversal_attribution_decision.json",
        {
            "lab_declaration": LAB_DECLARATION,
            "readiness_decision": report["readiness_decision"],
            "status": report["status"],
            "post_sprint_oop_underpowered": report["post_sprint_oop_underpowered"],
            "fixed_shortlist_oop_validation_ready_for_stable": False,
            "stable_promotion_ready": False,
            "formal_training_ready": False,
            "qmt_ready": False,
            "order_intent_ready": False,
            "automatic_promotion_ready": False,
            "threshold_selection_allowed": False,
            "threshold_tuned_on_post_sprint": False,
            "threshold_sensitivity_is_diagnostic_only": True,
            "p0_blockers": report["p0_blockers"],
            "p1_warnings": report["p1_warnings"],
        },
    )
    write_csv(out_dir / "post_sprint_date_error_attribution.csv", date_rows, date_columns())
    write_csv(out_dir / "post_sprint_etf_error_attribution.csv", etf_rows, etf_columns())
    write_csv(out_dir / "post_sprint_threshold_sensitivity_detail.csv", threshold_rows, threshold_columns())
    write_csv(out_dir / "post_sprint_label_regime_table.csv", label_rows, label_columns())
    write_csv(out_dir / "post_sprint_feature_shift_focus.csv", feature_rows_table, feature_columns())
    docs_json, docs_md = docs_report(report)
    write_json(repo_root / "docs/research/aetfq3_intraday_oop_post_sprint_reversal_attribution.json", docs_json)
    (repo_root / "docs/research/aetfq3_intraday_oop_post_sprint_reversal_attribution.md").write_text(docs_md, encoding="utf-8")
    (out_dir / "post_sprint_reversal_attribution_report.md").write_text(docs_md, encoding="utf-8")


def docs_report(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    docs = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_oop_post_sprint_reversal_attribution",
        "status": report["status"],
        "readiness_decision": report["readiness_decision"],
        "sample_power": report["sample_power"],
        "date_level_attribution": report["date_level_attribution"],
        "etf_level_attribution": report["etf_level_attribution"],
        "threshold_sensitivity_attribution": report["threshold_sensitivity_attribution"],
        "label_regime_attribution": report["label_regime_attribution"],
        "feature_shift_attribution": report["feature_shift_attribution"],
        "fixed_shortlist_oop_validation_ready_for_stable": False,
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
        "# Intraday OOP Post-Sprint Reversal Attribution",
        "",
        "Lab-only row-level forensic attribution. It does not train, tune, change thresholds, save model/scaler, connect QMT, generate OrderIntent, or create Stable evidence.",
        "",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- status: {report['status']}",
        f"- post_sprint_oop_underpowered: {str(report['post_sprint_oop_underpowered']).lower()}",
        f"- post_sprint_group_count: {report['sample_power'].get('post_sprint_group_count')}",
        f"- date_dominates_reversal: {report['date_level_attribution'].get('date_dominates_reversal')}",
        f"- front_back_regime_reversal_observed: {report['date_level_attribution'].get('front_back_regime_reversal_observed')}",
        f"- etf_concentration_primary: {report['etf_level_attribution'].get('etf_concentration_primary')}",
        f"- threshold_selection_allowed: {str(report['threshold_selection_allowed']).lower()}",
        f"- stable_promotion_ready: {str(report['stable_promotion_ready']).lower()}",
    ]
    return docs, "\n".join(lines) + "\n"


def date_columns() -> list[str]:
    return [
        "anchor_date",
        "group_count",
        "group_share",
        "label_positive_rate",
        "prediction_positive_rate",
        "fp",
        "fn",
        "tp",
        "tn",
        "error_rate",
        "balanced_accuracy",
        "probability_mean",
        "probability_min",
        "probability_max",
        "date_contribution_to_total_error",
        "whether_date_dominates_reversal",
        "dominant_error_type",
    ]


def etf_columns() -> list[str]:
    return [
        "etf_code",
        "group_count",
        "group_share",
        "label_positive_rate",
        "prediction_positive_rate",
        "fp",
        "fn",
        "tp",
        "tn",
        "error_rate",
        "error_share",
        "balanced_accuracy",
        "probability_mean",
        "whether_etf_dominates_reversal",
    ]


def threshold_columns() -> list[str]:
    return [
        "split_name",
        "threshold",
        "row_count",
        "balanced_accuracy",
        "fp",
        "fn",
        "tp",
        "tn",
        "error_rate",
        "prediction_positive_rate",
        "threshold_selection_allowed",
        "threshold_tuned_on_post_sprint",
        "threshold_sensitivity_is_diagnostic_only",
    ]


def label_columns() -> list[str]:
    return [
        "split_name",
        "group_count",
        "positive_rate",
        "absolute_delta_vs_train",
        "absolute_delta_vs_pre",
        "whether_label_shift_observed",
        "whether_post_sprint_label_regime_is_outlier",
    ]


def feature_columns() -> list[str]:
    return [
        "feature",
        "feature_group",
        "train_mean",
        "pre_mean",
        "post_mean",
        "train_vs_post_smd",
        "pre_vs_post_smd",
        "abs_train_vs_post_smd",
        "abs_pre_vs_post_smd",
        "whether_feature_shift_likely_contributes_to_reversal",
    ]


def dedupe(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(values))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--oop-dir", type=Path, default=DEFAULT_OOP_DIR)
    parser.add_argument("--instability-dir", type=Path, default=DEFAULT_INSTABILITY_DIR)
    parser.add_argument("--manual-inbox", type=Path, default=DEFAULT_MANUAL_INBOX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_attribution(args.oop_dir, args.instability_dir, args.manual_inbox, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI emits auditable Lab blocker.
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
