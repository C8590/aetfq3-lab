from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Sequence

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore", message="X does not have valid feature names.*")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_group_level_past_only_feature_expansion_dryrun import (  # noqa: E402
    CORE_FEATURES,
    CROSS_SECTIONAL_FEATURES,
    OPTIONAL_ANCHOR_FEATURES,
    PAST_DAILY_FEATURES,
    append_cross_sectional_features,
    build_groups,
    calculate_anchor_features,
    clean_number,
    to_float,
)
from tools.lab.intraday_signal_recovery_sprint1 import compute_past_daily_feature_values  # noqa: E402
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts  # noqa: E402


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_fixed_shortlist_oop_no_save_validation"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_fixed_shortlist_oop_no_save_validation")
DEFAULT_MANUAL_INBOX = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox")
DEFAULT_OUT_DIR = ALLOWED_OUTPUT_DIR
MANUAL_CSV_NAME = "historical_5m_manual_export.csv"
ROW_LEVEL_PREDICTION_FILE = "fixed_shortlist_oop_row_level_predictions.csv"
SPRINT_START = "2026-04-09"
SPRINT_END = "2026-06-03"
BASE_39_FEATURES = CORE_FEATURES + OPTIONAL_ANCHOR_FEATURES + CROSS_SECTIONAL_FEATURES
ALL_LABELS = ["label_ret3d_gt_100bp", "label_safe_positive_3d"]
SHORTLIST = [
    {
        "family_id": "label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy",
        "label_policy": "label_ret3d_gt_100bp",
        "feature_set": "base_39_plus_scale_transform_policy",
        "model_family": "logistic_balanced_scaled_variants",
        "transform_policy": "scale_transform_policy",
    },
    {
        "family_id": "label_ret3d_gt_100bp|base_39_plus_past_daily_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy",
        "label_policy": "label_ret3d_gt_100bp",
        "feature_set": "base_39_plus_past_daily_plus_scale_transform_policy",
        "model_family": "logistic_balanced_scaled_variants",
        "transform_policy": "scale_transform_policy",
    },
    {
        "family_id": "label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy",
        "label_policy": "label_safe_positive_3d",
        "feature_set": "base_39_plus_scale_transform_policy",
        "model_family": "logistic_balanced_scaled_variants",
        "transform_policy": "scale_transform_policy",
    },
]
DECISION_COMPLETED = "FIXED_SHORTLIST_OOP_NO_SAVE_VALIDATION_COMPLETED_REVIEW_REQUIRED"
DECISION_SURVIVES = "FIXED_SHORTLIST_OOP_DIAGNOSTIC_SIGNAL_SURVIVES_STRICT_OOP_REVIEW_REQUIRED"
DECISION_NO_SIGNAL = "FIXED_SHORTLIST_OOP_NO_DIAGNOSTIC_SIGNAL_SURVIVAL_REVIEW_REQUIRED"
DECISION_BLOCKED_DATA = "FIXED_SHORTLIST_OOP_VALIDATION_BLOCKED_DATA_QUALITY"
DECISION_BLOCKED_LABEL = "FIXED_SHORTLIST_OOP_VALIDATION_BLOCKED_LABEL_DEFINITION_MISMATCH"
DECISION_BLOCKED_SPLIT = "FIXED_SHORTLIST_OOP_VALIDATION_BLOCKED_INSUFFICIENT_TRAIN_OR_OOP"
DECISION_BLOCKED_LEAKAGE = "FIXED_SHORTLIST_OOP_VALIDATION_BLOCKED_LEAKAGE_RISK"
MODEL_NAMES = ["dummy_most_frequent", "dummy_stratified", "logistic_balanced_scaled"]
FORBIDDEN_OUTPUT_NAMES = {
    "model.pkl",
    "scaler.pkl",
    "checkpoint.pt",
    "model.joblib",
    "scaler.joblib",
}


class FixedShortlistOopError(RuntimeError):
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
            raise FixedShortlistOopError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise FixedShortlistOopError(f"CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise FixedShortlistOopError(f"CSV has no header: {path}")
    return rows, columns


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FixedShortlistOopError(f"JSON cannot be read: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FixedShortlistOopError(f"JSON parse failed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FixedShortlistOopError(f"JSON root must be object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def manual_csv_path(manual_inbox: Path, repo_root: Path = REPO_ROOT) -> Path:
    inbox = resolve_repo_path(manual_inbox, repo_root)
    path = inbox / MANUAL_CSV_NAME
    if not path.exists():
        raise FixedShortlistOopError(f"manual CSV not found: {path}")
    return path


def validate_manual_manifest(manual_inbox: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    manifest_path = resolve_repo_path(manual_inbox, repo_root) / "MANIFEST.json"
    manifest = load_json(manifest_path)
    blockers: list[str] = []
    expected_false = [
        "training_allowed",
        "stable_effect_allowed",
        "contains_secret",
        "contains_order_intent",
        "contains_live_order",
        "contains_account",
        "contains_position",
        "contains_order",
        "contains_trade",
        "qmt_related",
    ]
    for field in expected_false:
        if manifest.get(field) is not False:
            blockers.append(f"MANIFEST.{field} must be false")
    if manifest.get("source_kind") != "broker_terminal_manual_export":
        blockers.append("MANIFEST.source_kind must be broker_terminal_manual_export")
    return {"path": str(manifest_path), "manifest": manifest, "p0_blockers": blockers, "passed": not blockers}


def build_daily_rows_from_groups(groups: dict[tuple[str, str], list[dict[str, str]]]) -> list[dict[str, Any]]:
    daily_rows: list[dict[str, Any]] = []
    for (trade_date, etf_code), group_rows in sorted(groups.items()):
        features = calculate_anchor_features(group_rows)
        daily_rows.append(
            {
                "trade_date": trade_date,
                "etf_code": etf_code,
                "close": features.get("close_last", ""),
                "volume": features.get("volume_sum", ""),
            }
        )
    return daily_rows


def append_outcomes_and_labels(rows: list[dict[str, Any]], daily_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        by_code[str(row.get("etf_code", ""))].append(row)
    for values in by_code.values():
        values.sort(key=lambda item: str(item.get("trade_date", "")))
    index_by_code_date = {
        (code, str(row.get("trade_date", ""))): index
        for code, values in by_code.items()
        for index, row in enumerate(values)
    }
    coverage = {"t_plus_1": 0, "t_plus_3": 0, "missing_t_plus_3": 0}
    for row in rows:
        code = str(row.get("etf_code", ""))
        trade_date = str(row.get("trade_date", ""))
        values = by_code.get(code, [])
        index = index_by_code_date.get((code, trade_date))
        anchor_close = to_float(row.get("close_last"))
        future_1 = values[index + 1] if index is not None and index + 1 < len(values) else None
        future_3 = values[index + 3] if index is not None and index + 3 < len(values) else None
        future_window = values[index + 1 : index + 4] if index is not None else []
        close_1 = to_float(future_1.get("close")) if future_1 else None
        close_3 = to_float(future_3.get("close")) if future_3 else None
        window_closes = [value for item in future_window if (value := to_float(item.get("close"))) is not None]
        future_return_1d = ratio_minus_one(close_1, anchor_close)
        future_return_3d = ratio_minus_one(close_3, anchor_close)
        max_drawdown_3d = ratio_minus_one(min(window_closes), anchor_close) if len(window_closes) == 3 else None
        row["future_return_1d"] = clean_number(future_return_1d)
        row["future_return_3d"] = clean_number(future_return_3d)
        row["max_drawdown_3d"] = clean_number(max_drawdown_3d)
        row["label_ret3d_gt_100bp"] = "" if future_return_3d is None else int(future_return_3d > 0.01)
        row["label_safe_positive_3d"] = (
            ""
            if future_return_3d is None or max_drawdown_3d is None
            else int(future_return_3d > 0 and max_drawdown_3d > -0.02)
        )
        row["t_plus_1_date"] = future_1.get("trade_date", "") if future_1 else ""
        row["t_plus_3_date"] = future_3.get("trade_date", "") if future_3 else ""
        row["t_plus_3_covered"] = future_return_3d is not None and max_drawdown_3d is not None
        coverage["t_plus_1"] += int(future_1 is not None)
        coverage["t_plus_3"] += int(row["t_plus_3_covered"])
        coverage["missing_t_plus_3"] += int(not row["t_plus_3_covered"])
    return coverage


def append_past_daily_features(rows: list[dict[str, Any]], daily_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        by_code[str(row.get("etf_code", ""))].append(row)
    for values in by_code.values():
        values.sort(key=lambda item: str(item.get("trade_date", "")))
    missing = {feature: 0 for feature in PAST_DAILY_FEATURES}
    generated = {feature: 0 for feature in PAST_DAILY_FEATURES}
    for row in rows:
        code = str(row.get("etf_code", ""))
        anchor = str(row.get("trade_date", ""))
        history = [item for item in by_code.get(code, []) if str(item.get("trade_date", "")) <= anchor]
        values = compute_past_daily_feature_values(history)
        for feature in PAST_DAILY_FEATURES:
            value = values.get(feature)
            row[feature] = clean_number(value)
            if value is None:
                missing[feature] += 1
            else:
                generated[feature] += 1
    return {
        "past_daily_features": PAST_DAILY_FEATURES,
        "generated_value_counts": generated,
        "missing_value_counts": missing,
        "past_only_rule": "daily history filtered with trade_date <= anchor_date",
    }


def build_feature_rows(bar_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    groups = build_groups(bar_rows)
    rows: list[dict[str, Any]] = []
    for (trade_date, etf_code), group_rows in sorted(groups.items()):
        last = group_rows[-1]
        row: dict[str, Any] = {
            "trade_date": trade_date,
            "anchor_date": trade_date,
            "etf_code": etf_code,
            "bar_count": len(group_rows),
            "last_bar_datetime": last.get("datetime", ""),
        }
        row.update(calculate_anchor_features(group_rows))
        rows.append(row)
    append_cross_sectional_features(rows)
    daily_rows = build_daily_rows_from_groups(groups)
    past_daily_report = append_past_daily_features(rows, daily_rows)
    coverage = append_outcomes_and_labels(rows, daily_rows)
    feature_quality = feature_quality_check(rows)
    return rows, {
        "group_count": len(rows),
        "daily_row_count": len(daily_rows),
        "past_daily_report": past_daily_report,
        "t_plus_coverage": coverage,
        "feature_quality": feature_quality,
    }


def feature_quality_check(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    all_features = BASE_39_FEATURES + PAST_DAILY_FEATURES
    missing_by_feature: dict[str, int] = {}
    for feature in all_features:
        missing_by_feature[feature] = sum(to_float(row.get(feature)) is None for row in rows)
    return {
        "base_39_count": len(BASE_39_FEATURES),
        "past_daily_count": len(PAST_DAILY_FEATURES),
        "missing_by_feature": missing_by_feature,
        "fully_available_base_39": [feature for feature in BASE_39_FEATURES if missing_by_feature[feature] == 0],
        "fully_available_past_daily": [feature for feature in PAST_DAILY_FEATURES if missing_by_feature[feature] == 0],
    }


def ratio_minus_one(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    value = numerator / denominator - 1.0
    return value if math.isfinite(value) else None


def label_ret3d_gt_100bp(future_return_3d: float | None) -> int | None:
    return None if future_return_3d is None else int(future_return_3d > 0.01)


def label_safe_positive_3d(future_return_3d: float | None, max_drawdown_3d: float | None) -> int | None:
    return None if future_return_3d is None or max_drawdown_3d is None else int(future_return_3d > 0 and max_drawdown_3d > -0.02)


def split_anchor_dates(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    train = sorted({str(row["trade_date"]) for row in rows if SPRINT_START <= str(row["trade_date"]) <= SPRINT_END})
    pre = sorted({str(row["trade_date"]) for row in rows if str(row["trade_date"]) < SPRINT_START})
    post = sorted({str(row["trade_date"]) for row in rows if str(row["trade_date"]) > SPRINT_END})
    return {"train": train, "pre_sprint_oop": pre, "post_sprint_oop": post, "combined_strict_oop": sorted([*pre, *post])}


def feature_columns_for_set(feature_set: str) -> list[str]:
    if feature_set == "base_39_plus_scale_transform_policy":
        return list(BASE_39_FEATURES)
    if feature_set == "base_39_plus_past_daily_plus_scale_transform_policy":
        return [*BASE_39_FEATURES, *PAST_DAILY_FEATURES]
    raise FixedShortlistOopError(f"unsupported fixed feature_set: {feature_set}")


def finite_model_rows(
    rows: Sequence[dict[str, Any]],
    dates: Sequence[str],
    label_policy: str,
    feature_columns: Sequence[str],
) -> list[dict[str, Any]]:
    date_set = set(dates)
    selected: list[dict[str, Any]] = []
    for row in rows:
        if str(row.get("trade_date", "")) not in date_set:
            continue
        if label_value(row.get(label_policy)) is None:
            continue
        if all(to_float(row.get(feature)) is not None for feature in feature_columns):
            selected.append(row)
    return selected


def build_split_payload(
    rows: Sequence[dict[str, Any]],
    label_policy: str,
    feature_columns: Sequence[str],
    split_dates: dict[str, list[str]],
) -> dict[str, Any]:
    train_rows = finite_model_rows(rows, split_dates["train"], label_policy, feature_columns)
    pre_rows = finite_model_rows(rows, split_dates["pre_sprint_oop"], label_policy, feature_columns)
    post_rows = finite_model_rows(rows, split_dates["post_sprint_oop"], label_policy, feature_columns)
    combined_rows = [*pre_rows, *post_rows]
    blockers: list[str] = []
    if not train_rows:
        blockers.append("train rows are empty")
    if not combined_rows:
        blockers.append("combined strict OOP rows are empty")
    if len(set(row["trade_date"] for row in train_rows) & set(row["trade_date"] for row in combined_rows)) > 0:
        blockers.append("train and OOP anchors overlap")
    for name, selected in (("train", train_rows), ("combined_strict_oop", combined_rows)):
        if selected and len(set(label_value(row.get(label_policy)) for row in selected)) < 2:
            blockers.append(f"{name} has single-class label distribution")
    return {
        "label_policy": label_policy,
        "feature_columns": list(feature_columns),
        "train_rows": train_rows,
        "pre_sprint_oop_rows": pre_rows,
        "post_sprint_oop_rows": post_rows,
        "combined_strict_oop_rows": combined_rows,
        "passed": not blockers,
        "p0_blockers": blockers,
    }


def train_only_scale(
    train_rows: Sequence[dict[str, Any]],
    eval_rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
) -> dict[str, Any]:
    x_train = rows_to_matrix(train_rows, feature_columns)
    x_eval = rows_to_matrix(eval_rows, feature_columns)
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval) if x_eval else []
    return {
        "x_train": x_train_scaled.tolist(),
        "x_eval": x_eval_scaled.tolist() if hasattr(x_eval_scaled, "tolist") else [],
        "audit": {
            "fit_scope": "train_only",
            "fit_row_count": len(train_rows),
            "eval_row_count": len(eval_rows),
            "fit_feature_count": len(feature_columns),
            "eval_fit_performed": False,
            "train_means": [float(value) for value in scaler.mean_.tolist()],
            "train_vars": [float(value) for value in scaler.var_.tolist()],
        },
    }


def fit_models_for_candidate(split: dict[str, Any]) -> dict[str, Any]:
    train_rows = split["train_rows"]
    y_train = rows_to_labels(train_rows, split["label_policy"])
    feature_columns = split["feature_columns"]
    eval_sets = {
        "pre_sprint_oop": split["pre_sprint_oop_rows"],
        "post_sprint_oop": split["post_sprint_oop_rows"],
        "combined_strict_oop": split["combined_strict_oop_rows"],
    }
    scaled_by_eval = {name: train_only_scale(train_rows, rows, feature_columns) for name, rows in eval_sets.items()}
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
    predictions_summary: list[dict[str, Any]] = []
    row_level_predictions: list[dict[str, Any]] = []
    for model_name, model in models.items():
        train_matrix = scaled_by_eval["combined_strict_oop"]["x_train"] if model_name == "logistic_balanced_scaled" else rows_to_matrix(train_rows, feature_columns)
        model.fit(train_matrix, y_train)
        metrics[model_name] = {}
        row_eval_sets = {
            "train": train_rows,
            **eval_sets,
        }
        for split_name, eval_rows in eval_sets.items():
            x_eval = matrix_for_model_split(model_name, split_name, eval_rows, train_matrix, feature_columns, scaled_by_eval)
            y_eval = rows_to_labels(eval_rows, split["label_policy"])
            predictions = [int(value) for value in model.predict(x_eval)] if eval_rows else []
            scores = probability_scores(model, x_eval, predictions)
            score = score_predictions(y_eval, predictions, scores)
            metrics[model_name][split_name] = score
            predictions_summary.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "row_count": len(eval_rows),
                    "prediction_distribution": json.dumps(score["prediction_distribution"], sort_keys=True),
                    "probability_min": score["probability_summary"]["min"],
                    "probability_max": score["probability_summary"]["max"],
                    "probability_mean": score["probability_summary"]["mean"],
                }
            )
        for split_name, eval_rows in row_eval_sets.items():
            x_eval = matrix_for_model_split(model_name, split_name, eval_rows, train_matrix, feature_columns, scaled_by_eval)
            predictions = [int(value) for value in model.predict(x_eval)] if eval_rows else []
            scores = probability_scores(model, x_eval, predictions)
            row_level_predictions.extend(build_row_level_prediction_rows(split_name, eval_rows, split["label_policy"], model_name, predictions, scores))
    return {
        "metrics": metrics,
        "scaler_audit": scaled_by_eval["combined_strict_oop"]["audit"],
        "prediction_summary_rows": predictions_summary,
        "row_level_prediction_rows": row_level_predictions,
    }


def matrix_for_model_split(
    model_name: str,
    split_name: str,
    eval_rows: Sequence[dict[str, Any]],
    train_matrix: Sequence[Sequence[float]],
    feature_columns: Sequence[str],
    scaled_by_eval: dict[str, dict[str, Any]],
) -> Sequence[Sequence[float]]:
    if model_name != "logistic_balanced_scaled":
        return rows_to_matrix(eval_rows, feature_columns)
    if split_name == "train":
        return train_matrix
    return scaled_by_eval[split_name]["x_eval"]


def build_row_level_prediction_rows(
    split_name: str,
    rows: Sequence[dict[str, Any]],
    label_policy: str,
    model_name: str,
    predictions: Sequence[int],
    scores: Sequence[float],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row, prediction, probability in zip(rows, predictions, scores, strict=True):
        label = label_value(row.get(label_policy))
        output.append(
            {
                "model": model_name,
                "label_policy": label_policy,
                "split_name": split_name,
                "anchor_date": row.get("anchor_date") or row.get("trade_date"),
                "etf_code": row.get("etf_code"),
                "label": label,
                "prediction": prediction,
                "probability": probability,
                "is_correct": label == prediction if label is not None else False,
                "error_type": prediction_error_type(label, prediction),
                "future_return_3d": row.get("future_return_3d"),
                "train_or_oop": "train" if split_name == "train" else "oop",
                "is_pre_sprint_oop": split_name == "pre_sprint_oop",
                "is_post_sprint_oop": split_name == "post_sprint_oop",
                "is_combined_oop": split_name == "combined_strict_oop",
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


def probability_scores(model: Any, x_eval: Sequence[Sequence[float]], predictions: Sequence[int]) -> list[float]:
    if not x_eval:
        return []
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x_eval)
        classes = [int(value) for value in model.classes_]
        if 1 in classes:
            index = classes.index(1)
            return [float(row[index]) for row in proba]
    return [float(value) for value in predictions]


def score_predictions(y_true: Sequence[int], predictions: Sequence[int], scores: Sequence[float]) -> dict[str, Any]:
    if not y_true:
        return empty_metrics()
    roc_auc = float(roc_auc_score(y_true, scores)) if len(set(y_true)) == 2 and len(set(scores)) > 1 else None
    pr_auc = float(average_precision_score(y_true, scores)) if len(set(y_true)) == 2 else None
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    prevalence = sum(y_true) / len(y_true)
    return {
        "row_count": len(y_true),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "label_prevalence": prevalence,
        "confusion_matrix": {
            "tn": int(matrix[0][0]),
            "fp": int(matrix[0][1]),
            "fn": int(matrix[1][0]),
            "tp": int(matrix[1][1]),
        },
        "prediction_distribution": distribution_from_values(predictions),
        "probability_summary": summarize_scores(scores),
    }


def empty_metrics() -> dict[str, Any]:
    return {
        "row_count": 0,
        "accuracy": None,
        "balanced_accuracy": None,
        "roc_auc": None,
        "pr_auc": None,
        "label_prevalence": None,
        "confusion_matrix": {"tn": 0, "fp": 0, "fn": 0, "tp": 0},
        "prediction_distribution": {"0": 0, "1": 0},
        "probability_summary": {"min": None, "max": None, "mean": None},
    }


def summarize_scores(scores: Sequence[float]) -> dict[str, float | None]:
    if not scores:
        return {"min": None, "max": None, "mean": None}
    return {"min": float(min(scores)), "max": float(max(scores)), "mean": float(mean(scores))}


def collapse_check(metrics: dict[str, Any]) -> dict[str, Any]:
    logistic = metrics["logistic_balanced_scaled"]
    flags: list[str] = []
    by_split: dict[str, Any] = {}
    for split_name, score in logistic.items():
        dist = score["prediction_distribution"]
        single_class = dist["0"] == 0 or dist["1"] == 0
        probability = score["probability_summary"]
        probability_collapse = probability["min"] == probability["max"] if probability["min"] is not None else True
        by_split[split_name] = {
            "single_class_prediction_collapse": single_class,
            "probability_collapse": probability_collapse,
            "prediction_distribution": dist,
            "probability_summary": probability,
        }
        if single_class:
            flags.append(f"{split_name}:single_class_prediction_collapse")
        if probability_collapse:
            flags.append(f"{split_name}:probability_collapse")
    return {"passed": not flags, "by_split": by_split, "flags": flags}


def dispersion_checks(rows_by_split: dict[str, list[dict[str, Any]]], label_policy: str, metrics: dict[str, Any]) -> dict[str, Any]:
    etf_level: dict[str, Any] = {}
    date_level: dict[str, Any] = {}
    for split_name, rows in rows_by_split.items():
        etf_level[split_name] = {
            code: distribution_from_values([label_value(row[label_policy]) for row in selected if label_value(row[label_policy]) is not None])
            for code, selected in group_by(rows, "etf_code").items()
        }
        date_level[split_name] = {
            date: distribution_from_values([label_value(row[label_policy]) for row in selected if label_value(row[label_policy]) is not None])
            for date, selected in group_by(rows, "trade_date").items()
        }
    pre_ba = metrics["logistic_balanced_scaled"]["pre_sprint_oop"].get("balanced_accuracy")
    post_ba = metrics["logistic_balanced_scaled"]["post_sprint_oop"].get("balanced_accuracy")
    return {
        "etf_level_label_distribution": etf_level,
        "date_level_label_distribution": date_level,
        "pre_post_divergence": {
            "pre_balanced_accuracy": pre_ba,
            "post_balanced_accuracy": post_ba,
            "opposite_conclusion": pre_ba is not None and post_ba is not None and ((pre_ba > 0.5 and post_ba < 0.5) or (pre_ba < 0.5 and post_ba > 0.5)),
        },
    }


def group_by(rows: Sequence[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key, ""))].append(row)
    return dict(grouped)


def summarize_split_manifest(rows: Sequence[dict[str, Any]], split_dates: dict[str, list[str]]) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "sprint_discovery_window": {"start": SPRINT_START, "end": SPRINT_END},
        "no_overlap_assertion": not (set(split_dates["train"]) & set(split_dates["combined_strict_oop"])),
        "train_anchor_dates": split_dates["train"],
        "pre_sprint_oop_anchor_dates": split_dates["pre_sprint_oop"],
        "post_sprint_oop_anchor_dates": split_dates["post_sprint_oop"],
        "combined_strict_oop_anchor_dates": split_dates["combined_strict_oop"],
    }
    for split_name, dates in split_dates.items():
        selected = [row for row in rows if str(row.get("trade_date", "")) in set(dates)]
        covered = [row for row in selected if row.get("t_plus_3_covered") is True]
        manifest[split_name] = {
            "anchor_count": len(dates),
            "etf_count": len({row.get("etf_code") for row in selected}),
            "group_count": len(selected),
            "t_plus_3_covered_group_count": len(covered),
            "label_distribution": {label: distribution_from_values([label_value(row[label]) for row in covered if label_value(row[label]) is not None]) for label in ALL_LABELS},
        }
    return manifest


def candidate_survives(candidate_result: dict[str, Any]) -> bool:
    combined_pass = combined_minimum_metrics_pass(candidate_result)
    pre = candidate_result["metrics"]["logistic_balanced_scaled"]["pre_sprint_oop"]
    post = candidate_result["metrics"]["logistic_balanced_scaled"]["post_sprint_oop"]
    divergence = candidate_result["dispersion_checks"]["pre_post_divergence"]
    return all(
        [
            combined_pass,
            divergence["opposite_conclusion"] is False,
            pre["row_count"] > 0,
            post["row_count"] > 0,
        ]
    )


def combined_minimum_metrics_pass(candidate_result: dict[str, Any]) -> bool:
    metrics = candidate_result["metrics"]
    logistic = metrics["logistic_balanced_scaled"]
    dummy = metrics["dummy_most_frequent"]
    combined = logistic["combined_strict_oop"]
    combined_dummy = dummy["combined_strict_oop"]
    collapse = candidate_result["collapse_check"]
    return all(
        [
            combined["balanced_accuracy"] is not None and combined_dummy["balanced_accuracy"] is not None,
            combined["balanced_accuracy"] > combined_dummy["balanced_accuracy"],
            combined["balanced_accuracy"] > 0.5,
            combined["roc_auc"] is not None and combined["roc_auc"] > 0.5,
            combined["pr_auc"] is not None and combined["label_prevalence"] is not None and combined["pr_auc"] >= combined["label_prevalence"],
            collapse["by_split"]["combined_strict_oop"]["single_class_prediction_collapse"] is False,
        ]
    )


def decide(candidate_results: Sequence[dict[str, Any]], blockers: Sequence[str]) -> str:
    if any("label definition" in blocker.lower() for blocker in blockers):
        return DECISION_BLOCKED_LABEL
    if any("leakage" in blocker.lower() or "overlap" in blocker.lower() for blocker in blockers):
        return DECISION_BLOCKED_LEAKAGE
    if any("train" in blocker.lower() or "oop" in blocker.lower() or "single-class" in blocker.lower() for blocker in blockers):
        return DECISION_BLOCKED_SPLIT
    if blockers:
        return DECISION_BLOCKED_DATA
    survived = [item for item in candidate_results if item["diagnostic_signal_survives_minimum_standard"]]
    if survived:
        return DECISION_SURVIVES
    if any(item.get("combined_strict_oop_minimum_metrics_pass") for item in candidate_results):
        return DECISION_COMPLETED
    if candidate_results:
        return DECISION_NO_SIGNAL
    return DECISION_COMPLETED


def build_docs_report(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    summary = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_fixed_shortlist_oop_no_save_validation",
        "status": report["status"],
        "readiness_decision": report["readiness_decision"],
        "source": report["input_sources"],
        "strict_oop": report["split_manifest"]["sprint_discovery_window"],
        "candidate_count": len(report["candidate_results"]),
        "survived_candidate_count": sum(item["diagnostic_signal_survives_minimum_standard"] for item in report["candidate_results"]),
        "formal_training": False,
        "model_saved": False,
        "scaler_saved": False,
        "stable_promotion_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "stable_evidence": False,
        "row_level_predictions_emitted": report["row_level_predictions_emitted"],
        "row_level_prediction_file": report["row_level_prediction_file"],
        "row_level_prediction_row_count": report["row_level_prediction_row_count"],
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday Fixed Shortlist OOP No-Save Validation",
        "",
        "Lab-only strict OOP diagnostic validation. It does not save models/scalers, does not write output/, does not connect QMT, does not generate OrderIntent, and is not Stable evidence.",
        "",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- train_anchor_count: {len(report['split_manifest']['train_anchor_dates'])}",
        f"- pre_sprint_oop_anchor_count: {len(report['split_manifest']['pre_sprint_oop_anchor_dates'])}",
        f"- post_sprint_oop_anchor_count: {len(report['split_manifest']['post_sprint_oop_anchor_dates'])}",
        f"- combined_strict_oop_anchor_count: {len(report['split_manifest']['combined_strict_oop_anchor_dates'])}",
        f"- candidate_count: {len(report['candidate_results'])}",
        f"- survived_candidate_count: {summary['survived_candidate_count']}",
        f"- row_level_predictions_emitted: {str(report['row_level_predictions_emitted']).lower()}",
        f"- row_level_prediction_row_count: {report['row_level_prediction_row_count']}",
        f"- model_saved: {str(report['model_saved']).lower()}",
        f"- scaler_saved: {str(report['scaler_saved']).lower()}",
        f"- stable_promotion_ready: {str(report['stable_promotion_ready']).lower()}",
        "",
        "## Candidate Summary",
    ]
    for item in report["candidate_results"]:
        combined = item["metrics"]["logistic_balanced_scaled"]["combined_strict_oop"]
        lines.append(
            f"- {item['family_id']}: balanced_accuracy={combined['balanced_accuracy']}, roc_auc={combined['roc_auc']}, pr_auc={combined['pr_auc']}, survives={str(item['diagnostic_signal_survives_minimum_standard']).lower()}"
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
    required_columns = {"trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume"}
    missing = sorted(required_columns - set(columns))
    if missing:
        blockers.append("manual CSV missing required columns: " + ", ".join(missing))

    feature_rows, build_report_payload = build_feature_rows(bar_rows)
    split_dates = split_anchor_dates(feature_rows)
    split_manifest = summarize_split_manifest(feature_rows, split_dates)
    if not split_manifest["no_overlap_assertion"]:
        blockers.append("split leakage: train and strict OOP anchor dates overlap")

    artifact_before = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_before["p0_blockers"])

    candidate_results: list[dict[str, Any]] = []
    metrics_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    row_level_prediction_rows: list[dict[str, Any]] = []
    if not blockers:
        for candidate in SHORTLIST:
            features = feature_columns_for_set(candidate["feature_set"])
            split = build_split_payload(feature_rows, candidate["label_policy"], features, split_dates)
            if not split["passed"]:
                blockers.extend([f"{candidate['family_id']}: {blocker}" for blocker in split["p0_blockers"]])
                continue
            model_result = fit_models_for_candidate(split)
            collapse = collapse_check(model_result["metrics"])
            dispersion = dispersion_checks(
                {
                    "pre_sprint_oop": split["pre_sprint_oop_rows"],
                    "post_sprint_oop": split["post_sprint_oop_rows"],
                    "combined_strict_oop": split["combined_strict_oop_rows"],
                },
                candidate["label_policy"],
                model_result["metrics"],
            )
            result = {
                **candidate,
                "feature_count": len(features),
                "train_group_count": len(split["train_rows"]),
                "pre_sprint_oop_group_count": len(split["pre_sprint_oop_rows"]),
                "post_sprint_oop_group_count": len(split["post_sprint_oop_rows"]),
                "combined_strict_oop_group_count": len(split["combined_strict_oop_rows"]),
                "train_label_distribution": distribution_from_values(rows_to_labels(split["train_rows"], candidate["label_policy"])),
                "pre_sprint_oop_label_distribution": distribution_from_values(rows_to_labels(split["pre_sprint_oop_rows"], candidate["label_policy"])),
                "post_sprint_oop_label_distribution": distribution_from_values(rows_to_labels(split["post_sprint_oop_rows"], candidate["label_policy"])),
                "combined_strict_oop_label_distribution": distribution_from_values(rows_to_labels(split["combined_strict_oop_rows"], candidate["label_policy"])),
                "metrics": model_result["metrics"],
                "scaler_audit": model_result["scaler_audit"],
                "collapse_check": collapse,
                "dispersion_checks": dispersion,
            }
            result["combined_strict_oop_minimum_metrics_pass"] = combined_minimum_metrics_pass(result)
            result["diagnostic_signal_survives_minimum_standard"] = candidate_survives(result)
            candidate_results.append(result)
            prediction_rows.extend({**row, "family_id": candidate["family_id"]} for row in model_result["prediction_summary_rows"])
            row_level_prediction_rows.extend(
                {
                    **row,
                    "candidate_id": candidate["family_id"],
                    "family_id": candidate["family_id"],
                    "label_policy": candidate["label_policy"],
                    "feature_set": candidate["feature_set"],
                    "model_family": candidate["model_family"],
                }
                for row in model_result["row_level_prediction_rows"]
            )
            for model_name, by_split in model_result["metrics"].items():
                for split_name, metric in by_split.items():
                    metrics_rows.append(flat_metric_row(candidate, model_name, split_name, metric))

    artifact_after = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_after["p0_blockers"])
    readiness_decision = decide(candidate_results, dedupe(blockers))
    p1_warnings = [
        "P1_DIAGNOSTIC_ONLY_NOT_FORMAL_MODEL_EVIDENCE",
        "P1_REQUIRES_HUMAN_REVIEW",
        "P1_NO_STABLE_PROMOTION_WITHOUT_PROMOTION_GATE",
    ]
    if candidate_results and not any(item["diagnostic_signal_survives_minimum_standard"] for item in candidate_results):
        p1_warnings.append("P1_STRICT_OOP_SIGNAL_NOT_SURVIVED")
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": "blocked" if readiness_decision.startswith("FIXED_SHORTLIST_OOP_VALIDATION_BLOCKED") else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "manual_inbox": str(manual_inbox),
            "manual_csv": str(csv_path),
            "manual_manifest": manifest_check["path"],
            "source": "broker terminal manual export",
            "stable_bundle": False,
        },
        "strict_oop_definition": {
            "discovery_anchor_start": SPRINT_START,
            "discovery_anchor_end": SPRINT_END,
            "oop_rule": "anchor_date < 2026-04-09 or anchor_date > 2026-06-03",
        },
        "readiness_decision": readiness_decision,
        "shortlist": SHORTLIST,
        "manual_manifest_check": {key: value for key, value in manifest_check.items() if key != "manifest"},
        "data_build_report": build_report_payload,
        "split_manifest": split_manifest,
        "candidate_results": candidate_results,
        "row_level_predictions_emitted": True,
        "row_level_prediction_file": ROW_LEVEL_PREDICTION_FILE,
        "row_level_prediction_row_count": len(row_level_prediction_rows),
        "row_level_prediction_counts_by_split": summarize_row_level_counts(row_level_prediction_rows),
        "row_level_prediction_required_columns": row_level_prediction_columns(),
        "artifact_check_before": artifact_before,
        "artifact_check_after": artifact_after,
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
    }
    docs_json, docs_md = build_docs_report(report)
    write_json(resolved_out_dir / "fixed_shortlist_oop_validation_report.json", report)
    write_json(resolved_out_dir / "fixed_shortlist_oop_split_manifest.json", split_manifest)
    write_json(
        resolved_out_dir / "fixed_shortlist_oop_decision.json",
        {
            "lab_declaration": LAB_DECLARATION,
            "readiness_decision": readiness_decision,
            "status": report["status"],
            "p0_blockers": report["p0_blockers"],
            "p1_warnings": report["p1_warnings"],
            "formal_training": False,
            "model_saved": False,
            "scaler_saved": False,
            "stable_promotion_ready": False,
            "qmt_ready": False,
            "order_intent_ready": False,
            "stable_evidence": False,
            "row_level_predictions_emitted": report["row_level_predictions_emitted"],
            "row_level_prediction_file": report["row_level_prediction_file"],
            "row_level_prediction_row_count": report["row_level_prediction_row_count"],
        },
    )
    write_csv(resolved_out_dir / "fixed_shortlist_oop_metrics.csv", metrics_rows, metric_columns())
    write_csv(
        resolved_out_dir / "fixed_shortlist_oop_predictions_summary.csv",
        prediction_rows,
        ["family_id", "model", "split", "row_count", "prediction_distribution", "probability_min", "probability_max", "probability_mean"],
    )
    write_csv(resolved_out_dir / ROW_LEVEL_PREDICTION_FILE, row_level_prediction_rows, row_level_prediction_columns())
    (resolved_out_dir / "fixed_shortlist_oop_validation_report.md").write_text(docs_md, encoding="utf-8")
    write_json(repo_root / "docs/research/aetfq3_intraday_fixed_shortlist_oop_no_save_validation.json", docs_json)
    (repo_root / "docs/research/aetfq3_intraday_fixed_shortlist_oop_no_save_validation.md").write_text(docs_md, encoding="utf-8")
    return report


def flat_metric_row(candidate: dict[str, Any], model: str, split_name: str, metric: dict[str, Any]) -> dict[str, Any]:
    return {
        "family_id": candidate["family_id"],
        "label_policy": candidate["label_policy"],
        "feature_set": candidate["feature_set"],
        "model": model,
        "split": split_name,
        "row_count": metric["row_count"],
        "accuracy": metric["accuracy"],
        "balanced_accuracy": metric["balanced_accuracy"],
        "roc_auc": metric["roc_auc"],
        "pr_auc": metric["pr_auc"],
        "label_prevalence": metric["label_prevalence"],
        "tn": metric["confusion_matrix"]["tn"],
        "fp": metric["confusion_matrix"]["fp"],
        "fn": metric["confusion_matrix"]["fn"],
        "tp": metric["confusion_matrix"]["tp"],
        "prediction_0": metric["prediction_distribution"]["0"],
        "prediction_1": metric["prediction_distribution"]["1"],
        "probability_min": metric["probability_summary"]["min"],
        "probability_max": metric["probability_summary"]["max"],
        "probability_mean": metric["probability_summary"]["mean"],
    }


def metric_columns() -> list[str]:
    return [
        "family_id",
        "label_policy",
        "feature_set",
        "model",
        "split",
        "row_count",
        "accuracy",
        "balanced_accuracy",
        "roc_auc",
        "pr_auc",
        "label_prevalence",
        "tn",
        "fp",
        "fn",
        "tp",
        "prediction_0",
        "prediction_1",
        "probability_min",
        "probability_max",
        "probability_mean",
    ]


def row_level_prediction_columns() -> list[str]:
    return [
        "candidate_id",
        "family_id",
        "label_policy",
        "feature_set",
        "model_family",
        "model",
        "split_name",
        "anchor_date",
        "etf_code",
        "label",
        "prediction",
        "probability",
        "is_correct",
        "error_type",
        "future_return_3d",
        "train_or_oop",
        "is_pre_sprint_oop",
        "is_post_sprint_oop",
        "is_combined_oop",
    ]


def summarize_row_level_counts(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in rows:
        grouped[(str(row.get("candidate_id", "")), str(row.get("model", "")), str(row.get("split_name", "")))] += 1
    return [
        {
            "candidate_id": candidate_id,
            "model": model,
            "split_name": split_name,
            "row_count": row_count,
        }
        for (candidate_id, model, split_name), row_count in sorted(grouped.items())
    ]


def rows_to_matrix(rows: Sequence[dict[str, Any]], feature_columns: Sequence[str]) -> list[list[float]]:
    return [[float(to_float(row.get(feature))) for feature in feature_columns] for row in rows]


def rows_to_labels(rows: Sequence[dict[str, Any]], label_policy: str) -> list[int]:
    return [int(label_value(row.get(label_policy))) for row in rows if label_value(row.get(label_policy)) is not None]


def label_value(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    label = int(number)
    return label if label in (0, 1) else None


def distribution_from_values(values: Sequence[int | None]) -> dict[str, int]:
    return {"0": sum(1 for value in values if value == 0), "1": sum(1 for value in values if value == 1)}


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
                "candidate_count": len(report["candidate_results"]),
                "survived_candidate_count": sum(item["diagnostic_signal_survives_minimum_standard"] for item in report["candidate_results"]),
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
