from __future__ import annotations

import argparse
import csv
import importlib.util
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
REPORT_TYPE = "intraday_signal_recovery_sprint1"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint1")
DEFAULT_SAMPLES = Path(
    ".local_research_outputs/aetfq3_lab/intraday_group_level_past_only_feature_expansion_dryrun/"
    "intraday_group_level_past_only_feature_samples.csv"
)
DEFAULT_MANIFEST = Path(
    ".local_research_outputs/aetfq3_lab/intraday_group_level_past_only_feature_expansion_dryrun/"
    "intraday_group_level_past_only_feature_manifest.json"
)
DEFAULT_TRANSFORM_POLICY = Path(
    ".local_research_outputs/aetfq3_lab/intraday_group_level_feature_scale_diagnostic/"
    "transform_policy_recommendation.json"
)
DEFAULT_TRANSFORM_SMOKE = Path(
    ".local_research_outputs/aetfq3_lab/intraday_group_level_transform_aware_no_save_smoke/"
    "intraday_group_level_transform_aware_no_save_smoke_report.json"
)
DEFAULT_DAILY = Path(
    ".local_artifact_backup/aetfq3_lab_sources/intraday_5m_public_larger_eligible_anchor_collection/"
    "future_window_daily_ohlcv.csv"
)

BASE_LABEL = "three_day_positive_label"
LABEL_POLICIES = [
    BASE_LABEL,
    "label_ret3d_gt_0bp",
    "label_ret3d_gt_20bp",
    "label_ret3d_gt_50bp",
    "label_ret3d_gt_100bp",
    "label_safe_positive_3d",
    "label_neutral_band_20bp",
    "label_neutral_band_50bp",
]
NEW_LABEL_POLICIES = [label for label in LABEL_POLICIES if label != BASE_LABEL]
OUTCOME_COLUMNS = ["future_return_1d", "future_return_3d", "max_drawdown_3d"]
PAST_DAILY_FEATURES = [
    "prev_1d_return",
    "prev_3d_return",
    "prev_5d_return",
    "prev_10d_return",
    "prev_5d_volatility",
    "prev_10d_volatility",
    "prev_5d_volume_zscore",
    "prev_close_to_5d_ma",
    "prev_close_to_10d_ma",
]
MODEL_NAMES = [
    "dummy_most_frequent",
    "dummy_stratified",
    "logistic_balanced_scaled",
    "logistic_log1p_scaled_balanced",
    "random_forest_shallow_no_save",
    "hist_gradient_boosting_no_save",
]
OPTIONAL_MODEL_IMPORTS = {
    "lightgbm_no_save": "lightgbm",
    "xgboost_no_save": "xgboost",
    "catboost_no_save": "catboost",
}
BOUNDARY_FALSE_FIELDS = {
    "formal_model_evidence": False,
    "stable_promotion_ready": False,
    "formal_training_ready": False,
    "qmt_ready": False,
    "order_intent_ready": False,
    "automatic_promotion_ready": False,
    "model_saved": False,
    "scaler_saved": False,
    "checkpoint_saved": False,
    "gpu_used": False,
    "torchrun_used": False,
    "qmt_used": False,
    "order_intent_generated": False,
    "stable_affected": False,
    "metrics_are_effectiveness_evidence": False,
    "not_trading_advice": True,
}

DECISION_CANDIDATE = "SIGNAL_RECOVERY_SPRINT1_DIAGNOSTIC_CANDIDATE_FOUND_REVIEW_REQUIRED"
DECISION_NONE = "SIGNAL_RECOVERY_SPRINT1_NO_DIAGNOSTIC_SIGNAL_CANDIDATE_FOUND"
DECISION_LEAKAGE = "SIGNAL_RECOVERY_SPRINT1_BLOCKED_LEAKAGE_P0"
DECISION_RUNTIME = "SIGNAL_RECOVERY_SPRINT1_BLOCKED_RUNTIME_ERROR"


class SignalRecoverySprintError(RuntimeError):
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
        raise SignalRecoverySprintError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SignalRecoverySprintError(f"JSON cannot be read: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SignalRecoverySprintError(f"JSON parse failed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SignalRecoverySprintError(f"JSON root must be object: {path}")
    return payload


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise SignalRecoverySprintError(f"CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise SignalRecoverySprintError(f"CSV has no header: {path}")
    return rows, columns


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def to_float(value: Any) -> float | None:
    text = str(value).strip()
    if text == "" or text.lower() in {"na", "nan", "none", "null"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def clean_number(value: float | int | None) -> float | int | str:
    if value is None:
        return ""
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def ratio_minus_one(numerator: float | None, denominator: float | None) -> float | None:
    ratio = safe_div(numerator, denominator)
    return None if ratio is None else ratio - 1.0


def generate_label_variants(rows: list[dict[str, Any]]) -> dict[str, Any]:
    null_counts = {label: 0 for label in NEW_LABEL_POLICIES}
    for row in rows:
        future_return_3d = to_float(row.get("future_return_3d"))
        max_drawdown_3d = to_float(row.get("max_drawdown_3d"))
        variants = {
            "label_ret3d_gt_0bp": None if future_return_3d is None else int(future_return_3d > 0),
            "label_ret3d_gt_20bp": None if future_return_3d is None else int(future_return_3d > 0.002),
            "label_ret3d_gt_50bp": None if future_return_3d is None else int(future_return_3d > 0.005),
            "label_ret3d_gt_100bp": None if future_return_3d is None else int(future_return_3d > 0.01),
            "label_safe_positive_3d": None
            if future_return_3d is None or max_drawdown_3d is None
            else int(future_return_3d > 0 and max_drawdown_3d > -0.02),
            "label_neutral_band_20bp": neutral_band_label(future_return_3d, 0.002),
            "label_neutral_band_50bp": neutral_band_label(future_return_3d, 0.005),
        }
        for label, value in variants.items():
            row[label] = clean_number(value)
            if value is None:
                null_counts[label] += 1
    return {
        "generated_label_policies": NEW_LABEL_POLICIES,
        "definition_source": "future_return_3d and max_drawdown_3d outcome columns; labels are excluded from feature_columns",
        "null_counts": null_counts,
    }


def neutral_band_label(future_return_3d: float | None, band: float) -> int | None:
    if future_return_3d is None or abs(future_return_3d) <= band:
        return None
    return int(future_return_3d > band)


def derive_anchor_date_split(rows: Sequence[dict[str, Any]], anchor_column: str = "trade_date") -> tuple[list[str], list[str]]:
    dates = sorted({str(row.get(anchor_column, "")).strip() for row in rows if str(row.get(anchor_column, "")).strip()})
    if len(dates) < 2:
        return dates, []
    train_count = max(1, int(len(dates) * 0.70))
    if train_count >= len(dates):
        train_count = len(dates) - 1
    return dates[:train_count], dates[train_count:]


def summarize_label_policies(
    rows: Sequence[dict[str, Any]],
    train_dates: Sequence[str],
    valid_dates: Sequence[str],
    anchor_column: str = "trade_date",
) -> dict[str, Any]:
    train_set = set(train_dates)
    valid_set = set(valid_dates)
    summaries: dict[str, Any] = {}
    for label in LABEL_POLICIES:
        values = [label_value(row.get(label)) for row in rows]
        non_null = [(row, value) for row, value in zip(rows, values) if value is not None]
        label_0_count = sum(1 for _, value in non_null if value == 0)
        label_1_count = sum(1 for _, value in non_null if value == 1)
        train_values = [
            value
            for row, value in non_null
            if str(row.get(anchor_column, "")).strip() in train_set
        ]
        valid_values = [
            value
            for row, value in non_null
            if str(row.get(anchor_column, "")).strip() in valid_set
        ]
        class_count = int(label_0_count > 0) + int(label_1_count > 0)
        min_class_count = min(label_0_count, label_1_count) if class_count == 2 else 0
        train_balance = distribution_from_values(train_values)
        valid_balance = distribution_from_values(valid_values)
        summaries[label] = {
            "row_count": len(non_null),
            "null_count": len(rows) - len(non_null),
            "label_0_count": label_0_count,
            "label_1_count": label_1_count,
            "positive_rate": safe_div(float(label_1_count), float(len(non_null))) if non_null else None,
            "class_count": class_count,
            "min_class_count": min_class_count,
            "train_class_balance": train_balance,
            "valid_class_balance": valid_balance,
            "eligible_for_diagnostic_smoke": is_label_policy_eligible(train_balance, valid_balance, min_class_count),
        }
    return summaries


def is_label_policy_eligible(train_balance: dict[str, int], valid_balance: dict[str, int], min_class_count: int) -> bool:
    return (
        min_class_count >= 2
        and train_balance["0"] > 0
        and train_balance["1"] > 0
        and valid_balance["0"] > 0
        and valid_balance["1"] > 0
    )


def label_value(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    rounded = int(number)
    if rounded not in (0, 1):
        return None
    return rounded


def recover_past_daily_features(
    sample_rows: list[dict[str, Any]],
    daily_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    daily_by_code: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in daily_rows:
        code = str(row.get("etf_code", "")).strip()
        if not code:
            continue
        daily_by_code[code].append(row)
    for rows in daily_by_code.values():
        rows.sort(key=lambda item: str(item.get("trade_date", "")).strip())

    skipped_counts = {feature: 0 for feature in PAST_DAILY_FEATURES}
    generated_counts = {feature: 0 for feature in PAST_DAILY_FEATURES}
    audit_examples: list[dict[str, Any]] = []
    for row in sample_rows:
        code = str(row.get("etf_code", "")).strip()
        anchor_date = str(row.get("trade_date") or row.get("anchor_date") or "").strip()
        history = [
            daily
            for daily in daily_by_code.get(code, [])
            if str(daily.get("trade_date", "")).strip() <= anchor_date
        ]
        values = compute_past_daily_feature_values(history)
        for feature in PAST_DAILY_FEATURES:
            value = values.get(feature)
            row[feature] = clean_number(value)
            if value is None:
                skipped_counts[feature] += 1
            else:
                generated_counts[feature] += 1
        if len(audit_examples) < 5:
            audit_examples.append(
                {
                    "trade_date": anchor_date,
                    "etf_code": code,
                    "history_end_date": str(history[-1].get("trade_date")) if history else None,
                    "history_row_count_at_or_before_anchor": len(history),
                    "used_future_rows": False,
                }
            )

    recovered_features = [feature for feature, count in generated_counts.items() if count > 0]
    fully_available_features = [
        feature for feature, count in generated_counts.items() if count == len(sample_rows) and sample_rows
    ]
    return {
        "attempted_features": PAST_DAILY_FEATURES,
        "recovered_features": recovered_features,
        "fully_available_features": fully_available_features,
        "generated_value_counts": generated_counts,
        "skipped_value_counts": skipped_counts,
        "skipped_features": [feature for feature, count in generated_counts.items() if count == 0],
        "past_only_rule": "daily rows filtered with trade_date <= anchor date D; no D+1/D+3 or future outcome fields used",
        "past_only_audit_examples": audit_examples,
    }


def compute_past_daily_feature_values(history: Sequence[dict[str, Any]]) -> dict[str, float | None]:
    closes = [to_float(row.get("close")) for row in history]
    volumes = [to_float(row.get("volume")) for row in history]
    closes_clean = [value for value in closes if value is not None]
    volumes_clean = [value for value in volumes if value is not None]
    latest_close = closes_clean[-1] if closes_clean else None

    def close_return(days: int) -> float | None:
        if len(closes_clean) <= days:
            return None
        return ratio_minus_one(closes_clean[-1], closes_clean[-days - 1])

    returns = [
        ratio_minus_one(closes_clean[index], closes_clean[index - 1])
        for index in range(1, len(closes_clean))
    ]
    returns = [value for value in returns if value is not None]

    def volatility(window: int) -> float | None:
        if len(returns) < window:
            return None
        return pstdev(returns[-window:])

    def close_to_ma(window: int) -> float | None:
        if latest_close is None or len(closes_clean) < window:
            return None
        return ratio_minus_one(latest_close, mean(closes_clean[-window:]))

    volume_window = volumes_clean[-5:] if len(volumes_clean) >= 5 else []
    volume_zscore = None
    if len(volume_window) == 5:
        volume_std = pstdev(volume_window)
        if volume_std != 0:
            volume_zscore = (volume_window[-1] - mean(volume_window)) / volume_std

    return {
        "prev_1d_return": close_return(1),
        "prev_3d_return": close_return(3),
        "prev_5d_return": close_return(5),
        "prev_10d_return": close_return(10),
        "prev_5d_volatility": volatility(5),
        "prev_10d_volatility": volatility(10),
        "prev_5d_volume_zscore": volume_zscore,
        "prev_close_to_5d_ma": close_to_ma(5),
        "prev_close_to_10d_ma": close_to_ma(10),
    }


def build_manifest(
    source_manifest: dict[str, Any],
    base_features: Sequence[str],
    recovered_past_features: Sequence[str],
) -> dict[str, Any]:
    feature_columns = dedupe([*base_features, *recovered_past_features])
    label_columns = dedupe([*string_list(source_manifest.get("label_columns")), *LABEL_POLICIES])
    outcome_columns = dedupe([*string_list(source_manifest.get("outcome_columns")), *OUTCOME_COLUMNS])
    manifest = {
        **source_manifest,
        "manifest_version": "intraday_signal_recovery_sprint1_v1",
        "sample_subtype": "intraday_signal_recovery_sprint1",
        "report_type": REPORT_TYPE,
        "feature_columns": feature_columns,
        "generated_feature_count": len(feature_columns),
        "generated_labels": LABEL_POLICIES,
        "label_columns": label_columns,
        "outcome_columns": outcome_columns,
        "label_generation_method": "intraday_signal_recovery_sprint1_dryrun_label_variants_v1",
        "feature_label_overlap_check": True,
        "label_generation_authorized": True,
        "supervised_training_allowed": False,
        "training_allowed": False,
        "stable_effect_allowed": False,
        "contains_order_intent": False,
        "contains_live_order": False,
        "contains_secret": False,
        "model_training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "formal_model_evidence": False,
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "not_trading_advice": True,
        "sprint_scope": "signal_recovery_diagnostic_only",
    }
    return manifest


def build_feature_set_variants(
    base_features: Sequence[str],
    recovered_past_features: Sequence[str],
) -> dict[str, dict[str, Any]]:
    past = list(recovered_past_features)
    return {
        "base_39_features": {
            "feature_columns": list(base_features),
            "past_daily_context_included": False,
            "scale_transform_policy_included": False,
        },
        "base_39_plus_past_daily": {
            "feature_columns": dedupe([*base_features, *past]),
            "past_daily_context_included": True,
            "scale_transform_policy_included": False,
        },
        "base_39_plus_scale_transform_policy": {
            "feature_columns": list(base_features),
            "past_daily_context_included": False,
            "scale_transform_policy_included": True,
        },
        "base_39_plus_past_daily_plus_scale_transform_policy": {
            "feature_columns": dedupe([*base_features, *past]),
            "past_daily_context_included": True,
            "scale_transform_policy_included": True,
        },
    }


def check_feature_set_leakage(
    feature_columns: Sequence[str],
    label_columns: Sequence[str],
    outcome_columns: Sequence[str],
) -> dict[str, Any]:
    feature_set = set(feature_columns)
    p0_blockers: list[str] = []
    label_overlap = sorted(feature_set & set(label_columns))
    outcome_overlap = sorted(feature_set & set(outcome_columns))
    future_features = sorted(column for column in feature_set if column.startswith("future_"))
    label_features = sorted(column for column in feature_set if "label" in column.lower())
    if label_overlap:
        p0_blockers.append("feature_columns intersects label_columns: " + ", ".join(label_overlap))
    if outcome_overlap:
        p0_blockers.append("feature_columns intersects outcome_columns: " + ", ".join(outcome_overlap))
    if future_features:
        p0_blockers.append("feature_columns contains future_* fields: " + ", ".join(future_features))
    if label_features:
        p0_blockers.append("feature_columns contains label-pattern fields: " + ", ".join(label_features))
    return {
        "passed": not p0_blockers,
        "feature_count": len(feature_columns),
        "p0_blockers": p0_blockers,
    }


def build_feature_set_report(
    variants: dict[str, dict[str, Any]],
    label_columns: Sequence[str],
    outcome_columns: Sequence[str],
) -> dict[str, Any]:
    return {
        name: {
            **payload,
            "leakage_check": check_feature_set_leakage(
                payload["feature_columns"],
                label_columns,
                outcome_columns,
            ),
        }
        for name, payload in variants.items()
    }


def run_diagnostic_suite(
    rows: Sequence[dict[str, Any]],
    label_summaries: dict[str, Any],
    feature_set_variants: dict[str, dict[str, Any]],
    transform_policy: dict[str, Any],
    train_dates: Sequence[str],
    valid_dates: Sequence[str],
    out_dir: Path,
) -> dict[str, Any]:
    artifact_check_before = check_model_artifacts(out_dir)
    optional_models = available_optional_models()
    model_names = [*MODEL_NAMES, *optional_models]
    results: list[dict[str, Any]] = []
    p0_blockers: list[str] = []
    p0_blockers.extend(artifact_check_before["p0_blockers"])

    for label_policy, label_summary in label_summaries.items():
        if not label_summary["eligible_for_diagnostic_smoke"]:
            continue
        for feature_set_name, feature_set in feature_set_variants.items():
            leakage_check = feature_set["leakage_check"]
            if not leakage_check["passed"]:
                p0_blockers.extend(leakage_check["p0_blockers"])
                continue
            split = build_split_payload(
                rows,
                label_policy,
                feature_set["feature_columns"],
                train_dates,
                valid_dates,
            )
            if not split["passed"]:
                results.append(
                    {
                        "label_policy": label_policy,
                        "feature_set": feature_set_name,
                        "status": "skipped",
                        "skip_reasons": split["p0_blockers"],
                        "feature_count": len(feature_set["feature_columns"]),
                    }
                )
                continue
            dummy_metrics: dict[str, Any] = {}
            dummy_predictions: dict[str, list[int]] = {}
            for model_name in model_names:
                try:
                    metrics, predictions = fit_and_score_model(
                        model_name,
                        split,
                        feature_set["feature_columns"],
                        transform_policy,
                        feature_set["scale_transform_policy_included"],
                    )
                    if model_name in {"dummy_most_frequent", "dummy_stratified"}:
                        dummy_metrics[model_name] = metrics
                        dummy_predictions[model_name] = predictions
                    results.append(
                        build_model_result(
                            label_policy,
                            feature_set_name,
                            feature_set,
                            model_name,
                            metrics,
                            predictions,
                            split,
                            leakage_check,
                            dummy_metrics,
                            dummy_predictions,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - diagnostic suite records combination failures.
                    results.append(
                        {
                            "label_policy": label_policy,
                            "feature_set": feature_set_name,
                            "model": model_name,
                            "status": "runtime_error",
                            "error": str(exc),
                            "feature_count": len(feature_set["feature_columns"]),
                        }
                    )

    artifact_check_after = check_model_artifacts(out_dir)
    p0_blockers.extend(artifact_check_after["p0_blockers"])
    candidates = [
        result
        for result in results
        if result.get("candidate_gate", {}).get("diagnostic_signal_candidate") is True
    ]
    return {
        "suite_scope": "no_save_diagnostic_fitting_only",
        "models_requested": MODEL_NAMES,
        "optional_models_available": optional_models,
        "optional_models_skipped": [
            name for name in OPTIONAL_MODEL_IMPORTS if name not in optional_models
        ],
        "combination_count": len(results),
        "results": results,
        "diagnostic_candidates": candidates,
        "artifact_check_before": artifact_check_before,
        "artifact_check_after": artifact_check_after,
        "p0_blockers": dedupe(p0_blockers),
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
    }


def available_optional_models() -> list[str]:
    return [
        model_name
        for model_name, module_name in OPTIONAL_MODEL_IMPORTS.items()
        if importlib.util.find_spec(module_name) is not None
    ]


def build_split_payload(
    rows: Sequence[dict[str, Any]],
    label_policy: str,
    feature_columns: Sequence[str],
    train_dates: Sequence[str],
    valid_dates: Sequence[str],
) -> dict[str, Any]:
    train_set = set(train_dates)
    valid_set = set(valid_dates)
    train_rows = [
        row for row in rows if str(row.get("trade_date", "")).strip() in train_set
    ]
    valid_rows = [
        row for row in rows if str(row.get("trade_date", "")).strip() in valid_set
    ]
    p0_blockers: list[str] = []
    x_train, y_train, train_used, train_dropped = rows_to_matrix_and_labels(
        train_rows,
        feature_columns,
        label_policy,
    )
    x_valid, y_valid, valid_used, valid_dropped = rows_to_matrix_and_labels(
        valid_rows,
        feature_columns,
        label_policy,
    )
    if not x_train or not x_valid:
        p0_blockers.append("train and valid rows must both be non-empty after null label/feature filtering")
    if len(set(y_train)) < 2:
        p0_blockers.append("train split must contain both classes")
    if len(set(y_valid)) < 2:
        p0_blockers.append("valid split must contain both classes")
    return {
        "passed": not p0_blockers,
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "train_rows_used": train_used,
        "valid_rows_used": valid_used,
        "train_rows_dropped": train_dropped,
        "valid_rows_dropped": valid_dropped,
        "train_label_distribution": distribution_from_values(y_train),
        "valid_label_distribution": distribution_from_values(y_valid),
        "valid_prevalence": safe_div(float(sum(y_valid)), float(len(y_valid))) if y_valid else None,
        "p0_blockers": p0_blockers,
    }


def rows_to_matrix_and_labels(
    rows: Sequence[dict[str, Any]],
    feature_columns: Sequence[str],
    label_policy: str,
) -> tuple[list[list[float]], list[int], int, int]:
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
    return matrix, labels, len(matrix), dropped


def fit_and_score_model(
    model_name: str,
    split: dict[str, Any],
    feature_columns: Sequence[str],
    transform_policy: dict[str, Any],
    scale_transform_policy_included: bool,
) -> tuple[dict[str, Any], list[int]]:
    x_train = split["x_train"]
    y_train = split["y_train"]
    x_valid = split["x_valid"]
    y_valid = split["y_valid"]
    x_train_model, x_valid_model = x_train, x_valid

    if model_name == "dummy_most_frequent":
        model = DummyClassifier(strategy="most_frequent")
    elif model_name == "dummy_stratified":
        model = DummyClassifier(strategy="stratified", random_state=42)
    elif model_name == "logistic_balanced_scaled":
        x_train_model, x_valid_model = train_only_standardize(x_train, x_valid)
        model = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced", random_state=42)
    elif model_name == "logistic_log1p_scaled_balanced":
        transformed = apply_log1p_policy(
            x_train,
            x_valid,
            feature_columns,
            transform_policy,
            scale_transform_policy_included,
        )
        x_train_model, x_valid_model = train_only_standardize(transformed[0], transformed[1])
        model = LogisticRegression(max_iter=200, solver="liblinear", class_weight="balanced", random_state=42)
    elif model_name == "random_forest_shallow_no_save":
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )
    elif model_name == "hist_gradient_boosting_no_save":
        model = HistGradientBoostingClassifier(
            max_iter=50,
            max_leaf_nodes=7,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
        )
    elif model_name == "lightgbm_no_save":
        from lightgbm import LGBMClassifier

        model = LGBMClassifier(
            n_estimators=30,
            max_depth=3,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            verbose=-1,
            device_type="cpu",
        )
    elif model_name == "xgboost_no_save":
        from xgboost import XGBClassifier

        model = XGBClassifier(
            n_estimators=30,
            max_depth=2,
            learning_rate=0.05,
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0,
        )
    elif model_name == "catboost_no_save":
        from catboost import CatBoostClassifier

        model = CatBoostClassifier(
            iterations=30,
            depth=3,
            learning_rate=0.05,
            random_seed=42,
            verbose=False,
            task_type="CPU",
            allow_writing_files=False,
        )
    else:
        raise SignalRecoverySprintError(f"unknown model: {model_name}")

    model.fit(x_train_model, y_train)
    predictions = [int(value) for value in model.predict(x_valid_model)]
    scores = positive_scores(model, x_valid_model, predictions)
    return score_predictions(y_valid, predictions, scores), predictions


def train_only_standardize(
    x_train: Sequence[Sequence[float]],
    x_valid: Sequence[Sequence[float]],
) -> tuple[list[list[float]], list[list[float]]]:
    scaler = StandardScaler()
    return scaler.fit_transform(x_train).tolist(), scaler.transform(x_valid).tolist()


def apply_log1p_policy(
    x_train: Sequence[Sequence[float]],
    x_valid: Sequence[Sequence[float]],
    feature_columns: Sequence[str],
    transform_policy: dict[str, Any],
    scale_transform_policy_included: bool,
) -> tuple[list[list[float]], list[list[float]]]:
    if not scale_transform_policy_included:
        return [list(row) for row in x_train], [list(row) for row in x_valid]
    transforms = transform_policy.get("recommended_transforms")
    recommended = string_list(transforms.get("log1p_recommended")) if isinstance(transforms, dict) else []
    log1p_indices = {
        index
        for index, feature in enumerate(feature_columns)
        if feature in recommended and is_raw_flow_feature(feature)
    }

    def transform(matrix: Sequence[Sequence[float]]) -> list[list[float]]:
        output: list[list[float]] = []
        for row in matrix:
            output.append(
                [
                    math.log1p(value) if index in log1p_indices and value >= 0 else value
                    for index, value in enumerate(row)
                ]
            )
        return output

    return transform(x_train), transform(x_valid)


def is_raw_flow_feature(feature: str) -> bool:
    name = feature.lower()
    return ("volume" in name or "amount" in name) and not any(token in name for token in ("ratio", "rank", "relative"))


def positive_scores(model: Any, x_valid: Sequence[Sequence[float]], predictions: Sequence[int]) -> list[float]:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_valid)
        return [float(row[1]) for row in probabilities]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_valid)
        return [float(value) for value in scores]
    return [float(value) for value in predictions]


def score_predictions(
    y_valid: Sequence[int],
    predictions: Sequence[int],
    scores: Sequence[float],
) -> dict[str, Any]:
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
        "prediction_distribution": distribution_from_values([int(value) for value in predictions]),
    }


def build_model_result(
    label_policy: str,
    feature_set_name: str,
    feature_set: dict[str, Any],
    model_name: str,
    metrics: dict[str, Any],
    predictions: list[int],
    split: dict[str, Any],
    leakage_check: dict[str, Any],
    dummy_metrics: dict[str, Any],
    dummy_predictions: dict[str, list[int]],
) -> dict[str, Any]:
    collapse = detect_collapse(predictions, dummy_predictions.get("dummy_most_frequent", []))
    gate = evaluate_candidate_gate(metrics, split["valid_prevalence"], dummy_metrics, collapse, leakage_check)
    if model_name.startswith("dummy_"):
        gate["checks"]["non_dummy_model"] = False
        gate["diagnostic_signal_candidate"] = False
        gate["candidate_label"] = "NOT_DIAGNOSTIC_SIGNAL_CANDIDATE"
    else:
        gate["checks"]["non_dummy_model"] = True
    return {
        "label_policy": label_policy,
        "feature_set": feature_set_name,
        "model": model_name,
        "status": "completed",
        "feature_count": len(feature_set["feature_columns"]),
        "train_rows_used": split["train_rows_used"],
        "valid_rows_used": split["valid_rows_used"],
        "train_rows_dropped": split["train_rows_dropped"],
        "valid_rows_dropped": split["valid_rows_dropped"],
        "train_label_distribution": split["train_label_distribution"],
        "valid_label_distribution": split["valid_label_distribution"],
        **metrics,
        "collapse_flag": collapse["collapse_flag"],
        "collapse_check": collapse,
        "compared_to_dummy_most_frequent": compare_to_dummy(metrics, dummy_metrics.get("dummy_most_frequent")),
        "compared_to_dummy_stratified": compare_to_dummy(metrics, dummy_metrics.get("dummy_stratified")),
        "candidate_gate": gate,
        "no_save_artifact_check": {"passed": True, "model_artifact_created": False},
        "leakage_check": leakage_check,
    }


def detect_collapse(predictions: Sequence[int], dummy_most_frequent_predictions: Sequence[int] | None = None) -> dict[str, Any]:
    distribution = distribution_from_values([int(value) for value in predictions])
    valid_prediction_contains_both_classes = distribution["0"] > 0 and distribution["1"] > 0
    matches_dummy = bool(dummy_most_frequent_predictions) and list(predictions) == list(dummy_most_frequent_predictions)
    collapse_flag = (not valid_prediction_contains_both_classes) or matches_dummy
    return {
        "collapse_flag": collapse_flag,
        "prediction_distribution": distribution,
        "valid_prediction_contains_both_classes": valid_prediction_contains_both_classes,
        "matches_dummy_most_frequent_predictions": matches_dummy,
    }


def compare_to_dummy(metrics: dict[str, Any], dummy_metrics: dict[str, Any] | None) -> dict[str, Any]:
    if not dummy_metrics:
        return {"available": False}
    return {
        "available": True,
        "balanced_accuracy_delta": metrics["balanced_accuracy"] - dummy_metrics["balanced_accuracy"],
        "accuracy_delta": metrics["accuracy"] - dummy_metrics["accuracy"],
        "roc_auc_delta": none_safe_delta(metrics.get("roc_auc"), dummy_metrics.get("roc_auc")),
        "pr_auc_delta": none_safe_delta(metrics.get("pr_auc"), dummy_metrics.get("pr_auc")),
    }


def none_safe_delta(value: Any, baseline: Any) -> float | None:
    if value is None or baseline is None:
        return None
    return float(value) - float(baseline)


def evaluate_candidate_gate(
    metrics: dict[str, Any],
    prevalence: float | None,
    dummy_metrics: dict[str, Any],
    collapse: dict[str, Any],
    leakage_check: dict[str, Any],
) -> dict[str, Any]:
    dummy_most_frequent = dummy_metrics.get("dummy_most_frequent", {})
    dummy_ba = dummy_most_frequent.get("balanced_accuracy")
    checks = {
        "no_collapse": collapse["collapse_flag"] is False,
        "balanced_accuracy_beats_dummy_by_0_03": dummy_ba is not None
        and metrics["balanced_accuracy"] >= float(dummy_ba) + 0.03,
        "roc_auc_at_least_0_53": metrics.get("roc_auc") is not None and metrics["roc_auc"] >= 0.53,
        "pr_auc_beats_prevalence_by_0_03": prevalence is not None
        and metrics.get("pr_auc") is not None
        and metrics["pr_auc"] >= prevalence + 0.03,
        "valid_prediction_contains_both_classes": collapse["valid_prediction_contains_both_classes"],
        "no_leakage": leakage_check["passed"],
        "no_artifact": True,
    }
    return {
        "candidate_label": "DIAGNOSTIC_SIGNAL_CANDIDATE" if all(checks.values()) else "NOT_DIAGNOSTIC_SIGNAL_CANDIDATE",
        "diagnostic_signal_candidate": all(checks.values()),
        "checks": checks,
        "formal_model_evidence": False,
    }


def build_decision(
    diagnostic_report: dict[str, Any],
    manifest_check: dict[str, Any],
) -> dict[str, Any]:
    p0_blockers = dedupe([
        *diagnostic_report.get("p0_blockers", []),
        *manifest_check.get("p0_blockers", []),
    ])
    if p0_blockers:
        decision = DECISION_LEAKAGE if any("feature_columns" in blocker for blocker in p0_blockers) else DECISION_RUNTIME
    elif diagnostic_report["diagnostic_candidates"]:
        decision = DECISION_CANDIDATE
    else:
        decision = DECISION_NONE
    return {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "sprint_decision": decision,
        "diagnostic_candidate_count": len(diagnostic_report["diagnostic_candidates"]),
        "diagnostic_candidates": diagnostic_report["diagnostic_candidates"],
        "formal_model_evidence": False,
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "requires_human_review": True,
        "promotion_gate_required": True,
        "p0_blockers": p0_blockers,
        "p1_warnings": [
            "P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED",
            "P1_EXTREME_FEATURE_SCALE_REVIEW_REQUIRED",
            "P1_TRAIN_VALID_FEATURE_SHIFT_REVIEW_REQUIRED",
        ],
        **BOUNDARY_FALSE_FIELDS,
    }


def run_sprint(
    samples_path: Path,
    manifest_path: Path,
    daily_path: Path,
    transform_policy_path: Path,
    transform_smoke_path: Path,
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_samples = resolve_repo_path(samples_path, repo_root)
    resolved_manifest = resolve_repo_path(manifest_path, repo_root)
    resolved_daily = resolve_repo_path(daily_path, repo_root)
    resolved_transform_policy = resolve_repo_path(transform_policy_path, repo_root)
    resolved_transform_smoke = resolve_repo_path(transform_smoke_path, repo_root)
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    for path, label in (
        (resolved_samples, "samples"),
        (resolved_manifest, "manifest"),
        (resolved_daily, "daily"),
        (resolved_transform_policy, "transform policy"),
        (resolved_transform_smoke, "transform-aware smoke report"),
    ):
        if not path.exists():
            raise SignalRecoverySprintError(f"{label} path does not exist: {path}")
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    rows, columns = load_csv_rows(resolved_samples)
    source_manifest = load_json(resolved_manifest)
    transform_policy = load_json(resolved_transform_policy)
    transform_smoke = load_json(resolved_transform_smoke)
    daily_rows, _ = load_csv_rows(resolved_daily)

    base_features = string_list(source_manifest.get("feature_columns"))
    train_dates = string_list(transform_smoke.get("train_anchor_dates"))
    valid_dates = string_list(transform_smoke.get("valid_anchor_dates"))
    if not train_dates or not valid_dates:
        train_dates, valid_dates = derive_anchor_date_split(rows)

    label_generation = generate_label_variants(rows)
    label_summaries = summarize_label_policies(rows, train_dates, valid_dates)
    feature_recovery = recover_past_daily_features(rows, daily_rows)
    recovered_features = feature_recovery["recovered_features"]
    manifest = build_manifest(source_manifest, base_features, recovered_features)
    manifest_path_out = resolved_out_dir / "signal_recovery_sprint1_manifest.json"
    write_json(manifest_path_out, manifest)
    manifest_leakage = check_label_manifest(manifest_path_out).to_summary()
    feature_set_variants = build_feature_set_variants(base_features, recovered_features)
    feature_set_report = build_feature_set_report(
        feature_set_variants,
        manifest["label_columns"],
        manifest["outcome_columns"],
    )
    p0_blockers = dedupe(
        [
            *manifest_leakage.get("p0_blockers", []),
            *[
                blocker
                for payload in feature_set_report.values()
                for blocker in payload["leakage_check"]["p0_blockers"]
            ],
        ]
    )

    sample_columns = dedupe([*columns, *NEW_LABEL_POLICIES, *PAST_DAILY_FEATURES])
    write_csv(resolved_out_dir / "signal_recovery_sprint1_feature_samples.csv", rows, sample_columns)

    label_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "signal_recovery_sprint1_label_policy_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "label_generation": label_generation,
        "label_policies": label_summaries,
        "train_anchor_dates": train_dates,
        "valid_anchor_dates": valid_dates,
        "formal_model_evidence": False,
        "stable_promotion_ready": False,
        "not_trading_advice": True,
    }
    feature_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "signal_recovery_sprint1_feature_recovery_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_daily_path": str(daily_path),
        "input_samples_path": str(samples_path),
        "feature_recovery": feature_recovery,
        "feature_set_variants": feature_set_report,
        "manifest_leakage_check": manifest_leakage,
        "p0_blockers": p0_blockers,
        "formal_model_evidence": False,
        "stable_promotion_ready": False,
        "not_trading_advice": True,
    }
    write_json(resolved_out_dir / "signal_recovery_sprint1_label_policy_report.json", label_report)
    write_json(resolved_out_dir / "signal_recovery_sprint1_feature_recovery_report.json", feature_report)

    diagnostic_report = {
        "suite_scope": "blocked_before_runtime",
        "results": [],
        "diagnostic_candidates": [],
        "p0_blockers": p0_blockers,
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
    }
    if not p0_blockers:
        diagnostic_report = run_diagnostic_suite(
            rows,
            label_summaries,
            feature_set_report,
            transform_policy,
            train_dates,
            valid_dates,
            resolved_out_dir,
        )
    diagnostic_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "signal_recovery_sprint1_diagnostic_smoke_report",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_model_evidence": False,
        "stable_promotion_ready": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "not_trading_advice": True,
        **diagnostic_report,
    }
    decision = build_decision(diagnostic_report, {"p0_blockers": p0_blockers})

    write_json(resolved_out_dir / "signal_recovery_sprint1_diagnostic_smoke_report.json", diagnostic_report)
    write_json(resolved_out_dir / "signal_recovery_sprint1_decision.json", decision)
    write_markdown_reports(resolved_out_dir, label_report, feature_report, diagnostic_report, decision)
    return {
        "label_report": label_report,
        "feature_report": feature_report,
        "diagnostic_report": diagnostic_report,
        "decision": decision,
    }


def write_markdown_reports(
    out_dir: Path,
    label_report: dict[str, Any],
    feature_report: dict[str, Any],
    diagnostic_report: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    md = [
        LAB_DECLARATION,
        "",
        "# Intraday Signal Recovery Sprint 1",
        "",
        "本输出是 Lab-only signal recovery diagnostic，不是正式训练，不保存模型或 scaler，不接 QMT，不生成 OrderIntent，不进入 Stable，不构成交易建议。",
        "",
        f"- sprint_decision: {decision['sprint_decision']}",
        f"- diagnostic_candidate_count: {decision['diagnostic_candidate_count']}",
        f"- label_policy_count: {len(label_report['label_policies'])}",
        f"- recovered_past_daily_features: {', '.join(feature_report['feature_recovery']['recovered_features']) or 'none'}",
        f"- diagnostic_combination_count: {diagnostic_report.get('combination_count', 0)}",
        f"- formal_model_evidence: {str(decision['formal_model_evidence']).lower()}",
        f"- stable_promotion_ready: {str(decision['stable_promotion_ready']).lower()}",
        f"- qmt_ready: {str(decision['qmt_ready']).lower()}",
        f"- order_intent_ready: {str(decision['order_intent_ready']).lower()}",
        "",
        "## Boundary",
        "",
        "- label variants are dry-run labels, not trading signals.",
        "- past daily features use only anchor date D or earlier daily rows.",
        "- no-save model suite is diagnostic only.",
        "- candidates, if any, require human review and promotion gate.",
    ]
    (out_dir / "signal_recovery_sprint1_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "signal_recovery_sprint1_label_policy_report.md").write_text(
        "\n".join([
            LAB_DECLARATION,
            "",
            "# Label Policy Report",
            "",
            "Dry-run label variants only; labels and outcomes are excluded from feature columns.",
            "",
            json.dumps(label_report["label_policies"], ensure_ascii=False, indent=2),
        ])
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "signal_recovery_sprint1_feature_recovery_report.md").write_text(
        "\n".join([
            LAB_DECLARATION,
            "",
            "# Feature Recovery Report",
            "",
            "Past daily feature recovery uses trade_date <= anchor date D only.",
            "",
            json.dumps(feature_report["feature_recovery"], ensure_ascii=False, indent=2),
        ])
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "signal_recovery_sprint1_diagnostic_smoke_report.md").write_text(
        "\n".join([
            LAB_DECLARATION,
            "",
            "# Diagnostic Smoke Report",
            "",
            "No-save diagnostic fitting only; metrics are not formal model evidence.",
            "",
            f"- combination_count: {diagnostic_report.get('combination_count', 0)}",
            f"- candidate_count: {len(diagnostic_report.get('diagnostic_candidates', []))}",
            f"- model_saved: {str(diagnostic_report.get('model_saved', False)).lower()}",
            f"- scaler_saved: {str(diagnostic_report.get('scaler_saved', False)).lower()}",
            f"- checkpoint_saved: {str(diagnostic_report.get('checkpoint_saved', False)).lower()}",
            f"- p0_blockers: {json.dumps(diagnostic_report.get('p0_blockers', []), ensure_ascii=False)}",
        ])
        + "\n",
        encoding="utf-8",
    )


def distribution_from_values(values: Sequence[int]) -> dict[str, int]:
    return {"0": sum(1 for value in values if value == 0), "1": sum(1 for value in values if value == 1)}


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
    parser.add_argument("--daily", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--transform-policy", type=Path, default=DEFAULT_TRANSFORM_POLICY)
    parser.add_argument("--transform-smoke", type=Path, default=DEFAULT_TRANSFORM_SMOKE)
    parser.add_argument("--out-dir", type=Path, default=ALLOWED_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = run_sprint(
            args.samples,
            args.manifest,
            args.daily,
            args.transform_policy,
            args.transform_smoke,
            args.out_dir,
        )
    except Exception as exc:  # noqa: BLE001 - CLI must return auditable blocker.
        print(
            json.dumps(
                {
                    "lab_declaration": LAB_DECLARATION,
                    "status": "failed",
                    "sprint_decision": DECISION_RUNTIME,
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
                "sprint_decision": payload["decision"]["sprint_decision"],
                "diagnostic_candidate_count": payload["decision"]["diagnostic_candidate_count"],
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
