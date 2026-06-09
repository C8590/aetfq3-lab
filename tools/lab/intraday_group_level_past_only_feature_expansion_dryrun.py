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
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_group_level_past_only_feature_expansion_dryrun"
SAMPLE_SUBTYPE = "intraday_group_level_past_only_feature_expansion_dryrun"
GROUP_LABEL_POLICY = "anchor_close_last_bar"
TARGET_COLUMN = "three_day_positive_label"
ALLOWED_OUTPUT_DIR = Path(
    ".local_research_outputs/aetfq3_lab/intraday_group_level_past_only_feature_expansion_dryrun"
)
DEFAULT_BAR_SAMPLES = Path(
    ".local_research_outputs/aetfq3_lab/intraday_larger_eligible_anchor_readiness/"
    "larger_eligible_anchor_label_samples.csv"
)
DEFAULT_GROUP_SAMPLES = Path(
    ".local_research_outputs/aetfq3_lab/intraday_group_level_sample_dryrun/"
    "intraday_group_level_samples.csv"
)
DEFAULT_INCONSISTENCY_REPORT = Path(
    ".local_research_outputs/aetfq3_lab/intraday_group_label_inconsistency_diagnostic/"
    "intraday_group_label_inconsistency_report.json"
)
DEFAULT_DESIGN = Path("docs/research/aetfq3_intraday_group_level_past_only_feature_expansion_design.json")
DEFAULT_LEAKAGE_CHECKLIST = Path("docs/research/aetfq3_intraday_group_level_feature_leakage_checklist.json")

CORE_FEATURES = [
    "open_first",
    "high_max",
    "low_min",
    "close_last",
    "volume_sum",
    "amount_sum",
    "vwap_day",
    "day_return",
    "high_low_range",
    "close_to_vwap",
    "intraday_return_mean",
    "intraday_return_std",
    "distance_to_vwap_mean",
    "distance_to_vwap_last",
    "volume_first_half_sum",
    "volume_second_half_sum",
    "amount_first_half_sum",
    "amount_second_half_sum",
]
OPTIONAL_ANCHOR_FEATURES = [
    "intraday_return_skew",
    "intraday_return_min",
    "intraday_return_max",
    "volume_second_half_ratio",
    "amount_second_half_ratio",
    "volume_spike_ratio",
    "amount_spike_ratio",
    "morning_return",
    "afternoon_return",
    "last_hour_return",
    "close_vs_morning_high",
    "close_vs_intraday_high",
    "close_vs_intraday_low",
    "vwap_slope_proxy",
    "price_above_vwap_bar_ratio",
]
CROSS_SECTIONAL_FEATURES = [
    "rank_day_return",
    "rank_volume_sum",
    "rank_amount_sum",
    "rank_close_to_vwap",
    "relative_return_to_universe_mean",
    "relative_volume_to_universe_mean",
]
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
GENERATED_OUTCOMES = ["future_return_1d", "future_return_3d", "max_drawdown_3d"]
GENERATED_LABELS = [TARGET_COLUMN]
BLOCKED_LABELS = ["buy_now_label", "wait_pullback_label", "cancel_buy_label"]
BOUNDARY_FALSE_FIELDS = [
    "supervised_training_allowed",
    "training_allowed",
    "stable_effect_allowed",
    "contains_order_intent",
    "contains_live_order",
    "contains_secret",
]

PASSED = "GROUP_LEVEL_PAST_ONLY_FEATURE_EXPANSION_DRY_RUN_PASSED_READINESS_REVIEW_REQUIRED"
PASSED_WITH_WARNINGS = "GROUP_LEVEL_PAST_ONLY_FEATURE_EXPANSION_DRY_RUN_PASSED_WITH_FEATURE_QUALITY_WARNINGS"
BLOCKED_TOO_FEW = "BLOCKED_FEATURE_GENERATION_TOO_FEW_FEATURES"
BLOCKED_MANIFEST = "BLOCKED_MANIFEST_LEAKAGE_P0"
BLOCKED_BOUNDARY = "BLOCKED_BOUNDARY_FLAG"
BLOCKED_READINESS = "BLOCKED_GROUP_LEVEL_READINESS"


class FeatureExpansionDryRunError(RuntimeError):
    pass


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def resolve_output_dir(
    out_dir: Path,
    repo_root: Path = REPO_ROOT,
    *,
    enforce_allowed_output_dir: bool = True,
) -> Path:
    repo_root = repo_root.resolve()
    resolved = resolve_repo_path(out_dir, repo_root).resolve()
    if enforce_allowed_output_dir:
        allowed = (repo_root / ALLOWED_OUTPUT_DIR).resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError as exc:
            raise FeatureExpansionDryRunError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FeatureExpansionDryRunError(f"JSON cannot be read: {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise FeatureExpansionDryRunError(f"JSON parse failed: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise FeatureExpansionDryRunError(f"JSON root must be object: {path}")
    return payload


def load_csv_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            columns = list(reader.fieldnames or [])
    except OSError as exc:
        raise FeatureExpansionDryRunError(f"CSV cannot be read: {path}: {exc}") from exc
    if not columns:
        raise FeatureExpansionDryRunError(f"CSV has no header: {path}")
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
    if text == "":
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    return number


def safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def safe_ratio_minus_one(numerator: float | None, denominator: float | None) -> float | None:
    value = safe_div(numerator, denominator)
    return None if value is None else value - 1.0


def clean_number(value: float | None) -> float | str:
    if value is None or not math.isfinite(value):
        return ""
    return value


def numeric_values(rows: Sequence[dict[str, str]], field: str) -> list[float]:
    return [value for row in rows if (value := to_float(row.get(field))) is not None]


def first_numeric(rows: Sequence[dict[str, str]], field: str) -> float | None:
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            return value
    return None


def last_numeric(rows: Sequence[dict[str, str]], field: str) -> float | None:
    for row in reversed(rows):
        value = to_float(row.get(field))
        if value is not None:
            return value
    return None


def sum_numeric(rows: Sequence[dict[str, str]], field: str) -> float | None:
    values = numeric_values(rows, field)
    if not values:
        return None
    return sum(values)


def skew(values: Sequence[float]) -> float | None:
    if len(values) < 3:
        return None
    avg = mean(values)
    std = pstdev(values)
    if std == 0:
        return None
    return sum(((value - avg) / std) ** 3 for value in values) / len(values)


def pct_returns_from_close(rows: Sequence[dict[str, str]]) -> list[float]:
    closes = numeric_values(rows, "close")
    returns: list[float] = []
    for previous, current in zip(closes, closes[1:]):
        value = safe_ratio_minus_one(current, previous)
        if value is not None:
            returns.append(value)
    return returns


def row_sort_key(row: dict[str, str]) -> tuple[str, int]:
    datetime_text = str(row.get("datetime", "")).strip()
    bar_index = int(to_float(row.get("bar_index")) or 0)
    return datetime_text, bar_index


def build_groups(rows: list[dict[str, str]]) -> dict[tuple[str, str], list[dict[str, str]]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        trade_date = str(row.get("trade_date", "")).strip()
        etf_code = str(row.get("etf_code", "")).strip()
        if trade_date and etf_code:
            groups[(trade_date, etf_code)].append(row)
    return {key: sorted(group_rows, key=row_sort_key) for key, group_rows in groups.items()}


def calculate_anchor_features(group_rows: list[dict[str, str]]) -> dict[str, Any]:
    first_half, second_half = split_halves(group_rows)
    morning = first_half
    afternoon = second_half
    last_hour = group_rows[-min(12, len(group_rows)) :] if group_rows else []

    open_first = first_numeric(group_rows, "open")
    close_last = last_numeric(group_rows, "close")
    highs = numeric_values(group_rows, "high")
    lows = numeric_values(group_rows, "low")
    volumes = numeric_values(group_rows, "volume")
    amounts = numeric_values(group_rows, "amount")
    high_max = max(highs) if highs else None
    low_min = min(lows) if lows else None
    volume_sum = sum(volumes) if volumes else None
    amount_sum = sum(amounts) if amounts else None
    vwap_day = safe_div(amount_sum, volume_sum)
    intraday_returns = numeric_values(group_rows, "intraday_return") or pct_returns_from_close(group_rows)
    distances = numeric_values(group_rows, "distance_to_vwap")
    if not distances:
        distances = [
            value
            for row in group_rows
            if (value := safe_ratio_minus_one(to_float(row.get("close")), to_float(row.get("vwap")))) is not None
        ]
    first_half_volume = sum_numeric(first_half, "volume")
    second_half_volume = sum_numeric(second_half, "volume")
    first_half_amount = sum_numeric(first_half, "amount")
    second_half_amount = sum_numeric(second_half, "amount")
    first_half_vwap = safe_div(first_half_amount, first_half_volume)
    second_half_vwap = safe_div(second_half_amount, second_half_volume)
    morning_highs = numeric_values(morning, "high")

    features = {
        "open_first": clean_number(open_first),
        "high_max": clean_number(high_max),
        "low_min": clean_number(low_min),
        "close_last": clean_number(close_last),
        "volume_sum": clean_number(volume_sum),
        "amount_sum": clean_number(amount_sum),
        "vwap_day": clean_number(vwap_day),
        "day_return": clean_number(safe_ratio_minus_one(close_last, open_first)),
        "high_low_range": clean_number(safe_ratio_minus_one(high_max, low_min)),
        "close_to_vwap": clean_number(safe_ratio_minus_one(close_last, vwap_day)),
        "intraday_return_mean": clean_number(mean(intraday_returns) if intraday_returns else None),
        "intraday_return_std": clean_number(pstdev(intraday_returns) if len(intraday_returns) > 1 else None),
        "intraday_return_skew": clean_number(skew(intraday_returns)),
        "intraday_return_min": clean_number(min(intraday_returns) if intraday_returns else None),
        "intraday_return_max": clean_number(max(intraday_returns) if intraday_returns else None),
        "distance_to_vwap_mean": clean_number(mean(distances) if distances else None),
        "distance_to_vwap_last": clean_number(distances[-1] if distances else None),
        "volume_first_half_sum": clean_number(first_half_volume),
        "volume_second_half_sum": clean_number(second_half_volume),
        "amount_first_half_sum": clean_number(first_half_amount),
        "amount_second_half_sum": clean_number(second_half_amount),
        "volume_second_half_ratio": clean_number(safe_div(second_half_volume, volume_sum)),
        "amount_second_half_ratio": clean_number(safe_div(second_half_amount, amount_sum)),
        "volume_spike_ratio": clean_number(safe_div(max(volumes) if volumes else None, mean(volumes) if volumes else None)),
        "amount_spike_ratio": clean_number(safe_div(max(amounts) if amounts else None, mean(amounts) if amounts else None)),
        "morning_return": clean_number(safe_ratio_minus_one(last_numeric(morning, "close"), open_first)),
        "afternoon_return": clean_number(
            safe_ratio_minus_one(close_last, first_numeric(afternoon, "close"))
        ),
        "last_hour_return": clean_number(safe_ratio_minus_one(close_last, first_numeric(last_hour, "close"))),
        "close_vs_morning_high": clean_number(safe_ratio_minus_one(close_last, max(morning_highs) if morning_highs else None)),
        "close_vs_intraday_high": clean_number(safe_ratio_minus_one(close_last, high_max)),
        "close_vs_intraday_low": clean_number(safe_ratio_minus_one(close_last, low_min)),
        "vwap_slope_proxy": clean_number(safe_ratio_minus_one(second_half_vwap, first_half_vwap)),
        "price_above_vwap_bar_ratio": clean_number(price_above_vwap_ratio(group_rows)),
    }
    return features


def split_halves(rows: Sequence[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    midpoint = len(rows) // 2
    return list(rows[:midpoint]), list(rows[midpoint:])


def price_above_vwap_ratio(rows: Sequence[dict[str, str]]) -> float | None:
    total = 0
    above = 0
    running_amount = 0.0
    running_volume = 0.0
    for row in rows:
        close = to_float(row.get("close"))
        amount = to_float(row.get("amount"))
        volume = to_float(row.get("volume"))
        bar_vwap = to_float(row.get("vwap"))
        if amount is not None and volume is not None:
            running_amount += amount
            running_volume += volume
        compare_vwap = bar_vwap if bar_vwap is not None else safe_div(running_amount, running_volume)
        if close is None or compare_vwap is None:
            continue
        total += 1
        above += int(close > compare_vwap)
    return above / total if total else None


def append_cross_sectional_features(rows: list[dict[str, Any]]) -> None:
    rows_by_date: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_date[str(row.get("trade_date", ""))].append(row)
    for date_rows in rows_by_date.values():
        add_rank(date_rows, "day_return", "rank_day_return")
        add_rank(date_rows, "volume_sum", "rank_volume_sum")
        add_rank(date_rows, "amount_sum", "rank_amount_sum")
        add_rank(date_rows, "close_to_vwap", "rank_close_to_vwap")
        add_relative_mean(date_rows, "day_return", "relative_return_to_universe_mean", subtract=True)
        add_relative_mean(date_rows, "volume_sum", "relative_volume_to_universe_mean", subtract=False)


def add_rank(rows: list[dict[str, Any]], source: str, target: str) -> None:
    values = [(idx, to_float(row.get(source))) for idx, row in enumerate(rows)]
    valid = sorted(((idx, value) for idx, value in values if value is not None), key=lambda item: item[1], reverse=True)
    for rank, (idx, _value) in enumerate(valid, start=1):
        rows[idx][target] = rank
    for idx, value in values:
        if value is None:
            rows[idx][target] = ""


def add_relative_mean(rows: list[dict[str, Any]], source: str, target: str, *, subtract: bool) -> None:
    values = [to_float(row.get(source)) for row in rows]
    valid = [value for value in values if value is not None]
    avg = mean(valid) if valid else None
    for row, value in zip(rows, values):
        if value is None or avg is None:
            row[target] = ""
        elif subtract:
            row[target] = value - avg
        else:
            row[target] = clean_number(safe_ratio_minus_one(value, avg))


def generate_feature_rows(bar_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    grouped = build_groups(bar_rows)
    output_rows: list[dict[str, Any]] = []
    for (trade_date, etf_code), group_rows in sorted(grouped.items()):
        first = group_rows[0]
        last = group_rows[-1]
        feature_values = calculate_anchor_features(group_rows)
        row: dict[str, Any] = {
            "trade_date": trade_date,
            "anchor_date": "",
            "etf_code": etf_code,
            "etf_name": first.get("etf_name", ""),
            "bar_count": len(group_rows),
            "last_bar_datetime": last.get("datetime", ""),
            "group_label_policy": GROUP_LABEL_POLICY,
        }
        row.update(feature_values)
        for outcome in GENERATED_OUTCOMES:
            row[outcome] = last.get(outcome, "")
        for label in BLOCKED_LABELS:
            row[label] = ""
        row[TARGET_COLUMN] = last.get(TARGET_COLUMN, "")
        row["label_status"] = last.get("label_status", "")
        row["label_horizon"] = last.get("label_horizon", "")
        output_rows.append(row)
    append_cross_sectional_features(output_rows)
    feature_columns = determine_generated_features(output_rows)
    skipped_features = [
        feature
        for feature in PAST_DAILY_FEATURES
        if feature not in feature_columns
    ]
    return output_rows, feature_columns, skipped_features


def determine_generated_features(rows: list[dict[str, Any]]) -> list[str]:
    candidates = CORE_FEATURES + OPTIONAL_ANCHOR_FEATURES + CROSS_SECTIONAL_FEATURES
    generated: list[str] = []
    for feature in candidates:
        if any(str(row.get(feature, "")).strip() != "" for row in rows):
            generated.append(feature)
    return generated


def build_manifest(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    skipped_features: list[str],
    design: dict[str, Any],
) -> dict[str, Any]:
    anchors = sorted({str(row["trade_date"]) for row in rows})
    manifest = {
        "manifest_version": "intraday_group_level_past_only_feature_expansion_dryrun_v1",
        "sample_type": "intraday_5m",
        "sample_subtype": SAMPLE_SUBTYPE,
        "group_level_sample": True,
        "group_key": ["trade_date", "etf_code"],
        "group_label_policy": GROUP_LABEL_POLICY,
        "feature_time_scope": "anchor_day_only_or_prior",
        "label_time_scope": "after_anchor_day",
        "intraday_live_decision_ready": False,
        "eligible_anchor_subset_only": True,
        "eligible_anchor_dates": anchors,
        "feature_columns": feature_columns,
        "generated_feature_count": len(feature_columns),
        "skipped_features": skipped_features,
        "generated_outcomes": GENERATED_OUTCOMES,
        "generated_labels": GENERATED_LABELS,
        "blocked_labels": BLOCKED_LABELS,
        "label_generated": True,
        "label_source_kind": "public_future_window_anchor_close_last_bar",
        "label_horizon": {
            "unit": "trading_day",
            "required_horizons": ["T+1", "T+3"],
            "source": "existing bar-level dry-run label_horizon copied from last anchor-day bar",
        },
        "label_generation_method": "anchor_close_last_bar_group_level_past_only_feature_expansion_dryrun_v1",
        "label_columns": BLOCKED_LABELS + GENERATED_LABELS,
        "outcome_columns": GENERATED_OUTCOMES,
        "label_status_column": "label_status",
        "insufficient_future_window_policy": "set label null when future window is unavailable",
        "feature_label_overlap_check": True,
        "label_generation_authorized": True,
        "design_source": "docs/research/aetfq3_intraday_group_level_past_only_feature_expansion_design.json",
        "design_next_allowed_action": design.get("next_allowed_action"),
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
        "not_trading_advice": True,
    }
    return manifest


def feature_quality_precheck(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    *,
    min_generated_features: int,
) -> dict[str, Any]:
    missing_count: dict[str, int] = {}
    inf_count: dict[str, int] = {}
    zero_variance: list[str] = []
    per_feature: dict[str, Any] = {}
    stds: list[float] = []
    for feature in feature_columns:
        values: list[float] = []
        missing = 0
        inf = 0
        for row in rows:
            raw = row.get(feature, "")
            if str(raw).strip() == "":
                missing += 1
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                missing += 1
                continue
            if not math.isfinite(value):
                inf += 1
                continue
            values.append(value)
        missing_count[feature] = missing
        inf_count[feature] = inf
        std = pstdev(values) if len(values) > 1 else 0.0
        if std == 0.0:
            zero_variance.append(feature)
        elif std > 0:
            stds.append(std)
        per_feature[feature] = {
            "count": len(values),
            "missing_count": missing,
            "inf_count": inf,
            "mean": mean(values) if values else None,
            "std": std,
            "min": min(values) if values else None,
            "max": max(values) if values else None,
        }
    feature_scale_ratio = max(stds) / min(stds) if stds and min(stds) > 0 else None
    extreme_scale_features = [
        feature
        for feature, stats in per_feature.items()
        if (stats["std"] or 0) > 0 and feature_scale_ratio and (max(stds) / stats["std"]) > 1000000
    ]
    split_diag = train_valid_standardized_mean_difference(rows, feature_columns)
    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    if len(feature_columns) < min_generated_features:
        p0_blockers.append(f"generated_feature_count must be >= {min_generated_features}")
    if zero_variance:
        p1_warnings.append("P1_ZERO_VARIANCE_FEATURES_REVIEW_REQUIRED")
    if extreme_scale_features:
        p1_warnings.append("P1_EXTREME_FEATURE_SCALE_REVIEW_REQUIRED")
    return {
        "passed": not p0_blockers,
        "generated_feature_count": len(feature_columns),
        "skipped_feature_count": 0,
        "missing_count": missing_count,
        "inf_count": inf_count,
        "zero_variance_features": zero_variance,
        "extreme_scale_features": extreme_scale_features,
        "feature_scale_ratio": feature_scale_ratio,
        "train_valid_standardized_mean_difference": split_diag,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }


def train_valid_standardized_mean_difference(
    rows: list[dict[str, Any]],
    feature_columns: list[str],
) -> dict[str, Any]:
    anchors = sorted({str(row.get("trade_date", "")) for row in rows if str(row.get("trade_date", ""))})
    if len(anchors) < 2:
        return {"available": False, "reason": "insufficient anchors"}
    split_index = max(1, min(len(anchors) - 1, int(len(anchors) * 0.7)))
    train_anchors = set(anchors[:split_index])
    valid_anchors = set(anchors[split_index:])
    train_rows = [row for row in rows if str(row.get("trade_date", "")) in train_anchors]
    valid_rows = [row for row in rows if str(row.get("trade_date", "")) in valid_anchors]
    per_feature: dict[str, Any] = {}
    for feature in feature_columns:
        train_values = [value for row in train_rows if (value := to_float(row.get(feature))) is not None]
        valid_values = [value for row in valid_rows if (value := to_float(row.get(feature))) is not None]
        pooled_std = pstdev(train_values + valid_values) if len(train_values + valid_values) > 1 else 0.0
        per_feature[feature] = {
            "train_mean": mean(train_values) if train_values else None,
            "valid_mean": mean(valid_values) if valid_values else None,
            "standardized_mean_difference": (
                (mean(train_values) - mean(valid_values)) / pooled_std
                if train_values and valid_values and pooled_std > 0
                else None
            ),
        }
    return {
        "available": True,
        "split_policy": "anchor_date_70_30",
        "train_anchor_count": len(train_anchors),
        "valid_anchor_count": len(valid_anchors),
        "per_feature": per_feature,
    }


def class_balance_precheck(
    rows: list[dict[str, Any]],
    *,
    min_group_count: int,
    min_anchor_count: int,
    min_etf_count: int,
    min_class_count: int,
) -> dict[str, Any]:
    labels = [normalize_label(row.get(TARGET_COLUMN, "")) for row in rows]
    label_null_count = sum(label is None for label in labels)
    label_0_count = sum(label == 0 for label in labels)
    label_1_count = sum(label == 1 for label in labels)
    class_count = int(label_0_count > 0) + int(label_1_count > 0)
    min_observed = min((count for count in (label_0_count, label_1_count) if count > 0), default=0)
    anchors = sorted({str(row.get("trade_date", "")) for row in rows if str(row.get("trade_date", ""))})
    etfs = sorted({str(row.get("etf_code", "")) for row in rows if str(row.get("etf_code", ""))})
    p0_blockers: list[str] = []
    if len(rows) < min_group_count:
        p0_blockers.append(f"group_count must be >= {min_group_count}")
    if len(anchors) < min_anchor_count:
        p0_blockers.append(f"anchor_count must be >= {min_anchor_count}")
    if len(etfs) < min_etf_count:
        p0_blockers.append(f"etf_count must be >= {min_etf_count}")
    if label_null_count:
        p0_blockers.append("label_null_count must be 0")
    if class_count < 2:
        p0_blockers.append("label class_count must be 2")
    if min_observed < min_class_count:
        p0_blockers.append(f"min_class_count must be >= {min_class_count}")
    return {
        "passed": not p0_blockers,
        "group_count": len(rows),
        "anchor_count": len(anchors),
        "etf_count": len(etfs),
        "label_null_count": label_null_count,
        "label_0_count": label_0_count,
        "label_1_count": label_1_count,
        "positive_rate": label_1_count / (label_0_count + label_1_count) if (label_0_count + label_1_count) else None,
        "class_count": class_count,
        "min_class_count": min_observed,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def normalize_label(value: Any) -> int | None:
    text = str(value).strip()
    if text in {"0", "0.0"}:
        return 0
    if text in {"1", "1.0"}:
        return 1
    return None


def run_readiness_precheck(
    rows: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_check: Any,
    feature_quality: dict[str, Any],
    class_balance: dict[str, Any],
    inherited_p1_warnings: Sequence[str],
) -> dict[str, Any]:
    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    p0_blockers.extend(feature_quality["p0_blockers"])
    p1_warnings.extend(feature_quality["p1_warnings"])
    p0_blockers.extend(class_balance["p0_blockers"])
    p1_warnings.extend(warning for warning in inherited_p1_warnings if warning not in p1_warnings)
    for field in BOUNDARY_FALSE_FIELDS:
        if manifest.get(field) is not False:
            p0_blockers.append(f"{field} must be false")
    if manifest.get("group_level_sample") is not True:
        p0_blockers.append("group_level_sample must be true")
    if manifest.get("group_key") != ["trade_date", "etf_code"]:
        p0_blockers.append("group_key must be ['trade_date','etf_code']")
    if manifest.get("group_label_policy") != GROUP_LABEL_POLICY:
        p0_blockers.append(f"group_label_policy must be {GROUP_LABEL_POLICY}")
    if manifest.get("intraday_live_decision_ready") is not False:
        p0_blockers.append("intraday_live_decision_ready must be false")
    split_check = split_feasibility(rows)
    p0_blockers.extend(split_check["p0_blockers"])

    if any(message.startswith("generated_feature_count") for message in feature_quality["p0_blockers"]):
        decision = BLOCKED_TOO_FEW
    elif manifest_check.p0_blockers:
        decision = BLOCKED_MANIFEST
    elif any("must be false" in blocker for blocker in p0_blockers):
        decision = BLOCKED_BOUNDARY
    elif class_balance["p0_blockers"] or split_check["p0_blockers"]:
        decision = BLOCKED_READINESS
    elif p1_warnings:
        decision = PASSED_WITH_WARNINGS
    else:
        decision = PASSED

    return {
        "status": "blocked" if decision.startswith("BLOCKED_") else "passed",
        "readiness_decision": decision,
        "target": TARGET_COLUMN,
        "group_level_sample": True,
        "group_key": ["trade_date", "etf_code"],
        "group_label_policy": GROUP_LABEL_POLICY,
        "feature_time_scope": "anchor_day_only_or_prior",
        "label_time_scope": "after_anchor_day",
        "intraday_live_decision_ready": False,
        "selected_split_policy": split_check["selected_split_policy"],
        "train_group_count": split_check["train_group_count"],
        "valid_group_count": split_check["valid_group_count"],
        "split_feasible": split_check["split_feasible"],
        "training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "not_trading_advice": True,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
        "split_check": split_check,
    }


def split_feasibility(rows: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = sorted({str(row.get("trade_date", "")) for row in rows if str(row.get("trade_date", ""))})
    if len(anchors) < 2:
        return {
            "selected_split_policy": None,
            "train_group_count": 0,
            "valid_group_count": 0,
            "split_feasible": False,
            "p0_blockers": ["time-based split requires at least two anchors"],
        }
    for policy_name, ratio in (("anchor_date_70_30", 0.7), ("anchor_date_60_40", 0.6)):
        split_index = max(1, min(len(anchors) - 1, int(len(anchors) * ratio)))
        train_anchors = set(anchors[:split_index])
        valid_anchors = set(anchors[split_index:])
        train_rows = [row for row in rows if str(row.get("trade_date", "")) in train_anchors]
        valid_rows = [row for row in rows if str(row.get("trade_date", "")) in valid_anchors]
        train_counts = count_labels(train_rows)
        valid_counts = count_labels(valid_rows)
        feasible = train_counts[0] > 0 and train_counts[1] > 0 and valid_counts[0] > 0 and valid_counts[1] > 0
        if feasible:
            return {
                "selected_split_policy": policy_name,
                "train_group_count": len(train_rows),
                "valid_group_count": len(valid_rows),
                "train_label_0_count": train_counts[0],
                "train_label_1_count": train_counts[1],
                "valid_label_0_count": valid_counts[0],
                "valid_label_1_count": valid_counts[1],
                "split_feasible": True,
                "p0_blockers": [],
            }
    return {
        "selected_split_policy": "anchor_date_60_40",
        "train_group_count": 0,
        "valid_group_count": 0,
        "split_feasible": False,
        "p0_blockers": ["time-based split train/valid must both contain class 0 and class 1"],
    }


def count_labels(rows: list[dict[str, Any]]) -> dict[int, int]:
    counts = {0: 0, 1: 0}
    for row in rows:
        label = normalize_label(row.get(TARGET_COLUMN))
        if label in counts:
            counts[label] += 1
    return counts


def inherited_group_label_p1_warnings(inconsistency_report: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    for warning in inconsistency_report.get("p1_warnings", []):
        if isinstance(warning, str) and warning and warning not in warnings:
            warnings.append(warning)
    rate = to_float(inconsistency_report.get("inconsistent_group_rate"))
    if rate is not None and rate > 0.10 and "P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED" not in warnings:
        warnings.append("P1_GROUP_LABEL_INCONSISTENCY_REVIEW_REQUIRED")
    return warnings


def build_report(
    samples_path: Path,
    manifest_path: Path,
    design_path: Path,
    leakage_checklist_path: Path,
    rows: list[dict[str, Any]],
    feature_columns: list[str],
    skipped_features: list[str],
    manifest_check: Any,
    feature_quality: dict[str, Any],
    class_balance: dict[str, Any],
    readiness: dict[str, Any],
    inherited_p1_warnings: Sequence[str],
) -> dict[str, Any]:
    return {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": readiness["status"],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_bar_samples_path": str(samples_path),
        "manifest_path": str(manifest_path),
        "design_path": str(design_path),
        "leakage_checklist_path": str(leakage_checklist_path),
        "group_level_sample": True,
        "group_key": ["trade_date", "etf_code"],
        "group_label_policy": GROUP_LABEL_POLICY,
        "feature_time_scope": "anchor_day_only_or_prior",
        "label_time_scope": "after_anchor_day",
        "intraday_live_decision_ready": False,
        "feature_columns": feature_columns,
        "generated_feature_count": len(feature_columns),
        "skipped_features": skipped_features,
        "skipped_feature_count": len(skipped_features),
        "generated_outcomes": GENERATED_OUTCOMES,
        "generated_labels": GENERATED_LABELS,
        "blocked_labels": BLOCKED_LABELS,
        "group_count": len(rows),
        "manifest_leakage_check": manifest_check.to_summary(),
        "feature_quality_precheck": feature_quality,
        "class_balance_precheck": class_balance,
        "inherited_p1_warnings": list(inherited_p1_warnings),
        "supervised_smoke_readiness_precheck": readiness,
        "readiness_decision": readiness["readiness_decision"],
        "training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "model_training_performed": False,
        "no_save_supervised_smoke_run": False,
        "hyperparameter_tuning": False,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "not_trading_advice": True,
        "p0_blockers": readiness["p0_blockers"],
        "p1_warnings": readiness["p1_warnings"],
    }


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday Group-Level Past-Only Feature Expansion Dry Run",
        "",
        "This Lab-only dry-run materializes past-only group-level features and runs leakage, feature quality, class-balance, and readiness checks. It does not run no-save smoke, train a model, connect QMT, generate OrderIntent, or enter Stable.",
        "",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- group_count: {report['group_count']}",
        f"- generated_feature_count: {report['generated_feature_count']}",
        f"- skipped_feature_count: {report['skipped_feature_count']}",
        f"- class_count: {report['class_balance_precheck']['class_count']}",
        f"- min_class_count: {report['class_balance_precheck']['min_class_count']}",
        f"- manifest_leakage_status: {report['manifest_leakage_check']['status']}",
        f"- training_allowed: {str(report['training_allowed']).lower()}",
        f"- stable_allowed: {str(report['stable_allowed']).lower()}",
        f"- qmt_allowed: {str(report['qmt_allowed']).lower()}",
        f"- order_intent_allowed: {str(report['order_intent_allowed']).lower()}",
        f"- metrics_are_effectiveness_evidence: {str(report['metrics_are_effectiveness_evidence']).lower()}",
        "",
        "## Skipped Features",
        "",
    ]
    lines.extend(f"- {feature}" for feature in report["skipped_features"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dryrun(
    bar_samples: Path = DEFAULT_BAR_SAMPLES,
    group_samples: Path = DEFAULT_GROUP_SAMPLES,
    group_inconsistency_report: Path | None = None,
    design_path: Path = DEFAULT_DESIGN,
    leakage_checklist_path: Path = DEFAULT_LEAKAGE_CHECKLIST,
    out_dir: Path = ALLOWED_OUTPUT_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    enforce_allowed_output_dir: bool = True,
    min_generated_features: int = 18,
    min_group_count: int = 200,
    min_anchor_count: int = 20,
    min_etf_count: int = 3,
    min_class_count: int = 1,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_bar_samples = resolve_repo_path(bar_samples, repo_root)
    resolved_group_samples = resolve_repo_path(group_samples, repo_root)
    resolved_group_inconsistency_report = (
        resolve_repo_path(group_inconsistency_report, repo_root) if group_inconsistency_report is not None else None
    )
    resolved_design = resolve_repo_path(design_path, repo_root)
    resolved_leakage_checklist = resolve_repo_path(leakage_checklist_path, repo_root)
    for path, label in (
        (resolved_bar_samples, "bar-samples"),
        (resolved_group_samples, "group-samples"),
        (resolved_design, "design"),
        (resolved_leakage_checklist, "leakage-checklist"),
    ):
        if not path.exists():
            raise FeatureExpansionDryRunError(f"{label} path does not exist: {path}")

    resolved_out_dir = resolve_output_dir(
        out_dir,
        repo_root,
        enforce_allowed_output_dir=enforce_allowed_output_dir,
    )
    resolved_out_dir.mkdir(parents=True, exist_ok=True)
    design = load_json(resolved_design)
    _leakage_checklist = load_json(resolved_leakage_checklist)
    inconsistency_report = (
        load_json(resolved_group_inconsistency_report)
        if resolved_group_inconsistency_report is not None and resolved_group_inconsistency_report.exists()
        else {}
    )
    bar_rows, _bar_columns = load_csv_rows(resolved_bar_samples)
    _group_rows, _group_columns = load_csv_rows(resolved_group_samples)

    rows, feature_columns, skipped_features = generate_feature_rows(bar_rows)
    feature_quality = feature_quality_precheck(rows, feature_columns, min_generated_features=min_generated_features)
    feature_quality["skipped_feature_count"] = len(skipped_features)
    class_balance = class_balance_precheck(
        rows,
        min_group_count=min_group_count,
        min_anchor_count=min_anchor_count,
        min_etf_count=min_etf_count,
        min_class_count=min_class_count,
    )
    manifest = build_manifest(rows, feature_columns, skipped_features, design)

    sample_path = resolved_out_dir / "intraday_group_level_past_only_feature_samples.csv"
    manifest_path = resolved_out_dir / "intraday_group_level_past_only_feature_manifest.json"
    report_path = resolved_out_dir / "intraday_group_level_past_only_feature_report.json"
    report_md_path = resolved_out_dir / "intraday_group_level_past_only_feature_report.md"
    quality_path = resolved_out_dir / "feature_quality_precheck.json"
    class_balance_path = resolved_out_dir / "class_balance_precheck.json"
    readiness_report_path = resolved_out_dir / "supervised_smoke_readiness_report.json"
    readiness_decision_path = resolved_out_dir / "readiness_decision.json"

    base_columns = [
        "trade_date",
        "anchor_date",
        "etf_code",
        "etf_name",
        "bar_count",
        "last_bar_datetime",
        "group_label_policy",
    ]
    label_columns = GENERATED_OUTCOMES + BLOCKED_LABELS + GENERATED_LABELS + ["label_status", "label_horizon"]
    write_csv(sample_path, rows, base_columns + feature_columns + label_columns)
    write_json(manifest_path, manifest)

    manifest_check = check_label_manifest(manifest_path)
    inherited_p1_warnings = inherited_group_label_p1_warnings(inconsistency_report)
    readiness = run_readiness_precheck(
        rows,
        manifest,
        manifest_check,
        feature_quality,
        class_balance,
        inherited_p1_warnings,
    )
    report = build_report(
        sample_path,
        manifest_path,
        resolved_design,
        resolved_leakage_checklist,
        rows,
        feature_columns,
        skipped_features,
        manifest_check,
        feature_quality,
        class_balance,
        readiness,
        inherited_p1_warnings,
    )
    write_json(quality_path, feature_quality)
    write_json(class_balance_path, class_balance)
    write_json(readiness_report_path, readiness)
    write_json(readiness_decision_path, {
        "readiness_decision": readiness["readiness_decision"],
        "status": readiness["status"],
        "training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "p0_blockers": readiness["p0_blockers"],
        "p1_warnings": readiness["p1_warnings"],
    })
    write_json(report_path, report)
    write_markdown_report(report_md_path, report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Lab-only intraday group-level past-only feature expansion dry-run."
    )
    parser.add_argument("--bar-samples", type=Path, default=DEFAULT_BAR_SAMPLES)
    parser.add_argument("--group-samples", type=Path, default=DEFAULT_GROUP_SAMPLES)
    parser.add_argument("--group-inconsistency-report", type=Path, default=DEFAULT_INCONSISTENCY_REPORT)
    parser.add_argument("--design", type=Path, default=DEFAULT_DESIGN)
    parser.add_argument("--leakage-checklist", type=Path, default=DEFAULT_LEAKAGE_CHECKLIST)
    parser.add_argument("--out-dir", type=Path, default=ALLOWED_OUTPUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_dryrun(
            bar_samples=args.bar_samples,
            group_samples=args.group_samples,
            group_inconsistency_report=args.group_inconsistency_report,
            design_path=args.design,
            leakage_checklist_path=args.leakage_checklist,
            out_dir=args.out_dir,
        )
    except FeatureExpansionDryRunError as exc:
        print(json.dumps({"status": "failed", "p0_blockers": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    print(
        json.dumps(
            {
                "status": report["status"],
                "readiness_decision": report["readiness_decision"],
                "generated_feature_count": report["generated_feature_count"],
                "skipped_feature_count": report["skipped_feature_count"],
                "group_count": report["group_count"],
                "manifest_leakage_status": report["manifest_leakage_check"]["status"],
                "training_allowed": False,
                "stable_allowed": False,
                "qmt_allowed": False,
                "order_intent_allowed": False,
                "metrics_are_effectiveness_evidence": False,
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
