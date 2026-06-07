from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


FORBIDDEN_FEATURE_COLUMNS = {
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
    "best_in_sector_1d",
    "best_in_sector_3d",
    "top_quantile_in_sector_3d",
    "avoid_in_sector",
    "pairwise_outperform_label",
    "trade_date",
    "sector",
    "etf_code",
    "etf_name",
    "ranking_group_id",
    "model_version",
    "feature_version",
}

PREDICTION_FIELDS = [
    "trade_date",
    "sector",
    "etf_code",
    "etf_name",
    "ranking_group_id",
    "target_label",
    "y_true",
    "model_name",
    "y_score",
    "y_pred",
    "split",
]

REPORT_TYPE = "table_ml_baseline_smoke"
TASK_SCOPE = "Lab-only no-save baseline smoke"


class BaselineSmokeError(RuntimeError):
    pass


@dataclass(frozen=True)
class SplitData:
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    train_dates: list[str]
    valid_dates: list[str]
    group_leakage: list[str] = field(default_factory=list)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BaselineSmokeError(f"JSON root must be an object: {path}")
    return value


def require_manifest_boundaries(manifest: dict[str, Any]) -> None:
    expected = {
        "training_allowed": False,
        "stable_effect_allowed": False,
        "advisory_only": True,
        "affects_stable_trading": False,
        "contains_secret": False,
        "contains_live_order": False,
        "contains_order_intent": False,
    }
    failures = [
        f"{field_name} must be {str(expected_value).lower()}"
        for field_name, expected_value in expected.items()
        if manifest.get(field_name) is not expected_value
    ]
    if failures:
        raise BaselineSmokeError("; ".join(failures))


def feature_columns_from_contract(contract: dict[str, Any]) -> list[str]:
    field_classification = contract.get("field_classification")
    if isinstance(field_classification, dict):
        raw = field_classification.get("candidate_features") or field_classification.get(
            "numeric_candidate_features"
        )
    else:
        raw = contract.get("feature_columns")

    if not isinstance(raw, list):
        raise BaselineSmokeError("feature contract must contain feature_columns or field_classification.candidate_features")

    feature_columns = [item for item in raw if isinstance(item, str) and item]
    if not feature_columns:
        raise BaselineSmokeError("feature contract has no feature columns")
    return feature_columns


def validate_feature_columns(df: pd.DataFrame, feature_columns: Sequence[str]) -> None:
    forbidden = sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS)
    if forbidden:
        raise BaselineSmokeError("forbidden columns in feature_columns: " + ", ".join(forbidden))

    missing = [column for column in feature_columns if column not in df.columns]
    if missing:
        raise BaselineSmokeError("feature columns missing from sample: " + ", ".join(missing))

    non_numeric: list[str] = []
    for column in feature_columns:
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.isna().any():
            non_numeric.append(column)
    if non_numeric:
        raise BaselineSmokeError("feature columns must be complete numeric columns: " + ", ".join(non_numeric))


def choose_target(df: pd.DataFrame, requested_target: str) -> str:
    if requested_target in df.columns:
        return requested_target
    fallback = "best_in_sector_3d"
    if fallback in df.columns:
        return fallback
    raise BaselineSmokeError(f"target not found and fallback unavailable: {requested_target}")


def chronological_split(df: pd.DataFrame) -> SplitData:
    dates = sorted(str(value) for value in df["trade_date"].unique())
    if len(dates) < 2:
        raise BaselineSmokeError("chronological split requires at least 2 dates")

    train_date_count = max(1, int(math.floor(len(dates) * 0.7)))
    if train_date_count >= len(dates):
        train_date_count = len(dates) - 1

    train_dates = dates[:train_date_count]
    valid_dates = dates[train_date_count:]
    train_df = df[df["trade_date"].astype(str).isin(train_dates)].copy()
    valid_df = df[df["trade_date"].astype(str).isin(valid_dates)].copy()
    if train_df.empty or valid_df.empty:
        raise BaselineSmokeError("chronological split produced an empty train or validation set")

    train_groups = set(train_df["ranking_group_id"].astype(str))
    valid_groups = set(valid_df["ranking_group_id"].astype(str))
    group_leakage = sorted(train_groups & valid_groups)
    if group_leakage:
        raise BaselineSmokeError("ranking_group_id appears in both train and validation: " + ", ".join(group_leakage))

    return SplitData(
        train_df=train_df,
        valid_df=valid_df,
        train_dates=train_dates,
        valid_dates=valid_dates,
        group_leakage=group_leakage,
    )


def standardize(train_values: np.ndarray, valid_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0)
    std[std == 0] = 1.0
    return (train_values - mean) / std, (valid_values - mean) / std


def fit_numpy_logistic_regression(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int = 200,
    learning_rate: float = 0.05,
    l2: float = 0.01,
) -> np.ndarray:
    x_train_i = np.column_stack([np.ones(x_train.shape[0]), x_train])
    weights = np.zeros(x_train_i.shape[1], dtype=float)
    for _ in range(epochs):
        probs = sigmoid(x_train_i @ weights)
        grad = (x_train_i.T @ (probs - y_train)) / len(y_train)
        grad[1:] += l2 * weights[1:]
        weights -= learning_rate * grad
    return weights


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def predict_scores(weights: np.ndarray, x_values: np.ndarray) -> np.ndarray:
    x_values_i = np.column_stack([np.ones(x_values.shape[0]), x_values])
    return sigmoid(x_values_i @ weights)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    return float((y_true == y_pred).mean()) if len(y_true) else None


def log_loss(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(y_true) == 0:
        return None
    eps = 1e-15
    score = np.clip(y_score, eps, 1 - eps)
    return float(-(y_true * np.log(score) + (1 - y_true) * np.log(1 - score)).mean())


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = y_score[y_true == 1]
    negatives = y_score[y_true == 0]
    if len(positives) == 0 or len(negatives) == 0:
        return None

    wins = 0.0
    for positive in positives:
        wins += float((positive > negatives).sum()) + 0.5 * float((positive == negatives).sum())
    return float(wins / (len(positives) * len(negatives)))


def class_balance(values: Sequence[int]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {str(value): int(counts.get(value, 0)) for value in (0, 1)}


def grouped_validation_summary(
    valid_df: pd.DataFrame,
    group_column: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> list[dict[str, Any]]:
    frame = valid_df[[group_column]].copy()
    frame["y_true"] = y_true
    frame["y_pred"] = y_pred
    frame["y_score"] = y_score

    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(group_column, sort=True):
        group_y_true = group["y_true"].to_numpy()
        group_y_pred = group["y_pred"].to_numpy()
        group_y_score = group["y_score"].to_numpy()
        rows.append(
            {
                group_column: str(key),
                "count": int(len(group)),
                "class_balance": class_balance(group_y_true),
                "accuracy": accuracy(group_y_true, group_y_pred),
                "roc_auc": roc_auc(group_y_true, group_y_score),
                "mean_score": float(np.mean(group_y_score)) if len(group_y_score) else None,
            }
        )
    return rows


def build_review_checklist() -> dict[str, Any]:
    return {
        "researched": "E sector internal ranking Lab-only no-save baseline smoke.",
        "data_source": "Local Lab sample, manifest, and feature contract.",
        "uses_stable_bundle": False,
        "future_leakage": "future / label / id / group fields are forbidden as features.",
        "affects_stable_trading": False,
        "read_only_advisory": False,
        "recommended_for_stable": False,
        "minimal_stable_merge": "not applicable",
        "do_not_submit_to_stable": True,
        "next_step": "Only expand no-save smoke coverage after human review.",
    }


def write_predictions(
    path: Path,
    valid_df: pd.DataFrame,
    *,
    target_label: str,
    model_name: str,
    y_true: np.ndarray,
    y_score: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    rows = []
    for index, row in valid_df.reset_index(drop=True).iterrows():
        rows.append(
            {
                "trade_date": row["trade_date"],
                "sector": row["sector"],
                "etf_code": str(row["etf_code"]),
                "etf_name": row["etf_name"],
                "ranking_group_id": row["ranking_group_id"],
                "target_label": target_label,
                "y_true": int(y_true[index]),
                "model_name": model_name,
                "y_score": f"{float(y_score[index]):.10f}",
                "y_pred": int(y_pred[index]),
                "split": "validation",
            }
        )

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def build_report_markdown(report: dict[str, Any]) -> str:
    metrics = report["metrics"][0]
    data = report["data"]
    split = report["split"]
    leakage = report["feature_leakage_check"]
    boundary = report["boundary"]
    auc = metrics["roc_auc"]
    auc_note = "valid has both classes" if auc is not None else "valid has a single class or insufficient samples"
    return f"""# E Sector Internal Ranking Baseline Smoke Report

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## 任务定位
Lab-only baseline smoke，不是正式训练，不是 advisory，不是 Stable 接口。

## 数据
- rows: {data["rows"]}
- features: {data["feature_count"]}
- date range: {data["date_start"]} to {data["date_end"]}
- train dates: {", ".join(data["train_dates"])}
- validation dates: {", ".join(data["validation_dates"])}

## Feature leakage check
- feature columns: see JSON report
- forbidden columns entered feature: {leakage["feature_forbidden_intersection"]}
- future labels only as labels: {str(leakage["future_labels_only_as_labels"]).lower()}

## Split
- chronological: true
- no shuffle: true
- train_count: {split["train_count"]}
- valid_count: {split["valid_count"]}
- group leakage check passed: {str(split["group_leakage_check_passed"]).lower()}

## Models
- numpy logistic regression smoke
- no-save: true
- no tuning: true

## Metrics
- target_label: `{metrics["target_label"]}`
- train class balance: {metrics["class_balance_train"]}
- valid class balance: {metrics["class_balance_valid"]}
- accuracy: {metrics["accuracy"]}
- roc_auc: {auc} ({auc_note})
- log_loss: {metrics["log_loss"]}

## Boundary
- no Stable: {str(boundary["no_stable"]).lower()}
- no QMT: {str(boundary["no_qmt"]).lower()}
- no OrderIntent: {str(boundary["no_order_intent"]).lower()}
- no output/: {str(boundary["no_output"]).lower()}
- no lab_advisory/: {str(boundary["no_lab_advisory"]).lower()}
- no model save: {str(boundary["no_model_save"]).lower()}
- no trading advice: {str(boundary["not_trading_advice"]).lower()}

## Review Checklist 自检
1. 研究了什么：E sector internal ranking Lab-only no-save baseline smoke 工具路径。
2. 数据来自哪里：本地 Lab sample、manifest 和 feature contract。
3. 是否来自 Stable bundle：否。
4. 是否有未来函数：未发现；future / label / id / group 字段未进入 feature。
5. 是否影响 Stable 正式交易：否。
6. 是否只读 advisory：否，本任务不是 advisory 包。
7. 是否建议进入 Stable：否。
8. 如果建议进入 Stable，最小合并方案是什么：不适用。
9. 不允许直接提交到 Stable：确认不允许。
10. 下一步建议是什么：仅在人工确认后扩展更多 no-save smoke 模型或更大样本。
"""


def run_baseline_smoke(
    sample_path: Path,
    manifest_path: Path,
    feature_contract_path: Path,
    target: str,
    out_dir: Path,
) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    require_manifest_boundaries(manifest)
    contract = load_json(feature_contract_path)
    feature_columns = feature_columns_from_contract(contract)

    df = pd.read_csv(sample_path, dtype={"etf_code": str})
    validate_feature_columns(df, feature_columns)
    target_label = choose_target(df, target)

    split = chronological_split(df)
    x_train_raw = split.train_df[list(feature_columns)].astype(float).to_numpy()
    x_valid_raw = split.valid_df[list(feature_columns)].astype(float).to_numpy()
    y_train = split.train_df[target_label].astype(int).to_numpy()
    y_valid = split.valid_df[target_label].astype(int).to_numpy()

    x_train, x_valid = standardize(x_train_raw, x_valid_raw)
    weights = fit_numpy_logistic_regression(x_train, y_train)
    train_score = predict_scores(weights, x_train)
    valid_score = predict_scores(weights, x_valid)
    train_pred = (train_score >= 0.5).astype(int)
    valid_pred = (valid_score >= 0.5).astype(int)

    model_name = "numpy_logistic_regression_smoke"
    prediction_path = out_dir / "sector_internal_ranking_baseline_predictions.csv"
    report_json_path = out_dir / "sector_internal_ranking_baseline_smoke_report.json"
    report_md_path = out_dir / "sector_internal_ranking_baseline_smoke_report.md"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "model_name": model_name,
        "train_count": int(len(split.train_df)),
        "valid_count": int(len(split.valid_df)),
        "feature_count": int(len(feature_columns)),
        "target_label": target_label,
        "class_balance_train": class_balance(y_train),
        "class_balance_valid": class_balance(y_valid),
        "train_accuracy": accuracy(y_train, train_pred),
        "accuracy": accuracy(y_valid, valid_pred),
        "roc_auc": roc_auc(y_valid, valid_score),
        "log_loss": log_loss(y_valid, valid_score),
        "by_date_validation_summary": grouped_validation_summary(
            split.valid_df, "trade_date", y_valid, valid_pred, valid_score
        ),
        "by_sector_validation_summary": grouped_validation_summary(
            split.valid_df, "sector", y_valid, valid_pred, valid_score
        ),
    }
    dates = sorted(str(value) for value in df["trade_date"].unique())
    report = {
        "report_type": REPORT_TYPE,
        "task_scope": TASK_SCOPE,
        "lab_only": True,
        "no_save": True,
        "no_tuning": True,
        "no_stable": True,
        "no_qmt": True,
        "no_order_intent": True,
        "no_output": True,
        "no_lab_advisory": True,
        "model_saved": False,
        "checkpoint_saved": False,
        "target_label": target_label,
        "feature_columns": list(feature_columns),
        "forbidden_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "train_count": int(len(split.train_df)),
        "valid_count": int(len(split.valid_df)),
        "split_method": "chronological",
        "group_leakage_check": "passed",
        "models": [
            {
                "model_name": model_name,
                "implementation": "numpy in-memory logistic regression",
                "no_save": True,
                "no_tuning": True,
                "parameters": {"epochs": 200, "learning_rate": 0.05, "l2": 0.01, "threshold": 0.5},
            }
        ],
        "metrics": [metrics],
        "prediction_file": str(prediction_path),
        "review_checklist": build_review_checklist(),
        "task": "sector_internal_ranking_baseline_smoke",
        "lab_boundary": "aetfq3-lab / Lab, not V2.1 Stable",
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "status": "passed",
        "data": {
            "sample_path": str(sample_path),
            "manifest_path": str(manifest_path),
            "feature_contract_path": str(feature_contract_path),
            "rows": int(len(df)),
            "feature_count": int(len(feature_columns)),
            "date_start": dates[0],
            "date_end": dates[-1],
            "train_dates": split.train_dates,
            "validation_dates": split.valid_dates,
        },
        "feature_leakage_check": {
            "feature_columns": list(feature_columns),
            "forbidden_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
            "feature_forbidden_intersection": sorted(set(feature_columns) & FORBIDDEN_FEATURE_COLUMNS),
            "future_labels_only_as_labels": True,
            "raw_sector_string_as_feature": False,
            "etf_identity_as_feature": False,
            "ranking_group_id_as_feature": False,
        },
        "split": {
            "type": "chronological",
            "shuffle": False,
            "train_date_count": len(split.train_dates),
            "validation_date_count": len(split.valid_dates),
            "train_count": int(len(split.train_df)),
            "valid_count": int(len(split.valid_df)),
            "group_leakage_check_passed": not split.group_leakage,
            "group_leakage": split.group_leakage,
        },
        "boundary": {
            "no_stable": True,
            "no_qmt": True,
            "no_order_intent": True,
            "no_output": True,
            "no_lab_advisory": True,
            "no_model_save": True,
            "no_checkpoint": True,
            "no_hyperparameter_search": True,
            "not_trading_advice": True,
        },
    }

    write_predictions(
        prediction_path,
        split.valid_df,
        target_label=target_label,
        model_name=model_name,
        y_true=y_valid,
        y_score=valid_score,
        y_pred=valid_pred,
    )
    report_json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_md_path.write_text(build_report_markdown(report), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only no-save table ML baseline smoke.")
    parser.add_argument("--sample", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--feature-contract", required=True, type=Path)
    parser.add_argument("--target", default="top_quantile_in_sector_3d")
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_baseline_smoke(
            sample_path=args.sample,
            manifest_path=args.manifest,
            feature_contract_path=args.feature_contract,
            target=args.target,
            out_dir=args.out_dir,
        )
    except BaselineSmokeError as exc:
        print(f"FAILED baseline_smoke_valid=false {exc}", file=sys.stderr)
        return 1

    metrics = report["metrics"][0]
    print(
        json.dumps(
            {
                "status": report["status"],
                "model_name": metrics["model_name"],
                "train_count": metrics["train_count"],
                "valid_count": metrics["valid_count"],
                "feature_count": metrics["feature_count"],
                "target_label": metrics["target_label"],
                "accuracy": metrics["accuracy"],
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
                "prediction_file": report["prediction_file"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
