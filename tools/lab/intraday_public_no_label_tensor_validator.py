from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_public_no_label_tensor_validation")
REPORT_TYPE = "intraday_public_no_label_tensor_validation"
REQUIRED_COLUMNS = [
    "trade_date",
    "datetime",
    "etf_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]
DEFAULT_FEATURE_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "intraday_return",
    "return_from_open",
    "distance_to_vwap",
]
EXPLICIT_FORBIDDEN_COLUMNS = {
    "max_drawdown_3d",
    "execution_return_to_close",
    "execution_return_to_next_open",
    "execution_drawdown_after_entry",
}


class PublicNoLabelTensorValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorShapeBundle:
    batch_size: int
    min_time_steps: int
    max_time_steps: int
    feature_count: int
    feature_columns: list[str]
    sequence_keys: list[dict[str, str]]
    nan_count: int
    inf_count: int


def is_forbidden_feature(column: str) -> bool:
    return column.startswith("future_") or column.endswith("_label") or column in EXPLICIT_FORBIDDEN_COLUMNS


def scan_forbidden_features(feature_columns: Sequence[str]) -> dict[str, Any]:
    forbidden = [column for column in feature_columns if is_forbidden_feature(column)]
    return {
        "passed": not forbidden,
        "forbidden_columns": forbidden,
        "forbidden_rules": [
            "all future_* fields",
            "all *_label fields",
            "max_drawdown_3d",
            "execution_return_to_close",
            "execution_return_to_next_open",
            "execution_drawdown_after_entry",
        ],
    }


def resolve_cli_output_dir(out_dir: Path, repo_root: Path = REPO_ROOT) -> Path:
    resolved = (repo_root / out_dir if not out_dir.is_absolute() else out_dir).resolve()
    allowed = (repo_root / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise PublicNoLabelTensorValidationError(f"out-dir must be under {ALLOWED_OUTPUT_ROOT}") from exc
    return resolved


def read_public_intraday_csv(input_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(input_path, dtype={"etf_code": str})
    except OSError as exc:
        raise PublicNoLabelTensorValidationError(f"input CSV cannot be read: {exc}") from exc


def ensure_required_columns(df: pd.DataFrame) -> None:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise PublicNoLabelTensorValidationError(f"missing required columns: {missing}")
    if df.empty:
        raise PublicNoLabelTensorValidationError("input CSV has no rows")


def enrich_public_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    numeric_columns = ["open", "high", "low", "close", "volume", "amount"]
    if "vwap" in enriched.columns:
        numeric_columns.append("vwap")
    for column in numeric_columns:
        enriched[column] = pd.to_numeric(enriched[column], errors="raise")

    if "vwap" not in enriched.columns:
        volume = enriched["volume"].replace(0, np.nan)
        enriched["vwap"] = enriched["amount"] / volume

    sort_columns = ["trade_date", "etf_code", "datetime"]
    if "bar_index" in enriched.columns:
        enriched["bar_index"] = pd.to_numeric(enriched["bar_index"], errors="raise")
        sort_columns = ["trade_date", "etf_code", "bar_index", "datetime"]
    enriched = enriched.sort_values(sort_columns).reset_index(drop=True)

    grouped = enriched.groupby(["etf_code", "trade_date"], sort=True)
    prior_close = grouped["close"].shift(1)
    enriched["intraday_return"] = (enriched["close"] / prior_close) - 1
    first_bar = prior_close.isna()
    enriched.loc[first_bar, "intraday_return"] = (
        enriched.loc[first_bar, "close"] / enriched.loc[first_bar, "open"]
    ) - 1
    open_by_group = grouped["open"].transform("first")
    enriched["return_from_open"] = (enriched["close"] / open_by_group) - 1
    enriched["distance_to_vwap"] = (enriched["close"] / enriched["vwap"]) - 1
    return enriched


def select_feature_columns(
    df: pd.DataFrame,
    requested_features: Sequence[str] | None = None,
) -> list[str]:
    features = list(requested_features or DEFAULT_FEATURE_COLUMNS)
    scan = scan_forbidden_features(features)
    if not scan["passed"]:
        raise PublicNoLabelTensorValidationError(f"forbidden feature columns: {scan['forbidden_columns']}")

    missing = [column for column in features if column not in df.columns]
    if missing:
        raise PublicNoLabelTensorValidationError(f"feature columns missing from CSV: {missing}")
    if not features:
        raise PublicNoLabelTensorValidationError("feature_count must be greater than 0")
    return features


def build_no_label_tensor_shape(
    input_path: Path,
    feature_columns: Sequence[str] | None = None,
    min_required_time_steps: int = 12,
) -> TensorShapeBundle:
    df = read_public_intraday_csv(input_path)
    ensure_required_columns(df)
    df = enrich_public_features(df)
    features = select_feature_columns(df, feature_columns)

    for column in features:
        df[column] = pd.to_numeric(df[column], errors="raise")

    sequence_keys: list[dict[str, str]] = []
    sequence_arrays: list[np.ndarray] = []
    time_step_counts: list[int] = []
    for (etf_code, trade_date), group in df.groupby(["etf_code", "trade_date"], sort=True):
        ordered = group.sort_values(["bar_index", "datetime"] if "bar_index" in group.columns else ["datetime"])
        time_step_counts.append(len(ordered))
        sequence_keys.append({"etf_code": str(etf_code), "trade_date": str(trade_date)})
        sequence_arrays.append(ordered[features].to_numpy(dtype=np.float32))

    if not sequence_arrays:
        raise PublicNoLabelTensorValidationError("no ETF/day groups constructed")

    min_time_steps = min(time_step_counts)
    max_time_steps = max(time_step_counts)
    if min_time_steps < min_required_time_steps:
        raise PublicNoLabelTensorValidationError(
            f"min_time_steps={min_time_steps} is below required {min_required_time_steps}"
        )
    if min_time_steps != max_time_steps:
        raise PublicNoLabelTensorValidationError(
            f"non-uniform time_steps: min={min_time_steps}, max={max_time_steps}"
        )

    tensor = np.stack(sequence_arrays)
    nan_count = int(np.isnan(tensor).sum())
    inf_count = int(np.isinf(tensor).sum())
    return TensorShapeBundle(
        batch_size=int(tensor.shape[0]),
        min_time_steps=int(min_time_steps),
        max_time_steps=int(max_time_steps),
        feature_count=int(tensor.shape[2]),
        feature_columns=features,
        sequence_keys=sequence_keys,
        nan_count=nan_count,
        inf_count=inf_count,
    )


def run_public_no_label_tensor_validation(
    input_path: Path,
    out_dir: Path,
    feature_columns: Sequence[str] | None = None,
) -> dict[str, Any]:
    p0_blockers: list[str] = []
    bundle: TensorShapeBundle | None = None
    rows_checked = etf_count = trade_date_count = 0
    try:
        df = read_public_intraday_csv(input_path)
        rows_checked = int(len(df))
        if "etf_code" in df.columns:
            etf_count = int(df["etf_code"].astype(str).nunique())
        if "trade_date" in df.columns:
            trade_date_count = int(df["trade_date"].astype(str).nunique())
        bundle = build_no_label_tensor_shape(input_path, feature_columns=feature_columns)
        if bundle.nan_count:
            p0_blockers.append(f"feature tensor contains NaN values: {bundle.nan_count}")
        if bundle.inf_count:
            p0_blockers.append(f"feature tensor contains Inf values: {bundle.inf_count}")
    except Exception as exc:
        p0_blockers.append(str(exc))

    selected_features = bundle.feature_columns if bundle else list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    scan = scan_forbidden_features(selected_features)
    tensor_shape_passed = bundle is not None and not p0_blockers and scan["passed"]
    report = {
        "report_type": REPORT_TYPE,
        "status": "passed" if tensor_shape_passed else "failed",
        "lab_only": True,
        "no_training": True,
        "no_qmt": True,
        "no_order_intent": True,
        "no_stable": True,
        "no_output": True,
        "no_lab_advisory": True,
        "model_saved": False,
        "checkpoint_saved": False,
        "order_intent_generated": False,
        "labels_required": False,
        "target_count": 0,
        "input_file": str(input_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_checked": rows_checked,
        "etf_count": etf_count,
        "trade_date_count": trade_date_count,
        "batch_size": bundle.batch_size if bundle else 0,
        "min_time_steps": bundle.min_time_steps if bundle else 0,
        "max_time_steps": bundle.max_time_steps if bundle else 0,
        "feature_count": bundle.feature_count if bundle else 0,
        "feature_columns": selected_features,
        "nan_count": bundle.nan_count if bundle else 0,
        "inf_count": bundle.inf_count if bundle else 0,
        "tensor_shape_passed": tensor_shape_passed,
        "forbidden_feature_passed": scan["passed"],
        "forbidden_columns": scan["forbidden_columns"],
        "sequence_keys": bundle.sequence_keys if bundle else [],
        "p0_blockers": p0_blockers,
    }
    write_reports(report, out_dir)
    return report


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intraday_public_no_label_tensor_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。",
        "",
        "# Intraday Public No-Label Tensor Validation Report",
        "",
        f"- status: {report['status']}",
        f"- rows_checked: {report['rows_checked']}",
        f"- etf_count: {report['etf_count']}",
        f"- trade_date_count: {report['trade_date_count']}",
        f"- batch_size: {report['batch_size']}",
        f"- min_time_steps: {report['min_time_steps']}",
        f"- max_time_steps: {report['max_time_steps']}",
        f"- feature_count: {report['feature_count']}",
        f"- labels_required: {str(report['labels_required']).lower()}",
        f"- target_count: {report['target_count']}",
        f"- nan_count: {report['nan_count']}",
        f"- inf_count: {report['inf_count']}",
        f"- tensor_shape_passed: {str(report['tensor_shape_passed']).lower()}",
        "- boundary: public OHLCV tensor shape validation only; no labels, training, QMT, OrderIntent, Stable, output/, lab_advisory, checkpoint, or model save.",
    ]
    (out_dir / "intraday_public_no_label_tensor_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab-only public intraday no-label tensor shape validation.")
    parser.add_argument("--input", required=True, type=Path, help="Public intraday 5m OHLCV CSV.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Ignored local report directory.")
    parser.add_argument("--feature-column", action="append", default=None, help="Optional feature override.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out_dir = resolve_cli_output_dir(args.out_dir)
        report = run_public_no_label_tensor_validation(args.input, out_dir, feature_columns=args.feature_column)
    except PublicNoLabelTensorValidationError as exc:
        print(json.dumps({"status": "failed", "p0_blockers": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": report["status"],
        "rows_checked": report["rows_checked"],
        "etf_count": report["etf_count"],
        "trade_date_count": report["trade_date_count"],
        "batch_size": report["batch_size"],
        "min_time_steps": report["min_time_steps"],
        "max_time_steps": report["max_time_steps"],
        "feature_count": report["feature_count"],
        "labels_required": report["labels_required"],
        "target_count": report["target_count"],
        "tensor_shape_passed": report["tensor_shape_passed"],
        "nan_count": report["nan_count"],
        "inf_count": report["inf_count"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
