from __future__ import annotations

import argparse
import io
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_long_history_data_lake"
DEFAULT_RAW_DIR = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_long_history_raw_exports")
DEFAULT_FALLBACK_MANUAL_INBOX = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox")
ALLOWED_ARTIFACT_DIR = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_long_history_data_lake")
ALLOWED_REPORT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_long_history_alpha_optimization")
DATA_SUFFIXES = {".csv", ".txt", ".zip", ".parquet"}
METADATA_FILE_NAMES = {"manifest.json", "sha256sums.txt", "source_note.md"}
STANDARD_COLUMNS = [
    "trade_date",
    "datetime",
    "etf_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "frequency",
    "source_file",
]
SIGNAL_CLOCKS = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "14:50"]
TIME_CENSORED_FEATURES = [
    "return_since_open",
    "high_so_far_vs_open",
    "low_so_far_vs_open",
    "close_now_vs_open",
    "close_now_vs_vwap_so_far",
    "volume_so_far",
    "amount_so_far",
    "last_3_bar_return",
    "last_6_bar_return",
    "intraday_volatility_so_far",
    "range_so_far",
    "drawdown_so_far",
    "rebound_from_low_so_far",
    "price_above_vwap_ratio_so_far",
    "volume_acceleration_so_far",
]
OUTCOME_COLUMNS = [
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
    "entry_to_exit_return",
]
LABEL_COLUMNS = [
    "ret_1d_gt_0bp",
    "ret_1d_gt_20bp",
    "ret_3d_gt_0bp",
    "ret_3d_gt_100bp",
    "safe_positive_3d",
    "drawdown_limited_positive_3d",
]
BOUNDARY_FIELDS = {
    "access_mode": "READ_ONLY",
    "final_action_change_allowed": False,
    "contains_live_order": False,
    "contains_secret": False,
    "requires_human_review": True,
    "promotion_gate_required": True,
    "formal_training": False,
    "formal_model_evidence": False,
    "stable_promotion_ready": False,
    "stable_evidence": False,
    "qmt_ready": False,
    "order_intent_ready": False,
    "automatic_promotion_ready": False,
    "model_saved": False,
    "scaler_saved": False,
    "checkpoint_saved": False,
    "gpu_used": False,
    "torchrun_used": False,
    "not_trading_advice": True,
}


class LongHistoryDataLakeError(RuntimeError):
    pass


@dataclass(frozen=True)
class DataLakeConfig:
    raw_dir: Path
    fallback_manual_inbox: Path
    out_artifact_dir: Path
    out_report_dir: Path


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def ensure_under(path: Path, allowed: Path, repo_root: Path = REPO_ROOT, label: str = "path") -> Path:
    repo_root = repo_root.resolve()
    resolved = resolve_repo_path(path, repo_root).resolve()
    allowed_resolved = resolve_repo_path(allowed, repo_root).resolve()
    try:
        resolved.relative_to(allowed_resolved)
    except ValueError as exc:
        raise LongHistoryDataLakeError(f"{label} must be under {allowed}") from exc
    return resolved


def resolve_artifact_dir(path: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    return ensure_under(path, ALLOWED_ARTIFACT_DIR, repo_root, "out-artifact-dir") if enforce else resolve_repo_path(path, repo_root).resolve()


def resolve_report_dir(path: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    return ensure_under(path, ALLOWED_REPORT_DIR, repo_root, "out-report-dir") if enforce else resolve_repo_path(path, repo_root).resolve()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            return str(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def discover_data_files(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.rglob("*")
        if path.is_file() and path.name.lower() not in METADATA_FILE_NAMES and path.suffix.lower() in DATA_SUFFIXES
    )


def infer_etf_code(text: str) -> str | None:
    match = re.search(r"(?:SH|SZ)?#?(\d{6})", text.upper())
    return match.group(1) if match else None


def normalize_etf_code(value: Any, fallback: str | None = None) -> str:
    text = str(value or "").strip().upper()
    match = re.search(r"(\d{6})", text)
    if match:
        return match.group(1)
    return fallback or text


COLUMN_ALIASES = {
    "trade_date": ["trade_date", "date", "tradedate", "交易日期", "日期", "自然日"],
    "datetime": ["datetime", "date_time", "time", "timestamp", "bar_time", "时间", "日期时间", "成交时间"],
    "etf_code": ["etf_code", "code", "symbol", "ticker", "证券代码", "代码", "标的代码"],
    "open": ["open", "开盘", "开盘价"],
    "high": ["high", "最高", "最高价"],
    "low": ["low", "最低", "最低价"],
    "close": ["close", "收盘", "收盘价", "最新价"],
    "volume": ["volume", "vol", "成交量", "数量"],
    "amount": ["amount", "turnover", "money", "成交额", "金额"],
    "vwap": ["vwap", "均价"],
}


def normalized_name(name: Any) -> str:
    return re.sub(r"[\s_\-/#.]+", "", str(name or "").strip().lower())


def map_columns(columns: Sequence[str]) -> dict[str, str]:
    by_normalized = {normalized_name(column): column for column in columns}
    mapping: dict[str, str] = {}
    for standard, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            found = by_normalized.get(normalized_name(alias))
            if found is not None:
                mapping[standard] = found
                break
    return mapping


def read_delimited_bytes(payload: bytes, source_name: str) -> pd.DataFrame:
    errors: list[str] = []
    for encoding in ("utf-8-sig", "utf-8", "gbk", "ansi"):
        try:
            text = payload.decode(encoding)
            return pd.read_csv(io.StringIO(text), sep=None, engine="python")
        except Exception as exc:  # noqa: BLE001 - collect parse attempts for source diagnostics.
            errors.append(f"{encoding}: {exc}")
    raise LongHistoryDataLakeError(f"cannot read delimited file {source_name}: {'; '.join(errors[:3])}")


def read_one_file(path: Path) -> list[tuple[str, pd.DataFrame]]:
    suffix = path.suffix.lower()
    if suffix == ".zip":
        frames: list[tuple[str, pd.DataFrame]] = []
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                inner_suffix = Path(name).suffix.lower()
                if inner_suffix not in DATA_SUFFIXES - {".zip"}:
                    continue
                payload = archive.read(name)
                source_name = f"{path.name}!{name}"
                if inner_suffix == ".parquet":
                    frames.append((source_name, pd.read_parquet(io.BytesIO(payload))))
                else:
                    frames.append((source_name, read_delimited_bytes(payload, source_name)))
        return frames
    if suffix == ".parquet":
        return [(path.name, pd.read_parquet(path))]
    return [(path.name, read_delimited_bytes(path.read_bytes(), path.name))]


def parse_datetime_series(df: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    if "datetime" in mapping:
        raw = df[mapping["datetime"]].astype(str).str.strip()
        parsed = pd.to_datetime(raw, errors="coerce")
        if parsed.notna().any():
            return parsed
    if "trade_date" not in mapping:
        return pd.Series(pd.NaT, index=df.index)
    date_raw = df[mapping["trade_date"]].astype(str).str.strip()
    if "datetime" in mapping:
        time_raw = df[mapping["datetime"]].astype(str).str.strip()
        return pd.to_datetime(date_raw + " " + time_raw, errors="coerce")
    return pd.to_datetime(date_raw, errors="coerce")


def detect_frequency(datetimes: pd.Series, rows_per_day: pd.Series | None = None) -> str:
    values = datetimes.dropna().sort_values().drop_duplicates()
    deltas = values.diff().dropna().dt.total_seconds()
    positive = deltas[deltas > 0]
    if not positive.empty:
        median_seconds = float(positive.median())
        if median_seconds <= 90:
            return "1m"
        if median_seconds <= 390:
            return "5m"
        if median_seconds <= 990:
            return "15m"
        if median_seconds <= 1890:
            return "30m"
        return f"{max(1, round(median_seconds / 60))}m"
    if rows_per_day is not None and not rows_per_day.empty:
        median_count = float(rows_per_day.median())
        if median_count >= 180:
            return "1m"
        if median_count >= 36:
            return "5m"
    return "unknown"


def standardize_frame(df: pd.DataFrame, source_file: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = map_columns(list(df.columns))
    missing = [column for column in ["datetime", "open", "high", "low", "close", "volume"] if column not in mapping]
    fallback_code = infer_etf_code(source_file)
    result = pd.DataFrame(index=df.index)
    result["datetime"] = parse_datetime_series(df, mapping)
    result["trade_date"] = (
        pd.to_datetime(df[mapping["trade_date"]], errors="coerce").dt.date.astype(str)
        if "trade_date" in mapping
        else result["datetime"].dt.date.astype(str)
    )
    result["etf_code"] = df[mapping["etf_code"]].map(lambda value: normalize_etf_code(value, fallback_code)) if "etf_code" in mapping else fallback_code
    for column in ["open", "high", "low", "close", "volume", "amount", "vwap"]:
        source = mapping.get(column)
        result[column] = pd.to_numeric(df[source], errors="coerce") if source else pd.NA
    if result["amount"].isna().all():
        result["amount"] = result["close"] * result["volume"]
    if result["vwap"].isna().all():
        result["vwap"] = result["amount"] / result["volume"].where(result["volume"] != 0)
    per_day = result.groupby(["etf_code", "trade_date"], dropna=False).size() if "trade_date" in result else None
    result["frequency"] = detect_frequency(result["datetime"], per_day)
    result["source_file"] = source_file
    result = result[STANDARD_COLUMNS]
    report = {
        "source_file": source_file,
        "input_columns": list(map(str, df.columns)),
        "column_mapping": mapping,
        "missing_fields": missing + ([] if fallback_code or "etf_code" in mapping else ["etf_code"]),
        "row_count": int(len(result)),
        "detected_frequency": str(result["frequency"].iloc[0]) if len(result) else "unknown",
    }
    return result, report


def load_standard_bars(input_dir: Path) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    frames: list[pd.DataFrame] = []
    source_reports: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in discover_data_files(input_dir):
        try:
            for source_name, frame in read_one_file(path):
                standard, report = standardize_frame(frame, source_name)
                frames.append(standard)
                source_reports.append({**report, "path": str(path)})
        except Exception as exc:  # noqa: BLE001 - source-level report should not abort all other files.
            errors.append({"path": str(path), "error": str(exc)})
    if not frames:
        return pd.DataFrame(columns=STANDARD_COLUMNS), source_reports, errors
    bars = pd.concat(frames, ignore_index=True)
    bars = bars.dropna(subset=["datetime", "etf_code", "open", "high", "low", "close"])
    bars["trade_date"] = pd.to_datetime(bars["datetime"], errors="coerce").dt.date.astype(str)
    bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
    return bars.sort_values(["frequency", "etf_code", "datetime", "source_file"]).reset_index(drop=True), source_reports, errors


def quality_report(bars: pd.DataFrame, source_reports: list[dict[str, Any]], errors: list[dict[str, Any]], created_at_utc: str) -> dict[str, Any]:
    if bars.empty:
        return {
            "lab_declaration": LAB_DECLARATION,
            "report_type": "long_history_data_quality_report",
            "created_at_utc": created_at_utc,
            "data_quality_passed": False,
            "blocker": "no usable long-history intraday bars loaded",
            "source_reports": source_reports,
            "source_errors": errors,
            **BOUNDARY_FIELDS,
        }
    key_cols = ["etf_code", "datetime", "frequency"]
    duplicate_count = int(bars.duplicated(key_cols, keep=False).sum())
    conflict_rows = 0
    for _key, group in bars.groupby(key_cols, dropna=False):
        if len(group) > 1 and group[["open", "high", "low", "close", "volume", "amount"]].nunique(dropna=False).max() > 1:
            conflict_rows += len(group)
    ohlc_bad = bars[
        (bars["high"] < bars[["open", "close"]].max(axis=1))
        | (bars["low"] > bars[["open", "close"]].min(axis=1))
        | (bars["high"] < bars["low"])
    ]
    negative_volume = int((bars["volume"].fillna(0) < 0).sum())
    negative_amount = int((bars["amount"].fillna(0) < 0).sum())
    monotonic_failures = []
    for (code, day, freq), group in bars.groupby(["etf_code", "trade_date", "frequency"], dropna=False):
        if not group["datetime"].is_monotonic_increasing:
            monotonic_failures.append({"etf_code": code, "trade_date": day, "frequency": freq, "row_count": int(len(group))})
    bars_per_day = (
        bars.groupby(["etf_code", "trade_date", "frequency"], dropna=False)
        .size()
        .reset_index(name="bar_count")
        .groupby("frequency")["bar_count"]
        .agg(["min", "median", "max"])
        .reset_index()
        .to_dict("records")
    )
    passed = duplicate_count == 0 and conflict_rows == 0 and len(ohlc_bad) == 0 and negative_volume == 0 and negative_amount == 0
    return {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "long_history_data_quality_report",
        "created_at_utc": created_at_utc,
        "data_quality_passed": passed,
        "row_count": int(len(bars)),
        "etf_count": int(bars["etf_code"].nunique()),
        "date_coverage": {"start": str(bars["trade_date"].min()), "end": str(bars["trade_date"].max()), "trading_day_count": int(bars["trade_date"].nunique())},
        "frequency_coverage": sorted(map(str, bars["frequency"].dropna().unique())),
        "duplicate_bar_rows": duplicate_count,
        "conflict_rows": int(conflict_rows),
        "datetime_monotonic_failures": monotonic_failures[:50],
        "ohlc_inconsistent_rows": int(len(ohlc_bad)),
        "volume_negative_rows": negative_volume,
        "amount_negative_rows": negative_amount,
        "bars_per_etf_day": bars_per_day,
        "source_reports": source_reports,
        "source_errors": errors,
        **BOUNDARY_FIELDS,
    }


def resample_bars(bars: pd.DataFrame, target_frequency: str) -> pd.DataFrame:
    minutes = int(target_frequency.removesuffix("m"))
    source = bars[bars["frequency"] == "1m"].copy()
    if source.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS)
    source["datetime"] = pd.to_datetime(source["datetime"], errors="coerce")
    source = source.dropna(subset=["datetime"])
    source["bucket"] = source["datetime"].dt.floor(f"{minutes}min")
    grouped = source.sort_values("datetime").groupby(["etf_code", "trade_date", "bucket"], as_index=False)
    output = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        amount=("amount", "sum"),
        source_file=("source_file", "first"),
    )
    output["datetime"] = output["bucket"]
    output["vwap"] = output["amount"] / output["volume"].where(output["volume"] != 0)
    output["frequency"] = target_frequency
    return output[STANDARD_COLUMNS].sort_values(["etf_code", "datetime"]).reset_index(drop=True)


def safe_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator in (None, 0):
        return None
    return float(numerator) / float(denominator) - 1.0


def clean_float(value: Any) -> float | None:
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def build_feature_rows(bars: pd.DataFrame, signal_clocks: Sequence[str] = SIGNAL_CLOCKS) -> list[dict[str, Any]]:
    if bars.empty:
        return []
    source = bars[bars["frequency"].isin(["5m", "1m"])].copy()
    if source.empty:
        source = bars.copy()
    if "5m" in set(source["frequency"]):
        source = source[source["frequency"] == "5m"].copy()
    source = source.sort_values(["etf_code", "datetime"])
    daily_close_rows = []
    for (code, day), group in source.groupby(["etf_code", "trade_date"], dropna=False):
        ordered = group.sort_values("datetime")
        daily_close_rows.append({"etf_code": code, "trade_date": day, "close": clean_float(ordered.iloc[-1]["close"])})
    daily_by_code: dict[str, list[dict[str, Any]]] = {}
    for row in daily_close_rows:
        daily_by_code.setdefault(str(row["etf_code"]), []).append(row)
    for values in daily_by_code.values():
        values.sort(key=lambda item: str(item["trade_date"]))
    daily_index = {(code, row["trade_date"]): index for code, values in daily_by_code.items() for index, row in enumerate(values)}
    rows: list[dict[str, Any]] = []
    for (code, day), group in source.groupby(["etf_code", "trade_date"], dropna=False):
        ordered = group.sort_values("datetime")
        day_open = clean_float(ordered.iloc[0]["open"])
        if day_open in (None, 0):
            continue
        day_values = daily_by_code.get(str(code), [])
        day_idx = daily_index.get((str(code), str(day)))
        future_1 = day_values[day_idx + 1] if day_idx is not None and day_idx + 1 < len(day_values) else None
        future_3 = day_values[day_idx + 3] if day_idx is not None and day_idx + 3 < len(day_values) else None
        future_window = day_values[day_idx + 1 : day_idx + 4] if day_idx is not None else []
        close_today = clean_float(ordered.iloc[-1]["close"])
        future_return_1d = safe_ratio(clean_float(future_1.get("close")) if future_1 else None, close_today)
        future_return_3d = safe_ratio(clean_float(future_3.get("close")) if future_3 else None, close_today)
        future_closes = [clean_float(item.get("close")) for item in future_window]
        future_closes = [value for value in future_closes if value is not None]
        max_drawdown_3d = safe_ratio(min(future_closes), close_today) if len(future_closes) == 3 else None
        for clock in signal_clocks:
            hh, mm = [int(part) for part in clock.split(":")]
            censored = ordered[(ordered["datetime"].dt.hour < hh) | ((ordered["datetime"].dt.hour == hh) & (ordered["datetime"].dt.minute <= mm))]
            if censored.empty:
                continue
            current_close = clean_float(censored.iloc[-1]["close"])
            high_so_far = clean_float(censored["high"].max())
            low_so_far = clean_float(censored["low"].min())
            amount_so_far = clean_float(censored["amount"].sum())
            volume_so_far = clean_float(censored["volume"].sum())
            vwap_so_far = amount_so_far / volume_so_far if amount_so_far is not None and volume_so_far not in (None, 0) else clean_float(censored.iloc[-1]["vwap"])
            closes = censored["close"].astype(float)
            returns = closes.pct_change().dropna()
            row = {
                "trade_date": str(day),
                "datetime": censored.iloc[-1]["datetime"].isoformat(),
                "signal_clock": clock,
                "etf_code": str(code),
                "frequency": str(censored.iloc[-1]["frequency"]),
                "source_file": str(censored.iloc[-1]["source_file"]),
                "return_since_open": safe_ratio(current_close, day_open),
                "high_so_far_vs_open": safe_ratio(high_so_far, day_open),
                "low_so_far_vs_open": safe_ratio(low_so_far, day_open),
                "close_now_vs_open": safe_ratio(current_close, day_open),
                "close_now_vs_vwap_so_far": safe_ratio(current_close, vwap_so_far),
                "volume_so_far": volume_so_far,
                "amount_so_far": amount_so_far,
                "last_3_bar_return": safe_ratio(clean_float(censored.iloc[-1]["close"]), clean_float(censored.iloc[-3]["close"])) if len(censored) >= 3 else None,
                "last_6_bar_return": safe_ratio(clean_float(censored.iloc[-1]["close"]), clean_float(censored.iloc[-6]["close"])) if len(censored) >= 6 else None,
                "intraday_volatility_so_far": float(returns.std(ddof=0)) if len(returns) else 0.0,
                "range_so_far": safe_ratio(high_so_far, low_so_far),
                "drawdown_so_far": safe_ratio(current_close, high_so_far),
                "rebound_from_low_so_far": safe_ratio(current_close, low_so_far),
                "price_above_vwap_ratio_so_far": float((censored["close"].astype(float) > censored["vwap"].astype(float)).mean()),
                "volume_acceleration_so_far": float(censored["volume"].tail(3).mean() / censored["volume"].head(max(1, min(3, len(censored)))).mean() - 1.0)
                if float(censored["volume"].head(max(1, min(3, len(censored)))).mean()) != 0
                else None,
                "future_return_1d": future_return_1d,
                "future_return_3d": future_return_3d,
                "max_drawdown_3d": max_drawdown_3d,
                "entry_to_exit_return": future_return_3d,
                "ret_1d_gt_0bp": None if future_return_1d is None else int(future_return_1d > 0),
                "ret_1d_gt_20bp": None if future_return_1d is None else int(future_return_1d > 0.002),
                "ret_3d_gt_0bp": None if future_return_3d is None else int(future_return_3d > 0),
                "ret_3d_gt_100bp": None if future_return_3d is None else int(future_return_3d > 0.01),
                "safe_positive_3d": None if future_return_3d is None or max_drawdown_3d is None else int(future_return_3d > 0 and max_drawdown_3d > -0.02),
                "drawdown_limited_positive_3d": None
                if future_return_3d is None or max_drawdown_3d is None
                else int(future_return_3d > 0 and max_drawdown_3d > -0.03),
                "t_plus_1_date": future_1.get("trade_date") if future_1 else "",
                "t_plus_3_date": future_3.get("trade_date") if future_3 else "",
            }
            rows.append(row)
    return rows


def feature_manifest(created_at_utc: str, feature_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "long_history_feature_manifest",
        "created_at_utc": created_at_utc,
        "signal_clocks": SIGNAL_CLOCKS,
        "time_censored_feature_columns": TIME_CENSORED_FEATURES,
        "eod_group_level_features": {"allowed": True, "usage_boundary": "past_only; not for live intraday conclusion"},
        "diagnostic_label_columns": LABEL_COLUMNS,
        "outcome_columns": OUTCOME_COLUMNS,
        "feature_row_count": len(feature_rows),
        "future_label_or_outcome_in_feature_columns": bool(set(TIME_CENSORED_FEATURES) & (set(LABEL_COLUMNS) | set(OUTCOME_COLUMNS))),
        **BOUNDARY_FIELDS,
    }


def run_data_lake(config: DataLakeConfig, repo_root: Path = REPO_ROOT, *, enforce_paths: bool = True) -> dict[str, Any]:
    created_at_utc = datetime.now(timezone.utc).isoformat()
    artifact_dir = resolve_artifact_dir(config.out_artifact_dir, repo_root, enforce=enforce_paths)
    report_dir = resolve_report_dir(config.out_report_dir, repo_root, enforce=enforce_paths)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = resolve_repo_path(config.raw_dir, repo_root)
    fallback_dir = resolve_repo_path(config.fallback_manual_inbox, repo_root)
    input_dir = raw_dir if discover_data_files(raw_dir) else fallback_dir
    bars, source_reports, errors = load_standard_bars(input_dir)
    bars.to_csv(artifact_dir / "long_history_bars.csv", index=False, lineterminator="\n")
    inventory_rows = [
        {
            "source_file": item["source_file"],
            "path": item["path"],
            "row_count": item["row_count"],
            "detected_frequency": item["detected_frequency"],
            "missing_fields": "|".join(item["missing_fields"]),
        }
        for item in source_reports
    ]
    pd.DataFrame(inventory_rows).to_csv(report_dir / "long_history_bar_inventory.csv", index=False, lineterminator="\n")
    q_report = quality_report(bars, source_reports, errors, created_at_utc)
    write_json(report_dir / "long_history_data_quality_report.json", q_report)
    resampled: dict[str, int] = {}
    for target in ["5m", "15m", "30m"]:
        frame = resample_bars(bars, target)
        if not frame.empty:
            frame.to_csv(artifact_dir / f"long_history_{target}_bars.csv", index=False, lineterminator="\n")
            resampled[target] = int(len(frame))
    base_5m = pd.read_csv(artifact_dir / "long_history_5m_bars.csv") if (artifact_dir / "long_history_5m_bars.csv").exists() else bars[bars["frequency"] == "5m"].copy()
    if base_5m.empty:
        base_5m = bars.copy()
    base_5m["datetime"] = pd.to_datetime(base_5m["datetime"], errors="coerce")
    feature_rows = build_feature_rows(base_5m)
    pd.DataFrame(feature_rows).to_csv(artifact_dir / "long_history_feature_rows.csv", index=False, lineterminator="\n")
    resample_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "long_history_resample_report",
        "created_at_utc": created_at_utc,
        "source_has_1m": bool((bars["frequency"] == "1m").any()) if not bars.empty else False,
        "resampled_row_counts": resampled,
        "rules": {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "amount": "sum", "vwap": "amount/volume when volume nonzero"},
        **BOUNDARY_FIELDS,
    }
    write_json(report_dir / "long_history_resample_report.json", resample_report)
    manifest = feature_manifest(created_at_utc, feature_rows)
    write_json(report_dir / "long_history_feature_manifest.json", manifest)
    return {
        "decision": "LONG_HISTORY_DATA_LAKE_COMPLETED" if not bars.empty else "LONG_HISTORY_DATA_LAKE_BLOCKED_NO_INPUT",
        "input_dir": str(input_dir),
        "artifact_dir": str(artifact_dir),
        "report_dir": str(report_dir),
        "bar_count": int(len(bars)),
        "feature_row_count": len(feature_rows),
        "quality_report": q_report,
        "resample_report": resample_report,
        "feature_manifest": manifest,
        **BOUNDARY_FIELDS,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Lab-only long-history intraday data lake.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--fallback-manual-inbox", type=Path, default=DEFAULT_FALLBACK_MANUAL_INBOX)
    parser.add_argument("--out-artifact-dir", type=Path, default=ALLOWED_ARTIFACT_DIR)
    parser.add_argument("--out-report-dir", type=Path, default=ALLOWED_REPORT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = DataLakeConfig(args.raw_dir, args.fallback_manual_inbox, args.out_artifact_dir, args.out_report_dir)
    report = run_data_lake(config)
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    return 0 if report["bar_count"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
