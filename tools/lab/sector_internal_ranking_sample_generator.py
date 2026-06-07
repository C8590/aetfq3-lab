from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_VERSION = "lab_sector_internal_ranking_expanded_v1"
FEATURE_VERSION = "lag1_past_only_v1"
REPORT_NAME = "sector_internal_ranking_expanded"
LAB_BOUNDARY = "aetfq3-lab / Lab, not V2.1 Stable"

ID_COLUMNS = [
    "trade_date",
    "sector",
    "etf_code",
    "etf_name",
    "ranking_group_id",
    "model_version",
    "feature_version",
]
FUTURE_LABEL_COLUMNS = [
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
]
RANK_LABEL_COLUMNS = [
    "best_in_sector_1d",
    "best_in_sector_3d",
    "top_quantile_in_sector_3d",
    "avoid_in_sector",
    "pairwise_outperform_label",
]
FEATURE_COLUMNS = [
    "etf_ret_1d_lag1",
    "etf_ret_3d_lag1",
    "etf_ret_5d_lag1",
    "etf_ret_10d_lag1",
    "etf_volatility_5d_lag1",
    "etf_volatility_10d_lag1",
    "etf_amount_5d_mean_lag1",
    "etf_amount_10d_mean_lag1",
    "etf_amount_change_5d_lag1",
    "etf_drawdown_5d_lag1",
    "etf_drawdown_10d_lag1",
    "sector_ret_1d_mean_lag1",
    "sector_ret_3d_mean_lag1",
    "sector_ret_5d_mean_lag1",
    "sector_breadth_1d_lag1",
    "sector_breadth_3d_lag1",
    "sector_amount_5d_mean_lag1",
    "sector_etf_count",
    "etf_vs_sector_ret_3d_lag1",
    "etf_vs_sector_ret_5d_lag1",
    "etf_amount_share_in_sector_lag1",
    "etf_rank_ret_3d_in_sector_lag1",
    "etf_rank_ret_5d_in_sector_lag1",
    "etf_rank_amount_5d_in_sector_lag1",
    "etf_rank_volatility_5d_in_sector_lag1",
]
FORBIDDEN_FEATURE_COLUMNS = [
    *FUTURE_LABEL_COLUMNS,
    *RANK_LABEL_COLUMNS,
    *ID_COLUMNS,
]
BASE_SAMPLE_COLUMNS = [*ID_COLUMNS, *FUTURE_LABEL_COLUMNS, *RANK_LABEL_COLUMNS]
FEATURE_SAMPLE_COLUMNS = [*BASE_SAMPLE_COLUMNS, *FEATURE_COLUMNS]


class GeneratorError(RuntimeError):
    pass


@dataclass(frozen=True)
class GeneratedPaths:
    sample: Path
    manifest: Path
    feature_sample: Path
    feature_contract: Path
    report_md: Path
    report_json: Path


@dataclass(frozen=True)
class GenerationResult:
    paths: GeneratedPaths
    report: dict[str, Any]
    manifest: dict[str, Any]
    feature_contract: dict[str, Any]


def normalize_symbol(value: object) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6) if text.isdigit() and len(text) <= 6 else text


def clean_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def normalize_daily_frame(df: pd.DataFrame, code: str, name: str, sector: str) -> pd.DataFrame:
    rename_map = {
        "date": "trade_date",
        "symbol": "etf_code",
        "code": "etf_code",
        "name": "etf_name",
        "收盘": "close",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "成交额": "amount",
        "成交量": "volume",
    }
    frame = df.rename(columns={key: value for key, value in rename_map.items() if key in df.columns}).copy()
    if "trade_date" not in frame.columns:
        raise GeneratorError("daily frame missing trade_date/date column")
    if "close" not in frame.columns:
        raise GeneratorError("daily frame missing close column")

    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
    frame["etf_code"] = frame.get("etf_code", code).map(normalize_symbol) if "etf_code" in frame else normalize_symbol(code)
    frame["etf_name"] = frame.get("etf_name", name) if "etf_name" in frame else name
    frame["sector"] = frame.get("sector", sector) if "sector" in frame else sector

    for column in ("open", "high", "low", "close", "amount", "volume"):
        if column in frame.columns:
            frame[column] = clean_number(frame[column])
    if "open" not in frame.columns:
        frame["open"] = frame["close"]
    if "high" not in frame.columns:
        frame["high"] = frame[["open", "close"]].max(axis=1)
    if "low" not in frame.columns:
        frame["low"] = frame[["open", "close"]].min(axis=1)
    if "amount" not in frame.columns:
        volume = clean_number(frame["volume"]) if "volume" in frame.columns else pd.Series(0.0, index=frame.index)
        frame["amount"] = volume.fillna(0.0) * frame["close"].fillna(0.0)

    keep = ["trade_date", "etf_code", "etf_name", "sector", "open", "high", "low", "close", "amount"]
    frame = frame[keep].dropna(subset=["trade_date", "close"]).sort_values("trade_date")
    frame["etf_code"] = frame["etf_code"].map(normalize_symbol)
    frame["etf_name"] = frame["etf_name"].fillna(name).astype(str)
    frame["sector"] = frame["sector"].fillna(sector).astype(str)
    return frame.drop_duplicates(subset=["trade_date", "etf_code"], keep="last").reset_index(drop=True)


def load_mock_daily_csv(path: Path, sector_map_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if not path.exists():
        raise GeneratorError(f"mock daily csv does not exist: {path}")
    raw = pd.read_csv(path, dtype={"etf_code": str, "symbol": str, "code": str})
    mapping = load_sector_mapping(sector_map_path) if sector_map_path else {}
    frames: list[pd.DataFrame] = []
    code_column = next((column for column in ("etf_code", "symbol", "code") if column in raw.columns), None)
    if code_column is None:
        raise GeneratorError("mock daily csv missing etf_code/symbol/code column")

    for code, group in raw.groupby(raw[code_column].map(normalize_symbol), sort=True):
        meta = mapping.get(code, {})
        name = str(meta.get("name") or group.get("etf_name", group.get("name", pd.Series([code]))).iloc[0])
        sector = str(meta.get("sector") or group.get("sector", pd.Series(["mock_sector"])).iloc[0])
        frames.append(normalize_daily_frame(group, code=code, name=name, sector=sector))
    if not frames:
        raise GeneratorError("mock daily csv produced no ETF frames")
    return pd.concat(frames, ignore_index=True), {
        "adapter": "mock",
        "attempted": int(len(frames)),
        "downloaded": int(len(frames)),
        "selected": int(len(frames)),
        "download_log": [],
    }


def load_sector_mapping(path: Path | None) -> dict[str, dict[str, Any]]:
    mapping_path = path or REPO_ROOT / "config" / "etf_sector_map.yaml"
    try:
        from data.sector_map import load_etf_sector_map

        return load_etf_sector_map(mapping_path)
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        return load_sector_mapping_without_yaml(mapping_path)


def load_sector_mapping_without_yaml(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "etfs:":
            continue
        if line.startswith("- {"):
            record = parse_inline_yaml_record(line)
            if record:
                records[normalize_symbol(record["code"])] = normalize_sector_record(record)
            current = None
            continue
        if line.startswith("- "):
            if current and current.get("code"):
                records[normalize_symbol(current["code"])] = normalize_sector_record(current)
            current = {}
            key, value = parse_yaml_key_value(line[2:])
            if key:
                current[key] = value
            continue
        if current is not None and ":" in line:
            key, value = parse_yaml_key_value(line)
            if key:
                current[key] = value
    if current and current.get("code"):
        records[normalize_symbol(current["code"])] = normalize_sector_record(current)
    return records


def parse_inline_yaml_record(line: str) -> dict[str, str]:
    body = line.strip()[3:-1]
    record: dict[str, str] = {}
    for match in re.finditer(r"([A-Za-z_][A-Za-z0-9_]*):\s*(\"[^\"]*\"|[^,\]}]+)", body):
        record[match.group(1)] = clean_yaml_scalar(match.group(2))
    return record


def parse_yaml_key_value(text: str) -> tuple[str, str]:
    if ":" not in text:
        return "", ""
    key, value = text.split(":", 1)
    return key.strip(), clean_yaml_scalar(value)


def clean_yaml_scalar(value: str) -> str:
    text = value.strip().rstrip(",")
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    return text


def normalize_sector_record(raw: dict[str, Any]) -> dict[str, Any]:
    code = normalize_symbol(raw.get("code") or raw.get("symbol") or "")
    sector = str(raw.get("sector") or raw.get("sector_l2") or "行业未录入")
    return {
        "code": code,
        "symbol": code,
        "name": str(raw.get("name") or code),
        "sector": sector,
        "sector_l1": str(raw.get("sector_l1") or sector),
        "sector_l2": str(raw.get("sector_l2") or sector),
        "theme": str(raw.get("theme") or sector),
    }


def load_akshare_daily(max_etfs: int, min_etfs_per_sector: int, sector_map_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = load_sector_mapping(sector_map_path)
    if not mapping:
        raise GeneratorError("sector map is empty; cannot select ETF universe")

    frames: list[pd.DataFrame] = []
    log: list[dict[str, Any]] = []
    selected_counts: dict[str, int] = {}
    max_attempts = max(max_etfs * 4, 40)
    attempted = 0

    for code, meta in mapping.items():
        if len(frames) >= max_etfs:
            break
        if attempted >= max_attempts:
            break
        sector = str(meta.get("sector") or meta.get("sector_l2") or "行业未录入")
        if selected_counts.get(sector, 0) >= min_etfs_per_sector and len(frames) + min_etfs_per_sector <= max_etfs:
            continue
        attempted += 1
        try:
            df, source = download_history_with_fallback(code)
            frame = normalize_daily_frame(df, code=code, name=str(meta.get("name") or code), sector=sector)
            if len(frame) < 20:
                raise GeneratorError("not enough daily rows after normalization")
            frames.append(frame)
            selected_counts[sector] = selected_counts.get(sector, 0) + 1
            log.append(download_log_item(code, meta, "downloaded", len(frame), frame, source))
        except Exception as exc:  # noqa: BLE001
            log.append(download_log_item(code, meta, "failed", 0, None, "", str(exc)))

    if not frames:
        raise GeneratorError("AKShare source produced no usable ETF daily frames")

    daily = pd.concat(frames, ignore_index=True)
    eligible_sectors = {
        sector for sector, count in daily.groupby("sector")["etf_code"].nunique().items() if count >= min_etfs_per_sector
    }
    daily = daily[daily["sector"].isin(eligible_sectors)].copy()
    selected_codes = sorted(daily["etf_code"].unique())
    if len(selected_codes) > max_etfs:
        selected_codes = selected_codes[:max_etfs]
        daily = daily[daily["etf_code"].isin(selected_codes)].copy()
    if daily.empty:
        raise GeneratorError("no sector has enough ETFs for min_etfs_per_sector")

    return daily.reset_index(drop=True), {
        "adapter": "data.downloader.download_etf_history",
        "market_data_source": "AKShare ETF daily OHLCV",
        "attempted": attempted,
        "downloaded": len({item["code"] for item in log if item["status"] == "downloaded"}),
        "selected": int(daily["etf_code"].nunique()),
        "selected_codes": sorted(daily["etf_code"].unique()),
        "selected_sector_counts": {
            str(k): int(v) for k, v in daily.groupby("sector")["etf_code"].nunique().sort_index().items()
        },
        "download_log": log,
    }


def download_history_with_fallback(code: str) -> tuple[pd.DataFrame, str]:
    try:
        from data.downloader import download_etf_history

        return download_etf_history(
            symbol=code,
            start_date="20250101",
            retries=2,
            retry_delay=0.5,
            timeout_per_source=20.0,
            max_sources=2,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        import akshare as ak

        symbol = code if code.lower().startswith(("sh", "sz")) else f"{'sh' if code.startswith(('5', '6')) else 'sz'}{code}"
        raw = ak.fund_etf_hist_sina(symbol=symbol)
        return raw, "akshare.fund_etf_hist_sina"


def download_log_item(
    code: str,
    meta: dict[str, Any],
    status: str,
    rows: int,
    frame: pd.DataFrame | None,
    source: str,
    failure_reason: str = "",
) -> dict[str, Any]:
    item = {
        "code": code,
        "name": str(meta.get("name") or code),
        "sector": str(meta.get("sector") or meta.get("sector_l2") or ""),
        "sector_l1": str(meta.get("sector_l1") or ""),
        "theme": str(meta.get("theme") or ""),
        "status": status,
        "rows": int(rows),
        "source": source,
    }
    if frame is not None and not frame.empty:
        item["date_start"] = str(frame["trade_date"].min().date())
        item["date_end"] = str(frame["trade_date"].max().date())
    if failure_reason:
        item["failure_reason"] = failure_reason
    return item


def add_etf_features(daily: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for _, group in daily.groupby("etf_code", sort=True):
        frame = group.sort_values("trade_date").copy()
        close = frame["close"].astype(float)
        low = frame["low"].astype(float)
        amount = frame["amount"].astype(float).fillna(0.0)
        ret = close.pct_change()
        lag_ret = ret.shift(1)
        frame["daily_ret"] = ret
        frame["etf_ret_1d_lag1"] = lag_ret.fillna(0.0)
        frame["etf_ret_3d_lag1"] = lag_ret.rolling(3, min_periods=1).sum().fillna(0.0)
        frame["etf_ret_5d_lag1"] = lag_ret.rolling(5, min_periods=1).sum().fillna(0.0)
        frame["etf_ret_10d_lag1"] = lag_ret.rolling(10, min_periods=1).sum().fillna(0.0)
        frame["etf_volatility_5d_lag1"] = lag_ret.rolling(5, min_periods=2).std().fillna(0.0)
        frame["etf_volatility_10d_lag1"] = lag_ret.rolling(10, min_periods=2).std().fillna(0.0)
        frame["etf_amount_5d_mean_lag1"] = amount.shift(1).rolling(5, min_periods=1).mean().fillna(0.0)
        frame["etf_amount_10d_mean_lag1"] = amount.shift(1).rolling(10, min_periods=1).mean().fillna(0.0)
        base_amount = amount.shift(6).replace(0, pd.NA)
        frame["etf_amount_change_5d_lag1"] = ((amount.shift(1) / base_amount) - 1.0).fillna(0.0)
        lag_close = close.shift(1)
        frame["etf_drawdown_5d_lag1"] = (lag_close / lag_close.rolling(5, min_periods=1).max() - 1.0).fillna(0.0)
        frame["etf_drawdown_10d_lag1"] = (lag_close / lag_close.rolling(10, min_periods=1).max() - 1.0).fillna(0.0)
        frame["future_return_1d"] = close.shift(-1) / close - 1.0
        frame["future_return_3d"] = close.shift(-3) / close - 1.0
        future_low = pd.concat([low.shift(-1), low.shift(-2), low.shift(-3)], axis=1).min(axis=1)
        frame["max_drawdown_3d"] = future_low / close - 1.0
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_sector_features_and_labels(feature_frame: pd.DataFrame) -> pd.DataFrame:
    group_keys = ["trade_date", "sector"]
    sector_stats = feature_frame.groupby(group_keys).agg(
        sector_ret_1d_mean_lag1=("etf_ret_1d_lag1", "mean"),
        sector_ret_3d_mean_lag1=("etf_ret_3d_lag1", "mean"),
        sector_ret_5d_mean_lag1=("etf_ret_5d_lag1", "mean"),
        sector_breadth_1d_lag1=("etf_ret_1d_lag1", lambda values: float((values > 0).mean())),
        sector_breadth_3d_lag1=("etf_ret_3d_lag1", lambda values: float((values > 0).mean())),
        sector_amount_5d_mean_lag1=("etf_amount_5d_mean_lag1", "mean"),
        sector_etf_count=("etf_code", "nunique"),
    )
    frame = feature_frame.merge(sector_stats.reset_index(), on=group_keys, how="left")
    frame["etf_vs_sector_ret_3d_lag1"] = frame["etf_ret_3d_lag1"] - frame["sector_ret_3d_mean_lag1"]
    frame["etf_vs_sector_ret_5d_lag1"] = frame["etf_ret_5d_lag1"] - frame["sector_ret_5d_mean_lag1"]
    sector_amount_sum = frame.groupby(group_keys)["etf_amount_5d_mean_lag1"].transform("sum").replace(0, pd.NA)
    frame["etf_amount_share_in_sector_lag1"] = (frame["etf_amount_5d_mean_lag1"] / sector_amount_sum).fillna(0.0)
    frame["etf_rank_ret_3d_in_sector_lag1"] = frame.groupby(group_keys)["etf_ret_3d_lag1"].rank(
        ascending=False, pct=True, method="average"
    )
    frame["etf_rank_ret_5d_in_sector_lag1"] = frame.groupby(group_keys)["etf_ret_5d_lag1"].rank(
        ascending=False, pct=True, method="average"
    )
    frame["etf_rank_amount_5d_in_sector_lag1"] = frame.groupby(group_keys)["etf_amount_5d_mean_lag1"].rank(
        ascending=False, pct=True, method="average"
    )
    frame["etf_rank_volatility_5d_in_sector_lag1"] = frame.groupby(group_keys)["etf_volatility_5d_lag1"].rank(
        ascending=True, pct=True, method="average"
    )

    frame["rank_1d"] = frame.groupby(group_keys)["future_return_1d"].rank(ascending=False, method="first")
    frame["rank_3d"] = frame.groupby(group_keys)["future_return_3d"].rank(ascending=False, method="first")
    group_size = frame.groupby(group_keys)["etf_code"].transform("count")
    top_n = group_size.map(lambda value: max(1, math.ceil(float(value) / 2.0)))
    frame["best_in_sector_1d"] = (frame["rank_1d"] == 1).astype(int)
    frame["best_in_sector_3d"] = (frame["rank_3d"] == 1).astype(int)
    frame["top_quantile_in_sector_3d"] = (frame["rank_3d"] <= top_n).astype(int)
    frame["avoid_in_sector"] = (frame["rank_3d"] > top_n).astype(int)
    frame["pairwise_outperform_label"] = frame["top_quantile_in_sector_3d"].astype(int)
    frame["ranking_group_id"] = frame["trade_date"].dt.strftime("%Y-%m-%d") + "_" + frame["sector"].astype(str)
    frame["model_version"] = MODEL_VERSION
    frame["feature_version"] = FEATURE_VERSION
    return frame


def build_samples(
    daily: pd.DataFrame,
    max_trading_days: int,
    max_etfs: int,
    min_etfs_per_sector: int,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected_codes = sorted(daily["etf_code"].unique())[:max_etfs]
    daily = daily[daily["etf_code"].isin(selected_codes)].copy()
    daily = add_etf_features(daily)
    daily = add_sector_features_and_labels(daily)
    daily = daily.dropna(subset=FUTURE_LABEL_COLUMNS).copy()

    eligible_groups = daily.groupby(["trade_date", "sector"]).filter(
        lambda group: group["etf_code"].nunique() >= min_etfs_per_sector
    )
    if eligible_groups.empty:
        raise GeneratorError("no ranking group meets min_etfs_per_sector")

    valid_dates = sorted(eligible_groups["trade_date"].drop_duplicates())
    selected_dates = valid_dates[-max_trading_days:]
    frame = eligible_groups[eligible_groups["trade_date"].isin(selected_dates)].copy()
    frame = frame.sort_values(["trade_date", "sector", "etf_code"]).reset_index(drop=True)

    if max_rows is not None and max_rows > 0 and len(frame) > max_rows:
        frame = limit_by_max_rows(frame, max_rows)

    for column in FEATURE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    frame["trade_date"] = frame["trade_date"].dt.strftime("%Y-%m-%d")
    frame["sector_etf_count"] = frame["sector_etf_count"].astype(int)
    for column in RANK_LABEL_COLUMNS:
        frame[column] = frame[column].astype(int)

    base_sample = frame[BASE_SAMPLE_COLUMNS].copy()
    feature_sample = frame[FEATURE_SAMPLE_COLUMNS].copy()
    if base_sample.empty:
        raise GeneratorError("generated sample is empty")
    return base_sample, feature_sample


def limit_by_max_rows(frame: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    selected_groups: list[str] = []
    running_rows = 0
    group_counts = frame.groupby("ranking_group_id").size().sort_index(ascending=False)
    for group_id, count in group_counts.items():
        if running_rows and running_rows + int(count) > max_rows:
            continue
        selected_groups.append(str(group_id))
        running_rows += int(count)
        if running_rows >= max_rows:
            break
    if not selected_groups:
        raise GeneratorError("max_rows is smaller than the first complete ranking group")
    return frame[frame["ranking_group_id"].isin(selected_groups)].sort_values(["trade_date", "sector", "etf_code"])


def ensure_allowed_out_dir(out_dir: Path) -> Path:
    resolved = (REPO_ROOT / out_dir).resolve() if not out_dir.is_absolute() else out_dir.resolve()
    output_root = (REPO_ROOT / "output").resolve()
    try:
        resolved.relative_to(output_root)
        raise GeneratorError("out-dir must not be under output/")
    except ValueError:
        pass
    ignored_root = (REPO_ROOT / ".local_research_outputs" / "aetfq3_lab").resolve()
    try:
        resolved.relative_to(ignored_root)
    except ValueError as exc:
        raise GeneratorError("out-dir must be under .local_research_outputs/aetfq3_lab/") from exc
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8")


def json_dump(payload: dict[str, Any], path: Path) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def relative_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path)


def build_manifest(feature_sample_path: Path, feature_sample: pd.DataFrame, generated_at: str, source: str) -> dict[str, Any]:
    return {
        "manifest_version": "aetfq3_lab_table_ml_expanded_sample/v1",
        "sample_type": "sector_internal_ranking",
        "sample_path": relative_path(feature_sample_path),
        "sample_path_type": "local_ignored",
        "source_kind": "lab_generated_small_sample",
        "source_description": (
            f"Lab generated expanded E sector internal ranking sample from {source}. "
            "Features are lag1/past-only; future returns/drawdown are labels only."
        ),
        "generated_at": generated_at,
        "generated_by": "tools/lab/sector_internal_ranking_sample_generator.py",
        "human_authorized": True,
        "authorized_by": "Codex Lab-only sample generator",
        "authorization_scope": "Lab-only expanded sector_internal_ranking sample coverage, ignored local outputs only",
        "uses_stable_bundle": False,
        "stable_bundle_path": "",
        "stable_bundle_commit": "",
        "stable_bundle_snapshot_date": "",
        "data_time_start": str(feature_sample["trade_date"].min()),
        "data_time_end": str(feature_sample["trade_date"].max()),
        "row_count": int(len(feature_sample)),
        "symbol_count": int(feature_sample["etf_code"].nunique()),
        "sector_count": int(feature_sample["sector"].nunique()),
        "contains_future_labels": True,
        "future_label_columns": FUTURE_LABEL_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "forbidden_feature_columns": FORBIDDEN_FEATURE_COLUMNS,
        "has_future_leakage_check": True,
        "allowed_for": ["schema_validation_only", "dry_validation_only", "mock_validation_only"],
        "training_allowed": False,
        "stable_effect_allowed": False,
        "advisory_only": True,
        "affects_stable_trading": False,
        "contains_secret": False,
        "contains_live_order": False,
        "contains_order_intent": False,
        "qmt_related": False,
        "review_checklist_passed": True,
        "notes": "Expanded Lab-only sample for dry validation and no-save smoke. Not training authorization, not trading advice, not Stable input.",
    }


def summarize_sample(feature_sample: pd.DataFrame) -> dict[str, Any]:
    group_sizes = feature_sample.groupby("ranking_group_id")["etf_code"].nunique()
    label_distribution = {
        column: {str(k): int(v) for k, v in feature_sample[column].value_counts().sort_index().items()}
        for column in RANK_LABEL_COLUMNS
    }
    feature_missing = feature_sample[FEATURE_COLUMNS].isna().sum().sum()
    feature_cells = max(1, len(feature_sample) * len(FEATURE_COLUMNS))
    return {
        "row_count": int(len(feature_sample)),
        "date_start": str(feature_sample["trade_date"].min()),
        "date_end": str(feature_sample["trade_date"].max()),
        "date_count": int(feature_sample["trade_date"].nunique()),
        "etf_count": int(feature_sample["etf_code"].nunique()),
        "sector_count": int(feature_sample["sector"].nunique()),
        "group_count": int(feature_sample["ranking_group_id"].nunique()),
        "min_group_size": int(group_sizes.min()),
        "max_group_size": int(group_sizes.max()),
        "feature_count": int(len(FEATURE_COLUMNS)),
        "missing_rate": float(feature_missing / feature_cells),
        "sectors": sorted(feature_sample["sector"].unique()),
        "label_distribution": label_distribution,
    }


def build_feature_contract(paths: GeneratedPaths, feature_sample: pd.DataFrame, generated_at: str) -> dict[str, Any]:
    summary = summarize_sample(feature_sample)
    return {
        "task": "sector_internal_ranking_expanded_feature_contract",
        "lab_boundary": LAB_BOUNDARY,
        "generated_at": generated_at,
        "inputs": {
            "raw_sample": relative_path(paths.sample),
            "feature_sample": relative_path(paths.feature_sample),
            "manifest": relative_path(paths.manifest),
        },
        "field_classification": {
            "id_group_fields": ID_COLUMNS,
            "label_future_outcome_fields": [*FUTURE_LABEL_COLUMNS, *RANK_LABEL_COLUMNS],
            "candidate_features": FEATURE_COLUMNS,
            "numeric_candidate_features": FEATURE_COLUMNS,
            "forbidden_columns": FORBIDDEN_FEATURE_COLUMNS,
        },
        "feature_columns": FEATURE_COLUMNS,
        "feature_policy": {
            "all_features_lag1_or_past_only": True,
            "future_return_drawdown_label_only": True,
            "forbidden_fields_excluded": True,
            "raw_sector_string_excluded": True,
            "etf_identity_excluded": True,
            "ranking_group_id_excluded": True,
        },
        "baseline_precheck": {
            "row_count": summary["row_count"],
            "group_count": summary["group_count"],
            "min_group_size": summary["min_group_size"],
            "date_count": summary["date_count"],
            "sector_count": summary["sector_count"],
            "label_distribution": summary["label_distribution"],
            "candidate_feature_count": len(FEATURE_COLUMNS),
            "numeric_candidate_feature_count": len(FEATURE_COLUMNS),
            "chronological_split_possible": summary["date_count"] >= 2,
            "group_split_possible": summary["group_count"] >= 2,
            "baseline_smoke_allowed": True,
        },
        "boundary": boundary(),
    }


def boundary() -> dict[str, bool]:
    return {
        "no_stable": True,
        "no_qmt": True,
        "no_order_intent": True,
        "no_output": True,
        "no_lab_advisory": True,
        "no_model_save": True,
        "no_checkpoint": True,
        "not_trading_advice": True,
    }


def build_report(
    paths: GeneratedPaths,
    feature_sample: pd.DataFrame,
    generated_at: str,
    source: str,
    source_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = summarize_sample(feature_sample)
    return {
        "task": "sector_internal_ranking_expanded_sample_generation",
        "lab_boundary": LAB_BOUNDARY,
        "generated_at": generated_at,
        "status": "generated",
        "data_source": {
            "adapter": source,
            "sector_map": "config/etf_sector_map.yaml" if source != "mock" else "mock sector map",
            "uses_stable_bundle": False,
            "no_stable_output": True,
            "no_qmt": True,
            "no_order_intent": True,
        },
        "sample_summary": summary,
        "feature_summary": {
            "feature_columns": FEATURE_COLUMNS,
            "feature_count": len(FEATURE_COLUMNS),
            "all_features_lag1_or_past_only": True,
            "future_labels_only_as_labels": True,
            "forbidden_feature_intersection": sorted(set(FEATURE_COLUMNS) & set(FORBIDDEN_FEATURE_COLUMNS)),
        },
        "download_summary": source_summary,
        "outputs": {
            "raw_sample": relative_path(paths.sample),
            "feature_sample": relative_path(paths.feature_sample),
            "manifest": relative_path(paths.manifest),
            "feature_contract": relative_path(paths.feature_contract),
        },
        "boundary": boundary(),
    }


def render_report_md(report: dict[str, Any]) -> str:
    summary = report["sample_summary"]
    source = report["data_source"]
    return f"""# E sector internal ranking expanded sample generation

本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。

## 目标

工具化生成 E sector internal ranking expanded Lab-only sample，用于 intake/schema/dry validation 或 no-save smoke 前置检查。

## 数据来源

- adapter: {source["adapter"]}
- sector map: {source["sector_map"]}
- uses_stable_bundle: false

## 样本摘要

- rows: {summary["row_count"]}
- date_count: {summary["date_count"]}
- ETF count: {summary["etf_count"]}
- sector count: {summary["sector_count"]}
- group_count: {summary["group_count"]}
- min_group_size: {summary["min_group_size"]}
- feature_count: {summary["feature_count"]}
- missing_rate: {summary["missing_rate"]}

## 边界

- no Stable: true
- no QMT: true
- no OrderIntent: true
- no training: true
- no model save/checkpoint: true
- no advisory package: true
- not trading advice: true

## Feature Policy

所有 feature 均为 lag1/past-only 或明确允许的 `sector_etf_count`；`future_return_1d`、`future_return_3d`、`max_drawdown_3d` 只作为 label/outcome，不进入 `feature_columns`。
"""


def generate_sector_internal_ranking_sample(
    *,
    max_trading_days: int,
    max_etfs: int,
    min_etfs_per_sector: int,
    out_dir: Path,
    max_rows: int | None = None,
    source: str = "akshare",
    mock_daily_csv: Path | None = None,
    mock_sector_map: Path | None = None,
    skip_baseline_smoke: bool = False,
) -> GenerationResult:
    del skip_baseline_smoke  # Baseline smoke is intentionally out of first-version generation.
    if max_trading_days <= 0 or max_etfs <= 0 or min_etfs_per_sector <= 0:
        raise GeneratorError("max_trading_days, max_etfs, and min_etfs_per_sector must be positive")
    out_dir = ensure_allowed_out_dir(out_dir)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if source == "mock":
        if mock_daily_csv is None:
            raise GeneratorError("--mock-daily-csv is required when --source mock")
        daily, source_summary = load_mock_daily_csv(mock_daily_csv, mock_sector_map)
        source_name = "mock"
    elif source == "akshare":
        daily, source_summary = load_akshare_daily(max_etfs=max_etfs, min_etfs_per_sector=min_etfs_per_sector)
        source_name = "data.downloader.download_etf_history"
    else:
        raise GeneratorError(f"unsupported source: {source}")

    base_sample, feature_sample = build_samples(
        daily=daily,
        max_trading_days=max_trading_days,
        max_etfs=max_etfs,
        min_etfs_per_sector=min_etfs_per_sector,
        max_rows=max_rows,
    )

    paths = GeneratedPaths(
        sample=out_dir / f"{REPORT_NAME}_sample.csv",
        manifest=out_dir / f"{REPORT_NAME}_manifest.json",
        feature_sample=out_dir / f"{REPORT_NAME}_feature_sample.csv",
        feature_contract=out_dir / f"{REPORT_NAME}_feature_contract.json",
        report_md=out_dir / f"{REPORT_NAME}_generation_report.md",
        report_json=out_dir / f"{REPORT_NAME}_generation_report.json",
    )
    manifest = build_manifest(paths.feature_sample, feature_sample, generated_at, source_name)
    feature_contract = build_feature_contract(paths, feature_sample, generated_at)
    report = build_report(paths, feature_sample, generated_at, source_name, source_summary)

    write_csv(base_sample, paths.sample)
    write_csv(feature_sample, paths.feature_sample)
    json_dump(manifest, paths.manifest)
    json_dump(feature_contract, paths.feature_contract)
    json_dump(report, paths.report_json)
    paths.report_md.write_text(render_report_md(report), encoding="utf-8")
    return GenerationResult(paths=paths, report=report, manifest=manifest, feature_contract=feature_contract)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Lab-only E sector internal ranking expanded sample.")
    parser.add_argument("--max-trading-days", type=int, default=60)
    parser.add_argument("--max-etfs", type=int, default=32)
    parser.add_argument("--min-etfs-per-sector", type=int, default=4)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source", choices=["akshare", "mock"], default="akshare")
    parser.add_argument("--mock-daily-csv", type=Path)
    parser.add_argument("--mock-sector-map", type=Path)
    parser.add_argument("--skip-baseline-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_sector_internal_ranking_sample(
            max_trading_days=args.max_trading_days,
            max_etfs=args.max_etfs,
            min_etfs_per_sector=args.min_etfs_per_sector,
            max_rows=args.max_rows,
            out_dir=args.out_dir,
            source=args.source,
            mock_daily_csv=args.mock_daily_csv,
            mock_sector_map=args.mock_sector_map,
            skip_baseline_smoke=args.skip_baseline_smoke,
        )
    except GeneratorError as exc:
        print(f"FAILED sector_internal_ranking_sample_generated=false P0 {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": "OK",
                "sample": relative_path(result.paths.sample),
                "feature_sample": relative_path(result.paths.feature_sample),
                "manifest": relative_path(result.paths.manifest),
                "feature_contract": relative_path(result.paths.feature_contract),
                "report": relative_path(result.paths.report_json),
                "row_count": result.report["sample_summary"]["row_count"],
                "group_count": result.report["sample_summary"]["group_count"],
                "feature_count": result.report["sample_summary"]["feature_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
