from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
DEFAULT_ETFS = ["159915", "510050", "510300", "510500", "512100", "588000", "159949", "512880"]
DEFAULT_ARTIFACT_DIR = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_signal_recovery_rolling_oop_pool")
DEFAULT_REPORT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_signal_recovery_rolling_oop_pool_readiness")
ALLOWED_OUTPUT_ROOTS = {".local_artifact_backup", ".local_research_outputs"}
FIVE_M_COLUMNS = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume", "amount", "vwap"]
DAILY_COLUMNS = ["trade_date", "etf_code", "open", "high", "low", "close", "volume", "amount"]
BOUNDARY_FIELDS = {
    "access_mode": "READ_ONLY",
    "final_action_change_allowed": False,
    "contains_live_order": False,
    "contains_secret": False,
    "requires_human_review": True,
    "promotion_gate_required": True,
    "formal_model_evidence": False,
    "stable_promotion_ready": False,
    "formal_training_ready": False,
    "qmt_ready": False,
    "order_intent_ready": False,
    "automatic_promotion_ready": False,
    "metrics_are_effectiveness_evidence": False,
    "model_training_allowed": False,
    "stable_allowed": False,
    "qmt_allowed": False,
    "order_intent_allowed": False,
    "model_saved": False,
    "scaler_saved": False,
    "checkpoint_saved": False,
    "gpu_used": False,
    "torchrun_used": False,
    "qmt_used": False,
    "order_intent_generated": False,
    "stable_affected": False,
    "not_trading_advice": True,
}


class RollingOopPoolCaptureError(RuntimeError):
    pass


@dataclass(frozen=True)
class CaptureConfig:
    etfs: list[str]
    artifact_dir: Path
    report_dir: Path
    sprint_anchor_start: str = "2026-04-09"
    sprint_anchor_end: str = "2026-06-03"
    min_oop_anchors: int = 10
    min_etfs: int = 5
    min_groups: int = 50
    complete_5m_bar_count: int = 48


@dataclass(frozen=True)
class MergeResult:
    merged: pd.DataFrame
    stats: dict[str, Any]


@dataclass(frozen=True)
class ReadinessResult:
    inventory: pd.DataFrame
    anchor_payload: dict[str, Any]
    decision_payload: dict[str, Any]


def prefixed_symbol(code: str) -> str:
    stripped = str(code).strip()
    return ("sz" if stripped.startswith(("15", "16", "18")) else "sh") + stripped


def parse_etfs(value: str) -> list[str]:
    etfs = [item.strip() for item in value.split(",") if item.strip()]
    if not etfs:
        raise RollingOopPoolCaptureError("ETF universe is empty")
    return etfs


def resolve_output_dir(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    repo_root = repo_root.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise RollingOopPoolCaptureError(f"output path must be inside repo: {path}") from exc
    if not relative.parts or relative.parts[0] not in ALLOWED_OUTPUT_ROOTS:
        raise RollingOopPoolCaptureError("output path must be under .local_artifact_backup or .local_research_outputs")
    return resolved


def normalize_rolling_5m_frame(etf_code: str, raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=FIVE_M_COLUMNS)
    if "day" not in raw.columns:
        raise RollingOopPoolCaptureError(f"{etf_code} rolling source missing day column")
    out = pd.DataFrame()
    parsed_dt = pd.to_datetime(raw["day"], errors="coerce")
    out["trade_date"] = parsed_dt.dt.strftime("%Y-%m-%d")
    out["datetime"] = parsed_dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["etf_code"] = str(etf_code)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in raw.columns:
            raise RollingOopPoolCaptureError(f"{etf_code} rolling source missing {column} column")
        out[column] = pd.to_numeric(raw[column], errors="coerce")
    out = out.dropna(subset=["trade_date", "datetime", "open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0)
    out["amount"] = out["amount"].fillna(0)
    out["vwap"] = out.apply(lambda row: float(row["amount"] / row["volume"]) if float(row["volume"]) > 0 else None, axis=1)
    return out[FIVE_M_COLUMNS].sort_values(["etf_code", "datetime"]).reset_index(drop=True)


def normalize_daily_frame(etf_code: str, raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame(columns=DAILY_COLUMNS)
    if "date" not in raw.columns:
        raise RollingOopPoolCaptureError(f"{etf_code} daily source missing date column")
    out = pd.DataFrame()
    parsed_dates = pd.to_datetime(raw["date"], errors="coerce")
    out["trade_date"] = parsed_dates.dt.strftime("%Y-%m-%d")
    out["etf_code"] = str(etf_code)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in raw.columns:
            raise RollingOopPoolCaptureError(f"{etf_code} daily source missing {column} column")
        out[column] = pd.to_numeric(raw[column], errors="coerce")
    out = out.dropna(subset=["trade_date", "open", "high", "low", "close"])
    out["volume"] = out["volume"].fillna(0)
    out["amount"] = out["amount"].fillna(0)
    return out[DAILY_COLUMNS].sort_values(["etf_code", "trade_date"]).reset_index(drop=True)


def load_csv(path: Path, columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=list(columns))
    frame = pd.read_csv(path, dtype={"etf_code": str})
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.NA
    return frame[list(columns)]


def canonicalize_frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in ["trade_date", "datetime", "etf_code"]:
        if column in out.columns:
            out[column] = out[column].astype(str)
    for column in ["open", "high", "low", "close", "volume", "amount", "vwap"]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")
    for column in columns:
        if column not in out.columns:
            out[column] = pd.NA
    return out[list(columns)]


def values_equal(left: Any, right: Any) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=1e-8)
    except (TypeError, ValueError):
        return str(left) == str(right)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
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


def append_only_merge(existing: pd.DataFrame, incoming: pd.DataFrame, key_cols: Sequence[str], columns: Sequence[str]) -> MergeResult:
    existing_clean = canonicalize_frame(existing, columns).drop_duplicates(list(key_cols), keep="first").reset_index(drop=True)
    incoming_clean = canonicalize_frame(incoming, columns).drop_duplicates(list(key_cols), keep="last").reset_index(drop=True)
    existing_index = {tuple(row[column] for column in key_cols): idx for idx, row in existing_clean.iterrows()}
    rows_to_add: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    identical_duplicates = 0

    for _, row in incoming_clean.iterrows():
        key = tuple(row[column] for column in key_cols)
        if key not in existing_index:
            rows_to_add.append(row.to_dict())
            continue
        old = existing_clean.loc[existing_index[key]]
        differing_columns = []
        for column in columns:
            if column in key_cols:
                continue
            if not values_equal(old[column], row[column]):
                differing_columns.append(
                    {
                        "column": column,
                        "existing": json_safe(old[column]),
                        "incoming": json_safe(row[column]),
                    }
                )
        if differing_columns:
            conflicts.append({"key": dict(zip(key_cols, key)), "differing_columns": differing_columns})
        else:
            identical_duplicates += 1

    add_frame = pd.DataFrame(rows_to_add, columns=list(columns))
    merged = pd.concat([existing_clean, add_frame], ignore_index=True) if not add_frame.empty else existing_clean.copy()
    merged = merged.sort_values(list(key_cols)).reset_index(drop=True)
    return MergeResult(
        merged=merged[list(columns)],
        stats={
            "existing_rows_before": int(len(existing_clean)),
            "incoming_rows": int(len(incoming_clean)),
            "added_rows": int(len(add_frame)),
            "identical_duplicate_rows": int(identical_duplicates),
            "conflict_count": int(len(conflicts)),
            "conflict_examples": conflicts[:20],
        },
    )


def compute_inventory(
    five_m_pool: pd.DataFrame,
    daily_pool: pd.DataFrame,
    *,
    sprint_anchor_start: str,
    sprint_anchor_end: str,
    complete_5m_bar_count: int = 48,
) -> pd.DataFrame:
    columns = [
        "trade_date",
        "etf_code",
        "bar_count",
        "complete_5m_bars",
        "daily_exists",
        "t1_daily_date",
        "t1_daily_exists",
        "t3_daily_date",
        "t3_daily_exists",
        "strict_oop_by_date",
        "eligible_strict_oop_etf_anchor",
    ]
    if five_m_pool.empty:
        return pd.DataFrame(columns=columns)

    five_m = canonicalize_frame(five_m_pool, FIVE_M_COLUMNS)
    daily = canonicalize_frame(daily_pool, DAILY_COLUMNS)
    bar_counts = five_m.groupby(["trade_date", "etf_code"], dropna=False).size().reset_index(name="bar_count")
    daily_dates_by_etf = {
        etf: sorted(group["trade_date"].dropna().astype(str).unique().tolist())
        for etf, group in daily.groupby("etf_code")
    }
    rows = []
    for _, row in bar_counts.iterrows():
        trade_date = str(row["trade_date"])
        etf_code = str(row["etf_code"])
        dates = daily_dates_by_etf.get(etf_code, [])
        later_dates = [item for item in dates if item > trade_date]
        t1_date = later_dates[0] if len(later_dates) >= 1 else ""
        t3_date = later_dates[2] if len(later_dates) >= 3 else ""
        strict_oop_by_date = trade_date < sprint_anchor_start or trade_date > sprint_anchor_end
        bar_count = int(row["bar_count"])
        rows.append(
            {
                "trade_date": trade_date,
                "etf_code": etf_code,
                "bar_count": bar_count,
                "complete_5m_bars": bar_count == complete_5m_bar_count,
                "daily_exists": trade_date in dates,
                "t1_daily_date": t1_date,
                "t1_daily_exists": bool(t1_date),
                "t3_daily_date": t3_date,
                "t3_daily_exists": bool(t3_date),
                "strict_oop_by_date": strict_oop_by_date,
                "eligible_strict_oop_etf_anchor": bool(
                    strict_oop_by_date and bar_count == complete_5m_bar_count and trade_date in dates and t1_date and t3_date
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["trade_date", "etf_code"]).reset_index(drop=True)


def build_readiness(
    five_m_pool: pd.DataFrame,
    daily_pool: pd.DataFrame,
    config: CaptureConfig,
    created_at_utc: str,
) -> ReadinessResult:
    inventory = compute_inventory(
        five_m_pool,
        daily_pool,
        sprint_anchor_start=config.sprint_anchor_start,
        sprint_anchor_end=config.sprint_anchor_end,
        complete_5m_bar_count=config.complete_5m_bar_count,
    )
    anchor_rows: list[dict[str, Any]] = []
    if not inventory.empty:
        for trade_date, group in inventory.groupby("trade_date"):
            eligible_etfs = sorted(group.loc[group["eligible_strict_oop_etf_anchor"] == True, "etf_code"].astype(str).tolist())
            complete_etfs = sorted(group.loc[group["complete_5m_bars"] == True, "etf_code"].astype(str).tolist())
            t3_etfs = sorted(group.loc[group["t3_daily_exists"] == True, "etf_code"].astype(str).tolist())
            strict_oop_by_date = bool(group["strict_oop_by_date"].any())
            anchor_rows.append(
                {
                    "anchor_date": str(trade_date),
                    "strict_oop_by_date": strict_oop_by_date,
                    "complete_5m_etf_count": int(len(complete_etfs)),
                    "daily_t3_covered_etf_count": int(len(t3_etfs)),
                    "eligible_etf_count": int(len(eligible_etfs)),
                    "eligible_etfs": eligible_etfs,
                    "eligible_group_count": int(len(eligible_etfs)),
                    "eligible_anchor": bool(strict_oop_by_date and len(eligible_etfs) >= config.min_etfs),
                }
            )

    eligible_anchors = [row for row in anchor_rows if row["eligible_anchor"]]
    eligible_dates = [row["anchor_date"] for row in eligible_anchors]
    eligible_etf_union = sorted({etf for row in eligible_anchors for etf in row["eligible_etfs"]})
    etf_count = len(eligible_etf_union)
    group_count = int(sum(row["eligible_group_count"] for row in eligible_anchors))
    t_daily_passed = bool(eligible_anchors and all(row["daily_t3_covered_etf_count"] >= config.min_etfs for row in eligible_anchors))

    threshold_failures = []
    if len(eligible_anchors) < config.min_oop_anchors:
        threshold_failures.append(f"eligible_oop_anchors {len(eligible_anchors)} < {config.min_oop_anchors}")
    if etf_count < config.min_etfs:
        threshold_failures.append(f"etf_count {etf_count} < {config.min_etfs}")
    if group_count < config.min_groups:
        threshold_failures.append(f"group_count {group_count} < {config.min_groups}")
    if not t_daily_passed:
        threshold_failures.append("T+1/T+3 daily coverage not passed for readiness threshold")

    if five_m_pool.empty or daily_pool.empty:
        decision = "ROLLING_OOP_POOL_BLOCKED_SOURCE_UNAVAILABLE"
        blocker = "One or more public source captures returned no usable pool rows."
    elif len(eligible_anchors) >= config.min_oop_anchors and etf_count >= config.min_etfs and group_count >= config.min_groups and t_daily_passed:
        decision = "ROLLING_OOP_POOL_READY_FOR_FIXED_SHORTLIST_VALIDATION"
        blocker = "none for readiness threshold; validation still requires separate explicit task and human review"
    elif len(eligible_anchors) > 0:
        decision = "ROLLING_OOP_POOL_LIMITED_ACCUMULATING"
        blocker = "Strict OOP anchors exist but fixed-shortlist validation threshold is not met."
    else:
        decision = "ROLLING_OOP_POOL_NO_ELIGIBLE_ANCHORS_YET"
        blocker = "No strict OOP anchor currently satisfies complete 5m bars plus T+1/T+3 daily coverage."

    thresholds = {"eligible_oop_anchors": config.min_oop_anchors, "etf_count": config.min_etfs, "group_count": config.min_groups}
    anchor_payload = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_rolling_oop_anchor_readiness",
        "created_at_utc": created_at_utc,
        "strict_oop_rule": {
            "sprint_anchor_start": config.sprint_anchor_start,
            "sprint_anchor_end": config.sprint_anchor_end,
            "complete_5m_bar_count_required": config.complete_5m_bar_count,
            "requires_t1_daily": True,
            "requires_t3_daily": True,
            "no_stable_bundle": True,
            "no_qmt": True,
        },
        "thresholds": thresholds,
        "anchors": anchor_rows,
        "eligible_oop_anchor_dates": eligible_dates,
        "eligible_oop_anchor_count": int(len(eligible_anchors)),
        "eligible_etf_union": eligible_etf_union,
        "etf_count": int(etf_count),
        "group_count": int(group_count),
        "t_plus_1_t_plus_3_daily_coverage_passed": t_daily_passed,
        **BOUNDARY_FIELDS,
    }
    decision_payload = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_rolling_oop_readiness_decision",
        "created_at_utc": created_at_utc,
        "readiness_decision": decision,
        "eligible_oop_anchor_dates": eligible_dates,
        "eligible_oop_anchor_count": int(len(eligible_anchors)),
        "etf_count": int(etf_count),
        "group_count": int(group_count),
        "thresholds": thresholds,
        "threshold_failure_reasons": threshold_failures,
        "t_plus_1_t_plus_3_daily_coverage_passed": t_daily_passed,
        "fixed_shortlist_validation_allowed_by_this_task": False,
        "blocker": blocker,
        "next_allowed_action": "Continue append-only rolling captures until thresholds are met; validation requires a separate no-save task after human review.",
        **BOUNDARY_FIELDS,
    }
    return ReadinessResult(inventory=inventory, anchor_payload=anchor_payload, decision_payload=decision_payload)


def fetch_rolling_5m(etfs: Sequence[str]) -> tuple[pd.DataFrame, list[dict[str, Any]], str]:
    import akshare as ak

    frames = []
    results = []
    for etf_code in etfs:
        symbol = prefixed_symbol(etf_code)
        try:
            raw = ak.stock_zh_a_minute(symbol=symbol, period="5", adjust="")
            normalized = normalize_rolling_5m_frame(etf_code, raw)
            frames.append(normalized)
            results.append(
                {
                    "etf_code": etf_code,
                    "symbol": symbol,
                    "source": "akshare.stock_zh_a_minute / Sina rolling 5m",
                    "status": "ok",
                    "raw_rows": int(len(raw)),
                    "standardized_rows": int(len(normalized)),
                    "date_min": None if normalized.empty else str(normalized["trade_date"].min()),
                    "date_max": None if normalized.empty else str(normalized["trade_date"].max()),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "etf_code": etf_code,
                    "symbol": symbol,
                    "source": "akshare.stock_zh_a_minute / Sina rolling 5m",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        time.sleep(0.15)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=FIVE_M_COLUMNS)
    return combined, results, getattr(ak, "__version__", "unknown")


def fetch_daily_ohlcv(etfs: Sequence[str], start_date: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    from data.downloader import download_etf_history

    frames = []
    results = []
    end_date = datetime.now().strftime("%Y%m%d")
    for etf_code in etfs:
        try:
            raw, source = download_etf_history(
                etf_code,
                start_date=start_date,
                end_date=end_date,
                retries=1,
                retry_delay=0.2,
                timeout_per_source=12,
                max_sources=1,
            )
            normalized = normalize_daily_frame(etf_code, raw)
            frames.append(normalized)
            results.append(
                {
                    "etf_code": etf_code,
                    "source": source,
                    "status": "ok",
                    "raw_rows": int(len(raw)),
                    "standardized_rows": int(len(normalized)),
                    "date_min": None if normalized.empty else str(normalized["trade_date"].min()),
                    "date_max": None if normalized.empty else str(normalized["trade_date"].max()),
                }
            )
        except Exception as exc:
            results.append(
                {
                    "etf_code": etf_code,
                    "source": "data.downloader.download_etf_history(max_sources=1)",
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        time.sleep(0.15)
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DAILY_COLUMNS)
    return combined, results


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_source_note(path: Path, config: CaptureConfig, created_at_utc: str, akshare_version: str) -> None:
    path.write_text(
        "\n".join(
            [
                "# Rolling OOP Pool Source Note",
                "",
                LAB_DECLARATION,
                "",
                f"- Created/updated UTC: {created_at_utc}",
                f"- AKShare version: `{akshare_version}`",
                "- Minute source: AKShare `stock_zh_a_minute`, Sina rolling 5m visible window.",
                "- Daily source: `data.downloader.download_etf_history()` with public AKShare daily OHLCV.",
                f"- ETF universe: `{', '.join(config.etfs)}`",
                "- 5m key: `etf_code + datetime`; daily key: `etf_code + trade_date`.",
                "- Append-only policy: existing rows are preserved; conflicts are recorded; old rows are kept.",
                f"- Sprint anchor overlap range: `{config.sprint_anchor_start}` to `{config.sprint_anchor_end}`.",
                "- Boundary: no Stable bundle, no QMT, no labels, no validation, no model, no training, no OrderIntent.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def write_outputs(
    config: CaptureConfig,
    created_at_utc: str,
    akshare_version: str,
    minute_results: list[dict[str, Any]],
    daily_results: list[dict[str, Any]],
    merge_5m: MergeResult,
    merge_daily: MergeResult,
    readiness: ReadinessResult,
) -> dict[str, Any]:
    artifact_dir = resolve_output_dir(config.artifact_dir)
    report_dir = resolve_output_dir(config.report_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    five_m_path = artifact_dir / "rolling_oop_5m_pool.csv"
    daily_path = artifact_dir / "rolling_oop_daily_ohlcv_pool.csv"
    inventory_path = artifact_dir / "POOL_INVENTORY.csv"
    source_note_path = artifact_dir / "source_note.md"
    manifest_path = artifact_dir / "POOL_MANIFEST.json"
    sha_path = artifact_dir / "SHA256SUMS.txt"
    report_md_path = report_dir / "rolling_oop_capture_report.md"
    report_json_path = report_dir / "rolling_oop_capture_report.json"
    anchor_json_path = report_dir / "rolling_oop_anchor_readiness.json"
    decision_json_path = report_dir / "rolling_oop_readiness_decision.json"

    write_csv(five_m_path, merge_5m.merged)
    write_csv(daily_path, merge_daily.merged)
    write_csv(inventory_path, readiness.inventory)
    write_source_note(source_note_path, config, created_at_utc, akshare_version)

    inventory_summary = {
        "five_m_rows": int(len(merge_5m.merged)),
        "daily_rows": int(len(merge_daily.merged)),
        "inventory_rows": int(len(readiness.inventory)),
        "five_m_date_min": None if merge_5m.merged.empty else str(merge_5m.merged["trade_date"].min()),
        "five_m_date_max": None if merge_5m.merged.empty else str(merge_5m.merged["trade_date"].max()),
        "daily_date_min": None if merge_daily.merged.empty else str(merge_daily.merged["trade_date"].min()),
        "daily_date_max": None if merge_daily.merged.empty else str(merge_daily.merged["trade_date"].max()),
        "etf_codes_with_5m_rows": sorted(merge_5m.merged["etf_code"].dropna().astype(str).unique().tolist())
        if not merge_5m.merged.empty
        else [],
        "etf_codes_with_daily_rows": sorted(merge_daily.merged["etf_code"].dropna().astype(str).unique().tolist())
        if not merge_daily.merged.empty
        else [],
    }
    report_payload = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_rolling_oop_capture_report",
        "created_at_utc": created_at_utc,
        "akshare_version": akshare_version,
        "python_executable": sys.executable,
        "task_scope": "rolling 5m capture, append-only pool, dedup/hash/inventory/OOP anchor readiness only",
        "minute_capture_results": minute_results,
        "daily_capture_results": daily_results,
        "merge_stats": {
            "rolling_oop_5m_pool": merge_5m.stats,
            "rolling_oop_daily_ohlcv_pool": merge_daily.stats,
        },
        "pool_inventory_summary": inventory_summary,
        "readiness_decision": readiness.decision_payload,
        **BOUNDARY_FIELDS,
    }
    report_md = "\n".join(
        [
            "# Rolling OOP Capture Report",
            "",
            LAB_DECLARATION,
            "",
            "## Scope",
            "Rolling Sina 5m capture / append-only pool / hash / inventory / OOP anchor readiness only. No OOP validation, no labels, no model, no training, no QMT, no OrderIntent, no Stable modification.",
            "",
            "## Capture Summary",
            f"- AKShare version: `{akshare_version}`",
            f"- ETF universe: `{', '.join(config.etfs)}`",
            f"- 5m pool rows after merge: `{inventory_summary['five_m_rows']}`",
            f"- Daily pool rows after merge: `{inventory_summary['daily_rows']}`",
            f"- 5m conflicts: `{merge_5m.stats['conflict_count']}`",
            f"- Daily conflicts: `{merge_daily.stats['conflict_count']}`",
            "",
            "## Strict OOP Readiness",
            f"- Eligible anchor dates: `{', '.join(readiness.decision_payload['eligible_oop_anchor_dates']) or 'none'}`",
            f"- Eligible anchor count: `{readiness.decision_payload['eligible_oop_anchor_count']}`",
            f"- ETF count: `{readiness.decision_payload['etf_count']}`",
            f"- Group count: `{readiness.decision_payload['group_count']}`",
            f"- Decision: `{readiness.decision_payload['readiness_decision']}`",
            "",
            "This is read-only Lab data-pool readiness evidence only; it is not model effectiveness evidence, trading advice, or Stable promotion evidence.",
            "",
        ]
    )
    write_json(anchor_json_path, readiness.anchor_payload)
    write_json(decision_json_path, readiness.decision_payload)
    write_json(report_json_path, report_payload)
    report_md_path.write_text(report_md, encoding="utf-8")

    manifest_payload = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_rolling_oop_pool_manifest",
        "created_at_utc": created_at_utc,
        "pool_dir": str(artifact_dir),
        "source_contract": {
            "minute_source": "AKShare stock_zh_a_minute / Sina rolling 5m",
            "daily_source": "data.downloader.download_etf_history / AKShare public daily OHLCV",
            "minute_key": ["etf_code", "datetime"],
            "daily_key": ["etf_code", "trade_date"],
            "append_only": True,
            "conflict_policy": "keep_existing_record_conflict_in_manifest_and_report",
        },
        "merge_stats": {
            "rolling_oop_5m_pool": merge_5m.stats,
            "rolling_oop_daily_ohlcv_pool": merge_daily.stats,
        },
        "readiness_decision": readiness.decision_payload["readiness_decision"],
        "eligible_oop_anchor_dates": readiness.decision_payload["eligible_oop_anchor_dates"],
        "eligible_oop_anchor_count": readiness.decision_payload["eligible_oop_anchor_count"],
        "etf_count": readiness.decision_payload["etf_count"],
        "group_count": readiness.decision_payload["group_count"],
        "hash_note": "SHA256SUMS.txt includes the final POOL_MANIFEST.json hash; manifest excludes self-hash and SHA256SUMS self-hash to avoid recursive hash churn.",
        **BOUNDARY_FIELDS,
    }
    tracked_artifacts = [five_m_path, daily_path, inventory_path, source_note_path]
    manifest_payload["files"] = {path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in tracked_artifacts}
    write_json(manifest_path, manifest_payload)
    sha_targets = tracked_artifacts + [manifest_path]
    sha_path.write_text("\n".join(f"{sha256_file(path)}  {path.name}" for path in sha_targets) + "\n", encoding="utf-8")

    return {
        "artifact_dir": str(artifact_dir),
        "report_dir": str(report_dir),
        "manifest_path": str(manifest_path),
        "report_json_path": str(report_json_path),
        "anchor_json_path": str(anchor_json_path),
        "decision_json_path": str(decision_json_path),
        "summary": {
            **inventory_summary,
            "readiness_decision": readiness.decision_payload["readiness_decision"],
            "eligible_oop_anchor_dates": readiness.decision_payload["eligible_oop_anchor_dates"],
            "eligible_oop_anchor_count": readiness.decision_payload["eligible_oop_anchor_count"],
            "etf_count": readiness.decision_payload["etf_count"],
            "group_count": readiness.decision_payload["group_count"],
            "five_m_conflicts": merge_5m.stats["conflict_count"],
            "daily_conflicts": merge_daily.stats["conflict_count"],
        },
    }


def run_capture(config: CaptureConfig) -> dict[str, Any]:
    artifact_dir = resolve_output_dir(config.artifact_dir)
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    five_m_incoming, minute_results, akshare_version = fetch_rolling_5m(config.etfs)
    five_m_start = "20260601" if five_m_incoming.empty else str(five_m_incoming["trade_date"].min()).replace("-", "")
    daily_incoming, daily_results = fetch_daily_ohlcv(config.etfs, five_m_start)

    existing_5m = load_csv(artifact_dir / "rolling_oop_5m_pool.csv", FIVE_M_COLUMNS)
    existing_daily = load_csv(artifact_dir / "rolling_oop_daily_ohlcv_pool.csv", DAILY_COLUMNS)
    merge_5m = append_only_merge(existing_5m, five_m_incoming, ["etf_code", "datetime"], FIVE_M_COLUMNS)
    merge_daily = append_only_merge(existing_daily, daily_incoming, ["etf_code", "trade_date"], DAILY_COLUMNS)
    readiness = build_readiness(merge_5m.merged, merge_daily.merged, config, created_at_utc)
    return write_outputs(config, created_at_utc, akshare_version, minute_results, daily_results, merge_5m, merge_daily, readiness)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab-only rolling OOP 5m pool capture tool")
    parser.add_argument("--out-artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--etfs", default=",".join(DEFAULT_ETFS))
    parser.add_argument("--sprint-anchor-start", default="2026-04-09")
    parser.add_argument("--sprint-anchor-end", default="2026-06-03")
    parser.add_argument("--min-oop-anchors", type=int, default=10)
    parser.add_argument("--min-etfs", type=int, default=5)
    parser.add_argument("--min-groups", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    config = CaptureConfig(
        etfs=parse_etfs(args.etfs),
        artifact_dir=args.out_artifact_dir,
        report_dir=args.out_report_dir,
        sprint_anchor_start=args.sprint_anchor_start,
        sprint_anchor_end=args.sprint_anchor_end,
        min_oop_anchors=args.min_oop_anchors,
        min_etfs=args.min_etfs,
        min_groups=args.min_groups,
    )
    result = run_capture(config)
    print(json.dumps(json_safe(result["summary"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
