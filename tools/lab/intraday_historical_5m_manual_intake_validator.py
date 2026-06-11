from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
DEFAULT_INBOX = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox")
DEFAULT_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_manual_intake")
ALLOWED_INBOX_ROOT = ".local_artifact_backup"
ALLOWED_OUT_ROOT = ".local_research_outputs"
ALLOWED_DATA_SUFFIXES = {".csv", ".zip", ".parquet"}
STANDARD_COLUMNS = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume", "amount", "vwap"]
REQUIRED_STANDARD_COLUMNS = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume"]
SOURCE_NOTE_FIELDS = [
    "source_name",
    "source_type",
    "export_method",
    "exported_at",
    "date_range",
    "etf_universe",
    "frequency",
    "whether_qmt_export",
    "whether_account_related",
    "whether_order_related",
    "whether_contains_trades_or_fills",
    "whether_contains_secret",
    "whether_stable_bundle",
    "human_authorized",
]
MANIFEST_REQUIRED_VALUES = {
    "sample_type": "intraday_5m_historical_manual_export",
    "frequency": "5m",
    "training_allowed": False,
    "stable_effect_allowed": False,
    "contains_secret": False,
    "contains_order_intent": False,
    "contains_live_order": False,
    "contains_account": False,
    "contains_position": False,
    "contains_order": False,
    "contains_trade": False,
    "human_authorized": True,
}
MANIFEST_REQUIRED_FIELDS = [
    "sample_type",
    "frequency",
    "source_kind",
    "source_note_path",
    "sha256_file",
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
    "qmt_mode",
    "human_authorized",
    "allowed_for",
]
FORBIDDEN_FIELD_NAMES = {
    "account",
    "position",
    "order",
    "trade",
    "fill",
    "orderintent",
    "target_weight",
    "final_buy_action",
}
SECRET_FIELD_TOKENS = {"secret", "token", "password", "passwd", "apikey", "api_key", "private_key"}
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
    "not_trading_advice": True,
}


class ManualIntakeValidatorError(RuntimeError):
    pass


@dataclass(frozen=True)
class IntakeConfig:
    inbox: Path
    out_dir: Path
    sprint_anchor_start: str = "2026-04-09"
    sprint_anchor_end: str = "2026-06-03"
    min_oop_anchors: int = 10
    min_etfs: int = 5
    min_groups: int = 50


def resolve_local_path(path: Path, expected_root: str, repo_root: Path = REPO_ROOT) -> Path:
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    repo_root = repo_root.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ManualIntakeValidatorError(f"path must be inside repo: {path}") from exc
    if not relative.parts or relative.parts[0] != expected_root:
        raise ManualIntakeValidatorError(f"path must be under {expected_root}: {path}")
    return resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(json_safe(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_waiting_outputs(config: IntakeConfig, out_dir: Path, created_at_utc: str) -> dict[str, Any]:
    inventory = pd.DataFrame(columns=["path", "file_name", "suffix", "bytes", "sha256", "role", "allowed_type"])
    inventory.to_csv(out_dir / "manual_intake_inventory.csv", index=False, lineterminator="\n")
    schema = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_manual_intake_schema_report",
        "created_at_utc": created_at_utc,
        "schema_passed": False,
        "blocker": "manual inbox not found",
        **BOUNDARY_FIELDS,
    }
    quality = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_manual_intake_data_quality_report",
        "created_at_utc": created_at_utc,
        "data_quality_passed": False,
        "blocker": "manual inbox not found",
        **BOUNDARY_FIELDS,
    }
    readiness = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_manual_intake_oop_readiness",
        "created_at_utc": created_at_utc,
        "strict_oop_anchor_count": 0,
        "strict_oop_anchor_dates": [],
        "etf_count": 0,
        "group_count": 0,
        "readiness_decision": "MANUAL_HISTORICAL_5M_PACKAGE_NOT_FOUND_WAITING_FOR_INPUT",
        "fixed_shortlist_oop_validation_ready": False,
        **BOUNDARY_FIELDS,
    }
    decision = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_manual_intake_readiness_decision",
        "created_at_utc": created_at_utc,
        "readiness_decision": "MANUAL_HISTORICAL_5M_PACKAGE_NOT_FOUND_WAITING_FOR_INPUT",
        "inbox": str(config.inbox),
        "blocker": "Manual/export historical 5m package not found; waiting for ignored inbox input.",
        "next_allowed_action": "Place a legal CSV/ZIP/parquet package plus source_note.md, SHA256SUMS.txt, and MANIFEST.json into the ignored manual inbox, then rerun this validator.",
        "fixed_shortlist_oop_validation_ready": False,
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "manual_intake_schema_report.json", schema)
    write_json(out_dir / "manual_intake_data_quality_report.json", quality)
    write_json(out_dir / "manual_intake_oop_readiness.json", readiness)
    write_json(out_dir / "manual_intake_readiness_decision.json", decision)
    write_report_md(out_dir / "manual_intake_report.md", decision, readiness, schema, quality)
    return decision


def inventory_inbox(inbox: Path) -> tuple[pd.DataFrame, dict[str, Path], list[Path]]:
    rows = []
    roles: dict[str, Path] = {}
    data_files: list[Path] = []
    for path in sorted(item for item in inbox.iterdir() if item.is_file()):
        suffix = path.suffix.lower()
        role = "ignored"
        if path.name == "source_note.md":
            role = "source_note"
            roles["source_note"] = path
        elif path.name == "SHA256SUMS.txt":
            role = "sha256sums"
            roles["sha256sums"] = path
        elif path.name == "MANIFEST.json":
            role = "manifest"
            roles["manifest"] = path
        elif suffix in ALLOWED_DATA_SUFFIXES:
            role = "data"
            data_files.append(path)
        rows.append(
            {
                "path": str(path),
                "file_name": path.name,
                "suffix": suffix,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
                "allowed_type": role != "ignored",
            }
        )
    return pd.DataFrame(rows), roles, data_files


def parse_sha256sums(path: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            hashes[parts[-1]] = parts[0].lower()
    return hashes


def validate_hashes(inbox: Path, roles: dict[str, Path], inventory: pd.DataFrame) -> tuple[bool, list[str]]:
    if "sha256sums" not in roles:
        return False, ["BLOCKED_MISSING_HASH: SHA256SUMS.txt is missing"]
    expected = parse_sha256sums(roles["sha256sums"])
    errors = []
    for _, row in inventory.iterrows():
        file_name = str(row["file_name"])
        if file_name == "SHA256SUMS.txt":
            continue
        if file_name not in expected:
            errors.append(f"BLOCKED_MISSING_HASH: {file_name} not listed in SHA256SUMS.txt")
            continue
        if str(row["sha256"]).lower() != expected[file_name].lower():
            errors.append(f"BLOCKED_HASH_MISMATCH: {file_name}")
    return not errors, errors


def load_manifest(roles: dict[str, Path]) -> tuple[dict[str, Any] | None, list[str]]:
    if "manifest" not in roles:
        return None, ["BLOCKED_UNAUTHORIZED_SOURCE: MANIFEST.json is missing"]
    try:
        return json.loads(roles["manifest"].read_text(encoding="utf-8")), []
    except json.JSONDecodeError as exc:
        return None, [f"BLOCKED_UNAUTHORIZED_SOURCE: MANIFEST.json invalid JSON: {exc}"]


def validate_manifest(manifest: dict[str, Any] | None) -> tuple[bool, list[str]]:
    if manifest is None:
        return False, ["MANIFEST.json is missing"]
    errors = []
    for field in MANIFEST_REQUIRED_FIELDS:
        if field not in manifest:
            errors.append(f"missing manifest field: {field}")
    for field, expected in MANIFEST_REQUIRED_VALUES.items():
        if manifest.get(field) != expected:
            errors.append(f"manifest {field} must be {expected!r}")
    if manifest.get("qmt_related") is True and manifest.get("qmt_mode") != "export_only":
        errors.append("qmt_related=true is only allowed with qmt_mode=export_only")
    if manifest.get("qmt_mode") not in {"export_only", "read_only", "not_qmt"}:
        errors.append("qmt_mode must be export_only/read_only/not_qmt")
    allowed_for = manifest.get("allowed_for")
    required_allowed = {"intake", "schema_validation", "oop_readiness"}
    if not isinstance(allowed_for, list) or not required_allowed.issubset(set(allowed_for)):
        errors.append("allowed_for must include intake, schema_validation, oop_readiness")
    return not errors, errors


def validate_source_note(roles: dict[str, Path]) -> tuple[bool, list[str], dict[str, str]]:
    if "source_note" not in roles:
        return False, ["BLOCKED_MISSING_SOURCE_NOTE: source_note.md is missing"], {}
    text = roles["source_note"].read_text(encoding="utf-8")
    parsed: dict[str, str] = {}
    for field in SOURCE_NOTE_FIELDS:
        match = re.search(rf"(?im)^\s*-?\s*{re.escape(field)}\s*[:=]\s*(.+?)\s*$", text)
        if match:
            parsed[field] = match.group(1).strip()
    missing = [field for field in SOURCE_NOTE_FIELDS if field not in parsed]
    errors = [f"source_note missing field: {field}" for field in missing]
    if parsed.get("frequency", "").lower() != "5m":
        errors.append("source_note frequency must be 5m")
    if parsed.get("human_authorized", "").lower() != "true":
        errors.append("source_note human_authorized must be true")
    for field in ["whether_account_related", "whether_order_related", "whether_contains_trades_or_fills", "whether_contains_secret", "whether_stable_bundle"]:
        if parsed.get(field, "").lower() == "true":
            errors.append(f"source_note {field} must be false")
    return not errors, errors, parsed


def is_forbidden_field_name(column: str) -> bool:
    lower = column.strip().lower()
    if lower == "trade_date":
        return False
    compact = re.sub(r"[^a-z0-9_]", "", lower)
    if compact in FORBIDDEN_FIELD_NAMES or compact in SECRET_FIELD_TOKENS:
        return True
    for token in FORBIDDEN_FIELD_NAMES | SECRET_FIELD_TOKENS:
        if compact.startswith(token + "_") or compact.endswith("_" + token) or f"_{token}_" in compact:
            return True
    return False


def detect_forbidden_fields(columns: Sequence[str]) -> list[str]:
    return sorted(column for column in columns if is_forbidden_field_name(str(column)))


def load_data_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype={"etf_code": str, "symbol": str, "code": str})
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".zip":
        frames = []
        with zipfile.ZipFile(path) as zf:
            for name in zf.namelist():
                lower = name.lower()
                if lower.endswith(".csv"):
                    with zf.open(name) as handle:
                        frames.append(pd.read_csv(handle, dtype={"etf_code": str, "symbol": str, "code": str}))
                elif lower.endswith(".parquet"):
                    with zf.open(name) as handle:
                        frames.append(pd.read_parquet(handle))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    raise ManualIntakeValidatorError(f"unsupported data file type: {path.name}")


def column_map_for(frame: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "trade_date": ["trade_date", "date", "交易日期"],
        "datetime": ["datetime", "time", "时间", "bar_time"],
        "etf_code": ["etf_code", "symbol", "code", "证券代码"],
        "open": ["open", "开盘"],
        "high": ["high", "最高"],
        "low": ["low", "最低"],
        "close": ["close", "收盘"],
        "volume": ["volume", "成交量"],
        "amount": ["amount", "成交额"],
        "vwap": ["vwap", "均价"],
    }
    normalized = {str(column).strip().lower(): str(column) for column in frame.columns}
    mapping = {}
    for standard, names in aliases.items():
        for name in names:
            found = normalized.get(name.lower())
            if found is not None:
                mapping[standard] = found
                break
    return mapping


def standardize_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = column_map_for(frame)
    missing = [column for column in REQUIRED_STANDARD_COLUMNS if column not in mapping]
    report = {"column_mapping": mapping, "missing_required_columns": missing}
    if missing:
        return pd.DataFrame(columns=STANDARD_COLUMNS), report
    out = pd.DataFrame()
    dt = pd.to_datetime(frame[mapping["datetime"]], errors="coerce")
    if "trade_date" in mapping:
        trade_dates = pd.to_datetime(frame[mapping["trade_date"]], errors="coerce")
        out["trade_date"] = trade_dates.dt.strftime("%Y-%m-%d")
    else:
        out["trade_date"] = dt.dt.strftime("%Y-%m-%d")
    out["datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    out["etf_code"] = frame[mapping["etf_code"]].astype(str).str.extract(r"(\d{6})", expand=False).fillna(frame[mapping["etf_code"]].astype(str))
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(frame[mapping[column]], errors="coerce")
    out["amount"] = pd.to_numeric(frame[mapping["amount"]], errors="coerce") if "amount" in mapping else pd.NA
    if "vwap" in mapping:
        out["vwap"] = pd.to_numeric(frame[mapping["vwap"]], errors="coerce")
    else:
        out["vwap"] = pd.NA
        if "amount" in mapping:
            volume = pd.to_numeric(out["volume"], errors="coerce")
            amount = pd.to_numeric(out["amount"], errors="coerce")
            out.loc[volume > 0, "vwap"] = amount[volume > 0] / volume[volume > 0]
    out = out.dropna(subset=["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume"])
    return out[STANDARD_COLUMNS], report


def validate_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "data_quality_passed": False,
            "rows": 0,
            "duplicate_bars": 0,
            "datetime_monotonic_by_etf": False,
            "ohlc_consistency_passed": False,
            "volume_nonnegative_passed": False,
            "amount_nonnegative_passed": None,
            "bars_per_etf_day": {},
            "p0_blockers": ["input data has no valid standardized rows"],
        }
    duplicate_count = int(frame.duplicated(["etf_code", "datetime"]).sum())
    monotonic = True
    for _, group in frame.sort_values(["etf_code", "datetime"]).groupby("etf_code"):
        dates = pd.to_datetime(group["datetime"], errors="coerce")
        if not dates.is_monotonic_increasing:
            monotonic = False
            break
    ohlc_passed = bool(
        (
            (frame["high"] >= frame[["open", "close", "low"]].max(axis=1))
            & (frame["low"] <= frame[["open", "close", "high"]].min(axis=1))
        ).all()
    )
    volume_passed = bool((pd.to_numeric(frame["volume"], errors="coerce") >= 0).all())
    amount_nonnull = pd.to_numeric(frame["amount"], errors="coerce").dropna()
    amount_passed = None if amount_nonnull.empty else bool((amount_nonnull >= 0).all())
    bar_counts = frame.groupby(["trade_date", "etf_code"]).size().to_dict()
    bars_per_day = {f"{date}|{etf}": int(count) for (date, etf), count in bar_counts.items()}
    blockers = []
    if duplicate_count:
        blockers.append(f"duplicate bars detected: {duplicate_count}")
    if not monotonic:
        blockers.append("datetime not monotonic by ETF")
    if not ohlc_passed:
        blockers.append("OHLC consistency failed")
    if not volume_passed:
        blockers.append("volume nonnegative check failed")
    if amount_passed is False:
        blockers.append("amount nonnegative check failed")
    return {
        "data_quality_passed": not blockers,
        "rows": int(len(frame)),
        "duplicate_bars": duplicate_count,
        "datetime_monotonic_by_etf": monotonic,
        "ohlc_consistency_passed": ohlc_passed,
        "volume_nonnegative_passed": volume_passed,
        "amount_nonnegative_passed": amount_passed,
        "bars_per_etf_day": bars_per_day,
        "min_bars_per_etf_day": int(min(bar_counts.values())) if bar_counts else 0,
        "max_bars_per_etf_day": int(max(bar_counts.values())) if bar_counts else 0,
        "p0_blockers": blockers,
    }


def compute_oop_readiness(frame: pd.DataFrame, config: IntakeConfig, data_quality_passed: bool, source_authorized: bool, no_forbidden_fields: bool) -> dict[str, Any]:
    complete = frame.groupby(["trade_date", "etf_code"]).size().reset_index(name="bar_count") if not frame.empty else pd.DataFrame(columns=["trade_date", "etf_code", "bar_count"])
    complete = complete[complete["bar_count"] >= 1]
    strict = complete[(complete["trade_date"] < config.sprint_anchor_start) | (complete["trade_date"] > config.sprint_anchor_end)]
    anchor_rows = []
    for trade_date, group in strict.groupby("trade_date"):
        etfs = sorted(group["etf_code"].astype(str).unique().tolist())
        anchor_rows.append({"anchor_date": str(trade_date), "eligible_etfs": etfs, "eligible_etf_count": len(etfs), "eligible_anchor": len(etfs) >= config.min_etfs})
    eligible = [row for row in anchor_rows if row["eligible_anchor"]]
    eligible_dates = [row["anchor_date"] for row in eligible]
    etf_union = sorted({etf for row in eligible for etf in row["eligible_etfs"]})
    group_count = int(sum(row["eligible_etf_count"] for row in eligible))
    threshold_passed = len(eligible) >= config.min_oop_anchors and len(etf_union) >= config.min_etfs and group_count >= config.min_groups
    fixed_ready = bool(threshold_passed and data_quality_passed and source_authorized and no_forbidden_fields)
    if fixed_ready:
        decision = "MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION"
    elif len(eligible) > 0 and data_quality_passed and source_authorized and no_forbidden_fields:
        decision = "MANUAL_HISTORICAL_5M_PACKAGE_LIMITED_REVIEW_REQUIRED"
    elif not threshold_passed and data_quality_passed and source_authorized and no_forbidden_fields:
        decision = "BLOCKED_INSUFFICIENT_OOP_ANCHORS"
    else:
        decision = "BLOCKED_DATA_QUALITY"
    return {
        "strict_oop_anchor_count": int(len(eligible)),
        "strict_oop_anchor_dates": eligible_dates,
        "anchor_rows": anchor_rows,
        "etf_count": int(len(etf_union)),
        "group_count": group_count,
        "pre_sprint_oop_count": int(sum(1 for row in eligible if row["anchor_date"] < config.sprint_anchor_start)),
        "post_sprint_oop_count": int(sum(1 for row in eligible if row["anchor_date"] > config.sprint_anchor_end)),
        "t_plus_1_t_plus_3_daily_coverage": "not_checked_no_daily_data_in_manual_package",
        "t_plus_1_t_plus_3_daily_coverage_passed": None,
        "thresholds": {"strict_oop_anchors": config.min_oop_anchors, "etf_count": config.min_etfs, "group_count": config.min_groups},
        "readiness_decision": decision,
        "fixed_shortlist_oop_validation_ready": fixed_ready,
    }


def final_decision(
    *,
    missing_source_note: bool,
    missing_hash: bool,
    hash_mismatch: bool,
    forbidden_fields: list[str],
    manifest_authorized: bool,
    source_note_authorized: bool,
    data_quality_passed: bool,
    oop_decision: str,
) -> str:
    if missing_source_note:
        return "BLOCKED_MISSING_SOURCE_NOTE"
    if missing_hash:
        return "BLOCKED_MISSING_HASH"
    if hash_mismatch:
        return "BLOCKED_HASH_MISMATCH"
    if forbidden_fields:
        return "BLOCKED_FORBIDDEN_FIELDS"
    if not manifest_authorized or not source_note_authorized:
        return "BLOCKED_UNAUTHORIZED_SOURCE"
    if not data_quality_passed:
        return "BLOCKED_DATA_QUALITY"
    return oop_decision


def write_report_md(path: Path, decision: dict[str, Any], readiness: dict[str, Any], schema: dict[str, Any], quality: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Historical 5m Manual Intake Report",
                "",
                LAB_DECLARATION,
                "",
                "## Decision",
                f"- readiness_decision: `{decision.get('readiness_decision')}`",
                f"- fixed_shortlist_oop_validation_ready: `{decision.get('fixed_shortlist_oop_validation_ready', False)}`",
                "",
                "## Schema",
                f"- schema_passed: `{schema.get('schema_passed', False)}`",
                f"- rows_checked: `{schema.get('rows_checked', 0)}`",
                "",
                "## Data Quality",
                f"- data_quality_passed: `{quality.get('data_quality_passed', False)}`",
                f"- duplicate_bars: `{quality.get('duplicate_bars', 0)}`",
                "",
                "## OOP Readiness",
                f"- strict_oop_anchor_count: `{readiness.get('strict_oop_anchor_count', 0)}`",
                f"- etf_count: `{readiness.get('etf_count', 0)}`",
                f"- group_count: `{readiness.get('group_count', 0)}`",
                "",
                "This is intake/readiness only. It is not model evidence, trading advice, Stable evidence, or permission to run validation in this task.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_validator(config: IntakeConfig) -> dict[str, Any]:
    inbox = resolve_local_path(config.inbox, ALLOWED_INBOX_ROOT)
    out_dir = resolve_local_path(config.out_dir, ALLOWED_OUT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at_utc = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    if not inbox.exists():
        return build_waiting_outputs(config, out_dir, created_at_utc)

    inventory, roles, data_files = inventory_inbox(inbox)
    inventory.to_csv(out_dir / "manual_intake_inventory.csv", index=False, lineterminator="\n")
    source_note_ok, source_note_errors, source_note = validate_source_note(roles)
    manifest, manifest_load_errors = load_manifest(roles)
    manifest_ok, manifest_errors = validate_manifest(manifest)
    hash_ok, hash_errors = validate_hashes(inbox, roles, inventory)
    missing_hash = any(error.startswith("BLOCKED_MISSING_HASH") for error in hash_errors)
    hash_mismatch = any(error.startswith("BLOCKED_HASH_MISMATCH") for error in hash_errors)

    frames = []
    load_errors = []
    for data_file in data_files:
        try:
            frames.append(load_data_file(data_file))
        except Exception as exc:
            load_errors.append(f"{data_file.name}: {type(exc).__name__}: {exc}")
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    forbidden_fields = detect_forbidden_fields(raw.columns)
    secret_fields = sorted(column for column in raw.columns if is_forbidden_field_name(str(column)) and any(token in str(column).lower() for token in SECRET_FIELD_TOKENS))
    standardized, schema_details = standardize_frame(raw)
    schema_passed = bool(not schema_details["missing_required_columns"] and not forbidden_fields and not secret_fields and not load_errors and not standardized.empty)
    quality = validate_data_quality(standardized)
    source_authorized = bool(source_note_ok and manifest_ok)
    readiness = compute_oop_readiness(standardized, config, quality["data_quality_passed"], source_authorized, not forbidden_fields and not secret_fields)
    decision_name = final_decision(
        missing_source_note="source_note" not in roles,
        missing_hash="sha256sums" not in roles or missing_hash,
        hash_mismatch=hash_mismatch,
        forbidden_fields=forbidden_fields + secret_fields,
        manifest_authorized=manifest_ok and not manifest_load_errors,
        source_note_authorized=source_note_ok,
        data_quality_passed=quality["data_quality_passed"],
        oop_decision=readiness["readiness_decision"],
    )
    fixed_ready = decision_name == "MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION"

    schema_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_manual_intake_schema_report",
        "created_at_utc": created_at_utc,
        "schema_passed": schema_passed,
        "rows_checked": int(len(standardized)),
        "data_files": [path.name for path in data_files],
        "column_mapping": schema_details["column_mapping"],
        "missing_required_columns": schema_details["missing_required_columns"],
        "forbidden_fields": forbidden_fields,
        "secret_fields": secret_fields,
        "load_errors": load_errors,
        "source_note_errors": source_note_errors,
        "manifest_errors": manifest_load_errors + manifest_errors,
        "hash_errors": hash_errors,
        **BOUNDARY_FIELDS,
    }
    quality_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_manual_intake_data_quality_report",
        "created_at_utc": created_at_utc,
        **quality,
        **BOUNDARY_FIELDS,
    }
    readiness_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_manual_intake_oop_readiness",
        "created_at_utc": created_at_utc,
        "strict_oop_rule": {
            "sprint_anchor_start": config.sprint_anchor_start,
            "sprint_anchor_end": config.sprint_anchor_end,
            "anchor_before_start_or_after_end": True,
            "no_stable_bundle": True,
            "no_qmt": True,
            "no_secret": True,
        },
        **readiness,
        **BOUNDARY_FIELDS,
    }
    decision_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_manual_intake_readiness_decision",
        "created_at_utc": created_at_utc,
        "readiness_decision": decision_name,
        "fixed_shortlist_oop_validation_ready": fixed_ready,
        "inbox": str(config.inbox),
        "source_authorized": source_authorized,
        "schema_passed": schema_passed,
        "data_quality_passed": quality["data_quality_passed"],
        "strict_oop_anchor_count": readiness["strict_oop_anchor_count"],
        "etf_count": readiness["etf_count"],
        "group_count": readiness["group_count"],
        "blockers": schema_report["source_note_errors"] + schema_report["manifest_errors"] + schema_report["hash_errors"] + quality["p0_blockers"],
        "next_allowed_action": "If ready, create a separate fixed-shortlist OOP no-save validation task after human review. This task does not run validation.",
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "manual_intake_schema_report.json", schema_report)
    write_json(out_dir / "manual_intake_data_quality_report.json", quality_report)
    write_json(out_dir / "manual_intake_oop_readiness.json", readiness_report)
    write_json(out_dir / "manual_intake_readiness_decision.json", decision_report)
    write_report_md(out_dir / "manual_intake_report.md", decision_report, readiness_report, schema_report, quality_report)
    return decision_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab-only historical 5m manual/export intake validator")
    parser.add_argument("--inbox", type=Path, default=DEFAULT_INBOX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--min-oop-anchors", type=int, default=10)
    parser.add_argument("--min-etfs", type=int, default=5)
    parser.add_argument("--min-groups", type=int, default=50)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = IntakeConfig(
        inbox=args.inbox,
        out_dir=args.out_dir,
        min_oop_anchors=args.min_oop_anchors,
        min_etfs=args.min_etfs,
        min_groups=args.min_groups,
    )
    result = run_validator(config)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
