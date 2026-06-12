from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import subprocess
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
DEFAULT_RAW_EXPORT_DIR = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports")
DEFAULT_MANUAL_INBOX = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox")
DEFAULT_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_broker_export_packager")
DEFAULT_VALIDATOR_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_manual_intake")
MANUAL_INTAKE_VALIDATOR = Path("tools/lab/intraday_historical_5m_manual_intake_validator.py")

ALLOWED_ARTIFACT_ROOT = ".local_artifact_backup"
ALLOWED_REPORT_ROOT = ".local_research_outputs"
ALLOWED_SUFFIXES = {".csv", ".txt", ".zip", ".parquet"}
MANAGED_INBOX_FILES = [
    "historical_5m_manual_export.csv",
    "source_note.md",
    "MANIFEST.json",
    "SHA256SUMS.txt",
]
TARGET_ETFS = {"159915", "510050", "510300", "510500", "512100", "588000", "159949", "512880"}
STANDARD_COLUMNS = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume", "amount", "vwap"]
REQUIRED_COLUMNS = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume"]
BOUNDARY_FIELDS = {
    "access_mode": "READ_ONLY",
    "final_action_change_allowed": False,
    "contains_live_order": False,
    "contains_secret": False,
    "requires_human_review": True,
    "promotion_gate_required": True,
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

FIELD_ALIASES = {
    "etf_code": ["证券代码", "代码", "symbol", "code", "etf_code", "__source_file_code"],
    "trade_date": ["日期", "交易日期", "date", "trade_date"],
    "datetime": ["时间", "日期时间", "datetime", "time", "bar_time"],
    "open": ["开盘", "open"],
    "high": ["最高", "high"],
    "low": ["最低", "low"],
    "close": ["收盘", "close"],
    "volume": ["成交量", "volume", "vol"],
    "amount": ["成交额", "amount", "turnover"],
}
FORBIDDEN_TOKENS = [
    "account",
    "账户",
    "资金",
    "balance",
    "position",
    "持仓",
    "order",
    "委托",
    "trade",
    "成交",
    "fill",
    "password",
    "token",
    "secret",
    "target_weight",
    "final_buy_action",
    "orderintent",
]


class BrokerExportPackagerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PackagerConfig:
    raw_export_dir: Path
    manual_inbox: Path
    out_dir: Path
    run_manual_intake_validator: bool = False
    validator_out_dir: Path = DEFAULT_VALIDATOR_OUT_DIR


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def resolve_under_repo(path: Path, expected_root: str, repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = REPO_ROOT
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    repo_root = repo_root.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise BrokerExportPackagerError(f"path must be inside repo: {path}") from exc
    if not relative.parts or relative.parts[0] != expected_root:
        raise BrokerExportPackagerError(f"path must be under {expected_root}: {path}")
    return resolved


def validate_paths(config: PackagerConfig) -> tuple[Path, Path, Path, Path]:
    raw_export_dir = resolve_under_repo(config.raw_export_dir, ALLOWED_ARTIFACT_ROOT)
    manual_inbox = resolve_under_repo(config.manual_inbox, ALLOWED_ARTIFACT_ROOT)
    out_dir = resolve_under_repo(config.out_dir, ALLOWED_REPORT_ROOT)
    validator_out_dir = resolve_under_repo(config.validator_out_dir, ALLOWED_REPORT_ROOT)
    return raw_export_dir, manual_inbox, out_dir, validator_out_dir


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_name(value: str) -> str:
    return re.sub(r"[\s_\-./()（）:：]+", "", str(value).strip().lower())


def contains_forbidden_token(value: str) -> bool:
    normalized = normalize_name(value)
    for token in FORBIDDEN_TOKENS:
        if normalize_name(token) in normalized:
            if normalize_name(token) == "trade" and "tradedate" in normalized:
                continue
            if token == "成交" and value in {"成交量", "成交额"}:
                continue
            return True
    return False


def detect_forbidden_values(values: Iterable[str]) -> list[str]:
    return sorted({str(value) for value in values if contains_forbidden_token(str(value))})


def decode_text_with_fallback(raw: bytes) -> str:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "cp936"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    return raw.decode("utf-8", errors="replace")


def header_score(line: str) -> int:
    normalized = normalize_name(line)
    score = 0
    for aliases in FIELD_ALIASES.values():
        if any(normalize_name(alias) in normalized for alias in aliases if not alias.startswith("__")):
            score += 1
    return score


def extract_etf_code_from_name(value: str) -> str | None:
    match = re.search(r"(\d{6})", value)
    return match.group(1) if match else None


def normalize_time_token(value: str) -> str:
    token = str(value).strip()
    if re.fullmatch(r"\d{3,4}", token):
        padded = token.zfill(4)
        return f"{padded[:2]}:{padded[2:]}"
    if re.fullmatch(r"\d{5,6}", token):
        padded = token.zfill(6)
        return f"{padded[:2]}:{padded[2:4]}:{padded[4:]}"
    return token


def parse_delimited_text(text: str, source_name: str) -> pd.DataFrame:
    lines = text.splitlines()
    header_index = 0
    for index, line in enumerate(lines[:50]):
        if header_score(line) >= 4:
            header_index = index
            break
    frame = pd.read_csv(
        io.StringIO(text),
        dtype=str,
        sep=r"\t|,",
        engine="python",
        skiprows=header_index,
    )
    frame.columns = [str(column).strip() for column in frame.columns]
    if "__source_file_code" not in frame.columns:
        frame["__source_file_code"] = extract_etf_code_from_name(source_name) or extract_etf_code_from_name(lines[0] if lines else "") or ""
    frame["__source_file_name"] = source_name
    return frame


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    return parse_delimited_text(decode_text_with_fallback(path.read_bytes()), path.name)


def read_zip_frames(path: Path) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    frames: list[pd.DataFrame] = []
    inner_names: list[str] = []
    errors: list[str] = []
    with zipfile.ZipFile(path) as zf:
        for name in sorted(zf.namelist()):
            if name.endswith("/"):
                continue
            inner_names.append(name)
            suffix = Path(name).suffix.lower()
            if suffix not in {".csv", ".txt", ".parquet"}:
                continue
            try:
                with zf.open(name) as handle:
                    if suffix in {".csv", ".txt"}:
                        frames.append(parse_delimited_text(decode_text_with_fallback(handle.read()), name))
                    else:
                        frame = pd.read_parquet(handle)
                        frame["__source_file_code"] = extract_etf_code_from_name(name) or ""
                        frame["__source_file_name"] = name
                        frames.append(frame)
            except Exception as exc:  # noqa: BLE001 - report exact import/read failure.
                errors.append(f"{path.name}!{name}: {type(exc).__name__}: {exc}")
    return frames, inner_names, errors


def load_export_file(path: Path) -> tuple[list[pd.DataFrame], list[str], list[str]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        return [read_csv_with_fallback(path)], [], []
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
        frame["__source_file_code"] = extract_etf_code_from_name(path.name) or ""
        frame["__source_file_name"] = path.name
        return [frame], [], []
    if suffix == ".zip":
        return read_zip_frames(path)
    return [], [], [f"unsupported suffix: {path.name}"]


def inventory_raw_exports(raw_export_dir: Path) -> tuple[pd.DataFrame, list[Path]]:
    if not raw_export_dir.exists():
        return pd.DataFrame(columns=["path", "file_name", "suffix", "bytes", "sha256", "allowed_type"]), []
    rows = []
    files: list[Path] = []
    for path in sorted(item for item in raw_export_dir.iterdir() if item.is_file()):
        suffix = path.suffix.lower()
        allowed = suffix in ALLOWED_SUFFIXES
        if allowed:
            files.append(path)
        rows.append(
            {
                "path": str(path),
                "file_name": path.name,
                "suffix": suffix,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "allowed_type": allowed,
            }
        )
    return pd.DataFrame(rows), files


def build_column_mapping(columns: Sequence[str]) -> dict[str, str]:
    normalized_columns = {normalize_name(str(column)): str(column) for column in columns}
    mapping: dict[str, str] = {}
    for standard, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            found = normalized_columns.get(normalize_name(alias))
            if found is not None:
                mapping[standard] = found
                break
    return mapping


def parse_datetime_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    raw_datetime = frame[mapping["datetime"]].astype(str).map(normalize_time_token)
    trade_date = frame[mapping["trade_date"]].astype(str).str.strip()
    datetime_has_date = raw_datetime.str.contains(r"\d{4}[-/]?\d{1,2}[-/]?\d{1,2}", regex=True, na=False)
    combined = raw_datetime.where(datetime_has_date, trade_date + " " + raw_datetime)
    return pd.to_datetime(combined, errors="coerce")


def standardize_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    mapping = build_column_mapping([str(column) for column in frame.columns])
    missing = [column for column in REQUIRED_COLUMNS if column not in mapping]
    report = {
        "source_columns": [str(column) for column in frame.columns],
        "column_mapping": mapping,
        "missing_required_columns": missing,
        "raw_rows_by_source_file": frame["__source_file_name"].value_counts().to_dict() if "__source_file_name" in frame.columns else {},
    }
    if missing:
        return pd.DataFrame(columns=STANDARD_COLUMNS), report

    parsed_datetime = parse_datetime_columns(frame, mapping)
    out = pd.DataFrame()
    out["trade_date"] = parsed_datetime.dt.strftime("%Y-%m-%d")
    out["datetime"] = parsed_datetime.dt.strftime("%Y-%m-%d %H:%M:%S")
    code_raw = frame[mapping["etf_code"]].astype(str)
    out["etf_code"] = code_raw.str.extract(r"(\d{6})", expand=False).fillna(code_raw.str.strip())
    for column in ["open", "high", "low", "close", "volume"]:
        out[column] = pd.to_numeric(frame[mapping[column]], errors="coerce")
    out["amount"] = pd.to_numeric(frame[mapping["amount"]], errors="coerce") if "amount" in mapping else pd.NA
    out["vwap"] = pd.NA
    amount = pd.to_numeric(out["amount"], errors="coerce")
    volume = pd.to_numeric(out["volume"], errors="coerce")
    out.loc[volume > 0, "vwap"] = amount[volume > 0] / volume[volume > 0]
    before_filter_rows = int(len(out))
    if "__source_file_name" in frame.columns:
        out["__source_file_name"] = frame["__source_file_name"].astype(str)
    out = out[out["etf_code"].isin(TARGET_ETFS)]
    report["target_etf_filtered_rows"] = before_filter_rows - int(len(out))
    out = out.dropna(subset=REQUIRED_COLUMNS)
    if "__source_file_name" in out.columns:
        report["standardized_rows_by_source_file"] = {str(key): int(value) for key, value in out["__source_file_name"].value_counts().items()}
    return out[STANDARD_COLUMNS], report


def validate_data_quality(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "data_quality_passed": False,
            "rows": 0,
            "duplicate_bars": 0,
            "p0_blockers": ["no standardized target ETF rows"],
        }
    duplicate_count = int(frame.duplicated(["etf_code", "datetime"]).sum())
    ohlc_passed = bool(
        (
            (frame["high"] >= frame[["open", "high", "low", "close"]].max(axis=1))
            & (frame["low"] <= frame[["open", "high", "low", "close"]].min(axis=1))
        ).all()
    )
    volume_passed = bool((pd.to_numeric(frame["volume"], errors="coerce") >= 0).all())
    amount_values = pd.to_numeric(frame["amount"], errors="coerce").dropna()
    amount_passed = None if amount_values.empty else bool((amount_values >= 0).all())
    blockers = []
    if duplicate_count:
        blockers.append(f"duplicate bars detected: {duplicate_count}")
    if not ohlc_passed:
        blockers.append("OHLC consistency failed")
    if not volume_passed:
        blockers.append("volume nonnegative check failed")
    if amount_passed is False:
        blockers.append("amount nonnegative check failed")
    counts = frame.groupby(["trade_date", "etf_code"]).size()
    return {
        "data_quality_passed": not blockers,
        "rows": int(len(frame)),
        "duplicate_bars": duplicate_count,
        "ohlc_consistency_passed": ohlc_passed,
        "volume_nonnegative_passed": volume_passed,
        "amount_nonnegative_passed": amount_passed,
        "amount_available": bool(not amount_values.empty),
        "amount_nonnull_rows": int(len(amount_values)),
        "trade_date_start": str(frame["trade_date"].min()),
        "trade_date_end": str(frame["trade_date"].max()),
        "etf_count": int(frame["etf_code"].nunique()),
        "bar_groups": int(len(counts)),
        "min_bars_per_etf_day": int(counts.min()) if len(counts) else 0,
        "max_bars_per_etf_day": int(counts.max()) if len(counts) else 0,
        "p0_blockers": blockers,
    }


def build_manifest(qmt_export_only: bool = False) -> dict[str, Any]:
    qmt_related = bool(qmt_export_only)
    return {
        "sample_type": "intraday_5m_historical_manual_export",
        "frequency": "5m",
        "source_kind": "broker_terminal_manual_export",
        "acquisition_mode": "user_manual_export",
        "source_note_path": "source_note.md",
        "sha256_file": "SHA256SUMS.txt",
        "training_allowed": False,
        "stable_effect_allowed": False,
        "contains_secret": False,
        "contains_order_intent": False,
        "contains_live_order": False,
        "contains_account": False,
        "contains_position": False,
        "contains_order": False,
        "contains_trade": False,
        "qmt_related": qmt_related,
        "qmt_mode": "export_only" if qmt_related else "not_qmt",
        "human_authorized": True,
        "allowed_for": ["intake", "schema_validation", "oop_readiness"],
    }


def build_source_note(frame: pd.DataFrame, created_at_utc: str, raw_files: Sequence[Path]) -> str:
    date_range = "none"
    etf_universe = []
    if not frame.empty:
        date_range = f"{frame['trade_date'].min()} to {frame['trade_date'].max()}"
        etf_universe = sorted(frame["etf_code"].astype(str).unique().tolist())
    return "\n".join(
        [
            "# Historical 5m Broker Terminal Manual Export Source Note",
            "",
            LAB_DECLARATION,
            "",
            "- source_name: user_broker_terminal_export",
            "- source_type: broker_terminal_manual_export",
            "- acquisition_mode: user_manual_export",
            "- export_method: user_manual_file_export",
            f"- exported_at: {created_at_utc}",
            f"- date_range: {date_range}",
            f"- etf_universe: {','.join(etf_universe)}",
            "- frequency: 5m",
            "- whether_qmt_export: false",
            "- whether_account_related: false",
            "- whether_order_related: false",
            "- whether_contains_trades_or_fills: false",
            "- whether_contains_secret: false",
            "- whether_stable_bundle: false",
            "- human_authorized: true",
            f"- raw_export_files: {','.join(path.name for path in raw_files)}",
            "",
            "This package contains historical ETF 5m OHLCV bars only. It is for Lab manual intake/schema validation/OOP readiness, not trading advice, not model evidence, and not Stable evidence.",
            "",
        ]
    )


def clean_managed_inbox_files(manual_inbox: Path) -> None:
    manual_inbox.mkdir(parents=True, exist_ok=True)
    for file_name in MANAGED_INBOX_FILES:
        target = manual_inbox / file_name
        if target.exists():
            target.unlink()


def write_manual_package(manual_inbox: Path, frame: pd.DataFrame, created_at_utc: str, raw_files: Sequence[Path]) -> dict[str, str]:
    clean_managed_inbox_files(manual_inbox)
    csv_path = manual_inbox / "historical_5m_manual_export.csv"
    source_note_path = manual_inbox / "source_note.md"
    manifest_path = manual_inbox / "MANIFEST.json"
    sha_path = manual_inbox / "SHA256SUMS.txt"
    frame.to_csv(csv_path, index=False, lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    source_note_path.write_text(build_source_note(frame, created_at_utc, raw_files), encoding="utf-8")
    write_json(manifest_path, build_manifest())
    hash_lines = []
    for path in [csv_path, source_note_path, manifest_path]:
        hash_lines.append(f"{sha256_file(path)}  {path.name}")
    sha_path.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    return {path.name: sha256_file(path) for path in [csv_path, source_note_path, manifest_path, sha_path]}


def write_report_md(path: Path, report: dict[str, Any], decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Broker Terminal Historical 5m Export Packager Report",
                "",
                LAB_DECLARATION,
                "",
                "## Decision",
                f"- decision: `{decision.get('decision')}`",
                f"- manual_intake_validator_decision: `{decision.get('manual_intake_validator_decision')}`",
                f"- fixed_shortlist_oop_validation_ready: `{decision.get('fixed_shortlist_oop_validation_ready')}`",
                "",
                "## Raw Export",
                f"- raw_export_dir_exists: `{report.get('raw_export_dir_exists')}`",
                f"- raw_allowed_file_count: `{report.get('raw_allowed_file_count')}`",
                "",
                "## Package",
                f"- package_generated: `{report.get('package_generated')}`",
                f"- standardized_rows: `{report.get('standardized_rows')}`",
                f"- etf_count: `{report.get('etf_count')}`",
                f"- date_range: `{report.get('date_range')}`",
                "",
                "## Boundary",
                "This is Lab manual packaging/readiness only. It does not connect to a broker, QMT, account, position, order, trade, fill, model training, OrderIntent, Stable, or advisory promotion.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def run_manual_validator(manual_inbox: Path, validator_out_dir: Path) -> dict[str, Any]:
    command = [
        sys.executable,
        str((REPO_ROOT / MANUAL_INTAKE_VALIDATOR).resolve()),
        "--inbox",
        str(manual_inbox),
        "--out-dir",
        str(validator_out_dir),
    ]
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=120, check=False)
    decision_path = validator_out_dir / "manual_intake_readiness_decision.json"
    validator_decision = None
    if decision_path.exists():
        try:
            validator_decision = json.loads(decision_path.read_text(encoding="utf-8")).get("readiness_decision")
        except json.JSONDecodeError:
            validator_decision = None
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout_excerpt": completed.stdout[:1000],
        "stderr_excerpt": completed.stderr[:1000],
        "decision_path": str(decision_path),
        "manual_intake_validator_decision": validator_decision,
    }


def build_waiting_result(config: PackagerConfig, raw_export_dir: Path, manual_inbox: Path, out_dir: Path, created_at_utc: str) -> dict[str, Any]:
    inventory = pd.DataFrame(columns=["path", "file_name", "suffix", "bytes", "sha256", "allowed_type"])
    inventory.to_csv(out_dir / "broker_export_inventory.csv", index=False, lineterminator="\n")
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_broker_export_package_report",
        "created_at_utc": created_at_utc,
        "raw_export_dir": str(raw_export_dir),
        "raw_export_dir_exists": raw_export_dir.exists(),
        "raw_allowed_file_count": 0,
        "package_generated": False,
        "manual_inbox": str(manual_inbox),
        "standardized_rows": 0,
        "etf_count": 0,
        "date_range": None,
        "validator_handoff_requested": config.run_manual_intake_validator,
        "validator_handoff_ran": False,
        **BOUNDARY_FIELDS,
    }
    decision = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_broker_export_package_decision",
        "created_at_utc": created_at_utc,
        "decision": "BROKER_EXPORT_PACKAGE_BLOCKED_WAITING_FOR_RAW_EXPORT",
        "failure_reason": "Raw export directory missing or contains no allowed CSV/ZIP/parquet files.",
        "package_generated": False,
        "manual_inbox": str(manual_inbox),
        "manual_intake_validator_command": None,
        "manual_intake_validator_decision": None,
        "fixed_shortlist_oop_validation_ready": False,
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "broker_export_package_report.json", report)
    write_json(out_dir / "broker_export_package_decision.json", decision)
    write_report_md(out_dir / "broker_export_package_report.md", report, decision)
    return decision


def package_exports(config: PackagerConfig) -> dict[str, Any]:
    raw_export_dir, manual_inbox, out_dir, validator_out_dir = validate_paths(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at_utc = utc_now()
    if not raw_export_dir.exists():
        return build_waiting_result(config, raw_export_dir, manual_inbox, out_dir, created_at_utc)

    inventory, raw_files = inventory_raw_exports(raw_export_dir)
    inventory.to_csv(out_dir / "broker_export_inventory.csv", index=False, lineterminator="\n")
    if not raw_files:
        return build_waiting_result(config, raw_export_dir, manual_inbox, out_dir, created_at_utc)

    filename_forbidden = detect_forbidden_values([path.name for path in raw_files])
    frames = []
    read_errors: list[str] = []
    zip_inner_names: list[str] = []
    for raw_file in raw_files:
        try:
            file_frames, inner_names, errors = load_export_file(raw_file)
            frames.extend(file_frames)
            zip_inner_names.extend(inner_names)
            read_errors.extend(errors)
        except Exception as exc:  # noqa: BLE001 - turn reader errors into bounded reports.
            read_errors.append(f"{raw_file.name}: {type(exc).__name__}: {exc}")
    zip_forbidden = detect_forbidden_values(zip_inner_names)
    raw = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    field_forbidden = detect_forbidden_values([str(column) for column in raw.columns])
    standardized, schema = standardize_frame(raw)
    quality = validate_data_quality(standardized)

    package_generated = False
    package_hashes: dict[str, str] = {}
    validator_result: dict[str, Any] | None = None
    blockers = []
    if read_errors:
        blockers.extend(read_errors)
    if filename_forbidden or zip_forbidden or field_forbidden:
        blockers.extend([f"forbidden raw file name: {name}" for name in filename_forbidden])
        blockers.extend([f"forbidden zip member name: {name}" for name in zip_forbidden])
        blockers.extend([f"forbidden field: {name}" for name in field_forbidden])
        decision_name = "BROKER_EXPORT_PACKAGE_BLOCKED_FORBIDDEN_FIELDS"
    elif schema["missing_required_columns"]:
        blockers.extend([f"missing required column: {name}" for name in schema["missing_required_columns"]])
        decision_name = "BROKER_EXPORT_PACKAGE_BLOCKED_SCHEMA_UNMAPPABLE"
    elif not quality["data_quality_passed"]:
        blockers.extend(quality["p0_blockers"])
        decision_name = "BROKER_EXPORT_PACKAGE_BLOCKED_DATA_QUALITY"
    else:
        package_hashes = write_manual_package(manual_inbox, standardized, created_at_utc, raw_files)
        package_generated = True
        decision_name = "BROKER_EXPORT_PACKAGE_READY_FOR_MANUAL_INTAKE_VALIDATOR"
        if config.run_manual_intake_validator:
            validator_out_dir.mkdir(parents=True, exist_ok=True)
            validator_result = run_manual_validator(manual_inbox, validator_out_dir)
            if validator_result.get("manual_intake_validator_decision") == "MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION":
                decision_name = "BROKER_EXPORT_PACKAGE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION"
            else:
                decision_name = "BROKER_EXPORT_PACKAGE_BLOCKED_VALIDATOR_NOT_READY"

    date_range = None
    if not standardized.empty:
        date_range = f"{standardized['trade_date'].min()} to {standardized['trade_date'].max()}"
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_broker_export_package_report",
        "created_at_utc": created_at_utc,
        "raw_export_dir": str(raw_export_dir),
        "raw_export_dir_exists": raw_export_dir.exists(),
        "raw_allowed_file_count": len(raw_files),
        "raw_files": [path.name for path in raw_files],
        "manual_inbox": str(manual_inbox),
        "package_generated": package_generated,
        "package_hashes": package_hashes,
        "source_column_mapping": schema.get("column_mapping", {}),
        "missing_required_columns": schema.get("missing_required_columns", []),
        "raw_rows_by_source_file": schema.get("raw_rows_by_source_file", {}),
        "standardized_rows_by_source_file": schema.get("standardized_rows_by_source_file", {}),
        "forbidden_file_names": filename_forbidden + zip_forbidden,
        "forbidden_fields": field_forbidden,
        "read_errors": read_errors,
        "standardized_rows": int(len(standardized)),
        "parsed_etf_codes": sorted(standardized["etf_code"].astype(str).unique().tolist()) if not standardized.empty else [],
        "etf_count": int(standardized["etf_code"].nunique()) if not standardized.empty else 0,
        "date_range": date_range,
        "data_quality": quality,
        "validator_handoff_requested": config.run_manual_intake_validator,
        "validator_handoff_ran": validator_result is not None,
        "validator_result": validator_result,
        **BOUNDARY_FIELDS,
    }
    manual_intake_validator_decision = None
    manual_intake_validator_command = None
    if validator_result is not None:
        manual_intake_validator_decision = validator_result.get("manual_intake_validator_decision")
        manual_intake_validator_command = validator_result.get("command")
    fixed_ready = decision_name == "BROKER_EXPORT_PACKAGE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION"
    decision = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_historical_5m_broker_export_package_decision",
        "created_at_utc": created_at_utc,
        "decision": decision_name,
        "failure_reason": "; ".join(blockers) if blockers else None,
        "package_generated": package_generated,
        "manual_inbox": str(manual_inbox),
        "manual_intake_validator_command": manual_intake_validator_command,
        "manual_intake_validator_decision": manual_intake_validator_decision,
        "fixed_shortlist_oop_validation_ready": fixed_ready,
        "next_allowed_action": "Only if validator returns MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION may a separate fixed-shortlist OOP no-save validation task be opened after human review.",
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "broker_export_package_report.json", report)
    write_json(out_dir / "broker_export_package_decision.json", decision)
    write_report_md(out_dir / "broker_export_package_report.md", report, decision)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab-only broker terminal historical ETF 5m export packager")
    parser.add_argument("--raw-export-dir", type=Path, default=DEFAULT_RAW_EXPORT_DIR)
    parser.add_argument("--manual-inbox", type=Path, default=DEFAULT_MANUAL_INBOX)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--validator-out-dir", type=Path, default=DEFAULT_VALIDATOR_OUT_DIR)
    parser.add_argument("--run-manual-intake-validator", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = PackagerConfig(
        raw_export_dir=args.raw_export_dir,
        manual_inbox=args.manual_inbox,
        out_dir=args.out_dir,
        validator_out_dir=args.validator_out_dir,
        run_manual_intake_validator=args.run_manual_intake_validator,
    )
    try:
        result = package_exports(config)
    except BrokerExportPackagerError as exc:
        print(f"BROKER_EXPORT_PACKAGER_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
