from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import struct
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
TARGET_ETFS = {"159915", "510050", "510300", "510500", "512100", "588000", "159949", "512880"}
DEFAULT_CLIENT_ROOTS = [
    Path(r"E:\CGS\T0002\export"),
    Path(r"E:\CGS\T0002"),
    Path(r"E:\XGS\T0002\export"),
    Path(r"E:\XGS\T0002"),
]
DEFAULT_RAW_EXPORT_DIR = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports")
DEFAULT_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_broker_client_local_data_rescue")
DEFAULT_RESCUED_DIR = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_raw_exports_rescued")
DEFAULT_PACKAGER_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_broker_export_packager")
DEFAULT_MANUAL_INBOX = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox")
DEFAULT_VALIDATOR_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_historical_5m_manual_intake")
PACKAGER_SCRIPT = Path("tools/lab/intraday_historical_5m_broker_export_packager.py")

STANDARD_COLUMNS = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume", "amount", "vwap"]
TEXT_SUFFIXES = {".csv", ".txt"}
INVENTORY_SUFFIXES = {".lc5", ".lc1", ".day", ".dat", ".csv", ".txt", ".zip", ".db", ".sqlite"}
FIVE_MIN_SUFFIXES = {".lc5", ".csv", ".txt", ".db", ".sqlite", ".dat"}
PATH_KEYWORDS = {"vipdoc", "fzline", "minline", "lday", "t0002", "export", "download", "history", "cache"}
FIVE_MIN_KEYWORDS = {"lc5", "fzline", "minline", "5m", "5min", "5分钟", "分钟线"}
FORBIDDEN_PATH_TOKENS = [
    "account",
    "position",
    "order",
    "trade",
    "fill",
    "secret",
    "token",
    "password",
    "cookie",
    "账户",
    "委托",
    "成交",
    "持仓",
    "资金",
]
FIELD_ALIASES = {
    "etf_code": ["证券代码", "代码", "symbol", "code", "etf_code", "sec_code"],
    "trade_date": ["日期", "交易日期", "date", "trade_date"],
    "datetime": ["日期时间", "datetime", "bar_time", "time"],
    "open": ["开盘", "open", "o"],
    "high": ["最高", "high", "h"],
    "low": ["最低", "low", "l"],
    "close": ["收盘", "close", "c"],
    "volume": ["成交量", "volume", "vol", "qty"],
    "amount": ["成交额", "amount", "turnover", "money"],
}
BOUNDARY_FIELDS = {
    "access_mode": "READ_ONLY",
    "broker_login_used": False,
    "broker_network_used": False,
    "qmt_used": False,
    "xtdata_used": False,
    "account_api_used": False,
    "position_api_used": False,
    "order_api_used": False,
    "trade_api_used": False,
    "fill_api_used": False,
    "contains_account": False,
    "contains_position": False,
    "contains_order": False,
    "contains_trade": False,
    "contains_fill": False,
    "contains_secret": False,
    "model_training_allowed": False,
    "labels_generated": False,
    "order_intent_generated": False,
    "automatic_order_allowed": False,
    "stable_affected": False,
    "stable_promotion_ready": False,
    "metrics_are_effectiveness_evidence": False,
    "not_trading_advice": True,
}


class BrokerClientLocalDataRescueError(RuntimeError):
    pass


@dataclass(frozen=True)
class RescueConfig:
    local_roots: Sequence[Path] = field(default_factory=lambda: [*DEFAULT_CLIENT_ROOTS, DEFAULT_RAW_EXPORT_DIR])
    export_roots: Sequence[Path] = field(default_factory=lambda: [Path(r"E:\CGS\T0002\export"), Path(r"E:\XGS\T0002\export"), DEFAULT_RAW_EXPORT_DIR])
    raw_export_dir: Path = DEFAULT_RAW_EXPORT_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    rescued_dir: Path = DEFAULT_RESCUED_DIR
    manual_inbox: Path = DEFAULT_MANUAL_INBOX
    packager_out_dir: Path = DEFAULT_PACKAGER_OUT_DIR
    validator_out_dir: Path = DEFAULT_VALIDATOR_OUT_DIR
    run_packager: bool = True
    run_packager_validator: bool = True


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_under_repo(path: Path, expected_root: str, repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = REPO_ROOT
    candidate = path if path.is_absolute() else repo_root / path
    resolved = candidate.resolve()
    repo_root = repo_root.resolve()
    try:
        relative = resolved.relative_to(repo_root)
    except ValueError as exc:
        raise BrokerClientLocalDataRescueError(f"path must be inside repo: {path}") from exc
    if not relative.parts or relative.parts[0] != expected_root:
        raise BrokerClientLocalDataRescueError(f"path must be under {expected_root}: {path}")
    return resolved


def resolve_input_path(path: Path, repo_root: Path | None = None) -> Path:
    if repo_root is None:
        repo_root = REPO_ROOT
    return (path if path.is_absolute() else repo_root / path).resolve()


def normalize_name(value: str) -> str:
    return re.sub(r"[\s_\-./\\()（）:：]+", "", str(value).strip().lower())


def contains_forbidden_path_token(value: str) -> bool:
    normalized = normalize_name(value)
    return any(normalize_name(token) in normalized for token in FORBIDDEN_PATH_TOKENS)


def path_is_forbidden(path: Path) -> bool:
    return contains_forbidden_path_token(str(path))


def interesting_inventory_path(path: Path) -> bool:
    normalized = normalize_name(str(path))
    suffix = path.suffix.lower()
    return (
        suffix in INVENTORY_SUFFIXES
        or any(code in normalized for code in TARGET_ETFS)
        or any(keyword in normalized for keyword in PATH_KEYWORDS)
    )


def candidate_reason(path: Path) -> str | None:
    normalized = normalize_name(str(path))
    suffix = path.suffix.lower()
    reasons = []
    has_target_code = any(code in normalized for code in TARGET_ETFS)
    has_5m_keyword = any(keyword in normalized for keyword in FIVE_MIN_KEYWORDS)
    has_market_keyword = any(keyword in normalized for keyword in PATH_KEYWORDS)
    if has_target_code:
        reasons.append("target_etf_code_in_path")
    if has_5m_keyword:
        reasons.append("5m_keyword_in_path")
    if suffix == ".lc5":
        reasons.append("tdx_lc5_suffix")
    elif suffix in TEXT_SUFFIXES and (has_target_code or has_5m_keyword):
        reasons.append("text_candidate")
    elif suffix in {".db", ".sqlite"} and (has_target_code or has_5m_keyword or has_market_keyword):
        reasons.append("db_under_market_data_keyword_path")
    elif suffix == ".dat" and (has_target_code or has_5m_keyword):
        reasons.append("dat_candidate_with_target_or_5m_hint")
    return ";".join(dict.fromkeys(reasons)) if reasons and suffix in INVENTORY_SUFFIXES else None


def decode_text_prefix(path: Path, max_bytes: int = 4096) -> tuple[str, str, bytes]:
    raw = path.read_bytes()[:max_bytes]
    for encoding in ("utf-8-sig", "utf-8", "gbk", "ansi"):
        try:
            return raw.decode("mbcs" if encoding == "ansi" else encoding), encoding, raw
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace"), "unknown", raw


def guess_delimiter(text: str) -> str:
    sample = text[:4096]
    if sample.count("\t") >= max(sample.count(","), sample.count(";"), sample.count("|")):
        return "TAB"
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return "TAB" if dialect.delimiter == "\t" else dialect.delimiter
    except csv.Error:
        return ","


def line_has_ohlcv_header(line: str) -> bool:
    return all(any(alias.lower() in line.lower() or alias in line for alias in aliases) for aliases in (FIELD_ALIASES["open"], FIELD_ALIASES["high"], FIELD_ALIASES["low"], FIELD_ALIASES["close"]))


def find_header_line_index(lines: Sequence[str]) -> int:
    for index, line in enumerate(lines[:10]):
        if line_has_ohlcv_header(line):
            return index
    return 0


def count_lines_bounded(path: Path, max_lines: int = 100000) -> tuple[int, bool]:
    lines = 0
    truncated = False
    with path.open("rb") as handle:
        for lines, _ in enumerate(handle, start=1):
            if lines >= max_lines:
                truncated = True
                break
    return lines, truncated


def analyze_export_file(path: Path, target_code: str) -> dict[str, Any]:
    text, encoding, _ = decode_text_prefix(path)
    delimiter = guess_delimiter(text)
    line_count, line_count_truncated = count_lines_bounded(path)
    nonempty_lines = [line for line in text.splitlines() if line.strip()]
    header_index = find_header_line_index(nonempty_lines)
    header = nonempty_lines[header_index] if nonempty_lines else ""
    data_lines = [line for line in nonempty_lines[header_index + 1 :] if not line.lstrip().startswith("#")]
    contains_source_note = "#数据来源" in text
    contains_target = target_code in text
    data_line_count_in_prefix = len(data_lines)
    has_ohlcv_headers = line_has_ohlcv_header(header)
    has_ohlcv_data = data_line_count_in_prefix > 0 and has_ohlcv_headers
    ordinary_stock_codes = re.findall(r"(?<!\d)([036]\d{5})(?!\d)", text)
    ordinary_non_target = [code for code in ordinary_stock_codes if code not in TARGET_ETFS]
    return {
        "path": str(path),
        "file_name": path.name,
        "target_etf": target_code,
        "file_exists": True,
        "bytes": path.stat().st_size,
        "line_count": line_count,
        "line_count_truncated": line_count_truncated,
        "header_only": has_ohlcv_headers and data_line_count_in_prefix == 0,
        "contains_source_note": contains_source_note,
        "contains_target_etf_code": contains_target,
        "has_ohlcv_header": has_ohlcv_headers,
        "has_ohlcv_data_rows_in_prefix": has_ohlcv_data,
        "encoding_guess": encoding,
        "delimiter_guess": delimiter,
        "only_a_share_non_etf_data_in_prefix": bool(ordinary_non_target and not contains_target),
        "diagnosis": "header_only_empty_export" if has_ohlcv_headers and data_line_count_in_prefix == 0 else "has_data_prefix" if has_ohlcv_data else "no_ohlcv_rows_in_export",
    }


def find_export_files_for_code(export_roots: Sequence[Path], code: str) -> list[Path]:
    matches: list[Path] = []
    for root in export_roots:
        resolved = resolve_input_path(root)
        if not resolved.exists():
            continue
        if resolved.is_file():
            if code in resolved.name and resolved.suffix.lower() in TEXT_SUFFIXES:
                matches.append(resolved)
            continue
        for current_root, dirs, files in os.walk(resolved, followlinks=False):
            dirs[:] = [name for name in dirs if not contains_forbidden_path_token(str(Path(current_root) / name))]
            for name in files:
                path = Path(current_root) / name
                if path_is_forbidden(path):
                    continue
                if code in name and path.suffix.lower() in TEXT_SUFFIXES:
                    matches.append(path)
    return sorted(dict.fromkeys(matches))


def diagnose_empty_exports(export_roots: Sequence[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for code in sorted(TARGET_ETFS):
        matches = find_export_files_for_code(export_roots, code)
        if not matches:
            rows.append(
                {
                    "path": "",
                    "file_name": "",
                    "target_etf": code,
                    "file_exists": False,
                    "bytes": 0,
                    "line_count": 0,
                    "line_count_truncated": False,
                    "header_only": False,
                    "contains_source_note": False,
                    "contains_target_etf_code": False,
                    "has_ohlcv_header": False,
                    "has_ohlcv_data_rows_in_prefix": False,
                    "encoding_guess": "",
                    "delimiter_guess": "",
                    "only_a_share_non_etf_data_in_prefix": False,
                    "diagnosis": "target_etf_file_missing",
                }
            )
            continue
        for path in matches:
            rows.append(analyze_export_file(path, code))
    return pd.DataFrame(rows)


def inventory_local_data_files(local_roots: Sequence[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for root in local_roots:
        resolved = resolve_input_path(root)
        root_exists = resolved.exists()
        if not root_exists:
            rows.append(
                {
                    "root": str(resolved),
                    "path": str(resolved),
                    "item_type": "root",
                    "file_name": resolved.name,
                    "suffix": resolved.suffix.lower(),
                    "bytes": 0,
                    "contains_target_code": False,
                    "contains_market_keyword": False,
                    "candidate_reason": "",
                    "skipped_for_safety": False,
                    "status": "root_missing",
                }
            )
            continue
        walk_roots = [resolved] if resolved.is_file() else None
        if walk_roots is not None:
            files = [resolved.name]
            current_roots = [(str(resolved.parent), [], files)]
        else:
            current_roots = os.walk(resolved, followlinks=False)
        for current_root, dirs, files in current_roots:
            safe_dirs = []
            for dirname in dirs:
                dir_path = Path(current_root) / dirname
                if path_is_forbidden(dir_path):
                    rows.append(
                        {
                            "root": str(resolved),
                            "path": str(dir_path),
                            "item_type": "directory",
                            "file_name": dirname,
                            "suffix": "",
                            "bytes": 0,
                            "contains_target_code": any(code in normalize_name(str(dir_path)) for code in TARGET_ETFS),
                            "contains_market_keyword": any(keyword in normalize_name(str(dir_path)) for keyword in PATH_KEYWORDS),
                            "candidate_reason": "",
                            "skipped_for_safety": True,
                            "status": "skipped_forbidden_path",
                        }
                    )
                    continue
                safe_dirs.append(dirname)
            dirs[:] = safe_dirs
            for filename in files:
                path = Path(current_root) / filename
                if path in seen:
                    continue
                seen.add(path)
                forbidden = path_is_forbidden(path)
                if not forbidden and not interesting_inventory_path(path):
                    continue
                reason = None if forbidden else candidate_reason(path)
                row = {
                    "root": str(resolved),
                    "path": str(path),
                    "item_type": "file",
                    "file_name": path.name,
                    "suffix": path.suffix.lower(),
                    "bytes": path.stat().st_size if path.exists() else 0,
                    "contains_target_code": any(code in normalize_name(str(path)) for code in TARGET_ETFS),
                    "contains_market_keyword": any(keyword in normalize_name(str(path)) for keyword in PATH_KEYWORDS),
                    "candidate_reason": reason or "",
                    "skipped_for_safety": forbidden,
                    "status": "skipped_forbidden_path" if forbidden else "candidate" if reason else "inventory_only",
                }
                rows.append(row)
                if reason:
                    candidate_rows.append(row)
    return pd.DataFrame(rows), pd.DataFrame(candidate_rows)


def build_column_mapping(columns: Sequence[str]) -> dict[str, str]:
    normalized_to_original = {normalize_name(column): column for column in columns}
    mapping: dict[str, str] = {}
    for standard, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            normalized = normalize_name(alias)
            if normalized in normalized_to_original:
                mapping[standard] = normalized_to_original[normalized]
                break
    return mapping


def parse_datetime_columns(frame: pd.DataFrame, mapping: dict[str, str]) -> pd.Series:
    if "datetime" in mapping and "trade_date" in mapping and mapping["datetime"] != mapping["trade_date"]:
        raw_time = frame[mapping["datetime"]].astype(str).str.strip()
        raw_date = frame[mapping["trade_date"]].astype(str).str.strip()
        needs_date = raw_time.str.fullmatch(r"\d{1,2}:?\d{2}(:\d{2})?").fillna(False)
        combined = raw_time.mask(needs_date, raw_date + " " + raw_time)
        return pd.to_datetime(combined, errors="coerce")
    if "datetime" in mapping:
        return pd.to_datetime(frame[mapping["datetime"]], errors="coerce")
    if "trade_date" in mapping:
        return pd.to_datetime(frame[mapping["trade_date"]], errors="coerce")
    return pd.Series(pd.NaT, index=frame.index)


def standardize_frame(frame: pd.DataFrame, source_path: Path | None = None) -> tuple[pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return pd.DataFrame(columns=STANDARD_COLUMNS), {"column_mapping": {}, "missing_required_columns": STANDARD_COLUMNS[:8]}
    mapping = build_column_mapping([str(column) for column in frame.columns])
    missing = [name for name in ["datetime", "etf_code", "open", "high", "low", "close", "volume"] if name not in mapping]
    if "datetime" not in mapping and "trade_date" in mapping:
        missing = [name for name in missing if name != "datetime"]
    if missing:
        return pd.DataFrame(columns=STANDARD_COLUMNS), {"column_mapping": mapping, "missing_required_columns": missing}
    standardized = pd.DataFrame(index=frame.index)
    dt = parse_datetime_columns(frame, mapping)
    standardized["datetime"] = dt.dt.strftime("%Y-%m-%d %H:%M:%S")
    standardized["trade_date"] = dt.dt.strftime("%Y-%m-%d")
    code_series = frame[mapping["etf_code"]].astype(str).str.extract(r"(\d{6})", expand=False)
    if code_series.isna().all() and source_path is not None:
        code_match = re.search(r"(\d{6})", source_path.name)
        if code_match:
            code_series = pd.Series(code_match.group(1), index=frame.index)
    standardized["etf_code"] = code_series.astype(str).str.zfill(6)
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column in mapping:
            standardized[column] = pd.to_numeric(frame[mapping[column]], errors="coerce")
        else:
            standardized[column] = pd.NA
    standardized = standardized[standardized["etf_code"].isin(TARGET_ETFS)].copy()
    if "amount" in standardized and standardized["amount"].notna().any():
        standardized["vwap"] = standardized["amount"] / standardized["volume"].replace({0: pd.NA})
    else:
        standardized["vwap"] = pd.NA
    standardized = standardized.dropna(subset=["datetime", "trade_date", "open", "high", "low", "close", "volume"])
    return standardized[STANDARD_COLUMNS], {"column_mapping": mapping, "missing_required_columns": []}


def parse_text_candidate(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    text, encoding, raw = decode_text_prefix(path)
    if not raw:
        return pd.DataFrame(columns=STANDARD_COLUMNS), {"parser": "text", "encoding": encoding, "status": "empty_file"}
    delimiter = guess_delimiter(text)
    sep = "\t" if delimiter == "TAB" else delimiter
    lines = text.splitlines()
    header_index = find_header_line_index([line for line in lines if line.strip()])
    parse_text = "\n".join([line for line in lines[header_index:] if line.strip() and not line.lstrip().startswith("#")])
    try:
        frame = pd.read_csv(io.StringIO(parse_text), dtype=str, sep=sep, engine="python")
    except Exception as exc:  # noqa: BLE001 - parser diagnosis only.
        return pd.DataFrame(columns=STANDARD_COLUMNS), {"parser": "text", "encoding": encoding, "status": f"read_error:{type(exc).__name__}:{exc}"}
    if len(frame) > 20:
        frame = frame.head(20)
    standardized, schema = standardize_frame(frame, path)
    return standardized, {"parser": "text", "encoding": encoding, "delimiter": delimiter, "rows_read": int(len(frame)), **schema}


def decode_tdx_lc5_date(value: int) -> str | None:
    year = value // 2048 + 2004
    month = (value % 2048) // 100
    day = value % 100
    if 1990 <= year <= 2100 and 1 <= month <= 12 and 1 <= day <= 31:
        return f"{year:04d}-{month:02d}-{day:02d}"
    return None


def parse_lc5_candidate(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = path.read_bytes()[: min(4096, 20 * 32)]
    if len(raw) < 32:
        return pd.DataFrame(columns=STANDARD_COLUMNS), {"parser": "tdx_lc5", "status": "too_short"}
    rows: list[dict[str, Any]] = []
    code_match = re.search(r"(\d{6})", path.name)
    etf_code = code_match.group(1) if code_match else ""
    for offset in range(0, min(len(raw), 20 * 32), 32):
        chunk = raw[offset : offset + 32]
        if len(chunk) < 32:
            break
        try:
            packed_date, packed_time, open_, high, low, close, amount, volume, _reserved = struct.unpack("<HHfffffII", chunk)
        except struct.error:
            break
        trade_date = decode_tdx_lc5_date(packed_date)
        hour = packed_time // 60
        minute = packed_time % 60
        if trade_date is None or not (0 <= hour <= 23 and 0 <= minute <= 59):
            continue
        rows.append(
            {
                "trade_date": trade_date,
                "datetime": f"{trade_date} {hour:02d}:{minute:02d}:00",
                "etf_code": etf_code,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "amount": amount,
                "vwap": amount / volume if volume else pd.NA,
            }
        )
    frame = pd.DataFrame(rows, columns=STANDARD_COLUMNS)
    frame = frame[frame["etf_code"].isin(TARGET_ETFS)].copy() if not frame.empty else frame
    return frame, {"parser": "tdx_lc5", "records_read": int(len(raw) // 32), "rows_parsed": int(len(frame))}


def parse_sqlite_candidate(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = path.read_bytes()[:100]
    if not raw.startswith(b"SQLite format 3"):
        return pd.DataFrame(columns=STANDARD_COLUMNS), {"parser": "sqlite", "status": "not_sqlite_magic"}
    uri = f"file:{path.as_posix()}?mode=ro"
    table_summaries: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []
    with sqlite3.connect(uri, uri=True) as conn:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()]
        for table in tables[:20]:
            if contains_forbidden_path_token(table):
                table_summaries.append({"table": table, "status": "skipped_forbidden_table_name"})
                continue
            columns = [row[1] for row in conn.execute(f"PRAGMA table_info({quote_sqlite_ident(table)})").fetchall()]
            table_summaries.append({"table": table, "columns": columns})
            mapping = build_column_mapping(columns)
            if not {"etf_code", "open", "high", "low", "close", "volume"}.issubset(mapping):
                continue
            query = f"SELECT * FROM {quote_sqlite_ident(table)} LIMIT 20"
            raw_frame = pd.read_sql_query(query, conn, dtype=str)
            standardized, _schema = standardize_frame(raw_frame, path)
            if not standardized.empty:
                frames.append(standardized)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=STANDARD_COLUMNS)
    return frame, {"parser": "sqlite", "tables": table_summaries, "rows_parsed": int(len(frame))}


def quote_sqlite_ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def parse_candidate(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    if path_is_forbidden(path):
        return pd.DataFrame(columns=STANDARD_COLUMNS), {"parser": "none", "status": "skipped_forbidden_path"}
    suffix = path.suffix.lower()
    if suffix in TEXT_SUFFIXES:
        return parse_text_candidate(path)
    if suffix == ".lc5":
        return parse_lc5_candidate(path)
    if suffix in {".db", ".sqlite"}:
        return parse_sqlite_candidate(path)
    return pd.DataFrame(columns=STANDARD_COLUMNS), {"parser": "none", "status": "unsupported_suffix"}


def validate_rescued_frame(frame: pd.DataFrame) -> dict[str, Any]:
    blockers: list[str] = []
    if frame.empty:
        blockers.append("no_parseable_ohlcv_rows")
    missing = [column for column in STANDARD_COLUMNS[:8] if column not in frame.columns]
    blockers.extend([f"missing_column:{column}" for column in missing])
    if not frame.empty:
        numeric = frame[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            blockers.append("non_numeric_ohlcv")
        if not ((numeric["high"] >= numeric["low"]) & (numeric["volume"] >= 0)).all():
            blockers.append("invalid_high_low_or_volume")
        if not frame["etf_code"].isin(TARGET_ETFS).all():
            blockers.append("non_target_etf_rows")
    return {"data_quality_passed": not blockers, "p0_blockers": blockers, "rows": int(len(frame))}


def write_rescued_package(rescued_dir: Path, raw_export_dir: Path, frame: pd.DataFrame, source_paths: Sequence[Path], created_at_utc: str) -> dict[str, Any]:
    rescued_dir.mkdir(parents=True, exist_ok=True)
    raw_export_dir.mkdir(parents=True, exist_ok=True)
    csv_path = rescued_dir / "rescued_historical_5m.csv"
    note_path = rescued_dir / "source_note.md"
    manifest_path = rescued_dir / "MANIFEST.json"
    sha_path = rescued_dir / "SHA256SUMS.txt"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig", lineterminator="\n")
    note_path.write_text(
        "\n".join(
            [
                "# Broker Client Local 5m Data Rescue Source Note",
                "",
                LAB_DECLARATION,
                "",
                f"- created_at_utc: {created_at_utc}",
                "- source_type: broker_client_local_market_data_file",
                "- access_mode: read_only_bounded_parser",
                "- frequency: 5m",
                f"- source_files: {', '.join(str(path) for path in source_paths)}",
                "- account_related: false",
                "- order_related: false",
                "- qmt_related: false",
                "- stable_related: false",
                "- human_authorized: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "lab_declaration": LAB_DECLARATION,
        "sample_type": "intraday_5m_historical_broker_client_local_rescue",
        "created_at_utc": created_at_utc,
        "frequency": "5m",
        "source_kind": "broker_client_local_market_data_file",
        "source_files": [str(path) for path in source_paths],
        "rescued_csv": str(csv_path),
        "training_allowed": False,
        "stable_effect_allowed": False,
        "contains_secret": False,
        "contains_order_intent": False,
        "contains_live_order": False,
        "contains_account": False,
        "contains_position": False,
        "contains_order": False,
        "contains_trade": False,
        "qmt_related": False,
        "qmt_mode": "not_qmt",
        "human_authorized": True,
        "allowed_for": "manual_intake_validation_after_human_review_only",
        **BOUNDARY_FIELDS,
    }
    write_json(manifest_path, manifest)
    sha_lines = []
    for path in [csv_path, note_path, manifest_path]:
        sha_lines.append(f"{sha256_file(path)}  {path.name}")
    sha_path.write_text("\n".join(sha_lines) + "\n", encoding="utf-8")
    handoff_path = raw_export_dir / "rescued_historical_5m.csv"
    shutil.copy2(csv_path, handoff_path)
    return {
        "rescued_csv": str(csv_path),
        "source_note": str(note_path),
        "manifest": str(manifest_path),
        "sha256sums": str(sha_path),
        "raw_export_handoff_csv": str(handoff_path),
    }


def run_packager(config: RescueConfig, raw_export_dir: Path, created_at_utc: str) -> dict[str, Any]:
    command = [
        sys.executable,
        str((REPO_ROOT / PACKAGER_SCRIPT).resolve()),
        "--raw-export-dir",
        str(raw_export_dir),
        "--manual-inbox",
        str(resolve_under_repo(config.manual_inbox, ".local_artifact_backup")),
        "--out-dir",
        str(resolve_under_repo(config.packager_out_dir, ".local_research_outputs")),
    ]
    if config.run_packager_validator:
        command.append("--run-manual-intake-validator")
        command.extend(["--validator-out-dir", str(resolve_under_repo(config.validator_out_dir, ".local_research_outputs"))])
    completed = subprocess.run(command, cwd=REPO_ROOT, capture_output=True, text=True, timeout=180, check=False)
    decision_path = resolve_under_repo(config.packager_out_dir, ".local_research_outputs") / "broker_export_package_decision.json"
    packager_decision = None
    if decision_path.exists():
        try:
            packager_decision = json.loads(decision_path.read_text(encoding="utf-8")).get("decision")
        except json.JSONDecodeError:
            packager_decision = None
    return {
        "created_at_utc": created_at_utc,
        "command": command,
        "returncode": completed.returncode,
        "stdout_excerpt": completed.stdout[:1000],
        "stderr_excerpt": completed.stderr[:1000],
        "decision_path": str(decision_path),
        "packager_decision": packager_decision,
    }


def write_report_md(path: Path, report: dict[str, Any], decision: dict[str, Any]) -> None:
    path.write_text(
        "\n".join(
            [
                "# Broker Client Local Data Rescue Report",
                "",
                LAB_DECLARATION,
                "",
                "## Rescue summary",
                f"- decision: `{decision.get('decision')}`",
                f"- failure_reason: `{decision.get('failure_reason')}`",
                f"- empty_export_rows: `{report.get('empty_export_rows')}`",
                f"- local_inventory_rows: `{report.get('local_inventory_rows')}`",
                f"- candidate_5m_files: `{report.get('candidate_5m_file_count')}`",
                f"- rescued_rows: `{report.get('rescued_rows')}`",
                "",
                "## Boundary",
                "Read-only local broker-client market-data inspection only. No broker login, GUI control, QMT, xtdata, account, position, order, trade, fill, model training, labels, OrderIntent, Stable output, or advisory action was used.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def decide(
    *,
    rescued_generated: bool,
    quality: dict[str, Any],
    candidate_count: int,
    parser_attempts: Sequence[dict[str, Any]],
    skipped_for_safety_count: int,
    empty_export_frame: pd.DataFrame,
    packager_result: dict[str, Any] | None,
) -> tuple[str, str | None]:
    if rescued_generated and packager_result:
        if packager_result.get("packager_decision") == "BROKER_EXPORT_PACKAGE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION":
            return "BROKER_CLIENT_LOCAL_5M_RESCUE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION", None
        return "BROKER_CLIENT_LOCAL_5M_RESCUE_READY_FOR_MANUAL_INTAKE", None
    if rescued_generated:
        return "BROKER_CLIENT_LOCAL_5M_RESCUE_READY_FOR_MANUAL_INTAKE", None
    if skipped_for_safety_count and candidate_count == 0:
        return "BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_FORBIDDEN_ACCOUNT_OR_TRADE_FILES", "Only forbidden account/order/trade-like local paths were encountered."
    if candidate_count == 0:
        if not empty_export_frame.empty and (empty_export_frame["diagnosis"] == "header_only_empty_export").any():
            return "BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_EMPTY_EXPORT_ONLY", "Export files exist but contain headers only; no local 5m market-data candidate files were found."
        return "BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_NO_LOCAL_5M_FILES_FOUND", "No local 5m candidate files were found under authorized roots."
    if parser_attempts and not any(attempt.get("rows_parsed", 0) for attempt in parser_attempts):
        return "BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_UNSUPPORTED_LOCAL_FORMAT", "Candidate files were found but no supported OHLCV 5m records were parsed within bounded reads."
    if not quality.get("data_quality_passed", False):
        return "BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_DATA_QUALITY", "; ".join(quality.get("p0_blockers", []))
    return "BROKER_CLIENT_LOCAL_5M_RESCUE_BLOCKED_UNSUPPORTED_LOCAL_FORMAT", "No rescue output was generated."


def rescue_local_data(config: RescueConfig) -> dict[str, Any]:
    out_dir = resolve_under_repo(config.out_dir, ".local_research_outputs")
    rescued_dir = resolve_under_repo(config.rescued_dir, ".local_artifact_backup")
    raw_export_dir = resolve_under_repo(config.raw_export_dir, ".local_artifact_backup")
    out_dir.mkdir(parents=True, exist_ok=True)
    created_at_utc = utc_now()

    empty_exports = diagnose_empty_exports(config.export_roots)
    inventory, candidates = inventory_local_data_files(config.local_roots)
    empty_exports.to_csv(out_dir / "empty_export_diagnosis.csv", index=False, lineterminator="\n")
    inventory.to_csv(out_dir / "local_data_file_inventory.csv", index=False, lineterminator="\n")
    candidates.to_csv(out_dir / "candidate_5m_data_files.csv", index=False, lineterminator="\n")

    parsed_frames: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    parser_attempts: list[dict[str, Any]] = []
    for _, row in candidates.iterrows():
        path = Path(str(row["path"]))
        frame, parser_info = parse_candidate(path)
        parser_info = {
            "path": str(path),
            "file_name": path.name,
            "candidate_reason": row.get("candidate_reason", ""),
            **parser_info,
            "rows_parsed": int(len(frame)),
        }
        parser_attempts.append(parser_info)
        if not frame.empty:
            parsed_frames.append(frame)
            source_paths.append(path)

    rescued_frame = pd.concat(parsed_frames, ignore_index=True) if parsed_frames else pd.DataFrame(columns=STANDARD_COLUMNS)
    if not rescued_frame.empty:
        rescued_frame = rescued_frame.drop_duplicates().sort_values(["etf_code", "datetime"]).reset_index(drop=True)
    quality = validate_rescued_frame(rescued_frame)

    rescued_package: dict[str, Any] | None = None
    packager_result: dict[str, Any] | None = None
    if quality["data_quality_passed"]:
        rescued_package = write_rescued_package(rescued_dir, raw_export_dir, rescued_frame, source_paths, created_at_utc)
        if config.run_packager:
            packager_result = run_packager(config, raw_export_dir, created_at_utc)

    skipped_for_safety_count = int(inventory["skipped_for_safety"].sum()) if "skipped_for_safety" in inventory else 0
    decision_name, failure_reason = decide(
        rescued_generated=rescued_package is not None,
        quality=quality,
        candidate_count=int(len(candidates)),
        parser_attempts=parser_attempts,
        skipped_for_safety_count=skipped_for_safety_count,
        empty_export_frame=empty_exports,
        packager_result=packager_result,
    )
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_broker_client_local_data_rescue_report",
        "created_at_utc": created_at_utc,
        "authorized_local_roots": [str(resolve_input_path(path)) for path in config.local_roots],
        "export_roots": [str(resolve_input_path(path)) for path in config.export_roots],
        "empty_export_rows": int(len(empty_exports)),
        "local_inventory_rows": int(len(inventory)),
        "candidate_5m_file_count": int(len(candidates)),
        "skipped_for_safety_count": skipped_for_safety_count,
        "parser_attempts": parser_attempts,
        "rescued_rows": int(len(rescued_frame)),
        "rescued_etf_count": int(rescued_frame["etf_code"].nunique()) if not rescued_frame.empty else 0,
        "rescued_package": rescued_package,
        "packager_result": packager_result,
        "data_quality": quality,
        **BOUNDARY_FIELDS,
    }
    decision = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "aetfq3_intraday_broker_client_local_data_rescue_decision",
        "created_at_utc": created_at_utc,
        "decision": decision_name,
        "failure_reason": failure_reason,
        "rescued_csv_generated": rescued_package is not None,
        "manual_intake_ready": decision_name in {
            "BROKER_CLIENT_LOCAL_5M_RESCUE_READY_FOR_MANUAL_INTAKE",
            "BROKER_CLIENT_LOCAL_5M_RESCUE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION",
        },
        "fixed_shortlist_oop_validation_ready": decision_name == "BROKER_CLIENT_LOCAL_5M_RESCUE_VALIDATOR_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION",
        "next_allowed_action": "Report readiness only; do not run OOP validation in this task.",
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "broker_client_local_data_rescue_report.json", report)
    write_json(out_dir / "rescue_decision.json", decision)
    write_report_md(out_dir / "broker_client_local_data_rescue_report.md", report, decision)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lab-only local broker-client ETF 5m data rescue and empty export diagnosis")
    parser.add_argument("--raw-export-dir", type=Path, default=DEFAULT_RAW_EXPORT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--rescued-dir", type=Path, default=DEFAULT_RESCUED_DIR)
    parser.add_argument("--manual-inbox", type=Path, default=DEFAULT_MANUAL_INBOX)
    parser.add_argument("--packager-out-dir", type=Path, default=DEFAULT_PACKAGER_OUT_DIR)
    parser.add_argument("--validator-out-dir", type=Path, default=DEFAULT_VALIDATOR_OUT_DIR)
    parser.add_argument("--skip-packager", action="store_true")
    parser.add_argument("--skip-packager-validator", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = RescueConfig(
        local_roots=[*DEFAULT_CLIENT_ROOTS, args.raw_export_dir],
        export_roots=[Path(r"E:\CGS\T0002\export"), Path(r"E:\XGS\T0002\export"), args.raw_export_dir],
        raw_export_dir=args.raw_export_dir,
        out_dir=args.out_dir,
        rescued_dir=args.rescued_dir,
        manual_inbox=args.manual_inbox,
        packager_out_dir=args.packager_out_dir,
        validator_out_dir=args.validator_out_dir,
        run_packager=not args.skip_packager,
        run_packager_validator=not args.skip_packager_validator,
    )
    try:
        result = rescue_local_data(config)
    except BrokerClientLocalDataRescueError as exc:
        print(f"BROKER_CLIENT_LOCAL_DATA_RESCUE_ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
