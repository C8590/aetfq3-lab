from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_dry_validation_intake_checker import (
    EXPLICIT_FORBIDDEN_FEATURES,
    is_forbidden_feature_name,
    load_manifest,
    IntakeCheckResult,
)


REQUIRED_FIELDS = {
    "trade_date",
    "datetime",
    "etf_code",
    "etf_name",
    "sector",
    "bar_index",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
}

NUMERIC_FIELDS = {"bar_index", "open", "high", "low", "close", "volume", "amount", "vwap"}


@dataclass
class SchemaValidationResult:
    ok: bool = True
    rows_checked: int = 0
    trade_date_count: int = 0
    etf_count: int = 0
    min_bars_per_etf_day: int = 0
    p0_blockers: list[str] = field(default_factory=list)
    p1_warnings: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.p0_blockers.append(message)

    def warn(self, message: str) -> None:
        self.p1_warnings.append(message)

    def to_summary(self) -> dict[str, Any]:
        return {
            "status": "passed" if self.ok else "failed",
            "rows_checked": self.rows_checked,
            "trade_date_count": self.trade_date_count,
            "etf_count": self.etf_count,
            "min_bars_per_etf_day": self.min_bars_per_etf_day,
            "schema_passed": self.ok,
            "p0_blockers": self.p0_blockers,
            "p1_warnings": self.p1_warnings,
        }


def validate_schema(input_path: Path, manifest_path: Path, repo_root: Path | None = None) -> SchemaValidationResult:
    repo_root = (repo_root or Path.cwd()).resolve()
    result = SchemaValidationResult()
    manifest_result = IntakeCheckResult()
    manifest = load_manifest(manifest_path, manifest_result)
    if manifest_result.p0_blockers:
        for blocker in manifest_result.p0_blockers:
            result.fail(blocker)
        return result

    fieldnames, rows = read_rows(input_path, result)
    if result.p0_blockers:
        return result

    result.rows_checked = len(rows)
    missing = sorted(REQUIRED_FIELDS - set(fieldnames))
    if missing:
        result.fail("missing required columns: " + ", ".join(missing))
        return result
    if not rows:
        result.fail("input CSV has no rows")
        return result

    check_feature_columns(manifest, fieldnames, result)
    check_rows(rows, result)
    check_group_bar_counts(rows, result)
    check_manifest_counts(manifest, result)
    return result


def read_rows(path: Path, result: SchemaValidationResult) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            rows = [dict(row) for row in reader]
    except OSError as exc:
        result.fail(f"input CSV cannot be read: {exc}")
        return [], []
    return fieldnames, rows


def check_feature_columns(
    manifest: dict[str, Any],
    fieldnames: Sequence[str],
    result: SchemaValidationResult,
) -> None:
    features = [item for item in manifest.get("feature_columns", []) if isinstance(item, str)]
    forbidden = set(manifest.get("forbidden_feature_columns", [])) | EXPLICIT_FORBIDDEN_FEATURES
    missing_features = sorted(set(features) - set(fieldnames))
    if missing_features:
        result.fail("feature_columns missing from CSV: " + ", ".join(missing_features))

    leaked = sorted(column for column in features if column in forbidden or is_forbidden_feature_name(column))
    if leaked:
        result.fail("feature_columns contains forbidden columns: " + ", ".join(leaked))


def check_rows(rows: Sequence[dict[str, str]], result: SchemaValidationResult) -> None:
    for index, row in enumerate(rows, start=2):
        numbers = parse_numeric_row(row, index, result)
        if not numbers:
            continue
        if numbers["high"] < max(numbers["open"], numbers["close"]):
            result.fail(f"bad OHLC at CSV row {index}: high < max(open, close)")
        if numbers["low"] > min(numbers["open"], numbers["close"]):
            result.fail(f"bad OHLC at CSV row {index}: low > min(open, close)")
        if numbers["volume"] < 0:
            result.fail(f"bad volume at CSV row {index}: volume < 0")
        if numbers["amount"] < 0:
            result.fail(f"bad amount at CSV row {index}: amount < 0")


def parse_numeric_row(row: dict[str, str], row_number: int, result: SchemaValidationResult) -> dict[str, float] | None:
    parsed: dict[str, float] = {}
    for field_name in NUMERIC_FIELDS:
        try:
            parsed[field_name] = float(row.get(field_name, ""))
        except ValueError:
            result.fail(f"{field_name} must be numeric at CSV row {row_number}: {row.get(field_name, '')}")
            return None
    return parsed


def check_group_bar_counts(rows: Sequence[dict[str, str]], result: SchemaValidationResult) -> None:
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    trade_dates: set[str] = set()
    etfs: set[str] = set()
    for row in rows:
        trade_date = row.get("trade_date", "")
        etf_code = row.get("etf_code", "")
        trade_dates.add(trade_date)
        etfs.add(etf_code)
        groups.setdefault((trade_date, etf_code), []).append(row)

    result.trade_date_count = len(trade_dates)
    result.etf_count = len(etfs)
    counts = [len(group_rows) for group_rows in groups.values()]
    result.min_bars_per_etf_day = min(counts) if counts else 0

    for key, group_rows in sorted(groups.items()):
        if len(group_rows) < 12:
            result.fail(f"insufficient bars for {key[0]}/{key[1]}: {len(group_rows)} < 12")
        bar_indices = [int(float(row["bar_index"])) for row in group_rows]
        if bar_indices != sorted(bar_indices):
            result.fail(f"bar_index is not monotonic for {key[0]}/{key[1]}")


def check_manifest_counts(manifest: dict[str, Any], result: SchemaValidationResult) -> None:
    expected_rows = manifest.get("row_count")
    if isinstance(expected_rows, int) and expected_rows != result.rows_checked:
        result.warn(f"manifest row_count={expected_rows} differs from CSV rows={result.rows_checked}")
    expected_etfs = manifest.get("etf_count")
    if isinstance(expected_etfs, int) and expected_etfs != result.etf_count:
        result.warn(f"manifest etf_count={expected_etfs} differs from CSV etf_count={result.etf_count}")
    expected_dates = manifest.get("trade_date_count")
    if isinstance(expected_dates, int) and expected_dates != result.trade_date_count:
        result.warn(
            f"manifest trade_date_count={expected_dates} differs from CSV trade_date_count={result.trade_date_count}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Lab-only synthetic intraday 5m CSV schema.")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_schema(args.input, args.manifest)
    print(json.dumps(result.to_summary(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
