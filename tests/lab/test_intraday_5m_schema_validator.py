from __future__ import annotations

import csv
from pathlib import Path

from tools.lab.intraday_5m_schema_validator import validate_schema


VALID_MANIFEST = Path("tests/fixtures/aetfq3_lab/mock_intraday_5m_manifest.json")
VALID_CSV = Path("tests/fixtures/aetfq3_lab/mock_intraday_5m_samples.csv")


def read_rows() -> tuple[list[str], list[dict[str, str]]]:
    with VALID_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), [dict(row) for row in reader]


def write_csv(tmp_path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> Path:
    path = tmp_path / "sample.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_valid_mock_csv_passed() -> None:
    result = validate_schema(VALID_CSV, VALID_MANIFEST)

    assert result.ok
    assert result.rows_checked == 48
    assert result.min_bars_per_etf_day == 12


def test_missing_required_column_fails(tmp_path: Path) -> None:
    fieldnames, rows = read_rows()
    fieldnames.remove("vwap")
    for row in rows:
        row.pop("vwap")
    path = write_csv(tmp_path, fieldnames, rows)

    result = validate_schema(path, VALID_MANIFEST)

    assert not result.ok
    assert any("missing required columns" in blocker for blocker in result.p0_blockers)


def test_bad_ohlc_fails(tmp_path: Path) -> None:
    fieldnames, rows = read_rows()
    rows[0]["high"] = "9.00"
    path = write_csv(tmp_path, fieldnames, rows)

    result = validate_schema(path, VALID_MANIFEST)

    assert not result.ok
    assert any("bad OHLC" in blocker for blocker in result.p0_blockers)


def test_insufficient_bars_fails(tmp_path: Path) -> None:
    fieldnames, rows = read_rows()
    rows = [row for row in rows if not (row["trade_date"] == "2026-06-01" and row["etf_code"] == "MOCK001" and row["bar_index"] == "12")]
    path = write_csv(tmp_path, fieldnames, rows)

    result = validate_schema(path, VALID_MANIFEST)

    assert not result.ok
    assert any("insufficient bars" in blocker for blocker in result.p0_blockers)
