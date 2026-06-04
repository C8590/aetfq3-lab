from __future__ import annotations

import csv
from pathlib import Path

from tools.lab.table_ml_schema_validator import (
    validate_chronological_split,
    validate_no_group_split_leakage,
    validate_sample,
)


FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "aetfq3_lab"
FALSE_DOWNGRADE_FIXTURE = FIXTURE_DIR / "mock_false_downgrade_samples.csv"
SECTOR_INTERNAL_FIXTURE = FIXTURE_DIR / "mock_sector_internal_ranking_samples.csv"


def test_valid_false_downgrade_mock_passes():
    result = validate_sample("false_downgrade", FALSE_DOWNGRADE_FIXTURE)

    assert result.ok
    assert result.rows_checked == 8
    assert result.p0_errors == []


def test_valid_sector_internal_ranking_mock_passes():
    result = validate_sample("sector_internal_ranking", SECTOR_INTERNAL_FIXTURE)

    assert result.ok
    assert result.rows_checked == 8
    assert result.p0_errors == []
    assert result.p1_warnings == []


def test_missing_required_field_fails(tmp_path):
    broken = tmp_path / "missing_required.csv"
    write_csv_without_column(FALSE_DOWNGRADE_FIXTURE, broken, "ml_action")

    result = validate_sample("false_downgrade", broken)

    assert not result.ok
    assert any("Missing required fields: ml_action" in error for error in result.p0_errors)


def test_duplicate_primary_key_fails(tmp_path):
    broken = tmp_path / "duplicate_key.csv"
    duplicate_first_data_row(FALSE_DOWNGRADE_FIXTURE, broken)

    result = validate_sample("false_downgrade", broken)

    assert not result.ok
    assert any("Duplicate primary key" in error for error in result.p0_errors)


def test_forbidden_future_feature_column_fails():
    result = validate_sample(
        "false_downgrade",
        FALSE_DOWNGRADE_FIXTURE,
        feature_columns=["v2_score", "future_return_3d", "sector_rank"],
    )

    assert not result.ok
    assert any("Forbidden future/label feature column: future_return_3d" in error for error in result.p0_errors)


def test_sector_internal_ranking_group_split_leakage_fails():
    train_rows = [{"ranking_group_id": "2026-01-02_tech"}, {"ranking_group_id": "2026-01-02_health"}]
    validation_rows = [{"ranking_group_id": "2026-01-02_tech"}, {"ranking_group_id": "2026-01-03_health"}]

    result = validate_no_group_split_leakage(train_rows, validation_rows)

    assert not result.ok
    assert result.p0_errors == ["ranking_group_id appears in both train and validation: 2026-01-02_tech"]


def test_ranking_group_single_member_warns(tmp_path):
    broken = tmp_path / "single_member_group.csv"
    write_first_rows(SECTOR_INTERNAL_FIXTURE, broken, row_count=1)

    result = validate_sample("sector_internal_ranking", broken)

    assert result.ok
    assert result.p1_warnings == ["ranking_group_id has fewer than 2 ETFs: 2026-01-02_tech"]


def test_chronological_split_requires_train_before_validation():
    result = validate_chronological_split("2026-01-03", "2026-01-03")

    assert not result.ok
    assert result.p0_errors == ["train_end_date must be earlier than valid_start_date"]


def write_csv_without_column(source: Path, target: Path, column_to_drop: str) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = [field for field in reader.fieldnames or [] if field != column_to_drop]
        rows = [{field: row[field] for field in fieldnames} for row in reader]

    with target.open("w", encoding="utf-8", newline="") as target_handle:
        writer = csv.DictWriter(target_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def duplicate_first_data_row(source: Path, target: Path) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    rows.append(dict(rows[0]))

    with target.open("w", encoding="utf-8", newline="") as target_handle:
        writer = csv.DictWriter(target_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_first_rows(source: Path, target: Path, row_count: int) -> None:
    with source.open("r", encoding="utf-8-sig", newline="") as source_handle:
        reader = csv.DictReader(source_handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)[:row_count]

    with target.open("w", encoding="utf-8", newline="") as target_handle:
        writer = csv.DictWriter(target_handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
