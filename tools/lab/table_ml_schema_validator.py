from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence


FALSE_DOWNGRADE_REQUIRED_FIELDS = {
    "trade_date",
    "etf_code",
    "etf_name",
    "sector",
    "v2_action",
    "ml_action",
    "model_version",
    "feature_version",
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
    "false_downgrade_1d",
    "false_downgrade_3d",
    "false_downgrade_lock3",
    "true_downgrade",
    "neutral_downgrade",
}

SECTOR_INTERNAL_REQUIRED_FIELDS = {
    "trade_date",
    "sector",
    "etf_code",
    "etf_name",
    "ranking_group_id",
    "model_version",
    "feature_version",
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
    "best_in_sector_1d",
    "best_in_sector_3d",
    "top_quantile_in_sector_3d",
    "avoid_in_sector",
    "pairwise_outperform_label",
}

FALSE_DOWNGRADE_LABEL_FIELDS = {
    "false_downgrade_1d",
    "false_downgrade_3d",
    "false_downgrade_lock3",
    "true_downgrade",
    "neutral_downgrade",
}

SECTOR_INTERNAL_LABEL_FIELDS = {
    "best_in_sector_1d",
    "best_in_sector_3d",
    "top_quantile_in_sector_3d",
    "avoid_in_sector",
    "pairwise_outperform_label",
}

COMMON_FORBIDDEN_FEATURE_FIELDS = {
    "future_best_etf_code",
    "future_sector_rank",
    "future_etf_rank",
    "pairwise_outperform_label",
}

FALSE_DOWNGRADE_FORBIDDEN_FEATURE_FIELDS = FALSE_DOWNGRADE_LABEL_FIELDS | {
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
}

SECTOR_INTERNAL_FORBIDDEN_FEATURE_FIELDS = SECTOR_INTERNAL_LABEL_FIELDS | {
    "future_return_1d",
    "future_return_3d",
    "max_drawdown_3d",
}

V2_ACTION_VALUES = {"PROBE", "BUY", "NO_BUY", "OBSERVE"}
ML_ACTION_VALUES = {"ML_DOWNGRADED", "AVOID", "NO_BUY", "KEEP_ORIGINAL", "UPGRADE_PROBE"}
BOOL_VALUES = {"0", "1", "true", "false", "True", "False", "TRUE", "FALSE"}


@dataclass
class ValidationResult:
    ok: bool = True
    p0_errors: list[str] = field(default_factory=list)
    p1_warnings: list[str] = field(default_factory=list)
    rows_checked: int = 0

    def fail(self, message: str) -> None:
        self.ok = False
        self.p0_errors.append(message)

    def warn(self, message: str) -> None:
        self.p1_warnings.append(message)


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def validate_sample(
    sample_type: str,
    input_path: Path,
    feature_columns: Sequence[str] | None = None,
) -> ValidationResult:
    fieldnames, rows = read_csv_rows(input_path)
    result = ValidationResult(rows_checked=len(rows))

    if sample_type == "false_downgrade":
        validate_false_downgrade(fieldnames, rows, result)
    elif sample_type == "sector_internal_ranking":
        validate_sector_internal_ranking(fieldnames, rows, result)
    else:
        result.fail(f"Unsupported sample_type: {sample_type}")

    if feature_columns:
        validate_feature_columns(sample_type, feature_columns, result)

    return result


def validate_false_downgrade(
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
    result: ValidationResult,
) -> None:
    require_fields(FALSE_DOWNGRADE_REQUIRED_FIELDS, fieldnames, result)
    if result.p0_errors:
        return

    require_unique_key(rows, ("trade_date", "etf_code", "model_version", "feature_version"), result)
    validate_allowed_values(rows, "v2_action", V2_ACTION_VALUES, result)
    validate_allowed_values(rows, "ml_action", ML_ACTION_VALUES, result)
    validate_bool_labels(rows, FALSE_DOWNGRADE_LABEL_FIELDS, result)
    validate_any_label_present(rows, FALSE_DOWNGRADE_LABEL_FIELDS, result)


def validate_sector_internal_ranking(
    fieldnames: Sequence[str],
    rows: Sequence[dict[str, str]],
    result: ValidationResult,
) -> None:
    require_fields(SECTOR_INTERNAL_REQUIRED_FIELDS, fieldnames, result)
    if result.p0_errors:
        return

    require_unique_key(rows, ("trade_date", "sector", "etf_code", "model_version", "feature_version"), result)
    validate_bool_labels(rows, SECTOR_INTERNAL_LABEL_FIELDS, result)
    validate_any_label_present(rows, SECTOR_INTERNAL_LABEL_FIELDS, result)
    validate_ranking_group_ids(rows, result)
    validate_group_member_counts(rows, result)


def require_fields(required: set[str], fieldnames: Sequence[str], result: ValidationResult) -> None:
    missing = sorted(required - set(fieldnames))
    if missing:
        result.fail("Missing required fields: " + ", ".join(missing))


def require_unique_key(
    rows: Sequence[dict[str, str]],
    key_fields: Sequence[str],
    result: ValidationResult,
) -> None:
    seen: dict[tuple[str, ...], int] = {}
    for index, row in enumerate(rows, start=2):
        key = tuple(row.get(field, "") for field in key_fields)
        if key in seen:
            result.fail(
                "Duplicate primary key at CSV rows "
                f"{seen[key]} and {index}: {dict(zip(key_fields, key, strict=True))}"
            )
        else:
            seen[key] = index


def validate_allowed_values(
    rows: Sequence[dict[str, str]],
    field_name: str,
    allowed_values: set[str],
    result: ValidationResult,
) -> None:
    for index, row in enumerate(rows, start=2):
        value = row.get(field_name, "")
        if value not in allowed_values:
            result.fail(f"Invalid {field_name} at CSV row {index}: {value}")


def validate_bool_labels(
    rows: Sequence[dict[str, str]],
    label_fields: Iterable[str],
    result: ValidationResult,
) -> None:
    for index, row in enumerate(rows, start=2):
        for field_name in label_fields:
            value = row.get(field_name, "")
            if value != "" and value not in BOOL_VALUES:
                result.fail(f"Label {field_name} must be 0/1 or boolean at CSV row {index}: {value}")


def validate_any_label_present(
    rows: Sequence[dict[str, str]],
    label_fields: Iterable[str],
    result: ValidationResult,
) -> None:
    label_list = list(label_fields)
    for index, row in enumerate(rows, start=2):
        if all(row.get(field_name, "") == "" for field_name in label_list):
            result.fail(f"All label fields are missing at CSV row {index}")


def expected_group_id_variants(trade_date: str, sector: str) -> set[str]:
    return {
        f"{trade_date}_{sector}",
        f"{trade_date}|{sector}",
        f"{trade_date}-{sector}",
        f"{trade_date}:{sector}",
    }


def validate_ranking_group_ids(rows: Sequence[dict[str, str]], result: ValidationResult) -> None:
    for index, row in enumerate(rows, start=2):
        group_id = row.get("ranking_group_id", "")
        variants = expected_group_id_variants(row.get("trade_date", ""), row.get("sector", ""))
        if group_id not in variants:
            result.fail(
                f"ranking_group_id mismatch at CSV row {index}: "
                f"{group_id} does not match trade_date + sector"
            )


def validate_group_member_counts(rows: Sequence[dict[str, str]], result: ValidationResult) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        group_id = row.get("ranking_group_id", "")
        counts[group_id] = counts.get(group_id, 0) + 1

    for group_id, count in sorted(counts.items()):
        if count < 2:
            result.warn(f"ranking_group_id has fewer than 2 ETFs: {group_id}")


def validate_feature_columns(
    sample_type: str,
    feature_columns: Sequence[str],
    result: ValidationResult,
) -> None:
    forbidden_exact = set(COMMON_FORBIDDEN_FEATURE_FIELDS)
    if sample_type == "false_downgrade":
        forbidden_exact |= FALSE_DOWNGRADE_FORBIDDEN_FEATURE_FIELDS
    elif sample_type == "sector_internal_ranking":
        forbidden_exact |= SECTOR_INTERNAL_FORBIDDEN_FEATURE_FIELDS

    for column in feature_columns:
        normalized = column.strip()
        if not normalized:
            continue
        if is_forbidden_feature(normalized, forbidden_exact):
            result.fail(f"Forbidden future/label feature column: {normalized}")


def is_forbidden_feature(column: str, forbidden_exact: set[str]) -> bool:
    return (
        column in forbidden_exact
        or column.startswith("future_")
        or column.startswith("max_drawdown_")
        or column.startswith("best_in_sector_")
        or column.startswith("top_quantile_")
        or column.startswith("true_future_")
    )


def validate_chronological_split(train_end_date: str, valid_start_date: str) -> ValidationResult:
    result = ValidationResult()
    train_end = parse_date(train_end_date, "train_end_date", result)
    valid_start = parse_date(valid_start_date, "valid_start_date", result)
    if train_end and valid_start and train_end >= valid_start:
        result.fail("train_end_date must be earlier than valid_start_date")
    return result


def validate_no_group_split_leakage(
    train_rows: Sequence[dict[str, str]],
    validation_rows: Sequence[dict[str, str]],
) -> ValidationResult:
    result = ValidationResult(rows_checked=len(train_rows) + len(validation_rows))
    train_groups = {row.get("ranking_group_id", "") for row in train_rows}
    validation_groups = {row.get("ranking_group_id", "") for row in validation_rows}
    leaked = sorted(group for group in train_groups & validation_groups if group)
    if leaked:
        result.fail("ranking_group_id appears in both train and validation: " + ", ".join(leaked))
    return result


def parse_date(value: str, field_name: str, result: ValidationResult) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        result.fail(f"{field_name} must use YYYY-MM-DD format: {value}")
        return None


def parse_feature_columns(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def print_result(result: ValidationResult) -> None:
    if result.ok:
        print(f"OK rows_checked={result.rows_checked}")
    else:
        print(f"FAILED rows_checked={result.rows_checked}")

    for error in result.p0_errors:
        print(f"P0 {error}")
    for warning in result.p1_warnings:
        print(f"P1 {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate AETF Q3 Lab table ML sample schema.")
    parser.add_argument("--sample-type", choices=["false_downgrade", "sector_internal_ranking"], required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--feature-columns", default="")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = validate_sample(
        sample_type=args.sample_type,
        input_path=args.input,
        feature_columns=parse_feature_columns(args.feature_columns),
    )
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
