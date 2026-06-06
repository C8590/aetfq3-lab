from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


REQUIRED_FIELDS = {
    "manifest_version",
    "sample_type",
    "sample_path",
    "sample_path_type",
    "source_kind",
    "source_description",
    "generated_at",
    "generated_by",
    "human_authorized",
    "authorized_by",
    "authorization_scope",
    "uses_stable_bundle",
    "stable_bundle_path",
    "stable_bundle_commit",
    "stable_bundle_snapshot_date",
    "data_time_start",
    "data_time_end",
    "row_count",
    "symbol_count",
    "sector_count",
    "contains_future_labels",
    "future_label_columns",
    "feature_columns",
    "forbidden_feature_columns",
    "has_future_leakage_check",
    "allowed_for",
    "training_allowed",
    "stable_effect_allowed",
    "advisory_only",
    "affects_stable_trading",
    "contains_secret",
    "contains_live_order",
    "contains_order_intent",
    "qmt_related",
    "review_checklist_passed",
    "notes",
}

SAMPLE_TYPES = {"false_downgrade", "sector_internal_ranking"}
SAMPLE_PATH_TYPES = {"local_ignored", "external_readonly", "stable_bundle_readonly"}
SOURCE_KINDS = {
    "manual_small_sample",
    "lab_generated_small_sample",
    "stable_bundle_extract",
    "external_authorized_extract",
}
ALLOWED_FOR_VALUES = {"schema_validation_only", "dry_validation_only", "mock_validation_only"}
LOCAL_IGNORED_ROOT = Path(".local_research_outputs/aetfq3_lab")


@dataclass
class IntakeCheckResult:
    ok: bool = True
    p0_errors: list[str] = field(default_factory=list)
    p1_warnings: list[str] = field(default_factory=list)

    def fail(self, message: str) -> None:
        self.ok = False
        self.p0_errors.append(message)

    def warn(self, message: str) -> None:
        self.p1_warnings.append(message)


def load_manifest(path: Path, result: IntakeCheckResult) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        result.fail(f"Manifest JSON parse failed: {exc}")
        return {}
    except OSError as exc:
        result.fail(f"Manifest cannot be read: {exc}")
        return {}

    if not isinstance(value, dict):
        result.fail("Manifest root must be a JSON object")
        return {}
    return value


def check_manifest(manifest_path: Path, repo_root: Path | None = None) -> IntakeCheckResult:
    result = IntakeCheckResult()
    repo_root = (repo_root or Path.cwd()).resolve()
    manifest = load_manifest(manifest_path, result)
    if not result.ok:
        return result

    check_required_fields(manifest, result)
    check_enum_fields(manifest, result)
    check_array_fields(manifest, result)
    check_boolean_boundaries(manifest, result)
    check_sample_path(manifest, repo_root, result)
    check_stable_bundle_metadata(manifest, result)
    check_feature_leakage(manifest, result)
    check_qmt_boundary(manifest, result)
    return result


def check_required_fields(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    missing = sorted(REQUIRED_FIELDS - set(manifest))
    if missing:
        result.fail("Missing required fields: " + ", ".join(missing))


def check_enum_fields(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    require_enum(manifest, "sample_type", SAMPLE_TYPES, result)
    require_enum(manifest, "sample_path_type", SAMPLE_PATH_TYPES, result)
    require_enum(manifest, "source_kind", SOURCE_KINDS, result)


def require_enum(
    manifest: dict[str, Any],
    field_name: str,
    allowed_values: set[str],
    result: IntakeCheckResult,
) -> None:
    value = manifest.get(field_name)
    if value is None:
        return
    if value not in allowed_values:
        result.fail(f"{field_name} must be one of {sorted(allowed_values)}: {value}")


def check_array_fields(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    for field_name in ("allowed_for", "future_label_columns", "feature_columns", "forbidden_feature_columns"):
        value = manifest.get(field_name)
        if value is not None and not isinstance(value, list):
            result.fail(f"{field_name} must be an array")

    allowed_for = manifest.get("allowed_for")
    if isinstance(allowed_for, list):
        for item in allowed_for:
            if item == "training" or item not in ALLOWED_FOR_VALUES:
                result.fail(f"allowed_for contains unsupported or prohibited value: {item}")


def check_boolean_boundaries(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    require_bool_value(manifest, "human_authorized", True, result)
    require_bool_value(manifest, "training_allowed", False, result)
    require_bool_value(manifest, "stable_effect_allowed", False, result)
    require_bool_value(manifest, "advisory_only", True, result)
    require_bool_value(manifest, "affects_stable_trading", False, result)
    require_bool_value(manifest, "contains_secret", False, result)
    require_bool_value(manifest, "contains_live_order", False, result)
    require_bool_value(manifest, "contains_order_intent", False, result)


def require_bool_value(
    manifest: dict[str, Any],
    field_name: str,
    expected: bool,
    result: IntakeCheckResult,
) -> None:
    value = manifest.get(field_name)
    if value is None:
        return
    if value is not expected:
        result.fail(f"{field_name} must be {str(expected).lower()}")


def check_sample_path(manifest: dict[str, Any], repo_root: Path, result: IntakeCheckResult) -> None:
    raw_sample_path = manifest.get("sample_path")
    if not isinstance(raw_sample_path, str) or not raw_sample_path:
        result.fail("sample_path must be a non-empty string")
        return

    sample_path = Path(raw_sample_path)
    resolved_sample_path = sample_path if sample_path.is_absolute() else repo_root / sample_path
    path_may_not_exist = manifest.get("path_may_not_exist_for_template", False)
    if path_may_not_exist is not True and not resolved_sample_path.exists():
        result.fail(f"sample_path does not exist: {raw_sample_path}")

    if manifest.get("sample_path_type") == "local_ignored":
        local_root = (repo_root / LOCAL_IGNORED_ROOT).resolve()
        try:
            resolved_sample_path.resolve().relative_to(local_root)
        except ValueError:
            result.fail("local_ignored sample_path must be under .local_research_outputs/aetfq3_lab/")


def check_stable_bundle_metadata(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    if manifest.get("uses_stable_bundle") is not True:
        return

    stable_bundle_path = manifest.get("stable_bundle_path")
    stable_bundle_commit = manifest.get("stable_bundle_commit")
    stable_bundle_snapshot_date = manifest.get("stable_bundle_snapshot_date")
    authorization_scope = manifest.get("authorization_scope")

    if not stable_bundle_path:
        result.fail("uses_stable_bundle=true requires stable_bundle_path")
    if not stable_bundle_commit and not stable_bundle_snapshot_date:
        result.fail("uses_stable_bundle=true requires stable_bundle_commit or stable_bundle_snapshot_date")
    if not authorization_scope:
        result.fail("uses_stable_bundle=true requires authorization_scope")


def check_feature_leakage(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    feature_columns = string_set(manifest.get("feature_columns"))
    future_label_columns = string_set(manifest.get("future_label_columns"))
    forbidden_feature_columns = string_set(manifest.get("forbidden_feature_columns"))

    leaked_future_labels = sorted(feature_columns & future_label_columns)
    if leaked_future_labels:
        result.fail("feature_columns contains future_label_columns: " + ", ".join(leaked_future_labels))

    leaked_forbidden = sorted(feature_columns & forbidden_feature_columns)
    if leaked_forbidden:
        result.fail("feature_columns contains forbidden_feature_columns: " + ", ".join(leaked_forbidden))

    prefix_leaks = sorted(column for column in feature_columns if is_future_or_label_like(column))
    if prefix_leaks:
        result.fail("feature_columns contains future/label-like columns: " + ", ".join(prefix_leaks))

    if manifest.get("has_future_leakage_check") is not True:
        result.warn("has_future_leakage_check is not true; manual review required before dry validation")


def string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def is_future_or_label_like(column: str) -> bool:
    return (
        column.startswith("future_")
        or column.startswith("max_drawdown_")
        or column.startswith("best_in_sector_")
        or column.startswith("top_quantile_")
        or column.endswith("_label")
        or column in {
            "false_downgrade_1d",
            "false_downgrade_3d",
            "false_downgrade_lock3",
            "true_downgrade",
            "neutral_downgrade",
            "avoid_in_sector",
        }
    )


def check_qmt_boundary(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    if manifest.get("qmt_related") is not True:
        return

    qmt_access_mode = manifest.get("qmt_access_mode")
    if qmt_access_mode in {"mock", "readonly"} and manifest.get("human_authorized") is True:
        result.warn("qmt_related=true is P1 only because qmt_access_mode is mock/readonly and authorized")
    else:
        result.fail("qmt_related=true is P0 unless readonly/mock and human-authorized")


def print_result(result: IntakeCheckResult) -> None:
    if result.ok:
        print("OK sample_intake_manifest_valid=true")
    else:
        print("FAILED sample_intake_manifest_valid=false")

    for error in result.p0_errors:
        print(f"P0 {error}")
    for warning in result.p1_warnings:
        print(f"P1 {warning}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate AETF Q3 Lab table ML sample intake manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_manifest(args.manifest)
    print_result(result)
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
