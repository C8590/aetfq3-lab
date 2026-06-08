from __future__ import annotations

import argparse
import json
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
    "uses_qmt_export",
    "qmt_export_path",
    "qmt_mode",
    "data_time_start",
    "data_time_end",
    "row_count",
    "etf_count",
    "trade_date_count",
    "bar_count_per_etf_day",
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

ALLOWED_SAMPLE_PATH_TYPES = {"repo_mock_fixture", "local_ignored", "external_readonly"}
ALLOWED_SOURCE_KINDS = {"synthetic_mock", "manual_mock", "external_readonly_mock"}
ALLOWED_FOR = {"dry_validation_only", "mock_validation_only"}
ALLOWED_QMT_MODES = {"readonly", "mock", "export_only"}
EXPLICIT_FORBIDDEN_FEATURES = {
    "max_drawdown_3d",
    "execution_return_to_close",
    "execution_return_to_next_open",
    "execution_drawdown_after_entry",
}


@dataclass
class IntakeCheckResult:
    ok: bool = True
    p0_blockers: list[str] = field(default_factory=list)
    p1_warnings: list[str] = field(default_factory=list)
    manifest: dict[str, Any] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.ok = False
        self.p0_blockers.append(message)

    def warn(self, message: str) -> None:
        self.p1_warnings.append(message)

    def to_summary(self) -> dict[str, Any]:
        return {
            "status": "passed" if self.ok else "failed",
            "intake_passed": self.ok,
            "sample_type": self.manifest.get("sample_type"),
            "sample_path": self.manifest.get("sample_path"),
            "training_allowed": self.manifest.get("training_allowed"),
            "stable_effect_allowed": self.manifest.get("stable_effect_allowed"),
            "advisory_only": self.manifest.get("advisory_only"),
            "affects_stable_trading": self.manifest.get("affects_stable_trading"),
            "contains_secret": self.manifest.get("contains_secret"),
            "contains_live_order": self.manifest.get("contains_live_order"),
            "contains_order_intent": self.manifest.get("contains_order_intent"),
            "qmt_related": self.manifest.get("qmt_related"),
            "p0_blockers": self.p0_blockers,
            "p1_warnings": self.p1_warnings,
        }


def load_manifest(path: Path, result: IntakeCheckResult) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.fail(f"manifest JSON parse failed: {exc}")
        return {}
    except OSError as exc:
        result.fail(f"manifest cannot be read: {exc}")
        return {}
    if not isinstance(payload, dict):
        result.fail("manifest root must be a JSON object")
        return {}
    return payload


def check_manifest(manifest_path: Path, repo_root: Path | None = None) -> IntakeCheckResult:
    result = IntakeCheckResult()
    repo_root = (repo_root or Path.cwd()).resolve()
    manifest = load_manifest(manifest_path, result)
    result.manifest = manifest
    if not result.ok:
        return result

    check_required_fields(manifest, result)
    check_enums(manifest, result)
    check_array_fields(manifest, result)
    check_boundaries(manifest, result)
    check_sample_path(manifest, repo_root, result)
    check_feature_leakage(manifest, result)
    check_qmt_mode(manifest, result)
    return result


def check_required_fields(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    missing = sorted(REQUIRED_FIELDS - set(manifest))
    if missing:
        result.fail("missing required fields: " + ", ".join(missing))


def check_enums(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    if manifest.get("sample_type") != "intraday_5m":
        result.fail("sample_type must be intraday_5m")
    require_enum(manifest, "sample_path_type", ALLOWED_SAMPLE_PATH_TYPES, result)
    require_enum(manifest, "source_kind", ALLOWED_SOURCE_KINDS, result)


def require_enum(
    manifest: dict[str, Any],
    field_name: str,
    allowed_values: set[str],
    result: IntakeCheckResult,
) -> None:
    value = manifest.get(field_name)
    if value is not None and value not in allowed_values:
        result.fail(f"{field_name} must be one of {sorted(allowed_values)}: {value}")


def check_array_fields(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    for field_name in ("allowed_for", "future_label_columns", "feature_columns", "forbidden_feature_columns"):
        value = manifest.get(field_name)
        if not isinstance(value, list):
            result.fail(f"{field_name} must be an array")

    allowed_for = manifest.get("allowed_for")
    if isinstance(allowed_for, list):
        invalid = [item for item in allowed_for if item not in ALLOWED_FOR]
        if invalid:
            result.fail("allowed_for contains unsupported values: " + ", ".join(map(str, invalid)))


def check_boundaries(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    require_bool(manifest, "human_authorized", True, result)
    require_bool(manifest, "training_allowed", False, result)
    require_bool(manifest, "stable_effect_allowed", False, result)
    require_bool(manifest, "advisory_only", True, result)
    require_bool(manifest, "affects_stable_trading", False, result)
    require_bool(manifest, "contains_secret", False, result)
    require_bool(manifest, "contains_live_order", False, result)
    require_bool(manifest, "contains_order_intent", False, result)
    require_bool(manifest, "has_future_leakage_check", True, result)
    require_bool(manifest, "review_checklist_passed", True, result)


def require_bool(
    manifest: dict[str, Any],
    field_name: str,
    expected: bool,
    result: IntakeCheckResult,
) -> None:
    value = manifest.get(field_name)
    if value is not expected:
        result.fail(f"{field_name} must be {str(expected).lower()}")


def check_sample_path(manifest: dict[str, Any], repo_root: Path, result: IntakeCheckResult) -> None:
    raw_path = manifest.get("sample_path")
    if not isinstance(raw_path, str) or not raw_path:
        result.fail("sample_path must be a non-empty string")
        return
    sample_path = Path(raw_path)
    resolved = sample_path if sample_path.is_absolute() else repo_root / sample_path
    if not resolved.exists():
        result.fail(f"sample_path does not exist: {raw_path}")


def check_feature_leakage(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    feature_columns = string_set(manifest.get("feature_columns"))
    future_label_columns = string_set(manifest.get("future_label_columns"))
    forbidden_columns = string_set(manifest.get("forbidden_feature_columns")) | EXPLICIT_FORBIDDEN_FEATURES

    leaked_future_labels = sorted(feature_columns & future_label_columns)
    if leaked_future_labels:
        result.fail("feature_columns intersects future_label_columns: " + ", ".join(leaked_future_labels))

    leaked_forbidden = sorted(feature_columns & forbidden_columns)
    if leaked_forbidden:
        result.fail("feature_columns intersects forbidden_feature_columns: " + ", ".join(leaked_forbidden))

    pattern_leaks = sorted(column for column in feature_columns if is_forbidden_feature_name(column))
    if pattern_leaks:
        result.fail("feature_columns contains future/label/outcome fields: " + ", ".join(pattern_leaks))


def string_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if isinstance(item, str)}


def is_forbidden_feature_name(column: str) -> bool:
    return (
        column.startswith("future_")
        or column.endswith("_label")
        or column in EXPLICIT_FORBIDDEN_FEATURES
    )


def check_qmt_mode(manifest: dict[str, Any], result: IntakeCheckResult) -> None:
    if manifest.get("qmt_related") is not True:
        return
    qmt_mode = manifest.get("qmt_mode")
    if qmt_mode not in ALLOWED_QMT_MODES:
        result.fail(f"qmt_related=true requires qmt_mode in {sorted(ALLOWED_QMT_MODES)}: {qmt_mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate Lab-only intraday dry validation manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_manifest(args.manifest)
    print(json.dumps(result.to_summary(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
