from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


REQUIRED_FIELDS = {
    "sample_type",
    "feature_columns",
    "label_generated",
    "label_source_kind",
    "label_horizon",
    "label_generation_method",
    "label_columns",
    "outcome_columns",
    "label_status_column",
    "insufficient_future_window_policy",
    "feature_label_overlap_check",
    "label_generation_authorized",
    "supervised_training_allowed",
    "training_allowed",
    "stable_effect_allowed",
    "contains_order_intent",
    "contains_live_order",
    "contains_secret",
}
EXPLICIT_FORBIDDEN_FEATURES = {
    "max_drawdown_3d",
    "execution_return_to_close",
    "execution_return_to_next_open",
    "execution_drawdown_after_entry",
    "expected_3d_return",
    "expected_3d_drawdown",
}


@dataclass
class LabelManifestLeakageResult:
    manifest_path: str
    p0_blockers: list[str] = field(default_factory=list)
    p1_warnings: list[str] = field(default_factory=list)
    feature_count: int = 0
    label_count: int = 0
    outcome_count: int = 0

    @property
    def ok(self) -> bool:
        return not self.p0_blockers

    def fail(self, message: str) -> None:
        self.p0_blockers.append(message)

    def warn(self, message: str) -> None:
        self.p1_warnings.append(message)

    def to_summary(self) -> dict[str, Any]:
        return {
            "status": "passed" if self.ok else "failed",
            "manifest_path": self.manifest_path,
            "p0_blockers": self.p0_blockers,
            "p1_warnings": self.p1_warnings,
            "feature_count": self.feature_count,
            "label_count": self.label_count,
            "outcome_count": self.outcome_count,
            "boundary_passed": self.ok,
        }


def load_manifest(path: Path, result: LabelManifestLeakageResult) -> dict[str, Any]:
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


def check_manifest(manifest_path: Path) -> LabelManifestLeakageResult:
    result = LabelManifestLeakageResult(manifest_path=str(manifest_path))
    manifest = load_manifest(manifest_path, result)
    if not manifest:
        return result

    check_required_fields(manifest, result)
    feature_columns = string_list(manifest.get("feature_columns"))
    label_columns = string_list(manifest.get("label_columns"))
    outcome_columns = string_list(manifest.get("outcome_columns"))
    result.feature_count = len(feature_columns)
    result.label_count = len(label_columns)
    result.outcome_count = len(outcome_columns)

    check_array_fields(manifest, result)
    check_required_values(manifest, result)
    check_feature_leakage(feature_columns, label_columns, outcome_columns, result)
    check_boundary_flags(manifest, result)
    return result


def check_required_fields(manifest: dict[str, Any], result: LabelManifestLeakageResult) -> None:
    missing = sorted(REQUIRED_FIELDS - set(manifest))
    if missing:
        result.fail("missing required fields: " + ", ".join(missing))


def check_array_fields(manifest: dict[str, Any], result: LabelManifestLeakageResult) -> None:
    for field_name in ("feature_columns", "label_columns", "outcome_columns"):
        value = manifest.get(field_name)
        if not isinstance(value, list):
            result.fail(f"{field_name} must be an array")
        elif not all(isinstance(item, str) and item for item in value):
            result.fail(f"{field_name} must contain non-empty strings only")


def check_required_values(manifest: dict[str, Any], result: LabelManifestLeakageResult) -> None:
    if manifest.get("sample_type") != "intraday_5m":
        result.fail("sample_type must be intraday_5m")
    require_non_empty(manifest, "label_source_kind", result)
    require_non_empty(manifest, "label_horizon", result)
    require_non_empty(manifest, "label_generation_method", result)
    require_non_empty(manifest, "label_status_column", result)
    require_non_empty(manifest, "insufficient_future_window_policy", result)
    if manifest.get("feature_label_overlap_check") is not True:
        result.fail("feature_label_overlap_check must be true")
    if manifest.get("label_generation_authorized") is not True:
        result.fail("label_generation_authorized must be true")


def require_non_empty(
    manifest: dict[str, Any],
    field_name: str,
    result: LabelManifestLeakageResult,
) -> None:
    value = manifest.get(field_name)
    if value is None or value == "" or value == [] or value == {}:
        result.fail(f"{field_name} must be present and non-empty")


def check_feature_leakage(
    feature_columns: Sequence[str],
    label_columns: Sequence[str],
    outcome_columns: Sequence[str],
    result: LabelManifestLeakageResult,
) -> None:
    feature_set = set(feature_columns)
    label_overlap = sorted(feature_set & set(label_columns))
    if label_overlap:
        result.fail("feature_columns intersects label_columns: " + ", ".join(label_overlap))

    outcome_overlap = sorted(feature_set & set(outcome_columns))
    if outcome_overlap:
        result.fail("feature_columns intersects outcome_columns: " + ", ".join(outcome_overlap))

    future_columns = sorted(column for column in feature_set if column.startswith("future_"))
    if future_columns:
        result.fail("feature_columns contains future_* fields: " + ", ".join(future_columns))

    label_pattern_columns = sorted(column for column in feature_set if column.endswith("_label"))
    if label_pattern_columns:
        result.fail("feature_columns contains *_label fields: " + ", ".join(label_pattern_columns))

    explicit_forbidden = sorted(feature_set & EXPLICIT_FORBIDDEN_FEATURES)
    if explicit_forbidden:
        result.fail("feature_columns contains forbidden outcome fields: " + ", ".join(explicit_forbidden))


def check_boundary_flags(manifest: dict[str, Any], result: LabelManifestLeakageResult) -> None:
    require_bool(manifest, "supervised_training_allowed", False, result)
    require_bool(manifest, "training_allowed", False, result)
    require_bool(manifest, "stable_effect_allowed", False, result)
    require_bool(manifest, "contains_order_intent", False, result)
    require_bool(manifest, "contains_live_order", False, result)
    require_bool(manifest, "contains_secret", False, result)


def require_bool(
    manifest: dict[str, Any],
    field_name: str,
    expected: bool,
    result: LabelManifestLeakageResult,
) -> None:
    if manifest.get(field_name) is not expected:
        result.fail(f"{field_name} must be {str(expected).lower()}")


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday label manifest leakage checks.")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = check_manifest(args.manifest)
    print(json.dumps(result.to_summary(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
