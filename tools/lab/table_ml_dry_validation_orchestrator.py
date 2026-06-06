from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.table_ml_sample_intake_checker import check_manifest, load_manifest, IntakeCheckResult
from tools.lab.table_ml_schema_validator import validate_sample


REQUIRED_SUMMARY_FIELDS = {
    "status",
    "sample_type",
    "sample_path",
    "rows_checked",
    "intake_passed",
    "schema_passed",
    "warnings",
    "p0_blockers",
    "p1_warnings",
    "advisory_only",
    "training_allowed",
    "affects_stable_trading",
}


@dataclass
class DryValidationSummary:
    status: str = "not_started"
    sample_type: str = ""
    sample_path: str = ""
    rows_checked: int = 0
    intake_passed: bool = False
    schema_passed: bool = False
    warnings: list[str] = field(default_factory=list)
    p0_blockers: list[str] = field(default_factory=list)
    p1_warnings: list[str] = field(default_factory=list)
    advisory_only: bool | None = None
    training_allowed: bool | None = None
    affects_stable_trading: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "sample_type": self.sample_type,
            "sample_path": self.sample_path,
            "rows_checked": self.rows_checked,
            "intake_passed": self.intake_passed,
            "schema_passed": self.schema_passed,
            "warnings": self.warnings,
            "p0_blockers": self.p0_blockers,
            "p1_warnings": self.p1_warnings,
            "advisory_only": self.advisory_only,
            "training_allowed": self.training_allowed,
            "affects_stable_trading": self.affects_stable_trading,
        }


def orchestrate_dry_validation(manifest_path: Path, repo_root: Path | None = None) -> DryValidationSummary:
    repo_root = (repo_root or Path.cwd()).resolve()
    summary = DryValidationSummary()
    manifest = read_manifest_for_summary(manifest_path, summary)
    if not manifest:
        summary.status = "intake_failed"
        return summary

    populate_manifest_summary_fields(summary, manifest)

    intake_result = check_manifest(manifest_path, repo_root=repo_root)
    summary.p0_blockers.extend(intake_result.p0_errors)
    summary.p1_warnings.extend(intake_result.p1_warnings)
    summary.warnings.extend(intake_result.p1_warnings)
    if not intake_result.ok:
        summary.status = classify_intake_failure(intake_result)
        return summary

    summary.intake_passed = True
    sample_type = str(manifest.get("sample_type", ""))
    sample_path = resolve_sample_path(str(manifest.get("sample_path", "")), repo_root)
    feature_columns = as_string_list(manifest.get("feature_columns"))

    schema_result = validate_sample(
        sample_type=sample_type,
        input_path=sample_path,
        feature_columns=feature_columns,
    )
    summary.rows_checked = schema_result.rows_checked
    summary.p0_blockers.extend(schema_result.p0_errors)
    summary.p1_warnings.extend(schema_result.p1_warnings)
    summary.warnings.extend(schema_result.p1_warnings)

    if not schema_result.ok:
        summary.status = classify_schema_failure(schema_result.p0_errors)
        return summary

    summary.schema_passed = True
    summary.status = "passed"
    return summary


def read_manifest_for_summary(manifest_path: Path, summary: DryValidationSummary) -> dict[str, Any]:
    result = IntakeCheckResult()
    manifest = load_manifest(manifest_path, result)
    if result.p0_errors:
        summary.p0_blockers.extend(result.p0_errors)
    if result.p1_warnings:
        summary.p1_warnings.extend(result.p1_warnings)
        summary.warnings.extend(result.p1_warnings)
    return manifest


def populate_manifest_summary_fields(summary: DryValidationSummary, manifest: dict[str, Any]) -> None:
    summary.sample_type = str(manifest.get("sample_type", ""))
    summary.sample_path = str(manifest.get("sample_path", ""))
    summary.advisory_only = bool_or_none(manifest.get("advisory_only"))
    summary.training_allowed = bool_or_none(manifest.get("training_allowed"))
    summary.affects_stable_trading = bool_or_none(manifest.get("affects_stable_trading"))


def bool_or_none(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def resolve_sample_path(raw_sample_path: str, repo_root: Path) -> Path:
    sample_path = Path(raw_sample_path)
    if sample_path.is_absolute():
        return sample_path
    return repo_root / sample_path


def as_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def classify_intake_failure(result: IntakeCheckResult) -> str:
    errors = " | ".join(result.p0_errors)
    if "sample_path does not exist" in errors:
        return "missing_sample_file"
    if "feature_columns contains" in errors:
        return "forbidden_future_feature"
    if (
        "human_authorized must be true" in errors
        or "training_allowed must be false" in errors
        or "affects_stable_trading must be false" in errors
    ):
        return "unauthorized_input"
    return "intake_failed"


def classify_schema_failure(p0_errors: Sequence[str]) -> str:
    errors = " | ".join(p0_errors)
    if "Forbidden future/label feature column" in errors:
        return "forbidden_future_feature"
    return "schema_failed"


def print_summary(summary: DryValidationSummary) -> None:
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only table ML dry validation gates.")
    parser.add_argument("--manifest", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = orchestrate_dry_validation(args.manifest)
    print_summary(summary)
    return 0 if summary.status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
