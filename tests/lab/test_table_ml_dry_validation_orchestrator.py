from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.table_ml_dry_validation_orchestrator import (
    REQUIRED_SUMMARY_FIELDS,
    orchestrate_dry_validation,
)


FALSE_DOWNGRADE_MANIFEST = (
    REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_dry_validation_manifest_false_downgrade.json"
)
SECTOR_MANIFEST = (
    REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_dry_validation_manifest_sector_internal_ranking.json"
)


def load_manifest(path: Path = FALSE_DOWNGRADE_MANIFEST) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_manifest(tmp_path: Path, manifest: dict):
    return orchestrate_dry_validation(write_json(tmp_path, manifest), repo_root=REPO_ROOT)


def test_valid_false_downgrade_manifest_passes():
    summary = orchestrate_dry_validation(FALSE_DOWNGRADE_MANIFEST, repo_root=REPO_ROOT)

    assert summary.status == "passed"
    assert summary.intake_passed
    assert summary.schema_passed
    assert summary.sample_type == "false_downgrade"
    assert summary.rows_checked == 8


def test_valid_sector_internal_ranking_manifest_passes():
    summary = orchestrate_dry_validation(SECTOR_MANIFEST, repo_root=REPO_ROOT)

    assert summary.status == "passed"
    assert summary.intake_passed
    assert summary.schema_passed
    assert summary.sample_type == "sector_internal_ranking"
    assert summary.rows_checked == 8


def test_intake_fail_does_not_call_schema_validator(tmp_path, monkeypatch):
    manifest = load_manifest()
    manifest["training_allowed"] = True

    def fail_if_called(*args, **kwargs):
        raise AssertionError("schema validator should not be called when intake fails")

    monkeypatch.setattr("tools.lab.table_ml_dry_validation_orchestrator.validate_sample", fail_if_called)
    summary = run_manifest(tmp_path, manifest)

    assert summary.status == "unauthorized_input"
    assert not summary.intake_passed
    assert not summary.schema_passed


def test_schema_fail_returns_schema_failed(tmp_path):
    invalid_csv = tmp_path / "invalid_schema.csv"
    invalid_csv.write_text("trade_date,etf_code\n2026-01-02,MOCK001\n", encoding="utf-8")
    manifest = load_manifest()
    manifest["sample_path"] = str(invalid_csv)

    summary = run_manifest(tmp_path, manifest)

    assert summary.status == "schema_failed"
    assert summary.intake_passed
    assert not summary.schema_passed
    assert any("Missing required fields" in blocker for blocker in summary.p0_blockers)


def test_forbidden_future_feature_fails(tmp_path):
    manifest = load_manifest()
    manifest["feature_columns"] = [*manifest["feature_columns"], "future_return_3d"]

    summary = run_manifest(tmp_path, manifest)

    assert summary.status == "forbidden_future_feature"
    assert not summary.intake_passed
    assert not summary.schema_passed


def test_training_allowed_true_fails(tmp_path):
    manifest = load_manifest()
    manifest["training_allowed"] = True

    summary = run_manifest(tmp_path, manifest)

    assert summary.status == "unauthorized_input"
    assert "training_allowed must be false" in summary.p0_blockers


def test_affects_stable_trading_true_fails(tmp_path):
    manifest = load_manifest()
    manifest["affects_stable_trading"] = True

    summary = run_manifest(tmp_path, manifest)

    assert summary.status == "unauthorized_input"
    assert "affects_stable_trading must be false" in summary.p0_blockers


def test_missing_sample_file_fails(tmp_path):
    manifest = load_manifest()
    manifest["sample_path"] = "tests/fixtures/aetfq3_lab/does_not_exist.csv"

    summary = run_manifest(tmp_path, manifest)

    assert summary.status == "missing_sample_file"
    assert not summary.intake_passed
    assert not summary.schema_passed
    assert any("sample_path does not exist" in blocker for blocker in summary.p0_blockers)


def test_cli_summary_json_contains_required_fields():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/lab/table_ml_dry_validation_orchestrator.py",
            "--manifest",
            "tests/fixtures/aetfq3_lab/mock_dry_validation_manifest_false_downgrade.json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert REQUIRED_SUMMARY_FIELDS <= set(summary)
    assert summary["status"] == "passed"
    assert summary["intake_passed"] is True
    assert summary["schema_passed"] is True
    assert summary["advisory_only"] is True
    assert summary["training_allowed"] is False
    assert summary["affects_stable_trading"] is False
