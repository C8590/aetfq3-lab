from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.table_ml_sample_intake_checker import check_manifest


FIXTURE_PATH = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_sample_intake_manifest.json"


def write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_manifest() -> dict:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def run_manifest(tmp_path: Path, manifest: dict):
    return check_manifest(write_manifest(tmp_path, manifest), repo_root=REPO_ROOT)


def test_valid_mock_manifest_passes():
    result = check_manifest(FIXTURE_PATH, repo_root=REPO_ROOT)

    assert result.ok
    assert result.p0_errors == []


def test_missing_required_field_fails(tmp_path):
    manifest = load_manifest()
    manifest.pop("manifest_version")

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert any("Missing required fields" in error for error in result.p0_errors)


def test_human_authorized_false_fails(tmp_path):
    manifest = load_manifest()
    manifest["human_authorized"] = False

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert "human_authorized must be true" in result.p0_errors


def test_training_allowed_true_fails(tmp_path):
    manifest = load_manifest()
    manifest["training_allowed"] = True

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert "training_allowed must be false" in result.p0_errors


def test_affects_stable_trading_true_fails(tmp_path):
    manifest = load_manifest()
    manifest["affects_stable_trading"] = True

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert "affects_stable_trading must be false" in result.p0_errors


def test_contains_secret_true_fails(tmp_path):
    manifest = load_manifest()
    manifest["contains_secret"] = True

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert "contains_secret must be false" in result.p0_errors


def test_contains_live_order_true_fails(tmp_path):
    manifest = load_manifest()
    manifest["contains_live_order"] = True

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert "contains_live_order must be false" in result.p0_errors


def test_contains_order_intent_true_fails(tmp_path):
    manifest = load_manifest()
    manifest["contains_order_intent"] = True

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert "contains_order_intent must be false" in result.p0_errors


def test_feature_columns_contains_future_label_fails(tmp_path):
    manifest = load_manifest()
    manifest["feature_columns"] = copy.copy(manifest["feature_columns"])
    manifest["feature_columns"].append("future_return_3d")

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert any("feature_columns contains future_label_columns" in error for error in result.p0_errors)


def test_uses_stable_bundle_without_metadata_fails(tmp_path):
    manifest = load_manifest()
    manifest["uses_stable_bundle"] = True
    manifest["stable_bundle_path"] = ""
    manifest["stable_bundle_commit"] = ""
    manifest["stable_bundle_snapshot_date"] = ""

    result = run_manifest(tmp_path, manifest)

    assert not result.ok
    assert "uses_stable_bundle=true requires stable_bundle_path" in result.p0_errors
    assert any(
        "uses_stable_bundle=true requires stable_bundle_commit or stable_bundle_snapshot_date" == error
        for error in result.p0_errors
    )
