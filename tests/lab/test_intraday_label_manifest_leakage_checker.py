from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


from tools.lab.intraday_label_manifest_leakage_checker import check_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_MANIFEST = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_label_manifest_valid.json"
BAD_FEATURE_OVERLAP = (
    REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_label_manifest_bad_feature_overlap.json"
)
CHECKER = REPO_ROOT / "tools/lab/intraday_label_manifest_leakage_checker.py"


def write_manifest(tmp_path: Path, overrides: dict[str, object]) -> Path:
    payload = json.loads(VALID_MANIFEST.read_text(encoding="utf-8"))
    payload.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_valid_manifest_passed() -> None:
    result = check_manifest(VALID_MANIFEST)

    assert result.ok
    assert result.p0_blockers == []
    assert result.feature_count == 11
    assert result.label_count == 4
    assert result.outcome_count == 8


def test_label_column_in_feature_columns_fails() -> None:
    result = check_manifest(BAD_FEATURE_OVERLAP)

    assert not result.ok
    assert any("feature_columns intersects label_columns" in item for item in result.p0_blockers)
    assert any("buy_now_label" in item for item in result.p0_blockers)


def test_outcome_column_in_feature_columns_fails(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"feature_columns": ["bar_index", "open", "execution_return_to_close"]})

    result = check_manifest(path)

    assert not result.ok
    assert any("feature_columns intersects outcome_columns" in item for item in result.p0_blockers)


def test_future_return_3d_in_feature_columns_fails(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"feature_columns": ["bar_index", "future_return_3d"]})

    result = check_manifest(path)

    assert not result.ok
    assert any("future_return_3d" in item for item in result.p0_blockers)


def test_supervised_training_allowed_true_fails(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"supervised_training_allowed": True})

    result = check_manifest(path)

    assert not result.ok
    assert any("supervised_training_allowed must be false" in item for item in result.p0_blockers)


def test_label_generation_authorized_false_fails(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"label_generation_authorized": False})

    result = check_manifest(path)

    assert not result.ok
    assert any("label_generation_authorized must be true" in item for item in result.p0_blockers)


def test_contains_order_intent_true_fails(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"contains_order_intent": True})

    result = check_manifest(path)

    assert not result.ok
    assert any("contains_order_intent must be false" in item for item in result.p0_blockers)


def test_missing_insufficient_future_window_policy_fails(tmp_path: Path) -> None:
    payload = json.loads(VALID_MANIFEST.read_text(encoding="utf-8"))
    payload.pop("insufficient_future_window_policy")
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    result = check_manifest(path)

    assert not result.ok
    assert any("missing required fields" in item for item in result.p0_blockers)
    assert any("insufficient_future_window_policy" in item for item in result.p0_blockers)


def test_checker_stdout_json_contains_expected_keys() -> None:
    completed = subprocess.run(
        [sys.executable, str(CHECKER), "--manifest", str(VALID_MANIFEST)],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(completed.stdout)

    assert payload["status"] == "passed"
    assert payload["boundary_passed"] is True
    assert set(payload) == {
        "status",
        "manifest_path",
        "p0_blockers",
        "p1_warnings",
        "feature_count",
        "label_count",
        "outcome_count",
        "boundary_passed",
    }
