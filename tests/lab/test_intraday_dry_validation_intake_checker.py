from __future__ import annotations

import json
from pathlib import Path

from tools.lab.intraday_dry_validation_intake_checker import check_manifest


VALID_MANIFEST = Path("tests/fixtures/aetfq3_lab/mock_intraday_5m_manifest.json")
BAD_FUTURE_MANIFEST = Path("tests/fixtures/aetfq3_lab/mock_intraday_5m_bad_future_feature_manifest.json")


def write_manifest(tmp_path: Path, overrides: dict[str, object]) -> Path:
    payload = json.loads(VALID_MANIFEST.read_text(encoding="utf-8"))
    payload.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_valid_mock_manifest_passed() -> None:
    result = check_manifest(VALID_MANIFEST)

    assert result.ok
    assert result.p0_blockers == []


def test_future_return_3d_in_feature_columns_fails() -> None:
    result = check_manifest(BAD_FUTURE_MANIFEST)

    assert not result.ok
    assert any("future_return_3d" in blocker for blocker in result.p0_blockers)


def test_training_allowed_true_fails(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"training_allowed": True})

    result = check_manifest(path)

    assert not result.ok
    assert any("training_allowed must be false" in blocker for blocker in result.p0_blockers)


def test_contains_order_intent_true_fails(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"contains_order_intent": True})

    result = check_manifest(path)

    assert not result.ok
    assert any("contains_order_intent must be false" in blocker for blocker in result.p0_blockers)


def test_qmt_related_with_invalid_qmt_mode_fails(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"qmt_related": True, "qmt_mode": "trade"})

    result = check_manifest(path)

    assert not result.ok
    assert any("qmt_related=true requires qmt_mode" in blocker for blocker in result.p0_blockers)
