from __future__ import annotations

import json
from pathlib import Path

from tools.lab.intraday_dry_validation_orchestrator import orchestrate
from tools.lab.intraday_dry_validation_report_reader import validate_report


VALID_MANIFEST = Path("tests/fixtures/aetfq3_lab/mock_intraday_5m_manifest.json")
OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_synthetic_dry_validation/pytest_reader")


def valid_report_path() -> Path:
    orchestrate(VALID_MANIFEST, OUT_DIR)
    return OUT_DIR / "intraday_synthetic_dry_validation_report.json"


def write_modified_report(tmp_path: Path, overrides: dict[str, object]) -> Path:
    payload = json.loads(valid_report_path().read_text(encoding="utf-8"))
    payload.update(overrides)
    path = tmp_path / "report.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_valid_report_passed() -> None:
    summary = validate_report(valid_report_path())

    assert summary["status"] == "passed"
    assert summary["reader_passed"]


def test_report_with_no_order_intent_false_fails(tmp_path: Path) -> None:
    path = write_modified_report(tmp_path, {"no_order_intent": False})

    summary = validate_report(path)

    assert summary["status"] == "failed"
    assert any("no_order_intent must be true" in blocker for blocker in summary["p0_blockers"])


def test_report_with_model_saved_true_fails(tmp_path: Path) -> None:
    path = write_modified_report(tmp_path, {"model_saved": True})

    summary = validate_report(path)

    assert summary["status"] == "failed"
    assert any("model_saved must be false" in blocker for blocker in summary["p0_blockers"])


def test_report_containing_order_intent_field_fails(tmp_path: Path) -> None:
    path = write_modified_report(tmp_path, {"OrderIntent": "forbidden"})

    summary = validate_report(path)

    assert summary["status"] == "failed"
    assert any("OrderIntent" in blocker for blocker in summary["p0_blockers"])
