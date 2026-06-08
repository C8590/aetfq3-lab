from __future__ import annotations

from pathlib import Path

from tools.lab.intraday_dry_validation_orchestrator import orchestrate


VALID_MANIFEST = Path("tests/fixtures/aetfq3_lab/mock_intraday_5m_manifest.json")
OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_synthetic_dry_validation/pytest_orchestrator")


def test_valid_manifest_end_to_end_passed() -> None:
    report = orchestrate(VALID_MANIFEST, OUT_DIR)

    assert report["status"] == "passed"
    assert report["intake_passed"]
    assert report["schema_passed"]
    assert report["forbidden_feature_passed"]
    assert report["tensor_shape_passed"]
    assert report["state_machine_passed"]


def test_output_report_contains_required_boundary_fields() -> None:
    report = orchestrate(VALID_MANIFEST, OUT_DIR)

    for field in ("lab_only", "no_stable", "no_qmt", "no_order_intent", "no_output", "no_lab_advisory", "no_training"):
        assert report[field] is True
    assert report["checkpoint_saved"] is False
    assert report["model_saved"] is False
    assert report["stable_effect_allowed"] is False


def test_no_order_intent_generated() -> None:
    report = orchestrate(VALID_MANIFEST, OUT_DIR)

    assert report["order_intent_generated"] is False


def test_no_checkpoint_generated() -> None:
    orchestrate(VALID_MANIFEST, OUT_DIR)

    assert not list(OUT_DIR.glob("*.pt"))
    assert not list(OUT_DIR.glob("*.pth"))
    assert not list(OUT_DIR.glob("*.ckpt"))
