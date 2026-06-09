from __future__ import annotations

import json
from pathlib import Path

from tools.lab.intraday_label_generation_intake_orchestrator import (
    BLOCKED_BOUNDARY_VIOLATION,
    BLOCKED_HASH_OR_SOURCE_NOTE,
    BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA,
    BLOCKED_MANIFEST_P0,
    IntakeOrchestratorError,
    orchestrate_intake,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
VALID_MANIFEST = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_label_generation_intake_manifest.json"
SUFFICIENT_FUTURE_WINDOW = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_future_window_daily_sufficient.csv"
INSUFFICIENT_FUTURE_WINDOW = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_future_window_daily_insufficient.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_label_generation_intake/pytest")
REQUIRED_ETFS = ["159915", "510050", "510300"]
REQUIRED_DATES = ["2026-06-09", "2026-06-10", "2026-06-11"]


def write_manifest(tmp_path: Path, overrides: dict[str, object]) -> Path:
    payload = json.loads(VALID_MANIFEST.read_text(encoding="utf-8"))
    payload.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_future_window_manifest(tmp_path: Path, source_path: Path, coverage_sufficient: bool) -> Path:
    source_note = tmp_path / "future_source_note.md"
    source_note.write_text("fixture source note\n", encoding="utf-8")
    hash_file = tmp_path / "SHA256SUMS.txt"
    hash_file.write_text("fixture-hash  future_window_daily_ohlcv.csv\n", encoding="utf-8")
    return write_manifest(
        tmp_path,
        {
            "future_window_source_kind": "public_future_window",
            "future_window_source_path": str(source_path),
            "future_window_source_note_path": str(source_note),
            "future_window_hash_path": str(hash_file),
            "future_window_required_etf_codes": REQUIRED_ETFS,
            "future_window_required_dates": REQUIRED_DATES,
            "future_window_required_coverage_end": "2026-06-11",
            "future_window_coverage_sufficient": coverage_sufficient,
        },
    )


def run_manifest(path: Path, name: str) -> dict[str, object]:
    return orchestrate_intake(path, OUT_ROOT / name)


def test_valid_intake_manifest_blocks_on_missing_future_window_source() -> None:
    report = run_manifest(VALID_MANIFEST, "valid_missing_future_window")

    assert report["readiness_decision"] == BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA
    assert report["manifest_leakage_check"]["status"] == "passed"
    assert report["hash_source_note_check"]["passed"] is True
    assert report["no_label_tensor_report_check"]["passed"] is True
    assert report["data_quality_report_check"]["passed"] is True
    assert report["future_window_readiness_check"]["passed"] is False
    assert report["presence_gate_passed"] is False
    assert report["coverage_gate_passed"] is False
    assert report["label_generation_performed"] is False


def test_sufficient_future_window_returns_ready(tmp_path: Path) -> None:
    path = write_future_window_manifest(tmp_path, SUFFICIENT_FUTURE_WINDOW, True)

    report = run_manifest(path, "sufficient_future_window")

    assert report["readiness_decision"] == "READY_FOR_LABEL_GENERATION_DRY_RUN"
    assert report["raw_presence_decision"] == "READY_FOR_LABEL_GENERATION_DRY_RUN"
    assert report["effective_readiness_decision"] == "READY_FOR_LABEL_GENERATION_DRY_RUN"
    assert report["presence_gate_passed"] is True
    assert report["coverage_gate_passed"] is True
    assert report["coverage_sufficient"] is True
    assert report["missing_future_dates_by_etf"] == {}


def test_insufficient_future_window_blocks_and_lists_missing_dates(tmp_path: Path) -> None:
    path = write_future_window_manifest(tmp_path, INSUFFICIENT_FUTURE_WINDOW, False)

    report = run_manifest(path, "insufficient_future_window")

    assert report["readiness_decision"] == BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA
    assert report["raw_presence_decision"] == "READY_FOR_LABEL_GENERATION_DRY_RUN"
    assert report["effective_readiness_decision"] == BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA
    assert report["presence_gate_passed"] is True
    assert report["coverage_gate_passed"] is False
    assert report["coverage_sufficient"] is False
    assert report["missing_future_dates_by_etf"] == {code: REQUIRED_DATES for code in REQUIRED_ETFS}


def test_retry_and_final_retry_out_dirs_are_allowed(tmp_path: Path) -> None:
    path = write_future_window_manifest(tmp_path, SUFFICIENT_FUTURE_WINDOW, True)

    retry_report = orchestrate_intake(
        path,
        Path(".local_research_outputs/aetfq3_lab/intraday_label_generation_intake_retry/pytest"),
    )
    final_retry_report = orchestrate_intake(
        path,
        Path(".local_research_outputs/aetfq3_lab/intraday_label_generation_intake_final_retry/pytest"),
    )

    assert retry_report["readiness_decision"] == "READY_FOR_LABEL_GENERATION_DRY_RUN"
    assert final_retry_report["readiness_decision"] == "READY_FOR_LABEL_GENERATION_DRY_RUN"


def test_rejects_non_lab_output_dirs(tmp_path: Path) -> None:
    path = write_future_window_manifest(tmp_path, SUFFICIENT_FUTURE_WINDOW, True)

    for out_dir in (
        Path("output/intraday_label_generation_intake"),
        Path("Stable/runtime/intraday_label_generation_intake"),
        tmp_path / "outside_repo",
    ):
        try:
            orchestrate_intake(path, out_dir)
        except IntakeOrchestratorError:
            pass
        else:
            raise AssertionError(f"out_dir should have been rejected: {out_dir}")


def test_bad_feature_overlap_blocks_manifest_p0(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"feature_columns": ["open", "buy_now_label"]})

    report = run_manifest(path, "bad_feature_overlap")

    assert report["readiness_decision"] == BLOCKED_MANIFEST_P0
    assert any("feature_columns intersects label_columns" in item for item in report["p0_blockers"])


def test_missing_source_note_blocks_hash_or_source_note(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifact"
    artifact_dir.mkdir()
    for filename in ("intraday_5m_export.csv", "EXPORT_MANIFEST.json", "SHA256SUMS.txt"):
        (artifact_dir / filename).write_text("placeholder\n", encoding="utf-8")
    path = write_manifest(
        tmp_path,
        {
            "public_artifact_dir": str(artifact_dir),
            "hash_source_validation_report_path": "",
        },
    )

    report = run_manifest(path, "missing_source_note")

    assert report["readiness_decision"] == BLOCKED_HASH_OR_SOURCE_NOTE
    assert any("source_note.md" in item for item in report["p0_blockers"])


def test_supervised_training_allowed_true_blocks_boundary(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"supervised_training_allowed": True})

    report = run_manifest(path, "supervised_training_allowed")

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_VIOLATION
    assert any("supervised_training_allowed must be false" in item for item in report["p0_blockers"])


def test_contains_order_intent_true_blocks_boundary(tmp_path: Path) -> None:
    path = write_manifest(tmp_path, {"contains_order_intent": True})

    report = run_manifest(path, "contains_order_intent")

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_VIOLATION
    assert any("contains_order_intent must be false" in item for item in report["p0_blockers"])


def test_output_json_required_fields() -> None:
    out_dir = OUT_ROOT / "output_json_required_fields"
    report = orchestrate_intake(VALID_MANIFEST, out_dir)
    report_path = REPO_ROOT / out_dir / "intraday_label_generation_intake_report.json"
    decision_path = REPO_ROOT / out_dir / "readiness_decision.json"

    assert report_path.exists()
    assert decision_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))

    for key in (
        "report_type",
        "status",
        "readiness_decision",
        "manifest_leakage_check",
        "hash_source_note_check",
        "no_label_tensor_report_check",
        "data_quality_report_check",
        "future_window_readiness_check",
        "boundary_check",
        "p0_blockers",
    ):
        assert key in payload
    assert decision["readiness_decision"] == report["readiness_decision"]
    assert decision["label_generation_performed"] is False
