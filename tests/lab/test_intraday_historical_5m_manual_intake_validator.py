from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tools.lab.intraday_historical_5m_manual_intake_validator import (
    ManualIntakeValidatorError,
    detect_forbidden_fields,
    resolve_local_path,
    run_validator,
    IntakeConfig,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_CSV = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_historical_5m_manual_export.csv"
FIXTURE_MANIFEST = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_historical_5m_manual_manifest.json"
LOCAL_ARTIFACT = REPO_ROOT / ".local_artifact_backup/pytest_manual_intake"
LOCAL_REPORT = REPO_ROOT / ".local_research_outputs/pytest_manual_intake"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_note_text() -> str:
    return "\n".join(
        [
            "source_name: pytest_fixture",
            "source_type: manual_export",
            "export_method: static_fixture",
            "exported_at: 2026-06-11T00:00:00Z",
            "date_range: 2026-03-02 to 2026-03-13",
            "etf_universe: 159915,510050,510300,510500,512100",
            "frequency: 5m",
            "whether_qmt_export: false",
            "whether_account_related: false",
            "whether_order_related: false",
            "whether_contains_trades_or_fills: false",
            "whether_contains_secret: false",
            "whether_stable_bundle: false",
            "human_authorized: true",
            "",
        ]
    )


def make_inbox(name: str, *, include_source_note: bool = True, csv_path: Path = FIXTURE_CSV, bad_hash: bool = False) -> tuple[Path, Path]:
    inbox = LOCAL_ARTIFACT / name
    out_dir = LOCAL_REPORT / name
    shutil.rmtree(inbox, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    inbox.mkdir(parents=True)
    shutil.copy2(csv_path, inbox / "manual_export.csv")
    shutil.copy2(FIXTURE_MANIFEST, inbox / "MANIFEST.json")
    if include_source_note:
        (inbox / "source_note.md").write_text(source_note_text(), encoding="utf-8")
    lines = []
    for path in sorted(p for p in inbox.iterdir() if p.name != "SHA256SUMS.txt"):
        digest = "0" * 64 if bad_hash and path.name == "manual_export.csv" else sha256(path)
        lines.append(f"{digest}  {path.name}")
    (inbox / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return inbox, out_dir


def run_for(name: str, **kwargs) -> tuple[dict[str, object], Path]:
    inbox, out_dir = make_inbox(name, **kwargs)
    result = run_validator(IntakeConfig(inbox=inbox, out_dir=out_dir))
    return result, out_dir


def test_missing_inbox_waiting_decision() -> None:
    inbox = LOCAL_ARTIFACT / "missing_inbox"
    out_dir = LOCAL_REPORT / "missing_inbox"
    shutil.rmtree(inbox, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)

    result = run_validator(IntakeConfig(inbox=inbox, out_dir=out_dir))

    assert result["readiness_decision"] == "MANUAL_HISTORICAL_5M_PACKAGE_NOT_FOUND_WAITING_FOR_INPUT"
    assert (out_dir / "manual_intake_readiness_decision.json").exists()


def test_valid_csv_manifest_schema_passed() -> None:
    result, out_dir = run_for("valid_schema")
    schema = json.loads((out_dir / "manual_intake_schema_report.json").read_text(encoding="utf-8"))

    assert schema["schema_passed"]
    assert schema["rows_checked"] == 50
    assert result["readiness_decision"] == "MANUAL_HISTORICAL_5M_PACKAGE_READY_FOR_FIXED_SHORTLIST_OOP_VALIDATION"


def test_missing_source_note_blocked() -> None:
    result, _ = run_for("missing_source_note", include_source_note=False)

    assert result["readiness_decision"] == "BLOCKED_MISSING_SOURCE_NOTE"


def test_hash_mismatch_blocked() -> None:
    result, _ = run_for("hash_mismatch", bad_hash=True)

    assert result["readiness_decision"] == "BLOCKED_HASH_MISMATCH"


def test_forbidden_field_account_order_trade_blocked(tmp_path: Path) -> None:
    df = pd.read_csv(FIXTURE_CSV)
    df["account"] = "forbidden"
    df["order_id"] = "forbidden"
    df["trade"] = "forbidden"
    csv_path = tmp_path / "forbidden.csv"
    df.to_csv(csv_path, index=False)

    result, _ = run_for("forbidden_fields", csv_path=csv_path)

    assert result["readiness_decision"] == "BLOCKED_FORBIDDEN_FIELDS"
    assert detect_forbidden_fields(["trade_date", "account", "order_id", "trade"]) == ["account", "order_id", "trade"]


def test_duplicate_bars_detected(tmp_path: Path) -> None:
    with FIXTURE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.append(rows[0].copy())
    csv_path = tmp_path / "duplicate.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result, out_dir = run_for("duplicate", csv_path=csv_path)
    quality = json.loads((out_dir / "manual_intake_data_quality_report.json").read_text(encoding="utf-8"))

    assert result["readiness_decision"] == "BLOCKED_DATA_QUALITY"
    assert quality["duplicate_bars"] == 1


def test_strict_oop_anchor_count_calculated_correctly() -> None:
    _, out_dir = run_for("oop_count")
    readiness = json.loads((out_dir / "manual_intake_oop_readiness.json").read_text(encoding="utf-8"))

    assert readiness["strict_oop_anchor_count"] == 10
    assert readiness["etf_count"] == 5
    assert readiness["group_count"] == 50


def test_readiness_threshold_blocks_limited_sample(tmp_path: Path) -> None:
    df = pd.read_csv(FIXTURE_CSV).head(25)
    csv_path = tmp_path / "limited.csv"
    df.to_csv(csv_path, index=False)

    result, _ = run_for("limited", csv_path=csv_path)

    assert result["readiness_decision"] == "MANUAL_HISTORICAL_5M_PACKAGE_LIMITED_REVIEW_REQUIRED"
    assert result["fixed_shortlist_oop_validation_ready"] is False


def test_non_local_input_path_rejected() -> None:
    with pytest.raises(ManualIntakeValidatorError):
        resolve_local_path(Path("tests/fixtures/aetfq3_lab"), ".local_artifact_backup")
