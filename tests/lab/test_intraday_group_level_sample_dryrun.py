from __future__ import annotations

import csv
import json
from pathlib import Path

import tools.lab.intraday_group_level_sample_dryrun as dryrun
from tools.lab.intraday_group_level_sample_dryrun import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    BLOCKED_SINGLE_CLASS,
    READY_CLASS_DIVERSE,
    run_group_level_dryrun,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_group_level_bar_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_sample_dryrun/pytest")


def base_source_manifest(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample_type": "intraday_5m",
        "eligible_anchor_dates": ["2026-01-02", "2026-01-03"],
        "feature_columns": [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "vwap",
            "intraday_return",
            "return_from_open",
            "distance_to_vwap",
        ],
        "label_generated": True,
        "label_source_kind": "public_future_window",
        "label_horizon": {"unit": "trading_day", "required_horizons": ["T+1", "T+3"]},
        "label_generation_method": "unit_test_three_day_positive_label_v1",
        "label_columns": [
            "buy_now_label",
            "wait_pullback_label",
            "cancel_buy_label",
            "three_day_positive_label",
        ],
        "outcome_columns": [
            "future_return_1d",
            "future_return_3d",
            "max_drawdown_3d",
            "execution_return_to_close",
            "execution_return_to_next_open",
            "execution_drawdown_after_entry",
            "expected_3d_return",
            "expected_3d_drawdown",
        ],
        "label_status_column": "label_status",
        "insufficient_future_window_policy": "set label null when future window is unavailable",
        "feature_label_overlap_check": True,
        "label_generation_authorized": True,
        "supervised_training_allowed": False,
        "training_allowed": False,
        "stable_effect_allowed": False,
        "contains_order_intent": False,
        "contains_live_order": False,
        "contains_secret": False,
    }
    if overrides:
        payload.update(overrides)
    return payload


def base_diagnostic(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "report_type": "intraday_majority_class_collapse_diagnostic",
        "diagnostic_decision": "DIAGNOSTIC_COMPLETED_GROUP_LEVEL_SAMPLE_RECOMMENDED",
        "diagnostic_flags": [
            "GROUP_REPEATED_LABEL_STRUCTURE_OBSERVED",
            "LOGISTIC_THRESHOLD_COLLAPSE_OBSERVED",
            "NO_FORMAL_MODEL_EVIDENCE",
        ],
        "model_saved": False,
        "checkpoint_saved": False,
        "stable_affected": False,
    }
    if overrides:
        payload.update(overrides)
    return payload


def write_json(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_manifest(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "manifest.json", base_source_manifest(overrides))


def write_diagnostic(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "diagnostic.json", base_diagnostic(overrides))


def configure_small_fixture_thresholds(monkeypatch) -> None:
    monkeypatch.setattr(dryrun, "MIN_GROUPS", 4)
    monkeypatch.setattr(dryrun, "MIN_ANCHORS", 2)
    monkeypatch.setattr(dryrun, "MIN_ETFS", 2)


def read_output_rows(out_dir: Path) -> list[dict[str, str]]:
    with (REPO_ROOT / out_dir / "intraday_group_level_samples.csv").open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def make_single_class_sample(tmp_path: Path) -> Path:
    out = tmp_path / "single_class.csv"
    with SAMPLES.open("r", encoding="utf-8", newline="") as src, out.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            row["three_day_positive_label"] = "1"
            writer.writerow(row)
    return out


def test_valid_bar_level_fixture_generates_group_level_sample(tmp_path: Path, monkeypatch) -> None:
    configure_small_fixture_thresholds(monkeypatch)
    out_dir = OUT_ROOT / "valid"

    report = run_group_level_dryrun(SAMPLES, write_manifest(tmp_path), write_diagnostic(tmp_path), out_dir)
    rows = read_output_rows(out_dir)

    assert report["readiness_decision"] == READY_CLASS_DIVERSE
    assert report["raw_bar_row_count"] == 12
    assert report["group_count"] == 4
    assert len(rows) == 4
    assert report["training_allowed"] is False


def test_last_bar_label_policy_is_used(tmp_path: Path, monkeypatch) -> None:
    configure_small_fixture_thresholds(monkeypatch)
    out_dir = OUT_ROOT / "last_bar"

    report = run_group_level_dryrun(SAMPLES, write_manifest(tmp_path), write_diagnostic(tmp_path), out_dir)
    rows = read_output_rows(out_dir)
    target = next(row for row in rows if row["trade_date"] == "2026-01-02" and row["etf_code"] == "510050")

    assert report["group_label_policy"] == "anchor_close_last_bar"
    assert report["intraday_live_decision_ready"] is False
    assert target["three_day_positive_label"] == "1"
    assert target["future_return_3d"] == "0.02"
    assert target["last_bar_datetime"] == "2026-01-02 09:45:00"


def test_future_outcome_label_in_feature_columns_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_fixture_thresholds(monkeypatch)

    report = run_group_level_dryrun(
        SAMPLES,
        write_manifest(tmp_path),
        write_diagnostic(tmp_path),
        OUT_ROOT / "leakage",
        feature_columns_override=dryrun.BASE_GROUP_FEATURE_COLUMNS
        + ["future_return_3d", "three_day_positive_label", "execution_return_to_close"],
    )

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("future_return_3d" in item for item in report["p0_blockers"])
    assert any("three_day_positive_label" in item for item in report["p0_blockers"])


def test_inconsistent_group_label_is_counted(tmp_path: Path, monkeypatch) -> None:
    configure_small_fixture_thresholds(monkeypatch)

    report = run_group_level_dryrun(SAMPLES, write_manifest(tmp_path), write_diagnostic(tmp_path), OUT_ROOT / "inconsistent")

    assert report["group_statistics"]["inconsistent_label_group_count"] == 1
    assert report["group_statistics"]["single_label_group_count"] == 3
    assert report["group_statistics"]["inconsistent_label_group_examples"][0]["etf_code"] == "510050"


def test_single_class_group_label_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_fixture_thresholds(monkeypatch)
    single_class = make_single_class_sample(tmp_path)

    report = run_group_level_dryrun(
        single_class,
        write_manifest(tmp_path),
        write_diagnostic(tmp_path),
        OUT_ROOT / "single_class",
    )

    assert report["readiness_decision"] == BLOCKED_SINGLE_CLASS
    assert report["class_balance_precheck"]["class_count"] == 1


def test_boundary_flag_true_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_fixture_thresholds(monkeypatch)

    report = run_group_level_dryrun(
        SAMPLES,
        write_manifest(tmp_path, {"training_allowed": True}),
        write_diagnostic(tmp_path),
        OUT_ROOT / "boundary",
    )

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("training_allowed must be false" in item for item in report["p0_blockers"])


def test_manifest_json_contains_group_level_fields(tmp_path: Path, monkeypatch) -> None:
    configure_small_fixture_thresholds(monkeypatch)
    out_dir = OUT_ROOT / "manifest_fields"

    run_group_level_dryrun(SAMPLES, write_manifest(tmp_path), write_diagnostic(tmp_path), out_dir)
    manifest = json.loads((REPO_ROOT / out_dir / "intraday_group_level_manifest.json").read_text(encoding="utf-8"))

    assert manifest["sample_subtype"] == "intraday_group_level_three_day_label_dryrun"
    assert manifest["group_level_sample"] is True
    assert manifest["group_key"] == ["trade_date", "etf_code"]
    assert manifest["group_label_policy"] == "anchor_close_last_bar"
    assert manifest["intraday_live_decision_ready"] is False
    assert manifest["supervised_training_allowed"] is False
    assert manifest["contains_order_intent"] is False


def test_no_model_artifacts_created(tmp_path: Path, monkeypatch) -> None:
    configure_small_fixture_thresholds(monkeypatch)
    out_dir = OUT_ROOT / "no_artifacts"

    report = run_group_level_dryrun(SAMPLES, write_manifest(tmp_path), write_diagnostic(tmp_path), out_dir)
    forbidden = [
        path
        for path in (REPO_ROOT / out_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in {".pkl", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"}
    ]

    assert report["model_saved"] is False
    assert report["checkpoint_saved"] is False
    assert forbidden == []
