from __future__ import annotations

import json
from pathlib import Path

from tools.lab.intraday_supervised_no_save_repeatability_check import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    BLOCKED_MODEL_ARTIFACT_CREATED,
    BLOCKED_READINESS_NOT_PASSED,
    READY,
    main,
    run_repeatability_check,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_supervised_smoke_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_supervised_no_save_repeatability_check/pytest")
FEATURE_COLUMNS = [
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
]


def base_manifest(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample_type": "intraday_5m",
        "feature_columns": FEATURE_COLUMNS,
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


def base_readiness(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "readiness_decision": "SUPERVISED_SMOKE_READINESS_PASSED_REVIEW_REQUIRED",
        "status": "passed",
        "selected_split_policy": "anchor_date_70_30",
        "train_anchor_dates": ["2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28"],
        "valid_anchor_dates": ["2026-06-01", "2026-06-02"],
        "split_feasible": True,
        "training_allowed": False,
        "stable_allowed": False,
        "order_intent_allowed": False,
        "qmt_allowed": False,
    }
    if overrides:
        payload.update(overrides)
    return payload


def base_baseline(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "report_type": "intraday_supervised_no_save_smoke",
        "smoke_scope": "lab_only_no_save",
        "status": "passed",
        "readiness_decision": "NO_SAVE_SUPERVISED_SMOKE_COMPLETED_REVIEW_REQUIRED",
        "target": "three_day_positive_label",
        "feature_columns": FEATURE_COLUMNS,
        "models_run": ["dummy_most_frequent", "dummy_stratified", "logistic_regression"],
        "train_anchor_dates": ["2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28"],
        "valid_anchor_dates": ["2026-06-01", "2026-06-02"],
        "train_rows": 16,
        "valid_rows": 8,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "metrics_are_effectiveness_evidence": False,
    }
    if overrides:
        payload.update(overrides)
    return payload


def write_json(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_manifest(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "manifest.json", base_manifest(overrides))


def write_readiness(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "readiness.json", base_readiness(overrides))


def write_baseline(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "baseline.json", base_baseline(overrides))


def test_repeatability_cli_on_fixture_succeeds(tmp_path: Path, capsys) -> None:
    manifest = write_manifest(tmp_path)
    readiness = write_readiness(tmp_path)
    baseline = write_baseline(tmp_path)
    out_dir = OUT_ROOT / "cli_success"

    exit_code = main(
        [
            "--samples",
            str(SAMPLES),
            "--manifest",
            str(manifest),
            "--readiness",
            str(readiness),
            "--baseline-smoke",
            str(baseline),
            "--out-dir",
            str(out_dir),
        ]
    )
    stdout = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout["readiness_decision"] == READY
    assert stdout["seeds"] == [7, 13, 42, 101, 2026]
    assert stdout["models_run"] == ["dummy_most_frequent", "dummy_stratified", "logistic_regression"]
    assert stdout["model_saved"] is False


def test_readiness_not_passed_blocks(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    readiness = write_readiness(tmp_path, {"readiness_decision": "BLOCKED_SPLIT_NOT_CLASS_DIVERSE", "status": "blocked"})
    baseline = write_baseline(tmp_path)

    report = run_repeatability_check(SAMPLES, manifest, readiness, baseline, OUT_ROOT / "readiness_blocked")

    assert report["readiness_decision"] == BLOCKED_READINESS_NOT_PASSED
    assert report["models_run"] == []


def test_label_in_feature_columns_blocks(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, {"feature_columns": FEATURE_COLUMNS + ["three_day_positive_label"]})
    readiness = write_readiness(tmp_path)
    baseline = write_baseline(tmp_path, {"feature_columns": FEATURE_COLUMNS + ["three_day_positive_label"]})

    report = run_repeatability_check(SAMPLES, manifest, readiness, baseline, OUT_ROOT / "label_feature")

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("three_day_positive_label" in item for item in report["p0_blockers"])


def test_outcome_in_feature_columns_blocks(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, {"feature_columns": FEATURE_COLUMNS + ["future_return_3d"]})
    readiness = write_readiness(tmp_path)
    baseline = write_baseline(tmp_path, {"feature_columns": FEATURE_COLUMNS + ["future_return_3d"]})

    report = run_repeatability_check(SAMPLES, manifest, readiness, baseline, OUT_ROOT / "outcome_feature")

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("future_return_3d" in item for item in report["p0_blockers"])


def test_training_allowed_true_blocks_boundary(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, {"training_allowed": True})
    readiness = write_readiness(tmp_path)
    baseline = write_baseline(tmp_path)

    report = run_repeatability_check(SAMPLES, manifest, readiness, baseline, OUT_ROOT / "training_allowed")

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("training_allowed must be false" in item for item in report["p0_blockers"])


def test_contains_order_intent_true_blocks_boundary(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, {"contains_order_intent": True})
    readiness = write_readiness(tmp_path)
    baseline = write_baseline(tmp_path)

    report = run_repeatability_check(SAMPLES, manifest, readiness, baseline, OUT_ROOT / "order_intent")

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("contains_order_intent must be false" in item for item in report["p0_blockers"])


def test_output_artifact_check_catches_pkl_and_pt(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    readiness = write_readiness(tmp_path)
    baseline = write_baseline(tmp_path)
    out_dir = REPO_ROOT / OUT_ROOT / "artifact_blocked"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    checkpoint_path = out_dir / "checkpoint.pt"
    model_path.write_text("forbidden", encoding="utf-8")
    checkpoint_path.write_text("forbidden", encoding="utf-8")

    report = run_repeatability_check(SAMPLES, manifest, readiness, baseline, OUT_ROOT / "artifact_blocked")

    assert report["readiness_decision"] == BLOCKED_MODEL_ARTIFACT_CREATED
    assert "model.pkl" in ",".join(report["artifact_check_before"]["found_model_artifacts"])
    assert "checkpoint.pt" in ",".join(report["artifact_check_before"]["found_model_artifacts"])
    model_path.unlink()
    checkpoint_path.unlink()


def test_report_json_includes_repeatability_boundary_fields(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    readiness = write_readiness(tmp_path)
    baseline = write_baseline(tmp_path)
    out_dir = OUT_ROOT / "report_fields"

    report = run_repeatability_check(SAMPLES, manifest, readiness, baseline, out_dir)
    payload = json.loads(
        (REPO_ROOT / out_dir / "intraday_supervised_no_save_repeatability_report.json").read_text(encoding="utf-8")
    )

    assert report["readiness_decision"] == READY
    for key in (
        "report_type",
        "smoke_scope",
        "target",
        "seeds",
        "models_run",
        "train_anchor_dates",
        "valid_anchor_dates",
        "train_rows",
        "valid_rows",
        "train_label_distribution",
        "valid_label_distribution",
        "metrics_by_seed",
        "metric_variability_summary",
        "model_saved",
        "checkpoint_saved",
        "gpu_used",
        "torchrun_used",
        "qmt_used",
        "order_intent_generated",
        "stable_affected",
        "not_trading_advice",
        "metrics_are_effectiveness_evidence",
        "automatic_promotion_ready",
    ):
        assert key in payload
    assert payload["model_saved"] is False
    assert payload["automatic_promotion_ready"] is False
