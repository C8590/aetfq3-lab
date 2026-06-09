from __future__ import annotations

import json
from pathlib import Path

from tools.lab.intraday_majority_class_collapse_diagnostic import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    REPORT_TYPE,
    main,
    run_diagnostic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_majority_collapse_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_majority_class_collapse_diagnostic/pytest")
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
        "feature_columns": FEATURE_COLUMNS,
        "split_check": {
            "selected_split_policy": "anchor_date_70_30",
            "train_anchor_dates": ["2026-01-02", "2026-01-03"],
            "valid_anchor_dates": ["2026-01-04", "2026-01-05"],
        },
        "training_allowed": False,
        "stable_allowed": False,
        "order_intent_allowed": False,
        "qmt_allowed": False,
    }
    if overrides:
        payload.update(overrides)
    return payload


def base_repeatability(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "report_type": "intraday_supervised_no_save_repeatability_check",
        "status": "passed",
        "readiness_decision": "NO_SAVE_SUPERVISED_SMOKE_REPEATABILITY_COMPLETED_REVIEW_REQUIRED",
        "target": "three_day_positive_label",
        "feature_columns": FEATURE_COLUMNS,
        "train_anchor_dates": ["2026-01-02", "2026-01-03"],
        "valid_anchor_dates": ["2026-01-04", "2026-01-05"],
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


def write_repeatability(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "repeatability.json", base_repeatability(overrides))


def test_cli_on_fixture_succeeds(tmp_path: Path, capsys) -> None:
    manifest = write_manifest(tmp_path)
    readiness = write_readiness(tmp_path)
    repeatability = write_repeatability(tmp_path)
    out_dir = OUT_ROOT / "cli_success"

    exit_code = main(
        [
            "--samples",
            str(SAMPLES),
            "--manifest",
            str(manifest),
            "--readiness",
            str(readiness),
            "--repeatability",
            str(repeatability),
            "--out-dir",
            str(out_dir),
        ]
    )
    stdout = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout["status"] == "passed"
    assert "NO_FORMAL_MODEL_EVIDENCE" in stdout["diagnostic_flags"]
    assert stdout["model_saved"] is False


def test_label_shift_is_counted(tmp_path: Path) -> None:
    report = run_diagnostic(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_repeatability(tmp_path),
        OUT_ROOT / "label_shift",
    )

    assert report["label_distribution"]["train"] == {"0": 3, "1": 9}
    assert report["label_distribution"]["valid"] == {"0": 9, "1": 3}
    assert report["train_valid_label_shift"]["observed"] is True
    assert "TRAIN_VALID_LABEL_SHIFT_OBSERVED" in report["diagnostic_flags"]


def test_repeated_label_group_structure_is_identified(tmp_path: Path) -> None:
    report = run_diagnostic(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_repeatability(tmp_path),
        OUT_ROOT / "repeated_groups",
    )

    assert report["sample_granularity"]["repeated_label_group_structure_observed"] is True
    assert report["sample_granularity"]["groups_with_single_repeated_label_count"] == 8
    assert "GROUP_REPEATED_LABEL_STRUCTURE_OBSERVED" in report["diagnostic_flags"]


def test_future_outcome_label_feature_columns_block(tmp_path: Path) -> None:
    leaked_features = FEATURE_COLUMNS + ["future_return_3d", "three_day_positive_label", "execution_return_to_close"]
    manifest = write_manifest(tmp_path, {"feature_columns": leaked_features})
    readiness = write_readiness(tmp_path, {"feature_columns": leaked_features})
    repeatability = write_repeatability(tmp_path, {"feature_columns": leaked_features})

    report = run_diagnostic(SAMPLES, manifest, readiness, repeatability, OUT_ROOT / "leaked_features")

    assert report["diagnostic_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert report["status"] == "blocked"
    assert any("future_return_3d" in item for item in report["p0_blockers"])
    assert any("three_day_positive_label" in item for item in report["p0_blockers"])


def test_training_allowed_true_blocks(tmp_path: Path) -> None:
    report = run_diagnostic(
        SAMPLES,
        write_manifest(tmp_path, {"training_allowed": True}),
        write_readiness(tmp_path),
        write_repeatability(tmp_path),
        OUT_ROOT / "training_allowed",
    )

    assert report["diagnostic_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("training_allowed must be false" in item for item in report["p0_blockers"])


def test_contains_order_intent_true_blocks(tmp_path: Path) -> None:
    report = run_diagnostic(
        SAMPLES,
        write_manifest(tmp_path, {"contains_order_intent": True}),
        write_readiness(tmp_path),
        write_repeatability(tmp_path),
        OUT_ROOT / "order_intent",
    )

    assert report["diagnostic_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("contains_order_intent must be false" in item for item in report["p0_blockers"])


def test_probability_collapse_case_is_identified(tmp_path: Path) -> None:
    report = run_diagnostic(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_repeatability(tmp_path),
        OUT_ROOT / "probability_collapse",
    )

    probability = report["logistic_probability_diagnostic"]
    assert probability["threshold_collapse_observed"] is True
    assert probability["prediction_distribution_at_threshold_0_5"] == {"0": 0, "1": 12}
    assert "LOGISTIC_THRESHOLD_COLLAPSE_OBSERVED" in report["diagnostic_flags"]


def test_report_json_includes_required_boundary_fields(tmp_path: Path) -> None:
    out_dir = OUT_ROOT / "report_fields"
    report = run_diagnostic(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_repeatability(tmp_path),
        out_dir,
    )
    payload = json.loads(
        (REPO_ROOT / out_dir / "intraday_majority_class_collapse_diagnostic_report.json").read_text(encoding="utf-8")
    )

    assert report["report_type"] == REPORT_TYPE
    for key in (
        "report_type",
        "input_sample",
        "target",
        "feature_columns",
        "label_distribution",
        "train_valid_label_shift",
        "sample_granularity",
        "feature_scale_diagnostic",
        "univariate_signal_diagnostic",
        "logistic_probability_diagnostic",
        "balanced_scaled_probe",
        "diagnostic_flags",
        "diagnostic_decision",
        "model_saved",
        "checkpoint_saved",
        "gpu_used",
        "torchrun_used",
        "qmt_used",
        "order_intent_generated",
        "stable_affected",
        "metrics_are_effectiveness_evidence",
        "not_trading_advice",
    ):
        assert key in payload
    assert payload["model_saved"] is False
    assert payload["stable_promotion_ready"] is False


def test_no_model_artifacts_created(tmp_path: Path) -> None:
    out_dir = OUT_ROOT / "no_artifacts"
    report = run_diagnostic(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_repeatability(tmp_path),
        out_dir,
    )
    resolved = REPO_ROOT / out_dir
    forbidden = [
        path
        for path in resolved.rglob("*")
        if path.is_file() and path.suffix.lower() in {".pkl", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"}
    ]

    assert report["model_saved"] is False
    assert report["checkpoint_saved"] is False
    assert forbidden == []
