from __future__ import annotations

import json
from pathlib import Path

from tools.lab.intraday_group_level_feature_scale_diagnostic import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    COMPLETED_TRANSFORM_RECOMMENDED,
    main,
    run_diagnostic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_group_level_feature_scale_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_feature_scale_diagnostic/pytest")
FEATURE_COLUMNS = [
    "volume_sum",
    "amount_sum",
    "volume_spike_ratio",
    "day_return",
    "rank_day_return",
    "constant_feature",
    "shift_feature",
]


def base_manifest(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "manifest_version": "intraday_group_level_past_only_feature_expansion_dryrun_v1",
        "sample_type": "intraday_5m",
        "sample_subtype": "intraday_group_level_past_only_feature_expansion_dryrun",
        "group_level_sample": True,
        "group_key": ["trade_date", "etf_code"],
        "group_label_policy": "anchor_close_last_bar",
        "feature_columns": FEATURE_COLUMNS,
        "generated_feature_count": len(FEATURE_COLUMNS),
        "label_columns": ["three_day_positive_label"],
        "outcome_columns": ["future_return_3d"],
        "supervised_training_allowed": False,
        "training_allowed": False,
        "stable_effect_allowed": False,
        "contains_order_intent": False,
        "contains_live_order": False,
        "contains_secret": False,
        "model_training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "not_trading_advice": True,
    }
    if overrides:
        payload.update(overrides)
    return payload


def base_readiness(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "passed",
        "readiness_decision": "GROUP_LEVEL_PAST_ONLY_FEATURE_EXPANSION_DRY_RUN_PASSED_WITH_FEATURE_QUALITY_WARNINGS",
        "selected_split_policy": "anchor_date_70_30",
        "train_anchor_dates": ["2026-01-02", "2026-01-03"],
        "valid_anchor_dates": ["2026-01-04", "2026-01-05"],
        "train_group_count": 4,
        "valid_group_count": 4,
        "training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "model_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "qmt_used": False,
        "order_intent_generated": False,
        "stable_affected": False,
        "not_trading_advice": True,
        "p0_blockers": [],
        "p1_warnings": ["P1_EXTREME_FEATURE_SCALE_REVIEW_REQUIRED"],
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


def out_dir(tmp_path: Path, name: str) -> Path:
    return OUT_ROOT / tmp_path.name / name


def test_valid_fixture_diagnostic_cli_succeeds(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--samples",
            str(SAMPLES),
            "--manifest",
            str(write_manifest(tmp_path)),
            "--readiness",
            str(write_readiness(tmp_path)),
            "--out-dir",
            str(out_dir(tmp_path, "cli_success")),
        ]
    )
    stdout = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout["decision"] == COMPLETED_TRANSFORM_RECOMMENDED
    assert "FEATURE_SCALE_RISK_CONFIRMED" in stdout["diagnostic_flags"]
    assert "NO_FORMAL_MODEL_EVIDENCE" in stdout["diagnostic_flags"]


def test_future_outcome_label_in_feature_columns_blocks(tmp_path: Path) -> None:
    report = run_diagnostic(
        SAMPLES,
        write_manifest(
            tmp_path,
            {"feature_columns": FEATURE_COLUMNS + ["three_day_positive_label", "future_return_3d", "future_bad"]},
        ),
        write_readiness(tmp_path),
        out_dir(tmp_path, "leakage"),
    )

    assert report["decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("three_day_positive_label" in item for item in report["p0_blockers"])
    assert any("future_return_3d" in item for item in report["p0_blockers"])
    assert any("future_bad" in item for item in report["p0_blockers"])


def test_volume_amount_scale_recommends_log1p(tmp_path: Path) -> None:
    report = run_diagnostic(SAMPLES, write_manifest(tmp_path), write_readiness(tmp_path), out_dir(tmp_path, "log1p"))

    assert "volume_sum" in report["recommended_transforms"]["log1p_recommended"]
    assert "amount_sum" in report["recommended_transforms"]["log1p_recommended"]
    assert "volume_spike_ratio" in report["recommended_transforms"]["log1p_recommended"]
    assert "LOG1P_TRANSFORM_RECOMMENDED" in report["diagnostic_flags"]


def test_zero_variance_feature_detected(tmp_path: Path) -> None:
    report = run_diagnostic(SAMPLES, write_manifest(tmp_path), write_readiness(tmp_path), out_dir(tmp_path, "zero_var"))

    assert "constant_feature" in report["feature_scale_summary"]["zero_variance_features"]
    assert report["feature_statistics"]["constant_feature"]["zero_variance"] is True
    assert "ZERO_VARIANCE_FEATURE_FOUND" in report["diagnostic_flags"]


def test_train_valid_shift_detected(tmp_path: Path) -> None:
    report = run_diagnostic(SAMPLES, write_manifest(tmp_path), write_readiness(tmp_path), out_dir(tmp_path, "shift"))

    assert "shift_feature" in report["feature_shift_summary"]["shifted_features"]
    assert "TRAIN_VALID_FEATURE_SHIFT_OBSERVED" in report["diagnostic_flags"]


def test_no_scaler_or_model_artifacts_created(tmp_path: Path) -> None:
    destination = REPO_ROOT / out_dir(tmp_path, "artifacts")
    report = run_diagnostic(SAMPLES, write_manifest(tmp_path), write_readiness(tmp_path), out_dir(tmp_path, "artifacts"))

    suffixes = {path.suffix.lower() for path in destination.rglob("*") if path.is_file()}
    assert report["model_saved"] is False
    assert report["scaler_saved"] is False
    assert not (suffixes & {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"})


def test_report_json_includes_boundary_fields(tmp_path: Path) -> None:
    destination = out_dir(tmp_path, "report_fields")
    report = run_diagnostic(SAMPLES, write_manifest(tmp_path), write_readiness(tmp_path), destination)
    payload = json.loads(
        (REPO_ROOT / destination / "intraday_group_level_feature_scale_diagnostic_report.json").read_text(
            encoding="utf-8"
        )
    )
    policy = json.loads((REPO_ROOT / destination / "transform_policy_recommendation.json").read_text(encoding="utf-8"))

    assert report["decision"] == COMPLETED_TRANSFORM_RECOMMENDED
    for key in (
        "report_type",
        "feature_count",
        "feature_columns",
        "feature_scale_summary",
        "feature_shift_summary",
        "recommended_transforms",
        "diagnostic_flags",
        "decision",
        "model_saved",
        "scaler_saved",
        "checkpoint_saved",
        "gpu_used",
        "torchrun_used",
        "qmt_used",
        "order_intent_generated",
        "stable_affected",
        "not_trading_advice",
    ):
        assert key in payload
    assert payload["training_allowed"] is False
    assert payload["stable_allowed"] is False
    assert payload["qmt_allowed"] is False
    assert payload["order_intent_allowed"] is False
    assert payload["metrics_are_effectiveness_evidence"] is False
    assert policy["policy_scope"] == "diagnostic_only"
    assert policy["train_only_fit_required"] is True
    assert policy["save_scaler"] is False
    assert policy["model_training_allowed"] is False
    assert policy["stable_allowed"] is False


def test_boundary_flag_true_blocks(tmp_path: Path) -> None:
    report = run_diagnostic(
        SAMPLES,
        write_manifest(tmp_path, {"stable_allowed": True}),
        write_readiness(tmp_path),
        out_dir(tmp_path, "boundary"),
    )

    assert report["decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("manifest.stable_allowed must be false" in item for item in report["p0_blockers"])
