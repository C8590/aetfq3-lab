from __future__ import annotations

import json
from pathlib import Path

from tools.lab.intraday_group_level_transform_aware_no_save_smoke import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    BLOCKED_MODEL_OR_SCALER_ARTIFACT_CREATED,
    MODEL_NAMES,
    P1_INCONSISTENCY,
    P1_SCALE,
    P1_SHIFT,
    READY,
    main,
    run_smoke,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_group_level_transform_smoke_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_transform_aware_no_save_smoke/pytest")
FEATURE_COLUMNS = [
    "volume_sum",
    "amount_sum",
    "volume_spike_ratio",
    "negative_amount",
    "day_return",
    "rank_day_return",
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
        "intraday_live_decision_ready": False,
        "feature_columns": FEATURE_COLUMNS,
        "label_generated": True,
        "label_source_kind": "public_future_window_anchor_close_last_bar",
        "label_horizon": {"unit": "trading_day", "required_horizons": ["T+1", "T+3"]},
        "label_generation_method": "anchor_close_last_bar_group_level_past_only_feature_expansion_dryrun_v1",
        "label_columns": ["three_day_positive_label"],
        "outcome_columns": ["future_return_3d"],
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
        "target": "three_day_positive_label",
        "group_level_sample": True,
        "group_key": ["trade_date", "etf_code"],
        "group_label_policy": "anchor_close_last_bar",
        "selected_split_policy": "anchor_date_70_30",
        "train_anchor_dates": ["2026-01-02", "2026-01-03"],
        "valid_anchor_dates": ["2026-01-04", "2026-01-05"],
        "train_group_count": 4,
        "valid_group_count": 4,
        "split_feasible": True,
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
        "p1_warnings": [P1_SCALE, P1_INCONSISTENCY, P1_SHIFT],
    }
    if overrides:
        payload.update(overrides)
    return payload


def base_policy(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "policy_scope": "diagnostic_only",
        "train_only_fit_required": True,
        "save_scaler": False,
        "model_training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "recommended_transforms": {
            "log1p_recommended": [
                "volume_sum",
                "amount_sum",
                "volume_spike_ratio",
                "negative_amount",
                "day_return",
            ],
            "standardize_recommended": FEATURE_COLUMNS,
            "clip_winsorize_review": ["volume_sum", "amount_sum"],
            "no_transform_or_bounded": ["volume_spike_ratio", "day_return", "rank_day_return"],
        },
    }
    if overrides:
        payload.update(overrides)
    return payload


def base_baseline() -> dict[str, object]:
    return {
        "report_type": "intraday_group_level_no_save_diagnostic_smoke",
        "readiness_decision": "GROUP_LEVEL_NO_SAVE_DIAGNOSTIC_SMOKE_COMPLETED_WITH_P1_LABEL_INCONSISTENCY_REVIEW_REQUIRED",
        "metrics": {
            "logistic_regression": {
                "balanced_accuracy": 0.5,
                "prediction_distribution": {"0": 0, "1": 4},
            },
            "logistic_regression_balanced_scaled": {
                "balanced_accuracy": 0.46,
                "prediction_distribution": {"0": 2, "1": 2},
            },
        },
        "majority_class_collapse_check": {
            "logistic_matches_dummy_most_frequent": True,
            "balanced_scaled_probe_reduces_collapse": True,
        },
    }


def write_json(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_manifest(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "manifest.json", base_manifest(overrides))


def write_readiness(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "readiness.json", base_readiness(overrides))


def write_policy(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "transform_policy.json", base_policy(overrides))


def write_baseline(tmp_path: Path) -> Path:
    return write_json(tmp_path, "baseline.json", base_baseline())


def out_dir(tmp_path: Path, name: str) -> Path:
    return OUT_ROOT / tmp_path.name / name


def test_fixture_smoke_cli_succeeds(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--samples",
            str(SAMPLES),
            "--manifest",
            str(write_manifest(tmp_path)),
            "--readiness",
            str(write_readiness(tmp_path)),
            "--transform-policy",
            str(write_policy(tmp_path)),
            "--baseline-smoke",
            str(write_baseline(tmp_path)),
            "--out-dir",
            str(out_dir(tmp_path, "cli_success")),
        ]
    )
    stdout = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout["readiness_decision"] == READY
    assert stdout["models_run"] == MODEL_NAMES
    assert "NO_FORMAL_MODEL_EVIDENCE" in stdout["diagnostic_flags"]


def test_log1p_only_applies_to_nonnegative_flow_features(tmp_path: Path) -> None:
    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_policy(tmp_path),
        write_baseline(tmp_path),
        out_dir(tmp_path, "log1p"),
    )

    assert report["log1p_features_applied"] == ["amount_sum", "volume_sum"]
    assert report["log1p_features_skipped"]["volume_spike_ratio"] == "not_amount_volume_raw_flow_feature"
    assert report["log1p_features_skipped"]["negative_amount"] == "negative_value_present"
    assert report["log1p_features_skipped"]["day_return"] == "not_amount_volume_raw_flow_feature"


def test_standard_scaler_fit_is_train_only(tmp_path: Path) -> None:
    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_policy(tmp_path),
        write_baseline(tmp_path),
        out_dir(tmp_path, "scaler"),
    )

    raw_audit = report["scaler_audit"]["raw_scaled"]
    assert report["standard_scaler_fit_scope"] == "train_only"
    assert raw_audit["fit_scope"] == "train_only"
    assert raw_audit["fit_row_count"] == 4
    assert raw_audit["transform_valid_row_count"] == 4
    assert raw_audit["valid_fit_performed"] is False
    assert raw_audit["train_scaled_abs_mean_max"] < 1e-12
    assert raw_audit["valid_scaled_abs_mean_max"] > 1.0


def test_future_outcome_label_in_features_blocks(tmp_path: Path) -> None:
    report = run_smoke(
        SAMPLES,
        write_manifest(
            tmp_path,
            {"feature_columns": FEATURE_COLUMNS + ["three_day_positive_label", "future_return_3d", "future_bad"]},
        ),
        write_readiness(tmp_path),
        write_policy(tmp_path),
        write_baseline(tmp_path),
        out_dir(tmp_path, "leakage"),
    )

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("three_day_positive_label" in item for item in report["p0_blockers"])
    assert any("future_return_3d" in item for item in report["p0_blockers"])
    assert any("future_bad" in item for item in report["p0_blockers"])


def test_boundary_flag_true_blocks(tmp_path: Path) -> None:
    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path, {"stable_allowed": True}),
        write_readiness(tmp_path),
        write_policy(tmp_path),
        write_baseline(tmp_path),
        out_dir(tmp_path, "boundary"),
    )

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("manifest.stable_allowed must be false" in item for item in report["p0_blockers"])


def test_artifact_scan_catches_model_and_scaler_files(tmp_path: Path) -> None:
    destination = REPO_ROOT / out_dir(tmp_path, "artifact")
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.pkl"
    scaler_path = destination / "scaler.joblib"
    model_path.write_text("forbidden", encoding="utf-8")
    scaler_path.write_text("forbidden", encoding="utf-8")

    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_policy(tmp_path),
        write_baseline(tmp_path),
        out_dir(tmp_path, "artifact"),
    )

    assert report["readiness_decision"] == BLOCKED_MODEL_OR_SCALER_ARTIFACT_CREATED
    assert "model.pkl" in ",".join(report["artifact_check_before"]["found_model_artifacts"])
    assert "scaler.joblib" in ",".join(report["artifact_check_before"]["found_model_artifacts"])
    model_path.unlink()
    scaler_path.unlink()


def test_report_json_includes_transform_boundary_fields(tmp_path: Path) -> None:
    destination = out_dir(tmp_path, "report")
    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_policy(tmp_path),
        write_baseline(tmp_path),
        destination,
    )
    payload = json.loads(
        (REPO_ROOT / destination / "intraday_group_level_transform_aware_no_save_smoke_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["readiness_decision"] == READY
    for key in (
        "report_type",
        "smoke_scope",
        "target",
        "transform_policy_applied",
        "log1p_features_applied",
        "log1p_features_skipped",
        "standard_scaler_fit_scope",
        "scaler_saved",
        "models_run",
        "train_group_count",
        "valid_group_count",
        "train_label_distribution",
        "valid_label_distribution",
        "metrics",
        "prediction_distribution_by_model",
        "collapse_check",
        "comparison_to_baseline_group_smoke",
        "model_saved",
        "checkpoint_saved",
        "gpu_used",
        "torchrun_used",
        "qmt_used",
        "order_intent_generated",
        "stable_affected",
        "metrics_are_effectiveness_evidence",
        "automatic_promotion_ready",
        "not_trading_advice",
    ):
        assert key in payload
    assert payload["scaler_saved"] is False
    assert payload["model_saved"] is False
    assert payload["metrics_are_effectiveness_evidence"] is False


def test_no_scaler_or_model_artifact_created(tmp_path: Path) -> None:
    destination = REPO_ROOT / out_dir(tmp_path, "no_artifacts")
    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path),
        write_policy(tmp_path),
        write_baseline(tmp_path),
        out_dir(tmp_path, "no_artifacts"),
    )

    suffixes = {path.suffix.lower() for path in destination.rglob("*") if path.is_file()}
    assert report["model_saved"] is False
    assert report["scaler_saved"] is False
    assert not (suffixes & {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"})
