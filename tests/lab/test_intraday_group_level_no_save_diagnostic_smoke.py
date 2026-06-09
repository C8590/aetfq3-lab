from __future__ import annotations

import json
from pathlib import Path

from tools.lab.intraday_group_level_no_save_diagnostic_smoke import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_GROUP_LEVEL_READINESS_NOT_PASSED,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    BLOCKED_MODEL_ARTIFACT_CREATED,
    P1_INCONSISTENCY,
    READY_WITH_P1,
    main,
    run_smoke,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_group_level_smoke_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_no_save_diagnostic_smoke/pytest")
FEATURE_COLUMNS = [
    "open_first",
    "high_max",
    "low_min",
    "close_last",
    "volume_sum",
    "amount_sum",
    "vwap_day",
    "day_return",
    "high_low_range",
    "close_to_vwap",
    "intraday_return_mean",
    "intraday_return_std",
    "distance_to_vwap_mean",
    "distance_to_vwap_last",
    "volume_first_half_sum",
    "volume_second_half_sum",
    "amount_first_half_sum",
    "amount_second_half_sum",
]


def base_manifest(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "manifest_version": "intraday_group_level_three_day_label_dryrun_v1",
        "sample_type": "intraday_5m",
        "sample_subtype": "intraday_group_level_three_day_label_dryrun",
        "group_level_sample": True,
        "group_key": ["trade_date", "etf_code"],
        "group_label_policy": "anchor_close_last_bar",
        "intraday_live_decision_ready": False,
        "feature_columns": FEATURE_COLUMNS,
        "label_generated": True,
        "label_source_kind": "public_future_window_anchor_close_last_bar",
        "label_horizon": {"unit": "trading_day", "required_horizons": ["T+1", "T+3"]},
        "label_generation_method": "anchor_close_last_bar_group_level_dryrun_v1",
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
        "readiness_decision": "GROUP_LEVEL_SUPERVISED_SMOKE_READINESS_PASSED_WITH_LABEL_INCONSISTENCY_REVIEW_REQUIRED",
        "status": "passed",
        "selected_split_policy": "anchor_date_70_30",
        "train_anchor_dates": ["2026-01-02", "2026-01-03"],
        "valid_anchor_dates": ["2026-01-04", "2026-01-05"],
        "train_group_count": 4,
        "valid_group_count": 4,
        "train_label_0_count": 2,
        "train_label_1_count": 2,
        "valid_label_0_count": 2,
        "valid_label_1_count": 2,
        "split_feasible": True,
        "training_allowed": False,
        "stable_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "automatic_promotion_ready": False,
        "metrics_are_effectiveness_evidence": False,
        "p0_blockers": [],
        "p1_warnings": [P1_INCONSISTENCY],
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


def test_fixture_smoke_cli_succeeds(tmp_path: Path, capsys) -> None:
    out_dir = OUT_ROOT / "cli_success"
    exit_code = main(
        [
            "--samples",
            str(SAMPLES),
            "--manifest",
            str(write_manifest(tmp_path)),
            "--readiness",
            str(write_readiness(tmp_path)),
            "--out-dir",
            str(out_dir),
        ]
    )
    stdout = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout["readiness_decision"] == READY_WITH_P1
    assert stdout["models_run"] == [
        "dummy_most_frequent",
        "dummy_stratified",
        "logistic_regression",
        "logistic_regression_balanced_scaled",
    ]
    assert P1_INCONSISTENCY in stdout["p1_warnings"]
    assert stdout["model_saved"] is False


def test_readiness_not_passed_blocks(tmp_path: Path) -> None:
    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path),
        write_readiness(tmp_path, {"readiness_decision": "BLOCKED_GROUP_LEVEL_SPLIT_NOT_CLASS_DIVERSE"}),
        OUT_ROOT / "readiness_blocked",
    )

    assert report["readiness_decision"] == BLOCKED_GROUP_LEVEL_READINESS_NOT_PASSED
    assert report["models_run"] == []


def test_p1_warning_is_preserved(tmp_path: Path) -> None:
    report = run_smoke(SAMPLES, write_manifest(tmp_path), write_readiness(tmp_path), OUT_ROOT / "p1_preserved")

    assert report["readiness_decision"] == READY_WITH_P1
    assert report["p1_warnings"] == [P1_INCONSISTENCY]


def test_label_or_outcome_in_feature_columns_blocks(tmp_path: Path) -> None:
    features = FEATURE_COLUMNS + ["three_day_positive_label", "future_return_3d"]

    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path, {"feature_columns": features}),
        write_readiness(tmp_path),
        OUT_ROOT / "leaky_features",
    )

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("three_day_positive_label" in item for item in report["p0_blockers"])
    assert any("future_return_3d" in item for item in report["p0_blockers"])


def test_boundary_flag_true_blocks(tmp_path: Path) -> None:
    report = run_smoke(
        SAMPLES,
        write_manifest(tmp_path, {"stable_allowed": True}),
        write_readiness(tmp_path),
        OUT_ROOT / "boundary",
    )

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("stable_allowed must be false" in item for item in report["p0_blockers"])


def test_artifact_check_catches_pkl_and_pt(tmp_path: Path) -> None:
    out_dir = REPO_ROOT / OUT_ROOT / "artifact_blocked"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    checkpoint_path = out_dir / "checkpoint.pt"
    model_path.write_text("forbidden", encoding="utf-8")
    checkpoint_path.write_text("forbidden", encoding="utf-8")

    report = run_smoke(SAMPLES, write_manifest(tmp_path), write_readiness(tmp_path), OUT_ROOT / "artifact_blocked")

    assert report["readiness_decision"] == BLOCKED_MODEL_ARTIFACT_CREATED
    assert "model.pkl" in ",".join(report["artifact_check_before"]["found_model_artifacts"])
    assert "checkpoint.pt" in ",".join(report["artifact_check_before"]["found_model_artifacts"])
    model_path.unlink()
    checkpoint_path.unlink()


def test_report_includes_no_save_boundary_fields(tmp_path: Path) -> None:
    out_dir = OUT_ROOT / "report_fields"

    report = run_smoke(SAMPLES, write_manifest(tmp_path), write_readiness(tmp_path), out_dir)
    payload = json.loads(
        (REPO_ROOT / out_dir / "intraday_group_level_no_save_diagnostic_smoke_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["readiness_decision"] == READY_WITH_P1
    for key in (
        "report_type",
        "smoke_scope",
        "target",
        "models_run",
        "group_level_sample",
        "group_label_policy",
        "intraday_live_decision_ready",
        "p1_warnings",
        "train_group_count",
        "valid_group_count",
        "train_label_distribution",
        "valid_label_distribution",
        "metrics",
        "majority_class_collapse_check",
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
    assert payload["metrics_are_effectiveness_evidence"] is False
    assert payload["automatic_promotion_ready"] is False
