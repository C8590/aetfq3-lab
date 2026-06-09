from __future__ import annotations

import csv
import json
from pathlib import Path

import tools.lab.intraday_group_level_supervised_smoke_readiness_precheck as precheck
from tools.lab.intraday_group_level_supervised_smoke_readiness_precheck import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_INSUFFICIENT_GROUPS,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    BLOCKED_SINGLE_CLASS,
    BLOCKED_SPLIT_NOT_CLASS_DIVERSE,
    P1_INCONSISTENCY,
    READY,
    READY_WITH_INCONSISTENCY,
    run_precheck,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
READY_SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_group_level_ready_samples.csv"
SINGLE_CLASS_SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_group_level_single_class_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_group_level_supervised_smoke_readiness_precheck/pytest")
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
        "eligible_anchor_subset_only": True,
        "eligible_anchor_dates": ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        "label_generation_scope": "dry_run_only",
        "label_generation_dryrun_allowed": True,
        "label_generation_performed": True,
        "generated_outcomes": ["future_return_1d", "future_return_3d", "max_drawdown_3d"],
        "generated_labels": ["three_day_positive_label"],
        "blocked_labels": ["buy_now_label", "wait_pullback_label", "cancel_buy_label"],
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
    }
    if overrides:
        payload.update(overrides)
    return payload


def base_group_report(inconsistent_count: int = 0) -> dict[str, object]:
    return {
        "report_type": "intraday_group_level_sample_dryrun",
        "status": "passed",
        "group_label_policy": "anchor_close_last_bar",
        "intraday_live_decision_ready": False,
        "raw_bar_row_count": 384,
        "group_count": 8,
        "group_statistics": {
            "single_label_group_count": 8 - inconsistent_count,
            "inconsistent_label_group_count": inconsistent_count,
            "null_label_group_count": 0,
        },
        "readiness_decision": "GROUP_LEVEL_SAMPLE_DRY_RUN_PASSED_CLASS_DIVERSE_REVIEW_REQUIRED",
    }


def configure_small_thresholds(monkeypatch) -> None:
    monkeypatch.setattr(precheck, "MIN_GROUPS", 8)
    monkeypatch.setattr(precheck, "MIN_ANCHORS", 4)
    monkeypatch.setattr(precheck, "MIN_ETFS", 2)
    monkeypatch.setattr(precheck, "MIN_CLASS_COUNT", 4)


def write_json(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_manifest(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    return write_json(tmp_path, "manifest.json", base_manifest(overrides))


def write_group_report(tmp_path: Path, inconsistent_count: int = 0) -> Path:
    return write_json(tmp_path, "group_report.json", base_group_report(inconsistent_count))


def make_split_valid_single_class_sample(tmp_path: Path) -> Path:
    out = tmp_path / "split_valid_single_class.csv"
    with READY_SAMPLES.open("r", encoding="utf-8", newline="") as src, out.open("w", encoding="utf-8", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if row["trade_date"] in {"2026-01-04", "2026-01-05"}:
                row["three_day_positive_label"] = "1"
            writer.writerow(row)
    return out


def test_valid_group_level_fixture_readiness_passed(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)

    report = run_precheck(
        READY_SAMPLES,
        write_manifest(tmp_path),
        write_group_report(tmp_path),
        OUT_ROOT / "valid",
    )

    assert report["readiness_decision"] == READY
    assert report["split_feasible"] is True
    assert report["train_label_0_count"] == 2
    assert report["valid_label_1_count"] == 2
    assert report["training_allowed"] is False


def test_single_class_group_level_fixture_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)

    report = run_precheck(
        SINGLE_CLASS_SAMPLES,
        write_manifest(tmp_path),
        write_group_report(tmp_path),
        OUT_ROOT / "single_class",
    )

    assert report["readiness_decision"] == BLOCKED_SINGLE_CLASS
    assert report["sample_check"]["class_count"] == 1


def test_label_in_feature_columns_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)
    features = FEATURE_COLUMNS + ["three_day_positive_label"]

    report = run_precheck(
        READY_SAMPLES,
        write_manifest(tmp_path, {"feature_columns": features}),
        write_group_report(tmp_path),
        OUT_ROOT / "label_feature",
    )

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("three_day_positive_label" in item for item in report["p0_blockers"])


def test_outcome_in_feature_columns_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)
    features = FEATURE_COLUMNS + ["future_return_3d", "max_drawdown_3d"]

    report = run_precheck(
        READY_SAMPLES,
        write_manifest(tmp_path, {"feature_columns": features}),
        write_group_report(tmp_path),
        OUT_ROOT / "outcome_feature",
    )

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("future_return_3d" in item for item in report["p0_blockers"])


def test_insufficient_group_count_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)
    monkeypatch.setattr(precheck, "MIN_GROUPS", 9)

    report = run_precheck(
        READY_SAMPLES,
        write_manifest(tmp_path),
        write_group_report(tmp_path),
        OUT_ROOT / "insufficient_groups",
    )

    assert report["readiness_decision"] == BLOCKED_INSUFFICIENT_GROUPS
    assert any("group_count must be" in item for item in report["p0_blockers"])


def test_split_valid_single_class_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)
    monkeypatch.setattr(precheck, "MIN_CLASS_COUNT", 2)
    split_bad = make_split_valid_single_class_sample(tmp_path)

    report = run_precheck(
        split_bad,
        write_manifest(tmp_path),
        write_group_report(tmp_path),
        OUT_ROOT / "split_single_class",
    )

    assert report["readiness_decision"] == BLOCKED_SPLIT_NOT_CLASS_DIVERSE
    assert report["split_feasible"] is False


def test_boundary_flag_true_blocks(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)

    report = run_precheck(
        READY_SAMPLES,
        write_manifest(tmp_path, {"training_allowed": True}),
        write_group_report(tmp_path),
        OUT_ROOT / "boundary",
    )

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("training_allowed must be false" in item for item in report["p0_blockers"])


def test_inconsistent_label_groups_emit_p1_warning(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)

    report = run_precheck(
        READY_SAMPLES,
        write_manifest(tmp_path),
        write_group_report(tmp_path, inconsistent_count=2),
        OUT_ROOT / "inconsistent",
    )

    assert report["readiness_decision"] == READY_WITH_INCONSISTENCY
    assert report["inconsistent_label_review"]["inconsistent_label_group_rate"] == 0.25
    assert P1_INCONSISTENCY in report["p1_warnings"]


def test_report_json_contains_required_boundary_fields(tmp_path: Path, monkeypatch) -> None:
    configure_small_thresholds(monkeypatch)
    out_dir = OUT_ROOT / "report_fields"

    report = run_precheck(READY_SAMPLES, write_manifest(tmp_path), write_group_report(tmp_path), out_dir)
    payload = json.loads(
        (REPO_ROOT / out_dir / "intraday_group_level_supervised_smoke_readiness_report.json").read_text(
            encoding="utf-8"
        )
    )

    assert report["readiness_decision"] == READY
    for key in (
        "report_type",
        "readiness_decision",
        "selected_split_policy",
        "train_anchor_dates",
        "valid_anchor_dates",
        "train_group_count",
        "valid_group_count",
        "train_label_0_count",
        "train_label_1_count",
        "valid_label_0_count",
        "valid_label_1_count",
        "split_feasible",
        "training_allowed",
        "stable_allowed",
        "qmt_allowed",
        "order_intent_allowed",
        "automatic_promotion_ready",
        "metrics_are_effectiveness_evidence",
        "model_saved",
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
    assert payload["metrics_are_effectiveness_evidence"] is False
