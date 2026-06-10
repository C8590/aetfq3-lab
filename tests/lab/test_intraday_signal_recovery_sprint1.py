from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.lab.intraday_signal_recovery_sprint1 import (
    BASE_LABEL,
    DECISION_CANDIDATE,
    LABEL_POLICIES,
    NEW_LABEL_POLICIES,
    build_decision,
    build_feature_set_variants,
    build_manifest,
    build_split_payload,
    check_feature_set_leakage,
    compute_past_daily_feature_values,
    detect_collapse,
    evaluate_candidate_gate,
    generate_label_variants,
    load_csv_rows,
    recover_past_daily_features,
    run_diagnostic_suite,
    summarize_label_policies,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_signal_recovery_sprint1_samples.csv"
OUT_ROOT = REPO_ROOT / ".local_research_outputs/aetfq3_lab/intraday_signal_recovery_sprint1/pytest"
pytestmark = pytest.mark.filterwarnings("ignore:X does not have valid feature names.*:UserWarning")


def base_rows() -> list[dict[str, object]]:
    rows, _ = load_csv_rows(FIXTURE)
    return rows


def base_manifest() -> dict[str, object]:
    return {
        "manifest_version": "test",
        "sample_type": "intraday_5m",
        "sample_subtype": "test",
        "feature_columns": ["feature_a", "feature_b"],
        "label_generated": True,
        "label_source_kind": "public_future_window_anchor_close_last_bar",
        "label_horizon": {"unit": "trading_day", "required_horizons": ["T+1", "T+3"]},
        "label_generation_method": "test",
        "label_columns": [BASE_LABEL],
        "outcome_columns": ["future_return_1d", "future_return_3d", "max_drawdown_3d"],
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


def test_label_policy_variants_generated() -> None:
    rows = base_rows()
    generation = generate_label_variants(rows)

    assert generation["generated_label_policies"] == NEW_LABEL_POLICIES
    for row in rows:
        for label in NEW_LABEL_POLICIES:
            assert label in row


def test_neutral_band_labels_can_contain_null_and_are_excluded() -> None:
    rows = base_rows()
    rows[0]["future_return_3d"] = "0.001"
    generate_label_variants(rows)
    split = build_split_payload(
        rows,
        "label_neutral_band_20bp",
        ["feature_a", "feature_b"],
        ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        ["2026-01-06", "2026-01-07"],
    )

    assert rows[0]["label_neutral_band_20bp"] == ""
    assert split["train_rows_dropped"] >= 1
    assert split["passed"]


def test_future_outcome_and_label_not_in_feature_columns() -> None:
    manifest = build_manifest(base_manifest(), ["feature_a", "feature_b"], ["prev_1d_return"])
    check = check_feature_set_leakage(
        manifest["feature_columns"],
        manifest["label_columns"],
        manifest["outcome_columns"],
    )
    bad_check = check_feature_set_leakage(
        ["feature_a", "future_return_3d", "label_ret3d_gt_20bp"],
        manifest["label_columns"],
        manifest["outcome_columns"],
    )

    assert check["passed"]
    assert not bad_check["passed"]


def test_past_daily_feature_uses_only_prior_dates() -> None:
    rows = [{"trade_date": "2026-01-04", "etf_code": "510300"}]
    daily_rows = [
        {"trade_date": "2026-01-02", "etf_code": "510300", "close": "100", "volume": "1000"},
        {"trade_date": "2026-01-03", "etf_code": "510300", "close": "110", "volume": "1100"},
        {"trade_date": "2026-01-04", "etf_code": "510300", "close": "121", "volume": "1200"},
        {"trade_date": "2026-01-05", "etf_code": "510300", "close": "999", "volume": "9999"},
    ]
    report = recover_past_daily_features(rows, daily_rows)

    assert rows[0]["prev_1d_return"] == pytest.approx(0.1)
    assert report["past_only_audit_examples"][0]["history_end_date"] == "2026-01-04"
    assert report["past_only_audit_examples"][0]["used_future_rows"] is False


def test_no_save_model_suite_creates_no_artifacts(tmp_path: Path) -> None:
    rows = base_rows()
    generate_label_variants(rows)
    label_summaries = summarize_label_policies(
        rows,
        ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        ["2026-01-06", "2026-01-07"],
    )
    variants = build_feature_set_variants(["feature_a", "feature_b"], [])
    feature_set_report = {
        name: {
            **payload,
            "leakage_check": {"passed": True, "feature_count": len(payload["feature_columns"]), "p0_blockers": []},
        }
        for name, payload in variants.items()
    }
    out_dir = OUT_ROOT / tmp_path.name
    out_dir.mkdir(parents=True, exist_ok=True)

    report = run_diagnostic_suite(
        rows,
        label_summaries,
        feature_set_report,
        {"recommended_transforms": {"log1p_recommended": [], "standardize_recommended": []}},
        ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        ["2026-01-06", "2026-01-07"],
        out_dir,
    )

    assert report["artifact_check_after"]["passed"]
    assert report["model_saved"] is False
    assert report["scaler_saved"] is False
    assert not list(out_dir.glob("*.pkl"))


def test_candidate_gate_works() -> None:
    gate = evaluate_candidate_gate(
        {
            "accuracy": 0.7,
            "balanced_accuracy": 0.61,
            "roc_auc": 0.56,
            "pr_auc": 0.63,
            "precision": 0.7,
            "recall": 0.7,
        },
        prevalence=0.5,
        dummy_metrics={"dummy_most_frequent": {"balanced_accuracy": 0.5}},
        collapse={"collapse_flag": False, "valid_prediction_contains_both_classes": True},
        leakage_check={"passed": True},
    )
    decision = build_decision({"diagnostic_candidates": [{"x": 1}], "p0_blockers": []}, {"p0_blockers": []})

    assert gate["diagnostic_signal_candidate"]
    assert gate["candidate_label"] == "DIAGNOSTIC_SIGNAL_CANDIDATE"
    assert decision["sprint_decision"] == DECISION_CANDIDATE


def test_collapse_detection_works() -> None:
    collapsed = detect_collapse([1, 1, 1, 1], [1, 1, 1, 1])
    not_collapsed = detect_collapse([0, 1, 0, 1], [1, 1, 1, 1])

    assert collapsed["collapse_flag"] is True
    assert collapsed["valid_prediction_contains_both_classes"] is False
    assert not_collapsed["collapse_flag"] is False


def test_boundary_fields_present_in_report_json(tmp_path: Path) -> None:
    decision = build_decision({"diagnostic_candidates": [], "p0_blockers": []}, {"p0_blockers": []})
    path = tmp_path / "decision.json"
    path.write_text(json.dumps(decision, ensure_ascii=False), encoding="utf-8")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["formal_model_evidence"] is False
    assert payload["stable_promotion_ready"] is False
    assert payload["formal_training_ready"] is False
    assert payload["qmt_ready"] is False
    assert payload["order_intent_ready"] is False
    assert payload["automatic_promotion_ready"] is False
    assert payload["requires_human_review"] is True
    assert set(LABEL_POLICIES)


def test_compute_past_daily_values_never_needs_future_row() -> None:
    values = compute_past_daily_feature_values(
        [
            {"trade_date": "2026-01-01", "close": "100", "volume": "100"},
            {"trade_date": "2026-01-02", "close": "105", "volume": "110"},
            {"trade_date": "2026-01-03", "close": "110", "volume": "120"},
        ]
    )

    assert values["prev_1d_return"] == pytest.approx(110 / 105 - 1)
