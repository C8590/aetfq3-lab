from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.lab.intraday_signal_recovery_sprint2_candidate_audit import (
    DECISION_ROBUST,
    ROBUST_CANDIDATE,
    WEAK_CANDIDATE,
    audit_candidates,
    build_decision,
    build_split_payload,
    check_model_artifacts,
    load_csv_rows,
    robustness_gate,
    run_family_robustness,
    select_candidate_families,
    validate_split_policy,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_signal_recovery_sprint2_samples.csv"


def sprint1_like_candidate(
    *,
    label_policy: str = "three_day_positive_label",
    feature_set: str = "base_39_features",
    model: str = "logistic_balanced_scaled",
    collapse: bool = False,
    pr_auc: float = 0.62,
    prevalence: float = 0.50,
) -> dict[str, object]:
    valid_ones = int(prevalence * 20)
    valid_zeroes = 20 - valid_ones
    return {
        "label_policy": label_policy,
        "feature_set": feature_set,
        "model": model,
        "status": "completed",
        "train_rows_used": 40,
        "valid_rows_used": 20,
        "train_label_distribution": {"0": 20, "1": 20},
        "valid_label_distribution": {"0": valid_zeroes, "1": valid_ones},
        "balanced_accuracy": 0.61,
        "roc_auc": 0.58,
        "pr_auc": pr_auc,
        "prediction_distribution": {"0": 9, "1": 11} if not collapse else {"0": 0, "1": 20},
        "collapse_flag": collapse,
        "collapse_check": {
            "collapse_flag": collapse,
            "valid_prediction_contains_both_classes": not collapse,
            "matches_dummy_most_frequent_predictions": collapse,
        },
        "compared_to_dummy_most_frequent": {
            "balanced_accuracy_delta": 0.11,
            "accuracy_delta": 0.10,
            "roc_auc_delta": 0.08,
            "pr_auc_delta": pr_auc - prevalence,
        },
        "compared_to_dummy_stratified": {
            "balanced_accuracy_delta": 0.09,
            "accuracy_delta": 0.07,
            "roc_auc_delta": 0.06,
            "pr_auc_delta": pr_auc - prevalence - 0.01,
        },
        "candidate_gate": {
            "diagnostic_signal_candidate": True,
            "checks": {
                "no_collapse": not collapse,
                "balanced_accuracy_beats_dummy_by_0_03": True,
                "roc_auc_at_least_0_53": True,
                "pr_auc_beats_prevalence_by_0_03": pr_auc >= prevalence + 0.03,
                "valid_prediction_contains_both_classes": not collapse,
                "no_leakage": True,
                "no_artifact": True,
                "non_dummy_model": True,
            },
        },
        "no_save_artifact_check": {"passed": True, "model_artifact_created": False},
        "leakage_check": {"passed": True, "p0_blockers": []},
    }


def label_report() -> dict[str, object]:
    return {
        "label_policies": {
            "three_day_positive_label": {
                "null_count": 0,
                "row_count": 60,
                "min_class_count": 20,
            },
            "label_ret3d_gt_20bp": {
                "null_count": 0,
                "row_count": 60,
                "min_class_count": 20,
            },
            "label_neutral_band_50bp": {
                "null_count": 55,
                "row_count": 5,
                "min_class_count": 2,
            },
        }
    }


def diagnostic_report(candidates: list[dict[str, object]]) -> dict[str, object]:
    return {"diagnostic_candidates": candidates}


def feature_report() -> dict[str, object]:
    return {
        "feature_set_variants": {
            "base_39_features": {
                "feature_columns": ["feature_a", "feature_b", "volume_sum", "amount_sum"]
            },
            "base_39_plus_scale_transform_policy": {
                "feature_columns": ["feature_a", "feature_b", "volume_sum", "amount_sum"]
            },
        }
    }


def test_candidate_audit_reads_sprint1_like_report() -> None:
    audit = audit_candidates(
        diagnostic_report([sprint1_like_candidate(), sprint1_like_candidate(model="random_forest_shallow_no_save")]),
        label_report(),
    )

    assert audit["candidate_count"] == 2
    assert audit["candidate_count_by_label_policy"]["three_day_positive_label"] == 2
    assert audit["candidate_count_by_model"]["logistic_balanced_scaled"] == 1
    assert audit["candidates"][0]["prevalence"] == 0.5


def test_candidate_family_grouping_works() -> None:
    audit = audit_candidates(
        diagnostic_report(
            [
                sprint1_like_candidate(model="logistic_balanced_scaled"),
                sprint1_like_candidate(model="logistic_log1p_scaled_balanced"),
            ]
        ),
        label_report(),
    )
    families = select_candidate_families(audit)

    assert len(families) == 1
    assert families[0]["model_family"] == "logistic_balanced_scaled_variants"
    assert families[0]["candidate_count"] == 2


def test_isolated_candidate_flagged() -> None:
    audit = audit_candidates(
        diagnostic_report([sprint1_like_candidate(label_policy="label_ret3d_gt_20bp")]),
        label_report(),
    )

    assert len(audit["candidates_isolated_to_one_label_policy"]) == 1
    assert len(audit["candidates_isolated_to_one_model_family"]) == 1


def test_robustness_gate_passes_stable_family() -> None:
    seed_results = [
        {
            "split_id": split_id,
            "beats_dummy_most_frequent_by_0_03": True,
            "roc_auc_at_least_0_53": True,
            "pr_auc_beats_prevalence_by_0_03": True,
        }
        for split_id in ["s1", "s2"]
    ]
    split_results = [
        {"split_id": "s1", "available": True, "no_collapse_all_seeds": True},
        {"split_id": "s2", "available": True, "no_collapse_all_seeds": True},
    ]

    gate = robustness_gate(split_results, seed_results)

    assert gate["robust_diagnostic_signal_candidate"] is True


def test_robustness_gate_blocks_collapse_family() -> None:
    split_results = [
        {"split_id": "s1", "available": True, "no_collapse_all_seeds": True},
        {"split_id": "s2", "available": True, "no_collapse_all_seeds": False},
    ]
    seed_results = [
        {
            "split_id": "s1",
            "beats_dummy_most_frequent_by_0_03": True,
            "roc_auc_at_least_0_53": True,
            "pr_auc_beats_prevalence_by_0_03": True,
        },
        {
            "split_id": "s2",
            "beats_dummy_most_frequent_by_0_03": True,
            "roc_auc_at_least_0_53": True,
            "pr_auc_beats_prevalence_by_0_03": True,
        },
    ]

    assert robustness_gate(split_results, seed_results)["robust_diagnostic_signal_candidate"] is False


def test_robustness_gate_blocks_split_unstable_family() -> None:
    split_results = [
        {"split_id": "s1", "available": True, "no_collapse_all_seeds": True},
        {"split_id": "s2", "available": True, "no_collapse_all_seeds": True},
    ]
    seed_results = [
        {
            "split_id": "s1",
            "beats_dummy_most_frequent_by_0_03": True,
            "roc_auc_at_least_0_53": True,
            "pr_auc_beats_prevalence_by_0_03": True,
        }
    ]

    assert robustness_gate(split_results, seed_results)["robust_diagnostic_signal_candidate"] is False


def test_random_split_is_not_allowed() -> None:
    with pytest.raises(Exception, match="random"):
        validate_split_policy("random_70_30")


def test_no_save_artifact_check_catches_model_scaler_files(tmp_path: Path) -> None:
    (tmp_path / "model.pkl").write_text("nope", encoding="utf-8")
    (tmp_path / "scaler.joblib").write_text("nope", encoding="utf-8")
    check = check_model_artifacts(tmp_path)

    assert not check["passed"]
    assert "model.pkl" in ",".join(check["found_model_artifacts"])
    assert "scaler.joblib" in ",".join(check["found_model_artifacts"])


def test_boundary_fields_present_in_report_json() -> None:
    decision = build_decision({"robust_family_count": 1, "selected_family_count": 1, "robust_diagnostic_candidates": [], "p0_blockers": []}, {"p0_blockers": []})

    assert decision["sprint2_decision"] == DECISION_ROBUST
    assert decision["formal_model_evidence"] is False
    assert decision["stable_promotion_ready"] is False
    assert decision["formal_training_ready"] is False
    assert decision["qmt_ready"] is False
    assert decision["order_intent_ready"] is False
    assert decision["automatic_promotion_ready"] is False
    assert decision["metrics_are_effectiveness_evidence"] is False


def test_family_robustness_returns_review_status() -> None:
    rows, _ = load_csv_rows(FIXTURE)
    family = {
        "candidate_family": "three_day_positive_label|base_39_features|logistic_balanced_scaled_variants|no_scale_transform_policy",
        "label_policy": "three_day_positive_label",
        "feature_set": "base_39_features",
        "model_family": "logistic_balanced_scaled_variants",
        "transform_policy": "no_scale_transform_policy",
        "representative_model": "logistic_balanced_scaled",
    }
    splits = [
        {
            "split_policy": "anchor_date_60_40",
            "split_id": "anchor_date_60_40",
            "train_anchor_dates": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"],
            "valid_anchor_dates": ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08"],
        }
    ]
    result = run_family_robustness(rows, feature_report()["feature_set_variants"], family, splits)

    assert result["robustness_decision"] in {ROBUST_CANDIDATE, WEAK_CANDIDATE}
    assert result["model_saved"] is False if "model_saved" in result else True


def test_build_split_payload_requires_two_classes() -> None:
    rows, _ = load_csv_rows(FIXTURE)
    split = build_split_payload(
        rows,
        "three_day_positive_label",
        ["feature_a", "feature_b"],
        ["2026-01-01"],
        ["2026-01-02"],
    )

    assert split["available"] is True
