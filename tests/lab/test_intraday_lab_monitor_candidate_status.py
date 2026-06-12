from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.lab.intraday_lab_monitor_candidate_status import (
    DECISION_ACTIVE,
    DECISION_BLOCKED_MISSING,
    DECISION_RETIRED,
    FOCUS_CANDIDATE_ID,
    decide,
    gate_status,
    resolve_output_dir,
    run_status,
)


def candidate(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "family_id": FOCUS_CANDIDATE_ID,
        "evaluated_fold_count": 11,
        "positive_fold_count": 8,
        "fraction_folds_balanced_accuracy_above_0_5": 0.7272727272727273,
        "fraction_folds_roc_auc_above_0_5": 0.7272727272727273,
        "fraction_folds_pr_auc_not_below_prevalence": 0.7272727272727273,
        "fraction_folds_non_collapse": 1.0,
        "fraction_folds_above_dummy": 0.7272727272727273,
        "balanced_accuracy_mean": 0.525,
        "roc_auc_mean": 0.5403,
        "pr_auc_mean": 0.4587,
    }
    base.update(overrides)
    return base


def attribution(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "readiness_decision": "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_LAB_MONITOR_CANDIDATE_REVIEW_READY",
        "fold_robustness_summary": {"month_concentration_observed": False},
        "etf_dispersion_summary": {"etf_concentration_observed": False},
        "threshold_sensitivity_summary": {"threshold_sensitivity_observed": False},
        "protocol_reconciliation_summary": {"rolling_origin_does_not_override_post_sprint_underpowered": True},
        "model_saved": False,
        "scaler_saved": False,
    }
    base.update(overrides)
    return base


def test_monitor_candidate_status_active() -> None:
    gates = gate_status(candidate(), attribution())

    assert gates["continuation_gate_passed"] is True
    assert decide(gates, []) == DECISION_ACTIVE


def test_continuation_gate_passed() -> None:
    gates = gate_status(candidate(), attribution())

    assert gates["continuation_gate_passed"] is True
    assert gates["retire_gate_triggered"] is False


def test_review_gate_triggered_by_low_fold_fraction() -> None:
    gates = gate_status(candidate(fraction_folds_balanced_accuracy_above_0_5=0.5), attribution())

    assert gates["continuation_gate_passed"] is False
    assert gates["review_gate_triggered"] is True
    assert "fold_balanced_accuracy_fraction_below_60pct" in gates["review_warning_reasons"]


def test_retire_gate_triggered() -> None:
    gates = gate_status(
        candidate(
            fraction_folds_above_dummy=0.3,
            balanced_accuracy_mean=0.49,
            roc_auc_mean=0.49,
        ),
        attribution(),
    )

    assert gates["retire_gate_triggered"] is True
    assert decide(gates, []) == DECISION_RETIRED


def test_missing_rolling_origin_output_blocked(tmp_path: Path) -> None:
    attrib = tmp_path / "attrib"
    attrib.mkdir()
    write_json(attrib / "rolling_origin_stability_attribution_report.json", attribution())

    report = run_status(tmp_path / "missing", attrib, tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["status_decision"] == DECISION_BLOCKED_MISSING
    assert report["stable_evidence"] is False


def test_missing_attribution_output_blocked(tmp_path: Path) -> None:
    rolling = tmp_path / "rolling"
    rolling.mkdir()
    write_json(rolling / "rolling_origin_walk_forward_report.json", {"aggregate_stability": [candidate()]})

    report = run_status(rolling, tmp_path / "missing", tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["status_decision"] == DECISION_BLOCKED_MISSING
    assert report["qmt_ready"] is False


def test_forbidden_next_tasks_include_stable_qmt_orderintent(tmp_path: Path) -> None:
    rolling, attrib = write_fixture_inputs(tmp_path)

    report = run_status(rolling, attrib, tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)
    forbidden = report["candidate_status"]["forbidden_next_tasks"]

    assert "Stable promotion" in forbidden
    assert "QMT connection" in forbidden
    assert "OrderIntent generation" in forbidden


def test_output_path_outside_local_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_output_dir(tmp_path)


def test_boundary_fields_all_false(tmp_path: Path) -> None:
    rolling, attrib = write_fixture_inputs(tmp_path)

    report = run_status(rolling, attrib, tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["stable_promotion_ready"] is False
    assert report["stable_evidence"] is False
    assert report["formal_training_ready"] is False
    assert report["qmt_ready"] is False
    assert report["order_intent_ready"] is False
    assert report["automatic_promotion_ready"] is False


def test_no_model_or_scaler_file_written(tmp_path: Path) -> None:
    rolling, attrib = write_fixture_inputs(tmp_path)
    out_dir = tmp_path / "out"

    run_status(rolling, attrib, out_dir, repo_root=tmp_path, enforce_output_dir=False)

    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))


def write_fixture_inputs(tmp_path: Path) -> tuple[Path, Path]:
    rolling = tmp_path / "rolling"
    attrib = tmp_path / "attrib"
    rolling.mkdir()
    attrib.mkdir()
    write_json(rolling / "rolling_origin_walk_forward_report.json", {"aggregate_stability": [candidate()]})
    write_json(attrib / "rolling_origin_stability_attribution_report.json", attribution())
    return rolling, attrib


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
