from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from tools.lab.intraday_lab_monitor_refresh_gate import (
    DECISION_BLOCKED_RETIRED,
    DECISION_DUE_MISSING,
    DECISION_DUE_NEW_ANCHORS,
    DECISION_DUE_NEW_MANUAL,
    DECISION_DUE_NEW_RAW,
    DECISION_NOT_DUE,
    DECISION_REVIEW,
    FOCUS_CANDIDATE_ID,
    FORBIDDEN_NEXT_TASKS,
    resolve_output_dir,
    run_refresh_gate,
)


def test_no_new_data_refresh_not_due(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_NOT_DUE
    assert report["refresh_due"] is False
    assert report["post_sprint_raw_group_count"] == 56
    assert report["post_sprint_evaluable_group_count"] == 32
    assert report["post_sprint_gate_group_count"] == 32
    assert report["post_sprint_group_count"] == 32
    assert report["group_count_basis"] == "evaluable_groups"
    assert report["next_allowed_task"] == "wait_for_new_data_or_manual_review"
    assert report["stable_evidence"] is False


def test_raw_groups_and_evaluable_groups_are_distinct(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path, post_sprint_group_count=56, post_sprint_evaluable_group_count=32)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["post_sprint_raw_group_count"] == 56
    assert report["post_sprint_evaluable_group_count"] == 32
    assert report["post_sprint_gate_group_count"] == 32
    assert report["t_plus_3_coverage_passed"] is False


def test_gate_uses_evaluable_groups_not_raw_groups(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path, post_sprint_anchor_count=10, post_sprint_group_count=80, post_sprint_evaluable_group_count=32)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_NOT_DUE
    assert report["post_sprint_raw_group_count"] == 80
    assert report["post_sprint_gate_group_count"] == 32
    assert report["rerun_gate_passed"] is False


def test_new_raw_export_detected_due_new_raw_export(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)
    (paths["raw_export_dir"] / "new_raw.csv").write_text("date,time,open\n", encoding="utf-8")
    set_mtime(paths["raw_export_dir"] / "new_raw.csv", 500)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_DUE_NEW_RAW
    assert report["new_raw_export_detected"] is True
    assert report["next_allowed_task"] == "run_broker_export_packager_and_manual_intake_validator"


def test_new_manual_package_detected_due_new_manual_package(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)
    for name in ["MANIFEST.json", "SHA256SUMS.txt", "historical_5m_manual_export.csv"]:
        set_mtime(paths["manual_inbox"] / name, 500)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_DUE_NEW_MANUAL
    assert report["new_manual_package_detected"] is True
    assert report["next_allowed_task"] == "run_fixed_shortlist_oop_and_rolling_origin_refresh"


def test_new_post_sprint_anchor_threshold_passed_due_new_anchors(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path, post_sprint_anchor_count=10, post_sprint_group_count=80, post_sprint_evaluable_group_count=80)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_DUE_NEW_ANCHORS
    assert report["post_sprint_anchor_count"] == 10
    assert report["post_sprint_raw_group_count"] == 80
    assert report["post_sprint_evaluable_group_count"] == 80
    assert report["post_sprint_gate_group_count"] == 80
    assert report["post_sprint_group_count"] == 80
    assert report["next_allowed_task"] == "rerun_fixed_shortlist_oop_no_save_validation_and_attribution"


def test_raw_groups_above_threshold_but_evaluable_below_threshold_not_ready(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path, post_sprint_anchor_count=10, post_sprint_group_count=80, post_sprint_evaluable_group_count=49)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_NOT_DUE
    assert report["post_sprint_raw_group_count"] == 80
    assert report["post_sprint_gate_group_count"] == 49
    assert report["rerun_gate_passed"] is False


def test_evaluable_group_count_unavailable_review_required(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path, evaluable_available=False)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_REVIEW
    assert report["evaluable_group_count_available"] is False
    assert report["refresh_reason"] == "evaluable_group_count_unavailable"


def test_t_plus_3_missing_groups_not_counted_as_evaluable(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path, post_sprint_group_count=56, post_sprint_evaluable_group_count=32)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["post_sprint_raw_group_count"] == 56
    assert report["post_sprint_evaluable_group_count"] == 32
    assert report["t_plus_3_coverage_passed"] is False


def test_missing_outputs_due_missing_outputs(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)
    (paths["rolling_origin_dir"] / "rolling_origin_decision.json").unlink()

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_DUE_MISSING
    assert "rolling_origin_decision" in report["missing_outputs"]


def test_retired_status_blocked_retired(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path, status_decision="LAB_MONITOR_CANDIDATE_STATUS_RETIRED_REVIEW_REQUIRED")

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_BLOCKED_RETIRED
    assert report["status"] == "blocked"


def test_forbidden_next_tasks_include_stable_qmt_orderintent_training(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert "stable_promotion" in report["forbidden_next_tasks"]
    assert "qmt_trading" in report["forbidden_next_tasks"]
    assert "order_intent_generation" in report["forbidden_next_tasks"]
    assert "formal_training" in report["forbidden_next_tasks"]
    assert set(FORBIDDEN_NEXT_TASKS).issubset(set(report["forbidden_next_tasks"]))


def test_output_path_outside_local_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_output_dir(tmp_path)


def test_boundary_fields_all_false(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)

    report = run_refresh_gate(**paths, out_dir=tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["stable_promotion_ready"] is False
    assert report["stable_evidence"] is False
    assert report["formal_training_ready"] is False
    assert report["qmt_ready"] is False
    assert report["order_intent_ready"] is False
    assert report["automatic_promotion_ready"] is False


def test_no_model_or_scaler_file_written(tmp_path: Path) -> None:
    paths = write_fixture_tree(tmp_path)
    out_dir = tmp_path / "out"

    run_refresh_gate(**paths, out_dir=out_dir, repo_root=tmp_path, enforce_output_dir=False)

    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))
    assert not list(out_dir.rglob("*.ckpt"))


def write_fixture_tree(
    tmp_path: Path,
    *,
    status_decision: str = "LAB_MONITOR_CANDIDATE_STATUS_ACTIVE_REVIEW_READY",
    post_sprint_anchor_count: int = 7,
    post_sprint_group_count: int = 56,
    post_sprint_evaluable_group_count: int = 32,
    evaluable_available: bool = True,
) -> dict[str, Path]:
    manual_inbox = tmp_path / "manual_inbox"
    raw_export_dir = tmp_path / "raw_exports"
    candidate_status_dir = tmp_path / "candidate_status"
    rolling_origin_dir = tmp_path / "rolling"
    attribution_dir = tmp_path / "attribution"
    fixed_oop_dir = tmp_path / "fixed_oop"
    reversal_dir = tmp_path / "reversal"
    for directory in [manual_inbox, raw_export_dir, candidate_status_dir, rolling_origin_dir, attribution_dir, fixed_oop_dir, reversal_dir]:
        directory.mkdir(parents=True)

    write_json(
        manual_inbox / "MANIFEST.json",
        {
            "sample_type": "historical_5m_manual_export",
            "training_allowed": False,
            "stable_effect_allowed": False,
            "contains_secret": False,
        },
    )
    (manual_inbox / "SHA256SUMS.txt").write_text("dummy  historical_5m_manual_export.csv\n", encoding="utf-8")
    write_manual_csv(manual_inbox / "historical_5m_manual_export.csv")
    (raw_export_dir / "old_raw.csv").write_text("date,time,open\n", encoding="utf-8")

    write_json(
        candidate_status_dir / "lab_monitor_candidate_status_report.json",
        {
            "status_decision": status_decision,
            "candidate_status": {
                "candidate_id": FOCUS_CANDIDATE_ID,
                "monitor_status": "LAB_MONITOR_CANDIDATE_REVIEW_READY",
            },
        },
    )
    write_json(candidate_status_dir / "lab_monitor_candidate_protocol_decision.json", {"status_decision": status_decision})
    rolling_report = {
        "readiness_decision": "ROLLING_ORIGIN_WALK_FORWARD_DIAGNOSTIC_STABILITY_OBSERVED_REVIEW_REQUIRED",
        "fold_manifest": [
            {
                "fold_id": "2026-04-30_to_2026-05",
                "validation_month": "2026-05",
                "validation_anchor_dates": ["2026-05-28", "2026-05-29"],
                "validation_anchor_count": 2,
                "validation_group_count": 16,
                "validation_etf_count": 8,
                "skipped": False,
                "skip_reasons": [],
            },
            {
                "fold_id": "2026-05-29_to_2026-06",
                "validation_month": "2026-06",
                "validation_anchor_dates": ["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"],
                "validation_anchor_count": post_sprint_anchor_count,
                "validation_group_count": post_sprint_group_count,
                "validation_etf_count": 8,
                "skipped": True,
                "skip_reasons": [] if post_sprint_anchor_count >= 10 else ["min_validation_anchors_not_met"],
            },
        ],
    }
    write_json(rolling_origin_dir / "rolling_origin_walk_forward_report.json", rolling_report)
    write_json(rolling_origin_dir / "rolling_origin_decision.json", {"readiness_decision": rolling_report["readiness_decision"]})
    write_json(rolling_origin_dir / "rolling_origin_fold_manifest.json", {"fold_manifest": rolling_report["fold_manifest"]})
    write_json(
        attribution_dir / "rolling_origin_stability_attribution_report.json",
        {"readiness_decision": "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_LAB_MONITOR_CANDIDATE_REVIEW_READY"},
    )
    write_json(
        attribution_dir / "rolling_origin_stability_attribution_decision.json",
        {"readiness_decision": "ROLLING_ORIGIN_STABILITY_ATTRIBUTION_LAB_MONITOR_CANDIDATE_REVIEW_READY"},
    )
    if evaluable_available:
        write_json(
            fixed_oop_dir / "fixed_shortlist_oop_split_manifest.json",
            {
                "post_sprint_oop": {
                    "anchor_count": post_sprint_anchor_count,
                    "etf_count": 8,
                    "group_count": post_sprint_group_count,
                    "t_plus_3_covered_group_count": post_sprint_evaluable_group_count,
                    "label_distribution": {"label_ret3d_gt_100bp": {"0": post_sprint_evaluable_group_count, "1": 0}},
                }
            },
        )
        write_post_sprint_row_level_csv(fixed_oop_dir / "fixed_shortlist_oop_row_level_predictions.csv", post_sprint_evaluable_group_count)
        write_json(
            reversal_dir / "post_sprint_reversal_attribution_report.json",
            {"sample_power": {"post_sprint_anchor_count": post_sprint_anchor_count, "post_sprint_group_count": post_sprint_evaluable_group_count}},
        )

    for path in [
        manual_inbox / "MANIFEST.json",
        manual_inbox / "SHA256SUMS.txt",
        manual_inbox / "historical_5m_manual_export.csv",
        raw_export_dir / "old_raw.csv",
    ]:
        set_mtime(path, 100)
    for path in [
        rolling_origin_dir / "rolling_origin_walk_forward_report.json",
        rolling_origin_dir / "rolling_origin_decision.json",
        rolling_origin_dir / "rolling_origin_fold_manifest.json",
        attribution_dir / "rolling_origin_stability_attribution_report.json",
        attribution_dir / "rolling_origin_stability_attribution_decision.json",
    ]:
        set_mtime(path, 200)
    for path in list(fixed_oop_dir.glob("*")) + list(reversal_dir.glob("*")):
        set_mtime(path, 200)
    for path in [
        candidate_status_dir / "lab_monitor_candidate_status_report.json",
        candidate_status_dir / "lab_monitor_candidate_protocol_decision.json",
    ]:
        set_mtime(path, 300)

    return {
        "manual_inbox": manual_inbox,
        "raw_export_dir": raw_export_dir,
        "candidate_status_dir": candidate_status_dir,
        "rolling_origin_dir": rolling_origin_dir,
        "attribution_dir": attribution_dir,
        "fixed_oop_dir": fixed_oop_dir,
        "reversal_dir": reversal_dir,
    }


def write_manual_csv(path: Path) -> None:
    rows = [
        "trade_date,datetime,etf_code,open,high,low,close,volume,amount",
        "2026-06-01,2026-06-01 09:35:00,510300,1,1,1,1,100,100",
        "2026-06-01,2026-06-01 09:35:00,159915,1,1,1,1,100,100",
        "2026-06-05,2026-06-05 09:35:00,510300,1,1,1,1,100,100",
    ]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_post_sprint_row_level_csv(path: Path, row_count: int) -> None:
    header = [
        "candidate_id",
        "model",
        "is_post_sprint_oop",
        "anchor_date",
        "etf_code",
        "label",
        "future_return_3d",
    ]
    rows = [",".join(header)]
    dates = ["2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10", "2026-06-11", "2026-06-12"]
    etfs = ["159915", "159949", "510050", "510300", "510500", "512100", "512880", "588000"]
    emitted = 0
    for date in dates:
        for etf in etfs:
            if emitted >= row_count:
                break
            rows.append(f"{FOCUS_CANDIDATE_ID},logistic_balanced_scaled,True,{date},{etf},1,0.01")
            emitted += 1
        if emitted >= row_count:
            break
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def set_mtime(path: Path, timestamp: int) -> None:
    os.utime(path, (timestamp, timestamp))
