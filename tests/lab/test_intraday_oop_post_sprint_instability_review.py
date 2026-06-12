from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.lab.intraday_oop_post_sprint_instability_review import (
    DECISION_BLOCKED_MISSING,
    BASE_39_FEATURES,
    date_concentration_check,
    decide,
    etf_breakdown_rows,
    feature_shift_rows,
    label_shift_rows,
    resolve_output_dir,
    run_review,
    sample_power_check,
    validate_oop_outputs,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_row(date: str, etf: str, label: int, value: float) -> dict[str, object]:
    row: dict[str, object] = {
        "trade_date": date,
        "etf_code": etf,
        "label_ret3d_gt_100bp": label,
        "label_safe_positive_3d": label,
        "future_return_3d": 0.02 if label else -0.01,
        "t_plus_3_covered": True,
    }
    for feature in BASE_39_FEATURES:
        row[feature] = value
    row["volume_sum"] = value * 1000
    row["amount_sum"] = value * 2000
    row["day_return"] = value / 100
    row["intraday_return_std"] = value / 200
    return row


def rows_by_split() -> dict[str, list[dict[str, object]]]:
    train = [make_row("2026-04-09", "159915", 0, 1.0), make_row("2026-04-09", "510050", 1, 1.1)]
    pre = [make_row("2026-04-01", "159915", 0, 1.2), make_row("2026-04-01", "510050", 1, 1.3)]
    post = [
        make_row("2026-06-04", "159915", 1, 10.0),
        make_row("2026-06-04", "159915", 1, 10.5),
        make_row("2026-06-05", "510050", 1, 11.0),
    ]
    return {"train": train, "pre_sprint_oop": pre, "post_sprint_oop": post, "combined_strict_oop": [*pre, *post]}


def test_post_sprint_sample_underpowered_flag() -> None:
    check = sample_power_check({"post_sprint_oop_anchor_dates": ["2026-06-04", "2026-06-05"]}, rows_by_split()["post_sprint_oop"])

    assert check["post_sprint_oop_underpowered"] is True


def test_label_distribution_shift_detection() -> None:
    rows, check = label_shift_rows(rows_by_split(), ["label_safe_positive_3d"])

    assert check["label_shift_observed"] is True
    assert any(row["split"] == "post_sprint_oop" and row["label_shift_observed"] for row in rows)


def test_etf_concentration_detection() -> None:
    _rows, check = etf_breakdown_rows(rows_by_split())

    assert check["etf_concentration_observed"] is True
    assert check["max_group_share"] > 0.5


def test_date_concentration_detection() -> None:
    anchor_rows = [
        {"anchor_date": "2026-06-04", "group_share_of_post": 0.8, "positive_rate": 1.0},
        {"anchor_date": "2026-06-05", "group_share_of_post": 0.2, "positive_rate": 0.5},
    ]

    check = date_concentration_check(anchor_rows)

    assert check["date_concentration_observed"] is True


def test_feature_shift_table_generation() -> None:
    rows, check = feature_shift_rows(rows_by_split(), BASE_39_FEATURES[:3])

    assert len(rows) == 3
    assert "post_vs_train_smd" in rows[0]
    assert check["feature_shift_observed"] is True


def test_missing_oop_outputs_blocked(tmp_path: Path) -> None:
    write_json(tmp_path / "fixed_shortlist_oop_validation_report.json", {})
    check = validate_oop_outputs(tmp_path)
    decision = decide({"data_quality_blocked": False}, missing_outputs=True)

    assert check["passed"] is False
    assert decision == DECISION_BLOCKED_MISSING


def test_output_path_outside_local_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_output_dir(tmp_path)


def test_boundary_fields_all_false_in_report(tmp_path: Path) -> None:
    oop_dir = tmp_path / "oop"
    manual_inbox = tmp_path / "manual"
    write_minimal_oop_outputs(oop_dir)
    write_minimal_manual_csv(manual_inbox)

    report = run_review(oop_dir, manual_inbox, tmp_path / "out", enforce_output_dir=False)

    assert report["formal_training_ready"] is False
    assert report["qmt_ready"] is False
    assert report["order_intent_ready"] is False
    assert report["automatic_promotion_ready"] is False
    assert report["stable_promotion_ready"] is False


def test_no_model_or_scaler_file_written(tmp_path: Path) -> None:
    oop_dir = tmp_path / "oop"
    manual_inbox = tmp_path / "manual"
    out_dir = tmp_path / "out"
    write_minimal_oop_outputs(oop_dir)
    write_minimal_manual_csv(manual_inbox)

    run_review(oop_dir, manual_inbox, out_dir, enforce_output_dir=False)

    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))


def write_minimal_oop_outputs(oop_dir: Path) -> None:
    write_json(
        oop_dir / "fixed_shortlist_oop_validation_report.json",
        {
            "candidate_results": [
                {
                    "family_id": "label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy",
                    "combined_strict_oop_minimum_metrics_pass": True,
                    "diagnostic_signal_survives_minimum_standard": False,
                }
            ]
        },
    )
    write_json(
        oop_dir / "fixed_shortlist_oop_split_manifest.json",
        {
            "train_anchor_dates": ["2026-04-09"],
            "pre_sprint_oop_anchor_dates": ["2026-04-01"],
            "post_sprint_oop_anchor_dates": ["2026-06-04"],
            "combined_strict_oop_anchor_dates": ["2026-04-01", "2026-06-04"],
        },
    )
    write_json(oop_dir / "fixed_shortlist_oop_decision.json", {"readiness_decision": "x"})
    columns = ["family_id", "label_policy", "feature_set", "model", "split", "row_count", "balanced_accuracy", "roc_auc", "pr_auc"]
    rows = [
        {
            "family_id": "label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy",
            "label_policy": "label_safe_positive_3d",
            "feature_set": "base_39_plus_scale_transform_policy",
            "model": "logistic_balanced_scaled",
            "split": split,
            "row_count": 1,
            "balanced_accuracy": 0.4,
            "roc_auc": 0.4,
            "pr_auc": 0.4,
        }
        for split in ["pre_sprint_oop", "post_sprint_oop", "combined_strict_oop"]
    ]
    write_csv(oop_dir / "fixed_shortlist_oop_metrics.csv", rows, columns)
    write_csv(
        oop_dir / "fixed_shortlist_oop_predictions_summary.csv",
        [
            {
                "family_id": "label_safe_positive_3d|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy",
                "model": "logistic_balanced_scaled",
                "split": "post_sprint_oop",
                "row_count": 1,
                "prediction_distribution": '{"0": 1, "1": 0}',
                "probability_min": 0.2,
                "probability_max": 0.2,
                "probability_mean": 0.2,
            }
        ],
        ["family_id", "model", "split", "row_count", "prediction_distribution", "probability_min", "probability_max", "probability_mean"],
    )


def write_minimal_manual_csv(manual_inbox: Path) -> None:
    columns = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume", "amount", "vwap"]
    rows = [
        {"trade_date": "2026-04-01", "datetime": "2026-04-01 09:35:00", "etf_code": "159915", "open": 1, "high": 1.1, "low": 0.9, "close": 1, "volume": 100, "amount": 100, "vwap": 1},
        {"trade_date": "2026-04-09", "datetime": "2026-04-09 09:35:00", "etf_code": "159915", "open": 1, "high": 1.1, "low": 0.9, "close": 1.1, "volume": 100, "amount": 110, "vwap": 1.1},
        {"trade_date": "2026-06-04", "datetime": "2026-06-04 09:35:00", "etf_code": "159915", "open": 1, "high": 1.1, "low": 0.9, "close": 1.2, "volume": 100, "amount": 120, "vwap": 1.2},
        {"trade_date": "2026-06-05", "datetime": "2026-06-05 09:35:00", "etf_code": "159915", "open": 1, "high": 1.1, "low": 0.9, "close": 1.3, "volume": 100, "amount": 130, "vwap": 1.3},
        {"trade_date": "2026-06-08", "datetime": "2026-06-08 09:35:00", "etf_code": "159915", "open": 1, "high": 1.1, "low": 0.9, "close": 1.4, "volume": 100, "amount": 140, "vwap": 1.4},
        {"trade_date": "2026-06-09", "datetime": "2026-06-09 09:35:00", "etf_code": "159915", "open": 1, "high": 1.1, "low": 0.9, "close": 1.5, "volume": 100, "amount": 150, "vwap": 1.5},
    ]
    write_csv(manual_inbox / "historical_5m_manual_export.csv", rows, columns)
