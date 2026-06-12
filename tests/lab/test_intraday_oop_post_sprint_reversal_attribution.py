from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.lab.intraday_oop_post_sprint_reversal_attribution import (
    DECISION_BLOCKED_MISSING_ROW,
    FOCUS_FAMILY_ID,
    date_attribution_rows,
    etf_attribution_rows,
    feature_shift_rows,
    label_regime_rows,
    resolve_output_dir,
    run_attribution,
    threshold_sensitivity_rows,
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


def base_row(split: str, date: str, etf: str, label: int, probability: float) -> dict[str, object]:
    prediction = int(probability >= 0.5)
    if label == 1 and prediction == 1:
        error_type = "TP"
    elif label == 0 and prediction == 0:
        error_type = "TN"
    elif label == 0 and prediction == 1:
        error_type = "FP"
    else:
        error_type = "FN"
    return {
        "candidate_id": FOCUS_FAMILY_ID,
        "family_id": FOCUS_FAMILY_ID,
        "label_policy": "label_safe_positive_3d",
        "feature_set": "base_39_plus_scale_transform_policy",
        "model_family": "logistic_balanced_scaled_variants",
        "model": "logistic_balanced_scaled",
        "split_name": split,
        "anchor_date": date,
        "etf_code": etf,
        "label": label,
        "prediction": prediction,
        "probability": probability,
        "is_correct": label == prediction,
        "error_type": error_type,
        "future_return_3d": 0.02 if label else -0.01,
        "train_or_oop": "train" if split == "train" else "oop",
        "is_pre_sprint_oop": split == "pre_sprint_oop",
        "is_post_sprint_oop": split == "post_sprint_oop",
        "is_combined_oop": split == "combined_strict_oop",
    }


def row_level_rows() -> list[dict[str, object]]:
    rows = [
        base_row("train", "2026-04-09", "159915", 0, 0.2),
        base_row("train", "2026-04-09", "510050", 1, 0.8),
        base_row("pre_sprint_oop", "2026-04-01", "159915", 0, 0.3),
        base_row("pre_sprint_oop", "2026-04-01", "510050", 1, 0.7),
        base_row("post_sprint_oop", "2026-06-04", "159915", 0, 0.8),
        base_row("post_sprint_oop", "2026-06-04", "510050", 0, 0.7),
        base_row("post_sprint_oop", "2026-06-08", "159915", 1, 0.2),
        base_row("post_sprint_oop", "2026-06-08", "510050", 1, 0.9),
    ]
    rows.extend({**row, "split_name": "combined_strict_oop", "is_combined_oop": True} for row in rows if row["split_name"] in {"pre_sprint_oop", "post_sprint_oop"})
    return rows


def test_date_level_fp_fn_attribution() -> None:
    rows, summary = date_attribution_rows(row_level_rows())

    assert rows[0]["anchor_date"] == "2026-06-04"
    assert rows[0]["fp"] == 2
    assert rows[1]["fn"] == 1
    assert summary["post_sprint_total_error"] == 3
    assert summary["front_back_regime_reversal_observed"] is True


def test_etf_error_share_concentration_requires_group_and_error_share() -> None:
    rows, summary = etf_attribution_rows(
        [
            base_row("post_sprint_oop", "2026-06-04", "159915", 0, 0.8),
            base_row("post_sprint_oop", "2026-06-05", "159915", 1, 0.2),
            base_row("post_sprint_oop", "2026-06-06", "159915", 0, 0.9),
            base_row("post_sprint_oop", "2026-06-07", "159915", 1, 0.1),
            base_row("post_sprint_oop", "2026-06-04", "510050", 0, 0.1),
            base_row("post_sprint_oop", "2026-06-05", "510050", 1, 0.9),
        ]
    )

    assert summary["etf_concentration_primary"] is True
    assert next(row for row in rows if row["etf_code"] == "159915")["whether_etf_dominates_reversal"] is True


def test_threshold_grid_sensitivity_and_forbidden_flags() -> None:
    rows, summary = threshold_sensitivity_rows(row_level_rows())

    assert len(rows) == 36
    assert {row["threshold"] for row in rows if row["split_name"] == "post_sprint_oop"} == {0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7}
    assert summary["threshold_selection_allowed"] is False
    assert summary["threshold_tuned_on_post_sprint"] is False
    assert summary["threshold_sensitivity_is_diagnostic_only"] is True


def test_label_regime_shift_attribution() -> None:
    rows, summary = label_regime_rows(row_level_rows())

    post = next(row for row in rows if row["split_name"] == "post_sprint_oop")
    assert post["positive_rate"] == 0.5
    assert summary["label_shift_primary"] is False


def test_feature_shift_attribution_table(tmp_path: Path) -> None:
    manual_inbox = tmp_path / "manual"
    write_manual_csv(manual_inbox)
    split_manifest = {
        "train_anchor_dates": ["2026-04-09"],
        "pre_sprint_oop_anchor_dates": ["2026-04-01"],
        "post_sprint_oop_anchor_dates": ["2026-06-04"],
    }

    rows, summary = feature_shift_rows(manual_inbox, split_manifest, tmp_path)

    assert rows
    assert {"volume", "amount", "intraday_return", "volatility", "position/range"} & {row["feature_group"] for row in rows}
    assert "feature_shift_primary" in summary


def test_missing_row_level_diagnostics_blocked(tmp_path: Path) -> None:
    report = run_attribution(tmp_path / "oop", tmp_path / "instability", tmp_path / "manual", tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["readiness_decision"] == DECISION_BLOCKED_MISSING_ROW
    assert report["status"] == "blocked"


def test_output_path_outside_local_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_output_dir(tmp_path)


def test_boundary_fields_all_false_and_no_model_or_scaler_written(tmp_path: Path) -> None:
    oop_dir = tmp_path / "oop"
    instability_dir = tmp_path / "instability"
    manual_inbox = tmp_path / "manual"
    out_dir = tmp_path / "out"
    write_complete_inputs(oop_dir, instability_dir, manual_inbox)

    report = run_attribution(oop_dir, instability_dir, manual_inbox, out_dir, repo_root=tmp_path, enforce_output_dir=False)

    assert report["formal_training_ready"] is False
    assert report["stable_promotion_ready"] is False
    assert report["qmt_ready"] is False
    assert report["order_intent_ready"] is False
    assert report["automatic_promotion_ready"] is False
    assert report["threshold_selection_allowed"] is False
    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))


def write_complete_inputs(oop_dir: Path, instability_dir: Path, manual_inbox: Path) -> None:
    rows = row_level_rows()
    write_csv(oop_dir / "fixed_shortlist_oop_row_level_predictions.csv", rows, list(rows[0].keys()))
    write_json(
        oop_dir / "fixed_shortlist_oop_validation_report.json",
        {
            "split_manifest": {
                "train_anchor_dates": ["2026-04-09"],
                "pre_sprint_oop_anchor_dates": ["2026-04-01"],
                "post_sprint_oop_anchor_dates": ["2026-06-04"],
            }
        },
    )
    write_json(
        instability_dir / "post_sprint_instability_review_report.json",
        {
            "sample_power": {
                "post_sprint_anchor_count": 1,
                "post_sprint_group_count": 4,
                "post_sprint_oop_underpowered": True,
                "minimum_anchor_count": 10,
                "minimum_group_count": 50,
            }
        },
    )
    write_manual_csv(manual_inbox)


def write_manual_csv(manual_inbox: Path) -> None:
    columns = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume", "amount", "vwap"]
    dates = ["2026-04-01", "2026-04-02", "2026-04-03", "2026-04-06", "2026-04-09", "2026-04-10", "2026-04-13", "2026-04-14", "2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09"]
    rows: list[dict[str, object]] = []
    for index, trade_date in enumerate(dates):
        rows.extend(day_bar_rows(trade_date, "159915", 1.0 + index * 0.03, 0.0006))
        rows.extend(day_bar_rows(trade_date, "510050", 2.0 - index * 0.01, -0.0004))
    write_csv(manual_inbox / "historical_5m_manual_export.csv", rows, columns)


def day_bar_rows(trade_date: str, etf_code: str, close_last: float, drift: float) -> list[dict[str, object]]:
    times = ["09:35:00", "09:40:00", "09:45:00", "09:50:00", "09:55:00", "10:00:00", "10:05:00", "10:10:00", "10:15:00", "10:20:00", "10:25:00", "10:30:00", "10:35:00", "10:40:00", "10:45:00", "10:50:00", "10:55:00", "11:00:00", "11:05:00", "11:10:00", "11:15:00", "11:20:00", "11:25:00", "11:30:00", "13:05:00", "13:10:00", "13:15:00", "13:20:00", "13:25:00", "13:30:00", "13:35:00", "13:40:00", "13:45:00", "13:50:00", "13:55:00", "14:00:00", "14:05:00", "14:10:00", "14:15:00", "14:20:00", "14:25:00", "14:30:00", "14:35:00", "14:40:00", "14:45:00", "14:50:00", "14:55:00", "15:00:00"]
    start = close_last - drift * 47
    return [bar_row(trade_date, etf_code, start + index * drift, time_text, index) for index, time_text in enumerate(times)]


def bar_row(trade_date: str, etf_code: str, close: float, time_text: str, index: int) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "datetime": f"{trade_date} {time_text}",
        "etf_code": etf_code,
        "open": close * 0.999,
        "high": close * 1.002,
        "low": close * 0.998,
        "close": close,
        "volume": 1000 + index,
        "amount": close * (1000 + index),
        "vwap": close,
    }
