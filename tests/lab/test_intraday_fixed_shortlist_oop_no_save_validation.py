from __future__ import annotations

from pathlib import Path
import csv
import json

import pytest

from tools.lab.intraday_fixed_shortlist_oop_no_save_validation import (
    ALLOWED_OUTPUT_DIR,
    DECISION_BLOCKED_SPLIT,
    BASE_39_FEATURES,
    ROW_LEVEL_PREDICTION_FILE,
    build_split_payload,
    collapse_check,
    decide,
    distribution_from_values,
    feature_columns_for_set,
    fit_models_for_candidate,
    label_ret3d_gt_100bp,
    label_safe_positive_3d,
    prediction_error_type,
    resolve_output_dir,
    row_level_prediction_columns,
    run_validation,
    split_anchor_dates,
    train_only_scale,
)


def make_row(trade_date: str, etf_code: str, label: int, feature_value: float) -> dict[str, object]:
    row: dict[str, object] = {
        "trade_date": trade_date,
        "anchor_date": trade_date,
        "etf_code": etf_code,
        "label_ret3d_gt_100bp": label,
        "label_safe_positive_3d": label,
        "future_return_3d": 0.02 if label else -0.01,
        "max_drawdown_3d": 0.001 if label else -0.03,
        "t_plus_3_covered": True,
    }
    for index, feature in enumerate(BASE_39_FEATURES):
        row[feature] = feature_value + index * 0.001
    return row


def sample_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date in ["2026-04-01", "2026-04-02", "2026-04-03"]:
        rows.append(make_row(date, "159915", 1, 1.0))
        rows.append(make_row(date, "510050", 0, -1.0))
    for date in ["2026-04-09", "2026-04-10", "2026-04-13", "2026-04-14"]:
        rows.append(make_row(date, "159915", 1, 1.2))
        rows.append(make_row(date, "510050", 0, -1.2))
    for date in ["2026-06-04", "2026-06-05", "2026-06-08"]:
        rows.append(make_row(date, "159915", 1, 1.4))
        rows.append(make_row(date, "510050", 0, -1.4))
    return rows


def test_split_has_no_overlap() -> None:
    split = split_anchor_dates(sample_rows())

    assert not set(split["train"]) & set(split["combined_strict_oop"])


def test_oop_anchors_are_only_outside_discovery_window() -> None:
    split = split_anchor_dates(sample_rows())

    assert all(date < "2026-04-09" or date > "2026-06-03" for date in split["combined_strict_oop"])
    assert all("2026-04-09" <= date <= "2026-06-03" for date in split["train"])


def test_train_only_scaler_uses_train_distribution_only() -> None:
    rows = sample_rows()
    train = [row for row in rows if row["trade_date"] == "2026-04-09"]
    eval_rows = [make_row("2026-06-04", "588000", 1, 999.0)]
    scaled = train_only_scale(train, eval_rows, BASE_39_FEATURES[:2])

    assert scaled["audit"]["fit_scope"] == "train_only"
    assert scaled["audit"]["eval_fit_performed"] is False
    assert scaled["audit"]["train_means"][0] == pytest.approx(0.0)


def test_label_policy_ret3d_gt_100bp() -> None:
    assert label_ret3d_gt_100bp(0.0101) == 1
    assert label_ret3d_gt_100bp(0.01) == 0
    assert label_ret3d_gt_100bp(None) is None


def test_label_policy_safe_positive_3d() -> None:
    assert label_safe_positive_3d(0.001, -0.019) == 1
    assert label_safe_positive_3d(0.001, -0.021) == 0
    assert label_safe_positive_3d(-0.001, 0.0) == 0
    assert label_safe_positive_3d(None, 0.0) is None


def test_single_class_prediction_collapse_is_detected() -> None:
    metrics = {
        "logistic_balanced_scaled": {
            "combined_strict_oop": {
                "prediction_distribution": {"0": 10, "1": 0},
                "probability_summary": {"min": 0.2, "max": 0.2, "mean": 0.2},
            }
        }
    }

    check = collapse_check(metrics)

    assert check["by_split"]["combined_strict_oop"]["single_class_prediction_collapse"] is True
    assert check["by_split"]["combined_strict_oop"]["probability_collapse"] is True


def test_dummy_baselines_are_generated() -> None:
    rows = sample_rows()
    split_dates = split_anchor_dates(rows)
    payload = build_split_payload(
        rows,
        "label_ret3d_gt_100bp",
        feature_columns_for_set("base_39_plus_scale_transform_policy"),
        split_dates,
    )
    result = fit_models_for_candidate(payload)

    assert "dummy_most_frequent" in result["metrics"]
    assert "dummy_stratified" in result["metrics"]
    assert result["metrics"]["dummy_most_frequent"]["combined_strict_oop"]["row_count"] > 0


def test_row_level_predictions_from_fit_have_complete_fields_and_errors() -> None:
    rows = sample_rows()
    split_dates = split_anchor_dates(rows)
    payload = build_split_payload(
        rows,
        "label_ret3d_gt_100bp",
        feature_columns_for_set("base_39_plus_scale_transform_policy"),
        split_dates,
    )

    result = fit_models_for_candidate(payload)
    row_predictions = result["row_level_prediction_rows"]

    assert row_predictions
    assert set(row_level_prediction_columns()) - {"candidate_id", "family_id", "feature_set", "model_family"} <= set(row_predictions[0])
    assert {row["split_name"] for row in row_predictions} == {"train", "pre_sprint_oop", "post_sprint_oop", "combined_strict_oop"}
    assert {row["error_type"] for row in row_predictions} <= {"TP", "TN", "FP", "FN", "NA"}
    assert prediction_error_type(1, 1) == "TP"
    assert prediction_error_type(0, 0) == "TN"
    assert prediction_error_type(0, 1) == "FP"
    assert prediction_error_type(1, 0) == "FN"


def test_run_validation_emits_row_level_file_and_counts_match_split_groups(tmp_path: Path) -> None:
    manual_inbox = tmp_path / "manual"
    out_dir = tmp_path / "out"
    write_manual_package(manual_inbox)

    report = run_validation(manual_inbox, out_dir, repo_root=tmp_path, enforce_output_dir=False)
    row_path = out_dir / ROW_LEVEL_PREDICTION_FILE

    assert report["row_level_predictions_emitted"] is True
    assert report["row_level_prediction_file"] == ROW_LEVEL_PREDICTION_FILE
    assert report["model_saved"] is False
    assert report["scaler_saved"] is False
    assert row_path.exists()
    with row_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert set(row_level_prediction_columns()).issubset(rows[0].keys())
    assert any(row["is_post_sprint_oop"] == "True" for row in rows)
    focus_candidate = report["candidate_results"][0]
    logistic_rows = [
        row
        for row in rows
        if row["candidate_id"] == focus_candidate["family_id"] and row["model"] == "logistic_balanced_scaled"
    ]
    counts = {split: sum(1 for row in logistic_rows if row["split_name"] == split) for split in {"train", "pre_sprint_oop", "post_sprint_oop", "combined_strict_oop"}}
    assert counts["train"] == focus_candidate["train_group_count"]
    assert counts["pre_sprint_oop"] == focus_candidate["pre_sprint_oop_group_count"]
    assert counts["post_sprint_oop"] == focus_candidate["post_sprint_oop_group_count"]
    assert counts["combined_strict_oop"] == focus_candidate["combined_strict_oop_group_count"]
    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))


def test_output_path_outside_local_research_outputs_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_output_dir(tmp_path)


def test_no_model_or_scaler_file_written(tmp_path: Path) -> None:
    rows = sample_rows()
    split_dates = split_anchor_dates(rows)
    payload = build_split_payload(
        rows,
        "label_ret3d_gt_100bp",
        feature_columns_for_set("base_39_plus_scale_transform_policy"),
        split_dates,
    )
    fit_models_for_candidate(payload)

    assert not list(tmp_path.rglob("*.pkl"))
    assert not list(tmp_path.rglob("*.joblib"))
    assert not list(tmp_path.rglob("*.pt"))


def test_blocked_decision_for_insufficient_train_or_oop() -> None:
    decision = decide([], ["train rows are empty"])

    assert decision == DECISION_BLOCKED_SPLIT
    assert distribution_from_values([0, 1, 1]) == {"0": 1, "1": 2}
    assert str(ALLOWED_OUTPUT_DIR).startswith(".local_research_outputs")


def write_manual_package(manual_inbox: Path) -> None:
    manual_inbox.mkdir(parents=True, exist_ok=True)
    (manual_inbox / "MANIFEST.json").write_text(
        json.dumps(
            {
                "source_kind": "broker_terminal_manual_export",
                "training_allowed": False,
                "stable_effect_allowed": False,
                "contains_secret": False,
                "contains_order_intent": False,
                "contains_live_order": False,
                "contains_account": False,
                "contains_position": False,
                "contains_order": False,
                "contains_trade": False,
                "qmt_related": False,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    columns = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume", "amount", "vwap"]
    dates = [
        "2026-04-01",
        "2026-04-02",
        "2026-04-03",
        "2026-04-06",
        "2026-04-07",
        "2026-04-08",
        "2026-04-09",
        "2026-04-10",
        "2026-04-13",
        "2026-04-14",
        "2026-04-15",
        "2026-04-16",
        "2026-06-04",
        "2026-06-05",
        "2026-06-08",
        "2026-06-09",
        "2026-06-10",
        "2026-06-11",
        "2026-06-12",
        "2026-06-15",
        "2026-06-16",
    ]
    rows: list[dict[str, object]] = []
    for index, trade_date in enumerate(dates):
        rows.extend(day_bar_rows(trade_date, "159915", 1.0 + index * 0.02, 0.0005))
        rows.extend(day_bar_rows(trade_date, "510050", 2.0 - index * 0.02, -0.0005))
    with (manual_inbox / "historical_5m_manual_export.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def day_bar_rows(trade_date: str, etf_code: str, close_last: float, drift: float) -> list[dict[str, object]]:
    morning = [f"{hour:02d}:{minute:02d}:00" for hour, minute in [(9, 35), (9, 40), (9, 45), (9, 50), (9, 55), (10, 0), (10, 5), (10, 10), (10, 15), (10, 20), (10, 25), (10, 30), (10, 35), (10, 40), (10, 45), (10, 50), (10, 55), (11, 0), (11, 5), (11, 10), (11, 15), (11, 20), (11, 25), (11, 30)]]
    afternoon = [f"{hour:02d}:{minute:02d}:00" for hour, minute in [(13, 5), (13, 10), (13, 15), (13, 20), (13, 25), (13, 30), (13, 35), (13, 40), (13, 45), (13, 50), (13, 55), (14, 0), (14, 5), (14, 10), (14, 15), (14, 20), (14, 25), (14, 30), (14, 35), (14, 40), (14, 45), (14, 50), (14, 55), (15, 0)]]
    times = morning + afternoon
    start = close_last - drift * 47
    return [bar_row(trade_date, etf_code, start + index * drift, time_text, index) for index, time_text in enumerate(times)]


def bar_row(trade_date: str, etf_code: str, close: float, time_text: str, index: int) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "datetime": f"{trade_date} {time_text}",
        "etf_code": etf_code,
        "open": close * (1 - 0.0002),
        "high": close * (1 + 0.001 + index * 0.000001),
        "low": close * (1 - 0.001),
        "close": close,
        "volume": 1000 + index,
        "amount": close * (1000 + index),
        "vwap": close,
    }
