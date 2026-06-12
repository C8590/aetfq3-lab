from __future__ import annotations

from pathlib import Path

import pytest

from tools.lab.intraday_fixed_shortlist_oop_no_save_validation import (
    ALLOWED_OUTPUT_DIR,
    DECISION_BLOCKED_SPLIT,
    BASE_39_FEATURES,
    build_split_payload,
    collapse_check,
    decide,
    distribution_from_values,
    feature_columns_for_set,
    fit_models_for_candidate,
    label_ret3d_gt_100bp,
    label_safe_positive_3d,
    resolve_output_dir,
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
