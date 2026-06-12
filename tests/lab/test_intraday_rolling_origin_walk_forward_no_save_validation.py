from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from tools.lab.intraday_fixed_shortlist_oop_no_save_validation import BASE_39_FEATURES, SHORTLIST
from tools.lab.intraday_rolling_origin_walk_forward_no_save_validation import (
    DECISION_NO_STABILITY,
    DECISION_STABLE,
    MIN_TRAIN_ANCHORS,
    aggregate_candidate_results,
    decide,
    make_fold_manifest,
    resolve_output_dir,
    row_level_prediction_columns,
    run_fold_candidate,
    run_validation,
    stability_observed,
    train_only_scale,
)


def make_row(trade_date: str, etf_code: str, label: int, feature_value: float, t_plus_3_date: str | None = None) -> dict[str, object]:
    row: dict[str, object] = {
        "trade_date": trade_date,
        "anchor_date": trade_date,
        "etf_code": etf_code,
        "label_ret3d_gt_100bp": label,
        "label_safe_positive_3d": label,
        "future_return_3d": 0.02 if label else -0.02,
        "max_drawdown_3d": -0.001 if label else -0.03,
        "t_plus_3_date": t_plus_3_date or trade_date,
        "t_plus_3_covered": True,
    }
    signed_feature = feature_value if label else -feature_value
    for index, feature in enumerate(BASE_39_FEATURES):
        row[feature] = signed_feature + index * 0.001
    return row


def business_dates(start: date, count: int) -> list[str]:
    output: list[str] = []
    current = start
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current.isoformat())
        current += timedelta(days=1)
    return output


def synthetic_feature_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    train_dates = business_dates(date(2025, 4, 1), 62)
    validation_dates = business_dates(date(2025, 7, 1), 12)
    etfs = ["159915", "159949", "510050", "510300", "510500", "512100"]
    for day_index, trade_date in enumerate(train_dates):
        for etf_index, etf_code in enumerate(etfs):
            label = (day_index + etf_index) % 2
            rows.append(make_row(trade_date, etf_code, label, 1.0 + day_index * 0.01, trade_date))
    for day_index, trade_date in enumerate(validation_dates):
        for etf_index, etf_code in enumerate(etfs):
            label = (day_index + etf_index) % 2
            rows.append(make_row(trade_date, etf_code, label, 1.7 + day_index * 0.01, trade_date))
    return rows


def test_fold_train_validation_time_no_overlap() -> None:
    folds = make_fold_manifest(synthetic_feature_rows(), ["2025-06-30"])
    fold = folds[0]

    assert fold["train_validation_no_overlap"] is True
    assert not set(fold["train_anchor_dates"]) & set(fold["validation_anchor_dates"])


def test_validation_window_strictly_later_than_cutoff() -> None:
    fold = make_fold_manifest(synthetic_feature_rows(), ["2025-06-30"])[0]

    assert fold["validation_strictly_after_cutoff"] is True
    assert all(anchor > "2025-06-30" for anchor in fold["validation_anchor_dates"])


def test_train_only_scaler_does_not_fit_validation_distribution() -> None:
    train = [make_row("2025-06-01", "159915", 1, 1.0), make_row("2025-06-02", "510050", 0, 1.0)]
    validation = [make_row("2025-07-01", "159915", 1, 999.0), make_row("2025-07-02", "510050", 0, 999.0)]

    scaled = train_only_scale(train, validation, BASE_39_FEATURES[:2])

    assert scaled["audit"]["fit_scope"] == "train_only"
    assert scaled["audit"]["validation_fit_performed"] is False
    assert scaled["audit"]["train_means"][0] == pytest.approx(0.0)


def test_min_train_validation_thresholds_are_enforced() -> None:
    rows = synthetic_feature_rows()[: (MIN_TRAIN_ANCHORS - 1) * 6]
    fold = make_fold_manifest(rows, ["2025-06-30"])[0]

    assert fold["skipped"] is True
    assert "min_train_anchors_not_met" in fold["skip_reasons"]
    assert "min_validation_anchors_not_met" in fold["skip_reasons"]


def test_dummy_baselines_are_generated_for_fold_candidate() -> None:
    rows = synthetic_feature_rows()
    fold = make_fold_manifest(rows, ["2025-06-30"])[0]

    result = run_fold_candidate(rows, fold, SHORTLIST[0])

    assert result["skipped"] is False
    assert "dummy_most_frequent" in result["metrics"]
    assert "dummy_stratified" in result["metrics"]
    assert result["metrics"]["dummy_most_frequent"]["validation"]["row_count"] > 0
    assert set(row_level_prediction_columns()).issubset(result["row_level_prediction_rows"][0])


def test_collapse_detection_is_reported() -> None:
    rows: list[dict[str, object]] = []
    train_dates = business_dates(date(2025, 4, 1), 62)
    validation_dates = business_dates(date(2025, 7, 1), 12)
    etfs = ["159915", "159949", "510050", "510300", "510500", "512100"]
    for index, trade_date in enumerate(train_dates):
        for etf_index, etf_code in enumerate(etfs):
            rows.append(make_row(trade_date, etf_code, etf_index % 2, 1.0 + index * 0.01, trade_date))
    for trade_date in validation_dates:
        for etf_code in etfs:
            rows.append(make_row(trade_date, etf_code, 1, 1.0, trade_date))
    fold = make_fold_manifest(rows, ["2025-06-30"])[0]

    result = run_fold_candidate(rows, fold, SHORTLIST[0])

    assert "single_class_prediction_collapse" in result["collapse_check"]["by_split"]["validation"]


def test_aggregate_fold_stability_decision() -> None:
    stable = {
        "family_id": "candidate",
        "evaluated_fold_count": 6,
        "fraction_folds_balanced_accuracy_above_0_5": 0.6,
        "fraction_folds_roc_auc_above_0_5": 0.6,
        "fraction_folds_pr_auc_not_below_prevalence": 0.6,
        "fraction_folds_non_collapse": 1.0,
    }
    unstable = {**stable, "fraction_folds_balanced_accuracy_above_0_5": 0.5}

    assert stability_observed(stable, True, True) is True
    assert stability_observed(unstable, True, True) is False
    assert decide([stable], [], True, True) == DECISION_STABLE
    assert decide([unstable], [], True, True) == DECISION_NO_STABILITY


def test_output_path_outside_local_research_outputs_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_output_dir(tmp_path)


def test_aggregate_results_include_baseline_fraction() -> None:
    rows = synthetic_feature_rows()
    fold = make_fold_manifest(rows, ["2025-06-30"])[0]
    result = run_fold_candidate(rows, fold, SHORTLIST[0])

    aggregate = aggregate_candidate_results([result])[0]

    assert aggregate["evaluated_fold_count"] == 1
    assert "fraction_folds_above_dummy" in aggregate
    assert aggregate["worst_fold"]["fold_id"] == fold["fold_id"]


def test_no_model_or_scaler_file_written_and_boundary_fields_false(tmp_path: Path) -> None:
    manual_inbox = tmp_path / "manual"
    out_dir = tmp_path / "out"
    write_manual_package(manual_inbox)

    report = run_validation(manual_inbox, out_dir, repo_root=tmp_path, enforce_output_dir=False)

    assert report["access_mode"] == "READ_ONLY"
    assert report["final_action_change_allowed"] is False
    assert report["contains_live_order"] is False
    assert report["contains_secret"] is False
    assert report["stable_promotion_ready"] is False
    assert report["qmt_ready"] is False
    assert report["order_intent_ready"] is False
    assert report["formal_training_ready"] is False
    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))


def write_manual_package(manual_inbox: Path) -> None:
    manual_inbox.mkdir(parents=True, exist_ok=True)
    (manual_inbox / "MANIFEST.json").write_text(
        """{
  "source_kind": "broker_terminal_manual_export",
  "training_allowed": false,
  "stable_effect_allowed": false,
  "contains_secret": false,
  "contains_order_intent": false,
  "contains_live_order": false,
  "contains_account": false,
  "contains_position": false,
  "contains_order": false,
  "contains_trade": false,
  "qmt_related": false
}
""",
        encoding="utf-8",
    )
    columns = ["trade_date", "datetime", "etf_code", "open", "high", "low", "close", "volume", "amount", "vwap"]
    dates = business_dates(date(2025, 4, 1), 90) + business_dates(date(2025, 7, 1), 16)
    etfs = ["159915", "159949", "510050", "510300", "510500", "512100"]
    with (manual_inbox / "historical_5m_manual_export.csv").open("w", encoding="utf-8", newline="") as f:
        import csv

        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for day_index, trade_date in enumerate(dates):
            for etf_index, etf_code in enumerate(etfs):
                base = 1.0 + etf_index * 0.1 + day_index * 0.002
                for bar_index in range(8):
                    close = base + bar_index * 0.0005
                    writer.writerow(
                        {
                            "trade_date": trade_date,
                            "datetime": f"{trade_date} 09:{35 + bar_index * 5:02d}:00",
                            "etf_code": etf_code,
                            "open": close,
                            "high": close + 0.001,
                            "low": close - 0.001,
                            "close": close,
                            "volume": 1000 + bar_index,
                            "amount": close * (1000 + bar_index),
                            "vwap": close,
                        }
                    )
