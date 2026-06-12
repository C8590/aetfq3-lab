from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.lab.intraday_rolling_origin_stability_attribution_review import (
    CLASS_BLOCKED,
    CLASS_MONITOR_READY,
    DECISION_BLOCKED_MISSING,
    FOCUS_FAMILY_ID,
    FOCUS_MODEL,
    classify_and_decide,
    etf_dispersion_rows,
    extract_winning_candidate,
    fold_robustness_rows,
    protocol_reconciliation_rows,
    resolve_output_dir,
    run_review,
    threshold_sensitivity_rows,
)


def test_winning_candidate_extracted_from_rolling_origin_output() -> None:
    report = {"aggregate_stability": [{"family_id": FOCUS_FAMILY_ID, "diagnostic_stability_observed": True}]}

    assert extract_winning_candidate(report)["family_id"] == FOCUS_FAMILY_ID


def test_fold_level_positive_fraction_computed() -> None:
    metrics = metric_rows([("2025-07", 0.60, 0.50), ("2025-08", 0.45, 0.50)])
    manifest = manifest_rows(["2025-07", "2025-08"])

    rows, summary = fold_robustness_rows(metrics, manifest)

    assert len(rows) == 2
    assert summary["positive_fold_count"] == 1
    assert rows[0]["prediction_positive_rate"] == pytest.approx(0.4)


def test_month_concentration_detection() -> None:
    metrics = metric_rows([("2025-07", 0.75, 0.50), ("2025-08", 0.51, 0.50), ("2025-09", 0.50, 0.50)])
    rows, summary = fold_robustness_rows(metrics, manifest_rows(["2025-07", "2025-08", "2025-09"]))

    assert rows[0]["positive_advantage_share"] > 0.35
    assert summary["month_concentration_observed"] is True


def test_etf_concentration_detection() -> None:
    rows = []
    rows.extend(row_level_rows_for_etf("159915", labels=[1, 1, 1, 1], predictions=[0, 0, 0, 0]))
    rows.extend(row_level_rows_for_etf("510050", labels=[0, 1, 0, 1], predictions=[0, 1, 0, 1]))
    rows.extend(row_level_rows_for_etf("588000", labels=[0, 1, 0, 1], predictions=[0, 1, 0, 1]))

    table, summary = etf_dispersion_rows(rows)

    assert summary["etf_concentration_observed"] is True
    assert any(row["etf_code"] == "159915" and row["whether_etf_concentration_observed"] for row in table)


def test_threshold_diagnostic_flags_forbid_tuning() -> None:
    rows = row_level_rows_for_etf("159915", labels=[0, 1, 0, 1], predictions=[0, 1, 0, 1], probabilities=[0.1, 0.8, 0.2, 0.9])

    table, summary = threshold_sensitivity_rows(rows)

    assert table
    assert summary["threshold_selection_allowed"] is False
    assert summary["threshold_tuned_on_walk_forward"] is False
    assert summary["threshold_sensitivity_is_diagnostic_only"] is True


def test_protocol_reconciliation_labels_backward_vs_walk_forward() -> None:
    rows = protocol_reconciliation_rows({"balanced_accuracy": 0.4778}, {"balanced_accuracy_mean": 0.5250})

    assert "backward-in-calendar" in rows[0]["validation_direction"]
    assert rows[1]["train_direction"] == "past-to-future expanding window"
    assert rows[1]["allowed_interpretation"] == "Lab diagnostic monitor candidate if no concentration is observed"


def test_missing_rolling_origin_outputs_blocked(tmp_path: Path) -> None:
    report = run_review(tmp_path / "missing", tmp_path / "fixed", tmp_path / "out", repo_root=tmp_path, enforce_output_dir=False)

    assert report["candidate_classification"] == CLASS_BLOCKED
    assert report["readiness_decision"] == DECISION_BLOCKED_MISSING
    assert report["stable_evidence"] is False


def test_output_path_outside_local_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_output_dir(tmp_path)


def test_boundary_fields_all_false_and_no_model_files_written(tmp_path: Path) -> None:
    rolling_dir = tmp_path / "rolling"
    fixed_dir = tmp_path / "fixed"
    out_dir = tmp_path / "out"
    write_fixture_outputs(rolling_dir, fixed_dir)

    report = run_review(rolling_dir, fixed_dir, out_dir, repo_root=tmp_path, enforce_output_dir=False)

    assert report["candidate_classification"] == CLASS_MONITOR_READY
    assert report["stable_promotion_ready"] is False
    assert report["stable_evidence"] is False
    assert report["formal_training_ready"] is False
    assert report["qmt_ready"] is False
    assert report["order_intent_ready"] is False
    assert report["automatic_promotion_ready"] is False
    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))


def test_classification_monitor_ready_when_no_concentration() -> None:
    candidate = {"evaluated_fold_count": 6, "fraction_folds_balanced_accuracy_above_0_5": 0.67}

    classification, _decision = classify_and_decide(
        {"fold_count": 6, "month_concentration_observed": False},
        {"month_dominates_decision": False},
        {"etf_concentration_observed": False},
        {"threshold_sensitivity_observed": False},
        candidate,
        [],
    )

    assert classification == CLASS_MONITOR_READY


def metric_rows(months: list[tuple[str, float, float]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for index, (month, logistic_ba, dummy_ba) in enumerate(months):
        fold_id = f"fold-{month}"
        cutoff = f"2025-{index + 1:02d}-28"
        common = {
            "fold_id": fold_id,
            "cutoff": cutoff,
            "validation_month": month,
            "family_id": FOCUS_FAMILY_ID,
            "label_policy": "label_ret3d_gt_100bp",
            "feature_set": "base_39_plus_scale_transform_policy",
            "split": "validation",
            "row_count": "10",
            "accuracy": str(logistic_ba),
            "roc_auc": str(logistic_ba),
            "pr_auc": str(logistic_ba),
            "label_prevalence": "0.5",
            "tn": "3",
            "fp": "2",
            "fn": "2",
            "tp": "3",
            "prediction_0": "6",
            "prediction_1": "4",
            "probability_min": "0.1",
            "probability_max": "0.9",
            "probability_mean": "0.5",
            "single_class_prediction_collapse": "False",
        }
        rows.append({**common, "model": FOCUS_MODEL, "balanced_accuracy": str(logistic_ba)})
        rows.append({**common, "model": "dummy_most_frequent", "balanced_accuracy": "0.5"})
        rows.append({**common, "model": "dummy_stratified", "balanced_accuracy": str(dummy_ba)})
    return rows


def manifest_rows(months: list[str]) -> list[dict[str, object]]:
    return [
        {
            "fold_id": f"fold-{month}",
            "validation_anchor_count": 10,
            "validation_group_count": 10,
        }
        for month in months
    ]


def row_level_rows_for_etf(
    etf_code: str,
    labels: list[int],
    predictions: list[int],
    probabilities: list[float] | None = None,
) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    probabilities = probabilities or [0.9 if prediction else 0.1 for prediction in predictions]
    for index, (label, prediction, probability) in enumerate(zip(labels, predictions, probabilities, strict=True)):
        output.append(
            {
                "fold_id": "fold-2025-07",
                "cutoff": "2025-06-30",
                "validation_month": "2025-07",
                "candidate_id": FOCUS_FAMILY_ID,
                "family_id": FOCUS_FAMILY_ID,
                "label_policy": "label_ret3d_gt_100bp",
                "feature_set": "base_39_plus_scale_transform_policy",
                "model_family": "logistic_balanced_scaled_variants",
                "model": FOCUS_MODEL,
                "anchor_date": f"2025-07-{index + 1:02d}",
                "etf_code": etf_code,
                "label": str(label),
                "prediction": str(prediction),
                "probability": str(probability),
                "is_correct": str(label == prediction),
                "error_type": error_type(label, prediction),
                "future_return_3d": "0.01",
                "t_plus_3_date": "2025-07-04",
                "train_or_oop": "validation",
            }
        )
    return output


def error_type(label: int, prediction: int) -> str:
    if label == prediction == 1:
        return "TP"
    if label == prediction == 0:
        return "TN"
    if label == 0 and prediction == 1:
        return "FP"
    return "FN"


def write_fixture_outputs(rolling_dir: Path, fixed_dir: Path) -> None:
    rolling_dir.mkdir(parents=True)
    fixed_dir.mkdir(parents=True)
    months = [f"2025-{month:02d}" for month in range(1, 7)]
    metrics = metric_rows([(month, 0.56, 0.5) for month in months])
    row_rows: list[dict[str, str]] = []
    for month in months:
        for etf in ["159915", "510050", "588000"]:
            row_rows.extend(row_level_rows_for_etf(etf, [0, 1, 0, 1], [0, 1, 0, 1]))
            for row in row_rows[-4:]:
                row["fold_id"] = f"fold-{month}"
                row["validation_month"] = month
    (rolling_dir / "rolling_origin_walk_forward_report.json").write_text(
        json.dumps(
            {
                "aggregate_stability": [
                    {
                        "family_id": FOCUS_FAMILY_ID,
                        "diagnostic_stability_observed": True,
                        "evaluated_fold_count": 6,
                        "balanced_accuracy_mean": 0.56,
                        "fraction_folds_balanced_accuracy_above_0_5": 1.0,
                    }
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (rolling_dir / "rolling_origin_fold_manifest.json").write_text(
        json.dumps({"fold_manifest": manifest_rows(months)}, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv_file(rolling_dir / "rolling_origin_fold_metrics.csv", metrics)
    write_csv_file(rolling_dir / "rolling_origin_row_level_predictions.csv", row_rows)
    write_csv_file(
        fixed_dir / "fixed_shortlist_oop_metrics.csv",
        [
            {
                "family_id": FOCUS_FAMILY_ID,
                "model": FOCUS_MODEL,
                "split": "combined_strict_oop",
                "balanced_accuracy": "0.4778",
                "roc_auc": "0.47",
                "pr_auc": "0.33",
            }
        ],
    )


def write_csv_file(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
