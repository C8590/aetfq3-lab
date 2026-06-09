from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.lab.intraday_group_level_past_only_feature_expansion_dryrun import (
    BLOCKED_LABELS,
    CORE_FEATURES,
    GENERATED_OUTCOMES,
    TARGET_COLUMN,
    generate_feature_rows,
    load_csv_rows,
    run_dryrun,
)


FIXTURE = Path("tests/fixtures/aetfq3_lab/mock_intraday_group_level_feature_expansion_bars.csv")
DESIGN = Path("docs/research/aetfq3_intraday_group_level_past_only_feature_expansion_design.json")
CHECKLIST = Path("docs/research/aetfq3_intraday_group_level_feature_leakage_checklist.json")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_generate_core_and_optional_features_from_anchor_day_only() -> None:
    rows, _columns = load_csv_rows(FIXTURE)
    output_rows, feature_columns, skipped = generate_feature_rows(rows)

    assert len(output_rows) == 4
    for feature in CORE_FEATURES:
        assert feature in feature_columns
    assert "volume_second_half_ratio" in feature_columns
    assert "price_above_vwap_bar_ratio" in feature_columns
    assert "rank_day_return" in feature_columns
    assert "prev_1d_return" in skipped


def test_labels_and_outcomes_are_not_feature_columns() -> None:
    rows, _columns = load_csv_rows(FIXTURE)
    _output_rows, feature_columns, _skipped = generate_feature_rows(rows)

    forbidden = set(GENERATED_OUTCOMES + BLOCKED_LABELS + [TARGET_COLUMN])
    assert not forbidden.intersection(feature_columns)


def test_dryrun_writes_manifest_and_reports(tmp_path: Path) -> None:
    report = run_dryrun(
        bar_samples=FIXTURE,
        group_samples=FIXTURE,
        design_path=DESIGN,
        leakage_checklist_path=CHECKLIST,
        out_dir=tmp_path,
        enforce_allowed_output_dir=False,
        min_generated_features=18,
        min_group_count=4,
        min_anchor_count=2,
        min_etf_count=2,
        min_class_count=2,
    )

    assert report["status"] == "passed"
    assert report["generated_feature_count"] >= 18
    assert report["class_balance_precheck"]["class_count"] == 2
    assert report["training_allowed"] is False
    assert report["stable_allowed"] is False
    assert (tmp_path / "intraday_group_level_past_only_feature_samples.csv").exists()
    assert (tmp_path / "intraday_group_level_past_only_feature_manifest.json").exists()
    assert (tmp_path / "feature_quality_precheck.json").exists()
    assert (tmp_path / "class_balance_precheck.json").exists()
    assert (tmp_path / "supervised_smoke_readiness_report.json").exists()
    assert (tmp_path / "readiness_decision.json").exists()


def test_manifest_contains_required_boundary_and_metadata(tmp_path: Path) -> None:
    run_dryrun(
        bar_samples=FIXTURE,
        group_samples=FIXTURE,
        design_path=DESIGN,
        leakage_checklist_path=CHECKLIST,
        out_dir=tmp_path,
        enforce_allowed_output_dir=False,
        min_group_count=4,
        min_anchor_count=2,
        min_etf_count=2,
        min_class_count=2,
    )
    manifest = json.loads((tmp_path / "intraday_group_level_past_only_feature_manifest.json").read_text(encoding="utf-8"))

    assert manifest["sample_type"] == "intraday_5m"
    assert manifest["sample_subtype"] == "intraday_group_level_past_only_feature_expansion_dryrun"
    assert manifest["group_level_sample"] is True
    assert manifest["group_key"] == ["trade_date", "etf_code"]
    assert manifest["group_label_policy"] == "anchor_close_last_bar"
    assert manifest["feature_time_scope"] == "anchor_day_only_or_prior"
    assert manifest["label_time_scope"] == "after_anchor_day"
    assert manifest["intraday_live_decision_ready"] is False
    assert manifest["generated_feature_count"] >= 18
    assert manifest["generated_outcomes"] == GENERATED_OUTCOMES
    assert manifest["generated_labels"] == [TARGET_COLUMN]
    assert manifest["blocked_labels"] == BLOCKED_LABELS
    assert manifest["training_allowed"] is False
    assert manifest["stable_effect_allowed"] is False
    assert manifest["contains_order_intent"] is False


def test_samples_have_prefixed_group_rows_and_no_blocked_label_values(tmp_path: Path) -> None:
    run_dryrun(
        bar_samples=FIXTURE,
        group_samples=FIXTURE,
        design_path=DESIGN,
        leakage_checklist_path=CHECKLIST,
        out_dir=tmp_path,
        enforce_allowed_output_dir=False,
        min_group_count=4,
        min_anchor_count=2,
        min_etf_count=2,
        min_class_count=2,
    )
    rows = read_csv(tmp_path / "intraday_group_level_past_only_feature_samples.csv")

    assert len(rows) == 4
    assert {row["group_label_policy"] for row in rows} == {"anchor_close_last_bar"}
    assert all(row["buy_now_label"] == "" for row in rows)
    assert all(row["wait_pullback_label"] == "" for row in rows)
    assert all(row["cancel_buy_label"] == "" for row in rows)
