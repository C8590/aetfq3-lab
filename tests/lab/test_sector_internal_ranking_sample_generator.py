from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.sector_internal_ranking_sample_generator import (  # noqa: E402
    FEATURE_COLUMNS,
    FUTURE_LABEL_COLUMNS,
    GeneratorError,
    generate_sector_internal_ranking_sample,
)


MOCK_DAILY = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_etf_daily_for_sector_ranking.csv"
MOCK_SECTOR_MAP = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_sector_map_for_sector_ranking.yaml"


def run_mock_generator(tmp_path: Path):
    out_dir = (
        REPO_ROOT
        / ".local_research_outputs/aetfq3_lab/sector_internal_ranking_expanded/pytest"
        / tmp_path.name
    )
    return generate_sector_internal_ranking_sample(
        max_trading_days=6,
        max_etfs=4,
        min_etfs_per_sector=2,
        out_dir=out_dir,
        source="mock",
        mock_daily_csv=MOCK_DAILY,
        mock_sector_map=MOCK_SECTOR_MAP,
        skip_baseline_smoke=True,
    )


def test_mock_data_generates_sample(tmp_path: Path):
    result = run_mock_generator(tmp_path)

    assert result.paths.sample.exists()
    assert result.paths.feature_sample.exists()
    assert result.paths.manifest.exists()
    sample = pd.read_csv(result.paths.feature_sample, dtype={"etf_code": str})
    assert not sample.empty
    assert result.report["sample_summary"]["row_count"] == len(sample)
    assert result.report["sample_summary"]["feature_count"] == len(FEATURE_COLUMNS)


def test_ranking_group_id_and_group_size(tmp_path: Path):
    result = run_mock_generator(tmp_path)
    sample = pd.read_csv(result.paths.feature_sample, dtype={"etf_code": str})

    assert (sample["ranking_group_id"] == sample["trade_date"] + "_" + sample["sector"]).all()
    group_sizes = sample.groupby("ranking_group_id")["etf_code"].nunique()
    assert int(group_sizes.min()) >= 2


def test_future_labels_not_in_feature_columns(tmp_path: Path):
    result = run_mock_generator(tmp_path)

    manifest = json.loads(result.paths.manifest.read_text(encoding="utf-8"))
    assert set(FUTURE_LABEL_COLUMNS).isdisjoint(manifest["feature_columns"])
    assert "top_quantile_in_sector_3d" not in manifest["feature_columns"]
    assert "pairwise_outperform_label" not in manifest["feature_columns"]


def test_feature_columns_are_lag1_past_only_or_allowed_count(tmp_path: Path):
    result = run_mock_generator(tmp_path)
    manifest = json.loads(result.paths.manifest.read_text(encoding="utf-8"))

    for column in manifest["feature_columns"]:
        assert column == "sector_etf_count" or column.endswith("_lag1")


def test_manifest_boundary_fields(tmp_path: Path):
    result = run_mock_generator(tmp_path)
    manifest = json.loads(result.paths.manifest.read_text(encoding="utf-8"))

    assert manifest["uses_stable_bundle"] is False
    assert manifest["training_allowed"] is False
    assert manifest["stable_effect_allowed"] is False
    assert manifest["advisory_only"] is True
    assert manifest["affects_stable_trading"] is False
    assert manifest["contains_order_intent"] is False
    assert manifest["qmt_related"] is False
    assert manifest["has_future_leakage_check"] is True
    assert manifest["sample_path"].startswith(".local_research_outputs/aetfq3_lab/")


def test_generated_json_is_parseable(tmp_path: Path):
    result = run_mock_generator(tmp_path)

    assert json.loads(result.paths.report_json.read_text(encoding="utf-8"))["status"] == "generated"
    assert json.loads(result.paths.feature_contract.read_text(encoding="utf-8"))["feature_columns"] == FEATURE_COLUMNS


def test_does_not_write_output_dir(tmp_path: Path):
    with pytest.raises(GeneratorError, match="out-dir must be under"):
        generate_sector_internal_ranking_sample(
            max_trading_days=6,
            max_etfs=4,
            min_etfs_per_sector=2,
            out_dir=tmp_path / "output",
            source="mock",
            mock_daily_csv=MOCK_DAILY,
            mock_sector_map=MOCK_SECTOR_MAP,
        )

    result = run_mock_generator(tmp_path)
    assert not (tmp_path / "output").exists()


def test_no_model_or_checkpoint_files(tmp_path: Path):
    result = run_mock_generator(tmp_path)
    generated_files = [path.name for path in result.paths.report_json.parent.iterdir()]

    assert all(not name.endswith((".pkl", ".joblib", ".cbm", ".model", ".bin", ".pt", ".pth")) for name in generated_files)
    assert all("checkpoint" not in name.lower() for name in generated_files)
