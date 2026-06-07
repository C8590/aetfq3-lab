from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.table_ml_baseline_smoke import BaselineSmokeError, parse_model_names, run_baseline_smoke
from tools.lab.table_ml_baseline_report_reader import find_prohibited_fields, summarize_report


SAMPLE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_sector_internal_ranking_feature_samples.csv"
MANIFEST = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_sector_internal_ranking_feature_manifest.json"
FEATURE_COLUMNS = [
    "etf_ret_1d_lag1",
    "etf_ret_3d_lag1",
    "etf_amount_5d_mean_lag1",
    "sector_ret_3d_mean_lag1",
    "etf_vs_sector_ret_3d_lag1",
]


def write_contract(tmp_path: Path, feature_columns: list[str] | None = None) -> Path:
    path = tmp_path / "feature_contract.json"
    payload = {
        "field_classification": {
            "candidate_features": feature_columns or FEATURE_COLUMNS,
            "numeric_candidate_features": feature_columns or FEATURE_COLUMNS,
        }
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_manifest(tmp_path: Path, **overrides) -> Path:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload.update(overrides)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def copy_sample(tmp_path: Path) -> Path:
    path = tmp_path / "sample.csv"
    shutil.copyfile(SAMPLE, path)
    return path


def test_mock_feature_sample_runs(tmp_path: Path):
    report = run_baseline_smoke(
        sample_path=SAMPLE,
        manifest_path=MANIFEST,
        feature_contract_path=write_contract(tmp_path),
        target="top_quantile_in_sector_3d",
        out_dir=tmp_path / "out",
    )

    assert report["status"] == "passed"
    assert report["report_type"] == "table_ml_baseline_smoke"
    assert report["task_scope"] == "Lab-only no-save baseline smoke"
    assert report["metrics"][0]["train_count"] == 14
    assert report["metrics"][0]["valid_count"] == 6
    assert report["metrics"][0]["feature_count"] == len(FEATURE_COLUMNS)
    assert [model["model_name"] for model in report["models"]] == ["numpy_logistic_regression_smoke"]
    assert Path(report["prediction_file"]).exists()


def test_parse_multi_model_aliases():
    assert parse_model_names("numpy_logistic,lightgbm,catboost,xgboost") == [
        "numpy_logistic",
        "lightgbm",
        "catboost",
        "xgboost",
    ]
    assert parse_model_names(["numpy_logistic_regression_smoke", "lightgbm_smoke"]) == [
        "numpy_logistic",
        "lightgbm",
    ]


def test_unknown_model_alias_fails():
    with pytest.raises(BaselineSmokeError, match="unknown model alias"):
        parse_model_names("numpy_logistic,not_a_model")


def test_forbidden_feature_fails(tmp_path: Path):
    with pytest.raises(BaselineSmokeError, match="forbidden columns"):
        run_baseline_smoke(
            sample_path=SAMPLE,
            manifest_path=MANIFEST,
            feature_contract_path=write_contract(tmp_path, [*FEATURE_COLUMNS, "future_return_3d"]),
            target="top_quantile_in_sector_3d",
            out_dir=tmp_path / "out",
        )


def test_training_allowed_true_fails(tmp_path: Path):
    manifest = write_manifest(tmp_path, training_allowed=True)

    with pytest.raises(BaselineSmokeError, match="training_allowed must be false"):
        run_baseline_smoke(
            sample_path=SAMPLE,
            manifest_path=manifest,
            feature_contract_path=write_contract(tmp_path),
            target="top_quantile_in_sector_3d",
            out_dir=tmp_path / "out",
        )


def test_group_leakage_check_fails(tmp_path: Path):
    sample = copy_sample(tmp_path)
    df = pd.read_csv(sample, dtype={"etf_code": str})
    df.loc[df["trade_date"] == "2026-05-08", "ranking_group_id"] = "2026-05-01_科技成长"
    df.to_csv(sample, index=False)

    with pytest.raises(BaselineSmokeError, match="ranking_group_id appears in both train and validation"):
        run_baseline_smoke(
            sample_path=sample,
            manifest_path=MANIFEST,
            feature_contract_path=write_contract(tmp_path),
            target="top_quantile_in_sector_3d",
            out_dir=tmp_path / "out",
        )


def test_output_json_contains_required_fields(tmp_path: Path):
    out_dir = tmp_path / "out"
    run_baseline_smoke(
        sample_path=SAMPLE,
        manifest_path=MANIFEST,
        feature_contract_path=write_contract(tmp_path),
        target="top_quantile_in_sector_3d",
        out_dir=out_dir,
    )

    report = json.loads((out_dir / "sector_internal_ranking_baseline_smoke_report.json").read_text(encoding="utf-8"))
    assert report["report_type"] == "table_ml_baseline_smoke"
    assert report["task_scope"] == "Lab-only no-save baseline smoke"
    assert report["lab_only"] is True
    assert report["no_save"] is True
    assert report["no_tuning"] is True
    assert report["no_stable"] is True
    assert report["no_qmt"] is True
    assert report["no_order_intent"] is True
    assert report["no_output"] is True
    assert report["no_lab_advisory"] is True
    assert report["model_saved"] is False
    assert report["checkpoint_saved"] is False
    assert report["target_label"] == "top_quantile_in_sector_3d"
    assert report["feature_columns"] == FEATURE_COLUMNS
    assert "future_return_3d" in report["forbidden_columns"]
    assert report["train_count"] == 14
    assert report["valid_count"] == 6
    assert report["split_method"] == "chronological"
    assert report["group_leakage_check"] == "passed"
    assert isinstance(report["models"], list) and report["models"]
    assert report["models"][0]["status"] == "passed"
    assert report["models"][0]["no_save"] is True
    assert report["models"][0]["no_tuning"] is True
    assert report["models"][0]["model_saved"] is False
    assert report["models"][0]["checkpoint_saved"] is False
    assert isinstance(report["metrics"], list) and report["metrics"]
    assert report["metrics"][0]["no_save"] is True
    assert report["metrics"][0]["no_tuning"] is True
    assert report["metrics"][0]["model_saved"] is False
    assert report["metrics"][0]["checkpoint_saved"] is False
    assert isinstance(report["review_checklist"], dict)
    assert report["task"] == "sector_internal_ranking_baseline_smoke"
    assert report["boundary"]["no_model_save"] is True
    assert report["feature_leakage_check"]["feature_forbidden_intersection"] == []
    assert report["split"]["group_leakage_check_passed"] is True
    assert report["metrics"][0]["target_label"] == "top_quantile_in_sector_3d"
    assert find_prohibited_fields(report) == []


def test_writer_output_passes_reader_contract(tmp_path: Path):
    out_dir = tmp_path / "out"
    run_baseline_smoke(
        sample_path=SAMPLE,
        manifest_path=MANIFEST,
        feature_contract_path=write_contract(tmp_path),
        target="top_quantile_in_sector_3d",
        out_dir=out_dir,
    )

    summary = summarize_report(out_dir / "sector_internal_ranking_baseline_smoke_report.json")
    assert summary["status"] == "OK"
    assert summary["report_type"] == "table_ml_baseline_smoke"
    assert summary["task_scope"] == "Lab-only no-save baseline smoke"
    assert summary["models"] == ["numpy_logistic_regression_smoke"]
    assert summary["train_count"] == 14
    assert summary["valid_count"] == 6


def test_multi_model_report_passes_reader_contract(tmp_path: Path):
    out_dir = tmp_path / "out"
    report = run_baseline_smoke(
        sample_path=SAMPLE,
        manifest_path=MANIFEST,
        feature_contract_path=write_contract(tmp_path),
        target="top_quantile_in_sector_3d",
        out_dir=out_dir,
        models="numpy_logistic,lightgbm,catboost,xgboost",
    )

    model_names = [model["model_name"] for model in report["models"]]
    assert model_names == [
        "numpy_logistic_regression_smoke",
        "lightgbm_smoke",
        "catboost_smoke",
        "xgboost_smoke",
    ]
    for model in report["models"]:
        assert model["status"] in {"passed", "skipped"}
        assert model["no_save"] is True
        assert model["no_tuning"] is True
        assert model["model_saved"] is False
        assert model["checkpoint_saved"] is False
        if model["status"] == "passed":
            assert model["train_count"] == 14
            assert model["valid_count"] == 6

    summary = summarize_report(out_dir / "sector_internal_ranking_baseline_smoke_report.json")
    assert summary["status"] == "OK"
    assert summary["models"] == model_names

    predictions = pd.read_csv(report["prediction_file"], dtype={"etf_code": str})
    assert set(predictions["model_name"]).issubset(set(model_names))
    assert "target_weight" not in predictions.columns
    assert "order_intent" not in predictions.columns


def test_missing_optional_model_dependency_skips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import tools.lab.table_ml_baseline_smoke as smoke

    real_import_module = smoke.importlib.import_module

    def fake_import_module(name: str, package: str | None = None) -> Any:
        if name in {"lightgbm", "catboost", "xgboost"}:
            raise ImportError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(smoke.importlib, "import_module", fake_import_module)
    report = run_baseline_smoke(
        sample_path=SAMPLE,
        manifest_path=MANIFEST,
        feature_contract_path=write_contract(tmp_path),
        target="top_quantile_in_sector_3d",
        out_dir=tmp_path / "out",
        models="numpy_logistic,lightgbm,catboost,xgboost",
    )

    statuses = {model["model_name"]: model["status"] for model in report["models"]}
    assert statuses["numpy_logistic_regression_smoke"] == "passed"
    assert statuses["lightgbm_smoke"] == "skipped"
    assert statuses["catboost_smoke"] == "skipped"
    assert statuses["xgboost_smoke"] == "skipped"


def test_catboost_smoke_does_not_leave_catboost_info(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    out_dir = tmp_path / "out"
    report = run_baseline_smoke(
        sample_path=SAMPLE,
        manifest_path=MANIFEST,
        feature_contract_path=write_contract(tmp_path),
        target="top_quantile_in_sector_3d",
        out_dir=out_dir,
        models="catboost",
    )

    assert report["models"][0]["model_name"] == "catboost_smoke"
    assert report["models"][0]["status"] in {"passed", "skipped"}
    assert not (tmp_path / "catboost_info").exists()
    assert not (out_dir / "catboost_info").exists()


def test_no_model_file_is_generated(tmp_path: Path):
    out_dir = tmp_path / "out"
    run_baseline_smoke(
        sample_path=SAMPLE,
        manifest_path=MANIFEST,
        feature_contract_path=write_contract(tmp_path),
        target="top_quantile_in_sector_3d",
        out_dir=out_dir,
    )

    generated = {path.name for path in out_dir.iterdir()}
    assert generated == {
        "sector_internal_ranking_baseline_predictions.csv",
        "sector_internal_ranking_baseline_smoke_report.json",
        "sector_internal_ranking_baseline_smoke_report.md",
    }
    assert not list(out_dir.glob("*.pkl"))
    assert not list(out_dir.glob("*.joblib"))
    assert not list(out_dir.glob("*.cbm"))
    assert not list(out_dir.glob("*.model"))
    assert not list(out_dir.glob("*.bst"))
    assert not list(out_dir.glob("*.txt"))
