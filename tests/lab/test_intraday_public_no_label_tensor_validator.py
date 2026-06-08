from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_public_no_label_tensor_validator import (
    DEFAULT_FEATURE_COLUMNS,
    PublicNoLabelTensorValidationError,
    build_no_label_tensor_shape,
    run_public_no_label_tensor_validation,
    scan_forbidden_features,
)


SAMPLE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_5m_no_label_samples.csv"


def assert_no_checkpoint_files(path: Path) -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt"}
    assert not [item for item in path.rglob("*") if item.suffix.lower() in forbidden_suffixes]
    assert not [item for item in path.rglob("*") if "checkpoint" in item.name.lower()]


def test_valid_no_label_ohlcv_mock_passes(tmp_path: Path) -> None:
    report = run_public_no_label_tensor_validation(SAMPLE, tmp_path / "report")

    assert report["status"] == "passed"
    assert report["tensor_shape_passed"] is True
    assert report["labels_required"] is False
    assert report["target_count"] == 0
    assert report["feature_columns"] == DEFAULT_FEATURE_COLUMNS


def test_missing_required_column_fails(tmp_path: Path) -> None:
    bad_csv = tmp_path / "missing_amount.csv"
    bad_csv.write_text(
        "trade_date,datetime,etf_code,open,high,low,close,volume\n"
        "2026-06-01,2026-06-01 09:30:00,MOCK001,10,11,9,10.5,1000\n",
        encoding="utf-8",
    )

    with pytest.raises(PublicNoLabelTensorValidationError, match="missing required columns"):
        build_no_label_tensor_shape(bad_csv)


def test_future_return_feature_fails(tmp_path: Path) -> None:
    bad_csv = tmp_path / "future_feature.csv"
    bad_csv.write_text(SAMPLE.read_text(encoding="utf-8").replace("\n", ",future_return_3d\n", 1), encoding="utf-8")

    scan = scan_forbidden_features([*DEFAULT_FEATURE_COLUMNS, "future_return_3d"])

    assert scan["passed"] is False
    assert scan["forbidden_columns"] == ["future_return_3d"]
    with pytest.raises(PublicNoLabelTensorValidationError, match="forbidden feature columns"):
        build_no_label_tensor_shape(bad_csv, feature_columns=[*DEFAULT_FEATURE_COLUMNS, "future_return_3d"])


def test_no_labels_still_passed() -> None:
    header = SAMPLE.read_text(encoding="utf-8").splitlines()[0].split(",")

    assert not [column for column in header if column.endswith("_label")]
    bundle = build_no_label_tensor_shape(SAMPLE)
    assert bundle.batch_size == 2


def test_tensor_shape_is_batch_time_features() -> None:
    bundle = build_no_label_tensor_shape(SAMPLE)

    assert bundle.batch_size == 2
    assert bundle.min_time_steps == 12
    assert bundle.max_time_steps == 12
    assert bundle.feature_count == len(DEFAULT_FEATURE_COLUMNS)
    assert len(bundle.sequence_keys) == 2


def test_nan_or_inf_is_reported_and_blocked(tmp_path: Path) -> None:
    nan_csv = tmp_path / "nan.csv"
    nan_csv.write_text(SAMPLE.read_text(encoding="utf-8").replace(",1000,", ",0,", 1), encoding="utf-8")

    report = run_public_no_label_tensor_validation(nan_csv, tmp_path / "nan_report")

    assert report["status"] == "failed"
    assert report["nan_count"] > 0
    assert report["tensor_shape_passed"] is False
    assert "NaN" in " ".join(report["p0_blockers"])


def test_report_does_not_generate_checkpoint_or_model(tmp_path: Path) -> None:
    out_dir = tmp_path / "safe_report"
    report = run_public_no_label_tensor_validation(SAMPLE, out_dir)

    assert report["model_saved"] is False
    assert report["checkpoint_saved"] is False
    assert_no_checkpoint_files(out_dir)


def test_report_does_not_generate_order_intent(tmp_path: Path) -> None:
    report = run_public_no_label_tensor_validation(SAMPLE, tmp_path / "order_report")

    assert report["no_order_intent"] is True
    assert report["order_intent_generated"] is False
    assert "OrderIntent" not in json.dumps(report, ensure_ascii=False)


def test_report_declares_no_output_write(tmp_path: Path) -> None:
    report = run_public_no_label_tensor_validation(SAMPLE, tmp_path / "output_report")

    assert report["no_output"] is True
    assert not (tmp_path / "output").exists()
