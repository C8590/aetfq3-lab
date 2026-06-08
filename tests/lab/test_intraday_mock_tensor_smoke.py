from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_mock_tensor_smoke import (
    DEFAULT_FEATURE_COLUMNS,
    IntradayMockSmokeError,
    MLPBaseline,
    GRUSmoke,
    TARGET_COLUMNS,
    TemporalCNNSmoke,
    build_sequence_tensor,
    run_intraday_mock_tensor_smoke,
    run_model_smoke,
    scan_forbidden_features,
)


SAMPLE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_5m_samples.csv"


def assert_no_checkpoint_files(path: Path) -> None:
    forbidden_suffixes = {".pt", ".pth", ".ckpt"}
    assert not [item for item in path.rglob("*") if item.suffix.lower() in forbidden_suffixes]
    assert not [item for item in path.rglob("*") if "checkpoint" in item.name.lower()]


def test_mock_csv_shape_can_build_tensor():
    bundle = build_sequence_tensor(SAMPLE)

    assert tuple(bundle.x.shape) == (4, 12, len(DEFAULT_FEATURE_COLUMNS))
    assert tuple(bundle.y.shape) == (4, len(TARGET_COLUMNS))
    assert len(bundle.sequence_keys) == 4


def test_valid_mock_feature_scan_passes():
    scan = scan_forbidden_features(DEFAULT_FEATURE_COLUMNS)

    assert scan["passed"] is True
    assert scan["forbidden_columns"] == []


def test_forbidden_future_feature_scan_fails():
    scan = scan_forbidden_features([*DEFAULT_FEATURE_COLUMNS, "future_return_3d"])

    assert scan["passed"] is False
    assert scan["forbidden_columns"] == ["future_return_3d"]

    with pytest.raises(IntradayMockSmokeError, match="forbidden feature columns"):
        build_sequence_tensor(SAMPLE, feature_columns=[*DEFAULT_FEATURE_COLUMNS, "future_return_3d"])


def test_mlp_smoke_runs():
    bundle = build_sequence_tensor(SAMPLE)
    _, time_steps, feature_count = bundle.x.shape
    target_count = bundle.y.shape[1]

    result = run_model_smoke(
        "mlp_smoke",
        MLPBaseline(time_steps, feature_count, target_count),
        bundle.x,
        bundle.y,
        device=__import__("torch").device("cpu"),
    )

    assert result["status"] == "passed"
    assert result["steps"] == 2
    assert result["model_saved"] is False
    assert result["checkpoint_saved"] is False


def test_gru_smoke_runs():
    bundle = build_sequence_tensor(SAMPLE)
    feature_count = bundle.x.shape[2]
    target_count = bundle.y.shape[1]

    result = run_model_smoke(
        "gru_smoke",
        GRUSmoke(feature_count, target_count),
        bundle.x,
        bundle.y,
        device=__import__("torch").device("cpu"),
    )

    assert result["status"] == "passed"
    assert result["final_loss"] is not None


def test_temporal_cnn_smoke_runs():
    bundle = build_sequence_tensor(SAMPLE)
    feature_count = bundle.x.shape[2]
    target_count = bundle.y.shape[1]

    result = run_model_smoke(
        "temporal_cnn_smoke",
        TemporalCNNSmoke(feature_count, target_count),
        bundle.x,
        bundle.y,
        device=__import__("torch").device("cpu"),
    )

    assert result["status"] == "passed"
    assert result["final_loss"] is not None


def test_cli_runner_writes_reports_and_no_checkpoint(tmp_path: Path):
    out_dir = tmp_path / "smoke"
    report = run_intraday_mock_tensor_smoke(SAMPLE, out_dir, device_name="cpu")

    assert report["status"] == "passed"
    assert report["batch_size"] == 4
    assert report["time_steps"] == 12
    assert report["feature_count"] == len(DEFAULT_FEATURE_COLUMNS)
    assert report["target_count"] == len(TARGET_COLUMNS)
    assert report["forbidden_feature_passed"] is True
    assert report["model_saved"] is False
    assert report["checkpoint_saved"] is False
    assert (out_dir / "intraday_mock_tensor_smoke_report.json").exists()
    assert (out_dir / "intraday_mock_tensor_smoke_report.md").exists()
    assert_no_checkpoint_files(out_dir)

