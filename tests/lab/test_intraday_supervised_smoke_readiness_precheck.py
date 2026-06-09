from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.lab.intraday_supervised_smoke_readiness_precheck import (
    BLOCKED_BOUNDARY_FLAG,
    BLOCKED_INSUFFICIENT_ROWS,
    BLOCKED_MANIFEST_LEAKAGE_P0,
    BLOCKED_SINGLE_CLASS_LABEL,
    BLOCKED_SPLIT_NOT_CLASS_DIVERSE,
    READY,
    main,
    run_precheck,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
READY_FIXTURE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_supervised_smoke_ready_samples.csv"
SINGLE_CLASS_FIXTURE = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_supervised_smoke_single_class_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_supervised_smoke_readiness_precheck/pytest")
ANCHOR_DATES = ["2026-05-25", "2026-05-26", "2026-05-27", "2026-05-28", "2026-05-29", "2026-06-01"]
ETF_CODES = ["159915", "510050"]


def base_manifest(overrides: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "sample_type": "intraday_5m",
        "feature_columns": [
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
            "vwap",
            "intraday_return",
            "return_from_open",
            "distance_to_vwap",
        ],
        "label_generated": True,
        "label_source_kind": "public_future_window",
        "label_horizon": {"unit": "trading_day", "required_horizons": ["T+1", "T+3"]},
        "label_generation_method": "unit_test_three_day_positive_label_v1",
        "label_columns": [
            "buy_now_label",
            "wait_pullback_label",
            "cancel_buy_label",
            "three_day_positive_label",
        ],
        "outcome_columns": [
            "future_return_1d",
            "future_return_3d",
            "max_drawdown_3d",
            "execution_return_to_close",
            "execution_return_to_next_open",
            "execution_drawdown_after_entry",
            "expected_3d_return",
            "expected_3d_drawdown",
        ],
        "label_status_column": "label_status",
        "insufficient_future_window_policy": "set label null when future window is unavailable",
        "feature_label_overlap_check": True,
        "label_generation_authorized": True,
        "supervised_training_allowed": False,
        "training_allowed": False,
        "stable_effect_allowed": False,
        "contains_order_intent": False,
        "contains_live_order": False,
        "contains_secret": False,
    }
    if overrides:
        payload.update(overrides)
    return payload


def write_manifest(tmp_path: Path, overrides: dict[str, object] | None = None) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(base_manifest(overrides), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_samples(tmp_path: Path, name: str, label_mode: str = "diverse", rows_per_etf_anchor: int = 50) -> Path:
    path = tmp_path / name
    columns = [
        "trade_date",
        "datetime",
        "etf_code",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "vwap",
        "intraday_return",
        "return_from_open",
        "distance_to_vwap",
        "future_return_1d",
        "future_return_3d",
        "max_drawdown_3d",
        "three_day_positive_label",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for anchor_index, anchor_date in enumerate(ANCHOR_DATES):
            for etf_index, etf_code in enumerate(ETF_CODES):
                for row_index in range(rows_per_etf_anchor):
                    if label_mode == "single_class":
                        label = 1
                    elif label_mode == "valid_single_class":
                        label = 1 if anchor_index >= 3 else (anchor_index + etf_index + row_index) % 2
                    else:
                        label = (anchor_index + etf_index + row_index) % 2
                    writer.writerow(
                        {
                            "trade_date": anchor_date,
                            "datetime": f"{anchor_date} 09:{35 + (row_index % 20):02d}:00",
                            "etf_code": etf_code,
                            "open": "1",
                            "high": "1",
                            "low": "1",
                            "close": "1",
                            "volume": "100",
                            "amount": "100",
                            "vwap": "1",
                            "intraday_return": "0",
                            "return_from_open": "0",
                            "distance_to_vwap": "0",
                            "future_return_1d": "0.01",
                            "future_return_3d": "0.02" if label else "-0.02",
                            "max_drawdown_3d": "0",
                            "three_day_positive_label": str(label),
                        }
                    )
    return path


def test_class_diverse_sample_returns_ready(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    samples = write_samples(tmp_path, "ready.csv")

    report = run_precheck(samples, manifest, OUT_ROOT / "ready")

    assert report["readiness_decision"] == READY
    assert report["split_feasible"] is True
    assert report["training_allowed"] is False
    assert report["stable_allowed"] is False
    assert report["order_intent_allowed"] is False
    assert report["qmt_allowed"] is False


def test_single_class_sample_blocks(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    samples = write_samples(tmp_path, "single.csv", label_mode="single_class")

    report = run_precheck(samples, manifest, OUT_ROOT / "single")

    assert report["readiness_decision"] == BLOCKED_SINGLE_CLASS_LABEL
    assert any("both class 0 and class 1" in item for item in report["p0_blockers"])


def test_label_in_feature_columns_blocks_manifest_p0(tmp_path: Path) -> None:
    feature_columns = list(base_manifest()["feature_columns"]) + ["three_day_positive_label"]
    manifest = write_manifest(tmp_path, {"feature_columns": feature_columns})
    samples = write_samples(tmp_path, "label_feature.csv")

    report = run_precheck(samples, manifest, OUT_ROOT / "label_feature")

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("three_day_positive_label" in item for item in report["p0_blockers"])


def test_outcome_in_feature_columns_blocks_manifest_p0(tmp_path: Path) -> None:
    feature_columns = list(base_manifest()["feature_columns"]) + ["future_return_3d"]
    manifest = write_manifest(tmp_path, {"feature_columns": feature_columns})
    samples = write_samples(tmp_path, "outcome_feature.csv")

    report = run_precheck(samples, manifest, OUT_ROOT / "outcome_feature")

    assert report["readiness_decision"] == BLOCKED_MANIFEST_LEAKAGE_P0
    assert any("future_return_3d" in item for item in report["p0_blockers"])


def test_insufficient_rows_blocks(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)

    report = run_precheck(READY_FIXTURE, manifest, OUT_ROOT / "insufficient_rows")

    assert report["readiness_decision"] == BLOCKED_INSUFFICIENT_ROWS
    assert any("row_count" in item for item in report["p0_blockers"])


def test_split_valid_single_class_blocks(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path)
    samples = write_samples(tmp_path, "split_bad.csv", label_mode="valid_single_class")

    report = run_precheck(samples, manifest, OUT_ROOT / "split_bad")

    assert report["readiness_decision"] == BLOCKED_SPLIT_NOT_CLASS_DIVERSE
    assert report["split_feasible"] is False


def test_training_allowed_true_blocks_boundary(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, {"training_allowed": True})
    samples = write_samples(tmp_path, "training_allowed.csv")

    report = run_precheck(samples, manifest, OUT_ROOT / "training_allowed")

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("training_allowed must be false" in item for item in report["p0_blockers"])


def test_contains_order_intent_true_blocks_boundary(tmp_path: Path) -> None:
    manifest = write_manifest(tmp_path, {"contains_order_intent": True})
    samples = write_samples(tmp_path, "order_intent.csv")

    report = run_precheck(samples, manifest, OUT_ROOT / "order_intent")

    assert report["readiness_decision"] == BLOCKED_BOUNDARY_FLAG
    assert any("contains_order_intent must be false" in item for item in report["p0_blockers"])


def test_stdout_and_report_json_contains_required_keys(tmp_path: Path, capsys) -> None:
    manifest = write_manifest(tmp_path)
    samples = write_samples(tmp_path, "stdout.csv")
    out_dir = OUT_ROOT / "stdout"

    exit_code = main(["--samples", str(samples), "--manifest", str(manifest), "--out-dir", str(out_dir)])
    stdout = json.loads(capsys.readouterr().out)
    report = json.loads((REPO_ROOT / out_dir / "intraday_supervised_smoke_readiness_report.json").read_text(encoding="utf-8"))

    assert exit_code == 0
    for key in (
        "readiness_decision",
        "selected_split_policy",
        "split_feasible",
        "training_allowed",
        "stable_allowed",
        "order_intent_allowed",
        "qmt_allowed",
    ):
        assert key in stdout
    for key in (
        "manifest_leakage_check",
        "sample_check",
        "split_check",
        "selected_split_policy",
        "train_anchor_dates",
        "valid_anchor_dates",
        "train_row_count",
        "valid_row_count",
        "train_label_0_count",
        "train_label_1_count",
        "valid_label_0_count",
        "valid_label_1_count",
        "split_feasible",
    ):
        assert key in report
