from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.table_ml_baseline_report_reader import (
    BaselineReportContractError,
    summarize_report,
)


MOCK_REPORT = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_baseline_smoke_report.json"


def load_report() -> dict:
    return json.loads(MOCK_REPORT.read_text(encoding="utf-8"))


def write_report(tmp_path: Path, report: dict) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_valid_mock_report_passes():
    summary = summarize_report(MOCK_REPORT)

    assert summary["status"] == "OK"
    assert summary["boundary_passed"] is True
    assert summary["models"] == ["numpy_logistic_regression_smoke"]
    assert "accuracy" in summary["metrics_keys"]


def test_missing_required_field_fails(tmp_path: Path):
    report = load_report()
    report.pop("lab_only")

    with pytest.raises(BaselineReportContractError, match="Missing required fields: lab_only"):
        summarize_report(write_report(tmp_path, report))


def test_lab_only_false_fails(tmp_path: Path):
    report = load_report()
    report["lab_only"] = False

    with pytest.raises(BaselineReportContractError, match="lab_only must be true"):
        summarize_report(write_report(tmp_path, report))


def test_model_saved_true_fails(tmp_path: Path):
    report = load_report()
    report["model_saved"] = True

    with pytest.raises(BaselineReportContractError, match="model_saved must be false"):
        summarize_report(write_report(tmp_path, report))


def test_no_order_intent_false_fails(tmp_path: Path):
    report = load_report()
    report["no_order_intent"] = False

    with pytest.raises(BaselineReportContractError, match="no_order_intent must be true"):
        summarize_report(write_report(tmp_path, report))


def test_order_intent_field_fails(tmp_path: Path):
    report = load_report()
    report["order_intent"] = {"symbol": "510300"}

    with pytest.raises(BaselineReportContractError, match="Prohibited trading fields present: order_intent"):
        summarize_report(write_report(tmp_path, report))


def test_split_method_non_chronological_fails(tmp_path: Path):
    report = load_report()
    report["split_method"] = "random"

    with pytest.raises(BaselineReportContractError, match="split_method must be chronological"):
        summarize_report(write_report(tmp_path, report))


def test_group_leakage_check_not_passed_fails(tmp_path: Path):
    report = load_report()
    report["group_leakage_check"] = "failed"

    with pytest.raises(BaselineReportContractError, match="group_leakage_check must be passed"):
        summarize_report(write_report(tmp_path, report))


def test_cli_outputs_summary_json():
    completed = subprocess.run(
        [
            sys.executable,
            "tools/lab/table_ml_baseline_report_reader.py",
            "--report",
            "tests/fixtures/aetfq3_lab/mock_baseline_smoke_report.json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["status"] == "OK"
    assert summary["boundary_passed"] is True
    assert summary["models"] == ["numpy_logistic_regression_smoke"]
    assert "metrics_keys" in summary
    assert "accuracy" in summary["metrics_keys"]
