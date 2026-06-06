from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tools.lab.table_ml_dry_validation_report_reader import summarize_report, summarize_summary


def valid_report(sample_path: str = "tests/fixtures/aetfq3_lab/mock_false_downgrade_samples.csv") -> dict:
    return {
        "advisory_only": True,
        "affects_stable_trading": False,
        "intake_passed": True,
        "p0_blockers": [],
        "p1_warnings": [],
        "rows_checked": 8,
        "sample_path": sample_path,
        "sample_type": "false_downgrade",
        "schema_passed": True,
        "status": "passed",
        "training_allowed": False,
        "warnings": [],
    }


def write_json(tmp_path: Path, value: dict, name: str = "report.json") -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_valid_single_report_outputs_passed(tmp_path):
    summary = summarize_report(write_json(tmp_path, valid_report()))

    assert summary["review_status"] == "OK"
    assert summary["status"] == "passed"
    assert summary["sample_type"] == "false_downgrade"
    assert summary["rows_checked"] == 8


def test_summary_outputs_all_passed(tmp_path):
    smoke_summary = {
        "reports": [
            valid_report(),
            {**valid_report(), "sample_type": "sector_internal_ranking"},
        ]
    }

    summary = summarize_summary(write_json(tmp_path, smoke_summary, "summary.json"))

    assert summary["review_status"] == "OK"
    assert summary["included_report_count"] == 2
    assert summary["all_passed"] is True
    assert summary["report_statuses"] == {
        "false_downgrade": "passed",
        "sector_internal_ranking": "passed",
    }


def test_report_with_p0_blockers_marks_p0_review_required(tmp_path):
    report = valid_report()
    report["p0_blockers"] = ["blocked"]

    summary = summarize_report(write_json(tmp_path, report))

    assert summary["review_status"] == "P0_REVIEW_REQUIRED"


def test_training_allowed_true_marks_p0_review_required(tmp_path):
    report = valid_report()
    report["training_allowed"] = True

    summary = summarize_report(write_json(tmp_path, report))

    assert summary["review_status"] == "P0_REVIEW_REQUIRED"


def test_affects_stable_trading_true_marks_p0_review_required(tmp_path):
    report = valid_report()
    report["affects_stable_trading"] = True

    summary = summarize_report(write_json(tmp_path, report))

    assert summary["review_status"] == "P0_REVIEW_REQUIRED"


def test_advisory_only_false_marks_p0_review_required(tmp_path):
    report = valid_report()
    report["advisory_only"] = False

    summary = summarize_report(write_json(tmp_path, report))

    assert summary["review_status"] == "P0_REVIEW_REQUIRED"


def test_reader_does_not_open_sample_path_csv(tmp_path, monkeypatch):
    sample_path = tmp_path / "must_not_open.csv"
    report_path = write_json(tmp_path, valid_report(str(sample_path)))
    opened_paths: list[Path] = []
    original_open = Path.open

    def guarded_open(self, *args, **kwargs):
        opened_paths.append(self)
        if self == sample_path:
            raise AssertionError("reader must not open sample_path")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)

    summary = summarize_report(report_path)

    assert summary["status"] == "passed"
    assert sample_path not in opened_paths


def test_format_json_outputs_valid_json(tmp_path):
    report_path = write_json(tmp_path, valid_report())
    completed = subprocess.run(
        [
            sys.executable,
            "tools/lab/table_ml_dry_validation_report_reader.py",
            "--report",
            str(report_path),
            "--format",
            "json",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    assert summary["review_status"] == "OK"
    assert summary["status"] == "passed"
