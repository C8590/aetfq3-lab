from __future__ import annotations

import csv
import json
from pathlib import Path

from tools.lab.intraday_group_label_inconsistency_diagnostic import (
    ACCEPTED,
    BAR_LEVEL_OUTCOME_DENOMINATOR_VARIATION,
    BLOCKED_DATA_QUALITY,
    DATA_QUALITY_SUSPECT,
    GROUP_POLICY_EXPECTED_DIAGNOSTIC_BEHAVIOR,
    TARGET_COLUMN,
    THRESHOLD_NEAR_ZERO_LABEL_FLIP,
    main,
    run_diagnostic,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BAR_SAMPLES = REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_group_label_inconsistency_samples.csv"
OUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_group_label_inconsistency_diagnostic/pytest")


def write_json(tmp_path: Path, name: str, payload: dict[str, object]) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_group_report(tmp_path: Path) -> Path:
    return write_json(
        tmp_path,
        "group_report.json",
        {
            "report_type": "intraday_group_level_sample_dryrun",
            "group_key": ["trade_date", "etf_code"],
            "group_label_policy": "anchor_close_last_bar",
            "intraday_live_decision_ready": False,
            "group_statistics": {
                "single_label_group_count": 3,
                "inconsistent_label_group_count": 1,
                "null_label_group_count": 0,
            },
        },
    )


def write_group_samples(tmp_path: Path, bad_close: bool = False, bad_label: bool = False) -> Path:
    path = tmp_path / "group_samples.csv"
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    with BAR_SAMPLES.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            groups.setdefault((row["trade_date"], row["etf_code"]), []).append(row)
    fieldnames = [
        "trade_date",
        "etf_code",
        "bar_count",
        "close_last",
        "future_return_3d",
        TARGET_COLUMN,
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key, rows in sorted(groups.items()):
            rows = sorted(rows, key=lambda row: row["datetime"])
            last = rows[-1]
            close_last = "9.999" if bad_close and key == ("2026-01-03", "510300") else last["close"]
            label = "0" if bad_label and key == ("2026-01-03", "510050") else last[TARGET_COLUMN]
            writer.writerow(
                {
                    "trade_date": key[0],
                    "etf_code": key[1],
                    "bar_count": len(rows),
                    "close_last": close_last,
                    "future_return_3d": last["future_return_3d"],
                    TARGET_COLUMN: label,
                }
            )
    return path


def test_consistent_group_is_identified(tmp_path: Path) -> None:
    report = run_diagnostic(BAR_SAMPLES, write_group_samples(tmp_path), write_group_report(tmp_path), OUT_ROOT / "consistent")
    group = next(item for item in report["group_diagnostics"] if item["trade_date"] == "2026-01-02" and item["etf_code"] == "510300")

    assert group["inconsistent_group"] is False
    assert group["label_unique_count"] == 1
    assert report["consistent_group_count"] == 3


def test_inconsistent_group_is_identified(tmp_path: Path) -> None:
    report = run_diagnostic(BAR_SAMPLES, write_group_samples(tmp_path), write_group_report(tmp_path), OUT_ROOT / "inconsistent")
    group = next(item for item in report["group_diagnostics"] if item["trade_date"] == "2026-01-02" and item["etf_code"] == "510050")

    assert group["inconsistent_group"] is True
    assert group["label_0_count"] == 1
    assert group["label_1_count"] == 2
    assert report["inconsistent_group_count"] == 1
    assert BAR_LEVEL_OUTCOME_DENOMINATOR_VARIATION in report["inconsistency_drivers"]
    assert GROUP_POLICY_EXPECTED_DIAGNOSTIC_BEHAVIOR in report["inconsistency_drivers"]


def test_first_bar_label_differs_from_last_bar_label_is_counted(tmp_path: Path) -> None:
    report = run_diagnostic(BAR_SAMPLES, write_group_samples(tmp_path), write_group_report(tmp_path), OUT_ROOT / "first_last")

    assert report["first_last_label_mismatch_group_count"] == 1
    assert report["group_diagnostics"][0]["first_bar_label_differs_from_last_bar_label"] is True


def test_last_bar_policy_matches_group_level_label(tmp_path: Path) -> None:
    report = run_diagnostic(BAR_SAMPLES, write_group_samples(tmp_path), write_group_report(tmp_path), OUT_ROOT / "last_bar")

    assert report["last_bar_policy_check"]["all_last_bar_labels_match_group_level_labels"] is True
    assert report["last_bar_policy_check"]["accepted_for_end_of_day_diagnostic"] is True
    assert report["policy_review_decision"] == ACCEPTED
    assert report["intraday_live_decision_ready"] is False


def test_near_zero_flip_is_marked(tmp_path: Path) -> None:
    report = run_diagnostic(BAR_SAMPLES, write_group_samples(tmp_path), write_group_report(tmp_path), OUT_ROOT / "near_zero")
    group = next(item for item in report["group_diagnostics"] if item["trade_date"] == "2026-01-02" and item["etf_code"] == "510050")

    assert group["near_zero_label_flip"] is True
    assert report["near_zero_flip_group_count"] == 1
    assert THRESHOLD_NEAR_ZERO_LABEL_FLIP in report["inconsistency_drivers"]


def test_data_quality_suspect_blocks(tmp_path: Path) -> None:
    report = run_diagnostic(
        BAR_SAMPLES,
        write_group_samples(tmp_path, bad_close=True),
        write_group_report(tmp_path),
        OUT_ROOT / "data_quality",
    )

    assert report["policy_review_decision"] == BLOCKED_DATA_QUALITY
    assert DATA_QUALITY_SUSPECT in report["inconsistency_drivers"]
    assert report["p0_blockers"] == ["data quality suspect groups found"]


def test_report_json_includes_boundary_fields(tmp_path: Path) -> None:
    out_dir = OUT_ROOT / "boundary_fields"

    report = run_diagnostic(BAR_SAMPLES, write_group_samples(tmp_path), write_group_report(tmp_path), out_dir)
    payload = json.loads(
        (REPO_ROOT / out_dir / "intraday_group_label_inconsistency_report.json").read_text(encoding="utf-8")
    )

    assert report["policy_review_decision"] == ACCEPTED
    for key in (
        "report_type",
        "group_key",
        "total_group_count",
        "inconsistent_group_count",
        "inconsistent_group_rate",
        "inconsistency_drivers",
        "last_bar_policy_check",
        "policy_review_decision",
        "intraday_live_decision_ready",
        "stable_promotion_ready",
        "formal_training_ready",
        "qmt_ready",
        "order_intent_ready",
        "metrics_are_effectiveness_evidence",
        "not_trading_advice",
    ):
        assert key in payload
    assert payload["stable_promotion_ready"] is False
    assert payload["not_trading_advice"] is True


def test_no_model_artifacts_created(tmp_path: Path) -> None:
    out_dir = OUT_ROOT / "no_artifacts"

    report = run_diagnostic(BAR_SAMPLES, write_group_samples(tmp_path), write_group_report(tmp_path), out_dir)
    forbidden = [
        path
        for path in (REPO_ROOT / out_dir).rglob("*")
        if path.is_file() and path.suffix.lower() in {".pkl", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"}
    ]

    assert report["model_saved"] is False
    assert report["checkpoint_saved"] is False
    assert forbidden == []


def test_cli_succeeds(tmp_path: Path, capsys) -> None:
    exit_code = main(
        [
            "--bar-samples",
            str(BAR_SAMPLES),
            "--group-samples",
            str(write_group_samples(tmp_path)),
            "--group-report",
            str(write_group_report(tmp_path)),
            "--out-dir",
            str(OUT_ROOT / "cli"),
        ]
    )
    stdout = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert stdout["policy_review_decision"] == ACCEPTED
    assert stdout["formal_training_ready"] is False
