from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tools.lab.intraday_long_history_data_lake import (
    DataLakeConfig,
    LABEL_COLUMNS,
    OUTCOME_COLUMNS,
    TIME_CENSORED_FEATURES,
    build_feature_rows,
    map_columns,
    resample_bars,
    resolve_artifact_dir,
    run_data_lake,
    standardize_frame,
)


def one_minute_bars() -> pd.DataFrame:
    rows = []
    for minute in range(5):
        close = 1.0 + minute * 0.01
        rows.append(
            {
                "trade_date": "2025-01-02",
                "datetime": f"2025-01-02 09:{30 + minute:02d}:00",
                "etf_code": "510300",
                "open": 1.0 + minute * 0.01,
                "high": close + 0.02,
                "low": close - 0.01,
                "close": close,
                "volume": 100 + minute,
                "amount": close * (100 + minute),
                "vwap": close,
                "frequency": "1m",
                "source_file": "SH#510300.csv",
            }
        )
    return pd.DataFrame(rows)


def test_chinese_and_english_schema_mapping() -> None:
    chinese = pd.DataFrame(
        [
            {
                "日期": "2025-01-02",
                "时间": "2025-01-02 09:35:00",
                "证券代码": "SH#510300",
                "开盘": 1.0,
                "最高": 1.1,
                "最低": 0.9,
                "收盘": 1.05,
                "成交量": 1000,
                "成交额": 1050,
            }
        ]
    )

    mapping = map_columns(chinese.columns)
    standard, report = standardize_frame(chinese, "SH#510300.csv")

    assert mapping["trade_date"] == "日期"
    assert mapping["etf_code"] == "证券代码"
    assert report["missing_fields"] == []
    assert standard.iloc[0]["etf_code"] == "510300"
    assert set(["trade_date", "datetime", "open", "close"]).issubset(standard.columns)


def test_one_minute_to_five_minute_resample_correctness() -> None:
    frame = resample_bars(one_minute_bars(), "5m")

    assert len(frame) == 1
    row = frame.iloc[0]
    assert row["open"] == pytest.approx(1.0)
    assert row["high"] == pytest.approx(1.06)
    assert row["low"] == pytest.approx(0.99)
    assert row["close"] == pytest.approx(1.04)
    assert row["volume"] == pytest.approx(510)
    assert row["vwap"] == pytest.approx(row["amount"] / row["volume"])


def test_time_censored_feature_only_uses_bars_at_or_before_clock() -> None:
    rows = []
    for day_offset, day in enumerate(["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]):
        for time_text, close in [("09:30:00", 1.0 + day_offset), ("10:00:00", 1.1 + day_offset), ("10:05:00", 9.9 + day_offset)]:
            rows.append(
                {
                    "trade_date": day,
                    "datetime": f"{day} {time_text}",
                    "etf_code": "510300",
                    "open": 1.0 + day_offset,
                    "high": close,
                    "low": 1.0 + day_offset,
                    "close": close,
                    "volume": 100,
                    "amount": close * 100,
                    "vwap": close,
                    "frequency": "5m",
                    "source_file": "sample.csv",
                }
            )
    frame = pd.DataFrame(rows)
    frame["datetime"] = pd.to_datetime(frame["datetime"])

    feature_rows = build_feature_rows(frame, ["10:00"])
    first = next(row for row in feature_rows if row["trade_date"] == "2025-01-02")

    assert first["close_now_vs_open"] == pytest.approx(0.10)
    assert first["high_so_far_vs_open"] == pytest.approx(0.10)


def test_future_labels_and_outcomes_are_excluded_from_feature_columns() -> None:
    forbidden = set(LABEL_COLUMNS) | set(OUTCOME_COLUMNS)

    assert not (set(TIME_CENSORED_FEATURES) & forbidden)


def test_output_path_outside_local_artifact_backup_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-artifact-dir must be under"):
        resolve_artifact_dir(tmp_path)


def test_run_data_lake_writes_boundary_false_reports(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    artifact = tmp_path / "lake"
    report = tmp_path / "report"
    raw.mkdir()
    one_minute_bars().to_csv(raw / "SH#510300.csv", index=False)

    result = run_data_lake(
        DataLakeConfig(raw, raw, artifact, report),
        repo_root=tmp_path,
        enforce_paths=False,
    )

    assert result["access_mode"] == "READ_ONLY"
    assert result["final_action_change_allowed"] is False
    assert result["contains_live_order"] is False
    assert result["contains_secret"] is False
    assert result["stable_promotion_ready"] is False
    assert (artifact / "long_history_feature_rows.csv").exists()
    assert (report / "long_history_data_quality_report.json").exists()
