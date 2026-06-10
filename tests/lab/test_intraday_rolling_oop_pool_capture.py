from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tools.lab.intraday_rolling_oop_pool_capture import (
    BOUNDARY_FIELDS,
    CaptureConfig,
    DAILY_COLUMNS,
    FIVE_M_COLUMNS,
    LAB_DECLARATION,
    RollingOopPoolCaptureError,
    append_only_merge,
    build_readiness,
    compute_inventory,
    resolve_output_dir,
    write_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def make_5m_pool(anchor_dates: list[str], etfs: list[str], bars_per_day: int = 48) -> pd.DataFrame:
    rows = []
    for anchor_date in anchor_dates:
        current = datetime.combine(date.fromisoformat(anchor_date), time(9, 35))
        for etf_index, etf_code in enumerate(etfs):
            for bar_index in range(bars_per_day):
                bar_dt = current + timedelta(minutes=5 * bar_index)
                close = 3.0 + etf_index * 0.1 + bar_index * 0.001
                volume = 1000 + bar_index
                amount = close * volume
                rows.append(
                    {
                        "trade_date": anchor_date,
                        "datetime": bar_dt.strftime("%Y-%m-%d %H:%M:%S"),
                        "etf_code": etf_code,
                        "open": close - 0.001,
                        "high": close + 0.002,
                        "low": close - 0.002,
                        "close": close,
                        "volume": volume,
                        "amount": amount,
                        "vwap": amount / volume,
                    }
                )
    return pd.DataFrame(rows, columns=FIVE_M_COLUMNS)


def make_daily_pool(anchor_dates: list[str], etfs: list[str], include_t3: bool = True) -> pd.DataFrame:
    date_map = {
        "2026-06-04": ["2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09"],
        "2026-06-05": ["2026-06-05", "2026-06-08", "2026-06-09", "2026-06-10"],
        "2026-06-03": ["2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08"],
    }
    rows = []
    for anchor_date in anchor_dates:
        dates = date_map[anchor_date] if include_t3 else date_map[anchor_date][:2]
        for etf_index, etf_code in enumerate(etfs):
            for trade_date in dates:
                close = 3.0 + etf_index * 0.1
                rows.append(
                    {
                        "trade_date": trade_date,
                        "etf_code": etf_code,
                        "open": close - 0.01,
                        "high": close + 0.02,
                        "low": close - 0.02,
                        "close": close,
                        "volume": 100000,
                        "amount": close * 100000,
                    }
                )
    return pd.DataFrame(rows, columns=DAILY_COLUMNS).drop_duplicates(["trade_date", "etf_code"]).reset_index(drop=True)


def config_for_test(min_oop_anchors: int = 1, min_etfs: int = 2, min_groups: int = 2) -> CaptureConfig:
    return CaptureConfig(
        etfs=["159915", "510300"],
        artifact_dir=Path(".local_artifact_backup/pytest_intraday_rolling_oop_pool_capture"),
        report_dir=Path(".local_research_outputs/pytest_intraday_rolling_oop_pool_capture"),
        min_oop_anchors=min_oop_anchors,
        min_etfs=min_etfs,
        min_groups=min_groups,
    )


def test_append_only_merge_does_not_duplicate_identical_rows() -> None:
    existing = pd.read_csv(REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_rolling_oop_5m_pool.csv", dtype={"etf_code": str})
    result = append_only_merge(existing, existing, ["etf_code", "datetime"], FIVE_M_COLUMNS)

    assert len(result.merged) == len(existing)
    assert result.stats["added_rows"] == 0
    assert result.stats["identical_duplicate_rows"] == len(existing)
    assert result.stats["conflict_count"] == 0


def test_append_only_merge_records_conflict_and_keeps_existing_row() -> None:
    existing = pd.read_csv(REPO_ROOT / "tests/fixtures/aetfq3_lab/mock_intraday_rolling_oop_5m_pool.csv", dtype={"etf_code": str})
    incoming = existing.copy()
    incoming.loc[0, "close"] = incoming.loc[0, "close"] + 0.1

    result = append_only_merge(existing, incoming, ["etf_code", "datetime"], FIVE_M_COLUMNS)

    assert len(result.merged) == len(existing)
    assert result.stats["conflict_count"] == 1
    key_match = result.merged["datetime"].eq(existing.loc[0, "datetime"]) & result.merged["etf_code"].eq(existing.loc[0, "etf_code"])
    assert result.merged.loc[key_match, "close"].iloc[0] == existing.loc[0, "close"]


def test_strict_oop_anchor_computation_counts_complete_future_anchor() -> None:
    etfs = ["159915", "510300"]
    five_m = make_5m_pool(["2026-06-04"], etfs)
    daily = make_daily_pool(["2026-06-04"], etfs)

    readiness = build_readiness(five_m, daily, config_for_test(), "2026-06-10T00:00:00+00:00")

    assert readiness.decision_payload["eligible_oop_anchor_dates"] == ["2026-06-04"]
    assert readiness.decision_payload["eligible_oop_anchor_count"] == 1
    assert readiness.decision_payload["group_count"] == 2
    assert readiness.decision_payload["readiness_decision"] == "ROLLING_OOP_POOL_READY_FOR_FIXED_SHORTLIST_VALIDATION"


def test_overlap_anchor_is_not_strict_oop() -> None:
    etfs = ["159915", "510300"]
    inventory = compute_inventory(
        make_5m_pool(["2026-06-03"], etfs),
        make_daily_pool(["2026-06-03"], etfs),
        sprint_anchor_start="2026-04-09",
        sprint_anchor_end="2026-06-03",
    )

    assert inventory["strict_oop_by_date"].eq(False).all()
    assert inventory["eligible_strict_oop_etf_anchor"].eq(False).all()


def test_t1_t3_missing_blocks_anchor() -> None:
    etfs = ["159915", "510300"]
    readiness = build_readiness(
        make_5m_pool(["2026-06-04"], etfs),
        make_daily_pool(["2026-06-04"], etfs, include_t3=False),
        config_for_test(),
        "2026-06-10T00:00:00+00:00",
    )

    assert readiness.decision_payload["eligible_oop_anchor_count"] == 0
    assert readiness.decision_payload["readiness_decision"] == "ROLLING_OOP_POOL_NO_ELIGIBLE_ANCHORS_YET"


def test_readiness_thresholds_limited_until_all_thresholds_pass() -> None:
    etfs = ["159915", "510300"]
    five_m = make_5m_pool(["2026-06-04", "2026-06-05"], etfs)
    daily = make_daily_pool(["2026-06-04", "2026-06-05"], etfs)
    limited = build_readiness(five_m, daily, config_for_test(min_oop_anchors=3, min_etfs=2, min_groups=6), "2026-06-10T00:00:00+00:00")
    ready = build_readiness(five_m, daily, config_for_test(min_oop_anchors=2, min_etfs=2, min_groups=4), "2026-06-10T00:00:00+00:00")

    assert limited.decision_payload["readiness_decision"] == "ROLLING_OOP_POOL_LIMITED_ACCUMULATING"
    assert "eligible_oop_anchors 2 < 3" in limited.decision_payload["threshold_failure_reasons"]
    assert ready.decision_payload["readiness_decision"] == "ROLLING_OOP_POOL_READY_FOR_FIXED_SHORTLIST_VALIDATION"


def test_non_local_output_path_is_rejected() -> None:
    with pytest.raises(RollingOopPoolCaptureError):
        resolve_output_dir(Path("reports/not_local"))


def test_report_json_contains_boundary_fields() -> None:
    etfs = ["159915", "510300"]
    config = config_for_test()
    five_m = make_5m_pool(["2026-06-04"], etfs)
    daily = make_daily_pool(["2026-06-04"], etfs)
    merge_5m = append_only_merge(pd.DataFrame(columns=FIVE_M_COLUMNS), five_m, ["etf_code", "datetime"], FIVE_M_COLUMNS)
    merge_daily = append_only_merge(pd.DataFrame(columns=DAILY_COLUMNS), daily, ["etf_code", "trade_date"], DAILY_COLUMNS)
    readiness = build_readiness(merge_5m.merged, merge_daily.merged, config, "2026-06-10T00:00:00+00:00")

    write_outputs(config, "2026-06-10T00:00:00+00:00", "test", [], [], merge_5m, merge_daily, readiness)
    report_path = REPO_ROOT / config.report_dir / "rolling_oop_capture_report.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert payload["lab_declaration"] == LAB_DECLARATION
    for field, expected in BOUNDARY_FIELDS.items():
        assert payload[field] == expected
