import json

import pandas as pd

from historical_ml.cli import main as historical_ml_main
from historical_ml.config import HistoricalMLConfig
from historical_ml.entry_adapter import HeuristicEntryAdapter
from historical_ml.replay_engine import HistoricalReplayEngine
from historical_ml.schemas import DAILY_ML_UNIVERSE_SAMPLE_COLUMNS, ENTRY_CANDIDATE_COLUMNS
from historical_ml.validators import assert_required_columns, assert_signal_execution_separation, assert_source_is_historical_replay
from historical_ml.tests.test_helpers import make_price_data


def test_replay_outputs_required_columns_and_separates_execution_date():
    prices = make_price_data(days=45)
    config = HistoricalMLConfig(min_history_days=5, replay_start=pd.Timestamp("2024-10-01").date(), replay_end=pd.Timestamp("2024-10-15").date())
    outputs = HistoricalReplayEngine(prices, config=config).run(config.replay_start, config.replay_end)
    samples = outputs["entry_candidate_samples"]
    assert not samples.empty
    assert_required_columns(samples, ENTRY_CANDIDATE_COLUMNS, "entry_candidate_samples")
    assert_signal_execution_separation(samples)
    assert set(samples["source"]) == {"historical_replay"}


def test_daily_ml_universe_samples_include_non_selected_etfs_and_write_summary(tmp_path):
    prices = make_price_data(days=80)
    config = HistoricalMLConfig(
        min_history_days=5,
        selected_sector_count=1,
        candidate_top_n_per_sector=1,
        max_selected_entries=1,
        replay_start=pd.Timestamp("2024-10-15").date(),
        replay_end=pd.Timestamp("2024-10-18").date(),
    )

    outputs = HistoricalReplayEngine(
        prices,
        config=config,
        entry_adapter=HeuristicEntryAdapter(),
    ).run(config.replay_start, config.replay_end, out_dir=tmp_path)

    samples = outputs["daily_ml_universe_samples"]
    assert not samples.empty
    assert_required_columns(samples, DAILY_ML_UNIVERSE_SAMPLE_COLUMNS, "daily_ml_universe_samples")
    assert_source_is_historical_replay(samples, "daily_ml_universe_samples")

    selected_count = int(samples["pre_selected"].sum())
    assert len(samples) > selected_count
    assert samples.loc[~samples["pre_selected"]].shape[0] > 0
    assert samples.groupby("trade_date")["is_valid_sample"].sum().min() >= 1

    generated = tmp_path / "generated"
    assert (generated / "daily_ml_universe_samples.csv").exists()
    summary = json.loads((generated / "daily_ml_universe_summary.json").read_text(encoding="utf-8"))
    assert summary["source"] == "historical_replay"
    assert summary["total_rows"] == len(samples)
    assert summary["total_pre_selected"] == selected_count
    assert len(summary["daily_stats"]) == samples["trade_date"].nunique()
    assert all(day["valid_samples"] >= day["pre_selected_count"] for day in summary["daily_stats"])


def test_daily_ml_universe_samples_do_not_change_when_future_prices_are_perturbed():
    prices = make_price_data(days=100)
    trade_date = pd.Timestamp("2024-11-15").date()
    config = HistoricalMLConfig(min_history_days=5, replay_start=trade_date, replay_end=trade_date)

    base = HistoricalReplayEngine(
        prices,
        config=config,
        entry_adapter=HeuristicEntryAdapter(),
    ).run(trade_date, trade_date)["daily_ml_universe_samples"]

    mutated = prices.copy()
    future_mask = pd.to_datetime(mutated["date"]) > pd.Timestamp(trade_date)
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        mutated.loc[future_mask, col] = mutated.loc[future_mask, col] * 1000 + 123

    changed_future = HistoricalReplayEngine(
        mutated,
        config=config,
        entry_adapter=HeuristicEntryAdapter(),
    ).run(trade_date, trade_date)["daily_ml_universe_samples"]

    compare_cols = [
        "code",
        "momentum_20",
        "momentum_60",
        "momentum_120",
        "momentum_score",
        "acceleration_score",
        "volatility_20",
        "drawdown_20",
        "drawdown_60",
        "sector_rank",
        "etf_rank",
        "pre_selected",
        "entry_raw_action",
        "final_action",
    ]
    pd.testing.assert_frame_equal(
        base[compare_cols].sort_values("code").reset_index(drop=True),
        changed_future[compare_cols].sort_values("code").reset_index(drop=True),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
    )


def test_run_all_writes_performance_reports(tmp_path):
    prices_path = tmp_path / "prices.csv"
    make_price_data(days=70).to_csv(prices_path, index=False)
    out_dir = tmp_path / "artifacts"

    rc = historical_ml_main(
        [
            "run-all",
            "--prices",
            str(prices_path),
            "--start",
            "2024-10-15",
            "--end",
            "2024-10-25",
            "--out",
            str(out_dir),
            "--format",
            "csv",
        ]
    )

    assert rc == 0
    report_json = out_dir / "historical_ml_performance_report.json"
    report_md = out_dir / "historical_ml_performance_report.md"
    timing_csv = out_dir / "replay_timing_summary.csv"
    assert report_json.exists()
    assert report_md.exists()
    assert timing_csv.exists()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["command"] == "run-all"
    assert payload["timing"]["feature_build_seconds"] >= 0
    assert payload["timing"]["label_build_seconds"] >= 0
    assert payload["row_counts"]["daily_ml_universe_samples"] > 0
    timing = pd.read_csv(timing_csv)
    assert {"load_data_seconds", "feature_build_seconds", "label_build_seconds", "total_seconds"}.issubset(timing.columns)
