from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from time import perf_counter

import pandas as pd

from .config import HistoricalMLConfig
from .daily_universe import build_daily_ml_universe_samples, write_daily_ml_universe_outputs
from .entry_adapter import EntryAdapter, RealEntryAdapter
from .feature_builder import ReplayFeatureCache
from .io_utils import reorder_columns, write_daily_partition, write_table
from .schemas import (
    DAILY_DECISION_SNAPSHOT_COLUMNS,
    DAILY_ETF_SAMPLE_COLUMNS,
    DAILY_ML_UNIVERSE_SAMPLE_COLUMNS,
    DAILY_SECTOR_SAMPLE_COLUMNS,
    ENTRY_CANDIDATE_COLUMNS,
)


@dataclass
class HistoricalReplayEngine:
    """Replay ETF history day by day and produce feature/candidate samples."""

    price_df: pd.DataFrame
    config: HistoricalMLConfig = HistoricalMLConfig()
    entry_adapter: Optional[EntryAdapter] = None
    last_timing: dict[str, float] = field(default_factory=dict, init=False)

    def __post_init__(self):
        start = perf_counter()
        self.price_df = self.price_df.copy()
        self.price_df["date"] = pd.to_datetime(self.price_df["date"]).dt.normalize()
        self.price_df["code"] = self.price_df["code"].astype(str)
        self.price_df = self.price_df.sort_values(["date", "code"]).reset_index(drop=True)
        self.trading_dates = sorted(pd.to_datetime(self.price_df["date"].unique()))
        self.last_timing["normalize_code_seconds"] = perf_counter() - start
        if self.entry_adapter is None:
            self.entry_adapter = RealEntryAdapter()

    def _trading_dates(self, start, end) -> list[pd.Timestamp]:
        start = pd.Timestamp(start).normalize()
        end = pd.Timestamp(end).normalize()
        return [d for d in self.trading_dates if start <= d <= end]

    def _next_trading_date(self, trade_date) -> pd.Timestamp | pd.NaT:
        trade_date = pd.Timestamp(trade_date).normalize()
        for d in self.trading_dates:
            if d > trade_date:
                return d
        return pd.NaT

    def run(self, start=None, end=None, out_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
        start = start or self.config.replay_start
        end = end or self.config.replay_end
        dates = self._trading_dates(start, end)
        if not dates:
            raise ValueError(f"no trading dates found between {start} and {end}")

        total_start = perf_counter()
        feature_start = perf_counter()
        feature_prices = self._feature_price_window(dates)
        feature_cache = ReplayFeatureCache.build(feature_prices, self.config, dates=dates)
        self.last_timing["feature_build_seconds"] = perf_counter() - feature_start

        all_etf = []
        all_sector = []
        all_snapshots = []
        all_candidates = []
        all_universe = []
        decision_seconds = 0.0
        write_seconds = 0.0

        for trade_date in dates:
            etf_samples, sector_samples = feature_cache.for_day(trade_date)
            if etf_samples.empty:
                continue

            execution_date = self._next_trading_date(trade_date)
            decision_start = perf_counter()
            candidates = self.entry_adapter.build_entry_candidates(
                etf_samples=etf_samples,
                sector_samples=sector_samples,
                signal_date=trade_date,
                execution_date=execution_date,
                config=self.config,
            )
            decision_seconds += perf_counter() - decision_start

            snapshot = self._build_snapshot(trade_date, execution_date, etf_samples, sector_samples, candidates)
            universe = build_daily_ml_universe_samples(etf_samples, candidates, self.config)
            etf_out = reorder_columns(etf_samples, DAILY_ETF_SAMPLE_COLUMNS)
            sector_out = reorder_columns(sector_samples, DAILY_SECTOR_SAMPLE_COLUMNS)
            cand_out = reorder_columns(candidates, ENTRY_CANDIDATE_COLUMNS)
            universe_out = reorder_columns(universe, DAILY_ML_UNIVERSE_SAMPLE_COLUMNS)

            all_etf.append(etf_out)
            all_sector.append(sector_out)
            all_snapshots.append(snapshot)
            all_candidates.append(cand_out)
            all_universe.append(universe_out)

            if out_dir and self.config.write_daily_partitions:
                write_start = perf_counter()
                write_daily_partition(etf_out, out_dir, "daily_etf_samples", trade_date, self.config.output_format)
                write_daily_partition(sector_out, out_dir, "daily_sector_samples", trade_date, self.config.output_format)
                write_daily_partition(snapshot, out_dir, "daily_decision_snapshot", trade_date, self.config.output_format)
                write_daily_partition(cand_out, out_dir, "entry_candidate_samples", trade_date, self.config.output_format)
                write_daily_partition(universe_out, out_dir, "daily_ml_universe_samples", trade_date, self.config.output_format)
                write_seconds += perf_counter() - write_start

        outputs = {
            "daily_etf_samples": pd.concat(all_etf, ignore_index=True) if all_etf else pd.DataFrame(),
            "daily_sector_samples": pd.concat(all_sector, ignore_index=True) if all_sector else pd.DataFrame(),
            "daily_decision_snapshot": pd.concat(all_snapshots, ignore_index=True) if all_snapshots else pd.DataFrame(),
            "entry_candidate_samples": pd.concat(all_candidates, ignore_index=True) if all_candidates else pd.DataFrame(),
            "daily_ml_universe_samples": pd.concat(all_universe, ignore_index=True) if all_universe else pd.DataFrame(),
        }

        if out_dir:
            write_start = perf_counter()
            write_table(outputs["daily_etf_samples"], out_dir, "daily_etf_samples", self.config.output_format)
            write_table(outputs["daily_sector_samples"], out_dir, "daily_sector_samples", self.config.output_format)
            write_table(outputs["daily_decision_snapshot"], out_dir, "daily_decision_snapshot", self.config.output_format)
            write_table(outputs["entry_candidate_samples"], out_dir, "entry_candidate_samples_unlabeled", self.config.output_format)
            write_daily_ml_universe_outputs(outputs["daily_ml_universe_samples"], out_dir, self.config)
            write_seconds += perf_counter() - write_start

        self.last_timing["replay_decision_seconds"] = decision_seconds
        self.last_timing["write_output_seconds"] = write_seconds
        self.last_timing["total_replay_seconds"] = perf_counter() - total_start

        return outputs

    def _feature_price_window(self, dates: list[pd.Timestamp]) -> pd.DataFrame:
        if not dates:
            return self.price_df
        warmup_days = max(
            max(self.config.momentum_windows),
            60,
            20,
            self.config.min_history_days,
        ) + self.config.acceleration_lag + 1
        first_idx = self.trading_dates.index(pd.Timestamp(dates[0]).normalize())
        last_idx = self.trading_dates.index(pd.Timestamp(dates[-1]).normalize())
        start_idx = max(0, first_idx - warmup_days)
        start_date = self.trading_dates[start_idx]
        end_date = self.trading_dates[last_idx]
        return self.price_df.loc[(self.price_df["date"] >= start_date) & (self.price_df["date"] <= end_date)].copy()

    def _build_snapshot(self, trade_date, execution_date, etf_samples, sector_samples, candidates) -> pd.DataFrame:
        trade_date = pd.Timestamp(trade_date).normalize()
        market_state = str(etf_samples["market_state"].iloc[0]) if not etf_samples.empty else "unknown"
        exclude = candidates.get("exclude_reason", pd.Series(dtype=str)).fillna("").astype(str)
        snapshot = pd.DataFrame(
            [
                {
                    "trade_date": trade_date,
                    "signal_date": trade_date,
                    "execution_date": execution_date,
                    "market_state": market_state,
                    "etf_count": int(len(etf_samples)),
                    "sector_count": int(len(sector_samples)),
                    "candidate_count": int(candidates.get("was_candidate", pd.Series(dtype=bool)).sum()),
                    "selected_count": int(candidates.get("was_selected", pd.Series(dtype=bool)).sum()),
                    "bought_count": int(candidates.get("was_bought", pd.Series(dtype=bool)).sum()),
                    "defense_block_count": int(exclude.str.contains("defense", case=False).sum()),
                    "filtered_count": int((~candidates.get("was_candidate", pd.Series(dtype=bool))).sum()),
                    "data_abnormal_count": int(exclude.str.contains("data_abnormal", case=False).sum()),
                    "source": self.config.source,
                }
            ]
        )
        return reorder_columns(snapshot, DAILY_DECISION_SNAPSHOT_COLUMNS)
