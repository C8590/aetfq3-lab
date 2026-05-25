from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import HistoricalMLConfig
from .daily_universe import generated_output_dir
from .io_utils import reorder_columns, write_table
from .schemas import DAILY_ML_UNIVERSE_SAMPLE_COLUMNS, ENTRY_CANDIDATE_COLUMNS, FUTURE_LABEL_COLUMNS, ML_ENTRY_LABEL_COLUMNS


def _build_future_label_frame(price_df: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]]:
    prices = price_df.copy()
    prices["date"] = pd.to_datetime(prices["date"]).dt.normalize()
    prices["code"] = prices["code"].astype(str)
    if "high" not in prices.columns:
        prices["high"] = prices["close"]
    if "low" not in prices.columns:
        prices["low"] = prices["close"]
    if "sector" not in prices.columns and "sector_level2" in prices.columns:
        prices["sector"] = prices["sector_level2"]
    if "sector" not in prices.columns:
        prices["sector"] = ""
    for column in ["close", "high", "low"]:
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")

    dates = list(pd.to_datetime(prices["date"].dropna().sort_values().unique()))
    codes = list(prices["code"].dropna().astype(str).sort_values().unique())
    if not dates or not codes:
        return pd.DataFrame(), pd.DataFrame(), dates

    grid_index = pd.MultiIndex.from_product([codes, dates], names=["code", "date"])
    grid = prices.set_index(["code", "date"]).reindex(grid_index).reset_index()
    grid["sector"] = grid.groupby("code")["sector"].transform(lambda s: s.ffill().bfill()).fillna("").astype(str)
    grouped = grid.groupby("code", sort=False)
    close = pd.to_numeric(grid["close"], errors="coerce")
    high = pd.to_numeric(grid["high"], errors="coerce")
    low = pd.to_numeric(grid["low"], errors="coerce")

    for horizon in horizons:
        grid[f"future_return_{horizon}d"] = grouped["close"].transform(
            lambda s, h=horizon: pd.to_numeric(s, errors="coerce").shift(-h) / pd.to_numeric(s, errors="coerce") - 1.0
        )
    grid["future_max_gain_10d"] = grouped["high"].transform(lambda s: _forward_window(pd.to_numeric(s, errors="coerce"), 10, "max")) / close - 1.0
    grid["future_max_drawdown_10d"] = grouped["low"].transform(lambda s: _forward_window(pd.to_numeric(s, errors="coerce"), 10, "min")) / close - 1.0
    grid["future_min_low_3d"] = grouped["low"].transform(lambda s: _forward_window(pd.to_numeric(s, errors="coerce"), 3, "min"))

    grid.loc[~close.gt(0), [c for c in grid.columns if c.startswith("future_")]] = np.nan
    grid["market_return_10d"] = grid.groupby("date")["future_return_10d"].transform("mean")
    sector_returns = (
        grid.groupby(["date", "sector"], dropna=False)["future_return_10d"]
        .mean()
        .reset_index(name="sector_return_10d")
    )
    grid = grid.merge(sector_returns, on=["date", "sector"], how="left")
    return grid, sector_returns, dates


def _forward_window(series: pd.Series, window: int, method: str) -> pd.Series:
    reverse = series.iloc[::-1].shift(1).rolling(window=window, min_periods=1)
    result = getattr(reverse, method)().iloc[::-1]
    result.index = series.index
    return result


def _next_date_map(dates: list[pd.Timestamp]) -> dict[pd.Timestamp, pd.Timestamp]:
    return {pd.Timestamp(date).normalize(): dates[idx + 1] for idx, date in enumerate(dates[:-1])}


@dataclass
class FutureLabeler:
    """Attach future performance labels after replay samples are generated.

    This stage intentionally uses future data and must run after feature sample
    generation.  It never feeds values back into the replay engine.
    """

    price_df: pd.DataFrame
    config: HistoricalMLConfig = HistoricalMLConfig()
    precomputed_label_cache: tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]] | None = None

    def __post_init__(self):
        self.price_df = self.price_df.copy()
        self.price_df["date"] = pd.to_datetime(self.price_df["date"]).dt.normalize()
        self.price_df["code"] = self.price_df["code"].astype(str)
        if "high" not in self.price_df.columns:
            self.price_df["high"] = self.price_df["close"]
        if "low" not in self.price_df.columns:
            self.price_df["low"] = self.price_df["close"]
        if "sector_l1" not in self.price_df.columns:
            self.price_df["sector_l1"] = self.price_df["sector"]
        self.price_df = self.price_df.sort_values(["date", "code"]).reset_index(drop=True)
        self.trading_dates = list(sorted(pd.to_datetime(self.price_df["date"].unique())))
        self.date_pos = {pd.Timestamp(d): i for i, d in enumerate(self.trading_dates)}
        self.by_code = {code: g.sort_values("date").set_index("date") for code, g in self.price_df.groupby("code")}
        horizons = tuple(sorted({int(h) for h in self.config.label_horizons if int(h) > 0} | {1, 3, 5, 10, 20}))
        if self.precomputed_label_cache is None:
            self.label_frame, self.sector_return_frame, self.label_dates = _build_future_label_frame(self.price_df, horizons)
        else:
            self.label_frame, self.sector_return_frame, self.label_dates = self.precomputed_label_cache
        self.label_date_pos = {pd.Timestamp(d): i for i, d in enumerate(self.label_dates)}
        self.next_dates = _next_date_map(self.label_dates)
        self.known_codes = set(self.price_df["code"].astype(str))

    def attach_labels(self, candidates: pd.DataFrame) -> pd.DataFrame:
        if candidates.empty:
            return candidates.copy()

        base = candidates.copy().reset_index(drop=True)
        base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce").dt.normalize()
        base["code"] = base["code"].astype(str)
        base["execution_date"] = pd.to_datetime(base.get("execution_date", pd.Series(pd.NaT, index=base.index)), errors="coerce").dt.normalize()
        date_set = set(self.label_dates)
        fallback_base = base["trade_date"].map(self.next_dates)
        base["_label_base_date"] = base["execution_date"].where(base["execution_date"].isin(date_set), fallback_base)
        base["_sample_sector"] = base.get("sector", pd.Series("", index=base.index)).fillna("").astype(str)

        labels = self._merge_label_columns(base, "_label_base_date", label_base_column="label_base_date", missing_status="missing_base_date_or_code")
        out = pd.concat([base.drop(columns=["_label_base_date", "_sample_sector"]).reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
        out = self._assign_auto_label(out)
        return reorder_columns(out, list(ENTRY_CANDIDATE_COLUMNS) + FUTURE_LABEL_COLUMNS)

    def _merge_label_columns(
        self,
        base: pd.DataFrame,
        base_date_column: str,
        *,
        label_base_column: str,
        missing_status: str,
    ) -> pd.DataFrame:
        lookup_columns = [
            "code",
            "date",
            "close",
            "future_return_1d",
            "future_return_3d",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "future_max_gain_10d",
            "future_max_drawdown_10d",
            "future_min_low_3d",
            "market_return_10d",
        ]
        lookup = self.label_frame[[c for c in lookup_columns if c in self.label_frame.columns]].copy()
        lookup = lookup.rename(columns={"close": "_label_close"})
        merged = base.merge(
            lookup,
            left_on=["code", base_date_column],
            right_on=["code", "date"],
            how="left",
        )
        sector_returns = self.sector_return_frame.rename(columns={"date": base_date_column, "sector": "_sample_sector"})
        merged = merged.merge(sector_returns, on=[base_date_column, "_sample_sector"], how="left")

        labels = pd.DataFrame(index=base.index)
        for column in FUTURE_LABEL_COLUMNS:
            if column not in {"outperform_market_10d", "outperform_sector_10d", "exit_within_3d", "auto_label", "label_status"}:
                labels[column] = np.nan
        labels["outperform_market_10d"] = False
        labels["outperform_sector_10d"] = False
        labels["exit_within_3d"] = False
        labels["auto_label"] = "unlabeled"
        labels["label_status"] = "ok"
        labels[label_base_column] = base[base_date_column]

        for column in [
            "future_return_1d",
            "future_return_3d",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "future_max_gain_10d",
            "future_max_drawdown_10d",
            "market_return_10d",
            "sector_return_10d",
        ]:
            labels[column] = merged.get(column, pd.Series(np.nan, index=merged.index)).to_numpy()

        known_base = base[base_date_column].isin(set(self.label_dates)) & base["code"].astype(str).isin(self.known_codes)
        labels.loc[~known_base, "label_status"] = missing_status
        label_close = pd.to_numeric(merged.get("_label_close", pd.Series(np.nan, index=merged.index)), errors="coerce")
        missing_price = known_base & label_close.isna()
        labels.loc[missing_price, "label_status"] = "missing_base_price"
        bad_price = known_base & label_close.notna() & ~label_close.gt(0)
        labels.loc[bad_price, "label_status"] = "bad_base_price"
        max_horizon = max(self.config.label_horizons)
        insufficient = base[base_date_column].map(self.label_date_pos).fillna(len(self.label_dates)).astype(int) + max_horizon >= len(self.label_dates)
        labels.loc[known_base & ~missing_price & ~bad_price & insufficient, "label_status"] = "insufficient_future_data"

        labels["outperform_market_10d"] = labels["future_return_10d"].notna() & labels["market_return_10d"].notna() & (
            labels["future_return_10d"] > labels["market_return_10d"]
        )
        labels["outperform_sector_10d"] = labels["future_return_10d"].notna() & labels["sector_return_10d"].notna() & (
            labels["future_return_10d"] > labels["sector_return_10d"]
        )
        draw3 = pd.to_numeric(merged.get("future_min_low_3d", pd.Series(np.nan, index=merged.index)), errors="coerce") / label_close - 1.0
        labels["exit_within_3d"] = (draw3 <= self.config.quick_failure_return_3d) | (
            pd.to_numeric(labels["future_return_3d"], errors="coerce") <= self.config.quick_failure_return_3d
        )
        return labels

    def _resolve_base_date(self, row) -> pd.Timestamp | None:
        execution_date = row.get("execution_date", pd.NaT)
        if pd.notna(execution_date):
            execution_date = pd.Timestamp(execution_date).normalize()
            if execution_date in self.date_pos:
                return execution_date
        trade_date = pd.Timestamp(row["trade_date"]).normalize()
        for d in self.trading_dates:
            if d > trade_date:
                return d
        return None

    def _label_one(self, row) -> dict:
        code = str(row["code"])
        sector = str(row.get("sector", ""))
        base_date = self._resolve_base_date(row)
        labels = {c: np.nan for c in FUTURE_LABEL_COLUMNS if c not in {"outperform_market_10d", "outperform_sector_10d", "exit_within_3d", "auto_label", "label_status"}}
        labels.update(
            {
                "outperform_market_10d": False,
                "outperform_sector_10d": False,
                "exit_within_3d": False,
                "auto_label": "unlabeled",
                "label_status": "ok",
            }
        )
        if base_date is None or code not in self.by_code:
            labels["label_status"] = "missing_base_date_or_code"
            return labels
        labels["label_base_date"] = base_date
        g = self.by_code[code]
        if base_date not in g.index or pd.isna(g.loc[base_date, "close"]):
            labels["label_status"] = "missing_base_price"
            return labels
        base_close = float(g.loc[base_date, "close"])
        if base_close <= 0:
            labels["label_status"] = "bad_base_price"
            return labels

        base_pos = self.date_pos[base_date]
        max_horizon = max(self.config.label_horizons)
        if base_pos + max_horizon >= len(self.trading_dates):
            labels["label_status"] = "insufficient_future_data"

        for h in self.config.label_horizons:
            if base_pos + h < len(self.trading_dates):
                target_date = self.trading_dates[base_pos + h]
                if target_date in g.index and pd.notna(g.loc[target_date, "close"]):
                    labels[f"future_return_{h}d"] = float(g.loc[target_date, "close"] / base_close - 1.0)

        horizon_dates_10 = self.trading_dates[base_pos + 1 : min(base_pos + 11, len(self.trading_dates))]
        future = g.loc[g.index.intersection(horizon_dates_10)]
        if not future.empty:
            labels["future_max_gain_10d"] = float(future["high"].max() / base_close - 1.0)
            labels["future_max_drawdown_10d"] = float(future["low"].min() / base_close - 1.0)

        # Equal-weight market and sector returns use only the label window; this is label side only.
        if base_pos + 10 < len(self.trading_dates):
            target_10d = self.trading_dates[base_pos + 10]
            labels["market_return_10d"] = self._basket_return(base_date, target_10d, sector=None)
            labels["sector_return_10d"] = self._basket_return(base_date, target_10d, sector=sector)
            if pd.notna(labels.get("future_return_10d")) and pd.notna(labels.get("market_return_10d")):
                labels["outperform_market_10d"] = bool(labels["future_return_10d"] > labels["market_return_10d"])
            if pd.notna(labels.get("future_return_10d")) and pd.notna(labels.get("sector_return_10d")):
                labels["outperform_sector_10d"] = bool(labels["future_return_10d"] > labels["sector_return_10d"])

        min_3d = labels.get("future_max_drawdown_10d")
        if base_pos + 3 < len(self.trading_dates):
            dates_3 = self.trading_dates[base_pos + 1 : base_pos + 4]
            f3 = g.loc[g.index.intersection(dates_3)]
            if not f3.empty:
                draw3 = float(f3["low"].min() / base_close - 1.0)
                ret3 = labels.get("future_return_3d")
                labels["exit_within_3d"] = bool(
                    draw3 <= self.config.quick_failure_return_3d
                    or (pd.notna(ret3) and ret3 <= self.config.quick_failure_return_3d)
                )
        return labels

    def _basket_return(self, base_date, target_date, sector: str | None) -> float:
        base = self.price_df.loc[self.price_df["date"] == base_date, ["code", "close", "sector"]].rename(columns={"close": "base_close"})
        target = self.price_df.loc[self.price_df["date"] == target_date, ["code", "close"]].rename(columns={"close": "target_close"})
        merged = base.merge(target, on="code", how="inner")
        if sector is not None:
            merged = merged.loc[merged["sector"].astype(str) == str(sector)]
        merged = merged.loc[(merged["base_close"] > 0) & merged["target_close"].notna()]
        if merged.empty:
            return np.nan
        return float((merged["target_close"] / merged["base_close"] - 1.0).mean())

    def _assign_auto_label(self, out: pd.DataFrame) -> pd.DataFrame:
        out = out.copy()
        out["auto_label"] = "neutral_entry"
        incomplete = out["label_status"].fillna("").astype(str) != "ok"
        good = (
            (out["future_return_10d"] >= self.config.good_return_10d)
            & (out["future_max_drawdown_10d"] > self.config.bad_drawdown_10d)
            & (out["outperform_market_10d"].astype(bool) | out["outperform_sector_10d"].astype(bool))
        )
        bad = (
            (out["future_return_10d"] <= self.config.bad_return_10d)
            | (out["future_max_drawdown_10d"] <= self.config.bad_drawdown_10d)
            | out["exit_within_3d"].astype(bool)
        )
        out.loc[good, "auto_label"] = "good_entry"
        out.loc[bad, "auto_label"] = "bad_entry"
        out.loc[incomplete, "auto_label"] = "unlabeled"
        return out


@dataclass
class MLEntryUniverseLabeler:
    """Attach training-only labels to daily full-universe ML samples.

    ``daily_ml_universe_samples`` is the feature_at_t table.  This labeler is
    the separate label_after_t stage: it may use future prices, but it never
    changes or feeds values back into replay or live entry fields.
    """

    price_df: pd.DataFrame
    config: HistoricalMLConfig = HistoricalMLConfig()
    label_horizons: tuple[int, ...] | None = None
    precomputed_label_cache: tuple[pd.DataFrame, pd.DataFrame, list[pd.Timestamp]] | None = None

    def __post_init__(self):
        self.price_df = self.price_df.copy()
        self.price_df["date"] = pd.to_datetime(self.price_df["date"]).dt.normalize()
        self.price_df["code"] = self.price_df["code"].astype(str)
        if "high" not in self.price_df.columns:
            self.price_df["high"] = self.price_df["close"]
        if "low" not in self.price_df.columns:
            self.price_df["low"] = self.price_df["close"]
        if "sector" not in self.price_df.columns and "sector_level2" in self.price_df.columns:
            self.price_df["sector"] = self.price_df["sector_level2"]
        if "sector" not in self.price_df.columns:
            self.price_df["sector"] = ""
        self.price_df = self.price_df.sort_values(["date", "code"]).reset_index(drop=True)
        self.trading_dates = list(sorted(pd.to_datetime(self.price_df["date"].unique())))
        self.date_pos = {pd.Timestamp(d): i for i, d in enumerate(self.trading_dates)}
        self.by_code = {code: g.sort_values("date").set_index("date") for code, g in self.price_df.groupby("code")}
        horizons = self.label_horizons or self.config.label_horizons
        self.horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
        for required in (1, 3, 5, 10, 20):
            if required not in self.horizons:
                self.horizons = tuple(sorted((*self.horizons, required)))
        if self.precomputed_label_cache is None:
            self.label_frame, self.sector_return_frame, self.label_dates = _build_future_label_frame(self.price_df, self.horizons)
        else:
            self.label_frame, self.sector_return_frame, self.label_dates = self.precomputed_label_cache
        self.label_date_pos = {pd.Timestamp(d): i for i, d in enumerate(self.label_dates)}
        self.next_dates = _next_date_map(self.label_dates)
        self.known_codes = set(self.price_df["code"].astype(str))

    def attach_labels(self, universe_samples: pd.DataFrame) -> pd.DataFrame:
        if universe_samples.empty:
            return universe_samples.copy()

        base = universe_samples.copy().reset_index(drop=True)
        base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce").dt.normalize()
        base["code"] = base["code"].astype(str)
        base["_sample_sector"] = base.get("sector_level2", base.get("sector", pd.Series("", index=base.index))).fillna("").astype(str)
        label_df = self._vectorized_labels(base)
        out = pd.concat([universe_samples.reset_index(drop=True), label_df.reset_index(drop=True)], axis=1)
        out = self._assign_auto_label(out)
        return reorder_columns(out, list(DAILY_ML_UNIVERSE_SAMPLE_COLUMNS) + ML_ENTRY_LABEL_COLUMNS)

    def _vectorized_labels(self, base: pd.DataFrame) -> pd.DataFrame:
        lookup_columns = [
            "code",
            "date",
            "close",
            "future_return_1d",
            "future_return_3d",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "future_max_gain_10d",
            "future_max_drawdown_10d",
            "market_return_10d",
        ]
        lookup = self.label_frame[[c for c in lookup_columns if c in self.label_frame.columns]].copy()
        lookup = lookup.rename(columns={"close": "_label_close"})
        merged = base.merge(lookup, left_on=["code", "trade_date"], right_on=["code", "date"], how="left")
        sector_returns = self.sector_return_frame.rename(columns={"date": "trade_date", "sector": "_sample_sector"})
        merged = merged.merge(sector_returns, on=["trade_date", "_sample_sector"], how="left")

        labels = pd.DataFrame(index=base.index)
        for column in ML_ENTRY_LABEL_COLUMNS:
            labels[column] = np.nan
        labels["feature_at_t"] = base["trade_date"]
        labels["label_after_t"] = base["trade_date"].map(self.next_dates)
        labels["outperform_market_10d"] = False
        labels["outperform_sector_10d"] = False
        labels["hit_stop_loss_10d"] = False
        labels["auto_label"] = "unlabeled"
        labels["label_reason_cn"] = "未来数据不足，暂不标注。"
        labels["label_status"] = "ok"

        for column in [
            "future_return_1d",
            "future_return_3d",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "future_max_gain_10d",
            "future_max_drawdown_10d",
        ]:
            labels[column] = merged.get(column, pd.Series(np.nan, index=merged.index)).to_numpy()
        labels["market_return_10d"] = merged.get("market_return_10d", pd.Series(np.nan, index=merged.index)).to_numpy()
        labels["sector_return_10d"] = merged.get("sector_return_10d", pd.Series(np.nan, index=merged.index)).to_numpy()

        known_base = base["trade_date"].isin(set(self.label_dates)) & base["code"].astype(str).isin(self.known_codes)
        labels.loc[~known_base, "label_status"] = "missing_feature_date_or_code"
        labels.loc[~known_base, "label_reason_cn"] = "缺少特征日或ETF价格序列，无法生成标签。"
        label_close = pd.to_numeric(merged.get("_label_close", pd.Series(np.nan, index=merged.index)), errors="coerce")
        missing_price = known_base & label_close.isna()
        labels.loc[missing_price, "label_status"] = "missing_feature_price"
        labels.loc[missing_price, "label_reason_cn"] = "特征日缺少收盘价，无法生成标签。"
        bad_price = known_base & label_close.notna() & ~label_close.gt(0)
        labels.loc[bad_price, "label_status"] = "bad_feature_price"
        labels.loc[bad_price, "label_reason_cn"] = "特征日收盘价无效，无法生成标签。"

        max_horizon = max(self.horizons)
        insufficient = base["trade_date"].map(self.label_date_pos).fillna(len(self.label_dates)).astype(int) + max_horizon >= len(self.label_dates)
        labels.loc[known_base & ~missing_price & ~bad_price & insufficient, "label_status"] = "insufficient_future_data"

        labels["outperform_market_10d"] = labels["future_return_10d"].notna() & labels["market_return_10d"].notna() & (
            labels["future_return_10d"] > labels["market_return_10d"]
        )
        labels["outperform_sector_10d"] = labels["future_return_10d"].notna() & labels["sector_return_10d"].notna() & (
            labels["future_return_10d"] > labels["sector_return_10d"]
        )
        labels["hit_stop_loss_10d"] = pd.to_numeric(labels["future_max_drawdown_10d"], errors="coerce") <= self.config.bad_drawdown_10d
        return labels

    def write_outputs(self, labeled: pd.DataFrame, out_dir: str | Path) -> Path:
        generated = generated_output_dir(out_dir)
        samples_path = write_table(labeled, generated, "ml_entry_labeled_samples", self.config.output_format)
        summary_path = generated / "ml_entry_label_summary.json"
        summary_path.write_text(json.dumps(self.label_summary(labeled), ensure_ascii=False, indent=2), encoding="utf-8")
        return samples_path

    def label_summary(self, labeled: pd.DataFrame) -> dict:
        counts = labeled.get("auto_label", pd.Series(dtype=str)).fillna("unlabeled").astype(str).value_counts()
        reason_counts = (
            labeled.groupby(["auto_label", "label_reason_cn"], dropna=False)
            .size()
            .reset_index(name="count")
            .to_dict(orient="records")
            if {"auto_label", "label_reason_cn"}.issubset(labeled.columns)
            else []
        )
        return {
            "source": self.config.source,
            "total_rows": int(len(labeled)),
            "label_distribution": {str(k): int(v) for k, v in counts.items()},
            "good_entry_count": int(counts.get("good_entry", 0)),
            "bad_entry_count": int(counts.get("bad_entry", 0)),
            "neutral_entry_count": int(counts.get("neutral_entry", 0)),
            "unlabeled_count": int(counts.get("unlabeled", 0)),
            "reason_distribution": reason_counts,
        }

    def _label_one(self, row) -> dict:
        feature_at_t = pd.Timestamp(row["trade_date"]).normalize()
        code = str(row["code"])
        sector = str(row.get("sector_level2", row.get("sector", "")))
        labels = {column: np.nan for column in ML_ENTRY_LABEL_COLUMNS}
        labels.update(
            {
                "feature_at_t": feature_at_t,
                "label_after_t": pd.NaT,
                "outperform_market_10d": False,
                "outperform_sector_10d": False,
                "hit_stop_loss_10d": False,
                "auto_label": "unlabeled",
                "label_reason_cn": "未来数据不足，暂不标注。",
                "label_status": "ok",
            }
        )
        if feature_at_t not in self.date_pos or code not in self.by_code:
            labels["label_status"] = "missing_feature_date_or_code"
            labels["label_reason_cn"] = "缺少特征日或ETF价格序列，无法生成标签。"
            return labels

        g = self.by_code[code]
        if feature_at_t not in g.index or pd.isna(g.loc[feature_at_t, "close"]):
            labels["label_status"] = "missing_feature_price"
            labels["label_reason_cn"] = "特征日缺少收盘价，无法生成标签。"
            return labels

        base_close = float(g.loc[feature_at_t, "close"])
        if base_close <= 0:
            labels["label_status"] = "bad_feature_price"
            labels["label_reason_cn"] = "特征日收盘价无效，无法生成标签。"
            return labels

        base_pos = self.date_pos[feature_at_t]
        if base_pos + 1 < len(self.trading_dates):
            labels["label_after_t"] = self.trading_dates[base_pos + 1]
        if base_pos + max(self.horizons) >= len(self.trading_dates):
            labels["label_status"] = "insufficient_future_data"

        for horizon in self.horizons:
            target_pos = base_pos + horizon
            if target_pos >= len(self.trading_dates):
                continue
            target_date = self.trading_dates[target_pos]
            if target_date in g.index and pd.notna(g.loc[target_date, "close"]):
                labels[f"future_return_{horizon}d"] = float(g.loc[target_date, "close"] / base_close - 1.0)

        window_10 = self.trading_dates[base_pos + 1 : min(base_pos + 11, len(self.trading_dates))]
        future_10 = g.loc[g.index.intersection(window_10)]
        if not future_10.empty:
            labels["future_max_gain_10d"] = float(future_10["high"].max() / base_close - 1.0)
            labels["future_max_drawdown_10d"] = float(future_10["low"].min() / base_close - 1.0)
            labels["hit_stop_loss_10d"] = bool(labels["future_max_drawdown_10d"] <= self.config.bad_drawdown_10d)

        if base_pos + 10 < len(self.trading_dates):
            target_10d = self.trading_dates[base_pos + 10]
            labels["market_return_10d"] = self._basket_return(feature_at_t, target_10d, sector=None)
            labels["sector_return_10d"] = self._basket_return(feature_at_t, target_10d, sector=sector)
            if pd.notna(labels.get("future_return_10d")) and pd.notna(labels.get("market_return_10d")):
                labels["outperform_market_10d"] = bool(labels["future_return_10d"] > labels["market_return_10d"])
            if pd.notna(labels.get("future_return_10d")) and pd.notna(labels.get("sector_return_10d")):
                labels["outperform_sector_10d"] = bool(labels["future_return_10d"] > labels["sector_return_10d"])
        return labels

    def _basket_return(self, base_date, target_date, sector: str | None) -> float:
        base = self.price_df.loc[self.price_df["date"] == base_date, ["code", "close", "sector"]].rename(columns={"close": "base_close"})
        target = self.price_df.loc[self.price_df["date"] == target_date, ["code", "close"]].rename(columns={"close": "target_close"})
        merged = base.merge(target, on="code", how="inner")
        if sector is not None:
            merged = merged.loc[merged["sector"].astype(str) == str(sector)]
        merged = merged.loc[(merged["base_close"] > 0) & merged["target_close"].notna()]
        if merged.empty:
            return np.nan
        return float((merged["target_close"] / merged["base_close"] - 1.0).mean())

    def _assign_auto_label(self, out: pd.DataFrame) -> pd.DataFrame:
        out = out.copy()
        out["auto_label"] = "neutral_entry"
        out["label_reason_cn"] = "未来10日表现未达到优秀或风险标签阈值，标记为中性样本。"

        incomplete = out["label_status"].fillna("").astype(str) != "ok"
        drawdown_ok = out["future_max_drawdown_10d"] > self.config.bad_drawdown_10d
        good = (
            (out["future_return_10d"] > 0)
            & drawdown_ok
            & out["outperform_market_10d"].astype(bool)
            & out["outperform_sector_10d"].astype(bool)
        )
        bad = (
            (out["future_max_drawdown_10d"] <= self.config.bad_drawdown_10d)
            | (out["future_return_5d"] <= self.config.quick_failure_return_3d)
            | (out["future_return_10d"] < out.get("market_return_10d", pd.Series(np.nan, index=out.index)))
            | out["hit_stop_loss_10d"].astype(bool)
        )

        out.loc[good, "auto_label"] = "good_entry"
        out.loc[good, "label_reason_cn"] = "未来10日收益为正，最大回撤可控，同时跑赢市场和所属板块。"
        out.loc[bad, "auto_label"] = "bad_entry"
        out.loc[bad, "label_reason_cn"] = "未来短期回撤过大、跑输市场，或10日内触发止损阈值。"
        out.loc[incomplete, "auto_label"] = "unlabeled"
        out.loc[incomplete, "label_reason_cn"] = "未来数据不足或价格缺失，暂不参与训练标签。"
        return out
