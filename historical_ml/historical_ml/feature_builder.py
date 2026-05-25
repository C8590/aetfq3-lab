from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import HistoricalMLConfig


def _safe_zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    std = s.std(skipna=True, ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean(skipna=True)) / std


def _max_drawdown(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").dropna()
    if len(v) < 2:
        return np.nan
    running_max = v.cummax()
    dd = v / running_max - 1.0
    return float(dd.min())


def _last_or_nan(values: pd.Series, idx_from_end: int) -> float:
    if len(values) <= idx_from_end:
        return np.nan
    return float(values.iloc[-1] / values.iloc[-1 - idx_from_end] - 1.0)


@dataclass
class ReplayFeatureCache:
    """Reusable feature matrix for historical replay.

    The cache is built from prices available at each row's trade date only.
    Future labels are intentionally not part of this table.
    """

    etf_features: pd.DataFrame
    sector_features: pd.DataFrame

    @classmethod
    def build(cls, price_df: pd.DataFrame, config: HistoricalMLConfig, dates: list[pd.Timestamp] | None = None) -> "ReplayFeatureCache":
        frame = _normalized_price_frame(price_df)
        if frame.empty:
            return cls(pd.DataFrame(), pd.DataFrame())

        features = _build_all_etf_features(frame, config)
        if dates is not None:
            wanted = set(pd.to_datetime(dates).normalize())
            features = features.loc[features["trade_date"].isin(wanted)].copy()
        sectors = build_sector_features(features, config)
        if not sectors.empty:
            features = features.merge(
                sectors[["trade_date", "sector", "sector_rank", "sector_state", "sector_score"]],
                on=["trade_date", "sector"],
                how="left",
            )
        else:
            features["sector_rank"] = np.nan
            features["sector_state"] = "unknown"
            features["sector_score"] = np.nan

        features["etf_rank"] = features.groupby(["trade_date", "sector"])["entry_score"].rank(
            ascending=False, method="first"
        ).astype(int)
        features["global_rank"] = features.groupby("trade_date")["entry_score"].rank(
            ascending=False, method="first"
        ).astype(int)
        features["source"] = config.source
        features = features.sort_values(["trade_date", "code"]).reset_index(drop=True)
        return cls(features, sectors)

    def for_day(self, trade_date) -> tuple[pd.DataFrame, pd.DataFrame]:
        trade_date = pd.Timestamp(trade_date).normalize()
        etf = self.etf_features.loc[self.etf_features["trade_date"] == trade_date].copy()
        sector = self.sector_features.loc[self.sector_features["trade_date"] == trade_date].copy()
        return etf.reset_index(drop=True), sector.reset_index(drop=True)

    def for_dates(self, dates: list[pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
        wanted = set(pd.to_datetime(dates).normalize())
        etf = self.etf_features.loc[self.etf_features["trade_date"].isin(wanted)].copy()
        sector = self.sector_features.loc[self.sector_features["trade_date"].isin(wanted)].copy()
        return etf.reset_index(drop=True), sector.reset_index(drop=True)


def _normalized_price_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    frame = price_df.copy()
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"]).dt.normalize()
    frame["code"] = frame["code"].astype(str)
    if "name" not in frame.columns:
        frame["name"] = frame["code"]
    if "sector" not in frame.columns:
        frame["sector"] = "UNKNOWN"
    if "sector_l1" not in frame.columns:
        frame["sector_l1"] = frame.get("sector_level1", frame["sector"])
    if "sector_level1" not in frame.columns:
        frame["sector_level1"] = frame["sector_l1"]
    if "sector_level2" not in frame.columns:
        frame["sector_level2"] = frame["sector"]
    for column in ["open", "high", "low", "close", "volume", "amount"]:
        if column not in frame.columns:
            frame[column] = frame["close"] if column in {"open", "high", "low"} else 0.0
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_values(["code", "date"]).reset_index(drop=True)


def _rolling_drawdown(values: pd.Series, window: int) -> pd.Series:
    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    out = np.full(len(arr), np.nan, dtype=float)
    for idx in range(1, len(arr)):
        start = max(0, idx - window + 1)
        clean = arr[start : idx + 1]
        clean = clean[np.isfinite(clean)]
        if clean.size < 2:
            continue
        running_max = np.maximum.accumulate(clean)
        out[idx] = float(np.min(clean / running_max - 1.0))
    return pd.Series(out, index=values.index)


def _zscore_by_date(frame: pd.DataFrame, column: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    means = values.groupby(frame["trade_date"]).transform("mean")
    stds = values.groupby(frame["trade_date"]).transform(lambda s: s.std(skipna=True, ddof=0))
    out = (values - means) / stds.replace(0, np.nan)
    return out.fillna(0.0)


def _build_all_etf_features(price_df: pd.DataFrame, config: HistoricalMLConfig) -> pd.DataFrame:
    w20, w60, w120 = config.momentum_windows
    frame = price_df.sort_values(["code", "date"]).copy()
    grouped = frame.groupby("code", sort=False)
    close = pd.to_numeric(frame["close"], errors="coerce")
    amount = pd.to_numeric(frame["amount"], errors="coerce")

    frame["trade_date"] = frame["date"]
    frame["_row_count"] = grouped.cumcount() + 1
    for window in (w20, w60, w120):
        frame[f"r{window}"] = grouped["close"].transform(lambda s, w=window: pd.to_numeric(s, errors="coerce") / pd.to_numeric(s, errors="coerce").shift(w) - 1.0)
        frame[f"ma{window}"] = grouped["close"].transform(
            lambda s, w=window: pd.to_numeric(s, errors="coerce").rolling(window=w, min_periods=2).mean()
        )

    frame["avg_amount_20d"] = grouped["amount"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(window=20, min_periods=1).mean()
    )
    frame["vol20"] = grouped["close"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").pct_change().rolling(window=20, min_periods=4).std(ddof=0)
    )
    frame["max_drawdown_20d"] = grouped["close"].transform(lambda s: _rolling_drawdown(pd.to_numeric(s, errors="coerce"), 20))
    frame["max_drawdown_60d"] = grouped["close"].transform(lambda s: _rolling_drawdown(pd.to_numeric(s, errors="coerce"), 60))
    frame["missing_ratio_60d"] = grouped["close"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").isna().rolling(window=60, min_periods=1).mean()
    )

    frame["data_quality_flag"] = "ok"
    frame.loc[frame["_row_count"] < config.min_history_days, "data_quality_flag"] = "insufficient_history"
    frame.loc[
        (frame["data_quality_flag"] == "ok") & (frame["missing_ratio_60d"] > config.max_missing_ratio_60d),
        "data_quality_flag",
    ] = "missing_data"
    frame.loc[
        (frame["data_quality_flag"] == "ok") & (frame["avg_amount_20d"] < config.min_avg_amount_20d),
        "data_quality_flag",
    ] = "low_liquidity"
    frame.loc[
        (frame["data_quality_flag"] == "ok") & (close.isna() | (close <= 0)),
        "data_quality_flag",
    ] = "bad_price"

    frame["abs_trend_score"] = 0.0
    valid_close = close.notna()
    frame.loc[valid_close & frame["r60"].gt(0), "abs_trend_score"] += 0.35
    frame.loc[valid_close & frame["r120"].gt(0), "abs_trend_score"] += 0.35
    frame.loc[valid_close & frame["ma60"].notna() & close.gt(frame["ma60"]), "abs_trend_score"] += 0.30

    rolling_min_20 = grouped["close"].transform(
        lambda s: pd.to_numeric(s, errors="coerce").rolling(window=20, min_periods=5).min()
    )
    runup_20 = close / rolling_min_20 - 1.0
    runup_20 = runup_20.where(rolling_min_20.gt(0), 0.0).fillna(0.0)
    ma60_gap = close / frame["ma60"] - 1.0
    ma60_gap = ma60_gap.where(frame["ma60"].gt(0), 0.0).fillna(0.0)
    frame["trend_maturity"] = np.clip(0.5 * (runup_20 / 0.25) + 0.5 * (ma60_gap.clip(lower=0) / 0.18), 0, 1)
    frame["overheat_score"] = np.clip(0.6 * runup_20.clip(lower=0) / 0.35 + 0.4 * ma60_gap.clip(lower=0) / 0.25, 0, 1)

    z20 = _zscore_by_date(frame, "r20")
    z60 = _zscore_by_date(frame, "r60")
    z120 = _zscore_by_date(frame, "r120")
    mw20, mw60, mw120 = config.momentum_weights
    frame["momentum_score"] = mw20 * z20 + mw60 * z60 + mw120 * z120
    frame["liquidity_score"] = _safe_zscore_by_group(np.log1p(frame["avg_amount_20d"].fillna(0)), frame["trade_date"])
    vol_for_risk = frame["vol20"].fillna(frame.groupby("trade_date")["vol20"].transform("median"))
    frame["risk_score"] = _safe_zscore_by_group(vol_for_risk.fillna(0), frame["trade_date"]) + _safe_zscore_by_group(
        (-frame["max_drawdown_60d"]).fillna(0), frame["trade_date"]
    )

    frame = _add_acceleration(frame, config)
    frame["market_state"] = _market_state_by_date(frame, config)
    frame["entry_score"] = (
        0.55 * frame["momentum_score"].fillna(0)
        + 0.20 * frame["acceleration_score"].fillna(0)
        + 0.10 * frame["abs_trend_score"].fillna(0)
        + 0.05 * frame["liquidity_score"].fillna(0)
        - 0.10 * frame["risk_score"].fillna(0)
        - 0.05 * frame["overheat_score"].fillna(0)
    )
    return frame


def _safe_zscore_by_group(values: pd.Series, group_key: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    means = numeric.groupby(group_key).transform("mean")
    stds = numeric.groupby(group_key).transform(lambda s: s.std(skipna=True, ddof=0))
    return ((numeric - means) / stds.replace(0, np.nan)).fillna(0.0)


def _add_acceleration(frame: pd.DataFrame, config: HistoricalMLConfig) -> pd.DataFrame:
    dates = list(pd.to_datetime(frame["trade_date"].dropna().sort_values().unique()))
    if not dates:
        frame["acceleration_score"] = 0.0
        return frame
    lag_lookup = pd.DataFrame({"trade_date": dates})
    lag_lookup["lag_trade_date"] = lag_lookup["trade_date"].shift(config.acceleration_lag)
    out = frame.merge(lag_lookup, on="trade_date", how="left")
    lag_scores = frame[["code", "trade_date", "momentum_score"]].rename(
        columns={"trade_date": "lag_trade_date", "momentum_score": "momentum_score_lag"}
    )
    out = out.merge(lag_scores, on=["code", "lag_trade_date"], how="left")
    out["acceleration_score"] = out["momentum_score"] - out["momentum_score_lag"].fillna(0.0)
    return out.drop(columns=["lag_trade_date"], errors="ignore")


def _market_state_by_date(frame: pd.DataFrame, config: HistoricalMLConfig) -> pd.Series:
    if config.market_index_code and config.market_index_code in set(frame["code"].astype(str)):
        market = frame.loc[frame["code"].astype(str) == config.market_index_code, ["trade_date", "r60", "r120"]].copy()
        breadth = (
            frame.assign(_above_ma60=frame["close"] > frame["ma60"])
            .groupby("trade_date")["_above_ma60"]
            .mean()
            .reset_index(name="breadth")
        )
        stats = frame[["trade_date"]].drop_duplicates().merge(market, on="trade_date", how="left").merge(breadth, on="trade_date", how="left")
    else:
        stats = frame.assign(_above_ma60=frame["close"] > frame["ma60"]).groupby("trade_date").agg(
            r60=("r60", "mean"),
            r120=("r120", "mean"),
            breadth=("_above_ma60", "mean"),
        ).reset_index()

    r60 = stats["r60"].fillna(0.0)
    r120 = stats["r120"].fillna(0.0)
    breadth = stats["breadth"].fillna(0.0)
    stats["market_state"] = np.select(
        [(r60 > 0) & (r120 > 0) & (breadth >= 0.55), (r60 < 0) & (r120 < 0) & (breadth <= 0.45)],
        ["offense", "defense"],
        default="neutral",
    )
    return frame[["trade_date"]].merge(stats[["trade_date", "market_state"]], on="trade_date", how="left")["market_state"].fillna("unknown")


def build_sector_features(etf_features: pd.DataFrame, config: HistoricalMLConfig) -> pd.DataFrame:
    if etf_features.empty:
        return pd.DataFrame()

    records = []
    for (trade_date, sector), group in etf_features.groupby(["trade_date", "sector"], sort=False):
        g = group.copy()
        n_top = max(1, int(np.ceil(len(g) * 0.30)))
        top = g.nlargest(n_top, "momentum_score")
        sector_momentum = float(top["momentum_score"].mean(skipna=True))
        sector_acceleration = float(g.loc[g["acceleration_score"] > 0, "acceleration_score"].mean(skipna=True))
        if pd.isna(sector_acceleration):
            sector_acceleration = 0.0
        breadth = (
            0.40 * float((g["r60"] > 0).mean())
            + 0.40 * float((g["close"] > g["ma60"]).mean())
            + 0.20 * float((g["acceleration_score"] > 0).mean())
        )
        sector_risk = float(g["risk_score"].fillna(0).mean())
        proxy = sector_momentum + 0.3 * sector_acceleration + 0.2 * breadth - 0.1 * sector_risk
        records.append(
            {
                "trade_date": trade_date,
                "sector": sector,
                "sector_l1": g["sector_l1"].iloc[0],
                "market_state": g["market_state"].iloc[0],
                "sector_momentum_score": sector_momentum,
                "sector_acceleration_score": sector_acceleration,
                "sector_breadth_score": breadth,
                "sector_risk_score": sector_risk,
                "sector_entry_success_proxy": proxy,
                "candidate_count": int(len(g)),
            }
        )

    sectors = pd.DataFrame(records)
    sectors["sector_score"] = (
        0.50 * _safe_zscore_by_group(sectors["sector_momentum_score"], sectors["trade_date"])
        + 0.20 * _safe_zscore_by_group(sectors["sector_acceleration_score"], sectors["trade_date"])
        + 0.20 * _safe_zscore_by_group(sectors["sector_breadth_score"], sectors["trade_date"])
        - 0.10 * _safe_zscore_by_group(sectors["sector_risk_score"], sectors["trade_date"])
    )
    sectors["sector_rank"] = sectors.groupby("trade_date")["sector_score"].rank(ascending=False, method="first").astype(int)
    sectors["sector_state"] = np.select(
        [sectors["sector_score"] >= 0.5, sectors["sector_score"] <= -0.5],
        ["strong", "weak"],
        default="neutral",
    )
    sectors["source"] = config.source
    return sectors.sort_values(["trade_date", "sector_rank", "sector"]).reset_index(drop=True)


def _basic_features_for_date(price_df: pd.DataFrame, trade_date, config: HistoricalMLConfig) -> pd.DataFrame:
    """Compute cross-sectional features for one date using rows <= trade_date only."""

    trade_date = pd.Timestamp(trade_date).normalize()
    history = price_df.loc[price_df["date"] <= trade_date].copy()
    today = history.loc[history["date"] == trade_date].copy()
    if today.empty:
        return pd.DataFrame()

    records = []
    w20, w60, w120 = config.momentum_windows
    for code, g in history.groupby("code", sort=False):
        g = g.sort_values("date")
        if g.empty or g["date"].iloc[-1] != trade_date:
            continue
        latest = g.iloc[-1]
        close = pd.to_numeric(g["close"], errors="coerce")
        high = pd.to_numeric(g.get("high", close), errors="coerce")
        low = pd.to_numeric(g.get("low", close), errors="coerce")
        amount = pd.to_numeric(g.get("amount", pd.Series(0, index=g.index)), errors="coerce")

        r20 = _last_or_nan(close, w20)
        r60 = _last_or_nan(close, w60)
        r120 = _last_or_nan(close, w120)
        ma20 = float(close.tail(w20).mean()) if len(close.dropna()) >= max(2, min(w20, len(close))) else np.nan
        ma60 = float(close.tail(w60).mean()) if len(close.dropna()) >= max(2, min(w60, len(close))) else np.nan
        ma120 = float(close.tail(w120).mean()) if len(close.dropna()) >= max(2, min(w120, len(close))) else np.nan
        avg_amount_20d = float(amount.tail(20).mean()) if len(amount.dropna()) else 0.0
        vol20 = float(close.pct_change().tail(20).std(ddof=0)) if len(close) >= 5 else np.nan
        maxdd20 = _max_drawdown(close.tail(20))
        maxdd60 = _max_drawdown(close.tail(60))
        missing_ratio_60d = float(close.tail(60).isna().mean()) if len(close) else 1.0
        data_quality_flag = "ok"
        if len(g) < config.min_history_days:
            data_quality_flag = "insufficient_history"
        elif missing_ratio_60d > config.max_missing_ratio_60d:
            data_quality_flag = "missing_data"
        elif avg_amount_20d < config.min_avg_amount_20d:
            data_quality_flag = "low_liquidity"
        elif pd.isna(latest["close"]) or latest["close"] <= 0:
            data_quality_flag = "bad_price"

        close_today = float(latest["close"]) if pd.notna(latest["close"]) else np.nan
        abs_trend_score = 0.0
        if pd.notna(close_today):
            abs_trend_score += 0.35 if pd.notna(r60) and r60 > 0 else 0.0
            abs_trend_score += 0.35 if pd.notna(r120) and r120 > 0 else 0.0
            abs_trend_score += 0.30 if pd.notna(ma60) and close_today > ma60 else 0.0

        # Higher trend_maturity means more mature / more likely chase-high risk.
        runup_20 = close_today / close.tail(20).min() - 1.0 if len(close.dropna()) >= 5 and close.tail(20).min() > 0 else 0.0
        ma60_gap = close_today / ma60 - 1.0 if pd.notna(ma60) and ma60 > 0 else 0.0
        trend_maturity = float(np.clip(0.5 * (runup_20 / 0.25) + 0.5 * (max(ma60_gap, 0) / 0.18), 0, 1))
        overheat_score = float(np.clip(0.6 * max(runup_20, 0) / 0.35 + 0.4 * max(ma60_gap, 0) / 0.25, 0, 1))

        records.append(
            {
                "trade_date": trade_date,
                "code": str(code),
                "name": str(latest.get("name", code)),
                "sector": str(latest.get("sector", "UNKNOWN")),
                "sector_l1": str(latest.get("sector_l1", latest.get("sector_level1", latest.get("sector", "UNKNOWN")))),
                "sector_level1": str(latest.get("sector_level1", latest.get("sector_l1", latest.get("sector", "UNKNOWN")))),
                "sector_level2": str(latest.get("sector_level2", latest.get("sector", "UNKNOWN"))),
                "close": close_today,
                "r20": r20,
                "r60": r60,
                "r120": r120,
                "ma20": ma20,
                "ma60": ma60,
                "ma120": ma120,
                "avg_amount_20d": avg_amount_20d,
                "vol20": vol20,
                "max_drawdown_20d": maxdd20,
                "max_drawdown_60d": maxdd60,
                "missing_ratio_60d": missing_ratio_60d,
                "abs_trend_score": abs_trend_score,
                "trend_maturity": trend_maturity,
                "overheat_score": overheat_score,
                "data_quality_flag": data_quality_flag,
            }
        )

    out = pd.DataFrame(records)
    if out.empty:
        return out

    z20 = _safe_zscore(out["r20"])
    z60 = _safe_zscore(out["r60"])
    z120 = _safe_zscore(out["r120"])
    mw20, mw60, mw120 = config.momentum_weights
    out["momentum_score"] = mw20 * z20 + mw60 * z60 + mw120 * z120
    out["liquidity_score"] = _safe_zscore(np.log1p(out["avg_amount_20d"].fillna(0)))
    out["risk_score"] = _safe_zscore(out["vol20"].fillna(out["vol20"].median())) + _safe_zscore((-out["max_drawdown_60d"]).fillna(0))
    return out


def compute_market_state(price_df: pd.DataFrame, trade_date, config: HistoricalMLConfig) -> str:
    """Classify market state from data available by trade_date."""

    trade_date = pd.Timestamp(trade_date).normalize()
    history = price_df.loc[price_df["date"] <= trade_date].copy()
    today_features = _basic_features_for_date(price_df, trade_date, config)
    if today_features.empty:
        return "unknown"

    if config.market_index_code and config.market_index_code in set(history["code"].astype(str)):
        g = history.loc[history["code"].astype(str) == config.market_index_code].sort_values("date")
        close = g["close"]
        r60 = _last_or_nan(close, 60)
        r120 = _last_or_nan(close, 120)
        breadth = float((today_features["close"] > today_features["ma60"]).mean())
    else:
        r60 = float(today_features["r60"].mean(skipna=True))
        r120 = float(today_features["r120"].mean(skipna=True))
        breadth = float((today_features["close"] > today_features["ma60"]).mean())

    if pd.isna(r60):
        r60 = 0.0
    if pd.isna(r120):
        r120 = 0.0

    if r60 > 0 and r120 > 0 and breadth >= 0.55:
        return "offense"
    if r60 < 0 and r120 < 0 and breadth <= 0.45:
        return "defense"
    return "neutral"


def build_sector_features_for_day(etf_features: pd.DataFrame, config: HistoricalMLConfig) -> pd.DataFrame:
    if etf_features.empty:
        return pd.DataFrame()

    records = []
    for sector, g in etf_features.groupby("sector", sort=False):
        g = g.copy()
        n_top = max(1, int(np.ceil(len(g) * 0.30)))
        top = g.nlargest(n_top, "momentum_score")
        sector_momentum = float(top["momentum_score"].mean(skipna=True))
        sector_acceleration = float(g.loc[g["acceleration_score"] > 0, "acceleration_score"].mean(skipna=True))
        if pd.isna(sector_acceleration):
            sector_acceleration = 0.0
        breadth = (
            0.40 * float((g["r60"] > 0).mean())
            + 0.40 * float((g["close"] > g["ma60"]).mean())
            + 0.20 * float((g["acceleration_score"] > 0).mean())
        )
        sector_risk = float((g["risk_score"].fillna(0)).mean())
        proxy = sector_momentum + 0.3 * sector_acceleration + 0.2 * breadth - 0.1 * sector_risk
        records.append(
            {
                "trade_date": g["trade_date"].iloc[0],
                "sector": sector,
                "sector_l1": g["sector_l1"].iloc[0],
                "market_state": g["market_state"].iloc[0],
                "sector_momentum_score": sector_momentum,
                "sector_acceleration_score": sector_acceleration,
                "sector_breadth_score": breadth,
                "sector_risk_score": sector_risk,
                "sector_entry_success_proxy": proxy,
                "candidate_count": int(len(g)),
            }
        )

    sectors = pd.DataFrame(records)
    sectors["sector_score"] = (
        0.50 * _safe_zscore(sectors["sector_momentum_score"])
        + 0.20 * _safe_zscore(sectors["sector_acceleration_score"])
        + 0.20 * _safe_zscore(sectors["sector_breadth_score"])
        - 0.10 * _safe_zscore(sectors["sector_risk_score"])
    )
    sectors["sector_rank"] = sectors["sector_score"].rank(ascending=False, method="first").astype(int)
    sectors["sector_state"] = np.select(
        [sectors["sector_score"] >= 0.5, sectors["sector_score"] <= -0.5],
        ["strong", "weak"],
        default="neutral",
    )
    sectors["source"] = config.source
    return sectors.sort_values("sector_rank").reset_index(drop=True)


def build_etf_features_for_day(price_df: pd.DataFrame, trade_date, config: HistoricalMLConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build ETF and sector feature samples for trade_date with no future data."""

    return ReplayFeatureCache.build(price_df, config).for_day(trade_date)
