import pandas as pd
import json

from historical_ml.config import HistoricalMLConfig
from historical_ml.labeler import MLEntryUniverseLabeler
from historical_ml.schemas import ML_ENTRY_LABEL_COLUMNS
from historical_ml.validators import assert_required_columns


def test_universe_labeler_produces_non_empty_label_distribution_and_reasons(tmp_path):
    prices = _label_price_data()
    samples = _daily_ml_universe_samples()
    config = HistoricalMLConfig(output_format="csv", label_horizons=(5, 10, 20))

    labeler = MLEntryUniverseLabeler(prices, config=config, label_horizons=(5, 10, 20))
    labeled = labeler.attach_labels(samples)
    labeler.write_outputs(labeled, tmp_path)

    assert_required_columns(labeled, ML_ENTRY_LABEL_COLUMNS, "ml_entry_labeled_samples")
    counts = labeled["auto_label"].value_counts().to_dict()
    assert counts["good_entry"] >= 1
    assert counts["bad_entry"] >= 1
    assert counts["neutral_entry"] >= 1
    assert labeled["label_reason_cn"].fillna("").str.len().min() > 0
    assert labeled["label_reason_cn"].map(_has_cjk).all()
    assert set(labeled["source"]) == {"historical_replay"}
    assert (tmp_path / "generated" / "ml_entry_labeled_samples.csv").exists()
    summary = json.loads((tmp_path / "generated" / "ml_entry_label_summary.json").read_text(encoding="utf-8"))
    assert summary["good_entry_count"] >= 1
    assert summary["bad_entry_count"] >= 1
    assert summary["neutral_entry_count"] >= 1


def test_universe_labeler_keeps_feature_at_t_fields_stable_when_future_prices_change():
    prices = _label_price_data()
    samples = _daily_ml_universe_samples()
    config = HistoricalMLConfig(label_horizons=(5, 10, 20))

    base = MLEntryUniverseLabeler(prices, config=config).attach_labels(samples)
    mutated = prices.copy()
    future_mask = pd.to_datetime(mutated["date"]) > pd.Timestamp("2024-01-02")
    mutated.loc[future_mask & (mutated["code"] == "GOOD"), ["close", "high", "low"]] *= 0.5
    changed = MLEntryUniverseLabeler(mutated, config=config).attach_labels(samples)

    feature_cols = [
        "trade_date",
        "code",
        "momentum_20",
        "momentum_60",
        "momentum_120",
        "momentum_score",
        "acceleration_score",
        "volatility_20",
        "drawdown_20",
        "drawdown_60",
        "pre_selected",
        "entry_raw_action",
        "final_action",
        "source",
    ]
    pd.testing.assert_frame_equal(base[feature_cols], changed[feature_cols])
    assert base.loc[base["code"] == "GOOD", "future_return_10d"].iloc[0] != changed.loc[changed["code"] == "GOOD", "future_return_10d"].iloc[0]
    assert "future_return_10d" not in samples.columns
    assert "auto_label" not in samples.columns


def test_universe_labeler_supports_horizon_configuration():
    prices = _label_price_data()
    samples = _daily_ml_universe_samples()
    labeled = MLEntryUniverseLabeler(
        prices,
        config=HistoricalMLConfig(label_horizons=(5, 10, 20)),
        label_horizons=(5, 10, 20),
    ).attach_labels(samples)

    assert labeled["future_return_5d"].notna().any()
    assert labeled["future_return_10d"].notna().any()
    assert labeled["future_return_20d"].notna().any()


def _daily_ml_universe_samples() -> pd.DataFrame:
    rows = []
    for code, sector in [
        ("GOOD", "TECH"),
        ("PEER", "TECH"),
        ("BAD", "CYCLICAL"),
        ("NEUTRAL", "UTILITY"),
    ]:
        rows.append(
            {
                "trade_date": pd.Timestamp("2024-01-02"),
                "code": code,
                "name": f"{code} ETF",
                "sector_level1": sector,
                "sector_level2": sector,
                "is_valid_sample": True,
                "exclude_reason": "",
                "momentum_20": 0.01,
                "momentum_60": 0.02,
                "momentum_120": 0.03,
                "momentum_score": 0.1,
                "acceleration_score": 0.01,
                "volatility_20": 0.02,
                "drawdown_20": -0.01,
                "drawdown_60": -0.02,
                "market_state": "offense",
                "sector_state": "strong",
                "sector_rank": 1,
                "etf_rank": 1,
                "pre_selected": code == "GOOD",
                "entry_raw_action": "BUY" if code == "GOOD" else "OBSERVE",
                "final_action": "BUY" if code == "GOOD" else "OBSERVE",
                "source": "historical_replay",
            }
        )
    return pd.DataFrame(rows)


def _has_cjk(value: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(value))


def _label_price_data() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=30)
    sector = {
        "GOOD": "TECH",
        "PEER": "TECH",
        "BAD": "CYCLICAL",
        "NEUTRAL": "UTILITY",
    }
    target_10d = {
        "GOOD": 112.0,
        "PEER": 100.0,
        "BAD": 90.0,
        "NEUTRAL": 101.0,
    }
    target_20d = {
        "GOOD": 120.0,
        "PEER": 101.0,
        "BAD": 92.0,
        "NEUTRAL": 102.0,
    }
    rows = []
    for code in sector:
        for i, date in enumerate(dates):
            if i <= 10:
                close = 100.0 + (target_10d[code] - 100.0) * i / 10
            else:
                close = target_10d[code] + (target_20d[code] - target_10d[code]) * min(i - 10, 10) / 10
            low = close * 0.995
            if code == "BAD" and 1 <= i <= 10:
                low = min(low, 94.0)
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": f"{code} ETF",
                    "sector": sector[code],
                    "sector_l1": sector[code],
                    "open": close,
                    "high": close * 1.005,
                    "low": low,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": 10_000_000,
                }
            )
    return pd.DataFrame(rows)
