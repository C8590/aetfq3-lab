from __future__ import annotations

import json

import pandas as pd

from historical_ml.cli import main
from historical_ml.ml_sim_historical_review import build_ml_sim_historical_review


def test_ml_sim_historical_review_is_walk_forward_and_writes_required_outputs(tmp_path):
    labeled = _samples()
    scoring = labeled.drop(
        columns=[
            "feature_at_t",
            "label_after_t",
            "future_return_1d",
            "future_return_3d",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "future_max_gain_10d",
            "future_max_drawdown_10d",
            "outperform_market_10d",
            "outperform_sector_10d",
            "hit_stop_loss_10d",
            "auto_label",
            "label_reason_cn",
            "label_status",
        ]
    )
    prices = _prices()

    result = build_ml_sim_historical_review(
        labeled_samples=labeled,
        scoring_samples=scoring,
        prices=prices,
        out_dir=tmp_path,
        start="2024-01-02",
        recent_days=12,
        min_train_dates=2,
    )

    assert not result.review_filled.empty
    assert result.report_json["leakage_control"]["walk_forward"] is True
    assert result.report_json["leakage_control"]["formal_entry_changed"] is False
    assert result.report_json["leakage_control"]["qmt_triggered"] is False
    assert "Top200" in set(result.effectiveness_summary["segment"])
    assert (tmp_path / "ml_sim_historical_review_filled.csv").exists()
    assert (tmp_path / "ml_sim_historical_effectiveness_summary.csv").exists()
    assert (tmp_path / "ml_sim_historical_review_report.md").exists()
    assert (tmp_path / "ml_sim_historical_review_report.json").exists()
    report_json = json.loads((tmp_path / "ml_sim_historical_review_report.json").read_text(encoding="utf-8"))
    assert report_json["recommendation"] in {
        "CONTINUE_SHADOW",
        "CONTINUE_ML_SIM",
        "TIGHTEN_ML_RECOVERED_THRESHOLD",
        "ALLOW_LIMITED_ACTIVE_SIM",
        "DISABLE_ML_ACTIVE_LAYER",
    }


def test_historical_walk_forward_score_for_date_ignores_future_rows(tmp_path):
    labeled = _samples()
    scoring = labeled.drop(
        columns=[
            "feature_at_t",
            "label_after_t",
            "future_return_1d",
            "future_return_3d",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "future_max_gain_10d",
            "future_max_drawdown_10d",
            "outperform_market_10d",
            "outperform_sector_10d",
            "hit_stop_loss_10d",
            "auto_label",
            "label_reason_cn",
            "label_status",
        ]
    )
    prices = _prices()
    base = build_ml_sim_historical_review(
        labeled_samples=labeled,
        scoring_samples=scoring,
        prices=prices,
        out_dir=tmp_path / "base",
        start="2024-01-02",
        min_train_dates=2,
    ).review_filled
    first_scored_date = pd.to_datetime(base["trade_date"]).min()

    mutated_labeled = labeled.copy()
    mutated_scoring = scoring.copy()
    future_labeled = pd.to_datetime(mutated_labeled["trade_date"]) > first_scored_date
    future_scoring = pd.to_datetime(mutated_scoring["trade_date"]) > first_scored_date
    for frame, mask in ((mutated_labeled, future_labeled), (mutated_scoring, future_scoring)):
        frame.loc[mask, "momentum_score"] *= -100
        frame.loc[mask, "momentum_20"] *= -100
        frame.loc[mask, "sector_state"] = "future_mutated"
    mutated_labeled.loc[future_labeled, "auto_label"] = "bad_entry"
    mutated_labeled.loc[future_labeled, "future_return_10d"] = -0.50
    mutated_labeled.loc[future_labeled, "future_max_drawdown_10d"] = -0.50

    mutated = build_ml_sim_historical_review(
        labeled_samples=mutated_labeled,
        scoring_samples=mutated_scoring,
        prices=prices,
        out_dir=tmp_path / "mutated",
        start="2024-01-02",
        min_train_dates=2,
    ).review_filled

    cols = ["trade_date", "code", "p_good_entry", "p_bad_entry", "ml_score", "ml_sim_action", "ml_adjustment_type"]
    left = base.loc[pd.to_datetime(base["trade_date"]).eq(first_scored_date), cols].sort_values("code").reset_index(drop=True)
    right = mutated.loc[pd.to_datetime(mutated["trade_date"]).eq(first_scored_date), cols].sort_values("code").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_ml_sim_historical_review_cli_runs(tmp_path):
    labeled = _samples()
    scoring = labeled.drop(
        columns=[
            "feature_at_t",
            "label_after_t",
            "future_return_1d",
            "future_return_3d",
            "future_return_5d",
            "future_return_10d",
            "future_return_20d",
            "future_max_gain_10d",
            "future_max_drawdown_10d",
            "outperform_market_10d",
            "outperform_sector_10d",
            "hit_stop_loss_10d",
            "auto_label",
            "label_reason_cn",
            "label_status",
        ]
    )
    labeled_path = tmp_path / "labeled.csv"
    scoring_path = tmp_path / "scoring.csv"
    prices_path = tmp_path / "prices.csv"
    labeled.to_csv(labeled_path, index=False)
    scoring.to_csv(scoring_path, index=False)
    _prices().to_csv(prices_path, index=False)

    rc = main(
        [
            "ml-sim-historical-review",
            "--labeled-samples",
            str(labeled_path),
            "--scoring-samples",
            str(scoring_path),
            "--prices",
            str(prices_path),
            "--out",
            str(tmp_path / "out"),
            "--start",
            "2024-01-02",
            "--recent-days",
            "12",
            "--min-train-dates",
            "2",
        ]
    )

    assert rc == 0
    assert (tmp_path / "out" / "ml_sim_historical_review_report.md").exists()


def _samples() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=32)
    specs = [
        ("GOOD1", "TECH", 0.90, "good_entry"),
        ("GOOD2", "TECH", 0.70, "good_entry"),
        ("BAD01", "VALUE", -0.90, "bad_entry"),
        ("BAD02", "VALUE", -0.70, "bad_entry"),
        ("MID01", "CYCLE", 0.10, "neutral_entry"),
        ("MID02", "CYCLE", -0.10, "neutral_entry"),
    ]
    rows = []
    for d_idx, trade_date in enumerate(dates):
        for c_idx, (code, sector, base_momentum, label) in enumerate(specs):
            momentum = base_momentum + ((d_idx % 3) - 1) * 0.01
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "name": f"{code} ETF",
                    "sector_level1": sector,
                    "sector_level2": sector,
                    "is_valid_sample": True,
                    "exclude_reason": "",
                    "momentum_20": momentum,
                    "momentum_60": momentum * 0.8,
                    "momentum_120": momentum * 0.5,
                    "momentum_score": momentum,
                    "acceleration_score": momentum * 0.2,
                    "volatility_20": 0.02 + c_idx * 0.001,
                    "drawdown_20": -0.01 - max(-momentum, 0) * 0.02,
                    "drawdown_60": -0.02 - max(-momentum, 0) * 0.03,
                    "market_state": "offense" if d_idx % 2 else "neutral",
                    "sector_state": "strong" if momentum > 0.5 else ("weak" if momentum < -0.5 else "neutral"),
                    "sector_rank": c_idx % 3 + 1,
                    "etf_rank": c_idx % 2 + 1,
                    "pre_selected": label == "good_entry",
                    "entry_raw_action": "PROBE" if label == "good_entry" else "OBSERVE",
                    "final_action": "PROBE" if label == "good_entry" else "OBSERVE",
                    "source": "historical_replay",
                    "feature_at_t": trade_date,
                    "label_after_t": trade_date + pd.offsets.BDay(10),
                    "future_return_1d": 0.01 if label == "good_entry" else (-0.01 if label == "bad_entry" else 0.0),
                    "future_return_3d": 0.03 if label == "good_entry" else (-0.03 if label == "bad_entry" else 0.005),
                    "future_return_5d": 0.05 if label == "good_entry" else (-0.05 if label == "bad_entry" else 0.006),
                    "future_return_10d": 0.08 if label == "good_entry" else (-0.08 if label == "bad_entry" else 0.01),
                    "future_return_20d": 0.12 if label == "good_entry" else (-0.10 if label == "bad_entry" else 0.015),
                    "future_max_gain_10d": 0.10 if label == "good_entry" else 0.01,
                    "future_max_drawdown_10d": -0.02 if label == "good_entry" else (-0.08 if label == "bad_entry" else -0.025),
                    "outperform_market_10d": label == "good_entry",
                    "outperform_sector_10d": label == "good_entry",
                    "hit_stop_loss_10d": label == "bad_entry",
                    "auto_label": label,
                    "label_reason_cn": "test label",
                    "label_status": "ok",
                }
            )
    return pd.DataFrame(rows)


def _prices() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=45)
    sectors = {"GOOD1": "TECH", "GOOD2": "TECH", "BAD01": "VALUE", "BAD02": "VALUE", "MID01": "CYCLE", "MID02": "CYCLE"}
    rows = []
    for code, sector in sectors.items():
        for idx, date in enumerate(dates):
            if code.startswith("GOOD"):
                close = 100.0 + idx * 0.8
            elif code.startswith("BAD"):
                close = 100.0 - idx * 0.5
            else:
                close = 100.0 + idx * 0.05
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": f"{code} ETF",
                    "sector": sector,
                    "sector_l1": sector,
                    "sector_level1": sector,
                    "sector_level2": sector,
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": 10_000_000,
                }
            )
    return pd.DataFrame(rows)
