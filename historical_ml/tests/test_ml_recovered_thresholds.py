from __future__ import annotations

import json

from historical_ml.cli import main
from historical_ml.ml_recovered_thresholds import build_ml_recovered_threshold_recommendation

import pandas as pd


def test_recovered_threshold_grid_writes_outputs_and_splits_pool(tmp_path):
    frame = _review_frame(strong_good=True)

    result = build_ml_recovered_threshold_recommendation(frame, out_dir=tmp_path)

    assert not result.grid.empty
    assert {"ML_STRONG_RECOVERED", "ML_WEAK_RECOVERED", "ML_NOISE_RECOVERED"} == set(
        result.report_json["recovered_classification_counts"]
    )
    assert result.report_json["hard_gate"]["formal_entry_changed"] is False
    assert result.report_json["hard_gate"]["final_buy_action_changed"] is False
    assert result.report_json["hard_gate"]["qmt_triggered"] is False
    assert result.report_json["best_condition"]["good_entry_rate"] > result.report_json["legacy_v21_buy_probe"]["good_entry_rate"]
    assert (tmp_path / "ml_recovered_threshold_grid.csv").exists()
    assert (tmp_path / "ml_recovered_threshold_recommendation.md").exists()
    assert (tmp_path / "ml_recovered_threshold_recommendation.json").exists()


def test_recovered_threshold_does_not_allow_active_when_good_rate_lags_legacy(tmp_path):
    frame = _review_frame(strong_good=False)

    result = build_ml_recovered_threshold_recommendation(frame, out_dir=tmp_path)

    assert result.recommendation in {"USE_ML_AS_RANKING_ONLY", "CONTINUE_SHADOW"}
    assert result.recommendation != "ALLOW_LIMITED_ACTIVE_SIM"
    assert result.report_json["best_condition"]["good_entry_rate"] <= result.report_json["legacy_v21_buy_probe"]["good_entry_rate"]


def test_recovered_threshold_cli_runs(tmp_path):
    input_path = tmp_path / "historical_review.csv"
    _review_frame(strong_good=True).to_csv(input_path, index=False)

    rc = main(["ml-recovered-thresholds", "--historical-review", str(input_path), "--out", str(tmp_path / "out")])

    assert rc == 0
    report = json.loads((tmp_path / "out" / "ml_recovered_threshold_recommendation.json").read_text(encoding="utf-8"))
    assert report["mode"] == "V2.1_ML_RECOVERED_THRESHOLD_OFFLINE_RECOMMENDATION"


def _review_frame(*, strong_good: bool) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=6)
    rows = []
    for d_idx, date in enumerate(dates):
        trade_date = date.strftime("%Y-%m-%d")
        for i in range(100):
            is_recovered = i < 80
            is_strong = i < 5
            is_legacy = 80 <= i < 90
            if is_legacy:
                label = "good_entry" if i in {80, 81, 82} else ("bad_entry" if i in {83, 84, 85} else "neutral_entry")
                ret = 0.01 if label == "good_entry" else (-0.02 if label == "bad_entry" else 0.0)
            elif is_strong:
                label = "good_entry" if strong_good else "bad_entry"
                ret = 0.05 if strong_good else 0.03
            elif is_recovered:
                label = "bad_entry" if i % 3 == 0 else "neutral_entry"
                ret = -0.01 if label == "bad_entry" else 0.002
            else:
                label = "neutral_entry"
                ret = 0.0
            p_good = 0.99 - i * 0.002 if is_recovered else 0.20
            p_bad = 0.001 + i * 0.001 if is_recovered else 0.50
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": f"{d_idx:02d}{i:04d}",
                    "name": f"ETF {i}",
                    "sector_level1": "TECH",
                    "sector_level2": "TECH",
                    "market_state": "neutral",
                    "sector_state": "strong" if i < 60 else "weak",
                    "legacy_action": "PROBE" if is_legacy else "OBSERVE",
                    "ml_sim_action": "PROBE" if is_recovered else ("PROBE" if is_legacy else "OBSERVE"),
                    "final_action": "PROBE" if is_legacy else "OBSERVE",
                    "ml_score": 100 - i,
                    "p_good_entry": p_good,
                    "p_bad_entry": p_bad,
                    "ml_adjustment_type": "ML_RECOVERED" if is_recovered else "ML_UNCHANGED",
                    "ml_adjustment_bucket": "ML_RECOVERED" if is_recovered else "ML_UNCHANGED",
                    "ml_rank_global": i + 1,
                    "ml_rank_sector": i + 1,
                    "future_return_3d": ret * 0.4,
                    "future_return_5d": ret * 0.7,
                    "future_return_10d": ret,
                    "future_max_drawdown_10d": -0.01 if is_strong else -0.04,
                    "outperform_market_10d": label == "good_entry",
                    "outperform_sector_10d": label == "good_entry",
                    "review_status": "READY",
                    "auto_label": label,
                    "is_valid_sample": True,
                    "exclude_reason": "",
                    "expected_drawdown_10d": -0.005 if is_strong else -0.04,
                    "momentum_score": 1.0 - i * 0.01,
                    "pre_selected": is_legacy,
                }
            )
    return pd.DataFrame(rows)
