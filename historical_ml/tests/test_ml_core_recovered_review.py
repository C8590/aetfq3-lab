from __future__ import annotations

import json

import pandas as pd

from historical_ml.cli import main
from historical_ml.ml_core_recovered_review import build_ml_core_recovered_review


def test_core_recovered_review_writes_manual_and_core_outputs(tmp_path):
    frame = _review_frame(core_good=True)

    result = build_ml_core_recovered_review(frame, out_dir=tmp_path)

    assert result.recommendation == "ALLOW_LIMITED_ACTIVE_SIM"
    assert not result.manual_review.empty
    assert not result.grid.empty
    assert result.report_json["hard_gate"]["formal_entry_changed"] is False
    assert result.report_json["hard_gate"]["final_buy_action_changed"] is False
    assert result.report_json["hard_gate"]["qmt_triggered"] is False
    assert result.report_json["best_core_condition"]["sample_count"] >= 100
    assert result.report_json["best_core_condition"]["good_entry_rate"] > result.report_json["legacy_v21_buy_probe"]["good_entry_rate"]
    assert result.report_json["best_core_condition"]["bad_entry_rate"] < result.report_json["legacy_v21_buy_probe"]["bad_entry_rate"]
    assert (tmp_path / "ml_strong_recovered_manual_review.csv").exists()
    assert (tmp_path / "ml_strong_recovered_manual_review.md").exists()
    assert (tmp_path / "ml_strong_recovered_manual_review.json").exists()
    assert (tmp_path / "ml_core_recovered_threshold_grid.csv").exists()
    assert (tmp_path / "ml_core_recovered_recommendation.md").exists()
    assert (tmp_path / "ml_core_recovered_recommendation.json").exists()


def test_core_recovered_review_continues_shadow_without_quality_lift(tmp_path):
    frame = _review_frame(core_good=False)

    result = build_ml_core_recovered_review(frame, out_dir=tmp_path)

    assert result.recommendation == "CONTINUE_SHADOW"
    assert result.report_json["core_candidate_counts"]["quality_and_sample_floor"] == 0


def test_core_recovered_cli_runs(tmp_path):
    input_path = tmp_path / "historical_review.csv"
    _review_frame(core_good=True).to_csv(input_path, index=False)

    rc = main(["ml-core-recovered-review", "--historical-review", str(input_path), "--out", str(tmp_path / "out")])

    assert rc == 0
    report = json.loads((tmp_path / "out" / "ml_core_recovered_recommendation.json").read_text(encoding="utf-8"))
    assert report["mode"] == "ML_CORE_RECOVERED_OFFLINE_RECOMMENDATION"


def _review_frame(*, core_good: bool) -> pd.DataFrame:
    dates = pd.bdate_range("2026-01-02", periods=40)
    rows = []
    for d_idx, date in enumerate(dates):
        trade_date = date.strftime("%Y-%m-%d")
        for i in range(120):
            is_recovered = i < 100
            is_core = i < 3
            is_legacy = 100 <= i < 110
            if is_legacy:
                label = "good_entry" if i in {100, 101} else ("bad_entry" if i in {102, 103, 104, 105} else "neutral_entry")
                ret = 0.01 if label == "good_entry" else (-0.03 if label == "bad_entry" else 0.0)
            elif is_core:
                label = "good_entry" if core_good else "bad_entry"
                ret = 0.06 if core_good else -0.03
            elif is_recovered:
                label = "bad_entry" if i % 4 == 0 else "neutral_entry"
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
                    "sector_level1": "TECH" if i % 2 == 0 else "CYCLICAL",
                    "sector_level2": "TECH",
                    "market_state": "neutral",
                    "sector_state": "strong" if i < 80 else "weak",
                    "legacy_action": "PROBE" if is_legacy else "OBSERVE",
                    "ml_sim_action": "PROBE" if is_recovered else ("PROBE" if is_legacy else "OBSERVE"),
                    "final_action": "PROBE" if is_legacy else "OBSERVE",
                    "ml_score": 100 - i,
                    "p_good_entry": p_good,
                    "p_bad_entry": p_bad,
                    "ml_adjustment_type": "ML_RECOVERED" if is_recovered else "ML_UNCHANGED",
                    "ml_adjustment_bucket": "ML_RECOVERED" if is_recovered else "ML_UNCHANGED",
                    "ml_rank_global": i + 1,
                    "ml_rank_sector": (i % 5) + 1,
                    "future_return_1d": ret * 0.2,
                    "future_return_3d": ret * 0.4,
                    "future_return_5d": ret * 0.7,
                    "future_return_10d": ret,
                    "future_max_drawdown_10d": -0.01 if is_core else -0.04,
                    "outperform_market_10d": label == "good_entry",
                    "outperform_sector_10d": label == "good_entry",
                    "review_status": "READY",
                    "auto_label": label,
                    "is_valid_sample": True,
                    "exclude_reason": "",
                    "expected_drawdown_10d": -0.005 if is_core else -0.04,
                    "momentum_score": 2.0 - i * 0.01,
                    "acceleration_score": 2.0 - i * 0.01,
                    "volatility_20": 0.02 + i * 0.0001,
                    "etf_rank": i + 1,
                    "pre_selected": is_legacy,
                }
            )
    return pd.DataFrame(rows)
