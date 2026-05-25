from __future__ import annotations

import json

import pandas as pd

from historical_ml.cli import main
from historical_ml.ml_sim_review import build_ml_sim_weekly_review


def test_ml_sim_weekly_review_backfills_ready_pending_and_missing_rows(tmp_path):
    prices = _prices()
    comparison = pd.DataFrame(
        [
            _row("2026-01-02", "GOOD", "OBSERVE", "PROBE", "ML_UPGRADED_TO_BUY_CANDIDATE", 90, 0.90, 0.02),
            _row("2026-01-02", "BAD", "PROBE", "OBSERVE", "ML_DOWNGRADED", -40, 0.10, 0.80),
            _row("2026-01-02", "LEGACY", "BUY", "BUY", "ML_UNCHANGED", 20, 0.45, 0.20),
            _row("2026-01-02", "NEUTRAL", "OBSERVE", "OBSERVE", "ML_UNCHANGED", 10, 0.30, 0.10),
            _row("2026-01-02", "RISKY", "OBSERVE", "PROBE", "ML_CONFLICT_WITH_RISK", 70, 0.70, 0.05),
        ]
    )
    review_queue = pd.concat(
        [
            comparison.loc[comparison["ml_adjustment_type"].ne("ML_UNCHANGED")],
            pd.DataFrame([_row("2026-01-14", "GOOD", "OBSERVE", "PROBE", "ML_UPGRADED_TO_BUY_CANDIDATE", 92, 0.92, 0.01)]),
            pd.DataFrame([_row("2026-01-02", "MISSING", "OBSERVE", "PROBE", "ML_UPGRADED_TO_BUY_CANDIDATE", 95, 0.95, 0.01)]),
        ],
        ignore_index=True,
    )

    result = build_ml_sim_weekly_review(
        review_queue=review_queue,
        comparison=comparison,
        prices=prices,
        out_dir=tmp_path,
        daily_decision_snapshot=pd.DataFrame([{"market_state": "offense"}]),
    )

    by_key = {(row["trade_date"], row["code"]): row for row in result.review_filled.to_dict(orient="records")}
    assert by_key[("2026-01-02", "000GOOD"[-6:])]["review_status"] == "READY"
    assert by_key[("2026-01-02", "000GOOD"[-6:])]["ml_adjustment_bucket"] == "ML_RECOVERED"
    assert by_key[("2026-01-02", "000GOOD"[-6:])]["auto_label"] == "good_entry"
    assert by_key[("2026-01-02", "000BAD"[-6:])]["auto_label"] == "bad_entry"
    assert "PENDING_NOT_ENOUGH_FUTURE_DATA" in set(result.review_filled["review_status"])
    assert "MISSING_PRICE" in set(result.review_filled["review_status"])
    assert result.recommendation == "ALLOW_LIMITED_ACTIVE_SIM"
    assert "ML_RECOVERED 是否有效" in result.report
    assert "ML_DOWNGRADED 是否有效" in result.report
    assert (tmp_path / "ml_sim_review_filled.csv").exists()
    assert (tmp_path / "ml_sim_weekly_review_report.md").exists()
    assert (tmp_path / "ml_sim_weekly_review_report.json").exists()
    assert (tmp_path / "ml_sim_effectiveness_summary.csv").exists()

    report_json = json.loads((tmp_path / "ml_sim_weekly_review_report.json").read_text(encoding="utf-8"))
    assert report_json["leakage_control"]["formal_entry_changed"] is False
    assert report_json["leakage_control"]["qmt_triggered"] is False


def test_ml_sim_weekly_review_cli_runs(tmp_path):
    prices = _prices()
    review = pd.DataFrame([_row("2026-01-02", "GOOD", "OBSERVE", "PROBE", "ML_UPGRADED_TO_BUY_CANDIDATE", 90, 0.90, 0.02)])
    price_path = tmp_path / "prices.csv"
    review_path = tmp_path / "review.csv"
    prices.to_csv(price_path, index=False, encoding="utf-8-sig")
    review.to_csv(review_path, index=False, encoding="utf-8-sig")

    rc = main(["ml-sim-review", "--review-queue", str(review_path), "--prices", str(price_path), "--out", str(tmp_path / "out")])

    assert rc == 0
    assert (tmp_path / "out" / "ml_sim_review_filled.csv").exists()


def _row(trade_date, code, legacy_action, ml_action, adjustment, score, p_good, p_bad):
    return {
        "trade_date": trade_date,
        "code": code,
        "name": f"{code} ETF",
        "sector_level1": "TECH" if code in {"GOOD", "RISKY"} else "VALUE",
        "sector_level2": "TECH" if code in {"GOOD", "RISKY"} else "VALUE",
        "legacy_action": legacy_action,
        "ml_sim_action": ml_action,
        "final_action": legacy_action,
        "ml_score": score,
        "p_good_entry": p_good,
        "p_bad_entry": p_bad,
        "ml_adjustment_type": adjustment,
        "review_priority": "P1",
    }


def _prices():
    dates = pd.bdate_range("2026-01-02", periods=15)
    targets = {
        "GOOD": 110.0,
        "BAD": 92.0,
        "LEGACY": 103.0,
        "NEUTRAL": 101.0,
        "RISKY": 100.0,
    }
    rows = []
    for code, target in targets.items():
        sector = "TECH" if code in {"GOOD", "RISKY"} else "VALUE"
        for idx, date in enumerate(dates):
            close = 100.0 + (target - 100.0) * min(idx, 10) / 10
            low = close * 0.995
            if code == "BAD" and 1 <= idx <= 10:
                low = min(low, 93.0)
            rows.append(
                {
                    "date": date,
                    "code": code,
                    "name": f"{code} ETF",
                    "sector": sector,
                    "sector_l1": sector,
                    "sector_level2": sector,
                    "open": close,
                    "high": close * 1.005,
                    "low": low,
                    "close": close,
                    "volume": 1_000_000,
                    "amount": 10_000_000,
                }
            )
    return pd.DataFrame(rows)
