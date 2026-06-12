from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tools.lab.intraday_long_history_alpha_risk_overlay_optimizer import (
    LAB_CANDIDATE_STATUS,
    PROMISING_STATUS,
    REJECTED_STATUS,
    RiskOverlayConfig,
    apply_drawdown_throttle,
    apply_exposure_and_returns,
    apply_top_k_and_filters,
    contribution_table,
    first_path_exit_return,
    gate_candidate_v1,
    resolve_out_dir,
    run_risk_overlay_optimizer,
    select_top_rejected_candidates,
    train_only_regime_thresholds,
)
from tools.lab.intraday_long_history_data_lake import TIME_CENSORED_FEATURES


def business_dates(start: date, count: int) -> list[str]:
    output: list[str] = []
    current = start
    while len(output) < count:
        if current.weekday() < 5:
            output.append(current.isoformat())
        current += timedelta(days=1)
    return output


def base_overlay(**overrides: object) -> dict[str, object]:
    overlay: dict[str, object] = {
        "top_k_per_day": 1,
        "min_probability": 0.50,
        "max_daily_sleeve_exposure": 0.25,
        "max_total_exposure": 0.75,
        "max_etf_weight_per_sleeve": 0.25,
        "holding_period": "1d",
        "stop_loss": None,
        "take_profit": None,
        "volatility_filter": "none",
        "liquidity_filter": "none",
        "drawdown_throttle": None,
        "threshold_search_lab_only": True,
        "diagnostic_only": True,
    }
    overlay.update(overrides)
    return overlay


def scored_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "datetime": pd.Timestamp("2025-01-02 10:00"), "etf_code": "510300", "signal_clock": "10:00", "score": 0.60, "future_return_1d_overlay": 0.01, "future_return_3d": 0.01, "entry_price": 1.0},
            {"trade_date": "2025-01-02", "datetime": pd.Timestamp("2025-01-02 10:00"), "etf_code": "510500", "signal_clock": "10:00", "score": 0.90, "future_return_1d_overlay": -0.10, "future_return_3d": -0.10, "entry_price": 1.0},
            {"trade_date": "2025-01-02", "datetime": pd.Timestamp("2025-01-02 10:00"), "etf_code": "159915", "signal_clock": "10:00", "score": 0.70, "future_return_1d_overlay": 0.20, "future_return_3d": 0.20, "entry_price": 1.0},
        ]
    )


def path_bars() -> pd.DataFrame:
    rows = []
    for trade_date, closes in {
        "2025-01-02": [1.00],
        "2025-01-03": [0.99, 0.974, 1.03, 1.06],
    }.items():
        for idx, close in enumerate(closes):
            rows.append(
                {
                    "trade_date": trade_date,
                    "datetime": pd.Timestamp(f"{trade_date} 10:{idx * 5:02d}"),
                    "etf_code": "510300",
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 100,
                    "amount": close * 100,
                    "vwap": close,
                    "frequency": "5m",
                    "source_file": "test.csv",
                }
            )
    return pd.DataFrame(rows)


def test_top_rejected_candidates_selected_correctly() -> None:
    leaderboard = pd.DataFrame(
        [
            {"candidate_id": "10:00|core_intraday_price|safe_positive_3d|3d|rule|exit|10bps|dummy_stratified", "candidate_status": REJECTED_STATUS, "net_total_return": 3.0, "calmar_like_ratio": 1.0, "profit_factor": 1.1},
            {"candidate_id": "10:30|core_intraday_price|safe_positive_3d|3d|rule|exit|10bps|dummy_stratified", "candidate_status": REJECTED_STATUS, "net_total_return": 1.0, "calmar_like_ratio": 5.0, "profit_factor": 1.2},
            {"candidate_id": "11:00|core_intraday_price|safe_positive_3d|3d|rule|exit|10bps|dummy_stratified", "candidate_status": LAB_CANDIDATE_STATUS, "net_total_return": 9.0, "calmar_like_ratio": 9.0, "profit_factor": 9.0},
        ]
    )

    selected = select_top_rejected_candidates(leaderboard, top_n=1)

    assert [row["signal_clock"] for row in selected] == ["10:00", "10:30"]


def test_drawdown_attribution_by_month() -> None:
    trades = pd.DataFrame(
        [
            {"month": "2025-01", "net_return": -0.10},
            {"month": "2025-01", "net_return": 0.02},
            {"month": "2025-02", "net_return": -0.03},
        ]
    )

    table = contribution_table(trades, "month")

    assert table.iloc[0]["month"] == "2025-01"
    assert table.iloc[0]["loss_sum"] == pytest.approx(-0.10)


def test_drawdown_attribution_by_etf() -> None:
    trades = pd.DataFrame(
        [
            {"etf_code": "510300", "net_return": -0.08},
            {"etf_code": "510500", "net_return": 0.03},
        ]
    )

    table = contribution_table(trades, "etf_code")

    assert table.iloc[0]["etf_code"] == "510300"


def test_top_k_uses_probability_only_not_future_returns() -> None:
    selected = apply_top_k_and_filters(scored_rows(), base_overlay(top_k_per_day=1))

    assert selected.iloc[0]["etf_code"] == "510500"
    assert selected.iloc[0]["future_return_1d_overlay"] == pytest.approx(-0.10)


def test_min_probability_filter_works_and_is_lab_only() -> None:
    overlay = base_overlay(min_probability=0.75)
    selected = apply_top_k_and_filters(scored_rows(), overlay)

    assert list(selected["etf_code"]) == ["510500"]
    assert overlay["threshold_search_lab_only"] is True


def test_exposure_cap_leq_100_percent() -> None:
    overlay = base_overlay(top_k_per_day="all", max_daily_sleeve_exposure=0.33, max_total_exposure=0.75, max_etf_weight_per_sleeve=0.50)
    trades = apply_exposure_and_returns(apply_top_k_and_filters(scored_rows(), overlay), pd.DataFrame(), overlay, cost_bps=10)

    assert trades.groupby("trade_date")["sleeve_weight"].sum().max() <= 1.0
    assert trades.groupby("trade_date")["sleeve_weight"].sum().max() <= 0.33


def test_stop_loss_triggers_first_future_5m_crossing_after_entry() -> None:
    ret, reason, exit_time = first_path_exit_return(
        path_bars(),
        etf_code="510300",
        entry_time=pd.Timestamp("2025-01-02 10:00"),
        entry_price=1.0,
        holding_days=1,
        stop_loss=-0.015,
        take_profit=None,
    )

    assert reason == "stop_loss"
    assert ret == pytest.approx(-0.026)
    assert "10:05" in exit_time


def test_take_profit_triggers_first_future_5m_crossing_after_entry() -> None:
    ret, reason, exit_time = first_path_exit_return(
        path_bars(),
        etf_code="510300",
        entry_time=pd.Timestamp("2025-01-02 10:00"),
        entry_price=1.0,
        holding_days=1,
        stop_loss=None,
        take_profit=0.02,
    )

    assert reason == "take_profit"
    assert ret == pytest.approx(0.03)
    assert "10:10" in exit_time


def test_trailing_volatility_filter_uses_train_past_only() -> None:
    daily = pd.DataFrame(
        {
            "etf_code": ["510300"] * 30,
            "trade_date": business_dates(date(2025, 1, 2), 30),
            "daily_close": [1 + idx * 0.01 for idx in range(30)],
            "daily_amount": [1000] * 30,
            "daily_return": [0.01] * 30,
        }
    )
    thresholds = train_only_regime_thresholds(daily, daily["trade_date"].tolist()[:25])

    assert thresholds["vol_q75"] == pytest.approx(0.0)


def test_drawdown_throttle_uses_past_nav_only() -> None:
    trades = pd.DataFrame(
        [
            {"trade_date": "2025-01-02", "net_return": -0.06},
            {"trade_date": "2025-01-03", "net_return": 0.10},
        ]
    )

    throttled = apply_drawdown_throttle(trades, -0.05)

    assert bool(throttled.iloc[0]["drawdown_throttle_applied"]) is False
    assert bool(throttled.iloc[1]["drawdown_throttle_applied"]) is True
    assert throttled.iloc[1]["net_return"] == pytest.approx(0.05)


def test_cost_sensitivity_reduces_return() -> None:
    overlay = base_overlay()
    trades_low = apply_exposure_and_returns(scored_rows().head(1), pd.DataFrame(), overlay, cost_bps=0)
    trades_high = apply_exposure_and_returns(scored_rows().head(1), pd.DataFrame(), overlay, cost_bps=10)

    assert trades_high.iloc[0]["net_return"] < trades_low.iloc[0]["net_return"]


def test_candidate_gate_passes_and_fails_correctly() -> None:
    good = {
        "net_total_return": 0.10,
        "max_drawdown": -0.20,
        "win_rate": 0.55,
        "profit_factor": 1.20,
        "monthly_win_rate": 0.60,
        "month_concentration": 0.40,
        "etf_concentration": 0.40,
    }
    bad = {**good, "max_drawdown": -0.40}

    assert gate_candidate_v1(good, v0_top_drawdown=-0.339, leakage_ok=True, artifact_saved=False)[0] is True
    assert gate_candidate_v1(bad, v0_top_drawdown=-0.339, leakage_ok=True, artifact_saved=False)[0] is False


def test_output_path_outside_local_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_out_dir(tmp_path)


def write_minimal_data_lake(data_lake: Path, v0_dir: Path) -> None:
    data_lake.mkdir(parents=True)
    v0_dir.mkdir(parents=True)
    dates = business_dates(date(2025, 1, 2), 96)
    etfs = ["510300", "510500", "159915"]
    feature_rows = []
    bar_rows = []
    for day_idx, trade_date in enumerate(dates):
        for etf_idx, code in enumerate(etfs):
            signal = 1 if etf_idx == day_idx % len(etfs) else 0
            close = 1.0 + day_idx * 0.002 + etf_idx * 0.01
            for minute, intraday_close in [("09:30", close), ("10:00", close + signal * 0.01), ("14:55", close + signal * 0.02)]:
                bar_rows.append(
                    {
                        "trade_date": trade_date,
                        "datetime": f"{trade_date} {minute}:00",
                        "etf_code": code,
                        "open": close,
                        "high": intraday_close + 0.001,
                        "low": intraday_close - 0.001,
                        "close": intraday_close,
                        "volume": 1000,
                        "amount": intraday_close * 1000,
                        "vwap": intraday_close,
                        "frequency": "5m",
                        "source_file": "test.csv",
                    }
                )
            row = {
                "trade_date": trade_date,
                "datetime": f"{trade_date}T10:00:00",
                "signal_clock": "10:00",
                "etf_code": code,
                "future_return_1d": 0.01 if signal else -0.004,
                "future_return_3d": 0.03 if signal else -0.01,
                "max_drawdown_3d": -0.01 if signal else -0.03,
                "ret_3d_gt_100bp": signal,
                "safe_positive_3d": signal,
            }
            for feature in TIME_CENSORED_FEATURES:
                row[feature] = float(signal) + day_idx * 0.001
            feature_rows.append(row)
    pd.DataFrame(feature_rows).to_csv(data_lake / "long_history_feature_rows.csv", index=False)
    pd.DataFrame(bar_rows).to_csv(data_lake / "long_history_bars.csv", index=False)
    pd.DataFrame(
        [
            {
                "candidate_id": "10:00|core_intraday_price|safe_positive_3d|3d|top1|exit|10bps|dummy_stratified",
                "candidate_status": REJECTED_STATUS,
                "signal_clock": "10:00",
                "feature_set": "core_intraday_price",
                "label_policy": "safe_positive_3d",
                "model_family": "dummy_stratified",
                "net_total_return": 1.0,
                "calmar_like_ratio": 1.0,
                "profit_factor": 1.2,
                "max_drawdown": -0.34,
                "win_rate": 0.6,
                "monthly_win_rate": 0.5,
            }
        ]
    ).to_csv(v0_dir / "long_history_candidate_leaderboard.csv", index=False)


def test_no_order_intent_no_artifacts_and_boundary_fields_false(tmp_path: Path) -> None:
    data_lake = tmp_path / "lake"
    v0_dir = tmp_path / "v0"
    out_dir = tmp_path / "out"
    write_minimal_data_lake(data_lake, v0_dir)

    report = run_risk_overlay_optimizer(
        RiskOverlayConfig(data_lake=data_lake, v0_dir=v0_dir, out_dir=out_dir, min_train_anchors=20, min_validation_anchors=5, min_validation_groups=5),
        repo_root=tmp_path,
        enforce_paths=False,
    )

    assert report["access_mode"] == "READ_ONLY"
    assert report["final_action_change_allowed"] is False
    assert report["contains_live_order"] is False
    assert report["contains_secret"] is False
    assert report["stable_promotion_ready"] is False
    assert report["order_intent_generated"] is False
    assert report["qmt_ready"] is False
    assert report["formal_training"] is False
    assert report["model_saved"] is False
    assert report["scaler_saved"] is False
    assert report["checkpoint_saved"] is False
    assert not list(out_dir.rglob("*OrderIntent*"))
    assert not list(out_dir.rglob("*order_intent*"))
    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))
    assert (out_dir / "risk_overlay_decision.json").exists()
