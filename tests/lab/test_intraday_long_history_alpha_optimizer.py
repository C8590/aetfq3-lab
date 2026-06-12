from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from tools.lab.intraday_long_history_alpha_optimizer import (
    OptimizerConfig,
    decision_from_results,
    gate_candidate,
    make_rolling_folds,
    max_drawdown,
    pnl_metrics,
    resolve_out_dir,
    run_optimizer,
    select_trades,
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


def synthetic_feature_frame() -> pd.DataFrame:
    dates = business_dates(date(2025, 1, 2), 96)
    etfs = ["510300", "510500", "159915"]
    rows = []
    for day_index, trade_date in enumerate(dates):
        for etf_index, code in enumerate(etfs):
            signal = 1 if etf_index == day_index % len(etfs) else 0
            future_return = 0.03 if signal else -0.01
            row = {
                "trade_date": trade_date,
                "datetime": f"{trade_date}T10:00:00",
                "signal_clock": "10:00",
                "etf_code": code,
                "future_return_3d": future_return,
                "max_drawdown_3d": -0.005 if signal else -0.03,
                "ret_3d_gt_100bp": signal,
                "safe_positive_3d": signal,
            }
            for feature in TIME_CENSORED_FEATURES:
                row[feature] = float(signal) + day_index * 0.001
            rows.append(row)
    return pd.DataFrame(rows)


def test_rolling_split_no_overlap() -> None:
    folds = make_rolling_folds(synthetic_feature_frame(), min_train_anchors=20, min_validation_anchors=5, min_validation_groups=5, embargo_days=3)
    usable = [fold for fold in folds if not fold["skipped"]]

    assert usable
    assert all(fold["train_validation_no_overlap"] for fold in usable)
    assert all(not (set(fold["train_anchor_dates"]) & set(fold["validation_anchor_dates"])) for fold in usable)


def test_embargo_works() -> None:
    folds = make_rolling_folds(synthetic_feature_frame(), min_train_anchors=20, min_validation_anchors=5, min_validation_groups=5, embargo_days=7)
    usable = [fold for fold in folds if not fold["skipped"]]

    assert usable
    for fold in usable:
        assert max(fold["train_anchor_dates"]) <= fold["train_cutoff"]
        assert fold["embargo_ok"] is True


def test_pnl_cost_reduces_gross_return() -> None:
    scored = pd.DataFrame(
        [
            {"trade_date": "2025-02-03", "etf_code": "510300", "score": 0.8, "future_return_3d": 0.02},
            {"trade_date": "2025-02-03", "etf_code": "510500", "score": 0.7, "future_return_3d": 0.01},
        ]
    )

    trades = select_trades(scored, cost_bps=10)

    assert len(trades) == 1
    assert trades.iloc[0]["net_return"] == pytest.approx(trades.iloc[0]["gross_return"] - 0.002)


def test_drawdown_calculation() -> None:
    assert max_drawdown([0.1, -0.2, 0.05]) == pytest.approx(-0.2)


def test_candidate_gate() -> None:
    metrics = {
        "net_total_return": 0.10,
        "max_drawdown": -0.05,
        "calmar_like_ratio": 2.0,
        "win_rate": 0.60,
        "profit_factor": 1.20,
        "monthly_win_rate": 0.60,
        "month_concentration": 0.30,
        "etf_concentration": 0.40,
    }

    passed, reasons = gate_candidate(metrics, leakage_ok=True, artifact_saved=False)

    assert passed is True
    assert reasons == []


def test_output_path_outside_local_research_outputs_rejected(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="out-dir must be under"):
        resolve_out_dir(tmp_path)


def test_no_model_or_scaler_artifacts_and_boundary_fields_false(tmp_path: Path) -> None:
    data_lake = tmp_path / "lake"
    out_dir = tmp_path / "out"
    data_lake.mkdir()
    synthetic_feature_frame().to_csv(data_lake / "long_history_feature_rows.csv", index=False)

    report = run_optimizer(
        OptimizerConfig(data_lake=data_lake, out_dir=out_dir, min_train_anchors=20, min_validation_anchors=5, min_validation_groups=5, max_candidates=2),
        repo_root=tmp_path,
        enforce_paths=False,
    )

    assert report["access_mode"] == "READ_ONLY"
    assert report["final_action_change_allowed"] is False
    assert report["contains_live_order"] is False
    assert report["contains_secret"] is False
    assert report["stable_promotion_ready"] is False
    assert report["order_intent_ready"] is False
    assert report["qmt_ready"] is False
    assert report["formal_training"] is False
    assert report["model_saved"] is False
    assert report["scaler_saved"] is False
    assert report["checkpoint_saved"] is False
    assert not list(out_dir.rglob("*.pkl"))
    assert not list(out_dir.rglob("*.joblib"))
    assert not list(out_dir.rglob("*.pt"))


def test_decision_never_marks_stable_candidate() -> None:
    result = {
        "candidate_gate_passed": True,
        "candidate_status": "LAB_DIAGNOSTIC_ALPHA_CANDIDATE_REVIEW_REQUIRED",
        "pnl_metrics": {},
    }
    fold = {"train_validation_no_overlap": True, "embargo_ok": True}

    assert decision_from_results([result], [fold], []) == "LONG_HISTORY_ALPHA_OPTIMIZATION_COMPLETED_CANDIDATES_FOUND_REVIEW_REQUIRED"
    assert "STABLE" not in result["candidate_status"]


def test_pnl_metrics_include_required_fields() -> None:
    trades = pd.DataFrame(
        [
            {"trade_date": "2025-02-03", "month": "2025-02", "etf_code": "510300", "gross_return": 0.02, "net_return": 0.018, "cost_paid": 0.002},
            {"trade_date": "2025-02-04", "month": "2025-02", "etf_code": "510500", "gross_return": -0.01, "net_return": -0.012, "cost_paid": 0.002},
        ]
    )
    validation = pd.DataFrame(
        [
            {"trade_date": "2025-02-03", "future_return_3d": 0.01},
            {"trade_date": "2025-02-04", "future_return_3d": 0.0},
        ]
    )

    metrics = pnl_metrics(trades, validation, cost_bps=10)

    assert "net_total_return" in metrics
    assert "max_drawdown" in metrics
    assert "win_rate" in metrics
    assert metrics["cost_paid"] == pytest.approx(0.004)
