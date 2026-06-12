from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from tools.lab.intraday_lab_monitor_candidate_paper_pnl_replay import (
    BOUNDARY_FIELDS,
    DECISION_MISSING_PREDICTIONS,
    ETF_UNIVERSE,
    FOCUS_CANDIDATE_ID,
    FOCUS_MODEL,
    PaperPnlReplayError,
    ReplayConfig,
    run_replay,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_ARTIFACT = REPO_ROOT / ".local_artifact_backup/pytest_paper_pnl"
LOCAL_RESEARCH_FIXTURE = REPO_ROOT / ".local_research_outputs/aetfq3_lab/pytest_paper_pnl"
LOCAL_REPORT = REPO_ROOT / ".local_research_outputs/aetfq3_lab/intraday_lab_monitor_candidate_paper_pnl_replay/pytest"


def reset_dirs(name: str) -> tuple[Path, Path, Path]:
    base = LOCAL_ARTIFACT / name
    inbox = base / "manual_inbox"
    rolling = LOCAL_RESEARCH_FIXTURE / name / "rolling_origin"
    out_dir = LOCAL_REPORT / name
    shutil.rmtree(base, ignore_errors=True)
    shutil.rmtree(LOCAL_RESEARCH_FIXTURE / name, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)
    inbox.mkdir(parents=True)
    rolling.mkdir(parents=True)
    return inbox, rolling, out_dir


def write_prices(inbox: Path, *, down_day: bool = False) -> None:
    rows = []
    dates = pd.date_range("2026-06-02", periods=7, freq="B")
    for date_index, date in enumerate(dates):
        for etf_index, code in enumerate(ETF_UNIVERSE):
            base = 1.0 + etf_index * 0.1 + date_index * (0.02 if not down_day else -0.01)
            rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "datetime": f"{date.strftime('%Y-%m-%d')} 09:35:00",
                    "etf_code": code,
                    "open": base,
                    "high": base + 0.01,
                    "low": base - 0.01,
                    "close": base + 0.005,
                    "volume": 1000,
                    "amount": 1000 * base,
                    "vwap": base,
                }
            )
            rows.append(
                {
                    "trade_date": date.strftime("%Y-%m-%d"),
                    "datetime": f"{date.strftime('%Y-%m-%d')} 14:55:00",
                    "etf_code": code,
                    "open": base + 0.01,
                    "high": base + 0.03,
                    "low": base,
                    "close": base + (0.04 if not down_day else -0.02),
                    "volume": 1000,
                    "amount": 1000 * base,
                    "vwap": base,
                }
            )
    pd.DataFrame(rows).to_csv(inbox / "manual_prices.csv", index=False)


def prediction_rows(*, positive: bool = True, include_other_candidate: bool = True, anchors: list[str] | None = None) -> list[dict[str, object]]:
    anchors = anchors or ["2026-06-01", "2026-06-02", "2026-06-03"]
    rows: list[dict[str, object]] = []
    for anchor_index, anchor_date in enumerate(anchors):
        for code in ETF_UNIVERSE:
            is_positive = positive and code in {"510050", "510300"} and anchor_index != 1
            rows.append(
                {
                    "fold_id": "pytest",
                    "candidate_id": FOCUS_CANDIDATE_ID,
                    "model": FOCUS_MODEL,
                    "anchor_date": anchor_date,
                    "etf_code": code,
                    "prediction": int(is_positive),
                    "probability": 0.7 if is_positive else 0.3,
                    "train_or_oop": "validation",
                }
            )
            rows.append(
                {
                    "fold_id": "pytest",
                    "candidate_id": FOCUS_CANDIDATE_ID,
                    "model": "dummy_stratified",
                    "anchor_date": anchor_date,
                    "etf_code": code,
                    "prediction": 1,
                    "probability": 1.0,
                    "train_or_oop": "validation",
                }
            )
        if include_other_candidate:
            rows.append(
                {
                    "fold_id": "pytest",
                    "candidate_id": "label_other|base|model|policy",
                    "model": FOCUS_MODEL,
                    "anchor_date": anchor_date,
                    "etf_code": "512880",
                    "prediction": 1,
                    "probability": 1.0,
                    "train_or_oop": "validation",
                }
            )
    return rows


def write_predictions(rolling: Path, **kwargs: object) -> None:
    pd.DataFrame(prediction_rows(**kwargs)).to_csv(rolling / "rolling_origin_row_level_predictions.csv", index=False)


def run_fixture(name: str, *, positive: bool = True, down_day: bool = False, anchors: list[str] | None = None) -> tuple[dict[str, object], Path]:
    inbox, rolling, out_dir = reset_dirs(name)
    write_prices(inbox, down_day=down_day)
    write_predictions(rolling, positive=positive, anchors=anchors)
    result = run_replay(ReplayConfig(manual_inbox=inbox, rolling_origin_dir=rolling, candidate_status_dir=rolling, out_dir=out_dir))
    return result, out_dir


def test_fixed_candidate_filtering_ignores_other_candidates_and_dummy_models() -> None:
    _, out_dir = run_fixture("candidate_filtering")
    trades = pd.read_csv(out_dir / "paper_pnl_simulated_trades.csv", dtype={"etf_code": str})

    assert set(trades["etf_code"]) == {"510050", "510300"}
    assert "512880" not in set(trades["etf_code"])


def test_prediction_one_generates_paper_signal() -> None:
    result, out_dir = run_fixture("positive_signal")
    trades = pd.read_csv(out_dir / "paper_pnl_simulated_trades.csv")

    assert result["coverage"]["positive_signal_rows"] > 0
    assert len(trades) > 0


def test_no_signal_day_stays_cash() -> None:
    _, out_dir = run_fixture("cash_sleeve")
    sleeves = pd.read_csv(out_dir / "paper_pnl_sleeves.csv")

    assert "cash" in set(sleeves["status"])
    assert sleeves.loc[sleeves["status"].eq("cash"), "budget_notional"].sum() == 0


def test_entry_uses_next_trading_day_first_price() -> None:
    _, out_dir = run_fixture("entry_timing")
    trades = pd.read_csv(out_dir / "paper_pnl_simulated_trades.csv")
    trade = trades[trades["anchor_date"].eq("2026-06-01")].iloc[0]

    assert trade["entry_date"] == "2026-06-02"
    assert trade["entry_price"] == pytest.approx(1.1)


def test_exit_uses_t_plus_3_last_5m_price() -> None:
    _, out_dir = run_fixture("exit_timing")
    trades = pd.read_csv(out_dir / "paper_pnl_simulated_trades.csv")
    trade = trades[trades["anchor_date"].eq("2026-06-01")].iloc[0]

    assert trade["exit_date"] == "2026-06-04"
    assert trade["exit_price"] == pytest.approx(1.18)


def test_costs_reduce_gross_return() -> None:
    _, out_dir = run_fixture("costs_reduce")
    trades = pd.read_csv(out_dir / "paper_pnl_simulated_trades.csv")

    assert (trades["net_return"] < trades["gross_return"]).all()


def test_overlapping_three_day_sleeves_do_not_exceed_full_budget_exposure() -> None:
    _, out_dir = run_fixture("overlap_budget", anchors=["2026-06-01", "2026-06-02", "2026-06-03", "2026-06-04"])
    nav = pd.read_csv(out_dir / "paper_pnl_nav.csv")

    assert nav["budget_exposure"].max() <= 1.0


def test_benchmark_comparison_generated() -> None:
    result, out_dir = run_fixture("benchmark")
    benchmark = pd.read_csv(out_dir / "paper_pnl_benchmark_comparison.csv")

    assert set(benchmark["benchmark"]) == {
        "cash",
        "equal_weight_8_etf",
        "signal_selected_gross_before_cost",
        "signal_selected_net_after_cost",
    }
    assert len(result["benchmark_comparison"]) == 4


def test_missing_row_level_predictions_blocked() -> None:
    inbox, rolling, out_dir = reset_dirs("missing_predictions")
    write_prices(inbox)

    result = run_replay(ReplayConfig(manual_inbox=inbox, rolling_origin_dir=rolling, candidate_status_dir=rolling, out_dir=out_dir))

    assert result["decision"] == DECISION_MISSING_PREDICTIONS
    assert (out_dir / "paper_pnl_decision.json").exists()


def test_output_path_outside_local_research_outputs_rejected(tmp_path: Path) -> None:
    inbox, rolling, _ = reset_dirs("bad_out")
    write_prices(inbox)
    write_predictions(rolling)

    with pytest.raises(PaperPnlReplayError):
        run_replay(ReplayConfig(manual_inbox=inbox, rolling_origin_dir=rolling, candidate_status_dir=rolling, out_dir=tmp_path))


def test_no_order_intent_file_generated() -> None:
    _, out_dir = run_fixture("no_order_intent")

    generated = [path.name.lower() for path in out_dir.iterdir()]
    assert not any("orderintent" in name or "order_intent" in name for name in generated)


def test_no_model_scaler_file_written() -> None:
    _, out_dir = run_fixture("no_model_scaler")

    generated = [path.name.lower() for path in out_dir.iterdir()]
    assert not any(name.endswith((".pkl", ".pt", ".pth", ".ckpt", ".joblib")) for name in generated)
    assert not any("scaler" in name or "model" in name for name in generated)


def test_boundary_fields_all_false_except_paper_trading_flag() -> None:
    result, out_dir = run_fixture("boundary")
    decision = json.loads((out_dir / "paper_pnl_decision.json").read_text(encoding="utf-8"))

    for field, expected in BOUNDARY_FIELDS.items():
        assert result[field] is expected
        assert decision[field] is expected
