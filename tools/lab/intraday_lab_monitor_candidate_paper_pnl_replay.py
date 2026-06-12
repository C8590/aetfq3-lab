from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
FOCUS_CANDIDATE_ID = "label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy"
FOCUS_MODEL = "logistic_balanced_scaled"
DEFAULT_MANUAL_INBOX = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_historical_5m_manual_inbox")
DEFAULT_ROLLING_ORIGIN_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_rolling_origin_walk_forward_no_save_validation")
DEFAULT_CANDIDATE_STATUS_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_lab_monitor_candidate_status")
DEFAULT_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_lab_monitor_candidate_paper_pnl_replay")
ROW_LEVEL_PREDICTIONS = "rolling_origin_row_level_predictions.csv"
INITIAL_CASH = 1_000_000.0
BASE_COST_BPS_PER_SIDE = 8.0
COST_SENSITIVITY_BPS_PER_SIDE = [0.0, 5.0, 8.0, 10.0, 20.0]
ETF_UNIVERSE = ["159915", "510050", "510300", "510500", "512100", "588000", "159949", "512880"]
REQUIRED_PREDICTION_COLUMNS = {
    "candidate_id",
    "anchor_date",
    "etf_code",
    "prediction",
}
REQUIRED_PRICE_COLUMNS = {
    "trade_date",
    "datetime",
    "etf_code",
    "open",
    "close",
}
BOUNDARY_FIELDS = {
    "paper_trading_only": True,
    "real_order_routing": False,
    "order_intent_generated": False,
    "stable_evidence": False,
    "stable_promotion_ready": False,
    "qmt_ready": False,
    "formal_training_ready": False,
    "order_intent_ready": False,
    "automatic_promotion_ready": False,
    "model_saved": False,
    "scaler_saved": False,
    "checkpoint_saved": False,
    "gpu_used": False,
    "torchrun_used": False,
    "stable_runtime_written": False,
    "output_written": False,
}
DECISION_COMPLETED = "PAPER_PNL_REPLAY_COMPLETED_REVIEW_REQUIRED"
DECISION_PROFIT = "PAPER_PNL_REPLAY_PROFITABILITY_OBSERVED_REVIEW_REQUIRED"
DECISION_NO_PROFIT = "PAPER_PNL_REPLAY_NO_PROFITABILITY_OBSERVED_REVIEW_REQUIRED"
DECISION_MISSING_PREDICTIONS = "PAPER_PNL_REPLAY_BLOCKED_MISSING_ROW_LEVEL_PREDICTIONS"
DECISION_PRICE_DATA = "PAPER_PNL_REPLAY_BLOCKED_PRICE_DATA"
DECISION_DATA_QUALITY = "PAPER_PNL_REPLAY_BLOCKED_DATA_QUALITY"
DECISION_SIGNAL_EMPTY = "PAPER_PNL_REPLAY_BLOCKED_SIGNAL_EMPTY"


class PaperPnlReplayError(RuntimeError):
    pass


@dataclass(frozen=True)
class ReplayConfig:
    manual_inbox: Path = DEFAULT_MANUAL_INBOX
    rolling_origin_dir: Path = DEFAULT_ROLLING_ORIGIN_DIR
    candidate_status_dir: Path = DEFAULT_CANDIDATE_STATUS_DIR
    out_dir: Path = DEFAULT_OUT_DIR
    initial_cash: float = INITIAL_CASH
    base_cost_bps_per_side: float = BASE_COST_BPS_PER_SIDE


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def ensure_under(path: Path, allowed_root: Path, label: str) -> Path:
    resolved = resolve_repo_path(path).resolve()
    allowed = resolve_repo_path(allowed_root).resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise PaperPnlReplayError(f"{label} must be under {allowed_root}") from exc
    return resolved


def resolve_inputs(config: ReplayConfig) -> ReplayConfig:
    return ReplayConfig(
        manual_inbox=ensure_under(config.manual_inbox, Path(".local_artifact_backup"), "manual-inbox"),
        rolling_origin_dir=ensure_under(config.rolling_origin_dir, Path(".local_research_outputs"), "rolling-origin-dir"),
        candidate_status_dir=ensure_under(config.candidate_status_dir, Path(".local_research_outputs"), "candidate-status-dir"),
        out_dir=ensure_under(config.out_dir, DEFAULT_OUT_DIR, "out-dir"),
        initial_cash=config.initial_cash,
        base_cost_bps_per_side=config.base_cost_bps_per_side,
    )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows), columns=list(columns))
    frame.to_csv(path, index=False)


def normalize_code(value: Any) -> str:
    return str(value).strip().split(".")[0].zfill(6)


def empty_outputs(out_dir: Path, decision: str, reason: str) -> dict[str, Any]:
    nav_columns = ["trade_date", "equity", "cash", "open_position_value", "daily_return", "exposure", "budget_exposure"]
    sleeve_columns = [
        "sleeve_id",
        "anchor_date",
        "entry_date",
        "exit_date",
        "positive_etf_count",
        "tradable_etf_count",
        "budget_notional",
        "status",
        "skip_reason",
        "gross_return",
        "net_return",
        "net_pnl",
    ]
    trade_columns = [
        "sleeve_id",
        "anchor_date",
        "etf_code",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "allocated_notional",
        "gross_return",
        "net_return",
        "gross_pnl",
        "net_pnl",
        "estimated_cost",
    ]
    write_csv(out_dir / "paper_pnl_nav.csv", [], nav_columns)
    write_csv(out_dir / "paper_pnl_sleeves.csv", [], sleeve_columns)
    write_csv(out_dir / "paper_pnl_simulated_trades.csv", [], trade_columns)
    write_csv(out_dir / "paper_pnl_monthly_returns.csv", [], ["month", "return"])
    write_csv(out_dir / "paper_pnl_etf_contribution.csv", [], ["etf_code", "gross_pnl", "net_pnl", "net_return_contribution"])
    write_csv(out_dir / "paper_pnl_cost_sensitivity.csv", [], ["cost_bps_per_side", "total_return", "max_drawdown"])
    write_csv(out_dir / "paper_pnl_benchmark_comparison.csv", [], ["benchmark", "total_return", "excess_vs_signal_net"])
    payload = {
        "lab_declaration": LAB_DECLARATION,
        "phase": "intraday_lab_monitor_candidate_paper_pnl_replay",
        "decision": decision,
        "status": "blocked",
        "blocked_reason": reason,
        "generated_at": utc_now(),
        "candidate_id": FOCUS_CANDIDATE_ID,
        "model": FOCUS_MODEL,
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "paper_pnl_replay_report.json", payload)
    write_json(out_dir / "paper_pnl_decision.json", payload)
    (out_dir / "paper_pnl_replay_report.md").write_text(
        "\n".join(
            [
                "# Paper PnL Replay Report",
                "",
                LAB_DECLARATION,
                "",
                f"Decision: `{decision}`",
                "",
                f"Blocked reason: `{reason}`",
                "",
                "This is a Lab paper-trading diagnostic only. It is not OOP validation, Stable evidence, QMT routing, or OrderIntent generation.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def load_predictions(rolling_origin_dir: Path) -> tuple[pd.DataFrame | None, str | None]:
    path = rolling_origin_dir / ROW_LEVEL_PREDICTIONS
    if not path.exists():
        return None, f"missing {ROW_LEVEL_PREDICTIONS}"
    frame = pd.read_csv(path, dtype={"etf_code": str})
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(frame.columns))
    if missing:
        return None, f"missing prediction columns: {missing}"
    if "model" in frame.columns:
        frame = frame[frame["model"].astype(str) == FOCUS_MODEL].copy()
    frame = frame[frame["candidate_id"].astype(str) == FOCUS_CANDIDATE_ID].copy()
    if "train_or_oop" in frame.columns:
        frame = frame[frame["train_or_oop"].astype(str).str.lower().eq("validation")].copy()
    frame["anchor_date"] = pd.to_datetime(frame["anchor_date"]).dt.date
    frame["etf_code"] = frame["etf_code"].map(normalize_code)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce").fillna(0).astype(int)
    return frame, None


def load_price_data(manual_inbox: Path) -> tuple[pd.DataFrame | None, str | None]:
    if not manual_inbox.exists():
        return None, "manual inbox missing"
    csv_paths = sorted(path for path in manual_inbox.glob("*.csv") if path.name.lower() != "sha256sums.csv")
    if not csv_paths:
        return None, "no csv price export found"
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        frame = pd.read_csv(path, dtype={"etf_code": str})
        missing = sorted(REQUIRED_PRICE_COLUMNS.difference(frame.columns))
        if missing:
            return None, f"{path.name} missing price columns: {missing}"
        frames.append(frame)
    prices = pd.concat(frames, ignore_index=True)
    prices["trade_date"] = pd.to_datetime(prices["trade_date"]).dt.date
    prices["datetime"] = pd.to_datetime(prices["datetime"])
    prices["etf_code"] = prices["etf_code"].map(normalize_code)
    prices["open"] = pd.to_numeric(prices["open"], errors="coerce")
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["trade_date", "datetime", "etf_code", "open", "close"])
    prices = prices[prices["etf_code"].isin(ETF_UNIVERSE)].copy()
    prices = prices.sort_values(["trade_date", "etf_code", "datetime"])
    duplicate_count = int(prices.duplicated(["etf_code", "datetime"]).sum())
    if duplicate_count:
        return None, f"duplicate etf_code+datetime bars: {duplicate_count}"
    if prices.empty:
        return None, "no ETF universe rows after standardization"
    return prices, None


def build_daily_price_maps(prices: pd.DataFrame) -> tuple[list[Any], dict[tuple[Any, str], dict[str, Any]]]:
    daily: dict[tuple[Any, str], dict[str, Any]] = {}
    for (trade_date, etf_code), group in prices.groupby(["trade_date", "etf_code"], sort=True):
        ordered = group.sort_values("datetime")
        daily[(trade_date, etf_code)] = {
            "entry_price": float(ordered.iloc[0]["open"]),
            "exit_price": float(ordered.iloc[-1]["close"]),
            "last_datetime": str(ordered.iloc[-1]["datetime"]),
        }
    return sorted(prices["trade_date"].unique()), daily


def next_trading_dates(anchor_date: Any, trading_days: Sequence[Any], count: int = 3) -> list[Any]:
    after = [day for day in trading_days if day > anchor_date]
    return after[:count]


def build_sleeve_plan(signals: pd.DataFrame, trading_days: Sequence[Any], daily_prices: dict[tuple[Any, str], dict[str, Any]]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []
    for index, (anchor_date, group) in enumerate(signals.groupby("anchor_date", sort=True), start=1):
        positives = sorted(group.loc[group["prediction"].eq(1), "etf_code"].unique().tolist())
        next_days = next_trading_dates(anchor_date, trading_days, 3)
        if len(next_days) < 3:
            plans.append(
                {
                    "sleeve_id": f"S{index:05d}",
                    "anchor_date": anchor_date,
                    "entry_date": None,
                    "exit_date": None,
                    "positive_etfs": positives,
                    "tradable_etfs": [],
                    "status": "skipped",
                    "skip_reason": "missing_t_plus_1_or_t_plus_3_trading_day",
                }
            )
            continue
        entry_date, exit_date = next_days[0], next_days[2]
        tradable = [
            etf
            for etf in positives
            if (entry_date, etf) in daily_prices and (exit_date, etf) in daily_prices
        ]
        status = "cash" if not positives else ("tradable" if tradable else "skipped")
        skip_reason = "" if status != "skipped" else "missing_entry_or_exit_5m_price"
        plans.append(
            {
                "sleeve_id": f"S{index:05d}",
                "anchor_date": anchor_date,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "positive_etfs": positives,
                "tradable_etfs": tradable,
                "status": status,
                "skip_reason": skip_reason,
            }
        )
    return plans


def close_price_for(daily_prices: dict[tuple[Any, str], dict[str, Any]], trade_date: Any, etf_code: str) -> float | None:
    item = daily_prices.get((trade_date, etf_code))
    return None if item is None else float(item["exit_price"])


def simulate_replay(
    signals: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    initial_cash: float,
    cost_bps_per_side: float,
) -> dict[str, Any]:
    trading_days, daily_prices = build_daily_price_maps(prices)
    plans = build_sleeve_plan(signals, trading_days, daily_prices)
    tradable_plans = [plan for plan in plans if plan["status"] in {"tradable", "cash"} and plan["entry_date"] is not None]
    if not tradable_plans:
        return {"blocked_reason": "no tradable paper sleeves", "plans": plans}
    first_day = min(plan["entry_date"] for plan in tradable_plans)
    last_day = max(plan["exit_date"] for plan in tradable_plans)
    replay_days = [day for day in trading_days if first_day <= day <= last_day]
    plans_by_entry: dict[Any, list[dict[str, Any]]] = {}
    for plan in plans:
        if plan["entry_date"] is not None:
            plans_by_entry.setdefault(plan["entry_date"], []).append(plan)

    cost_rate = cost_bps_per_side / 10_000.0
    cash = float(initial_cash)
    last_equity = float(initial_cash)
    open_positions: list[dict[str, Any]] = []
    sleeve_records: dict[str, dict[str, Any]] = {}
    trade_records: list[dict[str, Any]] = []
    nav_rows: list[dict[str, Any]] = []

    for trade_date in replay_days:
        active_budget = sum(float(pos["allocated_notional"]) for pos in open_positions)
        for plan in plans_by_entry.get(trade_date, []):
            positive_count = len(plan["positive_etfs"])
            tradable_count = len(plan["tradable_etfs"])
            budget_notional = 0.0
            if plan["status"] == "tradable" and tradable_count:
                available_budget = max(0.0, last_equity - active_budget)
                budget_notional = min(last_equity / 3.0, available_budget)
                per_etf_notional = budget_notional / tradable_count if tradable_count else 0.0
                for etf_code in plan["tradable_etfs"]:
                    entry_price = float(daily_prices[(plan["entry_date"], etf_code)]["entry_price"])
                    shares = per_etf_notional * (1.0 - cost_rate) / entry_price if entry_price > 0 else 0.0
                    cash -= per_etf_notional
                    active_budget += per_etf_notional
                    open_positions.append(
                        {
                            "sleeve_id": plan["sleeve_id"],
                            "anchor_date": plan["anchor_date"],
                            "entry_date": plan["entry_date"],
                            "exit_date": plan["exit_date"],
                            "etf_code": etf_code,
                            "entry_price": entry_price,
                            "allocated_notional": per_etf_notional,
                            "shares": shares,
                        }
                    )
            sleeve_records[plan["sleeve_id"]] = {
                "sleeve_id": plan["sleeve_id"],
                "anchor_date": str(plan["anchor_date"]),
                "entry_date": "" if plan["entry_date"] is None else str(plan["entry_date"]),
                "exit_date": "" if plan["exit_date"] is None else str(plan["exit_date"]),
                "positive_etf_count": positive_count,
                "tradable_etf_count": tradable_count,
                "budget_notional": budget_notional,
                "status": plan["status"],
                "skip_reason": plan["skip_reason"],
                "gross_return": 0.0,
                "net_return": 0.0,
                "net_pnl": 0.0,
            }

        exiting: list[dict[str, Any]] = []
        remaining: list[dict[str, Any]] = []
        position_value = 0.0
        for pos in open_positions:
            close_price = close_price_for(daily_prices, trade_date, pos["etf_code"])
            if close_price is None:
                close_price = float(pos["entry_price"])
            liquidation_value = float(pos["shares"]) * close_price * (1.0 - cost_rate)
            if pos["exit_date"] == trade_date:
                gross_value = float(pos["allocated_notional"]) * close_price / float(pos["entry_price"])
                gross_pnl = gross_value - float(pos["allocated_notional"])
                net_pnl = liquidation_value - float(pos["allocated_notional"])
                estimated_cost = gross_value - liquidation_value
                cash += liquidation_value
                trade_records.append(
                    {
                        "sleeve_id": pos["sleeve_id"],
                        "anchor_date": str(pos["anchor_date"]),
                        "etf_code": pos["etf_code"],
                        "entry_date": str(pos["entry_date"]),
                        "exit_date": str(pos["exit_date"]),
                        "entry_price": float(pos["entry_price"]),
                        "exit_price": close_price,
                        "allocated_notional": float(pos["allocated_notional"]),
                        "gross_return": gross_pnl / float(pos["allocated_notional"]) if pos["allocated_notional"] else 0.0,
                        "net_return": net_pnl / float(pos["allocated_notional"]) if pos["allocated_notional"] else 0.0,
                        "gross_pnl": gross_pnl,
                        "net_pnl": net_pnl,
                        "estimated_cost": estimated_cost,
                    }
                )
                exiting.append(pos)
            else:
                position_value += liquidation_value
                remaining.append(pos)
        open_positions = remaining

        for pos in exiting:
            related = [row for row in trade_records if row["sleeve_id"] == pos["sleeve_id"]]
            budget = sum(float(row["allocated_notional"]) for row in related)
            net_pnl = sum(float(row["net_pnl"]) for row in related)
            gross_pnl = sum(float(row["gross_pnl"]) for row in related)
            if pos["sleeve_id"] in sleeve_records and budget:
                sleeve_records[pos["sleeve_id"]]["gross_return"] = gross_pnl / budget
                sleeve_records[pos["sleeve_id"]]["net_return"] = net_pnl / budget
                sleeve_records[pos["sleeve_id"]]["net_pnl"] = net_pnl

        equity = cash + position_value
        daily_return = equity / last_equity - 1.0 if last_equity else 0.0
        budget_exposure = sum(float(pos["allocated_notional"]) for pos in open_positions) / equity if equity else 0.0
        exposure = position_value / equity if equity else 0.0
        nav_rows.append(
            {
                "trade_date": str(trade_date),
                "equity": equity,
                "cash": cash,
                "open_position_value": position_value,
                "daily_return": daily_return,
                "exposure": exposure,
                "budget_exposure": min(budget_exposure, 1.0),
            }
        )
        last_equity = equity

    nav = pd.DataFrame(nav_rows)
    trades = pd.DataFrame(trade_records)
    sleeves = pd.DataFrame(sleeve_records.values())
    if trades.empty:
        return {"blocked_reason": "no simulated trades generated", "plans": plans}
    return {
        "nav": nav,
        "trades": trades,
        "sleeves": sleeves,
        "plans": plans,
        "cost_bps_per_side": cost_bps_per_side,
    }


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    cumulative_max = equity.cummax()
    drawdown = equity / cumulative_max - 1.0
    return float(drawdown.min())


def monthly_returns(nav: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    if nav.empty:
        return pd.DataFrame(columns=["month", "return"])
    data = nav.copy()
    data["month"] = pd.to_datetime(data["trade_date"]).dt.to_period("M").astype(str)
    rows: list[dict[str, Any]] = []
    prior = initial_cash
    for month, group in data.groupby("month", sort=True):
        ending = float(group.iloc[-1]["equity"])
        rows.append({"month": month, "return": ending / prior - 1.0 if prior else 0.0})
        prior = ending
    return pd.DataFrame(rows)


def etf_contribution(trades: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["etf_code", "gross_pnl", "net_pnl", "net_return_contribution"])
    grouped = trades.groupby("etf_code", as_index=False).agg({"gross_pnl": "sum", "net_pnl": "sum"})
    grouped["net_return_contribution"] = grouped["net_pnl"] / initial_cash
    return grouped.sort_values("net_pnl", ascending=False)


def equal_weight_benchmark(prices: pd.DataFrame, start_day: str, end_day: str) -> float:
    data = prices.copy()
    data["trade_date_str"] = data["trade_date"].astype(str)
    data = data[(data["trade_date_str"] >= start_day) & (data["trade_date_str"] <= end_day)]
    if data.empty:
        return 0.0
    daily_close = (
        data.sort_values(["trade_date", "etf_code", "datetime"])
        .groupby(["trade_date_str", "etf_code"], as_index=False)
        .tail(1)[["trade_date_str", "etf_code", "close"]]
    )
    pivot = daily_close.pivot(index="trade_date_str", columns="etf_code", values="close").sort_index()
    pivot = pivot[[code for code in ETF_UNIVERSE if code in pivot.columns]].dropna(how="any")
    if len(pivot) < 2:
        return 0.0
    returns = pivot.pct_change().dropna().mean(axis=1)
    return float((1.0 + returns).prod() - 1.0)


def compute_metrics(result: dict[str, Any], prices: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    nav: pd.DataFrame = result["nav"]
    trades: pd.DataFrame = result["trades"]
    sleeves: pd.DataFrame = result["sleeves"]
    monthly = monthly_returns(nav, initial_cash)
    etf = etf_contribution(trades, initial_cash)
    total_return = float(nav.iloc[-1]["equity"] / initial_cash - 1.0)
    daily_returns = nav["daily_return"].astype(float)
    volatility = float(daily_returns.std(ddof=0) * math.sqrt(252)) if len(daily_returns) > 1 else 0.0
    sharpe = float(daily_returns.mean() / daily_returns.std(ddof=0) * math.sqrt(252)) if daily_returns.std(ddof=0) > 0 else 0.0
    days = max(len(nav), 1)
    annualized = float((1.0 + total_return) ** (252 / days) - 1.0) if total_return > -1.0 else -1.0
    mdd = max_drawdown(nav["equity"].astype(float))
    calmar = annualized / abs(mdd) if mdd < 0 else 0.0
    gross_return = float(trades["gross_pnl"].sum() / initial_cash)
    net_return = float(trades["net_pnl"].sum() / initial_cash)
    total_estimated_cost = float(trades["estimated_cost"].sum())
    average_holding_days = float(
        (
            pd.to_datetime(trades["exit_date"]) - pd.to_datetime(trades["entry_date"])
        ).dt.days.mean()
    )
    turnover = float((trades["allocated_notional"].sum() + (trades["allocated_notional"] + trades["gross_pnl"]).sum()) / nav["equity"].mean())
    benchmark_equal_weight = equal_weight_benchmark(prices, str(nav.iloc[0]["trade_date"]), str(nav.iloc[-1]["trade_date"]))
    benchmark_rows = [
        {"benchmark": "cash", "total_return": 0.0, "excess_vs_signal_net": -total_return},
        {"benchmark": "equal_weight_8_etf", "total_return": benchmark_equal_weight, "excess_vs_signal_net": benchmark_equal_weight - total_return},
        {"benchmark": "signal_selected_gross_before_cost", "total_return": gross_return, "excess_vs_signal_net": gross_return - total_return},
        {"benchmark": "signal_selected_net_after_cost", "total_return": total_return, "excess_vs_signal_net": 0.0},
    ]
    worst_month = "" if monthly.empty else str(monthly.sort_values("return").iloc[0]["month"])
    best_month = "" if monthly.empty else str(monthly.sort_values("return", ascending=False).iloc[0]["month"])
    winning_months = int((monthly["return"] > 0).sum()) if not monthly.empty else 0
    losing_months = int((monthly["return"] < 0).sum()) if not monthly.empty else 0
    month_concentration = False
    if not monthly.empty and total_return > 0:
        month_concentration = float(monthly["return"].max()) > max(0.6 * total_return, 0.0)
    etf_concentration_flag = False
    if not etf.empty:
        denom = float(etf["net_pnl"].abs().sum())
        etf_concentration_flag = denom > 0 and float(etf["net_pnl"].abs().max() / denom) > 0.5
    return {
        "metrics": {
            "total_return": total_return,
            "annualized_return": annualized,
            "max_drawdown": mdd,
            "volatility": volatility,
            "sharpe_like_diagnostic": sharpe,
            "calmar_like_diagnostic": calmar,
            "win_month_count": winning_months,
            "losing_month_count": losing_months,
            "trade_count": int(len(trades)),
            "average_holding_days": average_holding_days,
            "turnover": turnover,
            "average_exposure": float(nav["exposure"].mean()),
            "max_budget_exposure": float(nav["budget_exposure"].max()),
            "gross_return": gross_return,
            "net_return": net_return,
            "total_estimated_cost": total_estimated_cost,
            "benchmark_excess_return_vs_equal_weight": total_return - benchmark_equal_weight,
            "worst_month": worst_month,
            "best_month": best_month,
        },
        "monthly_returns": monthly,
        "etf_contribution": etf,
        "benchmark_comparison": pd.DataFrame(benchmark_rows),
        "concentration_checks": {
            "month_concentration_observed": bool(month_concentration),
            "single_etf_contribution_concentration_observed": bool(etf_concentration_flag),
        },
    }


def cost_sensitivity(signals: pd.DataFrame, prices: pd.DataFrame, initial_cash: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for cost in COST_SENSITIVITY_BPS_PER_SIDE:
        result = simulate_replay(signals, prices, initial_cash=initial_cash, cost_bps_per_side=cost)
        if "blocked_reason" in result:
            rows.append({"cost_bps_per_side": cost, "total_return": None, "max_drawdown": None, "trade_count": 0})
            continue
        rows.append(
            {
                "cost_bps_per_side": cost,
                "total_return": float(result["nav"].iloc[-1]["equity"] / initial_cash - 1.0),
                "max_drawdown": max_drawdown(result["nav"]["equity"].astype(float)),
                "trade_count": int(len(result["trades"])),
            }
        )
    return pd.DataFrame(rows)


def choose_decision(metrics: dict[str, Any], sensitivity: pd.DataFrame) -> str:
    values = metrics["metrics"]
    concentration = metrics["concentration_checks"]
    cost10 = sensitivity.loc[sensitivity["cost_bps_per_side"].eq(10.0), "total_return"]
    cost10_ok = not cost10.empty and pd.notna(cost10.iloc[0]) and float(cost10.iloc[0]) > -0.10
    profitability_observed = all(
        [
            values["total_return"] > 0,
            math.isfinite(values["max_drawdown"]),
            values["total_return"] > 0.0,
            not concentration["month_concentration_observed"],
            not concentration["single_etf_contribution_concentration_observed"],
            cost10_ok,
            BOUNDARY_FIELDS["order_intent_generated"] is False,
            BOUNDARY_FIELDS["stable_evidence"] is False,
        ]
    )
    return DECISION_PROFIT if profitability_observed else DECISION_NO_PROFIT


def write_report(out_dir: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    lines = [
        "# Lab Monitor Candidate Paper PnL Replay",
        "",
        LAB_DECLARATION,
        "",
        "## Purpose",
        "",
        "This is a Lab-only paper trading diagnostic for the registered monitor candidate. It is not OOP validation, Stable evidence, QMT routing, OrderIntent generation, training, or promotion.",
        "",
        "## Signal / Execution Protocol",
        "",
        f"- Candidate: `{FOCUS_CANDIDATE_ID}`",
        f"- Model row filter: `{FOCUS_MODEL}`",
        "- Signal: existing row-level `prediction=1`; threshold remains the validator default 0.5.",
        "- Entry: next trading day first available 5m open.",
        "- Exit: T+3 trading day last available 5m close.",
        "- Sleeve budget: one third of current paper equity, no leverage, no short.",
        f"- Base cost: {BASE_COST_BPS_PER_SIDE:.0f} bps per side.",
        "",
        "## Result",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Total return: `{metrics['total_return']:.6f}`",
        f"- Max drawdown: `{metrics['max_drawdown']:.6f}`",
        f"- Trade count: `{metrics['trade_count']}`",
        f"- Average exposure: `{metrics['average_exposure']:.6f}`",
        f"- Max budget exposure: `{metrics['max_budget_exposure']:.6f}`",
        "",
        "## Boundary",
        "",
        "- No QMT, no account data, no OrderIntent, no real order, no Stable effect, no model/scaler/checkpoint save.",
        "",
    ]
    (out_dir / "paper_pnl_replay_report.md").write_text("\n".join(lines), encoding="utf-8")


def run_replay(config: ReplayConfig) -> dict[str, Any]:
    config = resolve_inputs(config)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    predictions, prediction_error = load_predictions(config.rolling_origin_dir)
    if prediction_error:
        return empty_outputs(config.out_dir, DECISION_MISSING_PREDICTIONS, prediction_error)
    assert predictions is not None
    prices, price_error = load_price_data(config.manual_inbox)
    if price_error:
        return empty_outputs(config.out_dir, DECISION_PRICE_DATA if "missing" in price_error or "no csv" in price_error else DECISION_DATA_QUALITY, price_error)
    assert prices is not None
    if predictions.empty:
        return empty_outputs(config.out_dir, DECISION_MISSING_PREDICTIONS, "no fixed candidate prediction rows")
    if int(predictions["prediction"].sum()) == 0:
        return empty_outputs(config.out_dir, DECISION_SIGNAL_EMPTY, "fixed candidate has no prediction=1 rows")

    result = simulate_replay(
        predictions,
        prices,
        initial_cash=config.initial_cash,
        cost_bps_per_side=config.base_cost_bps_per_side,
    )
    if "blocked_reason" in result:
        return empty_outputs(config.out_dir, DECISION_PRICE_DATA, str(result["blocked_reason"]))
    metrics = compute_metrics(result, prices, config.initial_cash)
    sensitivity = cost_sensitivity(predictions, prices, config.initial_cash)
    decision = choose_decision(metrics, sensitivity)

    result["nav"].to_csv(config.out_dir / "paper_pnl_nav.csv", index=False)
    result["sleeves"].to_csv(config.out_dir / "paper_pnl_sleeves.csv", index=False)
    result["trades"].to_csv(config.out_dir / "paper_pnl_simulated_trades.csv", index=False)
    metrics["monthly_returns"].to_csv(config.out_dir / "paper_pnl_monthly_returns.csv", index=False)
    metrics["etf_contribution"].to_csv(config.out_dir / "paper_pnl_etf_contribution.csv", index=False)
    sensitivity.to_csv(config.out_dir / "paper_pnl_cost_sensitivity.csv", index=False)
    metrics["benchmark_comparison"].to_csv(config.out_dir / "paper_pnl_benchmark_comparison.csv", index=False)

    payload = {
        "lab_declaration": LAB_DECLARATION,
        "phase": "intraday_lab_monitor_candidate_paper_pnl_replay",
        "status": "completed",
        "decision": decision,
        "generated_at": utc_now(),
        "candidate_id": FOCUS_CANDIDATE_ID,
        "model": FOCUS_MODEL,
        "signal_source": str((config.rolling_origin_dir / ROW_LEVEL_PREDICTIONS).relative_to(REPO_ROOT)),
        "price_source": str(config.manual_inbox.relative_to(REPO_ROOT)),
        "execution_protocol": {
            "prediction_positive_means_long_signal": True,
            "threshold": 0.5,
            "threshold_tuned": False,
            "entry": "next_trading_day_first_available_5m_open",
            "exit": "t_plus_3_trading_day_last_available_5m_close",
            "initial_cash": config.initial_cash,
            "sleeve_budget_fraction": 1.0 / 3.0,
            "no_leverage": True,
            "no_short": True,
            "cash_earns_zero": True,
            "base_cost_bps_per_side": config.base_cost_bps_per_side,
        },
        "coverage": {
            "prediction_rows": int(len(predictions)),
            "positive_signal_rows": int(predictions["prediction"].sum()),
            "price_rows": int(len(prices)),
            "price_start": str(prices["trade_date"].min()),
            "price_end": str(prices["trade_date"].max()),
            "nav_start": str(result["nav"].iloc[0]["trade_date"]),
            "nav_end": str(result["nav"].iloc[-1]["trade_date"]),
        },
        "metrics": metrics["metrics"],
        "concentration_checks": metrics["concentration_checks"],
        "cost_sensitivity": sensitivity.to_dict(orient="records"),
        "benchmark_comparison": metrics["benchmark_comparison"].to_dict(orient="records"),
        **BOUNDARY_FIELDS,
    }
    write_json(config.out_dir / "paper_pnl_replay_report.json", payload)
    write_json(config.out_dir / "paper_pnl_decision.json", payload)
    write_report(config.out_dir, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lab-only monitor candidate paper PnL replay.")
    parser.add_argument("--manual-inbox", type=Path, default=DEFAULT_MANUAL_INBOX)
    parser.add_argument("--rolling-origin-dir", type=Path, default=DEFAULT_ROLLING_ORIGIN_DIR)
    parser.add_argument("--candidate-status-dir", type=Path, default=DEFAULT_CANDIDATE_STATUS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_replay(
        ReplayConfig(
            manual_inbox=args.manual_inbox,
            rolling_origin_dir=args.rolling_origin_dir,
            candidate_status_dir=args.candidate_status_dir,
            out_dir=args.out_dir,
        )
    )
    print(json.dumps({"decision": payload["decision"], "status": payload["status"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
