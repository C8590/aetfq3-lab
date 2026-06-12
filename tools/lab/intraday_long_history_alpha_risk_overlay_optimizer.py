from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_long_history_alpha_optimizer import (  # noqa: E402
    FEATURE_SETS,
    fit_predict_proba,
    make_rolling_folds,
    model_artifacts_present,
    pnl_metrics,
)
from tools.lab.intraday_long_history_data_lake import (  # noqa: E402
    BOUNDARY_FIELDS,
    LAB_DECLARATION,
    LABEL_COLUMNS,
    OUTCOME_COLUMNS,
    TIME_CENSORED_FEATURES,
    json_safe,
    resolve_repo_path,
    write_json,
)


REPORT_TYPE = "intraday_long_history_alpha_risk_overlay_optimizer"
ALLOWED_DATA_LAKE = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_long_history_data_lake")
ALLOWED_V0_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_long_history_alpha_optimization")
ALLOWED_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_long_history_alpha_risk_overlay_optimizer")
DECISION_CANDIDATES = "LONG_HISTORY_RISK_OVERLAY_OPTIMIZATION_COMPLETED_CANDIDATES_FOUND_REVIEW_REQUIRED"
DECISION_PROMISING = "LONG_HISTORY_RISK_OVERLAY_OPTIMIZATION_PROMISING_BUT_DRAWDOWN_UNCONTROLLED"
DECISION_NO_CANDIDATES = "LONG_HISTORY_RISK_OVERLAY_OPTIMIZATION_COMPLETED_NO_CANDIDATES_FOUND"
DECISION_MISSING = "LONG_HISTORY_RISK_OVERLAY_OPTIMIZATION_BLOCKED_MISSING_V0_OUTPUTS"
DECISION_DATA_QUALITY = "LONG_HISTORY_RISK_OVERLAY_OPTIMIZATION_BLOCKED_DATA_QUALITY"
DECISION_LEAKAGE = "LONG_HISTORY_RISK_OVERLAY_OPTIMIZATION_BLOCKED_LEAKAGE_RISK"
DECISION_RUNTIME = "LONG_HISTORY_RISK_OVERLAY_OPTIMIZATION_BLOCKED_RUNTIME_ERROR"
LAB_CANDIDATE_STATUS = "LAB_DIAGNOSTIC_ALPHA_CANDIDATE_REVIEW_REQUIRED"
PROMISING_STATUS = "PROMISING_BUT_DRAWDOWN_UNCONTROLLED_REVIEW_REQUIRED"
REJECTED_STATUS = "LAB_DIAGNOSTIC_ALPHA_REJECTED_OR_INSUFFICIENT_REVIEW"
MODEL_ARTIFACT_SUFFIXES = {".pkl", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"}
ORDER_INTENT_TOKENS = {"orderintent", "order_intent"}


class RiskOverlayOptimizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class RiskOverlayConfig:
    data_lake: Path
    v0_dir: Path
    out_dir: Path
    mode: str = "bounded_search"
    min_train_anchors: int = 60
    min_validation_anchors: int = 10
    min_validation_groups: int = 50
    embargo_days: int = 3
    base_cost_bps: float = 10.0
    max_base_candidates: int = 8


def ensure_under(path: Path, allowed: Path, repo_root: Path = REPO_ROOT, label: str = "path") -> Path:
    resolved = resolve_repo_path(path, repo_root).resolve()
    allowed_resolved = resolve_repo_path(allowed, repo_root).resolve()
    try:
        resolved.relative_to(allowed_resolved)
    except ValueError as exc:
        raise RiskOverlayOptimizerError(f"{label} must be under {allowed}") from exc
    return resolved


def resolve_data_lake(path: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    return ensure_under(path, ALLOWED_DATA_LAKE, repo_root, "data-lake") if enforce else resolve_repo_path(path, repo_root).resolve()


def resolve_v0_dir(path: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    return ensure_under(path, ALLOWED_V0_DIR, repo_root, "v0-dir") if enforce else resolve_repo_path(path, repo_root).resolve()


def resolve_out_dir(path: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    return ensure_under(path, ALLOWED_OUT_DIR, repo_root, "out-dir") if enforce else resolve_repo_path(path, repo_root).resolve()


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_feature_rows(data_lake: Path) -> pd.DataFrame:
    frame = read_csv_if_exists(data_lake / "long_history_feature_rows.csv")
    if not frame.empty:
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["etf_code"] = frame["etf_code"].astype(str).str.zfill(6)
        frame["datetime"] = pd.to_datetime(frame["datetime"], errors="coerce")
    return frame


def load_bars(data_lake: Path) -> pd.DataFrame:
    path = data_lake / "long_history_5m_bars.csv"
    if not path.exists():
        path = data_lake / "long_history_bars.csv"
    bars = read_csv_if_exists(path)
    if not bars.empty:
        bars["trade_date"] = bars["trade_date"].astype(str)
        bars["etf_code"] = bars["etf_code"].astype(str).str.zfill(6)
        bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
        bars = bars.sort_values(["etf_code", "datetime"]).reset_index(drop=True)
    return bars


def parse_candidate_id(candidate_id: str) -> dict[str, Any]:
    parts = candidate_id.split("|")
    if len(parts) < 8:
        raise RiskOverlayOptimizerError(f"invalid candidate_id: {candidate_id}")
    return {
        "candidate_id": candidate_id,
        "signal_clock": parts[0],
        "feature_set": parts[1],
        "label_policy": parts[2],
        "holding_period": parts[3],
        "entry_rule": parts[4],
        "exit_rule": parts[5],
        "cost_bps": float(str(parts[6]).removesuffix("bps")),
        "model_family": parts[7],
    }


def select_top_rejected_candidates(leaderboard: pd.DataFrame, top_n: int = 10) -> list[dict[str, Any]]:
    if leaderboard.empty:
        return []
    rejected = leaderboard[leaderboard["candidate_status"] != LAB_CANDIDATE_STATUS].copy()
    selected_ids: list[str] = []
    for metric in ["net_total_return", "calmar_like_ratio", "profit_factor"]:
        if metric in rejected.columns:
            ordered = rejected.sort_values(metric, ascending=False).head(top_n)
            selected_ids.extend(ordered["candidate_id"].astype(str).tolist())
    unique_ids = list(dict.fromkeys(selected_ids))
    rows = []
    for candidate_id in unique_ids:
        record = rejected[rejected["candidate_id"].astype(str) == candidate_id].iloc[0].to_dict()
        rows.append({**parse_candidate_id(candidate_id), "v0_metrics": record})
    return rows


def feature_columns_for_set(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise RiskOverlayOptimizerError(f"unknown feature_set: {feature_set}")
    return list(FEATURE_SETS[feature_set])


def finite_candidate_rows(frame: pd.DataFrame, candidate: dict[str, Any], dates: Sequence[str]) -> pd.DataFrame:
    columns = feature_columns_for_set(candidate["feature_set"])
    needed = ["trade_date", "etf_code", "signal_clock", candidate["label_policy"], "future_return_1d", "future_return_3d", "max_drawdown_3d"] + columns
    subset = frame[(frame["trade_date"].isin(dates)) & (frame["signal_clock"] == candidate["signal_clock"])].copy()
    if any(column not in subset.columns for column in needed):
        return pd.DataFrame(columns=needed)
    return subset.dropna(subset=[candidate["label_policy"], "future_return_3d"] + columns)


def score_base_candidate(feature_rows: pd.DataFrame, folds: Sequence[dict[str, Any]], candidate: dict[str, Any]) -> pd.DataFrame:
    scored_frames: list[pd.DataFrame] = []
    for fold in folds:
        if fold.get("skipped"):
            continue
        train = finite_candidate_rows(feature_rows, candidate, fold["train_anchor_dates"])
        validation = finite_candidate_rows(feature_rows, candidate, fold["validation_anchor_dates"])
        if train.empty or validation.empty:
            continue
        probabilities, _error = fit_predict_proba(train, validation, candidate)
        if probabilities is None:
            continue
        scored = validation.copy()
        scored["score"] = probabilities
        scored["fold_id"] = fold["fold_id"]
        scored["validation_month"] = fold["validation_month"]
        scored["candidate_id"] = candidate["candidate_id"]
        scored_frames.append(scored)
    return pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame()


def daily_close_table(bars: pd.DataFrame) -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["etf_code", "trade_date", "daily_close", "daily_amount", "daily_return"])
    daily = (
        bars.sort_values("datetime")
        .groupby(["etf_code", "trade_date"], as_index=False)
        .agg(daily_close=("close", "last"), daily_amount=("amount", "sum"))
        .sort_values(["etf_code", "trade_date"])
    )
    daily["etf_code"] = daily["etf_code"].astype(str).str.zfill(6)
    daily["daily_return"] = daily.groupby("etf_code")["daily_close"].pct_change()
    return daily


def append_holding_returns(scored: pd.DataFrame, daily: pd.DataFrame, max_days: int = 3) -> pd.DataFrame:
    if scored.empty:
        return scored
    output = scored.copy()
    by_code = {code: group.reset_index(drop=True) for code, group in daily.groupby("etf_code")}
    index_by_code_date = {(str(code), str(row["trade_date"])): idx for code, group in by_code.items() for idx, row in group.iterrows()}
    returns: dict[int, list[float | None]] = {day: [] for day in range(1, max_days + 1)}
    for row in output.itertuples(index=False):
        code = str(getattr(row, "etf_code"))
        trade_date = str(getattr(row, "trade_date"))
        group = by_code.get(code)
        idx = index_by_code_date.get((code, trade_date))
        entry_close = None if group is None or idx is None else float(group.iloc[idx]["daily_close"])
        for day in range(1, max_days + 1):
            exit_close = None if group is None or idx is None or idx + day >= len(group) else float(group.iloc[idx + day]["daily_close"])
            returns[day].append(None if entry_close in (None, 0) or exit_close is None else exit_close / entry_close - 1.0)
    for day, values in returns.items():
        output[f"future_return_{day}d_overlay"] = values
    return output


def add_entry_prices(scored: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    if scored.empty or bars.empty:
        return scored
    key_bars = bars[["etf_code", "datetime", "close"]].rename(columns={"close": "entry_price"}).copy()
    output = scored.copy()
    output["datetime"] = pd.to_datetime(output["datetime"], errors="coerce")
    return output.merge(key_bars, on=["etf_code", "datetime"], how="left")


def first_path_exit_return(
    bars: pd.DataFrame,
    *,
    etf_code: str,
    entry_time: pd.Timestamp,
    entry_price: float,
    holding_days: int,
    stop_loss: float | None,
    take_profit: float | None,
) -> tuple[float | None, str, str]:
    if bars.empty or not isinstance(entry_time, pd.Timestamp) or entry_price in (0, None) or pd.isna(entry_price):
        return None, "missing_path", ""
    horizon = entry_time + timedelta(days=max(1, holding_days) + 3)
    path = bars[(bars["etf_code"].astype(str) == str(etf_code)) & (bars["datetime"] > entry_time) & (bars["datetime"] <= horizon)].copy()
    if path.empty:
        return None, "missing_path", ""
    entry_day = entry_time.date().isoformat()
    future_days = sorted(day for day in path["trade_date"].astype(str).unique() if day > entry_day)
    if future_days:
        allowed_days = set(future_days[:holding_days])
        path = path[path["trade_date"].astype(str).isin(allowed_days)]
    if path.empty:
        return None, "missing_path", ""
    for row in path.itertuples(index=False):
        close = float(getattr(row, "close"))
        ret = close / entry_price - 1.0
        if stop_loss is not None and ret <= stop_loss:
            return ret, "stop_loss", str(getattr(row, "datetime"))
        if take_profit is not None and ret >= take_profit:
            return ret, "take_profit", str(getattr(row, "datetime"))
    final = path.iloc[-1]
    return float(final["close"]) / entry_price - 1.0, "time_exit", str(final["datetime"])


def train_only_regime_thresholds(daily: pd.DataFrame, train_dates: Sequence[str]) -> dict[str, float]:
    train = daily[daily["trade_date"].astype(str).isin(set(train_dates))].copy()
    if train.empty:
        return {"vol_q75": math.inf, "amount_q25": -math.inf}
    train["trailing_vol20"] = train.groupby("etf_code")["daily_return"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    train["trailing_amount20"] = train.groupby("etf_code")["daily_amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    return {
        "vol_q75": float(train["trailing_vol20"].dropna().quantile(0.75)) if train["trailing_vol20"].notna().any() else math.inf,
        "amount_q25": float(train["trailing_amount20"].dropna().quantile(0.25)) if train["trailing_amount20"].notna().any() else -math.inf,
    }


def append_past_regime_metrics(scored: pd.DataFrame, daily: pd.DataFrame, folds: Sequence[dict[str, Any]]) -> pd.DataFrame:
    if scored.empty or daily.empty:
        return scored
    daily_metrics = daily.copy()
    daily_metrics["trailing_vol20"] = daily_metrics.groupby("etf_code")["daily_return"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).std())
    daily_metrics["trailing_amount20"] = daily_metrics.groupby("etf_code")["daily_amount"].transform(lambda s: s.shift(1).rolling(20, min_periods=5).mean())
    fold_thresholds = {fold["fold_id"]: train_only_regime_thresholds(daily, fold["train_anchor_dates"]) for fold in folds if not fold.get("skipped")}
    output = scored.merge(
        daily_metrics[["etf_code", "trade_date", "trailing_vol20", "trailing_amount20"]],
        on=["etf_code", "trade_date"],
        how="left",
    )
    output["train_only_vol_q75"] = output["fold_id"].map(lambda value: fold_thresholds.get(value, {}).get("vol_q75", math.inf))
    output["train_only_amount_q25"] = output["fold_id"].map(lambda value: fold_thresholds.get(value, {}).get("amount_q25", -math.inf))
    output["regime_threshold_scope"] = "train_only_by_fold"
    return output


def overlay_grid() -> list[dict[str, Any]]:
    base = {
        "top_k_per_day": 1,
        "min_probability": 0.50,
        "max_daily_sleeve_exposure": 0.25,
        "max_total_exposure": 0.75,
        "max_etf_weight_per_sleeve": 0.25,
        "holding_period": "3d",
        "stop_loss": None,
        "take_profit": None,
        "volatility_filter": "none",
        "liquidity_filter": "none",
        "drawdown_throttle": None,
        "threshold_search_lab_only": True,
        "diagnostic_only": True,
    }
    configs = [base]
    for top_k in [2, 3, "all"]:
        configs.append({**base, "top_k_per_day": top_k})
    for threshold in [0.55, 0.60, 0.65]:
        configs.append({**base, "min_probability": threshold})
    for holding in ["1d", "2d"]:
        configs.append({**base, "holding_period": holding})
    for stop_loss in [-0.015, -0.025, -0.04]:
        configs.append({**base, "stop_loss": stop_loss})
    for take_profit in [0.02, 0.035, 0.05]:
        configs.append({**base, "take_profit": take_profit})
    configs.append({**base, "volatility_filter": "skip_highest_vol_quartile"})
    configs.append({**base, "liquidity_filter": "skip_lowest_liquidity_quartile"})
    for throttle in [-0.05, -0.10]:
        configs.append({**base, "drawdown_throttle": throttle})
    configs.append({**base, "top_k_per_day": 2, "min_probability": 0.55, "holding_period": "2d", "stop_loss": -0.025, "take_profit": 0.035})
    return configs


def apply_top_k_and_filters(scored: pd.DataFrame, overlay: dict[str, Any]) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    rows = scored[scored["score"] >= float(overlay["min_probability"])].copy()
    if overlay.get("volatility_filter") == "skip_highest_vol_quartile":
        rows = rows[(rows["trailing_vol20"].isna()) | (rows["trailing_vol20"] <= rows["train_only_vol_q75"])]
    if overlay.get("liquidity_filter") == "skip_lowest_liquidity_quartile":
        rows = rows[(rows["trailing_amount20"].isna()) | (rows["trailing_amount20"] >= rows["train_only_amount_q25"])]
    rows = rows.sort_values(["trade_date", "score"], ascending=[True, False])
    top_k = overlay["top_k_per_day"]
    if top_k != "all":
        rows = rows.groupby("trade_date", as_index=False).head(int(top_k))
    return rows


def apply_exposure_and_returns(trades: pd.DataFrame, bars: pd.DataFrame, overlay: dict[str, Any], cost_bps: float) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    output = trades.copy().sort_values(["trade_date", "score"], ascending=[True, False]).reset_index(drop=True)
    holding_days = int(str(overlay["holding_period"]).removesuffix("d"))
    daily_counts = output.groupby("trade_date")["etf_code"].transform("count").clip(lower=1)
    per_trade_daily_cap = float(overlay["max_daily_sleeve_exposure"]) / daily_counts.astype(float)
    per_trade_total_cap = float(overlay["max_total_exposure"]) / daily_counts.astype(float)
    output["sleeve_weight"] = np.minimum.reduce(
        [
            per_trade_daily_cap.to_numpy(),
            per_trade_total_cap.to_numpy(),
            np.full(len(output), float(overlay["max_etf_weight_per_sleeve"])),
        ]
    )
    raw_returns: list[float | None] = []
    exit_reasons: list[str] = []
    exit_times: list[str] = []
    stop_loss = overlay.get("stop_loss")
    take_profit = overlay.get("take_profit")
    bars_by_code = {str(code): group for code, group in bars.groupby("etf_code")} if not bars.empty and (stop_loss is not None or take_profit is not None) else {}
    for row in output.itertuples(index=False):
        if stop_loss is not None or take_profit is not None:
            code = str(getattr(row, "etf_code"))
            path_return, reason, exit_time = first_path_exit_return(
                bars_by_code.get(code, pd.DataFrame()),
                etf_code=code,
                entry_time=getattr(row, "datetime"),
                entry_price=float(getattr(row, "entry_price")) if not pd.isna(getattr(row, "entry_price")) else math.nan,
                holding_days=holding_days,
                stop_loss=stop_loss,
                take_profit=take_profit,
            )
            raw_returns.append(path_return)
            exit_reasons.append(reason)
            exit_times.append(exit_time)
        else:
            value = getattr(row, f"future_return_{holding_days}d_overlay", None)
            raw_returns.append(None if pd.isna(value) else float(value))
            exit_reasons.append("time_exit")
            exit_times.append("")
    output["raw_return"] = raw_returns
    output["exit_reason"] = exit_reasons
    output["exit_time"] = exit_times
    output = output.dropna(subset=["raw_return"]).copy()
    output["gross_return"] = output["raw_return"].astype(float) * output["sleeve_weight"]
    output["cost_paid"] = 2.0 * float(cost_bps) / 10000.0 * output["sleeve_weight"]
    output["net_return"] = output["gross_return"] - output["cost_paid"]
    output["month"] = output["trade_date"].astype(str).str[:7]
    throttle = overlay.get("drawdown_throttle")
    if throttle is not None:
        output = apply_drawdown_throttle(output, float(throttle))
    return output


def apply_drawdown_throttle(trades: pd.DataFrame, threshold: float) -> pd.DataFrame:
    output = trades.copy().sort_values("trade_date").reset_index(drop=True)
    nav = 1.0
    high = 1.0
    throttled_returns = []
    throttle_flags = []
    for row in output.itertuples(index=False):
        drawdown_before_trade = nav / high - 1.0
        factor = 0.5 if drawdown_before_trade <= threshold else 1.0
        throttle_flags.append(factor < 1.0)
        ret = float(getattr(row, "net_return")) * factor
        throttled_returns.append(ret)
        nav *= 1.0 + ret
        high = max(high, nav)
    output["drawdown_throttle_applied"] = throttle_flags
    output["net_return"] = throttled_returns
    return output


def max_drawdown_period(trades: pd.DataFrame) -> tuple[str, str, float]:
    if trades.empty:
        return "", "", 0.0
    nav = 1.0
    peak = 1.0
    peak_date = ""
    worst = 0.0
    worst_start = ""
    worst_end = ""
    for row in trades.sort_values("trade_date").itertuples(index=False):
        nav *= 1.0 + float(getattr(row, "net_return"))
        date = str(getattr(row, "trade_date"))
        if nav > peak:
            peak = nav
            peak_date = date
        drawdown = nav / peak - 1.0
        if drawdown < worst:
            worst = drawdown
            worst_start = peak_date
            worst_end = date
    return worst_start, worst_end, worst


def contribution_table(trades: pd.DataFrame, group_column: str) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[group_column, "net_return_sum", "trade_count", "loss_sum"])
    table = (
        trades.assign(loss_component=trades["net_return"].clip(upper=0.0))
        .groupby(group_column, as_index=False)
        .agg(net_return_sum=("net_return", "sum"), trade_count=("net_return", "size"), loss_sum=("loss_component", "sum"))
        .sort_values("net_return_sum")
    )
    return table


def attribution_row(candidate: dict[str, Any], trades: pd.DataFrame, v0_row: dict[str, Any], gate_reasons: Sequence[str]) -> dict[str, Any]:
    metrics = pnl_metrics(trades, trades, float(candidate.get("cost_bps", 10.0)))
    start, end, period_dd = max_drawdown_period(trades)
    return {
        "candidate_id": candidate["candidate_id"],
        "net_total_return": metrics["net_total_return"],
        "max_drawdown": metrics["max_drawdown"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "monthly_win_rate": metrics["monthly_win_rate"],
        "positive_month_fraction": metrics["monthly_win_rate"],
        "worst_month": metrics["worst_month"],
        "best_month": metrics["best_month"],
        "worst_drawdown_period": f"{start}..{end}:{period_dd:.6f}" if start or end else "",
        "etf_contribution": json.dumps(metrics["etf_contribution"], ensure_ascii=False, sort_keys=True),
        "month_contribution": json.dumps(contribution_table(trades, "month").set_index("month")["net_return_sum"].to_dict(), ensure_ascii=False, sort_keys=True)
        if not trades.empty
        else "{}",
        "signal_clock_contribution": json.dumps(contribution_table(trades, "signal_clock").set_index("signal_clock")["net_return_sum"].to_dict(), ensure_ascii=False, sort_keys=True)
        if not trades.empty
        else "{}",
        "holding_period_contribution": candidate.get("holding_period", ""),
        "cost_impact": metrics["cost_paid"],
        "rejection_reason": "|".join(gate_reasons),
        "whether_return_promising": bool(float(v0_row.get("net_total_return", metrics["net_total_return"])) > 0),
        "whether_drawdown_problem": bool(metrics["max_drawdown"] < -0.25),
        "whether_monthly_stability_problem": bool(metrics["monthly_win_rate"] < 0.55),
    }


def evaluate_overlay(scored: pd.DataFrame, bars: pd.DataFrame, overlay: dict[str, Any], base_candidate: dict[str, Any], v0_top_drawdown: float, cost_bps: float) -> dict[str, Any]:
    filtered = apply_top_k_and_filters(scored, overlay)
    trades = apply_exposure_and_returns(filtered, bars, overlay, cost_bps)
    metrics = pnl_metrics(trades, scored, cost_bps)
    candidate_passed, reasons = gate_candidate_v1(metrics, v0_top_drawdown=v0_top_drawdown, leakage_ok=True, artifact_saved=False)
    promising = metrics["net_total_return"] > 0 and metrics["profit_factor"] > 1.05 and metrics["win_rate"] > 0.50
    if candidate_passed:
        status = LAB_CANDIDATE_STATUS
    elif promising and metrics["max_drawdown"] < -0.25:
        status = PROMISING_STATUS
    else:
        status = REJECTED_STATUS
    result_id = f"{base_candidate['candidate_id']}|overlay={overlay_name(overlay)}"
    return {
        "overlay_candidate_id": result_id,
        "base_candidate_id": base_candidate["candidate_id"],
        "candidate_status": status,
        "candidate_gate_passed": candidate_passed,
        "candidate_gate_reasons": reasons,
        "overlay": overlay,
        "pnl_metrics": metrics,
        "trade_count": int(len(trades)),
        "trades": trades,
    }


def overlay_name(overlay: dict[str, Any]) -> str:
    return (
        f"k{overlay['top_k_per_day']}_p{overlay['min_probability']}_h{overlay['holding_period']}"
        f"_sl{overlay['stop_loss']}_tp{overlay['take_profit']}_vol{overlay['volatility_filter']}"
        f"_liq{overlay['liquidity_filter']}_thr{overlay['drawdown_throttle']}"
    )


def gate_candidate_v1(metrics: dict[str, Any], *, v0_top_drawdown: float, leakage_ok: bool, artifact_saved: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics.get("net_total_return", 0.0) <= 0:
        reasons.append("net_total_return_not_positive")
    drawdown = float(metrics.get("max_drawdown", 0.0))
    baseline_abs = abs(float(v0_top_drawdown))
    improved_25pct = baseline_abs > 0 and abs(drawdown) <= baseline_abs * 0.75
    if not (improved_25pct or drawdown >= -0.25):
        reasons.append("drawdown_not_improved_25pct_or_below_25pct")
    if metrics.get("win_rate", 0.0) <= 0.50:
        reasons.append("win_rate_not_above_50pct")
    if metrics.get("profit_factor", 0.0) <= 1.05:
        reasons.append("profit_factor_not_above_1_05")
    if metrics.get("monthly_win_rate", 0.0) < 0.55:
        reasons.append("monthly_win_rate_below_55pct")
    if metrics.get("monthly_win_rate", 0.0) < 0.55:
        reasons.append("positive_month_fraction_below_55pct")
    if metrics.get("month_concentration", 0.0) > 0.70:
        reasons.append("month_concentration_too_high")
    if metrics.get("etf_concentration", 0.0) > 0.70:
        reasons.append("etf_concentration_too_high")
    if not leakage_ok:
        reasons.append("leakage_risk")
    if artifact_saved:
        reasons.append("artifact_saved")
    return not reasons, reasons


def composite_rank(leaderboard: pd.DataFrame) -> pd.DataFrame:
    if leaderboard.empty:
        return leaderboard
    output = leaderboard.copy()
    output["rank_net_total_return"] = output["net_total_return"].rank(ascending=True, pct=True)
    output["rank_calmar_like_ratio"] = output["calmar_like_ratio"].rank(ascending=True, pct=True)
    output["rank_profit_factor"] = output["profit_factor"].replace(math.inf, np.nan).fillna(output["profit_factor"].replace(math.inf, np.nan).max()).rank(ascending=True, pct=True)
    output["rank_monthly_win_rate"] = output["monthly_win_rate"].rank(ascending=True, pct=True)
    output["penalty_max_drawdown"] = output["max_drawdown"].abs()
    output["penalty_month_concentration"] = output["month_concentration"]
    output["penalty_etf_concentration"] = output["etf_concentration"]
    output["composite_score"] = (
        output["rank_net_total_return"]
        + output["rank_calmar_like_ratio"]
        + output["rank_profit_factor"]
        + output["rank_monthly_win_rate"]
        - output["penalty_max_drawdown"]
        - output["penalty_month_concentration"]
        - output["penalty_etf_concentration"]
    )
    return output.sort_values(["candidate_status", "composite_score", "net_total_return"], ascending=[True, False, False])


def result_to_row(result: dict[str, Any]) -> dict[str, Any]:
    metrics = result["pnl_metrics"]
    overlay = result["overlay"]
    return {
        "overlay_candidate_id": result["overlay_candidate_id"],
        "base_candidate_id": result["base_candidate_id"],
        "candidate_status": result["candidate_status"],
        "top_k_per_day": overlay["top_k_per_day"],
        "min_probability": overlay["min_probability"],
        "holding_period": overlay["holding_period"],
        "stop_loss": overlay["stop_loss"],
        "take_profit": overlay["take_profit"],
        "volatility_filter": overlay["volatility_filter"],
        "liquidity_filter": overlay["liquidity_filter"],
        "drawdown_throttle": overlay["drawdown_throttle"],
        "net_total_return": metrics["net_total_return"],
        "annualized_return": metrics["annualized_return"],
        "max_drawdown": metrics["max_drawdown"],
        "calmar_like_ratio": metrics["calmar_like_ratio"],
        "win_rate": metrics["win_rate"],
        "profit_factor": metrics["profit_factor"],
        "monthly_win_rate": metrics["monthly_win_rate"],
        "positive_month_fraction": metrics["monthly_win_rate"],
        "turnover": metrics["turnover"],
        "average_exposure": metrics["average_exposure"],
        "cost_paid": metrics["cost_paid"],
        "month_concentration": metrics["month_concentration"],
        "etf_concentration": metrics["etf_concentration"],
        "trade_count": metrics["trade_count"],
        "candidate_gate_reasons": "|".join(result["candidate_gate_reasons"]),
    }


def order_intent_files_present(out_dir: Path) -> list[str]:
    if not out_dir.exists():
        return []
    return [str(path) for path in out_dir.rglob("*") if path.is_file() and any(token in path.name.lower() for token in ORDER_INTENT_TOKENS)]


def cost_sensitivity(results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for result in results[:20]:
        trades = result["trades"]
        if trades.empty:
            continue
        for bps in [0, 5, 10, 20]:
            adjusted = trades.copy()
            weight = adjusted.get("sleeve_weight", pd.Series([1.0] * len(adjusted))).astype(float)
            adjusted["cost_paid"] = 2.0 * bps / 10000.0 * weight
            adjusted["net_return"] = adjusted["gross_return"].astype(float) - adjusted["cost_paid"]
            metrics = pnl_metrics(adjusted, adjusted, bps)
            rows.append(
                {
                    "overlay_candidate_id": result["overlay_candidate_id"],
                    "cost_bps_per_side": bps,
                    "net_total_return": metrics["net_total_return"],
                    "max_drawdown": metrics["max_drawdown"],
                    "profit_factor": metrics["profit_factor"],
                }
            )
    return pd.DataFrame(rows)


def decision_from_results(results: Sequence[dict[str, Any]], folds: Sequence[dict[str, Any]], artifacts: Sequence[str], order_intents: Sequence[str]) -> str:
    if artifacts or order_intents:
        return DECISION_RUNTIME
    if not folds:
        return DECISION_MISSING
    if any(not fold.get("train_validation_no_overlap", True) or not fold.get("embargo_ok", True) for fold in folds):
        return DECISION_LEAKAGE
    if any(result["candidate_status"] == LAB_CANDIDATE_STATUS for result in results):
        return DECISION_CANDIDATES
    if any(result["candidate_status"] == PROMISING_STATUS for result in results):
        return DECISION_PROMISING
    return DECISION_NO_CANDIDATES


def write_report_md(path: Path, decision: dict[str, Any], leaderboard: pd.DataFrame, attribution: pd.DataFrame) -> None:
    lines = [
        "# AETF Q3 Lab Long-History Alpha Risk Overlay Optimizer",
        "",
        LAB_DECLARATION,
        "",
        "Lab-only no-save research. Not Stable evidence. No QMT. No OrderIntent.",
        "",
        f"- decision: {decision['decision']}",
        f"- candidate_found_count: {decision['candidate_found_count']}",
        f"- promising_but_drawdown_uncontrolled_count: {decision['promising_but_drawdown_uncontrolled_count']}",
        f"- stable_promotion_ready: {decision['stable_promotion_ready']}",
        "",
        "## Top Rejected Attribution",
        "",
        f"- attributed_candidates: {len(attribution)}",
        "",
        "## Risk Overlay Leaderboard",
        "",
    ]
    for row in leaderboard.head(5).to_dict("records"):
        lines.append(
            f"- {row['candidate_status']} | net={row['net_total_return']:.6f} | dd={row['max_drawdown']:.6f} | "
            f"win={row['win_rate']:.4f} | monthly={row['monthly_win_rate']:.4f} | {row['overlay_candidate_id']}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_risk_overlay_optimizer(config: RiskOverlayConfig, repo_root: Path = REPO_ROOT, *, enforce_paths: bool = True) -> dict[str, Any]:
    created_at_utc = datetime.now(timezone.utc).isoformat()
    data_lake = resolve_data_lake(config.data_lake, repo_root, enforce=enforce_paths)
    v0_dir = resolve_v0_dir(config.v0_dir, repo_root, enforce=enforce_paths)
    out_dir = resolve_out_dir(config.out_dir, repo_root, enforce=enforce_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_error = ""
    try:
        leaderboard_v0 = read_csv_if_exists(v0_dir / "long_history_candidate_leaderboard.csv")
        feature_rows = load_feature_rows(data_lake)
        bars = load_bars(data_lake)
        if leaderboard_v0.empty or feature_rows.empty or bars.empty:
            raise RiskOverlayOptimizerError("missing v0 leaderboard, feature rows, or bars")
        top_candidates = select_top_rejected_candidates(leaderboard_v0)[: config.max_base_candidates]
        folds = make_rolling_folds(
            feature_rows,
            min_train_anchors=config.min_train_anchors,
            min_validation_anchors=config.min_validation_anchors,
            min_validation_groups=config.min_validation_groups,
            embargo_days=config.embargo_days,
        )
        daily = daily_close_table(bars)
        results: list[dict[str, Any]] = []
        attribution_rows: list[dict[str, Any]] = []
        month_frames: list[pd.DataFrame] = []
        etf_frames: list[pd.DataFrame] = []
        clock_frames: list[pd.DataFrame] = []
        v0_top_drawdown = float(leaderboard_v0.sort_values("net_total_return", ascending=False).iloc[0]["max_drawdown"])
        for candidate in top_candidates:
            scored = score_base_candidate(feature_rows, folds, candidate)
            if scored.empty:
                continue
            scored = append_holding_returns(add_entry_prices(scored, bars), daily, 3)
            scored = append_past_regime_metrics(scored, daily, folds)
            base_overlay = overlay_grid()[0]
            base_trades = apply_exposure_and_returns(apply_top_k_and_filters(scored, base_overlay), bars, base_overlay, float(config.base_cost_bps))
            base_metrics = pnl_metrics(base_trades, scored, config.base_cost_bps)
            _passed, base_reasons = gate_candidate_v1(base_metrics, v0_top_drawdown=v0_top_drawdown, leakage_ok=True, artifact_saved=False)
            attribution_rows.append(attribution_row(candidate, base_trades, candidate["v0_metrics"], base_reasons))
            month_table = contribution_table(base_trades, "month")
            month_table["candidate_id"] = candidate["candidate_id"]
            month_frames.append(month_table)
            etf_table = contribution_table(base_trades, "etf_code")
            etf_table["candidate_id"] = candidate["candidate_id"]
            etf_frames.append(etf_table)
            clock_table = contribution_table(base_trades, "signal_clock")
            clock_table["candidate_id"] = candidate["candidate_id"]
            clock_frames.append(clock_table)
            for overlay in overlay_grid():
                results.append(evaluate_overlay(scored, bars, overlay, candidate, v0_top_drawdown, config.base_cost_bps))
        artifacts = model_artifacts_present(out_dir)
        order_intents = order_intent_files_present(out_dir)
        decision_value = decision_from_results(results, folds, artifacts, order_intents)
    except RiskOverlayOptimizerError as exc:
        feature_rows = pd.DataFrame()
        top_candidates = []
        folds = []
        results = []
        attribution_rows = []
        month_frames = []
        etf_frames = []
        clock_frames = []
        artifacts = model_artifacts_present(out_dir)
        order_intents = order_intent_files_present(out_dir)
        runtime_error = str(exc)
        decision_value = DECISION_MISSING
    except Exception as exc:  # noqa: BLE001
        feature_rows = pd.DataFrame()
        top_candidates = []
        folds = []
        results = []
        attribution_rows = []
        month_frames = []
        etf_frames = []
        clock_frames = []
        artifacts = model_artifacts_present(out_dir)
        order_intents = order_intent_files_present(out_dir)
        runtime_error = str(exc)
        decision_value = DECISION_RUNTIME
    result_rows = [result_to_row(result) for result in results]
    leaderboard = composite_rank(pd.DataFrame(result_rows))
    attribution = pd.DataFrame(attribution_rows)
    attribution.to_csv(out_dir / "top_rejected_candidate_attribution.csv", index=False, lineterminator="\n")
    pd.concat(month_frames, ignore_index=True).to_csv(out_dir / "drawdown_month_attribution.csv", index=False, lineterminator="\n") if month_frames else pd.DataFrame().to_csv(out_dir / "drawdown_month_attribution.csv", index=False)
    pd.concat(etf_frames, ignore_index=True).to_csv(out_dir / "drawdown_etf_attribution.csv", index=False, lineterminator="\n") if etf_frames else pd.DataFrame().to_csv(out_dir / "drawdown_etf_attribution.csv", index=False)
    pd.concat(clock_frames, ignore_index=True).to_csv(out_dir / "drawdown_clock_attribution.csv", index=False, lineterminator="\n") if clock_frames else pd.DataFrame().to_csv(out_dir / "drawdown_clock_attribution.csv", index=False)
    leaderboard.to_csv(out_dir / "risk_overlay_candidate_leaderboard.csv", index=False, lineterminator="\n")
    leaderboard.to_csv(out_dir / "risk_overlay_candidate_pnl_summary.csv", index=False, lineterminator="\n")
    risk_columns = [
        "overlay_candidate_id",
        "candidate_status",
        "max_drawdown",
        "month_concentration",
        "etf_concentration",
        "candidate_gate_reasons",
        "composite_score",
    ]
    leaderboard[[column for column in risk_columns if column in leaderboard.columns]].to_csv(out_dir / "risk_overlay_candidate_risk_summary.csv", index=False, lineterminator="\n")
    cost_frame = cost_sensitivity(sorted(results, key=lambda item: item["pnl_metrics"]["net_total_return"], reverse=True))
    cost_frame.to_csv(out_dir / "risk_overlay_cost_sensitivity.csv", index=False, lineterminator="\n")
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "risk_overlay_optimizer_report",
        "created_at_utc": created_at_utc,
        "data_lake": str(data_lake),
        "v0_dir": str(v0_dir),
        "feature_row_count": int(len(feature_rows)),
        "top_rejected_candidate_count": len(top_candidates),
        "risk_overlay_candidate_count": len(results),
        "fold_count": len(folds),
        "usable_fold_count": int(sum(1 for fold in folds if not fold.get("skipped"))),
        "threshold_search_lab_only": True,
        "primary_split": "rolling_origin_walk_forward",
        "composite_score_formula": "rank(net_total_return)+rank(Calmar-like)+rank(profit_factor)+rank(monthly_win_rate)-penalty(max_drawdown)-penalty(month_concentration)-penalty(etf_concentration)",
        "future_label_or_outcome_in_feature_columns": bool(set(TIME_CENSORED_FEATURES) & (set(LABEL_COLUMNS) | set(OUTCOME_COLUMNS))),
        "top_leaderboard": leaderboard.head(25).to_dict("records") if not leaderboard.empty else [],
        "runtime_error": runtime_error,
        "model_artifacts_detected": artifacts,
        "order_intent_files_detected": order_intents,
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "risk_overlay_optimizer_report.json", report)
    decision = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "risk_overlay_decision",
        "created_at_utc": created_at_utc,
        "decision": decision_value,
        "candidate_found_count": int(sum(1 for result in results if result["candidate_status"] == LAB_CANDIDATE_STATUS)),
        "promising_but_drawdown_uncontrolled_count": int(sum(1 for result in results if result["candidate_status"] == PROMISING_STATUS)),
        "stable_promotion_ready": False,
        "stable_evidence": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "order_intent_generated": False,
        "formal_training": False,
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        **{
            key: value
            for key, value in BOUNDARY_FIELDS.items()
            if key
            not in {
                "stable_promotion_ready",
                "stable_evidence",
                "qmt_ready",
                "order_intent_ready",
                "automatic_promotion_ready",
                "formal_training",
                "model_saved",
                "scaler_saved",
                "checkpoint_saved",
            }
        },
    }
    write_json(out_dir / "risk_overlay_decision.json", decision)
    write_report_md(out_dir / "risk_overlay_optimizer_report.md", decision, leaderboard, attribution)
    return {**decision, "report": report}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab-only long-history intraday risk overlay optimizer.")
    parser.add_argument("--data-lake", type=Path, default=ALLOWED_DATA_LAKE)
    parser.add_argument("--v0-dir", type=Path, default=ALLOWED_V0_DIR)
    parser.add_argument("--out-dir", type=Path, default=ALLOWED_OUT_DIR)
    parser.add_argument("--mode", default="bounded_search", choices=["bounded_search"])
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_risk_overlay_optimizer(RiskOverlayConfig(args.data_lake, args.v0_dir, args.out_dir, args.mode))
    print(json.dumps(json_safe({key: value for key, value in report.items() if key != "report"}), ensure_ascii=False, indent=2))
    return 0 if not str(report["decision"]).endswith("BLOCKED_RUNTIME_ERROR") else 2


if __name__ == "__main__":
    raise SystemExit(main())
