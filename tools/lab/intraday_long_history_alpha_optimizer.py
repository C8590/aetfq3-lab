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

from tools.lab.intraday_long_history_data_lake import (  # noqa: E402
    BOUNDARY_FIELDS,
    LAB_DECLARATION,
    LABEL_COLUMNS,
    OUTCOME_COLUMNS,
    TIME_CENSORED_FEATURES,
    build_feature_rows,
    json_safe,
    resolve_repo_path,
    write_json,
)


REPORT_TYPE = "intraday_long_history_alpha_optimizer"
ALLOWED_DATA_LAKE = Path(".local_artifact_backup/aetfq3_lab_sources/intraday_long_history_data_lake")
ALLOWED_OUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_long_history_alpha_optimization")
DECISION_CANDIDATES = "LONG_HISTORY_ALPHA_OPTIMIZATION_COMPLETED_CANDIDATES_FOUND_REVIEW_REQUIRED"
DECISION_NO_CANDIDATES = "LONG_HISTORY_ALPHA_OPTIMIZATION_COMPLETED_NO_CANDIDATES_FOUND"
DECISION_WAITING = "LONG_HISTORY_ALPHA_OPTIMIZATION_BLOCKED_WAITING_LONG_HISTORY_PACKAGE"
DECISION_DATA_QUALITY = "LONG_HISTORY_ALPHA_OPTIMIZATION_BLOCKED_DATA_QUALITY"
DECISION_LEAKAGE = "LONG_HISTORY_ALPHA_OPTIMIZATION_BLOCKED_LEAKAGE_RISK"
DECISION_RUNTIME = "LONG_HISTORY_ALPHA_OPTIMIZATION_BLOCKED_RUNTIME_ERROR"
MODEL_ARTIFACT_SUFFIXES = {".pkl", ".joblib", ".pt", ".pth", ".ckpt", ".onnx"}
FEATURE_SETS = {
    "core_intraday_price": [
        "return_since_open",
        "close_now_vs_vwap_so_far",
        "intraday_volatility_so_far",
        "drawdown_so_far",
        "rebound_from_low_so_far",
    ],
    "price_volume_censored": [
        "return_since_open",
        "high_so_far_vs_open",
        "low_so_far_vs_open",
        "volume_so_far",
        "amount_so_far",
        "volume_acceleration_so_far",
    ],
    "full_time_censored_intraday": TIME_CENSORED_FEATURES,
}
DEFAULT_SIGNAL_CLOCKS = ["10:00", "10:30", "11:00", "11:30", "13:30", "14:00", "14:30", "14:50"]
DEFAULT_LABELS = ["ret_3d_gt_100bp", "safe_positive_3d"]
DEFAULT_MODEL_FAMILIES = ["dummy_most_frequent", "dummy_stratified", "logistic_balanced_scaled", "hist_gradient_boosting", "random_forest_shallow"]


class AlphaOptimizerError(RuntimeError):
    pass


@dataclass(frozen=True)
class OptimizerConfig:
    data_lake: Path
    out_dir: Path
    mode: str = "bounded_search"
    min_train_anchors: int = 60
    min_validation_anchors: int = 10
    min_validation_groups: int = 50
    embargo_days: int = 3
    cost_bps: float = 10.0
    max_candidates: int = 96


def ensure_under(path: Path, allowed: Path, repo_root: Path = REPO_ROOT, label: str = "path") -> Path:
    resolved = resolve_repo_path(path, repo_root).resolve()
    allowed_resolved = resolve_repo_path(allowed, repo_root).resolve()
    try:
        resolved.relative_to(allowed_resolved)
    except ValueError as exc:
        raise AlphaOptimizerError(f"{label} must be under {allowed}") from exc
    return resolved


def resolve_data_lake(path: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    return ensure_under(path, ALLOWED_DATA_LAKE, repo_root, "data-lake") if enforce else resolve_repo_path(path, repo_root).resolve()


def resolve_out_dir(path: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    return ensure_under(path, ALLOWED_OUT_DIR, repo_root, "out-dir") if enforce else resolve_repo_path(path, repo_root).resolve()


def load_feature_frame(data_lake: Path) -> pd.DataFrame:
    feature_path = data_lake / "long_history_feature_rows.csv"
    if feature_path.exists():
        frame = pd.read_csv(feature_path)
    else:
        bars_path = data_lake / "long_history_5m_bars.csv"
        if not bars_path.exists():
            bars_path = data_lake / "long_history_bars.csv"
        if not bars_path.exists():
            return pd.DataFrame()
        bars = pd.read_csv(bars_path)
        bars["datetime"] = pd.to_datetime(bars["datetime"], errors="coerce")
        frame = pd.DataFrame(build_feature_rows(bars))
    if frame.empty:
        return frame
    frame["trade_date"] = frame["trade_date"].astype(str)
    return frame


def model_artifacts_present(out_dir: Path) -> list[str]:
    if not out_dir.exists():
        return []
    return [str(path) for path in out_dir.rglob("*") if path.is_file() and path.suffix.lower() in MODEL_ARTIFACT_SUFFIXES]


def feature_columns_for_set(feature_set: str) -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise AlphaOptimizerError(f"unknown feature_set: {feature_set}")
    return list(FEATURE_SETS[feature_set])


def candidate_grid(cost_bps: float, max_candidates: int = 96) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for signal_clock in DEFAULT_SIGNAL_CLOCKS:
        for feature_set in FEATURE_SETS:
            for label_policy in DEFAULT_LABELS:
                for model_family in DEFAULT_MODEL_FAMILIES:
                    rows.append(
                        {
                            "candidate_id": f"{signal_clock}|{feature_set}|{label_policy}|3d|top1_prob50|close_after_holding|{cost_bps:g}bps|{model_family}",
                            "signal_clock": signal_clock,
                            "feature_set": feature_set,
                            "label_policy": label_policy,
                            "holding_period": "3d",
                            "entry_rule": "top1_per_day_probability_ge_0_50",
                            "exit_rule": "close_after_holding_period",
                            "cost_bps": float(cost_bps),
                            "model_family": model_family,
                        }
                    )
    return rows[:max_candidates]


def month_start(month: str) -> datetime:
    return datetime.strptime(month + "-01", "%Y-%m-%d")


def make_rolling_folds(
    feature_rows: pd.DataFrame,
    *,
    min_train_anchors: int = 60,
    min_validation_anchors: int = 10,
    min_validation_groups: int = 50,
    embargo_days: int = 3,
) -> list[dict[str, Any]]:
    if feature_rows.empty:
        return []
    dates = sorted(feature_rows["trade_date"].dropna().astype(str).unique())
    months = sorted({date[:7] for date in dates})
    folds: list[dict[str, Any]] = []
    for validation_month in months[1:]:
        validation_start = month_start(validation_month)
        train_cutoff = (validation_start - timedelta(days=embargo_days)).date().isoformat()
        train_dates = [date for date in dates if date <= train_cutoff]
        validation_dates = [date for date in dates if date.startswith(validation_month)]
        validation_rows = feature_rows[feature_rows["trade_date"].isin(validation_dates)]
        skip_reasons: list[str] = []
        if len(train_dates) < min_train_anchors:
            skip_reasons.append("min_train_anchors_not_met")
        if len(validation_dates) < min_validation_anchors:
            skip_reasons.append("min_validation_anchors_not_met")
        if len(validation_rows) < min_validation_groups:
            skip_reasons.append("min_validation_groups_not_met")
        no_overlap = not (set(train_dates) & set(validation_dates))
        embargo_ok = not train_dates or max(train_dates) <= train_cutoff
        if not no_overlap:
            skip_reasons.append("train_validation_overlap")
        if not embargo_ok:
            skip_reasons.append("embargo_failed")
        folds.append(
            {
                "fold_id": f"train_to_{train_cutoff}_validate_{validation_month}",
                "train_window_type": "expanding",
                "validation_month": validation_month,
                "embargo_days": embargo_days,
                "train_cutoff": train_cutoff,
                "train_anchor_dates": train_dates,
                "validation_anchor_dates": validation_dates,
                "train_anchor_count": len(train_dates),
                "validation_anchor_count": len(validation_dates),
                "validation_group_count": int(len(validation_rows)),
                "train_validation_no_overlap": no_overlap,
                "embargo_ok": embargo_ok,
                "skipped": bool(skip_reasons),
                "skip_reasons": skip_reasons,
            }
        )
    return folds


def finite_candidate_rows(frame: pd.DataFrame, candidate: dict[str, Any], dates: Sequence[str]) -> pd.DataFrame:
    cols = feature_columns_for_set(candidate["feature_set"])
    needed = ["trade_date", "etf_code", "signal_clock", candidate["label_policy"], "future_return_3d", "max_drawdown_3d"] + cols
    subset = frame[(frame["trade_date"].isin(dates)) & (frame["signal_clock"] == candidate["signal_clock"])].copy()
    missing = [column for column in needed if column not in subset.columns]
    if missing:
        return pd.DataFrame(columns=needed)
    subset = subset.dropna(subset=[candidate["label_policy"], "future_return_3d"] + cols)
    return subset


def rows_to_matrix(frame: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    return frame[list(feature_columns)].astype(float).to_numpy()


def fit_predict_proba(train: pd.DataFrame, validation: pd.DataFrame, candidate: dict[str, Any]) -> tuple[np.ndarray | None, str | None]:
    labels = train[candidate["label_policy"]].astype(int).to_numpy()
    if len(set(labels.tolist())) < 2:
        return None, "train_single_class"
    x_train = rows_to_matrix(train, feature_columns_for_set(candidate["feature_set"]))
    x_validation = rows_to_matrix(validation, feature_columns_for_set(candidate["feature_set"]))
    family = candidate["model_family"]
    try:
        if family.startswith("dummy"):
            from sklearn.dummy import DummyClassifier

            strategy = "most_frequent" if family == "dummy_most_frequent" else "stratified"
            model = DummyClassifier(strategy=strategy, random_state=7)
            model.fit(x_train, labels)
            return model.predict_proba(x_validation)[:, 1], None
        if family == "logistic_balanced_scaled":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler

            model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=400, random_state=7))
            model.fit(x_train, labels)
            return model.predict_proba(x_validation)[:, 1], None
        if family == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            model = HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=15, learning_rate=0.05, random_state=7)
            model.fit(x_train, labels)
            return model.predict_proba(x_validation)[:, 1], None
        if family == "random_forest_shallow":
            from sklearn.ensemble import RandomForestClassifier

            model = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=10, random_state=7, n_jobs=1)
            model.fit(x_train, labels)
            return model.predict_proba(x_validation)[:, 1], None
        return None, f"unsupported_model_family:{family}"
    except Exception as exc:  # noqa: BLE001 - no-save search records model failure per candidate/fold.
        return None, f"model_error:{exc}"


def select_trades(scored: pd.DataFrame, cost_bps: float) -> pd.DataFrame:
    if scored.empty:
        return scored.assign(cost_paid=[])
    selected = (
        scored[scored["score"] >= 0.50]
        .sort_values(["trade_date", "score"], ascending=[True, False])
        .groupby("trade_date", as_index=False)
        .head(1)
        .copy()
    )
    if selected.empty:
        selected = scored.sort_values(["trade_date", "score"], ascending=[True, False]).groupby("trade_date", as_index=False).head(1).copy()
        selected = selected[selected["score"] >= 0.50]
    roundtrip_cost = 2.0 * float(cost_bps) / 10000.0
    selected["gross_return"] = selected["future_return_3d"].astype(float)
    selected["cost_paid"] = roundtrip_cost
    selected["net_return"] = selected["gross_return"] - selected["cost_paid"]
    selected["month"] = selected["trade_date"].astype(str).str[:7]
    return selected


def max_drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + float(value)
        peak = max(peak, equity)
        if peak:
            worst = min(worst, equity / peak - 1.0)
    return worst


def pnl_metrics(trades: pd.DataFrame, validation_rows: pd.DataFrame, cost_bps: float) -> dict[str, Any]:
    if trades.empty:
        return {
            "trade_count": 0,
            "net_total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "calmar_like_ratio": 0.0,
            "sharpe_like_diagnostic": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "average_trade_return": 0.0,
            "monthly_win_rate": 0.0,
            "turnover": 0.0,
            "average_exposure": 0.0,
            "cost_paid": 0.0,
            "net_vs_equal_weight_benchmark": 0.0,
            "net_vs_cash": 0.0,
            "worst_month": "",
            "best_month": "",
            "etf_contribution": {},
            "month_concentration": 0.0,
            "etf_concentration": 0.0,
        }
    returns = trades["net_return"].astype(float).tolist()
    gross = trades["gross_return"].astype(float).tolist()
    total = float(np.prod([1.0 + item for item in returns]) - 1.0)
    active_days = max(1, trades["trade_date"].nunique())
    annualized = float((1.0 + total) ** (252.0 / active_days) - 1.0) if total > -1.0 else -1.0
    drawdown = max_drawdown(returns)
    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns, ddof=0))
    positives = [value for value in returns if value > 0]
    negatives = [value for value in returns if value < 0]
    monthly = trades.groupby("month")["net_return"].sum().sort_index()
    etf_contribution = trades.groupby("etf_code")["net_return"].sum().sort_values(ascending=False).to_dict()
    abs_month = monthly.abs()
    abs_etf = pd.Series(etf_contribution).abs() if etf_contribution else pd.Series(dtype=float)
    benchmark = validation_rows.groupby("trade_date")["future_return_3d"].mean().mean() if not validation_rows.empty else 0.0
    return {
        "trade_count": int(len(trades)),
        "net_total_return": total,
        "annualized_return": annualized,
        "max_drawdown": drawdown,
        "calmar_like_ratio": annualized / abs(drawdown) if drawdown < 0 else (math.inf if annualized > 0 else 0.0),
        "sharpe_like_diagnostic": mean_return / std_return * math.sqrt(252.0) if std_return else 0.0,
        "win_rate": float(len(positives) / len(returns)),
        "profit_factor": float(sum(positives) / abs(sum(negatives))) if negatives else (math.inf if positives else 0.0),
        "average_trade_return": mean_return,
        "monthly_win_rate": float((monthly > 0).mean()) if len(monthly) else 0.0,
        "turnover": float(len(trades) / max(1, len(validation_rows))),
        "average_exposure": float(active_days / max(1, validation_rows["trade_date"].nunique())),
        "cost_paid": float(len(trades) * 2.0 * float(cost_bps) / 10000.0),
        "net_vs_equal_weight_benchmark": mean_return - float(benchmark),
        "net_vs_cash": mean_return,
        "gross_total_return": float(np.prod([1.0 + item for item in gross]) - 1.0),
        "worst_month": str(monthly.idxmin()) if len(monthly) else "",
        "best_month": str(monthly.idxmax()) if len(monthly) else "",
        "etf_contribution": {str(key): float(value) for key, value in etf_contribution.items()},
        "month_concentration": float(abs_month.max() / abs_month.sum()) if len(abs_month) and abs_month.sum() else 0.0,
        "etf_concentration": float(abs_etf.max() / abs_etf.sum()) if len(abs_etf) and abs_etf.sum() else 0.0,
    }


def gate_candidate(metrics: dict[str, Any], *, leakage_ok: bool, artifact_saved: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if metrics.get("net_total_return", 0.0) <= 0:
        reasons.append("net_total_return_not_positive")
    if metrics.get("max_drawdown", 0.0) < -0.20:
        reasons.append("max_drawdown_not_controlled")
    if metrics.get("calmar_like_ratio", 0.0) <= 0:
        reasons.append("calmar_like_not_positive")
    if metrics.get("win_rate", 0.0) <= 0.50:
        reasons.append("win_rate_not_above_50pct")
    if metrics.get("profit_factor", 0.0) <= 1.05:
        reasons.append("profit_factor_not_above_1_05")
    if metrics.get("monthly_win_rate", 0.0) < 0.55:
        reasons.append("positive_monthly_fraction_below_55pct")
    if metrics.get("month_concentration", 0.0) > 0.70:
        reasons.append("month_concentration_too_high")
    if metrics.get("etf_concentration", 0.0) > 0.70:
        reasons.append("etf_concentration_too_high")
    if not leakage_ok:
        reasons.append("leakage_risk")
    if artifact_saved:
        reasons.append("artifact_saved")
    return not reasons, reasons


def evaluate_candidate(feature_rows: pd.DataFrame, folds: Sequence[dict[str, Any]], candidate: dict[str, Any]) -> dict[str, Any]:
    fold_reports: list[dict[str, Any]] = []
    trade_frames: list[pd.DataFrame] = []
    leakage_ok = True
    for fold in folds:
        if fold["skipped"]:
            continue
        train = finite_candidate_rows(feature_rows, candidate, fold["train_anchor_dates"])
        validation = finite_candidate_rows(feature_rows, candidate, fold["validation_anchor_dates"])
        blockers: list[str] = []
        if train["trade_date"].nunique() < 1 or validation["trade_date"].nunique() < 1:
            blockers.append("empty_train_or_validation_after_filters")
        if set(train["trade_date"].astype(str)) & set(validation["trade_date"].astype(str)):
            blockers.append("train_validation_overlap")
            leakage_ok = False
        if len(validation) < 1:
            blockers.append("empty_validation")
        if blockers:
            fold_reports.append({**fold, "candidate_id": candidate["candidate_id"], "skipped": True, "skip_reasons": blockers})
            continue
        probabilities, error = fit_predict_proba(train, validation, candidate)
        if probabilities is None:
            fold_reports.append({**fold, "candidate_id": candidate["candidate_id"], "skipped": True, "skip_reasons": [error or "model_failed"]})
            continue
        scored = validation.copy()
        scored["score"] = probabilities
        trades = select_trades(scored, float(candidate["cost_bps"]))
        if not trades.empty:
            trades["candidate_id"] = candidate["candidate_id"]
            trades["fold_id"] = fold["fold_id"]
            trade_frames.append(trades)
        fold_reports.append(
            {
                **fold,
                "candidate_id": candidate["candidate_id"],
                "skipped": False,
                "train_group_count": int(len(train)),
                "validation_group_count": int(len(validation)),
                "trade_count": int(len(trades)),
            }
        )
    trades_all = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    validation_pool = feature_rows[feature_rows["signal_clock"] == candidate["signal_clock"]].copy()
    metrics = pnl_metrics(trades_all, validation_pool, float(candidate["cost_bps"]))
    gate_passed, gate_reasons = gate_candidate(metrics, leakage_ok=leakage_ok, artifact_saved=False)
    return {
        **candidate,
        "evaluated_fold_count": int(sum(1 for fold in fold_reports if not fold.get("skipped"))),
        "skipped_fold_count": int(sum(1 for fold in fold_reports if fold.get("skipped"))),
        "candidate_status": "LAB_DIAGNOSTIC_ALPHA_CANDIDATE_REVIEW_REQUIRED" if gate_passed else "LAB_DIAGNOSTIC_ALPHA_REJECTED_OR_INSUFFICIENT_REVIEW",
        "candidate_gate_passed": gate_passed,
        "candidate_gate_reasons": gate_reasons,
        "pnl_metrics": metrics,
        "fold_reports": fold_reports,
    }


def leaderboard_rows(results: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        metrics = result["pnl_metrics"]
        rows.append(
            {
                "candidate_id": result["candidate_id"],
                "candidate_status": result["candidate_status"],
                "signal_clock": result["signal_clock"],
                "feature_set": result["feature_set"],
                "label_policy": result["label_policy"],
                "model_family": result["model_family"],
                "net_total_return": metrics["net_total_return"],
                "annualized_return": metrics["annualized_return"],
                "max_drawdown": metrics["max_drawdown"],
                "calmar_like_ratio": metrics["calmar_like_ratio"],
                "win_rate": metrics["win_rate"],
                "profit_factor": metrics["profit_factor"],
                "monthly_win_rate": metrics["monthly_win_rate"],
                "trade_count": metrics["trade_count"],
                "cost_paid": metrics["cost_paid"],
                "net_vs_equal_weight_benchmark": metrics["net_vs_equal_weight_benchmark"],
            }
        )
    return sorted(rows, key=lambda item: (item["candidate_status"] != "LAB_DIAGNOSTIC_ALPHA_CANDIDATE_REVIEW_REQUIRED", -item["net_total_return"]))


def decision_from_results(results: Sequence[dict[str, Any]], folds: Sequence[dict[str, Any]], artifacts: Sequence[str]) -> str:
    if artifacts:
        return DECISION_RUNTIME
    if not folds:
        return DECISION_WAITING
    if any(not fold["train_validation_no_overlap"] or not fold["embargo_ok"] for fold in folds):
        return DECISION_LEAKAGE
    if any(result["candidate_gate_passed"] for result in results):
        return DECISION_CANDIDATES
    return DECISION_NO_CANDIDATES


def run_optimizer(config: OptimizerConfig, repo_root: Path = REPO_ROOT, *, enforce_paths: bool = True) -> dict[str, Any]:
    created_at_utc = datetime.now(timezone.utc).isoformat()
    data_lake = resolve_data_lake(config.data_lake, repo_root, enforce=enforce_paths)
    out_dir = resolve_out_dir(config.out_dir, repo_root, enforce=enforce_paths)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "long_history_optimizer_config",
        "created_at_utc": created_at_utc,
        "mode": config.mode,
        "search_dimensions": ["signal_clock", "feature_set", "label_policy", "holding_period", "entry_rule", "exit_rule", "cost_bps", "model_family"],
        "forbidden_search_dimensions": ["Stable BUY/PROBE", "target_weight", "final_buy_action", "OrderIntent", "QMT execution"],
        "validation_protocol": {
            "split": "rolling_origin_walk_forward",
            "fold_unit": "monthly",
            "train_window": "expanding",
            "min_train_anchors": config.min_train_anchors,
            "min_validation_anchors": config.min_validation_anchors,
            "min_validation_groups": config.min_validation_groups,
            "embargo_days": config.embargo_days,
        },
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "long_history_optimizer_config.json", config_report)
    try:
        feature_rows = load_feature_frame(data_lake)
        if feature_rows.empty:
            raise AlphaOptimizerError("long_history_feature_rows.csv or bars not found")
        folds = make_rolling_folds(
            feature_rows,
            min_train_anchors=config.min_train_anchors,
            min_validation_anchors=config.min_validation_anchors,
            min_validation_groups=config.min_validation_groups,
            embargo_days=config.embargo_days,
        )
        candidates = candidate_grid(config.cost_bps, config.max_candidates)
        results = [evaluate_candidate(feature_rows, folds, candidate) for candidate in candidates]
        artifacts = model_artifacts_present(out_dir)
        decision = decision_from_results(results, folds, artifacts)
    except AlphaOptimizerError as exc:
        feature_rows = pd.DataFrame()
        folds = []
        candidates = []
        results = []
        artifacts = model_artifacts_present(out_dir)
        decision = DECISION_WAITING if "not found" in str(exc) else DECISION_DATA_QUALITY
    except Exception as exc:  # noqa: BLE001 - top-level CLI report should close out runtime failures.
        feature_rows = pd.DataFrame()
        folds = []
        candidates = []
        results = []
        artifacts = model_artifacts_present(out_dir)
        decision = DECISION_RUNTIME
        runtime_error = str(exc)
    else:
        runtime_error = ""
    leaderboard = leaderboard_rows(results)
    pd.DataFrame(leaderboard).to_csv(out_dir / "long_history_candidate_leaderboard.csv", index=False, lineterminator="\n")
    pnl_rows = [{**{k: v for k, v in row.items() if k != "candidate_status"}, "candidate_status": row["candidate_status"]} for row in leaderboard]
    pd.DataFrame(pnl_rows).to_csv(out_dir / "long_history_candidate_pnl_summary.csv", index=False, lineterminator="\n")
    risk_rows = [
        {
            "candidate_id": result["candidate_id"],
            "candidate_status": result["candidate_status"],
            "max_drawdown": result["pnl_metrics"]["max_drawdown"],
            "worst_month": result["pnl_metrics"]["worst_month"],
            "best_month": result["pnl_metrics"]["best_month"],
            "month_concentration": result["pnl_metrics"]["month_concentration"],
            "etf_concentration": result["pnl_metrics"]["etf_concentration"],
            "candidate_gate_reasons": "|".join(result["candidate_gate_reasons"]),
        }
        for result in results
    ]
    pd.DataFrame(risk_rows).to_csv(out_dir / "long_history_candidate_risk_summary.csv", index=False, lineterminator="\n")
    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "long_history_alpha_search_report",
        "created_at_utc": created_at_utc,
        "data_lake": str(data_lake),
        "feature_row_count": int(len(feature_rows)),
        "candidate_count": len(candidates),
        "fold_count": len(folds),
        "usable_fold_count": int(sum(1 for fold in folds if not fold.get("skipped"))),
        "folds": folds,
        "leaderboard": leaderboard[:25],
        "candidate_results": results,
        "runtime_error": runtime_error,
        "model_artifacts_detected": artifacts,
        "future_label_or_outcome_in_feature_columns": bool(set(TIME_CENSORED_FEATURES) & (set(LABEL_COLUMNS) | set(OUTCOME_COLUMNS))),
        **BOUNDARY_FIELDS,
    }
    write_json(out_dir / "long_history_alpha_search_report.json", report)
    decision_report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": "long_history_alpha_optimization_decision",
        "created_at_utc": created_at_utc,
        "decision": decision,
        "candidate_status_allowed": "LAB_DIAGNOSTIC_ALPHA_CANDIDATE_REVIEW_REQUIRED",
        "candidate_found_count": int(sum(1 for result in results if result["candidate_gate_passed"])),
        "stable_promotion_ready": False,
        "stable_evidence": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "order_intent_generated": False,
        "qmt_ready": False,
        "formal_training": False,
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        **{key: value for key, value in BOUNDARY_FIELDS.items() if key not in {"stable_promotion_ready", "qmt_ready", "order_intent_ready", "formal_training", "model_saved", "scaler_saved", "checkpoint_saved"}},
    }
    write_json(out_dir / "long_history_alpha_optimization_decision.json", decision_report)
    return {**decision_report, "search_report": report}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab-only no-save bounded intraday alpha search.")
    parser.add_argument("--data-lake", type=Path, default=ALLOWED_DATA_LAKE)
    parser.add_argument("--out-dir", type=Path, default=ALLOWED_OUT_DIR)
    parser.add_argument("--mode", default="bounded_search", choices=["bounded_search"])
    parser.add_argument("--max-candidates", type=int, default=96)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_optimizer(OptimizerConfig(data_lake=args.data_lake, out_dir=args.out_dir, mode=args.mode, max_candidates=args.max_candidates))
    print(json.dumps(json_safe(report), ensure_ascii=False, indent=2))
    return 0 if not str(report["decision"]).startswith("LONG_HISTORY_ALPHA_OPTIMIZATION_BLOCKED_RUNTIME") else 2


if __name__ == "__main__":
    raise SystemExit(main())
