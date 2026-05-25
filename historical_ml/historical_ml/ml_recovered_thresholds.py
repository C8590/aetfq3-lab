from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import ensure_dir, read_table


DECISIONS = {
    "CONTINUE_SHADOW",
    "TIGHTEN_ML_RECOVERED_THRESHOLD",
    "USE_ML_AS_RANKING_ONLY",
    "USE_ML_AS_FILTER_ONLY",
    "ALLOW_LIMITED_ACTIVE_SIM",
    "DISABLE_ML_RECOVERY",
}

GRID_COLUMNS = [
    "candidate_id",
    "p_good_top_pct",
    "p_bad_max_quantile",
    "ml_rank_global_max",
    "ml_rank_sector_max",
    "drawdown_filter",
    "market_filter",
    "sector_filter",
    "momentum_filter",
    "data_quality_filter",
    "sample_count",
    "daily_average_count",
    "future_return_3d",
    "future_return_5d",
    "future_return_10d",
    "good_entry_rate",
    "bad_entry_rate",
    "future_max_drawdown_10d",
    "outperform_market_10d",
    "outperform_sector_10d",
    "precision_lift_vs_legacy",
    "bad_rate_delta_vs_legacy",
    "return_delta_vs_legacy",
    "passes_active_sim_hard_gate",
]


@dataclass(frozen=True)
class RecoveredThresholdResult:
    grid: pd.DataFrame
    report: str
    report_json: dict[str, Any]
    recommendation: str
    output_paths: dict[str, Path]


def build_ml_recovered_threshold_recommendation_from_file(
    *,
    historical_review_path: str | Path,
    out_dir: str | Path,
) -> RecoveredThresholdResult:
    frame = read_table(historical_review_path)
    return build_ml_recovered_threshold_recommendation(frame, out_dir=out_dir)


def build_ml_recovered_threshold_recommendation(
    historical_review: pd.DataFrame,
    *,
    out_dir: str | Path,
) -> RecoveredThresholdResult:
    frame = _prepare_frame(historical_review)
    total_trade_dates = int(frame["trade_date"].nunique()) if not frame.empty else 0
    legacy = _metrics(_legacy_pool(frame), "legacy_v21_buy_probe", "legacy_v21_buy_probe", total_trade_dates=total_trade_dates)
    original = _metrics(_original_recovered_pool(frame), "original_ml_recovered", "original_ml_recovered", legacy=legacy, total_trade_dates=total_trade_dates)
    random_baseline = _metrics(
        _random_baseline(frame, original["sample_count"]),
        "random_baseline",
        "random_baseline",
        legacy=legacy,
        total_trade_dates=total_trade_dates,
    )
    grid = _build_grid(frame, legacy)
    best = _select_best_grid_row(grid, legacy)
    comparison = _comparison_rows(frame, best, legacy, original, random_baseline)
    classes = _classify_recovered_pool(frame, best)
    recommendation = _recommend(best, legacy, original)
    report_json = _build_report_json(
        frame=frame,
        grid=grid,
        best=best,
        legacy=legacy,
        original=original,
        random_baseline=random_baseline,
        comparison=comparison,
        classes=classes,
        recommendation=recommendation,
    )
    report = _build_report_markdown(report_json, grid, comparison)

    out = ensure_dir(out_dir)
    paths = {
        "grid": out / "ml_recovered_threshold_grid.csv",
        "report_md": out / "ml_recovered_threshold_recommendation.md",
        "report_json": out / "ml_recovered_threshold_recommendation.json",
    }
    grid.to_csv(paths["grid"], index=False, encoding="utf-8-sig")
    paths["report_md"].write_text(report, encoding="utf-8")
    paths["report_json"].write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return RecoveredThresholdResult(grid, report, report_json, recommendation, paths)


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame = frame.loc[frame.get("review_status", "").astype(str).eq("READY")].copy()
    if frame.empty:
        return frame
    for column in [
        "ml_score",
        "p_good_entry",
        "p_bad_entry",
        "ml_rank_global",
        "ml_rank_sector",
        "expected_drawdown_10d",
        "momentum_score",
        "future_return_3d",
        "future_return_5d",
        "future_return_10d",
        "future_max_drawdown_10d",
    ]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["outperform_market_10d", "outperform_sector_10d", "is_valid_sample", "pre_selected"]:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].map(_truthy)
    for column in ["market_state", "sector_state", "legacy_action", "ml_adjustment_type", "ml_adjustment_bucket", "auto_label", "exclude_reason"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    frame["_is_original_recovered"] = frame["ml_adjustment_bucket"].str.upper().eq("ML_RECOVERED") | frame["ml_adjustment_type"].str.upper().eq("ML_RECOVERED")
    frame["_is_legacy_buy_probe"] = frame["legacy_action"].str.upper().isin(["BUY", "PROBE"])
    frame["_daily_count"] = frame.groupby("trade_date")["code"].transform("count")
    frame["_p_good_rank"] = frame.groupby("trade_date")["p_good_entry"].rank(ascending=False, method="first")
    frame["_p_bad_q30"] = frame.groupby("trade_date")["p_bad_entry"].transform(lambda s: s.quantile(0.30))
    frame["_p_bad_q40"] = frame.groupby("trade_date")["p_bad_entry"].transform(lambda s: s.quantile(0.40))
    frame["_p_bad_q50"] = frame.groupby("trade_date")["p_bad_entry"].transform(lambda s: s.quantile(0.50))
    frame["_drawdown_q50"] = frame.groupby("trade_date")["expected_drawdown_10d"].transform(lambda s: s.quantile(0.50))
    frame["_drawdown_q60"] = frame.groupby("trade_date")["expected_drawdown_10d"].transform(lambda s: s.quantile(0.60))
    frame["_drawdown_q70"] = frame.groupby("trade_date")["expected_drawdown_10d"].transform(lambda s: s.quantile(0.70))
    frame["_momentum_q40"] = frame.groupby("trade_date")["momentum_score"].transform(lambda s: s.quantile(0.40))
    frame["_momentum_q50"] = frame.groupby("trade_date")["momentum_score"].transform(lambda s: s.quantile(0.50))
    frame["_momentum_q60"] = frame.groupby("trade_date")["momentum_score"].transform(lambda s: s.quantile(0.60))
    return frame


def _build_grid(frame: pd.DataFrame, legacy: dict[str, Any]) -> pd.DataFrame:
    recovered = _original_recovered_pool(frame)
    if recovered.empty:
        return pd.DataFrame(columns=GRID_COLUMNS)
    total_trade_dates = int(frame["trade_date"].nunique())
    p_good_pcts = [0.01, 0.03, 0.05, 0.10]
    p_bad_quantiles = [0.50, 0.40, 0.30]
    global_ranks = [20, 50, 100]
    sector_ranks = [1, 3, 5]
    drawdown_filters = ["above_median", "above_p60"]
    market_filters = ["no_defense_strong"]
    sector_filters = ["no_weak_strong"]
    momentum_filters = ["above_p40", "above_median"]
    data_quality_filters = ["valid_only"]

    rows: list[dict[str, Any]] = []
    idx = 0
    for values in product(
        p_good_pcts,
        p_bad_quantiles,
        global_ranks,
        sector_ranks,
        drawdown_filters,
        market_filters,
        sector_filters,
        momentum_filters,
        data_quality_filters,
    ):
        idx += 1
        p_good_pct, p_bad_q, global_rank, sector_rank, drawdown_filter, market_filter, sector_filter, momentum_filter, data_quality_filter = values
        mask = _condition_mask(
            recovered,
            p_good_top_pct=p_good_pct,
            p_bad_max_quantile=p_bad_q,
            ml_rank_global_max=global_rank,
            ml_rank_sector_max=sector_rank,
            drawdown_filter=drawdown_filter,
            market_filter=market_filter,
            sector_filter=sector_filter,
            momentum_filter=momentum_filter,
            data_quality_filter=data_quality_filter,
        )
        metrics = _metrics(recovered.loc[mask], f"grid_{idx:04d}", "ML_STRONG_RECOVERED", legacy=legacy, total_trade_dates=total_trade_dates)
        metrics.update(
            {
                "candidate_id": f"grid_{idx:04d}",
                "p_good_top_pct": p_good_pct,
                "p_bad_max_quantile": p_bad_q,
                "ml_rank_global_max": global_rank,
                "ml_rank_sector_max": sector_rank,
                "drawdown_filter": drawdown_filter,
                "market_filter": market_filter,
                "sector_filter": sector_filter,
                "momentum_filter": momentum_filter,
                "data_quality_filter": data_quality_filter,
                "passes_active_sim_hard_gate": bool(
                    metrics["sample_count"] > 0
                    and metrics["good_entry_rate"] > legacy["good_entry_rate"]
                    and metrics["bad_entry_rate"] <= legacy["bad_entry_rate"]
                ),
            }
        )
        rows.append(metrics)
    grid = pd.DataFrame(rows)
    return grid[GRID_COLUMNS].sort_values(
        ["passes_active_sim_hard_gate", "good_entry_rate", "bad_entry_rate", "future_return_10d", "sample_count"],
        ascending=[False, False, True, False, False],
    ).reset_index(drop=True)


def _condition_mask(
    frame: pd.DataFrame,
    *,
    p_good_top_pct: float,
    p_bad_max_quantile: float,
    ml_rank_global_max: int,
    ml_rank_sector_max: int,
    drawdown_filter: str,
    market_filter: str,
    sector_filter: str,
    momentum_filter: str,
    data_quality_filter: str,
) -> pd.Series:
    top_n = np.ceil(frame["_daily_count"] * float(p_good_top_pct)).clip(lower=1)
    mask = (
        frame["_p_good_rank"].le(top_n)
        & frame["p_bad_entry"].le(frame[f"_p_bad_q{int(p_bad_max_quantile * 100)}"])
        & frame["ml_rank_global"].le(int(ml_rank_global_max))
        & frame["ml_rank_sector"].le(int(ml_rank_sector_max))
    )
    if drawdown_filter == "above_median":
        mask &= frame["expected_drawdown_10d"].ge(frame["_drawdown_q50"])
    elif drawdown_filter == "above_p60":
        mask &= frame["expected_drawdown_10d"].ge(frame["_drawdown_q60"])
    elif drawdown_filter == "above_p70":
        mask &= frame["expected_drawdown_10d"].ge(frame["_drawdown_q70"])
    if market_filter == "no_defense_strong":
        mask &= ~frame["market_state"].str.lower().eq("defense")
    if sector_filter == "no_weak_strong":
        mask &= ~frame["sector_state"].str.lower().eq("weak")
    if momentum_filter == "above_p40":
        mask &= frame["momentum_score"].ge(frame["_momentum_q40"])
    elif momentum_filter == "above_median":
        mask &= frame["momentum_score"].ge(frame["_momentum_q50"])
    elif momentum_filter == "above_p60":
        mask &= frame["momentum_score"].ge(frame["_momentum_q60"])
    if data_quality_filter == "valid_only":
        mask &= frame["is_valid_sample"] & ~frame["exclude_reason"].str.lower().str.contains("invalid", na=False)
    return mask.fillna(False)


def _comparison_rows(
    frame: pd.DataFrame,
    best: pd.Series,
    legacy: dict[str, Any],
    original: dict[str, Any],
    random_baseline: dict[str, Any],
) -> pd.DataFrame:
    recovered = _original_recovered_pool(frame)
    total_trade_dates = int(frame["trade_date"].nunique()) if not frame.empty else 0
    if best.empty:
        strong = recovered.iloc[0:0].copy()
    else:
        strong = recovered.loc[
            _condition_mask(
                recovered,
                p_good_top_pct=float(best["p_good_top_pct"]),
                p_bad_max_quantile=float(best["p_bad_max_quantile"]),
                ml_rank_global_max=int(best["ml_rank_global_max"]),
                ml_rank_sector_max=int(best["ml_rank_sector_max"]),
                drawdown_filter=str(best["drawdown_filter"]),
                market_filter=str(best["market_filter"]),
                sector_filter=str(best["sector_filter"]),
                momentum_filter=str(best["momentum_filter"]),
                data_quality_filter=str(best["data_quality_filter"]),
            )
        ].copy()
    rows = [
        original,
        _metrics(_daily_top(strong, 20), "ml_strong_recovered_top20", "ML_STRONG_RECOVERED Top20", legacy=legacy, total_trade_dates=total_trade_dates),
        _metrics(_daily_top(strong, 50), "ml_strong_recovered_top50", "ML_STRONG_RECOVERED Top50", legacy=legacy, total_trade_dates=total_trade_dates),
        _metrics(_daily_top(strong, 100), "ml_strong_recovered_top100", "ML_STRONG_RECOVERED Top100", legacy=legacy, total_trade_dates=total_trade_dates),
        legacy,
        random_baseline,
    ]
    return pd.DataFrame(rows)


def _select_best_grid_row(grid: pd.DataFrame, legacy: dict[str, Any]) -> pd.Series:
    if grid.empty:
        return pd.Series(dtype=object)
    viable = grid.loc[
        grid["passes_active_sim_hard_gate"]
        & grid["future_return_10d"].ge(legacy["future_return_10d"])
        & grid["daily_average_count"].ge(0.25)
    ].copy()
    if viable.empty:
        viable = grid.loc[grid["sample_count"].gt(0)].copy()
    if viable.empty:
        return pd.Series(dtype=object)
    viable["_score"] = (
        (viable["good_entry_rate"] - legacy["good_entry_rate"]) * 5.0
        - (viable["bad_entry_rate"] - legacy["bad_entry_rate"]) * 3.0
        + (viable["future_return_10d"] - legacy["future_return_10d"]) * 2.0
        + np.log1p(viable["sample_count"]) * 0.001
    )
    return viable.sort_values(["_score", "sample_count"], ascending=[False, False]).iloc[0]


def _classify_recovered_pool(frame: pd.DataFrame, best: pd.Series) -> dict[str, Any]:
    recovered = _original_recovered_pool(frame)
    if recovered.empty or best.empty:
        return {"ML_STRONG_RECOVERED": 0, "ML_WEAK_RECOVERED": 0, "ML_NOISE_RECOVERED": int(len(recovered))}
    strong_mask = _condition_mask(
        recovered,
        p_good_top_pct=float(best["p_good_top_pct"]),
        p_bad_max_quantile=float(best["p_bad_max_quantile"]),
        ml_rank_global_max=int(best["ml_rank_global_max"]),
        ml_rank_sector_max=int(best["ml_rank_sector_max"]),
        drawdown_filter=str(best["drawdown_filter"]),
        market_filter=str(best["market_filter"]),
        sector_filter=str(best["sector_filter"]),
        momentum_filter=str(best["momentum_filter"]),
        data_quality_filter=str(best["data_quality_filter"]),
    )
    weak_mask = (
        ~strong_mask
        & recovered["_p_good_rank"].le(np.ceil(recovered["_daily_count"] * 0.10).clip(lower=1))
        & recovered["p_bad_entry"].le(recovered["_p_bad_q50"])
        & recovered["ml_rank_global"].le(100)
        & recovered["ml_rank_sector"].le(5)
        & ~recovered["market_state"].str.lower().eq("defense")
        & ~recovered["sector_state"].str.lower().eq("weak")
    )
    return {
        "ML_STRONG_RECOVERED": int(strong_mask.sum()),
        "ML_WEAK_RECOVERED": int(weak_mask.sum()),
        "ML_NOISE_RECOVERED": int((~strong_mask & ~weak_mask).sum()),
    }


def _recommend(best: pd.Series, legacy: dict[str, Any], original: dict[str, Any]) -> str:
    if int(original["sample_count"]) == 0:
        return "DISABLE_ML_RECOVERY"
    if best.empty or int(best.get("sample_count") or 0) == 0:
        return "DISABLE_ML_RECOVERY"
    if float(best["good_entry_rate"]) <= float(legacy["good_entry_rate"]):
        if float(best["future_return_10d"]) > float(legacy["future_return_10d"]):
            return "USE_ML_AS_RANKING_ONLY"
        return "CONTINUE_SHADOW"
    if float(best["bad_entry_rate"]) > float(legacy["bad_entry_rate"]):
        return "USE_ML_AS_RANKING_ONLY"
    if float(best["future_return_10d"]) < float(legacy["future_return_10d"]):
        return "USE_ML_AS_FILTER_ONLY"
    if float(best["daily_average_count"]) < 1.0:
        return "TIGHTEN_ML_RECOVERED_THRESHOLD"
    return "ALLOW_LIMITED_ACTIVE_SIM"


def _metrics(
    frame: pd.DataFrame,
    candidate_id: str,
    label: str,
    *,
    legacy: dict[str, Any] | None = None,
    total_trade_dates: int = 0,
) -> dict[str, Any]:
    out = {
        "candidate_id": candidate_id,
        "label": label,
        "sample_count": int(len(frame)),
        "daily_average_count": _daily_average_count(frame, total_trade_dates=total_trade_dates),
        "future_return_3d": _mean(frame.get("future_return_3d")),
        "future_return_5d": _mean(frame.get("future_return_5d")),
        "future_return_10d": _mean(frame.get("future_return_10d")),
        "good_entry_rate": _rate(frame.get("auto_label"), "good_entry"),
        "bad_entry_rate": _rate(frame.get("auto_label"), "bad_entry"),
        "future_max_drawdown_10d": _mean(frame.get("future_max_drawdown_10d")),
        "outperform_market_10d": _truth_rate(frame.get("outperform_market_10d")),
        "outperform_sector_10d": _truth_rate(frame.get("outperform_sector_10d")),
    }
    if legacy:
        out["precision_lift_vs_legacy"] = _round(out["good_entry_rate"] - legacy["good_entry_rate"])
        out["bad_rate_delta_vs_legacy"] = _round(out["bad_entry_rate"] - legacy["bad_entry_rate"])
        out["return_delta_vs_legacy"] = _round(out["future_return_10d"] - legacy["future_return_10d"])
    else:
        out["precision_lift_vs_legacy"] = 0.0
        out["bad_rate_delta_vs_legacy"] = 0.0
        out["return_delta_vs_legacy"] = 0.0
    return out


def _legacy_pool(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["_is_legacy_buy_probe"]].copy()


def _original_recovered_pool(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["_is_original_recovered"]].copy()


def _random_baseline(frame: pd.DataFrame, sample_count: int) -> pd.DataFrame:
    if frame.empty or sample_count <= 0:
        return frame.iloc[0:0].copy()
    per_day = max(1, int(round(sample_count / max(frame["trade_date"].nunique(), 1))))
    return frame.groupby("trade_date", group_keys=False).apply(
        lambda group: group.sample(n=min(per_day, len(group)), random_state=20260525)
    )


def _daily_top(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    top = frame.copy()
    top["_top_rank"] = top.groupby("trade_date")["ml_score"].rank(ascending=False, method="first")
    return top.loc[top["_top_rank"].le(int(top_n))].drop(columns=["_top_rank"])


def _build_report_json(
    *,
    frame: pd.DataFrame,
    grid: pd.DataFrame,
    best: pd.Series,
    legacy: dict[str, Any],
    original: dict[str, Any],
    random_baseline: dict[str, Any],
    comparison: pd.DataFrame,
    classes: dict[str, Any],
    recommendation: str,
) -> dict[str, Any]:
    return {
        "mode": "V2.1_ML_RECOVERED_THRESHOLD_OFFLINE_RECOMMENDATION",
        "recommendation": recommendation,
        "allowed_recommendations": sorted(DECISIONS),
        "source_rows": int(len(frame)),
        "trade_date_count": int(frame["trade_date"].nunique()) if not frame.empty else 0,
        "original_ml_recovered_count": int(original["sample_count"]),
        "legacy_v21_buy_probe": legacy,
        "original_ml_recovered": original,
        "random_baseline": random_baseline,
        "best_condition": {} if best.empty else _jsonable(best.drop(labels=["_score"], errors="ignore").to_dict()),
        "recovered_classification_counts": classes,
        "comparison": _jsonable(comparison.replace({np.nan: None}).to_dict(orient="records")),
        "hard_gate": {
            "require_good_entry_rate_above_legacy_for_active_sim": True,
            "require_bad_entry_rate_not_above_legacy_for_active_sim": True,
            "formal_entry_changed": False,
            "final_buy_action_changed": False,
            "buy_probe_threshold_changed": False,
            "qmt_triggered": False,
            "market_data_refreshed": False,
            "data_cache_written": False,
        },
        "grid_rows": int(len(grid)),
    }


def _build_report_markdown(report_json: dict[str, Any], grid: pd.DataFrame, comparison: pd.DataFrame) -> str:
    best = report_json.get("best_condition") or {}
    lines = [
        "# ML_RECOVERED Threshold Offline Recommendation",
        "",
        "## control_center Conclusion",
        "",
        f"- recommendation: {report_json['recommendation']}",
        "- formal_entry_change: no",
        "- final_buy_action_change: no",
        "- qmt_triggered: no",
        "- BUY/PROBE threshold_change: no",
        f"- source_rows: {report_json['source_rows']}",
        f"- original_ml_recovered_count: {report_json['original_ml_recovered_count']}",
        "",
        "## Best Offline Condition",
        "",
        _dict_lines(best),
        "",
        "## Recovered Pool Split",
        "",
        _dict_lines(report_json["recovered_classification_counts"]),
        "",
        "## Required Comparison",
        "",
        _markdown_table(comparison),
        "",
        "## Top Grid Candidates",
        "",
        _markdown_table(grid.head(20)),
        "",
        "## Boundary",
        "",
        "- This is an offline historical_ml recommendation only.",
        "- It does not modify entry rules, final_buy_action, BUY/PROBE thresholds, QMT, market data, or data/cache.",
        "- Active-sim is forbidden unless good_entry_rate is above legacy and bad_entry_rate is not above legacy.",
        "",
    ]
    return "\n".join(lines)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_no data_"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    columns = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in display.columns) + " |")
    return "\n".join(lines)


def _dict_lines(values: dict[str, Any]) -> str:
    if not values:
        return "- none"
    return "\n".join(f"- {key}: {value}" for key, value in values.items())


def _daily_average_count(frame: pd.DataFrame, *, total_trade_dates: int) -> float:
    if frame.empty or total_trade_dates <= 0:
        return 0.0
    return _round(len(frame) / total_trade_dates)


def _mean(series: pd.Series | None) -> float:
    if series is None:
        return 0.0
    values = pd.to_numeric(series, errors="coerce").dropna()
    return _round(values.mean()) if not values.empty else 0.0


def _rate(series: pd.Series | None, value: str) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return _round(series.fillna("").astype(str).eq(value).mean())


def _truth_rate(series: pd.Series | None) -> float:
    if series is None or len(series) == 0:
        return 0.0
    return _round(series.map(_truthy).mean())


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y", "selected"}


def _round(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return round(float(value), 6)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if pd.isna(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    return value
