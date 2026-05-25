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
    "USE_ML_AS_RANKING_ONLY",
    "USE_ML_AS_FILTER_ONLY",
    "TIGHTEN_ML_RECOVERED_THRESHOLD",
    "ALLOW_LIMITED_ACTIVE_SIM",
    "DISABLE_ML_RECOVERY",
}

METRIC_COLUMNS = [
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
]

GRID_COLUMNS = [
    "candidate_id",
    "p_good_top_pct",
    "p_bad_max_quantile",
    "ml_rank_global_max",
    "ml_rank_sector_max",
    "momentum_min_quantile",
    "acceleration_min_quantile",
    "liquidity_filter",
    "drawdown_filter",
    *METRIC_COLUMNS,
    "quality_beats_legacy",
    "meets_sample_floor",
    "meets_target_daily_average",
    "ranking_only_signal",
]

MANUAL_REVIEW_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "sector_level1",
    "sector_level2",
    "ml_score",
    "p_good_entry",
    "p_bad_entry",
    "ml_rank_global",
    "ml_rank_sector",
    "momentum_score",
    "acceleration_score",
    "market_state",
    "sector_state",
    "future_return_1d",
    "future_return_3d",
    "future_return_5d",
    "future_return_10d",
    "future_max_drawdown_10d",
    "outperform_market_10d",
    "outperform_sector_10d",
    "是否集中在少数日期",
    "是否集中在少数行业",
    "是否存在数据异常",
    "是否属于高弹性脉冲",
    "是否符合右侧买点",
    "人工复核建议",
    "manual_review_note",
]


@dataclass(frozen=True)
class MLCoreRecoveredReviewResult:
    manual_review: pd.DataFrame
    grid: pd.DataFrame
    recommendation: str
    report: str
    report_json: dict[str, Any]
    output_paths: dict[str, Path]


def build_ml_core_recovered_review_from_file(
    *,
    historical_review_path: str | Path,
    out_dir: str | Path,
) -> MLCoreRecoveredReviewResult:
    return build_ml_core_recovered_review(read_table(historical_review_path), out_dir=out_dir)


def build_ml_core_recovered_review(
    historical_review: pd.DataFrame,
    *,
    out_dir: str | Path,
) -> MLCoreRecoveredReviewResult:
    frame = _prepare_frame(historical_review)
    total_trade_dates = int(frame["trade_date"].nunique()) if not frame.empty else 0

    legacy = _metrics(_legacy_pool(frame), "legacy_v21_buy_probe", "legacy buy/probe", total_trade_dates=total_trade_dates)
    original = _metrics(_original_recovered_pool(frame), "original_ml_recovered", "original ML_RECOVERED", legacy=legacy, total_trade_dates=total_trade_dates)
    random_baseline = _metrics(
        _random_baseline(frame, int(original["sample_count"])),
        "random_baseline",
        "random baseline",
        legacy=legacy,
        total_trade_dates=total_trade_dates,
    )

    strong_condition = _strong_condition()
    strong_pool = _filtered_recovered_pool(frame, strong_condition).copy()
    strong = _metrics(strong_pool, "ml_strong_recovered", "ML_STRONG_RECOVERED", legacy=legacy, total_trade_dates=total_trade_dates)
    manual_review, manual_summary = _build_manual_review(strong_pool)

    grid = _build_core_grid(frame, legacy, total_trade_dates=total_trade_dates)
    best = _select_best_core_row(grid, legacy)
    core_condition = {} if best.empty else _condition_from_grid_row(best)
    core_pool = _filtered_recovered_pool(frame, core_condition).copy() if core_condition else frame.iloc[0:0].copy()
    recommendation = _recommend(grid, best, legacy, original)
    comparison = _comparison_rows(
        frame=frame,
        core_pool=core_pool,
        legacy=legacy,
        random_baseline=random_baseline,
        original=original,
        strong=strong,
        total_trade_dates=total_trade_dates,
    )

    report_json = _build_report_json(
        frame=frame,
        manual_summary=manual_summary,
        legacy=legacy,
        original=original,
        random_baseline=random_baseline,
        strong=strong,
        grid=grid,
        best=best,
        comparison=comparison,
        recommendation=recommendation,
    )
    report = _build_report_markdown(report_json, grid, comparison)

    out = ensure_dir(out_dir)
    paths = {
        "manual_review_csv": out / "ml_strong_recovered_manual_review.csv",
        "manual_review_md": out / "ml_strong_recovered_manual_review.md",
        "manual_review_json": out / "ml_strong_recovered_manual_review.json",
        "grid_csv": out / "ml_core_recovered_threshold_grid.csv",
        "recommendation_md": out / "ml_core_recovered_recommendation.md",
        "recommendation_json": out / "ml_core_recovered_recommendation.json",
    }
    manual_review.to_csv(paths["manual_review_csv"], index=False, encoding="utf-8-sig")
    paths["manual_review_md"].write_text(_build_manual_review_markdown(manual_review, manual_summary), encoding="utf-8")
    paths["manual_review_json"].write_text(
        json.dumps(
            {
                "mode": "ML_STRONG_RECOVERED_MANUAL_REVIEW_OFFLINE",
                "manual_summary": manual_summary,
                "rows": _jsonable(manual_review.replace({np.nan: None}).to_dict(orient="records")),
                "hard_gate": _hard_gate(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    grid.to_csv(paths["grid_csv"], index=False, encoding="utf-8-sig")
    paths["recommendation_md"].write_text(report, encoding="utf-8")
    paths["recommendation_json"].write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")

    return MLCoreRecoveredReviewResult(
        manual_review=manual_review,
        grid=grid,
        recommendation=recommendation,
        report=report,
        report_json=report_json,
        output_paths=paths,
    )


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    if "trade_date" not in frame.columns:
        frame["trade_date"] = ""
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if "review_status" in frame.columns:
        frame = frame.loc[frame["review_status"].fillna("").astype(str).eq("READY")].copy()
    if frame.empty:
        return frame

    numeric_columns = [
        "ml_score",
        "p_good_entry",
        "p_bad_entry",
        "ml_rank_global",
        "ml_rank_sector",
        "momentum_score",
        "acceleration_score",
        "expected_drawdown_10d",
        "future_return_1d",
        "future_return_3d",
        "future_return_5d",
        "future_return_10d",
        "future_max_drawdown_10d",
        "etf_rank",
        "volatility_20",
    ]
    for column in numeric_columns:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    text_columns = [
        "code",
        "name",
        "sector_level1",
        "sector_level2",
        "market_state",
        "sector_state",
        "legacy_action",
        "ml_adjustment_type",
        "ml_adjustment_bucket",
        "auto_label",
        "exclude_reason",
    ]
    for column in text_columns:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)

    bool_columns = ["outperform_market_10d", "outperform_sector_10d", "is_valid_sample"]
    for column in bool_columns:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].map(_truthy)

    frame["_is_original_recovered"] = frame["ml_adjustment_bucket"].str.upper().eq("ML_RECOVERED") | frame[
        "ml_adjustment_type"
    ].str.upper().eq("ML_RECOVERED")
    frame["_is_legacy_buy_probe"] = frame["legacy_action"].str.upper().isin(["BUY", "PROBE"])
    frame["_daily_all_count"] = frame.groupby("trade_date")["code"].transform("count").clip(lower=1)
    frame["_p_good_rank_all"] = frame.groupby("trade_date")["p_good_entry"].rank(ascending=False, method="first")

    for quantile in (0.30, 0.40, 0.50):
        suffix = int(quantile * 100)
        frame[f"_p_bad_q{suffix}"] = frame.groupby("trade_date")["p_bad_entry"].transform(lambda s: s.quantile(quantile))
        frame[f"_momentum_q{suffix}"] = frame.groupby("trade_date")["momentum_score"].transform(lambda s: s.quantile(quantile))
        frame[f"_acceleration_q{suffix}"] = frame.groupby("trade_date")["acceleration_score"].transform(lambda s: s.quantile(quantile))
        frame[f"_drawdown_q{suffix}"] = frame.groupby("trade_date")["expected_drawdown_10d"].transform(lambda s: s.quantile(quantile))

    frame["_drawdown_q60"] = frame.groupby("trade_date")["expected_drawdown_10d"].transform(lambda s: s.quantile(0.60))
    frame["_volatility_q80"] = frame.groupby("trade_date")["volatility_20"].transform(lambda s: s.quantile(0.80))
    frame["_acceleration_q80"] = frame.groupby("trade_date")["acceleration_score"].transform(lambda s: s.quantile(0.80))
    return frame


def _strong_condition() -> dict[str, Any]:
    return {
        "p_good_top_pct": 0.01,
        "p_bad_max_quantile": 0.50,
        "ml_rank_global_max": 20,
        "ml_rank_sector_max": 1,
        "momentum_min_quantile": 0.40,
        "acceleration_min_quantile": None,
        "liquidity_filter": "none",
        "drawdown_filter": "expected_drawdown_ge_p50",
    }


def _build_core_grid(frame: pd.DataFrame, legacy: dict[str, Any], *, total_trade_dates: int) -> pd.DataFrame:
    recovered = _base_core_pool(frame)
    if recovered.empty:
        return pd.DataFrame(columns=GRID_COLUMNS)

    p_good_masks = {
        pct: recovered["_p_good_rank_all"].le(np.ceil(recovered["_daily_all_count"] * pct).clip(lower=1))
        for pct in [0.01, 0.03, 0.05]
    }
    p_bad_masks = {
        quantile: recovered["p_bad_entry"].le(recovered[f"_p_bad_q{int(quantile * 100)}"])
        for quantile in [0.30, 0.40, 0.50]
    }
    global_rank_masks = {rank: recovered["ml_rank_global"].le(rank) for rank in [20, 50, 100]}
    sector_rank_masks = {rank: recovered["ml_rank_sector"].le(rank) for rank in [1, 3]}
    momentum_masks = {
        quantile: recovered["momentum_score"].ge(recovered[f"_momentum_q{int(quantile * 100)}"])
        for quantile in [0.30, 0.40, 0.50]
    }
    acceleration_masks = {
        quantile: recovered["acceleration_score"].ge(recovered[f"_acceleration_q{int(quantile * 100)}"])
        for quantile in [0.30, 0.40, 0.50]
    }
    liquidity_masks = {
        "etf_rank_top_500": recovered["etf_rank"].le(500),
        "etf_rank_top_750": recovered["etf_rank"].le(750),
    }
    drawdown_masks = {
        "expected_drawdown_ge_p30": recovered["expected_drawdown_10d"].ge(recovered["_drawdown_q30"]),
        "expected_drawdown_ge_p50": recovered["expected_drawdown_10d"].ge(recovered["_drawdown_q50"]),
    }

    rows: list[dict[str, Any]] = []
    idx = 0
    for values in product(
        [0.01, 0.03, 0.05],
        [0.30, 0.40, 0.50],
        [20, 50, 100],
        [1, 3],
        [0.30, 0.40, 0.50],
        [0.30, 0.40, 0.50],
        ["etf_rank_top_500", "etf_rank_top_750"],
        ["expected_drawdown_ge_p30", "expected_drawdown_ge_p50"],
    ):
        idx += 1
        condition = {
            "p_good_top_pct": values[0],
            "p_bad_max_quantile": values[1],
            "ml_rank_global_max": values[2],
            "ml_rank_sector_max": values[3],
            "momentum_min_quantile": values[4],
            "acceleration_min_quantile": values[5],
            "liquidity_filter": values[6],
            "drawdown_filter": values[7],
        }
        mask = (
            p_good_masks[values[0]]
            & p_bad_masks[values[1]]
            & global_rank_masks[values[2]]
            & sector_rank_masks[values[3]]
            & momentum_masks[values[4]]
            & acceleration_masks[values[5]]
            & liquidity_masks[values[6]]
            & drawdown_masks[values[7]]
        )
        pool = recovered.loc[mask.fillna(False)]
        metrics = _metrics(pool, f"core_grid_{idx:04d}", "ML_CORE_RECOVERED candidate", legacy=legacy, total_trade_dates=total_trade_dates)
        metrics.update(condition)
        metrics["candidate_id"] = f"core_grid_{idx:04d}"
        metrics["quality_beats_legacy"] = bool(
            metrics["good_entry_rate"] > legacy["good_entry_rate"]
            and metrics["bad_entry_rate"] < legacy["bad_entry_rate"]
        )
        metrics["meets_sample_floor"] = bool(metrics["sample_count"] >= 100 and metrics["daily_average_count"] >= 0.5)
        metrics["meets_target_daily_average"] = bool(1.0 <= metrics["daily_average_count"] <= 3.0)
        metrics["ranking_only_signal"] = bool(
            metrics["future_return_10d"] > legacy["future_return_10d"]
            and metrics["good_entry_rate"] <= legacy["good_entry_rate"]
        )
        rows.append(metrics)

    grid = pd.DataFrame(rows)
    return grid[GRID_COLUMNS].sort_values(
        [
            "quality_beats_legacy",
            "meets_sample_floor",
            "meets_target_daily_average",
            "good_entry_rate",
            "bad_entry_rate",
            "future_return_10d",
            "sample_count",
        ],
        ascending=[False, False, False, False, True, False, False],
    ).reset_index(drop=True)


def _condition_from_grid_row(row: pd.Series) -> dict[str, Any]:
    return {
        "p_good_top_pct": float(row["p_good_top_pct"]),
        "p_bad_max_quantile": float(row["p_bad_max_quantile"]),
        "ml_rank_global_max": int(row["ml_rank_global_max"]),
        "ml_rank_sector_max": int(row["ml_rank_sector_max"]),
        "momentum_min_quantile": float(row["momentum_min_quantile"]),
        "acceleration_min_quantile": float(row["acceleration_min_quantile"]),
        "liquidity_filter": str(row["liquidity_filter"]),
        "drawdown_filter": str(row["drawdown_filter"]),
    }


def _filtered_recovered_pool(frame: pd.DataFrame, condition: dict[str, Any]) -> pd.DataFrame:
    if frame.empty or not condition:
        return frame.iloc[0:0].copy()

    p_good_top_pct = float(condition["p_good_top_pct"])
    p_bad_suffix = int(float(condition["p_bad_max_quantile"]) * 100)
    top_n = np.ceil(frame["_daily_all_count"] * p_good_top_pct).clip(lower=1)
    mask = (
        _base_core_mask(frame)
        & frame["_p_good_rank_all"].le(top_n)
        & frame["p_bad_entry"].le(frame[f"_p_bad_q{p_bad_suffix}"])
        & frame["ml_rank_global"].le(int(condition["ml_rank_global_max"]))
        & frame["ml_rank_sector"].le(int(condition["ml_rank_sector_max"]))
        & ~frame["market_state"].str.lower().eq("defense")
        & ~frame["sector_state"].str.lower().eq("weak")
        & frame["is_valid_sample"]
        & ~frame["exclude_reason"].str.lower().str.contains("invalid", na=False)
    )

    momentum_quantile = condition.get("momentum_min_quantile")
    if momentum_quantile is not None:
        mask &= frame["momentum_score"].ge(frame[f"_momentum_q{int(float(momentum_quantile) * 100)}"])

    acceleration_quantile = condition.get("acceleration_min_quantile")
    if acceleration_quantile is not None:
        mask &= frame["acceleration_score"].ge(frame[f"_acceleration_q{int(float(acceleration_quantile) * 100)}"])

    liquidity_filter = str(condition.get("liquidity_filter") or "none")
    if liquidity_filter == "etf_rank_top_500":
        mask &= frame["etf_rank"].le(500)
    elif liquidity_filter == "etf_rank_top_750":
        mask &= frame["etf_rank"].le(750)

    drawdown_filter = str(condition.get("drawdown_filter") or "none")
    if drawdown_filter == "expected_drawdown_ge_p30":
        mask &= frame["expected_drawdown_10d"].ge(frame["_drawdown_q30"])
    elif drawdown_filter == "expected_drawdown_ge_p50":
        mask &= frame["expected_drawdown_10d"].ge(frame["_drawdown_q50"])
    elif drawdown_filter == "expected_drawdown_ge_p60":
        mask &= frame["expected_drawdown_10d"].ge(frame["_drawdown_q60"])

    return frame.loc[mask.fillna(False)].copy()


def _base_core_pool(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[_base_core_mask(frame)].copy()


def _base_core_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["_is_original_recovered"]
        & ~frame["market_state"].str.lower().eq("defense")
        & ~frame["sector_state"].str.lower().eq("weak")
        & frame["is_valid_sample"]
        & ~frame["exclude_reason"].str.lower().str.contains("invalid", na=False)
    ).fillna(False)


def _build_manual_review(strong_pool: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    if strong_pool.empty:
        return pd.DataFrame(columns=MANUAL_REVIEW_COLUMNS), _manual_summary(strong_pool)

    review = strong_pool.sort_values(["trade_date", "ml_score", "p_good_entry"], ascending=[True, False, False]).copy()
    summary = _manual_summary(review)
    top_dates = set(summary["top_trade_dates"].keys())
    top_sectors = set(summary["top_sector_level1"].keys())

    review["是否集中在少数日期"] = review["trade_date"].isin(top_dates) & bool(summary["date_concentration"])
    review["是否集中在少数行业"] = review["sector_level1"].isin(top_sectors) & bool(summary["sector_concentration"])
    review["是否存在数据异常"] = (
        ~review["is_valid_sample"]
        | review["exclude_reason"].str.lower().str.contains("invalid|nan|missing", na=False)
        | review[["ml_score", "p_good_entry", "p_bad_entry", "future_return_10d"]].isna().any(axis=1)
    )
    review["是否属于高弹性脉冲"] = (
        review["acceleration_score"].ge(review["_acceleration_q80"])
        | review["volatility_20"].ge(review["_volatility_q80"])
        | review["future_return_1d"].abs().ge(0.03)
    ).fillna(False)
    review["是否符合右侧买点"] = (
        review["momentum_score"].gt(0)
        & review["acceleration_score"].ge(review["_acceleration_q30"])
        & review["p_bad_entry"].le(review["_p_bad_q50"])
        & ~review["market_state"].str.lower().eq("defense")
        & ~review["sector_state"].str.lower().eq("weak")
    ).fillna(False)
    review["人工复核建议"] = ""
    review["manual_review_note"] = ""
    return review[MANUAL_REVIEW_COLUMNS], summary


def _manual_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "sample_count": 0,
            "unique_trade_dates": 0,
            "unique_sector_level1": 0,
            "date_concentration": False,
            "sector_concentration": False,
            "top_trade_dates": {},
            "top_sector_level1": {},
        }

    date_counts = frame["trade_date"].value_counts().head(5)
    sector_counts = frame["sector_level1"].fillna("").replace("", "UNKNOWN").value_counts().head(5)
    count = int(len(frame))
    return {
        "sample_count": count,
        "unique_trade_dates": int(frame["trade_date"].nunique()),
        "unique_sector_level1": int(frame["sector_level1"].nunique()),
        "date_concentration": bool((date_counts.iloc[0] / count) >= 0.30 or frame["trade_date"].nunique() <= 3),
        "sector_concentration": bool((sector_counts.iloc[0] / count) >= 0.50 or frame["sector_level1"].nunique() <= 2),
        "top_trade_dates": {str(k): int(v) for k, v in date_counts.items()},
        "top_sector_level1": {str(k): int(v) for k, v in sector_counts.items()},
    }


def _comparison_rows(
    *,
    frame: pd.DataFrame,
    core_pool: pd.DataFrame,
    legacy: dict[str, Any],
    random_baseline: dict[str, Any],
    original: dict[str, Any],
    strong: dict[str, Any],
    total_trade_dates: int,
) -> pd.DataFrame:
    rows = [
        legacy,
        random_baseline,
        original,
        strong,
        _metrics(_daily_top(core_pool, 20), "ml_core_recovered_top20", "candidate ML_CORE_RECOVERED Top20", legacy=legacy, total_trade_dates=total_trade_dates),
        _metrics(_daily_top(core_pool, 50), "ml_core_recovered_top50", "candidate ML_CORE_RECOVERED Top50", legacy=legacy, total_trade_dates=total_trade_dates),
        _metrics(_daily_top(core_pool, 100), "ml_core_recovered_top100", "candidate ML_CORE_RECOVERED Top100", legacy=legacy, total_trade_dates=total_trade_dates),
        _metrics(core_pool, "ml_core_recovered_all", "candidate ML_CORE_RECOVERED all", legacy=legacy, total_trade_dates=total_trade_dates),
    ]
    return pd.DataFrame(rows)


def _select_best_core_row(grid: pd.DataFrame, legacy: dict[str, Any]) -> pd.Series:
    if grid.empty:
        return pd.Series(dtype=object)

    viable = grid.loc[grid["quality_beats_legacy"] & grid["meets_sample_floor"]].copy()
    if viable.empty:
        viable = grid.loc[grid["quality_beats_legacy"] & grid["sample_count"].gt(0)].copy()
    if viable.empty:
        viable = grid.loc[grid["sample_count"].gt(0)].copy()
    if viable.empty:
        return pd.Series(dtype=object)

    target_penalty = (viable["daily_average_count"] - 2.0).abs().where(viable["meets_target_daily_average"], 2.0)
    viable["_score"] = (
        (viable["good_entry_rate"] - legacy["good_entry_rate"]) * 6.0
        - (viable["bad_entry_rate"] - legacy["bad_entry_rate"]) * 4.0
        + (viable["future_return_10d"] - legacy["future_return_10d"]) * 2.0
        + np.log1p(viable["sample_count"]) * 0.002
        - target_penalty * 0.01
    )
    return viable.sort_values(["_score", "sample_count"], ascending=[False, False]).iloc[0]


def _recommend(grid: pd.DataFrame, best: pd.Series, legacy: dict[str, Any], original: dict[str, Any]) -> str:
    if int(original["sample_count"]) == 0:
        return "DISABLE_ML_RECOVERY"
    if best.empty or int(best.get("sample_count") or 0) == 0:
        return "CONTINUE_SHADOW"

    qualifying = grid.loc[grid["quality_beats_legacy"] & grid["meets_sample_floor"]]
    if qualifying.empty:
        ranking_only = grid.loc[grid["meets_sample_floor"] & grid["ranking_only_signal"]]
        if not ranking_only.empty:
            return "USE_ML_AS_RANKING_ONLY"
        return "CONTINUE_SHADOW"

    if bool(best["meets_target_daily_average"]):
        return "ALLOW_LIMITED_ACTIVE_SIM"
    return "TIGHTEN_ML_RECOVERED_THRESHOLD"


def _build_report_json(
    *,
    frame: pd.DataFrame,
    manual_summary: dict[str, Any],
    legacy: dict[str, Any],
    original: dict[str, Any],
    random_baseline: dict[str, Any],
    strong: dict[str, Any],
    grid: pd.DataFrame,
    best: pd.Series,
    comparison: pd.DataFrame,
    recommendation: str,
) -> dict[str, Any]:
    return {
        "mode": "ML_CORE_RECOVERED_OFFLINE_RECOMMENDATION",
        "recommendation": recommendation,
        "allowed_recommendations": sorted(DECISIONS),
        "source_rows": int(len(frame)),
        "trade_date_count": int(frame["trade_date"].nunique()) if not frame.empty else 0,
        "manual_review_summary": manual_summary,
        "legacy_v21_buy_probe": legacy,
        "random_baseline": random_baseline,
        "original_ml_recovered": original,
        "ml_strong_recovered": strong,
        "best_core_condition": {} if best.empty else _jsonable(best.drop(labels=["_score"], errors="ignore").to_dict()),
        "comparison": _jsonable(comparison.replace({np.nan: None}).to_dict(orient="records")),
        "grid_rows": int(len(grid)),
        "core_candidate_counts": {
            "quality_beats_legacy": int(grid["quality_beats_legacy"].sum()) if not grid.empty else 0,
            "meets_sample_floor": int(grid["meets_sample_floor"].sum()) if not grid.empty else 0,
            "quality_and_sample_floor": int((grid["quality_beats_legacy"] & grid["meets_sample_floor"]).sum()) if not grid.empty else 0,
            "target_daily_average": int(grid["meets_target_daily_average"].sum()) if not grid.empty else 0,
        },
        "hard_gate": _hard_gate(),
    }


def _build_report_markdown(report_json: dict[str, Any], grid: pd.DataFrame, comparison: pd.DataFrame) -> str:
    lines = [
        "# ML_CORE_RECOVERED Offline Recommendation",
        "",
        "## control_center Conclusion",
        "",
        f"- recommendation: {report_json['recommendation']}",
        "- formal_entry_change: no",
        "- final_buy_action_change: no",
        "- BUY/PROBE threshold_change: no",
        "- QMT triggered: no",
        "- market_data_refreshed: no",
        "- data_cache_written: no",
        f"- source_rows: {report_json['source_rows']}",
        f"- trade_date_count: {report_json['trade_date_count']}",
        "",
        "## Best ML_CORE_RECOVERED Condition",
        "",
        _dict_lines(report_json["best_core_condition"]),
        "",
        "## Required Comparison",
        "",
        _markdown_table(comparison),
        "",
        "## Core Candidate Counts",
        "",
        _dict_lines(report_json["core_candidate_counts"]),
        "",
        "## Top Grid Candidates",
        "",
        _markdown_table(grid.head(20)),
        "",
        "## Boundary",
        "",
        "- Offline historical_ml report only.",
        "- No BUY/PROBE threshold, final_buy_action, formal entry, QMT, market data refresh, or data/cache change.",
        "- If no core condition beats legacy on good_entry_rate and bad_entry_rate, the only allowed conclusion is CONTINUE_SHADOW.",
        "- If the signal only improves return while good_entry_rate does not beat legacy, the only allowed conclusion is USE_ML_AS_RANKING_ONLY.",
        "",
    ]
    return "\n".join(lines)


def _build_manual_review_markdown(review: pd.DataFrame, summary: dict[str, Any]) -> str:
    lines = [
        "# ML_STRONG_RECOVERED Manual Review",
        "",
        "## Summary",
        "",
        _dict_lines(summary),
        "",
        "## Manual Review Queue",
        "",
        _markdown_table(review),
        "",
        "## Boundary",
        "",
        "- Offline manual review list only.",
        "- Reserved manual fields are intentionally blank for human review.",
        "- No formal entry, final_buy_action, BUY/PROBE threshold, QMT, market data refresh, or data/cache change.",
        "",
    ]
    return "\n".join(lines)


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
    return frame.groupby("trade_date", group_keys=False).apply(lambda group: group.sample(n=min(per_day, len(group)), random_state=20260525))


def _daily_top(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    top = frame.copy()
    top["_top_rank"] = top.groupby("trade_date")["ml_score"].rank(ascending=False, method="first")
    return top.loc[top["_top_rank"].le(int(top_n))].drop(columns=["_top_rank"])


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


def _hard_gate() -> dict[str, bool]:
    return {
        "formal_entry_changed": False,
        "final_buy_action_changed": False,
        "buy_probe_threshold_changed": False,
        "qmt_triggered": False,
        "market_data_refreshed": False,
        "data_cache_written": False,
        "offline_recommendation_only": True,
    }


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
