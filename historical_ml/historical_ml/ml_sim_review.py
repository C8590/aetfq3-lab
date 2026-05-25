from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import HistoricalMLConfig
from .io_utils import ensure_dir, read_price_data, read_table


REVIEW_FILLED_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "sector_level1",
    "sector_level2",
    "market_state",
    "sector_state",
    "legacy_action",
    "ml_sim_action",
    "final_action",
    "ml_score",
    "p_good_entry",
    "p_bad_entry",
    "ml_adjustment_type",
    "ml_adjustment_bucket",
    "review_priority",
    "future_return_1d",
    "future_return_3d",
    "future_return_5d",
    "future_return_10d",
    "future_max_drawdown_3d",
    "future_max_drawdown_5d",
    "future_max_drawdown_10d",
    "outperform_market_3d",
    "outperform_market_5d",
    "outperform_market_10d",
    "outperform_sector_3d",
    "outperform_sector_5d",
    "outperform_sector_10d",
    "auto_label",
    "review_status",
    "review_reason_cn",
]

EFFECTIVENESS_COLUMNS = [
    "category",
    "segment",
    "group_key",
    "horizon",
    "row_count",
    "ready_count",
    "pending_count",
    "missing_price_count",
    "avg_future_return",
    "avg_future_max_drawdown",
    "good_entry_rate",
    "bad_entry_rate",
    "outperform_market_rate",
    "outperform_sector_rate",
    "avg_ml_score",
    "avg_p_good_entry",
    "avg_p_bad_entry",
]

DECISIONS = {
    "CONTINUE_SHADOW",
    "CONTINUE_ML_SIM",
    "TIGHTEN_ML_RECOVERED_THRESHOLD",
    "ALLOW_LIMITED_ACTIVE_SIM",
}


@dataclass(frozen=True)
class MLSimWeeklyReviewResult:
    review_filled: pd.DataFrame
    effectiveness_summary: pd.DataFrame
    report: str
    report_json: dict[str, Any]
    recommendation: str
    output_paths: dict[str, Path]


def build_ml_sim_weekly_review_from_files(
    *,
    review_queue_path: str | Path,
    prices_path: str | Path,
    out_dir: str | Path,
    comparison_path: str | Path | None = None,
    summary_path: str | Path | None = None,
    daily_decision_snapshot_path: str | Path | None = None,
    config: HistoricalMLConfig = HistoricalMLConfig(),
) -> MLSimWeeklyReviewResult:
    review_queue = read_table(review_queue_path)
    prices = read_price_data(prices_path)
    comparison = read_table(comparison_path) if comparison_path and Path(comparison_path).exists() else None
    summary = _read_json(summary_path) if summary_path and Path(summary_path).exists() else {}
    daily_snapshot = read_table(daily_decision_snapshot_path) if daily_decision_snapshot_path and Path(daily_decision_snapshot_path).exists() else None
    return build_ml_sim_weekly_review(
        review_queue=review_queue,
        prices=prices,
        out_dir=out_dir,
        comparison=comparison,
        summary=summary,
        daily_decision_snapshot=daily_snapshot,
        config=config,
    )


def build_ml_sim_weekly_review(
    *,
    review_queue: pd.DataFrame,
    prices: pd.DataFrame,
    out_dir: str | Path,
    comparison: pd.DataFrame | None = None,
    summary: dict[str, Any] | None = None,
    daily_decision_snapshot: pd.DataFrame | None = None,
    config: HistoricalMLConfig = HistoricalMLConfig(),
) -> MLSimWeeklyReviewResult:
    summary = summary or {}
    market_state = _snapshot_market_state(daily_decision_snapshot)
    label_frame, label_dates, known_codes = _build_label_frame(prices, horizons=(1, 3, 5, 10))

    review_filled = _fill_rows(review_queue, label_frame, label_dates, known_codes, market_state, config=config)
    analysis_source = comparison if comparison is not None and not comparison.empty else review_queue
    analysis_frame = _fill_rows(analysis_source, label_frame, label_dates, known_codes, market_state, config=config)

    effectiveness = _build_effectiveness_summary(analysis_frame)
    report_json = _build_report_json(review_filled, analysis_frame, effectiveness, summary)
    recommendation = _recommend(report_json)
    report_json["recommendation"] = recommendation
    report_json["allowed_recommendations"] = sorted(DECISIONS)
    report = _build_report_markdown(report_json, effectiveness)

    out = ensure_dir(out_dir)
    paths = {
        "review_filled": out / "ml_sim_review_filled.csv",
        "weekly_report_md": out / "ml_sim_weekly_review_report.md",
        "weekly_report_json": out / "ml_sim_weekly_review_report.json",
        "effectiveness_summary": out / "ml_sim_effectiveness_summary.csv",
    }
    review_filled.to_csv(paths["review_filled"], index=False, encoding="utf-8-sig")
    effectiveness.to_csv(paths["effectiveness_summary"], index=False, encoding="utf-8-sig")
    paths["weekly_report_md"].write_text(report, encoding="utf-8")
    paths["weekly_report_json"].write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")

    return MLSimWeeklyReviewResult(
        review_filled=review_filled,
        effectiveness_summary=effectiveness,
        report=report,
        report_json=report_json,
        recommendation=recommendation,
        output_paths=paths,
    )


def _build_label_frame(prices: pd.DataFrame, horizons: Iterable[int]) -> tuple[pd.DataFrame, list[pd.Timestamp], set[str]]:
    frame = prices.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["code"] = frame["code"].astype(str).str.zfill(6)
    if "sector" not in frame.columns:
        frame["sector"] = frame.get("sector_level2", "")
    if "sector_level2" not in frame.columns:
        frame["sector_level2"] = frame["sector"]
    for column in ("close", "high", "low"):
        if column not in frame.columns:
            frame[column] = frame["close"]
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["sector"] = frame["sector"].fillna(frame["sector_level2"]).fillna("").astype(str)
    frame = frame.sort_values(["code", "date"]).drop_duplicates(["code", "date"], keep="last")
    dates = list(pd.to_datetime(frame["date"].dropna().sort_values().unique()))
    codes = list(frame["code"].dropna().astype(str).sort_values().unique())
    if not dates or not codes:
        return pd.DataFrame(columns=["code", "date"]), dates, set()

    index = pd.MultiIndex.from_product([codes, dates], names=["code", "date"])
    grid = frame.set_index(["code", "date"]).reindex(index).reset_index()
    grid["sector"] = grid.groupby("code")["sector"].transform(lambda value: value.ffill().bfill()).fillna("").astype(str)
    grouped = grid.groupby("code", sort=False)
    close = pd.to_numeric(grid["close"], errors="coerce")

    for horizon in sorted({int(h) for h in horizons}):
        grid[f"future_return_{horizon}d"] = grouped["close"].transform(
            lambda value, h=horizon: pd.to_numeric(value, errors="coerce").shift(-h) / pd.to_numeric(value, errors="coerce") - 1.0
        )
        grid[f"future_max_drawdown_{horizon}d"] = (
            grouped["low"].transform(lambda value, h=horizon: _forward_window(pd.to_numeric(value, errors="coerce"), h, "min")) / close - 1.0
        )
        grid[f"market_return_{horizon}d"] = grid.groupby("date")[f"future_return_{horizon}d"].transform("mean")
        sector_returns = grid.groupby(["date", "sector"], dropna=False)[f"future_return_{horizon}d"].mean().rename(f"sector_return_{horizon}d")
        grid = grid.merge(sector_returns.reset_index(), on=["date", "sector"], how="left")

    future_columns = [column for column in grid.columns if column.startswith("future_")]
    grid.loc[~close.gt(0), future_columns] = np.nan
    return grid, dates, set(codes)


def _forward_window(series: pd.Series, window: int, method: str) -> pd.Series:
    reverse = series.iloc[::-1].shift(1).rolling(window=window, min_periods=1)
    result = getattr(reverse, method)().iloc[::-1]
    result.index = series.index
    return result


def _fill_rows(
    rows: pd.DataFrame,
    label_frame: pd.DataFrame,
    label_dates: list[pd.Timestamp],
    known_codes: set[str],
    market_state: str,
    *,
    config: HistoricalMLConfig,
) -> pd.DataFrame:
    base = rows.copy().reset_index(drop=True)
    if base.empty:
        return pd.DataFrame(columns=REVIEW_FILLED_COLUMNS)

    base["trade_date"] = pd.to_datetime(base["trade_date"], errors="coerce").dt.normalize()
    base["code"] = base["code"].astype(str).str.zfill(6)
    if "sector_level1" not in base.columns:
        base["sector_level1"] = base.get("sector", "")
    if "sector_level2" not in base.columns:
        base["sector_level2"] = base.get("sector", base.get("sector_level1", ""))
    if "market_state" not in base.columns:
        base["market_state"] = market_state
    base["market_state"] = base["market_state"].fillna("").astype(str)
    if market_state:
        base.loc[base["market_state"].eq(""), "market_state"] = market_state
    if "sector_state" not in base.columns:
        base["sector_state"] = base["sector_level2"]
    base["sector_state"] = base["sector_state"].fillna(base["sector_level2"]).astype(str)
    base["ml_adjustment_bucket"] = base.get("ml_adjustment_type", pd.Series("", index=base.index)).map(_adjustment_bucket)

    generated_columns = []
    for horizon in (1, 3, 5, 10):
        generated_columns.extend(
            [
                f"future_return_{horizon}d",
                f"future_max_drawdown_{horizon}d",
                f"market_return_{horizon}d",
                f"sector_return_{horizon}d",
                f"outperform_market_{horizon}d",
                f"outperform_sector_{horizon}d",
            ]
        )
    base = base.drop(columns=[column for column in generated_columns if column in base.columns], errors="ignore")

    lookup_columns = ["code", "date", "close"]
    for horizon in (1, 3, 5, 10):
        lookup_columns.extend(
            [
                f"future_return_{horizon}d",
                f"future_max_drawdown_{horizon}d",
                f"market_return_{horizon}d",
                f"sector_return_{horizon}d",
            ]
        )
    lookup = label_frame[[column for column in lookup_columns if column in label_frame.columns]].rename(columns={"close": "_label_close"})
    filled = base.merge(lookup, left_on=["code", "trade_date"], right_on=["code", "date"], how="left").drop(columns=["date"], errors="ignore")

    label_date_pos = {date: idx for idx, date in enumerate(label_dates)}
    known_date = filled["trade_date"].isin(set(label_dates))
    known_code = filled["code"].isin(known_codes)
    base_price = pd.to_numeric(filled.get("_label_close", pd.Series(np.nan, index=filled.index)), errors="coerce")
    enough_future = filled["trade_date"].map(label_date_pos).fillna(len(label_dates)).astype(int) + 10 < len(label_dates)

    filled["review_status"] = "READY"
    filled.loc[~known_date | ~known_code | base_price.isna() | ~base_price.gt(0), "review_status"] = "MISSING_PRICE"
    filled.loc[filled["review_status"].eq("READY") & ~enough_future, "review_status"] = "PENDING_NOT_ENOUGH_FUTURE_DATA"
    missing_future = pd.to_numeric(filled.get("future_return_10d"), errors="coerce").isna()
    filled.loc[filled["review_status"].eq("READY") & missing_future, "review_status"] = "MISSING_PRICE"

    for horizon in (3, 5, 10):
        filled[f"outperform_market_{horizon}d"] = (
            pd.to_numeric(filled.get(f"future_return_{horizon}d"), errors="coerce")
            > pd.to_numeric(filled.get(f"market_return_{horizon}d"), errors="coerce")
        )
        filled[f"outperform_sector_{horizon}d"] = (
            pd.to_numeric(filled.get(f"future_return_{horizon}d"), errors="coerce")
            > pd.to_numeric(filled.get(f"sector_return_{horizon}d"), errors="coerce")
        )

    filled["auto_label"] = _auto_label(filled, config=config)
    filled["review_reason_cn"] = filled.apply(_review_reason, axis=1)
    for column in ("ml_score", "p_good_entry", "p_bad_entry"):
        if column in filled.columns:
            filled[column] = pd.to_numeric(filled[column], errors="coerce")
    filled["trade_date"] = filled["trade_date"].dt.strftime("%Y-%m-%d")

    for column in REVIEW_FILLED_COLUMNS:
        if column not in filled.columns:
            filled[column] = ""
    internal_columns = {"_label_close", "future_max_drawdown_1d"}
    rest = [
        column
        for column in filled.columns
        if column not in REVIEW_FILLED_COLUMNS
        and column not in internal_columns
        and not column.startswith("market_return_")
        and not column.startswith("sector_return_")
    ]
    return filled[REVIEW_FILLED_COLUMNS + rest]


def _auto_label(frame: pd.DataFrame, *, config: HistoricalMLConfig) -> pd.Series:
    status_ok = frame["review_status"].eq("READY")
    ret10 = pd.to_numeric(frame.get("future_return_10d"), errors="coerce")
    ret3 = pd.to_numeric(frame.get("future_return_3d"), errors="coerce")
    dd10 = pd.to_numeric(frame.get("future_max_drawdown_10d"), errors="coerce")
    dd3 = pd.to_numeric(frame.get("future_max_drawdown_3d"), errors="coerce")
    good = status_ok & ret10.ge(config.good_return_10d) & dd10.gt(config.bad_drawdown_10d) & (
        frame.get("outperform_market_10d", False).astype(bool) | frame.get("outperform_sector_10d", False).astype(bool)
    )
    bad = status_ok & (ret10.le(config.bad_return_10d) | dd10.le(config.bad_drawdown_10d) | ret3.le(config.quick_failure_return_3d) | dd3.le(config.quick_failure_return_3d))
    labels = pd.Series("unlabeled", index=frame.index)
    labels.loc[status_ok] = "neutral_entry"
    labels.loc[good] = "good_entry"
    labels.loc[bad] = "bad_entry"
    return labels


def _review_reason(row: pd.Series) -> str:
    status = str(row.get("review_status") or "")
    bucket = str(row.get("ml_adjustment_bucket") or "")
    if status == "READY":
        return f"未来10个交易日数据已齐备，可用于事后复盘；调整类型={bucket}。"
    if status == "PENDING_NOT_ENOUGH_FUTURE_DATA":
        return "距离样本日不足10个交易日，保持待复盘，禁止强行打标签。"
    if status == "MISSING_PRICE":
        return "样本日或未来窗口价格缺失，暂不能纳入有效性统计。"
    return "复盘状态未知。"


def _build_effectiveness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket in ("ML_RECOVERED", "ML_DOWNGRADED", "ML_UNCHANGED", "ML_CONFLICT_WITH_RISK"):
        sub = frame.loc[frame["ml_adjustment_bucket"].eq(bucket)]
        for horizon in (1, 3, 5, 10):
            rows.append(_summary_row(sub, "adjustment", bucket, bucket, horizon))

    recovered = frame.loc[frame["ml_adjustment_bucket"].eq("ML_RECOVERED")].sort_values(["ml_score", "p_good_entry"], ascending=[False, False])
    for top_n in (20, 50, 100):
        rows.append(_summary_row(recovered.head(top_n), "ml_recovered_topn", f"Top{top_n}", "ML_RECOVERED", 10))

    legacy = frame.loc[frame.get("legacy_action", pd.Series("", index=frame.index)).astype(str).str.upper().isin(["BUY", "PROBE"])]
    rows.append(_summary_row(legacy, "legacy_vs_recovered", "legacy_v21_selected", "legacy_v21_selected", 10))
    rows.append(_summary_row(recovered, "legacy_vs_recovered", "ml_recovered", "ML_RECOVERED", 10))

    for column in ("ml_score", "p_good_entry", "p_bad_entry"):
        for label, sub in _score_bins(frame, column):
            rows.append(_summary_row(sub, f"{column}_bin", label, label, 10))

    for column, category in (("market_state", "market_state"), ("sector_state", "sector_state")):
        if column in frame.columns:
            for value, sub in frame.groupby(frame[column].fillna("").astype(str), dropna=False):
                rows.append(_summary_row(sub, category, value or "UNKNOWN", value or "UNKNOWN", 10))

    return pd.DataFrame(rows, columns=EFFECTIVENESS_COLUMNS)


def _summary_row(frame: pd.DataFrame, category: str, segment: str, group_key: str, horizon: int) -> dict[str, Any]:
    status = frame.get("review_status", pd.Series("", index=frame.index))
    ready = frame.loc[status.eq("READY")].copy()
    ret = pd.to_numeric(ready.get(f"future_return_{horizon}d"), errors="coerce")
    dd = pd.to_numeric(ready.get(f"future_max_drawdown_{horizon}d"), errors="coerce")
    return {
        "category": category,
        "segment": segment,
        "group_key": group_key,
        "horizon": f"{horizon}d",
        "row_count": int(len(frame)),
        "ready_count": int(len(ready)),
        "pending_count": int(status.eq("PENDING_NOT_ENOUGH_FUTURE_DATA").sum()),
        "missing_price_count": int(status.eq("MISSING_PRICE").sum()),
        "avg_future_return": _round(ret.mean()),
        "avg_future_max_drawdown": _round(dd.mean()),
        "good_entry_rate": _round(ready.get("auto_label", pd.Series("", index=ready.index)).eq("good_entry").mean()) if not ready.empty else np.nan,
        "bad_entry_rate": _round(ready.get("auto_label", pd.Series("", index=ready.index)).eq("bad_entry").mean()) if not ready.empty else np.nan,
        "outperform_market_rate": _round(ready.get(f"outperform_market_{horizon}d", pd.Series(False, index=ready.index)).astype(bool).mean()) if not ready.empty else np.nan,
        "outperform_sector_rate": _round(ready.get(f"outperform_sector_{horizon}d", pd.Series(False, index=ready.index)).astype(bool).mean()) if not ready.empty else np.nan,
        "avg_ml_score": _round(pd.to_numeric(frame.get("ml_score"), errors="coerce").mean()),
        "avg_p_good_entry": _round(pd.to_numeric(frame.get("p_good_entry"), errors="coerce").mean()),
        "avg_p_bad_entry": _round(pd.to_numeric(frame.get("p_bad_entry"), errors="coerce").mean()),
    }


def _score_bins(frame: pd.DataFrame, column: str) -> list[tuple[str, pd.DataFrame]]:
    if column not in frame.columns:
        return []
    values = pd.to_numeric(frame[column], errors="coerce")
    valid = frame.loc[values.notna()].copy()
    if valid.empty:
        return []
    valid["_bin_value"] = values.loc[valid.index]
    try:
        bins = pd.qcut(valid["_bin_value"], q=min(5, valid["_bin_value"].nunique()), duplicates="drop")
    except ValueError:
        return [(f"{column}:all", valid.drop(columns=["_bin_value"]))]
    out: list[tuple[str, pd.DataFrame]] = []
    for idx, (interval, sub) in enumerate(valid.groupby(bins, observed=False), start=1):
        out.append((f"{column}:bin{idx}:{interval}", sub.drop(columns=["_bin_value"])))
    return out


def _build_report_json(review_filled: pd.DataFrame, analysis_frame: pd.DataFrame, effectiveness: pd.DataFrame, summary: dict[str, Any]) -> dict[str, Any]:
    total = int(len(analysis_frame))
    ready = int(analysis_frame["review_status"].eq("READY").sum()) if "review_status" in analysis_frame.columns else 0
    pending = int(analysis_frame["review_status"].eq("PENDING_NOT_ENOUGH_FUTURE_DATA").sum()) if "review_status" in analysis_frame.columns else 0
    missing = int(analysis_frame["review_status"].eq("MISSING_PRICE").sum()) if "review_status" in analysis_frame.columns else 0
    recovered_count = int(analysis_frame["ml_adjustment_bucket"].eq("ML_RECOVERED").sum()) if "ml_adjustment_bucket" in analysis_frame.columns else 0
    downgraded_count = int(analysis_frame["ml_adjustment_bucket"].eq("ML_DOWNGRADED").sum()) if "ml_adjustment_bucket" in analysis_frame.columns else 0
    recovered_ratio = recovered_count / total if total else 0.0
    return {
        "mode": "V2.1_ML_SIM_WEEKLY_REVIEW",
        "source_summary": summary,
        "total_rows": total,
        "review_queue_rows": int(len(review_filled)),
        "ready_count": ready,
        "pending_count": pending,
        "missing_price_count": missing,
        "ml_recovered_count": recovered_count,
        "ml_downgraded_count": downgraded_count,
        "ml_recovered_ratio": _round(recovered_ratio),
        "ml_recovered_overwide": recovered_count > 100 or recovered_ratio >= 0.30,
        "topn": _effectiveness_records(effectiveness, "ml_recovered_topn"),
        "legacy_vs_recovered": _effectiveness_records(effectiveness, "legacy_vs_recovered"),
        "adjustment_summary": _effectiveness_records(effectiveness, "adjustment"),
        "leakage_control": {
            "future_labels_used_for": "post_trade_review_only",
            "formal_entry_changed": False,
            "qmt_triggered": False,
            "threshold_changed": False,
        },
    }


def _recommend(report_json: dict[str, Any]) -> str:
    if int(report_json.get("ready_count") or 0) == 0:
        return "CONTINUE_ML_SIM"
    lookup = {(row["category"], row["segment"]): row for row in report_json.get("topn", []) + report_json.get("legacy_vs_recovered", [])}
    legacy = lookup.get(("legacy_vs_recovered", "legacy_v21_selected"), {})
    top_rows = [lookup.get(("ml_recovered_topn", f"Top{n}"), {}) for n in (20, 50, 100)]
    legacy_ret = _float(legacy.get("avg_future_return"))
    legacy_good = _float(legacy.get("good_entry_rate"))
    legacy_bad = _float(legacy.get("bad_entry_rate"))
    usable_top_rows = [row for row in top_rows if int(row.get("ready_count") or 0) > 0]
    recovered_better = bool(usable_top_rows) and all(
        _float(row.get("avg_future_return")) > legacy_ret
        and _float(row.get("good_entry_rate")) >= legacy_good
        and _float(row.get("bad_entry_rate")) <= legacy_bad
        for row in usable_top_rows
    )
    if recovered_better and not report_json.get("ml_recovered_overwide"):
        return "ALLOW_LIMITED_ACTIVE_SIM"
    if report_json.get("ml_recovered_overwide"):
        return "TIGHTEN_ML_RECOVERED_THRESHOLD"
    return "CONTINUE_SHADOW"


def _build_report_markdown(report_json: dict[str, Any], effectiveness: pd.DataFrame) -> str:
    ready = report_json["ready_count"]
    overwide_text = "是，恢复池过宽，需要按 Top20/Top50/Top100 分层观察。" if report_json["ml_recovered_overwide"] else "否，恢复池宽度暂未超过警戒线。"
    recovered_answer = "当前没有 READY 样本，不能证明 ML_RECOVERED 已提升 entry 精准度。" if ready == 0 else _recovered_answer(effectiveness)
    downgraded_answer = "当前没有 READY 样本，不能证明 ML_DOWNGRADED 已过滤坏买点。" if ready == 0 else _downgraded_answer(effectiveness)
    lines = [
        "# V2.1 ML_SIM 周复盘报告",
        "",
        "## control_center 结论",
        "",
        f"- recommendation: {report_json['recommendation']}",
        "- formal_entry_change: no",
        "- qmt_triggered: no",
        "- threshold_change: no",
        f"- ML_RECOVERED 是否有效: {recovered_answer}",
        f"- ML_DOWNGRADED 是否有效: {downgraded_answer}",
        f"- recovered 数量是否过宽: {overwide_text} 当前 {report_json['ml_recovered_count']} / {report_json['total_rows']} ({report_json['ml_recovered_ratio']:.2%})。",
        "",
        "## 数据与防泄漏边界",
        "",
        "- 本报告只读取 ML_SIM 观测输出和本地历史行情，用未来窗口做事后复盘。",
        "- future_return / max_drawdown / outperform 字段不得回写 entry 当日决策，不得调整 BUY/PROBE 阈值。",
        "- 最近不足 10 个交易日的样本标记为 PENDING_NOT_ENOUGH_FUTURE_DATA，不强行标注 good_entry 或 bad_entry。",
        "",
        "## 样本状态",
        "",
        f"- total_rows: {report_json['total_rows']}",
        f"- review_queue_rows: {report_json['review_queue_rows']}",
        f"- READY: {report_json['ready_count']}",
        f"- PENDING_NOT_ENOUGH_FUTURE_DATA: {report_json['pending_count']}",
        f"- MISSING_PRICE: {report_json['missing_price_count']}",
        "",
        "## ML_RECOVERED TopN",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].eq("ml_recovered_topn")]),
        "",
        "## legacy_v21 selected vs ml_recovered",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].eq("legacy_vs_recovered")]),
        "",
        "## 调整类型表现",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].eq("adjustment") & effectiveness["horizon"].eq("10d")]),
        "",
        "## 分层表现",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].isin(["ml_score_bin", "p_good_entry_bin", "p_bad_entry_bin"])]),
        "",
        "## market_state / sector_state 分组",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].isin(["market_state", "sector_state"])]),
        "",
    ]
    return "\n".join(lines)


def _recovered_answer(effectiveness: pd.DataFrame) -> str:
    legacy = _one_metric(effectiveness, "legacy_vs_recovered", "legacy_v21_selected")
    top20 = _one_metric(effectiveness, "ml_recovered_topn", "Top20")
    if not legacy or not top20 or int(top20.get("ready_count") or 0) == 0:
        return "READY TopN 样本不足，不能证明优于 legacy。"
    if _float(top20.get("avg_future_return")) > _float(legacy.get("avg_future_return")) and _float(top20.get("good_entry_rate")) >= _float(legacy.get("good_entry_rate")):
        return "Top20 暂时优于 legacy，但仍需结合坏买点率和更长样本确认。"
    return "Top20 未优于 legacy，不允许扩大 ML 权限。"


def _downgraded_answer(effectiveness: pd.DataFrame) -> str:
    down = _one_metric(effectiveness, "adjustment", "ML_DOWNGRADED")
    legacy = _one_metric(effectiveness, "legacy_vs_recovered", "legacy_v21_selected")
    if not down or int(down.get("ready_count") or 0) == 0:
        return "ML_DOWNGRADED READY 样本不足，继续观察，不自动改正式交易。"
    if _float(down.get("avg_future_return")) < _float(legacy.get("avg_future_return")) or _float(down.get("avg_future_max_drawdown")) < _float(legacy.get("avg_future_max_drawdown")):
        return "降级样本后续表现更弱，可保留降级观察，但不能自动改正式交易。"
    return "降级样本未明显更差，不能证明过滤有效。"


def _one_metric(effectiveness: pd.DataFrame, category: str, segment: str) -> dict[str, Any]:
    rows = effectiveness.loc[effectiveness["category"].eq(category) & effectiveness["segment"].eq(segment) & effectiveness["horizon"].eq("10d")]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_无可展示数据_"
    display = df.copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    columns = [str(column) for column in display.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in display.columns) + " |")
    return "\n".join(lines)


def _adjustment_bucket(value: object) -> str:
    text = str(value or "").strip().upper()
    if text in {"ML_RECOVERED", "ML_UPGRADED_TO_BUY_CANDIDATE"}:
        return "ML_RECOVERED"
    if text in {"ML_DOWNGRADED", "ML_FILTERED_BAD_ENTRY"}:
        return "ML_DOWNGRADED"
    if text == "ML_CONFLICT_WITH_RISK":
        return "ML_CONFLICT_WITH_RISK"
    return "ML_UNCHANGED"


def _snapshot_market_state(daily_decision_snapshot: pd.DataFrame | None) -> str:
    if daily_decision_snapshot is None or daily_decision_snapshot.empty or "market_state" not in daily_decision_snapshot.columns:
        return ""
    return str(daily_decision_snapshot.iloc[-1].get("market_state") or "")


def _effectiveness_records(effectiveness: pd.DataFrame, category: str) -> list[dict[str, Any]]:
    return effectiveness.loc[effectiveness["category"].eq(category)].replace({np.nan: None}).to_dict(orient="records")


def _read_json(path: str | Path | None) -> dict[str, Any]:
    if not path:
        return {}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _round(value: Any) -> float:
    if pd.isna(value):
        return np.nan
    return round(float(value), 6)


def _float(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)
