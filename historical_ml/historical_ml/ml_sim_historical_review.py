from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import HistoricalMLConfig
from .io_utils import ensure_dir, read_price_data, read_table
from .ml_entry_model import (
    build_entry_feature_matrix,
    prepare_entry_quality_samples,
    prepare_entry_quality_scoring_samples,
    validate_no_label_leakage_features,
    _rank_scores,
    _score_one_date,
)
from .ml_sim_review import (
    EFFECTIVENESS_COLUMNS,
    REVIEW_FILLED_COLUMNS,
    _build_label_frame,
    _fill_rows,
    _markdown_table,
    _round,
    _score_bins,
    _summary_row,
)


HISTORICAL_REVIEW_OUTPUT_COLUMNS = [
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
    "ml_rank_global",
    "ml_rank_sector",
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
    "review_status",
    "review_reason_cn",
]

HISTORICAL_DECISIONS = {
    "CONTINUE_SHADOW",
    "CONTINUE_ML_SIM",
    "TIGHTEN_ML_RECOVERED_THRESHOLD",
    "ALLOW_LIMITED_ACTIVE_SIM",
    "DISABLE_ML_ACTIVE_LAYER",
}


@dataclass(frozen=True)
class MLSimHistoricalReviewResult:
    review_filled: pd.DataFrame
    effectiveness_summary: pd.DataFrame
    report: str
    report_json: dict[str, Any]
    recommendation: str
    output_paths: dict[str, Path]


def build_ml_sim_historical_review_from_files(
    *,
    labeled_samples_path: str | Path,
    scoring_samples_path: str | Path,
    prices_path: str | Path,
    out_dir: str | Path,
    start: str = "2024-09-24",
    end: str | None = None,
    recent_days: int | None = None,
    min_train_dates: int = 20,
    config: HistoricalMLConfig = HistoricalMLConfig(),
) -> MLSimHistoricalReviewResult:
    labeled = read_table(labeled_samples_path)
    scoring = read_table(scoring_samples_path)
    prices = read_price_data(prices_path)
    return build_ml_sim_historical_review(
        labeled_samples=labeled,
        scoring_samples=scoring,
        prices=prices,
        out_dir=out_dir,
        start=start,
        end=end,
        recent_days=recent_days,
        min_train_dates=min_train_dates,
        config=config,
    )


def build_ml_sim_historical_review(
    *,
    labeled_samples: pd.DataFrame,
    scoring_samples: pd.DataFrame,
    prices: pd.DataFrame,
    out_dir: str | Path,
    start: str = "2024-09-24",
    end: str | None = None,
    recent_days: int | None = None,
    min_train_dates: int = 20,
    config: HistoricalMLConfig = HistoricalMLConfig(),
) -> MLSimHistoricalReviewResult:
    label_frame, label_dates, known_codes = _build_label_frame(prices, horizons=(1, 3, 5, 10))
    if not label_dates:
        raise ValueError("no price trading dates available for ML_SIM historical review")

    score_dates, date_pos = _review_dates(scoring_samples, label_dates, start=start, end=end, recent_days=recent_days)
    train_samples = prepare_entry_quality_samples(labeled_samples)
    score_samples = prepare_entry_quality_scoring_samples(scoring_samples)
    train_samples["_label_ready_pos"] = train_samples["trade_date"].map(date_pos) + 10

    rows: list[pd.DataFrame] = []
    skipped: list[dict[str, Any]] = []
    for score_date in score_dates:
        score_pos = date_pos[pd.Timestamp(score_date).normalize()]
        train = train_samples.loc[
            train_samples["_label_ready_pos"].le(score_pos)
            & train_samples["trade_date"].lt(score_date)
            & train_samples["auto_label"].isin(["good_entry", "bad_entry", "neutral_entry"])
        ].copy()
        train_date_count = int(train["trade_date"].dropna().nunique())
        score_df = score_samples.loc[
            score_samples["trade_date"].eq(score_date) & score_samples["is_valid_sample"].map(_truthy)
        ].copy()
        if train_date_count < min_train_dates or score_df.empty:
            skipped.append(
                {
                    "trade_date": _date_text(score_date),
                    "train_date_count": train_date_count,
                    "score_rows": int(len(score_df)),
                    "reason": "not_enough_known_training_labels" if train_date_count < min_train_dates else "no_feature_ready_rows",
                }
            )
            continue

        feature_names = list(build_entry_feature_matrix(train).columns)
        validate_no_label_leakage_features(feature_names)
        scored = _rank_scores(_score_one_date(train, score_df, feature_names))
        rows.append(_ml_sim_rows_for_date(scored))

    comparison = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=HISTORICAL_REVIEW_OUTPUT_COLUMNS)
    review_filled = _fill_rows(comparison, label_frame, label_dates, known_codes, "", config=config)
    review_filled = _order_review_columns(review_filled)
    effectiveness = _build_historical_effectiveness_summary(review_filled)
    report_json = _build_historical_report_json(
        review_filled=review_filled,
        effectiveness=effectiveness,
        requested_dates=score_dates,
        skipped_dates=skipped,
        min_train_dates=min_train_dates,
        recent_days=recent_days,
        label_dates=label_dates,
    )
    recommendation = _historical_recommendation(report_json)
    report_json["recommendation"] = recommendation
    report_json["allowed_recommendations"] = sorted(HISTORICAL_DECISIONS)
    report = _build_historical_report_markdown(report_json, effectiveness)

    out = ensure_dir(out_dir)
    paths = {
        "review_filled": out / "ml_sim_historical_review_filled.csv",
        "effectiveness_summary": out / "ml_sim_historical_effectiveness_summary.csv",
        "report_md": out / "ml_sim_historical_review_report.md",
        "report_json": out / "ml_sim_historical_review_report.json",
    }
    review_filled.to_csv(paths["review_filled"], index=False, encoding="utf-8-sig")
    effectiveness.to_csv(paths["effectiveness_summary"], index=False, encoding="utf-8-sig")
    paths["report_md"].write_text(report, encoding="utf-8")
    paths["report_json"].write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return MLSimHistoricalReviewResult(review_filled, effectiveness, report, report_json, recommendation, paths)


def _review_dates(
    scoring_samples: pd.DataFrame,
    label_dates: list[pd.Timestamp],
    *,
    start: str,
    end: str | None,
    recent_days: int | None,
) -> tuple[list[pd.Timestamp], dict[pd.Timestamp, int]]:
    date_pos = {pd.Timestamp(date).normalize(): idx for idx, date in enumerate(label_dates)}
    latest_labelable = label_dates[-11] if len(label_dates) > 10 else label_dates[-1]
    start_ts = pd.Timestamp(start).normalize()
    end_ts = min(pd.Timestamp(end).normalize() if end else latest_labelable, latest_labelable)
    sample_dates = pd.to_datetime(scoring_samples["trade_date"], errors="coerce").dt.normalize().dropna().unique()
    dates = [pd.Timestamp(date).normalize() for date in sorted(sample_dates) if start_ts <= pd.Timestamp(date).normalize() <= end_ts]
    dates = [date for date in dates if date in date_pos]
    if recent_days is not None and recent_days > 0:
        dates = dates[-int(recent_days) :]
    return dates, date_pos


def _ml_sim_rows_for_date(scored: pd.DataFrame) -> pd.DataFrame:
    rows = scored.copy()
    rows["legacy_action"] = rows.apply(_legacy_action, axis=1)
    actions = rows.apply(_ml_sim_action, axis=1, result_type="expand")
    rows["ml_sim_action"] = actions[0]
    rows["ml_adjustment_type"] = actions[1]
    rows["final_action"] = rows["legacy_action"]
    rows["review_priority"] = np.where(rows["ml_adjustment_type"].eq("ML_UNCHANGED"), "P2", "P1")
    rows["ml_adjustment_reason_cn"] = rows.apply(_ml_adjustment_reason, axis=1)
    rows["trade_date"] = pd.to_datetime(rows["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in HISTORICAL_REVIEW_OUTPUT_COLUMNS + ["review_priority", "ml_adjustment_reason_cn"]:
        if column not in rows.columns:
            rows[column] = ""
    return rows


def _ml_sim_action(row: pd.Series) -> tuple[str, str]:
    legacy = str(row.get("legacy_action") or "OBSERVE").upper()
    suggestion = str(row.get("ml_action_suggestion") or "NO_ML").upper()
    if suggestion == "UPGRADE_PROBE" and legacy in {"OBSERVE", "REJECT", "AVOID", "BLOCKED"}:
        return "PROBE", "ML_RECOVERED"
    if suggestion in {"DOWNGRADE_WATCH", "WAIT_PULLBACK"} and legacy in {"BUY", "PROBE"}:
        return "OBSERVE", "ML_DOWNGRADED"
    if suggestion == "FORBID_CHASE" and legacy in {"BUY", "PROBE"}:
        return "AVOID", "ML_DOWNGRADED"
    return legacy, "ML_UNCHANGED"


def _legacy_action(row: pd.Series) -> str:
    for column in ("final_action", "entry_raw_action", "final_buy_action"):
        text = str(row.get(column) or "").strip().upper()
        if text in {"BUY", "PROBE", "OBSERVE", "REJECT", "AVOID", "BLOCKED"}:
            return text
    if _truthy(row.get("pre_selected")):
        return "PROBE"
    return "OBSERVE"


def _ml_adjustment_reason(row: pd.Series) -> str:
    adjustment = str(row.get("ml_adjustment_type") or "ML_UNCHANGED")
    suggestion = str(row.get("ml_action_suggestion") or "NO_ML")
    if adjustment == "ML_RECOVERED":
        return f"历史 walk-forward: ML score 将 legacy={row.get('legacy_action')} 恢复为 PROBE; suggestion={suggestion}."
    if adjustment == "ML_DOWNGRADED":
        return f"历史 walk-forward: ML score 将 legacy={row.get('legacy_action')} 降级为 {row.get('ml_sim_action')}; suggestion={suggestion}."
    return "历史 walk-forward: ML_SIM 保持 legacy_v21 动作。"


def _order_review_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in HISTORICAL_REVIEW_OUTPUT_COLUMNS:
        if column not in out.columns:
            out[column] = ""
    preferred = HISTORICAL_REVIEW_OUTPUT_COLUMNS + [
        "ml_adjustment_bucket",
        "review_priority",
        "auto_label",
        "ml_action_suggestion",
        "ml_reason_code",
        "ml_reason_cn",
        "ml_adjustment_reason_cn",
    ]
    for column in preferred:
        if column not in out.columns:
            out[column] = ""
    rest = [column for column in out.columns if column not in preferred and column in REVIEW_FILLED_COLUMNS]
    extra = [column for column in out.columns if column not in preferred and column not in rest]
    return out[preferred + rest + extra]


def _build_historical_effectiveness_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for bucket in ("ML_RECOVERED", "ML_DOWNGRADED", "ML_UNCHANGED", "ML_CONFLICT_WITH_RISK"):
        sub = frame.loc[frame["ml_adjustment_bucket"].eq(bucket)]
        for horizon in (1, 3, 5, 10):
            rows.append(_summary_row(sub, "adjustment", bucket, bucket, horizon))

    recovered = frame.loc[frame["ml_adjustment_bucket"].eq("ML_RECOVERED")].sort_values(
        ["trade_date", "ml_rank_global", "ml_score"], ascending=[True, True, False]
    )
    for top_n in (20, 50, 100, 200):
        top = _daily_top(recovered, top_n)
        rows.append(_summary_row(top, "ml_recovered_topn", f"Top{top_n}", "ML_RECOVERED", 10))

    legacy = frame.loc[frame["legacy_action"].fillna("").astype(str).str.upper().isin(["BUY", "PROBE"])]
    legacy_probe = frame.loc[frame["legacy_action"].fillna("").astype(str).str.upper().eq("PROBE")]
    rows.append(_summary_row(legacy, "legacy_vs_recovered", "legacy_v21_buy_probe", "legacy_v21_buy_probe", 10))
    rows.append(_summary_row(legacy_probe, "legacy_vs_recovered", "legacy_v21_probe", "legacy_v21_probe", 10))
    for top_n in (20, 50, 100, 200):
        rows.append(_summary_row(_daily_top(recovered, top_n), "legacy_vs_recovered", f"ml_recovered_top{top_n}", "ML_RECOVERED", 10))

    for column in ("ml_score", "p_good_entry", "p_bad_entry"):
        for label, sub in _score_bins(frame, column):
            rows.append(_summary_row(sub, f"{column}_bin", label, label, 10))

    for column, category in (("market_state", "market_state"), ("sector_state", "sector_state")):
        if column in frame.columns:
            for value, sub in frame.groupby(frame[column].fillna("").astype(str), dropna=False):
                rows.append(_summary_row(sub, category, value or "UNKNOWN", value or "UNKNOWN", 10))

    return pd.DataFrame(rows, columns=EFFECTIVENESS_COLUMNS)


def _daily_top(frame: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    ranked = frame.copy()
    ranked["_daily_recovered_rank"] = ranked.groupby("trade_date")["ml_score"].rank(ascending=False, method="first")
    return ranked.loc[ranked["_daily_recovered_rank"].le(top_n)].drop(columns=["_daily_recovered_rank"])


def _build_historical_report_json(
    *,
    review_filled: pd.DataFrame,
    effectiveness: pd.DataFrame,
    requested_dates: Iterable[pd.Timestamp],
    skipped_dates: list[dict[str, Any]],
    min_train_dates: int,
    recent_days: int | None,
    label_dates: list[pd.Timestamp],
) -> dict[str, Any]:
    total = int(len(review_filled))
    ready = int(review_filled["review_status"].eq("READY").sum()) if "review_status" in review_filled.columns else 0
    recovered_count = int(review_filled["ml_adjustment_bucket"].eq("ML_RECOVERED").sum()) if "ml_adjustment_bucket" in review_filled.columns else 0
    downgraded_count = int(review_filled["ml_adjustment_bucket"].eq("ML_DOWNGRADED").sum()) if "ml_adjustment_bucket" in review_filled.columns else 0
    scored_dates = sorted(pd.to_datetime(review_filled.get("trade_date", pd.Series(dtype=str)), errors="coerce").dropna().dt.normalize().unique())
    return {
        "mode": "V2.1_ML_SIM_HISTORICAL_REVIEW",
        "requested_start": _date_text(min(requested_dates)) if requested_dates else "",
        "requested_end": _date_text(max(requested_dates)) if requested_dates else "",
        "latest_trade_date": _date_text(label_dates[-1]),
        "latest_labelable_trade_date": _date_text(label_dates[-11] if len(label_dates) > 10 else label_dates[-1]),
        "recent_days": recent_days,
        "min_train_dates": min_train_dates,
        "requested_trade_date_count": len(list(requested_dates)),
        "scored_trade_date_count": len(scored_dates),
        "first_scored_trade_date": _date_text(scored_dates[0]) if scored_dates else "",
        "last_scored_trade_date": _date_text(scored_dates[-1]) if scored_dates else "",
        "skipped_trade_date_count": len(skipped_dates),
        "skipped_trade_dates_sample": skipped_dates[:20],
        "total_rows": total,
        "ready_count": ready,
        "pending_count": int(review_filled["review_status"].eq("PENDING_NOT_ENOUGH_FUTURE_DATA").sum()) if "review_status" in review_filled.columns else 0,
        "missing_price_count": int(review_filled["review_status"].eq("MISSING_PRICE").sum()) if "review_status" in review_filled.columns else 0,
        "ml_recovered_count": recovered_count,
        "ml_downgraded_count": downgraded_count,
        "ml_recovered_ratio": _round(recovered_count / total) if total else 0.0,
        "ml_recovered_overwide": bool(total and recovered_count / total >= 0.30),
        "topn": _records(effectiveness, "ml_recovered_topn"),
        "legacy_vs_recovered": _records(effectiveness, "legacy_vs_recovered"),
        "adjustment_summary": _records(effectiveness, "adjustment"),
        "leakage_control": {
            "walk_forward": True,
            "training_label_rule": "training rows require train_trade_date + 10 trading days <= as_of_date",
            "feature_inputs": "feature_at_t only; future_return_*, future_max_*, outperform_*, auto_label, label_status, code, and name are rejected as model features",
            "future_labels_used_for": "label_after_t/post_review_metrics_only",
            "formal_entry_changed": False,
            "final_buy_action_changed": False,
            "qmt_triggered": False,
            "buy_probe_threshold_changed": False,
            "data_cache_written": False,
            "future_function_risk": False,
        },
    }


def _historical_recommendation(report_json: dict[str, Any]) -> str:
    if report_json.get("leakage_control", {}).get("future_function_risk"):
        return "DISABLE_ML_ACTIVE_LAYER"
    if int(report_json.get("ready_count") or 0) == 0:
        return "CONTINUE_ML_SIM"

    lookup = {(row["category"], row["segment"]): row for row in report_json.get("legacy_vs_recovered", [])}
    legacy = lookup.get(("legacy_vs_recovered", "legacy_v21_buy_probe"), {})
    top20 = lookup.get(("legacy_vs_recovered", "ml_recovered_top20"), {})
    top50 = lookup.get(("legacy_vs_recovered", "ml_recovered_top50"), {})
    top100 = lookup.get(("legacy_vs_recovered", "ml_recovered_top100"), {})
    top200 = lookup.get(("legacy_vs_recovered", "ml_recovered_top200"), {})
    top20_ok = _beats_legacy(top20, legacy)
    top50_ok = _beats_legacy(top50, legacy)
    broad_worse = not _beats_legacy(top100, legacy) or not _beats_legacy(top200, legacy)
    if top20_ok and top50_ok and (broad_worse or report_json.get("ml_recovered_overwide")):
        return "TIGHTEN_ML_RECOVERED_THRESHOLD"
    if top20_ok and top50_ok:
        return "ALLOW_LIMITED_ACTIVE_SIM"
    return "CONTINUE_SHADOW"


def _beats_legacy(row: dict[str, Any], legacy: dict[str, Any]) -> bool:
    return (
        int(row.get("ready_count") or 0) > 0
        and _float(row.get("avg_future_return")) > _float(legacy.get("avg_future_return"))
        and _float(row.get("good_entry_rate")) >= _float(legacy.get("good_entry_rate"))
        and _float(row.get("bad_entry_rate")) <= _float(legacy.get("bad_entry_rate"))
    )


def _build_historical_report_markdown(report_json: dict[str, Any], effectiveness: pd.DataFrame) -> str:
    lines = [
        "# V2.1_ML_SIM 历史效果回看复盘",
        "",
        "## control_center 结论",
        "",
        f"- recommendation: {report_json['recommendation']}",
        "- formal_entry_change: no",
        "- final_buy_action_change: no",
        "- qmt_triggered: no",
        "- BUY/PROBE threshold_change: no",
        f"- recovered_pool: {report_json['ml_recovered_count']} / {report_json['total_rows']} ({report_json['ml_recovered_ratio']:.2%})",
        f"- recovered_overwide: {report_json['ml_recovered_overwide']}",
        "",
        "## walk-forward 防泄漏边界",
        "",
        f"- scored_trade_date_count: {report_json['scored_trade_date_count']}",
        f"- scored_range: {report_json['first_scored_trade_date']} to {report_json['last_scored_trade_date']}",
        f"- latest_trade_date: {report_json['latest_trade_date']}",
        f"- latest_labelable_trade_date: {report_json['latest_labelable_trade_date']}",
        f"- min_train_dates: {report_json['min_train_dates']}",
        "- 每个 as_of_date 的训练样本必须满足 train_trade_date + 10 个交易日 <= as_of_date。",
        "- future_return / future_drawdown / outperform 只在打分后用于 label_after_t 复盘统计。",
        "",
        "## 样本状态",
        "",
        f"- total_rows: {report_json['total_rows']}",
        f"- READY: {report_json['ready_count']}",
        f"- PENDING_NOT_ENOUGH_FUTURE_DATA: {report_json['pending_count']}",
        f"- MISSING_PRICE: {report_json['missing_price_count']}",
        f"- skipped_trade_date_count: {report_json['skipped_trade_date_count']}",
        "",
        "## ML_RECOVERED TopN",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].eq("ml_recovered_topn")]),
        "",
        "## legacy_v21 selected / PROBE vs ML_RECOVERED TopN",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].eq("legacy_vs_recovered")]),
        "",
        "## ML_DOWNGRADED 与调整类型",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].eq("adjustment") & effectiveness["horizon"].eq("10d")]),
        "",
        "## score / probability 分位数",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].isin(["ml_score_bin", "p_good_entry_bin", "p_bad_entry_bin"])]),
        "",
        "## market_state / sector_state",
        "",
        _markdown_table(effectiveness.loc[effectiveness["category"].isin(["market_state", "sector_state"])]),
        "",
    ]
    return "\n".join(lines)


def _records(effectiveness: pd.DataFrame, category: str) -> list[dict[str, Any]]:
    return effectiveness.loc[effectiveness["category"].eq(category)].replace({np.nan: None}).to_dict(orient="records")


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y", "selected", "buy", "probe"}


def _float(value: Any) -> float:
    if value is None or pd.isna(value):
        return 0.0
    return float(value)


def _date_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d")
