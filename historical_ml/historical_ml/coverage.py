from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .io_utils import ensure_dir, read_table


def normalize_etf_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if not text:
        return ""
    if text.startswith(("SH", "SZ")):
        text = text[2:]
    if "." in text:
        text = text.split(".", 1)[0]
    match = re.search(r"\d{6}", text)
    if match:
        return match.group(0)
    digits = re.sub(r"\D", "", text)
    return digits.zfill(6)[-6:] if digits else ""


def generate_ml_universe_coverage_report(
    *,
    prices_path: str | Path,
    pre_selection_path: str | Path,
    ml_scores_path: str | Path,
    daily_universe_path: str | Path,
    entry_signal_path: str | Path | None = None,
    order_intent_path: str | Path | None = None,
    out_dir: str | Path,
    trade_date: str | None = None,
) -> dict[str, Any]:
    prices = _read_optional(prices_path)
    pre = _read_optional(pre_selection_path)
    scores = _read_optional(ml_scores_path)
    universe = _read_optional(daily_universe_path)
    entry = _read_optional(entry_signal_path) if entry_signal_path else pd.DataFrame()
    orders = _read_optional(order_intent_path) if order_intent_path else pd.DataFrame()

    target_date = _resolve_trade_date(trade_date, pre, entry, universe, scores)
    all_codes = _code_set(pre, "symbol", "etf_code", "code")
    price_codes = _code_set(prices, "code", "symbol", "etf_code")
    score_today = _filter_date(scores, target_date)
    universe_today = _filter_date(universe, target_date)
    entry_today = _filter_date(entry, target_date)

    historical_covered = all_codes & price_codes
    feature_ready_codes = _feature_ready_codes(universe_today)
    scored_codes = _code_set(score_today, "code", "symbol", "etf_code")
    entry_codes = _code_set(entry_today, "symbol", "etf_code", "code") or all_codes
    direct_hit_codes = entry_codes & scored_codes

    reason_rows = _missing_reason_rows(
        all_codes=entry_codes,
        historical_covered=historical_covered,
        feature_ready=feature_ready_codes,
        scored=scored_codes,
        universe_today=universe_today,
        target_date=target_date,
    )
    missing_reasons = [row["missing_reason"] for row in reason_rows if row["missing_reason"] != "scored"]
    reason_counts = {str(k): int(v) for k, v in pd.Series(missing_reasons, dtype=str).value_counts().sort_index().items()}
    if not reason_counts:
        reason_counts = {"none": 0}

    report = {
        "trade_date": target_date,
        "all_market_valid_etf_count": len(all_codes),
        "historical_price_covered_etf_count": len(historical_covered),
        "historical_price_total_etf_count": len(price_codes),
        "ml_feature_ready_etf_count": len(feature_ready_codes),
        "ml_scored_etf_count": len(scored_codes),
        "ml_score_direct_hit_count": len(direct_hit_codes),
        "ml_score_missing_count": max(len(entry_codes) - len(direct_hit_codes), 0),
        "ml_score_missing_reason_distribution": reason_counts,
        "broad_recall_pool_count": _truthy_count(pre, "broad_recall_selected"),
        "ml_recovered_pool_count": _truthy_count(pre, "ml_recovered"),
        "entry_candidate_pool_count": len(entry_codes),
        "order_intent_count": len(orders),
        "price_date_min": _date_min(prices, "date"),
        "price_date_max": _date_max(prices, "date"),
        "daily_universe_date_max": _date_max(universe, "trade_date"),
        "ml_score_date_max": _date_max(scores, "trade_date"),
        "code_format_mismatch_count": _code_format_mismatch_count(pre, prices),
        "missing_historical_price_codes": sorted(entry_codes - historical_covered),
        "feature_missing_codes": sorted(entry_codes - feature_ready_codes),
        "unscored_feature_ready_codes": sorted(feature_ready_codes - scored_codes),
    }

    out_path = ensure_dir(out_dir)
    (out_path / "historical_ml_universe_coverage_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    _write_markdown_report(report, out_path / "historical_ml_universe_coverage_report.md")
    pd.DataFrame(reason_rows).to_csv(out_path / "ml_score_coverage_report.csv", index=False, encoding="utf-8-sig")
    return report


def _missing_reason_rows(
    *,
    all_codes: set[str],
    historical_covered: set[str],
    feature_ready: set[str],
    scored: set[str],
    universe_today: pd.DataFrame,
    target_date: str,
) -> list[dict[str, Any]]:
    reasons_by_code: dict[str, str] = {}
    if not universe_today.empty and "code" in universe_today.columns:
        tmp = universe_today.copy()
        tmp["_code"] = tmp["code"].map(normalize_etf_code)
        for _, row in tmp.iterrows():
            code = str(row["_code"])
            if not code:
                continue
            if not _truthy(row.get("is_valid_sample", True)):
                reasons_by_code[code] = str(row.get("exclude_reason") or "feature_invalid_sample")

    rows: list[dict[str, Any]] = []
    for code in sorted(all_codes):
        if code in scored:
            reason = "scored"
        elif code not in historical_covered:
            reason = "missing_historical_price"
        elif code not in feature_ready:
            reason = reasons_by_code.get(code, "feature_missing_or_not_generated_for_trade_date")
        else:
            reason = "feature_ready_but_not_scored"
        rows.append(
            {
                "trade_date": target_date,
                "code": code,
                "has_historical_price": code in historical_covered,
                "feature_ready": code in feature_ready,
                "ml_scored": code in scored,
                "direct_hit": code in scored,
                "missing_reason": reason,
            }
        )
    return rows


def _write_markdown_report(report: Mapping[str, Any], path: Path) -> None:
    distribution = report.get("ml_score_missing_reason_distribution") or {}
    lines = [
        "# historical_ml_universe_coverage_report",
        "",
        f"- trade_date: {report.get('trade_date')}",
        f"- all_market_valid_etf_count: {report.get('all_market_valid_etf_count')}",
        f"- historical_price_covered_etf_count: {report.get('historical_price_covered_etf_count')}",
        f"- historical_price_total_etf_count: {report.get('historical_price_total_etf_count')}",
        f"- ml_feature_ready_etf_count: {report.get('ml_feature_ready_etf_count')}",
        f"- ml_scored_etf_count: {report.get('ml_scored_etf_count')}",
        f"- ml_score_direct_hit_count: {report.get('ml_score_direct_hit_count')}",
        f"- ml_score_missing_count: {report.get('ml_score_missing_count')}",
        "",
        "## missing_reason_distribution",
        "",
    ]
    lines.extend(f"- {key}: {value}" for key, value in distribution.items())
    lines.extend(
        [
            "",
            "## diagnostics",
            "",
            f"- price_date_min: {report.get('price_date_min')}",
            f"- price_date_max: {report.get('price_date_max')}",
            f"- daily_universe_date_max: {report.get('daily_universe_date_max')}",
            f"- ml_score_date_max: {report.get('ml_score_date_max')}",
            f"- code_format_mismatch_count: {report.get('code_format_mismatch_count')}",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _read_optional(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return read_table(p)


def _resolve_trade_date(explicit: str | None, *frames: pd.DataFrame) -> str:
    if explicit:
        return str(pd.Timestamp(explicit).date())
    for frame in frames:
        date_col = "trade_date" if "trade_date" in frame.columns else ("date" if "date" in frame.columns else "")
        if date_col:
            dates = pd.to_datetime(frame[date_col], errors="coerce").dropna()
            if not dates.empty:
                return str(dates.max().date())
    return str(pd.Timestamp.today().date())


def _filter_date(frame: pd.DataFrame, trade_date: str) -> pd.DataFrame:
    if frame.empty:
        return frame
    date_col = "trade_date" if "trade_date" in frame.columns else ("date" if "date" in frame.columns else "")
    if not date_col:
        return frame
    dates = pd.to_datetime(frame[date_col], errors="coerce").dt.date.astype(str)
    return frame.loc[dates == trade_date].copy()


def _code_set(frame: pd.DataFrame, *columns: str) -> set[str]:
    if frame.empty:
        return set()
    for column in columns:
        if column in frame.columns:
            return {code for code in frame[column].map(normalize_etf_code) if code}
    return set()


def _feature_ready_codes(frame: pd.DataFrame) -> set[str]:
    if frame.empty:
        return set()
    ready = frame.copy()
    if "is_valid_sample" in ready.columns:
        ready = ready.loc[ready["is_valid_sample"].map(_truthy)].copy()
    required = [
        "momentum_20",
        "momentum_60",
        "momentum_120",
        "momentum_score",
        "acceleration_score",
        "volatility_20",
        "drawdown_20",
        "drawdown_60",
        "sector_rank",
        "etf_rank",
    ]
    for column in required:
        if column not in ready.columns:
            return set()
        ready[column] = pd.to_numeric(ready[column], errors="coerce")
    ready = ready.dropna(subset=required, how="all")
    return _code_set(ready, "code", "symbol", "etf_code")


def _truthy_count(frame: pd.DataFrame, column: str) -> int:
    if frame.empty or column not in frame.columns:
        return 0
    return int(frame[column].map(_truthy).sum())


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "selected", "是"}


def _date_min(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    return "" if dates.empty else str(dates.min().date())


def _date_max(frame: pd.DataFrame, column: str) -> str:
    if frame.empty or column not in frame.columns:
        return ""
    dates = pd.to_datetime(frame[column], errors="coerce").dropna()
    return "" if dates.empty else str(dates.max().date())


def _code_format_mismatch_count(pre: pd.DataFrame, prices: pd.DataFrame) -> int:
    if pre.empty or prices.empty:
        return 0
    pre_raw = _raw_codes(pre, "symbol", "etf_code", "code")
    price_raw = _raw_codes(prices, "code", "symbol", "etf_code")
    pre_norm = {normalize_etf_code(value) for value in pre_raw}
    price_norm = {normalize_etf_code(value) for value in price_raw}
    raw_overlap = pre_raw & price_raw
    normalized_overlap = pre_norm & price_norm
    return max(len(normalized_overlap) - len({normalize_etf_code(value) for value in raw_overlap}), 0)


def _raw_codes(frame: pd.DataFrame, *columns: str) -> set[str]:
    for column in columns:
        if column in frame.columns:
            return {str(value or "").strip() for value in frame[column] if str(value or "").strip()}
    return set()
