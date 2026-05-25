from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io_utils import ensure_dir, read_table
from .ml_core_recovered_review import _metrics


DECISIONS = {
    "CONTINUE_SHADOW",
    "USE_ML_AS_RANKING_ONLY",
    "USE_ML_AS_FILTER_ONLY",
    "TIGHTEN_ML_RECOVERED_THRESHOLD",
    "ALLOW_LIMITED_ACTIVE_SIM",
    "DISABLE_ML_RECOVERY",
}

UNKNOWN_TEXT = "行业未录入"


@dataclass(frozen=True)
class SectorMappingAuditResult:
    sector_audit: pd.DataFrame
    coverage_report: pd.DataFrame
    after_audit: pd.DataFrame
    recommendation: str
    report_json: dict[str, Any]
    output_paths: dict[str, Path]


def build_sector_mapping_audit_from_files(
    *,
    historical_review_path: str | Path,
    out_dir: str | Path,
    sector_map_path: str | Path = Path("config") / "etf_sector_map.yaml",
    universe_path: str | Path | None = Path("output") / "etf_universe_snapshot.csv",
    price_path: str | Path | None = Path("data") / "etf_daily.csv",
) -> SectorMappingAuditResult:
    return build_sector_mapping_audit(
        historical_review=read_table(historical_review_path),
        out_dir=out_dir,
        sector_map_path=sector_map_path,
        universe_path=universe_path,
        price_path=price_path,
    )


def build_sector_mapping_audit(
    *,
    historical_review: pd.DataFrame,
    out_dir: str | Path,
    sector_map_path: str | Path = Path("config") / "etf_sector_map.yaml",
    universe_path: str | Path | None = Path("output") / "etf_universe_snapshot.csv",
    price_path: str | Path | None = Path("data") / "etf_daily.csv",
) -> SectorMappingAuditResult:
    frame = _prepare_frame(historical_review)
    total_trade_dates = int(frame["trade_date"].nunique()) if not frame.empty else 0
    sector_map = _load_sector_map(sector_map_path)
    universe = _read_optional_table(universe_path)
    price = _read_optional_table(price_path)

    strong_pre = _strong_pool(frame)
    sector_audit = _build_strong_sector_audit(strong_pre, sector_map=sector_map, universe=universe, price=price)
    coverage_report = _build_coverage_report(frame, sector_map=sector_map, universe=universe, price=price, strong_pre=strong_pre)

    legacy = _metrics(_legacy_pool(frame), "legacy_v21_buy_probe", "legacy buy/probe", total_trade_dates=total_trade_dates)
    original = _metrics(_original_recovered_pool(frame), "original_ml_recovered", "original ML_RECOVERED", legacy=legacy, total_trade_dates=total_trade_dates)
    unmapped_keys = _sector_audit_unmapped_keys(sector_audit)
    strong_keys = strong_pre.apply(lambda row: (str(row.get("trade_date")), _normalize_code(row.get("code"))), axis=1)
    unmapped_mask = strong_keys.isin(unmapped_keys) if not strong_pre.empty else pd.Series(False, index=strong_pre.index)
    unmapped_review = strong_pre.loc[unmapped_mask].copy()
    strong_after = strong_pre.loc[~unmapped_mask].copy()

    rows = [
        legacy,
        original,
        _metrics(strong_pre, "ml_strong_recovered_before_sector_audit", "ML_STRONG_RECOVERED before sector audit", legacy=legacy, total_trade_dates=total_trade_dates),
        _metrics(unmapped_review, "ml_unmapped_review", "ML_UNMAPPED_REVIEW", legacy=legacy, total_trade_dates=total_trade_dates),
        _metrics(strong_after, "ml_strong_recovered_after_sector_audit", "ML_STRONG_RECOVERED after excluding unmapped sector", legacy=legacy, total_trade_dates=total_trade_dates),
    ]
    after_audit = pd.DataFrame(rows)
    recommendation = _recommend_after_audit(after_audit)
    p1_risks = _p1_risks(sector_audit, coverage_report, strong_after)
    report_json = _build_report_json(
        frame=frame,
        sector_audit=sector_audit,
        coverage_report=coverage_report,
        after_audit=after_audit,
        recommendation=recommendation,
        p1_risks=p1_risks,
    )

    out = ensure_dir(out_dir)
    paths = {
        "sector_audit_csv": out / "ml_strong_recovered_sector_audit.csv",
        "sector_audit_md": out / "ml_strong_recovered_sector_audit.md",
        "sector_audit_json": out / "ml_strong_recovered_sector_audit.json",
        "coverage_csv": out / "sector_mapping_coverage_report.csv",
        "coverage_md": out / "sector_mapping_coverage_report.md",
        "after_audit_csv": out / "ml_recovered_threshold_after_sector_audit.csv",
        "after_audit_md": out / "ml_recovered_threshold_after_sector_audit.md",
        "after_audit_json": out / "ml_recovered_threshold_after_sector_audit.json",
    }
    sector_audit.to_csv(paths["sector_audit_csv"], index=False, encoding="utf-8-sig")
    coverage_report.to_csv(paths["coverage_csv"], index=False, encoding="utf-8-sig")
    after_audit.to_csv(paths["after_audit_csv"], index=False, encoding="utf-8-sig")
    paths["sector_audit_md"].write_text(_build_sector_audit_markdown(sector_audit, report_json), encoding="utf-8")
    paths["coverage_md"].write_text(_build_coverage_markdown(coverage_report, report_json), encoding="utf-8")
    paths["after_audit_md"].write_text(_build_after_audit_markdown(after_audit, report_json), encoding="utf-8")
    paths["sector_audit_json"].write_text(
        json.dumps(
            {
                "mode": "ML_STRONG_RECOVERED_SECTOR_AUDIT",
                "summary": report_json["sector_audit_summary"],
                "rows": _jsonable(sector_audit.replace({np.nan: None}).to_dict(orient="records")),
                "hard_gate": _hard_gate(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["after_audit_json"].write_text(json.dumps(report_json, ensure_ascii=False, indent=2), encoding="utf-8")
    return SectorMappingAuditResult(sector_audit, coverage_report, after_audit, recommendation, report_json, paths)


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame["trade_date"] = pd.to_datetime(frame.get("trade_date"), errors="coerce").dt.strftime("%Y-%m-%d")
    if "review_status" in frame.columns:
        frame = frame.loc[frame["review_status"].fillna("").astype(str).eq("READY")].copy()
    for column in [
        "ml_score",
        "p_good_entry",
        "p_bad_entry",
        "ml_rank_global",
        "ml_rank_sector",
        "momentum_score",
        "expected_drawdown_10d",
        "future_return_3d",
        "future_return_5d",
        "future_return_10d",
        "future_max_drawdown_10d",
    ]:
        if column not in frame.columns:
            frame[column] = np.nan
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["outperform_market_10d", "outperform_sector_10d", "is_valid_sample"]:
        if column not in frame.columns:
            frame[column] = False
        frame[column] = frame[column].map(_truthy)
    for column in [
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
    ]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    frame["code"] = frame["code"].map(_normalize_code)
    frame["_is_original_recovered"] = frame["ml_adjustment_bucket"].str.upper().eq("ML_RECOVERED") | frame[
        "ml_adjustment_type"
    ].str.upper().eq("ML_RECOVERED")
    frame["_is_legacy_buy_probe"] = frame["legacy_action"].str.upper().isin(["BUY", "PROBE"])
    frame["_daily_count"] = frame.groupby("trade_date")["code"].transform("count").clip(lower=1)
    frame["_p_good_rank"] = frame.groupby("trade_date")["p_good_entry"].rank(ascending=False, method="first")
    for quantile in (0.30, 0.40, 0.50):
        suffix = int(quantile * 100)
        frame[f"_p_bad_q{suffix}"] = frame.groupby("trade_date")["p_bad_entry"].transform(lambda s: s.quantile(quantile))
        frame[f"_momentum_q{suffix}"] = frame.groupby("trade_date")["momentum_score"].transform(lambda s: s.quantile(quantile))
        frame[f"_drawdown_q{suffix}"] = frame.groupby("trade_date")["expected_drawdown_10d"].transform(lambda s: s.quantile(quantile))
    return frame


def _strong_pool(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    top_n = np.ceil(frame["_daily_count"] * 0.01).clip(lower=1)
    mask = (
        frame["_is_original_recovered"]
        & frame["_p_good_rank"].le(top_n)
        & frame["p_bad_entry"].le(frame["_p_bad_q50"])
        & frame["ml_rank_global"].le(20)
        & frame["ml_rank_sector"].le(1)
        & frame["momentum_score"].ge(frame["_momentum_q40"])
        & frame["expected_drawdown_10d"].ge(frame["_drawdown_q50"])
        & ~frame["market_state"].str.lower().eq("defense")
        & ~frame["sector_state"].str.lower().eq("weak")
        & frame["is_valid_sample"]
        & ~frame["exclude_reason"].str.lower().str.contains("invalid", na=False)
    )
    return frame.loc[mask.fillna(False)].copy()


def _build_strong_sector_audit(
    strong: pd.DataFrame,
    *,
    sector_map: dict[str, dict[str, Any]],
    universe: pd.DataFrame,
    price: pd.DataFrame,
) -> pd.DataFrame:
    if strong.empty:
        return pd.DataFrame(columns=_sector_audit_columns())
    universe_by_code = _latest_by_code(universe)
    price_by_key = _price_by_key(price)
    rows: list[dict[str, Any]] = []
    for _, row in strong.sort_values(["trade_date", "code"]).iterrows():
        code = _normalize_code(row.get("code"))
        map_record = sector_map.get(code)
        universe_record = universe_by_code.get(code, {})
        price_record = price_by_key.get((str(row.get("trade_date")), code), {})
        suggestion = _suggest_mapping(code=code, name=str(row.get("name") or universe_record.get("name") or ""))
        map_missing = map_record is None
        code_format_issue = code != str(row.get("code", "")).strip()
        current_map_l1 = (map_record or {}).get("sector_l1", UNKNOWN_TEXT)
        current_map_l2 = (map_record or {}).get("sector_l2", UNKNOWN_TEXT)
        formal_map_resolves_sector = bool(map_record and not (_is_unmapped(current_map_l1) or _is_unmapped(current_map_l2)))
        historical_unmapped = _is_unmapped(row.get("sector_level1")) or _is_unmapped(row.get("sector_level2"))
        unmapped = historical_unmapped and not formal_map_resolves_sector
        rows.append(
            {
                "trade_date": row.get("trade_date", ""),
                "code": code,
                "name": row.get("name", ""),
                "historical_sector_level1": row.get("sector_level1", ""),
                "historical_sector_level2": row.get("sector_level2", ""),
                "historical_price_sector_l1": price_record.get("sector_l1", ""),
                "historical_price_sector_l2": price_record.get("sector_l2", ""),
                "current_map_sector_l1": current_map_l1,
                "current_map_sector_l2": current_map_l2,
                "current_universe_asset_class": universe_record.get("asset_class", ""),
                "current_universe_category": universe_record.get("category", ""),
                "current_universe_tracking_index": universe_record.get("tracking_index", ""),
                "is_sector_unmapped": bool(unmapped),
                "map_missing": bool(map_missing),
                "code_format_issue": bool(code_format_issue),
                "name_parse_failed": bool(suggestion["parse_failed"]),
                "unclassified_asset_reason": suggestion["asset_reason"],
                "is_new_or_unmaintained": bool(map_missing and code in universe_by_code),
                "suggested_asset_class": suggestion["asset_class"],
                "suggested_sector_l1": suggestion["sector_l1"],
                "suggested_sector_l2": suggestion["sector_l2"],
                "suggested_theme": suggestion["theme"],
                "suggested_risk_group": suggestion["risk_group"],
                "suggestion_confidence": suggestion["confidence"],
                "ml_review_bucket_after_audit": "ML_UNMAPPED_REVIEW" if unmapped else "ML_STRONG_RECOVERED",
                "audit_reason": _audit_reason(map_missing, code_format_issue, suggestion),
            }
        )
    return pd.DataFrame(rows, columns=_sector_audit_columns())


def _build_coverage_report(
    frame: pd.DataFrame,
    *,
    sector_map: dict[str, dict[str, Any]],
    universe: pd.DataFrame,
    price: pd.DataFrame,
    strong_pre: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        _coverage_row("historical_review_ready_rows", frame, code_col="code", sector_l1="sector_level1", sector_l2="sector_level2"),
        _coverage_row("historical_review_original_ml_recovered", _original_recovered_pool(frame), code_col="code", sector_l1="sector_level1", sector_l2="sector_level2"),
        _coverage_row("ml_strong_recovered_before_audit", strong_pre, code_col="code", sector_l1="sector_level1", sector_l2="sector_level2"),
        _coverage_row("data_etf_daily_rows", price, code_col="code", sector_l1="sector_l1", sector_l2="sector_l2"),
        _coverage_row("current_universe_rows", universe, code_col="symbol", sector_l1="category", sector_l2="tracking_index"),
    ]
    map_codes = pd.DataFrame({"code": sorted(sector_map)})
    rows.append(
        {
            "dataset": "formal_sector_map_codes",
            "row_count": int(len(map_codes)),
            "unique_code_count": int(map_codes["code"].nunique()) if not map_codes.empty else 0,
            "unmapped_row_count": 0,
            "unmapped_unique_code_count": 0,
            "unmapped_row_rate": 0.0,
            "unmapped_code_rate": 0.0,
            "p1_data_quality_risk": False,
            "notes": "formal config code coverage only; missing current universe codes are checked separately",
        }
    )
    universe_codes = {_normalize_code(code) for code in universe.get("symbol", pd.Series(dtype=str)).dropna().astype(str)} if not universe.empty else set()
    missing_universe_codes = sorted(code for code in universe_codes if code and code not in sector_map)
    rows.append(
        {
            "dataset": "current_universe_missing_from_formal_sector_map",
            "row_count": int(len(universe_codes)),
            "unique_code_count": int(len(universe_codes)),
            "unmapped_row_count": int(len(missing_universe_codes)),
            "unmapped_unique_code_count": int(len(missing_universe_codes)),
            "unmapped_row_rate": _round(len(missing_universe_codes) / len(universe_codes)) if universe_codes else 0.0,
            "unmapped_code_rate": _round(len(missing_universe_codes) / len(universe_codes)) if universe_codes else 0.0,
            "p1_data_quality_risk": bool(universe_codes and len(missing_universe_codes) / len(universe_codes) > 0.05),
            "notes": ",".join(missing_universe_codes[:20]),
        }
    )
    return pd.DataFrame(rows)


def _coverage_row(dataset: str, df: pd.DataFrame, *, code_col: str, sector_l1: str, sector_l2: str) -> dict[str, Any]:
    if df.empty or code_col not in df.columns:
        return {
            "dataset": dataset,
            "row_count": 0,
            "unique_code_count": 0,
            "unmapped_row_count": 0,
            "unmapped_unique_code_count": 0,
            "unmapped_row_rate": 0.0,
            "unmapped_code_rate": 0.0,
            "p1_data_quality_risk": False,
            "notes": "no data",
        }
    l1 = df[sector_l1] if sector_l1 in df.columns else pd.Series("", index=df.index)
    l2 = df[sector_l2] if sector_l2 in df.columns else pd.Series("", index=df.index)
    unmapped_mask = _is_unmapped_series(l1) | _is_unmapped_series(l2)
    codes = df[code_col].map(_normalize_code)
    unique_count = int(codes.nunique())
    unmapped_codes = codes.loc[unmapped_mask].dropna().unique()
    row_rate = _round(float(unmapped_mask.mean())) if len(df) else 0.0
    code_rate = _round(len(unmapped_codes) / unique_count) if unique_count else 0.0
    return {
        "dataset": dataset,
        "row_count": int(len(df)),
        "unique_code_count": unique_count,
        "unmapped_row_count": int(unmapped_mask.sum()),
        "unmapped_unique_code_count": int(len(unmapped_codes)),
        "unmapped_row_rate": row_rate,
        "unmapped_code_rate": code_rate,
        "p1_data_quality_risk": bool(row_rate > 0.05 or code_rate > 0.05),
        "notes": "unmapped rate exceeds 5%" if row_rate > 0.05 or code_rate > 0.05 else "",
    }


def _suggest_mapping(*, code: str, name: str) -> dict[str, Any]:
    text = str(name or "")
    if not text.strip():
        return _suggestion("其他", "其他", "待人工分类", "待人工分类", "待人工分类", "name_missing", 0.0, parse_failed=True)
    if any(token in text for token in ["天天金", "日利", "添益", "货币", "现金"]):
        return _suggestion("货币", "防御资产", "债券现金", _theme_from_name(text), "防御资产", "money_or_cash_etf", 0.8)
    if any(token in text for token in ["中证1000", "A股", "A500", "沪深300", "中证500"]):
        return _suggestion("权益", "宽基指数", "中小盘宽基" if "1000" in text else "大盘宽基", _theme_from_name(text), "全市场", "broad_or_cross_market_equity", 0.8)
    if any(token in text for token in ["科创", "新材料", "芯片", "半导体", "人工智能"]):
        return _suggestion("权益", "行业主题", "科技成长", _theme_from_name(text), "科技成长", "thematic_growth_etf", 0.75)
    if "ETF" in text:
        return _suggestion("权益", "行业主题", "待人工分类", _theme_from_name(text), "待人工分类", "new_or_unclassified_etf", 0.4)
    return _suggestion("其他", "其他", "待人工分类", _theme_from_name(text), "待人工分类", "name_parse_uncertain", 0.2, parse_failed=True)


def _suggestion(
    asset_class: str,
    sector_l1: str,
    sector_l2: str,
    theme: str,
    risk_group: str,
    asset_reason: str,
    confidence: float,
    *,
    parse_failed: bool = False,
) -> dict[str, Any]:
    return {
        "asset_class": asset_class,
        "sector_l1": sector_l1,
        "sector_l2": sector_l2,
        "theme": theme,
        "risk_group": risk_group,
        "asset_reason": asset_reason,
        "confidence": confidence,
        "parse_failed": parse_failed,
    }


def _theme_from_name(name: str) -> str:
    text = str(name).replace("ETF", "")
    for brand in ["华泰柏瑞", "华泰", "南方", "华宝", "易方达", "华夏", "国泰", "广发", "富国", "博时"]:
        text = text.replace(brand, "")
    return text.strip() or str(name)


def _recommend_after_audit(after_audit: pd.DataFrame) -> str:
    strong_after = after_audit.loc[after_audit["candidate_id"].eq("ml_strong_recovered_after_sector_audit")]
    if strong_after.empty or int(strong_after.iloc[0]["sample_count"]) == 0:
        return "CONTINUE_SHADOW"
    legacy = after_audit.loc[after_audit["candidate_id"].eq("legacy_v21_buy_probe")].iloc[0]
    row = strong_after.iloc[0]
    if row["good_entry_rate"] <= legacy["good_entry_rate"] or row["bad_entry_rate"] >= legacy["bad_entry_rate"]:
        return "CONTINUE_SHADOW"
    if row["daily_average_count"] < 0.5 or row["sample_count"] < 100:
        return "CONTINUE_SHADOW"
    return "ALLOW_LIMITED_ACTIVE_SIM"


def _p1_risks(sector_audit: pd.DataFrame, coverage_report: pd.DataFrame, strong_after: pd.DataFrame) -> list[str]:
    risks: list[str] = []
    if not sector_audit.empty and bool(sector_audit["is_sector_unmapped"].all()):
        risks.append("ML_STRONG_RECOVERED 100% 行业未录入，存在行业映射缺失或样本池偏置风险。")
    if strong_after.empty:
        risks.append("排除行业未录入后 ML_STRONG_RECOVERED 消失，不能证明强恢复条件稳定。")
    high = coverage_report.loc[coverage_report["p1_data_quality_risk"].astype(bool)] if not coverage_report.empty else pd.DataFrame()
    for _, row in high.iterrows():
        risks.append(f"{row['dataset']} 行业映射缺失率过高：row_rate={row['unmapped_row_rate']}, code_rate={row['unmapped_code_rate']}")
    return risks


def _build_report_json(
    *,
    frame: pd.DataFrame,
    sector_audit: pd.DataFrame,
    coverage_report: pd.DataFrame,
    after_audit: pd.DataFrame,
    recommendation: str,
    p1_risks: list[str],
) -> dict[str, Any]:
    strong_after = after_audit.loc[after_audit["candidate_id"].eq("ml_strong_recovered_after_sector_audit")]
    return {
        "mode": "ML_RECOVERED_THRESHOLD_AFTER_SECTOR_AUDIT",
        "recommendation": recommendation,
        "allowed_recommendations": sorted(DECISIONS),
        "source_rows": int(len(frame)),
        "sector_audit_summary": {
            "strong_before_count": int(len(sector_audit)),
            "strong_unmapped_count": int(sector_audit["is_sector_unmapped"].sum()) if not sector_audit.empty else 0,
            "unique_strong_codes": int(sector_audit["code"].nunique()) if not sector_audit.empty else 0,
            "map_missing_codes": sorted(sector_audit.loc[sector_audit["map_missing"], "code"].unique().tolist()) if not sector_audit.empty else [],
            "strong_after_excluding_unmapped_count": int(strong_after.iloc[0]["sample_count"]) if not strong_after.empty else 0,
            "depends_on_unmapped_sector_bias": bool(not sector_audit.empty and sector_audit["is_sector_unmapped"].all()),
        },
        "p0_blocker": [],
        "p1_high_risk": p1_risks,
        "comparison": _jsonable(after_audit.replace({np.nan: None}).to_dict(orient="records")),
        "coverage_report": _jsonable(coverage_report.replace({np.nan: None}).to_dict(orient="records")),
        "hard_gate": _hard_gate(),
    }


def _build_sector_audit_markdown(sector_audit: pd.DataFrame, report_json: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ML_STRONG_RECOVERED Sector Audit",
            "",
            "## Summary",
            "",
            _dict_lines(report_json["sector_audit_summary"]),
            "",
            "## P1 High Risk",
            "",
            _list_lines(report_json["p1_high_risk"]),
            "",
            "## Audit Rows",
            "",
            _markdown_table(sector_audit),
            "",
            "## Boundary",
            "",
            "- Offline pre_selection sector mapping audit only.",
            "- No formal sector map file, entry rule, final_buy_action, QMT, market data refresh, or data/cache change.",
            "",
        ]
    )


def _build_coverage_markdown(coverage: pd.DataFrame, report_json: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Sector Mapping Coverage Report",
            "",
            "## P1 High Risk",
            "",
            _list_lines(report_json["p1_high_risk"]),
            "",
            "## Coverage",
            "",
            _markdown_table(coverage),
            "",
        ]
    )


def _build_after_audit_markdown(after_audit: pd.DataFrame, report_json: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# ML_RECOVERED Threshold After Sector Audit",
            "",
            "## control_center Conclusion",
            "",
            f"- recommendation: {report_json['recommendation']}",
            "- formal_entry_change: no",
            "- final_buy_action_change: no",
            "- qmt_triggered: no",
            "- market_data_refreshed: no",
            "- data_cache_written: no",
            "",
            "## Sector Audit Summary",
            "",
            _dict_lines(report_json["sector_audit_summary"]),
            "",
            "## Required Comparison",
            "",
            _markdown_table(after_audit),
            "",
            "## Boundary",
            "",
            "- 行业未录入样本不进入 ML_STRONG_RECOVERED after sector audit.",
            "- 行业未录入样本仅进入 ML_UNMAPPED_REVIEW.",
            "- If strong recovered disappears after excluding unmapped sector, conclusion remains CONTINUE_SHADOW.",
            "",
        ]
    )


def _load_sector_map(path: str | Path) -> dict[str, dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, dict[str, Any]] = {}
    for item in raw.get("etfs", []) or []:
        if isinstance(item, dict):
            code = _normalize_code(item.get("code") or item.get("symbol"))
            if code:
                out[code] = dict(item)
    return out


def _read_optional_table(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    if str(path).strip() == "":
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return read_table(p)


def _latest_by_code(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty:
        return {}
    frame = df.copy()
    code_col = "symbol" if "symbol" in frame.columns else "code"
    if code_col not in frame.columns:
        return {}
    frame["_code"] = frame[code_col].map(_normalize_code)
    if "latest_date" in frame.columns:
        frame = frame.sort_values("latest_date")
    return {str(row["_code"]): row.drop(labels=["_code"], errors="ignore").to_dict() for _, row in frame.dropna(subset=["_code"]).iterrows()}


def _price_by_key(df: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    if df.empty or "date" not in df.columns or "code" not in df.columns:
        return {}
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    frame["code"] = frame["code"].map(_normalize_code)
    return {(str(row["date"]), str(row["code"])): row.to_dict() for _, row in frame.iterrows()}


def _legacy_pool(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["_is_legacy_buy_probe"]].copy()


def _original_recovered_pool(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["_is_original_recovered"]].copy()


def _is_unmapped_series(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).map(_is_unmapped)


def _sector_audit_unmapped_keys(sector_audit: pd.DataFrame) -> set[tuple[str, str]]:
    if sector_audit.empty:
        return set()
    flags = sector_audit.loc[sector_audit["is_sector_unmapped"].astype(bool)].copy()
    if flags.empty:
        return set()
    return set(zip(flags["trade_date"].fillna("").astype(str), flags["code"].map(_normalize_code)))


def _is_unmapped(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text or text in {"nan", "none", "unknown"}:
        return True
    return "未录入" in text or "鏈綍鍏" in text


def _audit_reason(map_missing: bool, code_format_issue: bool, suggestion: dict[str, Any]) -> str:
    reasons: list[str] = []
    if map_missing:
        reasons.append("formal_sector_map_missing")
    if code_format_issue:
        reasons.append("code_format_normalized")
    if suggestion.get("parse_failed"):
        reasons.append("name_parse_failed")
    reasons.append(str(suggestion.get("asset_reason") or "classification_suggestion"))
    return "|".join(reasons)


def _sector_audit_columns() -> list[str]:
    return [
        "trade_date",
        "code",
        "name",
        "historical_sector_level1",
        "historical_sector_level2",
        "historical_price_sector_l1",
        "historical_price_sector_l2",
        "current_map_sector_l1",
        "current_map_sector_l2",
        "current_universe_asset_class",
        "current_universe_category",
        "current_universe_tracking_index",
        "is_sector_unmapped",
        "map_missing",
        "code_format_issue",
        "name_parse_failed",
        "unclassified_asset_reason",
        "is_new_or_unmaintained",
        "suggested_asset_class",
        "suggested_sector_l1",
        "suggested_sector_l2",
        "suggested_theme",
        "suggested_risk_group",
        "suggestion_confidence",
        "ml_review_bucket_after_audit",
        "audit_reason",
    ]


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.startswith(("SH", "SZ")):
        text = text[2:]
    if "." in text:
        text = text.split(".", 1)[0]
    match = re.search(r"\d{6}", text)
    return match.group(0) if match else text


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "1.0", "true", "yes", "y", "selected", "是"}


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
        "formal_sector_map_changed": False,
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


def _list_lines(values: list[str]) -> str:
    if not values:
        return "- 无"
    return "\n".join(f"- {value}" for value in values)


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
