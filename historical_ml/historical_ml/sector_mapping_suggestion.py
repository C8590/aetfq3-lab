from __future__ import annotations

import json
import math
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .io_utils import ensure_dir, read_table


SUGGESTION_COLUMNS = [
    "code",
    "name",
    "asset_class",
    "level1",
    "level2",
    "level3",
    "mapping_source",
    "mapping_confidence",
    "mapping_reason_cn",
    "needs_manual_review",
    "priority_score",
    "priority_reason_cn",
]

TOP100_REVIEW_COLUMNS = [
    "code",
    "name",
    "suggested_asset_class",
    "suggested_level1",
    "suggested_level2",
    "suggested_level3",
    "mapping_source",
    "mapping_confidence",
    "mapping_reason_cn",
    "priority_score",
    "priority_reason_cn",
    "appears_in_ml_strong_recovered",
    "appears_in_ml_core_recovered",
    "appears_in_entry_candidate_pool",
    "historical_sample_count",
    "amount_rank_or_liquidity_hint",
    "manual_confirm_status",
    "manual_note",
]

HIGH_CONFIDENCE_THRESHOLD = 0.80
MANUAL_REVIEW_THRESHOLD = 0.80
UNKNOWN_LEVEL = "待人工分类"


@dataclass(frozen=True)
class SectorMappingSuggestionResult:
    suggestion: pd.DataFrame
    manual_review_queue: pd.DataFrame
    priority_review: pd.DataFrame
    coverage: dict[str, Any]
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class SectorMappingTop100ReviewResult:
    review: pd.DataFrame
    patch_draft: dict[str, Any]
    summary: dict[str, Any]
    output_paths: dict[str, Path]


@dataclass(frozen=True)
class SectorMappingApplyTop100Result:
    accepted: pd.DataFrame
    rejected: pd.DataFrame
    summary: dict[str, Any]
    coverage: dict[str, Any]
    output_paths: dict[str, Path]


def build_sector_mapping_suggestions_from_files(
    *,
    sector_map_path: str | Path,
    universe_path: str | Path,
    out_dir: str | Path,
    price_path: str | Path | None = None,
    strong_audit_path: str | Path | None = None,
    historical_review_path: str | Path | None = None,
    current_position_path: str | Path | None = None,
    entry_signal_path: str | Path | None = None,
) -> SectorMappingSuggestionResult:
    return build_sector_mapping_suggestions(
        sector_map=_load_sector_map(sector_map_path),
        universe=read_table(universe_path),
        out_dir=out_dir,
        price=_read_optional_table(price_path),
        strong_audit=_read_optional_table(strong_audit_path),
        historical_review=_read_optional_table(historical_review_path),
        current_position=_read_current_position(current_position_path),
        entry_signal=_read_optional_table(entry_signal_path),
    )


def build_sector_mapping_top100_review_from_files(
    *,
    priority_review_path: str | Path,
    sector_map_path: str | Path,
    out_dir: str | Path,
    universe_path: str | Path | None = None,
    price_path: str | Path | None = None,
    strong_audit_path: str | Path | None = None,
    historical_review_path: str | Path | None = None,
    entry_signal_path: str | Path | None = None,
    top_n: int = 100,
) -> SectorMappingTop100ReviewResult:
    return build_sector_mapping_top100_review(
        priority_review=read_table(priority_review_path),
        sector_map=_load_sector_map(sector_map_path),
        out_dir=out_dir,
        universe=_read_optional_table(universe_path),
        price=_read_optional_table(price_path),
        strong_audit=_read_optional_table(strong_audit_path),
        historical_review=_read_optional_table(historical_review_path),
        entry_signal=_read_optional_table(entry_signal_path),
        top_n=top_n,
    )


def apply_sector_mapping_top100_from_files(
    *,
    sector_map_path: str | Path,
    review_path: str | Path,
    patch_draft_path: str | Path,
    out_dir: str | Path,
    universe_path: str | Path | None = None,
    historical_review_path: str | Path | None = None,
    price_path: str | Path | None = None,
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> SectorMappingApplyTop100Result:
    sector_path = Path(sector_map_path)
    sector_data = yaml.safe_load(sector_path.read_text(encoding="utf-8")) or {}
    patch_draft = yaml.safe_load(Path(patch_draft_path).read_text(encoding="utf-8")) or {}
    result = apply_sector_mapping_top100(
        sector_data=sector_data,
        review=read_table(review_path),
        patch_draft=patch_draft,
        out_dir=out_dir,
        universe=_read_optional_table(universe_path),
        historical_review=_read_optional_table(historical_review_path),
        price=_read_optional_table(price_path),
        high_confidence_threshold=high_confidence_threshold,
    )
    if not result.accepted.empty:
        _append_sector_map_entries(sector_path, result.accepted.attrs["patch_items"])
    return result


def apply_sector_mapping_top100(
    *,
    sector_data: dict[str, Any],
    review: pd.DataFrame,
    patch_draft: dict[str, Any],
    out_dir: str | Path,
    universe: pd.DataFrame | None = None,
    historical_review: pd.DataFrame | None = None,
    price: pd.DataFrame | None = None,
    high_confidence_threshold: float = HIGH_CONFIDENCE_THRESHOLD,
) -> SectorMappingApplyTop100Result:
    existing_codes = {
        _normalize_code(item.get("code") or item.get("symbol"))
        for item in sector_data.get("etfs", []) or []
        if isinstance(item, dict)
    }
    existing_codes = {code for code in existing_codes if code}
    patch_by_code = {
        _normalize_code(item.get("code")): dict(item)
        for item in patch_draft.get("etfs", []) or []
        if isinstance(item, dict) and _normalize_code(item.get("code"))
    }
    prepared = _prepare_top100_apply_review(review)
    accepted_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    accepted_patch_items: list[dict[str, Any]] = []
    for _, row in prepared.iterrows():
        code = str(row["code"])
        reasons = _top100_reject_reasons(row, existing_codes, patch_by_code, high_confidence_threshold)
        if reasons:
            rejected_rows.append({**row.to_dict(), "reject_reason": "|".join(reasons)})
            continue
        patch_item = _validated_patch_item(patch_by_code[code], row)
        accepted_rows.append(row.to_dict())
        accepted_patch_items.append(patch_item)

    accepted = pd.DataFrame(accepted_rows, columns=prepared.columns)
    rejected = pd.DataFrame(rejected_rows)
    accepted.attrs["patch_items"] = accepted_patch_items
    coverage = _coverage_after_apply(universe, existing_codes, accepted_patch_items)
    summary = _apply_summary(
        prepared=prepared,
        accepted=accepted,
        rejected=rejected,
        coverage=coverage,
        accepted_patch_items=accepted_patch_items,
    )

    out = ensure_dir(out_dir)
    paths = {
        "summary_md": out / "sector_mapping_apply_top100_summary.md",
        "summary_json": out / "sector_mapping_apply_top100_summary.json",
        "coverage_md": out / "sector_mapping_coverage_after_apply.md",
        "coverage_json": out / "sector_mapping_coverage_after_apply.json",
        "threshold_md": out / "ml_recovered_threshold_after_sector_apply.md",
        "threshold_json": out / "ml_recovered_threshold_after_sector_apply.json",
    }
    paths["summary_md"].write_text(_apply_summary_markdown(summary, accepted, rejected), encoding="utf-8")
    paths["summary_json"].write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["coverage_md"].write_text(_coverage_after_apply_markdown(coverage), encoding="utf-8")
    paths["coverage_json"].write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    threshold_report = _sector_apply_threshold_report(
        sector_data=sector_data,
        accepted_patch_items=accepted_patch_items,
        historical_review=historical_review,
        price=price,
        universe=universe,
        out_dir=out,
    )
    paths["threshold_md"].write_text(_threshold_after_apply_markdown(threshold_report), encoding="utf-8")
    paths["threshold_json"].write_text(json.dumps(threshold_report, ensure_ascii=False, indent=2), encoding="utf-8")
    return SectorMappingApplyTop100Result(accepted, rejected, summary, coverage, paths)


def build_sector_mapping_top100_review(
    *,
    priority_review: pd.DataFrame,
    sector_map: dict[str, dict[str, Any]],
    out_dir: str | Path,
    universe: pd.DataFrame | None = None,
    price: pd.DataFrame | None = None,
    strong_audit: pd.DataFrame | None = None,
    historical_review: pd.DataFrame | None = None,
    entry_signal: pd.DataFrame | None = None,
    top_n: int = 100,
) -> SectorMappingTop100ReviewResult:
    priority = _prepare_priority_review(priority_review)
    formal_codes = set(sector_map)
    strong_codes = _code_set(strong_audit, "code")
    core_codes = _extract_core_recovered_codes(historical_review)
    entry_codes = _entry_candidate_codes(entry_signal)
    historical_counts = _code_counts(historical_review)
    liquidity_hints = _liquidity_hints(universe, price)

    reason_text = priority["priority_reason_cn"].fillna("").astype(str)
    priority["appears_in_ml_strong_recovered"] = priority["code"].isin(strong_codes) | reason_text.str.contains("ML_STRONG_RECOVERED", na=False)
    priority["appears_in_ml_core_recovered"] = priority["code"].isin(core_codes) | reason_text.str.contains("ML_CORE_RECOVERED", na=False)
    priority["appears_in_entry_candidate_pool"] = priority["code"].isin(entry_codes) | reason_text.str.contains("entry_candidate_pool", na=False)
    priority["_sort_bucket"] = priority.apply(_top100_sort_bucket, axis=1)
    priority = priority.loc[~priority["code"].isin(formal_codes)].copy()
    priority = priority.sort_values(["_sort_bucket", "priority_score", "mapping_confidence", "code"], ascending=[True, False, False, True])
    selected = priority.head(top_n).copy()

    review = pd.DataFrame(
        {
            "code": selected["code"],
            "name": selected["name"],
            "suggested_asset_class": selected["asset_class"],
            "suggested_level1": selected["level1"],
            "suggested_level2": selected["level2"],
            "suggested_level3": selected["level3"],
            "mapping_source": selected["mapping_source"],
            "mapping_confidence": selected["mapping_confidence"],
            "mapping_reason_cn": selected["mapping_reason_cn"],
            "priority_score": selected["priority_score"],
            "priority_reason_cn": selected["priority_reason_cn"],
            "appears_in_ml_strong_recovered": selected["appears_in_ml_strong_recovered"],
            "appears_in_ml_core_recovered": selected["appears_in_ml_core_recovered"],
            "appears_in_entry_candidate_pool": selected["appears_in_entry_candidate_pool"],
            "historical_sample_count": selected["code"].map(historical_counts).fillna(0).astype(int),
            "amount_rank_or_liquidity_hint": selected["code"].map(liquidity_hints).fillna("no_liquidity_hint"),
            "manual_confirm_status": "PENDING",
            "manual_note": "",
        },
        columns=TOP100_REVIEW_COLUMNS,
    )
    patch = _patch_draft_from_review(review)
    summary = _top100_summary(review, patch, formal_codes)

    out = ensure_dir(out_dir)
    paths = {
        "review_csv": out / "sector_mapping_top100_review.csv",
        "review_md": out / "sector_mapping_top100_review.md",
        "review_json": out / "sector_mapping_top100_review.json",
        "patch_yaml": out / "sector_mapping_top100_patch_draft.yaml",
        "summary_md": out / "sector_mapping_top100_patch_summary.md",
    }
    review.to_csv(paths["review_csv"], index=False, encoding="utf-8-sig")
    paths["review_md"].write_text(_top100_review_markdown(review, summary), encoding="utf-8")
    paths["review_json"].write_text(
        json.dumps(
            {
                "mode": "SECTOR_MAPPING_TOP100_MANUAL_REVIEW",
                "summary": summary,
                "rows": _jsonable(review.replace({np.nan: None}).to_dict(orient="records")),
                "hard_gate": _hard_gate(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["patch_yaml"].write_text(yaml.safe_dump(patch, allow_unicode=True, sort_keys=False), encoding="utf-8")
    paths["summary_md"].write_text(_top100_summary_markdown(summary, patch, review), encoding="utf-8")
    return SectorMappingTop100ReviewResult(review, patch, summary, paths)


def build_sector_mapping_suggestions(
    *,
    sector_map: dict[str, dict[str, Any]],
    universe: pd.DataFrame,
    out_dir: str | Path,
    price: pd.DataFrame | None = None,
    strong_audit: pd.DataFrame | None = None,
    historical_review: pd.DataFrame | None = None,
    current_position: dict[str, Any] | None = None,
    entry_signal: pd.DataFrame | None = None,
) -> SectorMappingSuggestionResult:
    universe_frame = _prepare_universe(universe)
    price_frame = _prepare_price(price)
    strong_codes = _code_set(strong_audit, "code")
    core_codes = _extract_core_recovered_codes(historical_review)
    historical_counts = _code_counts(historical_review)
    holding_codes = _holding_codes(current_position)
    entry_candidate_codes = _entry_candidate_codes(entry_signal)
    liquidity_scores = _liquidity_scores(universe_frame, price_frame)

    rows: list[dict[str, Any]] = []
    for _, item in universe_frame.sort_values("code").iterrows():
        code = str(item["code"])
        formal = sector_map.get(code)
        if formal:
            mapping = _formal_mapping(formal, fallback_name=str(item.get("name") or ""))
        else:
            mapping = _rule_mapping(item)
        priority_score, priority_reason = _priority(
            code=code,
            formal=bool(formal),
            needs_manual=bool(mapping["needs_manual_review"]),
            holding_codes=holding_codes,
            entry_candidate_codes=entry_candidate_codes,
            strong_codes=strong_codes,
            core_codes=core_codes,
            liquidity_scores=liquidity_scores,
            historical_counts=historical_counts,
        )
        rows.append(
            {
                **{column: mapping[column] for column in SUGGESTION_COLUMNS if column not in {"code", "priority_score", "priority_reason_cn"}},
                "code": code,
                "priority_score": priority_score,
                "priority_reason_cn": priority_reason,
            }
        )

    suggestion = pd.DataFrame(rows, columns=SUGGESTION_COLUMNS).sort_values(
        ["priority_score", "mapping_confidence", "code"], ascending=[False, False, True]
    )
    manual_review = suggestion.loc[suggestion["needs_manual_review"].astype(bool)].copy()
    priority_review = suggestion.loc[~suggestion["mapping_source"].eq("formal_config")].copy()
    priority_review = priority_review.sort_values(["priority_score", "mapping_confidence", "code"], ascending=[False, False, True])
    coverage = _coverage(
        suggestion=suggestion,
        universe=universe_frame,
        sector_map=sector_map,
        strong_codes=strong_codes,
        core_codes=core_codes,
        entry_candidate_codes=entry_candidate_codes,
        holding_codes=holding_codes,
    )

    out = ensure_dir(out_dir)
    paths = {
        "suggestion_csv": out / "sector_mapping_suggestion.csv",
        "suggestion_md": out / "sector_mapping_suggestion.md",
        "suggestion_json": out / "sector_mapping_suggestion.json",
        "manual_review_csv": out / "sector_mapping_manual_review_queue.csv",
        "priority_review_csv": out / "sector_mapping_priority_review.csv",
        "coverage_md": out / "sector_mapping_coverage_after_suggestion.md",
        "coverage_json": out / "sector_mapping_coverage_after_suggestion.json",
    }
    suggestion.to_csv(paths["suggestion_csv"], index=False, encoding="utf-8-sig")
    manual_review.to_csv(paths["manual_review_csv"], index=False, encoding="utf-8-sig")
    priority_review.to_csv(paths["priority_review_csv"], index=False, encoding="utf-8-sig")
    paths["suggestion_md"].write_text(_suggestion_markdown(suggestion, coverage), encoding="utf-8")
    paths["suggestion_json"].write_text(
        json.dumps(
            {
                "mode": "SECTOR_MAPPING_SUGGESTION_OFFLINE",
                "coverage": coverage,
                "rows": _jsonable(suggestion.replace({np.nan: None}).to_dict(orient="records")),
                "hard_gate": _hard_gate(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    paths["coverage_md"].write_text(_coverage_markdown(coverage, manual_review, priority_review), encoding="utf-8")
    paths["coverage_json"].write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")

    return SectorMappingSuggestionResult(suggestion, manual_review, priority_review, coverage, paths)


def _prepare_universe(universe: pd.DataFrame) -> pd.DataFrame:
    frame = universe.copy()
    code_col = "symbol" if "symbol" in frame.columns else "code"
    if code_col not in frame.columns:
        raise ValueError("universe missing symbol/code column")
    frame["code"] = frame[code_col].map(_normalize_code)
    for column in ["name", "asset_class", "category", "tracking_index", "avg_amount_20", "spot_amount"]:
        if column not in frame.columns:
            frame[column] = ""
    frame["name"] = frame["name"].fillna("").astype(str)
    frame["asset_class"] = frame["asset_class"].fillna("").astype(str)
    frame["category"] = frame["category"].fillna("").astype(str)
    frame["tracking_index"] = frame["tracking_index"].fillna("").astype(str)
    frame["avg_amount_20"] = pd.to_numeric(frame["avg_amount_20"], errors="coerce")
    frame["spot_amount"] = pd.to_numeric(frame["spot_amount"], errors="coerce")
    frame = frame.loc[frame["code"].astype(str).str.len().gt(0)].copy()
    return frame.drop_duplicates("code", keep="last")


def _prepare_price(price: pd.DataFrame | None) -> pd.DataFrame:
    if price is None or price.empty:
        return pd.DataFrame(columns=["code", "amount"])
    frame = price.copy()
    if "code" not in frame.columns:
        return pd.DataFrame(columns=["code", "amount"])
    frame["code"] = frame["code"].map(_normalize_code)
    if "date" in frame.columns:
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        frame = frame.sort_values("date")
    if "amount" not in frame.columns:
        frame["amount"] = np.nan
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce")
    return frame.dropna(subset=["code"])


def _prepare_priority_review(priority_review: pd.DataFrame) -> pd.DataFrame:
    frame = priority_review.copy()
    required = ["code", "name", "asset_class", "level1", "level2", "level3", "mapping_source", "mapping_confidence", "mapping_reason_cn", "priority_score", "priority_reason_cn"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"priority review missing required columns: {missing}")
    frame["code"] = frame["code"].map(_normalize_code)
    for column in ["name", "asset_class", "level1", "level2", "level3", "mapping_source", "mapping_reason_cn", "priority_reason_cn"]:
        frame[column] = frame[column].fillna("").astype(str)
    frame["mapping_confidence"] = pd.to_numeric(frame["mapping_confidence"], errors="coerce").fillna(0.0)
    frame["priority_score"] = pd.to_numeric(frame["priority_score"], errors="coerce").fillna(0.0)
    return frame.loc[frame["code"].astype(str).str.len().gt(0)].drop_duplicates("code", keep="first")


def _formal_mapping(item: dict[str, Any], *, fallback_name: str) -> dict[str, Any]:
    level2 = _text(item.get("sector_l2") or item.get("sector") or item.get("theme") or UNKNOWN_LEVEL)
    return {
        "name": _text(item.get("name") or fallback_name),
        "asset_class": _text(item.get("asset_class") or UNKNOWN_LEVEL),
        "level1": _text(item.get("sector_l1") or item.get("sector") or level2),
        "level2": level2,
        "level3": _text(item.get("theme") or level2),
        "mapping_source": "formal_config",
        "mapping_confidence": 1.0,
        "mapping_reason_cn": "已有正式 sector map，保留正式映射，不覆盖。",
        "needs_manual_review": False,
    }


def _rule_mapping(item: pd.Series) -> dict[str, Any]:
    name = _text(item.get("name"))
    asset_hint = _text(item.get("asset_class"))
    category = _text(item.get("category"))
    tracking = _text(item.get("tracking_index"))
    text = f"{name} {asset_hint} {category} {tracking}"
    normalized = text.upper()

    if _has(text, ["货币", "现金", "天天", "日利", "添益", "保证金"]):
        return _mapping(name, "货币", "防御资产", "债券现金", "货币现金", "universe_asset_class+money_keyword", 0.92)
    if _has(text, ["国债", "政金债", "政策性金融债", "信用债", "公司债", "地方债", "城投债", "可转债", "短融", "债"]):
        level3 = _first_theme(text, [("可转债", "可转债"), ("信用债", "信用债"), ("公司债", "信用债"), ("地方债", "地方债"), ("政金", "政策金融债"), ("国债", "国债")], "债券")
        return _mapping(name, "债券", "防御资产", "债券现金", level3, "bond_keyword", 0.90)
    if _has(text, ["黄金", "豆粕", "有色金属期货", "商品", "能源化工", "原油"]):
        level2 = "黄金商品" if "黄金" in text else ("能源商品" if _has(text, ["原油", "能源化工"]) else "资源周期")
        level3 = _first_theme(text, [("黄金", "黄金"), ("豆粕", "农产品"), ("有色", "有色金属"), ("原油", "原油")], "商品")
        return _mapping(name, "商品", "商品资产", level2, level3, "commodity_keyword", 0.88)
    if asset_hint == "跨境" or category == "跨境" or _has(text, ["QDII", "港股", "恒生", "纳斯达克", "标普", "日经", "德国", "法国", "印度", "东南亚", "沙特", "中概", "美股", "海外"]):
        level2 = "海外科技" if _has(text, ["纳斯达克", "科技", "互联网", "中概", "软件"]) else "海外宽基"
        if _has(text, ["港股", "恒生", "恒生科技"]):
            level2 = "港股科技" if "科技" in text else "港股宽基"
        level3 = _first_theme(text, [("纳斯达克", "纳斯达克"), ("标普", "标普500"), ("日经", "日经225"), ("恒生科技", "恒生科技"), ("恒生", "恒生指数"), ("中概", "中概互联")], tracking or "跨境")
        return _mapping(name, "跨境", "海外资产", level2, level3, "cross_border_keyword", 0.86)
    if _has(text, ["红利", "低波", "高股息", "价值", "央企", "国企"]):
        level2 = "红利低波" if _has(text, ["红利", "低波", "高股息"]) else "价值风格"
        level3 = _first_theme(text, [("红利", "红利"), ("低波", "低波"), ("高股息", "高股息"), ("价值", "价值"), ("央企", "央企"), ("国企", "国企")], tracking or level2)
        return _mapping(name, "权益", "防御资产", level2, level3, "style_defensive_keyword", 0.84)
    industry = _industry_mapping(text)
    if industry:
        level2, level3, confidence = industry
        return _mapping(name, "权益", "行业主题", level2, level3, "industry_theme_keyword", confidence)
    broad = _broad_market_level(text, normalized, category)
    if broad:
        level2, level3 = broad
        return _mapping(name, "权益", "宽基指数", level2, level3, "broad_market_keyword", 0.86)
    if category == "行业主题":
        return _mapping(name, "权益", "行业主题", UNKNOWN_LEVEL, tracking or name, "industry_theme_unclassified", 0.45)
    if category == "风格":
        level2 = "成长风格" if "成长" in text else ("价值风格" if "价值" in text else UNKNOWN_LEVEL)
        return _mapping(name, "权益", "宽基指数", level2, tracking or level2, "style_category_keyword", 0.72)
    return _mapping(name, "其他", UNKNOWN_LEVEL, UNKNOWN_LEVEL, tracking or name, "low_confidence_unclassified", 0.20)


def _broad_market_level(text: str, normalized: str, category: str) -> tuple[str, str] | None:
    if category == "宽基" or _has(text, ["沪深300", "中证500", "中证1000", "中证2000", "上证50", "科创50", "创业板", "深证100", "A500", "A100", "MSCI中国A股"]):
        if _has(text, ["中证1000", "中证2000", "创业板", "科创", "科创创业"]):
            level2 = "中小盘宽基" if not _has(text, ["科创50", "创业板"]) else "科技成长"
            return level2, _first_theme(text, [("科创50", "科创50"), ("创业板", "创业板"), ("中证1000", "中证1000"), ("中证2000", "中证2000")], "宽基")
        if _has(text, ["沪深300", "上证50", "A50", "A500", "A100", "深证100", "上证180"]) or "A股" in text or "A股" in normalized:
            return "大盘宽基", _first_theme(text, [("沪深300", "沪深300"), ("上证50", "上证50"), ("中证A500", "中证A500"), ("A500", "中证A500"), ("A100", "中证A100"), ("深证100", "深证100")], "A股")
        return "全市场宽基", _first_theme(text, [("中证500", "中证500"), ("中证1000", "中证1000"), ("中证2000", "中证2000"), ("创业板", "创业板"), ("科创", "科创")], "宽基")
    return None


def _industry_mapping(text: str) -> tuple[str, str, float] | None:
    rules = [
        (["半导体", "芯片", "集成电路"], "科技成长", "芯片半导体", 0.86),
        (["人工智能", "AI", "机器人", "软件", "计算机", "通信", "云计算", "大数据", "数据", "游戏", "传媒", "互联网"], "科技成长", "数字科技", 0.83),
        (["医药", "医疗", "生物", "创新药", "疫苗", "中药"], "医药消费", "医药医疗", 0.84),
        (["消费", "食品", "酒", "家电", "旅游", "养殖", "农业", "粮食", "畜牧"], "医药消费", "消费农业", 0.82),
        (["新能源", "光伏", "电池", "汽车", "储能", "风电", "新材料"], "先进制造", "新能源制造", 0.84),
        (["军工", "国防", "高端装备", "机床", "机械", "工业母机", "工业软件", "智能制造"], "先进制造", "高端制造", 0.83),
        (["证券", "银行", "金融", "保险", "地产", "房地产"], "金融地产", "金融地产", 0.84),
        (["有色", "煤炭", "钢铁", "化工", "石油", "稀土", "资源", "基建", "建筑", "材料"], "资源周期", "资源周期", 0.83),
        (["电力", "公用", "环保", "水务", "绿电"], "公用环保", "公用环保", 0.81),
    ]
    for tokens, level2, level3, confidence in rules:
        if _has(text, tokens):
            return level2, level3, confidence
    return None


def _mapping(
    name: str,
    asset_class: str,
    level1: str,
    level2: str,
    level3: str,
    source: str,
    confidence: float,
) -> dict[str, Any]:
    needs_review = bool(confidence < MANUAL_REVIEW_THRESHOLD or UNKNOWN_LEVEL in {level1, level2, level3})
    return {
        "name": name,
        "asset_class": asset_class,
        "level1": level1,
        "level2": level2,
        "level3": level3,
        "mapping_source": source,
        "mapping_confidence": round(float(confidence), 3),
        "mapping_reason_cn": _reason(source, confidence, needs_review),
        "needs_manual_review": needs_review,
    }


def _priority(
    *,
    code: str,
    formal: bool,
    needs_manual: bool,
    holding_codes: set[str],
    entry_candidate_codes: set[str],
    strong_codes: set[str],
    core_codes: set[str],
    liquidity_scores: dict[str, float],
    historical_counts: dict[str, int],
) -> tuple[float, str]:
    score = 0.0
    reasons: list[str] = []
    if code in holding_codes:
        score += 1000.0
        reasons.append("当前持仓")
    if code in entry_candidate_codes:
        score += 600.0
        reasons.append("当前 entry_candidate_pool")
    if code in strong_codes:
        score += 500.0
        reasons.append("ML_STRONG_RECOVERED 涉及")
    if code in core_codes:
        score += 450.0
        reasons.append("ML_CORE_RECOVERED 涉及")
    if code in liquidity_scores:
        score += liquidity_scores[code] * 250.0
        if liquidity_scores[code] >= 0.80:
            reasons.append("成交额靠前")
    if historical_counts.get(code, 0) > 0:
        frequency_score = min(1.0, math.log1p(historical_counts[code]) / math.log1p(max(historical_counts.values() or [1])))
        score += frequency_score * 150.0
        if frequency_score >= 0.80:
            reasons.append("历史样本出现频率高")
    if not formal:
        score += 10.0
        reasons.append("当前 universe 缺正式 map")
    if needs_manual:
        score += 20.0
        reasons.append("低置信度需人工复核")
    return round(score, 3), "；".join(reasons) if reasons else "正式映射已存在"


def _coverage(
    *,
    suggestion: pd.DataFrame,
    universe: pd.DataFrame,
    sector_map: dict[str, dict[str, Any]],
    strong_codes: set[str],
    core_codes: set[str],
    entry_candidate_codes: set[str],
    holding_codes: set[str],
) -> dict[str, Any]:
    total = int(len(universe))
    formal_mask = suggestion["mapping_source"].eq("formal_config")
    suggested_mask = ~formal_mask & suggestion["level2"].ne(UNKNOWN_LEVEL) & suggestion["mapping_confidence"].gt(0)
    high_conf_mask = suggested_mask & suggestion["mapping_confidence"].ge(HIGH_CONFIDENCE_THRESHOLD) & ~suggestion["needs_manual_review"].astype(bool)
    unmapped_remaining = suggestion.loc[~formal_mask & ~suggested_mask]
    manual_review = suggestion.loc[suggestion["needs_manual_review"].astype(bool)]
    formal_in_universe = int(formal_mask.sum())
    missing_total = max(total - formal_in_universe, 0)
    composition = _composition(universe, sector_map)
    source_counts = suggestion["mapping_source"].value_counts().to_dict()
    return {
        "mode": "SECTOR_MAPPING_COVERAGE_AFTER_SUGGESTION",
        "source_universe_count": total,
        "current_formal_coverage": _coverage_obj(formal_in_universe, total),
        "suggested_coverage": _coverage_obj(int(suggested_mask.sum()), missing_total),
        "high_confidence_suggested_coverage": _coverage_obj(int(high_conf_mask.sum()), missing_total),
        "manual_review_count": int(len(manual_review)),
        "unmapped_remaining_count": int(len(unmapped_remaining)),
        "formal_map_total_codes": int(len(sector_map)),
        "formal_map_codes_not_in_universe": sorted(set(sector_map) - set(universe["code"])),
        "missing_formal_map_count": missing_total,
        "missing_formal_map_composition": composition,
        "mapping_source_counts": {str(k): int(v) for k, v in source_counts.items()},
        "priority_signal_counts": {
            "current_holdings": len(holding_codes),
            "entry_candidate_pool": len(entry_candidate_codes),
            "ml_strong_recovered_codes": len(strong_codes),
            "ml_core_recovered_codes": len(core_codes),
        },
        "control_center_recommendation": "CONTINUE_SHADOW",
        "hard_gate": _hard_gate(),
        "active_sim_sector_gate": {
            "allow_unknown_sector_into_ml_strong_recovered": False,
            "allow_low_confidence_suggestion_into_ml_strong_recovered": False,
            "allow_unknown_sector_into_active_sim": False,
        },
    }


def _composition(universe: pd.DataFrame, sector_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = universe.loc[~universe["code"].isin(sector_map)].copy()
    return {
        "by_asset_class": {str(k): int(v) for k, v in missing["asset_class"].fillna("").value_counts().items()},
        "by_category": {str(k): int(v) for k, v in missing["category"].fillna("").value_counts().items()},
        "top_tracking_index": {str(k): int(v) for k, v in missing["tracking_index"].fillna("").value_counts().head(20).items()},
    }


def _coverage_obj(count: int, total: int) -> dict[str, Any]:
    return {"count": int(count), "total": int(total), "rate": round(count / total, 6) if total else 0.0}


def _liquidity_scores(universe: pd.DataFrame, price: pd.DataFrame) -> dict[str, float]:
    liquidity = universe[["code", "avg_amount_20", "spot_amount"]].copy()
    liquidity["amount"] = liquidity["avg_amount_20"].fillna(liquidity["spot_amount"])
    if not price.empty and "amount" in price.columns:
        latest = price.groupby("code", as_index=False)["amount"].mean()
        liquidity = liquidity.merge(latest.rename(columns={"amount": "price_amount"}), on="code", how="left")
        liquidity["amount"] = liquidity["amount"].fillna(liquidity["price_amount"])
    liquidity["amount"] = pd.to_numeric(liquidity["amount"], errors="coerce").fillna(0.0)
    if liquidity.empty or liquidity["amount"].max() <= 0:
        return {}
    liquidity["score"] = liquidity["amount"].rank(pct=True)
    return {str(row["code"]): float(row["score"]) for _, row in liquidity.iterrows()}


def _liquidity_hints(universe: pd.DataFrame | None, price: pd.DataFrame | None) -> dict[str, str]:
    universe_frame = _prepare_universe(universe) if universe is not None and not universe.empty else pd.DataFrame()
    price_frame = _prepare_price(price)
    if universe_frame.empty and price_frame.empty:
        return {}
    rows: list[dict[str, Any]] = []
    if not universe_frame.empty:
        amount = universe_frame[["code", "avg_amount_20", "spot_amount"]].copy()
        amount["amount"] = amount["avg_amount_20"].fillna(amount["spot_amount"])
        rows.extend(amount[["code", "amount"]].to_dict(orient="records"))
    if not price_frame.empty and "amount" in price_frame.columns:
        latest = price_frame.groupby("code", as_index=False)["amount"].mean()
        rows.extend(latest[["code", "amount"]].to_dict(orient="records"))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}
    frame["amount"] = pd.to_numeric(frame["amount"], errors="coerce").fillna(0.0)
    frame = frame.sort_values("amount", ascending=False).drop_duplicates("code", keep="first")
    total = int(len(frame))
    frame["rank"] = frame["amount"].rank(method="min", ascending=False).astype(int)
    frame["pct"] = 1.0 - ((frame["rank"] - 1) / total) if total else 0.0
    return {
        str(row["code"]): f"amount_rank={int(row['rank'])}/{total}; pct={float(row['pct']):.2%}; amount={float(row['amount']):.2f}"
        for _, row in frame.iterrows()
    }


def _extract_core_recovered_codes(historical_review: pd.DataFrame | None) -> set[str]:
    if historical_review is None or historical_review.empty:
        return set()
    try:
        from .ml_core_recovered_review import (
            _build_core_grid,
            _condition_from_grid_row,
            _filtered_recovered_pool,
            _legacy_pool,
            _metrics,
            _prepare_frame,
            _select_best_core_row,
        )

        frame = _prepare_frame(historical_review)
        if frame.empty:
            return set()
        total_trade_dates = int(frame["trade_date"].nunique())
        legacy = _metrics(_legacy_pool(frame), "legacy_v21_buy_probe", "legacy buy/probe", total_trade_dates=total_trade_dates)
        grid = _build_core_grid(frame, legacy, total_trade_dates=total_trade_dates)
        best = _select_best_core_row(grid, legacy)
        if best.empty:
            return set()
        pool = _filtered_recovered_pool(frame, _condition_from_grid_row(best))
        return set(pool["code"].map(_normalize_code).dropna().astype(str))
    except Exception:
        return set()


def _holding_codes(current_position: dict[str, Any] | None) -> set[str]:
    if not current_position or current_position.get("current_empty"):
        return set()
    holdings = current_position.get("holdings") or []
    codes = set()
    for item in holdings:
        if isinstance(item, dict) and float(item.get("shares") or 0) > 0:
            codes.add(_normalize_code(item.get("symbol") or item.get("code")))
    return {code for code in codes if code}


def _entry_candidate_codes(entry_signal: pd.DataFrame | None) -> set[str]:
    if entry_signal is None or entry_signal.empty:
        return set()
    frame = entry_signal.copy()
    code_col = "symbol" if "symbol" in frame.columns else "code"
    if code_col not in frame.columns:
        return set()
    action_cols = [column for column in ["final_buy_action", "raw_entry_action", "buy_action"] if column in frame.columns]
    mask = pd.Series(False, index=frame.index)
    for column in action_cols:
        text = frame[column].fillna("").astype(str).str.upper()
        mask |= text.isin(["BUY", "PROBE"]) | text.str.contains("买入", na=False)
    if "position_size" in frame.columns:
        mask |= pd.to_numeric(frame["position_size"], errors="coerce").fillna(0).gt(0)
    return set(frame.loc[mask, code_col].map(_normalize_code).dropna().astype(str))


def _code_counts(frame: pd.DataFrame | None) -> dict[str, int]:
    if frame is None or frame.empty or "code" not in frame.columns:
        return {}
    codes = frame["code"].map(_normalize_code)
    return {str(k): int(v) for k, v in codes.value_counts().items()}


def _code_set(frame: pd.DataFrame | None, column: str) -> set[str]:
    if frame is None or frame.empty or column not in frame.columns:
        return set()
    return set(frame[column].map(_normalize_code).dropna().astype(str))


def _top100_sort_bucket(row: pd.Series) -> int:
    if bool(row.get("appears_in_entry_candidate_pool")):
        return 0
    if bool(row.get("appears_in_ml_strong_recovered")):
        return 1
    if bool(row.get("appears_in_ml_core_recovered")):
        return 2
    return 3


def _patch_draft_from_review(review: pd.DataFrame) -> dict[str, Any]:
    etfs: list[dict[str, Any]] = []
    for _, row in review.iterrows():
        level2 = str(row["suggested_level2"] or UNKNOWN_LEVEL)
        level3 = str(row["suggested_level3"] or level2)
        aliases = []
        for value in [level3, level2, str(row["name"] or "")]:
            value = value.strip()
            if value and value != UNKNOWN_LEVEL and value not in aliases:
                aliases.append(value)
        etfs.append(
            {
                "code": str(row["code"]),
                "name": str(row["name"]),
                "asset_class": str(row["suggested_asset_class"]),
                "sector_l1": str(row["suggested_level1"]),
                "sector_l2": level2,
                "sector": level2,
                "theme": level3,
                "risk_group": _risk_group_for(row),
                "aliases": aliases[:4],
            }
        )
    return {
        "version": 1,
        "description": "DRAFT ONLY: top100 sector mapping candidates pending manual confirmation; do not auto-apply.",
        "etfs": etfs,
    }


def _risk_group_for(row: pd.Series) -> str:
    level1 = str(row["suggested_level1"] or "")
    level2 = str(row["suggested_level2"] or "")
    level3 = str(row["suggested_level3"] or "")
    if level1 in {"防御资产", "商品资产", "海外资产"}:
        return level2 or level1
    if level2 and level2 != UNKNOWN_LEVEL:
        return level2
    if level3 and level3 != UNKNOWN_LEVEL:
        return level3
    return UNKNOWN_LEVEL


def _top100_summary(review: pd.DataFrame, patch: dict[str, Any], formal_codes: set[str]) -> dict[str, Any]:
    patch_codes = [str(item.get("code")) for item in patch.get("etfs", [])]
    duplicate_formal_codes = sorted(set(patch_codes) & formal_codes)
    high_conf = review["mapping_confidence"].ge(HIGH_CONFIDENCE_THRESHOLD)
    pending_low_conf = ~high_conf | review[["suggested_level1", "suggested_level2", "suggested_level3"]].eq(UNKNOWN_LEVEL).any(axis=1)
    return {
        "mode": "SECTOR_MAPPING_TOP100_PATCH_DRAFT",
        "review_count": int(len(review)),
        "patch_candidate_count": int(len(patch_codes)),
        "high_confidence_candidate_count": int(high_conf.sum()),
        "manual_pending_count": int(len(review)),
        "low_confidence_or_unknown_count": int(pending_low_conf.sum()),
        "duplicate_formal_code_count": int(len(duplicate_formal_codes)),
        "duplicate_formal_codes": duplicate_formal_codes,
        "ml_strong_recovered_codes_in_review": sorted(review.loc[review["appears_in_ml_strong_recovered"], "code"].astype(str).unique().tolist()),
        "ml_core_recovered_codes_in_review": sorted(review.loc[review["appears_in_ml_core_recovered"], "code"].astype(str).unique().tolist()),
        "entry_candidate_pool_codes_in_review": sorted(review.loc[review["appears_in_entry_candidate_pool"], "code"].astype(str).unique().tolist()),
        "control_center_recommendation": "CONTINUE_SHADOW",
        "hard_gate": _hard_gate(),
        "active_sim_sector_gate": {
            "allow_unknown_sector_into_ml_strong_recovered": False,
            "allow_low_confidence_suggestion_into_ml_strong_recovered": False,
            "allow_unknown_sector_into_active_sim": False,
        },
    }


def _prepare_top100_apply_review(review: pd.DataFrame) -> pd.DataFrame:
    frame = review.copy()
    rename = {
        "suggested_asset_class": "asset_class",
        "suggested_level1": "level1",
        "suggested_level2": "level2",
        "suggested_level3": "level3",
    }
    frame = frame.rename(columns=rename)
    required = [
        "code",
        "name",
        "asset_class",
        "level1",
        "level2",
        "level3",
        "mapping_confidence",
        "manual_confirm_status",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"top100 review missing required columns: {missing}")
    frame["code"] = frame["code"].map(_normalize_code)
    for column in ["name", "asset_class", "level1", "level2", "level3", "mapping_source", "mapping_reason_cn", "priority_reason_cn", "manual_confirm_status", "manual_note"]:
        if column not in frame.columns:
            frame[column] = ""
        frame[column] = frame[column].fillna("").astype(str)
    frame["manual_confirm_status"] = frame["manual_confirm_status"].replace("", "PENDING")
    frame["mapping_confidence"] = pd.to_numeric(frame["mapping_confidence"], errors="coerce").fillna(0.0)
    if "priority_score" not in frame.columns:
        frame["priority_score"] = 0.0
    frame["priority_score"] = pd.to_numeric(frame["priority_score"], errors="coerce").fillna(0.0)
    return frame.loc[frame["code"].astype(str).str.len().gt(0)].drop_duplicates("code", keep="first")


def _top100_reject_reasons(
    row: pd.Series,
    existing_codes: set[str],
    patch_by_code: dict[str, dict[str, Any]],
    high_confidence_threshold: float,
) -> list[str]:
    reasons: list[str] = []
    code = str(row["code"])
    if code in existing_codes:
        reasons.append("formal_code_conflict")
    if code not in patch_by_code:
        reasons.append("missing_from_patch_draft")
    if float(row["mapping_confidence"]) < high_confidence_threshold:
        reasons.append("low_confidence")
    if str(row["manual_confirm_status"]).strip().upper() == "REJECTED":
        reasons.append("manual_rejected")
    if any(UNKNOWN_LEVEL in str(row.get(column, "")) for column in ["asset_class", "level1", "level2", "level3"]):
        reasons.append("contains_pending_manual_classification")
    return reasons


def _validated_patch_item(item: dict[str, Any], row: pd.Series) -> dict[str, Any]:
    level2 = str(row["level2"])
    level3 = str(row["level3"])
    aliases = item.get("aliases") if isinstance(item.get("aliases"), list) else []
    clean_aliases = []
    for alias in aliases + [level3, level2]:
        alias_text = str(alias or "").strip()
        if alias_text and alias_text != UNKNOWN_LEVEL and alias_text not in clean_aliases:
            clean_aliases.append(alias_text)
    return {
        "code": str(row["code"]),
        "name": str(row["name"]),
        "asset_class": str(row["asset_class"]),
        "sector_l1": str(row["level1"]),
        "sector_l2": level2,
        "sector": level2,
        "theme": level3,
        "risk_group": str(item.get("risk_group") or _risk_group_for(row)),
        "aliases": clean_aliases[:4],
    }


def _append_sector_map_entries(path: Path, entries: list[dict[str, Any]]) -> None:
    if not entries:
        return
    original = path.read_text(encoding="utf-8")
    item_prefix = _sector_map_item_prefix(original)
    chunks: list[str] = []
    for item in entries:
        dumped = yaml.safe_dump([item], allow_unicode=True, sort_keys=False, width=120)
        chunks.append("\n".join(f"{item_prefix}{line}" if line.strip() else line for line in dumped.splitlines()))
    append_text = ("\n" if original and not original.endswith("\n") else "") + "\n".join(chunks) + "\n"
    candidate = original + append_text
    parsed = yaml.safe_load(candidate) or {}
    parsed_codes = [_normalize_code(item.get("code")) for item in parsed.get("etfs", []) if isinstance(item, dict)]
    for item in entries:
        if _normalize_code(item.get("code")) not in parsed_codes:
            raise ValueError(f"failed to append sector map code: {item.get('code')}")
    path.write_text(candidate, encoding="utf-8")


def _sector_map_item_prefix(text: str) -> str:
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "etfs:":
            for next_line in lines[idx + 1 :]:
                stripped = next_line.lstrip()
                if stripped.startswith("- "):
                    return next_line[: len(next_line) - len(stripped)]
                if stripped and not next_line.startswith((" ", "\t")):
                    return ""
    return "  "


def _coverage_after_apply(universe: pd.DataFrame | None, existing_codes: set[str], accepted_patch_items: list[dict[str, Any]]) -> dict[str, Any]:
    universe_frame = _prepare_universe(universe) if universe is not None and not universe.empty else pd.DataFrame()
    universe_codes = set(universe_frame["code"]) if not universe_frame.empty else set()
    accepted_codes = {_normalize_code(item.get("code")) for item in accepted_patch_items}
    accepted_codes = {code for code in accepted_codes if code}
    before_formal_in_universe = len(existing_codes & universe_codes) if universe_codes else len(existing_codes)
    after_codes = existing_codes | accepted_codes
    after_formal_in_universe = len(after_codes & universe_codes) if universe_codes else len(after_codes)
    total = len(universe_codes) if universe_codes else 0
    return {
        "mode": "SECTOR_MAPPING_COVERAGE_AFTER_APPLY",
        "source_universe_count": int(total),
        "accepted_apply_count": int(len(accepted_codes)),
        "current_formal_coverage_before_apply": _coverage_obj(before_formal_in_universe, total),
        "current_formal_coverage_after_apply": _coverage_obj(after_formal_in_universe, total),
        "formal_coverage_delta_count": int(after_formal_in_universe - before_formal_in_universe),
        "formal_map_total_before_apply": int(len(existing_codes)),
        "formal_map_total_after_apply": int(len(after_codes)),
        "remaining_missing_formal_map_count": int(max(total - after_formal_in_universe, 0)),
        "hard_gate": _apply_hard_gate(),
    }


def _apply_summary(
    *,
    prepared: pd.DataFrame,
    accepted: pd.DataFrame,
    rejected: pd.DataFrame,
    coverage: dict[str, Any],
    accepted_patch_items: list[dict[str, Any]],
) -> dict[str, Any]:
    reject_counts = rejected["reject_reason"].str.get_dummies(sep="|").sum().astype(int).to_dict() if not rejected.empty else {}
    return {
        "mode": "SECTOR_MAPPING_APPLY_TOP100",
        "source_review_count": int(len(prepared)),
        "accepted_count": int(len(accepted)),
        "rejected_count": int(len(rejected)),
        "rejected_reason_counts": {str(k): int(v) for k, v in reject_counts.items()},
        "accepted_codes": [str(item.get("code")) for item in accepted_patch_items],
        "low_confidence_or_pending_not_applied": int(
            len(
                rejected.loc[
                    rejected.get("reject_reason", pd.Series(dtype=str))
                    .fillna("")
                    .astype(str)
                    .str.contains("low_confidence|contains_pending_manual_classification", regex=True)
                ]
            )
            if not rejected.empty
            else 0
        ),
        "coverage": coverage,
        "control_center_recommendation": "CONTINUE_SHADOW",
        "hard_gate": _apply_hard_gate(),
        "active_sim_sector_gate": {
            "allow_unknown_sector_into_ml_strong_recovered": False,
            "allow_low_confidence_suggestion_into_ml_strong_recovered": False,
            "allow_unknown_sector_into_active_sim": False,
        },
    }


def _sector_apply_threshold_report(
    *,
    sector_data: dict[str, Any],
    accepted_patch_items: list[dict[str, Any]],
    historical_review: pd.DataFrame | None,
    price: pd.DataFrame | None,
    universe: pd.DataFrame | None,
    out_dir: Path,
) -> dict[str, Any]:
    if historical_review is None or historical_review.empty:
        return {"mode": "ML_RECOVERED_THRESHOLD_AFTER_SECTOR_APPLY", "error": "historical_review_missing", "hard_gate": _hard_gate()}
    from .sector_mapping_audit import build_sector_mapping_audit

    temp_map = {str(item.get("code")): dict(item) for item in (sector_data.get("etfs", []) or []) if isinstance(item, dict)}
    for item in accepted_patch_items:
        temp_map[str(item.get("code"))] = dict(item)
    with tempfile.TemporaryDirectory(prefix="sector_apply_audit_", dir=str(out_dir)) as tmp_name:
        tmp_dir = Path(tmp_name)
        temp_path = tmp_dir / "sector_map.yaml"
        temp_path.write_text(yaml.safe_dump({"etfs": list(temp_map.values())}, allow_unicode=True, sort_keys=False), encoding="utf-8")
        result = build_sector_mapping_audit(
            historical_review=historical_review,
            out_dir=tmp_dir,
            sector_map_path=temp_path,
            universe_path=None,
            price_path=None,
        )
    report = dict(result.report_json)
    report["mode"] = "ML_RECOVERED_THRESHOLD_AFTER_SECTOR_APPLY"
    report["sector_apply_summary"] = {
        "accepted_apply_count": int(len(accepted_patch_items)),
        "accepted_codes": [str(item.get("code")) for item in accepted_patch_items],
    }
    report["apply_hard_gate"] = _apply_hard_gate()
    return report


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
    if path is None or str(path).strip() == "":
        return pd.DataFrame()
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return read_table(p)


def _read_current_position(path: str | Path | None) -> dict[str, Any]:
    if path is None or str(path).strip() == "":
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return raw if isinstance(raw, dict) else {}


def _suggestion_markdown(suggestion: pd.DataFrame, coverage: dict[str, Any]) -> str:
    top = suggestion.head(40)
    return "\n".join(
        [
            "# Sector Mapping Suggestion",
            "",
            "## control_center Conclusion",
            "",
            "- recommendation: CONTINUE_SHADOW",
            "- formal sector map changed: no",
            "- entry/final_buy_action changed: no",
            "- QMT triggered: no",
            "- market data refreshed: no",
            "- data/cache written: no",
            "",
            "## Coverage Summary",
            "",
            _dict_lines(
                {
                    "source_universe_count": coverage["source_universe_count"],
                    "current_formal_coverage": coverage["current_formal_coverage"],
                    "suggested_coverage": coverage["suggested_coverage"],
                    "high_confidence_suggested_coverage": coverage["high_confidence_suggested_coverage"],
                    "manual_review_count": coverage["manual_review_count"],
                    "unmapped_remaining_count": coverage["unmapped_remaining_count"],
                }
            ),
            "",
            "## Top Priority Suggestions",
            "",
            _markdown_table(top),
            "",
            "## Boundary",
            "",
            "- Suggestions are offline only and are not written back to config/etf_sector_map.yaml.",
            "- 行业未录入、板块未知、低置信度建议不得进入 ML_STRONG_RECOVERED 或 active_sim.",
        ]
    )


def _coverage_markdown(coverage: dict[str, Any], manual_review: pd.DataFrame, priority_review: pd.DataFrame) -> str:
    composition = coverage["missing_formal_map_composition"]
    return "\n".join(
        [
            "# Sector Mapping Coverage After Suggestion",
            "",
            "## Required Coverage Fields",
            "",
            _dict_lines(
                {
                    "current_formal_coverage": coverage["current_formal_coverage"],
                    "suggested_coverage": coverage["suggested_coverage"],
                    "high_confidence_suggested_coverage": coverage["high_confidence_suggested_coverage"],
                    "manual_review_count": coverage["manual_review_count"],
                    "unmapped_remaining_count": coverage["unmapped_remaining_count"],
                }
            ),
            "",
            "## 1402 Missing Formal Map Composition",
            "",
            "### By Asset Class",
            "",
            _dict_lines(composition["by_asset_class"]),
            "",
            "### By Category",
            "",
            _dict_lines(composition["by_category"]),
            "",
            "### Top Tracking Index",
            "",
            _dict_lines(composition["top_tracking_index"]),
            "",
            "## Manual Review Queue Preview",
            "",
            _markdown_table(manual_review.head(30)),
            "",
            "## Priority Review Preview",
            "",
            _markdown_table(priority_review.head(30)),
        ]
    )


def _top100_review_markdown(review: pd.DataFrame, summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Sector Mapping Top100 Manual Review",
            "",
            "## control_center Conclusion",
            "",
            "- recommendation: CONTINUE_SHADOW",
            "- formal sector map changed: no",
            "- entry/final_buy_action changed: no",
            "- QMT triggered: no",
            "- market data refreshed: no",
            "- data/cache written: no",
            "- ML_STRONG_RECOVERED / ML_CORE_RECOVERED remain blocked from active_sim by sector hard gate.",
            "",
            "## Summary",
            "",
            _dict_lines(summary),
            "",
            "## Review Rows",
            "",
            _markdown_table(review),
        ]
    )


def _top100_summary_markdown(summary: dict[str, Any], patch: dict[str, Any], review: pd.DataFrame) -> str:
    high_conf = review.loc[review["mapping_confidence"].ge(HIGH_CONFIDENCE_THRESHOLD)]
    needs_attention = review.loc[
        review["mapping_confidence"].lt(HIGH_CONFIDENCE_THRESHOLD)
        | review[["suggested_level1", "suggested_level2", "suggested_level3"]].eq(UNKNOWN_LEVEL).any(axis=1)
    ]
    return "\n".join(
        [
            "# Sector Mapping Top100 Patch Draft Summary",
            "",
            "## Patch Draft Status",
            "",
            _dict_lines(summary),
            "",
            "## High Confidence Candidates",
            "",
            _markdown_table(high_conf[["code", "name", "suggested_asset_class", "suggested_level1", "suggested_level2", "suggested_level3", "mapping_confidence", "priority_reason_cn"]].head(50)),
            "",
            "## Needs Manual Attention",
            "",
            _markdown_table(needs_attention[["code", "name", "suggested_asset_class", "suggested_level1", "suggested_level2", "suggested_level3", "mapping_confidence", "priority_reason_cn"]].head(50)),
            "",
            "## YAML Patch Draft",
            "",
            f"- file: output/sector_mapping_top100_patch_draft.yaml",
            f"- etfs: {len(patch.get('etfs', []))}",
            "- apply_status: do not apply until manual confirmation.",
        ]
    )


def _apply_summary_markdown(summary: dict[str, Any], accepted: pd.DataFrame, rejected: pd.DataFrame) -> str:
    rejected_preview = rejected[["code", "name", "mapping_confidence", "level2", "level3", "manual_confirm_status", "reject_reason"]] if not rejected.empty else rejected
    accepted_preview = accepted[["code", "name", "mapping_confidence", "level1", "level2", "level3", "manual_confirm_status"]] if not accepted.empty else accepted
    return "\n".join(
        [
            "# Sector Mapping Apply Top100 Summary",
            "",
            "## control_center Conclusion",
            "",
            "- recommendation: CONTINUE_SHADOW",
            "- formal sector map changed: yes, append-only high-confidence confirmed candidates.",
            "- entry/final_buy_action changed: no",
            "- QMT triggered: no",
            "- market data refreshed: no",
            "- data/cache written: no",
            "- Low-confidence and pending-classification rows were not applied.",
            "",
            "## Summary",
            "",
            _dict_lines(summary),
            "",
            "## Accepted",
            "",
            _markdown_table(accepted_preview),
            "",
            "## Rejected / Not Applied",
            "",
            _markdown_table(rejected_preview.head(80) if not rejected_preview.empty else rejected_preview),
        ]
    )


def _coverage_after_apply_markdown(coverage: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Sector Mapping Coverage After Apply",
            "",
            _dict_lines(coverage),
            "",
            "## Boundary",
            "",
            "- Coverage is formal-map coverage only after append-only high-confidence Top100 apply.",
            "- This report does not permit low-confidence rows into ML_STRONG_RECOVERED or active_sim.",
        ]
    )


def _threshold_after_apply_markdown(report: dict[str, Any]) -> str:
    summary = report.get("sector_audit_summary", {})
    comparison = pd.DataFrame(report.get("comparison", []))
    p1 = report.get("p1_high_risk", [])
    return "\n".join(
        [
            "# ML Recovered Threshold After Sector Apply",
            "",
            "## control_center Conclusion",
            "",
            f"- recommendation: {report.get('recommendation', 'CONTINUE_SHADOW')}",
            "- entry/final_buy_action changed: no",
            "- QMT triggered: no",
            "- market data refreshed: no",
            "- data/cache written: no",
            "",
            "## Sector Audit Summary",
            "",
            _dict_lines(summary),
            "",
            "## P1 High Risk",
            "",
            _list_lines(p1),
            "",
            "## Comparison",
            "",
            _markdown_table(comparison),
        ]
    )


def _reason(source: str, confidence: float, needs_review: bool) -> str:
    prefix = {
        "universe_asset_class+money_keyword": "依据 universe 资产类别和货币/现金关键词建议归入货币现金。",
        "bond_keyword": "依据债券类关键词建议归入债券现金。",
        "commodity_keyword": "依据商品/黄金/期货关键词建议归入商品资产。",
        "cross_border_keyword": "依据跨境、海外指数或 QDII 关键词建议归入海外资产。",
        "style_defensive_keyword": "依据红利、低波、价值、央国企等风格关键词建议归入防御资产。",
        "broad_market_keyword": "依据宽基分类或主流宽基指数关键词建议归入宽基指数。",
        "industry_theme_keyword": "依据行业主题关键词建议归入行业主题。",
        "industry_theme_unclassified": "仅能确认行业主题大类，细分行业置信度不足。",
        "style_category_keyword": "依据风格分类给出风格建议，仍需人工确认细分。",
        "low_confidence_unclassified": "名称、跟踪指数和 universe 分类不足以可靠归类。",
    }.get(source, "基于规则生成建议。")
    if needs_review:
        return f"{prefix} 置信度 {confidence:.2f}，进入人工复核队列。"
    return f"{prefix} 置信度 {confidence:.2f}。"


def _first_theme(text: str, pairs: list[tuple[str, str]], fallback: str) -> str:
    for token, value in pairs:
        if token in text:
            return value
    return fallback


def _has(text: str, tokens: list[str]) -> bool:
    return any(token and token in text for token in tokens)


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _normalize_code(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text.endswith(".0") and text.replace(".", "", 1).isdigit():
        text = text[:-2]
    if text.startswith(("SH", "SZ")):
        text = text[2:]
    if "." in text:
        text = text.split(".", 1)[0]
    match = re.search(r"\d{6}", text)
    if match:
        return match.group(0)
    if text.isdigit() and len(text) <= 6:
        return text.zfill(6)
    return text


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


def _apply_hard_gate() -> dict[str, bool]:
    gate = _hard_gate()
    gate["formal_sector_map_changed"] = True
    gate["offline_recommendation_only"] = False
    return gate


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
        return "- 无"
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
