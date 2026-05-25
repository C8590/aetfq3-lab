from __future__ import annotations

import json

import pandas as pd
import yaml

from historical_ml.cli import main
from historical_ml.sector_mapping_suggestion import apply_sector_mapping_top100_from_files, build_sector_mapping_suggestions, build_sector_mapping_top100_review


def test_sector_mapping_suggestion_keeps_formal_map_and_builds_review_queue(tmp_path):
    result = build_sector_mapping_suggestions(
        sector_map={
            "510300": {
                "code": "510300",
                "name": "沪深300ETF",
                "asset_class": "权益",
                "sector_l1": "宽基指数",
                "sector_l2": "大盘宽基",
                "theme": "沪深300",
            }
        },
        universe=_universe_frame(),
        price=pd.DataFrame(),
        strong_audit=pd.DataFrame([{"code": "588000"}]),
        historical_review=_historical_review_frame(),
        current_position={"current_empty": False, "holdings": [{"symbol": "511880", "shares": 100}]},
        entry_signal=pd.DataFrame([{"symbol": "588000", "final_buy_action": "PROBE", "position_size": 0.3}]),
        out_dir=tmp_path,
    )

    rows = result.suggestion.set_index("code")
    assert rows.loc["510300", "mapping_source"] == "formal_config"
    assert rows.loc["510300", "needs_manual_review"] == False
    assert rows.loc["511880", "asset_class"] == "货币"
    assert rows.loc["511880", "mapping_confidence"] >= 0.8
    assert rows.loc["159999", "needs_manual_review"] == True
    assert "当前持仓" in rows.loc["511880", "priority_reason_cn"]
    assert "当前 entry_candidate_pool" in rows.loc["588000", "priority_reason_cn"]
    assert "ML_STRONG_RECOVERED 涉及" in rows.loc["588000", "priority_reason_cn"]
    assert result.coverage["current_formal_coverage"]["count"] == 1
    assert result.coverage["manual_review_count"] >= 1
    assert result.coverage["unmapped_remaining_count"] >= 1
    assert result.coverage["hard_gate"]["formal_sector_map_changed"] is False
    assert (tmp_path / "sector_mapping_suggestion.csv").exists()
    assert (tmp_path / "sector_mapping_manual_review_queue.csv").exists()
    assert (tmp_path / "sector_mapping_coverage_after_suggestion.json").exists()


def test_sector_mapping_suggestion_cli_runs(tmp_path):
    universe_path = tmp_path / "universe.csv"
    _universe_frame().to_csv(universe_path, index=False)
    map_path = tmp_path / "sector_map.yaml"
    map_path.write_text(
        yaml.safe_dump({"etfs": [{"code": "510300", "asset_class": "权益", "sector_l1": "宽基指数", "sector_l2": "大盘宽基", "theme": "沪深300"}]}, allow_unicode=True),
        encoding="utf-8",
    )
    review_path = tmp_path / "review.csv"
    _historical_review_frame().to_csv(review_path, index=False)
    strong_path = tmp_path / "strong.csv"
    pd.DataFrame([{"code": "588000"}]).to_csv(strong_path, index=False)
    entry_path = tmp_path / "entry.csv"
    pd.DataFrame([{"symbol": "588000", "final_buy_action": "PROBE", "position_size": 0.3}]).to_csv(entry_path, index=False)
    position_path = tmp_path / "current_position.yaml"
    position_path.write_text(yaml.safe_dump({"current_empty": True, "holdings": []}, allow_unicode=True), encoding="utf-8")

    rc = main(
        [
            "sector-mapping-suggestion",
            "--out",
            str(tmp_path / "out"),
            "--sector-map",
            str(map_path),
            "--universe",
            str(universe_path),
            "--prices",
            "",
            "--strong-audit",
            str(strong_path),
            "--historical-review",
            str(review_path),
            "--current-position",
            str(position_path),
            "--entry-signal",
            str(entry_path),
        ]
    )

    assert rc == 0
    coverage = json.loads((tmp_path / "out" / "sector_mapping_coverage_after_suggestion.json").read_text(encoding="utf-8"))
    assert "current_formal_coverage" in coverage
    assert "suggested_coverage" in coverage
    assert "high_confidence_suggested_coverage" in coverage
    assert "manual_review_count" in coverage
    assert "unmapped_remaining_count" in coverage
    assert coverage["control_center_recommendation"] == "CONTINUE_SHADOW"


def test_top100_review_package_prioritizes_flags_and_writes_parseable_patch(tmp_path):
    priority = pd.DataFrame(
        [
            _priority_row("510300", "沪深300ETF", 999, "当前 universe 缺正式 map"),
            _priority_row("159558", "半导体设备ETF易方达", 900, "当前 entry_candidate_pool；当前 universe 缺正式 map"),
            _priority_row("563330", "A股ETF华泰柏瑞", 800, "ML_STRONG_RECOVERED 涉及；当前 universe 缺正式 map", confidence=0.45, level2="待人工分类"),
            _priority_row("159638", "高端装备ETF嘉实", 700, "ML_CORE_RECOVERED 涉及；当前 universe 缺正式 map", level2="先进制造", level3="高端制造"),
            _priority_row("159999", "测试ETF", 600, "当前 universe 缺正式 map", confidence=0.45, level2="待人工分类"),
        ]
    )
    result = build_sector_mapping_top100_review(
        priority_review=priority,
        sector_map={"510300": {"code": "510300", "sector_l1": "宽基指数", "sector_l2": "大盘宽基"}},
        out_dir=tmp_path,
        universe=_universe_frame(),
        price=pd.DataFrame(),
        strong_audit=pd.DataFrame([{"code": "563330"}]),
        historical_review=_historical_review_frame_for_top100(),
        entry_signal=pd.DataFrame([{"symbol": "159558", "final_buy_action": "PROBE", "position_size": 0.3}]),
        top_n=4,
    )

    codes = result.review["code"].tolist()
    assert codes[:3] == ["159558", "563330", "159638"]
    assert "510300" not in codes
    assert result.summary["duplicate_formal_code_count"] == 0
    assert result.review.loc[result.review["code"].eq("563330"), "appears_in_ml_strong_recovered"].iloc[0] == True
    assert result.review.loc[result.review["code"].eq("159558"), "appears_in_entry_candidate_pool"].iloc[0] == True
    assert set(result.review["manual_confirm_status"]) == {"PENDING"}
    assert all(item["code"] != "510300" for item in result.patch_draft["etfs"])
    parsed = yaml.safe_load((tmp_path / "sector_mapping_top100_patch_draft.yaml").read_text(encoding="utf-8"))
    assert parsed["version"] == 1
    assert len(parsed["etfs"]) == len(result.review)
    assert (tmp_path / "sector_mapping_top100_review.csv").exists()
    assert (tmp_path / "sector_mapping_top100_review.json").exists()
    assert (tmp_path / "sector_mapping_top100_patch_summary.md").exists()


def test_top100_review_cli_runs(tmp_path):
    priority_path = tmp_path / "priority.csv"
    pd.DataFrame(
        [
            _priority_row("159558", "半导体设备ETF易方达", 900, "当前 entry_candidate_pool；当前 universe 缺正式 map"),
            _priority_row("563330", "A股ETF华泰柏瑞", 800, "ML_STRONG_RECOVERED 涉及；当前 universe 缺正式 map", confidence=0.45, level2="待人工分类"),
        ]
    ).to_csv(priority_path, index=False)
    map_path = tmp_path / "sector_map.yaml"
    map_path.write_text(yaml.safe_dump({"etfs": []}, allow_unicode=True), encoding="utf-8")
    universe_path = tmp_path / "universe.csv"
    _universe_frame().to_csv(universe_path, index=False)
    review_path = tmp_path / "review.csv"
    _historical_review_frame_for_top100().to_csv(review_path, index=False)
    strong_path = tmp_path / "strong.csv"
    pd.DataFrame([{"code": "563330"}]).to_csv(strong_path, index=False)
    entry_path = tmp_path / "entry.csv"
    pd.DataFrame([{"symbol": "159558", "final_buy_action": "PROBE", "position_size": 0.3}]).to_csv(entry_path, index=False)

    rc = main(
        [
            "sector-mapping-top100-review",
            "--out",
            str(tmp_path / "out"),
            "--priority-review",
            str(priority_path),
            "--sector-map",
            str(map_path),
            "--universe",
            str(universe_path),
            "--prices",
            "",
            "--strong-audit",
            str(strong_path),
            "--historical-review",
            str(review_path),
            "--entry-signal",
            str(entry_path),
            "--top-n",
            "2",
        ]
    )

    assert rc == 0
    patch = yaml.safe_load((tmp_path / "out" / "sector_mapping_top100_patch_draft.yaml").read_text(encoding="utf-8"))
    assert len(patch["etfs"]) == 2
    report = json.loads((tmp_path / "out" / "sector_mapping_top100_review.json").read_text(encoding="utf-8"))
    assert report["summary"]["control_center_recommendation"] == "CONTINUE_SHADOW"


def test_apply_top100_appends_only_high_confidence_non_pending_rows(tmp_path):
    map_path = tmp_path / "sector_map.yaml"
    map_path.write_text(
        yaml.safe_dump({"version": 1, "etfs": [{"code": "510300", "name": "沪深300ETF", "asset_class": "权益", "sector_l1": "宽基指数", "sector_l2": "大盘宽基", "sector": "大盘宽基", "theme": "沪深300", "risk_group": "全市场"}]}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    review = pd.DataFrame(
        [
            {
                "code": "159558",
                "name": "半导体设备ETF易方达",
                "suggested_asset_class": "权益",
                "suggested_level1": "行业主题",
                "suggested_level2": "科技成长",
                "suggested_level3": "芯片半导体",
                "mapping_confidence": 0.86,
                "manual_confirm_status": "PENDING",
            },
            {
                "code": "563330",
                "name": "A股ETF华泰柏瑞",
                "suggested_asset_class": "权益",
                "suggested_level1": "行业主题",
                "suggested_level2": "待人工分类",
                "suggested_level3": "A股",
                "mapping_confidence": 0.45,
                "manual_confirm_status": "PENDING",
            },
            {
                "code": "510300",
                "name": "沪深300ETF",
                "suggested_asset_class": "权益",
                "suggested_level1": "宽基指数",
                "suggested_level2": "大盘宽基",
                "suggested_level3": "沪深300",
                "mapping_confidence": 0.90,
                "manual_confirm_status": "PENDING",
            },
        ]
    )
    review_path = tmp_path / "review.csv"
    review.to_csv(review_path, index=False)
    patch_path = tmp_path / "patch.yaml"
    patch_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "etfs": [
                    {"code": "159558", "name": "半导体设备ETF易方达", "asset_class": "权益", "sector_l1": "行业主题", "sector_l2": "科技成长", "sector": "科技成长", "theme": "芯片半导体", "risk_group": "科技成长", "aliases": ["芯片半导体"]},
                    {"code": "563330", "name": "A股ETF华泰柏瑞", "asset_class": "权益", "sector_l1": "行业主题", "sector_l2": "待人工分类", "sector": "待人工分类", "theme": "A股", "risk_group": "待人工分类", "aliases": ["A股"]},
                    {"code": "510300", "name": "沪深300ETF", "asset_class": "权益", "sector_l1": "宽基指数", "sector_l2": "大盘宽基", "sector": "大盘宽基", "theme": "沪深300", "risk_group": "全市场", "aliases": ["沪深300"]},
                ],
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    universe_path = tmp_path / "universe.csv"
    pd.DataFrame(
        [
            {"symbol": "510300", "name": "沪深300ETF"},
            {"symbol": "159558", "name": "半导体设备ETF易方达"},
            {"symbol": "563330", "name": "A股ETF华泰柏瑞"},
        ]
    ).to_csv(universe_path, index=False)
    historical_path = tmp_path / "historical.csv"
    _historical_review_frame_for_top100().to_csv(historical_path, index=False)

    result = apply_sector_mapping_top100_from_files(
        sector_map_path=map_path,
        review_path=review_path,
        patch_draft_path=patch_path,
        out_dir=tmp_path,
        universe_path=universe_path,
        historical_review_path=historical_path,
        price_path=None,
    )

    parsed = yaml.safe_load(map_path.read_text(encoding="utf-8"))
    codes = [item["code"] for item in parsed["etfs"]]
    assert codes.count("159558") == 1
    assert "563330" not in codes
    assert codes.count("510300") == 1
    assert result.summary["accepted_count"] == 1
    assert result.summary["rejected_reason_counts"]["low_confidence"] == 1
    assert result.summary["rejected_reason_counts"]["contains_pending_manual_classification"] == 1
    assert result.summary["rejected_reason_counts"]["formal_code_conflict"] == 1
    assert (tmp_path / "sector_mapping_apply_top100_summary.json").exists()
    assert (tmp_path / "sector_mapping_coverage_after_apply.json").exists()
    assert (tmp_path / "ml_recovered_threshold_after_sector_apply.json").exists()


def _priority_row(code: str, name: str, score: float, reason: str, *, confidence: float = 0.86, level2: str = "科技成长", level3: str = "芯片半导体") -> dict[str, object]:
    return {
        "code": code,
        "name": name,
        "asset_class": "权益",
        "level1": "行业主题",
        "level2": level2,
        "level3": level3,
        "mapping_source": "industry_theme_keyword",
        "mapping_confidence": confidence,
        "mapping_reason_cn": "测试建议",
        "needs_manual_review": confidence < 0.8,
        "priority_score": score,
        "priority_reason_cn": reason,
    }


def _universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"symbol": "510300.SH", "name": "沪深300ETF", "asset_class": "A股股票", "category": "宽基", "tracking_index": "沪深300", "avg_amount_20": 100},
            {"symbol": "511880", "name": "银华日利货币ETF", "asset_class": "货币", "category": "货币", "tracking_index": "货币", "avg_amount_20": 200},
            {"symbol": "588000", "name": "科创50ETF", "asset_class": "A股股票", "category": "宽基", "tracking_index": "科创50", "avg_amount_20": 300},
            {"symbol": "159999", "name": "测试ETF", "asset_class": "A股股票", "category": "行业主题", "tracking_index": "", "avg_amount_20": 1},
            {"symbol": "159558", "name": "半导体设备ETF易方达", "asset_class": "A股股票", "category": "行业主题", "tracking_index": "半导体设备", "avg_amount_20": 400},
            {"symbol": "563330", "name": "A股ETF华泰柏瑞", "asset_class": "A股股票", "category": "行业主题", "tracking_index": "A股", "avg_amount_20": 500},
            {"symbol": "159638", "name": "高端装备ETF嘉实", "asset_class": "A股股票", "category": "行业主题", "tracking_index": "高端装备", "avg_amount_20": 600},
        ]
    )


def _historical_review_frame() -> pd.DataFrame:
    rows = []
    for idx, code in enumerate(["588000", "511880", "159999"]):
        rows.append(
            {
                "trade_date": f"2026-01-0{idx + 2}",
                "code": code,
                "name": code,
                "review_status": "READY",
                "legacy_action": "OBSERVE",
                "ml_adjustment_type": "ML_RECOVERED",
                "ml_adjustment_bucket": "ML_RECOVERED",
                "market_state": "neutral",
                "sector_state": "strong",
                "is_valid_sample": True,
                "exclude_reason": "",
                "p_good_entry": 0.9 - idx * 0.1,
                "p_bad_entry": 0.01,
                "ml_rank_global": idx + 1,
                "ml_rank_sector": 1,
                "momentum_score": 1.0,
                "acceleration_score": 1.0,
                "expected_drawdown_10d": -0.01,
                "etf_rank": 100,
                "future_return_10d": 0.01,
                "auto_label": "good_entry",
                "sector_level1": "行业未录入",
                "sector_level2": "行业未录入",
            }
        )
    return pd.DataFrame(rows)


def _historical_review_frame_for_top100() -> pd.DataFrame:
    frame = _historical_review_frame()
    extra = frame.iloc[[0]].copy()
    extra["code"] = "159638"
    extra["name"] = "高端装备ETF嘉实"
    extra["ml_rank_sector"] = 1
    extra["etf_rank"] = 100
    return pd.concat([frame, extra], ignore_index=True)
