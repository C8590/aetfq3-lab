from __future__ import annotations

import json

import pandas as pd
import yaml

from historical_ml.cli import main
from historical_ml.sector_mapping_audit import build_sector_mapping_audit


def test_sector_audit_moves_unmapped_strong_to_review_bucket(tmp_path):
    frame = _review_frame(include_mapped_strong=False)
    sector_map_path = tmp_path / "sector_map.yaml"
    sector_map_path.write_text(yaml.safe_dump({"etfs": []}, allow_unicode=True), encoding="utf-8")
    universe_path = tmp_path / "universe.csv"
    _universe_frame().to_csv(universe_path, index=False)
    price_path = tmp_path / "prices.csv"
    _price_frame().to_csv(price_path, index=False)

    result = build_sector_mapping_audit(
        historical_review=frame,
        out_dir=tmp_path,
        sector_map_path=sector_map_path,
        universe_path=universe_path,
        price_path=price_path,
    )

    assert result.recommendation == "CONTINUE_SHADOW"
    assert result.report_json["sector_audit_summary"]["strong_before_count"] == 4
    assert result.report_json["sector_audit_summary"]["strong_unmapped_count"] == 4
    assert result.report_json["sector_audit_summary"]["strong_after_excluding_unmapped_count"] == 0
    assert set(result.sector_audit["ml_review_bucket_after_audit"]) == {"ML_UNMAPPED_REVIEW"}
    assert result.report_json["hard_gate"]["formal_sector_map_changed"] is False
    assert result.report_json["hard_gate"]["final_buy_action_changed"] is False
    assert result.report_json["hard_gate"]["qmt_triggered"] is False
    assert (tmp_path / "ml_strong_recovered_sector_audit.csv").exists()
    assert (tmp_path / "sector_mapping_coverage_report.csv").exists()
    assert (tmp_path / "ml_recovered_threshold_after_sector_audit.json").exists()


def test_sector_audit_keeps_mapped_strong_but_not_active_when_tiny(tmp_path):
    frame = _review_frame(include_mapped_strong=True)
    sector_map_path = tmp_path / "sector_map.yaml"
    sector_map_path.write_text(
        yaml.safe_dump(
            {
                "etfs": [
                    {"code": "510300", "sector_l1": "宽基指数", "sector_l2": "大盘宽基", "sector": "大盘宽基"},
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    result = build_sector_mapping_audit(
        historical_review=frame,
        out_dir=tmp_path,
        sector_map_path=sector_map_path,
        universe_path=None,
        price_path=None,
    )

    assert result.report_json["sector_audit_summary"]["strong_after_excluding_unmapped_count"] == 1
    assert result.recommendation == "CONTINUE_SHADOW"


def test_sector_audit_formal_map_resolves_historical_unmapped_strong(tmp_path):
    frame = _review_frame(include_mapped_strong=False)
    sector_map_path = tmp_path / "sector_map.yaml"
    sector_map_path.write_text(
        yaml.safe_dump(
            {
                "etfs": [
                    {"code": "516300", "sector_l1": "宽基指数", "sector_l2": "中小盘宽基", "sector": "中小盘宽基"},
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    result = build_sector_mapping_audit(
        historical_review=frame,
        out_dir=tmp_path,
        sector_map_path=sector_map_path,
        universe_path=None,
        price_path=None,
    )

    assert result.report_json["sector_audit_summary"]["strong_before_count"] == 4
    assert result.report_json["sector_audit_summary"]["strong_unmapped_count"] == 0
    assert result.report_json["sector_audit_summary"]["strong_after_excluding_unmapped_count"] == 4


def test_sector_audit_cli_runs(tmp_path):
    input_path = tmp_path / "historical_review.csv"
    _review_frame(include_mapped_strong=False).to_csv(input_path, index=False)
    map_path = tmp_path / "sector_map.yaml"
    map_path.write_text(yaml.safe_dump({"etfs": []}, allow_unicode=True), encoding="utf-8")

    rc = main(
        [
            "sector-mapping-audit",
            "--historical-review",
            str(input_path),
            "--out",
            str(tmp_path / "out"),
            "--sector-map",
            str(map_path),
            "--universe",
            "",
            "--prices",
            "",
        ]
    )

    assert rc == 0
    report = json.loads((tmp_path / "out" / "ml_recovered_threshold_after_sector_audit.json").read_text(encoding="utf-8"))
    assert report["mode"] == "ML_RECOVERED_THRESHOLD_AFTER_SECTOR_AUDIT"


def _review_frame(*, include_mapped_strong: bool) -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2026-01-02", periods=4)
    for d_idx, date in enumerate(dates):
        trade_date = date.strftime("%Y-%m-%d")
        for i in range(40):
            is_strong = i == 0
            code = "510300" if include_mapped_strong and d_idx == 0 and is_strong else ("516300" if is_strong else f"159{i:03d}")
            sector = "宽基指数" if code == "510300" else "行业未录入"
            is_legacy = 20 <= i < 25
            label = "good_entry" if is_strong else ("good_entry" if is_legacy and i == 20 else ("bad_entry" if is_legacy and i in {21, 22} else "neutral_entry"))
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "name": "中证1000ETF华泰柏瑞" if code == "516300" else "沪深300ETF",
                    "sector_level1": sector,
                    "sector_level2": sector,
                    "market_state": "neutral",
                    "sector_state": "strong",
                    "legacy_action": "PROBE" if is_legacy else "OBSERVE",
                    "ml_adjustment_type": "ML_RECOVERED" if is_strong else "ML_UNCHANGED",
                    "ml_adjustment_bucket": "ML_RECOVERED" if is_strong else "ML_UNCHANGED",
                    "ml_score": 100 - i,
                    "p_good_entry": 0.99 - i * 0.01,
                    "p_bad_entry": 0.01 + i * 0.01,
                    "ml_rank_global": i + 1,
                    "ml_rank_sector": 1,
                    "momentum_score": 1.0 - i * 0.01,
                    "expected_drawdown_10d": -0.005 if is_strong else -0.02,
                    "future_return_3d": 0.02 if is_strong else 0.0,
                    "future_return_5d": 0.03 if is_strong else 0.0,
                    "future_return_10d": 0.05 if is_strong else 0.0,
                    "future_max_drawdown_10d": -0.01,
                    "outperform_market_10d": is_strong,
                    "outperform_sector_10d": is_strong,
                    "review_status": "READY",
                    "auto_label": label,
                    "is_valid_sample": True,
                    "exclude_reason": "",
                }
            )
    return pd.DataFrame(rows)


def _universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "symbol": "516300",
                "name": "中证1000ETF华泰柏瑞",
                "asset_class": "A股股票",
                "category": "宽基",
                "tracking_index": "中证1000",
                "latest_date": "2026-01-02",
            }
        ]
    )


def _price_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": "2026-01-02", "code": "516300", "sector_l1": "行业未录入", "sector_l2": "行业未录入"},
            {"date": "2026-01-05", "code": "516300", "sector_l1": "行业未录入", "sector_l2": "行业未录入"},
        ]
    )
