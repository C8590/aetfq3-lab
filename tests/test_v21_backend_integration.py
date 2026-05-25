from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from contracts.v21_schema import DAILY_DECISION_FIELDS, ML_SIM_COMPARISON_FIELDS, ORDER_INTENT_FIELDS, RISK_GATE_FIELDS
from signal.v21_orchestrator import run_v21_backend_pipeline


def _pre_rows():
    return [
        {
            "trade_date": "2026-05-20",
            "symbol": "159915",
            "name": "创业板ETF",
            "sector": "成长",
            "market_state": "进攻",
            "score": 88,
            "rank": 1,
            "selected": True,
            "reason": "进入候选池。",
        }
    ]


def _entry_rows(action: str = "标准买入", weight: float = 0.3):
    return [
        {
            "trade_date": "2026-05-20",
            "symbol": "159915",
            "name": "创业板ETF",
            "market_state": "进攻",
            "buy_action": action,
            "position_size": weight,
            "confidence": 0.8,
            "entry_reason": "趋势成熟度：确认期；买点质量：回踩确认；理由：测试买入。",
            "source_file": "entry_signal.csv",
        }
    ]


def _risk(level: str = "R0", *, freeze: bool = False, manual: bool = False):
    return {
        "risk_date": "2026-05-20",
        "risk_level": level,
        "risk_score": 85 if level in {"R3", "R4", "P0"} else 5,
        "freeze_entry": freeze,
        "equity_cap_override": 0.0 if freeze else 1.0,
        "manual_takeover_required": manual,
        "affected_sectors": ["成长"] if freeze else [],
        "active_events": [{"title": "测试风险", "affected_assets": ["159915"]}] if freeze else [],
        "explain": "测试风险门控说明。",
    }


def test_risk_warning_freeze_blocks_actual_buy(tmp_path: Path) -> None:
    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk("R3", freeze=True),
        entry_rows=_entry_rows(),
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    decision = result["daily_decision"]
    assert decision["freeze_entry"] is True
    assert decision["actual_buy_etfs"] == []
    assert any(item["risk_check_passed"] is False for item in result["order_intent"])


def test_exit_clear_has_priority_over_new_buy(tmp_path: Path) -> None:
    exit_rows = [
        {
            "trade_date": "2026-05-20",
            "symbol": "159915",
            "name": "创业板ETF",
            "sell_action": "清仓",
            "reduce_ratio": 1.0,
            "exit_reason": "风险退出：跌破趋势线。",
        }
    ]
    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows(),
        exit_rows=exit_rows,
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[{"symbol": "159915", "name": "创业板ETF", "shares": 1000, "cost_price": 1.0, "current_price": 1.2}],
        qmt_execution_available=True,
    )

    assert result["daily_decision"]["actual_buy_etfs"] == []
    assert any(item["side"] == "SELL" for item in result["order_intent"])
    entry_action = result["daily_decision"]["entry_actions"][0]
    assert entry_action["raw_entry_action"] == "BUY"
    assert entry_action["final_buy_action"] == "BLOCKED"
    assert entry_action["exit_priority_blocked"] is True
    assert "exit" in entry_action["final_block_reason"]
    assert entry_action["active_exit_count"] == 1
    assert entry_action["actual_position_exit_count"] == 1
    assert entry_action["has_real_position_to_exit"] is True
    assert entry_action["blocked_by_exit_symbols"] == ["159915"]
    assert entry_action["exit_action_type"] == "清仓退出"
    assert "159915" in entry_action["exit_block_reason"]
    assert "解除条件" in entry_action["final_block_reason"]
    assert "159915" in result["daily_decision"]["exit_block_release_condition"]


def test_exit_signal_without_real_position_does_not_block_raw_probe(tmp_path: Path) -> None:
    exit_rows = [
        {
            "trade_date": "2026-05-20",
            "symbol": "159915",
            "name": "创业板ETF",
            "sell_action": "clear",
            "reduce_ratio": 1.0,
            "exit_reason": "risk exit from stale exit_signal.csv",
        }
    ]
    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows("probe_buy", 0.3),
        exit_rows=exit_rows,
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    decision = result["daily_decision"]
    entry_action = decision["entry_actions"][0]
    exit_action = decision["exit_actions"][0]
    assert decision["active_exit_count"] == 1
    assert decision["actual_position_exit_count"] == 0
    assert decision["exit_priority_blocked"] is False
    assert entry_action["raw_entry_action"] == "PROBE"
    assert entry_action["final_buy_action"] == "PROBE"
    assert entry_action["exit_priority_blocked"] is False
    assert entry_action["actual_buy"] is True
    assert exit_action["active_exit"] is True
    assert exit_action["has_real_position_to_exit"] is False
    assert exit_action["actual_exit"] is False


def test_historical_ml_missing_degrades_without_interrupt(tmp_path: Path) -> None:
    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows("观察", 0.0),
        exit_rows=[],
        learning_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    assert result["historical_ml_summary"] == []
    assert "historical_ml" in result["daily_decision"]["fallback_reason"]
    assert (tmp_path / "historical_ml_summary.csv").exists()


def test_historical_ml_generated_parameter_suggestions_enter_v21_snapshot(tmp_path: Path) -> None:
    suggestions_path = tmp_path / "artifacts" / "historical_ml_61" / "generated" / "entry_calibration_suggestions.csv"
    suggestions_path.parent.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "suggestion_id": "CAL-001",
                "parameter_area": "trend_maturity",
                "current_pattern": "overheat stage has elevated bad_rate",
                "suggested_action": "require pullback confirmation",
                "confidence": "medium",
                "sample_count": 100,
                "bad_rate": 0.42,
                "notes": "observe only",
            }
        ]
    ).to_csv(suggestions_path, index=False, encoding="utf-8-sig")

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows("瑙傚療", 0.0),
        exit_rows=[],
        learning_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    assert len(result["historical_ml_summary"]) == 1
    item = result["historical_ml_summary"][0]
    assert item["entry_quality"] == "trend_maturity"
    assert item["calibration_suggestion"] == "require pullback confirmation"
    assert item["ml_action_suggestion"] == "KEEP_ORIGINAL"
    assert "historical_ml" not in result["daily_decision"]["fallback_reason"]
    assert json.loads((tmp_path / "historical_ml_summary.json").read_text(encoding="utf-8"))


def test_qmt_execution_missing_writes_draft_fallback(tmp_path: Path) -> None:
    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows(),
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=False,
    )

    assert "qmt_execution" in result["daily_decision"]["fallback_reason"]
    assert all(item["execution_mode"] in {"DRAFT", "MANUAL_CONFIRM", "SIMULATION"} for item in result["order_intent"])


def test_daily_decision_and_risk_gate_write_csv_json(tmp_path: Path) -> None:
    run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows("观察", 0.0),
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    daily_csv = pd.read_csv(tmp_path / "daily_decision_snapshot.csv")
    risk_csv = pd.read_csv(tmp_path / "risk_gate_snapshot.csv")
    daily_json = json.loads((tmp_path / "daily_decision_snapshot.json").read_text(encoding="utf-8"))
    risk_json = json.loads((tmp_path / "risk_gate_snapshot.json").read_text(encoding="utf-8"))
    assert set(DAILY_DECISION_FIELDS).issubset(daily_csv.columns)
    assert set(RISK_GATE_FIELDS).issubset(risk_csv.columns)
    assert set(DAILY_DECISION_FIELDS).issubset(daily_json.keys())
    assert set(RISK_GATE_FIELDS).issubset(risk_json.keys())


def test_ml_observation_fields_pass_through_v21_snapshot_without_changing_trade_fields(tmp_path: Path) -> None:
    entry_rows = _entry_rows("标准买入", 0.3)
    entry_rows[0].update(
        {
            "ml_entry_advice": "建议等待回踩",
            "ml_confidence": 0.73,
            "ml_reason": "历史样本提示当前买点偏急，仅供观察。",
            "ml_action_suggestion": "WAIT_PULLBACK",
        }
    )

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=entry_rows,
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    decision = result["daily_decision"]
    entry_action = decision["entry_actions"][0]
    candidate = decision["candidate_etfs"][0]
    assert decision["ml_observation_status"].startswith("ML 观察模式已启用")
    assert "仅供观察，不自动修改交易参数" in decision["ml_entry_advice"]
    assert entry_action["entry_action"] == "标准买入"
    assert entry_action["target_weight"] == 0.3
    assert entry_action["ml_entry_advice"] == "建议等待回踩"
    assert entry_action["ml_action_suggestion"] == "WAIT_PULLBACK"
    assert candidate["ml_entry_advice"] == "建议等待回踩"
    daily_json = json.loads((tmp_path / "daily_decision_snapshot.json").read_text(encoding="utf-8"))
    assert daily_json["entry_actions"][0]["ml_reason"] == "历史样本提示当前买点偏急，仅供观察。"


def test_entry_probe_blocked_by_exit_priority_preserves_raw_probe(tmp_path: Path) -> None:
    exit_rows = [
        {
            "trade_date": "2026-05-20",
            "symbol": "159915",
            "name": "创业板ETF",
            "sell_action": "清仓",
            "reduce_ratio": 1.0,
            "exit_reason": "风险退出：测试。",
        }
    ]
    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows("试探买入", 0.3),
        exit_rows=exit_rows,
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[{"symbol": "159915", "name": "创业板ETF", "shares": 1000, "cost_price": 1.0, "current_price": 1.2}],
        qmt_execution_available=True,
    )

    entry_action = result["daily_decision"]["entry_actions"][0]
    assert entry_action["raw_entry_action"] == "PROBE"
    assert entry_action["raw_entry_target_weight"] == 0.3
    assert entry_action["final_buy_action"] == "BLOCKED"
    assert entry_action["final_target_weight"] == 0.0
    assert entry_action["exit_priority_blocked"] is True
    assert "exit" in entry_action["final_block_reason"]
    assert entry_action["actual_position_exit_count"] == 1
    assert entry_action["blocked_by_exit_symbols"] == ["159915"]
    assert "解除条件" in entry_action["final_block_reason"]


def test_positive_ml_suggestion_does_not_change_raw_or_final_action(tmp_path: Path) -> None:
    entry_rows = _entry_rows("观察", 0.0)
    entry_rows[0].update(
        {
            "ml_entry_advice": "建议升级小仓试探",
            "ml_confidence": 0.91,
            "ml_reason": "ML 只观察，不改变 entry。",
            "ml_action_suggestion": "UPGRADE_PROBE",
        }
    )

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=entry_rows,
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    entry_action = result["daily_decision"]["entry_actions"][0]
    assert entry_action["ml_action_suggestion"] == "UPGRADE_PROBE"
    assert entry_action["raw_entry_action"] == "OBSERVE"
    assert entry_action["final_buy_action"] == "OBSERVE"
    assert entry_action["actual_buy"] is False
    assert result["entry_diagnostics"][0]["ml_observation_notice"] == "ML 参数级建议已读取 / 未直接参与交易裁决"


def test_active_sim_ml_recovered_probe_participates_in_sim_layer(tmp_path: Path) -> None:
    entry_rows = _entry_rows("观察", 0.0)
    entry_rows[0].update(
        {
            "raw_entry_action": "OBSERVE",
            "raw_entry_target_weight": 0.0,
            "rule_action": "OBSERVE",
            "ml_adjusted_action": "PROBE",
            "final_buy_action": "PROBE",
            "final_target_weight": 0.3,
            "ml_score": 68.0,
            "p_good_entry": 0.78,
            "p_bad_entry": 0.05,
            "ml_action_suggestion": "UPGRADE_PROBE",
            "ml_decision_mode": "active_sim",
            "ml_adjustment": "ML_RECOVERED",
            "ml_adjustment_reason_cn": "active_sim: ML_RECOVERED PROBE from ml_entry_scores.",
        }
    )

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=entry_rows,
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    entry_action = result["daily_decision"]["entry_actions"][0]
    assert entry_action["rule_action"] == "OBSERVE"
    assert entry_action["ml_adjustment"] == "ML_RECOVERED"
    assert entry_action["final_buy_action"] == "PROBE"
    assert entry_action["actual_buy"] is True
    assert result["order_intent"][0]["action"] == "DRAFT_BUY"
    assert result["order_intent"][0]["execution_mode"] in {"DRAFT", "MANUAL_CONFIRM", "SIMULATION"}
    assert result["order_intent"][0]["requires_manual_confirm"] is True


def test_active_sim_ml_downgraded_observe_blocks_sim_buy(tmp_path: Path) -> None:
    entry_rows = _entry_rows()
    entry_rows[0].update(
        {
            "raw_entry_action": "BUY",
            "raw_entry_target_weight": 0.3,
            "rule_action": "BUY",
            "ml_adjusted_action": "OBSERVE",
            "final_buy_action": "OBSERVE",
            "final_target_weight": 0.0,
            "ml_score": -52.0,
            "p_good_entry": 0.08,
            "p_bad_entry": 0.73,
            "ml_action_suggestion": "DOWNGRADE_WATCH",
            "ml_decision_mode": "active_sim",
            "ml_adjustment": "ML_DOWNGRADED",
            "ml_adjustment_reason_cn": "active_sim: ML_DOWNGRADED OBSERVE because ml_entry_scores show weaker entry quality.",
        }
    )

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=entry_rows,
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    entry_action = result["daily_decision"]["entry_actions"][0]
    assert entry_action["rule_action"] == "BUY"
    assert entry_action["ml_adjustment"] == "ML_DOWNGRADED"
    assert entry_action["final_buy_action"] == "OBSERVE"
    assert entry_action["actual_buy"] is False
    assert result["daily_decision"]["actual_buy_etfs"] == []


def test_active_sim_ml_recovered_probe_is_blocked_by_risk_warning(tmp_path: Path) -> None:
    entry_rows = _entry_rows("观察", 0.0)
    entry_rows[0].update(
        {
            "raw_entry_action": "OBSERVE",
            "raw_entry_target_weight": 0.0,
            "rule_action": "OBSERVE",
            "ml_adjusted_action": "PROBE",
            "final_buy_action": "PROBE",
            "final_target_weight": 0.3,
            "ml_action_suggestion": "UPGRADE_PROBE",
            "ml_decision_mode": "active_sim",
            "ml_adjustment": "ML_RECOVERED",
        }
    )

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk("R3", freeze=True),
        entry_rows=entry_rows,
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    entry_action = result["daily_decision"]["entry_actions"][0]
    assert entry_action["final_buy_action"] == "BLOCKED"
    assert entry_action["risk_gate_blocked"] is True
    assert entry_action["actual_buy"] is False
    assert result["daily_decision"]["actual_buy_etfs"] == []


def test_ml_sim_daily_comparison_is_observation_only_and_writes_review_outputs(tmp_path: Path) -> None:
    pre_rows = [
        {**_pre_rows()[0], "symbol": "159915", "selected": True, "sector_level1": "growth", "sector_level2": "startup"},
        {**_pre_rows()[0], "symbol": "512000", "name": "券商ETF", "selected": False, "sector_level1": "finance", "sector_level2": "broker"},
    ]
    entry_rows = [
        {
            **_entry_rows()[0],
            "symbol": "159915",
            "raw_entry_action": "BUY",
            "rule_action": "BUY",
            "final_buy_action": "BUY",
            "final_target_weight": 0.3,
            "ml_score": -55,
            "p_good_entry": 0.05,
            "p_bad_entry": 0.75,
            "ml_action_suggestion": "DOWNGRADE_WATCH",
            "ml_decision_mode": "shadow",
        },
        {
            **_entry_rows("观察", 0.0)[0],
            "symbol": "512000",
            "raw_entry_action": "OBSERVE",
            "rule_action": "OBSERVE",
            "final_buy_action": "OBSERVE",
            "final_target_weight": 0.0,
            "ml_score": 72,
            "p_good_entry": 0.82,
            "p_bad_entry": 0.04,
            "ml_action_suggestion": "UPGRADE_PROBE",
            "ml_decision_mode": "shadow",
        },
    ]

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=pre_rows,
        risk_gate=_risk(),
        entry_rows=entry_rows,
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    by_code = {row["code"]: row for row in result["ml_sim_daily_comparison"]}
    assert by_code["159915"]["legacy_action"] == "BUY"
    assert by_code["159915"]["ml_sim_action"] == "OBSERVE"
    assert by_code["159915"]["final_action"] == "BUY"
    assert by_code["159915"]["ml_adjustment_type"] == "ML_DOWNGRADED"
    assert by_code["512000"]["ml_sim_action"] == "PROBE"
    assert by_code["512000"]["final_action"] == "OBSERVE"
    assert by_code["512000"]["ml_adjustment_type"] == "ML_UPGRADED_TO_BUY_CANDIDATE"
    assert "ML_SIM 仅观察" in by_code["512000"]["ml_adjustment_reason_cn"]
    assert result["daily_decision"]["entry_actions"][0]["final_buy_action"] == "BUY"
    assert result["ml_sim_summary"]["ml_recovered_count"] == 1
    assert result["ml_sim_summary"]["ml_downgraded_count"] == 1
    assert result["ml_sim_summary"]["top_ml_recovered"][0]["code"] == "512000"
    assert result["ml_sim_summary"]["top_ml_downgraded"][0]["code"] == "159915"
    assert len(result["ml_sim_review_queue"]) == 2
    assert (tmp_path / "ml_sim_daily_comparison.csv").exists()
    assert (tmp_path / "ml_sim_daily_comparison.json").exists()
    assert (tmp_path / "ml_sim_summary.json").exists()
    assert (tmp_path / "ml_sim_review_queue.csv").exists()
    written = pd.read_csv(tmp_path / "ml_sim_daily_comparison.csv")
    assert set(ML_SIM_COMPARISON_FIELDS).issubset(written.columns)


def test_ml_sim_recovered_candidate_marks_risk_conflict_under_freeze(tmp_path: Path) -> None:
    entry_rows = _entry_rows("观察", 0.0)
    entry_rows[0].update(
        {
            "raw_entry_action": "OBSERVE",
            "rule_action": "OBSERVE",
            "final_buy_action": "OBSERVE",
            "ml_score": 80,
            "p_good_entry": 0.88,
            "p_bad_entry": 0.02,
            "ml_action_suggestion": "UPGRADE_PROBE",
            "ml_decision_mode": "shadow",
        }
    )

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk("R3", freeze=True),
        entry_rows=entry_rows,
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    row = result["ml_sim_daily_comparison"][0]
    assert row["ml_sim_action"] == "PROBE"
    assert row["ml_adjustment_type"] == "ML_CONFLICT_WITH_RISK"
    assert row["risk_blocked"] is True
    assert row["order_intent_in_ml_sim"] is False


def test_order_intent_defaults_to_manual_confirm(tmp_path: Path) -> None:
    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows(),
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    assert result["order_intent"]
    assert all(item["requires_manual_confirm"] is True for item in result["order_intent"])


def test_candidate_pool_can_exceed_legacy_top5_while_order_intents_remain_limited(tmp_path: Path) -> None:
    pre_rows = []
    entry_rows = []
    for i in range(8):
        symbol = f"{510000 + i:06d}"
        selected = i < 5
        pre_rows.append(
            {
                "trade_date": "2026-05-20",
                "symbol": symbol,
                "name": f"ETF{i}",
                "sector": "成长" if i < 4 else "消费",
                "market_state": "杩涙敾",
                "score": 90 - i,
                "rank": i + 1,
                "selected": selected,
                "candidate_pool_flag": True,
                "candidate_source": "LEGACY_TOP5" if selected else "BROAD_RECALL",
                "legacy_selected": selected,
                "broad_recall_selected": True,
                "ml_recovered": False,
                "candidate_pool_rank": i + 1,
                "reason": "candidate pool test",
            }
        )
        entry_rows.append(
            {
                "trade_date": "2026-05-20",
                "symbol": symbol,
                "name": f"ETF{i}",
                "market_state": "杩涙敾",
                "buy_action": "璇曟帰涔板叆" if selected else "瑙傚療",
                "position_size": 0.2 if selected else 0.0,
                "confidence": 0.5,
                "raw_entry_action": "PROBE" if selected else "OBSERVE",
                "raw_entry_target_weight": 0.2 if selected else 0.0,
                "final_buy_action": "PROBE" if selected else "OBSERVE",
                "final_target_weight": 0.2 if selected else 0.0,
                "entry_reason": "entry test",
                "source_file": "entry_signal.csv",
            }
        )

    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=pre_rows,
        risk_gate=_risk(),
        entry_rows=entry_rows,
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    decision = result["daily_decision"]
    assert len(decision["candidate_etfs"]) == 8
    assert len(result["order_intent"]) <= 5
    assert any(item["candidate_source"] == "BROAD_RECALL" for item in decision["candidate_etfs"])
    assert any(item["legacy_selected"] is False for item in decision["candidate_etfs"])
    assert all(item["side"] == "BUY" for item in result["order_intent"])


def test_frontend_output_fields_are_stable(tmp_path: Path) -> None:
    run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows(),
        exit_rows=[],
        learning_rows=[],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    daily = pd.read_csv(tmp_path / "daily_decision_snapshot.csv")
    order = pd.read_csv(tmp_path / "order_intent.csv")
    risk = pd.read_csv(tmp_path / "risk_gate_snapshot.csv")
    assert list(daily.columns) == list(DAILY_DECISION_FIELDS)
    assert list(order.columns) == list(ORDER_INTENT_FIELDS)
    assert list(risk.columns) == list(RISK_GATE_FIELDS)


def test_r3_r4_p0_risk_freezes_or_requires_manual(tmp_path: Path) -> None:
    for level in ("R3", "R4", "P0"):
        result = run_v21_backend_pipeline(
            output_dir=tmp_path / level,
            pre_selection_rows=_pre_rows(),
            risk_gate=_risk(level),
            entry_rows=_entry_rows(),
            exit_rows=[],
            learning_rows=[],
            historical_ml_rows=[],
            holdings=[],
            qmt_execution_available=True,
        )
        risk = result["risk_gate"]
        assert risk["freeze_entry"] is True or risk["manual_takeover_required"] is True


def test_post_924_regime_is_preserved(tmp_path: Path) -> None:
    result = run_v21_backend_pipeline(
        output_dir=tmp_path,
        pre_selection_rows=_pre_rows(),
        risk_gate=_risk(),
        entry_rows=_entry_rows("观察", 0.0),
        exit_rows=[],
        learning_rows=[
            {
                "trade_date": "2026-05-20",
                "symbol": "159915",
                "name": "创业板ETF",
                "return_pct": 0.02,
                "failure_attribution": "买点太差",
                "lesson": "测试复盘。",
                "adjustment": "仅建议，不改参数。",
            }
        ],
        historical_ml_rows=[],
        holdings=[],
        qmt_execution_available=True,
    )

    assert result["learning_summary"][0]["post_924_regime"] is True
