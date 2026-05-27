"""V2.1 backend integration orchestrator.

This layer reads or receives outputs from the seven project modules and writes a
stable controller-level snapshot for the future frontend. It only arbitrates and
serializes; it does not rewrite module formulas, entry thresholds, or QMT safety
rules.
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from data.trading_calendar import get_next_trading_day
from signal.entry.diagnostics import write_entry_diagnostics, write_entry_signal_coverage_report
from signal.execution_plan import EXECUTION_PLAN_FIELDS, build_execution_plan
from contracts.v21_schema import (
    DAILY_DECISION_FIELDS,
    ML_SIM_COMPARISON_FIELDS,
    ORDER_INTENT_FIELDS,
    PORTFOLIO_SNAPSHOT_FIELDS,
    RISK_GATE_FIELDS,
    SIGNAL_VERSION,
    TRAINING_SAMPLE_FIELDS,
    DailyDecision,
    PortfolioSnapshot,
    TrainingSample,
    V21OrderIntent,
    V21RiskGate,
)


OUTPUT_FILES = (
    "daily_decision_snapshot.csv",
    "daily_decision_snapshot.json",
    "risk_gate_snapshot.csv",
    "risk_gate_snapshot.json",
    "portfolio_snapshot.csv",
    "portfolio_snapshot.json",
    "order_intent.csv",
    "order_intent.json",
    "learning_summary.csv",
    "learning_summary.json",
    "historical_ml_summary.csv",
    "historical_ml_summary.json",
    "ml_sim_daily_comparison.csv",
    "ml_sim_daily_comparison.json",
    "ml_sim_summary.json",
    "ml_sim_review_queue.csv",
    "entry_diagnostics.csv",
    "entry_diagnostics.json",
    "entry_signal_coverage_report.csv",
    "entry_signal_coverage_report.md",
    "execution_plan.csv",
    "execution_plan.json",
    "tomorrow_trade_plan.md",
    "tomorrow_trade_plan.json",
    "v21_backend_status.json",
)

SAFE_EXECUTION_MODES = {"SIMULATION", "DRAFT", "MANUAL_CONFIRM"}
RISK_FREEZE_LEVELS = {"R3", "R4", "P0"}
ML_OBSERVATION_NOTICE = "仅供观察，不自动修改交易参数。"
HISTORICAL_ML_SUGGESTIONS_FILE = Path("artifacts") / "historical_ml_61" / "generated" / "entry_calibration_suggestions.csv"
HISTORICAL_ML_COVERAGE_REPORT_FILE = Path("artifacts") / "historical_ml_61" / "generated" / "historical_ml_universe_coverage_report.json"
HISTORICAL_ML_ENTRY_SCORES_FILE = Path("artifacts") / "historical_ml_61" / "generated" / "ml_entry_scores.csv"
ML_SIM_NOTICE = "ML_SIM 仅观察，不作为正式交易指令。"


def run_v21_backend_pipeline(
    *,
    output_dir: str | Path = "output",
    trade_date: str | pd.Timestamp | None = None,
    pre_selection_rows: Sequence[Mapping[str, Any]] | None = None,
    risk_gate: Mapping[str, Any] | Any | None = None,
    entry_rows: Sequence[Mapping[str, Any]] | None = None,
    exit_rows: Sequence[Mapping[str, Any]] | None = None,
    learning_rows: Sequence[Mapping[str, Any]] | None = None,
    historical_ml_rows: Sequence[Mapping[str, Any]] | None = None,
    holdings: Sequence[Mapping[str, Any]] | None = None,
    qmt_execution_available: bool | None = None,
    qmt_status: Mapping[str, Any] | None = None,
    account_total_asset: float | None = None,
) -> dict[str, Any]:
    """Build and write all V2.1 backend integration snapshots.

    Direct row arguments are primarily for tests and higher-level callers. When
    they are omitted, the orchestrator reads existing module outputs under
    ``output_dir`` and degrades gracefully when an optional source is absent.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated_at = _now()
    warnings: list[str] = []
    fallback_reasons: list[str] = []

    pre_rows = _rows_or_csv(pre_selection_rows, out_dir / "pre_selection_result.csv", warnings, "pre_selection")
    entry = _rows_or_csv(entry_rows, out_dir / "entry_signal.csv", warnings, "entry")
    exits = _rows_or_csv(exit_rows, out_dir / "exit_signal.csv", warnings, "exit")
    learning = _rows_or_csv(learning_rows, out_dir / "learning_report.csv", warnings, "learning")
    historical = _resolve_historical_rows(historical_ml_rows, out_dir, warnings, fallback_reasons)
    portfolio_holdings = _resolve_holdings(holdings, warnings, fallback_reasons)
    qmt_available, qmt_note = _resolve_qmt_status(qmt_execution_available, qmt_status)
    if qmt_note:
        warnings.append(qmt_note)
        fallback_reasons.append(qmt_note)

    effective_date = _resolve_trade_date(trade_date, pre_rows, entry, exits, learning, risk_gate)
    expected_execution_date = _resolve_expected_execution_date(effective_date)
    v21_risk = _build_risk_gate(risk_gate, out_dir, effective_date, warnings, fallback_reasons)

    risk_level = str(v21_risk.risk_level or "R0").upper()
    if risk_level in RISK_FREEZE_LEVELS and not (v21_risk.freeze_entry or v21_risk.manual_takeover_required):
        v21_risk = V21RiskGate(**{**v21_risk.to_dict(), "freeze_entry": True})
        warnings.append("风险等级达到 R3/R4/P0，总控已强制冻结 entry。")

    market_state = _first_text(pre_rows, "market_state", default=_first_text(entry, "market_state", default="未知"))
    selected_rows = [row for row in pre_rows if _truthy(row.get("selected"))]
    candidate_pool_rows = _candidate_pool_rows(pre_rows)
    selected_symbols = {_symbol(row.get("symbol") or row.get("etf_code") or row.get("code")) for row in selected_rows}
    selected_sectors = _unique(row.get("sector") for row in selected_rows)
    entry_by_symbol = {
        _symbol(row.get("symbol") or row.get("etf_code") or row.get("code")): row
        for row in entry
    }
    candidate_etfs = [
        _candidate_payload(
            row,
            entry_by_symbol.get(_symbol(row.get("symbol") or row.get("etf_code") or row.get("code")), {}),
        )
        for row in candidate_pool_rows
    ]

    portfolio = _build_portfolio_snapshot(
        holdings=portfolio_holdings,
        exit_rows=exits,
        pre_rows=pre_rows,
        trade_date=effective_date,
        account_total_asset=account_total_asset,
    )
    exit_actions = _build_exit_actions(exits, portfolio)
    exit_block = _build_exit_priority_block(exit_actions)
    high_priority_exit = bool(exit_block["exit_priority_blocked"])
    entry_actions = _build_entry_actions(entry, selected_symbols, v21_risk, exit_block)
    actual_buy_etfs = [item for item in entry_actions if item["actual_buy"]]
    ml_observation_status = _ml_observation_status(entry)
    ml_entry_advice = _ml_entry_advice_summary(entry, selected_symbols)
    portfolio_actions = _build_portfolio_actions(portfolio, exit_actions)

    learning_summary = [_learning_sample(row, entry, exits).to_dict() for row in learning]
    historical_summary = [_historical_sample(row).to_dict() for row in historical]

    order_intents = _build_order_intents(
        trade_date=effective_date,
        entry_actions=entry_actions,
        exit_actions=exit_actions,
        portfolio=portfolio,
        risk=v21_risk,
        qmt_available=qmt_available,
        qmt_note=qmt_note,
    )
    execution_plan = build_execution_plan(
        trade_date=effective_date,
        expected_execution_date=expected_execution_date,
        entry_actions=entry_actions,
        exit_actions=exit_actions,
        portfolio=[item.to_dict() for item in portfolio],
        risk_gate=v21_risk.to_dict(),
    )
    funnel_counts = _build_ml_funnel_counts(
        out_dir=out_dir,
        pre_rows=pre_rows,
        entry_rows=entry,
        entry_actions=entry_actions,
        candidate_etfs=candidate_etfs,
        order_intents=order_intents,
    )

    allow_entry = not v21_risk.freeze_entry and not v21_risk.manual_takeover_required and not high_priority_exit
    if high_priority_exit:
        fallback_reasons.append(str(exit_block["exit_block_reason"]))
    if v21_risk.freeze_entry:
        fallback_reasons.append("风险门控冻结买入，entry 信号只保留为观察和解释，不进入实际买入。")
    if not historical:
        fallback_reasons.append("historical_ml 暂无可用摘要，总控已降级为空建议，不中断今日决策。")

    decision = DailyDecision(
        trade_date=effective_date,
        signal_version=SIGNAL_VERSION,
        market_state=market_state,
        risk_level=v21_risk.risk_level,
        risk_score=int(_number(v21_risk.risk_score)),
        allow_entry=allow_entry,
        freeze_entry=bool(v21_risk.freeze_entry),
        manual_takeover_required=bool(v21_risk.manual_takeover_required),
        active_exit_count=int(exit_block["active_exit_count"]),
        actual_position_exit_count=int(exit_block["actual_position_exit_count"]),
        exit_priority_blocked=high_priority_exit,
        exit_block_reason=str(exit_block["exit_block_reason"]),
        exit_block_release_condition=str(exit_block["exit_block_release_condition"]),
        blocked_by_exit_symbols=list(exit_block["blocked_by_exit_symbols"]),
        has_real_position_to_exit=bool(exit_block["has_real_position_to_exit"]),
        exit_action_type=str(exit_block["exit_action_type"]),
        selected_sectors=selected_sectors,
        ml_observation_status=ml_observation_status,
        ml_entry_advice=ml_entry_advice,
        **funnel_counts,
        candidate_etfs=candidate_etfs,
        actual_buy_etfs=actual_buy_etfs,
        entry_actions=entry_actions,
        exit_actions=exit_actions,
        portfolio_actions=portfolio_actions,
        learning_summary=learning_summary,
        historical_ml_summary=historical_summary,
        order_intent_summary=order_intents,
        explain=_decision_explain(market_state, v21_risk, candidate_etfs, actual_buy_etfs, exit_actions, exit_block),
        warnings=_unique(warnings),
        fallback_reason=_join_reason(fallback_reasons),
        generated_at=generated_at,
    ).to_dict()
    entry_diagnostics = write_entry_diagnostics(
        output_dir=out_dir,
        trade_date=effective_date,
        pre_selection_rows=pre_rows,
        entry_actions=entry_actions,
        risk=v21_risk.to_dict(),
    )
    entry_coverage = write_entry_signal_coverage_report(output_dir=out_dir, diagnostics_rows=entry_diagnostics)

    _write_table(out_dir / "daily_decision_snapshot.csv", DAILY_DECISION_FIELDS, [decision])
    _write_json(out_dir / "daily_decision_snapshot.json", decision)
    _write_table(out_dir / "risk_gate_snapshot.csv", RISK_GATE_FIELDS, [v21_risk.to_dict()])
    _write_json(out_dir / "risk_gate_snapshot.json", v21_risk.to_dict())
    _write_table(out_dir / "portfolio_snapshot.csv", PORTFOLIO_SNAPSHOT_FIELDS, [item.to_dict() for item in portfolio])
    _write_json(out_dir / "portfolio_snapshot.json", [item.to_dict() for item in portfolio])
    _write_table(out_dir / "order_intent.csv", ORDER_INTENT_FIELDS, order_intents)
    _write_json(out_dir / "order_intent.json", order_intents)
    _write_table(out_dir / "execution_plan.csv", EXECUTION_PLAN_FIELDS, execution_plan)
    _write_json(out_dir / "execution_plan.json", execution_plan)
    _write_table(out_dir / "learning_summary.csv", TRAINING_SAMPLE_FIELDS, learning_summary)
    _write_json(out_dir / "learning_summary.json", learning_summary)
    _write_table(out_dir / "historical_ml_summary.csv", TRAINING_SAMPLE_FIELDS, historical_summary)
    _write_json(out_dir / "historical_ml_summary.json", historical_summary)
    ml_sim_comparison, ml_sim_summary, ml_sim_review_queue = _build_ml_sim_outputs(
        out_dir=out_dir,
        trade_date=effective_date,
        pre_rows=pre_rows,
        entry_rows=entry,
        entry_actions=entry_actions,
        risk=v21_risk,
        exit_block=exit_block,
        order_intents=order_intents,
    )
    _write_table(out_dir / "ml_sim_daily_comparison.csv", ML_SIM_COMPARISON_FIELDS, ml_sim_comparison)
    _write_json(out_dir / "ml_sim_daily_comparison.json", ml_sim_comparison)
    _write_json(out_dir / "ml_sim_summary.json", ml_sim_summary)
    _write_table(out_dir / "ml_sim_review_queue.csv", ML_SIM_COMPARISON_FIELDS, ml_sim_review_queue)

    status = {
        "trade_date": effective_date,
        "expected_execution_date": expected_execution_date,
        "signal_version": SIGNAL_VERSION,
        "status": "completed_with_fallback" if fallback_reasons else "completed",
        "generated_at": generated_at,
        "output_files": list(OUTPUT_FILES),
        "module_order": [
            "pre_selection",
            "risk_warning",
            "entry",
            "exit",
            "learning",
            "historical_ml",
            "qmt_execution",
        ],
        "priority_rules": [
            "RiskGate/P0/R4/R3 风险优先于所有买入信号。",
            "持仓真实风险和 exit 风险退出优先于新增买入。",
            "learning/historical_ml 只给建议，不自动修改交易参数。",
            "qmt_execution 只消费总控订单意图，不反向改变策略判断。",
        ],
        "fallback_reason": decision["fallback_reason"],
        "warnings": decision["warnings"],
        "entry_coverage": entry_coverage,
        "strategy_logic_modified": False,
        "entry_threshold_modified": False,
        "live_auto_order_enabled": False,
        "qmt_execution_available": qmt_available,
        "funnel_counts": funnel_counts,
        "ml_sim_summary": ml_sim_summary,
    }
    tomorrow_plan = _build_tomorrow_trade_plan(
        decision=decision,
        risk_gate=v21_risk.to_dict(),
        order_intents=order_intents,
        execution_plan=execution_plan,
        expected_execution_date=expected_execution_date,
    )
    _write_json(out_dir / "tomorrow_trade_plan.json", tomorrow_plan)
    (out_dir / "tomorrow_trade_plan.md").write_text(_tomorrow_trade_plan_markdown(tomorrow_plan), encoding="utf-8")
    _write_json(out_dir / "v21_backend_status.json", status)

    return {
        "daily_decision": decision,
        "risk_gate": v21_risk.to_dict(),
        "portfolio_snapshot": [item.to_dict() for item in portfolio],
        "order_intent": order_intents,
        "execution_plan": execution_plan,
        "learning_summary": learning_summary,
        "historical_ml_summary": historical_summary,
        "ml_sim_daily_comparison": ml_sim_comparison,
        "ml_sim_summary": ml_sim_summary,
        "ml_sim_review_queue": ml_sim_review_queue,
        "entry_diagnostics": entry_diagnostics,
        "entry_coverage": entry_coverage,
        "tomorrow_trade_plan": tomorrow_plan,
        "status": status,
    }


def _resolve_expected_execution_date(trade_date: str) -> str:
    try:
        return get_next_trading_day(pd.Timestamp(trade_date)).isoformat()
    except Exception:  # noqa: BLE001
        return (pd.Timestamp(trade_date) + pd.offsets.BDay(1)).date().isoformat()


def _entry_action_code(item: Mapping[str, Any]) -> str:
    for key in ("final_buy_action", "entry_action", "raw_entry_action", "action"):
        value = str(item.get(key) or "").strip().upper()
        if value:
            return value
    return ""


def _compact_action(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "etf_code": _symbol(item.get("etf_code") or item.get("symbol") or item.get("code")),
        "etf_name": item.get("etf_name") or item.get("name") or "",
        "action": _entry_action_code(item) or item.get("exit_action") or item.get("action") or "",
        "target_weight": item.get("final_target_weight", item.get("target_weight", "")),
        "reason": item.get("final_block_reason") or item.get("explain") or item.get("exit_reason") or "",
        "ml_readonly": ML_OBSERVATION_NOTICE,
    }


def _build_tomorrow_trade_plan(
    *,
    decision: Mapping[str, Any],
    risk_gate: Mapping[str, Any],
    order_intents: Sequence[Mapping[str, Any]],
    execution_plan: Sequence[Mapping[str, Any]],
    expected_execution_date: str,
) -> dict[str, Any]:
    entry_actions = _mapping_records(decision.get("entry_actions"))
    exit_actions = _mapping_records(decision.get("exit_actions"))
    buy_or_probe = [
        _compact_action(item)
        for item in entry_actions
        if _entry_action_code(item) in {"BUY", "PROBE"} and _bool(item.get("actual_buy"))
    ]
    observe = [
        _compact_action(item)
        for item in entry_actions
        if _entry_action_code(item) in {"OBSERVE", "NO_BUY", "NONE", "AVOID", "BLOCKED"} or not _bool(item.get("actual_buy"))
    ]
    exit_plan = [_compact_action(item) for item in exit_actions if str(item.get("exit_action") or item.get("action") or "").strip()]
    buy_execution_plan = [dict(item) for item in execution_plan if str(item.get("plan_side") or "").upper() == "BUY"]
    sell_execution_plan = [dict(item) for item in execution_plan if str(item.get("plan_side") or "").upper() == "SELL"]
    return {
        "mode": "V2.1 Stable",
        "data_date": decision.get("trade_date", ""),
        "signal_generated_at": decision.get("generated_at", ""),
        "expected_execution_date": expected_execution_date,
        "allow_buy": bool(decision.get("allow_entry")),
        "risk_gate": {
            "risk_level": risk_gate.get("risk_level", decision.get("risk_level", "")),
            "risk_score": risk_gate.get("risk_score", decision.get("risk_score", "")),
            "freeze_entry": bool(risk_gate.get("freeze_entry", decision.get("freeze_entry", False))),
            "manual_takeover_required": bool(risk_gate.get("manual_takeover_required", decision.get("manual_takeover_required", False))),
        },
        "exit_priority_blocked": bool(decision.get("exit_priority_blocked")),
        "exit_block_reason": decision.get("exit_block_reason", ""),
        "tomorrow_buy_or_probe_candidates": buy_or_probe,
        "tomorrow_sell_reduce_clear_advice": exit_plan,
        "execution_plan_summary": {
            "buy_plan_count": len(buy_execution_plan),
            "sell_plan_count": len(sell_execution_plan),
            "buy_actions": buy_execution_plan,
            "sell_actions": sell_execution_plan,
        },
        "hold_observation": observe,
        "qmt_order_drafts": [dict(item) for item in order_intents],
        "qmt_safety": "仅生成 DRAFT/MANUAL_CONFIRM/SIMULATION 草稿，不连接真实 QMT，不自动提交订单。",
        "ml_readonly_notice": ML_OBSERVATION_NOTICE,
    }


def _tomorrow_trade_plan_markdown(plan: Mapping[str, Any]) -> str:
    execution_summary = plan.get("execution_plan_summary") if isinstance(plan.get("execution_plan_summary"), Mapping) else {}
    buy_plans = _mapping_records(execution_summary.get("buy_actions"))
    sell_plans = _mapping_records(execution_summary.get("sell_actions"))
    lines = [
        "# V2.1 Stable 明日交易计划",
        "",
        f"- 数据日期：{plan.get('data_date', '')}",
        f"- 信号生成时间：{plan.get('signal_generated_at', '')}",
        f"- 预计执行日：{plan.get('expected_execution_date', '')}",
        f"- 是否允许买入：{'是' if plan.get('allow_buy') else '否'}",
        "",
        "## 买卖执行计划",
        "",
    ]
    if buy_plans:
        for item in buy_plans:
            lines.append(
                "- "
                f"{item.get('etf_code', '')} {item.get('etf_name', '')}："
                f"{item.get('execution_action', '')}；买入方式：{item.get('buy_method', '')}；"
                f"目标仓位：{item.get('target_weight', '')}；高开处理：{item.get('high_open_handling', '')}；"
                f"等待回踩：{item.get('wait_pullback_condition', '')}；"
                f"取消条件：{item.get('cancel_buy_condition', '')}；风险提示：{item.get('risk_note', '')}"
            )
    else:
        lines.append("- 当前无买入执行计划。")
    if sell_plans:
        for item in sell_plans:
            lines.append(
                "- "
                f"{item.get('etf_code', '')} {item.get('etf_name', '')}："
                f"{item.get('execution_action', '')}；卖出条件：{item.get('sell_condition', '')}；"
                f"利润保护：{item.get('profit_protection_placeholder', '')}；风险提示：{item.get('risk_note', '')}"
            )
    else:
        lines.append("- NO_SELL_PLAN")
    lines.extend(["", "## QMT 安全边界", "", str(plan.get("qmt_safety", "")), ""])
    return "\n".join(lines)


def _mapping_records(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return _mapping_records(parsed)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _rows_or_csv(
    rows: Sequence[Mapping[str, Any]] | None,
    path: Path,
    warnings: list[str],
    module_name: str,
) -> list[dict[str, Any]]:
    if rows is not None:
        return [dict(row) for row in rows]
    if not path.exists():
        warnings.append(f"{module_name} 暂无输出文件，已按空数据降级。")
        return []
    try:
        return pd.read_csv(path, dtype=str).fillna("").to_dict("records")
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"{module_name} 输出读取失败，已按空数据降级：{exc}")
        return []


def _resolve_historical_rows(
    rows: Sequence[Mapping[str, Any]] | None,
    out_dir: Path,
    warnings: list[str],
    fallback_reasons: list[str],
) -> list[dict[str, Any]]:
    if rows is not None:
        return [dict(row) for row in rows]
    candidates = (
        out_dir / HISTORICAL_ML_SUGGESTIONS_FILE,
        out_dir.parent / HISTORICAL_ML_SUGGESTIONS_FILE,
        out_dir / "entry_calibration_suggestions.csv",
        out_dir / "historical_ml_summary.csv",
        out_dir.parent / "artifacts" / "historical_ml_61" / "entry_calibration_suggestions.csv",
        out_dir.parent / "historical_ml" / "output" / "entry_calibration_suggestions.csv",
        out_dir.parent / "historical_ml" / "artifacts" / "entry_calibration_suggestions.csv",
    )
    seen: set[Path] = set()
    for path in candidates:
        resolved = path if path.is_absolute() else Path.cwd() / path
        try:
            resolved = resolved.resolve()
        except OSError:
            pass
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            try:
                return pd.read_csv(resolved, dtype=str).fillna("").to_dict("records")
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"historical_ml 摘要读取失败，已降级为空建议：{exc}")
                fallback_reasons.append("historical_ml 摘要读取失败，总控不中断。")
                return []
    warnings.append("historical_ml 摘要缺失，已降级为空建议。")
    return []


def _build_ml_funnel_counts(
    *,
    out_dir: Path,
    pre_rows: Sequence[Mapping[str, Any]],
    entry_rows: Sequence[Mapping[str, Any]],
    entry_actions: Sequence[Mapping[str, Any]],
    candidate_etfs: Sequence[Mapping[str, Any]],
    order_intents: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    coverage = _load_ml_coverage_report(out_dir)
    pre_symbols = {_symbol(row.get("symbol") or row.get("etf_code") or row.get("code")) for row in pre_rows}
    pre_symbols.discard("")
    entry_symbols = {_symbol(row.get("symbol") or row.get("etf_code") or row.get("code")) for row in entry_rows}
    entry_symbols.discard("")
    entry_pool_count = len(entry_symbols) or len(entry_actions) or len(candidate_etfs)
    ml_direct_hit = sum(1 for item in entry_actions if _has_ml_score(item))
    missing_count = max(entry_pool_count - ml_direct_hit, 0)
    missing_distribution = coverage.get("ml_score_missing_reason_distribution")
    if not isinstance(missing_distribution, Mapping) or not missing_distribution:
        missing_distribution = {"missing_ml_score": missing_count} if missing_count else {"none": 0}

    return {
        "all_market_valid_etf_count": _count_from_coverage(coverage, "all_market_valid_etf_count", len(pre_symbols) or entry_pool_count),
        "historical_price_covered_etf_count": _count_from_coverage(coverage, "historical_price_covered_etf_count", ml_direct_hit),
        "ml_feature_ready_etf_count": _count_from_coverage(coverage, "ml_feature_ready_etf_count", ml_direct_hit),
        "ml_scored_etf_count": _count_from_coverage(coverage, "ml_scored_etf_count", ml_direct_hit),
        "ml_score_direct_hit_count": _count_from_coverage(coverage, "ml_score_direct_hit_count", ml_direct_hit),
        "ml_score_missing_count": _count_from_coverage(coverage, "ml_score_missing_count", missing_count),
        "ml_score_missing_reason_distribution": {str(k): int(_number(v, 0)) for k, v in dict(missing_distribution).items()},
        "broad_recall_pool_count": _count_from_coverage(coverage, "broad_recall_pool_count", sum(1 for row in pre_rows if _bool(row.get("broad_recall_selected")))),
        "ml_recovered_pool_count": _count_from_coverage(
            coverage,
            "ml_recovered_pool_count",
            sum(1 for row in pre_rows if _bool(row.get("ml_recovered")))
            + sum(1 for row in entry_actions if str(row.get("ml_adjustment") or "").upper() == "ML_RECOVERED"),
        ),
        "entry_candidate_pool_count": _count_from_coverage(coverage, "entry_candidate_pool_count", entry_pool_count),
        "order_intent_count": _count_from_coverage(coverage, "order_intent_count", len(order_intents)),
    }


def _load_ml_coverage_report(out_dir: Path) -> dict[str, Any]:
    candidates = (
        out_dir / "historical_ml_universe_coverage_report.json",
        out_dir / HISTORICAL_ML_COVERAGE_REPORT_FILE,
        out_dir.parent / HISTORICAL_ML_COVERAGE_REPORT_FILE,
    )
    seen: set[Path] = set()
    for path in candidates:
        resolved = path if path.is_absolute() else Path.cwd() / path
        try:
            resolved = resolved.resolve()
        except OSError:
            pass
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            continue
        try:
            payload = json.loads(resolved.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(payload, Mapping):
            return dict(payload)
    return {}


def _count_from_coverage(coverage: Mapping[str, Any], key: str, default: int) -> int:
    value = coverage.get(key)
    if value in (None, ""):
        return int(default)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return int(default)


def _has_ml_score(item: Mapping[str, Any]) -> bool:
    value = item.get("ml_score")
    return value not in (None, "") and str(value).strip() != ""


def _resolve_holdings(
    holdings: Sequence[Mapping[str, Any]] | None,
    warnings: list[str],
    fallback_reasons: list[str],
) -> list[dict[str, Any]]:
    if holdings is not None:
        return [dict(item) for item in holdings]
    path = Path("config") / "current_position.yaml"
    if not path.exists():
        warnings.append("当前持仓文件缺失，PortfolioSnapshot 按空持仓输出。")
        fallback_reasons.append("当前持仓文件缺失，持仓页和订单意图按空持仓降级。")
        return []
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if payload.get("current_empty"):
            return []
        return [dict(item) for item in payload.get("holdings", []) or [] if isinstance(item, Mapping)]
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"当前持仓读取失败，PortfolioSnapshot 按空持仓输出：{exc}")
        fallback_reasons.append("当前持仓读取失败，持仓相关输出已降级。")
        return []


def _resolve_qmt_status(
    qmt_execution_available: bool | None,
    qmt_status: Mapping[str, Any] | None,
) -> tuple[bool, str]:
    if qmt_execution_available is not None:
        return bool(qmt_execution_available), "" if qmt_execution_available else "qmt_execution 缺失或不可用，仅生成订单草稿和人工确认说明。"
    if qmt_status:
        return True, ""
    snapshot = Path("runtime") / "qmt_execution" / "qmt_readonly_snapshot.json"
    if snapshot.exists():
        return True, ""
    return False, "qmt_execution 只读快照缺失，总控不提交订单，仅输出 DRAFT/MANUAL_CONFIRM 草稿。"


def _build_risk_gate(
    raw_gate: Mapping[str, Any] | Any | None,
    out_dir: Path,
    trade_date: str,
    warnings: list[str],
    fallback_reasons: list[str],
) -> V21RiskGate:
    payload: dict[str, Any] = {}
    source = "risk_warning"
    if raw_gate is not None:
        payload = raw_gate.to_dict() if hasattr(raw_gate, "to_dict") else dict(raw_gate)
        source = str(payload.get("source") or source)
    elif (out_dir / "risk_gate.json").exists():
        try:
            payload = json.loads((out_dir / "risk_gate.json").read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"RiskGate JSON 读取失败，已按 R0 降级：{exc}")
            fallback_reasons.append("RiskGate 读取失败，总控按 R0 降级但保留 warning。")
    elif (out_dir / "risk_warning_next_day.csv").exists():
        rows = _rows_or_csv(None, out_dir / "risk_warning_next_day.csv", warnings, "risk_warning")
        payload = rows[0] if rows else {}
    else:
        warnings.append("risk_warning 输出缺失，总控按 R0 保守降级。")
        fallback_reasons.append("risk_warning 输出缺失，RiskGate 按 R0 降级。")

    risk_events = payload.get("risk_events") or payload.get("active_events") or []
    if isinstance(risk_events, str):
        try:
            risk_events = json.loads(risk_events)
        except json.JSONDecodeError:
            risk_events = []
    affected_etfs = _unique(
        asset
        for event in risk_events if isinstance(event, Mapping)
        for asset in _as_list(event.get("affected_assets") or event.get("affected_etfs"))
    )
    level = str(payload.get("risk_level") or "R0").upper()
    freeze = _bool(payload.get("freeze_entry")) or level in RISK_FREEZE_LEVELS
    manual = _bool(payload.get("manual_takeover_required")) or level in {"R4", "P0"}
    return V21RiskGate(
        trade_date=str(payload.get("trade_date") or payload.get("risk_date") or trade_date),
        risk_level=level,
        risk_score=int(_number(payload.get("risk_score"), 0)),
        freeze_entry=freeze,
        equity_cap_override=float(_number(payload.get("equity_cap_override"), 0.0 if freeze else 1.0)),
        manual_takeover_required=manual,
        affected_sectors=_as_list(payload.get("affected_sectors")),
        affected_etfs=affected_etfs,
        risk_events=[dict(item) for item in risk_events if isinstance(item, Mapping)],
        explain=str(payload.get("explain") or "风险门控无详细说明，按当前风险等级执行。"),
        source=source,
    )


def _build_entry_actions(
    entry_rows: Sequence[Mapping[str, Any]],
    selected_symbols: set[str],
    risk: V21RiskGate,
    exit_block: Mapping[str, Any],
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in entry_rows:
        symbol = _symbol(row.get("symbol") or row.get("etf_code") or row.get("code"))
        action = str(row.get("buy_action") or row.get("entry_action") or row.get("action") or "")
        target_weight = _ratio(row.get("position_size") or row.get("target_weight") or row.get("suggested_weight"))
        raw_action = _entry_intent_label(row.get("raw_entry_action") or action)
        rule_action = _entry_intent_label(row.get("rule_action") or raw_action)
        ml_adjusted_action = _entry_intent_label(row.get("ml_adjusted_action") or row.get("final_buy_action") or rule_action)
        ml_decision_mode = str(row.get("ml_decision_mode") or "shadow").strip().lower()
        raw_target_weight = _ratio(row.get("raw_entry_target_weight") if row.get("raw_entry_target_weight") not in (None, "") else target_weight)
        raw_confidence = _number(row.get("raw_entry_confidence"), _number(row.get("confidence"), ""))
        raw_reason = str(row.get("raw_entry_reason") or row.get("entry_reason") or row.get("explain") or "")
        raw_block_reason = str(row.get("raw_entry_block_reason") or (raw_reason if raw_action not in {"BUY", "PROBE"} else ""))
        row_final_action = _entry_intent_label(row.get("final_buy_action") or ml_adjusted_action or raw_action)
        row_final_weight = _ratio(row.get("final_target_weight") if row.get("final_target_weight") not in (None, "") else target_weight)
        active_sim_buy = ml_decision_mode == "active_sim" and row_final_action in {"BUY", "PROBE"} and row_final_weight > 0
        intended_buy = (symbol in selected_symbols and raw_action in {"BUY", "PROBE"} and raw_target_weight > 0) or active_sim_buy
        risk_gate_blocked = _bool(row.get("risk_gate_blocked")) or (
            intended_buy and (bool(risk.freeze_entry) or bool(risk.manual_takeover_required))
        )
        exit_priority_blocked = intended_buy and _bool(exit_block.get("exit_priority_blocked"))
        final_block_reason = str(row.get("final_block_reason") or row.get("block_reason") or "")
        control_override_reason = str(row.get("control_override_reason") or "")
        if risk_gate_blocked:
            final_block_reason = final_block_reason or (
                "RiskGate 要求人工接管，entry 不进入实际买入。"
                if risk.manual_takeover_required
                else "RiskGate 冻结买入，entry 不进入实际买入。"
            )
            control_override_reason = control_override_reason or final_block_reason
        elif exit_priority_blocked:
            exit_reason = str(exit_block.get("exit_block_reason") or "exit 优先处理，暂停新增买入")
            release_condition = str(exit_block.get("exit_block_release_condition") or "")
            final_block_reason = final_block_reason or (
                f"{exit_reason}；解除条件：{release_condition}" if release_condition else exit_reason
            )
            control_override_reason = control_override_reason or final_block_reason

        if intended_buy and (risk_gate_blocked or exit_priority_blocked):
            final_action = "BLOCKED"
            final_target_weight = 0.0
            actionable = False
        elif intended_buy:
            final_action = row_final_action
            if final_action not in {"BUY", "PROBE"}:
                if not (ml_decision_mode == "active_sim" and str(row.get("ml_adjustment") or "").startswith("ML_DOWNGRADED")):
                    final_action = raw_action
            final_target_weight = row_final_weight
            actionable = final_action in {"BUY", "PROBE"} and final_target_weight > 0
        else:
            final_action = row_final_action
            if final_action in {"BUY", "PROBE", "BLOCKED"} and not final_block_reason:
                final_action = raw_action if raw_action in {"OBSERVE", "REJECT", "AVOID"} else "OBSERVE"
            final_target_weight = 0.0
            actionable = False
        actions.append(
            {
                "etf_code": symbol,
                "etf_name": str(row.get("name") or row.get("etf_name") or ""),
                "entry_action": action,
                "target_weight": target_weight,
                "confidence": _number(row.get("confidence"), ""),
                "raw_entry_action": raw_action,
                "raw_entry_target_weight": raw_target_weight,
                "raw_entry_confidence": raw_confidence,
                "raw_entry_reason": raw_reason,
                "raw_entry_block_reason": raw_block_reason,
                "rule_action": rule_action,
                "ml_adjusted_action": ml_adjusted_action,
                "final_buy_action": final_action,
                "final_target_weight": final_target_weight,
                "final_block_reason": final_block_reason,
                "expected_open_gap_pct": row.get("expected_open_gap_pct", ""),
                "expected_open_gap": row.get("expected_open_gap", ""),
                "open_gap_pct": row.get("open_gap_pct", ""),
                "next_open_gap_pct": row.get("next_open_gap_pct", ""),
                "control_override_reason": control_override_reason,
                "active_exit_count": int(_number(exit_block.get("active_exit_count"), 0)),
                "actual_position_exit_count": int(_number(exit_block.get("actual_position_exit_count"), 0)),
                "exit_priority_blocked": exit_priority_blocked,
                "exit_block_reason": str(exit_block.get("exit_block_reason") or ""),
                "exit_block_release_condition": str(exit_block.get("exit_block_release_condition") or ""),
                "blocked_by_exit_symbols": list(exit_block.get("blocked_by_exit_symbols") or []),
                "has_real_position_to_exit": bool(exit_block.get("has_real_position_to_exit")),
                "exit_action_type": str(exit_block.get("exit_action_type") or ""),
                "risk_gate_blocked": risk_gate_blocked,
                "ml_score": row.get("ml_score", ""),
                "p_good_entry": row.get("p_good_entry", ""),
                "p_bad_entry": row.get("p_bad_entry", ""),
                "ml_entry_advice": str(row.get("ml_entry_advice") or "无ML建议"),
                "ml_confidence": _number(row.get("ml_confidence"), 0),
                "ml_reason": str(row.get("ml_reason") or "未找到历史校准建议，维持原 entry 判断。"),
                "ml_action_suggestion": str(row.get("ml_action_suggestion") or "NO_ML"),
                "ml_decision_mode": ml_decision_mode,
                "ml_adjustment": str(row.get("ml_adjustment") or ""),
                "ml_adjustment_reason_cn": str(row.get("ml_adjustment_reason_cn") or ""),
                "ml_observation_notice": ML_OBSERVATION_NOTICE,
                "pre_selected": symbol in selected_symbols,
                "intended_buy": intended_buy,
                "actual_buy": actionable,
                "block_reason": final_block_reason,
                "explain": str(row.get("entry_reason") or row.get("explain") or "entry 输出无额外说明。"),
                "source_signal": str(row.get("source_file") or "entry_signal.csv"),
            }
        )
    return actions


def _ml_advice_active(row: Mapping[str, Any]) -> bool:
    action = str(row.get("ml_action_suggestion") or "").strip().upper()
    advice = str(row.get("ml_entry_advice") or "").strip()
    try:
        confidence = float(row.get("ml_confidence") or 0)
    except (TypeError, ValueError):
        confidence = 0.0
    return bool((action and action != "NO_ML") or confidence > 0 or (advice and advice != "无ML建议"))


def _ml_observation_status(entry_rows: Sequence[Mapping[str, Any]]) -> str:
    if not entry_rows:
        return f"ML 观察模式未启用（无 entry 输出；{ML_OBSERVATION_NOTICE}）"
    if not all("ml_entry_advice" in row for row in entry_rows):
        return f"ML 观察模式未启用（entry 输出缺少 ML 字段；{ML_OBSERVATION_NOTICE}）"
    if any(_ml_advice_active(row) for row in entry_rows):
        return f"ML 观察模式已启用（{ML_OBSERVATION_NOTICE}）"
    return f"ML 观察模式已启用（当前无ML建议，维持原 entry 判断；{ML_OBSERVATION_NOTICE}）"


def _ml_entry_advice_summary(entry_rows: Sequence[Mapping[str, Any]], selected_symbols: set[str]) -> str:
    items: list[str] = []
    for row in entry_rows:
        symbol = _symbol(row.get("symbol") or row.get("etf_code") or row.get("code"))
        if symbol not in selected_symbols:
            continue
        items.append(
            f"{symbol}:{row.get('ml_entry_advice', '无ML建议')}"
            f"（置信度{row.get('ml_confidence', 0)}，动作建议{row.get('ml_action_suggestion', 'NO_ML')}；{ML_OBSERVATION_NOTICE}）"
        )
    return " | ".join(items) if items else f"无ML建议（{ML_OBSERVATION_NOTICE}）"


def _build_ml_sim_outputs(
    *,
    out_dir: Path,
    trade_date: str,
    pre_rows: Sequence[Mapping[str, Any]],
    entry_rows: Sequence[Mapping[str, Any]],
    entry_actions: Sequence[Mapping[str, Any]],
    risk: V21RiskGate,
    exit_block: Mapping[str, Any],
    order_intents: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    pre_by_symbol = {_symbol(row.get("symbol") or row.get("etf_code") or row.get("code")): row for row in pre_rows}
    action_by_symbol = {str(row.get("etf_code") or ""): row for row in entry_actions}
    score_by_symbol = _load_ml_entry_scores(out_dir, trade_date)
    legacy_order_symbols = {
        str(row.get("etf_code") or "")
        for row in order_intents
        if str(row.get("side") or "").upper() == "BUY" and str(row.get("action") or "").upper() != "BLOCKED_BUY"
    }
    risk_level = str(risk.risk_level or "R0").upper()
    global_risk_block = risk_level in RISK_FREEZE_LEVELS or bool(risk.freeze_entry) or bool(risk.manual_takeover_required)
    global_exit_block = _bool(exit_block.get("exit_priority_blocked"))

    rows: list[dict[str, Any]] = []
    for entry_row in entry_rows:
        symbol = _symbol(entry_row.get("symbol") or entry_row.get("etf_code") or entry_row.get("code"))
        if not symbol:
            continue
        pre_row = pre_by_symbol.get(symbol, {})
        action_row = action_by_symbol.get(symbol, {})
        score_row = score_by_symbol.get(symbol, {})
        merged = {**score_row, **dict(entry_row), **action_row}
        legacy_action = _entry_intent_label(
            entry_row.get("rule_action")
            or entry_row.get("raw_entry_action")
            or entry_row.get("buy_action")
            or entry_row.get("entry_action")
        )
        official_final = _entry_intent_label(action_row.get("final_buy_action") or entry_row.get("final_buy_action") or legacy_action)
        ml_sim_action, ml_reason = _ml_sim_action(legacy_action, merged)
        risk_blocked = _bool(action_row.get("risk_gate_blocked")) or global_risk_block
        exit_blocked = _bool(action_row.get("exit_priority_blocked")) or global_exit_block
        adjustment_type = _ml_sim_adjustment_type(
            legacy_action=legacy_action,
            ml_sim_action=ml_sim_action,
            entry_row=merged,
            pre_selected=_truthy(pre_row.get("selected")) or _bool(action_row.get("pre_selected")),
            risk_blocked=risk_blocked,
            exit_blocked=exit_blocked,
        )
        reason = str(merged.get("ml_adjustment_reason_cn") or merged.get("ml_reason_cn") or merged.get("ml_reason") or ml_reason or ML_SIM_NOTICE)
        if adjustment_type == "ML_CONFLICT_WITH_RISK":
            reason = f"{ML_SIM_NOTICE} ML_SIM wants a buy/probe candidate, but RiskGate or exit priority blocks it. {reason}"
        elif adjustment_type == "ML_MISSING_SCORE":
            reason = f"{ML_SIM_NOTICE} No usable ml_entry_scores row; legacy_v21 remains unchanged."
        else:
            reason = f"{ML_SIM_NOTICE} {reason}"
        row = {
            "trade_date": trade_date,
            "code": symbol,
            "name": str(merged.get("etf_name") or merged.get("name") or pre_row.get("name") or ""),
            "sector_level1": str(pre_row.get("sector_level1") or pre_row.get("sector_l1") or pre_row.get("sector") or ""),
            "sector_level2": str(pre_row.get("sector_level2") or pre_row.get("sector") or ""),
            "legacy_action": legacy_action,
            "ml_sim_action": ml_sim_action,
            "final_action": official_final,
            "ml_score": merged.get("ml_score", ""),
            "p_good_entry": merged.get("p_good_entry", ""),
            "p_bad_entry": merged.get("p_bad_entry", ""),
            "ml_adjustment_type": adjustment_type,
            "ml_adjustment_reason_cn": reason,
            "risk_level": risk_level,
            "risk_blocked": risk_blocked,
            "exit_blocked": exit_blocked,
            "order_intent_in_legacy": symbol in legacy_order_symbols,
            "order_intent_in_ml_sim": _ml_sim_order_intent_flag(ml_sim_action, risk_blocked, exit_blocked),
            "review_priority": _ml_sim_review_priority(adjustment_type, merged),
            "future_return_1d": "",
            "future_return_3d": "",
            "future_return_5d": "",
            "future_return_10d": "",
            "future_max_drawdown_10d": "",
            "outperform_market_10d": "",
            "outperform_sector_10d": "",
        }
        rows.append(row)

    rows = sorted(rows, key=lambda item: (str(item["review_priority"]), -_number(item.get("ml_score"), -999999), str(item["code"])))
    review_queue = [
        row
        for row in rows
        if row["ml_adjustment_type"] not in {"ML_UNCHANGED"}
        or row["review_priority"] in {"P0", "P1"}
    ]
    summary = _ml_sim_summary(trade_date, rows, review_queue)
    return rows, summary, review_queue


def _load_ml_entry_scores(out_dir: Path, trade_date: str) -> dict[str, dict[str, Any]]:
    candidates = (
        out_dir / "ml_entry_scores.csv",
        out_dir / HISTORICAL_ML_ENTRY_SCORES_FILE,
        out_dir.parent / HISTORICAL_ML_ENTRY_SCORES_FILE,
    )
    seen: set[Path] = set()
    for path in candidates:
        resolved = path if path.is_absolute() else Path.cwd() / path
        try:
            resolved = resolved.resolve()
        except OSError:
            pass
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.exists():
            continue
        try:
            frame = pd.read_csv(resolved, dtype=str).fillna("")
        except Exception:
            continue
        if "trade_date" in frame.columns:
            dated = frame.loc[frame["trade_date"].astype(str).str.slice(0, 10).eq(str(trade_date)[:10])]
            if not dated.empty:
                frame = dated
        return {
            _symbol(row.get("code") or row.get("etf_code") or row.get("symbol")): dict(row)
            for row in frame.to_dict("records")
            if _symbol(row.get("code") or row.get("etf_code") or row.get("symbol"))
        }
    return {}


def _ml_sim_action(legacy_action: str, row: Mapping[str, Any]) -> tuple[str, str]:
    suggestion = str(row.get("ml_action_suggestion") or "NO_ML").strip().upper()
    if suggestion == "UPGRADE_PROBE" and legacy_action not in {"BUY", "PROBE"}:
        return "PROBE", "ML score suggests recovering this ETF into a simulation-only probe candidate."
    if suggestion in {"DOWNGRADE_WATCH", "WAIT_PULLBACK"} and legacy_action in {"BUY", "PROBE"}:
        return "OBSERVE", "ML score suggests downgrading a legacy buy/probe to observation in simulation."
    if suggestion == "FORBID_CHASE" and legacy_action in {"BUY", "PROBE"}:
        return "AVOID", "ML score warns against chasing this legacy buy/probe in simulation."
    return legacy_action, "ML_SIM keeps the legacy_v21 action."


def _ml_sim_adjustment_type(
    *,
    legacy_action: str,
    ml_sim_action: str,
    entry_row: Mapping[str, Any],
    pre_selected: bool,
    risk_blocked: bool,
    exit_blocked: bool,
) -> str:
    if ml_sim_action in {"BUY", "PROBE"} and (risk_blocked or exit_blocked):
        return "ML_CONFLICT_WITH_RISK"
    if not _has_ml_score(entry_row):
        return "ML_MISSING_SCORE"
    suggestion = str(entry_row.get("ml_action_suggestion") or "").strip().upper()
    if suggestion == "UPGRADE_PROBE" and legacy_action not in {"BUY", "PROBE"}:
        return "ML_RECOVERED" if pre_selected else "ML_UPGRADED_TO_BUY_CANDIDATE"
    if suggestion == "FORBID_CHASE" and legacy_action in {"BUY", "PROBE"}:
        return "ML_FILTERED_BAD_ENTRY"
    if suggestion in {"DOWNGRADE_WATCH", "WAIT_PULLBACK"} and legacy_action in {"BUY", "PROBE"}:
        return "ML_DOWNGRADED"
    if ml_sim_action != legacy_action:
        return "ML_DOWNGRADED" if legacy_action in {"BUY", "PROBE"} else "ML_RECOVERED"
    return "ML_UNCHANGED"


def _ml_sim_order_intent_flag(action: str, risk_blocked: bool, exit_blocked: bool) -> bool:
    return action in {"BUY", "PROBE"} and not risk_blocked and not exit_blocked


def _ml_sim_review_priority(adjustment_type: str, row: Mapping[str, Any]) -> str:
    if adjustment_type == "ML_CONFLICT_WITH_RISK":
        return "P0"
    if adjustment_type in {"ML_RECOVERED", "ML_UPGRADED_TO_BUY_CANDIDATE", "ML_DOWNGRADED", "ML_FILTERED_BAD_ENTRY"}:
        return "P1"
    if adjustment_type == "ML_MISSING_SCORE":
        return "P2"
    score = _number(row.get("ml_score"), 0.0)
    return "P2" if abs(score) >= 40 else "P3"


def _ml_sim_summary(trade_date: str, rows: Sequence[Mapping[str, Any]], review_queue: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in rows:
        key = str(row.get("ml_adjustment_type") or "ML_UNCHANGED")
        counts[key] = counts.get(key, 0) + 1
    recovered = [
        dict(row)
        for row in sorted(rows, key=lambda item: _number(item.get("ml_score"), -999999), reverse=True)
        if row.get("ml_adjustment_type") in {"ML_RECOVERED", "ML_UPGRADED_TO_BUY_CANDIDATE"}
    ][:10]
    downgraded = [
        dict(row)
        for row in sorted(rows, key=lambda item: _number(item.get("p_bad_entry"), 0), reverse=True)
        if row.get("ml_adjustment_type") in {"ML_DOWNGRADED", "ML_FILTERED_BAD_ENTRY"}
    ][:10]
    return {
        "trade_date": trade_date,
        "mode": "V2.1_ML_SIM",
        "observation_notice": ML_SIM_NOTICE,
        "total_rows": len(rows),
        "review_queue_count": len(review_queue),
        "ml_recovered_count": counts.get("ML_RECOVERED", 0) + counts.get("ML_UPGRADED_TO_BUY_CANDIDATE", 0),
        "ml_downgraded_count": counts.get("ML_DOWNGRADED", 0) + counts.get("ML_FILTERED_BAD_ENTRY", 0),
        "adjustment_counts": counts,
        "top_ml_recovered": recovered,
        "top_ml_downgraded": downgraded,
        "official_decision_policy": "legacy_v21/current control_center safety remains final; ML_SIM does not trigger QMT.",
    }


def _build_exit_actions(exit_rows: Sequence[Mapping[str, Any]], portfolio: Sequence[PortfolioSnapshot]) -> list[dict[str, Any]]:
    portfolio_by_symbol = {item.etf_code: item for item in portfolio}
    actions: list[dict[str, Any]] = []
    for row in exit_rows:
        symbol = _symbol(row.get("symbol") or row.get("etf_code") or row.get("code"))
        holding = portfolio_by_symbol.get(symbol)
        action = str(row.get("sell_action") or row.get("exit_action") or row.get("action") or "")
        reduce_ratio = _ratio(row.get("reduce_ratio") or row.get("delta_weight"))
        active_exit = _is_exit_action(action, reduce_ratio)
        high_priority_exit = _is_high_priority_exit(row)
        has_real_position = _portfolio_has_real_position(holding)
        actionable = active_exit and has_real_position
        actual_position_exit = actionable
        priority_blocking = actual_position_exit and high_priority_exit
        explain = str(row.get("exit_reason") or row.get("explain") or "exit 输出无额外说明。")
        action_type = _exit_action_type(action, explain, reduce_ratio)
        actions.append(
            {
                "etf_code": symbol,
                "etf_name": str(row.get("name") or row.get("etf_name") or ""),
                "exit_action": action,
                "exit_action_type": action_type,
                "reduce_ratio": reduce_ratio,
                "active_exit": active_exit,
                "high_priority_exit": high_priority_exit,
                "has_real_position_to_exit": has_real_position,
                "actual_position_exit": actual_position_exit,
                "priority_blocking_exit": priority_blocking,
                "actual_exit": actionable,
                "exit_block_reason": _exit_item_reason(symbol, row, action_type, explain) if priority_blocking else "",
                "exit_block_release_condition": _exit_release_condition(symbol) if priority_blocking else "",
                "explain": explain,
                "source_signal": str(row.get("source_file") or "exit_signal.csv"),
            }
        )
    return actions


def _build_exit_priority_block(exit_actions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    active = [item for item in exit_actions if _bool(item.get("active_exit"))]
    actual = [item for item in active if _bool(item.get("actual_position_exit"))]
    blocking = [item for item in actual if _bool(item.get("priority_blocking_exit"))]
    symbols = [str(item.get("etf_code") or "") for item in blocking if str(item.get("etf_code") or "")]
    reasons = [str(item.get("exit_block_reason") or "") for item in blocking if str(item.get("exit_block_reason") or "")]
    releases = [str(item.get("exit_block_release_condition") or "") for item in blocking if str(item.get("exit_block_release_condition") or "")]
    action_types = _unique(item.get("exit_action_type") for item in blocking)
    return {
        "active_exit_count": len(active),
        "actual_position_exit_count": len(actual),
        "exit_priority_blocked": bool(blocking),
        "exit_block_reason": "；".join(reasons),
        "exit_block_release_condition": "；".join(releases),
        "blocked_by_exit_symbols": symbols,
        "has_real_position_to_exit": bool(actual),
        "exit_action_type": " | ".join(action_types),
    }


def _build_portfolio_snapshot(
    *,
    holdings: Sequence[Mapping[str, Any]],
    exit_rows: Sequence[Mapping[str, Any]],
    pre_rows: Sequence[Mapping[str, Any]],
    trade_date: str,
    account_total_asset: float | None,
) -> list[PortfolioSnapshot]:
    exit_by_symbol = {_symbol(row.get("symbol") or row.get("etf_code") or row.get("code")): row for row in exit_rows}
    pre_by_symbol = {_symbol(row.get("symbol") or row.get("etf_code") or row.get("code")): row for row in pre_rows}
    values: list[float] = []
    normalized: list[dict[str, Any]] = []
    for item in holdings:
        symbol = _symbol(item.get("symbol") or item.get("etf_code") or item.get("code"))
        shares = _number(item.get("shares") or item.get("quantity"), 0.0)
        cost = _number(item.get("cost_price") or item.get("average_buy_price"), 0.0)
        exit_row = exit_by_symbol.get(symbol, {})
        current_price = _number(item.get("current_price") or item.get("last_price") or exit_row.get("sell_price") or cost, cost)
        value = shares * current_price if shares and current_price else 0.0
        values.append(value)
        normalized.append({**dict(item), "_symbol": symbol, "_shares": shares, "_cost": cost, "_current_price": current_price, "_value": value})
    total_asset = account_total_asset if account_total_asset and account_total_asset > 0 else sum(values)
    snapshots: list[PortfolioSnapshot] = []
    for item in normalized:
        symbol = item["_symbol"]
        exit_row = exit_by_symbol.get(symbol, {})
        pre_row = pre_by_symbol.get(symbol, {})
        current_price = item["_current_price"]
        cost = item["_cost"]
        shares = item["_shares"]
        value = item["_value"]
        pnl = (current_price - cost) * shares if current_price and cost and shares else ""
        pnl_pct = (current_price / cost - 1.0) if current_price and cost else ""
        current_weight = value / total_asset if total_asset else _ratio(item.get("current_weight"))
        exit_action = str(exit_row.get("sell_action") or exit_row.get("exit_action") or "")
        snapshots.append(
            PortfolioSnapshot(
                trade_date=trade_date,
                etf_code=symbol,
                etf_name=str(item.get("name") or item.get("etf_name") or exit_row.get("name") or symbol),
                current_weight=round(current_weight, 6),
                target_weight=_ratio(item.get("target_weight")),
                cost_price=cost or "",
                current_price=current_price or "",
                pnl=round(pnl, 4) if isinstance(pnl, (int, float)) else "",
                pnl_pct=round(pnl_pct, 6) if isinstance(pnl_pct, (int, float)) else "",
                holding_days=item.get("holding_days") or item.get("holding_days_count") or "",
                sector=str(item.get("sector") or pre_row.get("sector") or ""),
                risk_status="需要退出关注" if _is_exit_action(exit_action, _ratio(exit_row.get("reduce_ratio"))) else "正常跟踪",
                exit_action=exit_action,
                explain=str(exit_row.get("exit_reason") or "持仓纳入 V2.1 总控快照。"),
            )
        )
    return snapshots


def _build_portfolio_actions(
    portfolio: Sequence[PortfolioSnapshot],
    exit_actions: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    exit_by_symbol = {item.get("etf_code"): item for item in exit_actions}
    actions: list[dict[str, Any]] = []
    for item in portfolio:
        exit_action = exit_by_symbol.get(item.etf_code, {})
        actions.append(
            {
                "etf_code": item.etf_code,
                "etf_name": item.etf_name,
                "current_weight": item.current_weight,
                "target_weight": item.target_weight,
                "action": exit_action.get("exit_action") or "继续持仓跟踪",
                "explain": exit_action.get("explain") or item.explain,
            }
        )
    return actions


def _build_order_intents(
    *,
    trade_date: str,
    entry_actions: Sequence[Mapping[str, Any]],
    exit_actions: Sequence[Mapping[str, Any]],
    portfolio: Sequence[PortfolioSnapshot],
    risk: V21RiskGate,
    qmt_available: bool,
    qmt_note: str,
) -> list[dict[str, Any]]:
    portfolio_by_symbol = {item.etf_code: item for item in portfolio}
    intents: list[dict[str, Any]] = []
    manual = True
    risk_block = ""
    if risk.freeze_entry:
        risk_block = "RiskGate 冻结买入，买入侧不得生成可执行订单。"
    if risk.manual_takeover_required:
        risk_block = "RiskGate 要求人工接管，所有订单意图必须人工确认。"

    for exit_action in exit_actions:
        if not exit_action.get("actual_exit"):
            continue
        symbol = str(exit_action.get("etf_code") or "")
        holding = portfolio_by_symbol.get(symbol)
        current_weight = holding.current_weight if holding else 0.0
        reduce_ratio = _ratio(exit_action.get("reduce_ratio")) or 1.0
        delta = -min(current_weight, current_weight * reduce_ratio if current_weight else reduce_ratio)
        intents.append(
            V21OrderIntent(
                trade_date=trade_date,
                etf_code=symbol,
                etf_name=str(exit_action.get("etf_name") or (holding.etf_name if holding else "")),
                action="DRAFT_EXIT",
                side="SELL",
                target_weight=max(current_weight + delta, 0.0),
                current_weight=current_weight,
                delta_weight=round(delta, 6),
                estimated_price=holding.current_price if holding else "",
                estimated_amount="",
                order_type="LIMIT",
                execution_mode="MANUAL_CONFIRM",
                requires_manual_confirm=manual,
                risk_check_passed=True,
                risk_block_reason="",
                source_signal=str(exit_action.get("source_signal") or "exit_signal.csv"),
                explain=f"exit 优先生成卖出草稿。{exit_action.get('explain') or ''}",
            ).to_dict()
        )

    for entry in entry_actions:
        symbol = str(entry.get("etf_code") or "")
        holding = portfolio_by_symbol.get(symbol)
        current_weight = holding.current_weight if holding else 0.0
        target_weight = _ratio(entry.get("final_target_weight") if entry.get("final_target_weight") not in (None, "") else entry.get("target_weight"))
        passed = bool(entry.get("actual_buy")) and not risk_block
        block_reason = str(entry.get("block_reason") or risk_block or "")
        if not entry.get("intended_buy") and not passed:
            continue
        if not passed and not block_reason:
            continue
        execution_mode = "DRAFT" if passed else "MANUAL_CONFIRM"
        if execution_mode not in SAFE_EXECUTION_MODES:
            execution_mode = "DRAFT"
        explain = str(entry.get("explain") or "")
        if not qmt_available:
            explain = f"{explain} qmt_execution 不可用，当前仅保留订单草稿和人工确认说明。".strip()
        intents.append(
            V21OrderIntent(
                trade_date=trade_date,
                etf_code=symbol,
                etf_name=str(entry.get("etf_name") or ""),
                action="DRAFT_BUY" if passed else "BLOCKED_BUY",
                side="BUY",
                target_weight=target_weight if passed else current_weight,
                current_weight=current_weight,
                delta_weight=round(max(target_weight - current_weight, 0.0), 6) if passed else 0.0,
                estimated_price="",
                estimated_amount="",
                order_type="LIMIT",
                execution_mode=execution_mode,
                requires_manual_confirm=True,
                risk_check_passed=passed,
                risk_block_reason=block_reason,
                source_signal=str(entry.get("source_signal") or "entry_signal.csv"),
                explain=explain or qmt_note or "买入订单意图由 V2.1 总控草稿生成。",
            ).to_dict()
        )
    if not intents and qmt_note:
        intents.append(
            V21OrderIntent(
                trade_date=trade_date,
                etf_code="",
                action="NO_ORDER",
                side="",
                execution_mode="DRAFT",
                requires_manual_confirm=True,
                risk_check_passed=False,
                risk_block_reason=qmt_note,
                source_signal="v21_orchestrator",
                explain="没有可执行买卖动作，总控仅写出 QMT 降级说明。",
            ).to_dict()
        )
    return intents


def _learning_sample(
    row: Mapping[str, Any],
    entry_rows: Sequence[Mapping[str, Any]],
    exit_rows: Sequence[Mapping[str, Any]],
) -> TrainingSample:
    symbol = _symbol(row.get("symbol") or row.get("etf_code") or row.get("code"))
    entry = _find_symbol(entry_rows, symbol)
    exit_row = _find_symbol(exit_rows, symbol)
    return TrainingSample(
        trade_date=str(row.get("trade_date") or ""),
        etf_code=symbol,
        etf_name=str(row.get("name") or row.get("etf_name") or ""),
        signal_type="learning",
        market_state=str(entry.get("market_state") or exit_row.get("market_state") or row.get("market_state") or ""),
        sector=str(row.get("sector") or ""),
        entry_action=str(entry.get("buy_action") or row.get("entry_action") or ""),
        exit_action=str(exit_row.get("sell_action") or row.get("exit_action") or ""),
        confidence=entry.get("confidence") or row.get("confidence") or "",
        ml_entry_advice=str(entry.get("ml_entry_advice") or row.get("ml_entry_advice") or "无ML建议"),
        ml_confidence=entry.get("ml_confidence") or row.get("ml_confidence") or 0,
        ml_reason=str(entry.get("ml_reason") or row.get("ml_reason") or "未找到历史校准建议，维持原 entry 判断。"),
        ml_action_suggestion=str(entry.get("ml_action_suggestion") or row.get("ml_action_suggestion") or "NO_ML"),
        trend_maturity=_extract_between(str(entry.get("entry_reason") or ""), "趋势成熟度：", "；"),
        entry_quality=_extract_between(str(entry.get("entry_reason") or ""), "买点质量：", "；"),
        post_924_regime=_post_924(row.get("trade_date")),
        ret_1d=row.get("ret_1d") or "",
        ret_3d=row.get("ret_3d") or "",
        ret_5d=row.get("ret_5d") or "",
        ret_10d=row.get("return_pct") or row.get("ret_10d") or "",
        hindsight_label=str(row.get("hindsight_label") or ""),
        failure_type=str(row.get("failure_attribution") or row.get("failure_type") or ""),
        calibration_suggestion=str(row.get("adjustment") or row.get("calibration_suggestion") or ""),
        explain=str(row.get("lesson") or row.get("explain") or ""),
    )


def _historical_sample(row: Mapping[str, Any]) -> TrainingSample:
    return TrainingSample(
        trade_date=str(row.get("trade_date") or row.get("signal_date") or ""),
        etf_code=_symbol(row.get("etf_code") or row.get("code") or row.get("symbol")),
        etf_name=str(row.get("etf_name") or row.get("name") or ""),
        signal_type=str(row.get("signal_type") or "historical_ml"),
        market_state=str(row.get("market_state") or row.get("affected_market_state") or ""),
        sector=str(row.get("sector") or row.get("affected_sector_state") or ""),
        entry_action=str(row.get("entry_action") or row.get("was_bought") or ""),
        exit_action=str(row.get("exit_action") or ""),
        confidence=row.get("confidence") or "",
        ml_entry_advice=str(row.get("ml_entry_advice") or "无ML建议"),
        ml_confidence=row.get("ml_confidence") or row.get("confidence") or 0,
        ml_reason=str(row.get("ml_reason") or _historical_ml_reason(row)),
        ml_action_suggestion=str(row.get("ml_action_suggestion") or _historical_ml_action(row)),
        trend_maturity=str(row.get("trend_maturity") or ""),
        entry_quality=str(row.get("entry_quality") or row.get("parameter_area") or ""),
        post_924_regime=_post_924(row.get("trade_date") or row.get("signal_date")),
        ret_1d=row.get("ret_1d") or row.get("future_return_1d") or "",
        ret_3d=row.get("ret_3d") or row.get("future_return_3d") or "",
        ret_5d=row.get("ret_5d") or row.get("future_return_5d") or "",
        ret_10d=row.get("ret_10d") or row.get("future_return_10d") or row.get("avg_future_return_10d") or "",
        hindsight_label=str(row.get("hindsight_label") or row.get("auto_label") or ""),
        failure_type=str(row.get("failure_type") or row.get("review_reason") or ""),
        calibration_suggestion=str(row.get("calibration_suggestion") or row.get("suggested_action") or row.get("notes") or ""),
        explain=str(row.get("explain") or row.get("notes") or row.get("suggestion_id") or ""),
    )


def _historical_ml_reason(row: Mapping[str, Any]) -> str:
    if row.get("parameter_area") or row.get("suggested_action"):
        return "historical_ml 输出参数级校准建议；总控只读汇总，不写回 entry 参数，不参与当日交易裁决。"
    return "未找到历史校准建议，维持原 entry 判断。"


def _historical_ml_action(row: Mapping[str, Any]) -> str:
    if row.get("parameter_area") or row.get("suggested_action"):
        return "KEEP_ORIGINAL"
    return "NO_ML"


def _write_table(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field, "")) for field in fields})


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _resolve_trade_date(
    explicit: str | pd.Timestamp | None,
    *sources: Any,
) -> str:
    if explicit is not None:
        return str(pd.Timestamp(explicit).date())
    for source in sources:
        if source is None:
            continue
        if isinstance(source, Mapping):
            value = source.get("trade_date") or source.get("risk_date")
            if value:
                return str(value)[:10]
        if isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
            value = _first_text(source, "trade_date", default="")
            if value:
                return value[:10]
    return datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()


def _decision_explain(
    market_state: str,
    risk: V21RiskGate,
    candidates: Sequence[Mapping[str, Any]],
    buys: Sequence[Mapping[str, Any]],
    exits: Sequence[Mapping[str, Any]],
    exit_block: Mapping[str, Any],
) -> str:
    parts = [
        f"今日市场状态为{market_state or '未知'}。",
        f"风险等级为{risk.risk_level}，风险分数{risk.risk_score}。",
    ]
    if risk.freeze_entry:
        parts.append("风险门控已冻结买入，entry 不得进入实际买入。")
    elif _bool(exit_block.get("exit_priority_blocked")):
        parts.append(f"{exit_block.get('exit_block_reason')}解除条件：{exit_block.get('exit_block_release_condition')}。")
    else:
        parts.append("风险门控未冻结买入，entry 可在候选池内形成订单草稿。")
    parts.append(f"候选 ETF 数量为{len(candidates)}，实际买入建议数量为{len(buys)}，退出动作数量为{len([item for item in exits if item.get('actual_exit')])}。")
    parts.append("learning 与 historical_ml 仅提供复盘和校准建议，不自动修改当日交易参数。")
    return "".join(parts)


def _candidate_payload(row: Mapping[str, Any], entry_row: Mapping[str, Any] | None = None) -> dict[str, Any]:
    entry_row = entry_row or {}
    return {
        "etf_code": _symbol(row.get("symbol") or row.get("etf_code") or row.get("code")),
        "etf_name": str(row.get("name") or row.get("etf_name") or ""),
        "sector": str(row.get("sector") or ""),
        "rank": row.get("rank") or "",
        "score": row.get("score") or "",
        "candidate_pool_flag": _bool(row.get("candidate_pool_flag")) if "candidate_pool_flag" in row else _truthy(row.get("selected")),
        "candidate_source": str(row.get("candidate_source") or ("LEGACY_TOP5" if _truthy(row.get("selected")) else "")),
        "legacy_selected": _bool(row.get("legacy_selected")) if "legacy_selected" in row else _truthy(row.get("selected")),
        "broad_recall_selected": _bool(row.get("broad_recall_selected")),
        "ml_recovered": _bool(row.get("ml_recovered")),
        "candidate_pool_rank": row.get("candidate_pool_rank") or "",
        "ml_entry_advice": str(entry_row.get("ml_entry_advice") or "无ML建议"),
        "ml_confidence": _number(entry_row.get("ml_confidence"), 0),
        "ml_reason": str(entry_row.get("ml_reason") or "未找到历史校准建议，维持原 entry 判断。"),
        "ml_action_suggestion": str(entry_row.get("ml_action_suggestion") or "NO_ML"),
        "ml_score": entry_row.get("ml_score", ""),
        "p_good_entry": entry_row.get("p_good_entry", ""),
        "p_bad_entry": entry_row.get("p_bad_entry", ""),
        "ml_decision_mode": str(entry_row.get("ml_decision_mode") or "shadow"),
        "ml_adjustment": str(entry_row.get("ml_adjustment") or ""),
        "ml_adjustment_reason_cn": str(entry_row.get("ml_adjustment_reason_cn") or ""),
        "rule_action": _entry_intent_label(entry_row.get("rule_action") or entry_row.get("raw_entry_action")),
        "ml_adjusted_action": _entry_intent_label(entry_row.get("ml_adjusted_action") or entry_row.get("final_buy_action")),
        "ml_observation_notice": ML_OBSERVATION_NOTICE,
        "explain": str(row.get("reason") or row.get("explain") or ""),
    }


def _candidate_pool_rows(pre_rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    if any("candidate_pool_flag" in row for row in pre_rows):
        return [row for row in pre_rows if _bool(row.get("candidate_pool_flag"))]
    return [row for row in pre_rows if _truthy(row.get("selected"))]


def _is_buy_action(action: str) -> bool:
    text = action.strip().lower()
    if not text:
        return False
    blocked = ("观察", "等待", "禁止", "冻结", "暂停", "watch", "wait", "forbid", "blocked")
    if any(token in text for token in blocked):
        return False
    return any(token in text for token in ("买入", "加仓", "buy", "probe", "standard", "add"))


def _entry_intent_label(value: Any) -> str:
    text = str(value or "").strip()
    upper = text.upper()
    lower = text.lower()
    if upper in {"BUY", "PROBE", "OBSERVE", "REJECT", "AVOID", "BLOCKED"}:
        return upper
    if "avoid" in lower:
        return "AVOID"
    if any(token in text for token in ("冻结", "阻断", "暂停")) or "blocked" in lower:
        return "BLOCKED"
    if "禁止" in text or "forbid" in lower or "reject" in lower:
        return "REJECT"
    if "试探" in text or "probe" in lower:
        return "PROBE"
    if any(token in text for token in ("标准买入", "加强买入", "加仓", "买入")) or any(
        token in lower for token in ("standard", "add", "buy")
    ):
        return "BUY"
    return "OBSERVE"


def _portfolio_has_real_position(holding: PortfolioSnapshot | None) -> bool:
    if holding is None:
        return False
    return _number(holding.current_weight, 0.0) > 0


def _exit_action_type(action: str, reason: str, reduce_ratio: float) -> str:
    text = f"{action} {reason}".lower()
    if any(token in text for token in ("清仓", "clear")) or reduce_ratio >= 0.999:
        return "清仓退出"
    if any(token in text for token in ("风险退出", "止损", "risk", "stop")):
        return "风险退出"
    if any(token in text for token in ("减仓", "reduce")) or reduce_ratio > 0:
        return "减仓退出"
    return "退出观察"


def _exit_item_reason(symbol: str, row: Mapping[str, Any], action_type: str, reason: str) -> str:
    name = str(row.get("name") or row.get("etf_name") or "")
    label = f"{symbol} {name}".strip()
    return f"exit 优先处理：{label} 存在实际持仓需要{action_type}，原因：{reason}"


def _exit_release_condition(symbol: str) -> str:
    return f"{symbol} 持仓卖出完成且持仓数量/权重归零，或下一次 exit 不再给出清仓/风险退出信号后解除"


def _is_exit_action(action: str, reduce_ratio: float = 0.0) -> bool:
    text = action.strip().lower()
    if not text:
        return reduce_ratio > 0
    if any(token in text for token in ("持有", "观察", "hold", "watch")) and not any(token in text for token in ("减", "卖", "清", "sell", "reduce", "clear", "exit")):
        return False
    return reduce_ratio > 0 or any(token in text for token in ("清仓", "减仓", "卖出", "退出", "止损", "sell", "reduce", "clear", "exit", "stop"))


def _is_high_priority_exit(row: Mapping[str, Any]) -> bool:
    text = str(row.get("sell_action") or row.get("exit_action") or row.get("action") or "").lower()
    reason = str(row.get("exit_reason") or row.get("explain") or "").lower()
    return any(token in text + reason for token in ("清仓", "风险退出", "止损", "clear", "stop", "risk"))


def _first_text(rows: Sequence[Mapping[str, Any]], key: str, *, default: str = "") -> str:
    for row in rows:
        value = row.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _find_symbol(rows: Sequence[Mapping[str, Any]], symbol: str) -> dict[str, Any]:
    for row in rows:
        if _symbol(row.get("symbol") or row.get("etf_code") or row.get("code")) == symbol:
            return dict(row)
    return {}


def _symbol(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text.isdigit() else text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "是", "入选", "selected"}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "是"}


def _number(value: Any, default: Any = 0.0) -> Any:
    if value in (None, ""):
        return default
    try:
        result = float(str(value).strip().rstrip("%"))
        if str(value).strip().endswith("%"):
            result /= 100.0
        if not math.isfinite(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _ratio(value: Any) -> float:
    number = _number(value, 0.0)
    return number / 100.0 if abs(number) > 1 else number


def _as_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                return _as_list(parsed)
            except json.JSONDecodeError:
                pass
        return [item.strip() for item in text.replace("、", ",").split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _unique(values: Any) -> list[str]:
    result: list[str] = []
    for value in values:
        for item in _as_list(value):
            if item and item not in result:
                result.append(item)
    return result


def _join_reason(reasons: Sequence[str]) -> str:
    unique = _unique(reasons)
    return "；".join(unique) if unique else "无"


def _extract_between(text: str, prefix: str, suffix: str) -> str:
    if prefix not in text:
        return ""
    rest = text.split(prefix, 1)[1]
    return rest.split(suffix, 1)[0].strip()


def _post_924(value: Any) -> bool:
    try:
        return pd.Timestamp(value) >= pd.Timestamp("2024-09-24")
    except Exception:  # noqa: BLE001
        return True


def _now() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(timespec="seconds")


if __name__ == "__main__":
    run_v21_backend_pipeline()
