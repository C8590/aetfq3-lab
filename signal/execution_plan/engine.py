"""Build first-version buy/sell execution plans from controller decisions.

This layer turns V2.1 entry/exit decisions into human-executable plans. It does
not change entry thresholds, final buy actions, QMT intents, or order routing.
Intraday and 5-minute fields are placeholders until a later data adapter exists.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


HIGH_OPEN_GAP_THRESHOLD = 0.015
DEFAULT_PROBE_WEIGHT = 0.30

EXECUTION_PLAN_FIELDS = (
    "trade_date",
    "expected_execution_date",
    "etf_code",
    "etf_name",
    "plan_side",
    "source_action",
    "execution_action",
    "execution_priority",
    "buy_method",
    "sell_method",
    "target_weight",
    "current_weight",
    "reduce_ratio",
    "high_open_gap_threshold",
    "expected_open_gap_pct",
    "high_open_handling",
    "wait_pullback_condition",
    "first_30min_confirmation",
    "cancel_buy_condition",
    "sell_condition",
    "profit_protection_placeholder",
    "intraday_confirm_placeholder",
    "intraday_exit_trigger_placeholder",
    "five_min_k_placeholder",
    "risk_note",
    "qmt_intent_status",
    "manual_confirm_required",
    "source_signal",
    "explain",
)

ACTIVE_SELL_ACTIONS = {"SELL", "REDUCE", "REDUCE_HALF", "REDUCE_ONE_THIRD", "CLEAR"}
HIGH_PRIORITY_SELL_ACTIONS = {"SELL", "CLEAR"}


def build_execution_plan(
    *,
    trade_date: str,
    expected_execution_date: str,
    entry_actions: Sequence[Mapping[str, Any]],
    exit_actions: Sequence[Mapping[str, Any]],
    portfolio: Sequence[Mapping[str, Any]] = (),
    risk_gate: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Return execution-plan rows for buy and sell sides."""

    risk_gate = risk_gate or {}
    portfolio_by_symbol = {str(item.get("etf_code") or item.get("symbol") or ""): item for item in portfolio}
    rows: list[dict[str, Any]] = []

    for entry in entry_actions:
        if _is_buy_plan_candidate(entry):
            rows.append(_buy_plan_row(trade_date, expected_execution_date, entry, risk_gate))

    sell_rows = [
        _sell_plan_row(
            trade_date,
            expected_execution_date,
            item,
            portfolio_by_symbol.get(str(item.get("etf_code") or "")),
        )
        for item in exit_actions
        if _is_sell_plan_candidate(item)
    ]
    rows.extend(sell_rows)
    if not sell_rows:
        rows.append(_no_sell_plan_row(trade_date, expected_execution_date))

    return rows


def _buy_plan_row(
    trade_date: str,
    expected_execution_date: str,
    entry: Mapping[str, Any],
    risk_gate: Mapping[str, Any],
) -> dict[str, Any]:
    action = str(entry.get("final_buy_action") or entry.get("raw_entry_action") or "").strip().upper()
    target_weight = _ratio(
        entry.get("final_target_weight")
        if entry.get("final_target_weight") not in (None, "")
        else entry.get("raw_entry_target_weight")
    )
    if action == "PROBE" and target_weight <= 0:
        target_weight = DEFAULT_PROBE_WEIGHT
    gap = _optional_ratio(
        entry.get("expected_open_gap_pct")
        or entry.get("expected_open_gap")
        or entry.get("open_gap_pct")
        or entry.get("next_open_gap_pct")
    )
    high_open = gap is not None and gap > HIGH_OPEN_GAP_THRESHOLD
    blocked = action == "BLOCKED" or _bool(entry.get("risk_gate_blocked")) or _bool(entry.get("exit_priority_blocked"))
    execution_action = "BUY_BLOCKED" if blocked else ("WAIT_PULLBACK" if high_open else ("PROBE_READY" if action == "PROBE" else "BUY_READY"))
    high_open_text = (
        f"预计高开 {gap:.2%}，超过 {HIGH_OPEN_GAP_THRESHOLD:.2%}，不追高，等待回踩。"
        if high_open
        else "预计高开未超过阈值或暂无盘前高开数据；日线版默认不追价，盘中高开超过阈值则改为 WAIT_PULLBACK。"
    )
    block_reason = str(entry.get("final_block_reason") or entry.get("block_reason") or "")
    return _base_row(
        trade_date,
        expected_execution_date,
        plan_side="BUY",
        etf_code=str(entry.get("etf_code") or ""),
        etf_name=str(entry.get("etf_name") or ""),
        source_action=action,
        execution_action=execution_action,
        execution_priority="RISK_OR_EXIT_BLOCKED" if blocked else "NORMAL_BUY_PLAN",
        target_weight=target_weight if not blocked else 0.0,
        current_weight=0.0,
        reduce_ratio=0.0,
        source_signal=str(entry.get("source_signal") or "entry_signal.csv"),
        explain=str(entry.get("explain") or ""),
        buy_method="试探仓分批买入" if action == "PROBE" else "按目标仓位分批买入",
        sell_method="",
        high_open_gap_threshold=HIGH_OPEN_GAP_THRESHOLD,
        expected_open_gap_pct="" if gap is None else round(gap, 6),
        high_open_handling=high_open_text,
        wait_pullback_condition="不追开盘急涨；等待回踩至昨收附近、日内均价附近或 5 分钟 K 走稳后再人工确认。",
        first_30min_confirmation="前 30 分钟只确认承接和波动，不自动下单；若放量冲高回落，取消追入。",
        cancel_buy_condition=_cancel_buy_condition(entry, risk_gate),
        sell_condition="",
        profit_protection_placeholder="",
        intraday_confirm_placeholder="预留：分时承接、开盘 30 分钟区间、5 分钟 K 突破/回踩确认。",
        intraday_exit_trigger_placeholder="",
        five_min_k_placeholder="预留：5min close/ma/volume 字段，第一版不读取真实 5 分钟 K。",
        risk_note=_buy_risk_note(entry, risk_gate, block_reason),
        qmt_intent_status="DRAFT_ONLY",
        manual_confirm_required=True,
    )


def _sell_plan_row(
    trade_date: str,
    expected_execution_date: str,
    exit_action: Mapping[str, Any],
    holding: Mapping[str, Any] | None,
) -> dict[str, Any]:
    action = _normalize_sell_action(exit_action.get("exit_action") or exit_action.get("action"))
    current_weight = _ratio((holding or {}).get("current_weight"))
    reduce_ratio = _ratio(exit_action.get("reduce_ratio")) or (1.0 if action in {"SELL", "CLEAR"} else 0.0)
    execution_action = "CLEAR_READY" if action == "CLEAR" else ("REDUCE_READY" if action == "REDUCE" else "SELL_READY")
    priority = "HIGH_PRIORITY_EXIT" if action in HIGH_PRIORITY_SELL_ACTIONS or _bool(exit_action.get("high_priority_exit")) else "NORMAL_EXIT_PLAN"
    return _base_row(
        trade_date,
        expected_execution_date,
        plan_side="SELL",
        etf_code=str(exit_action.get("etf_code") or ""),
        etf_name=str(exit_action.get("etf_name") or ""),
        source_action=action,
        execution_action=execution_action,
        execution_priority=priority,
        target_weight=max(current_weight * (1 - reduce_ratio), 0.0) if current_weight else 0.0,
        current_weight=current_weight,
        reduce_ratio=reduce_ratio,
        source_signal=str(exit_action.get("source_signal") or "exit_signal.csv"),
        explain=str(exit_action.get("explain") or ""),
        buy_method="",
        sell_method="风险/退出优先，人工确认后分批卖出或清仓。",
        high_open_gap_threshold=HIGH_OPEN_GAP_THRESHOLD,
        expected_open_gap_pct="",
        high_open_handling="卖出计划不因高开追买；若高开后快速转弱，优先执行风险卖出。",
        wait_pullback_condition="",
        first_30min_confirmation="前 30 分钟确认流动性和跌破/反抽失败，不自动提交订单。",
        cancel_buy_condition="",
        sell_condition=str(exit_action.get("explain") or "exit 信号触发卖出/减仓/清仓计划。"),
        profit_protection_placeholder="预留：浮盈回撤保护、分段止盈、移动止损线。",
        intraday_confirm_placeholder="",
        intraday_exit_trigger_placeholder="预留：跌破日内均价、5 分钟 K 放量破位、反抽失败触发。",
        five_min_k_placeholder="预留：5min close/ma/volume 字段，第一版不读取真实 5 分钟 K。",
        risk_note="exit 清仓/风险退出优先级高于新增买入；QMT 只生成 DRAFT，必须人工确认。",
        qmt_intent_status="DRAFT_ONLY",
        manual_confirm_required=True,
    )


def _no_sell_plan_row(trade_date: str, expected_execution_date: str) -> dict[str, Any]:
    return _base_row(
        trade_date,
        expected_execution_date,
        plan_side="SELL",
        etf_code="",
        etf_name="",
        source_action="NO_EXIT_SIGNAL",
        execution_action="NO_SELL_PLAN",
        execution_priority="NO_ACTIVE_EXIT",
        target_weight=0.0,
        current_weight=0.0,
        reduce_ratio=0.0,
        source_signal="exit_signal.csv",
        explain="当前无 SELL/REDUCE/CLEAR 类 exit 信号，明日卖出计划为 NO_SELL_PLAN。",
        buy_method="",
        sell_method="无卖出计划；继续观察持仓和风险门控。",
        high_open_gap_threshold=HIGH_OPEN_GAP_THRESHOLD,
        expected_open_gap_pct="",
        high_open_handling="",
        wait_pullback_condition="",
        first_30min_confirmation="",
        cancel_buy_condition="",
        sell_condition="NO_SELL_PLAN",
        profit_protection_placeholder="预留：持仓浮盈后的利润保护规则。",
        intraday_confirm_placeholder="",
        intraday_exit_trigger_placeholder="预留：日内跌破触发和 5 分钟 K 风险确认。",
        five_min_k_placeholder="预留：5min close/ma/volume 字段，第一版不读取真实 5 分钟 K。",
        risk_note="无卖出计划不等于无风险；若 RiskGate 升级或 exit 信号出现，卖出优先级覆盖买入。",
        qmt_intent_status="DRAFT_ONLY",
        manual_confirm_required=True,
    )


def _base_row(
    trade_date: str,
    expected_execution_date: str,
    **values: Any,
) -> dict[str, Any]:
    row = {field: "" for field in EXECUTION_PLAN_FIELDS}
    row.update(values)
    row["trade_date"] = trade_date
    row["expected_execution_date"] = expected_execution_date
    return row


def _is_buy_plan_candidate(entry: Mapping[str, Any]) -> bool:
    action = str(entry.get("final_buy_action") or entry.get("raw_entry_action") or "").strip().upper()
    return action in {"BUY", "PROBE", "BLOCKED"} and (_bool(entry.get("intended_buy")) or _bool(entry.get("actual_buy")))


def _is_sell_plan_candidate(item: Mapping[str, Any]) -> bool:
    action = _normalize_sell_action(item.get("exit_action") or item.get("action"))
    return action in ACTIVE_SELL_ACTIONS and (_bool(item.get("active_exit")) or _bool(item.get("actual_exit")) or _ratio(item.get("reduce_ratio")) > 0)


def _normalize_sell_action(value: Any) -> str:
    text = str(value or "").strip().upper()
    if any(token in text for token in ("CLEAR", "清仓")):
        return "CLEAR"
    if any(token in text for token in ("REDUCE", "减仓", "一半", "三分之一")):
        return "REDUCE"
    if any(token in text for token in ("SELL", "卖出", "退出")):
        return "SELL"
    return text


def _cancel_buy_condition(entry: Mapping[str, Any], risk_gate: Mapping[str, Any]) -> str:
    parts = [
        "开盘高开超过阈值且未回踩确认",
        "前 30 分钟放量冲高回落或跌破日内均价",
        "候选 ETF 失去相对强势或板块主线转弱",
    ]
    if _bool(risk_gate.get("freeze_entry")) or _bool(risk_gate.get("manual_takeover_required")):
        parts.append("RiskGate 冻结买入或要求人工接管")
    if _bool(entry.get("exit_priority_blocked")):
        parts.append("exit 优先级阻断尚未解除")
    return "；".join(parts) + "。"


def _buy_risk_note(entry: Mapping[str, Any], risk_gate: Mapping[str, Any], block_reason: str) -> str:
    notes = ["PROBE/BUY 只转为执行计划，不改变 final_buy_action；QMT 只输出 DRAFT，人工确认。"]
    if block_reason:
        notes.append(block_reason)
    risk_level = str(risk_gate.get("risk_level") or "").upper()
    if risk_level in {"R3", "R4", "P0"}:
        notes.append(f"当前 {risk_level} 风险优先，禁止普通买入。")
    return " ".join(notes)


def _ratio(value: Any) -> float:
    number = _optional_ratio(value)
    return 0.0 if number is None else max(number, 0.0)


def _optional_ratio(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) > 1.0:
        number = number / 100.0
    return number


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "是"}
