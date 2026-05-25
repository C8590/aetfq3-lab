"""Entry intent diagnostics and coverage reports."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


ENTRY_DIAGNOSTIC_FIELDS = (
    "trade_date",
    "code",
    "name",
    "sector",
    "market_state",
    "risk_level",
    "risk_score",
    "sector_rank",
    "etf_rank",
    "momentum_score",
    "acceleration_score",
    "entry_score",
    "trend_maturity",
    "buy_quality",
    "overheat_flag",
    "defense_block",
    "data_quality_status",
    "liquidity_status",
    "raw_entry_action",
    "raw_entry_target_weight",
    "raw_entry_confidence",
    "raw_entry_reason",
    "raw_entry_block_reason",
    "final_buy_action",
    "final_target_weight",
    "final_block_reason",
    "control_override_reason",
    "active_exit_count",
    "actual_position_exit_count",
    "exit_priority_blocked",
    "exit_block_reason",
    "exit_block_release_condition",
    "blocked_by_exit_symbols",
    "has_real_position_to_exit",
    "exit_action_type",
    "risk_gate_blocked",
    "ml_score",
    "p_good_entry",
    "p_bad_entry",
    "ml_entry_advice",
    "ml_action_suggestion",
    "ml_decision_mode",
    "ml_adjustment",
    "ml_adjustment_reason_cn",
    "rule_action",
    "ml_adjusted_action",
    "ml_observation_notice",
)

COVERAGE_FIELDS = (
    "total_trade_days",
    "offense_days",
    "risk_r0_r1_days",
    "days_with_candidates",
    "raw_buy_days",
    "raw_probe_days",
    "raw_observe_only_days",
    "final_buy_days",
    "final_blocked_days",
    "exit_priority_blocked_days",
    "risk_gate_blocked_days",
    "no_signal_days",
    "block_reason_distribution",
    "observe_reason_distribution",
)


def write_entry_diagnostics(
    *,
    output_dir: str | Path,
    trade_date: str,
    pre_selection_rows: Sequence[Mapping[str, Any]],
    entry_actions: Sequence[Mapping[str, Any]],
    risk: Mapping[str, Any],
) -> list[dict[str, Any]]:
    out_dir = Path(output_dir)
    action_by_symbol = {_symbol(row.get("etf_code") or row.get("symbol") or row.get("code")): row for row in entry_actions}
    rows: list[dict[str, Any]] = []
    for pre_row in pre_selection_rows:
        symbol = _symbol(pre_row.get("symbol") or pre_row.get("etf_code") or pre_row.get("code"))
        action = action_by_symbol.get(symbol, {})
        if not _truthy(pre_row.get("selected")) and not action:
            continue
        reason = str(action.get("raw_entry_reason") or action.get("explain") or pre_row.get("reason") or "")
        maturity = str(action.get("trend_maturity") or _extract_between(reason, "趋势成熟度：", "；"))
        quality = str(action.get("buy_quality") or _extract_between(reason, "买点质量：", "；"))
        raw_action = _entry_action_label(action.get("raw_entry_action") or action.get("entry_action"))
        final_action = _final_action_label(action.get("final_buy_action"), action.get("actual_buy"), action.get("block_reason"))
        rows.append(
            {
                "trade_date": str(pre_row.get("trade_date") or action.get("trade_date") or trade_date)[:10],
                "code": symbol,
                "name": str(pre_row.get("name") or action.get("etf_name") or action.get("name") or ""),
                "sector": str(pre_row.get("sector") or ""),
                "market_state": str(pre_row.get("market_state") or action.get("market_state") or ""),
                "risk_level": str(risk.get("risk_level") or "R0").upper(),
                "risk_score": risk.get("risk_score", 0),
                "sector_rank": _first(pre_row, "sector_rank", "_sector_rank", "rank_in_sector"),
                "etf_rank": _first(pre_row, "etf_rank", "_etf_rank", "rank"),
                "momentum_score": _first(pre_row, "momentum_score", "momentum", "score"),
                "acceleration_score": _first(pre_row, "acceleration_score", "acceleration"),
                "entry_score": _first(pre_row, "entry_score", "score"),
                "trend_maturity": maturity,
                "buy_quality": quality,
                "overheat_flag": _is_overheated(maturity, quality, reason),
                "defense_block": _is_defense(pre_row.get("market_state")) or raw_action == "REJECT",
                "data_quality_status": _first(pre_row, "data_quality_status", "data_quality_flag", default="正常"),
                "liquidity_status": _first(pre_row, "liquidity_status", "liquidity_flag", default="正常"),
                "raw_entry_action": raw_action,
                "raw_entry_target_weight": _number(action.get("raw_entry_target_weight", action.get("target_weight"))),
                "raw_entry_confidence": _number(action.get("raw_entry_confidence", action.get("confidence"))),
                "raw_entry_reason": reason,
                "raw_entry_block_reason": str(action.get("raw_entry_block_reason") or (reason if raw_action not in {"BUY", "PROBE"} else "")),
                "final_buy_action": final_action,
                "final_target_weight": _number(action.get("final_target_weight", action.get("target_weight"))),
                "final_block_reason": str(action.get("final_block_reason") or action.get("block_reason") or ""),
                "control_override_reason": str(action.get("control_override_reason") or ""),
                "active_exit_count": int(_number(action.get("active_exit_count"))),
                "actual_position_exit_count": int(_number(action.get("actual_position_exit_count"))),
                "exit_priority_blocked": _truthy(action.get("exit_priority_blocked")),
                "exit_block_reason": str(action.get("exit_block_reason") or ""),
                "exit_block_release_condition": str(action.get("exit_block_release_condition") or ""),
                "blocked_by_exit_symbols": _json_value(action.get("blocked_by_exit_symbols")),
                "has_real_position_to_exit": _truthy(action.get("has_real_position_to_exit")),
                "exit_action_type": str(action.get("exit_action_type") or ""),
                "risk_gate_blocked": _truthy(action.get("risk_gate_blocked")),
                "ml_score": action.get("ml_score", ""),
                "p_good_entry": action.get("p_good_entry", ""),
                "p_bad_entry": action.get("p_bad_entry", ""),
                "ml_entry_advice": str(action.get("ml_entry_advice") or ""),
                "ml_action_suggestion": str(action.get("ml_action_suggestion") or "NO_ML"),
                "ml_decision_mode": str(action.get("ml_decision_mode") or "shadow"),
                "ml_adjustment": str(action.get("ml_adjustment") or ""),
                "ml_adjustment_reason_cn": str(action.get("ml_adjustment_reason_cn") or ""),
                "rule_action": str(action.get("rule_action") or raw_action),
                "ml_adjusted_action": str(action.get("ml_adjusted_action") or action.get("final_buy_action") or raw_action),
                "ml_observation_notice": _ml_observation_notice(action),
            }
        )

    _write_csv(out_dir / "entry_diagnostics.csv", ENTRY_DIAGNOSTIC_FIELDS, rows)
    (out_dir / "entry_diagnostics.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows


def write_entry_signal_coverage_report(
    *,
    output_dir: str | Path,
    diagnostics_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    out_dir = Path(output_dir)
    frame = _coverage_frame(out_dir, diagnostics_rows)
    summary = _coverage_summary(frame)
    _write_csv(out_dir / "entry_signal_coverage_report.csv", COVERAGE_FIELDS, [summary])
    (out_dir / "entry_signal_coverage_report.md").write_text(_coverage_markdown(summary), encoding="utf-8")
    return summary


def _coverage_frame(out_dir: Path, diagnostics_rows: Sequence[Mapping[str, Any]] | None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    if (out_dir / "signal_cases.csv").exists():
        cases = pd.read_csv(out_dir / "signal_cases.csv", dtype=str).fillna("")
        frames.append(
            pd.DataFrame(
                {
                    "trade_date": cases.get("trade_date", ""),
                    "code": cases.get("etf_code", cases.get("symbol", "")),
                    "market_state": cases.get("market_state", ""),
                    "risk_level": cases.get("risk_level", "R0"),
                    "candidate_flag": cases.apply(_case_has_candidate, axis=1),
                    "raw_entry_action": cases.get("raw_entry_action", cases.get("entry_action", "")),
                    "final_buy_action": cases.get("final_buy_action", cases.get("entry_action", "")),
                    "exit_priority_blocked": cases.get("exit_priority_blocked", False),
                    "risk_gate_blocked": cases.get("risk_gate_blocked", False),
                    "final_block_reason": cases.get("final_block_reason", ""),
                    "raw_entry_block_reason": cases.get("raw_entry_block_reason", cases.get("reason", "")),
                }
            )
        )

    if diagnostics_rows:
        diag_frame = pd.DataFrame([dict(row) for row in diagnostics_rows])
    elif (out_dir / "entry_diagnostics.csv").exists():
        diag_frame = pd.read_csv(out_dir / "entry_diagnostics.csv", dtype=str).fillna("")
    else:
        diag_frame = pd.DataFrame(columns=ENTRY_DIAGNOSTIC_FIELDS)
    if not diag_frame.empty:
        diag_frame = diag_frame.copy()
        diag_frame["candidate_flag"] = diag_frame.apply(_diagnostic_has_candidate, axis=1)
        if frames:
            diag_dates = set(diag_frame["trade_date"].astype(str).str[:10])
            frames = [frame[~frame["trade_date"].astype(str).str[:10].isin(diag_dates)] for frame in frames]
        frames.append(diag_frame)

    frame = pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame(columns=ENTRY_DIAGNOSTIC_FIELDS)
    for column in ("raw_entry_action", "final_buy_action"):
        if column in frame:
            frame[column] = frame[column].map(_entry_action_label)
    if "risk_level" not in frame:
        frame["risk_level"] = "R0"
    return frame.fillna("")


def _coverage_summary(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or "trade_date" not in frame:
        return {field: 0 if field.endswith("days") or field == "total_trade_days" else "{}" for field in COVERAGE_FIELDS}

    frame = frame.copy()
    frame["trade_date"] = frame["trade_date"].astype(str).str[:10]
    frame = frame[frame["trade_date"].astype(bool)]
    grouped = list(frame.groupby("trade_date", dropna=False))
    total_days = len(grouped)
    qualifying_days: list[tuple[str, pd.DataFrame]] = []
    offense_days = 0
    risk_r0_r1_days = 0
    candidate_days = 0

    for date, group in grouped:
        offense = group["market_state"].map(_is_attack).any() if "market_state" in group else False
        risk_ok = group["risk_level"].astype(str).str.upper().isin(["R0", "R1"]).any()
        has_candidate = group.get("candidate_flag", pd.Series([True] * len(group))).map(_truthy).any()
        offense_days += int(offense)
        risk_r0_r1_days += int(risk_ok)
        candidate_days += int(has_candidate)
        if offense and risk_ok and has_candidate:
            qualifying_days.append((date, group))

    def count_days(predicate: Any) -> int:
        return sum(1 for _, group in qualifying_days if predicate(group))

    raw_buy_days = count_days(lambda group: group["raw_entry_action"].map(lambda x: x == "BUY").any())
    raw_probe_days = count_days(lambda group: group["raw_entry_action"].map(lambda x: x == "PROBE").any())
    final_buy_days = count_days(lambda group: group["final_buy_action"].map(lambda x: x in {"BUY", "PROBE"}).any())
    final_blocked_days = count_days(lambda group: group["final_buy_action"].map(lambda x: x == "BLOCKED").any())
    exit_blocked_days = count_days(lambda group: group.get("exit_priority_blocked", pd.Series(dtype=str)).map(_truthy).any())
    risk_blocked_days = count_days(lambda group: group.get("risk_gate_blocked", pd.Series(dtype=str)).map(_truthy).any())
    raw_observe_only_days = count_days(lambda group: not group["raw_entry_action"].map(lambda x: x in {"BUY", "PROBE"}).any())
    no_signal_days = count_days(
        lambda group: not group["raw_entry_action"].map(lambda x: x in {"BUY", "PROBE"}).any()
        and not group["final_buy_action"].map(lambda x: x in {"BUY", "PROBE", "BLOCKED"}).any()
    )

    block_counter = Counter(
        _short_reason(value)
        for value in frame.get("final_block_reason", pd.Series(dtype=str)).astype(str)
        if str(value).strip()
    )
    observe_counter = Counter(
        _short_reason(value)
        for value in frame.get("raw_entry_block_reason", pd.Series(dtype=str)).astype(str)
        if str(value).strip()
    )
    return {
        "total_trade_days": total_days,
        "offense_days": offense_days,
        "risk_r0_r1_days": risk_r0_r1_days,
        "days_with_candidates": candidate_days,
        "raw_buy_days": raw_buy_days,
        "raw_probe_days": raw_probe_days,
        "raw_observe_only_days": raw_observe_only_days,
        "final_buy_days": final_buy_days,
        "final_blocked_days": final_blocked_days,
        "exit_priority_blocked_days": exit_blocked_days,
        "risk_gate_blocked_days": risk_blocked_days,
        "no_signal_days": no_signal_days,
        "block_reason_distribution": json.dumps(dict(block_counter.most_common(20)), ensure_ascii=False),
        "observe_reason_distribution": json.dumps(dict(observe_counter.most_common(20)), ensure_ascii=False),
    }


def _coverage_markdown(summary: Mapping[str, Any]) -> str:
    raw_total = int(summary.get("raw_buy_days", 0) or 0) + int(summary.get("raw_probe_days", 0) or 0)
    if raw_total == 0:
        diagnosis = "结论：进攻且 R0/R1 的样本中 raw BUY/PROBE 为 0，entry 规则仍偏保守，需要继续修正。"
    elif int(summary.get("final_buy_days", 0) or 0) == 0:
        diagnosis = "结论：raw entry 有买入意图，但 final BUY/PROBE 为 0，需要重点解释总控、exit 或 RiskGate 阻断。"
    else:
        diagnosis = "结论：raw entry 和 final 裁决均出现过买入或试探信号，不能判定 entry 本体完全失败。"
    lines = [
        "# Entry Signal Coverage Report",
        "",
        f"- total_trade_days: {summary.get('total_trade_days', 0)}",
        f"- offense_days: {summary.get('offense_days', 0)}",
        f"- risk_r0_r1_days: {summary.get('risk_r0_r1_days', 0)}",
        f"- days_with_candidates: {summary.get('days_with_candidates', 0)}",
        f"- raw_buy_days: {summary.get('raw_buy_days', 0)}",
        f"- raw_probe_days: {summary.get('raw_probe_days', 0)}",
        f"- raw_observe_only_days: {summary.get('raw_observe_only_days', 0)}",
        f"- final_buy_days: {summary.get('final_buy_days', 0)}",
        f"- final_blocked_days: {summary.get('final_blocked_days', 0)}",
        f"- exit_priority_blocked_days: {summary.get('exit_priority_blocked_days', 0)}",
        f"- risk_gate_blocked_days: {summary.get('risk_gate_blocked_days', 0)}",
        f"- no_signal_days: {summary.get('no_signal_days', 0)}",
        "",
        "## Required Answers",
        "",
        f"1. market_state=进攻 且 RiskGate=R0/R1 的日期里，entry 原始 BUY/PROBE 出现 {raw_total} 天。",
        f"2. raw_buy_days + raw_probe_days = {raw_total}。",
        f"3. final_buy_days = {summary.get('final_buy_days', 0)}；final_blocked_days = {summary.get('final_blocked_days', 0)}。",
        "4. 只有实际买入为 0 但 raw entry 有买入意图时，不能判定 entry 本体失败。",
        "",
        diagnosis,
        "",
        "## Distributions",
        "",
        f"- block_reason_distribution: {summary.get('block_reason_distribution', '{}')}",
        f"- observe_reason_distribution: {summary.get('observe_reason_distribution', '{}')}",
        "",
    ]
    return "\n".join(lines)


def _write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _json_value(value: Any) -> str:
    if isinstance(value, (list, tuple, set, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def _entry_action_label(value: Any) -> str:
    text = str(value or "").strip()
    upper = text.upper()
    lower = text.lower()
    if upper in {"BUY", "PROBE", "OBSERVE", "REJECT", "AVOID", "BLOCKED"}:
        return upper
    if "avoid" in lower:
        return "AVOID"
    if "blocked" in lower or "阻断" in text or "冻结" in text:
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


def _final_action_label(value: Any, actual_buy: Any, block_reason: Any) -> str:
    label = _entry_action_label(value)
    if _truthy(actual_buy):
        return label if label in {"BUY", "PROBE"} else "BUY"
    if str(block_reason or "").strip():
        return "BLOCKED"
    return label


def _extract_between(text: str, prefix: str, suffix: str) -> str:
    if prefix not in text:
        return ""
    rest = text.split(prefix, 1)[1]
    return rest.split(suffix, 1)[0].strip()


def _first(row: Mapping[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return default


def _number(value: Any) -> float:
    try:
        text = str(value).strip()
        if text.endswith("%"):
            return float(text[:-1]) / 100
        return float(text)
    except (TypeError, ValueError):
        return 0.0


def _symbol(value: Any) -> str:
    text = str(value or "").strip()
    return text.zfill(6) if text.isdigit() else text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "是", "selected"}


def _is_attack(value: Any) -> bool:
    return str(value or "").strip().lower() in {"进攻", "attack"}


def _is_defense(value: Any) -> bool:
    return str(value or "").strip().lower() in {"防守", "defense", "defensive"}


def _is_overheated(maturity: str, quality: str, reason: str) -> bool:
    text = f"{maturity} {quality} {reason}"
    return any(token in text for token in ("过热", "追高", "连续冲高"))


def _case_has_candidate(row: Mapping[str, Any]) -> bool:
    action = _entry_action_label(row.get("raw_entry_action") or row.get("entry_action"))
    if action in {"BUY", "PROBE"}:
        return True
    try:
        if _number(row.get("target_weight")) > 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    reason = str(row.get("reason") or "")
    return "入选" in reason and "未进入预选" not in reason


def _diagnostic_has_candidate(row: Mapping[str, Any]) -> bool:
    action = _entry_action_label(row.get("raw_entry_action"))
    if action in {"BUY", "PROBE"}:
        return True
    reason = str(row.get("raw_entry_reason") or row.get("raw_entry_block_reason") or "")
    return "入选" in reason and "未进入预选" not in reason


def _ml_observation_notice(action: Mapping[str, Any]) -> str:
    active = str(action.get("ml_action_suggestion") or "").strip().upper() != "NO_ML"
    if active:
        return "ML 参数级建议已读取 / 未直接参与交易裁决"
    return "ML 参数级建议未读取 / 未直接参与交易裁决"


def _short_reason(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text[:80]
