from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


STATES = {
    "WAIT_OPEN",
    "OPEN_GAP_TOO_HIGH",
    "INTRADAY_CONFIRMING",
    "PROBE_READY",
    "WAIT_PULLBACK",
    "CANCEL_BUY",
    "HOLD_LOCKED",
    "PROFIT_PROTECT",
    "EXIT_READY",
}
EVENTS = {
    "OPEN_PRINTED",
    "GAP_TOO_HIGH",
    "OPEN_RANGE_CONFIRMED",
    "VWAP_PULLBACK",
    "VWAP_RECLAIM",
    "BREAK_OPEN_LOW",
    "SECTOR_CONFIRM",
    "SECTOR_FADE",
    "RISK_CANCEL",
    "LOCK_HOLD_START",
    "PROFIT_PROTECT_TRIGGER",
    "EXIT_SIGNAL_RESEARCH_ONLY",
}


class IntradayWatchDryRunError(RuntimeError):
    pass


@dataclass(frozen=True)
class Transition:
    from_state: str
    event: str
    to_state: str


def transition_state(state: str, event: str) -> str:
    if event not in EVENTS:
        raise IntradayWatchDryRunError(f"unknown event: {event}")
    if state not in STATES:
        raise IntradayWatchDryRunError(f"unknown state: {state}")

    if event == "RISK_CANCEL":
        return "CANCEL_BUY"
    if event == "BREAK_OPEN_LOW":
        return "CANCEL_BUY"
    if event == "SECTOR_FADE" and state in {"INTRADAY_CONFIRMING", "PROBE_READY", "WAIT_PULLBACK"}:
        return "CANCEL_BUY"
    if event == "GAP_TOO_HIGH" and state == "WAIT_OPEN":
        return "OPEN_GAP_TOO_HIGH"
    if event == "OPEN_PRINTED" and state == "WAIT_OPEN":
        return "INTRADAY_CONFIRMING"
    if event == "OPEN_RANGE_CONFIRMED" and state in {"WAIT_OPEN", "INTRADAY_CONFIRMING"}:
        return "INTRADAY_CONFIRMING"
    if event == "SECTOR_CONFIRM" and state in {"INTRADAY_CONFIRMING", "WAIT_PULLBACK"}:
        return "PROBE_READY"
    if event == "VWAP_PULLBACK" and state in {"OPEN_GAP_TOO_HIGH", "PROBE_READY", "INTRADAY_CONFIRMING"}:
        return "WAIT_PULLBACK"
    if event == "VWAP_RECLAIM" and state == "WAIT_PULLBACK":
        return "PROBE_READY"
    if event == "LOCK_HOLD_START" and state == "PROBE_READY":
        return "HOLD_LOCKED"
    if event == "PROFIT_PROTECT_TRIGGER" and state == "HOLD_LOCKED":
        return "PROFIT_PROTECT"
    if event == "EXIT_SIGNAL_RESEARCH_ONLY" and state in {"HOLD_LOCKED", "PROFIT_PROTECT"}:
        return "EXIT_READY"
    return state


def load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        events = payload.get("events")
    else:
        events = payload
    if not isinstance(events, list) or not events:
        raise IntradayWatchDryRunError("events payload must contain a non-empty events list")
    for event in events:
        if not isinstance(event, dict) or "event_id" not in event:
            raise IntradayWatchDryRunError("each event must be an object with event_id")
    return events


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "intraday_watch_dryrun_report.json"
    md_path = out_dir / "intraday_watch_dryrun_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。",
        "",
        "# Intraday Watch State Machine Dry-run Report",
        "",
        f"- status: {report['status']}",
        f"- event_count: {report['event_count']}",
        f"- terminal_state: {report['terminal_state']}",
        f"- advisory_only: {str(report['advisory_only']).lower()}",
        f"- qmt_allowed: {str(report['qmt_allowed']).lower()}",
        f"- order_intent_generated: {str(report['order_intent_generated']).lower()}",
        f"- stable_effect_allowed: {str(report['stable_effect_allowed']).lower()}",
        f"- visited_states: {', '.join(report['visited_states'])}",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def run_intraday_watch_dryrun(events_path: Path, out_dir: Path) -> dict[str, Any]:
    events = load_events(events_path)
    state = "WAIT_OPEN"
    visited_states = [state]
    transitions: list[Transition] = []

    for item in events:
        event_id = str(item["event_id"])
        next_state = transition_state(state, event_id)
        transitions.append(Transition(from_state=state, event=event_id, to_state=next_state))
        state = next_state
        if state not in visited_states:
            visited_states.append(state)

    report = {
        "report_type": "intraday_watch_state_machine_dryrun",
        "task_scope": "Lab-only state machine dry-run; not a trading engine",
        "status": "passed",
        "lab_only": True,
        "mock_only": True,
        "reads_real_intraday": False,
        "advisory_only": True,
        "qmt_allowed": False,
        "order_intent_generated": False,
        "stable_effect_allowed": False,
        "auto_trade_allowed": False,
        "contains_live_order": False,
        "does_not_generate_order_intent": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "events_file": str(events_path),
        "event_count": len(events),
        "visited_states": visited_states,
        "terminal_state": state,
        "known_states": sorted(STATES),
        "known_events": sorted(EVENTS),
        "transitions": [transition.__dict__ for transition in transitions],
        "planned_runtime_outputs_not_generated": [
            "intraday_watch_snapshot.json",
            "intraday_watch_events.csv",
            "intraday_watch_strategy_report.md",
        ],
    }
    write_reports(report, out_dir)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday watch state machine dry-run.")
    parser.add_argument("--events", required=True, type=Path, help="Mock event JSON fixture.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Ignored local output directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_intraday_watch_dryrun(args.events, args.out_dir)
    print(json.dumps({
        "status": report["status"],
        "visited_states": report["visited_states"],
        "event_count": report["event_count"],
        "terminal_state": report["terminal_state"],
        "advisory_only": report["advisory_only"],
        "qmt_allowed": report["qmt_allowed"],
        "order_intent_generated": report["order_intent_generated"],
        "stable_effect_allowed": report["stable_effect_allowed"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

