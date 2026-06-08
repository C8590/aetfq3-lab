from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


REQUIRED_FIELDS = {
    "report_type",
    "task_scope",
    "status",
    "lab_only",
    "no_stable",
    "no_qmt",
    "no_order_intent",
    "no_training",
    "checkpoint_saved",
    "model_saved",
    "order_intent_generated",
    "stable_effect_allowed",
    "qmt_allowed",
    "intake_passed",
    "schema_passed",
    "forbidden_feature_passed",
    "tensor_shape_passed",
    "state_machine_passed",
    "rows_checked",
    "batch_size",
    "time_steps",
    "feature_count",
    "target_count",
    "visited_states",
}

FORBIDDEN_TOKENS = {
    "OrderIntent",
    "target_weight",
    "final_buy_action",
    "qmt_order",
    "live_order",
    "trade_instruction",
}


def read_report(report_path: Path) -> dict[str, Any]:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("report root must be a JSON object")
    return payload


def validate_report(report_path: Path) -> dict[str, Any]:
    report = read_report(report_path)
    p0_blockers: list[str] = []
    missing = sorted(REQUIRED_FIELDS - set(report))
    if missing:
        p0_blockers.append("missing required report fields: " + ", ".join(missing))

    require_value(report, "lab_only", True, p0_blockers)
    require_value(report, "no_stable", True, p0_blockers)
    require_value(report, "no_qmt", True, p0_blockers)
    require_value(report, "no_order_intent", True, p0_blockers)
    require_value(report, "no_training", True, p0_blockers)
    require_value(report, "checkpoint_saved", False, p0_blockers)
    require_value(report, "model_saved", False, p0_blockers)
    require_value(report, "order_intent_generated", False, p0_blockers)
    require_value(report, "stable_effect_allowed", False, p0_blockers)
    require_value(report, "qmt_allowed", False, p0_blockers)

    forbidden_hits = scan_forbidden_tokens(report)
    if forbidden_hits:
        p0_blockers.append("forbidden report tokens found: " + ", ".join(forbidden_hits))

    summary = {
        "report_type": "intraday_synthetic_report_reader_check",
        "source_report": str(report_path),
        "status": "passed" if not p0_blockers else "failed",
        "reader_passed": not p0_blockers,
        "lab_only": report.get("lab_only"),
        "no_stable": report.get("no_stable"),
        "no_qmt": report.get("no_qmt"),
        "no_order_intent": report.get("no_order_intent"),
        "no_training": report.get("no_training"),
        "checkpoint_saved": report.get("checkpoint_saved"),
        "model_saved": report.get("model_saved"),
        "order_intent_generated": report.get("order_intent_generated"),
        "stable_effect_allowed": report.get("stable_effect_allowed"),
        "qmt_allowed": report.get("qmt_allowed"),
        "p0_blockers": p0_blockers,
    }
    write_reader_check(summary, report_path.parent)
    return summary


def require_value(report: dict[str, Any], field_name: str, expected: Any, p0_blockers: list[str]) -> None:
    if field_name in report and report.get(field_name) != expected:
        p0_blockers.append(f"{field_name} must be {json.dumps(expected)}")


def scan_forbidden_tokens(value: Any) -> list[str]:
    hits: set[str] = set()
    scan_value(value, hits)
    return sorted(hits)


def scan_value(value: Any, hits: set[str]) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            scan_string(str(key), hits)
            scan_value(nested, hits)
    elif isinstance(value, list):
        for item in value:
            scan_value(item, hits)
    elif isinstance(value, str):
        scan_string(value, hits)


def scan_string(value: str, hits: set[str]) -> None:
    for token in FORBIDDEN_TOKENS:
        if token in value:
            hits.add(token)


def write_reader_check(summary: dict[str, Any], out_dir: Path) -> None:
    json_path = out_dir / "intraday_synthetic_report_reader_check.json"
    md_path = out_dir / "intraday_synthetic_report_reader_check.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。",
        "",
        "# Intraday Synthetic Report Reader Check",
        "",
        f"- status: {summary['status']}",
        f"- reader_passed: {str(summary['reader_passed']).lower()}",
        f"- p0_blockers: {len(summary['p0_blockers'])}",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read and validate Lab-only intraday synthetic dry validation report.")
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = validate_report(args.report)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if summary["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
