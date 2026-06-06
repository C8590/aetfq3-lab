from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence


P0_REVIEW_REQUIRED = "P0_REVIEW_REQUIRED"
NEEDS_REVIEW = "NEEDS_REVIEW"
OK = "OK"


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def summarize_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    p0_blockers = list_value(report.get("p0_blockers"))
    p1_warnings = list_value(report.get("p1_warnings"))
    status = string_value(report.get("status"))
    advisory_only = bool_value(report.get("advisory_only"))
    training_allowed = bool_value(report.get("training_allowed"))
    affects_stable_trading = bool_value(report.get("affects_stable_trading"))

    flags = review_flags(
        status=status,
        p0_blockers=p0_blockers,
        p1_warnings=p1_warnings,
        advisory_only=advisory_only,
        training_allowed=training_allowed,
        affects_stable_trading=affects_stable_trading,
    )

    return {
        "kind": "report",
        "review_status": review_status(flags),
        "review_flags": flags,
        "status": status,
        "sample_type": string_value(report.get("sample_type")),
        "rows_checked": int_value(report.get("rows_checked")),
        "intake_passed": bool_value(report.get("intake_passed")),
        "schema_passed": bool_value(report.get("schema_passed")),
        "p0_blockers_count": len(p0_blockers),
        "p1_warnings_count": len(p1_warnings),
        "advisory_only": advisory_only,
        "training_allowed": training_allowed,
        "affects_stable_trading": affects_stable_trading,
    }


def summarize_summary(path: Path) -> dict[str, Any]:
    summary = load_json(path)
    reports = normalize_summary_reports(summary)
    statuses = {report["name"]: report["status"] for report in reports}
    p0_count = sum(int_value(report.get("p0_blockers_count")) for report in reports)
    p1_count = sum(int_value(report.get("p1_warnings_count")) for report in reports)
    all_passed = bool(reports) and all(report["status"] == "passed" for report in reports)
    advisory_only = summary_advisory_only(summary, reports)
    training_allowed_false = summary_training_allowed_false(summary, reports)
    affects_stable_trading_false = summary_affects_stable_trading_false(summary, reports)

    flags: list[str] = []
    if not all_passed:
        flags.append(NEEDS_REVIEW)
    if p0_count > 0 or advisory_only is False or not training_allowed_false or not affects_stable_trading_false:
        flags.append(P0_REVIEW_REQUIRED)
    elif p1_count > 0:
        flags.append(NEEDS_REVIEW)

    return {
        "kind": "summary",
        "review_status": review_status(flags),
        "review_flags": dedupe(flags),
        "included_report_count": len(reports),
        "report_statuses": statuses,
        "all_passed": all_passed,
        "has_p0": p0_count > 0,
        "has_p1": p1_count > 0,
        "advisory_only": advisory_only,
        "training_allowed_false": training_allowed_false,
        "affects_stable_trading_false": affects_stable_trading_false,
    }


def normalize_summary_reports(summary: dict[str, Any]) -> list[dict[str, Any]]:
    raw_reports = summary.get("reports")
    if isinstance(raw_reports, list):
        return [normalize_report_item(index, item) for index, item in enumerate(raw_reports, start=1)]

    result = summary.get("result")
    if isinstance(result, dict):
        reports: list[dict[str, Any]] = []
        if "false_downgrade_status" in result:
            reports.append(
                {
                    "name": "false_downgrade",
                    "status": string_value(result.get("false_downgrade_status")),
                    "rows_checked": int_value(result.get("false_downgrade_rows_checked")),
                    "p0_blockers_count": 0,
                    "p1_warnings_count": 0,
                }
            )
        if "sector_internal_ranking_status" in result:
            reports.append(
                {
                    "name": "sector_internal_ranking",
                    "status": string_value(result.get("sector_internal_ranking_status")),
                    "rows_checked": int_value(result.get("sector_internal_ranking_rows_checked")),
                    "p0_blockers_count": 0,
                    "p1_warnings_count": 0,
                }
            )
        if reports:
            return reports

    outputs = list_value(summary.get("outputs"))
    return [
        {
            "name": f"report_{index}",
            "status": "",
            "rows_checked": 0,
            "p0_blockers_count": 0,
            "p1_warnings_count": 0,
        }
        for index, _ in enumerate(outputs, start=1)
    ]


def normalize_report_item(index: int, item: Any) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {
            "name": f"report_{index}",
            "status": "",
            "rows_checked": 0,
            "p0_blockers_count": 0,
            "p1_warnings_count": 0,
        }
    return {
        "name": string_value(item.get("sample_type")) or string_value(item.get("name")) or f"report_{index}",
        "status": string_value(item.get("status")),
        "rows_checked": int_value(item.get("rows_checked")),
        "p0_blockers_count": len(list_value(item.get("p0_blockers")))
        if "p0_blockers" in item
        else int_value(item.get("p0_blockers_count")),
        "p1_warnings_count": len(list_value(item.get("p1_warnings")))
        if "p1_warnings" in item
        else int_value(item.get("p1_warnings_count")),
        "advisory_only": bool_value(item.get("advisory_only")),
        "training_allowed": bool_value(item.get("training_allowed")),
        "affects_stable_trading": bool_value(item.get("affects_stable_trading")),
    }


def summary_advisory_only(summary: dict[str, Any], reports: list[dict[str, Any]]) -> bool | None:
    report_values = [report.get("advisory_only") for report in reports if report.get("advisory_only") is not None]
    if report_values:
        return all(value is True for value in report_values)

    review_checklist = dict_value(summary.get("review_checklist"))
    if "readonly_advisory" in review_checklist:
        return bool_value(review_checklist.get("readonly_advisory"))
    return None


def summary_training_allowed_false(summary: dict[str, Any], reports: list[dict[str, Any]]) -> bool:
    report_values = [
        report.get("training_allowed") for report in reports if report.get("training_allowed") is not None
    ]
    if report_values:
        return all(value is False for value in report_values)

    boundary = dict_value(summary.get("boundary"))
    if "no_training" in boundary:
        return bool_value(boundary.get("no_training")) is True
    return False


def summary_affects_stable_trading_false(summary: dict[str, Any], reports: list[dict[str, Any]]) -> bool:
    report_values = [
        report.get("affects_stable_trading")
        for report in reports
        if report.get("affects_stable_trading") is not None
    ]
    if report_values:
        return all(value is False for value in report_values)

    review_checklist = dict_value(summary.get("review_checklist"))
    if "affects_stable_trading" in review_checklist:
        return bool_value(review_checklist.get("affects_stable_trading")) is False
    return False


def review_flags(
    status: str,
    p0_blockers: list[Any],
    p1_warnings: list[Any],
    advisory_only: bool | None,
    training_allowed: bool | None,
    affects_stable_trading: bool | None,
) -> list[str]:
    flags: list[str] = []
    if status != "passed":
        flags.append(NEEDS_REVIEW)
    if p0_blockers or training_allowed is True or affects_stable_trading is True or advisory_only is False:
        flags.append(P0_REVIEW_REQUIRED)
    elif p1_warnings:
        flags.append(NEEDS_REVIEW)
    return dedupe(flags)


def review_status(flags: list[str]) -> str:
    if P0_REVIEW_REQUIRED in flags:
        return P0_REVIEW_REQUIRED
    if NEEDS_REVIEW in flags:
        return NEEDS_REVIEW
    return OK


def format_text(summary: dict[str, Any]) -> str:
    if summary["kind"] == "report":
        lines = [
            f"review_status: {summary['review_status']}",
            f"status: {summary['status']}",
            f"sample_type: {summary['sample_type']}",
            f"rows_checked: {summary['rows_checked']}",
            f"intake_passed: {str(summary['intake_passed']).lower()}",
            f"schema_passed: {str(summary['schema_passed']).lower()}",
            f"p0_blockers count: {summary['p0_blockers_count']}",
            f"p1_warnings count: {summary['p1_warnings_count']}",
            f"advisory_only: {str(summary['advisory_only']).lower()}",
            f"training_allowed: {str(summary['training_allowed']).lower()}",
            f"affects_stable_trading: {str(summary['affects_stable_trading']).lower()}",
        ]
    else:
        lines = [
            f"review_status: {summary['review_status']}",
            f"included report count: {summary['included_report_count']}",
            "report statuses: " + json.dumps(summary["report_statuses"], ensure_ascii=False, sort_keys=True),
            f"all_passed: {str(summary['all_passed']).lower()}",
            f"has_p0: {str(summary['has_p0']).lower()}",
            f"has_p1: {str(summary['has_p1']).lower()}",
            f"advisory_only: {str(summary['advisory_only']).lower()}",
            f"training_allowed=false: {str(summary['training_allowed_false']).lower()}",
            f"affects_stable_trading=false: {str(summary['affects_stable_trading_false']).lower()}",
        ]
    if summary["review_flags"]:
        lines.append("review_flags: " + ", ".join(summary["review_flags"]))
    return "\n".join(lines)


def print_summary(summary: dict[str, Any], output_format: str) -> None:
    if output_format == "json":
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_text(summary))


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def int_value(value: Any) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def bool_value(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read Lab-only table ML dry validation JSON reports.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--report", type=Path)
    source.add_argument("--summary", type=Path)
    parser.add_argument("--format", choices=["text", "json"], default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.report:
        summary = summarize_report(args.report)
    else:
        summary = summarize_summary(args.summary)
    print_summary(summary, args.format)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
