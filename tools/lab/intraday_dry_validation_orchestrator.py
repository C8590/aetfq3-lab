from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_5m_schema_validator import validate_schema
from tools.lab.intraday_dry_validation_intake_checker import check_manifest, is_forbidden_feature_name
from tools.lab.intraday_mock_tensor_smoke import build_sequence_tensor, scan_forbidden_features
from tools.lab.intraday_watch_state_machine_dryrun import load_events, transition_state


REPORT_TYPE = "intraday_synthetic_dry_validation"
TASK_SCOPE = "Lab-only synthetic intraday dry validation"
DEFAULT_EVENTS_FIXTURE = Path("tests/fixtures/aetfq3_lab/mock_intraday_watch_events.json")
ALLOWED_OUTPUT_ROOT = Path(".local_research_outputs/aetfq3_lab/intraday_synthetic_dry_validation")


class OrchestratorError(RuntimeError):
    pass


def orchestrate(manifest_path: Path, out_dir: Path, repo_root: Path | None = None) -> dict[str, Any]:
    repo_root = (repo_root or REPO_ROOT).resolve()
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    manifest_path = resolve_path(manifest_path, repo_root)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_path = resolve_path(Path(str(manifest.get("sample_path", ""))), repo_root)

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []

    intake_result = check_manifest(manifest_path, repo_root=repo_root)
    p0_blockers.extend(intake_result.p0_blockers)
    p1_warnings.extend(intake_result.p1_warnings)
    intake_passed = intake_result.ok

    schema_result = validate_schema(sample_path, manifest_path, repo_root=repo_root) if intake_passed else None
    schema_passed = bool(schema_result and schema_result.ok)
    if schema_result:
        p0_blockers.extend(schema_result.p0_blockers)
        p1_warnings.extend(schema_result.p1_warnings)

    feature_columns = [item for item in manifest.get("feature_columns", []) if isinstance(item, str)]
    forbidden_feature_report = run_forbidden_feature_scan(feature_columns)
    forbidden_feature_passed = bool(forbidden_feature_report["passed"])
    if not forbidden_feature_passed:
        p0_blockers.append(
            "forbidden feature scan failed: " + ", ".join(forbidden_feature_report["forbidden_columns"])
        )

    tensor_report = run_tensor_shape_validation(sample_path, feature_columns)
    tensor_shape_passed = bool(tensor_report["passed"])
    if not tensor_shape_passed:
        p0_blockers.extend(tensor_report["p0_blockers"])

    state_report = run_state_machine_validation(resolve_path(DEFAULT_EVENTS_FIXTURE, repo_root))
    state_machine_passed = bool(state_report["passed"])
    if not state_machine_passed:
        p0_blockers.extend(state_report["p0_blockers"])

    rows_checked = schema_result.rows_checked if schema_result else 0
    report = {
        "report_type": REPORT_TYPE,
        "task_scope": TASK_SCOPE,
        "status": "passed" if not p0_blockers else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path.relative_to(repo_root)),
        "sample_path": str(sample_path.relative_to(repo_root)),
        "lab_only": True,
        "mock_only": True,
        "synthetic_only": True,
        "real_intraday_data_used": False,
        "no_stable": True,
        "no_qmt": True,
        "no_order_intent": True,
        "no_output": True,
        "no_lab_advisory": True,
        "no_training": True,
        "checkpoint_saved": False,
        "model_saved": False,
        "intake_passed": intake_passed,
        "schema_passed": schema_passed,
        "forbidden_feature_passed": forbidden_feature_passed,
        "tensor_shape_passed": tensor_shape_passed,
        "state_machine_passed": state_machine_passed,
        "order_intent_generated": False,
        "qmt_allowed": False,
        "stable_effect_allowed": False,
        "rows_checked": rows_checked,
        "trade_date_count": schema_result.trade_date_count if schema_result else 0,
        "etf_count": schema_result.etf_count if schema_result else 0,
        "min_bars_per_etf_day": schema_result.min_bars_per_etf_day if schema_result else 0,
        "batch_size": tensor_report["batch_size"],
        "time_steps": tensor_report["time_steps"],
        "feature_count": tensor_report["feature_count"],
        "target_count": tensor_report["target_count"],
        "visited_states": state_report["visited_states"],
        "terminal_state": state_report["terminal_state"],
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }
    write_reports(report, resolved_out_dir, tensor_report, state_report)
    return report


def resolve_path(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def resolve_output_dir(out_dir: Path, repo_root: Path) -> Path:
    resolved = resolve_path(out_dir, repo_root).resolve()
    allowed_root = (repo_root / ALLOWED_OUTPUT_ROOT).resolve()
    try:
        resolved.relative_to(allowed_root)
    except ValueError as exc:
        raise OrchestratorError(f"out-dir must be under {ALLOWED_OUTPUT_ROOT}") from exc
    return resolved


def run_forbidden_feature_scan(feature_columns: Sequence[str]) -> dict[str, Any]:
    explicit_scan = scan_forbidden_features(feature_columns)
    pattern_hits = [column for column in feature_columns if is_forbidden_feature_name(column)]
    forbidden = sorted(set(explicit_scan["forbidden_columns"]) | set(pattern_hits))
    return {
        "passed": not forbidden,
        "forbidden_columns": forbidden,
        "feature_columns": list(feature_columns),
    }


def run_tensor_shape_validation(sample_path: Path, feature_columns: Sequence[str]) -> dict[str, Any]:
    try:
        bundle = build_sequence_tensor(sample_path, feature_columns=feature_columns)
    except Exception as exc:  # pragma: no cover - exercised through failure tests if needed.
        return {
            "passed": False,
            "batch_size": 0,
            "time_steps": 0,
            "feature_count": 0,
            "target_count": 0,
            "p0_blockers": [f"tensor shape validation failed: {exc}"],
            "checkpoint_saved": False,
            "model_saved": False,
        }

    batch_size, time_steps, feature_count = bundle.x.shape
    target_count = bundle.y.shape[1]
    return {
        "report_type": "intraday_synthetic_tensor_shape_report",
        "passed": True,
        "batch_size": int(batch_size),
        "time_steps": int(time_steps),
        "feature_count": int(feature_count),
        "target_count": int(target_count),
        "feature_columns": bundle.feature_columns,
        "target_columns": bundle.target_columns,
        "sequence_keys": bundle.sequence_keys,
        "no_training": True,
        "checkpoint_saved": False,
        "model_saved": False,
        "p0_blockers": [],
    }


def run_state_machine_validation(events_path: Path) -> dict[str, Any]:
    try:
        events = load_events(events_path)
        state = "WAIT_OPEN"
        visited_states = [state]
        transitions: list[dict[str, str]] = []
        for item in events:
            event_id = str(item["event_id"])
            next_state = transition_state(state, event_id)
            transitions.append({"from_state": state, "event": event_id, "to_state": next_state})
            state = next_state
            if state not in visited_states:
                visited_states.append(state)
    except Exception as exc:  # pragma: no cover - exercised through failure tests if needed.
        return {
            "passed": False,
            "visited_states": [],
            "terminal_state": "",
            "p0_blockers": [f"state machine validation failed: {exc}"],
        }

    return {
        "report_type": "intraday_synthetic_watch_dryrun_report",
        "passed": True,
        "events_file": str(events_path.relative_to(REPO_ROOT)),
        "event_count": len(events),
        "visited_states": visited_states,
        "terminal_state": state,
        "transitions": transitions,
        "advisory_only": True,
        "qmt_allowed": False,
        "order_intent_generated": False,
        "stable_effect_allowed": False,
        "p0_blockers": [],
    }


def write_reports(
    report: dict[str, Any],
    out_dir: Path,
    tensor_report: dict[str, Any],
    state_report: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intraday_synthetic_dry_validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "intraday_synthetic_tensor_shape_report.json").write_text(
        json.dumps(tensor_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "intraday_synthetic_watch_dryrun_report.json").write_text(
        json.dumps(state_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。",
        "",
        "# Intraday Synthetic Dry Validation Report",
        "",
        f"- status: {report['status']}",
        f"- intake_passed: {str(report['intake_passed']).lower()}",
        f"- schema_passed: {str(report['schema_passed']).lower()}",
        f"- forbidden_feature_passed: {str(report['forbidden_feature_passed']).lower()}",
        f"- tensor_shape_passed: {str(report['tensor_shape_passed']).lower()}",
        f"- state_machine_passed: {str(report['state_machine_passed']).lower()}",
        f"- rows_checked: {report['rows_checked']}",
        f"- batch_size: {report['batch_size']}",
        f"- time_steps: {report['time_steps']}",
        f"- feature_count: {report['feature_count']}",
        f"- target_count: {report['target_count']}",
        f"- visited_states: {', '.join(report['visited_states'])}",
        "- boundary: mock/synthetic only; no Stable, QMT, OrderIntent, output/, lab_advisory, training, model save, or checkpoint.",
    ]
    (out_dir / "intraday_synthetic_dry_validation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only synthetic intraday dry validation gates.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = orchestrate(args.manifest, args.out_dir)
    except OrchestratorError as exc:
        print(json.dumps({"status": "failed", "p0_blockers": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": report["status"],
        "intake_passed": report["intake_passed"],
        "schema_passed": report["schema_passed"],
        "forbidden_feature_passed": report["forbidden_feature_passed"],
        "tensor_shape_passed": report["tensor_shape_passed"],
        "state_machine_passed": report["state_machine_passed"],
        "rows_checked": report["rows_checked"],
        "batch_size": report["batch_size"],
        "time_steps": report["time_steps"],
        "feature_count": report["feature_count"],
        "target_count": report["target_count"],
        "visited_states": report["visited_states"],
    }, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
