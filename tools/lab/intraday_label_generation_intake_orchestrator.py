from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.lab.intraday_label_manifest_leakage_checker import check_manifest as check_label_manifest


ALLOWED_OUTPUT_BASE = Path(".local_research_outputs/aetfq3_lab")
ALLOWED_OUTPUT_DIR_PREFIX = "intraday_label_generation_intake"
READY_FOR_LABEL_GENERATION_DRY_RUN = "READY_FOR_LABEL_GENERATION_DRY_RUN"
BLOCKED_MISSING_FUTURE_WINDOW_SOURCE = "BLOCKED_MISSING_FUTURE_WINDOW_SOURCE"
BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA = "BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA"
BLOCKED_MANIFEST_P0 = "BLOCKED_MANIFEST_P0"
BLOCKED_HASH_OR_SOURCE_NOTE = "BLOCKED_HASH_OR_SOURCE_NOTE"
BLOCKED_BOUNDARY_VIOLATION = "BLOCKED_BOUNDARY_VIOLATION"
BOUNDARY_FALSE_FIELDS = [
    "supervised_training_allowed",
    "training_allowed",
    "stable_effect_allowed",
    "contains_order_intent",
    "contains_live_order",
    "contains_secret",
]
REQUIRED_PUBLIC_ARTIFACTS = [
    "intraday_5m_export.csv",
    "EXPORT_MANIFEST.json",
    "source_note.md",
    "SHA256SUMS.txt",
]


class IntakeOrchestratorError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise IntakeOrchestratorError(f"JSON parse failed for {path}: {exc}") from exc
    except OSError as exc:
        raise IntakeOrchestratorError(f"JSON cannot be read: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntakeOrchestratorError(f"JSON root must be object: {path}")
    return payload


def resolve_repo_path(raw_path: str | None, repo_root: Path = REPO_ROOT) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    return path if path.is_absolute() else repo_root / path


def resolve_output_dir(out_dir: Path, repo_root: Path = REPO_ROOT) -> Path:
    resolved = (repo_root / out_dir if not out_dir.is_absolute() else out_dir).resolve()
    repo_root = repo_root.resolve()
    allowed_base = (repo_root / ALLOWED_OUTPUT_BASE).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise IntakeOrchestratorError("out-dir must stay inside repository root") from exc
    try:
        relative_to_base = resolved.relative_to(allowed_base)
    except ValueError as exc:
        raise IntakeOrchestratorError(f"out-dir must be under {ALLOWED_OUTPUT_BASE}") from exc
    first_part = relative_to_base.parts[0] if relative_to_base.parts else ""
    if not first_part.startswith(ALLOWED_OUTPUT_DIR_PREFIX):
        raise IntakeOrchestratorError(
            f"out-dir must be under {ALLOWED_OUTPUT_BASE}/{ALLOWED_OUTPUT_DIR_PREFIX}*"
        )
    return resolved


def orchestrate_intake(manifest_path: Path, out_dir: Path, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    resolved_manifest = resolve_repo_path(str(manifest_path), repo_root)
    if resolved_manifest is None:
        raise IntakeOrchestratorError("manifest path is required")
    resolved_out_dir = resolve_output_dir(out_dir, repo_root)
    manifest = load_json(resolved_manifest)

    manifest_check = check_label_manifest(resolved_manifest)
    boundary_check = run_boundary_check(manifest)
    hash_source_check = run_hash_source_note_check(manifest, repo_root)
    tensor_report_check = run_no_label_tensor_report_check(manifest, repo_root)
    data_quality_check = run_data_quality_report_check(manifest, repo_root)
    future_window_check = run_future_window_readiness_check(manifest, repo_root)

    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    p0_blockers.extend(manifest_check.p0_blockers)
    p1_warnings.extend(manifest_check.p1_warnings)
    for check in (boundary_check, hash_source_check, tensor_report_check, data_quality_check, future_window_check):
        p0_blockers.extend(check["p0_blockers"])
        p1_warnings.extend(check["p1_warnings"])

    readiness_decision = decide_readiness(
        manifest_check_passed=manifest_check.ok,
        boundary_passed=boundary_check["passed"],
        hash_source_passed=hash_source_check["passed"],
        tensor_report_passed=tensor_report_check["passed"],
        data_quality_passed=data_quality_check["passed"],
        future_window_presence_passed=future_window_check["presence_gate_passed"],
        future_window_coverage_passed=future_window_check["coverage_gate_passed"],
    )
    raw_presence_decision = decide_readiness(
        manifest_check_passed=manifest_check.ok,
        boundary_passed=boundary_check["passed"],
        hash_source_passed=hash_source_check["passed"],
        tensor_report_passed=tensor_report_check["passed"],
        data_quality_passed=data_quality_check["passed"],
        future_window_presence_passed=future_window_check["presence_gate_passed"],
        future_window_coverage_passed=True,
    )
    report = {
        "report_type": "intraday_label_generation_intake",
        "status": "passed" if readiness_decision == READY_FOR_LABEL_GENERATION_DRY_RUN else "blocked",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "readiness_decision": readiness_decision,
        "raw_presence_decision": raw_presence_decision,
        "effective_readiness_decision": readiness_decision,
        "presence_gate_passed": future_window_check["presence_gate_passed"],
        "coverage_gate_passed": future_window_check["coverage_gate_passed"],
        "coverage_sufficient": future_window_check["coverage_sufficient"],
        "missing_future_dates_by_etf": future_window_check["missing_future_dates_by_etf"],
        "lab_only": True,
        "intake_only": True,
        "label_generation_performed": False,
        "no_training": True,
        "no_torchrun": True,
        "checkpoint_saved": False,
        "model_saved": False,
        "no_qmt": True,
        "no_order_intent": True,
        "no_stable": True,
        "no_output": True,
        "no_lab_advisory": True,
        "manifest_leakage_check": manifest_check.to_summary(),
        "hash_source_note_check": hash_source_check,
        "no_label_tensor_report_check": tensor_report_check,
        "data_quality_report_check": data_quality_check,
        "future_window_readiness_check": future_window_check,
        "boundary_check": boundary_check,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
        "readiness_does_not_authorize_supervised_training": True,
        "not_trading_advice": True,
    }
    write_reports(report, resolved_out_dir)
    return report


def decide_readiness(
    manifest_check_passed: bool,
    boundary_passed: bool,
    hash_source_passed: bool,
    tensor_report_passed: bool,
    data_quality_passed: bool,
    future_window_presence_passed: bool,
    future_window_coverage_passed: bool,
) -> str:
    if not boundary_passed:
        return BLOCKED_BOUNDARY_VIOLATION
    if not manifest_check_passed:
        return BLOCKED_MANIFEST_P0
    if not hash_source_passed:
        return BLOCKED_HASH_OR_SOURCE_NOTE
    if not tensor_report_passed or not data_quality_passed:
        return BLOCKED_BOUNDARY_VIOLATION
    if not future_window_presence_passed or not future_window_coverage_passed:
        return BLOCKED_INSUFFICIENT_FUTURE_WINDOW_DATA
    return READY_FOR_LABEL_GENERATION_DRY_RUN


def run_boundary_check(manifest: dict[str, Any]) -> dict[str, Any]:
    p0_blockers: list[str] = []
    for field_name in BOUNDARY_FALSE_FIELDS:
        if manifest.get(field_name) is not False:
            p0_blockers.append(f"{field_name} must be false")
    return {
        "passed": not p0_blockers,
        "checked_fields": BOUNDARY_FALSE_FIELDS,
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_hash_source_note_check(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    artifact_dir = resolve_repo_path(str(manifest.get("public_artifact_dir", "")), repo_root)
    if artifact_dir is None or not artifact_dir.exists():
        return {
            "passed": False,
            "artifact_dir": str(manifest.get("public_artifact_dir")),
            "required_artifacts": REQUIRED_PUBLIC_ARTIFACTS,
            "p0_blockers": ["public_artifact_dir missing or does not exist"],
            "p1_warnings": [],
        }

    found_files: dict[str, bool] = {}
    for filename in REQUIRED_PUBLIC_ARTIFACTS:
        path = artifact_dir / filename
        found_files[filename] = path.exists()
        if not path.exists():
            p0_blockers.append(f"missing public artifact: {filename}")
        elif filename in {"source_note.md", "SHA256SUMS.txt"} and not path.read_text(encoding="utf-8").strip():
            p0_blockers.append(f"public artifact is empty: {filename}")

    hash_report_path = resolve_repo_path(str(manifest.get("hash_source_validation_report_path", "")), repo_root)
    hash_report_status = "not_checked"
    hash_matched = None
    if hash_report_path and hash_report_path.exists():
        try:
            hash_report = load_json(hash_report_path)
            hash_report_status = str(hash_report.get("status"))
            hash_matched = hash_report.get("hash_matched")
            if hash_report.get("status") != "passed" or hash_report.get("hash_matched") is not True:
                p0_blockers.append("hash_source_validation_report is not passed with hash_matched=true")
        except IntakeOrchestratorError as exc:
            p0_blockers.append(str(exc))
    else:
        p1_warnings.append("hash_source_validation_report_path missing; presence check used SHA256SUMS/source_note only")

    return {
        "passed": not p0_blockers,
        "artifact_dir": str(artifact_dir),
        "found_files": found_files,
        "hash_source_validation_report_path": str(hash_report_path) if hash_report_path else None,
        "hash_report_status": hash_report_status,
        "hash_matched": hash_matched,
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }


def run_no_label_tensor_report_check(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    p0_blockers: list[str] = []
    report_path = resolve_repo_path(str(manifest.get("no_label_tensor_report_path", "")), repo_root)
    if report_path is None or not report_path.exists():
        return {
            "passed": False,
            "report_path": str(manifest.get("no_label_tensor_report_path")),
            "p0_blockers": ["no_label_tensor_report_path missing or does not exist"],
            "p1_warnings": [],
        }
    try:
        report = load_json(report_path)
    except IntakeOrchestratorError as exc:
        return {"passed": False, "report_path": str(report_path), "p0_blockers": [str(exc)], "p1_warnings": []}

    expected = {
        "status": "passed",
        "tensor_shape_passed": True,
        "labels_required": False,
        "target_count": 0,
        "no_training": True,
        "no_qmt": True,
        "no_order_intent": True,
        "no_stable": True,
        "model_saved": False,
        "checkpoint_saved": False,
    }
    for field_name, expected_value in expected.items():
        if report.get(field_name) != expected_value:
            p0_blockers.append(f"no-label tensor report {field_name} must be {expected_value!r}")
    if report.get("p0_blockers"):
        p0_blockers.append("no-label tensor report contains p0_blockers")
    return {
        "passed": not p0_blockers,
        "report_path": str(report_path),
        "rows_checked": report.get("rows_checked"),
        "batch_size": report.get("batch_size"),
        "min_time_steps": report.get("min_time_steps"),
        "max_time_steps": report.get("max_time_steps"),
        "feature_count": report.get("feature_count"),
        "p0_blockers": p0_blockers,
        "p1_warnings": [],
    }


def run_data_quality_report_check(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    p0_blockers: list[str] = []
    report_path = resolve_repo_path(str(manifest.get("data_quality_report_path", "")), repo_root)
    if report_path is None or not report_path.exists():
        return {
            "passed": False,
            "report_path": str(manifest.get("data_quality_report_path")),
            "p0_blockers": ["data_quality_report_path missing or does not exist"],
            "p1_warnings": [],
        }
    try:
        report = load_json(report_path)
    except IntakeOrchestratorError as exc:
        return {"passed": False, "report_path": str(report_path), "p0_blockers": [str(exc)], "p1_warnings": []}

    required_true = [
        "required_columns_present",
        "ohlc_consistency_passed",
        "volume_amount_nonnegative_passed",
        "datetime_monotonic_per_etf_day",
        "calculated_vwap_possible",
        "no_training",
        "no_trading_advice",
    ]
    if report.get("status") != "passed":
        p0_blockers.append("data quality report status must be passed")
    for field_name in required_true:
        if report.get(field_name) is not True:
            p0_blockers.append(f"data quality report {field_name} must be true")
    if report.get("p0_blockers"):
        p0_blockers.append("data quality report contains p0_blockers")
    return {
        "passed": not p0_blockers,
        "report_path": str(report_path),
        "row_count": report.get("row_count"),
        "etf_count": report.get("etf_count"),
        "trade_date_count": report.get("trade_date_count"),
        "p0_blockers": p0_blockers,
        "p1_warnings": list(report.get("p1_warnings", [])) if isinstance(report.get("p1_warnings"), list) else [],
    }


def run_future_window_readiness_check(manifest: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    p0_blockers: list[str] = []
    p1_warnings: list[str] = []
    source_kind = manifest.get("future_window_source_kind")
    source_path = resolve_repo_path(str(manifest.get("future_window_source_path", "")), repo_root)
    source_note_path = resolve_repo_path(str(manifest.get("future_window_source_note_path", "")), repo_root)
    hash_path = resolve_repo_path(str(manifest.get("future_window_hash_path", "")), repo_root)

    if source_kind not in {"human_exported_future_window", "public_future_window", "public_market_data_export"}:
        p0_blockers.append(
            "future_window_source_kind must be human_exported_future_window, public_future_window, or public_market_data_export"
        )
    for field_name, path in (
        ("future_window_source_path", source_path),
        ("future_window_source_note_path", source_note_path),
        ("future_window_hash_path", hash_path),
    ):
        if path is None or not path.exists():
            p0_blockers.append(f"{field_name} missing or does not exist")
        elif not path.is_file():
            p0_blockers.append(f"{field_name} must point to a file")
        elif path.stat().st_size == 0:
            p0_blockers.append(f"{field_name} is empty")

    if manifest.get("label_generation_authorized") is not True:
        p0_blockers.append("label_generation_authorized must be true before label generation dry-run")
    if manifest.get("label_generated") is not False:
        p0_blockers.append("label_generated must remain false at intake-only stage")
    if manifest.get("readiness_only") is not True:
        p1_warnings.append("readiness_only should be true for intake-only orchestration")

    presence_gate_passed = not p0_blockers
    coverage_check = run_future_window_coverage_audit(manifest, source_path, repo_root) if presence_gate_passed else {
        "coverage_gate_passed": False,
        "coverage_sufficient": False,
        "required_etf_codes": [],
        "required_future_dates": [],
        "required_coverage_end": manifest.get("future_window_required_coverage_end"),
        "missing_future_dates_by_etf": {},
        "coverage_p0_blockers": ["future-window presence gate failed; coverage audit not runnable"],
    }
    p0_blockers.extend(coverage_check["coverage_p0_blockers"])

    return {
        "passed": not p0_blockers,
        "presence_gate_passed": presence_gate_passed,
        "coverage_gate_passed": coverage_check["coverage_gate_passed"],
        "coverage_sufficient": coverage_check["coverage_sufficient"],
        "future_window_source_kind": source_kind,
        "future_window_source_path": str(source_path) if source_path else None,
        "future_window_source_note_path": str(source_note_path) if source_note_path else None,
        "future_window_hash_path": str(hash_path) if hash_path else None,
        "required_etf_codes": coverage_check["required_etf_codes"],
        "required_future_dates": coverage_check["required_future_dates"],
        "required_coverage_end": coverage_check["required_coverage_end"],
        "missing_future_dates_by_etf": coverage_check["missing_future_dates_by_etf"],
        "p0_blockers": p0_blockers,
        "p1_warnings": p1_warnings,
    }


def run_future_window_coverage_audit(
    manifest: dict[str, Any],
    source_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    p0_blockers: list[str] = []
    required_dates = string_list(manifest.get("future_window_required_dates"))
    required_coverage_end = manifest.get("future_window_required_coverage_end")
    if required_coverage_end and str(required_coverage_end) not in required_dates:
        required_dates.append(str(required_coverage_end))
    required_dates = sorted(set(required_dates))
    if not required_dates:
        p0_blockers.append("future_window_required_dates or future_window_required_coverage_end must be provided")

    required_etf_codes = resolve_required_etf_codes(manifest, repo_root, source_path)
    if not required_etf_codes:
        p0_blockers.append("required ETF codes cannot be resolved from manifest or public 5m export")

    observed_by_etf: dict[str, set[str]] = {}
    missing_future_dates_by_etf: dict[str, list[str]] = {}
    source_columns: list[str] = []
    if source_path is None or not source_path.exists():
        p0_blockers.append("future_window_source_path missing or does not exist")
    else:
        try:
            with source_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                source_columns = list(reader.fieldnames or [])
                missing_columns = [column for column in ("trade_date", "etf_code") if column not in source_columns]
                if missing_columns:
                    p0_blockers.append("future-window daily source missing required columns: " + ", ".join(missing_columns))
                else:
                    for row in reader:
                        etf_code = str(row.get("etf_code", "")).strip()
                        trade_date = str(row.get("trade_date", "")).strip()
                        if etf_code and trade_date:
                            observed_by_etf.setdefault(etf_code, set()).add(trade_date)
        except OSError as exc:
            p0_blockers.append(f"future-window daily source cannot be read: {exc}")

    for etf_code in required_etf_codes:
        observed_dates = observed_by_etf.get(etf_code, set())
        missing = [trade_date for trade_date in required_dates if trade_date not in observed_dates]
        if missing:
            missing_future_dates_by_etf[etf_code] = missing

    if missing_future_dates_by_etf:
        p0_blockers.append("future-window coverage missing required dates for one or more ETFs")

    manifest_coverage_flag = manifest.get("future_window_coverage_sufficient")
    if manifest_coverage_flag is True and missing_future_dates_by_etf:
        p0_blockers.append("future_window_coverage_sufficient=true conflicts with coverage audit")

    coverage_gate_passed = not p0_blockers
    return {
        "coverage_gate_passed": coverage_gate_passed,
        "coverage_sufficient": coverage_gate_passed,
        "required_etf_codes": required_etf_codes,
        "required_future_dates": required_dates,
        "required_coverage_end": str(required_coverage_end) if required_coverage_end else None,
        "source_columns": source_columns,
        "missing_future_dates_by_etf": missing_future_dates_by_etf,
        "coverage_p0_blockers": p0_blockers,
    }


def resolve_required_etf_codes(manifest: dict[str, Any], repo_root: Path, source_path: Path | None) -> list[str]:
    for field_name in ("future_window_required_etf_codes", "required_etf_codes", "etf_codes"):
        values = string_list(manifest.get(field_name))
        if values:
            return sorted(set(values))

    public_artifact_dir = resolve_repo_path(str(manifest.get("public_artifact_dir", "")), repo_root)
    public_5m_path = public_artifact_dir / "intraday_5m_export.csv" if public_artifact_dir else None
    values = read_etf_codes_from_csv(public_5m_path) if public_5m_path and public_5m_path.exists() else []
    if values:
        return values

    return read_etf_codes_from_csv(source_path) if source_path and source_path.exists() else []


def read_etf_codes_from_csv(path: Path | None) -> list[str]:
    if path is None:
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            if "etf_code" not in (reader.fieldnames or []):
                return []
            return sorted({str(row.get("etf_code", "")).strip() for row in reader if str(row.get("etf_code", "")).strip()})
    except OSError:
        return []


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "intraday_label_generation_intake_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    decision = {
        "readiness_decision": report["readiness_decision"],
        "raw_presence_decision": report["raw_presence_decision"],
        "effective_readiness_decision": report["effective_readiness_decision"],
        "status": report["status"],
        "presence_gate_passed": report["presence_gate_passed"],
        "coverage_gate_passed": report["coverage_gate_passed"],
        "coverage_sufficient": report["coverage_sufficient"],
        "missing_future_dates_by_etf": report["missing_future_dates_by_etf"],
        "label_generation_performed": False,
        "model_training_allowed": False,
        "qmt_allowed": False,
        "order_intent_allowed": False,
        "stable_allowed": False,
        "p0_blockers": report["p0_blockers"],
    }
    (out_dir / "readiness_decision.json").write_text(
        json.dumps(decision, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。",
        "",
        "# Intraday Label Generation Intake Report",
        "",
        f"- status: {report['status']}",
        f"- readiness_decision: {report['readiness_decision']}",
        f"- raw_presence_decision: {report['raw_presence_decision']}",
        f"- effective_readiness_decision: {report['effective_readiness_decision']}",
        f"- manifest_leakage_status: {report['manifest_leakage_check']['status']}",
        f"- hash_source_note_passed: {str(report['hash_source_note_check']['passed']).lower()}",
        f"- no_label_tensor_report_passed: {str(report['no_label_tensor_report_check']['passed']).lower()}",
        f"- data_quality_report_passed: {str(report['data_quality_report_check']['passed']).lower()}",
        f"- future_window_presence_gate_passed: {str(report['presence_gate_passed']).lower()}",
        f"- future_window_coverage_gate_passed: {str(report['coverage_gate_passed']).lower()}",
        f"- coverage_sufficient: {str(report['coverage_sufficient']).lower()}",
        f"- missing_future_dates_by_etf: {json.dumps(report['missing_future_dates_by_etf'], ensure_ascii=False, sort_keys=True)}",
        "- boundary: intake-only; no label generation, training, QMT, OrderIntent, Stable, output/, lab_advisory, checkpoint, or trading advice.",
    ]
    (out_dir / "intraday_label_generation_intake_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday label generation intake orchestration.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = orchestrate_intake(args.manifest, args.out_dir)
    except IntakeOrchestratorError as exc:
        print(json.dumps({"status": "failed", "readiness_decision": BLOCKED_BOUNDARY_VIOLATION, "p0_blockers": [str(exc)]}, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps({
        "status": report["status"],
        "readiness_decision": report["readiness_decision"],
        "manifest_path": report["manifest_path"],
        "p0_blockers": report["p0_blockers"],
        "p1_warnings": report["p1_warnings"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
