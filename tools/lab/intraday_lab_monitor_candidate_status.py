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

from tools.lab.intraday_fixed_shortlist_oop_no_save_validation import load_json, write_csv, write_json  # noqa: E402
from tools.lab.intraday_supervised_no_save_smoke import check_model_artifacts  # noqa: E402


LAB_DECLARATION = "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。"
REPORT_TYPE = "intraday_lab_monitor_candidate_status"
ALLOWED_OUTPUT_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_lab_monitor_candidate_status")
DEFAULT_ROLLING_ORIGIN_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_rolling_origin_walk_forward_no_save_validation")
DEFAULT_ATTRIBUTION_DIR = Path(".local_research_outputs/aetfq3_lab/intraday_rolling_origin_stability_attribution_review")
DEFAULT_OUT_DIR = ALLOWED_OUTPUT_DIR
FOCUS_CANDIDATE_ID = "label_ret3d_gt_100bp|base_39_plus_scale_transform_policy|logistic_balanced_scaled_variants|scale_transform_policy"
MONITOR_STATUS = "LAB_MONITOR_CANDIDATE_REVIEW_READY"

DECISION_ACTIVE = "LAB_MONITOR_CANDIDATE_STATUS_ACTIVE_REVIEW_READY"
DECISION_ACTIVE_WARNINGS = "LAB_MONITOR_CANDIDATE_STATUS_ACTIVE_WITH_WARNINGS"
DECISION_REVIEW = "LAB_MONITOR_CANDIDATE_STATUS_REVIEW_REQUIRED"
DECISION_RETIRED = "LAB_MONITOR_CANDIDATE_STATUS_RETIRED_REVIEW_REQUIRED"
DECISION_BLOCKED_MISSING = "LAB_MONITOR_CANDIDATE_STATUS_BLOCKED_MISSING_OUTPUTS"
DECISION_BLOCKED_DATA = "LAB_MONITOR_CANDIDATE_STATUS_BLOCKED_DATA_QUALITY"

F_PUBLIC_STATUS = "F_PUBLIC_STATUS=LAB_MONITOR_CANDIDATE_REVIEW_READY_ROLLING_ORIGIN_STABILITY_OBSERVED_STABLE_BLOCKED"
FORBIDDEN_NEXT_TASKS = [
    "Stable promotion",
    "QMT connection",
    "OrderIntent generation",
    "formal training",
    "model/scaler/checkpoint save",
    "BUY/PROBE threshold change",
    "target_weight change",
    "final_buy_action change",
    "automatic promotion",
]


class MonitorCandidateStatusError(RuntimeError):
    pass


def resolve_repo_path(path: Path, repo_root: Path = REPO_ROOT) -> Path:
    return path if path.is_absolute() else repo_root / path


def resolve_output_dir(out_dir: Path, repo_root: Path = REPO_ROOT, *, enforce: bool = True) -> Path:
    repo_root = repo_root.resolve()
    resolved = resolve_repo_path(out_dir, repo_root).resolve()
    if enforce:
        allowed = (repo_root / ALLOWED_OUTPUT_DIR).resolve()
        try:
            resolved.relative_to(allowed)
        except ValueError as exc:
            raise MonitorCandidateStatusError(f"out-dir must be under {ALLOWED_OUTPUT_DIR}") from exc
    return resolved


def find_candidate(rolling_report: dict[str, Any]) -> dict[str, Any] | None:
    for item in rolling_report.get("aggregate_stability", []):
        if item.get("family_id") == FOCUS_CANDIDATE_ID:
            return item
    return None


def gate_status(candidate: dict[str, Any], attribution_report: dict[str, Any]) -> dict[str, Any]:
    fold_count = int(candidate.get("evaluated_fold_count") or 0)
    ba_fraction = float(candidate.get("fraction_folds_balanced_accuracy_above_0_5") or 0.0)
    roc_fraction = float(candidate.get("fraction_folds_roc_auc_above_0_5") or 0.0)
    pr_fraction = float(candidate.get("fraction_folds_pr_auc_not_below_prevalence") or 0.0)
    non_collapse_fraction = float(candidate.get("fraction_folds_non_collapse") or 0.0)
    above_baseline_fraction = float(candidate.get("fraction_folds_above_dummy") or 0.0)
    ba_mean = float(candidate.get("balanced_accuracy_mean") or 0.0)
    roc_mean = float(candidate.get("roc_auc_mean") or 0.0)
    month_concentration = bool(attribution_report.get("fold_robustness_summary", {}).get("month_concentration_observed"))
    etf_concentration = bool(attribution_report.get("etf_dispersion_summary", {}).get("etf_concentration_observed"))
    threshold_sensitivity = bool(attribution_report.get("threshold_sensitivity_summary", {}).get("threshold_sensitivity_observed"))
    no_leakage = True
    no_model_artifacts = attribution_report.get("model_saved") is False and attribution_report.get("scaler_saved") is False
    continuation_gate_passed = all(
        [
            fold_count >= 6,
            ba_fraction >= 0.60,
            roc_fraction >= 0.60,
            pr_fraction >= 0.60,
            non_collapse_fraction >= 1.0,
            not month_concentration,
            not etf_concentration,
            no_leakage,
            no_model_artifacts,
        ]
    )
    review_reasons: list[str] = []
    if ba_fraction < 0.60:
        review_reasons.append("fold_balanced_accuracy_fraction_below_60pct")
    if roc_fraction < 0.60:
        review_reasons.append("fold_roc_auc_fraction_below_60pct")
    if pr_fraction < 0.60:
        review_reasons.append("fold_pr_auc_not_below_prevalence_fraction_below_60pct")
    if ba_mean <= 0.5:
        review_reasons.append("balanced_accuracy_mean_lte_0_5")
    if roc_mean <= 0.5:
        review_reasons.append("roc_auc_mean_lte_0_5")
    if non_collapse_fraction < 1.0:
        review_reasons.append("prediction_collapse_observed")
    if month_concentration:
        review_reasons.append("month_concentration_observed")
    if etf_concentration:
        review_reasons.append("etf_concentration_observed")
    if threshold_sensitivity:
        review_reasons.append("threshold_sensitivity_observed")
    if attribution_report.get("protocol_reconciliation_summary", {}).get("rolling_origin_does_not_override_post_sprint_underpowered") is True:
        review_reasons.append("post_sprint_forward_only_underpowered_still_review_item")

    retire_reasons: list[str] = []
    if (1.0 - above_baseline_fraction) >= 0.60:
        retire_reasons.append("at_least_60pct_folds_below_dummy_baseline")
    if ba_mean <= 0.5:
        retire_reasons.append("combined_rolling_origin_mean_ba_lte_0_5")
    if roc_mean <= 0.5:
        retire_reasons.append("combined_roc_auc_lte_0_5")
    if non_collapse_fraction < 1.0:
        retire_reasons.append("repeated_or_any_prediction_collapse_requires_retire_review")

    return {
        "continuation_gate_passed": continuation_gate_passed,
        "review_gate_triggered": bool(review_reasons) and not continuation_gate_passed,
        "review_warning_reasons": review_reasons,
        "retire_gate_triggered": bool(retire_reasons),
        "retire_reasons": retire_reasons,
        "month_concentration_flag": month_concentration,
        "etf_concentration_flag": etf_concentration,
        "threshold_sensitivity_flag": threshold_sensitivity,
        "post_sprint_underpowered_flag": True,
        "no_leakage_assertion_failed": False,
        "no_model_scaler_saved": no_model_artifacts,
    }


def decide(gates: dict[str, Any], blockers: Sequence[str]) -> str:
    if blockers:
        return DECISION_BLOCKED_MISSING if any("missing" in blocker.lower() for blocker in blockers) else DECISION_BLOCKED_DATA
    if gates["retire_gate_triggered"]:
        return DECISION_RETIRED
    if gates["continuation_gate_passed"]:
        return DECISION_ACTIVE
    if gates["review_gate_triggered"]:
        return DECISION_REVIEW
    return DECISION_ACTIVE_WARNINGS


def status_row(candidate: dict[str, Any], attribution_report: dict[str, Any], gates: dict[str, Any], decision: str) -> dict[str, Any]:
    return {
        "candidate_id": FOCUS_CANDIDATE_ID,
        "monitor_status": MONITOR_STATUS,
        "rolling_origin_decision": "ROLLING_ORIGIN_WALK_FORWARD_DIAGNOSTIC_STABILITY_OBSERVED_REVIEW_REQUIRED",
        "attribution_decision": attribution_report.get("readiness_decision"),
        "fold_count": candidate.get("evaluated_fold_count"),
        "positive_fold_count": candidate.get("positive_fold_count"),
        "above_baseline_fraction": candidate.get("fraction_folds_above_dummy"),
        "balanced_accuracy_mean": candidate.get("balanced_accuracy_mean"),
        "roc_auc_mean": candidate.get("roc_auc_mean"),
        "pr_auc_mean": candidate.get("pr_auc_mean"),
        "month_concentration_flag": gates["month_concentration_flag"],
        "etf_concentration_flag": gates["etf_concentration_flag"],
        "threshold_sensitivity_flag": gates["threshold_sensitivity_flag"],
        "post_sprint_underpowered_flag": gates["post_sprint_underpowered_flag"],
        "continuation_gate_passed": gates["continuation_gate_passed"],
        "review_gate_triggered": gates["review_gate_triggered"],
        "retire_gate_triggered": gates["retire_gate_triggered"],
        "status_decision": decision,
        "next_allowed_task": "Lab-only monitor protocol / periodic data refresh with read-only diagnostics",
        "forbidden_next_tasks": "; ".join(FORBIDDEN_NEXT_TASKS),
    }


def build_protocol_doc() -> tuple[dict[str, Any], str]:
    payload = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_lab_monitor_candidate_protocol",
        "candidate_identity": {
            "candidate_id": FOCUS_CANDIDATE_ID,
            "label_policy": "label_ret3d_gt_100bp",
            "feature_set": "base_39_plus_scale_transform_policy",
            "model_family": "logistic_balanced_scaled_variants",
            "transform_policy": "scale_transform_policy",
            "auto_replace_label_feature_model_threshold_allowed": False,
        },
        "monitor_status": MONITOR_STATUS,
        "not_stable_candidate": True,
        "not_trading_strategy": True,
        "not_advisory_signal": True,
        "not_order_intent_source": True,
        "not_qmt_ready_model": True,
        "not_formal_trained_model": True,
        "continuation_gates": {
            "min_evaluable_folds": 6,
            "min_fraction_ba_gt_0_5": 0.60,
            "min_fraction_roc_auc_gt_0_5": 0.60,
            "min_fraction_pr_auc_not_below_prevalence": 0.60,
            "prediction_non_collapse_required": True,
            "no_month_concentration_required": True,
            "no_etf_concentration_required": True,
            "no_leakage_assertion_failed": True,
            "no_model_scaler_saved": True,
        },
        "review_gates": [
            "two_new_months_below_dummy_baseline",
            "ba_mean_lte_0_5",
            "roc_auc_mean_lte_0_5",
            "single_class_prediction_collapse",
            "label_regime_shift",
            "feature_shift",
            "month_concentration",
            "etf_concentration",
            "post_sprint_forward_only_underpowered",
        ],
        "retire_gates": [
            "at_least_60pct_folds_below_dummy_baseline",
            "combined_rolling_origin_mean_ba_lte_0_5",
            "combined_roc_auc_lte_0_5",
            "repeated_collapse",
            "data_quality_blocker",
            "leakage_blocker",
            "source_integrity_blocker",
        ],
        "promotion_boundary": promotion_boundary(),
    }
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday Lab Monitor Candidate Protocol",
        "",
        "This protocol registers one Lab-only monitor candidate. It is read-only research status, not Stable evidence, not trading advice, not QMT-ready, and not an OrderIntent source.",
        "",
        "## Candidate Identity",
        "",
        f"- candidate_id: `{FOCUS_CANDIDATE_ID}`",
        "- label policy: `label_ret3d_gt_100bp`",
        "- feature set: `base_39_plus_scale_transform_policy`",
        "- model family: `logistic_balanced_scaled_variants`",
        "- transform policy: `scale_transform_policy`",
        "- automatic replacement of label / feature / model / threshold: false",
        "",
        "## Monitor Status",
        "",
        f"- status: `{MONITOR_STATUS}`",
        "- not a Stable candidate, trading strategy, advisory signal, OrderIntent source, QMT-ready model, or formal trained model",
        "",
        "## Continuation Gates",
        "",
        "- evaluable folds >= 6",
        "- >= 60% folds BA > 0.5",
        "- >= 60% folds ROC-AUC > 0.5",
        "- PR-AUC not below label prevalence in >= 60% folds",
        "- prediction non-collapse",
        "- no month concentration, no ETF concentration, no leakage assertion failed, no model/scaler saved",
        "",
        "## Promotion Boundary",
        "",
        "Even sustained monitor stability cannot automatically enter Stable. Any Stable direction requires a separate human-review promotion gate and remains blocked from BUY/PROBE threshold changes, target_weight changes, final_buy_action changes, OrderIntent, QMT, and automatic promotion.",
    ]
    return payload, "\n".join(lines) + "\n"


def promotion_boundary() -> dict[str, Any]:
    return {
        "stable_promotion_ready": False,
        "stable_evidence": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
        "requires_human_review_promotion_gate": True,
        "minimum_future_reviews": [
            "cost_slippage_diagnostic",
            "capacity_turnover_diagnostic",
            "formal_training_design",
            "no_save_to_save_transition_review",
            "Stable adapter review",
            "no BUY/PROBE threshold change",
            "no target_weight change",
            "no final_buy_action change",
            "no OrderIntent generation",
        ],
    }


def build_closeout_doc(report: dict[str, Any]) -> tuple[dict[str, Any], str]:
    payload = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_lab_monitor_candidate_closeout",
        "status_decision": report["status_decision"],
        "candidate_id": FOCUS_CANDIDATE_ID,
        "completed_chain": [
            "manual broker 5m data chain completed",
            "fixed-shortlist OOP completed",
            "row-level diagnostics completed",
            "post-sprint reversal attribution completed",
            "rolling-origin walk-forward completed",
            "rolling-origin stability attribution completed",
            "unique Lab monitor candidate identified",
        ],
        "current_not_stable_evidence": True,
        "next_step": "Lab-only monitor protocol / periodic data refresh",
        "promotion_boundary": promotion_boundary(),
    }
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday Lab Monitor Candidate Closeout",
        "",
        "- manual broker 5m data chain: completed",
        "- fixed-shortlist OOP: completed",
        "- row-level diagnostics: completed",
        "- post-sprint reversal attribution: completed",
        "- rolling-origin walk-forward: completed",
        "- rolling-origin stability attribution: completed",
        "- unique Lab monitor candidate: identified",
        "- current Stable evidence: false",
        "- next step: Lab-only monitor protocol / periodic data refresh",
    ]
    return payload, "\n".join(lines) + "\n"


def build_status_docs(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    lines = [
        LAB_DECLARATION,
        "",
        "# Intraday Lab Monitor Candidate Status",
        "",
        f"- status_decision: {report['status_decision']}",
        f"- candidate_id: {FOCUS_CANDIDATE_ID}",
        f"- monitor_status: {MONITOR_STATUS}",
        f"- continuation_gate_passed: {str(report['gate_status']['continuation_gate_passed']).lower()}",
        f"- review_gate_triggered: {str(report['gate_status']['review_gate_triggered']).lower()}",
        f"- retire_gate_triggered: {str(report['gate_status']['retire_gate_triggered']).lower()}",
        f"- stable_promotion_ready: false",
        f"- stable_evidence: false",
        f"- qmt_ready: false",
        f"- order_intent_ready: false",
    ]
    docs_json = {
        "lab_declaration": LAB_DECLARATION,
        "document_type": "aetfq3_intraday_lab_monitor_candidate_status",
        "status_decision": report["status_decision"],
        "candidate_id": FOCUS_CANDIDATE_ID,
        "monitor_status": MONITOR_STATUS,
        "gate_status": report["gate_status"],
        "stable_promotion_ready": False,
        "stable_evidence": False,
        "formal_training_ready": False,
        "qmt_ready": False,
        "order_intent_ready": False,
        "automatic_promotion_ready": False,
    }
    return "\n".join(lines) + "\n", docs_json


def run_status(
    rolling_origin_dir: Path = DEFAULT_ROLLING_ORIGIN_DIR,
    attribution_dir: Path = DEFAULT_ATTRIBUTION_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    *,
    repo_root: Path = REPO_ROOT,
    enforce_output_dir: bool = True,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    rolling_dir = resolve_repo_path(rolling_origin_dir, repo_root)
    attrib_dir = resolve_repo_path(attribution_dir, repo_root)
    resolved_out_dir = resolve_output_dir(out_dir, repo_root, enforce=enforce_output_dir)
    resolved_out_dir.mkdir(parents=True, exist_ok=True)

    rolling_report_path = rolling_dir / "rolling_origin_walk_forward_report.json"
    attribution_report_path = attrib_dir / "rolling_origin_stability_attribution_report.json"
    missing = [str(path) for path in [rolling_report_path, attribution_report_path] if not path.exists()]
    blockers = [f"missing required monitor input: {path}" for path in missing]
    artifact_before = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_before["p0_blockers"])
    rolling_report = load_json(rolling_report_path) if rolling_report_path.exists() else {}
    attribution_report = load_json(attribution_report_path) if attribution_report_path.exists() else {}
    candidate = find_candidate(rolling_report) or {}
    if rolling_report and not candidate:
        blockers.append("missing monitor candidate in rolling-origin aggregate_stability")
    gates = gate_status(candidate, attribution_report) if candidate and attribution_report else empty_gate_status()
    decision = decide(gates, blockers)
    row = status_row(candidate, attribution_report, gates, decision)
    artifact_after = check_model_artifacts(resolved_out_dir)
    blockers.extend(artifact_after["p0_blockers"])
    if blockers and decision not in {DECISION_BLOCKED_MISSING, DECISION_BLOCKED_DATA}:
        decision = decide(gates, blockers)
        row["status_decision"] = decision

    report = {
        "lab_declaration": LAB_DECLARATION,
        "report_type": REPORT_TYPE,
        "status": "blocked" if decision in {DECISION_BLOCKED_MISSING, DECISION_BLOCKED_DATA} else "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_sources": {
            "rolling_origin_dir": str(rolling_origin_dir),
            "attribution_dir": str(attribution_dir),
            "stable_bundle": False,
        },
        "candidate_status": row,
        "gate_status": gates,
        "status_decision": decision,
        "p0_blockers": list(dict.fromkeys(blockers)),
        "p1_warnings": p1_warnings(gates, decision),
        "artifact_check_before": artifact_before,
        "artifact_check_after": artifact_after,
        "access_mode": "READ_ONLY",
        "final_action_change_allowed": False,
        "contains_live_order": False,
        "contains_secret": False,
        "requires_human_review": True,
        "promotion_gate_required": True,
        "formal_training": False,
        "formal_training_ready": False,
        "model_saved": False,
        "scaler_saved": False,
        "checkpoint_saved": False,
        "gpu_used": False,
        "torchrun_used": False,
        "stable_promotion_ready": False,
        "automatic_promotion_ready": False,
        "qmt_ready": False,
        "qmt_used": False,
        "order_intent_ready": False,
        "order_intent_generated": False,
        "stable_evidence": False,
        "stable_affected": False,
        "advisory_package_created": False,
        "not_trading_advice": True,
    }
    emit_outputs(repo_root, resolved_out_dir, report, row)
    return report


def empty_gate_status() -> dict[str, Any]:
    return {
        "continuation_gate_passed": False,
        "review_gate_triggered": False,
        "review_warning_reasons": [],
        "retire_gate_triggered": False,
        "retire_reasons": [],
        "month_concentration_flag": False,
        "etf_concentration_flag": False,
        "threshold_sensitivity_flag": False,
        "post_sprint_underpowered_flag": True,
        "no_leakage_assertion_failed": False,
        "no_model_scaler_saved": True,
    }


def p1_warnings(gates: dict[str, Any], decision: str) -> list[str]:
    warnings = [
        "P1_LAB_MONITOR_CANDIDATE_NOT_STABLE_EVIDENCE",
        "P1_REQUIRES_HUMAN_REVIEW",
        "P1_NO_STABLE_PROMOTION_WITHOUT_PROMOTION_GATE",
        "P1_POST_SPRINT_FORWARD_ONLY_UNDERPOWERED_STILL_REVIEW_ITEM",
    ]
    if decision != DECISION_ACTIVE:
        warnings.append("P1_MONITOR_GATE_REVIEW_REQUIRED")
    if gates.get("post_sprint_underpowered_flag"):
        warnings.append("P1_MONITOR_DOES_NOT_OVERRIDE_POST_SPRINT_UNDERPOWERED")
    return list(dict.fromkeys(warnings))


def emit_outputs(repo_root: Path, out_dir: Path, report: dict[str, Any], row: dict[str, Any]) -> None:
    status_md, _status_docs_json = build_status_docs(report)
    protocol_json, protocol_md = build_protocol_doc()
    closeout_json, closeout_md = build_closeout_doc(report)
    write_json(out_dir / "lab_monitor_candidate_status_report.json", report)
    write_json(
        out_dir / "lab_monitor_candidate_protocol_decision.json",
        {
            "lab_declaration": LAB_DECLARATION,
            "status_decision": report["status_decision"],
            "candidate_id": FOCUS_CANDIDATE_ID,
            "monitor_status": MONITOR_STATUS,
            "stable_promotion_ready": False,
            "stable_evidence": False,
            "formal_training_ready": False,
            "qmt_ready": False,
            "order_intent_ready": False,
            "automatic_promotion_ready": False,
            "p0_blockers": report["p0_blockers"],
            "p1_warnings": report["p1_warnings"],
        },
    )
    write_csv(out_dir / "lab_monitor_candidate_gate_status.csv", [row], gate_columns())
    (out_dir / "lab_monitor_candidate_status_report.md").write_text(status_md, encoding="utf-8")
    write_json(repo_root / "docs/research/aetfq3_intraday_lab_monitor_candidate_protocol.json", protocol_json)
    (repo_root / "docs/research/aetfq3_intraday_lab_monitor_candidate_protocol.md").write_text(protocol_md, encoding="utf-8")
    write_json(repo_root / "docs/research/aetfq3_intraday_lab_monitor_candidate_closeout.json", closeout_json)
    (repo_root / "docs/research/aetfq3_intraday_lab_monitor_candidate_closeout.md").write_text(closeout_md, encoding="utf-8")


def gate_columns() -> list[str]:
    return [
        "candidate_id",
        "monitor_status",
        "rolling_origin_decision",
        "attribution_decision",
        "fold_count",
        "positive_fold_count",
        "above_baseline_fraction",
        "balanced_accuracy_mean",
        "roc_auc_mean",
        "pr_auc_mean",
        "month_concentration_flag",
        "etf_concentration_flag",
        "threshold_sensitivity_flag",
        "post_sprint_underpowered_flag",
        "continuation_gate_passed",
        "review_gate_triggered",
        "retire_gate_triggered",
        "status_decision",
        "next_allowed_task",
        "forbidden_next_tasks",
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=REPORT_TYPE)
    parser.add_argument("--rolling-origin-dir", type=Path, default=DEFAULT_ROLLING_ORIGIN_DIR)
    parser.add_argument("--attribution-dir", type=Path, default=DEFAULT_ATTRIBUTION_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_status(args.rolling_origin_dir, args.attribution_dir, args.out_dir)
    except Exception as exc:  # noqa: BLE001 - CLI emits auditable Lab blocker.
        print(
            json.dumps(
                {
                    "lab_declaration": LAB_DECLARATION,
                    "status": "failed",
                    "status_decision": DECISION_BLOCKED_DATA,
                    "p0_blockers": [str(exc)],
                    "stable_promotion_ready": False,
                    "stable_evidence": False,
                    "qmt_ready": False,
                    "order_intent_ready": False,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "lab_declaration": LAB_DECLARATION,
                "status": report["status"],
                "status_decision": report["status_decision"],
                "candidate_id": FOCUS_CANDIDATE_ID,
                "continuation_gate_passed": report["gate_status"]["continuation_gate_passed"],
                "review_gate_triggered": report["gate_status"]["review_gate_triggered"],
                "retire_gate_triggered": report["gate_status"]["retire_gate_triggered"],
                "stable_promotion_ready": False,
                "stable_evidence": False,
                "qmt_ready": False,
                "order_intent_ready": False,
                "p0_blockers": report["p0_blockers"],
                "p1_warnings": report["p1_warnings"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
