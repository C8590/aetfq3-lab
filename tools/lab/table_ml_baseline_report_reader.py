from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence


OK = "OK"
P0_REVIEW_REQUIRED = "P0_REVIEW_REQUIRED"

REQUIRED_CONTRACT_FIELDS = {
    "report_type",
    "task_scope",
    "lab_only",
    "no_save",
    "no_tuning",
    "no_stable",
    "no_qmt",
    "no_order_intent",
    "no_output",
    "no_lab_advisory",
    "model_saved",
    "checkpoint_saved",
    "target_label",
    "feature_columns",
    "forbidden_columns",
    "train_count",
    "valid_count",
    "split_method",
    "group_leakage_check",
    "models",
    "metrics",
    "prediction_file",
    "review_checklist",
}

PROHIBITED_FIELDS = {
    "order_intent",
    "target_weight",
    "final_buy_action",
    "stable_action",
    "qmt_order",
    "live_order",
    "trade_instruction",
}


class BaselineReportContractError(RuntimeError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8-sig") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise BaselineReportContractError("Report JSON root must be an object")
    return value


def summarize_report(path: Path) -> dict[str, Any]:
    report = load_json(path)
    prohibited_paths = find_prohibited_fields(report)
    if prohibited_paths:
        raise BaselineReportContractError("Prohibited trading fields present: " + ", ".join(prohibited_paths))

    normalized = normalize_report(report)
    missing = sorted(field for field in REQUIRED_CONTRACT_FIELDS if field not in normalized)
    if missing:
        raise BaselineReportContractError("Missing required fields: " + ", ".join(missing))

    errors = validate_contract_values(normalized)
    if errors:
        raise BaselineReportContractError("; ".join(errors))

    metrics = list_value(normalized.get("metrics"))
    models = list_value(normalized.get("models"))
    return {
        "status": OK,
        "report_type": str_value(normalized.get("report_type")),
        "task_scope": str_value(normalized.get("task_scope")),
        "boundary_passed": True,
        "models": model_names(models),
        "model_count": len(models),
        "metrics_keys": sorted({key for metric in metrics if isinstance(metric, dict) for key in metric}),
        "target_label": str_value(normalized.get("target_label")),
        "train_count": int_value(normalized.get("train_count")),
        "valid_count": int_value(normalized.get("valid_count")),
        "split_method": str_value(normalized.get("split_method")),
        "group_leakage_check": str_value(normalized.get("group_leakage_check")),
        "prediction_file": str_value(normalized.get("prediction_file")),
        "feature_count": len(list_value(normalized.get("feature_columns"))),
        "forbidden_count": len(list_value(normalized.get("forbidden_columns"))),
    }


def normalize_report(report: dict[str, Any]) -> dict[str, Any]:
    if REQUIRED_CONTRACT_FIELDS <= set(report):
        return dict(report)

    if not (
        isinstance(report.get("boundary"), dict)
        or isinstance(report.get("feature_leakage_check"), dict)
        or isinstance(report.get("split"), dict)
    ):
        return dict(report)

    boundary = dict_value(report.get("boundary"))
    leakage = dict_value(report.get("feature_leakage_check"))
    split = dict_value(report.get("split"))
    metrics = list_value(report.get("metrics"))
    first_metric = metrics[0] if metrics and isinstance(metrics[0], dict) else {}
    models = list_value(report.get("models"))

    return {
        "report_type": "table_ml_baseline_smoke",
        "task_scope": "Lab-only no-save baseline smoke",
        "lab_only": "aetfq3-lab / Lab" in str_value(report.get("lab_boundary")),
        "no_save": all(bool_value(model.get("no_save")) is True for model in models if isinstance(model, dict))
        and bool_value(boundary.get("no_model_save")) is True,
        "no_tuning": all(bool_value(model.get("no_tuning")) is True for model in models if isinstance(model, dict))
        and bool_value(boundary.get("no_hyperparameter_search")) is True,
        "no_stable": bool_value(boundary.get("no_stable")),
        "no_qmt": bool_value(boundary.get("no_qmt")),
        "no_order_intent": bool_value(boundary.get("no_order_intent")),
        "no_output": bool_value(boundary.get("no_output")),
        "no_lab_advisory": bool_value(boundary.get("no_lab_advisory")),
        "model_saved": False if bool_value(boundary.get("no_model_save")) is True else None,
        "checkpoint_saved": False if bool_value(boundary.get("no_checkpoint")) is True else None,
        "target_label": first_metric.get("target_label"),
        "feature_columns": leakage.get("feature_columns"),
        "forbidden_columns": leakage.get("forbidden_columns"),
        "train_count": first_metric.get("train_count") or split.get("train_count"),
        "valid_count": first_metric.get("valid_count") or split.get("valid_count"),
        "split_method": split.get("type"),
        "group_leakage_check": "passed" if bool_value(split.get("group_leakage_check_passed")) is True else "failed",
        "models": models,
        "metrics": metrics,
        "prediction_file": report.get("prediction_file"),
        "review_checklist": report.get("review_checklist", {}),
    }


def validate_contract_values(report: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if report.get("report_type") != "table_ml_baseline_smoke":
        errors.append("report_type must be table_ml_baseline_smoke")
    if report.get("task_scope") != "Lab-only no-save baseline smoke":
        errors.append("task_scope must be Lab-only no-save baseline smoke")

    expected_true = {
        "lab_only",
        "no_save",
        "no_tuning",
        "no_stable",
        "no_qmt",
        "no_order_intent",
        "no_output",
        "no_lab_advisory",
    }
    expected_false = {"model_saved", "checkpoint_saved"}
    for field_name in sorted(expected_true):
        if report.get(field_name) is not True:
            errors.append(f"{field_name} must be true")
    for field_name in sorted(expected_false):
        if report.get(field_name) is not False:
            errors.append(f"{field_name} must be false")

    if report.get("split_method") != "chronological":
        errors.append("split_method must be chronological")
    if report.get("group_leakage_check") != "passed":
        errors.append("group_leakage_check must be passed")
    if not isinstance(report.get("feature_columns"), list) or not report.get("feature_columns"):
        errors.append("feature_columns must be a non-empty array")
    if not isinstance(report.get("forbidden_columns"), list) or not report.get("forbidden_columns"):
        errors.append("forbidden_columns must be a non-empty array")
    if not isinstance(report.get("models"), list) or not report.get("models"):
        errors.append("models must be a non-empty array")
    if not isinstance(report.get("metrics"), list) or not report.get("metrics"):
        errors.append("metrics must be a non-empty array")
    if int_value(report.get("train_count")) <= 0:
        errors.append("train_count must be positive")
    if int_value(report.get("valid_count")) <= 0:
        errors.append("valid_count must be positive")
    if not str_value(report.get("target_label")):
        errors.append("target_label must be non-empty")
    if not str_value(report.get("prediction_file")):
        errors.append("prediction_file must be non-empty")
    if not isinstance(report.get("review_checklist"), dict):
        errors.append("review_checklist must be an object")
    return errors


def find_prohibited_fields(value: Any, prefix: str = "") -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in PROHIBITED_FIELDS:
                if item not in (None, "", [], {}):
                    paths.append(path)
            paths.extend(find_prohibited_fields(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(find_prohibited_fields(item, f"{prefix}[{index}]"))
    return paths


def model_names(models: list[Any]) -> list[str]:
    result: list[str] = []
    for model in models:
        if isinstance(model, dict):
            name = str_value(model.get("model_name"))
            if name:
                result.append(name)
        elif isinstance(model, str):
            result.append(model)
    return result


def dict_value(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def list_value(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def bool_value(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def str_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def int_value(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Read and validate Lab-only baseline smoke report contract.")
    parser.add_argument("--report", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_report(args.report)
    except (BaselineReportContractError, OSError, json.JSONDecodeError) as exc:
        print(f"FAILED baseline_report_contract_valid=false {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
