from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .io_utils import ensure_dir


TIMING_COLUMNS = [
    "load_data_seconds",
    "normalize_code_seconds",
    "feature_build_seconds",
    "label_build_seconds",
    "model_train_seconds",
    "score_inference_seconds",
    "write_output_seconds",
    "total_seconds",
]


def empty_timing() -> dict[str, float]:
    return {column: 0.0 for column in TIMING_COLUMNS}


def write_performance_outputs(
    out_dir: str | Path,
    timing: dict[str, Any],
    *,
    command: str,
    row_counts: dict[str, int] | None = None,
    notes: list[str] | None = None,
) -> tuple[Path, Path, Path]:
    out = ensure_dir(out_dir)
    normalized = empty_timing()
    for key, value in timing.items():
        if key in normalized:
            normalized[key] = round(float(value or 0.0), 6)
    extra = {key: value for key, value in timing.items() if key not in normalized}
    payload = {
        "command": command,
        "baseline_before_optimization": {
            "full_universe_replay": "user-reported > 10 minutes without completed result",
        },
        "performance_targets": {
            "smoke_30_trading_days_seconds": 30,
            "full_replay_seconds": 180,
            "shadow_inference_seconds": 15,
        },
        "timing": normalized,
        "extra_timing": extra,
        "row_counts": row_counts or {},
        "notes": notes or [],
    }

    json_path = out / "historical_ml_performance_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = out / "replay_timing_summary.csv"
    pd.DataFrame([{**{"command": command}, **normalized, **extra}]).to_csv(csv_path, index=False, encoding="utf-8-sig")

    md_path = out / "historical_ml_performance_report.md"
    md_path.write_text(_markdown_report(payload), encoding="utf-8")
    return md_path, json_path, csv_path


def _markdown_report(payload: dict[str, Any]) -> str:
    timing = payload["timing"]
    rows = "\n".join(f"| {key} | {value:.6f} |" for key, value in timing.items())
    extra = payload.get("extra_timing") or {}
    extra_rows = "\n".join(f"| {key} | {float(value):.6f} |" for key, value in extra.items() if _is_number(value))
    row_counts = payload.get("row_counts") or {}
    count_rows = "\n".join(f"| {key} | {value} |" for key, value in row_counts.items())
    notes = "\n".join(f"- {note}" for note in payload.get("notes") or ["无"])
    return "\n".join(
        [
            "# historical_ml_performance_report",
            "",
            "## Summary",
            "",
            "- 主责: aetfv2_05_historical_ml",
            "- 协作: aetfv2_08_control_center, aetfv2_01_pre_selection, aetfv2_02_entry",
            "- 优化前基线: full-universe replay 用户观测为超过 10 分钟未完成。",
            "- 优化后路径: 预构建 code/trade_date 特征矩阵，按交易日复用；标签通过前向窗口表一次性 merge。",
            "- 边界: 不修改 BUY/PROBE 阈值，不修改 entry final_buy_action，不连接 QMT，不刷新行情。",
            "",
            "## Timing",
            "",
            "| stage | seconds |",
            "|---|---:|",
            rows,
            "",
            "## Extra Timing",
            "",
            "| stage | seconds |",
            "|---|---:|",
            extra_rows or "| 无 | 0.000000 |",
            "",
            "## Row Counts",
            "",
            "| table | rows |",
            "|---|---:|",
            count_rows or "| 无 | 0 |",
            "",
            "## Notes",
            "",
            notes,
            "",
            "## No Future Function Boundary",
            "",
            "- replay feature_at_t only uses price rows at or before trade_date.",
            "- label_after_t is generated after replay and is not merged back into entry decisions.",
            "- ml_entry_scores remains a shadow ETF code-level scoring artifact.",
            "",
        ]
    )


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True
