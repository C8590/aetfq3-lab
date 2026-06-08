from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


REPORT_TYPE = "intraday_mock_tensor_smoke"
TASK_SCOPE = "Lab-only mock tensor smoke; not formal training"
TARGET_COLUMNS = [
    "buy_now_label",
    "wait_pullback_label",
    "cancel_buy_label",
    "three_day_positive_label",
]
DEFAULT_FEATURE_COLUMNS = [
    "bar_index",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "vwap",
    "prev_close",
    "open_price",
    "open_gap_pct",
    "intraday_return",
    "return_from_open",
    "drawdown_from_open",
    "distance_to_vwap",
    "pullback_to_vwap",
    "break_open_low",
    "reclaim_vwap",
    "sector_top1_flag",
    "candidate_rank",
]
EXPLICIT_FORBIDDEN_COLUMNS = {
    "max_drawdown_3d",
    "execution_return_to_close",
    "execution_return_to_next_open",
    "execution_drawdown_after_entry",
}


class IntradayMockSmokeError(RuntimeError):
    pass


@dataclass(frozen=True)
class TensorBundle:
    x: torch.Tensor
    y: torch.Tensor
    sequence_keys: list[dict[str, str]]
    feature_columns: list[str]
    target_columns: list[str]


class MLPBaseline(nn.Module):
    def __init__(self, time_steps: int, feature_count: int, target_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(time_steps * feature_count, 16),
            nn.ReLU(),
            nn.Linear(16, target_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GRUSmoke(nn.Module):
    def __init__(self, feature_count: int, target_count: int) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size=feature_count, hidden_size=12, batch_first=True)
        self.head = nn.Linear(12, target_count)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.head(hidden[-1])


class TemporalCNNSmoke(nn.Module):
    def __init__(self, feature_count: int, target_count: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(feature_count, 12, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(12, target_count),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2))


def is_forbidden_feature(column: str) -> bool:
    return column.startswith("future_") or column.endswith("_label") or column in EXPLICIT_FORBIDDEN_COLUMNS


def scan_forbidden_features(feature_columns: Sequence[str]) -> dict[str, Any]:
    forbidden = [column for column in feature_columns if is_forbidden_feature(column)]
    return {
        "passed": not forbidden,
        "forbidden_columns": forbidden,
        "forbidden_rules": [
            "all future_* fields",
            "all *_label fields",
            "max_drawdown_3d",
            "execution_return_to_close",
            "execution_return_to_next_open",
            "execution_drawdown_after_entry",
        ],
    }


def ensure_columns(df: pd.DataFrame, columns: Sequence[str], purpose: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise IntradayMockSmokeError(f"missing {purpose} columns: {missing}")


def build_sequence_tensor(
    input_path: Path,
    feature_columns: Sequence[str] | None = None,
    target_columns: Sequence[str] | None = None,
    min_time_steps: int = 12,
) -> TensorBundle:
    df = pd.read_csv(input_path, dtype={"etf_code": str})
    if df.empty:
        raise IntradayMockSmokeError("input CSV is empty")

    features = list(feature_columns or DEFAULT_FEATURE_COLUMNS)
    targets = list(target_columns or TARGET_COLUMNS)
    scan = scan_forbidden_features(features)
    if not scan["passed"]:
        raise IntradayMockSmokeError(f"forbidden feature columns: {scan['forbidden_columns']}")

    required = ["trade_date", "datetime", "etf_code", *features, *targets]
    ensure_columns(df, required, "mock intraday")

    for column in features + targets:
        df[column] = pd.to_numeric(df[column], errors="raise")

    sequence_arrays: list[np.ndarray] = []
    target_arrays: list[np.ndarray] = []
    sequence_keys: list[dict[str, str]] = []
    for (trade_date, etf_code), group in df.groupby(["trade_date", "etf_code"], sort=True):
        ordered = group.sort_values(["bar_index", "datetime"])
        if len(ordered) < min_time_steps:
            raise IntradayMockSmokeError(
                f"sequence {trade_date}/{etf_code} has {len(ordered)} bars, expected at least {min_time_steps}"
            )
        window = ordered.iloc[:min_time_steps]
        sequence_arrays.append(window[features].to_numpy(dtype=np.float32))
        target_arrays.append(window.iloc[-1][targets].to_numpy(dtype=np.float32))
        sequence_keys.append({"trade_date": str(trade_date), "etf_code": str(etf_code)})

    if not sequence_arrays:
        raise IntradayMockSmokeError("no sequences constructed")

    x = torch.tensor(np.stack(sequence_arrays), dtype=torch.float32)
    y = torch.tensor(np.stack(target_arrays), dtype=torch.float32)
    return TensorBundle(x=x, y=y, sequence_keys=sequence_keys, feature_columns=features, target_columns=targets)


def run_model_smoke(
    name: str,
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    device: torch.device,
    steps: int = 2,
) -> dict[str, Any]:
    model = model.to(device)
    x = x.to(device)
    y = y.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.BCEWithLogitsLoss()
    final_loss = None
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().cpu().item())

    return {
        "model_name": name,
        "status": "passed",
        "steps": steps,
        "final_loss": final_loss,
        "no_save": True,
        "checkpoint_saved": False,
        "model_saved": False,
        "notes": "final_loss is smoke-only and must not be interpreted as model performance",
    }


def write_reports(report: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "intraday_mock_tensor_smoke_report.json"
    md_path = out_dir / "intraday_mock_tensor_smoke_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "本任务属于 aetfq3-lab / Lab，不属于 V2.1 Stable。",
        "本文件只用于 Lab research，不是 Stable 交易规则，不接 QMT，不生成 OrderIntent，不自动下单。",
        "",
        "# Intraday Mock Tensor Smoke Report",
        "",
        f"- status: {report['status']}",
        f"- batch_size: {report['batch_size']}",
        f"- time_steps: {report['time_steps']}",
        f"- feature_count: {report['feature_count']}",
        f"- target_count: {report['target_count']}",
        f"- forbidden_feature_passed: {str(report['forbidden_feature_passed']).lower()}",
        f"- device: {report['device']}",
        f"- cuda_available: {str(report['cuda_available']).lower()}",
        "- note: final_loss is smoke-only, not model performance.",
    ]
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")


def run_intraday_mock_tensor_smoke(
    input_path: Path,
    out_dir: Path,
    device_name: str | None = None,
) -> dict[str, Any]:
    bundle = build_sequence_tensor(input_path)
    scan = scan_forbidden_features(bundle.feature_columns)
    if not scan["passed"]:
        raise IntradayMockSmokeError(f"forbidden feature columns: {scan['forbidden_columns']}")

    cuda_available = torch.cuda.is_available()
    resolved_device = device_name or ("cuda" if cuda_available else "cpu")
    device = torch.device(resolved_device)
    batch_size, time_steps, feature_count = bundle.x.shape
    target_count = bundle.y.shape[1]

    model_results = [
        run_model_smoke("mlp_smoke", MLPBaseline(time_steps, feature_count, target_count), bundle.x, bundle.y, device),
        run_model_smoke("gru_smoke", GRUSmoke(feature_count, target_count), bundle.x, bundle.y, device),
        run_model_smoke("temporal_cnn_smoke", TemporalCNNSmoke(feature_count, target_count), bundle.x, bundle.y, device),
    ]

    report = {
        "report_type": REPORT_TYPE,
        "task_scope": TASK_SCOPE,
        "status": "passed",
        "lab_only": True,
        "mock_only": True,
        "reads_real_intraday": False,
        "formal_training": False,
        "torchrun_used": False,
        "no_save": True,
        "model_saved": False,
        "checkpoint_saved": False,
        "no_stable": True,
        "no_qmt": True,
        "no_order_intent": True,
        "no_output": True,
        "no_lab_advisory": True,
        "input_file": str(input_path),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "batch_size": int(batch_size),
        "time_steps": int(time_steps),
        "feature_count": int(feature_count),
        "target_count": int(target_count),
        "feature_columns": bundle.feature_columns,
        "target_columns": bundle.target_columns,
        "sequence_keys": bundle.sequence_keys,
        "forbidden_feature_passed": bool(scan["passed"]),
        "forbidden_columns": scan["forbidden_columns"],
        "device": str(device),
        "cuda_available": bool(cuda_available),
        "models": model_results,
    }
    write_reports(report, out_dir)
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lab-only intraday mock tensor smoke.")
    parser.add_argument("--input", required=True, type=Path, help="Mock 5m CSV fixture; must not be real market data.")
    parser.add_argument("--out-dir", required=True, type=Path, help="Ignored local output directory.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None, help="Optional smoke device override.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_intraday_mock_tensor_smoke(args.input, args.out_dir, args.device)
    print(json.dumps({
        "status": report["status"],
        "batch_size": report["batch_size"],
        "time_steps": report["time_steps"],
        "feature_count": report["feature_count"],
        "target_count": report["target_count"],
        "forbidden_feature_passed": report["forbidden_feature_passed"],
        "device": report["device"],
        "cuda_available": report["cuda_available"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

