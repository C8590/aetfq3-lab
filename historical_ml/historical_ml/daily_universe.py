from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import HistoricalMLConfig
from .io_utils import ensure_dir, reorder_columns, write_table
from .schemas import DAILY_ML_UNIVERSE_SAMPLE_COLUMNS


def build_daily_ml_universe_samples(
    etf_samples: pd.DataFrame,
    entry_candidates: pd.DataFrame,
    config: HistoricalMLConfig,
) -> pd.DataFrame:
    """Build the full daily ETF universe table for ML from replay-only inputs."""

    if etf_samples.empty:
        return pd.DataFrame(columns=DAILY_ML_UNIVERSE_SAMPLE_COLUMNS)

    base = etf_samples.copy()
    base["trade_date"] = pd.to_datetime(base["trade_date"]).dt.normalize()
    base["code"] = base["code"].astype(str)

    decisions = _decision_columns(entry_candidates)
    if not decisions.empty:
        base = base.merge(decisions, on=["trade_date", "code"], how="left")
    else:
        base["pre_selected"] = False
        base["decision_exclude_reason"] = ""
        base["entry_raw_action"] = "OBSERVE"
        base["final_action"] = "OBSERVE"

    data_quality = base.get("data_quality_flag", pd.Series("ok", index=base.index)).fillna("ok").astype(str)
    valid = data_quality.eq("ok")
    decision_reason = base.get("decision_exclude_reason", pd.Series("", index=base.index)).fillna("").astype(str)
    invalid_reason = "invalid:" + data_quality

    out = pd.DataFrame(
        {
            "trade_date": base["trade_date"],
            "code": base["code"],
            "name": base.get("name", base["code"]).astype(str),
            "sector_level1": _first_existing(base, "sector_level1", "sector_l1", "sector"),
            "sector_level2": _first_existing(base, "sector_level2", "sector"),
            "is_valid_sample": valid,
            "exclude_reason": decision_reason.where(valid, invalid_reason),
            "momentum_20": base.get("r20"),
            "momentum_60": base.get("r60"),
            "momentum_120": base.get("r120"),
            "momentum_score": base.get("momentum_score"),
            "acceleration_score": base.get("acceleration_score"),
            "volatility_20": base.get("vol20"),
            "drawdown_20": base.get("max_drawdown_20d"),
            "drawdown_60": base.get("max_drawdown_60d"),
            "market_state": base.get("market_state"),
            "sector_state": base.get("sector_state"),
            "sector_rank": base.get("sector_rank"),
            "etf_rank": base.get("etf_rank"),
            "pre_selected": _bool_series(base.get("pre_selected", pd.Series(False, index=base.index))),
            "entry_raw_action": base.get("entry_raw_action", pd.Series("OBSERVE", index=base.index)).fillna("OBSERVE"),
            "final_action": base.get("final_action", pd.Series("OBSERVE", index=base.index)).fillna("OBSERVE"),
            "source": config.source,
        }
    )
    return reorder_columns(out, DAILY_ML_UNIVERSE_SAMPLE_COLUMNS)


def build_daily_ml_universe_summary(samples: pd.DataFrame, config: HistoricalMLConfig) -> dict[str, Any]:
    if samples.empty:
        return {
            "source": config.source,
            "total_rows": 0,
            "trade_date_count": 0,
            "total_valid_samples": 0,
            "total_pre_selected": 0,
            "samples_to_pre_selected_ratio": None,
            "daily_stats": [],
        }

    frame = samples.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"]).dt.strftime("%Y-%m-%d")
    frame["is_valid_sample"] = _bool_series(frame["is_valid_sample"])
    frame["pre_selected"] = _bool_series(frame["pre_selected"])

    daily_stats = []
    for trade_date, group in frame.groupby("trade_date", sort=True):
        valid_count = int(group["is_valid_sample"].sum())
        pre_selected_count = int(group["pre_selected"].sum())
        daily_stats.append(
            {
                "trade_date": str(trade_date),
                "total_samples": int(len(group)),
                "valid_samples": valid_count,
                "invalid_samples": int(len(group) - valid_count),
                "pre_selected_count": pre_selected_count,
                "non_selected_valid_samples": int((group["is_valid_sample"] & ~group["pre_selected"]).sum()),
                "entry_raw_action_distribution": _value_counts(group.get("entry_raw_action")),
                "final_action_distribution": _value_counts(group.get("final_action")),
            }
        )

    total_pre_selected = int(frame["pre_selected"].sum())
    return {
        "source": config.source,
        "total_rows": int(len(frame)),
        "trade_date_count": int(frame["trade_date"].nunique()),
        "total_valid_samples": int(frame["is_valid_sample"].sum()),
        "total_pre_selected": total_pre_selected,
        "samples_to_pre_selected_ratio": None if total_pre_selected == 0 else round(len(frame) / total_pre_selected, 6),
        "daily_stats": daily_stats,
    }


def generated_output_dir(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    return out if out.name == "generated" else out / "generated"


def write_daily_ml_universe_outputs(
    samples: pd.DataFrame,
    out_dir: str | Path,
    config: HistoricalMLConfig,
) -> tuple[Path, Path]:
    generated = ensure_dir(generated_output_dir(out_dir))
    samples_path = write_table(samples, generated, "daily_ml_universe_samples", config.output_format)
    summary = build_daily_ml_universe_summary(samples, config)
    summary_path = generated / "daily_ml_universe_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return samples_path, summary_path


def _decision_columns(entry_candidates: pd.DataFrame) -> pd.DataFrame:
    if entry_candidates.empty:
        return pd.DataFrame()
    decisions = entry_candidates.copy()
    decisions["trade_date"] = pd.to_datetime(decisions["trade_date"]).dt.normalize()
    decisions["code"] = decisions["code"].astype(str)
    decisions["pre_selected"] = _bool_series(decisions.get("was_selected", pd.Series(False, index=decisions.index)))
    decisions["decision_exclude_reason"] = decisions.get("exclude_reason", pd.Series("", index=decisions.index)).fillna("").astype(str)
    decisions["entry_raw_action"] = decisions.get("entry_raw_action", pd.Series("OBSERVE", index=decisions.index)).fillna("OBSERVE")
    decisions["final_action"] = decisions.get("final_action", pd.Series("OBSERVE", index=decisions.index)).fillna("OBSERVE")
    return decisions[["trade_date", "code", "pre_selected", "decision_exclude_reason", "entry_raw_action", "final_action"]]


def _first_existing(df: pd.DataFrame, *columns: str) -> pd.Series:
    for column in columns:
        if column in df.columns:
            return df[column].fillna("").astype(str)
    return pd.Series("", index=df.index)


def _bool_series(values: Any) -> pd.Series:
    if isinstance(values, pd.Series):
        if values.dtype == bool:
            return values.fillna(False)
        return values.fillna(False).map(lambda value: str(value).strip().lower() in {"1", "true", "yes", "y", "selected"})
    return pd.Series(bool(values))


def _value_counts(values: pd.Series | None) -> dict[str, int]:
    if values is None:
        return {}
    return {str(k): int(v) for k, v in values.fillna("").astype(str).value_counts().sort_index().items()}
