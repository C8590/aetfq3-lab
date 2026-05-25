from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .io_utils import ensure_dir, read_table, write_table


POLICY_ORDER = ("legacy_v21", "ml_shadow", "ml_candidate_expansion", "ml_active_sim")
POLICY_LABELS = {
    "legacy_v21": "A legacy_v21",
    "ml_shadow": "B ml_shadow",
    "ml_candidate_expansion": "C ml_candidate_expansion",
    "ml_active_sim": "D ml_active_sim",
}
COMPARISON_COLUMNS = [
    "policy",
    "policy_label",
    "evaluation_start",
    "evaluation_end",
    "trade_date_count",
    "sample_count",
    "daily_probe_count",
    "daily_buy_count",
    "good_entry_precision_top5",
    "good_entry_precision_top20",
    "bad_entry_rate",
    "future_10d_return",
    "future_10d_max_drawdown",
    "outperform_market_rate",
    "outperform_sector_rate",
    "turnover",
    "missed_winner_count",
    "false_positive_count",
]
REQUIRED_SAMPLE_COLUMNS = {
    "trade_date",
    "code",
    "auto_label",
    "label_status",
    "future_return_10d",
    "future_max_drawdown_10d",
    "outperform_market_10d",
    "outperform_sector_10d",
}
REQUIRED_SCORE_COLUMNS = {"trade_date", "code", "ml_score", "ml_action_suggestion"}
FORBIDDEN_POLICY_SCORE_PREFIXES = ("future_return_", "future_max_", "outperform_")
FORBIDDEN_POLICY_SCORE_COLUMNS = {"auto_label", "label_status", "label_reason_cn", "hit_stop_loss_10d"}


@dataclass(frozen=True)
class PolicyComparisonResult:
    comparison: pd.DataFrame
    report: str
    policy_rows: pd.DataFrame
    report_path: Path
    csv_path: Path


def compare_ml_policies_from_files(
    *,
    labeled_samples_path: str | Path,
    ml_scores_path: str | Path,
    out_dir: str | Path,
    output_format: str = "csv",
    top_ns: Iterable[int] = (5, 20),
) -> PolicyComparisonResult:
    samples = read_table(labeled_samples_path)
    scores = read_table(ml_scores_path)
    return compare_ml_policies(samples, scores, out_dir=out_dir, output_format=output_format, top_ns=top_ns)


def compare_ml_policies(
    labeled_samples: pd.DataFrame,
    ml_scores: pd.DataFrame,
    *,
    out_dir: str | Path,
    output_format: str = "csv",
    top_ns: Iterable[int] = (5, 20),
) -> PolicyComparisonResult:
    frame = _prepare_frame(labeled_samples, ml_scores)
    policy_rows = pd.concat([_apply_policy(frame, policy) for policy in POLICY_ORDER], ignore_index=True)
    comparison = pd.DataFrame([_policy_metrics(policy_rows, policy, top_ns=top_ns) for policy in POLICY_ORDER])
    comparison = comparison[COMPARISON_COLUMNS]

    report = build_policy_comparison_report(comparison, policy_rows)
    reports_dir = ensure_dir(Path(out_dir) / "reports")
    csv_path = write_table(comparison, reports_dir, "ml_policy_comparison", output_format)
    report_path = reports_dir / "ml_policy_comparison_report.md"
    report_path.write_text(report, encoding="utf-8")
    return PolicyComparisonResult(
        comparison=comparison,
        report=report,
        policy_rows=policy_rows,
        report_path=report_path,
        csv_path=csv_path,
    )


def build_policy_comparison_report(comparison: pd.DataFrame, policy_rows: pd.DataFrame) -> str:
    legacy = _row_for(comparison, "legacy_v21")
    active = _row_for(comparison, "ml_active_sim")
    shadow = _row_for(comparison, "ml_shadow")
    candidate = _row_for(comparison, "ml_candidate_expansion")
    decision = _active_sim_decision(legacy, active)
    probe_lines = _probe_change_lines(comparison)
    failure_lines = _failure_mode_lines(policy_rows)
    advantage = _advantage_summary(legacy, shadow, candidate, active)

    lines = [
        "# ML Policy Comparison Report",
        "",
        "## Control Center Conclusion",
        "",
        f"- active_sim_permission: {decision}",
        f"- summary: {advantage}",
        "- formal_parameter_change: no. This report is offline evidence only and does not modify entry thresholds, QMT, or live trading parameters.",
        "",
        "## Time Split And Leakage Control",
        "",
        "- Evaluation uses only dates that have walk-forward `ml_entry_scores` rows.",
        "- `ml_entry_scores` is validated as policy-safe: future_return_*, future_max_*, outperform_*, auto_label, and label_status columns are rejected.",
        "- Each scored date must be produced by training data strictly earlier than that date; labels are used only after replay for metrics.",
        "- Replay behavior columns are preserved for `legacy_v21`; future_return_*, outperform_*, and auto_label are never used to choose policy actions.",
        "",
        "## Metric Table",
        "",
        _markdown_table(comparison),
        "",
        "## PROBE Quantity Change",
        "",
        *probe_lines,
        "",
        "## Failure Modes",
        "",
        *failure_lines,
        "",
        "## Version Notes",
        "",
        "- A legacy_v21: uses historical replay rule actions and replay candidate coverage.",
        "- B ml_shadow: ranks by ML score for observation, but BUY/PROBE actions remain identical to legacy.",
        "- C ml_candidate_expansion: expands review coverage with ML top20 candidates, but does not change BUY/PROBE actions.",
        "- D ml_active_sim: applies ML UPGRADE_PROBE and DOWNGRADE/WAIT/FORBID suggestions to simulation-only BUY/PROBE/OBSERVE/AVOID layering.",
        "",
    ]
    return "\n".join(lines)


def _prepare_frame(labeled_samples: pd.DataFrame, ml_scores: pd.DataFrame) -> pd.DataFrame:
    _require_columns(labeled_samples, REQUIRED_SAMPLE_COLUMNS, "labeled_samples")
    _require_columns(ml_scores, REQUIRED_SCORE_COLUMNS, "ml_scores")
    _assert_policy_scores_are_label_free(ml_scores)

    samples = labeled_samples.copy()
    scores = ml_scores.copy()
    samples["trade_date"] = pd.to_datetime(samples["trade_date"], errors="coerce").dt.normalize()
    scores["trade_date"] = pd.to_datetime(scores["trade_date"], errors="coerce").dt.normalize()
    samples["code"] = samples["code"].astype(str)
    scores["code"] = scores["code"].astype(str)
    scores = scores.sort_values(["trade_date", "code"]).drop_duplicates(["trade_date", "code"], keep="last")

    evaluation_dates = set(scores["trade_date"].dropna().unique())
    samples = samples.loc[samples["trade_date"].isin(evaluation_dates)].copy()
    samples = samples.loc[samples["label_status"].fillna("").astype(str).eq("ok")].copy()
    if samples.empty:
        raise ValueError("no label_status=ok samples overlap with walk-forward ml_entry_scores dates")

    for optional in ("ml_rank_global", "p_good_entry", "p_bad_entry"):
        if optional not in scores.columns:
            scores[optional] = np.nan
    frame = samples.merge(
        scores[["trade_date", "code", "ml_score", "ml_rank_global", "ml_action_suggestion", "p_good_entry", "p_bad_entry"]],
        on=["trade_date", "code"],
        how="inner",
    )
    if frame.empty:
        raise ValueError("labeled samples and ml_entry_scores have no matching trade_date/code rows")

    frame["ml_score"] = pd.to_numeric(frame["ml_score"], errors="coerce").fillna(-np.inf)
    if "ml_rank_global" not in frame.columns or frame["ml_rank_global"].isna().all():
        frame["ml_rank_global"] = frame.groupby("trade_date")["ml_score"].rank(ascending=False, method="first")
    frame["ml_rank_global"] = pd.to_numeric(frame["ml_rank_global"], errors="coerce")
    for col in ["future_return_10d", "future_max_drawdown_10d"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    for col in ["outperform_market_10d", "outperform_sector_10d", "was_candidate", "was_selected", "was_bought", "pre_selected", "is_valid_sample"]:
        if col not in frame.columns:
            frame[col] = False
        frame[col] = frame[col].map(_truthy)
    if "entry_score" not in frame.columns:
        frame["entry_score"] = frame.get("momentum_score", 0)
    frame["entry_score"] = pd.to_numeric(frame["entry_score"], errors="coerce").fillna(0.0)
    frame["legacy_action"] = frame.apply(_legacy_action, axis=1)
    frame["legacy_covered"] = (
        frame["was_candidate"] | frame["was_selected"] | frame["was_bought"] | frame["pre_selected"] | frame["legacy_action"].isin(["BUY", "PROBE"])
    )
    frame["is_good"] = frame["auto_label"].astype(str).eq("good_entry")
    frame["is_bad"] = frame["auto_label"].astype(str).eq("bad_entry")
    frame["is_winner"] = frame["is_good"] | frame["outperform_market_10d"] | frame["outperform_sector_10d"]
    return frame.sort_values(["trade_date", "code"]).reset_index(drop=True)


def _apply_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    out = frame.copy()
    out["policy"] = policy
    if policy == "legacy_v21":
        out["policy_action"] = out["legacy_action"]
        out["policy_covered"] = out["legacy_covered"]
        out["policy_rank_score"] = out.apply(_legacy_rank_score, axis=1)
        out["policy_adjustment"] = "LEGACY_RULE"
    elif policy == "ml_shadow":
        out["policy_action"] = out["legacy_action"]
        out["policy_covered"] = out["legacy_covered"]
        out["policy_rank_score"] = out["ml_score"]
        out["policy_adjustment"] = "ML_SHADOW_ONLY"
    elif policy == "ml_candidate_expansion":
        out["policy_action"] = out["legacy_action"]
        top20 = out.groupby("trade_date")["ml_score"].rank(ascending=False, method="first").le(20)
        upgrade = out["ml_action_suggestion"].fillna("").astype(str).str.upper().eq("UPGRADE_PROBE")
        out["policy_covered"] = out["legacy_covered"] | top20 | upgrade
        out["policy_rank_score"] = out["ml_score"]
        out["policy_adjustment"] = np.where(out["legacy_covered"], "LEGACY_COVERED", "ML_CANDIDATE_EXPANDED")
    elif policy == "ml_active_sim":
        adjusted = out.apply(_active_sim_action, axis=1, result_type="expand")
        out["policy_action"] = adjusted[0]
        out["policy_adjustment"] = adjusted[1]
        out["policy_covered"] = out["legacy_covered"] | out["policy_action"].isin(["BUY", "PROBE"])
        out["policy_rank_score"] = out["ml_score"]
    else:
        raise ValueError(f"unknown policy: {policy}")
    out["policy_actionable"] = out["policy_action"].isin(["BUY", "PROBE"])
    return out


def _policy_metrics(policy_rows: pd.DataFrame, policy: str, top_ns: Iterable[int]) -> dict[str, Any]:
    frame = policy_rows.loc[policy_rows["policy"].eq(policy)].copy()
    dates = sorted(frame["trade_date"].dropna().unique())
    actionable = frame.loc[frame["policy_actionable"]].copy()
    covered = frame.loc[frame["policy_covered"]].copy()
    metrics = {
        "policy": policy,
        "policy_label": POLICY_LABELS[policy],
        "evaluation_start": "" if not dates else pd.Timestamp(dates[0]).strftime("%Y-%m-%d"),
        "evaluation_end": "" if not dates else pd.Timestamp(dates[-1]).strftime("%Y-%m-%d"),
        "trade_date_count": len(dates),
        "sample_count": len(frame),
        "daily_probe_count": _daily_count(frame, "PROBE"),
        "daily_buy_count": _daily_count(frame, "BUY"),
        "bad_entry_rate": _rate(actionable["is_bad"]) if not actionable.empty else 0.0,
        "future_10d_return": _mean(actionable["future_return_10d"]),
        "future_10d_max_drawdown": _mean(actionable["future_max_drawdown_10d"]),
        "outperform_market_rate": _rate(actionable["outperform_market_10d"]) if not actionable.empty else 0.0,
        "outperform_sector_rate": _rate(actionable["outperform_sector_10d"]) if not actionable.empty else 0.0,
        "turnover": _turnover(frame),
        "missed_winner_count": int((frame["is_winner"] & ~frame["policy_covered"]).sum()),
        "false_positive_count": int((frame["policy_actionable"] & frame["is_bad"]).sum()),
    }
    for top_n in top_ns:
        metrics[f"good_entry_precision_top{top_n}"] = _top_precision(frame, int(top_n))
    return metrics


def _top_precision(frame: pd.DataFrame, top_n: int) -> float:
    values: list[float] = []
    for _, group in frame.groupby("trade_date", sort=True):
        pool = group.loc[group["policy_covered"]].copy()
        if pool.empty:
            pool = group.copy()
        top = pool.sort_values(["policy_rank_score", "entry_score"], ascending=[False, False]).head(top_n)
        if not top.empty:
            values.append(float(top["is_good"].mean()))
    return round(float(np.mean(values)), 6) if values else 0.0


def _daily_count(frame: pd.DataFrame, action: str) -> float:
    if frame.empty:
        return 0.0
    daily = frame.assign(_hit=frame["policy_action"].eq(action)).groupby("trade_date")["_hit"].sum()
    return round(float(daily.mean()), 6) if not daily.empty else 0.0


def _turnover(frame: pd.DataFrame) -> float:
    prev: set[str] | None = None
    values: list[float] = []
    for _, group in frame.groupby("trade_date", sort=True):
        current = set(group.loc[group["policy_actionable"], "code"].astype(str))
        if prev is not None:
            union = current | prev
            changed = current.symmetric_difference(prev)
            values.append(0.0 if not union else len(changed) / len(union))
        prev = current
    return round(float(np.mean(values)), 6) if values else 0.0


def _active_sim_action(row: pd.Series) -> tuple[str, str]:
    rule = str(row.get("legacy_action") or "OBSERVE").upper()
    suggestion = str(row.get("ml_action_suggestion") or "NO_ML").upper()
    if suggestion == "UPGRADE_PROBE" and rule in {"OBSERVE", "REJECT", "AVOID"}:
        return "PROBE", "ML_RECOVERED"
    if suggestion in {"DOWNGRADE_WATCH", "WAIT_PULLBACK"} and rule in {"BUY", "PROBE"}:
        return "OBSERVE", "ML_DOWNGRADED"
    if suggestion == "FORBID_CHASE" and rule in {"BUY", "PROBE"}:
        return "AVOID", "ML_DOWNGRADED"
    return rule, "ML_KEEP_RULE"


def _legacy_action(row: pd.Series) -> str:
    for col in ("final_action", "entry_raw_action", "raw_entry_action", "final_buy_action"):
        value = row.get(col)
        action = _action_label(value)
        if action:
            return action
    if _truthy(row.get("was_bought")):
        return "BUY"
    if _truthy(row.get("was_selected")) or _truthy(row.get("pre_selected")):
        return "PROBE"
    return "OBSERVE"


def _action_label(value: Any) -> str:
    text = str(value or "").strip()
    upper = text.upper()
    lower = text.lower()
    if upper in {"BUY", "PROBE", "OBSERVE", "REJECT", "AVOID", "BLOCKED"}:
        return upper
    if "probe" in lower or "试探" in text:
        return "PROBE"
    if "buy" in lower or "买入" in text or "加仓" in text:
        return "BUY"
    if "avoid" in lower:
        return "AVOID"
    if "reject" in lower or "forbid" in lower or "禁止" in text:
        return "REJECT"
    if "block" in lower or "阻断" in text:
        return "BLOCKED"
    return "OBSERVE"


def _legacy_rank_score(row: pd.Series) -> float:
    action_bonus = {"BUY": 3.0, "PROBE": 2.0, "OBSERVE": 0.0, "REJECT": -1.0, "AVOID": -1.0, "BLOCKED": -2.0}.get(
        str(row.get("legacy_action") or "OBSERVE"), 0.0
    )
    covered_bonus = 1.0 if _truthy(row.get("legacy_covered")) else 0.0
    return action_bonus * 1_000_000.0 + covered_bonus * 10_000.0 + float(row.get("entry_score") or 0.0)


def _active_sim_decision(legacy: pd.Series, active: pd.Series) -> str:
    legacy_return = float(legacy.get("future_10d_return") or 0.0)
    active_return = float(active.get("future_10d_return") or 0.0)
    legacy_bad = float(legacy.get("bad_entry_rate") or 0.0)
    active_bad = float(active.get("bad_entry_rate") or 0.0)
    legacy_false = float(legacy.get("false_positive_count") or 0.0)
    active_false = float(active.get("false_positive_count") or 0.0)
    if active_return >= legacy_return and active_bad <= legacy_bad and active_false <= legacy_false:
        return "ALLOW_ACTIVE_SIM_FOR_V21_SIMULATION"
    if active_return >= legacy_return and active_bad <= legacy_bad * 1.15:
        return "REVIEW_ACTIVE_SIM_WITH_MANUAL_GUARDRAILS"
    return "DO_NOT_ALLOW_ACTIVE_SIM_YET"


def _advantage_summary(legacy: pd.Series, shadow: pd.Series, candidate: pd.Series, active: pd.Series) -> str:
    parts = []
    if float(shadow.get("good_entry_precision_top5") or 0.0) > float(legacy.get("good_entry_precision_top5") or 0.0):
        parts.append("ML top5 ranking precision is better than legacy.")
    else:
        parts.append("ML top5 ranking precision is not better than legacy.")
    if int(candidate.get("missed_winner_count") or 0) < int(legacy.get("missed_winner_count") or 0):
        parts.append("Candidate expansion reduces missed winners.")
    if float(active.get("bad_entry_rate") or 0.0) > float(legacy.get("bad_entry_rate") or 0.0):
        parts.append("active_sim increases bad_entry risk.")
    return " ".join(parts)


def _probe_change_lines(comparison: pd.DataFrame) -> list[str]:
    legacy_probe = float(_row_for(comparison, "legacy_v21").get("daily_probe_count") or 0.0)
    lines = []
    for policy in POLICY_ORDER:
        row = _row_for(comparison, policy)
        probe = float(row.get("daily_probe_count") or 0.0)
        delta = probe - legacy_probe
        if policy == "ml_shadow":
            reason = "shadow does not change actions, so probe count should remain near legacy."
        elif policy == "ml_candidate_expansion":
            reason = "candidate expansion changes review coverage only, not BUY/PROBE actions."
        elif policy == "ml_active_sim":
            reason = "active_sim can recover OBSERVE to PROBE and downgrade risky BUY/PROBE to OBSERVE/AVOID."
        else:
            reason = "baseline replay rule behavior."
        lines.append(f"- {POLICY_LABELS[policy]}: daily_probe_count={probe:.4f}, delta_vs_legacy={delta:.4f}. {reason}")
    return lines


def _failure_mode_lines(policy_rows: pd.DataFrame) -> list[str]:
    active = policy_rows.loc[policy_rows["policy"].eq("ml_active_sim")]
    recovered_bad = int(((active["policy_adjustment"] == "ML_RECOVERED") & active["is_bad"]).sum())
    downgraded_good = int(((active["policy_adjustment"] == "ML_DOWNGRADED") & active["is_good"]).sum())
    missed = active.loc[active["is_winner"] & ~active["policy_covered"]].copy()
    false_positive = active.loc[active["policy_actionable"] & active["is_bad"]].copy()
    lines = [
        f"- ML_RECOVERED false positives: {recovered_bad}.",
        f"- ML_DOWNGRADED good entries: {downgraded_good}.",
        f"- Remaining missed winners: {len(missed)}; top codes: {_top_values(missed, 'code')}.",
        f"- Active-sim false positives: {len(false_positive)}; top codes: {_top_values(false_positive, 'code')}.",
    ]
    if recovered_bad:
        lines.append("- Failure pattern: ML recovery can over-trust high score OBSERVE rows that later label as bad_entry.")
    if downgraded_good:
        lines.append("- Failure pattern: ML downgrade can suppress valid legacy opportunities.")
    if len(missed):
        lines.append("- Failure pattern: winners still outside candidate/active coverage need pre_selection recall review, not automatic parameter changes.")
    return lines


def _markdown_table(df: pd.DataFrame) -> str:
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda value: "" if pd.isna(value) else f"{value:.6f}")
    columns = [str(col) for col in display.columns]
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        rows.append("| " + " | ".join(str(row.get(col, "")) for col in display.columns) + " |")
    return "\n".join(rows)


def _row_for(df: pd.DataFrame, policy: str) -> pd.Series:
    rows = df.loc[df["policy"].eq(policy)]
    if rows.empty:
        return pd.Series(dtype=object)
    return rows.iloc[0]


def _top_values(df: pd.DataFrame, column: str, limit: int = 5) -> str:
    if df.empty or column not in df:
        return "none"
    counts = df[column].fillna("").astype(str).value_counts().head(limit)
    return ", ".join(f"{key}:{int(value)}" for key, value in counts.items()) if not counts.empty else "none"


def _rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return round(float(series.map(_truthy).mean()), 6)


def _mean(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return round(float(values.mean()), 6) if not values.empty else 0.0


def _truthy(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    text = str(value).strip().lower()
    if text in {"1", "1.0", "true", "yes", "y", "selected", "buy", "probe", "是"}:
        return True
    try:
        return bool(float(text))
    except (TypeError, ValueError):
        return False


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}")


def _assert_policy_scores_are_label_free(scores: pd.DataFrame) -> None:
    leaked = []
    for column in scores.columns:
        text = str(column)
        if text in FORBIDDEN_POLICY_SCORE_COLUMNS or any(text.startswith(prefix) for prefix in FORBIDDEN_POLICY_SCORE_PREFIXES):
            leaked.append(text)
    if leaked:
        raise ValueError(f"ml_scores contains future label columns and cannot drive policy comparison: {sorted(leaked)}")
