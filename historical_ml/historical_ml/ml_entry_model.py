from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .daily_universe import generated_output_dir
from .io_utils import read_table, write_table
from .ml_baseline import _fit_logistic, _roc_auc


MODEL_VERSION = "entry_quality_numpy_logistic_v1"
FEATURE_VERSION = "daily_ml_universe_features_v1"
FEATURE_COLUMNS_NUMERIC = [
    "momentum_20",
    "momentum_60",
    "momentum_120",
    "momentum_score",
    "acceleration_score",
    "volatility_20",
    "drawdown_20",
    "drawdown_60",
    "sector_rank",
    "etf_rank",
]
FEATURE_COLUMNS_CATEGORICAL = [
    "market_state",
    "sector_state",
    "sector_level1",
    "sector_level2",
    "pre_selected",
]
FORBIDDEN_FEATURE_PREFIXES = (
    "future_return_",
    "future_max_",
    "outperform_",
)
FORBIDDEN_FEATURE_EXACT = {
    "auto_label",
    "label_reason_cn",
    "label_status",
    "label_after_t",
    "hit_stop_loss_10d",
    "expected_return_10d",
    "expected_drawdown_10d",
    "p_good_entry",
    "p_bad_entry",
    "ml_score",
    "entry_raw_action",
    "final_action",
    "code",
    "name",
}
SCORE_COLUMNS = [
    "trade_date",
    "code",
    "name",
    "p_good_entry",
    "p_bad_entry",
    "expected_return_10d",
    "expected_drawdown_10d",
    "ml_score",
    "ml_rank_global",
    "ml_rank_sector",
    "ml_action_suggestion",
    "ml_reason_code",
    "ml_reason_cn",
    "model_version",
    "feature_version",
]


@dataclass(frozen=True)
class EntryModelSplit:
    train_start: str
    train_end: str
    validation_start: str
    validation_end: str
    test_start: str
    test_end: str
    train_dates: list[pd.Timestamp]
    validation_dates: list[pd.Timestamp]
    test_dates: list[pd.Timestamp]
    note: str


@dataclass
class EntryQualityModelResult:
    scores: pd.DataFrame
    report: str
    split: EntryModelSplit
    feature_names: list[str]
    walk_forward_metrics: pd.DataFrame


def train_entry_quality_model_from_file(
    samples_path: str | Path,
    out_dir: str | Path,
    output_format: str = "csv",
    top_n: int = 5,
    min_train_dates: int = 5,
) -> EntryQualityModelResult:
    samples = read_table(samples_path)
    return train_entry_quality_model(
        samples,
        out_dir=out_dir,
        output_format=output_format,
        top_n=top_n,
        min_train_dates=min_train_dates,
    )


def score_entry_quality_universe_from_files(
    *,
    labeled_samples_path: str | Path,
    scoring_samples_path: str | Path,
    out_dir: str | Path,
    output_format: str = "csv",
    score_date: str | None = None,
    min_train_dates: int = 5,
) -> pd.DataFrame:
    labeled = read_table(labeled_samples_path)
    scoring = read_table(scoring_samples_path)
    return score_entry_quality_universe(
        labeled,
        scoring,
        out_dir=out_dir,
        output_format=output_format,
        score_date=score_date,
        min_train_dates=min_train_dates,
    )


def score_entry_quality_universe(
    labeled_samples: pd.DataFrame,
    scoring_samples: pd.DataFrame,
    *,
    out_dir: str | Path,
    output_format: str = "csv",
    score_date: str | None = None,
    min_train_dates: int = 5,
) -> pd.DataFrame:
    """Train on historical labels and score the requested current universe.

    This path is intentionally separate from walk-forward validation: validation
    scores prove model behavior, while this function produces today's shadow
    inference rows for every feature-ready ETF sample.
    """

    train = prepare_entry_quality_samples(labeled_samples)
    if train.empty:
        raise ValueError("no label_status=ok ml_entry_labeled_samples available for entry quality model")

    score_df = prepare_entry_quality_scoring_samples(scoring_samples)
    if score_df.empty:
        raise ValueError("no feature-ready daily_ml_universe_samples available for entry quality scoring")
    target_date = pd.Timestamp(score_date).normalize() if score_date else pd.Timestamp(score_df["trade_date"].max()).normalize()
    score_df = score_df.loc[score_df["trade_date"] == target_date].copy()
    score_df = score_df.loc[score_df["is_valid_sample"].map(_truthy)].copy()
    if score_df.empty:
        raise ValueError(f"no feature-ready samples available for score_date={target_date.date()}")

    train = train.loc[train["trade_date"] < target_date].copy()
    if len(train["trade_date"].dropna().unique()) < min_train_dates:
        raise ValueError(
            f"not enough historical training dates before {target_date.date()}: "
            f"{len(train['trade_date'].dropna().unique())} < {min_train_dates}"
        )

    feature_names = list(build_entry_feature_matrix(train).columns)
    validate_no_label_leakage_features(feature_names)
    scores = _score_one_date(train, score_df, feature_names)
    scores["split"] = "current_inference"
    scores = _rank_scores(scores)
    scores_out = scores[SCORE_COLUMNS].copy()
    generated = generated_output_dir(out_dir)
    write_table(scores_out, generated, "ml_entry_scores", output_format)
    return scores_out


def train_entry_quality_model(
    labeled_samples: pd.DataFrame,
    out_dir: str | Path,
    output_format: str = "csv",
    top_n: int = 5,
    min_train_dates: int = 5,
) -> EntryQualityModelResult:
    df = prepare_entry_quality_samples(labeled_samples)
    if df.empty:
        raise ValueError("no label_status=ok ml_entry_labeled_samples available for entry quality model")

    split = build_train_validation_test_split(df)
    feature_template = build_entry_feature_matrix(df)
    feature_names = list(feature_template.columns)
    validate_no_label_leakage_features(feature_names)
    scored_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    score_dates = split.validation_dates + split.test_dates

    for score_date in score_dates:
        train = df.loc[df["trade_date"] < score_date].copy()
        score_df = df.loc[df["trade_date"] == score_date].copy()
        if len(train["trade_date"].dropna().unique()) < min_train_dates or score_df.empty:
            continue
        scored = _score_one_date(train, score_df, feature_names)
        scored["split"] = "validation" if score_date in set(split.validation_dates) else "test"
        scored_frames.append(scored)
        metric_rows.append(_walk_forward_metric_row(scored, top_n=top_n))

    scores = pd.concat(scored_frames, ignore_index=True) if scored_frames else pd.DataFrame(columns=SCORE_COLUMNS)
    if not scores.empty:
        scores = _rank_scores(scores)
    scores_out = scores[SCORE_COLUMNS].copy() if not scores.empty else pd.DataFrame(columns=SCORE_COLUMNS)
    report = build_model_report(df, split, scores, pd.DataFrame(metric_rows), feature_names, top_n=top_n)

    generated = generated_output_dir(out_dir)
    write_table(scores_out, generated, "ml_entry_scores", output_format)
    (generated / "model_report.md").write_text(report, encoding="utf-8")
    return EntryQualityModelResult(
        scores=scores_out,
        report=report,
        split=split,
        feature_names=feature_names,
        walk_forward_metrics=pd.DataFrame(metric_rows),
    )


def prepare_entry_quality_samples(samples: pd.DataFrame) -> pd.DataFrame:
    df = samples.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    if "label_status" not in df.columns:
        df["label_status"] = "ok"
    df = df.loc[df["label_status"].fillna("").astype(str).eq("ok")].copy()
    if "is_valid_sample" in df.columns:
        df = df.loc[df["is_valid_sample"].map(_truthy)].copy()
    df = df.loc[df["auto_label"].isin(["good_entry", "bad_entry", "neutral_entry"])].copy()
    df["is_good_entry"] = (df["auto_label"] == "good_entry").astype(int)
    df["is_bad_entry"] = (df["auto_label"] == "bad_entry").astype(int)
    for col in FEATURE_COLUMNS_NUMERIC + ["future_return_10d", "future_max_drawdown_10d"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FEATURE_COLUMNS_CATEGORICAL + ["code", "name"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).replace("", "unknown")
    df["pre_selected"] = df["pre_selected"].map(lambda value: "true" if _truthy(value) else "false")
    return df.sort_values(["trade_date", "code"]).reset_index(drop=True)


def prepare_entry_quality_scoring_samples(samples: pd.DataFrame) -> pd.DataFrame:
    df = samples.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.normalize()
    df = df.dropna(subset=["trade_date"]).copy()
    if "is_valid_sample" not in df.columns:
        df["is_valid_sample"] = True
    for col in FEATURE_COLUMNS_NUMERIC:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in FEATURE_COLUMNS_CATEGORICAL + ["code", "name"]:
        if col not in df.columns:
            df[col] = "unknown"
        df[col] = df[col].fillna("unknown").astype(str).replace("", "unknown")
    df["pre_selected"] = df["pre_selected"].map(lambda value: "true" if _truthy(value) else "false")
    return df.sort_values(["trade_date", "code"]).reset_index(drop=True)


def build_train_validation_test_split(df: pd.DataFrame) -> EntryModelSplit:
    dates = list(pd.to_datetime(df["trade_date"].dropna().sort_values().unique()))
    if not dates:
        return EntryModelSplit("", "", "", "", "", "", [], [], [], "no valid dates")
    n = len(dates)
    train_end_idx = max(0, min(n - 3, int(np.floor(n * 0.60)) - 1))
    validation_end_idx = max(train_end_idx + 1, min(n - 2, int(np.floor(n * 0.80)) - 1))
    train_dates = dates[: train_end_idx + 1]
    validation_dates = dates[train_end_idx + 1 : validation_end_idx + 1]
    test_dates = dates[validation_end_idx + 1 :]
    if not validation_dates and len(test_dates) > 1:
        validation_dates = test_dates[:1]
        test_dates = test_dates[1:]
    if not test_dates and validation_dates:
        test_dates = validation_dates[-1:]
        validation_dates = validation_dates[:-1]
    return EntryModelSplit(
        train_start=_date_min(train_dates),
        train_end=_date_max(train_dates),
        validation_start=_date_min(validation_dates),
        validation_end=_date_max(validation_dates),
        test_start=_date_min(test_dates),
        test_end=_date_max(test_dates),
        train_dates=train_dates,
        validation_dates=validation_dates,
        test_dates=test_dates,
        note="chronological 60/20/20 split; walk-forward scores train only on dates before each scored date",
    )


def build_entry_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    pieces: list[pd.DataFrame] = []
    numeric = tmp[FEATURE_COLUMNS_NUMERIC].apply(pd.to_numeric, errors="coerce")
    med = numeric.median(numeric_only=True).fillna(0.0)
    pieces.append(numeric.fillna(med).astype(float))
    categorical = tmp[FEATURE_COLUMNS_CATEGORICAL].copy()
    for col in FEATURE_COLUMNS_CATEGORICAL:
        categorical[col] = categorical[col].fillna("unknown").astype(str).replace("", "unknown")
    pieces.append(pd.get_dummies(categorical, columns=FEATURE_COLUMNS_CATEGORICAL, prefix_sep="=", dtype=float))
    matrix = pd.concat(pieces, axis=1)
    matrix = matrix.loc[:, ~matrix.columns.duplicated()].astype(float)
    validate_no_label_leakage_features(matrix.columns)
    return matrix


def validate_no_label_leakage_features(feature_names) -> None:
    leaked = []
    for name in feature_names:
        text = str(name)
        base = text.split("=", 1)[0]
        if text in FORBIDDEN_FEATURE_EXACT or base in FORBIDDEN_FEATURE_EXACT:
            leaked.append(text)
        if any(text.startswith(prefix) or base.startswith(prefix) for prefix in FORBIDDEN_FEATURE_PREFIXES):
            leaked.append(text)
    if leaked:
        raise ValueError(f"forbidden entry model feature leakage detected: {sorted(set(leaked))}")


def build_model_report(
    df: pd.DataFrame,
    split: EntryModelSplit,
    scores: pd.DataFrame,
    walk_forward_metrics: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 5,
) -> str:
    lines = [
        "# model_report",
        "",
        "This is a shadow-mode ETF code-level entry quality model. It does not change entry buy_action, final_action, QMT, or control_center decisions.",
        "",
        "## Versions",
        "",
        f"- model_version: {MODEL_VERSION}",
        f"- feature_version: {FEATURE_VERSION}",
        "- model_backend: numpy_logistic_fallback",
        "",
        "## Split",
        "",
        f"- train: {split.train_start} to {split.train_end}",
        f"- validation: {split.validation_start} to {split.validation_end}",
        f"- test: {split.test_start} to {split.test_end}",
        f"- note: {split.note}",
        "",
        "## Walk-Forward Validation",
        "",
        _walk_forward_table(walk_forward_metrics),
        "",
        "## TopN Precision And Bad Entry Rate",
        "",
        _topn_table(scores, top_n=top_n),
        "",
        "## Baseline Comparison",
        "",
        _baseline_comparison(df, scores, top_n=top_n),
        "",
        "## Feature Contract",
        "",
        f"- feature_count: {len(feature_names)}",
        "- feature inputs are feature_at_t only; future_return_*, future_max_*, outperform_*, auto_label, label_reason_cn, code, and name are forbidden as model features.",
        "- output is ETF code-level shadow score: p_good_entry, p_bad_entry, expected_return_10d, expected_drawdown_10d, and ml_score.",
        "",
    ]
    return "\n".join(lines)


def _score_one_date(train: pd.DataFrame, score_df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    x_train = build_entry_feature_matrix(train).reindex(columns=feature_names, fill_value=0.0).to_numpy(dtype=float)
    x_score = build_entry_feature_matrix(score_df).reindex(columns=feature_names, fill_value=0.0).to_numpy(dtype=float)
    good_model = _fit_logistic(x_train, train["is_good_entry"].to_numpy(dtype=int))
    bad_model = _fit_logistic(x_train, train["is_bad_entry"].to_numpy(dtype=int))
    p_good = good_model.predict_proba(x_score)
    p_bad = bad_model.predict_proba(x_score)
    expected_return, expected_drawdown = _expected_outcomes(train, p_good, p_bad)
    out = score_df.copy()
    out["p_good_entry"] = p_good
    out["p_bad_entry"] = p_bad
    out["expected_return_10d"] = expected_return
    out["expected_drawdown_10d"] = expected_drawdown
    out["ml_score"] = 100.0 * (out["p_good_entry"] - out["p_bad_entry"]) + 20.0 * out["expected_return_10d"].fillna(0.0)
    suggestions = out.apply(_suggestion_row, axis=1)
    out["ml_action_suggestion"] = [item[0] for item in suggestions]
    out["ml_reason_code"] = [item[1] for item in suggestions]
    out["ml_reason_cn"] = [item[2] for item in suggestions]
    out["model_version"] = MODEL_VERSION
    out["feature_version"] = FEATURE_VERSION
    return out


def _expected_outcomes(train: pd.DataFrame, p_good: np.ndarray, p_bad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    neutral_weight = np.clip(1.0 - p_good - p_bad, 0.0, 1.0)
    means = {}
    for label in ["good_entry", "bad_entry", "neutral_entry"]:
        subset = train.loc[train["auto_label"] == label]
        means[(label, "return")] = float(subset["future_return_10d"].mean()) if not subset.empty else float(train["future_return_10d"].mean())
        means[(label, "drawdown")] = float(subset["future_max_drawdown_10d"].mean()) if not subset.empty else float(train["future_max_drawdown_10d"].mean())
    expected_return = (
        p_good * means[("good_entry", "return")]
        + p_bad * means[("bad_entry", "return")]
        + neutral_weight * means[("neutral_entry", "return")]
    )
    expected_drawdown = (
        p_good * means[("good_entry", "drawdown")]
        + p_bad * means[("bad_entry", "drawdown")]
        + neutral_weight * means[("neutral_entry", "drawdown")]
    )
    return expected_return, expected_drawdown


def _suggestion_row(row: pd.Series) -> tuple[str, str, str]:
    p_good = float(row.get("p_good_entry", 0.0))
    p_bad = float(row.get("p_bad_entry", 0.0))
    score = float(row.get("ml_score", 0.0))
    if p_bad >= 0.45 and p_bad >= p_good + 0.10:
        return ("DOWNGRADE_WATCH", "HIGH_BAD_RISK", "bad_entry 概率偏高，shadow 模式建议 entry 侧重点复核。")
    if p_good >= 0.45 and p_good >= p_bad + 0.10 and score > 0:
        return ("UPGRADE_PROBE", "HIGH_GOOD_QUALITY", "good_entry 概率占优，shadow 模式提示可进入买点质量复核。")
    if score >= 0:
        return ("KEEP_ORIGINAL", "NEUTRAL_POSITIVE", "模型分数略偏正，shadow 模式保持原 entry 判断。")
    return ("KEEP_ORIGINAL", "NEUTRAL_CAUTION", "模型分数偏谨慎，shadow 模式仅提示观察，不改变原 entry 判断。")


def _rank_scores(scores: pd.DataFrame) -> pd.DataFrame:
    out = scores.copy()
    out["ml_rank_global"] = out.groupby("trade_date")["ml_score"].rank(ascending=False, method="first").astype(int)
    sector_col = "sector_level2" if "sector_level2" in out.columns else "sector_level1"
    out["ml_rank_sector"] = out.groupby(["trade_date", sector_col])["ml_score"].rank(ascending=False, method="first").astype(int)
    for col in ["p_good_entry", "p_bad_entry", "expected_return_10d", "expected_drawdown_10d", "ml_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").round(6)
    return out.sort_values(["trade_date", "ml_rank_global", "code"]).reset_index(drop=True)


def _walk_forward_metric_row(scored: pd.DataFrame, top_n: int) -> dict[str, Any]:
    top = scored.nsmallest(top_n, "ml_rank_global") if "ml_rank_global" in scored.columns else scored.nlargest(top_n, "ml_score")
    label = scored["auto_label"].astype(str)
    good_y = (label == "good_entry").astype(int).to_numpy()
    bad_y = (label == "bad_entry").astype(int).to_numpy()
    return {
        "trade_date": str(pd.Timestamp(scored["trade_date"].iloc[0]).date()),
        "split": str(scored.get("split", pd.Series([""])).iloc[0]),
        "sample_count": int(len(scored)),
        f"top{top_n}_precision": float((top["auto_label"] == "good_entry").mean()) if not top.empty else np.nan,
        f"top{top_n}_bad_entry_rate": float((top["auto_label"] == "bad_entry").mean()) if not top.empty else np.nan,
        "bad_entry_rate": float((label == "bad_entry").mean()) if len(label) else np.nan,
        "good_auc": _roc_auc(good_y, scored["p_good_entry"].to_numpy(dtype=float)),
        "bad_auc": _roc_auc(bad_y, scored["p_bad_entry"].to_numpy(dtype=float)),
    }


def _walk_forward_table(metrics: pd.DataFrame) -> str:
    if metrics.empty:
        return "(no walk-forward scores)"
    summary = metrics.groupby("split", dropna=False).agg(
        scored_days=("trade_date", "count"),
        sample_count=("sample_count", "sum"),
        topN_precision=(metrics.columns[3], "mean"),
        topN_bad_entry_rate=(metrics.columns[4], "mean"),
        bad_entry_rate=("bad_entry_rate", "mean"),
    )
    return summary.reset_index().to_markdown(index=False)


def _topn_table(scores: pd.DataFrame, top_n: int) -> str:
    if scores.empty:
        return "(no scores)"
    top = scores.loc[scores["ml_rank_global"] <= top_n].copy()
    table = top.groupby("split", dropna=False).agg(
        rows=("auto_label", "count"),
        topN_precision=("auto_label", lambda s: float((s == "good_entry").mean())),
        bad_entry_rate=("auto_label", lambda s: float((s == "bad_entry").mean())),
        avg_ml_score=("ml_score", "mean"),
    )
    return table.reset_index().to_markdown(index=False)


def _baseline_comparison(df: pd.DataFrame, scores: pd.DataFrame, top_n: int) -> str:
    if scores.empty:
        return "(no scores)"
    model_top = scores.loc[scores["ml_rank_global"] <= top_n]
    model_precision = float((model_top["auto_label"] == "good_entry").mean()) if not model_top.empty else np.nan
    model_bad_rate = float((model_top["auto_label"] == "bad_entry").mean()) if not model_top.empty else np.nan
    scored_dates = set(pd.to_datetime(scores["trade_date"]).dt.normalize())
    baseline_source = df.loc[df["trade_date"].isin(scored_dates)].copy()
    rank_col = "momentum_score" if "momentum_score" in baseline_source.columns else "etf_rank"
    ascending = False if rank_col == "momentum_score" else True
    baseline_source["_baseline_rank"] = baseline_source.groupby("trade_date")[rank_col].rank(ascending=ascending, method="first")
    baseline_top = baseline_source.loc[baseline_source["_baseline_rank"] <= top_n]
    baseline_precision = float((baseline_top["auto_label"] == "good_entry").mean()) if not baseline_top.empty else np.nan
    baseline_bad_rate = float((baseline_top["auto_label"] == "bad_entry").mean()) if not baseline_top.empty else np.nan
    table = pd.DataFrame(
        [
            {
                "method": "ml_score_topN",
                "topN_precision": model_precision,
                "bad_entry_rate": model_bad_rate,
                "sample_count": int(len(model_top)),
            },
            {
                "method": f"{rank_col}_baseline_topN",
                "topN_precision": baseline_precision,
                "bad_entry_rate": baseline_bad_rate,
                "sample_count": int(len(baseline_top)),
            },
        ]
    )
    return table.to_markdown(index=False)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "selected"}


def _date_min(dates: list[pd.Timestamp]) -> str:
    return "" if not dates else str(pd.Timestamp(min(dates)).date())


def _date_max(dates: list[pd.Timestamp]) -> str:
    return "" if not dates else str(pd.Timestamp(max(dates)).date())
