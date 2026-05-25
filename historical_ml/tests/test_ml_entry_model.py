import pandas as pd

from historical_ml.cli import main
from historical_ml.ml_entry_model import (
    FEATURE_VERSION,
    MODEL_VERSION,
    build_entry_feature_matrix,
    prepare_entry_quality_samples,
    score_entry_quality_universe,
    train_entry_quality_model,
)


def test_entry_quality_model_writes_code_level_scores_and_report(tmp_path):
    samples = _labeled_samples()

    result = train_entry_quality_model(samples, tmp_path, top_n=2, min_train_dates=4)
    scores = result.scores

    assert not scores.empty
    assert (tmp_path / "generated" / "ml_entry_scores.csv").exists()
    assert (tmp_path / "generated" / "model_report.md").exists()
    assert set(scores["model_version"]) == {MODEL_VERSION}
    assert set(scores["feature_version"]) == {FEATURE_VERSION}
    assert scores[["trade_date", "code"]].drop_duplicates().shape[0] == len(scores)
    assert scores["p_good_entry"].between(0, 1).all()
    assert scores["p_bad_entry"].between(0, 1).all()
    assert (scores["ml_action_suggestion"] == "NO_ML").mean() < 0.10
    assert {"validation", "test"}.issubset(set(result.walk_forward_metrics["split"]))
    assert "TopN Precision And Bad Entry Rate" in result.report
    assert "Baseline Comparison" in result.report
    assert "shadow-mode" in result.report


def test_entry_quality_model_uses_chronological_train_validation_test_split(tmp_path):
    result = train_entry_quality_model(_labeled_samples(), tmp_path, top_n=2, min_train_dates=4)
    split = result.split

    assert split.train_dates
    assert split.validation_dates
    assert split.test_dates
    assert max(split.train_dates) < min(split.validation_dates)
    assert max(split.validation_dates) < min(split.test_dates)
    assert "walk-forward" in split.note


def test_entry_quality_model_features_exclude_future_labels_and_identity():
    df = prepare_entry_quality_samples(_labeled_samples())
    features = build_entry_feature_matrix(df)

    names = list(features.columns)
    assert "code" not in names
    assert "name" not in names
    assert not any(name.startswith("future_return_") for name in names)
    assert not any(name.startswith("future_max_") for name in names)
    assert not any(name.startswith("outperform_") for name in names)
    assert "auto_label" not in names
    assert "label_reason_cn" not in names


def test_walk_forward_score_for_date_does_not_change_when_future_rows_are_perturbed(tmp_path):
    samples = _labeled_samples()
    base = train_entry_quality_model(samples, tmp_path / "base", top_n=2, min_train_dates=4).scores
    first_scored_date = pd.to_datetime(base["trade_date"]).min()

    mutated = samples.copy()
    future_mask = pd.to_datetime(mutated["trade_date"]) > first_scored_date
    mutated.loc[future_mask, "momentum_score"] *= -100
    mutated.loc[future_mask, "momentum_20"] *= -100
    mutated.loc[future_mask, "auto_label"] = "bad_entry"
    mutated.loc[future_mask, "future_return_10d"] = -0.50
    mutated_scores = train_entry_quality_model(mutated, tmp_path / "mutated", top_n=2, min_train_dates=4).scores

    cols = ["trade_date", "code", "p_good_entry", "p_bad_entry", "ml_score", "ml_action_suggestion"]
    left = base.loc[pd.to_datetime(base["trade_date"]) == first_scored_date, cols].sort_values("code").reset_index(drop=True)
    right = mutated_scores.loc[pd.to_datetime(mutated_scores["trade_date"]) == first_scored_date, cols].sort_values("code").reset_index(drop=True)
    pd.testing.assert_frame_equal(left, right)


def test_cli_train_entry_model_runs(tmp_path):
    samples = tmp_path / "ml_entry_labeled_samples.csv"
    _labeled_samples().to_csv(samples, index=False)

    rc = main(["train-entry-model", "--samples", str(samples), "--out", str(tmp_path), "--top-n", "2", "--min-train-dates", "4"])

    assert rc == 0
    assert (tmp_path / "generated" / "ml_entry_scores.csv").exists()
    assert (tmp_path / "generated" / "model_report.md").exists()


def test_current_universe_scoring_covers_feature_ready_untrained_codes(tmp_path):
    labeled = _labeled_samples()
    scoring = labeled.loc[pd.to_datetime(labeled["trade_date"]) == pd.to_datetime(labeled["trade_date"]).max()].copy()
    scoring["trade_date"] = pd.Timestamp("2024-02-06")
    scoring["auto_label"] = ""
    scoring["label_status"] = ""
    scoring.loc[scoring.index[:2], "code"] = ["NEW1", "NEW2"]

    scores = score_entry_quality_universe(
        labeled,
        scoring,
        out_dir=tmp_path,
        score_date="2024-02-06",
        min_train_dates=4,
    )

    assert {"NEW1", "NEW2"}.issubset(set(scores["code"]))
    assert len(scores) == len(scoring)
    assert set(scores["model_version"]) == {MODEL_VERSION}


def _labeled_samples() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=24)
    specs = [
        ("GOOD1", "科技", 0.90, "good_entry"),
        ("GOOD2", "消费", 0.75, "good_entry"),
        ("BAD1", "金融", -0.85, "bad_entry"),
        ("BAD2", "周期", -0.65, "bad_entry"),
        ("MID1", "科技", 0.10, "neutral_entry"),
        ("MID2", "消费", -0.05, "neutral_entry"),
    ]
    rows = []
    for d_idx, trade_date in enumerate(dates):
        for c_idx, (code, sector, base_momentum, label) in enumerate(specs):
            seasonal = ((d_idx % 5) - 2) * 0.01
            momentum = base_momentum + seasonal
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "name": f"{code} ETF",
                    "sector_level1": sector,
                    "sector_level2": sector,
                    "is_valid_sample": True,
                    "exclude_reason": "",
                    "momentum_20": momentum,
                    "momentum_60": momentum * 0.8,
                    "momentum_120": momentum * 0.5,
                    "momentum_score": momentum,
                    "acceleration_score": momentum * 0.2,
                    "volatility_20": 0.02 + c_idx * 0.002,
                    "drawdown_20": -0.01 - max(-momentum, 0) * 0.02,
                    "drawdown_60": -0.02 - max(-momentum, 0) * 0.03,
                    "market_state": "offense" if d_idx % 3 else "neutral",
                    "sector_state": "strong" if momentum > 0.5 else ("weak" if momentum < -0.5 else "neutral"),
                    "sector_rank": c_idx % 3 + 1,
                    "etf_rank": c_idx % 2 + 1,
                    "pre_selected": label == "good_entry",
                    "entry_raw_action": "BUY" if label == "good_entry" else "OBSERVE",
                    "final_action": "BUY" if label == "good_entry" else "OBSERVE",
                    "source": "historical_replay",
                    "feature_at_t": trade_date,
                    "label_after_t": trade_date + pd.offsets.BDay(1),
                    "future_return_1d": 0.01 if label == "good_entry" else (-0.01 if label == "bad_entry" else 0.0),
                    "future_return_3d": 0.03 if label == "good_entry" else (-0.03 if label == "bad_entry" else 0.005),
                    "future_return_5d": 0.05 if label == "good_entry" else (-0.05 if label == "bad_entry" else 0.006),
                    "future_return_10d": 0.08 if label == "good_entry" else (-0.08 if label == "bad_entry" else 0.01),
                    "future_return_20d": 0.12 if label == "good_entry" else (-0.10 if label == "bad_entry" else 0.015),
                    "future_max_gain_10d": 0.10 if label == "good_entry" else 0.01,
                    "future_max_drawdown_10d": -0.02 if label == "good_entry" else (-0.08 if label == "bad_entry" else -0.025),
                    "outperform_market_10d": label == "good_entry",
                    "outperform_sector_10d": label == "good_entry",
                    "hit_stop_loss_10d": label == "bad_entry",
                    "auto_label": label,
                    "label_reason_cn": "测试标签原因",
                    "label_status": "ok",
                }
            )
    return pd.DataFrame(rows)
