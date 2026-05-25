from __future__ import annotations

import pandas as pd

from historical_ml.cli import main
from historical_ml.policy_comparison import compare_ml_policies


def test_policy_comparison_writes_report_and_csv(tmp_path):
    samples, scores = _samples_and_scores()

    result = compare_ml_policies(samples, scores, out_dir=tmp_path)

    assert (tmp_path / "reports" / "ml_policy_comparison_report.md").exists()
    assert (tmp_path / "reports" / "ml_policy_comparison.csv").exists()
    assert set(result.comparison["policy"]) == {"legacy_v21", "ml_shadow", "ml_candidate_expansion", "ml_active_sim"}
    assert "active_sim_permission" in result.report
    assert "PROBE Quantity Change" in result.report
    assert "Failure Modes" in result.report
    assert "ML top5 ranking precision" in result.report
    active = result.comparison.loc[result.comparison["policy"].eq("ml_active_sim")].iloc[0]
    legacy = result.comparison.loc[result.comparison["policy"].eq("legacy_v21")].iloc[0]
    assert active["daily_probe_count"] > legacy["daily_probe_count"]
    assert "ML_RECOVERED false positives" in result.report


def test_policy_actions_do_not_change_when_future_labels_are_perturbed(tmp_path):
    samples, scores = _samples_and_scores()
    base = compare_ml_policies(samples, scores, out_dir=tmp_path / "base").policy_rows

    mutated = samples.copy()
    mutated["future_return_10d"] = mutated["future_return_10d"] * -100
    mutated["future_max_drawdown_10d"] = mutated["future_max_drawdown_10d"] * 100
    mutated["auto_label"] = "bad_entry"
    changed = compare_ml_policies(mutated, scores, out_dir=tmp_path / "changed").policy_rows

    cols = ["policy", "trade_date", "code", "policy_action", "policy_covered", "policy_adjustment"]
    pd.testing.assert_frame_equal(
        base[cols].sort_values(cols[:3]).reset_index(drop=True),
        changed[cols].sort_values(cols[:3]).reset_index(drop=True),
    )


def test_policy_comparison_rejects_future_columns_in_score_table(tmp_path):
    samples, scores = _samples_and_scores()
    scores["future_return_10d"] = 0.99

    try:
        compare_ml_policies(samples, scores, out_dir=tmp_path)
    except ValueError as exc:
        assert "future label columns" in str(exc)
    else:
        raise AssertionError("expected future label leakage to be rejected")


def test_cli_compare_policies_runs(tmp_path):
    samples, scores = _samples_and_scores()
    sample_path = tmp_path / "ml_entry_labeled_samples.csv"
    score_path = tmp_path / "ml_entry_scores.csv"
    samples.to_csv(sample_path, index=False, encoding="utf-8-sig")
    scores.to_csv(score_path, index=False, encoding="utf-8-sig")

    rc = main(["compare-policies", "--samples", str(sample_path), "--scores", str(score_path), "--out", str(tmp_path / "artifacts")])

    assert rc == 0
    assert (tmp_path / "artifacts" / "reports" / "ml_policy_comparison_report.md").exists()
    assert (tmp_path / "artifacts" / "reports" / "ml_policy_comparison.csv").exists()


def _samples_and_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = pd.bdate_range("2025-01-02", periods=3)
    specs = [
        ("BAD_LEGACY", "PROBE", 98, "bad_entry", -0.05, False, False, "KEEP_ORIGINAL"),
        ("GOOD_RULE", "BUY", 86, "good_entry", 0.06, True, True, "KEEP_ORIGINAL"),
        ("GOOD_RECOVER", "OBSERVE", 60, "good_entry", 0.09, True, True, "UPGRADE_PROBE"),
        ("BAD_RECOVER", "OBSERVE", 58, "bad_entry", -0.04, False, False, "UPGRADE_PROBE"),
        ("GOOD_DOWN", "PROBE", 80, "good_entry", 0.07, True, False, "DOWNGRADE_WATCH"),
        ("NEUTRAL", "OBSERVE", 50, "neutral_entry", 0.01, False, False, "KEEP_ORIGINAL"),
    ]
    rows = []
    score_rows = []
    for d_idx, trade_date in enumerate(dates):
        for idx, (code, action, entry_score, label, ret, out_mkt, out_sector, suggestion) in enumerate(specs):
            was_actionable = action in {"BUY", "PROBE"}
            rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "name": code,
                    "sector_level1": "sector",
                    "sector_level2": "sector",
                    "is_valid_sample": True,
                    "pre_selected": was_actionable,
                    "was_candidate": was_actionable,
                    "was_selected": was_actionable,
                    "was_bought": was_actionable and action == "BUY",
                    "entry_score": entry_score - d_idx,
                    "momentum_score": entry_score - d_idx,
                    "entry_raw_action": action,
                    "final_action": action,
                    "label_status": "ok",
                    "auto_label": label,
                    "future_return_10d": ret + d_idx * 0.002,
                    "future_max_drawdown_10d": -0.02 if label == "good_entry" else -0.08,
                    "outperform_market_10d": out_mkt,
                    "outperform_sector_10d": out_sector,
                    "source": "historical_replay",
                }
            )
            score = {
                "GOOD_RECOVER": 0.95,
                "GOOD_RULE": 0.88,
                "GOOD_DOWN": 0.82,
                "NEUTRAL": 0.40,
                "BAD_RECOVER": 0.35,
                "BAD_LEGACY": 0.05,
            }[code]
            score_rows.append(
                {
                    "trade_date": trade_date,
                    "code": code,
                    "ml_score": score,
                    "ml_rank_global": idx + 1,
                    "p_good_entry": max(score, 0.01),
                    "p_bad_entry": 1 - score,
                    "ml_action_suggestion": suggestion,
                }
            )
    scores = pd.DataFrame(score_rows)
    scores["ml_rank_global"] = scores.groupby("trade_date")["ml_score"].rank(ascending=False, method="first")
    return pd.DataFrame(rows), scores
