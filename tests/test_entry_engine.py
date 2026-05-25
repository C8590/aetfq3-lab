from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from contracts.signal_schema import BuyAction, MarketState
from signal.entry.engine import EntryEngine, OUTPUT_FILE, REQUIRED_OUTPUT_FIELDS


def _row(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "trade_date": "2026-05-18",
        "symbol": "510300",
        "name": "沪深300ETF",
        "sector": "宽基权益",
        "market_state": MarketState.BALANCED.value,
        "score": 82,
        "rank": 1,
        "selected": True,
        "reason": "动量排名靠前",
        "generated_at": "2026-05-18T15:10:00",
        "close": 4.2,
    }
    data.update(overrides)
    return data


def _write_ml_file(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class EntryEngineTest(unittest.TestCase):
    def test_run_writes_contract_fields_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = EntryEngine(generated_at="2026-05-18T16:00:00").run(
                [
                    _row(
                        market_state=MarketState.BALANCED.value,
                        score=78,
                        momentum_20=0.04,
                        momentum_60=0.06,
                        breakout_confirmed=True,
                    )
                ],
                output_dir=tmp,
            )
            output_path = Path(tmp) / OUTPUT_FILE
            with output_path.open("r", encoding="utf-8-sig", newline="") as file:
                written = list(csv.DictReader(file))

        self.assertEqual(list(rows[0].keys()), list(REQUIRED_OUTPUT_FIELDS))
        self.assertEqual(list(written[0].keys()), list(REQUIRED_OUTPUT_FIELDS))
        self.assertEqual(rows[0]["buy_action"], BuyAction.STANDARD_BUY.value)
        self.assertEqual(rows[0]["buy_price"], "4.200")
        self.assertIn("理由：", rows[0]["entry_reason"])
        self.assertIn("警示：", rows[0]["entry_reason"])
        self.assertIn("ml_entry_advice", rows[0])
        self.assertIn("ml_confidence", rows[0])
        self.assertIn("ml_reason", rows[0])
        self.assertIn("ml_action_suggestion", rows[0])

    def test_defensive_market_forbids_active_equity_buy(self) -> None:
        rows = EntryEngine(generated_at="fixed").run(
            [
                _row(
                    market_state=MarketState.DEFENSE.value,
                    score=92,
                    momentum_20=0.08,
                    momentum_60=0.12,
                    pullback_confirmed=True,
                )
            ],
            output_dir=tempfile.mkdtemp(),
        )

        self.assertEqual(rows[0]["buy_action"], BuyAction.FORBID_BUY.value)
        self.assertEqual(rows[0]["position_size"], 0.0)
        self.assertIn("市场状态为防守", rows[0]["entry_reason"])
        self.assertIn("禁止新开权益仓位", rows[0]["entry_reason"])

    def test_main_uptrend_pullback_can_add_to_target_weight(self) -> None:
        row = EntryEngine(first_buy_weight=0.25, target_weight=0.80, generated_at="fixed").run(
            [
                _row(
                    market_state=MarketState.ATTACK.value,
                    score=92,
                    momentum_20=0.08,
                    momentum_60=0.13,
                    momentum_120=0.10,
                    days_above_ma60=30,
                    pullback_confirmed=True,
                    distance_ma20=0.01,
                )
            ],
            output_dir=tempfile.mkdtemp(),
        )[0]

        self.assertEqual(row["buy_action"], BuyAction.ADD_BUY.value)
        self.assertEqual(row["position_size"], 0.8)
        self.assertIn("趋势成熟度：主升期", row["entry_reason"])
        self.assertIn("买点质量：回踩确认", row["entry_reason"])
        self.assertIn("目标权重：80%", row["entry_reason"])

    def test_overheated_trend_waits_for_pullback_without_heavy_new_position(self) -> None:
        row = EntryEngine(first_buy_weight=0.30, target_weight=0.90, generated_at="fixed").run(
            [
                _row(
                    market_state=MarketState.ATTACK.value,
                    score=94,
                    momentum_20=0.12,
                    momentum_60=0.18,
                    momentum_120=0.11,
                    distance_ma20=0.085,
                    pct_chg=0.03,
                    consecutive_up_days=3,
                )
            ],
            output_dir=tempfile.mkdtemp(),
        )[0]

        self.assertEqual(row["buy_action"], BuyAction.WAIT_PULLBACK.value)
        self.assertEqual(row["position_size"], 0.0)
        self.assertIn("趋势成熟度：过热期", row["entry_reason"])
        self.assertIn("不允许新开重仓", row["entry_reason"])

    def test_attack_r0_strong_candidate_allows_probe_instead_of_plain_watch(self) -> None:
        row = EntryEngine(first_buy_weight=0.30, target_weight=0.90, generated_at="fixed").run(
            [
                _row(
                    market_state=MarketState.ATTACK.value,
                    score=42,
                    rank=1,
                    selected=True,
                    momentum_20=0.01,
                    momentum_60=0.01,
                    distance_ma20=0.01,
                    pct_chg=0.005,
                    consecutive_up_days=1,
                )
            ],
            output_dir=tempfile.mkdtemp(),
        )[0]

        self.assertEqual(row["raw_entry_action"], "PROBE")
        self.assertEqual(row["final_buy_action"], "PROBE")
        self.assertGreater(row["raw_entry_target_weight"], 0)
        self.assertNotEqual(row["buy_action"], BuyAction.WATCH.value)

    def test_attack_candidate_overheat_does_not_emit_raw_probe(self) -> None:
        row = EntryEngine(generated_at="fixed").run(
            [
                _row(
                    market_state=MarketState.ATTACK.value,
                    score=92,
                    rank=1,
                    selected=True,
                    momentum_20=0.18,
                    distance_ma20=0.12,
                    pct_chg=0.075,
                )
            ],
            output_dir=tempfile.mkdtemp(),
        )[0]

        self.assertIn(row["raw_entry_action"], {"OBSERVE", "REJECT"})
        self.assertEqual(row["raw_entry_target_weight"], 0.0)

    def test_forbid_chasing_after_excessive_surge(self) -> None:
        row = EntryEngine(generated_at="fixed").run(
            [
                _row(
                    market_state=MarketState.ATTACK.value,
                    score=96,
                    momentum_20=0.18,
                    momentum_60=0.16,
                    distance_ma20=0.12,
                    pct_chg=0.075,
                )
            ],
            output_dir=tempfile.mkdtemp(),
        )[0]

        self.assertEqual(row["buy_action"], BuyAction.FORBID_BUY.value)
        self.assertEqual(row["position_size"], 0.0)
        self.assertIn("买点质量：禁止追高", row["entry_reason"])

    def test_reads_pre_selection_csv_when_rows_are_not_provided(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            input_path = Path(tmp) / "pre_selection_result.csv"
            with input_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=list(_row().keys()))
                writer.writeheader()
                writer.writerow(_row(selected=False, score=55))

            rows = EntryEngine(generated_at="fixed").run(output_dir=tmp)

        self.assertEqual(rows[0]["buy_action"], BuyAction.WATCH.value)
        self.assertIn("未进入预选候选池", rows[0]["entry_reason"])

    def test_recall_pool_rows_enter_entry_but_do_not_expand_buy_count(self) -> None:
        pre_rows = []
        for i in range(8):
            pre_rows.append(
                _row(
                    symbol=f"{510000 + i:06d}",
                    name=f"ETF{i}",
                    market_state=MarketState.ATTACK.value,
                    score=80 - i,
                    rank=i + 1,
                    selected=i < 5,
                    candidate_pool_flag=True,
                    candidate_source="LEGACY_TOP5" if i < 5 else "BROAD_RECALL",
                    legacy_selected=i < 5,
                    broad_recall_selected=True,
                    ml_recovered=False,
                    candidate_pool_rank=i + 1,
                    momentum_20=0.02,
                    momentum_60=0.03,
                    distance_ma20=0.01,
                )
            )

        rows = EntryEngine(generated_at="fixed").run(pre_rows, output_dir=tempfile.mkdtemp())

        self.assertEqual(len(rows), 8)
        actionable = [row for row in rows if row["raw_entry_action"] in {"BUY", "PROBE"} and row["raw_entry_target_weight"] > 0]
        self.assertLessEqual(len(actionable), 5)
        self.assertTrue(all(row["buy_action"] == BuyAction.WATCH.value for row in rows[5:]))

    def test_missing_ml_file_uses_default_fields_without_changing_entry_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            row = EntryEngine(generated_at="fixed").run(
                [
                    _row(
                        score=78,
                        momentum_20=0.04,
                        momentum_60=0.06,
                        breakout_confirmed=True,
                    )
                ],
                output_dir=tmp,
            )[0]

        self.assertEqual(row["buy_action"], BuyAction.STANDARD_BUY.value)
        self.assertEqual(row["ml_entry_advice"], "无ML建议")
        self.assertEqual(row["ml_confidence"], 0.0)
        self.assertEqual(row["ml_reason"], "未找到历史校准建议，维持原 entry 判断。")
        self.assertEqual(row["ml_action_suggestion"], "NO_ML")

    def test_matching_ml_file_populates_advice_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ml_path = Path(tmp) / "artifacts" / "historical_ml_61" / "generated" / "entry_calibration_suggestions.csv"
            _write_ml_file(
                ml_path,
                [
                    {
                        "etf_code": "510300",
                        "ml_entry_advice": "建议等待回踩",
                        "ml_confidence": 0.73,
                        "ml_reason": "历史样本显示当前乖离偏高，回踩后胜率更好。",
                        "ml_action_suggestion": "WAIT_PULLBACK",
                    }
                ],
            )

            row = EntryEngine(generated_at="fixed").run([_row()], output_dir=tmp)[0]

        self.assertEqual(row["ml_entry_advice"], "建议等待回踩")
        self.assertEqual(row["ml_confidence"], 0.73)
        self.assertEqual(row["ml_reason"], "历史样本显示当前乖离偏高，回踩后胜率更好。")
        self.assertEqual(row["ml_action_suggestion"], "WAIT_PULLBACK")

    def test_ml_upgrade_probe_does_not_rewrite_original_entry_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ml_path = Path(tmp) / "artifacts" / "historical_ml_61" / "generated" / "entry_calibration_suggestions.csv"
            _write_ml_file(
                ml_path,
                [
                    {
                        "code": "510300",
                        "advice": "建议升级小仓试探",
                        "confidence": 0.66,
                        "reason": "类似启动形态试探仓收益风险比较好。",
                        "action_suggestion": "UPGRADE_PROBE",
                    }
                ],
            )

            row = EntryEngine(generated_at="fixed").run([_row(selected=False, score=55)], output_dir=tmp)[0]

        self.assertEqual(row["buy_action"], BuyAction.WATCH.value)
        self.assertEqual(row["ml_action_suggestion"], "UPGRADE_PROBE")
        self.assertEqual(row["ml_entry_advice"], "建议升级小仓试探")

    def test_shadow_ml_entry_scores_do_not_change_final_buy_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ml_path = Path(tmp) / "artifacts" / "historical_ml_61" / "generated" / "ml_entry_scores.csv"
            _write_ml_file(
                ml_path,
                [
                    {
                        "code": "510300",
                        "p_good_entry": 0.82,
                        "p_bad_entry": 0.08,
                        "ml_score": 74.5,
                        "ml_action_suggestion": "UPGRADE_PROBE",
                    }
                ],
            )

            row = EntryEngine(generated_at="fixed", ml_decision_mode="shadow").run([_row(selected=False, score=55)], output_dir=tmp)[0]

        self.assertEqual(row["rule_action"], "OBSERVE")
        self.assertEqual(row["final_buy_action"], "OBSERVE")
        self.assertEqual(row["ml_adjusted_action"], "OBSERVE")
        self.assertEqual(row["ml_decision_mode"], "shadow")
        self.assertEqual(row["ml_adjustment"], "ML_SHADOW_ONLY")
        self.assertEqual(row["ml_score"], 74.5)
        self.assertEqual(row["p_good_entry"], 0.82)
        self.assertEqual(row["p_bad_entry"], 0.08)

    def test_ml_entry_scores_match_current_trade_date_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ml_path = Path(tmp) / "artifacts" / "historical_ml_61" / "generated" / "ml_entry_scores.csv"
            _write_ml_file(
                ml_path,
                [
                    {
                        "trade_date": "2026-05-17",
                        "code": "510300",
                        "p_good_entry": 0.82,
                        "p_bad_entry": 0.08,
                        "ml_score": 74.5,
                        "ml_action_suggestion": "UPGRADE_PROBE",
                    }
                ],
            )

            row = EntryEngine(generated_at="fixed", ml_decision_mode="shadow").run([_row()], output_dir=tmp)[0]

        self.assertEqual(row["ml_score"], "")
        self.assertEqual(row["ml_action_suggestion"], "NO_ML")

    def test_active_sim_ml_scores_can_recover_observe_to_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ml_path = Path(tmp) / "artifacts" / "historical_ml_61" / "generated" / "ml_entry_scores.csv"
            _write_ml_file(
                ml_path,
                [
                    {
                        "code": "510300",
                        "p_good_entry": 0.76,
                        "p_bad_entry": 0.05,
                        "ml_score": 62.0,
                        "ml_action_suggestion": "UPGRADE_PROBE",
                    }
                ],
            )

            row = EntryEngine(first_buy_weight=0.25, generated_at="fixed", ml_decision_mode="active_sim").run(
                [_row(selected=False, score=55)],
                output_dir=tmp,
            )[0]

        self.assertEqual(row["rule_action"], "OBSERVE")
        self.assertEqual(row["ml_adjustment"], "ML_RECOVERED")
        self.assertEqual(row["ml_adjusted_action"], "PROBE")
        self.assertEqual(row["final_buy_action"], "PROBE")
        self.assertEqual(row["final_target_weight"], 0.25)
        self.assertIn("ML_RECOVERED PROBE", row["ml_adjustment_reason_cn"])

    def test_active_sim_ml_scores_can_downgrade_buy_to_observe(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ml_path = Path(tmp) / "artifacts" / "historical_ml_61" / "generated" / "ml_entry_scores.csv"
            _write_ml_file(
                ml_path,
                [
                    {
                        "code": "510300",
                        "p_good_entry": 0.09,
                        "p_bad_entry": 0.72,
                        "ml_score": -41.0,
                        "ml_action_suggestion": "DOWNGRADE_WATCH",
                    }
                ],
            )

            row = EntryEngine(generated_at="fixed", ml_decision_mode="active_sim").run(
                [_row(score=78, momentum_20=0.04, momentum_60=0.06, breakout_confirmed=True)],
                output_dir=tmp,
            )[0]

        self.assertEqual(row["rule_action"], "BUY")
        self.assertEqual(row["ml_adjustment"], "ML_DOWNGRADED")
        self.assertEqual(row["ml_adjusted_action"], "OBSERVE")
        self.assertEqual(row["final_buy_action"], "OBSERVE")
        self.assertEqual(row["final_target_weight"], 0.0)
        self.assertIn("ML_DOWNGRADED OBSERVE", row["ml_adjustment_reason_cn"])

    def test_parameter_level_ml_suggestions_are_read_only_observation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ml_path = Path(tmp) / "artifacts" / "historical_ml_61" / "generated" / "entry_calibration_suggestions.csv"
            _write_ml_file(
                ml_path,
                [
                    {
                        "suggestion_id": "CAL-001",
                        "parameter_area": "trend_maturity",
                        "current_pattern": "overheat stage has elevated bad_rate",
                        "suggested_action": "require pullback confirmation",
                        "confidence": "high",
                    }
                ],
            )

            row = EntryEngine(generated_at="fixed").run(
                [_row(score=78, momentum_20=0.04, momentum_60=0.06, breakout_confirmed=True)],
                output_dir=tmp,
            )[0]

        self.assertEqual(row["buy_action"], BuyAction.STANDARD_BUY.value)
        self.assertEqual(row["final_buy_action"], "BUY")
        self.assertGreater(row["final_target_weight"], 0)
        self.assertEqual(row["ml_action_suggestion"], "KEEP_ORIGINAL")
        self.assertIn("parameter-level suggestions loaded", row["ml_entry_advice"])
        self.assertIn("does not change buy_action", row["ml_reason"])

    def test_ml_downgrade_watch_does_not_rewrite_original_entry_action(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            ml_path = Path(tmp) / "artifacts" / "historical_ml_61" / "generated" / "entry_calibration_suggestions.csv"
            _write_ml_file(
                ml_path,
                [
                    {
                        "symbol": "510300",
                        "entry_advice": "建议降级观察",
                        "ml_confidence": 0.81,
                        "ml_reason": "历史校准认为当前突破后失败率偏高。",
                        "ml_action_suggestion": "DOWNGRADE_WATCH",
                    }
                ],
            )

            row = EntryEngine(generated_at="fixed").run(
                [_row(score=78, momentum_20=0.04, momentum_60=0.06, breakout_confirmed=True)],
                output_dir=tmp,
            )[0]

        self.assertEqual(row["buy_action"], BuyAction.STANDARD_BUY.value)
        self.assertEqual(row["ml_action_suggestion"], "DOWNGRADE_WATCH")
        self.assertEqual(row["ml_entry_advice"], "建议降级观察")

    def test_output_fields_match_interface_contract_with_ml_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            rows = EntryEngine(generated_at="fixed").run([_row()], output_dir=tmp)
            output_path = Path(tmp) / OUTPUT_FILE
            with output_path.open("r", encoding="utf-8-sig", newline="") as file:
                written = list(csv.DictReader(file))

        self.assertEqual(list(rows[0].keys()), list(REQUIRED_OUTPUT_FIELDS))
        self.assertEqual(list(written[0].keys()), list(REQUIRED_OUTPUT_FIELDS))
        self.assertEqual(
            list(REQUIRED_OUTPUT_FIELDS)[-11:-2],
            [
                "ml_entry_advice",
                "ml_confidence",
                "ml_reason",
                "ml_action_suggestion",
                "ml_decision_mode",
                "ml_adjustment",
                "ml_adjustment_reason_cn",
                "rule_action",
                "ml_adjusted_action",
            ],
        )


if __name__ == "__main__":
    unittest.main()
