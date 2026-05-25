"""Entry engine for right-side double-layer momentum buy signals."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

from contracts.signal_schema import BuyAction, ENTRY_SIGNAL_FIELDS, MarketState

OUTPUT_FILE = "entry_signal.csv"
INPUT_FILE = "pre_selection_result.csv"
ML_SUGGESTIONS_FILE = Path("artifacts") / "historical_ml_61" / "generated" / "entry_calibration_suggestions.csv"
ML_ENTRY_SCORES_FILE = Path("artifacts") / "historical_ml_61" / "generated" / "ml_entry_scores.csv"
PARAMETER_LEVEL_ML_KEY = "__PARAMETER_LEVEL__"
REQUIRED_OUTPUT_FIELDS = ENTRY_SIGNAL_FIELDS
DEFAULT_ML_ADVICE = "无ML建议"
DEFAULT_ML_REASON = "未找到历史校准建议，维持原 entry 判断。"
VALID_ML_DECISION_MODES = {"disabled", "shadow", "advisory", "active_sim"}
VALID_ML_ACTION_SUGGESTIONS = {
    "NO_ML",
    "KEEP_ORIGINAL",
    "UPGRADE_PROBE",
    "DOWNGRADE_WATCH",
    "WAIT_PULLBACK",
    "FORBID_CHASE",
}


@dataclass(frozen=True)
class EntryDecision:
    buy_action: str
    position_size: float
    confidence: float
    maturity: str
    quality: str
    reason: str
    warning: str


@dataclass(frozen=True)
class MLAdvice:
    ml_entry_advice: str = DEFAULT_ML_ADVICE
    ml_confidence: float = 0.0
    ml_reason: str = DEFAULT_ML_REASON
    ml_action_suggestion: str = "NO_ML"
    ml_score: float | str = ""
    p_good_entry: float | str = ""
    p_bad_entry: float | str = ""


class EntryEngine:
    """Produce entry_signal.csv from pre_selection_result.csv."""

    def __init__(
        self,
        first_buy_weight: float = 0.30,
        target_weight: float = 1.00,
        generated_at: str | None = None,
        ml_suggestions_path: str | Path | None = None,
        ml_scores_path: str | Path | None = None,
        ml_decision_mode: str = "shadow",
    ) -> None:
        self.first_buy_weight = _clip_ratio(first_buy_weight)
        self.target_weight = _clip_ratio(target_weight)
        self.generated_at = generated_at
        self.ml_suggestions_path = Path(ml_suggestions_path) if ml_suggestions_path is not None else None
        self.ml_scores_path = Path(ml_scores_path) if ml_scores_path is not None else None
        self.ml_decision_mode = _normalize_ml_decision_mode(ml_decision_mode)

    def run(
        self,
        pre_selection_rows: Sequence[Mapping[str, Any]] | None = None,
        output_dir: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """Return rows that match REQUIRED_OUTPUT_FIELDS and write entry_signal.csv."""
        out_dir = Path(output_dir) if output_dir is not None else Path("output")
        rows = list(pre_selection_rows) if pre_selection_rows is not None else self._read_pre_selection(out_dir)
        if any("candidate_pool_flag" in row for row in rows):
            candidate_rows = [row for row in rows if _truthy(row.get("candidate_pool_flag"))]
            rows = candidate_rows or rows
        trade_dates = {_text(row.get("trade_date"))[:10] for row in rows if _text(row.get("trade_date"))}
        ml_suggestions = {} if self.ml_decision_mode == "disabled" else self._read_ml_suggestions(out_dir, trade_dates)
        generated_at = self.generated_at or datetime.now().isoformat(timespec="seconds")

        results = [self._build_output_row(row, generated_at, ml_suggestions) for row in rows]
        if self.ml_decision_mode == "advisory":
            results.sort(key=lambda item: _number(item.get("ml_score"), default=-999999.0), reverse=True)
        out_dir.mkdir(parents=True, exist_ok=True)
        self._write_csv(out_dir / OUTPUT_FILE, results)
        return results

    def _read_pre_selection(self, output_dir: Path) -> list[dict[str, Any]]:
        input_path = output_dir / INPUT_FILE
        if not input_path.exists():
            raise FileNotFoundError(f"未找到上游预选结果文件: {input_path}")
        with input_path.open("r", encoding="utf-8-sig", newline="") as file:
            return [dict(row) for row in csv.DictReader(file)]

    def _read_ml_suggestions(self, output_dir: Path, trade_dates: set[str] | None = None) -> dict[str, MLAdvice]:
        input_path = self._resolve_ml_suggestions_path(output_dir)
        suggestions: dict[str, MLAdvice] = {}
        parameter_rows: list[dict[str, Any]] = []
        if input_path is not None:
            with input_path.open("r", encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    symbol = _symbol(_first_text(row, "etf_code", "code", "symbol"))
                    if symbol:
                        suggestions[symbol] = _ml_advice_from_row(row)
                    elif _is_parameter_level_ml_row(row):
                        parameter_rows.append(dict(row))
        if parameter_rows:
            suggestions[PARAMETER_LEVEL_ML_KEY] = _parameter_level_ml_advice(parameter_rows)
        for score_path in self._resolve_ml_score_paths(output_dir):
            with score_path.open("r", encoding="utf-8-sig", newline="") as file:
                for row in csv.DictReader(file):
                    if not _score_row_matches_trade_dates(row, trade_dates):
                        continue
                    symbol = _symbol(_first_text(row, "etf_code", "code", "symbol"))
                    if symbol:
                        suggestions[symbol] = _merge_ml_advice(suggestions.get(symbol, MLAdvice()), _ml_advice_from_row(row))
        return suggestions

    def _resolve_ml_suggestions_path(self, output_dir: Path) -> Path | None:
        candidates: list[Path] = []
        if self.ml_suggestions_path is not None:
            candidates.append(self.ml_suggestions_path)
        candidates.extend(
            [
                output_dir / ML_SUGGESTIONS_FILE,
                output_dir.parent / ML_SUGGESTIONS_FILE,
            ]
        )

        seen: set[Path] = set()
        for candidate in candidates:
            path = candidate if candidate.is_absolute() else Path.cwd() / candidate
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                return resolved
        return None

    def _resolve_ml_score_paths(self, output_dir: Path) -> list[Path]:
        candidates: list[Path] = []
        if self.ml_scores_path is not None:
            candidates.append(self.ml_scores_path)
        candidates.extend(
            [
                output_dir / ML_ENTRY_SCORES_FILE,
                output_dir.parent / ML_ENTRY_SCORES_FILE,
            ]
        )

        paths: list[Path] = []
        seen: set[Path] = set()
        for candidate in candidates:
            path = candidate if candidate.is_absolute() else Path.cwd() / candidate
            try:
                resolved = path.resolve()
            except OSError:
                resolved = path
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.exists():
                paths.append(resolved)
        return paths

    def _build_output_row(
        self,
        row: Mapping[str, Any],
        generated_at: str,
        ml_suggestions: Mapping[str, MLAdvice] | None = None,
    ) -> dict[str, Any]:
        decision = self._decide(row)
        ml_lookup = ml_suggestions or {}
        ml_advice = ml_lookup.get(_symbol(row.get("symbol")), ml_lookup.get(PARAMETER_LEVEL_ML_KEY, MLAdvice()))
        entry_reason = (
            f"趋势成熟度：{decision.maturity}；买点质量：{decision.quality}；"
            f"理由：{decision.reason}；警示：{decision.warning}；"
            f"首买权重：{self.first_buy_weight:.0%}；目标权重：{self.target_weight:.0%}"
        )
        raw_action = _raw_entry_action(decision.buy_action)
        ml_adjusted_action, ml_adjustment, ml_adjustment_reason, final_weight = self._apply_ml_decision(
            rule_action=raw_action,
            rule_weight=round(decision.position_size, 4),
            ml_advice=ml_advice,
        )
        final_action = ml_adjusted_action
        raw_block_reason = "" if raw_action in {"BUY", "PROBE"} else decision.reason
        final_block_reason = raw_block_reason if final_action not in {"BUY", "PROBE"} else ""
        if self.ml_decision_mode == "active_sim" and ml_adjustment.startswith("ML_DOWNGRADED"):
            final_block_reason = ml_adjustment_reason
        return {
            "trade_date": _text(row.get("trade_date")),
            "symbol": _symbol(row.get("symbol")),
            "name": _text(row.get("name")),
            "market_state": _normalize_market_state(row.get("market_state")),
            "buy_action": decision.buy_action,
            "buy_price": _format_price(_buy_price(row)),
            "position_size": round(decision.position_size, 4),
            "confidence": round(decision.confidence, 4),
            "entry_reason": entry_reason,
            "raw_entry_action": raw_action,
            "raw_entry_target_weight": round(decision.position_size, 4),
            "raw_entry_confidence": round(decision.confidence, 4),
            "raw_entry_reason": entry_reason,
            "raw_entry_block_reason": raw_block_reason,
            "final_buy_action": final_action,
            "final_target_weight": round(final_weight, 4),
            "final_block_reason": final_block_reason,
            "control_override_reason": "",
            "exit_priority_blocked": False,
            "risk_gate_blocked": False,
            "ml_score": ml_advice.ml_score,
            "p_good_entry": ml_advice.p_good_entry,
            "p_bad_entry": ml_advice.p_bad_entry,
            "ml_entry_advice": ml_advice.ml_entry_advice,
            "ml_confidence": round(ml_advice.ml_confidence, 4),
            "ml_reason": ml_advice.ml_reason,
            "ml_action_suggestion": ml_advice.ml_action_suggestion,
            "ml_decision_mode": self.ml_decision_mode,
            "ml_adjustment": ml_adjustment,
            "ml_adjustment_reason_cn": ml_adjustment_reason,
            "rule_action": raw_action,
            "ml_adjusted_action": ml_adjusted_action,
            "source_file": INPUT_FILE,
            "generated_at": generated_at,
        }

    def _apply_ml_decision(
        self,
        *,
        rule_action: str,
        rule_weight: float,
        ml_advice: MLAdvice,
    ) -> tuple[str, str, str, float]:
        action = str(ml_advice.ml_action_suggestion or "NO_ML").strip().upper()
        if self.ml_decision_mode == "disabled":
            return rule_action, "ML_DISABLED", "ML decision mode disabled; rule_action kept.", rule_weight
        if action in {"", "NO_ML", "KEEP_ORIGINAL"}:
            return rule_action, "ML_KEEP_RULE", "ML has no actionable override; rule_action kept.", rule_weight
        if self.ml_decision_mode == "shadow":
            return rule_action, "ML_SHADOW_ONLY", "shadow mode only displays ML; final_buy_action kept from rule_action.", rule_weight
        if self.ml_decision_mode == "advisory":
            return rule_action, "ML_ADVISORY_ONLY", "advisory mode may affect explanation or ordering only; final_buy_action kept from rule_action.", rule_weight

        if action == "UPGRADE_PROBE" and rule_action in {"OBSERVE", "REJECT", "AVOID"}:
            return "PROBE", "ML_RECOVERED", "active_sim: ML_RECOVERED PROBE from ml_entry_scores.", self.first_buy_weight
        if action in {"DOWNGRADE_WATCH", "WAIT_PULLBACK"} and rule_action in {"BUY", "PROBE"}:
            return "OBSERVE", "ML_DOWNGRADED", "active_sim: ML_DOWNGRADED OBSERVE because ml_entry_scores show weaker entry quality.", 0.0
        if action == "FORBID_CHASE" and rule_action in {"BUY", "PROBE"}:
            return "AVOID", "ML_DOWNGRADED", "active_sim: ML_DOWNGRADED AVOID because ml_entry_scores warn against chasing.", 0.0
        return rule_action, "ML_KEEP_RULE", "active_sim: ML did not change the rule_action.", rule_weight

    def _decide(self, row: Mapping[str, Any]) -> EntryDecision:
        selected = _truthy(row.get("selected"))
        market_state = _normalize_market_state(row.get("market_state"))
        score = _score(row.get("score"))
        maturity = self._trend_maturity(row)
        quality = self._buy_point_quality(row, maturity)

        if market_state == MarketState.DEFENSE.value and _is_equity_etf(row):
            return EntryDecision(
                buy_action=BuyAction.FORBID_BUY.value,
                position_size=0.0,
                confidence=0.15,
                maturity=maturity,
                quality="防守过滤",
                reason="市场状态为防守，权益 ETF 不触发主动买入。",
                warning="防守期优先控制回撤，禁止新开权益仓位。",
            )

        if not selected:
            return EntryDecision(
                buy_action=BuyAction.WATCH.value,
                position_size=0.0,
                confidence=0.20,
                maturity=maturity,
                quality=quality,
                reason="未进入预选候选池，仅保留观察。",
                warning="等待重新入选且买点质量改善后再评估。",
            )

        if quality == "禁止追高":
            return EntryDecision(
                buy_action=BuyAction.FORBID_BUY.value,
                position_size=0.0,
                confidence=0.20,
                maturity=maturity,
                quality=quality,
                reason="短线涨幅或均线乖离过大，右侧信号已偏离合理买点。",
                warning="禁止追高，等待充分回踩后重新确认。",
            )

        if maturity == "过热期":
            return EntryDecision(
                buy_action=BuyAction.WAIT_PULLBACK.value,
                position_size=0.0,
                confidence=0.35,
                maturity=maturity,
                quality=quality if quality != "普通确认" else "连续冲高",
                reason="趋势已经进入过热区，不能把新仓一次性打满。",
                warning="不允许新开重仓，只能等待回踩确认。",
            )

        if quality == "连续冲高":
            return EntryDecision(
                buy_action=BuyAction.WAIT_PULLBACK.value,
                position_size=0.0,
                confidence=0.40,
                maturity=maturity,
                quality=quality,
                reason="趋势方向仍强，但短线连续上冲后盈亏比下降。",
                warning="等待价格回到 20 日均线或前高附近再分批买入。",
            )

        if maturity == "启动期":
            probe_allowed = (
                (quality == "突破确认" and score >= 65)
                or (market_state == MarketState.ATTACK.value and selected and score >= 35)
            )
            action = BuyAction.PROBE_BUY.value if probe_allowed else BuyAction.WATCH.value
            size = self.first_buy_weight if action == BuyAction.PROBE_BUY.value else 0.0
            return EntryDecision(
                buy_action=action,
                position_size=size,
                confidence=0.50 if size else 0.32,
                maturity=maturity,
                quality=quality,
                reason="右侧趋势刚启动，进攻市场的入选强势候选允许先做小仓试探；其余情况继续观察强度。",
                warning="启动期容易假突破，首买后需等待二次确认。",
            )

        if maturity == "确认期":
            action = BuyAction.STANDARD_BUY.value if quality in {"突破确认", "回踩确认"} else BuyAction.PROBE_BUY.value
            size = self._standard_weight() if action == BuyAction.STANDARD_BUY.value else self.first_buy_weight
            return EntryDecision(
                buy_action=action,
                position_size=size,
                confidence=0.66 if action == BuyAction.STANDARD_BUY.value else 0.55,
                maturity=maturity,
                quality=quality,
                reason="中短期动量与趋势位置已完成确认，适合按计划分批介入。",
                warning="若买入后跌回关键均线，应停止加仓并等待退出模块处理。",
            )

        action = BuyAction.ADD_BUY.value if market_state == MarketState.ATTACK.value and quality == "回踩确认" else BuyAction.STANDARD_BUY.value
        size = self.target_weight if action == BuyAction.ADD_BUY.value else self._standard_weight()
        return EntryDecision(
            buy_action=action,
            position_size=size,
            confidence=0.82 if action == BuyAction.ADD_BUY.value else 0.74,
            maturity=maturity,
            quality=quality,
            reason="趋势处于主升期，双层动量保持共振，买点质量可执行。",
            warning="主升期仍需避免追涨，仓位上限不得超过目标权重。",
        )

    def _standard_weight(self) -> float:
        return _clip_ratio(max(self.first_buy_weight, self.target_weight * 0.60))

    def _trend_maturity(self, row: Mapping[str, Any]) -> str:
        score = _score(row.get("score"))
        m20 = _first_ratio(row, "momentum_20", "momentum20", "short_momentum")
        m60 = _first_ratio(row, "momentum_60", "momentum60", "mid_momentum")
        m120 = _first_ratio(row, "momentum_120", "momentum120", "long_momentum")
        distance20 = _first_ratio(row, "distance_ma20", "price_vs_ma20", "ma20_distance")
        distance60 = _first_ratio(row, "distance_ma60", "price_vs_ma60", "ma60_distance")
        days20 = _first_number(row, "days_above_ma20", "above_ma20_days")
        days60 = _first_number(row, "days_above_ma60", "above_ma60_days")
        pct_chg = _first_ratio(row, "pct_chg", "daily_return", "change_pct")
        consecutive_up = _first_number(row, "consecutive_up_days", "up_days")

        if (
            distance20 >= 0.08
            or distance60 >= 0.16
            or pct_chg >= 0.055
            or m20 >= 0.16
            or consecutive_up >= 5
        ):
            return "过热期"
        if score >= 85 and m20 > 0 and m60 > 0 and (m120 >= 0 or days60 >= 20):
            return "主升期"
        if score >= 70 and (m20 > 0 or m60 > 0 or days20 >= 5):
            return "确认期"
        return "启动期"

    def _buy_point_quality(self, row: Mapping[str, Any], maturity: str) -> str:
        pct_chg = _first_ratio(row, "pct_chg", "daily_return", "change_pct")
        distance20 = _first_ratio(row, "distance_ma20", "price_vs_ma20", "ma20_distance")
        distance60 = _first_ratio(row, "distance_ma60", "price_vs_ma60", "ma60_distance")
        consecutive_up = _first_number(row, "consecutive_up_days", "up_days")
        breakout = _truthy(_first_present(row, "breakout", "breakout_confirmed", "new_high"))
        pullback = _truthy(_first_present(row, "pullback", "pullback_confirmed"))
        from_high = _first_ratio(row, "pullback_pct", "from_high_pct", "drawdown_from_high")
        has_from_high = _has_any(row, "pullback_pct", "from_high_pct", "drawdown_from_high")
        has_distance20 = _has_any(row, "distance_ma20", "price_vs_ma20", "ma20_distance")
        score = _score(row.get("score"))

        if pct_chg >= 0.07 or distance20 >= 0.10 or distance60 >= 0.20:
            return "禁止追高"
        if maturity == "过热期" or consecutive_up >= 4 or pct_chg >= 0.045:
            return "连续冲高"
        if pullback or (has_from_high and 0.015 <= from_high <= 0.08) or (has_distance20 and -0.03 <= distance20 <= 0.025):
            return "回踩确认"
        if breakout or pct_chg >= 0.015 or score >= 78:
            return "突破确认"
        return "普通确认"

    def _write_csv(self, path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(REQUIRED_OUTPUT_FIELDS), extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _symbol(value: Any) -> str:
    text = _text(value)
    return text.zfill(6) if text.isdigit() else text


def _number(value: Any, default: float = 0.0) -> float:
    if value in (None, ""):
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if number != number:
        return default
    return number


def _ratio(value: Any, default: float = 0.0) -> float:
    number = _number(value, default=default)
    if abs(number) > 1:
        return number / 100.0
    return number


def _score(value: Any) -> float:
    number = _number(value)
    return number * 100 if 0 < number <= 1 else number


def _first_number(row: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in row and _text(row.get(key)) != "":
            return _number(row.get(key))
    return 0.0


def _first_ratio(row: Mapping[str, Any], *keys: str) -> float:
    for key in keys:
        if key in row and _text(row.get(key)) != "":
            return _ratio(row.get(key))
    return 0.0


def _first_present(row: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in row:
            return row.get(key)
    return None


def _first_text(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        if key in row and _text(row.get(key)) != "":
            return _text(row.get(key))
    return ""


def _score_row_matches_trade_dates(row: Mapping[str, Any], trade_dates: set[str] | None) -> bool:
    if not trade_dates:
        return True
    row_date = _first_text(row, "trade_date", "date")
    if not row_date:
        return True
    return row_date[:10] in trade_dates


def _has_any(row: Mapping[str, Any], *keys: str) -> bool:
    return any(key in row and _text(row.get(key)) != "" for key in keys)


def _ml_advice_from_row(row: Mapping[str, Any]) -> MLAdvice:
    raw_action = _first_text(row, "ml_action_suggestion", "action_suggestion")
    advice = _first_text(row, "ml_entry_advice", "entry_advice", "advice")
    action = _normalize_ml_action_suggestion(raw_action, advice)
    p_good = _first_present(row, "p_good_entry", "prob_good_entry")
    p_bad = _first_present(row, "p_bad_entry", "prob_bad_entry")
    ml_score = _first_present(row, "ml_score", "score")
    return MLAdvice(
        ml_entry_advice=advice or _advice_from_ml_action(action),
        ml_confidence=_clip_ratio(_first_present(row, "ml_confidence", "confidence")),
        ml_reason=_first_text(row, "ml_reason", "reason") or "历史校准建议存在但原因字段缺失，维持原 entry 判断。",
        ml_action_suggestion=action,
        ml_score="" if ml_score in (None, "") else round(_number(ml_score), 6),
        p_good_entry="" if p_good in (None, "") else round(_clip_ratio(p_good), 6),
        p_bad_entry="" if p_bad in (None, "") else round(_clip_ratio(p_bad), 6),
    )


def _merge_ml_advice(base: MLAdvice, override: MLAdvice) -> MLAdvice:
    return MLAdvice(
        ml_entry_advice=override.ml_entry_advice or base.ml_entry_advice,
        ml_confidence=override.ml_confidence if override.ml_confidence else base.ml_confidence,
        ml_reason=override.ml_reason or base.ml_reason,
        ml_action_suggestion=override.ml_action_suggestion if override.ml_action_suggestion != "NO_ML" else base.ml_action_suggestion,
        ml_score=override.ml_score if override.ml_score != "" else base.ml_score,
        p_good_entry=override.p_good_entry if override.p_good_entry != "" else base.p_good_entry,
        p_bad_entry=override.p_bad_entry if override.p_bad_entry != "" else base.p_bad_entry,
    )


def _is_parameter_level_ml_row(row: Mapping[str, Any]) -> bool:
    return any(_text(row.get(key)) for key in ("parameter_area", "current_pattern", "suggested_action", "suggestion_id"))


def _parameter_level_ml_advice(rows: Sequence[Mapping[str, Any]]) -> MLAdvice:
    count = len(rows)
    snippets: list[str] = []
    scores: list[float] = []
    for row in rows[:5]:
        area = _first_text(row, "parameter_area", "suggestion_id") or "parameter"
        action = _first_text(row, "suggested_action", "notes", "current_pattern") or "review"
        snippets.append(f"{area}: {action}")
    for row in rows:
        scores.append(_ml_confidence_score(_first_present(row, "confidence", "ml_confidence")))
    confidence = max(scores) if scores else 0.0
    suffix = "; ".join(snippets)
    if count > len(snippets):
        suffix = f"{suffix}; +{count - len(snippets)} more"
    return MLAdvice(
        ml_entry_advice=f"ML parameter-level suggestions loaded ({count}); observe only: {suffix}",
        ml_confidence=confidence,
        ml_reason="historical_ml suggestions are parameter-level and have no ETF code/symbol binding; entry displays them read-only and does not change buy_action, final_buy_action, or target_weight.",
        ml_action_suggestion="KEEP_ORIGINAL",
    )


def _ml_confidence_score(value: Any) -> float:
    text = _text(value).lower()
    if text in {"high", "strong"}:
        return 0.9
    if text in {"medium", "mid"}:
        return 0.6
    if text in {"low", "weak"}:
        return 0.3
    return _clip_ratio(value)


def _normalize_ml_action_suggestion(raw_action: str, advice: str = "") -> str:
    action = raw_action.strip().upper().replace("-", "_").replace(" ", "_")
    if action in VALID_ML_ACTION_SUGGESTIONS:
        return action

    text = f"{raw_action} {advice}"
    if "升级" in text or "试探" in text or "UPGRADE" in action:
        return "UPGRADE_PROBE"
    if "降级" in text or "观察" in text or "DOWNGRADE" in action:
        return "DOWNGRADE_WATCH"
    if "回踩" in text or "PULLBACK" in action:
        return "WAIT_PULLBACK"
    if "追高" in text or "禁止" in text or "CHASE" in action or "FORBID" in action:
        return "FORBID_CHASE"
    if "维持" in text or "保持" in text or "KEEP" in action:
        return "KEEP_ORIGINAL"
    return "KEEP_ORIGINAL"


def _normalize_ml_decision_mode(value: Any) -> str:
    mode = str(value or "shadow").strip().lower()
    return mode if mode in VALID_ML_DECISION_MODES else "shadow"


def _advice_from_ml_action(action: str) -> str:
    return {
        "KEEP_ORIGINAL": "建议维持原判断",
        "UPGRADE_PROBE": "建议升级小仓试探",
        "DOWNGRADE_WATCH": "建议降级观察",
        "WAIT_PULLBACK": "建议等待回踩",
        "FORBID_CHASE": "建议禁止追高",
    }.get(action, DEFAULT_ML_ADVICE)


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _text(value).lower()
    return text in {"1", "true", "yes", "y", "入选", "是", "selected", "通过"}


def _clip_ratio(value: Any) -> float:
    ratio = _ratio(value)
    return min(max(ratio, 0.0), 1.0)


def _normalize_market_state(value: Any) -> str:
    text = _text(value)
    if text in {MarketState.ATTACK.value, "attack", "进攻"}:
        return MarketState.ATTACK.value
    if text in {MarketState.BALANCED.value, "balanced", "balance", "均衡"}:
        return MarketState.BALANCED.value
    if text in {MarketState.DEFENSE.value, "defense", "defensive", "防守"}:
        return MarketState.DEFENSE.value
    return text or MarketState.BALANCED.value


def _is_equity_etf(row: Mapping[str, Any]) -> bool:
    text = " ".join(_text(row.get(key)) for key in ("symbol", "name", "sector", "reason")).lower()
    defensive_keywords = ("货币", "现金", "债", "国债", "短融", "银华日利", "511880", "511990")
    return not any(keyword.lower() in text for keyword in defensive_keywords)


def _buy_price(row: Mapping[str, Any]) -> float | None:
    for key in ("buy_price", "close", "latest_price", "price"):
        value = _number(row.get(key), default=-1.0)
        if value > 0:
            return value
    return None


def _format_price(value: float | None) -> str:
    return "" if value is None else f"{value:.3f}"


def _raw_entry_action(action: Any) -> str:
    text = _text(action).strip().lower()
    if not text:
        return "OBSERVE"
    if _text(action) == BuyAction.PROBE_BUY.value or "probe" in text or "试探" in _text(action):
        return "PROBE"
    if _text(action) in {BuyAction.STANDARD_BUY.value, BuyAction.ADD_BUY.value} or any(
        token in text for token in ("standard", "add", "buy")
    ):
        return "BUY"
    if _text(action) == BuyAction.FORBID_BUY.value or "forbid" in text or "禁止" in _text(action) or "reject" in text:
        return "REJECT"
    return "OBSERVE"
