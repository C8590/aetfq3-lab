"""Shared signal interface enums and CSV field contracts.

This module defines names only. It intentionally contains no strategy logic.
"""

from enum import Enum


class TextEnum(str, Enum):
    """String-valued enum with readable values in serialized outputs."""

    def __str__(self) -> str:
        return self.value


class MarketState(TextEnum):
    """Market state used by signal modules."""

    ATTACK = "进攻"
    BALANCED = "均衡"
    DEFENSE = "防守"


class BuyAction(TextEnum):
    """Entry-side action contract."""

    WATCH = "观察"
    WAIT_PULLBACK = "等待回踩"
    PROBE_BUY = "试探买入"
    STANDARD_BUY = "标准买入"
    ADD_BUY = "加强买入"
    FORBID_BUY = "禁止买入"


class SellAction(TextEnum):
    """Exit-side action contract."""

    HOLD = "持有"
    CAUTIOUS_HOLD = "谨慎持有"
    REDUCE_ONE_THIRD = "减仓三分之一"
    REDUCE_HALF = "减仓一半"
    CLEAR = "清仓"
    COOL_DOWN = "冷却"


class FailureAttribution(TextEnum):
    """Learning-side failure attribution contract."""

    BOUGHT_LATE_STAGE = "买在尾段"
    POOR_ENTRY = "买点太差"
    MARKET_TURNED_DEFENSIVE = "市场转防守"
    SAME_SECTOR_CONCENTRATION = "同板块集中"
    FREQUENT_ROTATION = "频繁换仓"
    SOLD_TOO_EARLY = "卖早"
    SOLD_TOO_LATE = "卖晚"
    DATA_OR_LIQUIDITY_ISSUE = "数据或流动性问题"


class FieldName(TextEnum):
    """Canonical CSV field names for the four signal modules."""

    TRADE_DATE = "trade_date"
    SYMBOL = "symbol"
    NAME = "name"
    SECTOR = "sector"
    MARKET_STATE = "market_state"
    SCORE = "score"
    RANK = "rank"
    SELECTED = "selected"
    CANDIDATE_POOL_FLAG = "candidate_pool_flag"
    CANDIDATE_SOURCE = "candidate_source"
    LEGACY_SELECTED = "legacy_selected"
    BROAD_RECALL_SELECTED = "broad_recall_selected"
    ML_RECOVERED = "ml_recovered"
    CANDIDATE_POOL_RANK = "candidate_pool_rank"
    REASON = "reason"
    BUY_ACTION = "buy_action"
    BUY_PRICE = "buy_price"
    POSITION_SIZE = "position_size"
    CONFIDENCE = "confidence"
    ENTRY_REASON = "entry_reason"
    RAW_ENTRY_ACTION = "raw_entry_action"
    RAW_ENTRY_TARGET_WEIGHT = "raw_entry_target_weight"
    RAW_ENTRY_CONFIDENCE = "raw_entry_confidence"
    RAW_ENTRY_REASON = "raw_entry_reason"
    RAW_ENTRY_BLOCK_REASON = "raw_entry_block_reason"
    FINAL_BUY_ACTION = "final_buy_action"
    FINAL_TARGET_WEIGHT = "final_target_weight"
    FINAL_BLOCK_REASON = "final_block_reason"
    CONTROL_OVERRIDE_REASON = "control_override_reason"
    EXIT_PRIORITY_BLOCKED = "exit_priority_blocked"
    RISK_GATE_BLOCKED = "risk_gate_blocked"
    ML_SCORE = "ml_score"
    P_GOOD_ENTRY = "p_good_entry"
    P_BAD_ENTRY = "p_bad_entry"
    ML_ENTRY_ADVICE = "ml_entry_advice"
    ML_CONFIDENCE = "ml_confidence"
    ML_REASON = "ml_reason"
    ML_ACTION_SUGGESTION = "ml_action_suggestion"
    ML_DECISION_MODE = "ml_decision_mode"
    ML_ADJUSTMENT = "ml_adjustment"
    ML_ADJUSTMENT_REASON_CN = "ml_adjustment_reason_cn"
    RULE_ACTION = "rule_action"
    ML_ADJUSTED_ACTION = "ml_adjusted_action"
    SELL_ACTION = "sell_action"
    SELL_PRICE = "sell_price"
    REDUCE_RATIO = "reduce_ratio"
    COOL_DOWN_DAYS = "cool_down_days"
    EXIT_REASON = "exit_reason"
    TRADE_ID = "trade_id"
    HOLDING_DAYS = "holding_days"
    RETURN_PCT = "return_pct"
    FAILURE_ATTRIBUTION = "failure_attribution"
    LESSON = "lesson"
    ADJUSTMENT = "adjustment"
    SOURCE_FILE = "source_file"
    GENERATED_AT = "generated_at"


PRE_SELECTION_RESULT_FIELDS = (
    FieldName.TRADE_DATE.value,
    FieldName.SYMBOL.value,
    FieldName.NAME.value,
    FieldName.SECTOR.value,
    FieldName.MARKET_STATE.value,
    FieldName.SCORE.value,
    FieldName.RANK.value,
    FieldName.SELECTED.value,
    FieldName.CANDIDATE_POOL_FLAG.value,
    FieldName.CANDIDATE_SOURCE.value,
    FieldName.LEGACY_SELECTED.value,
    FieldName.BROAD_RECALL_SELECTED.value,
    FieldName.ML_RECOVERED.value,
    FieldName.CANDIDATE_POOL_RANK.value,
    FieldName.REASON.value,
    FieldName.GENERATED_AT.value,
)

ENTRY_SIGNAL_FIELDS = (
    FieldName.TRADE_DATE.value,
    FieldName.SYMBOL.value,
    FieldName.NAME.value,
    FieldName.MARKET_STATE.value,
    FieldName.BUY_ACTION.value,
    FieldName.BUY_PRICE.value,
    FieldName.POSITION_SIZE.value,
    FieldName.CONFIDENCE.value,
    FieldName.ENTRY_REASON.value,
    FieldName.RAW_ENTRY_ACTION.value,
    FieldName.RAW_ENTRY_TARGET_WEIGHT.value,
    FieldName.RAW_ENTRY_CONFIDENCE.value,
    FieldName.RAW_ENTRY_REASON.value,
    FieldName.RAW_ENTRY_BLOCK_REASON.value,
    FieldName.FINAL_BUY_ACTION.value,
    FieldName.FINAL_TARGET_WEIGHT.value,
    FieldName.FINAL_BLOCK_REASON.value,
    FieldName.CONTROL_OVERRIDE_REASON.value,
    FieldName.EXIT_PRIORITY_BLOCKED.value,
    FieldName.RISK_GATE_BLOCKED.value,
    FieldName.ML_SCORE.value,
    FieldName.P_GOOD_ENTRY.value,
    FieldName.P_BAD_ENTRY.value,
    FieldName.ML_ENTRY_ADVICE.value,
    FieldName.ML_CONFIDENCE.value,
    FieldName.ML_REASON.value,
    FieldName.ML_ACTION_SUGGESTION.value,
    FieldName.ML_DECISION_MODE.value,
    FieldName.ML_ADJUSTMENT.value,
    FieldName.ML_ADJUSTMENT_REASON_CN.value,
    FieldName.RULE_ACTION.value,
    FieldName.ML_ADJUSTED_ACTION.value,
    FieldName.SOURCE_FILE.value,
    FieldName.GENERATED_AT.value,
)

EXIT_SIGNAL_FIELDS = (
    FieldName.TRADE_DATE.value,
    FieldName.SYMBOL.value,
    FieldName.NAME.value,
    FieldName.MARKET_STATE.value,
    FieldName.SELL_ACTION.value,
    FieldName.SELL_PRICE.value,
    FieldName.REDUCE_RATIO.value,
    FieldName.COOL_DOWN_DAYS.value,
    FieldName.EXIT_REASON.value,
    FieldName.SOURCE_FILE.value,
    FieldName.GENERATED_AT.value,
)

LEARNING_REPORT_FIELDS = (
    FieldName.TRADE_DATE.value,
    FieldName.TRADE_ID.value,
    FieldName.SYMBOL.value,
    FieldName.NAME.value,
    FieldName.HOLDING_DAYS.value,
    FieldName.RETURN_PCT.value,
    FieldName.FAILURE_ATTRIBUTION.value,
    FieldName.LESSON.value,
    FieldName.ADJUSTMENT.value,
    FieldName.SOURCE_FILE.value,
    FieldName.GENERATED_AT.value,
)
