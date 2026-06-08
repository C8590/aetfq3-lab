from __future__ import annotations

from datetime import date
from typing import Sequence

import pandas as pd


REAL_PROVIDER_TEMPLATE_ERROR = "real provider requires separate human authorization and safety review"
REQUIRED_COLUMNS = [
    "trade_date",
    "datetime",
    "etf_code",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
]


class Intraday5mProviderTemplate:
    def get_5m_bars(self, symbols: Sequence[str], start_date: date, end_date: date) -> pd.DataFrame:
        raise NotImplementedError(REAL_PROVIDER_TEMPLATE_ERROR)

    @staticmethod
    def validate_output_schema(df: pd.DataFrame) -> None:
        missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
        if missing:
            raise ValueError(f"missing required 5m bar columns: {missing}")

        if df.empty:
            raise ValueError("5m bar frame is empty")

        pd.to_datetime(df["trade_date"], errors="raise")
        pd.to_datetime(df["datetime"], errors="raise")
        for column in ["open", "high", "low", "close", "volume", "amount"]:
            pd.to_numeric(df[column], errors="raise")

    @staticmethod
    def provider_capabilities() -> dict[str, bool]:
        return {
            "market_data_only": True,
            "supports_account": False,
            "supports_position": False,
            "supports_order": False,
            "supports_trade": False,
            "supports_submit_order": False,
            "supports_cancel_order": False,
            "supports_order_intent": False,
            "requires_secret": False,
            "requires_live_session": False,
        }
