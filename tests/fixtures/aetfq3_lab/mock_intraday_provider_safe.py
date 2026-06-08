from __future__ import annotations


class MockIntradayProviderSafe:
    def get_5m_bars(self, symbols, start_date, end_date):
        return []

    def get_history_bars(self, symbols, start_date, end_date, interval="5m"):
        return []
