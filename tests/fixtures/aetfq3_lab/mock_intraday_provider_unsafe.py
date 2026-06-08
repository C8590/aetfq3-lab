from __future__ import annotations


class MockIntradayProviderUnsafe:
    def get_account(self):
        return {}

    def submit_order(self, payload):
        return payload

    def cancel_order(self, identifier):
        return identifier

    def get_positions(self):
        return []

    def order_intent(self):
        return {}
