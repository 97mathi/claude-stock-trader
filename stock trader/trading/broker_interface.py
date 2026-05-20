"""
Placeholder for real-money broker integration.

When you're ready to trade with real money:
  1. Pick a broker (Zerodha Kite, Upstox, Angel One SmartAPI).
  2. Implement the methods below to call the broker's REST/WebSocket API.
  3. In gui/app.py replace `self.portfolio = Portfolio(...)` with this class
     when DEFAULT_MODE == "real".

Both classes intentionally expose the SAME interface as Portfolio so the
rest of the app keeps working without changes.

IMPORTANT:
  - Test thoroughly on paper trading first.
  - Most Indian brokers REQUIRE you to handle the daily login (TOTP)
    yourself; the API token is short-lived.
  - Never check API keys into source control — load from environment.
"""

from __future__ import annotations


class RealBrokerNotConfigured(Exception):
    pass


class RealBrokerAdapter:
    """Stub — raises until you implement the integration."""

    def __init__(self):
        raise RealBrokerNotConfigured(
            "Real trading is not configured yet. "
            "Implement broker_interface.RealBrokerAdapter "
            "before switching DEFAULT_MODE to 'real'."
        )

    # Same method signatures as Portfolio so the GUI can swap them in:
    def get_cash(self) -> float: ...
    def get_positions(self) -> list: ...
    def get_position(self, symbol: str): ...
    def buy(self, symbol, quantity, price, predicted_target,
            expected_return_pct, horizon, reason="Model signal"): ...
    def sell(self, symbol, price, reason="Manual sell"): ...
    def get_trades(self, limit: int = 100) -> list: ...
    def total_value(self, live_prices) -> dict: ...
