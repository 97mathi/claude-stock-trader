"""
Sector momentum signal.

Idea: a stock tends to move with its sector. If the stock's sector index has
positive momentum over the last ~20 trading days, that's a tailwind (+); if
negative, a headwind (-).

Score = recent sector return scaled into [-1, +1].
Uses live-ish daily data from Yahoo Finance. Neutral if the index is missing.
"""

from __future__ import annotations
from data.fetcher import get_index_history
from data.sectors import sector_for
from signals.base import Signal, neutral

# A 20-day move of +/-8% maps to +/-1.0 score (tunable).
_SCALE = 0.08


def sector_signal(symbol: str) -> Signal:
    index_sym = sector_for(symbol)
    hist = get_index_history(index_sym, period="3mo")
    if hist is None or hist.empty or "Close" not in hist.columns or len(hist) < 21:
        return neutral("sector", f"index {index_sym} unavailable")

    closes = hist["Close"]
    recent = float(closes.iloc[-1])
    past = float(closes.iloc[-21])
    if past <= 0:
        return neutral("sector", "bad index data")

    ret_20d = (recent - past) / past
    score = max(-1.0, min(1.0, ret_20d / _SCALE))
    return Signal(
        name="sector",
        score=score,
        note=f"{index_sym} 20d {ret_20d*100:+.1f}%",
        available=True,
    )
