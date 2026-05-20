"""
Fundamentals signal (P/E, margins, growth, leverage).

This is a SANITY / quality signal, not a precise valuation model. It rewards
stocks that look reasonably valued and healthy, and penalizes very expensive
or weak ones.

Components (each contributes to the score):
  - Trailing P/E : very high P/E (>40) is a mild negative; reasonable (10-25)
                   is a mild positive; negative earnings is a negative.
  - Profit margin: higher is better.
  - Earnings growth: positive growth is good.
  - Debt/equity  : very high leverage is a negative.

All Yahoo fundamentals can be missing for NSE names - missing parts are
skipped, and if nothing is available the signal is neutral.
"""

from __future__ import annotations
from data.fetcher import get_fundamentals
from signals.base import Signal, neutral


def _pe_score(pe) -> float | None:
    if pe is None:
        return None
    if pe < 0:
        return -0.6                 # losing money
    if pe < 10:
        return 0.5                  # cheap
    if pe < 25:
        return 0.3                  # reasonable
    if pe < 40:
        return -0.1                 # getting pricey
    return -0.5                     # very expensive


def _margin_score(m) -> float | None:
    if m is None:
        return None
    if m <= 0:
        return -0.5
    if m < 0.05:
        return -0.1
    if m < 0.15:
        return 0.2
    return 0.5


def _growth_score(g) -> float | None:
    if g is None:
        return None
    if g <= -0.10:
        return -0.5
    if g < 0:
        return -0.2
    if g < 0.10:
        return 0.2
    return 0.5


def _de_score(de) -> float | None:
    if de is None:
        return None
    # Yahoo reports debt/equity as a percentage-ish number (e.g. 80 = 0.8x)
    de_ratio = de / 100.0 if de > 5 else de
    if de_ratio > 2.0:
        return -0.5
    if de_ratio > 1.0:
        return -0.2
    return 0.2


def fundamentals_signal(symbol: str) -> Signal:
    f = get_fundamentals(symbol)
    parts = []
    notes = []

    pe = _pe_score(f.get("trailing_pe"))
    if pe is not None:
        parts.append(pe)
        notes.append(f"PE={f['trailing_pe']:.1f}" if f.get("trailing_pe") else "PE n/a")

    m = _margin_score(f.get("profit_margins"))
    if m is not None:
        parts.append(m)

    g = _growth_score(f.get("earnings_growth"))
    if g is not None:
        parts.append(g)

    de = _de_score(f.get("debt_to_equity"))
    if de is not None:
        parts.append(de)

    if not parts:
        return neutral("fundamentals", "no fundamentals available")

    score = sum(parts) / len(parts)
    score = max(-1.0, min(1.0, score))
    note = ", ".join(notes) if notes else "blended"
    return Signal(name="fundamentals", score=score, note=note, available=True)
