"""
Macro regime signal (RBI rate / inflation / GDP).

India has no reliable free programmatic source for these, so the values come
from config.MACRO_INPUTS (you update them when the RBI / MoSPI publish new
numbers). This signal turns the macro picture into a single risk-on / risk-off
score applied to EVERY stock equally.

Logic (each piece nudges the score):
  - RBI last move: "cut" = bullish (+), "hike" = bearish (-), "hold" = 0.
  - Inflation vs target: well above target = bearish, near/below = bullish.
  - GDP growth + trend: strong & rising = bullish, weak & falling = bearish.

Returns one Signal reused for all stocks in a cycle.
"""

from __future__ import annotations
import config
from signals.base import Signal


def macro_signal() -> Signal:
    m = config.MACRO_INPUTS
    parts = []
    notes = []

    # RBI policy stance
    move = str(m.get("rbi_last_move", "hold")).lower()
    if move == "cut":
        parts.append(0.6); notes.append("RBI cut(+)")
    elif move == "hike":
        parts.append(-0.6); notes.append("RBI hike(-)")
    else:
        parts.append(0.0); notes.append("RBI hold")

    # Inflation vs target
    cpi = m.get("cpi_inflation_pct")
    target = m.get("inflation_target_pct", 4.0)
    if cpi is not None:
        gap = cpi - target
        if gap > 2.0:
            parts.append(-0.6); notes.append(f"CPI {cpi:.1f}% hot(-)")
        elif gap > 0.5:
            parts.append(-0.2); notes.append(f"CPI {cpi:.1f}% warm")
        else:
            parts.append(0.3); notes.append(f"CPI {cpi:.1f}% ok(+)")

    # GDP growth + trend
    gdp = m.get("gdp_growth_pct")
    trend = str(m.get("gdp_trend", "stable")).lower()
    if gdp is not None:
        base = 0.4 if gdp >= 6.0 else (0.0 if gdp >= 4.0 else -0.4)
        if trend == "rising":
            base += 0.2
        elif trend == "falling":
            base -= 0.2
        parts.append(max(-1.0, min(1.0, base)))
        notes.append(f"GDP {gdp:.1f}% {trend}")

    score = sum(parts) / len(parts) if parts else 0.0
    score = max(-1.0, min(1.0, score))
    return Signal(name="macro", score=score,
                  note="; ".join(notes), available=True)
