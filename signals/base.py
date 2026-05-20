"""
Shared types for the signals layer.

A Signal is one data source's opinion about a stock, normalized to roughly
[-1, +1]:
    +1 = strongly bullish, 0 = neutral, -1 = strongly bearish.

Each signal also carries a short human-readable `note` explaining the score,
and an `available` flag (False = data was missing, treated as neutral).
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Signal:
    name: str
    score: float          # roughly -1..+1
    note: str = ""
    available: bool = True

    def clamped(self) -> float:
        return max(-1.0, min(1.0, self.score))


def neutral(name: str, note: str = "no data") -> Signal:
    return Signal(name=name, score=0.0, note=note, available=False)
