"""
Aggregator: blends the LSTM price forecast with all the alternative-data
signals into ONE combined score the agent uses to rank stocks.

Each signal is in [-1, +1]. The LSTM "edge" (predicted % return) is first
normalized into the same range, then everything is combined as a weighted
average using config.SIGNAL_WEIGHTS.

The result also keeps every individual signal so the GUI/ledger can show
exactly WHY a stock was picked or skipped.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import config
from signals.base import Signal
from signals.sector import sector_signal
from signals.fundamentals import fundamentals_signal
from signals.sentiment import sentiment_signal
from signals.macro import macro_signal
from signals.correlation import correlation_signal


# An LSTM edge of +/-5% maps to +/-1.0 normalized score.
_LSTM_EDGE_SCALE = 5.0


@dataclass
class StockScore:
    symbol: str
    combined: float                       # final blended score
    lstm_edge_pct: float                  # raw predicted % return
    signals: dict[str, Signal] = field(default_factory=dict)
    max_abs_corr: float = 0.0
    most_correlated_with: str = ""

    def explain(self) -> str:
        bits = [f"score={self.combined:+.3f}",
                f"lstm={self.lstm_edge_pct:+.2f}%"]
        for name, s in self.signals.items():
            if name == "lstm":
                continue
            flag = "" if s.available else "(n/a)"
            bits.append(f"{name}={s.score:+.2f}{flag}")
        return " | ".join(bits)


def _lstm_signal(edge_pct: float) -> Signal:
    norm = max(-1.0, min(1.0, edge_pct / _LSTM_EDGE_SCALE))
    return Signal("lstm", norm, f"forecast {edge_pct:+.2f}%", True)


def score_stock(symbol: str,
                lstm_edge_pct: float,
                holdings: list[str],
                macro: Signal | None = None) -> StockScore:
    """
    Build the combined score for one stock.
    `macro` can be passed in (computed once per cycle) to avoid recomputing.
    """
    signals: dict[str, Signal] = {}
    signals["lstm"] = _lstm_signal(lstm_edge_pct)
    signals["sector"] = sector_signal(symbol)
    signals["fundamentals"] = fundamentals_signal(symbol)
    signals["sentiment"] = sentiment_signal(symbol)
    signals["macro"] = macro if macro is not None else macro_signal()

    corr = correlation_signal(symbol, holdings)
    signals["correlation"] = corr.signal

    # Weighted average (weights from config). Missing signals still count as 0,
    # which is the safe neutral default.
    weights = config.SIGNAL_WEIGHTS
    total_w = sum(weights.get(k, 0.0) for k in signals)
    if total_w <= 0:
        combined = 0.0
    else:
        combined = sum(weights.get(k, 0.0) * s.clamped()
                       for k, s in signals.items()) / total_w

    return StockScore(
        symbol=symbol,
        combined=combined,
        lstm_edge_pct=lstm_edge_pct,
        signals=signals,
        max_abs_corr=corr.max_abs_corr,
        most_correlated_with=corr.most_correlated_with,
    )
