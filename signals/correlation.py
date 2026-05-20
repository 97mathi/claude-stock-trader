"""
Correlation / diversification signal.

Goal: avoid loading up on stocks that all move together (concentration risk).

For a candidate stock, we look at how correlated its daily returns are with the
stocks already held. High correlation with existing holdings = bad for
diversification = negative score. Low/negative correlation = good = positive.

If there are no holdings yet, correlation is neutral (nothing to diversify
against). Returns the MAX absolute correlation too, so the agent can hard-block
buys above config.AGENT_MAX_CORRELATION.
"""

from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

from data.fetcher import get_daily_history
from signals.base import Signal, neutral


@dataclass
class CorrelationResult:
    signal: Signal
    max_abs_corr: float          # 0..1, highest correlation with any holding
    most_correlated_with: str    # symbol


_cache: dict[str, pd.Series] = {}


def _returns(symbol: str) -> pd.Series | None:
    if symbol in _cache:
        return _cache[symbol]
    hist = get_daily_history(symbol, "6mo")
    if hist is None or hist.empty or "Close" not in hist.columns:
        return None
    rets = hist["Close"].pct_change().dropna()
    _cache[symbol] = rets
    return rets


def clear_cache():
    _cache.clear()


def correlation_signal(candidate: str,
                       holdings: list[str]) -> CorrelationResult:
    if not holdings:
        return CorrelationResult(
            signal=Signal("correlation", 0.2, "no holdings - free to diversify"),
            max_abs_corr=0.0, most_correlated_with="",
        )

    cand_ret = _returns(candidate)
    if cand_ret is None:
        return CorrelationResult(
            signal=neutral("correlation", "no return data"),
            max_abs_corr=0.0, most_correlated_with="",
        )

    max_abs = 0.0
    worst = ""
    for h in holdings:
        if h == candidate:
            return CorrelationResult(
                signal=Signal("correlation", -1.0, "already held"),
                max_abs_corr=1.0, most_correlated_with=h,
            )
        h_ret = _returns(h)
        if h_ret is None:
            continue
        joined = pd.concat([cand_ret, h_ret], axis=1, join="inner").dropna()
        if len(joined) < 20:
            continue
        c = float(joined.iloc[:, 0].corr(joined.iloc[:, 1]))
        if abs(c) > max_abs:
            max_abs = abs(c)
            worst = h

    # Map correlation to score: low corr -> +, high corr -> -
    score = max(-1.0, min(1.0, 1.0 - 2.0 * max_abs))
    note = (f"max corr {max_abs:.2f} with {worst}" if worst
            else "no overlap")
    return CorrelationResult(
        signal=Signal("correlation", score, note),
        max_abs_corr=max_abs, most_correlated_with=worst,
    )
