"""
News sentiment signal.

Two data sources, tried in order:
  1. Yahoo Finance per-ticker news headlines (free, no key) - always tried.
  2. NewsAPI.org (if config.NEWSAPI_KEY or env NEWSAPI_KEY is set) - richer.

Headlines are scored with VADER (a rule-based sentiment analyzer tuned for
short social/news text). If the `vaderSentiment` package isn't installed, a
tiny built-in keyword lexicon is used as a fallback. If no headlines are
found at all, the signal is neutral (0).

Score = average headline sentiment, in [-1, +1].
"""

from __future__ import annotations
import os
import config
from data.fetcher import get_recent_news
from signals.base import Signal, neutral


# ---- sentiment scorer (VADER if available, else tiny lexicon) ----

def _make_scorer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        an = SentimentIntensityAnalyzer()
        return lambda text: an.polarity_scores(text)["compound"]
    except Exception:
        pos = {"surge", "jump", "gain", "rise", "beat", "profit", "growth",
               "upgrade", "record", "strong", "rally", "soar", "bullish",
               "wins", "approval", "expansion", "high"}
        neg = {"fall", "drop", "loss", "miss", "decline", "downgrade", "weak",
               "plunge", "slump", "bearish", "fraud", "probe", "cut", "lawsuit",
               "ban", "default", "crash", "low"}

        def score(text: str) -> float:
            t = text.lower()
            p = sum(1 for w in pos if w in t)
            n = sum(1 for w in neg if w in t)
            if p == n == 0:
                return 0.0
            return (p - n) / (p + n)
        return score


_SCORER = _make_scorer()


def _newsapi_headlines(symbol: str) -> list[str]:
    """Fetch headlines from NewsAPI.org if a key is configured."""
    key = config.NEWSAPI_KEY or os.environ.get("NEWSAPI_KEY", "")
    if not key:
        return []
    try:
        import urllib.request
        import urllib.parse
        import json as _json
        query = symbol.replace(".NS", "").replace(".BO", "")
        params = urllib.parse.urlencode({
            "q": query,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": key,
        })
        url = f"https://newsapi.org/v2/everything?{params}"
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = _json.loads(resp.read().decode())
        return [a.get("title", "") for a in data.get("articles", [])]
    except Exception:
        return []


def sentiment_signal(symbol: str) -> Signal:
    headlines = [n["title"] for n in get_recent_news(symbol) if n.get("title")]
    headlines += _newsapi_headlines(symbol)
    headlines = [h for h in headlines if h]

    if not headlines:
        return neutral("sentiment", "no headlines found")

    scores = [_SCORER(h) for h in headlines]
    avg = sum(scores) / len(scores)
    avg = max(-1.0, min(1.0, avg))
    return Signal(
        name="sentiment",
        score=avg,
        note=f"{len(headlines)} headlines, avg {avg:+.2f}",
        available=True,
    )
