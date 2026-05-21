"""
Data fetcher: pulls historical and live prices from Yahoo Finance.

Live-price strategy (fastest-first waterfall):
  1. yf.Ticker.fast_info["last_price"]   — single lightweight HTTP call (~0.3 s)
  2. yf.download 1-min bars fallback     — heavier but reliable (~3 s)

Batch fetching (get_latest_prices) runs all symbols in parallel threads
(max 10 at once) so 50 Nifty stocks take ~3-5 s instead of ~30 s.

Delay note:
  Yahoo Finance delays NSE data by ~15 minutes on their free feed.
  This is fine for paper trading and swing-trade monitoring.
  For real intraday trading swap to a broker WebSocket feed
  (Zerodha Kite / Upstox / Angel SmartAPI) — see broker_interface.py.
"""

from __future__ import annotations
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf


# ------------------------------------------------------------------ #
#  Historical data                                                     #
# ------------------------------------------------------------------ #

def get_daily_history(symbol: str, period: str = "2y") -> pd.DataFrame:
    """
    Daily OHLCV (Open, High, Low, Close, Volume) bars for the past `period`.
    Used to train the swing-trading LSTM.
    period examples: '6mo', '1y', '2y', '5y', 'max'.
    """
    df = yf.download(symbol, period=period, interval="1d",
                     progress=False, auto_adjust=True)
    return _flatten(df)


def get_hourly_history(symbol: str, period: str = "60d") -> pd.DataFrame:
    """
    Hourly OHLCV bars. Used to train the intraday LSTM.
    Yahoo limits hourly data to ~730 days; 60d is a safe default.
    """
    df = yf.download(symbol, period=period, interval="1h",
                     progress=False, auto_adjust=True)
    return _flatten(df)


# ------------------------------------------------------------------ #
#  Live / near-live prices                                            #
# ------------------------------------------------------------------ #

def get_latest_price(symbol: str) -> float | None:
    """
    Best-effort latest price for one symbol.

    Waterfall:
      1. fast_info["last_price"]  — quickest path (~0.3 s per call)
      2. 1-min bar download       — heavier fallback (~3 s)

    ⚠ Yahoo NSE data is delayed ~15 min on the free feed.
    Fine for paper trading; use a broker WebSocket for real intraday.
    """
    # -- Fast path --
    try:
        t    = yf.Ticker(symbol)
        info = getattr(t, "fast_info", None)
        if info is not None:
            price = info.get("last_price") if hasattr(info, "get") \
                else getattr(info, "last_price", None)
            if price and float(price) > 0:
                return float(price)
    except Exception:
        pass

    # -- Fallback: last close of today's 1-min bars --
    try:
        df = yf.download(symbol, period="1d", interval="1m",
                         progress=False, auto_adjust=True)
        df = _flatten(df)
        if len(df) > 0:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass

    return None


def get_latest_prices(symbols: list[str],
                      max_workers: int = 10) -> dict[str, float | None]:
    """
    Parallel batch price fetch — all symbols fetched concurrently.

    Uses a ThreadPoolExecutor so 50 stocks take ~3-5 s instead of ~30 s.
    max_workers=10 keeps Yahoo from rate-limiting (60 req/min limit).
    """
    results: dict[str, float | None] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_to_sym = {pool.submit(get_latest_price, s): s
                         for s in symbols}
        for future in as_completed(future_to_sym):
            sym = future_to_sym[future]
            try:
                results[sym] = future.result()
            except Exception:
                results[sym] = None

    return results


# ------------------------------------------------------------------ #
#  Sector indices & fundamentals                                       #
# ------------------------------------------------------------------ #

def get_index_history(index_symbol: str, period: str = "3mo") -> pd.DataFrame:
    """Daily history for a sector/market index (e.g. '^NSEBANK')."""
    try:
        df = yf.download(index_symbol, period=period, interval="1d",
                         progress=False, auto_adjust=True)
        return _flatten(df)
    except Exception:
        return pd.DataFrame()


def get_fundamentals(symbol: str) -> dict:
    """
    Best-effort fundamentals from Yahoo Finance .info.
    Returns a dict with keys that may be None if unavailable.
    Yahoo's NSE fundamentals can be patchy — callers must handle None.
    """
    out = {
        "trailing_pe": None, "forward_pe": None,
        "price_to_book": None, "profit_margins": None,
        "earnings_growth": None, "debt_to_equity": None,
        "sector": None,
    }
    try:
        info = yf.Ticker(symbol).info or {}
        out["trailing_pe"]    = info.get("trailingPE")
        out["forward_pe"]     = info.get("forwardPE")
        out["price_to_book"]  = info.get("priceToBook")
        out["profit_margins"] = info.get("profitMargins")
        out["earnings_growth"]= info.get("earningsGrowth")
        out["debt_to_equity"] = info.get("debtToEquity")
        out["sector"]         = info.get("sector")
    except Exception:
        pass
    return out


def get_recent_news(symbol: str, limit: int = 20) -> list[dict]:
    """
    Yahoo Finance attaches recent news to each ticker. Free, no key.
    Returns list of {title, publisher, link, time}. Empty if none.
    (The dedicated sentiment signal can also use NewsAPI if a key is set.)
    """
    try:
        items = yf.Ticker(symbol).news or []
        out = []
        for it in items[:limit]:
            out.append({
                "title":     it.get("title", ""),
                "publisher": it.get("publisher", ""),
                "link":      it.get("link", ""),
                "time":      it.get("providerPublishTime", 0),
            })
        return out
    except Exception:
        return []


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns columns as a MultiIndex.
    Flatten so we have clean 'Open','High','Low','Close','Volume' columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.dropna(how="all")
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"]
            if c in df.columns]
    return df[keep].copy()
