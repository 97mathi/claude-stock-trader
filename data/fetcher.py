"""
Data fetcher.

Live prices  →  data/price_cache.py  (NSE scrape, ~1-3 min delay, no HTTP here)
Historical OHLCV  →  Yahoo Finance (used only for LSTM training, not live prices)

get_latest_price()  and  get_latest_prices()  are thin wrappers that read from
the PriceCache singleton started by the GUI at launch.  They return None during
the first ~30 s before the cache's first cycle completes.
"""

from __future__ import annotations

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
    Returns the latest known price for one symbol.

    Source priority:
      1. PriceCache (NSE scrape, ~1-3 min delay) — used if cache has ANY value
         for this symbol, even if stale. NSE data is always fresher than Yahoo.
      2. None — returned when cache is empty for this symbol (first ~30 s after
         app start, before the first scrape cycle completes).

    Yahoo Finance is NOT used here — its ~15-min delay makes it worse than any
    cached NSE value. Yahoo is only used for historical OHLCV (model training).
    """
    from data.price_cache import price_cache
    return price_cache.get(symbol)


def get_latest_prices(symbols: list[str],
                      max_workers: int = 10) -> dict[str, float | None]:
    """
    Batch price read — all values come from the in-memory PriceCache instantly.
    No HTTP calls; the cache background thread handles fetching independently.
    `max_workers` kept for API compatibility but is unused.
    """
    from data.price_cache import price_cache
    return price_cache.get_many(symbols)


# ------------------------------------------------------------------ #
#  Sector indices & fundamentals                                       #
# ------------------------------------------------------------------ #

def get_current_rsi(symbol: str, period: int = 14) -> float | None:
    """
    Compute the current RSI for a symbol using recent daily closes.
    Used by the agent to skip overbought stocks at entry.
    Returns None if data is insufficient or an error occurs.
    """
    try:
        from ta.momentum import RSIIndicator
        df = get_daily_history(symbol, period="60d")
        if df.empty or len(df) < period + 5:
            return None
        rsi_series = RSIIndicator(close=df["Close"], window=period).rsi()
        val = rsi_series.iloc[-1]
        return float(val) if pd.notna(val) else None
    except Exception:
        return None


def get_nifty_trend(sma_period: int = 20) -> dict:
    """
    Check whether the Nifty 50 index is in an uptrend.
    Returns a dict:
      uptrend    : bool  — True if Nifty close > N-day SMA (safe to buy)
      current    : float | None — latest Nifty close
      sma        : float | None — current SMA value
      pct_vs_sma : float — how far above/below SMA as a % (positive = above)

    Fails gracefully (uptrend=True) if Yahoo can't serve the data,
    so a network glitch doesn't freeze the entire buy scan.
    """
    try:
        df = get_daily_history("^NSEI", period="60d")
        if df.empty or len(df) < sma_period:
            return {"uptrend": True, "current": None, "sma": None, "pct_vs_sma": 0.0}
        current = float(df["Close"].iloc[-1])
        sma     = float(df["Close"].rolling(sma_period).mean().iloc[-1])
        pct     = (current - sma) / sma * 100.0
        return {"uptrend": current > sma, "current": current,
                "sma": sma, "pct_vs_sma": pct}
    except Exception:
        return {"uptrend": True, "current": None, "sma": None, "pct_vs_sma": 0.0}


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
