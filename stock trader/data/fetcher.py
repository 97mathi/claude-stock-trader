"""
Data fetcher: pulls historical and live prices from Yahoo Finance.

Why Yahoo Finance?
- Free, no API key, supports NSE/BSE stocks via .NS / .BO suffix.
- Good enough for paper trading and model training.

For production-grade real trading you'd swap this for a broker feed
(Zerodha Kite WebSocket, Upstox, etc.) — see broker_interface.py.
"""

from __future__ import annotations
import time
import pandas as pd
import yfinance as yf


# ----- Historical data -----

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


# ----- Live / near-live data -----

def get_latest_price(symbol: str) -> float | None:
    """
    Best-effort latest price. yfinance .fast_info is quickest;
    falls back to last 1-minute bar.
    Note: Yahoo data is delayed ~15 min for NSE — fine for paper trading,
    not for real intraday trading.
    """
    try:
        t = yf.Ticker(symbol)
        info = getattr(t, "fast_info", None)
        if info and "last_price" in info and info["last_price"]:
            return float(info["last_price"])
    except Exception:
        pass

    # Fallback: pull a 1-day, 1-minute history and use the last close
    try:
        df = yf.download(symbol, period="1d", interval="1m",
                         progress=False, auto_adjust=True)
        df = _flatten(df)
        if len(df) > 0:
            return float(df["Close"].iloc[-1])
    except Exception:
        pass
    return None


def get_latest_prices(symbols: list[str]) -> dict[str, float | None]:
    """Batch version — fetches one symbol at a time with a small pause to be polite."""
    out = {}
    for s in symbols:
        out[s] = get_latest_price(s)
        time.sleep(0.05)
    return out


# ----- Sector indices & fundamentals -----

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
    Yahoo's NSE fundamentals can be patchy - callers must handle None.
    """
    out = {
        "trailing_pe": None, "forward_pe": None,
        "price_to_book": None, "profit_margins": None,
        "earnings_growth": None, "debt_to_equity": None,
        "sector": None,
    }
    try:
        info = yf.Ticker(symbol).info or {}
        out["trailing_pe"] = info.get("trailingPE")
        out["forward_pe"] = info.get("forwardPE")
        out["price_to_book"] = info.get("priceToBook")
        out["profit_margins"] = info.get("profitMargins")
        out["earnings_growth"] = info.get("earningsGrowth")
        out["debt_to_equity"] = info.get("debtToEquity")
        out["sector"] = info.get("sector")
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
                "title": it.get("title", ""),
                "publisher": it.get("publisher", ""),
                "link": it.get("link", ""),
                "time": it.get("providerPublishTime", 0),
            })
        return out
    except Exception:
        return []


# ----- Helpers -----

def _flatten(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance sometimes returns columns as a MultiIndex.
    Flatten so we have clean 'Open','High','Low','Close','Volume' columns.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Drop fully-empty rows
    df = df.dropna(how="all")
    # Standard column names
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].copy()
