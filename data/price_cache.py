"""
NSE price cache — async scraper that runs in a background thread.

Architecture
------------
  App start  →  price_cache.start(symbols, interval=30)
                    └─ daemon thread → asyncio loop → scrape all symbols every 30 s
                          └─ stores {symbol: {price, timestamp}} in _cache dict

  Any module  →  price_cache.get("RELIANCE.NS")  →  float | None  (instant)

Design rules
------------
- Returns last known price regardless of age (staleness is better than Yahoo's
  15-min delay once the cache is warm).
- Returns None only if the symbol has NEVER been scraped (first ~30 s after start).
- No Yahoo fallback for live prices — Yahoo is used only for historical OHLCV.
- If aiohttp is not installed, all get() calls return None silently (app still works).

NSE scraping
------------
NSE's internal JSON API (used by their own website) returns near-real-time prices
(typically 1-3 min delay). It requires browser-like headers and a session cookie
obtained by visiting the homepage first.
URL: https://www.nseindia.com/api/quote-equity?symbol=RELIANCE  (no .NS suffix)
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from typing import Callable

from nifty50 import NIFTY_50 as _DEFAULT_UNIVERSE

logger = logging.getLogger(__name__)

try:
    import aiohttp
    _AIOHTTP_OK = True
except ImportError:
    _AIOHTTP_OK = False
    logger.warning("aiohttp not installed — price cache disabled. "
                   "Run: pip install aiohttp")

_NSE_QUOTE = "https://www.nseindia.com/api/quote-equity?symbol={sym}"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate",   # no br — aiohttp needs Brotli pkg for that
    "Referer":         "https://www.nseindia.com/",
    "Connection":      "keep-alive",
}


def _strip_suffix(symbol: str) -> str:
    """RELIANCE.NS → RELIANCE,  INFY.BO → INFY"""
    return symbol.split(".")[0].upper()


class PriceCache:
    """
    Singleton async price cache backed by NSE scraping.
    Use the module-level `price_cache` instance — don't instantiate directly.
    """

    _instance: PriceCache | None = None

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}   # {symbol: {price, timestamp}}
        self._lock   = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running   = False
        self._symbols:  list[str] = []
        self._interval  = 30
        self._last_run:  datetime | None = None
        self._last_count = 0                 # symbols updated in last cycle
        # Optional GUI callback: fn(updated_count, total) called after each cycle
        self.on_update: Callable[[int, int], None] | None = None

    @classmethod
    def instance(cls) -> PriceCache:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------ #
    #  Control                                                             #
    # ------------------------------------------------------------------ #

    def start(self, symbols: list[str] | None = None, interval: int = 30) -> None:
        """Start the background scraping thread. Safe to call multiple times.
        symbols defaults to NIFTY_50 from nifty50.py — the same list used by
        the trainer — so training and scraping always stay in sync.
        """
        if self._running:
            return
        if not _AIOHTTP_OK:
            return
        self._symbols  = list(symbols or _DEFAULT_UNIVERSE)
        self._interval = max(10, interval)
        self._running  = True
        self._thread   = threading.Thread(
            target=self._loop, name="PriceCache", daemon=True)
        self._thread.start()
        logger.info("PriceCache started — %d symbols, interval %ds",
                    len(self._symbols), self._interval)

    def stop(self) -> None:
        self._running = False

    def is_running(self) -> bool:
        return (self._running
                and self._thread is not None
                and self._thread.is_alive())

    # ------------------------------------------------------------------ #
    #  Read API                                                            #
    # ------------------------------------------------------------------ #

    def get(self, symbol: str) -> float | None:
        """Last known price. None only if this symbol has never been scraped."""
        with self._lock:
            entry = self._cache.get(symbol)
            return float(entry["price"]) if entry else None

    def get_many(self, symbols: list[str]) -> dict[str, float | None]:
        """Batch read — instant, no HTTP calls."""
        with self._lock:
            return {s: (float(self._cache[s]["price"])
                        if s in self._cache else None)
                    for s in symbols}

    def last_updated(self, symbol: str) -> datetime | None:
        with self._lock:
            e = self._cache.get(symbol)
            return e["timestamp"] if e else None

    def status(self) -> dict:
        """Summary dict for the GUI status label."""
        with self._lock:
            total   = len(self._symbols)
            cached  = len(self._cache)
            last    = self._last_run
        return {
            "running":    self.is_running(),
            "cached":     cached,
            "total":      total,
            "last_run":   last,
            "last_count": self._last_count,
        }

    # ------------------------------------------------------------------ #
    #  Background loop                                                     #
    # ------------------------------------------------------------------ #

    def _loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while self._running:
            try:
                updated = loop.run_until_complete(self._fetch_all())
                self._last_count = updated
                self._last_run   = datetime.now()
                if self.on_update:
                    self.on_update(updated, len(self._symbols))
            except Exception as exc:
                logger.debug("PriceCache cycle error: %s", exc)
            # Sleep in small steps so stop() is noticed quickly
            for _ in range(self._interval * 2):
                if not self._running:
                    break
                time.sleep(0.5)
        loop.close()

    async def _fetch_all(self) -> int:
        timeout   = aiohttp.ClientTimeout(total=25)
        connector = aiohttp.TCPConnector(ssl=False, limit=20)
        async with aiohttp.ClientSession(
            headers=_HEADERS, timeout=timeout, connector=connector
        ) as session:
            tasks   = [self._fetch_one(session, s) for s in self._symbols]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            return sum(1 for r in results if r is True)

    async def _fetch_one(self, session: aiohttp.ClientSession,
                         symbol: str) -> bool:
        nse_sym = _strip_suffix(symbol)
        url = _NSE_QUOTE.format(sym=nse_sym)
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return False
                data  = await resp.json(content_type=None)
                price = (data.get("priceInfo") or {}).get("lastPrice")
                if price is None:
                    return False
                with self._lock:
                    self._cache[symbol] = {
                        "price":     float(price),
                        "timestamp": datetime.now(),
                    }
                return True
        except Exception:
            return False


# Module-level singleton — import this everywhere
price_cache = PriceCache.instance()
