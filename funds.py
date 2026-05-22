"""
Tradeable fund / ETF universe for the Indian market (NSE, Yahoo Finance .NS).

Why ETFs instead of individual stocks?
  Individual stocks can go to zero. ETFs own a basket of stocks, so one
  company blowing up barely moves the fund. They are also more liquid,
  more predictable, and follow broader market trends which the LSTM can
  learn more reliably.

Risk ranking (from safest to most volatile):
  Tier 1 — Nifty 50 broad index ETFs  (tracks 50 largest companies)
  Tier 2 — Sectoral / thematic ETFs   (tracks one industry)
  Tier 3 — Mid/small cap ETFs         (tracks smaller companies)
  Tier 4 — Commodity ETFs             (gold, silver — non-equity)
  Tier 5 — International ETFs         (US/global — currency risk added)

All symbols verified tradeable on NSE with Yahoo Finance as of 2025.
Add or remove symbols here — price_cache, trainer, and agent all read
from this file automatically.
"""

from __future__ import annotations

# ── Tier 1: Broad market / Nifty 50 ETFs ─────────────────────────────────────
# Safest. Track the 50 largest and most liquid Indian companies.
# Price moves are smooth, predictable, and closely follow the index.
NIFTY50_ETFS: list[str] = [
    "NIFTYBEES.NS",    # Nippon India ETF Nifty BeES      — oldest, most liquid
    "SETFNIF50.NS",    # SBI ETF Nifty 50
    "KOTAKNIFTY.NS",   # Kotak Nifty 50 ETF
    "HDFCNIFTY.NS",    # HDFC Nifty 50 ETF
    "LICNETFN50.NS",   # LIC MF ETF Nifty 50
    "ICICIB22.NS",     # ICICI Prudential Nifty ETF
]

# Nifty Next 50 — one step below Nifty 50, still large-cap
NIFTY_NEXT50_ETFS: list[str] = [
    "JUNIORBEES.NS",   # Nippon India ETF Junior BeES     — Nifty Next 50
    "SETFNN50.NS",     # SBI ETF Nifty Next 50
]

# ── Tier 2: Sectoral ETFs ─────────────────────────────────────────────────────
# Moderate risk. Move with their sector. Good when a specific sector is in favour.
SECTORAL_ETFS: list[str] = [
    "BANKBEES.NS",     # Nippon India ETF Bank BeES       — Bank Nifty (banking)
    "ITBEES.NS",       # Nippon India ETF IT BeES         — Nifty IT (technology)
    "PHARMABEES.NS",   # Nippon India ETF Pharma BeES     — Nifty Pharma
    "PSUBNKBEES.NS",   # Nippon India ETF PSU Bank BeES   — PSU banks
    "INFRABEES.NS",    # Nippon India ETF Infra BeES      — Infrastructure
    "CONSUMBEES.NS",   # Nippon India ETF Consumption     — Consumer stocks
    "NETFENERGY.NS",   # Nippon India ETF Energy          — Energy sector
    "AUTOBEES.NS",     # Nippon India ETF Auto            — Automobile sector
]

# ── Tier 3: Mid / small cap ETFs ──────────────────────────────────────────────
# Higher risk, higher potential return. More volatile than large-cap ETFs.
MIDCAP_ETFS: list[str] = [
    "MIDCAPBEES.NS",   # Nippon India ETF Nifty Midcap 150
    "MOM50.NS",        # Motilal Oswal Nifty Midcap 50 ETF
]

# ── Tier 4: Commodity ETFs ────────────────────────────────────────────────────
# Safe-haven assets. Move independently of the stock market.
# Good for diversification — add 1–2 alongside equity ETFs.
COMMODITY_ETFS: list[str] = [
    "GOLDBEES.NS",     # Nippon India ETF Gold BeES       — tracks gold price
    "SILVERBEES.NS",   # Nippon India ETF Silver BeES     — tracks silver price
    "HNGSNGBEES.NS",   # Hang Seng BeES                   — Hong Kong equities
]

# ── Tier 5: International ETFs ────────────────────────────────────────────────
# US/global exposure. Adds currency (USD/INR) risk on top of market risk.
INTERNATIONAL_ETFS: list[str] = [
    "MON100.NS",       # Motilal Oswal Nasdaq 100 ETF    — US tech
    "NETFIT.NS",       # SBI Nifty IT ETF (tech focus)
]

# ── Named selections shown in the GUI ────────────────────────────────────────
FUND_CATEGORIES: dict[str, list[str]] = {
    "Nifty 50 ETFs  (Tier 1 — safest)":        NIFTY50_ETFS,
    "Nifty Next 50 ETFs  (Tier 1 — safe)":     NIFTY_NEXT50_ETFS,
    "Sectoral ETFs  (Tier 2 — moderate)":      SECTORAL_ETFS,
    "Mid/Small Cap ETFs  (Tier 3 — higher)":   MIDCAP_ETFS,
    "Commodity ETFs  (Tier 4 — independent)":  COMMODITY_ETFS,
    "International ETFs  (Tier 5 — currency)": INTERNATIONAL_ETFS,
}

# Flat list of all ETFs across every category
ALL_FUNDS: list[str] = [
    sym
    for syms in FUND_CATEGORIES.values()
    for sym in syms
]

# Recommended starter set — broad Nifty 50 ETFs only (lowest risk)
RECOMMENDED_FUNDS: list[str] = NIFTY50_ETFS + NIFTY_NEXT50_ETFS
