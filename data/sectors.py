"""
Maps each Nifty 50 stock to its NSE sector index (Yahoo Finance symbol).

Used by the sector-momentum signal: if a stock's sector index is trending up,
that's a tailwind for the stock; if it's trending down, a headwind.

Note: Yahoo's coverage of Indian sector indices is imperfect. If an index
can't be fetched, the sector signal simply returns neutral (0) for that stock.
"""

# Common NSE sector index tickers on Yahoo Finance
NIFTY_BANK = "^NSEBANK"
NIFTY_IT = "^CNXIT"
NIFTY_AUTO = "^CNXAUTO"
NIFTY_PHARMA = "^CNXPHARMA"
NIFTY_FMCG = "^CNXFMCG"
NIFTY_METAL = "^CNXMETAL"
NIFTY_ENERGY = "^CNXENERGY"
NIFTY_FIN = "NIFTY_FIN_SERVICE.NS"
NIFTY_REALTY = "^CNXREALTY"
NIFTY_INFRA = "^CNXINFRA"
NIFTY_50 = "^NSEI"   # fallback / broad market

# stock symbol -> sector index symbol
STOCK_TO_SECTOR = {
    "RELIANCE.NS": NIFTY_ENERGY,
    "TCS.NS": NIFTY_IT,
    "HDFCBANK.NS": NIFTY_BANK,
    "ICICIBANK.NS": NIFTY_BANK,
    "INFY.NS": NIFTY_IT,
    "HINDUNILVR.NS": NIFTY_FMCG,
    "ITC.NS": NIFTY_FMCG,
    "SBIN.NS": NIFTY_BANK,
    "BHARTIARTL.NS": NIFTY_50,
    "KOTAKBANK.NS": NIFTY_BANK,
    "LT.NS": NIFTY_INFRA,
    "AXISBANK.NS": NIFTY_BANK,
    "BAJFINANCE.NS": NIFTY_FIN,
    "ASIANPAINT.NS": NIFTY_50,
    "MARUTI.NS": NIFTY_AUTO,
    "HCLTECH.NS": NIFTY_IT,
    "SUNPHARMA.NS": NIFTY_PHARMA,
    "TITAN.NS": NIFTY_50,
    "M&M.NS": NIFTY_AUTO,
    "ULTRACEMCO.NS": NIFTY_INFRA,
    "WIPRO.NS": NIFTY_IT,
    "NESTLEIND.NS": NIFTY_FMCG,
    "ADANIENT.NS": NIFTY_50,
    "NTPC.NS": NIFTY_ENERGY,
    "POWERGRID.NS": NIFTY_ENERGY,
    "ONGC.NS": NIFTY_ENERGY,
    "TATAMOTORS.NS": NIFTY_AUTO,
    "TATASTEEL.NS": NIFTY_METAL,
    "JSWSTEEL.NS": NIFTY_METAL,
    "COALINDIA.NS": NIFTY_ENERGY,
    "BAJAJFINSV.NS": NIFTY_FIN,
    "GRASIM.NS": NIFTY_50,
    "INDUSINDBK.NS": NIFTY_BANK,
    "HDFCLIFE.NS": NIFTY_FIN,
    "SBILIFE.NS": NIFTY_FIN,
    "DRREDDY.NS": NIFTY_PHARMA,
    "CIPLA.NS": NIFTY_PHARMA,
    "BRITANNIA.NS": NIFTY_FMCG,
    "DIVISLAB.NS": NIFTY_PHARMA,
    "EICHERMOT.NS": NIFTY_AUTO,
    "HEROMOTOCO.NS": NIFTY_AUTO,
    "BAJAJ-AUTO.NS": NIFTY_AUTO,
    "TECHM.NS": NIFTY_IT,
    "ADANIPORTS.NS": NIFTY_INFRA,
    "TATACONSUM.NS": NIFTY_FMCG,
    "APOLLOHOSP.NS": NIFTY_PHARMA,
    "UPL.NS": NIFTY_50,
    "BPCL.NS": NIFTY_ENERGY,
    "HINDALCO.NS": NIFTY_METAL,
    "LTIM.NS": NIFTY_IT,
}


def sector_for(symbol: str) -> str:
    """Return the sector index symbol for a stock, or the broad index."""
    return STOCK_TO_SECTOR.get(symbol, NIFTY_50)
