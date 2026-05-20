"""
Nifty 50 symbols pre-loaded into the app.
Yahoo Finance uses suffix '.NS' for NSE-listed Indian stocks.
Example: Reliance Industries -> 'RELIANCE.NS'.
"""

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "M&M.NS", "ULTRACEMCO.NS",
    "WIPRO.NS", "NESTLEIND.NS", "ADANIENT.NS", "NTPC.NS", "POWERGRID.NS",
    "ONGC.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "COALINDIA.NS",
    "BAJAJFINSV.NS", "GRASIM.NS", "INDUSINDBK.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "DRREDDY.NS", "CIPLA.NS", "BRITANNIA.NS", "DIVISLAB.NS", "EICHERMOT.NS",
    "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "TECHM.NS", "ADANIPORTS.NS", "TATACONSUM.NS",
    "APOLLOHOSP.NS", "UPL.NS", "BPCL.NS", "HINDALCO.NS", "LTIM.NS",
]


def is_indian_symbol(symbol: str) -> bool:
    """Return True if symbol looks like an NSE/BSE ticker."""
    return symbol.endswith(".NS") or symbol.endswith(".BO")


def normalize_symbol(symbol: str) -> str:
    """Make sure a user-typed symbol has the .NS suffix for NSE."""
    s = symbol.strip().upper()
    if "." in s:
        return s
    return s + ".NS"
