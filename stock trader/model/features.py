"""
Technical indicators ("features") added to raw price data so the LSTM
has richer signals to learn from.

Plain-English glossary:
  - SMA (Simple Moving Average): average price over N days. Smooths noise.
  - EMA (Exponential Moving Average): like SMA but recent days count more.
  - RSI (Relative Strength Index): 0-100 score. Above 70 = overbought
    (price may dip), below 30 = oversold (price may bounce).
  - MACD (Moving Average Convergence Divergence): difference between two
    EMAs. Positive & rising = upward momentum.
  - Returns: % change between consecutive bars. The model often learns
    better from returns than raw prices.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator, SMAIndicator
from ta.volatility import BollingerBands


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take a DataFrame of OHLCV bars and add technical-indicator columns.
    Returns a new DataFrame with NaN rows dropped.
    """
    if df is None or df.empty or "Close" not in df.columns:
        return pd.DataFrame()

    out = df.copy()
    close = out["Close"]

    # Trend
    out["SMA_10"] = SMAIndicator(close, window=10).sma_indicator()
    out["SMA_30"] = SMAIndicator(close, window=30).sma_indicator()
    out["EMA_12"] = EMAIndicator(close, window=12).ema_indicator()
    out["EMA_26"] = EMAIndicator(close, window=26).ema_indicator()

    # Momentum
    out["RSI_14"] = RSIIndicator(close, window=14).rsi()

    macd = MACD(close)
    out["MACD"] = macd.macd()
    out["MACD_signal"] = macd.macd_signal()
    out["MACD_diff"] = macd.macd_diff()

    # Volatility
    bb = BollingerBands(close, window=20, window_dev=2)
    out["BB_high"] = bb.bollinger_hband()
    out["BB_low"] = bb.bollinger_lband()
    out["BB_pct"] = bb.bollinger_pband()    # 0..1, where price sits in the band

    # Returns
    out["Return_1"] = close.pct_change(1)
    out["Return_5"] = close.pct_change(5)

    return out.dropna().copy()


FEATURE_COLUMNS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_10", "SMA_30", "EMA_12", "EMA_26",
    "RSI_14",
    "MACD", "MACD_signal", "MACD_diff",
    "BB_high", "BB_low", "BB_pct",
    "Return_1", "Return_5",
]
TARGET_COLUMN = "Close"
