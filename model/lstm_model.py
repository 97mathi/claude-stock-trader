"""
LSTM (Long Short-Term Memory) prediction engine — PyTorch implementation.

What it does, in plain English:
  1. Take historical bars + technical indicators.
  2. Scale all numbers to 0..1 (neural networks learn better that way).
  3. Slice the data into rolling windows: [previous 60 bars] -> [next bar's close].
  4. Train a small LSTM network.
  5. Iteratively predict N steps into the future (feed the prediction back in).

Why PyTorch instead of TensorFlow?
  - PyTorch supports newer Python versions (3.9-3.13) — TensorFlow lags behind.
  - The model architecture and behavior are equivalent.

The model is saved per-symbol-per-horizon so you only train once per stock
and re-use it; you can retrain whenever you want by hitting "Retrain".
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import MinMaxScaler

from .features import add_indicators, FEATURE_COLUMNS, TARGET_COLUMN
from data.fetcher import get_daily_history, get_hourly_history
import config


Horizon = Literal["swing", "intraday"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Prediction:
    symbol: str
    horizon: Horizon
    current_price: float
    predicted_price: float        # price at the end of the forecast horizon
    predicted_path: list[float]   # step-by-step forecast values
    expected_return_pct: float    # (predicted - current) / current * 100
    confidence: float             # rough 0..1 score (1 - normalized recent error)


# -------------------- model definition --------------------

class LSTMRegressor(nn.Module):
    """
    Two-layer LSTM with dropout, followed by a small dense head.
    Input shape: (batch, lookback, n_features)
    Output: predicted scaled close for the next bar.
    """

    def __init__(self, n_features: int, hidden1: int = 64, hidden2: int = 32,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size=n_features, hidden_size=hidden1,
                             batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2,
                             batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden2, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])   # take last time-step
        out = self.relu(self.fc1(out))
        return self.fc2(out).squeeze(-1)


# -------------------- helpers --------------------

def _make_windows(arr: np.ndarray, lookback: int, target_idx: int):
    """Slice scaled data into (X, y) supervised pairs."""
    X, y = [], []
    for i in range(lookback, len(arr)):
        X.append(arr[i - lookback:i, :])
        y.append(arr[i, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def _model_path(symbol: str, horizon: Horizon) -> str:
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    safe = symbol.replace(".", "_").replace("&", "and")
    return os.path.join(config.MODEL_DIR, f"{safe}_{horizon}.pt")


def _meta_path(symbol: str, horizon: Horizon) -> str:
    safe = symbol.replace(".", "_").replace("&", "and")
    return os.path.join(config.MODEL_DIR, f"{safe}_{horizon}.json")


# -------------------- training --------------------

def train(symbol: str, horizon: Horizon = "swing",
          epochs: int | None = None) -> dict:
    """
    Train (or retrain) the LSTM for one symbol & horizon.
    Returns a small dict with training stats.
    """
    epochs = epochs or config.LSTM_EPOCHS
    lookback = config.LSTM_LOOKBACK_DAYS

    raw = get_daily_history(symbol, "5y") if horizon == "swing" \
        else get_hourly_history(symbol, "60d")
    if raw is None or raw.empty:
        raise RuntimeError(f"No data returned for {symbol}")

    df = add_indicators(raw)
    if len(df) < lookback + 30:
        raise RuntimeError(
            f"Not enough data for {symbol}: only {len(df)} usable bars.")

    feats = df[FEATURE_COLUMNS].values
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(feats)
    target_idx = FEATURE_COLUMNS.index(TARGET_COLUMN)

    X, y = _make_windows(scaled, lookback, target_idx)
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    train_dl = DataLoader(train_ds, batch_size=config.LSTM_BATCH_SIZE,
                          shuffle=True)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    model = LSTMRegressor(n_features=X.shape[2]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = 4
    bad_epochs = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(loss_fn(val_pred, y_val_t).item())

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break  # early stopping

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Persist model weights + meta
    torch.save({"state_dict": model.state_dict(),
                "n_features": X.shape[2]}, _model_path(symbol, horizon))

    meta = {
        "symbol": symbol,
        "horizon": horizon,
        "feature_columns": FEATURE_COLUMNS,
        "target_column": TARGET_COLUMN,
        "lookback": lookback,
        "scaler_min": scaler.data_min_.tolist(),
        "scaler_max": scaler.data_max_.tolist(),
        "val_loss": float(best_val if best_val != float("inf") else 0.99),
        "train_rows": int(len(X_train)),
    }
    with open(_meta_path(symbol, horizon), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# -------------------- prediction --------------------

def _load(symbol: str, horizon: Horizon):
    """Load saved model + meta, training first if missing."""
    mpath = _model_path(symbol, horizon)
    metapath = _meta_path(symbol, horizon)
    if not (os.path.exists(mpath) and os.path.exists(metapath)):
        train(symbol, horizon)

    with open(metapath) as f:
        meta = json.load(f)

    checkpoint = torch.load(mpath, map_location=DEVICE)
    model = LSTMRegressor(n_features=checkpoint["n_features"]).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, meta


def predict(symbol: str, horizon: Horizon = "swing") -> Prediction:
    """
    Run a multi-step forecast for `symbol`.
    Trains the model on the fly if no saved one exists.
    """
    model, meta = _load(symbol, horizon)
    lookback = meta["lookback"]
    feature_cols = meta["feature_columns"]
    target_idx = feature_cols.index(meta["target_column"])

    raw = get_daily_history(symbol, "1y") if horizon == "swing" \
        else get_hourly_history(symbol, "60d")
    df = add_indicators(raw)
    if len(df) < lookback:
        raise RuntimeError(
            f"Not enough recent data for {symbol} to predict.")

    feats = df[feature_cols].values
    data_min = np.array(meta["scaler_min"])
    data_max = np.array(meta["scaler_max"])
    data_range = data_max - data_min + 1e-9
    scaled = (feats - data_min) / data_range

    window = scaled[-lookback:, :].copy().astype(np.float32)
    steps = config.SWING_HORIZON_DAYS if horizon == "swing" \
        else config.INTRADAY_HORIZON_BARS

    path_scaled = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.from_numpy(window).unsqueeze(0).to(DEVICE)
            next_close_scaled = float(model(x).item())
            path_scaled.append(next_close_scaled)

            # Synthetic next row: keep prior features, update Close.
            next_row = window[-1, :].copy()
            next_row[target_idx] = next_close_scaled
            window = np.vstack([window[1:], next_row])

    # Inverse-scale the predicted close path
    path_unscaled = [
        float(v * data_range[target_idx] + data_min[target_idx])
        for v in path_scaled
    ]

    current_price = float(df[meta["target_column"]].iloc[-1])
    predicted_price = path_unscaled[-1]
    expected_return = (predicted_price - current_price) / current_price * 100.0

    # Very simple "confidence": 1 - clipped validation loss.
    val_loss = float(meta.get("val_loss", 0.01))
    confidence = max(0.0, min(1.0, 1.0 - val_loss * 10))

    return Prediction(
        symbol=symbol,
        horizon=horizon,
        current_price=current_price,
        predicted_price=predicted_price,
        predicted_path=path_unscaled,
        expected_return_pct=expected_return,
        confidence=confidence,
    )
