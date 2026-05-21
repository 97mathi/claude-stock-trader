"""
LSTM (Long Short-Term Memory) prediction engine — PyTorch implementation.

What it does, in plain English:
  1. Take historical bars + technical indicators.
  2. Scale all numbers to 0..1 (neural networks learn better that way).
  3. Slice the data into rolling windows: [previous 60 bars] -> [next bar's close].
  4. Walk-forward search: try 4 architectures over 2 expanding folds, pick the
     one with lowest combined MAE + directional-error. Directional accuracy is
     weighted 60% because for trading we care more about up/down direction than
     exact price level.
  5. Final training on full 85% split with the winning architecture.
  6. Iteratively predict N steps into the future (feed the prediction back in).

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


# -------------------- architecture search candidates --------------------
# Four candidates covering small/medium/large and varying regularisation.
# The walk-forward search picks the winner for each stock/horizon pair.

_ARCH_CANDIDATES: list[dict] = [
    {"hidden1": 32,  "hidden2": 16, "dropout": 0.1},   # small – fast, low capacity
    {"hidden1": 64,  "hidden2": 32, "dropout": 0.2},   # medium – former default
    {"hidden1": 128, "hidden2": 64, "dropout": 0.2},   # large  – higher capacity
    {"hidden1": 64,  "hidden2": 32, "dropout": 0.3},   # medium + stronger dropout
]

_WF_FOLDS = 2        # number of walk-forward folds per architecture
_WF_EPOCH_DIV = 3    # fold epochs = full_epochs // this  (keeps search fast)


@dataclass
class Prediction:
    symbol: str
    horizon: Horizon
    current_price: float
    predicted_price: float        # price at the end of the forecast horizon
    predicted_path: list[float]   # step-by-step forecast values
    expected_return_pct: float    # (predicted - current) / current * 100
    confidence: float             # directional accuracy from walk-forward (0..1)


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


# -------------------- single-fold training --------------------

def _train_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                X_val: np.ndarray, y_val: np.ndarray,
                arch: dict, epochs: int,
                patience: int = 4) -> tuple[LSTMRegressor, float]:
    """
    Train one LSTMRegressor on a train/val split.
    Returns (best_model, best_val_loss).
    Uses early stopping with `patience` epochs of no improvement.
    """
    model = LSTMRegressor(
        n_features=X_tr.shape[2],
        hidden1=arch["hidden1"],
        hidden2=arch["hidden2"],
        dropout=arch["dropout"],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    train_dl = DataLoader(train_ds, batch_size=config.LSTM_BATCH_SIZE,
                          shuffle=True)
    X_val_t = torch.from_numpy(X_val).to(DEVICE)
    y_val_t = torch.from_numpy(y_val).to(DEVICE)

    best_val = float("inf")
    best_state: dict | None = None
    bad = 0

    for _ in range(epochs):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            vl = float(loss_fn(model(X_val_t), y_val_t).item())

        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# -------------------- walk-forward evaluation --------------------

def _walk_forward_score(X: np.ndarray, y: np.ndarray,
                         arch: dict, epochs: int) -> dict:
    """
    Expanding-window walk-forward cross-validation for one architecture.

    Data is divided into (_WF_FOLDS + 1) equal chunks.
    Fold k trains on chunks [0..k] and validates on chunk [k+1].
    This mimics real usage: the model always sees only past data.

    Scoring (lower combined = better):
      combined = 0.4 * avg_MAE + 0.6 * (1 - avg_directional_accuracy)

    Directional accuracy is weighted 60% because knowing whether the stock
    will go UP or DOWN is more actionable than knowing the exact price.

    Returns dict: {"mae", "dir_acc", "combined"}
    """
    n = len(X)
    chunk = n // (_WF_FOLDS + 1)
    if chunk < 20:
        # Not enough data — return neutral score (won't beat a real result)
        return {"mae": 0.05, "dir_acc": 0.5, "combined": 0.325}

    fold_epochs = max(8, epochs // _WF_EPOCH_DIV)
    all_mae: list[float] = []
    all_dir: list[float] = []

    for fold in range(_WF_FOLDS):
        train_end = chunk * (fold + 1)
        val_end   = train_end + chunk

        X_tr  = X[:train_end].astype(np.float32)
        y_tr  = y[:train_end].astype(np.float32)
        X_val = X[train_end:val_end].astype(np.float32)
        y_val = y[train_end:val_end].astype(np.float32)

        if len(X_tr) < 30 or len(X_val) < 5:
            continue

        model, _ = _train_fold(X_tr, y_tr, X_val, y_val,
                                arch, fold_epochs, patience=3)

        model.eval()
        with torch.no_grad():
            preds = model(
                torch.from_numpy(X_val).to(DEVICE)
            ).cpu().numpy()

        mae = float(np.mean(np.abs(preds - y_val)))

        # Directional accuracy: did we predict up/down vs previous bar correctly?
        if len(y_val) > 1:
            actual_dir = np.sign(np.diff(y_val))
            pred_dir   = np.sign(preds[1:] - y_val[:-1])
            dir_acc    = float(np.mean(actual_dir == pred_dir))
        else:
            dir_acc = 0.5

        all_mae.append(mae)
        all_dir.append(dir_acc)

    avg_mae = float(np.mean(all_mae)) if all_mae else 0.05
    avg_dir = float(np.mean(all_dir)) if all_dir else 0.5
    combined = 0.4 * avg_mae + 0.6 * (1.0 - avg_dir)

    return {"mae": avg_mae, "dir_acc": avg_dir, "combined": combined}


# -------------------- training --------------------

def _existing_wf_combined(symbol: str, horizon: Horizon) -> float | None:
    """
    Return the walk-forward combined score stored in the saved model's meta JSON,
    or None if the file doesn't exist or was trained before walk-forward was added.
    Used by train() to decide whether to overwrite a saved model.
    """
    meta_path = _meta_path(symbol, horizon)
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        arch_search = meta.get("arch_search", [])
        if arch_search:
            # Best (lowest) combined score from the saved arch search leaderboard
            return min(r.get("combined", float("inf")) for r in arch_search)
    except Exception:
        pass
    return None   # old model without walk-forward data → always upgrade


def train(symbol: str, horizon: Horizon = "swing",
          epochs: int | None = None) -> dict:
    """
    Train (or retrain) the LSTM for one symbol & horizon.

    Steps:
      1. Fetch historical data and compute features.
      2. Walk-forward search across _ARCH_CANDIDATES — pick the architecture
         with the best combined MAE + directional-accuracy score.
      3. Compare against the currently saved model's walk-forward score.
         If the new model is NOT better, keep the old one and return early.
      4. Save weights (.pt) + metadata (.json) only when the new model wins.

    Returns the metadata dict (same as what goes into the .json file).
    The dict includes "retrain_skipped": True if the old model was kept.
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

    # ---- Walk-forward architecture search ----
    best_arch = _ARCH_CANDIDATES[1]   # medium as safe fallback
    best_combined = float("inf")
    arch_results: list[dict] = []

    for arch in _ARCH_CANDIDATES:
        score = _walk_forward_score(X, y, arch, epochs)
        arch_results.append({**arch, **score})
        if score["combined"] < best_combined:
            best_combined = score["combined"]
            best_arch = arch

    best_wf = next(
        (r for r in arch_results if r["hidden1"] == best_arch["hidden1"]
         and r["dropout"] == best_arch["dropout"]), {}
    )
    best_dir_acc = float(best_wf.get("dir_acc", 0.5))

    # ---- Compare against existing saved model ----
    # Only overwrite the saved .pt if the new model is provably better.
    # "Better" = lower walk-forward combined score (MAE + directional error).
    # Old models trained before walk-forward was added have no score → always
    # upgrade them (None means no baseline to defend).
    old_combined = _existing_wf_combined(symbol, horizon)
    if old_combined is not None and best_combined >= old_combined:
        # New model did not beat the saved one — keep the existing file.
        with open(_meta_path(symbol, horizon)) as f:
            kept_meta = json.load(f)
        kept_meta["retrain_skipped"] = True
        kept_meta["new_combined"]    = round(best_combined, 6)
        kept_meta["old_combined"]    = round(old_combined, 6)
        return kept_meta

    # ---- Final training on full 85% split with winning architecture ----
    split = int(len(X) * 0.85)
    X_train = X[:split].astype(np.float32)
    y_train = y[:split].astype(np.float32)
    X_val   = X[split:].astype(np.float32)
    y_val   = y[split:].astype(np.float32)

    model, best_val = _train_fold(
        X_train, y_train, X_val, y_val,
        best_arch, epochs, patience=4,
    )

    # ---- Persist model weights + meta (new model won) ----
    torch.save(
        {"state_dict": model.state_dict(),
         "n_features":  X.shape[2],
         "arch":        best_arch},
        _model_path(symbol, horizon),
    )

    meta = {
        "symbol":           symbol,
        "horizon":          horizon,
        "feature_columns":  FEATURE_COLUMNS,
        "target_column":    TARGET_COLUMN,
        "lookback":         lookback,
        "scaler_min":       scaler.data_min_.tolist(),
        "scaler_max":       scaler.data_max_.tolist(),
        "val_loss":         float(best_val if best_val != float("inf") else 0.99),
        "train_rows":       int(len(X_train)),
        "arch":             best_arch,
        "wf_dir_acc":       best_dir_acc,     # directional accuracy from search
        "arch_search":      arch_results,     # full leaderboard for inspection
    }
    with open(_meta_path(symbol, horizon), "w") as f:
        json.dump(meta, f, indent=2)

    return meta


# -------------------- prediction --------------------

def _load(symbol: str, horizon: Horizon):
    """Load saved model + meta, training first if missing."""
    mpath    = _model_path(symbol, horizon)
    metapath = _meta_path(symbol, horizon)
    if not (os.path.exists(mpath) and os.path.exists(metapath)):
        train(symbol, horizon)

    with open(metapath) as f:
        meta = json.load(f)

    checkpoint = torch.load(mpath, map_location=DEVICE, weights_only=False)
    # Backward-compatible: old checkpoints saved without "arch" key
    arch = checkpoint.get("arch", {"hidden1": 64, "hidden2": 32, "dropout": 0.2})

    model = LSTMRegressor(
        n_features=checkpoint["n_features"],
        hidden1=arch["hidden1"],
        hidden2=arch["hidden2"],
        dropout=arch["dropout"],
    ).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, meta


def predict(symbol: str, horizon: Horizon = "swing") -> Prediction:
    """
    Run a multi-step forecast for `symbol`.
    Trains the model on the fly if no saved one exists.
    """
    model, meta = _load(symbol, horizon)
    lookback     = meta["lookback"]
    feature_cols = meta["feature_columns"]
    target_idx   = feature_cols.index(meta["target_column"])

    raw = get_daily_history(symbol, "1y") if horizon == "swing" \
        else get_hourly_history(symbol, "60d")
    df = add_indicators(raw)
    if len(df) < lookback:
        raise RuntimeError(
            f"Not enough recent data for {symbol} to predict.")

    feats      = df[feature_cols].values
    data_min   = np.array(meta["scaler_min"])
    data_max   = np.array(meta["scaler_max"])
    data_range = data_max - data_min + 1e-9
    scaled     = (feats - data_min) / data_range

    window = scaled[-lookback:, :].copy().astype(np.float32)
    steps  = config.SWING_HORIZON_DAYS if horizon == "swing" \
        else config.INTRADAY_HORIZON_BARS

    path_scaled: list[float] = []
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

    current_price   = float(df[meta["target_column"]].iloc[-1])
    predicted_price = path_unscaled[-1]
    expected_return = (predicted_price - current_price) / current_price * 100.0

    # Confidence = walk-forward directional accuracy (0..1).
    # Old models that don't have wf_dir_acc fall back to a val_loss proxy.
    if "wf_dir_acc" in meta:
        confidence = float(meta["wf_dir_acc"])
    else:
        val_loss   = float(meta.get("val_loss", 0.01))
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
