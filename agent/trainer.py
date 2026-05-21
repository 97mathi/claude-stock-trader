"""
Training manager.

Responsibilities:
  - Bulk-train LSTM models for the whole Nifty 50 universe at first startup.
  - Skip stocks that already have a saved model (so startup is fast after the
    first run - models persist on disk under model/trained/).
  - Retrain models flagged by the accuracy tracker as predicting poorly.

All long-running work reports progress through an optional callback so the GUI
can show a progress bar without freezing.
"""

from __future__ import annotations
import os
from typing import Callable

import config
from nifty50 import NIFTY_50
from model.lstm_model import train, _model_path, _meta_path
from agent.accuracy import AccuracyTracker


ProgressCb = Callable[[str, int, int], None]   # (message, done, total)


def _model_exists(symbol: str, horizon: str) -> bool:
    return (os.path.exists(_model_path(symbol, horizon)) and
            os.path.exists(_meta_path(symbol, horizon)))


def untrained_symbols(horizon: str,
                      universe: list[str] | None = None) -> list[str]:
    universe = universe or NIFTY_50
    return [s for s in universe if not _model_exists(s, horizon)]


def train_universe(horizon: str | None = None,
                   universe: list[str] | None = None,
                   only_missing: bool = True,
                   progress: ProgressCb | None = None) -> dict:
    """
    Train models for the whole universe for the given horizon.

    only_missing=True  -> skip stocks that already have a saved model.
    Returns a summary dict: {trained: [...], skipped: [...], failed: [...]}.
    """
    horizon = horizon or config.AGENT_DEFAULT_HORIZON
    universe = universe or NIFTY_50

    todo = untrained_symbols(horizon, universe) if only_missing else list(universe)
    skipped = [s for s in universe if s not in todo]

    trained, failed = [], []
    total = len(todo)
    for i, symbol in enumerate(todo, start=1):
        if progress:
            progress(f"Training {symbol} ({horizon})", i - 1, total)
        try:
            train(symbol, horizon)   # type: ignore
            trained.append(symbol)
        except Exception as e:
            failed.append((symbol, str(e)))
        if progress:
            progress(f"Done {symbol}", i, total)

    return {"trained": trained, "skipped": skipped, "failed": failed,
            "horizon": horizon}


def retrain_stale(progress: ProgressCb | None = None) -> dict:
    """
    Retrain only the models the accuracy tracker says are predicting poorly.
    """
    tracker = AccuracyTracker()
    tracker.resolve_due()
    stale = tracker.models_needing_retrain()

    trained, kept, failed = [], [], []
    total = len(stale)
    for i, (symbol, horizon) in enumerate(stale, start=1):
        if progress:
            progress(f"Retraining {symbol} ({horizon})", i - 1, total)
        try:
            # force_save=True: old model already proven bad by MAPE tracker,
            # always overwrite with the new walk-forward-selected model.
            meta = train(symbol, horizon, force_save=True)   # type: ignore
            trained.append((symbol, horizon))
        except Exception as e:
            failed.append((symbol, str(e)))
        if progress:
            progress(f"Done {symbol}", i, total)

    return {"retrained": trained, "kept": kept, "failed": failed, "checked": total}
