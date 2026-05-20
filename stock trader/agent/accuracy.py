"""
Accuracy tracker: logs every prediction, later compares it to what actually
happened, and reports which models are predicting poorly so they can be
retrained.

Flow:
  1. When the agent makes a prediction, call `log_prediction(...)`. We store
     the predicted price and the date it's "due" (when the forecast horizon
     ends).
  2. Later, call `resolve_due(...)` (or it runs automatically each cycle): for
     any prediction whose due date has passed, fetch the actual price and
     record the error.
  3. `models_needing_retrain(...)` returns symbols whose recent average error
     (MAPE - Mean Absolute Percentage Error) is above the threshold.

Stored in a small SQLite DB (config.ACCURACY_DB_PATH).
"""

from __future__ import annotations
import os
import sqlite3
import datetime as dt
from dataclasses import dataclass

import config
from data.fetcher import get_latest_price


@dataclass
class AccuracyStat:
    symbol: str
    horizon: str
    n_samples: int
    mape: float            # mean absolute % error over resolved predictions
    needs_retrain: bool


class AccuracyTracker:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.ACCURACY_DB_PATH
        self._init_db()

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    made_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    horizon TEXT NOT NULL,
                    price_at_pred REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    due_at TEXT NOT NULL,
                    actual_price REAL,
                    abs_pct_error REAL,
                    resolved INTEGER NOT NULL DEFAULT 0
                );
            """)
            c.commit()

    # ----- logging -----

    def log_prediction(self, symbol: str, horizon: str,
                       price_at_pred: float, predicted_price: float):
        now = dt.datetime.now()
        horizon_days = (config.SWING_HORIZON_DAYS if horizon == "swing"
                        else max(1, config.INTRADAY_HORIZON_BARS // 6))
        due = now + dt.timedelta(days=horizon_days)
        with self._conn() as c:
            c.execute("""
                INSERT INTO predictions
                (made_at, symbol, horizon, price_at_pred, predicted_price, due_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (now.isoformat(timespec="seconds"), symbol, horizon,
                  price_at_pred, predicted_price, due.isoformat(timespec="seconds")))
            c.commit()

    # ----- resolving -----

    def resolve_due(self) -> int:
        """
        Fill in actual prices for any predictions whose due date has passed.
        Returns how many were resolved this call.
        """
        now_iso = dt.datetime.now().isoformat(timespec="seconds")
        with self._conn() as c:
            rows = c.execute("""
                SELECT id, symbol, predicted_price, price_at_pred
                FROM predictions
                WHERE resolved = 0 AND due_at <= ?
            """, (now_iso,)).fetchall()

            resolved = 0
            for pid, symbol, predicted, _ in rows:
                actual = get_latest_price(symbol)
                if actual is None or actual <= 0:
                    continue
                err = abs(actual - predicted) / actual * 100.0
                c.execute("""
                    UPDATE predictions
                    SET actual_price = ?, abs_pct_error = ?, resolved = 1
                    WHERE id = ?
                """, (actual, err, pid))
                resolved += 1
            c.commit()
        return resolved

    # ----- reporting -----

    def stats(self) -> list[AccuracyStat]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT symbol, horizon,
                       COUNT(*) AS n,
                       AVG(abs_pct_error) AS mape
                FROM predictions
                WHERE resolved = 1
                GROUP BY symbol, horizon
            """).fetchall()
        out = []
        for symbol, horizon, n, mape in rows:
            mape = float(mape or 0.0)
            needs = (n >= config.RETRAIN_MIN_SAMPLES and
                     mape > config.RETRAIN_ERROR_THRESHOLD_PCT)
            out.append(AccuracyStat(symbol, horizon, int(n), mape, needs))
        return out

    def models_needing_retrain(self) -> list[tuple[str, str]]:
        """Return [(symbol, horizon), ...] whose recent error is too high."""
        return [(s.symbol, s.horizon) for s in self.stats() if s.needs_retrain]

    def overall_mape(self) -> float | None:
        with self._conn() as c:
            row = c.execute(
                "SELECT AVG(abs_pct_error) FROM predictions WHERE resolved=1"
            ).fetchone()
        return float(row[0]) if row and row[0] is not None else None
