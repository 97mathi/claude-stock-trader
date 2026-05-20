"""
Paper-trading portfolio backed by SQLite.

What's stored:
  - wallet      : single row with current cash balance
  - positions   : one row per currently-held stock (symbol, qty, buy price,
                  prediction_at_buy, predicted_target, stop_loss, take_profit)
  - trades      : append-only log of every BUY / SELL

This same interface (`buy`, `sell`, `get_positions`, etc.) is what a real
broker adapter would implement later — see broker_interface.py.
"""

from __future__ import annotations
import os
import sqlite3
import datetime as dt
from dataclasses import dataclass

import config


# -------------------- data classes --------------------

@dataclass
class Position:
    symbol: str
    quantity: int
    buy_price: float
    buy_date: str
    predicted_target: float    # what the LSTM said when we bought
    expected_return_pct: float
    stop_loss: float
    take_profit: float
    horizon: str               # "swing" or "intraday"

    @property
    def invested(self) -> float:
        return self.quantity * self.buy_price


@dataclass
class Trade:
    id: int
    timestamp: str
    side: str           # "BUY" or "SELL"
    symbol: str
    quantity: int
    price: float
    reason: str
    pnl: float          # only meaningful on SELL


# -------------------- portfolio --------------------

class Portfolio:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.DB_PATH
        self._init_db()

    # ----- internal -----

    def _conn(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self):
        fresh = not os.path.exists(self.db_path)
        with self._conn() as c:
            c.executescript("""
                CREATE TABLE IF NOT EXISTS wallet (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    cash REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity INTEGER NOT NULL,
                    buy_price REAL NOT NULL,
                    buy_date TEXT NOT NULL,
                    predicted_target REAL NOT NULL,
                    expected_return_pct REAL NOT NULL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    horizon TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    side TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    reason TEXT NOT NULL,
                    pnl REAL NOT NULL DEFAULT 0
                );
            """)
            if fresh:
                c.execute("INSERT INTO wallet (id, cash) VALUES (1, ?)",
                          (config.INITIAL_PAPER_CASH,))
            c.commit()

    # ----- wallet -----

    def get_cash(self) -> float:
        with self._conn() as c:
            row = c.execute("SELECT cash FROM wallet WHERE id = 1").fetchone()
            return float(row[0]) if row else 0.0

    def set_cash(self, amount: float):
        with self._conn() as c:
            c.execute("UPDATE wallet SET cash = ? WHERE id = 1", (amount,))
            c.commit()

    def reset(self, cash: float | None = None):
        """Wipe history and start over."""
        cash = cash if cash is not None else config.INITIAL_PAPER_CASH
        with self._conn() as c:
            c.executescript(
                "DELETE FROM trades; DELETE FROM positions;")
            c.execute("UPDATE wallet SET cash = ? WHERE id = 1", (cash,))
            c.commit()

    def clear_trades(self):
        """Delete all trade records. Positions and wallet are untouched."""
        with self._conn() as c:
            c.execute("DELETE FROM trades")
            c.commit()

    def set_total_capital(self, amount: float):
        """
        Set the TOTAL capital available to invest. This wipes positions and
        trade history and restarts the wallet with `amount` as cash.
        Use when you want to (re)fund the paper account with a chosen amount.
        """
        if amount <= 0:
            raise ValueError("Capital must be positive.")
        config.TOTAL_INVESTMENT_AMOUNT = amount
        self.reset(cash=amount)

    # ----- positions -----

    def get_positions(self) -> list[Position]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT symbol, quantity, buy_price, buy_date,
                       predicted_target, expected_return_pct,
                       stop_loss, take_profit, horizon
                FROM positions
            """).fetchall()
        return [Position(*r) for r in rows]

    def get_position(self, symbol: str) -> Position | None:
        for p in self.get_positions():
            if p.symbol == symbol:
                return p
        return None

    # ----- trades -----

    def invested_total(self) -> float:
        """Sum of (buy_price * qty) across open positions."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(buy_price * quantity), 0) FROM positions"
            ).fetchone()
        return float(row[0]) if row else 0.0

    def buy(self, symbol: str, quantity: int, price: float,
            predicted_target: float, expected_return_pct: float,
            horizon: str, reason: str = "Model signal",
            equity_at_buy: float | None = None):
        cost = quantity * price
        cash = self.get_cash()
        if cost > cash:
            raise ValueError(
                f"Not enough cash. Need Rs.{cost:,.2f}, have Rs.{cash:,.2f}.")

        # ---- 75% invested cap ----
        # Equity (cash + market value of holdings). We accept an optional
        # `equity_at_buy` from the caller (already-computed mark-to-market).
        # If not provided, fall back to (cash + invested-at-cost) as an
        # approximation.
        current_invested = self.invested_total()
        equity = equity_at_buy if equity_at_buy is not None \
            else (cash + current_invested)
        invested_cap = equity * config.MAX_INVESTED_PCT
        if current_invested + cost > invested_cap + 1e-6:
            raise ValueError(
                f"Buy would push invested to "
                f"Rs.{current_invested + cost:,.2f} "
                f"(>{config.MAX_INVESTED_PCT*100:.0f}% of equity "
                f"Rs.{equity:,.2f}). Reserve must be kept.")

        existing = self.get_position(symbol)
        if existing:
            raise ValueError(
                f"Already holding {symbol}. Sell first or pick another stock.")

        # Stop-loss = lose-limit. Take-profit = MAX-profit ceiling (horizon-aware).
        # The MIN-profit floor is computed at evaluation time in monitor.py,
        # not stored — so changing config later affects open positions too.
        _, max_pct = config.profit_goals_for(horizon)
        stop_loss = price * (1 - config.STOP_LOSS_PCT)
        take_profit = price * (1 + max_pct)
        now = dt.datetime.now().isoformat(timespec="seconds")

        with self._conn() as c:
            c.execute("UPDATE wallet SET cash = cash - ? WHERE id = 1",
                      (cost,))
            c.execute("""
                INSERT INTO positions
                (symbol, quantity, buy_price, buy_date,
                 predicted_target, expected_return_pct,
                 stop_loss, take_profit, horizon)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (symbol, quantity, price, now,
                  predicted_target, expected_return_pct,
                  stop_loss, take_profit, horizon))
            c.execute("""
                INSERT INTO trades
                (timestamp, side, symbol, quantity, price, reason, pnl)
                VALUES (?, 'BUY', ?, ?, ?, ?, 0)
            """, (now, symbol, quantity, price, reason))
            c.commit()

    def sell(self, symbol: str, price: float, reason: str = "Manual sell"):
        pos = self.get_position(symbol)
        if not pos:
            raise ValueError(f"No open position for {symbol}.")

        proceeds = pos.quantity * price
        pnl = (price - pos.buy_price) * pos.quantity
        now = dt.datetime.now().isoformat(timespec="seconds")

        with self._conn() as c:
            c.execute("UPDATE wallet SET cash = cash + ? WHERE id = 1",
                      (proceeds,))
            c.execute("DELETE FROM positions WHERE symbol = ?", (symbol,))
            c.execute("""
                INSERT INTO trades
                (timestamp, side, symbol, quantity, price, reason, pnl)
                VALUES (?, 'SELL', ?, ?, ?, ?, ?)
            """, (now, symbol, pos.quantity, price, reason, pnl))
            c.commit()
        return pnl

    def realized_pnl_since(self, since_iso: str) -> float:
        """Sum of P&L on SELL trades closed at-or-after `since_iso` (ISO date)."""
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades "
                "WHERE side='SELL' AND timestamp >= ?",
                (since_iso,)
            ).fetchone()
        return float(row[0]) if row else 0.0

    def period_goal_status(self, horizon: str,
                           live_prices: dict[str, float] | None = None) -> dict:
        """
        Has the profit goal for the current period been reached?

        Period = today (intraday) or this calendar week (swing).
        Goal amount = TOTAL_INVESTMENT_AMOUNT * GOAL_TARGET_PCT_OF_CAPITAL.
        Progress = realized P&L this period + unrealized P&L on open positions
                   of this horizon.
        Returns a dict the agent and GUI both use.
        """
        import datetime as dt
        live_prices = live_prices or {}

        if horizon == "intraday":
            since = dt.date.today().isoformat()
            label = "Today"
        else:
            monday = dt.date.today() - dt.timedelta(days=dt.date.today().weekday())
            since = monday.isoformat()
            label = "This week"

        realized = self.realized_pnl_since(since)

        # Unrealized P&L on this horizon's open positions
        unrealized = 0.0
        for p in self.get_positions():
            if p.horizon != horizon:
                continue
            cur = live_prices.get(p.symbol, p.buy_price)
            unrealized += (cur - p.buy_price) * p.quantity

        progress = realized + unrealized
        capital = config.TOTAL_INVESTMENT_AMOUNT
        goal_amount = capital * config.GOAL_TARGET_PCT_OF_CAPITAL
        reached = progress >= goal_amount

        return {
            "horizon": horizon,
            "label": label,
            "since": since,
            "realized": realized,
            "unrealized": unrealized,
            "progress": progress,
            "goal_amount": goal_amount,
            "reached": reached,
            "pct_of_goal": (progress / goal_amount * 100.0) if goal_amount > 0 else 0.0,
        }

    def get_trades(self, limit: int = 100) -> list[Trade]:
        with self._conn() as c:
            rows = c.execute("""
                SELECT id, timestamp, side, symbol, quantity, price, reason, pnl
                FROM trades ORDER BY id DESC LIMIT ?
            """, (limit,)).fetchall()
        return [Trade(*r) for r in rows]

    # ----- summary -----

    def total_value(self, live_prices: dict[str, float]) -> dict:
        """
        Live equity = cash + sum(qty * live price) for each holding.
        Returns a small summary dict for the GUI.
        """
        cash = self.get_cash()
        positions = self.get_positions()
        invested = 0.0
        market_value = 0.0
        unrealized = 0.0
        for p in positions:
            invested += p.invested
            cur = live_prices.get(p.symbol, p.buy_price)
            mv = p.quantity * cur
            market_value += mv
            unrealized += mv - p.invested

        # Realized P&L from trade log
        with self._conn() as c:
            row = c.execute(
                "SELECT COALESCE(SUM(pnl),0) FROM trades WHERE side='SELL'"
            ).fetchone()
        realized = float(row[0]) if row else 0.0

        return {
            "cash": cash,
            "invested": invested,
            "market_value": market_value,
            "equity": cash + market_value,
            "unrealized_pnl": unrealized,
            "realized_pnl": realized,
            "total_pnl": realized + unrealized,
            "n_positions": len(positions),
        }
