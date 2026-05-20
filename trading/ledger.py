"""
Ledger & performance report.

Reads the trade log from the Portfolio's SQLite DB and produces:
  - A detailed, chronological ledger (every BUY/SELL with reason and P&L).
  - Aggregate performance stats: win rate, profit factor, average win/loss,
    best/worst trade, totals - overall and broken down by horizon and symbol.
  - CSV export of the full ledger.
  - A plain-text / markdown report you can save or print.

Nothing here changes state; it's read-only reporting on top of portfolio.db.
"""

from __future__ import annotations
import csv
import sqlite3
import datetime as dt
from dataclasses import dataclass, field

import config


@dataclass
class TradeRow:
    id: int
    timestamp: str
    side: str
    symbol: str
    quantity: int
    price: float
    reason: str
    pnl: float


@dataclass
class PerfStats:
    n_trades: int = 0
    n_buys: int = 0
    n_sells: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0       # sum of positive P&L
    gross_loss: float = 0.0         # sum of negative P&L (negative number)
    net_pnl: float = 0.0
    profit_factor: float = 0.0      # gross_profit / abs(gross_loss)
    avg_win: float = 0.0
    avg_loss: float = 0.0
    best: float = 0.0
    worst: float = 0.0
    by_symbol: dict = field(default_factory=dict)   # symbol -> net pnl
    by_horizon_note: str = ""


class Ledger:
    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or config.DB_PATH

    def _rows(self) -> list[TradeRow]:
        with sqlite3.connect(self.db_path) as c:
            try:
                data = c.execute("""
                    SELECT id, timestamp, side, symbol, quantity, price, reason, pnl
                    FROM trades ORDER BY id ASC
                """).fetchall()
            except sqlite3.OperationalError:
                return []
        return [TradeRow(*r) for r in data]

    # ----- detailed ledger -----

    def all_trades(self) -> list[TradeRow]:
        return self._rows()

    # ----- performance -----

    def stats(self) -> PerfStats:
        rows = self._rows()
        s = PerfStats()
        s.n_trades = len(rows)
        sells = [r for r in rows if r.side == "SELL"]
        s.n_buys = sum(1 for r in rows if r.side == "BUY")
        s.n_sells = len(sells)

        wins = [r.pnl for r in sells if r.pnl > 0]
        losses = [r.pnl for r in sells if r.pnl < 0]
        s.wins = len(wins)
        s.losses = len(losses)
        s.gross_profit = sum(wins)
        s.gross_loss = sum(losses)
        s.net_pnl = s.gross_profit + s.gross_loss
        s.win_rate = (s.wins / s.n_sells * 100.0) if s.n_sells else 0.0
        s.profit_factor = (s.gross_profit / abs(s.gross_loss)
                           if s.gross_loss != 0 else
                           (float("inf") if s.gross_profit > 0 else 0.0))
        s.avg_win = (s.gross_profit / s.wins) if s.wins else 0.0
        s.avg_loss = (s.gross_loss / s.losses) if s.losses else 0.0
        s.best = max([r.pnl for r in sells], default=0.0)
        s.worst = min([r.pnl for r in sells], default=0.0)

        by_symbol: dict[str, float] = {}
        for r in sells:
            by_symbol[r.symbol] = by_symbol.get(r.symbol, 0.0) + r.pnl
        s.by_symbol = dict(sorted(by_symbol.items(),
                                  key=lambda kv: kv[1], reverse=True))
        return s

    # ----- exports -----

    def export_csv(self, path: str) -> str:
        rows = self._rows()
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "timestamp", "side", "symbol", "quantity",
                        "price", "pnl", "reason"])
            for r in rows:
                w.writerow([r.id, r.timestamp, r.side, r.symbol, r.quantity,
                            f"{r.price:.2f}", f"{r.pnl:.2f}", r.reason])
        return path

    def report_markdown(self) -> str:
        s = self.stats()
        rows = self._rows()
        now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

        pf = ("inf" if s.profit_factor == float("inf")
              else f"{s.profit_factor:.2f}")

        lines = []
        lines.append(f"# Trading Report")
        lines.append(f"_Generated {now}_\n")
        lines.append(f"Total capital set: Rs. {config.TOTAL_INVESTMENT_AMOUNT:,.2f}\n")

        lines.append("## Summary")
        lines.append("")
        lines.append(f"- Trades logged: {s.n_trades} ({s.n_buys} buys, {s.n_sells} sells)")
        lines.append(f"- Closed trades: {s.n_sells}  |  Wins: {s.wins}  |  Losses: {s.losses}")
        lines.append(f"- Win rate: {s.win_rate:.1f}%")
        lines.append(f"- Net P&L: Rs. {s.net_pnl:+,.2f}")
        lines.append(f"- Gross profit: Rs. {s.gross_profit:+,.2f}  |  Gross loss: Rs. {s.gross_loss:+,.2f}")
        lines.append(f"- Profit factor: {pf}")
        lines.append(f"- Average win: Rs. {s.avg_win:+,.2f}  |  Average loss: Rs. {s.avg_loss:+,.2f}")
        lines.append(f"- Best trade: Rs. {s.best:+,.2f}  |  Worst trade: Rs. {s.worst:+,.2f}")
        lines.append("")

        if s.by_symbol:
            lines.append("## Net P&L by symbol")
            lines.append("")
            lines.append("| Symbol | Net P&L (Rs.) |")
            lines.append("|--------|---------------|")
            for sym, pnl in s.by_symbol.items():
                lines.append(f"| {sym} | {pnl:+,.2f} |")
            lines.append("")

        lines.append("## Full ledger")
        lines.append("")
        lines.append("| # | Time | Side | Symbol | Qty | Price | P&L | Reason |")
        lines.append("|---|------|------|--------|-----|-------|-----|--------|")
        for r in rows:
            lines.append(
                f"| {r.id} | {r.timestamp} | {r.side} | {r.symbol} | "
                f"{r.quantity} | {r.price:.2f} | {r.pnl:+.2f} | {r.reason} |")
        lines.append("")
        return "\n".join(lines)

    def export_report(self, path: str) -> str:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.report_markdown())
        return path
