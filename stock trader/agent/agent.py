"""
The autonomous trading agent.

One "cycle" does the following, fully automatically:

  1. RESOLVE accuracy: fill in actual prices for past predictions so we know
     which models are drifting.

  2. MANAGE existing positions: run the monitor (stop-loss / max-profit /
     fresh-prediction) and auto-sell whatever it flags. (Skipped only if the
     period goal is reached AND config.FREEZE_ALL_AFTER_GOAL is True.)

  3. CHECK the goal circuit-breaker: if this horizon's period profit goal is
     already reached, STOP - do not open new positions until the period resets.

  4. SCAN the universe: for every Nifty 50 stock not already held, run the LSTM
     prediction and blend it with sentiment / sector / fundamentals / macro /
     correlation into one combined score.

  5. RANK & FILTER: keep candidates above the score threshold, drop ones too
     correlated with current holdings, sort best-first.

  6. BUY: size each pick with the 75% / per-stock rules and execute on the
     paper portfolio, up to AGENT_MAX_NEW_BUYS_PER_CYCLE.

Everything is reported back as an AgentReport for the GUI and ledger.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

import config
from nifty50 import NIFTY_50
from data.fetcher import get_latest_prices, get_latest_price
from model.lstm_model import predict
from signals.aggregator import score_stock, StockScore
from signals.macro import macro_signal
from signals import correlation as corr_mod
from trading.portfolio import Portfolio
from trading.monitor import evaluate_all, auto_execute
from trading.sizer import suggest_size
from agent.accuracy import AccuracyTracker


ProgressCb = Callable[[str, int, int], None]


@dataclass
class AgentReport:
    horizon: str
    goal_reached: bool
    goal_status: dict
    sells: list[str] = field(default_factory=list)
    buys: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    ranked: list[StockScore] = field(default_factory=list)
    resolved_predictions: int = 0
    messages: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"[{self.horizon}] goal_reached={self.goal_reached} | "
                f"{len(self.buys)} buys, {len(self.sells)} sells, "
                f"{len(self.ranked)} scored")


class TradingAgent:
    def __init__(self, portfolio: Portfolio | None = None,
                 universe: list[str] | None = None):
        self.portfolio = portfolio or Portfolio()
        self.universe = universe or NIFTY_50
        self.tracker = AccuracyTracker()

    # -----------------------------------------------------------------
    def run_cycle(self, horizon: str | None = None,
                  progress: ProgressCb | None = None,
                  mode: str = "paper") -> AgentReport:
        horizon = horizon or config.AGENT_DEFAULT_HORIZON
        report = AgentReport(horizon=horizon, goal_reached=False, goal_status={})

        if mode == "real":
            report.messages.append(
                "Real mode not implemented - agent runs in paper mode only.")

        # ---- 1) resolve accuracy ----
        if progress:
            progress("Resolving past predictions", 0, 1)
        try:
            report.resolved_predictions = self.tracker.resolve_due()
        except Exception as e:
            report.messages.append(f"Accuracy resolve failed: {e}")

        # ---- live prices for held positions ----
        positions = self.portfolio.get_positions()
        held = [p.symbol for p in positions]
        live_held = get_latest_prices(held) if held else {}
        for p in positions:
            if live_held.get(p.symbol) is None:
                live_held[p.symbol] = p.buy_price

        # ---- goal circuit-breaker status ----
        goal = self.portfolio.period_goal_status(horizon, live_held)
        report.goal_status = goal
        report.goal_reached = goal["reached"]

        # ---- 2) manage existing positions (sells) ----
        freeze_sells = goal["reached"] and config.FREEZE_ALL_AFTER_GOAL
        if not freeze_sells and positions:
            if progress:
                progress("Checking open positions", 0, 1)
            try:
                decisions = evaluate_all(self.portfolio, do_fresh_prediction=True)
                report.sells = auto_execute(self.portfolio, decisions)
            except Exception as e:
                report.messages.append(f"Position management failed: {e}")

        # ---- 3) circuit-breaker: stop new buys if goal reached ----
        if goal["reached"]:
            report.messages.append(
                f"{goal['label']} goal reached "
                f"(Rs.{goal['progress']:+,.0f} / Rs.{goal['goal_amount']:,.0f}). "
                "No new positions until the period resets.")
            return report

        # ---- 4) scan universe & score ----
        corr_mod.clear_cache()           # fresh correlation data each cycle
        macro = macro_signal()           # compute once per cycle
        held_now = [p.symbol for p in self.portfolio.get_positions()]
        candidates = [s for s in self.universe if s not in held_now]

        ranked: list[StockScore] = []
        total = len(candidates)
        for i, symbol in enumerate(candidates, start=1):
            if progress:
                progress(f"Scoring {symbol}", i, total)
            try:
                pred = predict(symbol, horizon)   # type: ignore
                # log prediction for future accuracy scoring
                try:
                    self.tracker.log_prediction(
                        symbol, horizon, pred.current_price, pred.predicted_price)
                except Exception:
                    pass
                sc = score_stock(symbol, pred.expected_return_pct,
                                 held_now, macro=macro)
                # stash the price so we don't refetch at buy time
                sc.signals["lstm"].note += f" @Rs.{pred.current_price:.2f}"
                sc._current_price = pred.current_price  # type: ignore
                ranked.append(sc)
            except Exception as e:
                report.skipped.append(f"{symbol}: {e}")

        ranked.sort(key=lambda s: s.combined, reverse=True)
        report.ranked = ranked

        # ---- 5) filter ----
        picks = []
        for sc in ranked:
            if sc.combined < config.AGENT_MIN_COMBINED_SCORE:
                continue
            if sc.max_abs_corr > config.AGENT_MAX_CORRELATION:
                report.skipped.append(
                    f"{sc.symbol}: too correlated ({sc.max_abs_corr:.2f}) "
                    f"with {sc.most_correlated_with}")
                continue
            picks.append(sc)
            if len(picks) >= config.AGENT_MAX_NEW_BUYS_PER_CYCLE:
                break

        # ---- 6) buy ----
        for sc in picks:
            price = getattr(sc, "_current_price", None) or get_latest_price(sc.symbol)
            if not price or price <= 0:
                report.skipped.append(f"{sc.symbol}: no price")
                continue

            # mark-to-market equity for sizing + 75% cap
            pos = self.portfolio.get_positions()
            lp = {p.symbol: live_held.get(p.symbol, p.buy_price) for p in pos}
            summary = self.portfolio.total_value(lp)

            sizing = suggest_size(
                equity=summary["equity"],
                current_invested=summary["invested"],
                cash=summary["cash"],
                current_price=price,
                expected_return_pct=sc.lstm_edge_pct,
            )
            if not sizing.ok or sizing.suggested_qty <= 0:
                report.skipped.append(f"{sc.symbol}: sizer skip ({sizing.notes[-1]})")
                continue

            try:
                self.portfolio.buy(
                    symbol=sc.symbol,
                    quantity=sizing.suggested_qty,
                    price=price,
                    predicted_target=price * (1 + sc.lstm_edge_pct / 100.0),
                    expected_return_pct=sc.lstm_edge_pct,
                    horizon=horizon,
                    reason=f"Agent score {sc.combined:+.2f} | {sc.explain()}",
                    equity_at_buy=summary["equity"],
                )
                report.buys.append(
                    f"BOUGHT {sizing.suggested_qty} x {sc.symbol} @ Rs.{price:.2f} "
                    f"(score {sc.combined:+.2f}, edge {sc.lstm_edge_pct:+.2f}%)")
            except Exception as e:
                report.skipped.append(f"{sc.symbol}: buy failed ({e})")

        return report
