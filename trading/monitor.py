"""
Position monitor / decision engine.

For each open position it answers ONE question: HOLD or SELL right now?

Decision ladder (first match wins):
  1. Trailing stop hit (price <= trailing_stop)          -> SELL  limit losses
  2. Stagnation: half horizon passed, barely moved       -> SELL  kill dead money
  3. Below breakeven (not stagnant)                      -> HOLD  wait for recovery
  4. MAX profit hit (P&L% >= MAX for horizon)            -> SELL  greed cap
  5. MIN profit reached (P&L in [MIN, MAX))              -> fresh LSTM prediction:
       - Forecast positive                               -> HOLD  let it run
       - Forecast flat/negative                          -> SELL  lock the gain
  6. Below MIN profit:
       a. Fresh forecast turned bearish                  -> SELL
       b. Past full horizon, below original target       -> SELL  thesis expired
       c. Otherwise                                      -> HOLD

evaluate_all() also updates the trailing stop in the DB whenever the live
price exceeds the stored high-water mark — so the stop always follows up.
"""

from __future__ import annotations
import datetime as dt
from dataclasses import dataclass

import config
from data.fetcher import get_latest_price
from trading.portfolio import Portfolio, Position
from model.lstm_model import predict, Prediction


@dataclass
class Decision:
    symbol: str
    action: str               # "HOLD" or "SELL"
    reason: str
    live_price: float
    buy_price: float
    pnl_pct: float
    min_target_price: float
    max_target_price: float
    horizon: str
    fresh_prediction: Prediction | None
    trailing_stop: float = 0.0   # current effective stop level (for GUI display)


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _days_since(iso_ts: str) -> float:
    """Calendar days since a buy date ISO string."""
    try:
        d = dt.datetime.fromisoformat(iso_ts)
        return (dt.datetime.now() - d).total_seconds() / 86_400.0
    except Exception:
        return 0.0


def _hours_since(iso_ts: str) -> float:
    """Hours since a buy date ISO string (for intraday precision)."""
    try:
        d = dt.datetime.fromisoformat(iso_ts)
        return (dt.datetime.now() - d).total_seconds() / 3_600.0
    except Exception:
        return 0.0


def _horizon_half_elapsed(position: Position) -> bool:
    """True when half of the planned trade duration has passed."""
    if position.horizon == "intraday":
        # Intraday horizon = INTRADAY_HORIZON_BARS hours (default 4)
        half_hours = config.INTRADAY_HORIZON_BARS / 2.0
        return _hours_since(position.buy_date) >= half_hours
    else:
        # Swing horizon = SWING_HORIZON_DAYS trading days
        half_days = config.SWING_HORIZON_DAYS * config.STAGNATION_DAYS_RATIO
        return _days_since(position.buy_date) >= half_days


def _past_full_horizon(position: Position) -> bool:
    """True when the full planned duration has elapsed."""
    if position.horizon == "intraday":
        return _hours_since(position.buy_date) >= config.INTRADAY_HORIZON_BARS
    else:
        return _days_since(position.buy_date) >= config.SWING_HORIZON_DAYS


# ------------------------------------------------------------------ #
#  Core decision engine                                                #
# ------------------------------------------------------------------ #

def evaluate(position: Position,
             live_price: float | None = None,
             do_fresh_prediction: bool = True) -> Decision:
    """
    Decide whether to hold or sell ONE position.
    `do_fresh_prediction=False` skips the LSTM call (faster bulk refresh
    when you only need the cheap risk checks).
    """
    min_pct, max_pct = config.profit_goals_for(position.horizon)
    min_target  = position.buy_price * (1 + min_pct)
    max_target  = position.buy_price * (1 + max_pct)
    trail_stop  = position.stop_loss   # already updated by evaluate_all()

    price = live_price if live_price is not None \
        else get_latest_price(position.symbol)

    if price is None or price <= 0:
        return Decision(
            symbol=position.symbol, action="HOLD",
            reason="Live price unavailable — skipping check",
            live_price=0.0, buy_price=position.buy_price,
            pnl_pct=0.0,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
            trailing_stop=trail_stop,
        )

    pnl_pct   = (price - position.buy_price) / position.buy_price * 100.0
    pnl_ratio = pnl_pct / 100.0

    # ── 1) Trailing stop-loss ─────────────────────────────────────────
    if price <= trail_stop:
        return Decision(
            symbol=position.symbol, action="SELL",
            reason=(f"Trailing stop hit — price Rs.{price:.2f} <= "
                    f"stop Rs.{trail_stop:.2f} "
                    f"(high Rs.{position.highest_price_seen:.2f} − "
                    f"{config.TRAILING_STOP_PCT*100:.0f}%)"),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
            trailing_stop=trail_stop,
        )

    # ── 2) Stagnation — dead money ────────────────────────────────────
    # Half the planned duration has passed and the stock hasn't moved
    # meaningfully in either direction. Exit and redeploy capital.
    if (_horizon_half_elapsed(position)
            and abs(pnl_pct) < config.STAGNATION_MIN_MOVE_PCT):
        half_label = (
            f"{config.INTRADAY_HORIZON_BARS / 2:.0f}h"
            if position.horizon == "intraday"
            else f"{config.SWING_HORIZON_DAYS * config.STAGNATION_DAYS_RATIO:.1f}d"
        )
        return Decision(
            symbol=position.symbol, action="SELL",
            reason=(f"Stagnation — {half_label} elapsed, moved only "
                    f"{pnl_pct:+.2f}% "
                    f"(threshold ±{config.STAGNATION_MIN_MOVE_PCT:.1f}%). "
                    "Dead money — redeploying capital."),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
            trailing_stop=trail_stop,
        )

    # ── 3) Below breakeven — hold until recovery ──────────────────────
    if pnl_pct < 0:
        return Decision(
            symbol=position.symbol, action="HOLD",
            reason=(f"Below breakeven ({pnl_pct:+.2f}%) — holding until "
                    f"min target Rs.{min_target:.2f}"),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
            trailing_stop=trail_stop,
        )

    # ── 4) Max profit reached — greed cap ────────────────────────────
    if pnl_ratio >= max_pct:
        return Decision(
            symbol=position.symbol, action="SELL",
            reason=(f"Max-profit cap hit: {pnl_pct:+.2f}% >= "
                    f"{max_pct*100:.1f}% "
                    f"({config.goal_period_label(position.horizon).lower()} ceiling)"),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
            trailing_stop=trail_stop,
        )

    # ── 5) In the profit zone [MIN, MAX) — fresh prediction decides ───
    if pnl_ratio >= min_pct:
        fresh = None
        if do_fresh_prediction:
            try:
                fresh = predict(position.symbol, position.horizon)  # type: ignore
            except Exception:
                fresh = None

        if fresh is None:
            return Decision(
                symbol=position.symbol, action="SELL",
                reason=(f"In profit zone ({pnl_pct:+.2f}%) but fresh forecast "
                        "unavailable — locking gain"),
                live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
                min_target_price=min_target, max_target_price=max_target,
                horizon=position.horizon, fresh_prediction=None,
                trailing_stop=trail_stop,
            )

        if fresh.expected_return_pct > 0:
            return Decision(
                symbol=position.symbol, action="HOLD",
                reason=(f"In profit zone ({pnl_pct:+.2f}%), fresh forecast "
                        f"{fresh.expected_return_pct:+.2f}% — let it run "
                        f"toward Rs.{max_target:.2f}"),
                live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
                min_target_price=min_target, max_target_price=max_target,
                horizon=position.horizon, fresh_prediction=fresh,
                trailing_stop=trail_stop,
            )
        else:
            return Decision(
                symbol=position.symbol, action="SELL",
                reason=(f"Profit locked at {pnl_pct:+.2f}% — fresh forecast "
                        f"{fresh.expected_return_pct:+.2f}% says don't risk it"),
                live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
                min_target_price=min_target, max_target_price=max_target,
                horizon=position.horizon, fresh_prediction=fresh,
                trailing_stop=trail_stop,
            )

    # ── 6) Below MIN profit ───────────────────────────────────────────
    fresh = None
    if do_fresh_prediction:
        try:
            fresh = predict(position.symbol, position.horizon)  # type: ignore
        except Exception:
            fresh = None

    # 6a) Fresh forecast turned bearish
    if fresh is not None:
        edge = fresh.expected_return_pct / 100.0
        if edge < config.MIN_HOLD_CONFIDENCE:
            return Decision(
                symbol=position.symbol, action="SELL",
                reason=(f"Model now predicts {fresh.expected_return_pct:+.2f}% — "
                        f"below hold threshold ({config.MIN_HOLD_CONFIDENCE*100:.1f}%)"),
                live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
                min_target_price=min_target, max_target_price=max_target,
                horizon=position.horizon, fresh_prediction=fresh,
                trailing_stop=trail_stop,
            )

    # 6b) Past full horizon with no progress toward target
    if _past_full_horizon(position) and price < position.predicted_target:
        return Decision(
            symbol=position.symbol, action="SELL",
            reason=(f"Past full horizon, below original target "
                    f"Rs.{position.predicted_target:.2f} — thesis expired"),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=fresh,
            trailing_stop=trail_stop,
        )

    # 6c) HOLD
    reason = (f"Below min goal ({pnl_pct:+.2f}%) — waiting for "
              f"Rs.{min_target:.2f}")
    if fresh is not None:
        reason += f" (fresh forecast {fresh.expected_return_pct:+.2f}%)"
    return Decision(
        symbol=position.symbol, action="HOLD",
        reason=reason,
        live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
        min_target_price=min_target, max_target_price=max_target,
        horizon=position.horizon, fresh_prediction=fresh,
        trailing_stop=trail_stop,
    )


# ------------------------------------------------------------------ #
#  Batch evaluation                                                    #
# ------------------------------------------------------------------ #

def evaluate_all(portfolio: Portfolio,
                 do_fresh_prediction: bool = True) -> list[Decision]:
    """
    Evaluate all open positions. Before evaluating each position, checks
    whether the live price has exceeded the stored high-water mark and
    updates the trailing stop in the DB if so.
    """
    decisions = []
    for position in portfolio.get_positions():
        # Fetch price once — reused for both the trail update and evaluate()
        live_price = get_latest_price(position.symbol)

        # Update trailing stop if price rose above previous high-water mark
        if live_price and live_price > position.highest_price_seen:
            portfolio.update_trailing_stop(position.symbol, live_price)
            # Reload so evaluate() sees the fresh stop_loss level
            refreshed = portfolio.get_position(position.symbol)
            if refreshed:
                position = refreshed

        decisions.append(
            evaluate(position, live_price=live_price,
                     do_fresh_prediction=do_fresh_prediction)
        )
    return decisions


# ------------------------------------------------------------------ #
#  Auto-execute                                                        #
# ------------------------------------------------------------------ #

def auto_execute(portfolio: Portfolio,
                 decisions: list[Decision]) -> list[str]:
    """
    Apply SELL decisions to the paper portfolio automatically.
    Returns a list of human-readable messages for the GUI log.
    """
    messages = []
    for d in decisions:
        if d.action == "SELL" and d.live_price > 0:
            try:
                pnl = portfolio.sell(d.symbol, d.live_price, reason=d.reason)
                messages.append(
                    f"SOLD {d.symbol} @ Rs.{d.live_price:.2f} "
                    f"(P&L Rs.{pnl:+,.2f}) — {d.reason}")
            except Exception as e:
                messages.append(f"Could not sell {d.symbol}: {e}")
    return messages
