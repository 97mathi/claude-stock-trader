"""
Position monitor / decision engine - goal-aware.

For each open position it answers ONE question: HOLD or SELL right now?

Decision order (first match wins):
  1. Stop-loss hit (price <= stop_loss)            -> SELL (limit losses)
  2. MAX profit hit (P&L% >= MAX for this horizon) -> SELL (don't get greedy)
  3. MIN profit reached (P&L% in [MIN, MAX))       -> Run a FRESH prediction:
       - Forecast positive                          -> HOLD  (let it run)
       - Forecast flat/negative                     -> SELL  (lock the gain)
  4. Below MIN profit:
       a. Fresh forecast turned bearish (< MIN_HOLD_CONFIDENCE) -> SELL
       b. Past horizon and below original target               -> SELL
       c. Otherwise                                            -> HOLD

Each Decision also carries the min/max target prices so the GUI can show
progress toward the goal.
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
    min_target_price: float   # buy_price * (1 + MIN_PROFIT_PCT)
    max_target_price: float   # buy_price * (1 + MAX_PROFIT_PCT)
    horizon: str
    fresh_prediction: Prediction | None


def _days_since(iso_ts: str) -> int:
    try:
        d = dt.datetime.fromisoformat(iso_ts)
        return (dt.datetime.now() - d).days
    except Exception:
        return 0


def evaluate(position: Position,
             live_price: float | None = None,
             do_fresh_prediction: bool = True) -> Decision:
    """
    Decide whether to hold or sell ONE position.
    `do_fresh_prediction=False` skips the LSTM call (faster bulk refresh
    when you only need the cheap risk checks).
    """
    min_pct, max_pct = config.profit_goals_for(position.horizon)
    min_target = position.buy_price * (1 + min_pct)
    max_target = position.buy_price * (1 + max_pct)

    price = live_price if live_price is not None \
        else get_latest_price(position.symbol)

    if price is None or price <= 0:
        return Decision(
            symbol=position.symbol, action="HOLD",
            reason="Live price unavailable - skipping check",
            live_price=0.0, buy_price=position.buy_price,
            pnl_pct=0.0,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
        )

    pnl_pct = (price - position.buy_price) / position.buy_price * 100.0
    pnl_ratio = pnl_pct / 100.0

    # ---------- 1) Stop loss ----------
    if price <= position.stop_loss:
        return Decision(
            symbol=position.symbol, action="SELL",
            reason=(f"Stop-loss hit (price Rs.{price:.2f} <= "
                    f"Rs.{position.stop_loss:.2f})"),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
        )

    # ---------- 1b) Below breakeven – hold until recovery ----------
    if pnl_pct < 0:
        return Decision(
            symbol=position.symbol, action="HOLD",
            reason=(f"Below breakeven ({pnl_pct:+.2f}%) — holding until "
                    f"min target Rs.{min_target:.2f}"),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
        )

    # ---------- 2) Max profit reached -> auto-sell (greed cap) ----------
    if pnl_ratio >= max_pct:
        return Decision(
            symbol=position.symbol, action="SELL",
            reason=(f"Max-profit goal hit: {pnl_pct:+.2f}% "
                    f">= {max_pct*100:.1f}% ({config.goal_period_label(position.horizon).lower()} cap)"),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=None,
        )

    # ---------- 3) Min profit reached -> fresh prediction decides ----------
    if pnl_ratio >= min_pct:
        fresh = None
        if do_fresh_prediction:
            try:
                fresh = predict(position.symbol, position.horizon)  # type: ignore
            except Exception:
                fresh = None

        if fresh is None:
            # Can't predict - be safe and lock in the gain.
            return Decision(
                symbol=position.symbol, action="SELL",
                reason=(f"In profit zone ({pnl_pct:+.2f}%) but fresh forecast "
                        "unavailable - locking gain"),
                live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
                min_target_price=min_target, max_target_price=max_target,
                horizon=position.horizon, fresh_prediction=None,
            )

        if fresh.expected_return_pct > 0:
            return Decision(
                symbol=position.symbol, action="HOLD",
                reason=(f"In profit zone ({pnl_pct:+.2f}%), fresh forecast "
                        f"{fresh.expected_return_pct:+.2f}% - let it run "
                        f"toward Rs.{max_target:.2f}"),
                live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
                min_target_price=min_target, max_target_price=max_target,
                horizon=position.horizon, fresh_prediction=fresh,
            )
        else:
            return Decision(
                symbol=position.symbol, action="SELL",
                reason=(f"Profit locked at {pnl_pct:+.2f}% - fresh forecast "
                        f"{fresh.expected_return_pct:+.2f}% says don't risk it"),
                live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
                min_target_price=min_target, max_target_price=max_target,
                horizon=position.horizon, fresh_prediction=fresh,
            )

    # ---------- 4) Below min profit - existing rules ----------
    fresh = None
    if do_fresh_prediction:
        try:
            fresh = predict(position.symbol, position.horizon)  # type: ignore
        except Exception:
            fresh = None

    # 4a) Fresh forecast turned bearish
    if fresh is not None:
        edge = fresh.expected_return_pct / 100.0
        if edge < config.MIN_HOLD_CONFIDENCE:
            return Decision(
                symbol=position.symbol, action="SELL",
                reason=(f"Model now predicts {fresh.expected_return_pct:+.2f}% - "
                        f"below hold threshold ({config.MIN_HOLD_CONFIDENCE*100:.1f}%)"),
                live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
                min_target_price=min_target, max_target_price=max_target,
                horizon=position.horizon, fresh_prediction=fresh,
            )

    # 4b) Past horizon with no progress to target
    days_held = _days_since(position.buy_date)
    horizon_days = (config.SWING_HORIZON_DAYS if position.horizon == "swing"
                    else max(1, config.INTRADAY_HORIZON_BARS // 6))
    if days_held >= horizon_days and price < position.predicted_target:
        return Decision(
            symbol=position.symbol, action="SELL",
            reason=(f"Past horizon ({days_held}d) and below original target "
                    f"(Rs.{position.predicted_target:.2f}) - thesis expired"),
            live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
            min_target_price=min_target, max_target_price=max_target,
            horizon=position.horizon, fresh_prediction=fresh,
        )

    # 4c) Hold
    reason = (f"Below min goal ({pnl_pct:+.2f}%) - waiting for "
              f"Rs.{min_target:.2f}")
    if fresh is not None:
        reason += f" (fresh forecast {fresh.expected_return_pct:+.2f}%)"
    return Decision(
        symbol=position.symbol, action="HOLD",
        reason=reason,
        live_price=price, buy_price=position.buy_price, pnl_pct=pnl_pct,
        min_target_price=min_target, max_target_price=max_target,
        horizon=position.horizon, fresh_prediction=fresh,
    )


def evaluate_all(portfolio: Portfolio,
                 do_fresh_prediction: bool = True) -> list[Decision]:
    return [evaluate(p, do_fresh_prediction=do_fresh_prediction)
            for p in portfolio.get_positions()]


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
                    f"(P&L Rs.{pnl:+,.2f}) - {d.reason}")
            except Exception as e:
                messages.append(f"Could not sell {d.symbol}: {e}")
    return messages
