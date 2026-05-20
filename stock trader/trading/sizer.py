"""
Position sizer: decides HOW MUCH to buy of a stock.

Rules (in order):
  1. Cap total invested across portfolio at MAX_INVESTED_PCT of equity
     (default 75% - 25% always kept as cash reserve).
  2. Cap any single position at MAX_POSITION_PCT of equity
     (default 10% - no all-in-one-stock).
  3. Within those caps, allocate proportionally to expected edge:
        target_pct = expected_return_pct * ALLOCATION_SCALE
     (higher predicted return -> larger slice of equity).
  4. If the resulting slice is below MIN_ALLOCATION_PCT of equity, skip
     the trade (too tiny to matter, and saves on slippage / mental load).

Returns a Sizing record with quantity, rupees committed, and a list of
human-readable reasons explaining how the number was computed.
"""

from __future__ import annotations
from dataclasses import dataclass, field

import config


@dataclass
class Sizing:
    suggested_qty: int
    suggested_rupees: float
    target_pct_of_equity: float       # uncapped, edge-driven target
    final_pct_of_equity: float        # what we'll actually use after caps
    available_budget: float           # rupees left under the 75% cap
    capital_used_pct: float           # how much of equity is currently invested
    notes: list[str] = field(default_factory=list)
    ok: bool = True                   # False -> don't trade

    @property
    def reason(self) -> str:
        return " | ".join(self.notes)


def suggest_size(
    *,
    equity: float,
    current_invested: float,
    cash: float,
    current_price: float,
    expected_return_pct: float,
) -> Sizing:
    """
    Compute the recommended quantity to buy.

    Parameters
    ----------
    equity            : cash + market_value of open positions (Rs.)
    current_invested  : sum of invested capital across open positions (Rs.)
    cash              : free cash in wallet (Rs.)
    current_price     : live price of the stock under consideration (Rs.)
    expected_return_pct : LSTM-forecasted % return (e.g. 2.5 for +2.5%)

    Returns
    -------
    A Sizing dataclass. If `ok` is False, you should NOT buy.
    """
    notes: list[str] = []

    if equity <= 0 or current_price <= 0:
        return Sizing(0, 0.0, 0.0, 0.0, 0.0, 0.0,
                      notes=["Equity or price not positive"], ok=False)

    # ---- 1) Total-invested cap (75% rule) ----
    invested_cap = equity * config.MAX_INVESTED_PCT
    available_budget = max(0.0, invested_cap - current_invested)
    capital_used_pct = current_invested / equity if equity > 0 else 0.0
    notes.append(
        f"Capital used: {capital_used_pct*100:.1f}% of equity "
        f"(cap {config.MAX_INVESTED_PCT*100:.0f}%)")

    if available_budget <= 0:
        return Sizing(0, 0.0, 0.0, 0.0, 0.0, capital_used_pct,
                      notes=notes + [
                          f"75% invested cap reached - reserve only "
                          f"(Rs.{cash:,.0f} cash, but cannot invest more)"],
                      ok=False)

    # ---- 2) Edge-driven target ----
    edge_pct = max(expected_return_pct, 0.0) / 100.0
    target_pct = edge_pct * config.ALLOCATION_SCALE   # of equity
    notes.append(
        f"Edge {expected_return_pct:+.2f}% x scale {config.ALLOCATION_SCALE:.1f} "
        f"-> target {target_pct*100:.1f}% of equity")

    # ---- 3) Single-position cap ----
    capped_pct = min(target_pct, config.MAX_POSITION_PCT)
    if capped_pct < target_pct:
        notes.append(
            f"Capped to single-position max {config.MAX_POSITION_PCT*100:.0f}%")

    # ---- 4) Minimum allocation threshold ----
    if capped_pct < config.MIN_ALLOCATION_PCT:
        notes.append(
            f"Target {capped_pct*100:.2f}% below minimum "
            f"{config.MIN_ALLOCATION_PCT*100:.1f}% - skip trade")
        return Sizing(0, 0.0, target_pct, capped_pct,
                      available_budget, capital_used_pct,
                      notes=notes, ok=False)

    # ---- 5) Convert % to rupees, respecting the remaining budget AND cash ----
    target_rupees = equity * capped_pct
    spendable = min(target_rupees, available_budget, cash)
    if spendable < target_rupees:
        if spendable == available_budget:
            notes.append(
                f"Reduced by 75% invested cap "
                f"(Rs.{available_budget:,.0f} remaining)")
        elif spendable == cash:
            notes.append(f"Reduced by available cash (Rs.{cash:,.0f})")

    qty = int(spendable // current_price)
    if qty <= 0:
        notes.append(
            f"Cannot afford 1 share at Rs.{current_price:,.2f} "
            f"with Rs.{spendable:,.0f} spendable")
        return Sizing(0, 0.0, target_pct, capped_pct,
                      available_budget, capital_used_pct,
                      notes=notes, ok=False)

    spent = qty * current_price
    final_pct = spent / equity
    notes.append(
        f"Buy {qty} shares at Rs.{current_price:,.2f} "
        f"= Rs.{spent:,.0f} ({final_pct*100:.2f}% of equity)")

    return Sizing(
        suggested_qty=qty,
        suggested_rupees=spent,
        target_pct_of_equity=target_pct,
        final_pct_of_equity=final_pct,
        available_budget=available_budget,
        capital_used_pct=capital_used_pct,
        notes=notes,
        ok=True,
    )
