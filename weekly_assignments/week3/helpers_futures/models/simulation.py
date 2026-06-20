"""Discrete-event simulation harness for the Avellaneda-Stoikov model.

Simulates a single trading day by stepping through a price series at fixed
time increments, computing optimal quotes, drawing Poisson-distributed fill
arrivals, and tracking cash X and inventory q.

Two strategies are provided:
  - AS (Avellaneda-Stoikov): inventory-adjusted reservation price + optimal spread
  - Symmetric baseline: fixed symmetric spread around mid, no inventory adjustment

Fixes vs. original implementation
----------------------------------
  1. Inventory clamp: fill probability is set to zero on the side that would
     breach the inventory limit, rather than drawing a fill and silently
     discarding it.
  2. Funding P&L: when params.funding_rate_per_s > 0, a per-step cash flow of
     inventory * rate * dt is applied, matching the exchange's 8-hour funding
     settlement (approximated as continuous accrual).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .avellaneda_stoikov import ASParams, bid_ask


@dataclass
class SimState:
    """Mutable simulation state carried across steps."""
    cash: float = 0.0
    inventory: int = 0
    pnl: list[float] = field(default_factory=list)
    inv_series: list[int] = field(default_factory=list)
    bid_series: list[float] = field(default_factory=list)
    ask_series: list[float] = field(default_factory=list)


def simulate_as(
    params: ASParams,
    price_series: np.ndarray,
    dt: float = 1.0,
    seed: int | None = 42,
) -> SimState:
    """Simulate a trading session using the A-S optimal quoting strategy.

    Parameters
    ----------
    params : ASParams
        Model parameters. When params.tau_risk is None, params.T should equal
        len(price_series) * dt. When tau_risk is set, T is unused by the model.
    price_series : array of shape (N,)
        Sequence of mid / mark price values (one per time step).
    dt : float
        Length of each time step (same units as params.T and intensity A).
    seed : int, optional
        RNG seed for reproducibility.

    Returns
    -------
    SimState with cash, inventory, and per-step series.
    """
    rng = np.random.default_rng(seed)
    state = SimState()

    for i, s in enumerate(price_series):
        t = i * dt
        q = max(params.q_min, min(params.q_max, state.inventory))

        bid, ask = bid_ask(s, q, t, params)
        state.bid_series.append(bid)
        state.ask_series.append(ask)

        delta_bid = s - bid
        delta_ask = ask - s

        prob_bid = min(1.0, params.A * np.exp(-params.kappa * delta_bid) * dt)
        prob_ask = min(1.0, params.A * np.exp(-params.kappa * delta_ask) * dt)

        # Do not draw fills on the side that would breach inventory limits.
        if q >= params.q_max:
            prob_bid = 0.0
        if q <= params.q_min:
            prob_ask = 0.0

        if rng.random() < prob_bid:
            state.cash -= bid
            state.inventory += 1

        if rng.random() < prob_ask:
            state.cash += ask
            state.inventory -= 1

        # Funding cost: longs pay, shorts receive, continuously accrued.
        if params.funding_rate_per_s != 0.0:
            state.cash -= state.inventory * params.funding_rate_per_s * dt

        mid_val = state.cash + state.inventory * s
        state.pnl.append(mid_val)
        state.inv_series.append(state.inventory)

    return state


def simulate_symmetric(
    params: ASParams,
    price_series: np.ndarray,
    fixed_half_spread: float | None = None,
    dt: float = 1.0,
    seed: int | None = 42,
) -> SimState:
    """Baseline: symmetric fixed-spread quoting around the mid-price.

    No inventory adjustment — quotes are always centered at mid.
    The fixed half-spread defaults to the A-S optimal spread at t=0.
    """
    from .avellaneda_stoikov import optimal_half_spread

    if fixed_half_spread is None:
        fixed_half_spread = optimal_half_spread(0.0, params)

    rng = np.random.default_rng(seed)
    state = SimState()

    for i, s in enumerate(price_series):
        bid = s - fixed_half_spread
        ask = s + fixed_half_spread

        state.bid_series.append(bid)
        state.ask_series.append(ask)

        prob_bid = min(1.0, params.A * np.exp(-params.kappa * fixed_half_spread) * dt)
        prob_ask = min(1.0, params.A * np.exp(-params.kappa * fixed_half_spread) * dt)

        if prob_bid > 0 and state.inventory < params.q_max:
            if rng.random() < prob_bid:
                state.cash -= bid
                state.inventory += 1

        if prob_ask > 0 and state.inventory > params.q_min:
            if rng.random() < prob_ask:
                state.cash += ask
                state.inventory -= 1

        mid_val = state.cash + state.inventory * s
        state.pnl.append(mid_val)
        state.inv_series.append(state.inventory)

    return state


def pnl_stats(state: SimState) -> dict[str, float]:
    """Compute summary statistics from a completed simulation."""
    pnl = np.array(state.pnl)
    inv = np.array(state.inv_series)
    returns = np.diff(pnl)
    mean = float(np.mean(returns))
    std = float(np.std(returns))
    sharpe = mean / std * np.sqrt(len(returns)) if std > 0 else 0.0
    max_dd = float(np.max(np.maximum.accumulate(pnl) - pnl))
    return {
        "final_pnl": float(pnl[-1]),
        "mean_step_return": mean,
        "std_step_return": std,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "mean_inventory": float(np.mean(inv)),
        "std_inventory": float(np.std(inv)),
    }
