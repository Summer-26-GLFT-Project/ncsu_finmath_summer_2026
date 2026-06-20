"""Avellaneda-Stoikov (2008) optimal market making model.

Reference: Avellaneda and Stoikov, "High-frequency trading in a limit order
book," Quantitative Finance 8(3), 217-224 (2008).

Setup
-----
Mid-price S follows arithmetic Brownian motion with volatility σ.
Order arrivals are Poisson with intensity Λ(δ) = A · exp(-κ · δ),
where δ is the quote distance from mid.
The MM maximises expected exponential utility of terminal wealth
over horizon T with risk-aversion parameter γ.

Asymptotic solution (T → ∞ or t ≪ T, used in practice)
-------------------------------------------------------
Reservation price:  r(s, q, t) = s - q · γ · σ² · τ - q · f · τ
Optimal half-spread: δ* = (γ · σ² · τ) / 2 + (1/γ) · ln(1 + γ/κ)

where τ = tau_risk (fixed) if set, else T - t.

Perpetual futures adaptation
-----------------------------
Two parameters extend the base model for instruments without a terminal date:

  tau_risk : float | None
      Fixed effective risk horizon (seconds). When set, replaces T−t in all
      formulas, making the model stationary (time-invariant). This is the
      infinite-horizon limit studied in Guéant, Lehalle & Fernandez-Tapia
      (2013). A typical value for crypto perp market making is 300–900 s.
      When None, the original finite-horizon behaviour is preserved.

  funding_rate_per_s : float
      Funding rate in USD per unit inventory per second (positive = longs pay
      shorts). Adds q·f·τ to the reservation price penalty, so a long position
      in a high positive-funding environment lowers r and makes the ask more
      competitive. Applied as a realised cash flow in simulation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ASParams:
    """Avellaneda-Stoikov model parameters."""
    gamma: float    # risk-aversion coefficient (CARA)
    sigma: float    # mid-price volatility (same units as price, per unit time)
    kappa: float    # order-arrival intensity decay parameter (κ in Λ=A·exp(-κδ))
    A: float        # order-arrival intensity at δ=0 (orders per unit time)
    T: float        # trading horizon (time units, e.g. seconds)
    q_min: int = -10  # minimum allowable inventory
    q_max: int = 10   # maximum allowable inventory

    # --- perpetual futures extensions ---
    tau_risk: float | None = None
    # When set, replaces T-t with this fixed value in all formulas (stationary
    # mode). When None, the original finite-horizon formula is used.

    funding_rate_per_s: float = 0.0
    # Funding rate: USD per unit inventory per second.
    # Typical Coincall BTC perp: ~0.01%/8h = 0.0001/28800 × price ≈ 0.23 USD/s
    # at BTC = 65,000. Zero disables the funding term (original model).


def _tau(t: float, params: ASParams) -> float:
    """Compute effective time-to-go, respecting stationary-mode override."""
    if params.tau_risk is not None:
        return params.tau_risk
    return params.T - t


def reservation_price(s: float, q: int | float, t: float, params: ASParams) -> float:
    """Reservation (fair) price adjusted for inventory risk and funding carry.

    r(s, q, t) = s - q · (γ · σ² + f) · τ

    where τ = tau_risk (fixed) if set, else T - t.

    Parameters
    ----------
    s : current mid / mark price
    q : current inventory (positive = long)
    t : current time (0 ≤ t ≤ T); ignored when tau_risk is set
    params : model parameters
    """
    tau = _tau(t, params)
    risk_aversion = params.gamma * params.sigma ** 2 + params.funding_rate_per_s
    return s - q * risk_aversion * tau


def optimal_half_spread(t: float, params: ASParams) -> float:
    """Optimal symmetric half-spread δ* around the reservation price.

    δ* = (γ · σ² · τ) / 2 + (1/γ) · ln(1 + γ/κ)

    The first term compensates for inventory risk (mean-reversion pressure).
    The second term is the spread component from order-flow competition.
    τ uses the stationary override when tau_risk is set.
    """
    tau = _tau(t, params)
    risk_term = params.gamma * params.sigma ** 2 * tau / 2.0
    flow_term = math.log(1.0 + params.gamma / params.kappa) / params.gamma
    return risk_term + flow_term


def bid_ask(
    s: float, q: int | float, t: float, params: ASParams
) -> tuple[float, float]:
    """Return optimal (bid, ask) quote pair.

    bid = r - δ*
    ask = r + δ*
    """
    r = reservation_price(s, q, t, params)
    half = optimal_half_spread(t, params)
    return r - half, r + half


# ---------------------------------------------------------------------------
# Vectorised versions for array inputs
# ---------------------------------------------------------------------------

def reservation_price_vec(
    s: np.ndarray, q: np.ndarray, t: np.ndarray, params: ASParams
) -> np.ndarray:
    if params.tau_risk is not None:
        tau = np.full_like(t, params.tau_risk)
    else:
        tau = params.T - t
    risk_aversion = params.gamma * params.sigma ** 2 + params.funding_rate_per_s
    return s - q * risk_aversion * tau


def optimal_half_spread_vec(t: np.ndarray, params: ASParams) -> np.ndarray:
    if params.tau_risk is not None:
        tau = np.full_like(t, params.tau_risk, dtype=float)
    else:
        tau = params.T - t
    risk = params.gamma * params.sigma ** 2 * tau / 2.0
    flow = math.log(1.0 + params.gamma / params.kappa) / params.gamma
    return risk + flow


def bid_ask_vec(
    s: np.ndarray, q: np.ndarray, t: np.ndarray, params: ASParams
) -> tuple[np.ndarray, np.ndarray]:
    r = reservation_price_vec(s, q, t, params)
    half = optimal_half_spread_vec(t, params)
    return r - half, r + half
