"""PS6 Q1: Black--Scholes Greeks with PyTorch autograd.

We implement the European call price once, then use PyTorch's automatic
differentiation to obtain first- and second-order Greeks:

    delta = dV/dS
    gamma = d^2V/dS^2
    vega  = dV/dsigma
    theta = -dV/dT        (calendar-time theta convention)
    vanna = d^2V/(dS dsigma)
    volga = d^2V/dsigma^2

The assignment asks for vanna/volga verification, so the experiment below
checks those two second-order Greeks against central finite differences across
a moneyness/maturity grid and reports the max relative error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


DTYPE = torch.float64
NORMAL = torch.distributions.Normal(
    torch.tensor(0.0, dtype=DTYPE), torch.tensor(1.0, dtype=DTYPE)
)


@dataclass(frozen=True)
class BSParams:
    """Black--Scholes inputs shared across the verification grid."""

    K: float = 100.0
    r: float = 0.02
    sigma: float = 0.40
    fd_step_s: float = 1e-2
    fd_step_sigma: float = 1e-4

    def validate(self) -> None:
        if self.K <= 0:
            raise ValueError("K must be positive")
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")
        if self.fd_step_s <= 0 or self.fd_step_sigma <= 0:
            raise ValueError("finite-difference steps must be positive")


def _tensor(x: float, *, requires_grad: bool = False) -> torch.Tensor:
    """Create a float64 scalar tensor."""

    return torch.tensor(float(x), dtype=DTYPE, requires_grad=requires_grad)


def bs_call_price(S: torch.Tensor, K: torch.Tensor, T: torch.Tensor, r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    """Autograd-friendly Black--Scholes European call price."""

    sqrt_T = torch.sqrt(T)
    d1 = (torch.log(S / K) + (r + 0.5 * sigma.square()) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * NORMAL.cdf(d1) - K * torch.exp(-r * T) * NORMAL.cdf(d2)


def autograd_call_greeks(S0: float, K: float, T0: float, r0: float, sigma0: float) -> dict[str, float]:
    """Return Black--Scholes price and Greeks from PyTorch autograd."""

    if S0 <= 0 or K <= 0 or T0 <= 0 or sigma0 <= 0:
        raise ValueError("S, K, T, and sigma must be positive")

    S = _tensor(S0, requires_grad=True)
    T = _tensor(T0, requires_grad=True)
    sigma = _tensor(sigma0, requires_grad=True)
    K_t = _tensor(K)
    r = _tensor(r0)

    price = bs_call_price(S, K_t, T, r, sigma)
    (delta,) = torch.autograd.grad(price, S, create_graph=True)
    (gamma,) = torch.autograd.grad(delta, S, create_graph=True)
    (vega,) = torch.autograd.grad(price, sigma, create_graph=True)
    (d_price_d_maturity,) = torch.autograd.grad(price, T, create_graph=True)
    (vanna,) = torch.autograd.grad(vega, S, create_graph=True)
    (volga,) = torch.autograd.grad(vega, sigma)

    return {
        "price": float(price.detach()),
        "delta": float(delta.detach()),
        "gamma": float(gamma.detach()),
        "vega": float(vega.detach()),
        "theta": float(-d_price_d_maturity.detach()),
        "d_price_d_maturity": float(d_price_d_maturity.detach()),
        "vanna": float(vanna.detach()),
        "volga": float(volga.detach()),
    }


def call_price_float(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black--Scholes call price as a Python float, useful for finite differences."""

    price = bs_call_price(_tensor(S), _tensor(K), _tensor(T), _tensor(r), _tensor(sigma))
    return float(price)


def finite_difference_vanna(S: float, K: float, T: float, r: float, sigma: float, h_s: float, h_sigma: float) -> float:
    """Central finite-difference estimate of d^2V/(dS dsigma)."""

    pp = call_price_float(S + h_s, K, T, r, sigma + h_sigma)
    pm = call_price_float(S + h_s, K, T, r, sigma - h_sigma)
    mp = call_price_float(S - h_s, K, T, r, sigma + h_sigma)
    mm = call_price_float(S - h_s, K, T, r, sigma - h_sigma)
    return (pp - pm - mp + mm) / (4.0 * h_s * h_sigma)


def finite_difference_volga(S: float, K: float, T: float, r: float, sigma: float, h_sigma: float) -> float:
    """Central finite-difference estimate of d^2V/dsigma^2."""

    up = call_price_float(S, K, T, r, sigma + h_sigma)
    mid = call_price_float(S, K, T, r, sigma)
    down = call_price_float(S, K, T, r, sigma - h_sigma)
    return (up - 2.0 * mid + down) / h_sigma**2


def verify_vanna_volga(
    params: BSParams | None = None,
    moneyness_grid: np.ndarray | None = None,
    maturity_grid: np.ndarray | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Verify autograd vanna/volga against finite differences on a grid."""

    params = params or BSParams()
    params.validate()
    if moneyness_grid is None:
        moneyness_grid = np.array([0.70, 0.85, 1.00, 1.15, 1.30])
    if maturity_grid is None:
        maturity_grid = np.array([0.10, 0.25, 0.50, 1.00, 2.00])

    rows: list[dict[str, float]] = []
    for moneyness in moneyness_grid:
        S = float(params.K * moneyness)
        for T in maturity_grid:
            greeks = autograd_call_greeks(S, params.K, float(T), params.r, params.sigma)
            fd_vanna = finite_difference_vanna(
                S, params.K, float(T), params.r, params.sigma, params.fd_step_s, params.fd_step_sigma
            )
            fd_volga = finite_difference_volga(
                S, params.K, float(T), params.r, params.sigma, params.fd_step_sigma
            )
            vanna_abs_error = abs(greeks["vanna"] - fd_vanna)
            volga_abs_error = abs(greeks["volga"] - fd_volga)
            rows.append(
                {
                    "moneyness": float(moneyness),
                    "S": S,
                    "K": params.K,
                    "T": float(T),
                    "r": params.r,
                    "sigma": params.sigma,
                    **greeks,
                    "fd_vanna": fd_vanna,
                    "fd_volga": fd_volga,
                    "vanna_abs_error": vanna_abs_error,
                    "volga_abs_error": volga_abs_error,
                    "vanna_rel_error": vanna_abs_error / max(abs(fd_vanna), 1e-12),
                    "volga_rel_error": volga_abs_error / max(abs(fd_volga), 1e-12),
                }
            )

    checks = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "grid_points": len(checks),
                "max_vanna_rel_error": checks["vanna_rel_error"].max(),
                "max_volga_rel_error": checks["volga_rel_error"].max(),
                "max_second_order_rel_error": max(
                    checks["vanna_rel_error"].max(), checks["volga_rel_error"].max()
                ),
                "mean_vanna_rel_error": checks["vanna_rel_error"].mean(),
                "mean_volga_rel_error": checks["volga_rel_error"].mean(),
                "fd_step_s": params.fd_step_s,
                "fd_step_sigma": params.fd_step_sigma,
            }
        ]
    )
    return checks, summary


def run_ps6_q1(output_dir: str | Path = "output") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full PS6 Q1 check and save CSV artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checks, summary = verify_vanna_volga()
    checks.to_csv(output_dir / "ps6_q1_bs_autograd_greeks_grid.csv", index=False)
    summary.to_csv(output_dir / "ps6_q1_bs_autograd_greeks_summary.csv", index=False)
    return checks, summary


if __name__ == "__main__":
    grid, summary_table = run_ps6_q1()
    row = summary_table.iloc[0]
    print("PS6 Q1: Black--Scholes Greeks via PyTorch autograd")
    print(f"grid points: {int(row['grid_points'])}")
    print(f"max vanna relative error: {row['max_vanna_rel_error']:.3e}")
    print(f"max volga relative error: {row['max_volga_rel_error']:.3e}")
    print(f"max second-order relative error: {row['max_second_order_rel_error']:.3e}")
    print("\nSample ATM, 1Y Greeks:")
    cols = ["price", "delta", "gamma", "vega", "theta", "vanna", "volga"]
    print(grid[(grid["moneyness"] == 1.0) & (grid["T"] == 1.0)][cols].to_string(index=False))
