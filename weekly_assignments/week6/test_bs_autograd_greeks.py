import numpy as np

from bs_autograd_greeks import (
    BSParams,
    autograd_call_greeks,
    run_ps6_q1,
    verify_vanna_volga,
)


def test_autograd_greeks_have_expected_signs_for_call():
    greeks = autograd_call_greeks(S0=100.0, K=100.0, T0=1.0, r0=0.02, sigma0=0.40)

    assert greeks["price"] > 0
    assert 0 < greeks["delta"] < 1
    assert greeks["gamma"] > 0
    assert greeks["vega"] > 0


def test_vanna_volga_match_finite_differences_on_grid():
    params = BSParams()
    checks, summary = verify_vanna_volga(
        params,
        moneyness_grid=np.array([0.85, 1.0, 1.15]),
        maturity_grid=np.array([0.25, 1.0, 2.0]),
    )

    assert len(checks) == 9
    assert summary.loc[0, "max_vanna_rel_error"] < 5e-5
    assert summary.loc[0, "max_volga_rel_error"] < 5e-5


def test_run_ps6_q1_writes_expected_csvs(tmp_path):
    checks, summary = run_ps6_q1(tmp_path)

    assert not checks.empty
    assert summary.loc[0, "grid_points"] == 25
    assert (tmp_path / "ps6_q1_bs_autograd_greeks_grid.csv").exists()
    assert (tmp_path / "ps6_q1_bs_autograd_greeks_summary.csv").exists()
