"""Model-internal checks on random balanced economies (no data needed)."""

import numpy as np
import pytest

from gravity_inverted import (Baseline, TargetNotAttainable, WelfareTarget, approx,
                              bilateral_tau, import_bound, invert_trade_flows,
                              solve_forward, solve_welfare_targets)
from gravity_inverted.equilibrium import balance_trade

SIGMA = 5.0


def random_flows(n, seed, home_bias=8.0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.2, 1.0, (n, n)) * rng.uniform(0.5, 3.0, n)[:, None]
    X[np.diag_indices(n)] *= home_bias
    return X


@pytest.fixture(scope="module")
def base():
    names = [f"C{i}" for i in range(7)]
    return balance_trade(names, random_flows(7, 1), SIGMA)


def test_balancing_closes_imbalances(base):
    assert base.is_balanced()
    assert np.isclose(base.Y.sum(), 1.0)


def test_forward_identity(base):
    eq = solve_forward(base, np.ones((base.n, base.n)), SIGMA)
    assert np.allclose(eq.p_hat, 1.0) and np.allclose(eq.W_hat, 1.0)


def test_lemma1_extended_acr(base):
    tau = bilateral_tau(base.n, {(0, 0): 0.97, (2, 0): 0.8, (0, 3): 1.2})
    eq = solve_forward(base, tau, SIGMA)
    pi_hat = np.diag(eq.shares) / base.domestic_share
    W_lemma = pi_hat ** (1 / (1 - SIGMA)) / np.diag(tau)
    assert np.allclose(eq.W_hat, W_lemma, rtol=1e-10)
    assert np.isclose((eq.Y if hasattr(eq, "Y") else base.Y * eq.p_hat).sum(), 1.0)


@pytest.mark.parametrize("kind", ["domestic", "import"])
@pytest.mark.parametrize("W", [0.99, 1.01, 1.05])
def test_unilateral_round_trip(base, kind, W):
    res = solve_welfare_targets(base, WelfareTarget("C2", W, kind), SIGMA)
    assert np.isclose(res.equilibrium.W_hat[2], W, rtol=1e-10)
    # instrument moves in the opposite direction of the target
    assert (res.instrument_hat["C2"] - 1) * (W - 1) < 0


def test_joint_targets(base):
    ts = [WelfareTarget("C0", 1.02, "domestic"), WelfareTarget("C1", 1.01, "import"),
          WelfareTarget("C4", 0.995, "domestic")]
    res = solve_welfare_targets(base, ts, SIGMA)
    for t in ts:
        assert np.isclose(res.equilibrium.W_hat[base.index(t.country)], t.W, rtol=1e-10)


def test_forward_then_invert_recovers_instrument(base):
    tau = bilateral_tau(base.n, {(3, 3): 0.95})
    W = solve_forward(base, tau, SIGMA).W_hat[3]
    res = solve_welfare_targets(base, WelfareTarget("C3", W, "domestic"), SIGMA)
    assert np.isclose(res.instrument_hat["C3"], 0.95, rtol=1e-9)


def test_proposition2_bound(base):
    b = import_bound(base, "C5", SIGMA)
    with pytest.raises(TargetNotAttainable):
        solve_welfare_targets(base, WelfareTarget("C5", b * 0.999, "import"), SIGMA)
    res = solve_welfare_targets(base, WelfareTarget("C5", b * 1.02, "import"), SIGMA)
    assert res.instrument_hat["C5"] > 1


def test_large_domestic_target_attainable(base):
    res = solve_welfare_targets(base, WelfareTarget("C6", 1.5, "domestic"), SIGMA)
    assert np.isclose(res.equilibrium.W_hat[6], 1.5)


def test_ge_ordering_vs_first_order(base):
    """Appendix D.1: GE makes the domestic instrument smaller and the import
    instrument larger than the first-order rules."""
    d = base.domestic_share[2]
    dom = solve_welfare_targets(base, WelfareTarget("C2", 1.01, "domestic"), SIGMA)
    imp = solve_welfare_targets(base, WelfareTarget("C2", 1.01, "import"), SIGMA)
    assert approx.first_order_domestic(d, 1.01) < dom.instrument_hat["C2"] < 1
    assert imp.instrument_hat["C2"] < approx.first_order_import(d, 1.01)


def test_lemma2_rationalizes_flows(base):
    """Forward-solve a shock, then invert the flows: recover the shock."""
    rng = np.random.default_rng(3)
    tau = np.exp(rng.normal(0, 0.1, (base.n, base.n)))
    np.fill_diagonal(tau, 1.0)
    eq = solve_forward(base, tau, SIGMA)
    inv = invert_trade_flows(base.countries, base.X, eq.X, SIGMA)
    assert np.allclose(inv.full_ge, tau, rtol=1e-9)
    assert np.allclose(inv.symmetric, np.sqrt(inv.full_ge * inv.full_ge.T))
