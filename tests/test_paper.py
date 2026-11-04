"""Replication of Anderson, Larch and Yotov (2026) with WIOD 2016 data.

Each test compares against a published table. Values are percentage changes.
Skipped when the aggregated WIOD file has not been built yet.
"""

import numpy as np
import pytest

from gravity_inverted import (WelfareTarget, bilateral_tau, invert_trade_flows,
                              solve_forward, solve_welfare_targets)
from gravity_inverted.data import has_wiod, wiod_baseline, wiod_flows

pytestmark = pytest.mark.skipif(not has_wiod(), reason="WIOD data not built")

SIGMA = 5.0
NAFTA = ("CAN", "MEX", "USA")
TOL2 = 0.0051  # tables rounded to two decimals
TOL4 = 0.000051


def pct(x):
    return (np.asarray(x) - 1.0) * 100.0


@pytest.fixture(scope="module")
def base():
    return wiod_baseline(2014, SIGMA)


def nafta_tau(base, sigma=SIGMA):
    t = np.exp(0.7 / (1.0 - sigma))
    idx = [base.index(c) for c in NAFTA]
    return bilateral_tau(base.n, {(i, j): t for i in idx for j in idx if i != j})


@pytest.fixture(scope="module")
def nafta(base):
    return solve_forward(base, nafta_tau(base), SIGMA)


@pytest.fixture(scope="module")
def us_barrier(base):
    j = base.index("USA")
    tau = np.ones((base.n, base.n))
    tau[np.arange(base.n) != j, j] = 1.5
    return solve_forward(base, tau, SIGMA)


# Table D1: domestic shares and exact instruments for a 1 % welfare gain
D1 = {  # country: (pi_jj, domestic exact, import exact)
    "LUX": (0.467, -1.60, -2.48), "IRL": (0.519, -1.48, -2.82),
    "MLT": (0.532, -1.46, -2.91), "HUN": (0.605, -1.34, -3.55),
    "BEL": (0.663, -1.26, -4.25), "NLD": (0.686, -1.23, -4.59),
    "DEU": (0.791, -1.13, -7.04), "CAN": (0.832, -1.10, -8.58),
    "MEX": (0.832, -1.10, -8.59), "FRA": (0.848, -1.09, -9.40),
    "GBR": (0.858, -1.08, -10.03), "JPN": (0.901, -1.05, -13.77),
    "USA": (0.930, -1.03, -18.35), "CHN": (0.935, -1.03, -19.18),
    "ROW": (0.827, -1.10, -8.49),
}


@pytest.mark.parametrize("c", sorted(D1))
def test_table_d1(base, c):
    share, dom, imp = D1[c]
    assert abs(base.domestic_share[base.index(c)] - share) < 0.00051
    r_d = solve_welfare_targets(base, WelfareTarget(c, 1.01, "domestic"), SIGMA)
    r_m = solve_welfare_targets(base, WelfareTarget(c, 1.01, "import"), SIGMA)
    assert abs(r_d.instrument_pct()[c] - dom) < TOL2
    assert abs(r_m.instrument_pct()[c] - imp) < TOL2


# Table C1 / Table 1 Panel A (four decimals)
C1 = {"CAN": (1.1906, -1.5155, 52.6767, -9.2074, 2.7478),
      "MEX": (2.5632, -0.3174, 57.3039, -8.4828, 2.8898),
      "USA": (0.0224, -0.3928, 22.0015, -1.6282, 0.4169),
      "DEU": (-0.0792, -0.0743, -0.1531, -0.0597, -0.0049),
      "IRL": (-0.0819, -0.0574, -0.1880, 0.0164, -0.0246)}


def test_table_c1_nafta(nafta):
    t = nafta.table()
    keys = ["producer_prices", "consumer_prices", "total_exports", "domestic_sales", "welfare"]
    for c, vals in C1.items():
        i = list(t["country"]).index(c)
        for k, v in zip(keys, vals):
            assert abs(t[k][i] - v) < TOL4, (c, k)


# Table 1 Panel B
T1B = {"CAN": (-2.96, -2.98, -19.17, -16.72),
       "MEX": (-3.11, -3.13, -19.89, -17.27),
       "USA": (-0.43, -0.45, -9.11, -5.71)}


@pytest.mark.parametrize("kind,col", [("domestic", 0), ("import", 2)])
def test_table1_unilateral(base, nafta, kind, col):
    for c in NAFTA:
        W = nafta.W_hat[base.index(c)]
        r = solve_welfare_targets(base, WelfareTarget(c, W, kind), SIGMA)
        assert abs(r.instrument_pct()[c] - T1B[c][col]) < TOL2


@pytest.mark.parametrize("kind,col", [("domestic", 1), ("import", 3)])
def test_table1_joint(base, nafta, kind, col):
    ts = [WelfareTarget(c, nafta.W_hat[base.index(c)], kind) for c in NAFTA]
    r = solve_welfare_targets(base, ts, SIGMA)
    for c in NAFTA:
        assert abs(r.instrument_pct()[c] - T1B[c][col]) < TOL2


# Table D2: trade effects of unilateral instruments
D2 = {"CAN": (-4.71, 2.38, 39.10, -17.28), "MEX": (-4.92, 2.52, 40.76, -18.11),
      "USA": (-0.75, 0.22, 17.97, -4.88)}


def test_table_d2(base, nafta):
    for c in NAFTA:
        i = base.index(c)
        W = nafta.W_hat[i]
        d = solve_welfare_targets(base, WelfareTarget(c, W, "domestic"), SIGMA).equilibrium
        m = solve_welfare_targets(base, WelfareTarget(c, W, "import"), SIGMA).equilibrium
        got = (pct(d.exports_hat[i]), pct(d.domestic_sales_hat[i]),
               pct(m.exports_hat[i]), pct(m.domestic_sales_hat[i]))
        assert np.allclose(got, D2[c], atol=TOL2), c


# Table C2 / Table 2 Panel A
C2 = {"USA": (15.0755, 16.3052, -51.4761, 20.0736, -1.0573),
      "CAN": (-4.2141, -2.9437, -29.8794, 0.9692, -1.3089),
      "CHN": (-3.7168, -3.6459, -7.7692, -3.4328, -0.0736)}


def test_table_c2_us_barrier(us_barrier):
    t = us_barrier.table()
    keys = ["producer_prices", "consumer_prices", "total_exports", "domestic_sales", "welfare"]
    for c, vals in C2.items():
        i = list(t["country"]).index(c)
        for k, v in zip(keys, vals):
            assert abs(t[k][i] - v) < TOL4, (c, k)


def test_table2_panel_b(base, us_barrier):
    j = base.index("USA")
    target = 2.0 - us_barrier.W_hat[j]  # same magnitude as the loss
    init = solve_welfare_targets(base, WelfareTarget("USA", target), SIGMA)
    post = solve_welfare_targets(us_barrier.as_baseline(), WelfareTarget("USA", target), SIGMA)
    e = init.equilibrium
    others = np.delete(pct(e.W_hat), j).mean()
    assert abs(init.instrument_pct()["USA"] - (-1.09)) < TOL2
    assert abs(post.instrument_pct()["USA"] - (-1.06)) < TOL2
    assert abs(pct(e.exports_hat[j]) - (-1.89)) < TOL2
    assert abs(pct(e.domestic_sales_hat[j]) - 0.55) < TOL2
    assert abs(others - (-0.008)) < 0.00051


# Table D4: robustness to sigma
@pytest.mark.parametrize("sigma,expect", [
    (3.0, {"CAN": (5.48, -5.78, -32.48)}),
    (7.0, {"CAN": (1.84, -1.99, -13.64)}),
])
def test_table_d4(sigma, expect):
    b = wiod_baseline(2014, sigma)
    eq = solve_forward(b, nafta_tau(b, sigma), sigma)
    for c, (w, dom, imp) in expect.items():
        W = eq.W_hat[b.index(c)]
        assert abs(pct(W) - w) < TOL2
        assert abs(solve_welfare_targets(b, WelfareTarget(c, W, "domestic"), sigma)
                   .instrument_pct()[c] - dom) < TOL2
        assert abs(solve_welfare_targets(b, WelfareTarget(c, W, "import"), sigma)
                   .instrument_pct()[c] - imp) < TOL2


# Table D6 / Figure 3: observed trade 2000-2014 (nominal output)
def test_table_d6():
    countries, X00 = wiod_flows(2000)
    _, X14 = wiod_flows(2014)
    inv = invert_trade_flows(countries, X00, X14, SIGMA)
    c, u = countries.index("CHN"), countries.index("USA")
    assert abs(inv.pct("mge")[c, u] - (-29.7)) < 0.051
    assert abs(inv.pct("mge")[u, c] - 2.2) < 0.051
    assert abs(inv.pct("full_ge")[c, u] - (-88.0)) < 0.051
    assert abs(inv.pct("full_ge")[u, c] - 497.8) < 0.051
    assert abs(pct(inv.symmetric[c, u]) - (-15.2)) < 0.051
    off = np.arange(len(countries)) != c
    assert abs(inv.pct("mge")[c, off].mean() - (-32.9)) < 0.051
    assert abs(inv.pct("full_ge")[c, off].mean() - (-79.1)) < 0.051
    assert abs(inv.pct("full_ge")[off, c].mean() - 268.3) < 0.051


# Table D6, Panel B, last column: producer prices net of real output growth
def test_table_d6_real_output():
    from gravity_inverted.data import has_real_output, wiod_real_output_growth
    if not has_real_output():
        pytest.skip("real output not built")
    countries, X00 = wiod_flows(2000)
    _, X14 = wiod_flows(2014)
    Q = wiod_real_output_growth(countries, 2000, 2014)
    idx = {c: countries.index(c) for c in ("CHN", "JPN", "USA", "ROW")}
    for c, v in {"CHN": 360.4, "JPN": -1.8, "USA": 19.8, "ROW": 44.1}.items():
        assert abs(pct(Q[idx[c]]) - v) < 0.051
    inv = invert_trade_flows(countries, X00, X14, SIGMA, Q_hat=Q)
    full = inv.pct("full_ge")
    c, u, j = idx["CHN"], idx["USA"], idx["JPN"]
    assert abs(full[c, u] - (-53.8)) < 0.051
    assert abs(full[u, c] - 55.5) < 0.051
    def avg_exp(k):
        return np.delete(full[k], k).mean()
    def avg_imp(k):
        return np.delete(full[:, k], k).mean()
    assert abs(avg_exp(c) - (-35.2)) < 0.051
    assert abs(avg_imp(c) - 8.3) < 0.051
    assert abs(avg_exp(j) - 132.3) < 0.051
    assert abs(avg_exp(u) - 59.7) < 0.051
    assert abs(avg_imp(u) - (-29.1)) < 0.051
