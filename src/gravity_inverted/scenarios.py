"""Ready-made experiments used by the app, kept here so they are tested.

Each function returns plain numbers (percentage changes) and pandas objects.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .equilibrium import Baseline
from .equilibrium import solve_forward
from .inverse import WelfareTarget, solve_welfare_targets
from .trade_flows import invert_trade_flows


def pct(x):
    return (np.asarray(x) - 1.0) * 100.0


@dataclass
class RouteResult:
    instrument: float  # required change, %
    exports: float  # own total exports, %
    home_sales: float  # own domestic sales, %
    others: pd.Series  # welfare change of all other economies, %


def two_routes(baseline: Baseline, code: str, goal_pct: float,
               sigma: float) -> dict[str, RouteResult]:
    """Domestic and uniform import instrument for one unilateral welfare goal."""
    W = 1 + goal_pct / 100
    i = baseline.index(code)
    out = {}
    for kind in ("domestic", "import"):
        r = solve_welfare_targets(baseline, WelfareTarget(code, W, kind), sigma)
        eq = r.equilibrium
        out[kind] = RouteResult(
            r.instrument_pct()[code], float(pct(eq.exports_hat[i])),
            float(pct(eq.domestic_sales_hat[i])),
            pd.Series(pct(eq.W_hat), index=baseline.countries).drop(code))
    return out


@dataclass
class WallResult:
    loss: float  # own welfare change from the wall, %
    others_wall: pd.Series  # others' welfare change from the wall, %
    restore: float  # domestic cut that restores pre-wall welfare, after the wall, %
    equivalent: float  # domestic cut giving a gain of the same size, from the start, %
    others_equivalent: pd.Series  # others' welfare change from that cut, %

    def spillover_ratio(self, min_abs: float = 1e-4) -> float | None:
        """Ratio of others' average losses, only when it is meaningful."""
        a, b = self.others_wall.mean(), self.others_equivalent.mean()
        if abs(b) < min_abs or np.sign(a) != np.sign(b):
            return None
        return float(a / b)


def wall_and_fix(baseline: Baseline, code: str, wall_pct: float, sigma: float) -> WallResult:
    """Uniform increase in a country's import costs and domestic alternatives.

    * ``restore``: starting from the equilibrium with the wall in place, the
      domestic trade cost change that brings welfare back to its pre-wall level
      (target 1 / loss).
    * ``equivalent``: starting from the original baseline, the domestic change
      that raises welfare by the size of the loss, as in Table 2 of the paper
      (target 2 - loss).
    """
    j = baseline.index(code)
    tau = np.ones((baseline.n, baseline.n))
    tau[np.arange(baseline.n) != j, j] = 1 + wall_pct / 100
    shock = solve_forward(baseline, tau, sigma)
    W = float(shock.W_hat[j])
    restore = solve_welfare_targets(shock.as_baseline(), WelfareTarget(code, 1 / W), sigma)
    equiv = solve_welfare_targets(baseline, WelfareTarget(code, 2 - W), sigma)
    s = lambda e: pd.Series(pct(e.W_hat), index=baseline.countries).drop(code)  # noqa: E731
    return WallResult(float(pct(W)), s(shock), restore.instrument_pct()[code],
                      equiv.instrument_pct()[code], s(equiv.equilibrium))


@dataclass
class PairResult:
    share_a_in_b: tuple[float, float]  # % of B's spending on A's goods, start and end
    share_b_in_a: tuple[float, float]
    symmetric: float  # both directions together (Novy index), %
    a_to_b: dict[str, float]  # mge, full_nominal, full_real
    b_to_a: dict[str, float]


def pair_since(countries, X0, X1, a: str, b: str, sigma: float,
               Q_hat: np.ndarray | None = None) -> PairResult:
    """Observed change in trade between two economies (Lemma 2)."""
    i, j = countries.index(a), countries.index(b)
    nom = invert_trade_flows(countries, X0, X1, sigma)
    real = invert_trade_flows(countries, X0, X1, sigma, Q_hat=Q_hat) if Q_hat is not None else None
    share = lambda X, e, m: float(X[e, m] / X[:, m].sum() * 100)  # noqa: E731

    def dirs(e, m):
        d = {"mge": float(nom.pct("mge")[e, m]), "full_nominal": float(nom.pct("full_ge")[e, m])}
        if real is not None:
            d["full_real"] = float(real.pct("full_ge")[e, m])
        return d

    return PairResult((share(X0, i, j), share(X1, i, j)), (share(X0, j, i), share(X1, j, i)),
                      float(pct(nom.symmetric[i, j])), dirs(i, j), dirs(j, i))


def sigma_band(make_baseline, fn, sigmas=(3.0, 5.0, 7.0)) -> dict[float, float]:
    """Evaluate ``fn(baseline, sigma)`` for several elasticities.

    The baseline is rebuilt for each sigma because removing trade imbalances
    (Dekle et al. 2007) depends on sigma.
    """
    return {s: fn(make_baseline(s), s) for s in sigmas}
