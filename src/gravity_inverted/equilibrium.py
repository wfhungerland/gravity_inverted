"""Structural gravity in changes (Anderson, Larch and Yotov 2026, Section 2.1).

Conventions
-----------
* ``X[i, j]`` is the flow from exporter ``i`` (row) to importer ``j`` (column),
  domestic sales on the diagonal.
* ``Y = X.sum(axis=1)`` is output, ``E = X.sum(axis=0)`` is expenditure.
* ``pi[i, j] = X[i, j] / E[j]`` are expenditure shares (columns sum to one).
* Hats are ratios counterfactual / baseline; ``(hat - 1) * 100`` is the
  percentage change reported in the paper's tables.
* World output is the numeraire: ``sum(Y_b * p_hat) == 1`` with ``sum(Y_b) == 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.optimize import root

ArrayF = np.ndarray


class SolverError(RuntimeError):
    """Raised when a general equilibrium system cannot be solved."""


@dataclass(frozen=True)
class Baseline:
    """A balanced baseline: expenditure shares and output, world output = 1."""

    countries: tuple[str, ...]
    pi: ArrayF  # N x N, columns sum to one
    Y: ArrayF  # N, sums to one

    def __post_init__(self) -> None:
        pi = np.asarray(self.pi, dtype=float)
        Y = np.asarray(self.Y, dtype=float)
        n = len(self.countries)
        if pi.shape != (n, n) or Y.shape != (n,):
            raise ValueError("pi must be N x N and Y of length N")
        if np.any(pi <= 0):
            raise ValueError("Assumption 1 requires all expenditure shares > 0")
        if not np.allclose(pi.sum(axis=0), 1.0, atol=1e-10):
            raise ValueError("expenditure shares must sum to one by importer")
        object.__setattr__(self, "pi", pi)
        object.__setattr__(self, "Y", Y / Y.sum())

    @property
    def n(self) -> int:
        return len(self.countries)

    @property
    def X(self) -> ArrayF:
        """Baseline flows in units of world output (balanced: E = Y)."""
        return self.pi * self.Y[None, :]

    @property
    def domestic_share(self) -> ArrayF:
        return np.diag(self.pi).copy()

    def index(self, country: str) -> int:
        return self.countries.index(country)

    def is_balanced(self, tol: float = 1e-9) -> bool:
        return bool(np.allclose(self.X.sum(axis=1), self.Y, atol=tol))

    @classmethod
    def from_flows(cls, countries: Sequence[str], X: ArrayF, sigma: float = 5.0,
                   balance: bool = True) -> "Baseline":
        """Build a baseline from a flow matrix.

        With ``balance=True`` trade imbalances are removed following
        Dekle, Eaton and Kortum (2007), as in the paper (Online Appendix B.3).
        """
        X = np.asarray(X, dtype=float)
        countries = tuple(countries)
        if balance:
            return balance_trade(countries, X, sigma)
        Y = X.sum(axis=1)
        E = X.sum(axis=0)
        if not np.allclose(Y, E, rtol=1e-9):
            raise ValueError("flows are unbalanced; use balance=True")
        return cls(countries, X / E[None, :], Y / Y.sum())


@dataclass
class Equilibrium:
    """Counterfactual equilibrium relative to a baseline."""

    baseline: Baseline
    sigma: float
    p_hat: ArrayF
    tau_hat: ArrayF  # N x N trade cost changes
    shares: ArrayF = field(repr=False)  # counterfactual expenditure shares
    P_hat: ArrayF = field(repr=False)

    @property
    def W_hat(self) -> ArrayF:
        """Welfare = real income, W = p / P (balanced trade)."""
        return self.p_hat / self.P_hat

    @property
    def X(self) -> ArrayF:
        """Counterfactual flows in units of baseline world output."""
        return self.shares * (self.baseline.Y * self.p_hat)[None, :]

    @property
    def exports_hat(self) -> ArrayF:
        Xb, Xc = self.baseline.X, self.X
        off = ~np.eye(self.baseline.n, dtype=bool)
        return (Xc * off).sum(axis=1) / (Xb * off).sum(axis=1)

    @property
    def domestic_sales_hat(self) -> ArrayF:
        return np.diag(self.X) / np.diag(self.baseline.X)

    def table(self) -> "dict[str, ArrayF]":
        """Percentage changes as in Tables C1-C3 of the paper."""
        pct = lambda a: (np.asarray(a) - 1.0) * 100.0  # noqa: E731
        return {
            "country": np.array(self.baseline.countries),
            "producer_prices": pct(self.p_hat),
            "consumer_prices": pct(self.P_hat),
            "total_exports": pct(self.exports_hat),
            "domestic_sales": pct(self.domestic_sales_hat),
            "welfare": pct(self.W_hat),
        }

    def as_baseline(self) -> Baseline:
        """Use this equilibrium as a new (balanced) baseline."""
        X = self.X
        return Baseline(self.baseline.countries, X / X.sum(axis=0)[None, :],
                        X.sum(axis=1))


# --------------------------------------------------------------------------
# Generic solver for Z(p_hat) = 0 on the simplex sum(Y p) = 1
# --------------------------------------------------------------------------

def solve_market_clearing(Y: ArrayF, shares_fn: Callable[[ArrayF], ArrayF],
                          x0: ArrayF | None = None, tol: float = 1e-13) -> ArrayF:
    """Solve ``Y_i p_i = sum_j a_ij(p) Y_j p_j`` with ``sum_i Y_i p_i = 1``.

    Unknowns are log prices. One market-clearing equation is redundant by
    Walras' law and is replaced by the numeraire condition, which makes the
    Jacobian non-singular (Appendix A.2, local solvability).
    """
    n = len(Y)

    def F(x: ArrayF) -> ArrayF:
        p = np.exp(x)
        a = shares_fn(p)
        income = Y * p
        z = (a @ income - income) / Y
        out = np.empty(n)
        out[:-1] = z[:-1]
        out[-1] = income.sum() - 1.0
        return out

    x0 = np.zeros(n) if x0 is None else np.log(x0)
    last = None
    for method in ("hybr", "lm"):
        sol = root(F, x0, method=method, tol=tol)
        last = sol
        if np.max(np.abs(F(sol.x))) < 1e-10:
            return np.exp(sol.x)
    raise SolverError(f"market clearing did not converge: {last.message}")


def cost_shares(pi: ArrayF, p_hat: ArrayF, tau_hat: ArrayF, sigma: float):
    """Counterfactual shares and price index changes, equation (2)."""
    terms = pi * (p_hat[:, None] * tau_hat) ** (1.0 - sigma)
    denom = terms.sum(axis=0)
    return terms / denom[None, :], denom ** (1.0 / (1.0 - sigma))


def solve_forward(baseline: Baseline, tau_hat: ArrayF, sigma: float = 5.0) -> Equilibrium:
    """Forward problem: given trade cost changes, solve (2)-(3) for prices."""
    tau_hat = np.asarray(tau_hat, dtype=float)
    if tau_hat.shape != (baseline.n, baseline.n) or np.any(tau_hat <= 0):
        raise ValueError("tau_hat must be a positive N x N matrix")
    p = solve_market_clearing(
        baseline.Y, lambda p: cost_shares(baseline.pi, p, tau_hat, sigma)[0])
    a, P = cost_shares(baseline.pi, p, tau_hat, sigma)
    return Equilibrium(baseline, sigma, p, tau_hat, a, P)


def balance_trade(countries: Sequence[str], X: ArrayF, sigma: float = 5.0) -> Baseline:
    """Remove trade imbalances (Dekle, Eaton and Kortum 2007).

    Observed shares ``pi_ij = X_ij / E_j`` and output ``Y_i`` are taken from the
    data. With unchanged trade costs and expenditure set equal to income,
    ``Y_i p_i = sum_j pi_ij(p) Y_j p_j`` is solved for ``p`` and the resulting
    flows ``pi_ij(p) Y_j p_j`` form the balanced baseline.
    """
    X = np.asarray(X, dtype=float)
    Y = X.sum(axis=1)
    E = X.sum(axis=0)
    total = Y.sum()
    Y, pi = Y / total, X / E[None, :]
    ones = np.ones_like(pi)
    p = solve_market_clearing(Y, lambda p: cost_shares(pi, p, ones, sigma)[0])
    a, _ = cost_shares(pi, p, ones, sigma)
    Xb = a * (Y * p)[None, :]
    return Baseline(tuple(countries), Xb / Xb.sum(axis=0)[None, :], Xb.sum(axis=1))


def bilateral_tau(n: int, changes: dict[tuple[int, int], float] | None = None) -> ArrayF:
    """Convenience: an N x N matrix of ones with selected entries changed."""
    t = np.ones((n, n))
    for (i, j), v in (changes or {}).items():
        t[i, j] = v
    return t
