"""Inverted gravity with welfare targets (Section 2.3, Appendix A.1).

Each country in the target set ``S`` picks one instrument:

* ``"domestic"`` (D): a change in its domestic trade costs, tau_jj.
* ``"import"`` (M): a uniform change in all of its import trade costs, tau_j.

Fixing the welfare target pins down the price index, which removes the
instrument from the expenditure shares (A2)/(A3). Market clearing then becomes
a system in producer prices alone. The instruments follow in closed form from
equations (5) and (6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from .equilibrium import (ArrayF, Baseline, Equilibrium, SolverError,
                          cost_shares, solve_forward, solve_market_clearing)

Instrument = Literal["domestic", "import"]


class TargetNotAttainable(ValueError):
    """The welfare target violates Proposition 1 or 2."""


@dataclass(frozen=True)
class WelfareTarget:
    country: str
    W: float  # target welfare ratio, e.g. 1.02 for +2 %
    instrument: Instrument = "domestic"


@dataclass
class InverseResult:
    """Solution of an inverse problem plus the implied forward equilibrium."""

    targets: tuple[WelfareTarget, ...]
    instrument_hat: dict[str, float]  # tau_jj_hat or tau_j_hat per target country
    equilibrium: Equilibrium

    def instrument_pct(self) -> dict[str, float]:
        return {k: (v - 1.0) * 100.0 for k, v in self.instrument_hat.items()}


def import_bound(baseline: Baseline, country: str, sigma: float) -> float:
    """Autarky welfare (pi_jj)^(1/(sigma-1)): Proposition 2, condition (7).

    A target with the import instrument is attainable iff it exceeds this.
    """
    j = baseline.index(country)
    return float(baseline.pi[j, j] ** (1.0 / (sigma - 1.0)))


def domestic_bound_first_order(baseline: Baseline, country: str, sigma: float) -> float:
    """Corollary 1(d): at unchanged producer prices the domestic instrument
    needs W^(sigma-1) > 1 - pi_jj. Exact attainability depends on p_hat
    (Proposition 1) and is checked after solving."""
    j = baseline.index(country)
    return float((1.0 - baseline.pi[j, j]) ** (1.0 / (sigma - 1.0)))


def _reduced_shares(baseline: Baseline, sigma: float, targets: Mapping[int, tuple[float, str]]):
    pi = baseline.pi
    n = baseline.n
    s = 1.0 - sigma
    ones = np.ones((n, n))
    idx = np.arange(n)

    def shares(p: ArrayF) -> ArrayF:
        a, _ = cost_shares(pi, p, ones, sigma)
        for j, (W, kind) in targets.items():
            off = idx != j
            col = np.empty(n)
            if kind == "domestic":  # (A2)
                col[off] = pi[off, j] * (p[off] / p[j]) ** s * W ** s
                col[j] = 1.0 - col[off].sum()
            else:  # (A3)
                sj = 1.0 - pi[j, j] * W ** s
                w = pi[off, j] * p[off] ** s
                col[off] = sj * w / w.sum()
                col[j] = 1.0 - sj
            a[:, j] = col
        return a

    return shares


def _instrument(baseline: Baseline, sigma: float, p: ArrayF, j: int, W: float,
                kind: str) -> float:
    pi = baseline.pi
    s = 1.0 - sigma
    off = np.arange(baseline.n) != j
    foreign = float((pi[off, j] * p[off] ** s).sum())
    if kind == "domestic":  # equation (5)
        bracket = (p[j] ** s * W ** (sigma - 1.0) - foreign) / pi[j, j]
        if bracket <= 0:
            raise TargetNotAttainable(
                f"{baseline.countries[j]}: implied domestic share is not positive "
                "(Proposition 1)")
        return float(bracket ** (1.0 / s) / p[j])
    # equation (6)
    bracket = p[j] ** s * (W ** (sigma - 1.0) - pi[j, j]) / foreign
    if bracket <= 0:
        raise TargetNotAttainable(
            f"{baseline.countries[j]}: target at or below autarky welfare (Proposition 2)")
    return float(bracket ** (1.0 / s))


def solve_welfare_targets(baseline: Baseline, targets: list[WelfareTarget] | WelfareTarget,
                          sigma: float = 5.0, continuation_steps: int = 8,
                          verify: bool = True) -> InverseResult:
    """Find the instrument changes that deliver the welfare targets.

    Unilateral targets pass a single ``WelfareTarget``; joint targets pass a
    list (one instrument per target country, Tinbergen counting).
    """
    if isinstance(targets, WelfareTarget):
        targets = [targets]
    targets = tuple(targets)
    if sigma <= 1:
        raise ValueError("Assumption 1 requires sigma > 1")
    if len({t.country for t in targets}) != len(targets):
        raise ValueError("one target per country")
    idx = {baseline.index(t.country): (float(t.W), t.instrument) for t in targets}
    for t in targets:
        if t.W <= 0:
            raise ValueError("welfare targets must be positive")
        if t.instrument == "import" and t.W <= import_bound(baseline, t.country, sigma):
            raise TargetNotAttainable(
                f"{t.country}: target {t.W:.4f} is at or below autarky welfare "
                f"{import_bound(baseline, t.country, sigma):.4f} (Proposition 2)")
        if t.instrument not in ("domestic", "import"):
            raise ValueError(f"unknown instrument {t.instrument!r}")

    # Continuation from the baseline (all targets = 1, p = 1), as in the
    # existence proof of Lemma A1. Usually one step suffices.
    p = np.ones(baseline.n)
    try:
        p = solve_market_clearing(baseline.Y, _reduced_shares(baseline, sigma, idx))
    except SolverError:
        for k in range(1, continuation_steps + 1):
            t = k / continuation_steps
            step = {j: (1.0 + t * (W - 1.0), kind) for j, (W, kind) in idx.items()}
            p = solve_market_clearing(baseline.Y, _reduced_shares(baseline, sigma, step), x0=p)

    inst = {baseline.countries[j]: _instrument(baseline, sigma, p, j, W, kind)
            for j, (W, kind) in idx.items()}

    tau = np.ones((baseline.n, baseline.n))
    for j, (_, kind) in idx.items():
        v = inst[baseline.countries[j]]
        if kind == "domestic":
            tau[j, j] = v
        else:
            off = np.arange(baseline.n) != j
            tau[off, j] = v

    a, P = cost_shares(baseline.pi, p, tau, sigma)
    eq = Equilibrium(baseline, sigma, p, tau, a, P)
    if verify:
        # The reduced system solution must be a forward equilibrium (A.1).
        fwd = solve_forward(baseline, tau, sigma)
        if not np.allclose(fwd.p_hat, p, rtol=1e-7, atol=1e-10):
            raise SolverError("inverse solution failed the forward round trip")
        for j, (W, _) in idx.items():
            if not np.isclose(fwd.W_hat[j], W, rtol=1e-8):
                raise SolverError("welfare target missed in forward round trip")
    return InverseResult(targets, inst, eq)
