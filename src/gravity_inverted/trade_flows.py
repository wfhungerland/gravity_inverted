"""Inverted gravity with trade-flow targets (Section 2.5, Lemma 2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ArrayF = np.ndarray


@dataclass
class TradeCostInversion:
    countries: tuple[str, ...]
    mge: ArrayF  # tau_ij^MGE hat, producer prices fixed
    full_ge: ArrayF  # tau_ij hat, full general equilibrium
    p_hat: ArrayF  # producer price changes used for the directional split

    @property
    def symmetric(self) -> ArrayF:
        """Novy (2013) index in changes, equation (11); identical for both."""
        return np.sqrt(self.mge * self.mge.T)

    def pct(self, which: str = "full_ge") -> ArrayF:
        return (getattr(self, which) - 1.0) * 100.0


def invert_trade_flows(countries, X_base: ArrayF, X_comp: ArrayF, sigma: float = 5.0,
                       Q_hat: ArrayF | None = None) -> TradeCostInversion:
    """Trade cost changes that rationalize observed flows, equation (10).

    ``X_base`` and ``X_comp`` are flow matrices (exporter rows, importer
    columns, domestic sales on the diagonal) for the base and comparison year.
    Domestic trade costs are normalized, tau_jj_hat = 1. Imbalances are fine.

    Without ``Q_hat`` all nominal output growth is attributed to producer
    prices, p_hat = Y_hat (endowment economy). With real output growth
    ``Q_hat``, p_hat = Y_hat / Q_hat (Online Appendix D.5).
    """
    Xb = np.asarray(X_base, dtype=float)
    Xc = np.asarray(X_comp, dtype=float)
    if Xb.shape != Xc.shape or Xb.shape[0] != Xb.shape[1]:
        raise ValueError("flow matrices must be square and of equal shape")
    if np.any(Xb <= 0) or np.any(Xc <= 0):
        raise ValueError("all flows must be strictly positive")
    X_hat = Xc / Xb
    Y_hat = Xc.sum(axis=1) / Xb.sum(axis=1)
    p_hat = Y_hat if Q_hat is None else Y_hat / np.asarray(Q_hat, dtype=float)
    mge = (np.diag(X_hat)[None, :] / X_hat) ** (1.0 / (sigma - 1.0))
    full = (p_hat[None, :] / p_hat[:, None]) * mge
    return TradeCostInversion(tuple(countries), mge, full, p_hat)
