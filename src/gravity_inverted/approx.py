"""First-order rules and exact decomposition (Corollary 1)."""

from __future__ import annotations

import numpy as np

from .equilibrium import Baseline


def first_order_domestic(pi_jj: float, W: float) -> float:
    """Corollary 1(b): ln tau_jj ~ -ln W / pi_jj (unchanged producer prices)."""
    return float(np.exp(-np.log(W) / pi_jj))


def first_order_import(pi_jj: float, W: float) -> float:
    """Corollary 1(b): ln tau_j ~ -ln W / (1 - pi_jj)."""
    return float(np.exp(-np.log(W) / (1.0 - pi_jj)))


def fixed_price_import(pi_jj: float, W: float, sigma: float) -> float:
    """Corollary 1(c): exact import instrument at unchanged producer prices."""
    return float(((W ** (sigma - 1.0) - pi_jj) / (1.0 - pi_jj)) ** (1.0 / (1.0 - sigma)))


def fixed_price_domestic(pi_jj: float, W: float, sigma: float) -> float:
    """Exact domestic instrument at unchanged producer prices (proof of Cor. 1)."""
    return float(((W ** (sigma - 1.0) - (1.0 - pi_jj)) / pi_jj) ** (1.0 / (1.0 - sigma)))


def direct_term_share(W: float, tau_jj_hat: float) -> float:
    """Share of the direct term -ln W in ln tau_jj, equation (8)."""
    return float(-np.log(W) / np.log(tau_jj_hat))


def first_order_table(baseline: Baseline, W: float = 1.01) -> dict[str, np.ndarray]:
    """First-order columns of Table D1 for every economy."""
    d = baseline.domestic_share
    return {
        "country": np.array(baseline.countries),
        "domestic_share": d,
        "domestic_first_order": (np.exp(-np.log(W) / d) - 1) * 100,
        "import_first_order": (np.exp(-np.log(W) / (1 - d)) - 1) * 100,
    }
