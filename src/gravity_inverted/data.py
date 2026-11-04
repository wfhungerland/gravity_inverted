"""Bundled data: WIOD 2016 release aggregated to bilateral country flows."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from .equilibrium import Baseline

DATA_FILE = Path(__file__).parent / "data" / "wiod2016_flows.csv.gz"


def has_wiod() -> bool:
    return DATA_FILE.exists()


@lru_cache(maxsize=1)
def _long() -> pd.DataFrame:
    if not has_wiod():
        raise FileNotFoundError(
            f"{DATA_FILE} not found; build it with scripts/build_wiod.py")
    return pd.read_csv(DATA_FILE)


def wiod_years() -> list[int]:
    return sorted(_long().year.unique().tolist())


def wiod_flows(year: int) -> tuple[tuple[str, ...], np.ndarray]:
    """Bilateral flow matrix (exporter rows, importer columns), US$ millions."""
    d = _long()
    m = d[d.year == year].pivot(index="exporter", columns="importer", values="value")
    countries = tuple(sorted(m.index))
    return countries, m.loc[list(countries), list(countries)].to_numpy()


def wiod_baseline(year: int = 2014, sigma: float = 5.0) -> Baseline:
    """Balanced baseline as in the paper (Dekle et al. 2007)."""
    countries, X = wiod_flows(year)
    return Baseline.from_flows(countries, X, sigma=sigma, balance=True)


def flows_from_csv(path_or_buffer, floor: bool = False):
    """Read user data: long CSV with columns exporter, importer, value.

    With ``floor=True`` zero or missing flows are replaced by a tiny positive
    value (1e-6 of the smallest positive flow), because Assumption 1 of the
    paper needs strictly positive shares; the number of replaced cells is
    returned as a third element.
    """
    d = pd.read_csv(path_or_buffer)
    need = {"exporter", "importer", "value"}
    if not need <= set(d.columns):
        raise ValueError(f"CSV needs columns {sorted(need)}")
    m = d.pivot_table(index="exporter", columns="importer", values="value", aggfunc="sum")
    countries = tuple(sorted(set(m.index) | set(m.columns)))
    m = m.reindex(index=countries, columns=countries)
    X = m.to_numpy(dtype=float, copy=True)
    if np.any(X < 0):
        raise ValueError("flows must not be negative")
    if np.any(np.isnan(np.diag(X))) or np.any(np.diag(X) <= 0):
        raise ValueError("every economy needs positive domestic sales (exporter == importer)")
    bad = np.isnan(X) | (X <= 0)
    if not floor:
        if bad.any():
            raise ValueError("every exporter-importer pair needs a positive value")
        return countries, X
    X[bad] = 1e-6 * X[~bad].min()
    return countries, X, int(bad.sum())

REAL_FILE = Path(__file__).parent / "data" / "wiod2016_real_output.csv"


def has_real_output() -> bool:
    return REAL_FILE.exists()


@lru_cache(maxsize=1)
def _real() -> pd.DataFrame:
    return pd.read_csv(REAL_FILE).pivot(index="country", columns="year", values="real_output")


def wiod_real_output_growth(countries, year0: int, year1: int) -> np.ndarray:
    """Real gross output growth Q_hat between two years (WIOD SEA, Appendix D.5)."""
    r = _real()
    return (r[year1] / r[year0]).loc[list(countries)].to_numpy()
