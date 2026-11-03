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


def flows_from_csv(path_or_buffer) -> tuple[tuple[str, ...], np.ndarray]:
    """Read user data: long CSV with columns exporter, importer, value."""
    d = pd.read_csv(path_or_buffer)
    need = {"exporter", "importer", "value"}
    if not need <= set(d.columns):
        raise ValueError(f"CSV needs columns {sorted(need)}")
    m = d.pivot_table(index="exporter", columns="importer", values="value", aggfunc="sum")
    countries = tuple(sorted(set(m.index) | set(m.columns)))
    m = m.reindex(index=countries, columns=countries)
    if m.isna().any().any():
        raise ValueError("every exporter-importer pair, incl. domestic sales, needs a value")
    return countries, m.to_numpy(dtype=float)
