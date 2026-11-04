"""Real gross output growth from the WIOD Socio-Economic Accounts (2016 release).

Input: ``Socio_Economic_Accounts.xlsx`` from https://doi.org/10.34894/PJ2M1C.
Output: ``src/gravity_inverted/data/wiod2016_real_output.csv`` with real gross
output per economy and year, index 2000 = 1.

Method as in the paper's Online Appendix D.5: per country, sectoral gross
output at 2010 prices, GO_2010 x GO_QI_t / 100, summed over sectors with
complete volume indices. The rest of the world is not covered by the accounts;
it gets the average growth of the 43 countries, weighted by their 2000 output
in US dollars (from the WIOD tables).

Usage::

    python scripts/build_sea.py /path/to/Socio_Economic_Accounts.xlsx
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src/gravity_inverted/data/wiod2016_real_output.csv"
sys.path.insert(0, str(ROOT / "src"))


def main(path: str) -> None:
    from gravity_inverted.data import wiod_flows, wiod_years

    d = pd.read_excel(path, sheet_name="DATA")
    years = [c for c in d.columns if isinstance(c, int)]
    go = d[d.variable == "GO"].set_index(["country", "code"])[years]
    qi = d[d.variable == "GO_QI"].set_index(["country", "code"])[years]
    complete = qi.notna().all(axis=1) & go[2010].notna() & (go[2010] > 0)
    real = (qi[complete].mul(go.loc[complete.index[complete], 2010], axis=0) / 100)
    real = real.groupby(level="country").sum()
    index = real.div(real[years[0]], axis=0)

    countries, X0 = wiod_flows(wiod_years()[0])
    y2000 = pd.Series(X0.sum(axis=1), index=countries)
    members = [c for c in index.index if c in y2000.index]
    w = y2000[members] / y2000[members].sum()
    index.loc["ROW"] = index.loc[members].mul(w, axis=0).sum()

    long = index.stack().rename("real_output").reset_index()
    long.columns = ["country", "year", "real_output"]
    long.to_csv(OUT, index=False, float_format="%.8f")
    n_sec = complete.groupby(level="country").sum()
    print(f"wrote {OUT}; sectors used: min {n_sec.min()} ({n_sec.idxmin()}), max {n_sec.max()}")


if __name__ == "__main__":
    main(sys.argv[1])
