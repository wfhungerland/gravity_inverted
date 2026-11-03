"""Aggregate the WIOD 2016 release to country-level bilateral flows.

Input: the R-format world input-output tables of the WIOD 2016 release
(Timmer et al. 2015), ``WIOT{year}_October16_ROW.RData``, from
https://doi.org/10.34894/PJ2M1C (file ``WIOTS_in_R.zip``).

Output: ``src/gravity_inverted/data/wiod2016_flows.csv.gz`` in long format
(year, exporter, importer, value in millions of US dollars).

As in the paper (Section 3), all 56 sectors are summed and gross flows,
including intermediate inputs, are treated as flows of final goods:
X_ij = sum over all sector rows of country i of all intermediate-use and
final-demand columns of country j.

Usage::

    python scripts/build_wiod.py /path/to/unzipped/WIOTS_in_R
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "src/gravity_inverted/data/wiod2016_flows.csv.gz"
COL = re.compile(r"^([A-Z]{3})(\d+)$")


def aggregate(path: Path) -> pd.DataFrame:
    import pyreadr

    df = next(iter(pyreadr.read_r(str(path)).values()))
    year = int(df["Year"].iloc[0])
    rows = df[pd.to_numeric(df["RNr"], errors="coerce").between(1, 56)]
    cols = [c for c in df.columns if (m := COL.match(c)) and 1 <= int(m.group(2)) <= 61]
    body = rows[cols].astype(float)
    body.index = rows["Country"].values
    by_exporter = body.groupby(level=0).sum()
    by_importer = by_exporter.T.groupby(lambda c: COL.match(c).group(1)).sum().T
    long = by_importer.stack().rename("value").reset_index()
    long.columns = ["exporter", "importer", "value"]
    long.insert(0, "year", year)
    return long


def main(folder: str) -> None:
    files = sorted(Path(folder).glob("WIOT*_*ROW.RData"))
    if not files:
        sys.exit(f"no WIOT*.RData files in {folder}")
    data = pd.concat([aggregate(f) for f in files], ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT, index=False, float_format="%.6f", compression="gzip")
    print(f"wrote {OUT} ({data.year.nunique()} years, {data.exporter.nunique()} economies)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
