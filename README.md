# gravity_inverted

**What would it take?** How big a change in trade costs does it take to reach
an economic goal? This tool runs the structural gravity model backwards. You
give it a target (a welfare gain, or an observed change in trade) and it
returns the trade cost changes that deliver it.

Live app: <https://gravity-inverted.streamlit.app> · expert view:
<https://gravity-inverted.streamlit.app/?view=expert>

It implements Anderson, J. E., Larch, M. and Yotov, Y. V. (2026),
*Inverted Gravity*, NBER Working Paper 35897,
<https://www.nber.org/papers/w35897>.

## What it answers

| Question | Paper | Module |
|---|---|---|
| By how much must **domestic trade costs** (infrastructure, regulation) fall to raise real income by X %? | Prop. 1, eq. (5) | `inverse.py` |
| By how much must **all import trade costs** fall for the same gain? | Prop. 2, eq. (6) | `inverse.py` |
| Same, for several countries at once (joint targets) | Props. 1–2 | `inverse.py` |
| What do the first-order rules say? | Corollary 1 | `approx.py` |
| Effects of a trade cost shock (forward model) | eqs. (2)–(3) | `equilibrium.py` |
| Which trade cost changes explain observed trade between two years? | Lemma 2, eqs. (10)–(11) | `trade_flows.py` |

Key results from the paper the tool reproduces: a 1 % welfare gain needs a
domestic trade cost cut of about 1–1.6 %, but an import cost cut of 2.5–19 %.
Welfare-equivalent instruments have opposite effects on trade volumes and very
different spillovers to partners.

## Quick start

```bash
pip install -e ".[app,dev]"
streamlit run app.py
```

The app opens in a plain-language view for lay readers and ministers, laid
out like the [Konjunkturmonitor](https://wfhungerland.github.io/konjunkturmonitor/):
three questions (get richer, trade walls, since 2000), one big number and one
answer sentence each, ranges for other elasticities, CSV download, and a link
in the address bar that reproduces every setting. `?view=expert` opens all
tables, formulas, mixed instruments, the elasticity, real-output deflation and
own data.

```python
from gravity_inverted import WelfareTarget, solve_welfare_targets
from gravity_inverted.data import wiod_baseline

base = wiod_baseline(2014, sigma=5)
res = solve_welfare_targets(base, WelfareTarget("DEU", 1.01, "domestic"), sigma=5)
res.instrument_pct()        # {'DEU': -1.13}
res.equilibrium.table()     # prices, exports, domestic sales, welfare for all economies
```

## Data

WIOD 2016 release (Timmer et al. 2015), 43 economies plus rest of world,
2000–2014, sectors summed to total bilateral flows. Real gross output growth comes from the WIOD Socio-Economic Accounts
(`scripts/build_sea.py`), used to separate price from volume growth for
observed trade. See [data/README.md](data/README.md) for the build steps. Own data can be uploaded in the app as a long CSV
(`exporter, importer, value`, including domestic sales).

All welfare experiments start from a baseline without trade imbalances
(Dekle, Eaton and Kortum 2007).

## Verification

`pytest` runs two suites:

* `tests/test_theory.py`: model identities on random economies (extended ACR
  formula, forward/inverse round trips, attainability bounds, Lemma 2
  recovers a known shock).
* `tests/test_paper.py`: reproduces Tables 1, 2, C1, C2, D1, D2, D4 and D6 of
  the paper from WIOD data. Skipped until the WIOD file is built.

## Solver

Fixing the welfare target pins down the importer's price index, so market
clearing becomes a system in producer prices alone (Appendix A.1). It is
solved in log prices with one market-clearing equation replaced by the
numéraire (world output = 1), which keeps the Jacobian non-singular
(Appendix A.2). If the direct solve fails, the solver walks the targets from
the baseline (continuation). Every inverse solution is checked by plugging the
instruments into the forward model.

## Limits

* Instruments are iceberg trade costs. Tariff revenue is not modelled
  (the paper's Online Appendix D.3 shows that revenue can flip the sign of the
  welfare effect for a large economy).
* One sector, no input–output linkages; gross flows are treated as final
  goods, so domestic shares are gross shares.
* The model says nothing about the cost of cutting domestic trade costs.
* For observed trade, the full GE split attributes all nominal output growth
  to producer prices unless real output growth is supplied (`Q_hat`); the app
  uses the WIOD Socio-Economic Accounts for this by default.
* Data end in 2014. Newer world input-output tables (e.g. OECD ICIO) could be
  loaded through the CSV upload in the expert view.

## Feedback

Problems and suggestions: <https://github.com/wfhungerland/gravity_inverted/issues>.

## Citation

See `CITATION.cff`.

Anderson, J. E., Larch, M. and Yotov, Y. V. (2026). *Inverted Gravity*.
NBER Working Paper 35897.

Code: MIT License. This is an independent implementation, not affiliated with
the authors.
