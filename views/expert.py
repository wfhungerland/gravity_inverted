"""Expert view: full tables, formulas, elasticity, mixed instruments and own data."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gravity_inverted import (Baseline, SolverError, TargetNotAttainable, WelfareTarget,
                              approx, import_bound, invert_trade_flows, solve_welfare_targets)
from gravity_inverted import data as gdata
from gravity_inverted.scenarios import wall_and_fix

from . import style
from .common import name, wiod_base, wiod_pair


def pct(x):
    return (np.asarray(x) - 1.0) * 100.0


@st.cache_resource(show_spinner=False)
def upload_base(raw: bytes, sigma: float):
    countries, X, n_floor = gdata.flows_from_csv(io.BytesIO(raw), floor=True)
    return Baseline.from_flows(countries, X, sigma=sigma, balance=True), n_floor


def outcome_frame(eq, label):
    t = eq.table()
    return pd.DataFrame({"country": t["country"], "instrument": label,
                         "producer_prices": t["producer_prices"],
                         "consumer_prices": t["consumer_prices"],
                         "exports": t["total_exports"], "domestic_sales": t["domestic_sales"],
                         "welfare": t["welfare"]})


def csv_button(df: pd.DataFrame, label: str, file: str, key: str):
    st.download_button(label, df.to_csv(index=False).encode(), file_name=file, mime="text/csv",
                       key=key)


def sidebar():
    st.sidebar.markdown("**Settings**")
    sigma = st.sidebar.slider("Elasticity of substitution σ", 1.5, 10.0, 5.0, 0.5,
                              help="Paper default 5; trade elasticity is σ − 1. The balanced "
                                   "baseline is rebuilt for each σ, so domestic shares move too.")
    sources = (["WIOD 2016 release"] if gdata.has_wiod() else []) + ["Upload CSV"]
    source = st.sidebar.radio("Data", sources)
    base, make = None, None
    if source == "WIOD 2016 release":
        year = int(st.sidebar.selectbox("Baseline year", gdata.wiod_years()[::-1]))
        base = wiod_base(year, sigma)
        make = lambda s: wiod_base(year, s)  # noqa: E731
    else:
        up = st.sidebar.file_uploader(
            "Long CSV: exporter, importer, value (incl. domestic sales)", type="csv")
        if up is not None:
            try:
                raw = up.getvalue()
                base, n_floor = upload_base(raw, sigma)
                make = lambda s: upload_base(raw, s)[0]  # noqa: E731
                if n_floor:
                    st.sidebar.warning(f"{n_floor} zero or missing flows were set to a tiny "
                                       "positive value (Assumption 1 needs positive shares).")
            except (ValueError, SolverError) as e:
                st.sidebar.error(str(e))
    st.sidebar.caption("Trade imbalances are removed before every welfare experiment "
                       "(Dekle, Eaton & Kortum 2007).")
    return sigma, base, make


# ------------------------------------------------------------------ tabs ----
def tab_welfare(base: Baseline, sigma: float, make):
    countries = list(base.countries)
    c1, c2 = st.columns([1, 2])
    with c1:
        dom = st.multiselect("Countries using domestic trade costs τ̂jj", countries,
                             default=["DEU"] if "DEU" in countries else countries[:1],
                             format_func=lambda c: f"{c} · {name(c)}")
        imp = st.multiselect("Countries using uniform import costs τ̂j",
                             [c for c in countries if c not in dom],
                             format_func=lambda c: f"{c} · {name(c)}")
        gain = st.number_input("Welfare target for each (% change in real income)", -10.0, 20.0,
                               1.0, 0.1, format="%.2f",
                               help="Import instrument: target must exceed autarky welfare "
                                    "(Prop. 2). Domestic instrument: losses are bounded by a "
                                    "positive domestic share (Prop. 1).")
        compare = st.checkbox("Also show each country with the other instrument, unilaterally",
                              value=True)
    W = 1 + gain / 100
    targets = [WelfareTarget(c, W, "domestic") for c in dom] + [WelfareTarget(c, W, "import") for c in imp]
    if not targets:
        c2.info("Pick at least one country.")
        return
    with c1:
        for c in imp:
            st.caption(f"{c}: import instrument needs target > autarky welfare "
                       f"{pct(import_bound(base, c, sigma)):.1f} %")
    try:
        res = solve_welfare_targets(base, targets, sigma)
    except (TargetNotAttainable, SolverError) as e:
        c2.warning(str(e))
        return
    rows = []
    for t in targets:
        d = base.domestic_share[base.index(t.country)]
        fo = approx.first_order_domestic(d, W) if t.instrument == "domestic" else approx.first_order_import(d, W)
        rows.append({"country": t.country, "instrument": t.instrument, "set": "joint" if len(targets) > 1 else "unilateral",
                     "required change (%)": res.instrument_pct()[t.country],
                     "first-order rule (%)": float(pct(fo)), "domestic share": d})
        if compare:
            other = "import" if t.instrument == "domestic" else "domestic"
            try:
                r2 = solve_welfare_targets(base, WelfareTarget(t.country, W, other), sigma)
                fo2 = approx.first_order_domestic(d, W) if other == "domestic" else approx.first_order_import(d, W)
                rows.append({"country": t.country, "instrument": other, "set": "unilateral",
                             "required change (%)": r2.instrument_pct()[t.country],
                             "first-order rule (%)": float(pct(fo2)), "domestic share": d})
            except (TargetNotAttainable, SolverError) as e:
                rows.append({"country": t.country, "instrument": other, "set": f"not attainable: {e}"})
    df = pd.DataFrame(rows)
    with c2:
        m = st.columns(min(3, len(targets)))
        for col, t in zip(m * len(targets), targets):
            col.metric(f"{t.country}: {t.instrument} instrument",
                       f"{res.instrument_pct()[t.country]:+.2f} %", delta_color="off",
                       delta=f"first order {float(pct(approx.first_order_domestic(base.domestic_share[base.index(t.country)], W) if t.instrument == 'domestic' else approx.first_order_import(base.domestic_share[base.index(t.country)], W))):+.2f} %")
        st.dataframe(df.round(4), hide_index=True, width="stretch")
        csv_button(df, "Download instruments (CSV)", "instruments.csv", "dl_inst")
    out = outcome_frame(res.equilibrium, "joint" if len(targets) > 1 else targets[0].instrument)
    st.subheader("Full general equilibrium")
    st.dataframe(out.round(4), hide_index=True, width="stretch", height=300)
    csv_button(out, "Download equilibrium (CSV)", "equilibrium.csv", "dl_eq")
    others = out[~out.country.isin([t.country for t in targets])]
    st.caption(f"Average welfare change of other economies: {others.welfare.mean():+.4f} %")

    st.subheader("Sensitivity to σ")
    first = targets[0].country
    sig = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
    rows = []
    for s in sig:
        b = make(s)
        for kind in ("domestic", "import"):
            try:
                v = solve_welfare_targets(b, WelfareTarget(first, W, kind), s).instrument_pct()[first]
            except (TargetNotAttainable, SolverError):
                v = np.nan
            rows.append({"sigma": s, "instrument": kind, "change": v,
                         "domestic share": b.domestic_share[b.index(first)]})
    sens = pd.DataFrame(rows)
    fig = go.Figure()
    for kind, color in (("domestic", style.PAL[0]), ("import", style.PAL[1])):
        d = sens[sens.instrument == kind]
        fig.add_scatter(x=d.sigma, y=d.change, name=f"{kind} instrument", mode="lines+markers",
                        line=dict(color=color, width=2.5))
    fig.update_xaxes(title="σ")
    st.plotly_chart(style.layout(fig, 320), width="stretch")
    st.caption(f"Unilateral instrument for {first}, same target. The baseline is re-balanced "
               f"for each σ, so the domestic share moves from "
               f"{sens['domestic share'].min():.3f} to {sens['domestic share'].max():.3f}.")


def tab_shock(base: Baseline, sigma: float):
    countries = list(base.countries)
    st.markdown("Raise one country's import trade costs (iceberg, no tariff revenue), then look "
                "at two domestic alternatives (Table 2 of the paper).")
    a, b = st.columns(2)
    who = a.selectbox("Country", countries, index=countries.index("USA") if "USA" in countries else 0,
                      format_func=lambda c: f"{c} · {name(c)}")
    shock = b.slider("Increase in all import trade costs (%)", 5, 200, 50, 5)
    try:
        r = wall_and_fix(base, who, shock, sigma)
    except (SolverError, TargetNotAttainable) as e:
        st.error(str(e))
        return
    m1, m2, m3 = st.columns(3)
    m1.metric(f"{who} welfare after shock", f"{r.loss:+.3f} %")
    m2.metric("Domestic change that restores pre-shock welfare", f"{r.restore:+.3f} %",
              help="Solved from the post-shock equilibrium, target 1/Ŵ.")
    m3.metric("Domestic change with a gain of equal size", f"{r.equivalent:+.3f} %",
              help="Solved from the initial baseline, target 2 − Ŵ (paper, Table 2).")
    o1, o2 = r.others_wall.mean(), r.others_equivalent.mean()
    ratio = r.spillover_ratio()
    st.markdown(f"Average welfare change of the other economies: **{o1:+.4f} %** with the barrier, "
                f"**{o2:+.4f} %** with the equal-size domestic instrument"
                + (f" (ratio {ratio:.0f})." if ratio else ". The ratio is not shown because the "
                   "averages are near zero or of opposite sign."))
    df = pd.DataFrame({"country": r.others_wall.index, "barrier": r.others_wall.values,
                       "domestic instead": r.others_equivalent.values}).sort_values("barrier")
    fig = go.Figure()
    fig.add_bar(name="import barrier", x=df.country, y=df.barrier, marker_color=style.WARN)
    fig.add_bar(name="domestic instrument instead", x=df.country, y=df["domestic instead"],
                marker_color=style.PAL[0])
    fig.update_layout(barmode="group")
    st.plotly_chart(style.layout(fig, 360), width="stretch")
    csv_button(df, "Download spillovers (CSV)", "spillovers.csv", "dl_sp")


def tab_observed(sigma: float):
    if not gdata.has_wiod():
        st.info("Needs the WIOD panel (scripts/build_wiod.py).")
        return
    yrs = gdata.wiod_years()
    a, b, c = st.columns(3)
    y0 = a.selectbox("Base year", yrs, index=0)
    y1 = b.selectbox("Comparison year", yrs, index=len(yrs) - 1)
    measure = c.radio("Measure", ["Full GE", "MGE", "Symmetric (Novy)"], horizontal=True)
    if y0 == y1:
        st.info("Pick two different years.")
        return
    prices = st.radio("Producer price growth p̂",
                      ["Nominal output growth, p̂ = Ŷ (paper, main text)",
                       "Net of real output growth, p̂ = Ŷ/Q̂ (Online Appendix D.5)"],
                      index=1 if gdata.has_real_output() else 0,
                      disabled=not gdata.has_real_output())
    names, X0, X1, Q = wiod_pair(y0, y1)
    use_real = prices.startswith("Net") and Q is not None
    inv = invert_trade_flows(names, X0, X1, sigma, Q_hat=Q if use_real else None)
    mat = {"Full GE": inv.full_ge, "MGE": inv.mge, "Symmetric (Novy)": inv.symmetric}[measure]
    z = np.log(mat)
    lim = np.nanpercentile(np.abs(z), 95) or 1.0
    compact = st.toggle("Compact heatmap (phones)", value=False)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(names), y=list(names), zmin=-lim, zmax=lim, colorscale="RdBu_r",
        customdata=pct(mat), hovertemplate="%{y} → %{x}: %{customdata:.1f} %<extra></extra>",
        colorbar=dict(title="log change")))
    fig.update_layout(height=420 if compact else 720, xaxis_title="Importer", yaxis_title="Exporter",
                      yaxis_autorange="reversed", font=dict(family=style.FONT, size=9 if compact else 11),
                      margin=dict(l=8, r=8, t=10, b=8))
    st.plotly_chart(fig, width="stretch")
    long = pd.DataFrame([(names[i], names[j], inv.pct("mge")[i, j], inv.pct("full_ge")[i, j],
                          float(pct(inv.symmetric[i, j])))
                         for i in range(len(names)) for j in range(len(names)) if i != j],
                        columns=["exporter", "importer", "mge_pct", "full_ge_pct", "symmetric_pct"])
    csv_button(long, "Download all pairs (CSV)", f"trade_costs_{y0}_{y1}.csv", "dl_tc")
    e, i = st.columns(2)
    ex = e.selectbox("Exporter", names, index=names.index("CHN") if "CHN" in names else 0)
    im = i.selectbox("Importer", names, index=names.index("USA") if "USA" in names else 1)
    p, q = names.index(ex), names.index(im)
    st.dataframe(pd.DataFrame({
        "measure": ["MGE", "Full GE", "Symmetric component"],
        f"{ex} → {im} (%)": [inv.pct("mge")[p, q], inv.pct("full_ge")[p, q], pct(inv.symmetric[p, q])],
        f"{im} → {ex} (%)": [inv.pct("mge")[q, p], inv.pct("full_ge")[q, p], pct(inv.symmetric[q, p])]}
    ).round(2), hide_index=True)
    st.caption("Domestic trade costs are normalized, so τ̂ij is relative to the importer's domestic "
               "trade costs. Flows in current US dollars; the full GE split depends on p̂, the "
               "symmetric component does not.")


def tab_method():
    st.markdown("**Model.** One-sector Armington endowment economy, CES demand with elasticity σ, "
                "iceberg trade costs. Welfare is real income W = p/P. Counterfactuals start from "
                "a baseline without trade imbalances (Dekle et al. 2007). World output is the "
                "numéraire. All results are long-run comparisons of equilibria.")
    st.markdown("**Extended ACR formula (Lemma 1).** Domestic trade costs enter welfare one for one:")
    st.latex(r"\widehat W_j = \widehat\pi_{jj}^{1/(1-\sigma)} \,/\, \widehat\tau_{jj}")
    st.markdown("**Instruments (eq. 5, 6).** Fixing the target pins down the price index, so market "
                "clearing becomes a system in producer prices alone with a unique solution "
                "(gross substitutability).")
    st.latex(r"\widehat\tau_{jj}=\frac1{\widehat p_j}\left[\frac{\widehat p_j^{1-\sigma}"
             r"\overline W_j^{\sigma-1}-\sum_{i\ne j}\pi^b_{ij}\widehat p_i^{1-\sigma}}"
             r"{\pi^b_{jj}}\right]^{\frac1{1-\sigma}},\qquad \widehat\tau_j=\left[\frac{"
             r"\widehat p_j^{1-\sigma}(\overline W_j^{\sigma-1}-\pi^b_{jj})}{\sum_{i\ne j}"
             r"\pi^b_{ij}\widehat p_i^{1-\sigma}}\right]^{\frac1{1-\sigma}}")
    st.markdown("**First-order rules (Corollary 1).** At unchanged producer prices:")
    st.latex(r"\ln\widehat\tau_{jj}\approx-\frac{\ln\overline W_j}{\pi^b_{jj}},\qquad"
             r"\ln\widehat\tau_j\approx-\frac{\ln\overline W_j}{1-\pi^b_{jj}}")
    st.markdown("**Attainability.** Import instrument: target above autarky welfare "
                r"$(\pi^b_{jj})^{1/(\sigma-1)}$ (Proposition 2). Domestic instrument: any welfare "
                "gain; losses are bounded because the implied domestic share must stay positive "
                "(Proposition 1).")
    st.markdown("**Observed trade (Lemma 2).**")
    st.latex(r"\widehat\tau_{ij}=\frac{\widehat p_j}{\widehat p_i}\left(\frac{\widehat X_{jj}}"
             r"{\widehat X_{ij}}\right)^{\frac1{\sigma-1}}=\frac{\widehat p_j}{\widehat p_i}"
             r"\,\widehat\tau^{MGE}_{ij},\qquad \widehat p_i=\widehat Y_i \text{ or } \widehat Y_i/\widehat Q_i")
    st.markdown("**Limits.** Instruments are iceberg costs, not tariffs with revenue. The model is "
                "silent on the resources needed to cut domestic trade costs. One sector, no "
                "input–output linkages; gross flows are treated as final goods, so domestic shares "
                "are gross shares. Data end in 2014.\n\n"
                "**Verification.** `pytest` reproduces Tables 1, 2, C1, C2, D1, D2, D4 and D6 of the "
                "paper from WIOD data, including the real-output variant of D6.")


def render() -> None:
    style.inject()
    st.markdown("<div style='height:2.5rem'></div>", unsafe_allow_html=True)
    style.header_bar("What would it take? · expert view", "σ, data and full tables in the sidebar",
                     [("Plain view", "?view=plain")], top="3.75rem")
    st.markdown('<div class="gi-lead">Inverted gravity: the trade cost changes that deliver '
                'welfare targets, and those that rationalize observed trade.</div>',
                unsafe_allow_html=True)
    sigma, base, make = sidebar()
    t_w, t_s, t_t, t_m = st.tabs(["Welfare targets", "Shock & compensation", "Observed trade", "Method"])
    with t_w:
        if base is None:
            st.info("Load data in the sidebar to run welfare experiments.")
        else:
            tab_welfare(base, sigma, make)
    with t_s:
        if base is None:
            st.info("Load data in the sidebar to run welfare experiments.")
        else:
            tab_shock(base, sigma)
    with t_t:
        tab_observed(sigma)
    with t_m:
        tab_method()
    style.footer()
