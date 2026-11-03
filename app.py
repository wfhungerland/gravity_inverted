"""Streamlit front end for the inverted gravity engine.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from gravity_inverted import (Baseline, SolverError, TargetNotAttainable, WelfareTarget,
                              approx, import_bound, invert_trade_flows, solve_forward,
                              solve_welfare_targets)
from gravity_inverted import data as gdata

PAPER = "https://www.nber.org/papers/w35897"
st.set_page_config(page_title="Inverted Gravity", layout="wide")


def pct(x):
    return (np.asarray(x) - 1.0) * 100.0


# ---------------------------------------------------------------- data ----
@st.cache_data(show_spinner=False)
def load_wiod(year: int, sigma: float):
    b = gdata.wiod_baseline(year, sigma)
    return b.countries, b.pi, b.Y


@st.cache_data(show_spinner=False)
def load_upload(raw: bytes, sigma: float):
    import io
    countries, X = gdata.flows_from_csv(io.BytesIO(raw))
    b = Baseline.from_flows(countries, X, sigma=sigma, balance=True)
    return b.countries, b.pi, b.Y


st.sidebar.title("Inverted Gravity")
st.sidebar.caption(f"Anderson, Larch & Yotov (2026), [NBER WP 35897]({PAPER})")
sigma = st.sidebar.slider("Elasticity of substitution σ", 1.5, 10.0, 5.0, 0.5,
                          help="Paper default: 5. Trade elasticity is σ − 1.")

sources = (["WIOD 2016 release"] if gdata.has_wiod() else []) + ["Upload CSV"]
source = st.sidebar.radio("Data", sources)
baseline = None
if source == "WIOD 2016 release":
    year = st.sidebar.selectbox("Baseline year", gdata.wiod_years()[::-1])
    baseline = Baseline(*load_wiod(int(year), sigma))
else:
    up = st.sidebar.file_uploader("Long CSV: exporter, importer, value (incl. domestic sales)",
                                  type="csv")
    if up is not None:
        try:
            baseline = Baseline(*load_upload(up.getvalue(), sigma))
        except (ValueError, SolverError) as e:
            st.sidebar.error(str(e))
st.sidebar.caption("Trade imbalances are removed before every welfare experiment "
                   "(Dekle, Eaton & Kortum 2007).")

st.title("Inverted Gravity")
st.markdown("**Start from the target, not the instrument.** How much must trade costs "
            "change to reach a given welfare gain, and which trade cost changes explain "
            "observed trade?")

tab_w, tab_s, tab_t, tab_m = st.tabs(
    ["Welfare targets", "Shock & compensation", "Observed trade", "Method"])

if baseline is None:
    for t in (tab_w, tab_s):
        t.info("Load data in the sidebar to run welfare experiments.")
countries = list(baseline.countries) if baseline is not None else []


def outcome_frame(eq, label):
    t = eq.table()
    return pd.DataFrame({"country": t["country"], "welfare": t["welfare"],
                         "exports": t["total_exports"],
                         "domestic_sales": t["domestic_sales"], "instrument": label})


# ------------------------------------------------------ welfare targets ----
with tab_w:
    if baseline is not None:
        c1, c2 = st.columns([1, 2])
        with c1:
            joint = st.multiselect("Countries pursuing a target", countries,
                                   default=["DEU"] if "DEU" in countries else countries[:1])
            gain = st.number_input("Welfare target (% change in real income)", -20.0, 50.0,
                                   1.0, 0.1, format="%.2f")
            W = 1.0 + gain / 100.0
        if not joint:
            st.info("Pick at least one country.")
        else:
            rows, eqs, problems = [], {}, []
            for kind, label in (("domestic", "Domestic trade costs τ̂jj"),
                                ("import", "Uniform import costs τ̂j")):
                try:
                    res = solve_welfare_targets(
                        baseline, [WelfareTarget(c, W, kind) for c in joint], sigma)
                    eqs[kind] = res.equilibrium
                    for c in joint:
                        d = baseline.domestic_share[baseline.index(c)]
                        fo = (approx.first_order_domestic(d, W) if kind == "domestic"
                              else approx.first_order_import(d, W))
                        rows.append({"country": c, "instrument": label,
                                     "required change (%)": res.instrument_pct()[c],
                                     "first-order rule (%)": pct(fo),
                                     "domestic share": d})
                except (TargetNotAttainable, SolverError) as e:
                    problems.append(f"{label}: {e}")
            with c1:
                for c in joint:
                    b = import_bound(baseline, c, sigma)
                    st.caption(f"{c}: import instrument needs target > autarky welfare "
                               f"{pct(b):.1f} %")
            with c2:
                for p in problems:
                    st.warning(p)
                if rows:
                    df = pd.DataFrame(rows)
                    for c in joint:
                        sub = df[df.country == c].set_index("instrument")
                        m = st.columns(len(sub))
                        for col, (lab, r) in zip(m, sub.iterrows()):
                            col.metric(f"{c}: {lab}", f"{r['required change (%)']:+.2f} %",
                                       f"first order {r['first-order rule (%)']:+.2f} %",
                                       delta_color="off")
                    st.dataframe(df.round(3), hide_index=True, width="stretch")
                    st.caption("Exact: reduced system of Propositions 1–2. First order: "
                               "Corollary 1(b), unchanged producer prices.")
            if eqs:
                out = pd.concat([outcome_frame(e, k) for k, e in eqs.items()])
                st.subheader("Footprint on trade and spillovers")
                own = out[out.country.isin(joint)].melt(
                    id_vars=["country", "instrument"], value_vars=["exports", "domestic_sales"])
                st.plotly_chart(px.bar(own, x="country", y="value", color="instrument",
                                       facet_col="variable", barmode="group",
                                       labels={"value": "% change"}), width="stretch")
                others = out[~out.country.isin(joint)]
                fig = px.bar(others.sort_values("welfare"), x="country", y="welfare",
                             color="instrument", barmode="group",
                             labels={"welfare": "welfare, % change"})
                st.plotly_chart(fig, width="stretch")
                avg = others.groupby("instrument").welfare.mean()
                st.caption("Average welfare change of other economies: " +
                           ", ".join(f"{k} {v:+.4f} %" for k, v in avg.items()))

# ------------------------------------------------ shock & compensation ----
with tab_s:
    if baseline is not None:
        st.markdown("Raise one country's import trade costs (iceberg, no tariff revenue), "
                    "then find the domestic trade cost cut that offsets the welfare loss "
                    "(Table 2 of the paper).")
        a, b = st.columns(2)
        who = a.selectbox("Country", countries,
                          index=countries.index("USA") if "USA" in countries else 0)
        shock = b.slider("Increase in all import trade costs (%)", 0, 200, 50, 5)
        j = baseline.index(who)
        tau = np.ones((baseline.n, baseline.n))
        tau[np.arange(baseline.n) != j, j] = 1 + shock / 100
        try:
            eq_s = solve_forward(baseline, tau, sigma)
            loss = eq_s.W_hat[j]
            comp = solve_welfare_targets(baseline, WelfareTarget(who, 2 - loss), sigma)
            m1, m2, m3 = st.columns(3)
            m1.metric(f"{who} welfare after shock", f"{pct(loss):+.2f} %")
            m2.metric("Offsetting domestic trade cost change",
                      f"{comp.instrument_pct()[who]:+.2f} %")
            o_s = np.delete(pct(eq_s.W_hat), j).mean()
            o_c = np.delete(pct(comp.equilibrium.W_hat), j).mean()
            m3.metric("Spillover ratio (others' avg. welfare)",
                      f"{o_s / o_c:.0f}×" if o_c else "–")
            df = pd.concat([outcome_frame(eq_s, "import barrier"),
                            outcome_frame(comp.equilibrium, "domestic instrument")])
            st.plotly_chart(px.bar(df[df.country != who].sort_values("welfare"),
                                   x="country", y="welfare", color="instrument",
                                   barmode="group", labels={"welfare": "% change"}),
                            width="stretch")
        except (SolverError, TargetNotAttainable) as e:
            st.error(str(e))

# ------------------------------------------------------- observed trade ----
with tab_t:
    if not gdata.has_wiod():
        st.info("Needs the WIOD panel (scripts/build_wiod.py).")
    else:
        yrs = gdata.wiod_years()
        a, b, c = st.columns(3)
        y0 = a.selectbox("Base year", yrs, index=0)
        y1 = b.selectbox("Comparison year", yrs, index=len(yrs) - 1)
        measure = c.radio("Measure", ["Full GE", "MGE", "Symmetric (Novy)"], horizontal=True)
        names, X0 = gdata.wiod_flows(y0)
        _, X1 = gdata.wiod_flows(y1)
        inv = invert_trade_flows(names, X0, X1, sigma)
        mat = {"Full GE": inv.full_ge, "MGE": inv.mge,
               "Symmetric (Novy)": inv.symmetric}[measure]
        z = np.log(mat)
        lim = np.nanpercentile(np.abs(z), 95)
        fig = go.Figure(go.Heatmap(
            z=z, x=names, y=names, zmin=-lim, zmax=lim, colorscale="RdBu_r",
            customdata=pct(mat), hovertemplate="%{y} → %{x}: %{customdata:.1f} %<extra></extra>",
            colorbar=dict(title="log change")))
        fig.update_layout(height=750, xaxis_title="Importer", yaxis_title="Exporter",
                          yaxis_autorange="reversed")
        st.plotly_chart(fig, width="stretch")
        e, i = st.columns(2)
        ex = e.selectbox("Exporter", names, index=names.index("CHN") if "CHN" in names else 0)
        im = i.selectbox("Importer", names, index=names.index("USA") if "USA" in names else 1)
        p, q = names.index(ex), names.index(im)
        st.dataframe(pd.DataFrame({
            "measure": ["MGE", "Full GE", "Symmetric component"],
            f"{ex} → {im} (%)": [inv.pct("mge")[p, q], inv.pct("full_ge")[p, q],
                                 pct(inv.symmetric[p, q])],
            f"{im} → {ex} (%)": [inv.pct("mge")[q, p], inv.pct("full_ge")[q, p],
                                 pct(inv.symmetric[q, p])]}).round(2), hide_index=True)
        st.caption("Full GE attributes all nominal output growth to producer prices "
                   "(p̂ = Ŷ). Domestic trade costs are normalized, so τ̂ij is relative to "
                   "the importer's domestic trade costs.")

# ---------------------------------------------------------------- method ----
with tab_m:
    st.markdown(rf"""
**Model.** One-sector Armington endowment economy, CES demand with elasticity σ,
iceberg trade costs τ<sub>ij</sub>. Welfare is real income W = p/P. Counterfactuals start
from a baseline without trade imbalances (Dekle et al. 2007). World output is the numéraire.

**Extended ACR formula (Lemma 1).** Domestic trade costs enter welfare one for one:
""", unsafe_allow_html=True)
    st.latex(r"\widehat W_j = \widehat\pi_{jj}^{1/(1-\sigma)} \,/\, \widehat\tau_{jj}")
    st.markdown("**Instruments (eq. 5, 6).** Fixing the target pins down the price index, "
                "so market clearing becomes a system in producer prices alone. "
                "It has a unique solution (gross substitutability).")
    st.latex(r"\widehat\tau_{jj}=\frac1{\widehat p_j}\left[\frac{\widehat p_j^{1-\sigma}"
             r"\overline W_j^{\sigma-1}-\sum_{i\ne j}\pi^b_{ij}\widehat p_i^{1-\sigma}}"
             r"{\pi^b_{jj}}\right]^{\frac1{1-\sigma}},\qquad \widehat\tau_j=\left[\frac{"
             r"\widehat p_j^{1-\sigma}(\overline W_j^{\sigma-1}-\pi^b_{jj})}{\sum_{i\ne j}"
             r"\pi^b_{ij}\widehat p_i^{1-\sigma}}\right]^{\frac1{1-\sigma}}")
    st.markdown("**First-order rules (Corollary 1).** At unchanged producer prices:")
    st.latex(r"\ln\widehat\tau_{jj}\approx-\frac{\ln\overline W_j}{\pi^b_{jj}},\qquad"
             r"\ln\widehat\tau_j\approx-\frac{\ln\overline W_j}{1-\pi^b_{jj}}")
    st.markdown("**Attainability.** Import instrument: target above autarky welfare "
                r"$(\pi^b_{jj})^{1/(\sigma-1)}$ (Proposition 2). Domestic instrument: any "
                "welfare gain; losses are bounded because the implied domestic share must "
                "stay positive (Proposition 1).")
    st.markdown("**Observed trade (Lemma 2).**")
    st.latex(r"\widehat\tau_{ij}=\frac{\widehat Y_j}{\widehat Y_i}\left(\frac{\widehat X_{jj}}"
             r"{\widehat X_{ij}}\right)^{\frac1{\sigma-1}}=\frac{\widehat Y_j}{\widehat Y_i}"
             r"\,\widehat\tau^{MGE}_{ij}")
    st.markdown("""
**Limits.** Instruments are iceberg costs, not tariffs with revenue. The model is silent on
the resources needed to cut domestic trade costs. One sector, no input–output linkages;
gross flows are treated as final goods, so domestic shares are gross shares.

**Verification.** `pytest` reproduces the paper's tables (C1, C2, D1, D2, D4, D6, Tables 1–2)
from the WIOD data.
""")
