"""Plain-language view for children, lay readers and ministers (KLM).

No symbols, one question per story, one sentence per answer. All numbers come
from the same engine as the expert view (WIOD 2014, sigma = 5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from gravity_inverted import WelfareTarget, invert_trade_flows, solve_forward, solve_welfare_targets
from gravity_inverted import data as gdata

SIGMA = 5.0
YEAR = 2014
PAPER = "https://www.nber.org/papers/w35897"

NAMES = {
    "AUS": "Australia", "AUT": "Austria", "BEL": "Belgium", "BGR": "Bulgaria",
    "BRA": "Brazil", "CAN": "Canada", "CHE": "Switzerland", "CHN": "China",
    "CYP": "Cyprus", "CZE": "Czechia", "DEU": "Germany", "DNK": "Denmark",
    "ESP": "Spain", "EST": "Estonia", "FIN": "Finland", "FRA": "France",
    "GBR": "United Kingdom", "GRC": "Greece", "HRV": "Croatia", "HUN": "Hungary",
    "IDN": "Indonesia", "IND": "India", "IRL": "Ireland", "ITA": "Italy",
    "JPN": "Japan", "KOR": "South Korea", "LTU": "Lithuania", "LUX": "Luxembourg",
    "LVA": "Latvia", "MEX": "Mexico", "MLT": "Malta", "NLD": "Netherlands",
    "NOR": "Norway", "POL": "Poland", "PRT": "Portugal", "ROU": "Romania",
    "ROW": "Rest of the world", "RUS": "Russia", "SVK": "Slovakia", "SVN": "Slovenia",
    "SWE": "Sweden", "TUR": "Türkiye", "TWN": "Taiwan", "USA": "United States",
}


THE = {"USA", "GBR", "NLD", "ROW"}


def name(code: str, article: bool = False) -> str:
    """Country name; with ``article`` adds "the" where English needs it."""
    n = NAMES.get(code, code)
    return f"the {n}" if article and code in THE else n


def span(lo: float, hi: float) -> str:
    """Plain wording for a range of percentage changes."""
    lo, hi = sorted((lo, hi))
    if hi < 0:
        return f"{num(hi)} to {num(lo)} % cheaper"
    if lo > 0:
        return f"{num(lo)} to {num(hi)} % more expensive"
    return f"somewhere between {num(lo)} % cheaper and {num(hi)} % more expensive"


def pct(x):
    return (np.asarray(x) - 1.0) * 100.0


def num(v: float) -> str:
    """Absolute value, rounded for reading: 1.1, 7, 0.02."""
    a = abs(float(v))
    if a >= 10:
        return f"{a:.0f}"
    if a >= 1:
        return f"{a:.1f}"
    if a >= 0.1:
        return f"{a:.2f}"
    return f"{a:.3f}"


def updown(v: float, up: str = "up", down: str = "down") -> str:
    return up if v > 0 else down


# ------------------------------------------------------------ cached math ---
@st.cache_resource(show_spinner=False)
def baseline():
    return gdata.wiod_baseline(YEAR, SIGMA)


@st.cache_data(show_spinner=False)
def richer(code: str, goal_pct: float):
    b = baseline()
    W = 1 + goal_pct / 100
    out = {}
    for kind in ("domestic", "import"):
        r = solve_welfare_targets(b, WelfareTarget(code, W, kind), SIGMA)
        eq = r.equilibrium
        i = b.index(code)
        out[kind] = {
            "cut": r.instrument_pct()[code],
            "exports": pct(eq.exports_hat[i]),
            "home_sales": pct(eq.domestic_sales_hat[i]),
            "others": pd.Series(pct(eq.W_hat), index=b.countries).drop(code),
        }
    return out


@st.cache_data(show_spinner=False)
def wall(code: str, wall_pct: float):
    b = baseline()
    j = b.index(code)
    tau = np.ones((b.n, b.n))
    tau[np.arange(b.n) != j, j] = 1 + wall_pct / 100
    shock = solve_forward(b, tau, SIGMA)
    loss = shock.W_hat[j]
    fix = solve_welfare_targets(b, WelfareTarget(code, 2 - loss), SIGMA)
    return {
        "loss": pct(loss),
        "fix": fix.instrument_pct()[code],
        "others_wall": pd.Series(pct(shock.W_hat), index=b.countries).drop(code),
        "others_fix": pd.Series(pct(fix.equilibrium.W_hat), index=b.countries).drop(code),
    }


@st.cache_data(show_spinner=False)
def since(a: str, b_: str, y0: int, y1: int):
    names, X0 = gdata.wiod_flows(y0)
    _, X1 = gdata.wiod_flows(y1)
    inv = invert_trade_flows(names, X0, X1, SIGMA)
    i, j = names.index(a), names.index(b_)
    share = lambda X, e, m: X[e, m] / X[:, m].sum() * 100  # noqa: E731
    return {
        "share_a_in_b": (share(X0, i, j), share(X1, i, j)),
        "share_b_in_a": (share(X0, j, i), share(X1, j, i)),
        "both": pct(inv.symmetric[i, j]),
        "a_to_b": (inv.pct("mge")[i, j], inv.pct("full_ge")[i, j]),
        "b_to_a": (inv.pct("mge")[j, i], inv.pct("full_ge")[j, i]),
    }


# ------------------------------------------------------------------ pieces ---
def world_map(values: pd.Series, title: str):
    v = values.drop("ROW", errors="ignore")
    df = pd.DataFrame({"iso": v.index, "change": v.values,
                       "country": [name(c) for c in v.index]})
    lim = max(float(np.abs(df.change).max()), 1e-6)
    fig = px.choropleth(df, locations="iso", color="change", hover_name="country",
                        color_continuous_scale="RdBu", range_color=(-lim, lim),
                        labels={"change": "richer / poorer (%)"})
    fig.update_traces(hovertemplate="%{hovertext}: %{z:+.3f} %<extra></extra>")
    fig.update_layout(title=title, margin=dict(l=0, r=0, t=40, b=0), height=380,
                      geo=dict(showframe=False, projection_type="natural earth"))
    return fig


def winners_losers(values: pd.Series, n: int = 3):
    """Text fallback for the map: the biggest winners and losers."""
    v = values.drop("ROW", errors="ignore").sort_values()
    fmt = lambda s: ", ".join(f"{name(c)} ({x:+.2f} %)" for c, x in s.items())  # noqa: E731
    gain = v[v > 0].tail(n)[::-1]
    lose = v[v < 0].head(n)
    st.markdown(f"**Gain most:** {fmt(gain) if len(gain) else 'nobody'}  \n"
                f"**Lose most:** {fmt(lose) if len(lose) else 'nobody'}")


def big_answer(text: str):
    st.markdown(f"<div style='font-size:1.35rem;line-height:1.5;margin:0.5rem 0 1rem'>"
                f"{text}</div>", unsafe_allow_html=True)


def pick(label: str, default: str, key: str, exclude: tuple[str, ...] = ("ROW",)):
    codes = sorted((c for c in baseline().countries if c not in exclude), key=name)
    return st.selectbox(label, codes, index=codes.index(default), format_func=name, key=key)


# ------------------------------------------------------------------ stories --
def story_richer():
    st.header("How can a country get richer through trade?")
    st.write("Pick a country and a goal. The calculator shows two ways to get there.")
    a, b = st.columns(2)
    with a:
        code = pick("Country", "DEU", "r_country")
    with b:
        goal = st.slider("Goal: people can buy this much more with their income (%)",
                         0.5, 5.0, 1.0, 0.5)
    r = richer(code, goal)
    d, m = r["domestic"], r["import"]
    ratio = m["cut"] / d["cut"]
    big_answer(f"To make people in <b>{name(code, True)}</b> <b>{goal:g} % richer</b>, "
               f"{name(code, True)} can either make <b>trade at home {num(d['cut'])} % cheaper</b>, "
               f"or make <b>all imports {num(m['cut'])} % cheaper</b>. "
               f"The import route needs a cut about <b>{ratio:.0f} times bigger</b>.")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Way 1: cheaper trade at home")
        st.metric("Cut in costs of moving goods inside the country", f"−{num(d['cut'])} %")
        st.write("Think of better roads and railways, faster permits, less paperwork.")
        st.write(f"Exports go {updown(d['exports'])} {num(d['exports'])} %, sales at home go "
                 f"{updown(d['home_sales'])} {num(d['home_sales'])} %.")
    with c2:
        st.subheader("Way 2: cheaper imports")
        st.metric("Cut in costs of buying from abroad", f"−{num(m['cut'])} %")
        st.write("Think of lower tariffs, quicker customs, fewer import rules.")
        st.write(f"Exports go {updown(m['exports'])} {num(m['exports'])} %, sales at home go "
                 f"{updown(m['home_sales'])} {num(m['home_sales'])} %.")
    st.write("Both ways make people equally richer, but they change trade in opposite "
             "directions.")
    way = st.radio("Show effect on other countries for", ["Way 1: home", "Way 2: imports"],
                   horizontal=True)
    others = d["others"] if way.startswith("Way 1") else m["others"]
    st.plotly_chart(world_map(others, "Who else gets richer (blue) or poorer (red)?"),
                    width="stretch")
    winners_losers(others)
    st.caption(f"Average for other countries: way 1 {d['others'].mean():+.3f} %, "
               f"way 2 {m['others'].mean():+.3f} %.")


def story_wall():
    st.header("What does a trade wall cost?")
    st.write("A trade wall makes everything bought from abroad more expensive, for "
             "example higher tariffs or stricter border checks.")
    a, b = st.columns(2)
    with a:
        code = pick("Country that builds the wall", "USA", "w_country")
    with b:
        size = st.slider("How much more expensive do imports get? (%)", 10, 100, 50, 10)
    r = wall(code, size)
    worst = r["others_wall"].sort_values().head(2)
    big_answer(f"If <b>{name(code, True)}</b> makes all imports <b>{size} % more expensive</b>, "
               f"its people get <b>{num(r['loss'])} % {updown(r['loss'], 'richer', 'poorer')}</b>. "
               f"Hardest hit abroad: {name(worst.index[0])} ({worst.iloc[0]:+.2f} %) and "
               f"{name(worst.index[1])} ({worst.iloc[1]:+.2f} %).")
    ratio = r["others_wall"].mean() / r["others_fix"].mean()
    st.write(f"**The alternative at home.** {name(code, True)[0].upper() + name(code, True)[1:]} could gain the same amount by "
             f"making trade at home **{num(r['fix'])} % cheaper**. That route hurts other "
             f"countries about **{ratio:.0f} times less** than the wall.")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(world_map(r["others_wall"], "Effect of the wall on others"),
                        width="stretch")
        winners_losers(r["others_wall"])
    with c2:
        st.plotly_chart(world_map(r["others_fix"], "Effect of cheaper trade at home on others"),
                        width="stretch")
        winners_losers(r["others_fix"])
    st.caption("The wall is counted as pure cost. Tariff money that a government collects "
               "is not included; with it, a large country can come out slightly ahead "
               "while partners lose more.")


def story_since():
    st.header("What has changed in trade since 2000?")
    st.write("Pick two countries. The calculator works out how much cheaper or more "
             "expensive it became to trade between them.")
    years = gdata.wiod_years()
    a, b, c = st.columns(3)
    with a:
        x = pick("Country A", "CHN", "s_a", exclude=())
    with b:
        y = pick("Country B", "USA", "s_b", exclude=())
    with c:
        y1 = st.select_slider("Up to year", years[1:], value=years[-1])
    if x == y:
        st.info("Pick two different countries.")
        return
    r = since(x, y, years[0], y1)
    s0, s1 = r["share_a_in_b"]
    big_answer(f"In {years[0]}, buyers in <b>{name(y, True)}</b> spent <b>{s0:.2f} %</b> of their "
               f"money on goods from <b>{name(x, True)}</b>. In {y1} it was <b>{s1:.2f} %</b>. "
               f"Overall, trade between the two got <b>{num(r['both'])} % "
               f"{updown(r['both'], 'more expensive', 'cheaper')}</b>.")
    st.write(f"**Direction matters.** Selling from {name(x, True)} to {name(y, True)} got "
             f"{span(*r['a_to_b'])}. Selling from {name(y, True)} to {name(x, True)} got "
             f"{span(*r['b_to_a'])}.")
    st.caption("All changes are measured against trade inside each country. "
               "The overall number is solid. The split by direction depends on how much "
               "of a country's growth came from higher prices, so we show a range: the "
               "lower end keeps prices fixed, the upper end counts all growth as price "
               "growth.")


def explainers():
    st.divider()
    st.subheader("Good to know")
    with st.expander("What does 'richer' mean here?"):
        st.write("How much people can buy with their income. If prices fall and income "
                 "stays the same, people are richer.")
    with st.expander("What are trade costs?"):
        st.write("Everything that makes it harder to get a product from the maker to the "
                 "buyer: distance, bad roads, tariffs, customs forms, different rules. "
                 "Picture a toll booth on every road a product travels. Trade costs exist "
                 "inside a country too, not only at the border.")
    with st.expander("Why is the home route so much smaller?"):
        st.write("Most of what people buy is made in their own country. A small saving on "
                 "that big part counts for a lot. Imports are a smaller part, so the cut "
                 "has to be bigger to have the same effect.")
    with st.expander("How does the calculator work?"):
        st.write("Most trade models answer: 'if we change this cost, what happens?' This "
                 "one runs the model backwards: 'we want this result, which cost change "
                 "gets us there?' It uses real trade data for 43 countries and the rest of "
                 "the world (2014) and lets all prices worldwide adjust.")
    with st.expander("Can I trust the numbers?"):
        st.write(f"The calculator reproduces the published tables of Anderson, Larch and "
                 f"Yotov (2026), [Inverted Gravity, NBER Working Paper 35897]({PAPER}), and "
                 f"is checked automatically on every change. It shows how big a change has "
                 f"to be. It does not say how to achieve it or what it costs to build roads "
                 f"or cut red tape.")


def render() -> None:
    if not gdata.has_wiod():
        st.error("Trade data is missing. See data/README.md.")
        return
    st.title("What would it take?")
    st.markdown("#### A trade calculator that starts from the goal.")
    t1, t2, t3 = st.tabs(["Get richer", "Trade walls", "Since 2000"])
    with t1:
        story_richer()
    with t2:
        story_wall()
    with t3:
        story_since()
    explainers()
