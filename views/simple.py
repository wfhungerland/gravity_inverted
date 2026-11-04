"""Plain-language view for children, lay readers and ministers (KLM).

Layout and wording follow the Konjunkturmonitor: a question per section, one
big number, one answer sentence, an explain box, charts in cards. All numbers
come from the same engine as the expert view (WIOD 2014, sigma = 5); the
ranges repeat the calculation for sigma = 3 and 7.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gravity_inverted import data as gdata
from gravity_inverted.scenarios import pair_since, sigma_band, two_routes, wall_and_fix

from . import style
from .common import name, num, signed, span, updown, wiod_base, wiod_pair

SIGMA = 5.0
YEAR = 2014
BAND = (3.0, 5.0, 7.0)


# ------------------------------------------------------------ cached math ---
@st.cache_data(show_spinner=False)
def routes(code: str, goal: float):
    r = two_routes(wiod_base(YEAR, SIGMA), code, goal, SIGMA)
    band = sigma_band(lambda s: wiod_base(YEAR, s),
                      lambda b, s: {k: v.instrument for k, v in two_routes(b, code, goal, s).items()},
                      BAND)
    return r, band


@st.cache_data(show_spinner=False)
def wall(code: str, size: float):
    return wall_and_fix(wiod_base(YEAR, SIGMA), code, size, SIGMA)


@st.cache_data(show_spinner=False)
def pair_path(a: str, b: str, y1: int):
    years = [y for y in gdata.wiod_years() if y <= y1]
    path = []
    for y in years[1:]:
        names, X0, X1, Q = wiod_pair(years[0], y)
        path.append((y, pair_since(names, X0, X1, a, b, SIGMA, Q)))
    return years[0], path


# ------------------------------------------------------------ url state -----
def qp(key: str, default, options=None, cast=str, lo=None, hi=None, step=None):
    """Read a shareable setting from the URL, falling back to the default."""
    try:
        v = cast(st.query_params.get(key, default))
    except (TypeError, ValueError):
        return default
    if options is not None and v not in options:
        return default
    if lo is not None and not (lo <= v <= hi):
        return default
    if step is not None and abs(round((v - lo) / step) * step + lo - v) > 1e-9:
        return default
    return v


def pick(label: str, key: str, default: str, codes: list[str]) -> str:
    start = qp(key, default, codes)
    return st.selectbox(label, codes, index=codes.index(start), format_func=name, key=f"k_{key}")


def rng(band: dict, kind: str) -> str:
    vals = sorted(abs(v[kind]) for v in band.values())
    lo, hi = num(vals[0]), num(vals[-1])
    if lo == hi:
        return f"Hardly depends on the assumptions ({lo} % in all cases)"
    return f"Range under other assumptions: {lo} to {hi} %"


# ------------------------------------------------------------ charts --------
def trade_bars(r) -> go.Figure:
    fig = go.Figure()
    for kind, label, color in (("domestic", "Way 1: home", style.PAL[0]),
                               ("import", "Way 2: imports", style.PAL[1])):
        fig.add_bar(name=label, x=["Exports", "Sales at home"], marker_color=color,
                    y=[r[kind].exports, r[kind].home_sales],
                    hovertemplate="%{x}: %{y:+.2f} %<extra></extra>")
    fig.update_layout(barmode="group")
    return style.layout(fig, 300)


def others_bars(series_by_label: dict[str, pd.Series], n: int = 10,
                colors=(style.PAL[0], style.PAL[1])) -> go.Figure:
    first = next(iter(series_by_label.values()))
    top = first.drop("ROW", errors="ignore").abs().sort_values(ascending=False).head(n).index
    fig = go.Figure()
    for (label, s), color in zip(series_by_label.items(), colors):
        fig.add_bar(name=label, y=[name(c) for c in top], x=s[top].values, orientation="h",
                    marker_color=color, hovertemplate="%{y}: %{x:+.3f} %<extra></extra>")
    fig.update_layout(barmode="group", height=70 + 22 * len(top) * len(series_by_label),
                      margin=dict(l=8, r=8, t=30, b=8), paper_bgcolor="#fff", plot_bgcolor="#fff",
                      font=dict(family=style.FONT, size=13),
                      legend=dict(orientation="h", yanchor="top", y=-0.02, x=0))
    fig.update_xaxes(side="top", gridcolor=style.GRID, zeroline=True, zerolinecolor=style.ZERO,
                     ticksuffix=" %")
    fig.update_yaxes(autorange="reversed")
    return fig


def lines(x, series: dict[str, list[float]], unit: str = "%") -> go.Figure:
    fig = go.Figure()
    for (label, ys), color in zip(series.items(), style.PAL):
        fig.add_scatter(x=x, y=ys, name=label, mode="lines+markers", line=dict(color=color, width=2.5),
                        marker=dict(size=5), hovertemplate="%{x}: %{y:.2f}" + f" {unit}<extra></extra>")
    return style.layout(fig, 300, unit)


# ------------------------------------------------------------ sections ------
def section_richer(codes):
    st.header("How much cheaper must trade get to make a country richer?", anchor="richer")
    a, b = st.columns(2)
    with a:
        code = pick("Country", "c", "DEU", codes)
    with b:
        goal = st.slider("Goal: people can buy this much more with their income (%)", 0.5, 5.0,
                         qp("g", 1.0, None, float, 0.5, 5.0, 0.5), 0.5, key="k_g")
    r, band = routes(code, goal)
    d, m = r["domestic"], r["import"]
    ratio = m.instrument / d.instrument
    style.answer(f"To make people in <b>{name(code, True)}</b> <b>{goal:g} % richer</b>, "
                 f"{name(code, True)} can make <b>trade at home {num(d.instrument)} % cheaper</b> or "
                 f"<b>all imports {num(m.instrument)} % cheaper</b>. The import route needs a cut "
                 f"about <b>{ratio:.0f} times as large</b>.")
    style.kpis([
        ("Way 1: cut in costs of trade at home", f"−{num(d.instrument)} %",
         rng(band, 'domestic')),
        ("Way 2: cut in costs of all imports", f"−{num(m.instrument)} %",
         rng(band, 'import')),
    ])
    style.explain("What does that mean?",
                  f"Way 1 works as if every good made and sold in {name(code, True)} got "
                  f"{num(d.instrument)} % cheaper to deliver: better roads and rail, faster permits, "
                  f"less paperwork. Way 2 works as if every imported good got {num(m.instrument)} % "
                  f"cheaper to deliver: lower tariffs, quicker customs. Most of what people buy is "
                  f"made at home, so a small saving there counts for a lot.")
    c1, c2 = st.columns(2)
    with c1:
        style.card_title("Same gain, opposite effect on trade",
                         f"Change in {name(code, True)}'s exports and sales at home, %")
        st.plotly_chart(trade_bars(r), width="stretch", config={"displayModeBar": False})
    with c2:
        style.card_title("Who else is affected?",
                         "Change in real income of the 10 most affected other countries, %")
        st.plotly_chart(others_bars({"Way 1: home": d.others, "Way 2: imports": m.others}),
                        width="stretch", config={"displayModeBar": False})
    style.note(f"On average, other countries change by {signed(d.others.mean(), 3)} % with way 1 "
               f"and {signed(m.others.mean(), 3)} % with way 2.")
    return {"country": code, "goal": goal, "r": r, "band": band}


def section_walls(codes):
    st.header("What does a trade wall cost, and what is the alternative?", anchor="walls")
    a, b = st.columns(2)
    with a:
        code = pick("Country that builds the wall", "w", "USA", codes)
    with b:
        size = st.slider("All imports become this much more expensive (%)", 10, 100,
                         qp("s", 50, None, int, 10, 100, 10), 10, key="k_s")
    r = wall(code, size)
    worst = r.others_wall.drop("ROW", errors="ignore").sort_values().head(2)
    style.answer(f"If <b>{name(code, True)}</b> makes all imports <b>{size} % more expensive</b>, "
                 f"its people get <b>{num(r.loss)} % {updown(r.loss, 'richer', 'poorer')}</b>. Hardest "
                 f"hit abroad: {name(worst.index[0], True)} ({signed(worst.iloc[0])} %) and "
                 f"{name(worst.index[1], True)} ({signed(worst.iloc[1])} %).")
    style.kpis([
        ("Real income after the wall", f"{signed(r.loss, 1)} %", "Compared with no wall"),
        ("Cut in home trade costs that makes up the loss", f"−{num(r.restore)} %",
         "Applied after the wall is built"),
        ("Other countries, on average", f"{signed(r.others_wall.mean())} %",
         "Real income, because of the wall"),
    ])
    ratio = r.spillover_ratio()
    tail = (f" That hurts other countries about <b>{ratio:.0f} times less</b> than the wall."
            if ratio and ratio > 1 else "")
    style.answer(f"<span style='font-size:16px'><b>The alternative:</b> instead of building the "
                 f"wall, {name(code, True)} could become just as much richer as the wall makes it "
                 f"poorer by cutting home trade costs by <b>{num(r.equivalent)} %</b>.{tail}</span>")
    style.card_title("Effect on the 10 most affected other countries",
                     "Change in real income, %")
    st.plotly_chart(others_bars({"Trade wall": r.others_wall, "Home route instead": r.others_equivalent},
                                colors=(style.WARN, style.PAL[0])),
                    width="stretch", config={"displayModeBar": False})
    style.explain("Not included:",
                  "The wall is counted as pure cost. Money a government collects from tariffs is "
                  "left out; with it, a large country can come out slightly ahead while its "
                  "partners lose more.")
    return {"country": code, "size": size, "r": r}


def section_since(all_codes):
    st.header("How has trade between two countries changed since 2000?", anchor="since")
    years = gdata.wiod_years()
    a, b, c = st.columns(3)
    with a:
        x = pick("Country A", "a", "CHN", all_codes)
    with b:
        y = pick("Country B", "b", "USA", all_codes)
    with c:
        y1 = st.select_slider("Up to year", years[1:], value=qp("y", years[-1], years[1:], int),
                              key="k_y")
    if x == y:
        st.info("Pick two different countries.")
        return None
    y0, path = pair_path(x, y, y1)
    last = path[-1][1]
    s0, s1 = last.share_a_in_b
    style.answer(f"In {y0}, buyers in <b>{name(y, True)}</b> spent <b>{s0:.2f} %</b> of their money "
                 f"on goods from <b>{name(x, True)}</b>; in {y1} it was <b>{s1:.2f} %</b>. Overall, "
                 f"trade between the two got <b>{num(last.symmetric)} % "
                 f"{updown(last.symmetric, 'more expensive', 'cheaper')}</b>.")
    keys = ("mge", "full_real") if "full_real" in last.a_to_b else ("mge", "full_nominal")
    ab = [last.a_to_b[k] for k in keys]
    ba = [last.b_to_a[k] for k in keys]
    style.answer(f"<span style='font-size:16px'><b>Direction matters.</b> Selling from "
                 f"{name(x, True)} to {name(y, True)} got {span(*ab)}. Selling from {name(y, True)} "
                 f"to {name(x, True)} got {span(*ba)}.</span>")
    yrs = [y0] + [p[0] for p in path]
    c1, c2 = st.columns(2)
    with c1:
        style.card_title("Share of spending on the other country's goods", "% of all spending")
        st.plotly_chart(lines(yrs, {
            f"{y} buys from {x}": [path[0][1].share_a_in_b[0]] + [p[1].share_a_in_b[1] for p in path],
            f"{x} buys from {y}": [path[0][1].share_b_in_a[0]] + [p[1].share_b_in_a[1] for p in path],
        }), width="stretch", config={"displayModeBar": False})
    with c2:
        style.card_title("Cost of trading between the two, both directions together",
                         f"Change since {y0}, %")
        st.plotly_chart(lines(yrs, {"Both directions": [0.0] + [p[1].symmetric for p in path]}),
                        width="stretch", config={"displayModeBar": False})
    style.explain("How to read this:",
                  "All changes are measured against trade inside each country. The overall number "
                  "is robust. The split by direction depends on how much of a country's growth came "
                  "from higher prices rather than more output, so we show a range: one end keeps "
                  "prices fixed, the other uses measured real output growth.")
    return {"a": x, "b": y, "y0": y0, "y1": y1, "r": last}


def explainers():
    st.header("Good to know", anchor="know")
    items = {
        "What does 'richer' mean here?":
            "How much people can buy with their income. If prices fall and income stays the "
            "same, people are richer.",
        "What are trade costs?":
            "Everything that makes it harder to get a product from the maker to the buyer: "
            "distance, bad roads, tariffs, customs forms, different rules. Picture a toll booth "
            "on every road a product travels. Trade costs exist inside a country too, not only "
            "at the border.",
        "Is this a forecast?":
            "No. It compares two long-run situations once all prices worldwide have adjusted. It "
            "says nothing about how fast that happens.",
        "Why data from 2014?":
            "The calculator needs consistent data on what every country sells at home and abroad. "
            "The World Input-Output Database (2016 release) provides this for 43 countries and the "
            "rest of the world up to 2014. Shares of home sales change slowly, so the size of the "
            "answers is a good guide, but exact numbers for today would differ.",
        "What does the range mean?":
            "The main number assumes a standard value for how easily buyers switch between "
            "suppliers from different countries. The range repeats the calculation with buyers "
            "who switch less easily and more easily.",
        "How does the calculator work?":
            "Most trade models answer: 'if we change this cost, what happens?' This one runs the "
            "model backwards: 'we want this result, which change in costs gets us there?'",
        "What it does not tell you":
            "How to cut trade costs, what that would cost, or how long it takes. It shows how big "
            "the change has to be. Tariff revenue and links between industries are left out.",
        "Can I trust the numbers?":
            f"The calculator reproduces the published tables of Anderson, Larch and Yotov (2026), "
            f"[Inverted Gravity, NBER Working Paper 35897]({style.PAPER}), and is checked "
            f"automatically on every change. The [code and tests]({style.REPO}) are public.",
    }
    cols = st.columns(2)
    for k, (q, t) in enumerate(items.items()):
        with cols[k % 2].expander(q):
            st.markdown(t)


def summary_csv(richer, walls, since) -> bytes:
    rows = []
    if richer:
        c, g, r, band = richer["country"], richer["goal"], richer["r"], richer["band"]
        for kind, label in (("domestic", "home trade costs"), ("import", "all import costs")):
            rows.append({"question": f"{name(c)}: {g:g} % richer", "result": f"change in {label}",
                         "value_pct": round(r[kind].instrument, 3),
                         "sigma_3": round(band[3.0][kind], 3), "sigma_7": round(band[7.0][kind], 3)})
    if walls:
        c, s, r = walls["country"], walls["size"], walls["r"]
        q = f"{name(c)}: imports {s} % dearer"
        rows += [{"question": q, "result": "own real income", "value_pct": round(r.loss, 3)},
                 {"question": q, "result": "others' real income (avg)", "value_pct": round(r.others_wall.mean(), 4)},
                 {"question": q, "result": "home-trade change restoring income", "value_pct": round(r.restore, 3)},
                 {"question": q, "result": "home-trade change, same-size gain", "value_pct": round(r.equivalent, 3)}]
    if since:
        r = since["r"]
        q = f"{name(since['a'])} and {name(since['b'])}, {since['y0']}-{since['y1']}"
        rows.append({"question": q, "result": "trade costs, both directions", "value_pct": round(r.symmetric, 2)})
        for k, v in r.a_to_b.items():
            rows.append({"question": q, "result": f"{since['a']} to {since['b']} ({k})", "value_pct": round(v, 2)})
        for k, v in r.b_to_a.items():
            rows.append({"question": q, "result": f"{since['b']} to {since['a']} ({k})", "value_pct": round(v, 2)})
    return pd.DataFrame(rows).to_csv(index=False).encode()


def render() -> None:
    style.inject(hide_streamlit_header=True)
    if not gdata.has_wiod():
        st.error("Trade data is missing. See data/README.md.")
        return
    style.header_bar("What would it take?", "Data: 2014 · long-run comparison",
                     [("Get richer", "richer"), ("Trade walls", "walls"), ("Since 2000", "since"),
                      ("Good to know", "know"), ("Expert view", "?view=expert")])
    st.markdown('<div class="gi-lead">Governments set goals: higher incomes, more trade, '
                'fewer barriers. This calculator runs a standard trade model backwards and shows '
                'how big a change in trade costs it takes to reach a goal. Pick your own '
                'countries in each section.</div>', unsafe_allow_html=True)
    top = st.container()

    base = wiod_base(YEAR, SIGMA)
    codes = sorted((c for c in base.countries if c != "ROW"), key=name)
    richer = section_richer(codes)
    walls = section_walls(codes)
    since = section_since(sorted(base.countries, key=name))
    explainers()

    with top:
        cards = []
        r = richer["r"]
        cards.append(style.question_card(
            f"How can {name(richer['country'], True)} get {richer['goal']:g} % richer?",
            f"−{num(r['domestic'].instrument)} %", "trade costs at home",
            f"imports: {r['import'].instrument / r['domestic'].instrument:.0f}× as much",
            f"or cut all import costs by {num(r['import'].instrument)} %", "Goal compared with 2014",
            "richer"))
        w = walls["r"]
        cards.append(style.question_card(
            f"What does a {walls['size']} % import wall cost {name(walls['country'], True)}?",
            f"{signed(w.loss, 1)} %", "real income",
            f"others {signed(w.others_wall.mean())} %",
            f"a {num(w.restore)} % cut in home trade costs makes up the loss",
            "Long-run comparison", "walls"))
        if since:
            s = since["r"]
            cards.append(style.question_card(
                f"{name(since['a'])} and {name(since['b'])}: trade since {since['y0']}?",
                f"{signed(s.symmetric, 0)} %", "trade costs", "both directions",
                f"Share of goods from {name(since['a'])} in spending: "
                f"{s.share_a_in_b[0]:.2f} → {s.share_a_in_b[1]:.2f} % ({name(since['b'], True)})",
                f"{since['y0']} to {since['y1']}", "since"))
        style.cards_grid(cards)

    st.download_button("Download these results (CSV)", summary_csv(richer, walls, since),
                       file_name="what_would_it_take.csv", mime="text/csv")
    style.note("The address bar always holds a link to exactly these settings, ready to share.")
    style.footer()

    st.query_params.update({
        "c": richer["country"], "g": str(richer["goal"]), "w": walls["country"],
        "s": str(walls["size"]),
        **({"a": since["a"], "b": since["b"], "y": str(since["y1"])} if since else {})})
