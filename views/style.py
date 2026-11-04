"""Shared look, following the Konjunkturmonitor house style.

Colours, type and components (question cards, explain boxes, chart cards)
mirror https://wfhungerland.github.io/konjunkturmonitor/. Charts follow the
same chart standard: value axis on the right, light horizontal grid, fixed
colour order, red only for unfavourable outcomes.
"""

from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

DARK, ACCENT, BLUE, WARN = "#002060", "#164a8c", "#0070C0", "#C00000"
PAL = ["#0070C0", "#D86DCD", "#00B0F0", "#51154A", "#47D359", "#7030A0", "#002060"]
GRID, ZERO, MUTED = "#D9D9D9", "#595959", "#3f4752"
FONT = "Source Sans 3, Segoe UI, system-ui, Roboto, Helvetica Neue, Arial, sans-serif"
REPO = "https://github.com/wfhungerland/gravity_inverted"
PAPER = "https://www.nber.org/papers/w35897"

CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">
<style>
:root { --bg:#f7f7f5; --card:#fff; --line:#e2e5e9; --accent:#164a8c; --dark:#002060; --muted:#3f4752; }
.stApp, .stApp p, .stApp label, .stApp li, .stApp input, .stApp button, .stApp h1, .stApp h2, .stApp h3,
.stApp summary p, .stApp [data-baseweb="select"] div {
  font-family: "Source Sans 3","Segoe UI",system-ui,-apple-system,Roboto,"Helvetica Neue",Arial,sans-serif; }
.stApp { background: var(--bg); }
.block-container { max-width: 1180px; padding-top: 1.2rem; }
h1, h2, h3 { color: #000; font-weight: 700; letter-spacing: 0; }
h2 { font-size: 26px !important; scroll-margin-top: 70px; }
a { color: var(--accent); }
.gi-bar { position: sticky; top: 0; z-index: 50; background: var(--card); border: 1px solid var(--line);
  border-radius: 0 0 10px 10px; margin: 0 0 14px; padding: 10px 16px; display: flex; flex-wrap: wrap; gap: 6px 18px; align-items: baseline; }
.gi-bar b { font-size: 19px; } .gi-bar .stand { font-size: 13px; color: var(--muted); }
.gi-bar nav { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 14px; margin-left: auto; }
.gi-lead { font-size: 19px; max-width: 46em; margin: 6px 0 14px; }
.gi-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 14px; margin-bottom: 8px; }
.gi-q { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px 12px;
  display: flex; flex-direction: column; gap: 6px; color: inherit !important; text-decoration: none !important; }
.gi-q:hover { border-color: var(--accent); }
.gi-q .ask { font-size: 17px; font-weight: 700; }
.gi-q .big { display: flex; align-items: baseline; gap: 8px; flex-wrap: wrap; }
.gi-q .num { font-weight: 700; font-size: 34px; line-height: 1; color: var(--dark); }
.gi-q .u { font-size: 15px; color: var(--muted); }
.gi-q .head { font-size: 14.5px; }
.gi-q .per { font-size: 12px; color: var(--muted); }
.gi-verdict { display: inline-block; font-size: 13px; color: var(--dark); background: #e6eef8; border-radius: 10px; padding: 1px 8px; }
.gi-answer { font-size: 18px; margin: 4px 0 10px; max-width: 48em; }
.gi-explain { background: var(--card); border-left: 3px solid var(--accent); padding: 8px 12px; margin: 8px 0 14px;
  max-width: 48em; font-size: 15px; }
.gi-explain b { color: var(--accent); }
.gi-card { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px 8px; margin: 6px 0 10px; }
.gi-card h3 { font-size: 16px !important; margin: 0 0 2px; padding: 0; }
.gi-card .sub, .gi-note { font-size: 13px; line-height: 1.4; color: var(--muted); }
.gi-kpis { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 14px; margin: 10px 0 14px; }
.gi-kpi { padding: 12px 14px; border: 1px solid #e3e6ea; border-radius: 8px; background: #fff; }
.gi-kpi .kn { font-size: 15px; color: var(--muted); }
.gi-kpi .kv { font-size: 30px; font-weight: 600; color: var(--dark); line-height: 1.15; }
.gi-kpi .ks { font-size: 13.5px; color: #595959; }
.gi-part { margin: 56px 0 4px; padding-top: 10px; border-top: 3px solid var(--dark); font-weight: 700; font-size: 17px;
  color: var(--dark); text-transform: uppercase; letter-spacing: .05em; }
.gi-foot { font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); margin-top: 40px; padding-top: 10px; }
div[data-testid="stExpander"] details { background: var(--card); border-radius: 8px; }
@media (max-width: 700px) { .gi-bar nav { margin-left: 0; } h2 { font-size: 22px !important; } .gi-q .num { font-size: 30px; } }
</style>
"""


HIDE_HEADER = """<style>header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 0 !important; }</style>"""


def inject(hide_streamlit_header: bool = False) -> None:
    """Load fonts and house CSS; the plain view also hides Streamlit's top bar."""
    st.markdown(CSS + (HIDE_HEADER if hide_streamlit_header else ""), unsafe_allow_html=True)


def esc(t) -> str:
    return html.escape(str(t))


def header_bar(title: str, stand: str, links: list[tuple[str, str]], top: str = "0") -> None:
    nav = " ".join(f'<a href="{a if a.startswith("?") else "#" + a}" target="_self">{esc(t)}</a>'
                   for t, a in links)
    st.markdown(f'<div class="gi-bar" style="top:{top}"><b>{esc(title)}</b><span class="stand">{esc(stand)}</span>'
                f'<nav>{nav}</nav></div>', unsafe_allow_html=True)


def question_card(ask: str, num: str, unit: str, verdict: str, head: str, per: str,
                  anchor: str) -> str:
    return (f'<a class="gi-q" href="#{anchor}" target="_self"><span class="ask">{esc(ask)}</span>'
            f'<span class="big"><span class="num">{esc(num)}</span><span class="u">{esc(unit)}</span>'
            f'<span class="gi-verdict">{esc(verdict)}</span></span>'
            f'<span class="head">{head}</span><span class="per">{esc(per)}</span></a>')


def cards_grid(cards: list[str]) -> None:
    st.markdown('<div class="gi-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def answer(text_html: str) -> None:
    st.markdown(f'<div class="gi-answer">{text_html}</div>', unsafe_allow_html=True)


def explain(title: str, text_html: str) -> None:
    st.markdown(f'<div class="gi-explain"><b>{esc(title)}</b> {text_html}</div>',
                unsafe_allow_html=True)


def kpis(items: list[tuple[str, str, str]]) -> None:
    body = "".join(f'<div class="gi-kpi"><div class="kn">{esc(n)}</div><div class="kv">{esc(v)}</div>'
                   f'<div class="ks">{s}</div></div>' for n, v, s in items)
    st.markdown(f'<div class="gi-kpis">{body}</div>', unsafe_allow_html=True)


def card_title(title: str, sub: str = "") -> None:
    st.markdown(f'<div class="gi-card" style="margin-bottom:0;border-bottom:0;border-radius:10px 10px 0 0">'
                f'<h3>{esc(title)}</h3><div class="sub">{sub}</div></div>', unsafe_allow_html=True)


def note(text_html: str) -> None:
    st.markdown(f'<div class="gi-note">{text_html}</div>', unsafe_allow_html=True)


def layout(fig: go.Figure, height: int = 340, unit: str = "%") -> go.Figure:
    """Chart standard: value axis right, light horizontal grid, no vertical grid."""
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=28, b=8), paper_bgcolor="#fff",
        plot_bgcolor="#fff", font=dict(family=FONT, size=13, color="#000"), colorway=PAL,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
        hoverlabel=dict(font_family=FONT))
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=ZERO, ticks="")
    fig.update_yaxes(side="right", gridcolor=GRID, zeroline=True, zerolinecolor=ZERO,
                     ticksuffix=f" {unit}" if unit else "")
    return fig


def hbar(labels, values, height: int | None = None, unit: str = "%") -> go.Figure:
    """Horizontal bars: blue for gains, red for losses (red only for bad news)."""
    colors = [BLUE if v >= 0 else WARN for v in values]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", marker_color=colors,
                           hovertemplate="%{y}: %{x:+.3f} %<extra></extra>"))
    h = height or max(220, 26 * len(labels) + 40)
    fig.update_layout(height=h, margin=dict(l=8, r=8, t=10, b=8), paper_bgcolor="#fff",
                      plot_bgcolor="#fff", font=dict(family=FONT, size=13), showlegend=False)
    fig.update_xaxes(side="top", gridcolor=GRID, zeroline=True, zerolinecolor=ZERO,
                     ticksuffix=f" {unit}")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def footer() -> None:
    st.markdown(
        f'<div class="gi-foot">Model: Anderson, Larch &amp; Yotov (2026), '
        f'<a href="{PAPER}">Inverted Gravity, NBER WP 35897</a>. Data: WIOD 2016 release '
        f'(Timmer et al. 2015), CC BY 4.0. Independent implementation, not affiliated with the '
        f'authors. <a href="{REPO}">Code and tests</a> · '
        f'<a href="{REPO}/issues/new">Report a problem or suggest an improvement</a></div>',
        unsafe_allow_html=True)
