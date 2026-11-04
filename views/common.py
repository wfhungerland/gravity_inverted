"""Helpers shared by both views: country names, number wording, cached data."""

from __future__ import annotations

import numpy as np
import streamlit as st

from gravity_inverted import data as gdata

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


def signed(v: float, d: int = 2) -> str:
    """Signed number with a true minus sign: −1.06, +0.09."""
    s = f"{float(v):+.{d}f}"
    return s.replace("-", "−")


def updown(v: float, up: str = "up", down: str = "down") -> str:
    return up if v > 0 else down



@st.cache_resource(show_spinner=False)
def wiod_base(year: int = 2014, sigma: float = 5.0):
    return gdata.wiod_baseline(year, sigma)


@st.cache_resource(show_spinner=False)
def wiod_pair(y0: int, y1: int):
    names, X0 = gdata.wiod_flows(y0)
    _, X1 = gdata.wiod_flows(y1)
    Q = gdata.wiod_real_output_growth(names, y0, y1) if gdata.has_real_output() else None
    return names, X0, X1, Q
