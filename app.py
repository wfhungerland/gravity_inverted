"""Streamlit front end for the inverted gravity engine.

Run with:  streamlit run app.py

The default view is written for children, lay readers and ministers.
The expert view keeps all tables, formulas, the elasticity and own data.
"""

import streamlit as st

from views import expert, simple

st.set_page_config(page_title="What would it take?", layout="wide")

st.sidebar.markdown("**What would it take?**")
expert_mode = st.sidebar.toggle("Expert mode", value=False,
                                help="Tables, formulas, elasticity of substitution, own data.")
st.sidebar.caption("Based on Anderson, Larch & Yotov (2026), "
                   "[Inverted Gravity, NBER WP 35897](https://www.nber.org/papers/w35897). "
                   "Data: WIOD 2016 release.")

if expert_mode:
    expert.render()
else:
    simple.render()
