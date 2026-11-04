"""Streamlit front end for the inverted gravity engine.

Run with:  streamlit run app.py

The default view is written for children, lay readers and ministers and
follows the Konjunkturmonitor layout. ``?view=expert`` opens the expert view
with all tables, formulas, the elasticity and own data.
"""

import streamlit as st

from views import expert, simple

st.set_page_config(page_title="What would it take?", layout="wide",
                   initial_sidebar_state="collapsed")

if st.query_params.get("view") == "expert":
    expert.render()
else:
    simple.render()
