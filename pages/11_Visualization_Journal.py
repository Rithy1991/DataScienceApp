from __future__ import annotations

import streamlit as st
from src.core.ui import render_footer

# Minimal bridge page to keep older switch_page targets working.
st.set_page_config(page_title="DataScope Pro - Visualization Journal", layout="wide", initial_sidebar_state="expanded")

try:
    st.switch_page("pages/8_Viz_Journal.py")
except Exception:
    st.error("Visualization Journal page is unavailable. Please open 'Visualization Journal' from the sidebar.")

render_footer()
