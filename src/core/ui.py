from __future__ import annotations

from typing import Dict, Optional

import pandas as pd
import streamlit as st

from src.core.config import AppConfig


def app_header(config: AppConfig, page_title: str, subtitle: str = None, icon: str = None) -> None:
    """Render gradient header for all pages with optional subtitle and icon."""
    st.set_page_config(page_title=f"{config.title} — {page_title}", layout="wide")
    
    # Determine icon based on page title if not provided
    if icon is None:
        icon_map = {
            "DS Assistant": "🤖",
            "Data Exploration": "🏠",
            "Data Analysis": "📊",
            "Tabular Machine Learning": "🎯",
            "Deep Learning": "⏰",
            "Visualization": "🎨",
            "Prediction": "🎯",
            "AI Insights": "💡",
            "Model Management": "📦",
            "Settings": "⚙️",
            "Data Science Academy": "🎓",
            "Feature Engineering": "🔨",
            "Data Visualization Academy": "🎨",
            "Data Cleaning": "🧼",
        }
        for key, emoji in icon_map.items():
            if key in page_title:
                icon = emoji
                break
        if icon is None:
            icon = "📊"
    
    # Default subtitle if not provided
    if subtitle is None:
        subtitle = "Your Complete Data Science Workspace"
    
    # Gradient header
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 25px;
                    border-radius: 15px;
                    margin-bottom: 20px;
                    box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3);">
            <h1 style="color: white; margin: 0; font-size: 32px; font-weight: 800;">
                {icon} DataScope Pro - {page_title}
            </h1>
            <p style="color: rgba(255,255,255,0.95); margin: 8px 0 0 0; font-size: 16px;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def instruction_block(title: str, lines: list[str], expanded: bool = False) -> None:
    """Render a collapsible instructions panel with bullet points."""
    with st.expander(title, expanded=expanded):
        st.markdown("\n".join(f"- {line}" for line in lines))


def sidebar_dataset_status(df: Optional[pd.DataFrame], clean_df: Optional[pd.DataFrame]) -> None:
    with st.sidebar:
        st.subheader("Session")
        if df is None:
            st.caption("Raw dataset: not loaded")
        else:
            st.caption(f"Raw dataset: {df.shape[0]:,} rows × {df.shape[1]:,} cols")

        if clean_df is None:
            st.caption("Clean dataset: not created")
        else:
            st.caption(f"Clean dataset: {clean_df.shape[0]:,} rows × {clean_df.shape[1]:,} cols")

        st.divider()
        st.caption("Tip: Use Data Cleaning → save cleaned data to session.")


def page_navigation(current_page: str) -> None:
    """Render next/previous page navigation buttons at the bottom of the page.
    
    Args:
        current_page: Current page number as string (e.g., "1", "2", "3", "10")
    """
    pages = {
        "0": ("app", "🏠 Home"),
        "1": ("1_DS_Assistant", "🤖 DS Assistant"),
        "2": ("2_Data_Analysis_EDA", "📊 Data Analysis / EDA"),
        "3": ("3_Data_Cleaning", "🧼 Data Cleaning"),
        "4": ("4_Feature_Engineering", "🔨 Feature Engineering"),
        "5": ("5_Tabular_Machine_Learning", "🎯 Tabular Machine Learning"),
        "6": ("6_Deep_Learning", "⏰ Deep Learning"),
        "7": ("7_Visualization", "🎨 Visualization"),
        "8": ("8_Viz_Journal", "📓 Visualization Journal"),
        "9": ("9_Prediction", "🎯 Prediction & Inference"),
        "10": ("10_AI_Insights", "💡 AI Insights"),
        "11": ("11_Model_Management", "📦 Model Management"),
        "12": ("12_DS_Academy", "🎓 DS Academy"),
        "13": ("13_Settings", "⚙️ Settings"),
    }
    
    page_order = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13"]
    
    if current_page not in page_order:
        return
    
    current_idx = page_order.index(current_page)
    prev_page = page_order[current_idx - 1] if current_idx > 0 else None
    next_page = page_order[current_idx + 1] if current_idx < len(page_order) - 1 else None
    
    st.divider()
    
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Previous button
    if prev_page:
        prev_file, prev_label = pages[prev_page]
        with col1:
            if st.button(f"⬅️ Previous: {prev_label}", key=f"prev_page_{current_page}", use_container_width=True):
                # Home page uses app.py, others use pages/filename.py
                page_path = f"{prev_file}.py" if prev_page == "0" else f"pages/{prev_file}.py"
                st.switch_page(page_path)
    
    # Current page indicator
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 10px;'><b>Page {current_idx} of {len(page_order)-1}</b></div>", unsafe_allow_html=True)
    
    # Next button
    if next_page:
        next_file, next_label = pages[next_page]
        with col3:
            if st.button(f"Next: {next_label} ➡️", key=f"next_page_{current_page}", use_container_width=True):
                # Home page uses app.py, others use pages/filename.py
                page_path = f"{next_file}.py" if next_page == "0" else f"pages/{next_file}.py"
                st.switch_page(page_path)
