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
    
    # Gradient header with consistent dimensions
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 18px 20px;
                    border-radius: 12px;
                    margin-bottom: 16px;
                    box-shadow: 0 10px 40px rgba(240, 147, 251, 0.3);">
            <div style="color: white; font-size: 24px; font-weight: 800; margin: 0;">
                {icon} DataScope Pro - {page_title}
            </div>
            <div style="color: rgba(255,255,255,0.95); font-size: 15px; opacity: 0.95; margin-top: 6px;">
                {subtitle}
            </div>
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
        st.subheader("Session & Progress")
        if df is None:
            st.caption("Raw dataset: not loaded")
        else:
            st.caption(f"Raw dataset: {df.shape[0]:,} rows × {df.shape[1]:,} cols")

        if clean_df is None:
            st.caption("Clean dataset: not created")
        else:
            st.caption(f"Clean dataset: {clean_df.shape[0]:,} rows × {clean_df.shape[1]:,} cols")

        # Visual progress indicator
        if df is not None and clean_df is None:
            st.progress(0.33, text="📊 Data Loaded — Next: Clean")
        elif df is not None and clean_df is not None:
            st.progress(0.67, text="✅ Data Cleaned — Next: Feature Engineering")
        else:
            st.progress(0.0, text="🏠 Start by Loading Data")

        st.divider()
        # Beginner vs Advanced mode toggle (persisted in session)
        if "ui_mode" not in st.session_state:
            st.session_state.ui_mode = "Beginner"
        st.selectbox(
            "Interface Mode",
            options=["Beginner", "Advanced"],
            index=0 if st.session_state.ui_mode == "Beginner" else 1,
            key="ui_mode",
            help="Beginner mode simplifies options and adds explanations. Advanced shows full controls.",
        )
        st.caption("💡 **Tip:** Load data → Clean → Feature Engineer → Train Model")


def _curated_sidebar_menu() -> None:
    """Render a curated sidebar menu in DS→ML learning order.
    Uses explicit navigation to ensure consistent ordering regardless of filename prefixes.
    """
    menu_items = [
        ("app.py", "🏠 Home"),
        ("pages/1_DS_Assistant.py", "🤖 DS Assistant / Workflow"),
        ("pages/2_Data_Analysis_EDA.py", "📊 Data Input & EDA"),
        ("pages/3_Data_Cleaning.py", "🧼 Data Cleaning & Preprocessing"),
        ("pages/4_Feature_Engineering.py", "🔨 Feature Engineering"),
        ("pages/14_Classification_Learning.py", "🧑‍🎓 Classification Learning"),
        ("pages/16_Regression_Learning.py", "🧑‍🎓 Regression Learning"),
        ("pages/15_Clustering_Learning.py", "🧑‍🎓 Clustering Learning"),
        ("pages/19_NLP_TFIDF_Sentiment.py", "🗣️ NLP: TF-IDF & Sentiment"),
        ("pages/5_Tabular_Machine_Learning.py", "🎯 Tabular ML (Classical)"),
        ("pages/6_Deep_Learning.py", "🧠 Deep Learning (Neural Networks)"),
        ("pages/9_Prediction.py", "🎯 Prediction & Inference"),
        ("pages/10_AI_Insights.py", "💡 AI Explanations & Insights"),
        ("pages/7_Visualization.py", "🎨 Visualization Studio"),
        ("pages/18_Sample_Report.py", "📄 Export & Reporting"),
        ("pages/17_Demo_Workflow.py", "🚀 Demo Workflow"),
        ("pages/12_DS_Academy.py", "🎓 Data Science Academy"),
        ("pages/20_Supervised_Learning.py", "🧠 Supervised Learning"),
        ("pages/21_Unsupervised_Learning.py", "🔍 Unsupervised Learning"),
        ("pages/22_ML_Academy.py", "🎓 ML Academy 2.0"),
        ("pages/23_ML_Platform_Studio.py", "💎 ML Platform Studio"),
        ("pages/13_Settings.py", "⚙️ Settings"),
    ]

    for idx, (path, label) in enumerate(menu_items):
        if st.button(label, key=f"curated_menu_{idx}", width="stretch"):
            st.switch_page(path)


def page_navigation(current_page: str) -> None:
    """Render next/previous page navigation buttons at the bottom of the page.
    
    Args:
        current_page: Current page number as string (e.g., "1", "2", "3", "10")
    """
    pages = {
        "0": ("app", "🏠 Home"),
        "1": ("1_DS_Assistant", "🤖 DS Assistant / Workflow"),
        "2": ("2_Data_Analysis_EDA", "📊 Data Input & EDA"),
        "3": ("3_Data_Cleaning", "🧼 Data Cleaning & Preprocessing"),
        "4": ("4_Feature_Engineering", "🔨 Feature Engineering"),
        "5": ("14_Classification_Learning", "🧑‍🎓 Classification Learning"),
        "6": ("16_Regression_Learning", "🧑‍🎓 Regression Learning"),
        "7": ("15_Clustering_Learning", "🧑‍🎓 Clustering Learning"),
        "8": ("19_NLP_TFIDF_Sentiment", "🗣️ NLP: TF-IDF & Sentiment"),
        "9": ("5_Tabular_Machine_Learning", "🎯 Tabular ML (Classical)"),
        "10": ("6_Deep_Learning", "🧠 Deep Learning (Neural Networks)"),
        "11": ("9_Prediction", "🎯 Prediction & Inference"),
        "12": ("10_AI_Insights", "💡 AI Explanations & Insights"),
        "13": ("7_Visualization", "🎨 Visualization Studio"),
        "14": ("18_Sample_Report", "📄 Export & Reporting"),
        "15": ("17_Demo_Workflow", "🚀 Demo Workflow"),
        "16": ("12_DS_Academy", "🎓 Data Science Academy"),
        "17": ("20_Supervised_Learning", "🧠 Supervised Learning"),
        "18": ("21_Unsupervised_Learning", "🔍 Unsupervised Learning"),
        "19": ("22_ML_Academy", "🎓 ML Academy 2.0"),
        "20": ("23_ML_Platform_Studio", "💎 ML Platform Studio"),
        "21": ("13_Settings", "⚙️ Settings"),
    }
    # Logical page order from beginner to advanced
    page_order = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"]
    
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
            if st.button(f"⬅️ Previous: {prev_label}", key=f"prev_page_{current_page}", width="stretch"):
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
            if st.button(f"Next: {next_label} ➡️", key=f"next_page_{current_page}", width="stretch"):
                # Home page uses app.py, others use pages/filename.py
                page_path = f"{next_file}.py" if next_page == "0" else f"pages/{next_file}.py"
                st.switch_page(page_path)

    render_footer()


def render_footer() -> None:
    """Render a global page footnote with creator/design attribution."""
    # Prevent duplicate rendering when called multiple times per page
    if st.session_state.get("_footer_rendered", False):
        return
    st.session_state["_footer_rendered"] = True

    st.markdown(
        """
        <div style="margin-top: 18px; padding: 12px; border-radius: 10px; background: #f8fafc; border: 1px solid #e9ecef;">
            <div style="font-size: 13px; color: #6c757d; text-align: center;">
                ✨ Created by <b>Mr. Hab Rithy</b> — attractive, modern design
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
