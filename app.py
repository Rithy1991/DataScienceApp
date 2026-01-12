from __future__ import annotations

import json
import io
from pathlib import Path

import pandas as pd
import streamlit as st

# Set page config to wide layout
st.set_page_config(page_title="DataScope Pro", layout="wide", initial_sidebar_state="expanded")

from src.core.config import load_config
from src.core.logging_utils import log_event, setup_logging
from src.core.state import get_clean_df, get_df, set_clean_df, set_df
from src.core.ui import app_header, sidebar_dataset_status, instruction_block, page_navigation, render_footer
from src.core.styles import inject_custom_css, render_stat_card
from src.core.premium_styles import inject_premium_css, get_plotly_theme
from src.core.modern_components import smart_data_preview, auto_data_profile, success_message_with_next_steps, data_quality_badge
from src.core.ai_helper import ai_sidebar_assistant
from src.core.platform_ui import platform_header, feature_card
from src.core.welcome import show_welcome_guide, show_page_tips
from src.data.cleaning import clean_pipeline, infer_datetime_columns, validate_dataframe
from src.data.loader import load_from_api, load_from_upload
from src.data.samples import get_sample_datasets, describe_sample_dataset
from src.storage.history import add_event


def _ensure_dirs(config):
    Path(config.registry_dir).mkdir(parents=True, exist_ok=True)
    Path(config.artifacts_dir).mkdir(parents=True, exist_ok=True)
    Path(config.logging_dir).mkdir(parents=True, exist_ok=True)
    Path("data").mkdir(parents=True, exist_ok=True)


config = load_config()
logger = setup_logging(config.logging_dir, config.logging_level)

def _in_streamlit_runtime() -> bool:
    """Return True when executed via `streamlit run`, False for plain imports.

    Streamlit pages execute at import-time; without this guard, `import app` will
    run UI code and can crash due to missing ScriptRunContext.
    """

    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        return get_script_run_ctx() is not None
    except Exception:
        try:
            from streamlit.runtime.scriptrunner_utils.script_run_context import (  # type: ignore
                get_script_run_ctx,
            )

            return get_script_run_ctx() is not None
        except Exception:
            return False


def main() -> None:
    config = load_config()
    logger = setup_logging(config.logging_dir, config.logging_level)
    _ensure_dirs(config)

    # Inject custom CSS
    inject_custom_css()

    # Add AI assistant
    ai_sidebar_assistant()

    # Show welcome guide for first-time users
    show_welcome_guide()

    st.markdown(
        """
        <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
            <div style="font-size: 24px; font-weight: 800;">🧹 Data Cleaning & Preparation</div>
            <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Upload a dataset or pick a sample to start exploring and cleaning.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    instruction_block(
        "How to use this page",
        [
            "Load data from a file, API URL, or sample dataset.",
            "Glance at validation and quick stats to confirm columns and row counts.",
            "Pick how to handle missing values, parse dates, and cap outliers.",
            "Run cleaning; the report lists what changed.",
            "Preview the cleaned table, export it, or reuse it on the other pages.",
        ],
    )

    raw_df = get_df(st.session_state)
    clean_df = get_clean_df(st.session_state)
    sidebar_dataset_status(raw_df, clean_df)

    # Show contextual tips
    show_page_tips("data_exploration")

    st.subheader("📥 Ingest Data")

    # Tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["Upload File", "API", "Sample Data"])

    with tab1:
        st.markdown("**Upload from local file** (CSV, Parquet, Excel)")
        uploaded = st.file_uploader(
            "Choose a file", type=["csv", "parquet", "xlsx", "xls"], accept_multiple_files=False, key="upload_file"
        )
        if st.button("📤 Load uploaded file", use_container_width=True, disabled=(uploaded is None)):
            res = load_from_upload(uploaded)
            if res is None:
                st.error("❌ Unsupported file type.")
            else:
                set_df(st.session_state, res.df, source=res.source)
                add_event(config.history_db_path, "data_load", f"Loaded dataset from {res.source}", json.dumps(res.meta))
                log_event(config.logging_dir, "data_load", {"source": res.source, **res.meta})
                st.success(f"✅ Loaded {res.df.shape[0]:,} rows × {res.df.shape[1]:,} cols")
                st.rerun()

    with tab2:
        st.markdown("**Load from REST API** (GET endpoint returning JSON or CSV)")
        api_url = st.text_input("API URL", value="", placeholder="https://api.example.com/data")
        api_headers = st.text_area("Optional headers (JSON)", value="{}", height=80)
        if st.button("🌐 Fetch API data", use_container_width=True, disabled=(api_url.strip() == "")):
            try:
                headers = json.loads(api_headers or "{}")
                res = load_from_api(api_url.strip(), headers=headers)
                set_df(st.session_state, res.df, source=res.source)
                add_event(config.history_db_path, "data_load", f"Loaded dataset from {res.source}", json.dumps(res.meta))
                log_event(config.logging_dir, "data_load", {"source": res.source, **res.meta})
                st.success(f"✅ Loaded {res.df.shape[0]:,} rows × {res.df.shape[1]:,} cols")
                st.rerun()
            except Exception as e:
                st.error(f"❌ API load failed: {e}")

    with tab3:
        st.markdown("**Quick start with sample datasets**")
        samples = get_sample_datasets()
        sample_name = st.selectbox("Choose a sample dataset", list(samples.keys()), key="sample_select")
        st.caption(describe_sample_dataset(sample_name))
        if st.button("📊 Load sample data", use_container_width=True):
            sample_df = samples[sample_name]
            set_df(st.session_state, sample_df, source=f"Sample: {sample_name}")
            add_event(config.history_db_path, "data_load", f"Loaded sample dataset: {sample_name}", json.dumps({"rows": len(sample_df), "cols": len(sample_df.columns)}))
            log_event(config.logging_dir, "data_load", {"source": f"Sample: {sample_name}", "rows": len(sample_df), "cols": len(sample_df.columns)})
            st.success(f"✅ Loaded sample: {sample_df.shape[0]:,} rows × {sample_df.shape[1]:,} cols")
            st.rerun()

    raw_df = get_df(st.session_state)
    if raw_df is None:
        st.info("💡 Load a dataset to begin — choose from file upload, API, or sample data above.")
        st.stop()

    # Narrow for type-checkers (Streamlit's st.stop doesn't narrow types).
    assert raw_df is not None

    st.subheader("✓ Validate")
    issues = validate_dataframe(raw_df)
    if issues:
        st.warning("⚠️ Validation issues detected")
        st.json(issues)
    else:
        st.success("✅ Basic validation passed")

    st.subheader("📊 Data Snapshot")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(render_stat_card("Rows", f"{raw_df.shape[0]:,}", icon="🗂️"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_stat_card("Columns", f"{raw_df.shape[1]:,}", icon="📑"), unsafe_allow_html=True)
    with c3:
        missing_total = int(raw_df.isna().sum().sum())
        st.markdown(render_stat_card("Missing values", f"{missing_total:,}", icon="⚠️"), unsafe_allow_html=True)

    st.subheader("👁️ Preview")
    max_rows = int(config.max_rows_preview)
    st.dataframe(raw_df.head(max_rows), use_container_width=True)

    st.divider()

    col1, col2 = st.columns([2, 1])
    with col1:
        st.info("👋 **Next Step:** Your data is loaded! Head over to the **Data Cleaning** page to handle missing values, duplicates, and outliers.")
    with col2:
        if st.button("Go to Data Cleaning ➡️", type="primary", use_container_width=True):
            st.switch_page("pages/3_Data_Cleaning.py")

if _in_streamlit_runtime():
    main()

# Page navigation
page_navigation("0")

# Redundant footer call to guarantee attribution on Home without duplication
render_footer()
