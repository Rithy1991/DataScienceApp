from __future__ import annotations

import warnings
import json
import io
from pathlib import Path

import pandas as pd
import streamlit as st

# Suppress scikit-learn warnings for educational context (imbalanced data)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._ranking')

# Set page config to wide layout
st.set_page_config(page_title="DataScope Pro", layout="wide", initial_sidebar_state="expanded")

from src.core.config import load_config
from src.core.logging_utils import log_event, setup_logging
from src.core.state import get_clean_df, get_df, set_clean_df, set_df
from src.core.ui import app_header, sidebar_dataset_status, instruction_block, page_navigation, render_footer
from src.core.styles import inject_custom_css, render_stat_card
from src.core.premium_styles import inject_premium_css, get_plotly_theme
from src.core.modern_components import smart_data_preview, auto_data_profile, success_message_with_next_steps, data_quality_badge
from src.core.platform_ui import platform_header, feature_card
from src.core.welcome import show_welcome_guide, show_page_tips
from src.core.flow_guidance import (
    render_pipeline_roadmap,
    render_completion_checklist,
    render_next_step_button,
)
from src.core.ml_best_practices import render_quick_tip, render_best_practices_section
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

    # Show welcome guide for first-time users
    show_welcome_guide()

    app_header(
        config,
        page_title="Home",
        subtitle="Your Complete Data Science Learning Platform",
        icon="🏠"
    )

    raw_df = get_df(st.session_state)
    clean_df = get_clean_df(st.session_state)
    sidebar_dataset_status(raw_df, clean_df)

    # Show contextual tips
    show_page_tips("data_exploration")

    # ========================================================================
    # VISUAL PIPELINE ROADMAP
    # ========================================================================
    
    st.markdown("## 🎯 Welcome to DataScope Pro")
    st.markdown(
        "Your complete platform for learning and practicing data science. "
        "Follow the pipeline below from **data loading** to **reporting** to master the entire workflow."
    )
    
    render_pipeline_roadmap()
    
    st.divider()

    # ========================================================================
    # DATA LOADING SECTION
    # ========================================================================
    
    instruction_block(
        "📍 Step 1: Load Your Data",
        [
            "Choose a data source: file upload, REST API, or sample dataset",
            "DataScope Pro will validate and profile your data automatically",
            "Once loaded, proceed to Data Cleaning to prepare for analysis",
        ],
    )

    st.subheader("📥 Load Data")

    # Tabs for different data sources
    tab1, tab2, tab3 = st.tabs(["Upload File", "API", "Sample Data"])

    with tab1:
        st.markdown("**Upload from your computer** (CSV, Parquet, Excel)")
        uploaded = st.file_uploader(
            "Choose a file", type=["csv", "parquet", "xlsx", "xls"], accept_multiple_files=False, key="upload_file"
        )
        if st.button("📤 Load uploaded file", width="stretch", disabled=(uploaded is None)):
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
        st.markdown("**Load from a REST API** (GET endpoint returning JSON or CSV)")
        api_url = st.text_input("API URL", value="", placeholder="https://api.example.com/data")
        api_headers = st.text_area("Optional headers (JSON)", value="{}", height=80)
        if st.button("🌐 Fetch API data", width="stretch", disabled=(api_url.strip() == "")):
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
        st.markdown("**Quick start with pre-loaded samples**")
        samples = get_sample_datasets()
        sample_name = st.selectbox("Choose a sample dataset", list(samples.keys()), key="sample_select")
        st.caption(describe_sample_dataset(sample_name))
        if st.button("📊 Load sample data", width="stretch"):
            sample_df = samples[sample_name]
            set_df(st.session_state, sample_df, source=f"Sample: {sample_name}")
            add_event(config.history_db_path, "data_load", f"Loaded sample dataset: {sample_name}", json.dumps({"rows": len(sample_df), "cols": len(sample_df.columns)}))
            log_event(config.logging_dir, "data_load", {"source": f"Sample: {sample_name}", "rows": len(sample_df), "cols": len(sample_df.columns)})
            st.success(f"✅ Loaded sample: {sample_df.shape[0]:,} rows × {sample_df.shape[1]:,} cols")
            st.rerun()

    raw_df = get_df(st.session_state)
    if raw_df is None:
        st.info("💡 **Start here:** Load a dataset using one of the options above to begin your data science journey.")
        st.stop()

    # Narrow for type-checkers (Streamlit's st.stop doesn't narrow types).
    assert raw_df is not None

    st.divider()

    # ========================================================================
    # DATA PREVIEW & VALIDATION
    # ========================================================================
    
    st.subheader("✓ Data Quality Check")
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

    st.subheader("👁️ Data Preview")
    max_rows = int(config.max_rows_preview)
    st.dataframe(raw_df.head(max_rows), width="stretch")

    st.divider()

    # ========================================================================
    # NEXT STEP GUIDANCE
    # ========================================================================
    
    st.markdown("## ➡️ Next: Data Cleaning")
    col1, col2 = st.columns(2)
    with col1:
        st.info(
            "Your data is loaded! The next step is **Data Cleaning**, where you'll:\n"
            "- Handle missing values (imputation or removal)\n"
            "- Remove duplicate rows\n"
            "- Detect and manage outliers\n\n"
            "This ensures your ML models train on clean, reliable data."
        )
    with col2:
        st.success(
            "**Why cleaning matters:**\n\n"
            "Poor data quality leads to biased models and unreliable predictions. "
            "Spending time on data preparation now saves hours of debugging later.\n\n"
            "✅ Let's get started!"
        )
    
    render_next_step_button(next_step_id=2)
    
    st.divider()
    
    # ========================================================================
    # COMPLETION CHECKLIST
    # ========================================================================
    
    st.markdown("## 📋 Your Progress")
    render_completion_checklist(st.session_state)
    
    st.divider()
    
    # ========================================================================
    # LEARNING RESOURCES
    # ========================================================================
    
    st.markdown("## 🎓 Learn As You Go")
    st.markdown("Master data science with these essential best practices:")
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_quick_tip("data_loading")
        render_best_practices_section("data_quality")
        render_best_practices_section("model_selection")
    
    with col2:
        st.info("💡 **Pro Tip:** The quality of your insights depends on the quality of your data. Take time to understand your data before diving into modeling!")
        render_best_practices_section("training")
        render_best_practices_section("common_mistakes")

if _in_streamlit_runtime():
    main()

# Page navigation
page_navigation("0")

# Redundant footer call to guarantee attribution on Home without duplication
render_footer()
