"""
Flow Guidance & Progress Tracking
Helps users understand where they are in the data science pipeline and what comes next.
"""
from __future__ import annotations

from typing import Any, MutableMapping, Optional, Tuple

import pandas as pd
import streamlit as st

from src.core.state import get_clean_df, get_df

# Data Science Pipeline steps
PIPELINE_STEPS = [
    {
        "id": 0,
        "emoji": "🏠",
        "name": "Load Data",
        "description": "Upload or select a dataset",
        "file": "app.py",
        "state_key": "raw_data_loaded",
    },
    {
        "id": 1,
        "emoji": "📊",
        "name": "Explore Data",
        "description": "Understand structure, types, and distributions",
        "file": "pages/2_Data_Analysis_EDA.py",
        "state_key": "data_explored",
    },
    {
        "id": 2,
        "emoji": "🧼",
        "name": "Clean Data",
        "description": "Handle missing, duplicates, and outliers",
        "file": "pages/3_Data_Cleaning.py",
        "state_key": "data_cleaned",
    },
    {
        "id": 3,
        "emoji": "🔨",
        "name": "Engineer Features",
        "description": "Create and select features for models",
        "file": "pages/4_Feature_Engineering.py",
        "state_key": "features_engineered",
    },
    {
        "id": 4,
        "emoji": "🎯",
        "name": "Train Model",
        "description": "Build and optimize machine learning models",
        "file": "pages/5_Tabular_Machine_Learning.py",
        "state_key": "model_trained",
    },
    {
        "id": 5,
        "emoji": "📈",
        "name": "Evaluate Model",
        "description": "Assess performance with metrics and visualizations",
        "file": "pages/5_Tabular_Machine_Learning.py",
        "state_key": "model_evaluated",
    },
    {
        "id": 6,
        "emoji": "🎯",
        "name": "Predict & Infer",
        "description": "Apply model to new data",
        "file": "pages/9_Prediction.py",
        "state_key": "predictions_made",
    },
    {
        "id": 7,
        "emoji": "📄",
        "name": "Report & Export",
        "description": "Generate findings and export results",
        "file": "pages/18_Sample_Report.py",
        "state_key": "report_generated",
    },
]


def get_current_pipeline_step(session_state: MutableMapping[Any, Any]) -> Optional[int]:
    """Determine which pipeline step the user is currently on based on data state.
    
    Returns:
        int: Current step ID (0-7), or None if indeterminate.
    """
    raw_df = get_df(session_state)
    clean_df = get_clean_df(session_state)
    
    if raw_df is None:
        return 0  # User hasn't loaded data yet
    
    if clean_df is None:
        return 1  # Data loaded but not cleaned
    
    # If we have cleaned data, user is past cleaning
    # Would need additional state to determine exact step
    return 2  # Default: assume at cleaning or beyond


def is_step_completed(session_state: MutableMapping[Any, Any], step_id: int) -> bool:
    """Check if a specific pipeline step is completed.
    
    Args:
        session_state: Streamlit session state.
        step_id: Step ID (0-7).
    
    Returns:
        bool: True if step is completed.
    """
    raw_df = get_df(session_state)
    clean_df = get_clean_df(session_state)
    
    if step_id <= 0:
        return raw_df is not None
    elif step_id <= 2:
        return clean_df is not None
    else:
        # Would need more state tracking for later steps
        return False


def render_pipeline_progress_sidebar() -> None:
    """Render a visual pipeline progress tracker in the sidebar."""
    with st.sidebar:
        st.markdown("### 📍 Your Data Science Journey")
        
        current_step = get_current_pipeline_step(st.session_state)
        
        # Render each pipeline step
        for step in PIPELINE_STEPS:
            is_completed = is_step_completed(st.session_state, step["id"])
            is_current = step["id"] == current_step
            
            if is_completed:
                status = "✅"
                color = "#06d6a0"  # Green
            elif is_current:
                status = "🔵"
                color = "#118ab2"  # Blue
            else:
                status = "⭕"
                color = "#cccccc"  # Gray
            
            # Render step with badge
            if is_current:
                st.markdown(
                    f"<div style='background-color: {color}; padding: 10px; border-radius: 8px; margin-bottom: 8px;'>"
                    f"<b>{status} {step['emoji']} {step['name']}</b><br>"
                    f"<small style='opacity: 0.9;'>{step['description']}</small>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"{status} {step['emoji']} {step['name']}"
                )
        
        st.divider()


def render_step_guidance(
    current_step_id: int,
    current_step_name: str,
    current_step_description: str,
    next_step_name: Optional[str] = None,
    next_step_description: Optional[str] = None,
) -> None:
    """Render 'where am I?' and 'what's next?' guidance on the current page.
    
    Args:
        current_step_id: ID of the current pipeline step.
        current_step_name: Name of current step.
        current_step_description: Description of what happens in this step.
        next_step_name: Name of next step (if any).
        next_step_description: Description of next step.
    """
    col1, col2 = st.columns([1, 1])
    
    # Current step info
    with col1:
        step = PIPELINE_STEPS[current_step_id]
        st.info(
            f"**📍 Current Step: {step['emoji']} {step['name']}**\n\n"
            f"{step['description']}"
        )
    
    # Next step preview
    with col2:
        if current_step_id < len(PIPELINE_STEPS) - 1:
            next_step = PIPELINE_STEPS[current_step_id + 1]
            st.success(
                f"**➡️ Next Step: {next_step['emoji']} {next_step['name']}**\n\n"
                f"{next_step['description']}"
            )
        else:
            st.success(
                f"**✅ You've reached the end of the pipeline!**\n\n"
                f"Export your findings or start a new analysis."
            )


def render_pipeline_roadmap() -> None:
    """Render a seamless visual roadmap of the data science pipeline.
    
    Used on home page to show the entire journey as one unified workflow.
    """
    st.markdown("## 🛣️ Your Complete Data Science Journey")
    st.markdown("A seamless end-to-end workflow from raw data to actionable insights:")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render all 8 steps in a unified flow across 2 rows
    # First row: 4 steps
    cols_row1 = st.columns(4)
    for i, step in enumerate(PIPELINE_STEPS[:4]):
        with cols_row1[i]:
            # Use different gradient based on step type
            if i < 2:  # Data steps
                gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
            elif i < 4:  # Preparation steps
                gradient = "linear-gradient(135deg, #8e54e9 0%, #4776e6 100%)"
            else:
                gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            
            st.markdown(
                f"""
                <div style='
                    background: {gradient};
                    padding: 20px;
                    border-radius: 12px;
                    color: white;
                    text-align: center;
                    min-height: 140px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    transition: transform 0.2s;
                '>
                    <div style='font-size: 38px; margin-bottom: 10px;'>{step['emoji']}</div>
                    <div style='font-weight: bold; font-size: 16px; margin-bottom: 8px;'>{step['name']}</div>
                    <div style='font-size: 13px; opacity: 0.95; line-height: 1.4;'>{step['description']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    # Add a flow connector
    st.markdown("<div style='text-align: center; font-size: 24px; margin: 10px 0;'>⬇️</div>", unsafe_allow_html=True)
    
    # Second row: 4 steps
    cols_row2 = st.columns(4)
    for i, step in enumerate(PIPELINE_STEPS[4:]):
        with cols_row2[i]:
            # Advanced ML steps use warmer gradients
            gradient = "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)"
            
            st.markdown(
                f"""
                <div style='
                    background: {gradient};
                    padding: 20px;
                    border-radius: 12px;
                    color: white;
                    text-align: center;
                    min-height: 140px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                    transition: transform 0.2s;
                '>
                    <div style='font-size: 38px; margin-bottom: 10px;'>{step['emoji']}</div>
                    <div style='font-weight: bold; font-size: 16px; margin-bottom: 8px;'>{step['name']}</div>
                    <div style='font-size: 13px; opacity: 0.95; line-height: 1.4;'>{step['description']}</div>
                </div>
                """,
                unsafe_allow_html=True
            )


def render_next_step_button(next_step_id: Optional[int]) -> bool:
    """Render a prominent 'Go to Next Step' button.
    
    Args:
        next_step_id: ID of the next step (or None if at the end).
    
    Returns:
        bool: True if the button was clicked.
    """
    if next_step_id is None or next_step_id >= len(PIPELINE_STEPS):
        return False
    
    next_step = PIPELINE_STEPS[next_step_id]
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(
            f"➡️ Continue to {next_step['name']}",
            type="primary",
            width="stretch",
            key=f"next_step_{next_step_id}",
        ):
            st.switch_page(next_step["file"])
            return True
    
    return False


def render_completion_checklist(session_state: MutableMapping[Any, Any]) -> None:
    """Render a checklist showing completion status of each pipeline step.
    
    Args:
        session_state: Streamlit session state.
    """
    st.markdown("## ✅ Completion Checklist")
    
    current_step = get_current_pipeline_step(session_state)
    
    for step in PIPELINE_STEPS:
        is_completed = is_step_completed(session_state, step["id"])
        is_current = step["id"] == current_step
        
        if is_completed:
            st.checkbox(
                f"{step['emoji']} {step['name']} — {step['description']}",
                value=True,
                disabled=True
            )
        elif is_current:
            st.markdown(
                f"🔵 **{step['emoji']} {step['name']} (Current)** — {step['description']}"
            )
        else:
            st.markdown(
                f"⭕ {step['emoji']} {step['name']} — {step['description']}"
            )
