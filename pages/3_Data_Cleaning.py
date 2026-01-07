"""
Data Cleaning Module
Handle missing values, duplicates, and outliers.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(page_title="DataScope Pro - Data Cleaning", layout="wide", initial_sidebar_state="expanded")

from src.core.config import load_config
from src.core.state import get_clean_df, get_df, set_clean_df
from src.core.ui import sidebar_dataset_status, page_navigation
from src.core.styles import inject_custom_css
from src.core.ai_helper import ai_sidebar_assistant
from src.core.platform_ui import module_section

def _save_changes(new_df: pd.DataFrame, message: str):
    """Save changes to clean dataframe state."""
    set_clean_df(st.session_state, new_df)
    st.toast(message, icon="✅")
    st.rerun()

def _missing_values_handler(df: pd.DataFrame):
    """Module for handling missing data."""
    st.subheader("🧹 Missing Values")
    
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    
    if missing.empty:
        st.success("✅ No missing values found!")
        return
        
    st.warning(f"Found missing values in {len(missing)} columns.")
    st.dataframe(missing.to_frame(name="Count").T)
    
    col_to_fix = st.selectbox("Select Column", missing.index)
    method = st.selectbox("Imputation Method", 
                          ["Drop Rows", "Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Custom Value", "Forward Fill"])
    
    if st.button(f"Apply '{method}' to {col_to_fix}", type="primary", use_container_width=True):
        temp_df = df.copy()
        if method == "Drop Rows":
            temp_df = temp_df.dropna(subset=[col_to_fix])
        elif method == "Fill with Mean":
            if pd.api.types.is_numeric_dtype(temp_df[col_to_fix]):
                temp_df[col_to_fix] = temp_df[col_to_fix].fillna(temp_df[col_to_fix].mean())
            else:
                st.error("Mean only works for numeric columns.")
                return
        elif method == "Fill with Median":
            if pd.api.types.is_numeric_dtype(temp_df[col_to_fix]):
                temp_df[col_to_fix] = temp_df[col_to_fix].fillna(temp_df[col_to_fix].median())
            else:
                st.error("Median only works for numeric columns.")
        elif method == "Fill with Mode":
            temp_df[col_to_fix] = temp_df[col_to_fix].fillna(temp_df[col_to_fix].mode()[0])
        elif method == "Forward Fill":
             temp_df[col_to_fix] = temp_df[col_to_fix].ffill()
        
        _save_changes(temp_df, f"Fixed missing values in {col_to_fix}")

def _duplicates_handler(df: pd.DataFrame):
    st.subheader("👯 Duplicates")
    
    n_dupes = df.duplicated().sum()
    if n_dupes == 0:
        st.success("✅ No duplicate rows found.")
    else:
        st.warning(f"Found {n_dupes} duplicate rows.")
        if st.button("Remove Duplicates", type="primary", use_container_width=True):
            temp_df = df.drop_duplicates()
            _save_changes(temp_df, f"Removed {n_dupes} duplicates")

def _outliers_handler(df: pd.DataFrame):
    st.subheader("📈 Outlier Management")
    
    num_cols = df.select_dtypes(include=np.number).columns
    if len(num_cols) == 0:
        st.info("No numeric columns for outlier check.")
        return
        
    target_col = st.selectbox("Select Column for Z-Score Check", num_cols)
    threshold = st.slider("Z-Score Threshold", 1.5, 5.0, 3.0)
    
    col_data = df[target_col].dropna()
    z_scores = ((col_data - col_data.mean()) / col_data.std()).abs()
    outliers = col_data[z_scores > threshold]
    
    st.write(f"Found **{len(outliers)}** outliers using Z-Score > {threshold}")
    
    if len(outliers) > 0:
        action = st.radio("Action", ["Clip Values", "Drop Rows containing Outliers"])
        if st.button("Handle Outliers", type="primary", use_container_width=True):
            temp_df = df.copy()
            if action == "Clip Values":
                lower = col_data.mean() - threshold * col_data.std()
                upper = col_data.mean() + threshold * col_data.std()
                temp_df[target_col] = temp_df[target_col].clip(lower, upper)
                msg = "Clipped outliers"
            else:
                temp_df = temp_df.drop(outliers.index)
                msg = f"Dropped {len(outliers)} outlier rows"
            
            _save_changes(temp_df, msg)

def main():
    config = load_config()
    inject_custom_css()
    ai_sidebar_assistant()
    
    st.markdown(
        """
        <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
            <div style="font-size: 24px; font-weight: 800;">🧼 Data Cleaning</div>
            <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Prepare your data for success.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    raw_df = get_df(st.session_state)
    clean_df = get_clean_df(st.session_state)
    sidebar_dataset_status(raw_df, clean_df)
    
    # Logic: If clean exists, work on clean. Else work on Raw (and save to Clean).
    df = clean_df if clean_df is not None else raw_df
    
    if df is None:
        st.warning("Please load a dataset first.")
        st.stop()
        
    tab1, tab2, tab3 = st.tabs(["Missing Data", "Duplicates", "Outliers"])
    
    with tab1:
        _missing_values_handler(df)
    with tab2:
        _duplicates_handler(df)
    with tab3:
        _outliers_handler(df)
        
    st.divider()
    # Logic: Page 13. Nav should be 13.
    # But wait, the standard navigation might not know about 13 yet.
    # I need to update src/core/ui.py to handle 13.
    page_navigation("3")

if __name__ == "__main__":
    main()
