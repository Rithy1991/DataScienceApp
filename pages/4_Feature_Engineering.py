"""
Feature Engineering Module
Transform raw data into powerful predictive features
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif

# Must set page config first
st.set_page_config(page_title="DataScope Pro - Feature Engineering", layout="wide", initial_sidebar_state="expanded")

from src.core.config import load_config
from src.core.state import get_clean_df, get_df, set_clean_df
from src.core.ui import sidebar_dataset_status, instruction_block, page_navigation
from src.core.styles import inject_custom_css
from src.core.ai_helper import ai_sidebar_assistant
from src.core.platform_ui import module_section

def _save_changes(new_df: pd.DataFrame, message: str = "Changes saved to clean dataset."):
    """Helper to save changes to session state."""
    set_clean_df(st.session_state, new_df)
    st.toast(message, icon="✅")
    st.rerun()

def _learning_module():
    """Educational content about feature engineering."""
    st.markdown("### 📚 Concepts & Best Practices")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Why Engineer Features?**\n\nAlgorithms are stupid. They only know what you show them. Feature engineering makes the 'signal' in your data obvious so the model doesn't have to guess.", icon="💡")
    with col2:
        st.success("**The Golden Rule**\n\nAlways split your data *before* complex engineering if possible, but for this app, we transform the full dataset to prepare it for the ML page.", icon="⭐")

    st.markdown("---")
    
    c1, c2, c3 = st.columns(3)
    c1.markdown("#### 1. Encoding")
    c1.caption("Machines don't speak text. Convert 'Red', 'Blue' into 0, 1 or [1,0], [0,1].")
    
    c2.markdown("#### 2. Scaling")
    c2.caption("If 'Salary' is 100,000 and 'Age' is 30, the model thinks Salary is 3000x more important. Scaling fixes this.")
    
    c3.markdown("#### 3. Selection")
    c3.caption("More data ≠ better. Removing noise (useless columns) often increases accuracy.")

def _categorical_encoding(df: pd.DataFrame):
    """Encoding module."""
    module_section("Categorical Encoding", "Convert text categories into numbers", "🏷️")
    
    obj_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not obj_cols:
        st.success("✅ No categorical columns found. Your data is already numeric!")
        return

    col_to_encode = st.selectbox("Select column to encode", obj_cols)
    method = st.radio("Method", ["One-Hot Encoding", "Label Encoding", "Frequency Encoding"], horizontal=True)
    
    if method == "One-Hot Encoding":
        st.caption("Creates a new binary column for each category. Best for: nominal data (Color, City) with low cardinality.")
        if st.checkbox(f"Preview One-Hot for '{col_to_encode}'?"):
            preview = pd.get_dummies(df, columns=[col_to_encode], prefix=col_to_encode)
            st.dataframe(preview.head())
            if st.button("Save One-Hot Changes"):
                _save_changes(preview, f"One-Hot Encoded {col_to_encode}")

    elif method == "Label Encoding":
        st.caption("Assigns a number (0, 1, 2) to each category. Best for: ordinal data (Low, Med, High).")
        if st.checkbox(f"Preview Label for '{col_to_encode}'?"):
            temp_df = df.copy()
            le = LabelEncoder()
            # Handle NaN for label encoder
            temp_df[f"{col_to_encode}_enc"] = le.fit_transform(temp_df[col_to_encode].astype(str))
            st.dataframe(temp_df[[col_to_encode, f"{col_to_encode}_enc"]].head())
            if st.button("Save Label Encoding"):
                temp_df = temp_df.drop(columns=[col_to_encode])
                temp_df = temp_df.rename(columns={f"{col_to_encode}_enc": col_to_encode})
                _save_changes(temp_df, f"Label Encoded {col_to_encode}")

    elif method == "Frequency Encoding":
        st.caption("Replaces category with its count/frequency. Best for: high cardinality (Zip Code).")
        if st.checkbox(f"Preview Frequency for '{col_to_encode}'?"):
            temp_df = df.copy()
            freq = temp_df[col_to_encode].value_counts(normalize=True)
            temp_df[f"{col_to_encode}_freq"] = temp_df[col_to_encode].map(freq)
            st.dataframe(temp_df[[col_to_encode, f"{col_to_encode}_freq"]].head())
            if st.button("Save Freq Encoding"):
                 temp_df = temp_df.drop(columns=[col_to_encode]).rename(columns={f"{col_to_encode}_freq": col_to_encode})
                 _save_changes(temp_df, f"Freq Encoded {col_to_encode}")

def _feature_scaling(df: pd.DataFrame):
    """Scaling module."""
    module_section("Feature Scaling", "Normalize range of numeric features", "📊")
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if not num_cols:
        st.info("No numeric columns to scale.")
        return

    cols_to_scale = st.multiselect("Select columns to scale", num_cols, default=num_cols[:3] if len(num_cols)>3 else num_cols)
    method = st.selectbox("Scaling Method", ["StandardScaler (Z-Score)", "MinMaxScaler (0-1)", "RobustScaler (IQR)"])
    
    if cols_to_scale:
        temp_df = df.copy()
        
        if method.startswith("Standard"):
            scaler = StandardScaler()
        elif method.startswith("MinMax"):
            scaler = MinMaxScaler()
        else:
            scaler = RobustScaler()
            
        if st.button(f"Preview {method.split()[0]}") or st.session_state.get('scaling_preview', False):
            st.session_state['scaling_preview'] = True
            
            # Preview logic
            scaled_vals = scaler.fit_transform(temp_df[cols_to_scale])
            preview_df = pd.DataFrame(scaled_vals, columns=cols_to_scale, index=temp_df.index)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Original Dist**")
                st.dataframe(df[cols_to_scale].describe().loc[['mean', 'std', 'min', 'max']])
            with col2:
                st.markdown("**Scaled Dist (Preview)**")
                st.dataframe(preview_df.describe().loc[['mean', 'std', 'min', 'max']])
            
            if st.button(f"Apply & Save Scaling"):
                temp_df[cols_to_scale] = scaled_vals
                st.session_state['scaling_preview'] = False
                _save_changes(temp_df, f"Scaled {len(cols_to_scale)} columns")

def _feature_selection(df: pd.DataFrame):
    """Selection module."""
    module_section("Feature Selection", "Remove low-value or redundant features", "🎯")
    
    st.markdown("#### Drop Columns Manually")
    to_drop = st.multiselect("Select columns to REMOVE", df.columns)
    if to_drop:
        st.warning(f"Dropping: {', '.join(to_drop)}")
        if st.button("Confirm Drop"):
            new_df = df.drop(columns=to_drop)
            _save_changes(new_df, f"Dropped {len(to_drop)} columns")
            
    st.divider()
    
    st.markdown("#### Auto-Selection (Correlation)")
    if len(df.select_dtypes(include=np.number).columns) > 1:
        target = st.selectbox("Select Target (to measure importance against)", df.columns)
        k = st.slider("Keep Top K Features", 1, len(df.columns)-1, 5)
        
        if st.button("Run Importance Analysis"):
            num_df = df.select_dtypes(include=np.number)
            if target in num_df.columns:
                corr = num_df.corrwith(num_df[target]).abs().sort_values(ascending=False)
                
                c1, c2 = st.columns([2,1])
                with c1:
                    fig = px.bar(corr, orientation='h', title="Feature Correlation with Target")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.dataframe(corr)
                
                # Suggestion
                top_k = corr.head(k+1).index.tolist() # +1 includes target itself
                st.info(f"Top {k} correlated features: {top_k}")
            else:
                st.warning("Target must be numeric for this correlation check.")

def _feature_creation(df: pd.DataFrame):
    """Creation module."""
    module_section("Feature Creation", "Engineer new signals", "⚗️")
    
    st.subheader("Math Operations")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) >= 2:
        c1, c2, c3 = st.columns(3)
        col_a = c1.selectbox("Col A", num_cols, key="create_a")
        op = c2.selectbox("Op", ["+", "-", "*", "/"], key="create_op")
        col_b = c3.selectbox("Col B", num_cols, key="create_b")
        
        new_name = st.text_input("New Column Name", f"{col_a}_{op}_{col_b}")
        
        if st.button("Calculate & Add"):
            try:
                temp_df = df.copy()
                if op == "+": temp_df[new_name] = temp_df[col_a] + temp_df[col_b]
                elif op == "-": temp_df[new_name] = temp_df[col_a] - temp_df[col_b]
                elif op == "*": temp_df[new_name] = temp_df[col_a] * temp_df[col_b]
                elif op == "/": temp_df[new_name] = temp_df[col_a] / (temp_df[col_b] + 1e-9)
                
                _save_changes(temp_df, f"Created {new_name}")
            except Exception as e:
                st.error(f"Error: {e}")
                
    st.subheader("Date Parsing")
    obj_cols = df.select_dtypes(include=['object']).columns.tolist()
    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    potential_dates = obj_cols + date_cols
    
    if potential_dates:
        dcol = st.selectbox("Select Date Column", potential_dates)
        if st.button("Extract Date Features (Year/Month/Day)"):
            temp_df = df.copy()
            # Ensure it's datetime
            temp_df[dcol] = pd.to_datetime(temp_df[dcol], errors='coerce')
            
            # create features
            temp_df[f"{dcol}_year"] = temp_df[dcol].dt.year
            temp_df[f"{dcol}_month"] = temp_df[dcol].dt.month
            temp_df[f"{dcol}_dow"] = temp_df[dcol].dt.dayofweek
            
            _save_changes(temp_df, "Extracted Date Features")

def main():
    config = load_config()
    inject_custom_css()
    ai_sidebar_assistant()
    
    st.markdown(
        """
        <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
            <div style="font-size: 24px; font-weight: 800;">🔨 Feature Engineering</div>
            <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Modify your dataset for better model performance. Changes are saved to the 'Clean' dataset.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    raw_df = get_df(st.session_state)
    clean_df = get_clean_df(st.session_state)
    sidebar_dataset_status(raw_df, clean_df)
    
    # Priority: Clean > Raw
    df = clean_df if clean_df is not None else raw_df
    
    if df is None:
        st.warning("⚠️ Please load a dataset first (Page 1 or 2).")
        st.stop()
        
    st.caption(f"Active Dataset: {len(df)} rows, {len(df.columns)} columns")
    st.dataframe(df.head())
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📚 Learn", "🏷️ Encoding", "📊 Scaling", "🎯 Selection", "⚗️ Creation"
    ])
    
    with tab1:
        _learning_module()
    with tab2:
        _categorical_encoding(df)
    with tab3:
        _feature_scaling(df)
    with tab4:
        _feature_selection(df)
    with tab5:
        _feature_creation(df)
        
    st.divider()
    page_navigation("4")

if __name__ == "__main__":
    main()
