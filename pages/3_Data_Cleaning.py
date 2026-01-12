"""
Data Cleaning Module
Handle missing values, duplicates, outliers, and data quality issues with strategy guidance.
"""
from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# Page Config
st.set_page_config(page_title="DataScope Pro - Data Cleaning", layout="wide", initial_sidebar_state="expanded")

from src.core.config import load_config
from src.core.state import get_clean_df, get_df, set_clean_df
from src.core.ui import sidebar_dataset_status, page_navigation, instruction_block
from src.core.standardized_ui import (
    standard_section_header,
    beginner_tip,
    concept_explainer,
    common_mistakes_panel,
    metric_card,
)
from src.core.styles import inject_custom_css
from src.core.ai_helper import ai_sidebar_assistant
from src.core.platform_ui import module_section

def _save_changes(new_df: pd.DataFrame, message: str):
    """Save changes to clean dataframe state."""
    set_clean_df(st.session_state, new_df)
    st.toast(message, icon="✅")
    st.rerun()

# ============================================================================
# CLEANING STRATEGY FRAMEWORK
# ============================================================================

def _show_cleaning_strategy_guide():
    """Educational guide for choosing cleaning strategies."""
    with st.expander("🎓 **Cleaning Strategy Guide**", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
**Numeric Columns (Missing):**
- **Mean/Median**: Use when ~5% missing, data is normally distributed
- **Forward Fill**: Time-series data with sequential patterns
- **Drop**: Use only if <2% missing AND data is MCAR (not biased)

**Categorical Columns (Missing):**
- **Mode**: Most frequent category (safe default)
- **Domain Default**: "Unknown", "N/A" (captures missingness explicitly)
- **Drop**: If >20% missing or creates sampling bias
            """)
        with col2:
            st.markdown("""
**Outliers (Strategy):**
- **Keep**: If valid business cases (e.g., large orders, extreme weather)
- **Clip**: If caused by measurement errors, cap at ±3σ
- **Transform**: Log/sqrt for skewed distributions
- **Separate Model**: Train on clean + separate on outliers

**Duplicates:**
- **Full Row**: Remove if truly identical (data entry error)
- **Key-based**: Keep most recent if timestamp available
- **Subset**: Check if duplicate on key columns only
            """)

def _show_data_quality_report(df: pd.DataFrame):
    """Display comprehensive data quality metrics."""
    st.subheader("📊 Data Quality Report")
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
    duplicates_pct = (df.duplicated().sum() / len(df)) * 100 if len(df) > 0 else 0
    numeric_cols = df.select_dtypes(include=np.number).columns
    
    # Outlier count (Z-score > 3)
    outlier_count = 0
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            if df[col].std() > 0:
                z_scores = ((df[col] - df[col].mean()) / df[col].std()).abs()
                outlier_count += (z_scores > 3).sum()
    
    with col1:
        metric_card(
            label="Completeness",
            value=f"{completeness:.1f}%",
            explanation="Non-null values"
        )
    with col2:
        metric_card(
            label="Duplicates",
            value=f"{duplicates_pct:.1f}%",
            explanation=f"{df.duplicated().sum()} rows"
        )
    with col3:
        metric_card(
            label="Outliers (Z>3)",
            value=f"{outlier_count}",
            explanation=f"{numeric_cols.size} numeric cols"
        )
    with col4:
        metric_card(
            label="Shape",
            value=f"{len(df):,} × {len(df.columns)}",
            explanation="Rows × Columns"
        )

def _save_cleaning_report(df: pd.DataFrame, original_df: pd.DataFrame, actions: list):
    """Generate cleaning report for audit trail."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "rows_before": len(original_df),
        "rows_after": len(df),
        "columns": len(df.columns),
        "missing_before": original_df.isnull().sum().sum(),
        "missing_after": df.isnull().sum().sum(),
        "duplicates_removed": original_df.duplicated().sum() - df.duplicated().sum(),
        "actions_taken": actions
    }
    return report

def _missing_values_handler(df: pd.DataFrame, original_df: pd.DataFrame):
    """Module for handling missing data with strategy guidance."""
    st.subheader("🧹 Missing Values Handler")
    
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    
    if missing.empty:
        st.success("✅ No missing values found!")
        return []
    
    st.warning(f"Found missing values in **{len(missing)} columns**.")
    
    # Show missing value distribution
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            missing.to_frame(name="Count").assign(
                Percentage=lambda x: (x["Count"] / len(df) * 100).round(1).astype(str) + "%"
            ),
            use_container_width=True
        )
    with col2:
        st.markdown("**Missing Data Types:**")
        for col in missing.index[:5]:
            dtype = df[col].dtype
            st.text(f"{col}: {dtype}")
    
    _show_cleaning_strategy_guide()
    
    col_to_fix = st.selectbox("Select Column to Fix", missing.index)
    col_dtype = pd.api.types.is_numeric_dtype(df[col_to_fix])
    col_type = "Numeric" if col_dtype else "Categorical"
    
    st.info(f"Column type: **{col_type}** | Missing: **{missing[col_to_fix]}** rows")
    
    # Strategy recommendation
    if col_dtype:
        method = st.selectbox(
            "Imputation Method", 
            ["Mean", "Median", "Mode", "Forward Fill", "Drop Rows", "Custom Value"],
            help="Mean/Median best for ~5% missing. Drop only if <2% missing."
        )
    else:
        method = st.selectbox(
            "Imputation Method",
            ["Mode", "Domain Default", "Drop Rows"],
            help="Mode (most frequent) is safest for categorical. Domain Default captures missingness."
        )
    
    custom_val = None
    if method == "Custom Value":
        custom_val = st.text_input(f"Enter custom value for {col_to_fix}")
    elif method == "Domain Default":
        custom_val = st.text_input(f"Enter domain default (e.g., 'Unknown', 'N/A')")
    
    if st.button(f"Apply '{method}' to {col_to_fix}", type="primary", use_container_width=True):
        with st.status(f"🧹 Applying {method}...", expanded=True) as status:
            missing_before = df[col_to_fix].isnull().sum()
            st.write(f"Processing **{col_to_fix}**: {missing_before} missing values found...")
            
            temp_df = df.copy()
            action_desc = f"{method} on {col_to_fix}"
            
            if method == "Drop Rows":
                rows_before = len(temp_df)
                temp_df = temp_df.dropna(subset=[col_to_fix])
                rows_removed = rows_before - len(temp_df)
                st.write(f"✅ Dropped {rows_removed} rows with missing {col_to_fix}")
                st.info(f"📊 **Report**: {rows_before:,} → {len(temp_df):,} rows | {rows_removed} rows removed")
                
            elif method == "Fill with Mean":
                if col_dtype:
                    fill_val = temp_df[col_to_fix].mean()
                    temp_df[col_to_fix] = temp_df[col_to_fix].fillna(fill_val)
                    st.write(f"✅ Filled {missing_before} values with mean: {fill_val:.2f}")
                    st.info(f"📊 **Report**: Filled with µ={fill_val:.2f}")
                else:
                    st.error("Mean only works for numeric columns.")
                    status.update(label="❌ Error", state="error")
                    return []
                    
            elif method == "Fill with Median":
                if col_dtype:
                    fill_val = temp_df[col_to_fix].median()
                    temp_df[col_to_fix] = temp_df[col_to_fix].fillna(fill_val)
                    st.write(f"✅ Filled {missing_before} values with median: {fill_val:.2f}")
                    st.info(f"📊 **Report**: Filled with Mdn={fill_val:.2f}")
                else:
                    st.error("Median only works for numeric columns.")
                    status.update(label="❌ Error", state="error")
                    return []
                    
            elif method == "Fill with Mode":
                fill_val = temp_df[col_to_fix].mode()[0]
                temp_df[col_to_fix] = temp_df[col_to_fix].fillna(fill_val)
                st.write(f"✅ Filled {missing_before} values with mode: {fill_val}")
                st.info(f"📊 **Report**: Mode frequency = {(temp_df[col_to_fix] == fill_val).sum()}")
                
            elif method == "Forward Fill":
                temp_df[col_to_fix] = temp_df[col_to_fix].ffill()
                st.write(f"✅ Applied forward fill (propagates last known value)")
                st.info(f"📊 **Report**: Propagated {missing_before} values forward")
                
            elif method == "Custom Value":
                if custom_val:
                    temp_df[col_to_fix] = temp_df[col_to_fix].fillna(custom_val)
                    st.write(f"✅ Filled {missing_before} values with: {custom_val}")
                    st.info(f"📊 **Report**: Custom fill = '{custom_val}'")
                    
            elif method == "Domain Default":
                if custom_val:
                    temp_df[col_to_fix] = temp_df[col_to_fix].fillna(custom_val)
                    st.write(f"✅ Filled {missing_before} values with domain default: {custom_val}")
                    st.info(f"📊 **Report**: Domain default = '{custom_val}' (captures missingness)")
            
            missing_after = temp_df[col_to_fix].isnull().sum()
            st.success(f"✅ Completed: {missing_before} → {missing_after} missing values")
            status.update(label="✅ Complete", state="complete")
            _save_changes(temp_df, f"Fixed missing values in {col_to_fix}")
            return [action_desc]
    
    return []

def _duplicates_handler(df: pd.DataFrame, original_df: pd.DataFrame):
    """Module for handling duplicate rows."""
    st.subheader("👯 Duplicates")
    
    n_dupes = df.duplicated().sum()
    if n_dupes == 0:
        st.success("✅ No duplicate rows found.")
        return []
    
    col1, col2 = st.columns(2)
    with col1:
        st.warning(f"Found **{n_dupes}** duplicate rows ({n_dupes/len(df)*100:.1f}% of data).")
    with col2:
        st.info("**Note**: Duplicates are determined by all columns. Use 'Subset' if needed.")
    
    # Option to check duplicates on subset of columns
    with st.expander("🔍 **Check Duplicates on Specific Columns**", expanded=False):
        subset_cols = st.multiselect("Select columns (leave empty for all):", df.columns)
        if subset_cols:
            n_subset_dupes = df.duplicated(subset=subset_cols).sum()
            st.write(f"Duplicates on {subset_cols}: **{n_subset_dupes}** rows")
    
    action = st.radio(
        "How to handle duplicates?",
        ["Remove All Duplicates", "Remove Duplicates (Keep First)", "Remove Duplicates (Keep Last)", "Show Duplicates Only"],
        help="Keep First: retains original entry. Keep Last: retains most recent."
    )
    
    if st.button("Apply Action", type="primary", use_container_width=True):
        with st.status("🔄 Handling duplicates...", expanded=True) as status:
            rows_before = len(df)
            temp_df = df.copy()
            
            if action == "Remove All Duplicates":
                temp_df = temp_df.drop_duplicates()
                removed = rows_before - len(temp_df)
                st.write(f"✅ Removed {removed} duplicate rows")
                st.info(f"📊 **Report**: {rows_before:,} rows → {len(temp_df):,} rows")
                action_desc = f"Removed {removed} full duplicates"
                
            elif action == "Remove Duplicates (Keep First)":
                temp_df = temp_df.drop_duplicates(keep='first')
                removed = rows_before - len(temp_df)
                st.write(f"✅ Removed {removed} duplicate rows (kept first occurrence)")
                st.info(f"📊 **Report**: {rows_before:,} rows → {len(temp_df):,} rows | Original entries preserved")
                action_desc = f"Removed {removed} duplicates (keep first)"
                
            elif action == "Remove Duplicates (Keep Last)":
                temp_df = temp_df.drop_duplicates(keep='last')
                removed = rows_before - len(temp_df)
                st.write(f"✅ Removed {removed} duplicate rows (kept last occurrence)")
                st.info(f"📊 **Report**: {rows_before:,} rows → {len(temp_df):,} rows | Most recent entries preserved")
                action_desc = f"Removed {removed} duplicates (keep last)"
                
            elif action == "Show Duplicates Only":
                dup_rows = df[df.duplicated(keep=False)].sort_values(by=list(df.columns))
                st.write(f"Showing {len(dup_rows)} rows that are duplicated:")
                st.dataframe(dup_rows, use_container_width=True)
                status.update(label="✅ Displayed", state="complete")
                return []
            
            st.success(f"✅ Deduplication complete!")
            status.update(label="✅ Duplicates handled", state="complete")
            _save_changes(temp_df, f"Removed duplicates from dataset")
            return [action_desc]
    
    return []

def _outliers_handler(df: pd.DataFrame, original_df: pd.DataFrame):
    """Module for handling outliers with strategy guidance."""
    st.subheader("📈 Outlier Detection & Management")
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) == 0:
        st.info("No numeric columns for outlier detection.")
        return []
    
    target_col = st.selectbox("Select Numeric Column", num_cols)
    method = st.selectbox(
        "Outlier Detection Method",
        ["Z-Score (|z| > σ)", "IQR (Interquartile Range)"],
        help="Z-Score: beyond ±3σ. IQR: beyond Q1-1.5×IQR or Q3+1.5×IQR"
    )
    
    col_data = df[target_col].dropna()
    
    if method == "Z-Score (|z| > σ)":
        threshold = st.slider("Z-Score Threshold", 1.0, 5.0, 3.0, step=0.5,
                             help="3.0 = beyond ±3σ (99.7% within). 2.0 = beyond ±2σ (95% within)")
        z_scores = ((col_data - col_data.mean()) / col_data.std()).abs()
        outlier_mask = z_scores > threshold
        outliers = col_data[outlier_mask]
        outlier_indices = outlier_mask[outlier_mask].index
    else:  # IQR
        Q1 = col_data.quantile(0.25)
        Q3 = col_data.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        outliers = col_data[outlier_mask]
        outlier_indices = outlier_mask[outlier_mask].index
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Q1 (25%)", f"{Q1:.2f}")
        with col2:
            st.metric("Q3 (75%)", f"{Q3:.2f}")
        with col3:
            st.metric("Lower Bound", f"{lower_bound:.2f}")
        with col4:
            st.metric("Upper Bound", f"{upper_bound:.2f}")
    
    st.warning(f"Found **{len(outliers)}** outliers ({len(outliers)/len(col_data)*100:.1f}% of data)")
    
    # Show outlier statistics
    if len(outliers) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Min Outlier", f"{outliers.min():.2f}")
        with col2:
            st.metric("Max Outlier", f"{outliers.max():.2f}")
        with col3:
            st.metric("Outlier Count", f"{len(outliers)}")
    
    if len(outliers) == 0:
        st.success("✅ No outliers detected!")
        return []
    
    # Show first 10 outliers
    with st.expander("👁️ **Preview Outlier Values**", expanded=False):
        st.dataframe(outliers.head(10).to_frame(), use_container_width=True)
    
    _show_cleaning_strategy_guide()
    
    action = st.radio(
        "How to handle outliers?",
        ["Clip (Cap at bounds)", "Drop Rows", "Log Transform", "Keep (Valid business cases)"],
        help="Clip: Safe for measurement errors. Drop: If truly invalid. Log: For right-skewed data."
    )
    
    if st.button("Apply Action", type="primary", use_container_width=True):
        with st.status(f"⚙️ Handling {len(outliers)} outliers...", expanded=True) as status:
            temp_df = df.copy()
            rows_before = len(temp_df)
            action_desc = f"Handled {len(outliers)} outliers with {action}"
            
            if action == "Clip (Cap at bounds)":
                if method == "Z-Score (|z| > σ)":
                    mean = col_data.mean()
                    std = col_data.std()
                    lower = mean - threshold * std
                    upper = mean + threshold * std
                else:
                    lower = lower_bound
                    upper = upper_bound
                
                temp_df[target_col] = temp_df[target_col].clip(lower, upper)
                st.write(f"✅ Clipped {len(outliers)} outliers to [{lower:.2f}, {upper:.2f}]")
                st.info(f"📊 **Report**: {rows_before:,} rows | {len(outliers)} values capped")
                
            elif action == "Drop Rows":
                temp_df = temp_df.drop(outlier_indices)
                rows_removed = rows_before - len(temp_df)
                st.write(f"✅ Removed {rows_removed} rows containing outliers")
                st.info(f"📊 **Report**: {rows_before:,} rows → {len(temp_df):,} rows | {rows_removed} rows deleted")
                
            elif action == "Log Transform":
                if col_data.min() <= 0:
                    st.warning("Log transform requires all positive values. Shifting data...")
                    shift = abs(col_data.min()) + 1
                    temp_df[target_col] = np.log1p(temp_df[target_col] + shift)
                    st.write(f"✅ Applied log(x + {shift}) transform")
                else:
                    temp_df[target_col] = np.log(temp_df[target_col])
                    st.write(f"✅ Applied log transform (compresses extreme values)")
                st.info(f"📊 **Report**: Distribution shape normalized | {rows_before:,} rows preserved")
                
            elif action == "Keep (Valid business cases)":
                st.success("✅ Outliers retained (flagged as valid data)")
                st.info(f"📊 **Report**: {rows_before:,} rows | {len(outliers)} potential outliers marked for review")
                return []
            
            st.success(f"✅ Outlier handling complete!")
            status.update(label="✅ Outliers handled", state="complete")
            _save_changes(temp_df, f"Applied {action} to {len(outliers)} outliers")
            return [action_desc]
    
    return []

def main():
    config = load_config()
    inject_custom_css()
    ai_sidebar_assistant()
    
    st.markdown(
        """
        <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
            <div style="font-size: 24px; font-weight: 800;">🧼 Data Cleaning</div>
            <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Prepare your data for success with strategic cleaning.</div>
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
    
    # ========================================================================
    # WORKFLOW STEPS
    # ========================================================================
    
    instruction_block(
        title="Data Cleaning Workflow",
        lines=[
            "**Assess Data Quality** → Identify missing, duplicates, outliers",
            "**Choose Strategy** → Different handling for numeric vs categorical",
            "**Apply Fixes** → Use appropriate imputation, removal, or transformation",
            "**Review Changes** → Compare before/after metrics",
            "**Document Actions** → Keep audit trail for reproducibility"
        ]
    )
    
    _show_data_quality_report(df)
    
    st.divider()
    
    # ========================================================================
    # INTERACTIVE CLEANING TABS
    # ========================================================================
    
    tab1, tab2, tab3, tab4 = st.tabs(["Missing Data", "Duplicates", "Outliers", "Summary"])
    
    original_df = raw_df if raw_df is not None else df.copy()
    all_actions = []
    
    with tab1:
        actions = _missing_values_handler(df, original_df)
        all_actions.extend(actions)
        
    with tab2:
        actions = _duplicates_handler(df, original_df)
        all_actions.extend(actions)
        
    with tab3:
        actions = _outliers_handler(df, original_df)
        all_actions.extend(actions)
        
    with tab4:
        st.subheader("📋 Cleaning Summary")
        current_clean_df = get_clean_df(st.session_state)
        if current_clean_df is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Original Dataset:**")
                st.text(f"Rows: {len(raw_df):,}\nCols: {len(raw_df.columns)}\nSize: {raw_df.memory_usage(deep=True).sum()/1024/1024:.1f}MB")
            with col2:
                st.write("**Cleaned Dataset:**")
                st.text(f"Rows: {len(current_clean_df):,}\nCols: {len(current_clean_df.columns)}\nSize: {current_clean_df.memory_usage(deep=True).sum()/1024/1024:.1f}MB")
            
            st.info(f"**Changes**: {len(raw_df) - len(current_clean_df):,} rows removed | {raw_df.isnull().sum().sum() - current_clean_df.isnull().sum().sum()} missing values fixed")
        
    st.divider()
    
    # ========================================================================
    # EDUCATIONAL CONTENT
    # ========================================================================
    
    st.markdown("## 🎓 Data Cleaning Guide")
    
    concept_explainer(
        title="Why Data Cleaning is Critical",
        explanation=(
            "Poor data quality leads to biased models, unreliable predictions, and wasted computational resources. "
            "Cleaning ensures your ML model learns from valid, representative signals."
        ),
        real_world_example=(
            "**Loan Risk Model**: Missing income (impute median), duplicate applications (remove), extreme debt-to-income "
            "ratios (clip to ±3σ) → Creates fair, accurate risk scores."
        ),
    )
    
    col1, col2 = st.columns(2)
    with col1:
        beginner_tip(
            "🎯 **Tip 1**: Prefer imputation over deletion. Removing rows loses information; imputation preserves sample size."
        )
        beginner_tip(
            "🎯 **Tip 2**: For missing values ~5%, mean/median is safe. For >20%, consider separate 'missing' indicator."
        )
    
    with col2:
        beginner_tip(
            "🎯 **Tip 3**: Outliers can be valid (e.g., celebrities in income data). Use domain knowledge, not just statistics."
        )
        beginner_tip(
            "🎯 **Tip 4**: Document ALL cleaning decisions. This audit trail is essential for model reproducibility."
        )
    
    common_mistakes_panel({
        "Filling text columns with numeric methods": "Use mode or domain defaults (e.g., 'Unknown') for categories.",
        "Dropping too many rows": "Consider imputation or separate model for missing patterns.",
        "Ignoring temporal patterns": "Forward-fill for time-series; otherwise, use mean/median.",
        "Creating target leakage": "Never use target values to guide cleaning decisions.",
        "Forgetting to record changes": "Keep cleaning reports for reproducibility and model auditing.",
        "Over-aggressive outlier removal": "Confirm whether extremes are valid before deletion.",
    })
    
    page_navigation("3")

if __name__ == "__main__":
    main()
