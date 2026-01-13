from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="DataScope Pro - EDA", layout="wide", initial_sidebar_state="expanded")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy import stats

from src.core.config import load_config
from src.core.state import get_clean_df, get_df, set_df, set_clean_df
from src.core.ui import app_header, sidebar_dataset_status, instruction_block, page_navigation
from src.core.standardized_ui import (
    standard_section_header,
    beginner_tip,
    concept_explainer,
    common_mistakes_panel,
)
from src.core.styles import render_stat_card, inject_custom_css
from src.core.premium_styles import inject_premium_css
from src.core.modern_components import smart_data_preview, auto_data_profile
from src.core.ai_helper import ai_help_button, ai_interpretation_box, ai_sidebar_assistant
from src.data.eda import correlation_matrix, describe_numeric, detect_anomaly_zscore, summarize, value_counts


config = load_config()

# Apply custom CSS
inject_custom_css()

# Add AI assistant for interpretation help
ai_sidebar_assistant()

app_header(
    config,
    page_title="Data Analysis & EDA",
    subtitle="Explore quality, patterns, and relationships before you model",
    icon="📊"
)

instruction_block(
    "🎯 Complete EDA Workflow - Step by Step",
    [
        "📊 **Step 1: Data Quality Check** - Review completeness, missing values, and data types",
        "🔍 **Step 2: Understand Distributions** - Identify skewness, outliers, and value ranges",
        "🔗 **Step 3: Find Relationships** - Discover correlations and feature interactions",
        "⚠️ **Step 4: Detect Anomalies** - Flag extreme values that may need attention",
        "📈 **Step 5: Statistical Testing** - Validate relationships and differences",
        "🎯 **Step 6: Profile Deep Dive** - Column-by-column analysis for comprehensive understanding",
        "⏰ **Step 7: Time Patterns** - Identify temporal trends if time data exists",
        "🔬 **Step 8: Advanced Analysis** - Feature variance, interactions, and transformations",
    ],
)

st.markdown("""
### 📚 Understanding EDA (Exploratory Data Analysis)

**What is EDA?**
EDA is your "data detective work" phase - systematically investigating your dataset to understand:
- **Quality**: Missing values, duplicates, errors
- **Structure**: Data types, value ranges, distributions
- **Relationships**: Correlations, patterns, dependencies
- **Anomalies**: Outliers, unusual values, data quality issues

**Why is EDA critical?**
- **Prevents modeling mistakes**: Catch data issues before they poison your model
- **Guides feature engineering**: Discover which transformations help
- **Validates assumptions**: Ensure data matches your expectations
- **Improves results**: Clean, well-understood data = better models

**Real-world example:**
Predicting customer churn? EDA reveals:
- 30% missing contract dates → Need imputation strategy
- Tenure highly skewed → Consider log transformation
- Support calls strongly correlated with churn → Key feature!
- Age has outliers (150 years) → Data entry errors to fix
""")

# Beginner quick start
st.success(
    "🚀 **Quick Start Guide**: (1) Check Data Quality metrics below → (2) Use tabs to explore patterns → "
    "(3) Review interpretation guides in each section → (4) Fix issues in Data Cleaning → (5) Proceed to modeling",
    icon="✅",
)

beginner_tip(
    "💡 **First-time user?** Focus on the Overview and Relationships tabs first. "
    "They give you the fastest insight into your data quality and structure."
)

raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)

sidebar_dataset_status(raw_df, clean_df)
df = clean_df if clean_df is not None else raw_df

# Sample dataset helper for quick start
if df is None:
    st.warning("No data loaded. Use the Cleaning page or load a quick sample to explore.")
    col_sample, col_upload = st.columns([1.2, 1])
    with col_sample:
        if st.button("Load sample dataset (Retail Sales)", type="primary", width="stretch"):
            sample = pd.DataFrame(
                {
                    "date": pd.date_range("2024-01-01", periods=180, freq="D"),
                    "store_id": np.random.choice(["A", "B", "C"], size=180),
                    "promo": np.random.choice([0, 1], size=180, p=[0.7, 0.3]),
                    "sales": np.random.normal(loc=12000, scale=3200, size=180).round(0),
                    "transactions": np.random.normal(loc=430, scale=90, size=180).round(0),
                }
            )
            set_df(st.session_state, sample, source="sample_retail")
            set_clean_df(st.session_state, sample.copy())
            st.success("Sample dataset loaded—scroll down to explore.")
            raw_df = sample
            clean_df = sample
            df = sample
    with col_upload:
        st.caption("Tip: You can still upload your own data in the Cleaning page.")

if df is None:
    st.info("Load data in Data Cleaning page first.")
    st.stop()

# Optional: allow users to edit data manually for this session
st.divider()
st.markdown("## ✍️ Manual Data Editing (optional)")
st.caption("Make quick fixes to the data in-session. You can switch between original and edited data.")

base_df = df
editable_limit = 500  # avoid huge interactive tables

if "eda_manual_df" not in st.session_state:
    st.session_state["eda_manual_df"] = base_df.copy().head(editable_limit)

data_mode = st.radio(
    "Choose data source",
    ["Use cleaned/raw data", "Use manually edited data"],
    horizontal=True,
    index=0,
)

with st.expander("Edit data (session-scoped)", expanded=False):
    if len(base_df) > editable_limit:
        st.warning(f"Editing limited to first {editable_limit} rows for performance. Add filters in Cleaning to narrow further.")
        edit_target = base_df.head(editable_limit).copy()
    else:
        edit_target = base_df.copy()

    edited = st.data_editor(
        edit_target,
        num_rows="dynamic",
        width="stretch",
        height=400,
    )
    st.session_state["eda_manual_df"] = edited
    st.caption("Edits persist only in this browser session. Use downloads to save externally if needed.")

# Pick active dataframe for downstream analysis
if data_mode == "Use manually edited data":
    if len(base_df) > editable_limit:
        # Combine edited head with untouched remainder to keep row count aligned
        df = pd.concat([st.session_state["eda_manual_df"], base_df.iloc[editable_limit:]], ignore_index=True)
    else:
        df = st.session_state["eda_manual_df"].copy()
else:
    df = base_df

# Automatic Data Profiling at the top
st.markdown("## 🔍 Automatic Data Insights")
auto_data_profile(df)

st.divider()

summary = summarize(df)

# Smart data preview
smart_data_preview(df, title="Dataset Quick View", show_stats=True, max_rows=10)

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📊 Overview", 
    "🔗 Relationships", 
    "📈 Distributions", 
    "⚠️ Anomalies",
    "🧪 Statistical Tests",
    "🎯 Data Profiling",
    "⏰ Time Series",
    "🔬 Advanced Analysis"
])

with tab1:
    st.markdown("## 📊 Step 1: Data Quality Overview")
    st.markdown("""
    **What to look for:**
    - **Rows (Sample Size)**: More rows = more reliable patterns (typically 100+ for simple models, 1000+ for complex)
    - **Columns (Features)**: Too many? Risk of overfitting. Too few? May miss important patterns
    - **Missing %**: Your data completeness score - lower is better
    - **Data Types**: Mix of numeric and categorical gives richer modeling options
    """)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("📊 Total Rows", f"{summary.n_rows:,}")
        if summary.n_rows < 100:
            st.caption("⚠️ Small sample - results may be unreliable")
        elif summary.n_rows < 1000:
            st.caption("✓ Moderate sample - good for simple models")
        else:
            st.caption("✅ Large sample - excellent for modeling")
    with c2:
        st.metric("📑 Total Columns", f"{summary.n_cols:,}")
        ratio = summary.n_rows / summary.n_cols if summary.n_cols > 0 else 0
        if ratio < 10:
            st.caption("⚠️ High dimension - consider feature selection")
        else:
            st.caption("✅ Good feature-to-sample ratio")
    with c3:
        pct_missing = (summary.missing_total / (summary.n_rows * summary.n_cols) * 100) if summary.n_rows * summary.n_cols > 0 else 0
        st.metric("⚠️ Missing Data %", f"{pct_missing:.1f}%")
        if pct_missing < 1:
            st.caption("✅ Excellent completeness")
        elif pct_missing < 5:
            st.caption("✓ Good - minor imputation needed")
        elif pct_missing < 20:
            st.caption("⚠️ Moderate - plan imputation strategy")
        else:
            st.caption("🚨 High - investigate data collection")
    with c4:
        st.metric("🔢 Numeric Columns", f"{len(summary.numeric_cols):,}")
        st.caption(f"📝 Categorical: {len(summary.categorical_cols)}")
    
    st.divider()
    
    st.subheader("Missingness Analysis")
    missing_df = (
        (df.isna().sum().sort_values(ascending=False))
        .reset_index()
        .rename(columns={"index": "column", 0: "missing"})
    )
    missing_df.columns = ["column", "missing"]
    missing_df = missing_df[missing_df["missing"] > 0]
    
    if missing_df.empty:
        st.success("✅ No missing values detected - data is complete!")
    else:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                missing_df.head(20), 
                x="column", 
                y="missing", 
                title="Top 20 columns with missing values",
                color="missing",
                color_continuous_scale="Reds"
            )
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            st.markdown("**Missing Data Strategy:**")
            st.markdown("""
            1. **< 1% missing**: Safe to drop or ignore
            2. **1-5% missing**: Impute with mean/median
            3. **5-20% missing**: Use advanced imputation
            4. **> 20% missing**: Consider dropping column
            """)
    
    st.divider()
    
    st.subheader("Numeric Summary Statistics")
    desc = describe_numeric(df)
    if desc.empty:
        st.info("No numeric columns in dataset.")
    else:
        st.dataframe(desc, width="stretch")
        
        with st.expander("📚 Understanding Statistics"):
            st.markdown("""
            - **Count**: Number of non-null values
            - **Mean**: Average value
            - **Std**: Standard deviation (spread)
            - **Min/Max**: Minimum and maximum values
            - **25%/50%/75%**: Quartiles (Q1, median, Q3)
            """)

with tab2:
    st.markdown("## 🔗 Step 2: Feature Relationships & Correlations")
    st.markdown("""
    **📚 Correlation Guide:**
    
    **What is correlation?**
    - Measures linear relationship strength between two numeric variables (-1 to +1)
    
    **How to interpret correlation values:**
    - **±0.9 to ±1.0**: Very strong relationship (almost perfect)
    - **±0.7 to ±0.9**: Strong relationship (highly correlated)
    - **±0.4 to ±0.7**: Moderate relationship (noticeable pattern)
    - **±0.2 to ±0.4**: Weak relationship (slight connection)
    - **0.0 to ±0.2**: Very weak/no linear relationship
    
    **Positive vs Negative:**
    - **Positive (+)**: Variables move together (age ↑ → experience ↑)
    - **Negative (−)**: Variables move oppositely (price ↑ → demand ↓)
    
    **Why it matters for modeling:**
    - **High correlations (>0.9)**: May indicate redundant features - consider removing one
    - **Moderate correlations (0.4-0.7)**: Good predictive features to keep
    - **Target correlations**: Features highly correlated with target are typically most important
    - **Multicollinearity**: When features correlate with each other, it can confuse some models
    
    **⚠️ Important caveat:**
    - Correlation ≠ Causation! High correlation doesn't prove one causes the other
    - Only captures LINEAR relationships (may miss complex patterns)
    """)
    
    if summary.numeric_cols:
        cols = st.multiselect(
            "Select numeric columns for correlation",
            options=summary.numeric_cols,
            default=summary.numeric_cols[: min(15, len(summary.numeric_cols))],
            help="Select up to 15 columns for clearer visualization"
        )
        
        corr = correlation_matrix(df, cols=cols)
        if not corr.empty:
            col1, col2 = st.columns([3, 1])
            
            with col2:
                mask_upper = st.checkbox("Mask upper triangle", value=True, help="Reduce redundancy in visualization")
                corr_threshold = st.slider("Show correlations > |threshold|", 0.0, 1.0, 0.3, 0.05)
            
            with col1:
                corr_to_plot = corr.copy()
                if mask_upper:
                    corr_to_plot = corr_to_plot.where(np.tril(np.ones_like(corr_to_plot), k=0).astype(bool))
                
                fig = px.imshow(
                    corr_to_plot,
                    text_auto=True,
                    aspect="auto",
                    title="Pearson Correlation Matrix",
                    color_continuous_scale="RdBu",
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig, width="stretch")
            
            # Find strong correlations
            st.subheader("🔍 Strong Correlations (> threshold)")
            strong_corrs = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > corr_threshold:
                        strong_corrs.append({
                            "Var1": corr.columns[i],
                            "Var2": corr.columns[j],
                            "Correlation": corr.iloc[i, j]
                        })
            
            if strong_corrs:
                strong_df = __import__('pandas').DataFrame(strong_corrs).sort_values('Correlation', key=abs, ascending=False)
                st.dataframe(strong_df, width="stretch")
            else:
                st.caption("No correlations exceed threshold")
    else:
        st.info("No numeric columns available")
    
    st.divider()
    
    st.subheader("Category vs Numeric")
    if summary.categorical_cols and summary.numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            cat_col = st.selectbox("Category column", options=summary.categorical_cols, key="cat_box")
        with col2:
            num_col = st.selectbox("Numeric column", options=summary.numeric_cols, key="num_box")
        
        fig = px.box(
            df,
            x=cat_col,
            y=num_col,
            points="outliers",
            title=f"📊 {num_col} distribution by {cat_col}",
            color=cat_col
        )
        fig.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Need categorical and numeric columns")

with tab3:
    st.markdown("## 📈 Step 3: Distribution Analysis & Patterns")
    st.markdown("""
    **📚 Understanding Distributions:**
    
    **Why distributions matter:**
    - Reveal data quality issues (gaps, outliers, errors)
    - Guide preprocessing decisions (scaling, transformations)
    - Indicate appropriate modeling techniques
    - Show if assumptions (like normality) are met
    
    **Distribution shapes to recognize:**
    - **Normal (bell curve)**: Symmetric, most data near center - ideal for many statistical methods
    - **Skewed right**: Long tail to right, mean > median - common in income, prices (consider log transform)
    - **Skewed left**: Long tail to left, mean < median - less common (consider reflection + log)
    - **Uniform**: Flat, all values equally likely - rare in real data
    - **Bimodal**: Two peaks - may indicate mixed populations (consider separate analysis)
    - **Heavy-tailed**: More extreme values than normal - use robust methods
    
    **Key statistics explained:**
    - **Skewness**: Measures asymmetry
      - Near 0: Symmetric
      - Positive: Right-skewed
      - Negative: Left-skewed
    - **Kurtosis**: Measures tail heaviness (outlier tendency)
      - Near 0: Normal-like tails
      - Positive: Heavy tails (more outliers)
      - Negative: Light tails (fewer outliers)
    - **CV (Coefficient of Variation)**: Std dev as % of mean
      - < 15%: Low variability
      - 15-30%: Moderate variability
      - > 30%: High variability
    """)
    
    if summary.numeric_cols:
        col1, col2 = st.columns(2)
        with col1:
            num_col = st.selectbox("Select numeric column", options=summary.numeric_cols)
        with col2:
            nbins = st.slider("Number of bins", 10, 100, 60)
        
        fig = px.histogram(
            df,
            x=num_col,
            nbins=nbins,
            title=f"Distribution of {num_col}",
            marginal="rug",
            hover_data={num_col: ":.4f"}
        )
        st.plotly_chart(fig, width="stretch")
        
        # Distribution insights
        col_data = df[num_col].dropna()
        if len(col_data) > 0:
            skewness = col_data.skew()
            kurtosis = col_data.kurtosis()
            
            st.subheader("Distribution Insights")
            i1, i2, i3 = st.columns(3)
            with i1:
                st.metric("Skewness", f"{skewness:.3f}")
                if abs(skewness) < 0.5:
                    st.caption("✅ Fairly symmetric")
                elif skewness > 0.5:
                    st.caption("⚠️ Right-skewed")
                else:
                    st.caption("⚠️ Left-skewed")
            with i2:
                st.metric("Kurtosis", f"{kurtosis:.3f}")
            with i3:
                st.metric("CV (Coefficient of Variation)", f"{(col_data.std() / col_data.mean() * 100):.1f}%")
    else:
        st.info("No numeric columns")
    
    st.divider()
    
    st.subheader("Categorical Value Counts")
    if summary.categorical_cols:
        cat_col = st.selectbox("Select categorical column", options=summary.categorical_cols, key="cat_vc")
        topn = st.slider("Top N categories", 5, 50, 20)
        
        vc = value_counts(df, cat_col, top_n=int(topn))
        
        fig = px.bar(
            vc,
            x="value",
            y="count",
            title=f"Top {topn} values: {cat_col}",
            color="count",
            color_continuous_scale="Viridis"
        )
        fig.update_layout(xaxis_tickangle=-45, height=400)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No categorical columns")
    
    st.divider()
    
    st.subheader("Pairwise Relationships")
    pair_cols = st.multiselect(
        "Select 2-6 numeric columns",
        options=summary.numeric_cols,
        default=summary.numeric_cols[: min(3, len(summary.numeric_cols))],
        max_selections=6
    )
    
    if pair_cols:
        fig = px.scatter_matrix(
            df[pair_cols].dropna(),
            dimensions=pair_cols,
            height=600,
            title="🔗 Pairwise Relationships",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption("Select 2-6 numeric columns to see pairwise scatter plots")

with tab4:
    st.markdown("## ⚠️ Step 4: Anomaly Detection & Outlier Analysis")
    st.markdown("""
    **📚 What are anomalies/outliers?**
    Data points that significantly deviate from the typical pattern. They can be:
    - **Errors**: Data entry mistakes, sensor failures (150-year-old customer)
    - **Rare events**: Legitimate but unusual (billionaire in income data)
    - **Important signals**: Fraud, disease, system failures
    
    **Z-Score Method Explained:**
    - **Z-score** = (value - mean) / std_dev
    - Measures how many standard deviations a point is from the mean
    - Works best for normally-distributed data
    
    **Threshold Guide:**
    - **Z = 2.0**: Flags ~5% of data (loose, catches more)
    - **Z = 2.5**: Flags ~1.2% of data (moderate)
    - **Z = 3.0**: Flags ~0.3% of data (strict, standard choice) ✅
    - **Z = 4.0**: Flags ~0.006% of data (very strict, only extremes)
    
    **What to do with anomalies:**
    1. **Investigate**: Check if they're errors or legitimate
    2. **Context matters**: A 150-year age is wrong; a $1M sale might be real
    3. **Options**:
       - **Remove**: If clearly errors
       - **Cap/floor**: Limit to reasonable max/min (winsorization)
       - **Transform**: Use robust scalers or log transforms
       - **Keep**: If they're important (fraud detection)
       - **Separate model**: Handle outliers with different logic
    
    **Best practices:**
    - Don't blindly remove all flagged points
    - Document decisions about outlier treatment
    - Check domain experts for context
    - Some algorithms (tree-based) handle outliers naturally
    """)
    
    if summary.numeric_cols:
        cols_a = st.multiselect(
            "Select numeric columns to scan",
            options=summary.numeric_cols,
            default=summary.numeric_cols[: min(5, len(summary.numeric_cols))]
        )
        
        col1, col2 = st.columns(2)
        with col1:
            z_threshold = st.slider("Z-score threshold", 2.0, 6.0, 3.0, 0.1, help="Higher = stricter anomaly detection")
        with col2:
            st.metric("Current threshold", f"{z_threshold:.1f}σ", help="Standard deviations from mean")
        
        if st.button("🔍 Scan for anomalies", width="stretch", disabled=(len(cols_a) == 0)):
            with st.spinner("Analyzing data for anomalies..."):
                anomalies = detect_anomaly_zscore(df, cols=cols_a, z=z_threshold)
                
                if anomalies.empty:
                    st.success("✅ No anomalies detected!")
                else:
                    st.warning(f"⚠️ Found {len(anomalies)} anomalies")
                    st.dataframe(anomalies, width="stretch")
                    
                    # Visualization
                    if len(cols_a) > 0:
                        fig = px.scatter(
                            df[[cols_a[0]]].reset_index(),
                            x="index",
                            y=cols_a[0],
                            title=f"Anomalies in {cols_a[0]} (red = anomalous)",
                            color=[i in anomalies.index for i in range(len(df))],
                            color_discrete_map={True: "red", False: "blue"}
                        )
                        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No numeric columns to analyze")
    
    st.divider()
    
    st.subheader("📚 Anomaly Detection Guide")
    with st.expander("Learn about Z-score anomaly detection"):
        st.markdown("""
        **Z-score**: Measures how many standard deviations a point is from the mean.
        
        - **Z = 2.0**: ~95% of data (2-3% are outliers)
        - **Z = 3.0**: ~99.7% of data (0.3% are outliers) ✅ **Default**
        - **Z = 4.0**: Very strict (0.01% outliers)
        
        **Interpretation:**
        - Flagged rows deviate significantly from the typical distribution
        - Could indicate data entry errors, sensor failures, or genuine extremes
        - Review context before removing
        """)

with tab5:
    st.markdown("## 🧪 Step 5: Statistical Hypothesis Testing")
    st.markdown("""
    **📚 Understanding Statistical Tests:**
    
    **What is hypothesis testing?**
    A formal method to determine if observed patterns are statistically significant (not due to random chance).
    
    **Key concepts:**
    - **Null Hypothesis (H0)**: "No effect" / "No difference" / "No relationship"
    - **Alternative Hypothesis (H1)**: "There IS an effect/difference/relationship"
    - **P-value**: Probability of seeing these results if H0 is true
      - **< 0.05**: Reject H0, result is "statistically significant" ✅
      - **≥ 0.05**: Fail to reject H0, not enough evidence
    - **α (alpha)**: Significance level, typically 0.05 (5% risk of false positive)
    
    **When to use each test:**
    
    🔹 **Normality Test (Shapiro-Wilk)**
    - **Purpose**: Check if data follows normal distribution
    - **When**: Before using parametric tests (t-test, ANOVA)
    - **Interpretation**: p > 0.05 → Normal; p < 0.05 → Not normal
    - **If not normal**: Use non-parametric tests or transformations
    
    🔹 **T-Test (Independent Samples)**
    - **Purpose**: Compare means of 2 groups
    - **Example**: Average salary of men vs women
    - **Assumptions**: Normal distribution, independent samples
    - **Interpretation**: p < 0.05 → Groups have significantly different means
    
    🔹 **ANOVA (One-Way)**
    - **Purpose**: Compare means across 3+ groups
    - **Example**: Test scores across multiple schools
    - **Interpretation**: p < 0.05 → At least one group differs
    - **Follow-up**: If significant, use post-hoc tests to find which pairs differ
    
    🔹 **Chi-Square Test**
    - **Purpose**: Test relationship between categorical variables
    - **Example**: Is gender related to product preference?
    - **Interpretation**: p < 0.05 → Variables are dependent (associated)
    
    🔹 **Correlation Test (Pearson)**
    - **Purpose**: Test if correlation is significant (not just by chance)
    - **Interpretation**: p < 0.05 → Correlation is real, not random
    
    **⚠️ Important caveats:**
    - **Statistical significance ≠ practical importance**: p < 0.05 with tiny effect may not matter
    - **Multiple testing**: Running many tests increases false positives (consider corrections)
    - **Sample size matters**: Very large samples can find "significant" trivial effects
    - **Context is key**: Always interpret results with domain knowledge
    """)
    
    test_type = st.radio(
        "Select test type",
        ["Normality Test", "T-Test (2 groups)", "ANOVA (3+ groups)", "Chi-Square Test", "Correlation Test"],
        horizontal=True
    )
    
    if test_type == "Normality Test":
        st.markdown("**Shapiro-Wilk Normality Test** - Tests if data follows normal distribution")
        if summary.numeric_cols:
            col = st.selectbox("Select numeric column", summary.numeric_cols, key="norm_test")
            if st.button("Run Test", key="btn_norm"):
                data = df[col].dropna()
                if len(data) > 3:
                    statistic, p_value = stats.shapiro(data)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Test Statistic", f"{statistic:.4f}")
                    with col2:
                        st.metric("P-Value", f"{p_value:.4f}")
                    with col3:
                        if p_value > 0.05:
                            st.success("✅ Normal")
                        else:
                            st.warning("⚠️ Not Normal")
                    
                    st.info(f"**Interpretation:** {'Data appears normally distributed' if p_value > 0.05 else 'Data does NOT follow normal distribution'} (α=0.05)")
                    
                    # QQ Plot
                    from scipy import stats as sp_stats
                    fig = go.Figure()
                    qq = sp_stats.probplot(data, dist="norm")
                    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][1], mode='markers', name='Data'))
                    fig.add_trace(go.Scatter(x=qq[0][0], y=qq[0][0], mode='lines', name='Normal', line=dict(color='red', dash='dash')))
                    fig.update_layout(title="Q-Q Plot", xaxis_title="Theoretical Quantiles", yaxis_title="Sample Quantiles", height=400)
                    st.plotly_chart(fig, width="stretch")
                else:
                    st.error("Need at least 3 data points")
        else:
            st.info("No numeric columns available")
    
    elif test_type == "T-Test (2 groups)":
        st.markdown("**Independent T-Test** - Compare means of two groups")
        if summary.categorical_cols and summary.numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox("Grouping column (2 categories)", summary.categorical_cols, key="ttest_group")
            with col2:
                value_col = st.selectbox("Value column (numeric)", summary.numeric_cols, key="ttest_val")
            
            if st.button("Run T-Test", key="btn_ttest"):
                groups = df[group_col].value_counts()
                if len(groups) == 2:
                    group_names = groups.index[:2]
                    group1 = df[df[group_col] == group_names[0]][value_col].dropna()
                    group2 = df[df[group_col] == group_names[1]][value_col].dropna()
                    
                    if len(group1) > 1 and len(group2) > 1:
                        statistic, p_value = stats.ttest_ind(group1, group2)
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(f"{group_names[0]} Mean", f"{group1.mean():.2f}")
                        with col2:
                            st.metric(f"{group_names[1]} Mean", f"{group2.mean():.2f}")
                        with col3:
                            st.metric("T-Statistic", f"{statistic:.4f}")
                        with col4:
                            st.metric("P-Value", f"{p_value:.4f}")
                        
                        if p_value < 0.05:
                            st.success("✅ **Significant difference** between groups (α=0.05)")
                        else:
                            st.info("⚪ No significant difference between groups (α=0.05)")
                    else:
                        st.error("Each group needs at least 2 data points")
                else:
                    st.error(f"Column {group_col} has {len(groups)} categories. Need exactly 2 for T-test")
        else:
            st.info("Need categorical and numeric columns")
    
    elif test_type == "ANOVA (3+ groups)":
        st.markdown("**One-Way ANOVA** - Compare means across multiple groups")
        if summary.categorical_cols and summary.numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                group_col = st.selectbox("Grouping column", summary.categorical_cols, key="anova_group")
            with col2:
                value_col = st.selectbox("Value column (numeric)", summary.numeric_cols, key="anova_val")
            
            if st.button("Run ANOVA", key="btn_anova"):
                groups = df.groupby(group_col)[value_col].apply(list).values
                groups = [g for g in groups if len(g) > 0]
                
                if len(groups) >= 2:
                    f_stat, p_value = stats.f_oneway(*groups)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("F-Statistic", f"{f_stat:.4f}")
                    with col2:
                        st.metric("P-Value", f"{p_value:.4f}")
                    with col3:
                        if p_value < 0.05:
                            st.success("✅ Significant")
                        else:
                            st.info("⚪ Not Significant")
                    
                    st.info(f"**Interpretation:** {'At least one group mean differs significantly' if p_value < 0.05 else 'No significant differences between group means'} (α=0.05)")
                else:
                    st.error("Need at least 2 groups")
        else:
            st.info("Need categorical and numeric columns")
    
    elif test_type == "Chi-Square Test":
        st.markdown("**Chi-Square Test of Independence** - Test relationship between categorical variables")
        if len(summary.categorical_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                cat1 = st.selectbox("First categorical column", summary.categorical_cols, key="chi_1")
            with col2:
                cat2 = st.selectbox("Second categorical column", [c for c in summary.categorical_cols if c != cat1], key="chi_2")
            
            if st.button("Run Chi-Square Test", key="btn_chi"):
                contingency = pd.crosstab(df[cat1], df[cat2])
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Chi-Square", f"{chi2:.2f}")
                with col2:
                    st.metric("P-Value", f"{p_value:.4f}")
                with col3:
                    st.metric("DoF", f"{dof}")
                with col4:
                    if p_value < 0.05:
                        st.success("✅ Associated")
                    else:
                        st.info("⚪ Independent")
                
                st.info(f"**Interpretation:** Variables are {'associated/dependent' if p_value < 0.05 else 'independent'} (α=0.05)")
                
                # Show contingency table
                with st.expander("View Contingency Table"):
                    st.dataframe(contingency, width="stretch")
        else:
            st.info("Need at least 2 categorical columns")
    
    elif test_type == "Correlation Test":
        st.markdown("**Pearson Correlation Significance Test** - Test if correlation is significant")
        if len(summary.numeric_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                num1 = st.selectbox("First numeric column", summary.numeric_cols, key="corr_1")
            with col2:
                num2 = st.selectbox("Second numeric column", [c for c in summary.numeric_cols if c != num1], key="corr_2")
            
            if st.button("Run Correlation Test", key="btn_corr"):
                data1 = df[num1].dropna()
                data2 = df[num2].dropna()
                common_idx = data1.index.intersection(data2.index)
                
                if len(common_idx) > 2:
                    r, p_value = stats.pearsonr(data1[common_idx], data2[common_idx])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Correlation (r)", f"{r:.4f}")
                    with col2:
                        st.metric("P-Value", f"{p_value:.4f}")
                    with col3:
                        if p_value < 0.05:
                            st.success("✅ Significant")
                        else:
                            st.info("⚪ Not Significant")
                    
                    strength = "Very Strong" if abs(r) > 0.8 else "Strong" if abs(r) > 0.6 else "Moderate" if abs(r) > 0.4 else "Weak" if abs(r) > 0.2 else "Very Weak"
                    direction = "positive" if r > 0 else "negative"
                    st.info(f"**Interpretation:** {strength} {direction} correlation {'(statistically significant)' if p_value < 0.05 else '(not statistically significant)'} at α=0.05")
                else:
                    st.error("Need at least 3 common data points")
        else:
            st.info("Need at least 2 numeric columns")

with tab6:
    st.subheader("🎯 Comprehensive Data Profiling")
    
    # Data types and memory usage
    st.markdown("### 📋 Column Details")
    
    profile_data = []
    for col in df.columns:
        col_data = df[col]
        profile_data.append({
            "Column": col,
            "Type": str(col_data.dtype),
            "Non-Null": col_data.notna().sum(),
            "Null": col_data.isna().sum(),
            "Null %": f"{(col_data.isna().sum() / len(col_data) * 100):.1f}%",
            "Unique": col_data.nunique(),
            "Unique %": f"{(col_data.nunique() / len(col_data) * 100):.1f}%" if len(col_data) > 0 else "0%",
        })
    
    profile_df = pd.DataFrame(profile_data)
    st.dataframe(profile_df, width="stretch", hide_index=True)
    
    st.divider()
    
    # Cardinality analysis
    st.markdown("### 🎲 Cardinality Analysis")
    st.caption("Understanding uniqueness of values")
    
    cardinality_data = []
    for col in df.columns:
        n_unique = df[col].nunique()
        n_total = len(df[col])
        ratio = n_unique / n_total if n_total > 0 else 0
        
        if ratio == 1.0:
            category = "🔑 Unique (ID-like)"
        elif ratio > 0.95:
            category = "🌟 High Cardinality"
        elif ratio > 0.5:
            category = "📊 Medium Cardinality"
        elif ratio > 0.05:
            category = "📁 Low Cardinality"
        else:
            category = "🏷️ Very Low Cardinality"
        
        cardinality_data.append({
            "Column": col,
            "Unique Values": n_unique,
            "Total Values": n_total,
            "Cardinality Ratio": f"{ratio:.2%}",
            "Category": category
        })
    
    cardinality_df = pd.DataFrame(cardinality_data).sort_values("Unique Values", ascending=False)
    st.dataframe(cardinality_df, width="stretch", hide_index=True)
    
    st.divider()
    
    # Data quality score
    st.markdown("### ✅ Data Quality Score")
    
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()
    completeness = (1 - missing_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Calculate other quality metrics
    duplicate_rows = df.duplicated().sum()
    duplicate_pct = (duplicate_rows / len(df) * 100) if len(df) > 0 else 0
    
    # Columns with high missing rate
    high_missing_cols = (df.isna().sum() / len(df) > 0.2).sum()
    
    # Overall quality score (simple heuristic)
    quality_score = (
        completeness * 0.4 +  # 40% weight on completeness
        (100 - duplicate_pct) * 0.3 +  # 30% weight on uniqueness
        ((df.shape[1] - high_missing_cols) / df.shape[1] * 100) * 0.3  # 30% weight on column quality
    )
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Overall Quality", f"{quality_score:.1f}/100", help="Composite score based on completeness, uniqueness, and column quality")
    with col2:
        st.metric("Completeness", f"{completeness:.1f}%", help="Percentage of non-missing values")
    with col3:
        st.metric("Duplicate Rows", f"{duplicate_rows:,}", delta=f"-{duplicate_pct:.1f}%" if duplicate_rows > 0 else "0%")
    with col4:
        st.metric("Problematic Columns", f"{high_missing_cols}", help="Columns with >20% missing data")
    
    # Quality visualization
    quality_metrics = {
        "Completeness": completeness,
        "Uniqueness": 100 - duplicate_pct,
        "Column Health": (df.shape[1] - high_missing_cols) / df.shape[1] * 100 if df.shape[1] > 0 else 0
    }
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=list(quality_metrics.keys()),
        y=list(quality_metrics.values()),
        marker_color=['#2ecc71' if v >= 80 else '#f39c12' if v >= 60 else '#e74c3c' for v in quality_metrics.values()],
        text=[f"{v:.1f}%" for v in quality_metrics.values()],
        textposition='auto',
    ))
    fig.update_layout(
        title="Data Quality Breakdown",
        yaxis_title="Score (%)",
        yaxis_range=[0, 100],
        height=350,
        showlegend=False
    )
    st.plotly_chart(fig, width="stretch")
    
    st.divider()
    
    # Column-by-column deep dive
    st.markdown("### 🔍 Column Deep Dive")
    selected_col = st.selectbox("Select column for detailed analysis", df.columns)
    
    col_data = df[selected_col]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Data Type", str(col_data.dtype))
    with col2:
        st.metric("Missing", f"{col_data.isna().sum():,}")
    with col3:
        st.metric("Unique", f"{col_data.nunique():,}")
    with col4:
        st.metric("Memory", f"{col_data.memory_usage(deep=True) / 1024:.1f} KB")
    
    if pd.api.types.is_numeric_dtype(col_data):
        st.markdown("**Numeric Statistics**")
        stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
        with stats_col1:
            st.metric("Min", f"{col_data.min():.2f}")
        with stats_col2:
            st.metric("Q1", f"{col_data.quantile(0.25):.2f}")
        with stats_col3:
            st.metric("Median", f"{col_data.median():.2f}")
        with stats_col4:
            st.metric("Q3", f"{col_data.quantile(0.75):.2f}")
        with stats_col5:
            st.metric("Max", f"{col_data.max():.2f}")
        
        # Histogram with statistics
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=col_data.dropna(), nbinsx=50, name="Distribution"))
        fig.update_layout(title=f"Distribution of {selected_col}", height=300)
        st.plotly_chart(fig, width="stretch")
    else:
        st.markdown("**Categorical Statistics**")
        top_values = col_data.value_counts().head(10)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(x=top_values.index.astype(str), y=top_values.values))
        fig.update_layout(title=f"Top 10 Values in {selected_col}", height=300)
        st.plotly_chart(fig, width="stretch")
        
        with st.expander("View All Value Counts"):
            st.dataframe(col_data.value_counts().reset_index(), width="stretch", hide_index=True)

with tab7:
    st.subheader("⏰ Time Series Analysis")
    st.caption("Analyze temporal patterns and trends in your data")
    
    # Detect potential time columns
    time_candidates = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            time_candidates.append(col)
        elif df[col].dtype == 'object':
            # Try parsing as datetime
            try:
                pd.to_datetime(df[col].head(100), errors='coerce')
                if df[col].head(100).apply(lambda x: pd.to_datetime(x, errors='coerce')).notna().sum() > 50:
                    time_candidates.append(col)
            except:
                pass
    
    if not time_candidates:
        st.info("No datetime columns detected. Select a column to attempt conversion.")
        time_col = st.selectbox("Select column (will attempt datetime conversion)", df.columns)
        if st.button("Convert to datetime"):
            try:
                df_temp = df.copy()
                df_temp[time_col] = pd.to_datetime(df_temp[time_col], errors='coerce')
                time_candidates = [time_col]
                st.success(f"✅ Converted {time_col} to datetime")
            except Exception as e:
                st.error(f"Conversion failed: {e}")
    
    if time_candidates:
        time_col = st.selectbox("Select time column", time_candidates)
        
        # Parse time column
        df_ts = df.copy()
        if not pd.api.types.is_datetime64_any_dtype(df_ts[time_col]):
            df_ts[time_col] = pd.to_datetime(df_ts[time_col], errors='coerce')
        
        df_ts = df_ts[df_ts[time_col].notna()].sort_values(time_col)
        
        if len(df_ts) > 0:
            st.markdown("### 📅 Time Range")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Start Date", df_ts[time_col].min().strftime('%Y-%m-%d'))
            with col2:
                st.metric("End Date", df_ts[time_col].max().strftime('%Y-%m-%d'))
            with col3:
                time_span = (df_ts[time_col].max() - df_ts[time_col].min()).days
                st.metric("Time Span", f"{time_span} days")
            with col4:
                st.metric("Data Points", f"{len(df_ts):,}")
            
            st.divider()
            
            # Temporal patterns
            st.markdown("### 📊 Temporal Patterns")
            
            df_ts['year'] = df_ts[time_col].dt.year
            df_ts['month'] = df_ts[time_col].dt.month
            df_ts['day_of_week'] = df_ts[time_col].dt.dayofweek
            df_ts['hour'] = df_ts[time_col].dt.hour
            df_ts['day'] = df_ts[time_col].dt.day
            
            pattern_type = st.radio(
                "Pattern type",
                ["Records per Year", "Records per Month", "Records per Day of Week", "Records per Hour"],
                horizontal=True
            )
            
            if pattern_type == "Records per Year":
                counts = df_ts['year'].value_counts().sort_index()
                fig = px.bar(x=counts.index, y=counts.values, labels={'x': 'Year', 'y': 'Count'}, title="Data Distribution by Year")
            elif pattern_type == "Records per Month":
                counts = df_ts['month'].value_counts().sort_index()
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                fig = px.bar(x=[month_names[i-1] for i in counts.index], y=counts.values, labels={'x': 'Month', 'y': 'Count'}, title="Seasonal Pattern (by Month)")
            elif pattern_type == "Records per Day of Week":
                counts = df_ts['day_of_week'].value_counts().sort_index()
                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                fig = px.bar(x=[day_names[i] for i in counts.index], y=counts.values, labels={'x': 'Day', 'y': 'Count'}, title="Weekly Pattern")
            else:
                counts = df_ts['hour'].value_counts().sort_index()
                fig = px.bar(x=counts.index, y=counts.values, labels={'x': 'Hour', 'y': 'Count'}, title="Daily Pattern (by Hour)")
            
            fig.update_layout(height=350)
            st.plotly_chart(fig, width="stretch")
            
            st.divider()
            
            # Time series decomposition (if numeric column selected)
            st.markdown("### 📈 Time Series Trend")
            if summary.numeric_cols:
                value_col = st.selectbox("Select value column", summary.numeric_cols, key="ts_value")
                
                df_ts_agg = df_ts.set_index(time_col)[value_col].resample('D').mean().dropna()
                
                if len(df_ts_agg) > 0:
                    # Plot time series
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df_ts_agg.index, y=df_ts_agg.values, mode='lines', name='Daily Average'))
                    
                    # Add rolling mean
                    if len(df_ts_agg) >= 7:
                        rolling_mean = df_ts_agg.rolling(window=7, center=True).mean()
                        fig.add_trace(go.Scatter(x=rolling_mean.index, y=rolling_mean.values, mode='lines', name='7-day Moving Average', line=dict(dash='dash')))
                    
                    fig.update_layout(title=f"{value_col} Over Time", xaxis_title="Date", yaxis_title=value_col, height=400)
                    st.plotly_chart(fig, width="stretch")
                    
                    # Trend statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        trend = "Increasing" if df_ts_agg.iloc[-1] > df_ts_agg.iloc[0] else "Decreasing"
                        st.metric("Overall Trend", trend)
                    with col2:
                        volatility = df_ts_agg.std() / df_ts_agg.mean() * 100 if df_ts_agg.mean() != 0 else 0
                        st.metric("Volatility (CV%)", f"{volatility:.1f}%")
                    with col3:
                        change_pct = ((df_ts_agg.iloc[-1] - df_ts_agg.iloc[0]) / df_ts_agg.iloc[0] * 100) if df_ts_agg.iloc[0] != 0 else 0
                        st.metric("Total Change", f"{change_pct:+.1f}%")
            else:
                st.info("No numeric columns available for trend analysis")
    else:
        st.info("No time columns detected in the dataset")

with tab8:
    st.subheader("🔬 Advanced Multivariate Analysis")
    
    # Feature importance using variance
    st.markdown("### 🎯 Feature Variance Analysis")
    st.caption("Higher variance features contain more information")
    
    if summary.numeric_cols:
        variance_data = []
        for col in summary.numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 1:
                variance = col_data.var()
                std = col_data.std()
                mean = col_data.mean()
                cv = (std / mean * 100) if mean != 0 else 0
                variance_data.append({
                    "Feature": col,
                    "Variance": variance,
                    "Std Dev": std,
                    "Mean": mean,
                    "CV%": cv
                })
        
        if variance_data:
            var_df = pd.DataFrame(variance_data).sort_values("Variance", ascending=False)
            
            fig = px.bar(var_df.head(20), x="Feature", y="Variance", title="Top 20 Features by Variance", color="Variance", color_continuous_scale="Viridis")
            fig.update_layout(xaxis_tickangle=-45, height=400)
            st.plotly_chart(fig, width="stretch")
            
            st.dataframe(var_df, width="stretch", hide_index=True)
    else:
        st.info("No numeric columns available")
    
    st.divider()
    
    # Principal Component Analysis (PCA) preview
    st.markdown("### 🔄 Dimensionality Insights")
    st.caption("Understanding feature relationships through variance")
    
    if len(summary.numeric_cols) >= 2:
        selected_features = st.multiselect(
            "Select features for analysis (2-20)",
            summary.numeric_cols,
            default=summary.numeric_cols[:min(5, len(summary.numeric_cols))],
            max_selections=20
        )
        
        if len(selected_features) >= 2 and st.button("Analyze Feature Space", key="pca_btn"):
            # Prepare data
            df_pca = df[selected_features].dropna()
            
            if len(df_pca) > 0:
                # Correlation matrix for selected features
                corr_selected = df_pca.corr()
                
                # Find highly correlated pairs
                high_corr_pairs = []
                for i in range(len(corr_selected.columns)):
                    for j in range(i+1, len(corr_selected.columns)):
                        corr_val = corr_selected.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            high_corr_pairs.append({
                                "Feature 1": corr_selected.columns[i],
                                "Feature 2": corr_selected.columns[j],
                                "Correlation": corr_val
                            })
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Features Selected", len(selected_features))
                    st.metric("High Correlations (>0.7)", len(high_corr_pairs))
                    
                    if high_corr_pairs:
                        st.markdown("**Redundant Features:**")
                        for pair in high_corr_pairs[:5]:
                            st.caption(f"• {pair['Feature 1']} ↔ {pair['Feature 2']}: {pair['Correlation']:.2f}")
                
                with col2:
                    # Variance explained estimate
                    normalized = (df_pca - df_pca.mean()) / df_pca.std()
                    cov_matrix = normalized.cov()
                    eigenvalues = np.linalg.eigvalsh(cov_matrix)
                    eigenvalues = sorted(eigenvalues, reverse=True)
                    
                    var_explained = eigenvalues / np.sum(eigenvalues) * 100
                    cumsum_var = np.cumsum(var_explained)
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(x=list(range(1, len(var_explained)+1)), y=var_explained, name="Individual"))
                    fig.add_trace(go.Scatter(x=list(range(1, len(cumsum_var)+1)), y=cumsum_var, mode='lines+markers', name="Cumulative", yaxis='y2'))
                    fig.update_layout(
                        title="Variance Explained by Components",
                        xaxis_title="Component",
                        yaxis_title="Variance Explained (%)",
                        yaxis2=dict(title="Cumulative (%)", overlaying='y', side='right'),
                        height=350
                    )
                    st.plotly_chart(fig, width="stretch")
                    
                    # Recommendation
                    n_components_90 = np.argmax(cumsum_var >= 90) + 1
                    st.info(f"💡 **Insight:** {n_components_90} components capture 90% of variance. Consider dimensionality reduction if working with {len(selected_features)} features.")
            else:
                st.warning("Not enough data after removing missing values")
    else:
        st.info("Need at least 2 numeric columns")
    
    st.divider()
    
    # Interaction detection
    st.markdown("### 🔗 Feature Interaction Detection")
    st.caption("Identify potential feature interactions for engineering")
    
    if len(summary.numeric_cols) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            feat1 = st.selectbox("First feature", summary.numeric_cols, key="int_feat1")
        with col2:
            feat2 = st.selectbox("Second feature", [c for c in summary.numeric_cols if c != feat1], key="int_feat2")
        
        if summary.numeric_cols and st.button("Check Interaction", key="int_btn"):
            # Create interaction term
            df_int = df[[feat1, feat2]].dropna()
            
            if len(df_int) > 0:
                # Multiply features
                df_int['interaction'] = df_int[feat1] * df_int[feat2]
                
                # 3D scatter plot
                fig = go.Figure(data=[go.Scatter3d(
                    x=df_int[feat1],
                    y=df_int[feat2],
                    z=df_int['interaction'],
                    mode='markers',
                    marker=dict(
                        size=3,
                        color=df_int['interaction'],
                        colorscale='Viridis',
                        showscale=True
                    )
                )])
                fig.update_layout(
                    title=f"Interaction: {feat1} × {feat2}",
                    scene=dict(
                        xaxis_title=feat1,
                        yaxis_title=feat2,
                        zaxis_title=f"{feat1} × {feat2}"
                    ),
                    height=500
                )
                st.plotly_chart(fig, width="stretch")
                
                # Statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Interaction Mean", f"{df_int['interaction'].mean():.2f}")
                with col2:
                    st.metric("Interaction Std", f"{df_int['interaction'].std():.2f}")
                with col3:
                    st.metric("Interaction Range", f"{df_int['interaction'].max() - df_int['interaction'].min():.2f}")
                
                st.success(f"💡 **Tip:** Consider creating engineered feature: `{feat1}_x_{feat2}` = {feat1} × {feat2}")
    else:
        st.info("Need at least 2 numeric columns")
    
    st.divider()
    
    # Data transformation suggestions
    st.markdown("### 🔧 Transformation Recommendations")
    st.caption("Suggested transformations to improve data distribution")
    
    if summary.numeric_cols:
        recommendations = []
        
        for col in summary.numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 10:
                skewness = col_data.skew()
                has_zeros = (col_data == 0).any()
                has_negatives = (col_data < 0).any()
                is_positive = (col_data > 0).all()
                
                suggestions = []
                
                # Skewness-based recommendations
                if abs(skewness) > 2:
                    if is_positive:
                        suggestions.append("Log transform (reduce right skew)")
                    suggestions.append("Box-Cox transform")
                    suggestions.append("Square root transform")
                
                # Range-based recommendations
                if col_data.max() - col_data.min() > 1000:
                    suggestions.append("Standardization (z-score)")
                    suggestions.append("Min-Max scaling")
                
                # Distribution-based
                if abs(skewness) < 0.5:
                    suggestions.append("Already well-distributed ✅")
                
                if suggestions:
                    recommendations.append({
                        "Feature": col,
                        "Skewness": f"{skewness:.2f}",
                        "Range": f"{col_data.max() - col_data.min():.1f}",
                        "Suggestions": " | ".join(suggestions[:3])
                    })
        
        if recommendations:
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, width="stretch", hide_index=True)
            
            st.info("💡 **Note:** Apply transformations in the Data Cleaning page or during model training")
        else:
            st.success("✅ Data distributions look good!")
    else:
        st.info("No numeric columns to analyze")

# Page navigation
standard_section_header("Learning Guide & Best Practices", "🎓")
concept_explainer(
    title="What is EDA?",
    explanation=(
        "Exploratory Data Analysis helps you understand data quality, distributions, relationships, and anomalies before modeling. "
        "Use EDA to decide cleaning steps, feature engineering, and whether your dataset fits the task."
    ),
    real_world_example=(
        "Customer churn: EDA shows missing contract dates, skewed tenure, and correlation between support calls and churn. "
        "You decide to impute dates, cap outliers, and include interactions before training."
    ),
)
beginner_tip("Tip: Focus on patterns, not perfection — EDA guides decisions, it doesn't prove causality.")
common_mistakes_panel({
    "Confusing correlation with causation": "Use domain knowledge and experiments to confirm causes.",
    "Ignoring missingness patterns": "Missing not at random can bias results — inspect by group.",
    "Overfitting conclusions to small samples": "Validate insights on holdout data or with statistical tests.",
    "Skipping outlier review": "Extreme values can distort models — consider clipping or robust methods.",
    "Not checking data types": "Incorrect types break aggregations and charts — fix early.",
})

page_navigation("2")
