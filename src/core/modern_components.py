"""
Modern UI Components Library
Smart, interactive components for professional data science UI
"""

from __future__ import annotations

import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def smart_data_preview(
    df: pd.DataFrame,
    title: str = "Data Preview",
    show_stats: bool = True,
    max_rows: int = 10
) -> None:
    """
    Smart data preview with automatic profiling and insights.
    
    Args:
        df: DataFrame to preview
        title: Preview section title
        show_stats: Whether to show automatic statistics
        max_rows: Maximum rows to display
    """
    with st.expander(f"📊 {title}", expanded=False):
        # Quick stats row
        if show_stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Rows",
                    f"{len(df):,}",
                    help="Total number of records in dataset"
                )
            
            with col2:
                st.metric(
                    "Columns",
                    df.shape[1],
                    help="Number of features/variables"
                )
            
            with col3:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100)
                st.metric(
                    "Missing",
                    f"{missing_pct:.1f}%",
                    help="Percentage of missing values",
                    delta=f"{-missing_pct:.1f}%" if missing_pct < 5 else None,
                    delta_color="inverse" if missing_pct > 5 else "normal"
                )
            
            with col4:
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.metric(
                    "Memory",
                    f"{memory_mb:.1f} MB",
                    help="Dataset memory usage"
                )
        
        # Data sample
        st.dataframe(
            df.head(max_rows),
            use_container_width=True,
            height=min(400, (max_rows + 1) * 35)
        )
        
        # Data types summary
        with st.expander("🔍 Column Details", expanded=False):
            dtypes_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.astype(str),
                'Non-Null': df.count().values,
                'Null Count': df.isnull().sum().values,
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(dtypes_df, use_container_width=True)


def auto_data_profile(df: pd.DataFrame) -> None:
    """
    Automatic data profiling with insights and recommendations.
    
    Args:
        df: DataFrame to profile
    """
    st.markdown("### 🔍 Automatic Data Profile")
    
    # Generate insights
    insights = []
    warnings = []
    recommendations = []
    
    # Check for missing data
    missing_cols = df.columns[df.isnull().any()].tolist()
    if missing_cols:
        pct = (df[missing_cols].isnull().sum().sum() / (len(df) * len(missing_cols)) * 100)
        if pct > 20:
            warnings.append(f"⚠️ High missing data ({pct:.1f}%) in {len(missing_cols)} columns")
            recommendations.append("Consider imputation or dropping columns with >50% missing values")
        else:
            insights.append(f"ℹ️ {len(missing_cols)} columns have missing values ({pct:.1f}%)")
    
    # Check for duplicates
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        warnings.append(f"{dup_count:,} duplicate rows detected ({dup_count/len(df)*100:.1f}%)")
        recommendations.append("Remove duplicates before modeling")
    
    # Check numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        insights.append(f"📊 {len(numeric_cols)} numeric features detected")
        
        # Check for outliers
        outlier_cols = []
        for col in numeric_cols:
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
            if outliers > len(df) * 0.05:  # More than 5% outliers
                outlier_cols.append(col)
        
        if outlier_cols:
            warnings.append(f"Potential outliers in {len(outlier_cols)} numeric columns")
            recommendations.append("Consider outlier treatment (capping, removal, or transformation)")
    
    # Check categorical columns
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        insights.append(f"🏷️ {len(cat_cols)} categorical features detected")
        
        high_card = [col for col in cat_cols if df[col].nunique() > 50]
        if high_card:
            warnings.append(f"High cardinality in {len(high_card)} categorical columns")
            recommendations.append("Consider grouping or encoding for high-cardinality features")
    
    # Check for imbalanced target (if last column looks like a target)
    if len(df.columns) > 0:
        last_col = df.columns[-1]
        if df[last_col].nunique() < 20:  # Likely categorical
            class_dist = df[last_col].value_counts(normalize=True)
            if class_dist.max() > 0.8 or class_dist.min() < 0.05:
                warnings.append(f"Potential class imbalance in '{last_col}'")
                recommendations.append("Use stratified sampling or class weights for imbalanced data")
    
    # Display insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### ✅ Insights")
        for insight in insights:
            st.success(insight, icon="ℹ️")
    
    with col2:
        st.markdown("#### ⚠️ Warnings")
        for warning in warnings:
            st.warning(warning, icon="⚠️")
    
    with col3:
        st.markdown("#### 💡 Recommendations")
        for rec in recommendations:
            st.info(rec, icon="💡")


def smart_metric_card(
    label: str,
    value: Any,
    delta: Optional[str] = None,
    help_text: Optional[str] = None,
    interpretation: Optional[str] = None,
    good_threshold: Optional[float] = None,
    compare_value: Optional[float] = None
) -> None:
    """
    Enhanced metric card with interpretation and context.
    
    Args:
        label: Metric name
        value: Metric value
        delta: Change indicator
        help_text: Tooltip help text
        interpretation: Plain English interpretation
        good_threshold: Threshold for good performance
        compare_value: Value to compare against
    """
    # Determine if metric is good/bad
    delta_color = "normal"
    if good_threshold is not None and isinstance(value, (int, float)):
        if value >= good_threshold:
            delta_color = "normal"
        else:
            delta_color = "inverse"
    
    st.metric(
        label=label,
        value=value,
        delta=delta,
        delta_color=delta_color,
        help=help_text
    )
    
    if interpretation:
        st.caption(f"💬 {interpretation}")


def comparison_table(
    data: List[Dict[str, Any]],
    highlight_best: Optional[str] = None,
    ascending: bool = False
) -> None:
    """
    Create a comparison table with best value highlighting.
    
    Args:
        data: List of dictionaries with comparison data
        highlight_best: Column to use for highlighting best performer
        ascending: Whether lower is better for highlight column
    """
    if not data:
        st.warning("No data to compare")
        return
    
    df = pd.DataFrame(data)
    
    if highlight_best and highlight_best in df.columns:
        # Find best performer
        if ascending:
            best_idx = df[highlight_best].idxmin()
        else:
            best_idx = df[highlight_best].idxmax()
        
        # Style the dataframe
        def highlight_row(row):
            if row.name == best_idx:
                return ['background-color: #d1fae5; font-weight: 600'] * len(row)
            return [''] * len(row)
        
        styled_df = df.style.apply(highlight_row, axis=1)
        st.dataframe(styled_df, use_container_width=True)
        
        st.caption(f"🏆 Best performer highlighted (by {highlight_best})")
    else:
        st.dataframe(df, use_container_width=True)


def guided_workflow(
    steps: List[Dict[str, str]],
    current_step: int = 0
) -> None:
    """
    Display a guided workflow with progress indicator.
    
    Args:
        steps: List of workflow steps with 'title' and 'description'
        current_step: Current active step index
    """
    st.markdown("### 📋 Workflow Steps")
    
    # Progress bar
    progress = (current_step + 1) / len(steps)
    st.progress(progress)
    st.caption(f"Step {current_step + 1} of {len(steps)}")
    
    # Steps
    for idx, step in enumerate(steps):
        status_icon = "✅" if idx < current_step else ("🔄" if idx == current_step else "⭕")
        opacity = "1.0" if idx <= current_step else "0.5"
        
        st.markdown(
            f"""
            <div style="opacity: {opacity}; padding: 1rem; margin: 0.5rem 0; 
                        border-left: 3px solid {'#667eea' if idx == current_step else '#cbd5e1'};">
                <div style="font-weight: 600; color: #1e293b;">
                    {status_icon} Step {idx + 1}: {step['title']}
                </div>
                <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.25rem;">
                    {step['description']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def interactive_tooltip(
    text: str,
    tooltip: str,
    icon: str = "ℹ️"
) -> None:
    """
    Display text with an interactive tooltip.
    
    Args:
        text: Main text to display
        tooltip: Tooltip content
        icon: Icon for tooltip indicator
    """
    st.markdown(
        f"""
        <div style="display: inline-flex; align-items: center; gap: 0.5rem;">
            <span style="font-weight: 500;">{text}</span>
            <span style="cursor: help;" title="{tooltip}">{icon}</span>
        </div>
        """,
        unsafe_allow_html=True
    )


def enhanced_chart(
    fig: go.Figure,
    title: Optional[str] = None,
    description: Optional[str] = None,
    insights: Optional[List[str]] = None
) -> None:
    """
    Display Plotly chart with enhanced styling and optional insights.
    
    Args:
        fig: Plotly figure
        title: Chart title
        description: Chart description
        insights: List of insights to display below chart
    """
    if title:
        st.markdown(f"### {title}")
    
    if description:
        st.caption(description)
    
    # Apply modern theme
    fig.update_layout(
        font=dict(family='Inter, sans-serif', size=12, color='#1e293b'),
        paper_bgcolor='white',
        plot_bgcolor='#f8fafc',
        hovermode='closest',
        margin=dict(l=60, r=40, t=60, b=60)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': ['select2d', 'lasso2d']
    })
    
    if insights:
        with st.expander("💡 Chart Insights", expanded=False):
            for insight in insights:
                st.info(insight, icon="💡")


def collapsible_advanced_options(
    options_dict: Dict[str, Any],
    title: str = "Advanced Options"
) -> Dict[str, Any]:
    """
    Create collapsible advanced options section.
    
    Args:
        options_dict: Dictionary of option names and default values
        title: Section title
    
    Returns:
        Dictionary of selected values
    """
    with st.expander(f"⚙️ {title}", expanded=False):
        st.caption("Configure advanced parameters (optional)")
        
        results = {}
        for key, default_value in options_dict.items():
            if isinstance(default_value, bool):
                results[key] = st.checkbox(key.replace('_', ' ').title(), value=default_value)
            elif isinstance(default_value, int):
                results[key] = st.number_input(key.replace('_', ' ').title(), value=default_value)
            elif isinstance(default_value, float):
                results[key] = st.slider(key.replace('_', ' ').title(), 
                                       min_value=0.0, max_value=1.0, value=default_value)
            elif isinstance(default_value, list):
                results[key] = st.multiselect(key.replace('_', ' ').title(), 
                                             options=default_value, default=default_value[0:1])
            else:
                results[key] = st.text_input(key.replace('_', ' ').title(), value=str(default_value))
        
        return results


def success_message_with_next_steps(
    message: str,
    next_steps: List[str],
    action_button: Optional[Dict[str, Any]] = None
) -> None:
    """
    Display success message with recommended next steps.
    
    Args:
        message: Success message
        next_steps: List of recommended actions
        action_button: Optional dict with 'label' and 'page' for navigation
    """
    st.success(f"✅ {message}", icon="🎉")
    
    st.markdown("#### 🎯 Next Steps")
    for idx, step in enumerate(next_steps, 1):
        st.markdown(f"{idx}. {step}")
    
    if action_button:
        if st.button(f"➡️ {action_button['label']}", type="primary", use_container_width=True):
            st.switch_page(f"pages/{action_button['page']}.py")


def data_quality_badge(
    quality_score: float,
    label: str = "Data Quality"
) -> None:
    """
    Display a data quality badge with color coding.
    
    Args:
        quality_score: Quality score (0-100)
        label: Badge label
    """
    if quality_score >= 80:
        color = "#10b981"
        grade = "Excellent"
        icon = "🌟"
    elif quality_score >= 60:
        color = "#f59e0b"
        grade = "Good"
        icon = "👍"
    elif quality_score >= 40:
        color = "#f97316"
        grade = "Fair"
        icon = "⚠️"
    else:
        color = "#ef4444"
        grade = "Poor"
        icon = "❌"
    
    st.markdown(
        f"""
        <div style="display: inline-block; background: {color}15; 
                    border-left: 4px solid {color}; padding: 0.75rem 1.25rem; 
                    border-radius: 8px; margin: 1rem 0;">
            <div style="font-size: 0.75rem; color: {color}; font-weight: 600; 
                        text-transform: uppercase; letter-spacing: 0.05em;">
                {label}
            </div>
            <div style="font-size: 1.5rem; font-weight: 700; color: {color}; margin-top: 0.25rem;">
                {icon} {grade} ({quality_score:.0f}/100)
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
