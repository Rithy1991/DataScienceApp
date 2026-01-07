"""
Streamlit Components for Universal Forecast Results
Ready-to-use UI components for displaying standardized forecast outputs
"""

from __future__ import annotations

from typing import List, Optional

import streamlit as st
import pandas as pd

from src.core.forecast_results import (
    ForecastResult,
    generate_ai_summary,
    create_forecast_visualization,
    create_forecast_table,
    create_model_comparison_chart,
)


def render_forecast_results(
    result: ForecastResult,
    show_ai_summary: bool = True,
    show_metrics: bool = True,
    show_table: bool = True,
    show_aggregations: bool = True,
) -> None:
    """
    Render complete standardized forecast results in Streamlit.
    
    Args:
        result: ForecastResult object
        show_ai_summary: Display AI-generated summary
        show_metrics: Show summary statistics
        show_table: Display forecast table
        show_aggregations: Show monthly/quarterly/yearly views
    """
    
    st.markdown("---")
    st.markdown(f"## 🎯 Forecast Results: {result.model_name}")
    
    # AI Summary (if available or requested)
    if show_ai_summary:
        if result.ai_summary is None:
            result.ai_summary = generate_ai_summary(result)
        
        st.markdown("### 🤖 AI Forecast Summary")
        st.info(result.ai_summary, icon="💡")
    
    # Main visualization
    st.markdown("### 📊 Interactive Forecast Chart")
    fig = create_forecast_visualization(result)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(
        "📖 **How to read:** Blue line shows historical data. Orange dashed line is the forecast. "
        "Shaded area shows confidence bounds—wider means more uncertainty."
    )
    
    # Summary statistics
    if show_metrics:
        st.markdown("### 📈 Forecast Statistics")
        stats = result.get_summary_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Forecast Horizon", f"{stats['forecast_horizon']} periods")
        with col2:
            st.metric("Average Forecast", f"{stats['mean_forecast']:.2f}")
        with col3:
            st.metric("Trend", stats['trend'])
        with col4:
            if 'uncertainty_pct' in stats:
                st.metric("Avg Uncertainty", f"±{stats['uncertainty_pct']:.1f}%")
            else:
                st.metric("Range", f"{stats['min_forecast']:.1f} - {stats['max_forecast']:.1f}")
        
        st.divider()
    
    # Time aggregations
    if show_aggregations and len(result.forecast_dates) > 7:
        st.markdown("### 📅 Forecast by Time Period")
        
        agg_view = st.selectbox(
            "View by",
            options=["Monthly", "Quarterly", "Yearly"],
            help="Aggregate forecast by time period"
        )
        
        aggregations = result.get_time_aggregations()
        
        if agg_view == "Monthly":
            agg_df = aggregations["monthly"]
        elif agg_view == "Quarterly":
            agg_df = aggregations["quarterly"]
        else:
            agg_df = aggregations["yearly"]
        
        # Display aggregated data
        display_df = agg_df.copy()
        display_df = display_df.drop(columns=["Date"], errors="ignore")
        
        # Round numeric columns
        numeric_cols = display_df.select_dtypes(include=["number"]).columns
        display_df[numeric_cols] = display_df[numeric_cols].round(2)
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        st.divider()
    
    # Detailed forecast table
    if show_table:
        st.markdown("### 📋 Detailed Forecast Table")
        
        with st.expander("View full forecast data", expanded=False):
            forecast_df = create_forecast_table(result, max_rows=100)
            st.dataframe(forecast_df, use_container_width=True, hide_index=True)
            
            # Download button
            csv = result.to_dataframe().to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast CSV",
                data=csv,
                file_name=f"forecast_{result.model_type}_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
            )
    
    # Model performance metrics
    if show_metrics and (result.train_metrics or result.test_metrics):
        st.markdown("### ✅ Model Performance")
        
        perf_col1, perf_col2 = st.columns(2)
        
        with perf_col1:
            if result.train_metrics:
                st.markdown("**Training Metrics**")
                for metric, value in result.train_metrics.items():
                    st.metric(metric.upper(), f"{value:.4f}")
        
        with perf_col2:
            if result.test_metrics:
                st.markdown("**Validation Metrics**")
                for metric, value in result.test_metrics.items():
                    st.metric(metric.upper(), f"{value:.4f}")


def render_model_comparison(
    results: List[ForecastResult],
    title: str = "Model Comparison Dashboard",
) -> None:
    """
    Render comparison dashboard for multiple forecast models.
    
    Args:
        results: List of ForecastResult objects to compare
        title: Dashboard title
    """
    
    st.markdown(f"## 🔬 {title}")
    
    if not results:
        st.warning("No forecast results to compare.")
        return
    
    if len(results) == 1:
        st.info("Add more models to enable comparison.")
        render_forecast_results(results[0])
        return
    
    # Comparison chart
    st.markdown("### 📊 Visual Comparison")
    fig = create_model_comparison_chart(results)
    st.plotly_chart(fig, use_container_width=True)
    
    st.caption(
        "📖 **Comparison view:** Each colored line represents a different model's forecast. "
        "Use this to see which models agree or disagree on future trends."
    )
    
    st.divider()
    
    # Performance comparison table
    st.markdown("### 📈 Performance Metrics Comparison")
    
    comparison_data = []
    for result in results:
        row = {
            "Model": result.model_name,
            "Type": result.model_type,
            "Trend": result.get_summary_stats()["trend"],
            "Mean Forecast": f"{result.get_summary_stats()['mean_forecast']:.2f}",
        }
        
        # Add metrics if available
        if result.test_metrics:
            for metric, value in result.test_metrics.items():
                row[metric.upper()] = f"{value:.4f}"
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # Individual model details
    st.markdown("### 📋 Individual Model Details")
    
    selected_model = st.selectbox(
        "Select model to view details",
        options=[f"{r.model_name} ({r.model_type})" for r in results],
    )
    
    selected_index = [i for i, r in enumerate(results) if f"{r.model_name} ({r.model_type})" == selected_model][0]
    selected_result = results[selected_index]
    
    # Show detailed results for selected model
    render_forecast_results(
        selected_result,
        show_ai_summary=True,
        show_metrics=True,
        show_table=True,
        show_aggregations=False,  # Skip aggregations in comparison view
    )


def render_forecast_quick_summary(result: ForecastResult) -> None:
    """
    Render compact forecast summary (for dashboards or quick views).
    
    Args:
        result: ForecastResult object
    """
    
    stats = result.get_summary_stats()
    
    st.markdown(f"#### {result.model_name}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Trend", stats["trend"])
    with col2:
        st.metric("Avg Forecast", f"{stats['mean_forecast']:.2f}")
    with col3:
        if "uncertainty_pct" in stats:
            st.metric("Uncertainty", f"±{stats['uncertainty_pct']:.1f}%")
        else:
            st.metric("Horizon", f"{stats['forecast_horizon']} periods")
    
    # Mini chart
    fig = create_forecast_visualization(result, height=300)
    st.plotly_chart(fig, use_container_width=True)


def render_forecast_success_message(result: ForecastResult) -> None:
    """
    Render success message after training with key highlights.
    
    Args:
        result: ForecastResult object
    """
    
    stats = result.get_summary_stats()
    
    st.success(
        f"✅ **Forecast Complete!** {result.model_name} ({result.model_type}) generated "
        f"{stats['forecast_horizon']} predictions from {stats['start_date']} to {stats['end_date']}. "
        f"Expected trend: **{stats['trend']}**.",
        icon="🎉"
    )
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 View Full Results", use_container_width=True):
            st.session_state.show_forecast_results = True
    
    with col2:
        csv = result.to_dataframe().to_csv(index=False)
        st.download_button(
            label="📥 Download Forecast",
            data=csv,
            file_name=f"forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )
    
    with col3:
        if st.button("🔄 Train Another Model", use_container_width=True):
            st.session_state.show_forecast_results = False
            st.rerun()
