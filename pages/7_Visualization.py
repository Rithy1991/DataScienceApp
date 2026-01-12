from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="DataScope Pro - Visualization", layout="wide", initial_sidebar_state="expanded")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

from src.core.config import load_config
from src.core.state import get_clean_df, get_df, set_df, set_clean_df
from src.core.ui import sidebar_dataset_status, instruction_block, page_navigation
from src.core.styles import render_stat_card, inject_custom_css
from src.core.premium_styles import inject_premium_css, get_plotly_theme
from src.core.modern_components import enhanced_chart
from src.core.ai_helper import ai_sidebar_assistant, ai_interpretation_box
from src.core.platform_ui import module_section
from src.core.standardized_ui import (
    standard_section_header,
    concept_explainer,
    beginner_tip,
    common_mistakes_panel,
)


config = load_config()

# Apply custom CSS
inject_custom_css()

# Add AI assistant
ai_sidebar_assistant()

st.markdown(
    """
    <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
        <div style="font-size: 24px; font-weight: 800;">🎨 Visualization Studio</div>
        <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Build interactive charts that make patterns easy to see and share.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

instruction_block(
    "How to use this page",
    [
        "Use cleaned data when you can; raw works if it's light.",
        "Pick a chart type, set X/Y, and add color or animation only if it helps readability.",
        "If a chart feels crowded, trim columns or filter rows before plotting.",
        "Try the gallery for quick histograms, scatter grids, correlations, and box plots.",
        "Adjust fields and styles, then export the view you like for reports.",
    ],
)

st.info(
    "Interactive plotting to move from raw tables to clear visuals with hover, zoom, and export built in.",
    icon="ℹ️",
)

cta1, cta2 = st.columns([1, 3])
with cta1:
    if st.button("📓 Visualization Journal", type="primary", use_container_width=True):
        st.switch_page("pages/8_Viz_Journal.py")
with cta2:
    st.caption("Learn visualization from beginner → advanced with interactive practice.")

raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)

sidebar_dataset_status(raw_df, clean_df)

df = clean_df if clean_df is not None else raw_df

if df is None:
    st.warning("No data loaded. Quickly load a sample to try the charts.")
    if st.button("Load sample dataset (Retail Sales)", type="primary", use_container_width=True):
        sample = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=120, freq="D"),
                "store": np.random.choice(["A", "B", "C"], size=120),
                "promo": np.random.choice([0, 1], size=120, p=[0.7, 0.3]),
                "sales": np.random.normal(loc=11000, scale=2800, size=120).round(0),
                "transactions": np.random.normal(loc=380, scale=80, size=120).round(0),
            }
        )
        set_df(st.session_state, sample, source="sample_retail")
        set_clean_df(st.session_state, sample.copy())
        st.success("Sample dataset loaded—scroll to build charts.")
        raw_df = sample
        clean_df = sample
        df = sample

if df is None:
    st.info("Load data in Data Cleaning page first.")
    st.stop()

# Data overview
cv1, cv2, cv3 = st.columns(3)
with cv1:
    st.markdown(render_stat_card("Rows", f"{df.shape[0]:,}", icon="🗂️"), unsafe_allow_html=True)
with cv2:
    st.markdown(render_stat_card("Columns", f"{df.shape[1]:,}", icon="📑"), unsafe_allow_html=True)
with cv3:
    num_cols_count = df.select_dtypes(include=["number"]).shape[1]
    st.markdown(render_stat_card("Numeric cols", f"{num_cols_count:,}", icon="📊"), unsafe_allow_html=True)

st.divider()

# Create tabs for different visualization approaches
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Custom Builder", "📚 Quick Gallery", "📊 Templates", "🎨 Chart Comparison", "💡 Learn"])

with tab1:
    st.subheader("Interactive Plot Builder")
    st.caption("Create custom visualizations with just a few clicks")
    
    col_chart, col_xy = st.columns(2)
    
    with col_chart:
        chart = st.selectbox(
            "Chart type",
            options=["Line", "Scatter", "Bar", "Histogram", "Box", "Heatmap", "Violin", "2D Density"],
            help="Choose the visualization type"
        )
    
    with col_xy:
        st.caption("")  # Spacing
    
    cols = list(df.columns)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        x = st.selectbox("X-axis", options=cols, index=0)
    
    with col2:
        y_idx = min(1, len(cols) - 1)
        y = st.selectbox("Y-axis", options=cols, index=y_idx)
    
    with col3:
        color = st.selectbox("Color by", options=["(none)"] + cols, index=0)
    
    with col4:
        hover = st.multiselect("Hover details", options=cols, default=[x, y] if x != y else [x], max_selections=5)
    
    # Advanced styling options
    st.divider()
    col_palette1, col_palette2 = st.columns(2)
    with col_palette1:
        palette = st.selectbox(
            "Color palette",
            options=["Default", "Viridis", "Plasma", "Cividis", "Blues", "Reds", "Greens", "Turbo", "Plotly"],
            help="Choose a color scheme for your chart"
        )
    with col_palette2:
        show_grid = st.checkbox("Show grid lines", value=True, help="Toggle grid visibility")
    
    # Build figure
    kwargs = {
        "x": x,
        "hover_data": hover,
        "title": f"{chart}: {x} vs {y}",
        "template": "plotly_white"
    }
    
    if color != "(none)":
        kwargs["color"] = color
    
    fig = None
    try:
        if chart == "Line":
            fig = px.line(df, y=y, **kwargs)
        elif chart == "Scatter":
            fig = px.scatter(df, y=y, **kwargs)
            fig.update_traces(marker=dict(size=6, opacity=0.7))
        elif chart == "Bar":
            fig = px.bar(df, y=y, **kwargs)
        elif chart == "Histogram":
            fig = px.histogram(df, x=x, color=None if color == "(none)" else color, nbins=50, title=f"Distribution: {x}")
        elif chart == "Box":
            fig = px.box(df, x=x, y=y, color=None if color == "(none)" else color, title=f"Box plot: {y} by {x}")
        elif chart == "Violin":
            fig = px.violin(df, x=x, y=y, color=None if color == "(none)" else color, title=f"Violin: {y} by {x}")
        elif chart == "2D Density":
            fig = px.density_contour(df, x=x, y=y, color=None if color == "(none)" else color, nbinsx=30, nbinsy=30)
        elif chart == "Heatmap":
            numeric = df.select_dtypes(include=["number"])
            if numeric.shape[1] < 2:
                st.error("Need at least 2 numeric columns for heatmap")
            else:
                corr = numeric.corr(numeric_only=True)
                fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title="Correlation matrix")
        
        if fig is not None:
            fig.update_layout(hovermode="closest", height=500)
            
            # Apply color palette if not default
            if palette != "Default":
                palette_map = {
                    "Viridis": "Viridis", "Plasma": "Plasma", "Cividis": "Cividis",
                    "Blues": "Blues", "Reds": "Reds", "Greens": "Greens",
                    "Turbo": "Turbo", "Plotly": "Plotly"
                }
                if chart in ["Heatmap"]:
                    fig.update_traces(colorscale=palette_map.get(palette, "Viridis"))
                elif color != "(none)":
                    fig.update_layout(coloraxis_colorscale=palette_map.get(palette, "Viridis"))
            
            # Apply grid settings
            if not show_grid:
                fig.update_xaxes(showgrid=False)
                fig.update_yaxes(showgrid=False)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Add explanation of what the chart shows
            st.divider()
            st.markdown("### 📖 What This Chart Shows")
            
            explanations = {
                "Line": f"A line chart showing the progression of **{y}** across **{x}**. Use this to track trends and changes over time or categories. The line connects data points sequentially.",
                "Scatter": f"A scatter plot displaying the relationship between **{x}** (horizontal) and **{y}** (vertical). Look for clusters or patterns. Each dot represents one row.",
                "Bar": f"A bar chart comparing **{y}** values across different **{x}** categories. Bars make it easy to compare magnitudes side-by-side.",
                "Histogram": f"A histogram showing the distribution of **{x}**. The height of each bar shows how many values fall in that range. Use to spot patterns, skew, or outliers.",
                "Box": f"A box plot comparing **{y}** across **{x}** categories. The box shows the middle 50% of values; the line inside is the median; whiskers extend to typical data bounds.",
                "Violin": f"A violin plot showing the full distribution of **{y}** within each **{x}** category. Wider sections show where values are more common.",
                "2D Density": f"A 2D density contour plot showing where points are concentrated in the **{x}** vs **{y}** space. Darker/warmer colors indicate higher density.",
                "Heatmap": "A correlation heatmap showing relationships between numeric columns. Red = positive correlation, blue = negative. Values range from -1 to +1."
            }
            
            explanation = explanations.get(chart, "Visualization of your data.")
            st.info(f"💡 {explanation}")
            
            # Export options
            st.divider()
            st.caption("Export Options")
            export_col1, export_col2, export_col3 = st.columns(3)
            with export_col1:
                if st.button("📥 Save as PNG", use_container_width=True):
                    st.info("💡 Click the camera icon 📷 in the chart toolbar above to save as PNG")
            with export_col2:
                html_str = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    "📄 Download HTML",
                    data=html_str,
                    file_name=f"{chart.lower()}_chart.html",
                    mime="text/html",
                    use_container_width=True
                )
            with export_col3:
                json_str = fig.to_json()
                st.download_button(
                    "📋 Download JSON",
                    data=json_str,
                    file_name=f"{chart.lower()}_data.json",
                    mime="application/json",
                    use_container_width=True
                )
    except Exception as e:
        st.error(f"Chart creation failed: {str(e)}")
    
    st.info("💡 **Tips:** Use color to add a third dimension. Hover to inspect values. Use zoom/pan buttons in the top-right.")

with tab2:
    st.subheader("Pre-built Visualization Gallery")
    st.caption("Explore ready-made charts for quick insights")
    
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    
    gallery_tabs = st.tabs(["📊 Distributions", "🔗 Scatter Matrix", "🔥 Correlation", "📦 Category Compare"])
    
    with gallery_tabs[0]:
        st.subheader("Numeric Distributions")
        if numeric_cols:
            cols_to_show = st.multiselect("Select columns", numeric_cols, default=numeric_cols[:3], max_selections=6)
            if cols_to_show:
                for col in cols_to_show:
                    fign = px.histogram(
                        df, x=col, nbins=50,
                        title=f"📊 Distribution: {col}",
                        color_discrete_sequence=["#0ea5e9"]
                    )
                    st.plotly_chart(fign, use_container_width=True)
                    st.caption(f"📖 Shows how **{col}** values are spread out. Look for peaks (common values), skew (asymmetry), or gaps in the data.")
        else:
            st.info("No numeric columns available")
    
    with gallery_tabs[1]:
        st.subheader("Pairwise Scatter Matrix")
        sm_cols = numeric_cols[: min(5, len(numeric_cols))]
        if len(sm_cols) >= 2:
            figsm = px.scatter_matrix(
                df[sm_cols].dropna(),
                dimensions=sm_cols,
                height=600,
                title="🔗 Pairwise Relationships"
            )
            st.plotly_chart(figsm, use_container_width=True)
            st.caption("📖 Each plot shows the relationship between two numeric columns. Diagonal shows individual distributions. Look for linear or curved patterns.")
        else:
            st.info("Need at least 2 numeric columns")
    
    with gallery_tabs[2]:
        st.subheader("Correlation Heatmap")
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr(numeric_only=True)
            figc = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="RdBu",
                zmin=-1, zmax=1,
                title="🔥 Correlation Matrix"
            )
            st.plotly_chart(figc, use_container_width=True)
            st.caption("📖 Red = positive correlation (both increase together). Blue = negative correlation (one increases, other decreases). Dark/faint = no correlation.")
        else:
            st.info("Need at least 2 numeric columns")
    
    with gallery_tabs[3]:
        st.subheader("Category vs Numeric")
        if categorical_cols and numeric_cols:
            cat_col = st.selectbox("Category", options=categorical_cols)
            num_col = st.selectbox("Numeric measure", options=numeric_cols)
            
            figb = px.box(
                df, x=cat_col, y=num_col,
                color=cat_col,
                points="outliers",
                title=f"📦 {num_col} Distribution by {cat_col}"
            )
            figb.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(figb, use_container_width=True)
            st.caption(f"📖 Compares **{num_col}** across **{cat_col}** groups. The box shows the middle 50%; dots are outliers. Helps spot group differences.")
        else:
            st.info("Need at least 1 category and 1 numeric column")

with tab3:
    st.subheader("� Chart Templates & Presets")
    st.caption("Ready-to-use visualization templates for common analysis scenarios")
    
    template_type = st.selectbox(
        "Select template",
        options=[
            "Business Dashboard",
            "Sales Analysis",
            "Time Series Monitoring",
            "Customer Segmentation",
            "Performance Scorecard"
        ]
    )
    
    if template_type == "Business Dashboard":
        st.markdown("### 📈 Business Dashboard Template")
        st.caption("Key metrics with trend indicators")
        
        if len(numeric_cols) >= 3:
            # KPI cards
            col1, col2, col3 = st.columns(3)
            with col1:
                val1 = df[numeric_cols[0]].mean()
                st.metric(numeric_cols[0], f"{val1:.2f}", delta=f"{val1*0.05:.1f}")
            with col2:
                val2 = df[numeric_cols[1]].sum()
                st.metric(numeric_cols[1], f"{val2:.0f}", delta=f"{val2*0.03:.0f}")
            with col3:
                val3 = df[numeric_cols[2]].max()
                st.metric(numeric_cols[2], f"{val3:.2f}", delta=f"-{val3*0.02:.1f}")
            
            # Trend chart
            fig_trend = go.Figure()
            for col in numeric_cols[:3]:
                fig_trend.add_trace(go.Scatter(
                    y=df[col].head(100),
                    name=col,
                    mode='lines+markers'
                ))
            fig_trend.update_layout(title="Trend Overview", height=400)
            st.plotly_chart(fig_trend, use_container_width=True)
        else:
            st.info("Need at least 3 numeric columns for this template")
    
    elif template_type == "Sales Analysis":
        st.markdown("### 💰 Sales Analysis Template")
        if numeric_cols and categorical_cols:
            sales_col = st.selectbox("Sales metric", numeric_cols)
            category_col = st.selectbox("Group by", categorical_cols)
            
            # Bar chart
            fig_sales = px.bar(
                df.groupby(category_col)[sales_col].sum().reset_index(),
                x=category_col,
                y=sales_col,
                title=f"{sales_col} by {category_col}",
                color=sales_col,
                color_continuous_scale="Greens"
            )
            st.plotly_chart(fig_sales, use_container_width=True)
            
            # Pie chart
            fig_pie = px.pie(
                df,
                names=category_col,
                values=sales_col,
                title=f"{sales_col} Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.info("Need both numeric and categorical columns")
    
    elif template_type == "Time Series Monitoring":
        st.markdown("### ⏰ Time Series Monitoring Template")
        if numeric_cols:
            metric_col = st.selectbox("Metric to monitor", numeric_cols)
            
            # Line chart with markers
            fig_ts = px.line(
                df.head(200),
                y=metric_col,
                title=f"{metric_col} Over Time",
                markers=True
            )
            fig_ts.add_hline(
                y=df[metric_col].mean(),
                line_dash="dash",
                line_color="red",
                annotation_text="Mean"
            )
            st.plotly_chart(fig_ts, use_container_width=True)
            
            # Summary stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[metric_col].mean():.2f}")
            with col2:
                st.metric("Std Dev", f"{df[metric_col].std():.2f}")
            with col3:
                st.metric("Min", f"{df[metric_col].min():.2f}")
            with col4:
                st.metric("Max", f"{df[metric_col].max():.2f}")
        else:
            st.info("Need numeric columns for time series")
    
    elif template_type == "Customer Segmentation":
        st.markdown("### 👥 Customer Segmentation Template")
        if len(numeric_cols) >= 2:
            x_seg = st.selectbox("X-axis (segment feature)", numeric_cols, index=0)
            y_seg = st.selectbox("Y-axis (segment feature)", numeric_cols, index=min(1, len(numeric_cols)-1))
            
            fig_seg = px.scatter(
                df,
                x=x_seg,
                y=y_seg,
                title=f"Segmentation: {x_seg} vs {y_seg}",
                marginal_x="histogram",
                marginal_y="histogram",
                opacity=0.6
            )
            st.plotly_chart(fig_seg, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns")
    
    else:  # Performance Scorecard
        st.markdown("### 🎯 Performance Scorecard Template")
        if numeric_cols:
            selected_metrics = st.multiselect(
                "Select metrics for scorecard",
                numeric_cols,
                default=numeric_cols[:min(5, len(numeric_cols))]
            )
            
            if selected_metrics:
                scorecard_data = []
                for metric in selected_metrics:
                    scorecard_data.append({
                        "Metric": metric,
                        "Current": df[metric].iloc[-1] if len(df) > 0 else 0,
                        "Average": df[metric].mean(),
                        "Target": df[metric].max() * 0.9,
                        "Status": "✅" if df[metric].iloc[-1] > df[metric].mean() else "⚠️"
                    })
                
                scorecard_df = pd.DataFrame(scorecard_data)
                st.dataframe(scorecard_df, use_container_width=True, hide_index=True)
                
                # Radar chart
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row["Current"] for _, row in scorecard_df.iterrows()],
                    theta=selected_metrics,
                    fill='toself',
                    name='Current Performance'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True)),
                    title="Performance Radar"
                )
                st.plotly_chart(fig_radar, use_container_width=True)
        else:
            st.info("Need numeric columns for scorecard")

with tab4:
    st.subheader("🎨 Side-by-Side Chart Comparison")
    st.caption("Compare different visualizations of your data simultaneously")
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown("#### Chart 1")
        chart1 = st.selectbox("Chart type", ["Line", "Scatter", "Bar", "Histogram"], key="chart1_type")
        x1 = st.selectbox("X-axis", cols, index=0, key="chart1_x")
        y1_idx = min(1, len(cols) - 1)
        y1 = st.selectbox("Y-axis", cols, index=y1_idx, key="chart1_y")
        
        try:
            if chart1 == "Line":
                fig1 = px.line(df, x=x1, y=y1, title=f"Line: {x1} vs {y1}")
            elif chart1 == "Scatter":
                fig1 = px.scatter(df, x=x1, y=y1, title=f"Scatter: {x1} vs {y1}")
            elif chart1 == "Bar":
                fig1 = px.bar(df, x=x1, y=y1, title=f"Bar: {x1} vs {y1}")
            else:  # Histogram
                fig1 = px.histogram(df, x=x1, title=f"Distribution: {x1}")
            
            fig1.update_layout(height=400)
            st.plotly_chart(fig1, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with comp_col2:
        st.markdown("#### Chart 2")
        chart2 = st.selectbox("Chart type", ["Line", "Scatter", "Bar", "Box"], key="chart2_type")
        x2 = st.selectbox("X-axis", cols, index=min(1, len(cols)-1), key="chart2_x")
        y2_idx = min(2, len(cols) - 1)
        y2 = st.selectbox("Y-axis", cols, index=y2_idx, key="chart2_y")
        
        try:
            if chart2 == "Line":
                fig2 = px.line(df, x=x2, y=y2, title=f"Line: {x2} vs {y2}")
            elif chart2 == "Scatter":
                fig2 = px.scatter(df, x=x2, y=y2, title=f"Scatter: {x2} vs {y2}")
            elif chart2 == "Bar":
                fig2 = px.bar(df, x=x2, y=y2, title=f"Bar: {x2} vs {y2}")
            else:  # Box
                fig2 = px.box(df, x=x2, y=y2, title=f"Box: {x2} vs {y2}")
            
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    st.divider()
    st.info("💡 **Tip**: Use comparison view to analyze relationships from different angles or spot patterns across visualizations")

with tab5:
    st.subheader("�📚 Data Visualization Guide")
    
    st.markdown("""
    ### Chart Selection by Use Case
    
    **When to use each chart:**
    
    | Chart Type | Best For | X-axis | Y-axis |
    |-----------|----------|--------|--------|
    | **Line** | Time series, trends | Time or continuous | Numeric |
    | **Scatter** | Relationships, clusters | Numeric | Numeric |
    | **Bar** | Comparisons, rankings | Categorical | Numeric |
    | **Histogram** | Distributions, frequency | Numeric | Count |
    | **Box** | Compare distributions | Categorical | Numeric |
    | **Heatmap** | Correlations, patterns | Numeric | Numeric |
    
    ### Color Best Practices
    
    - ✅ Add a **third dimension** to your plot
    - ✅ Use **diverging scales** for correlations (-1 to +1)
    - ✅ Use **sequential scales** for distributions (0 to max)
    - ⚠️ Avoid too many colors (limits readability)
    
    ### Interactive Features
    
    All Plotly charts support:
    - 🔍 **Zoom**: Box select or scroll
    - 👆 **Pan**: Click and drag
    - 📌 **Hover**: Inspect values
    - 💾 **Export**: Camera icon saves PNG
    - 👁️ **Toggle**: Click legend items
    """)
    
    st.divider()
    
    st.subheader("💡 Pro Tips")
    st.markdown("""
    1. **Reduce clutter**: Limit X-axis categories (use top 10)
    2. **Normalize scales**: When comparing variables with different ranges
    3. **Highlight key data**: Use color to emphasize important groups
    4. **Add context**: Include units, sources, and explanations
    5. **Test for colorblindness**: Avoid red-green combinations
    """)

# Page navigation
standard_section_header("Learning Guide & Best Practices", "🎓")
concept_explainer(
    title="Data Visualization",
    explanation=(
        "Visualization communicates insights clearly. Choose chart types based on data and audience, and annotate to provide context."
    ),
    real_world_example=(
        "Customer segmentation: Use scatter plots with clusters, bar charts for segment sizes, and annotated line charts for trends."
    ),
)
beginner_tip("Tip: One message per chart — reduce clutter and highlight what matters.")
common_mistakes_panel({
    "Misleading axes": "Always start at zero for bars; label units & scales.",
    "Too many categories": "Aggregate or filter to top-N for readability.",
    "Confusing colors": "Use consistent palettes; avoid red/green conflicts.",
    "Missing context": "Add titles, captions, and sources.",
    "Overanimated graphics": "Animation can distract; use sparingly.",
})

page_navigation("7")
