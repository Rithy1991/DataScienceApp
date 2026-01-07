from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta

st.set_page_config(
    page_title="DataScope Pro - Visualization Journal",
    layout="wide",
    initial_sidebar_state="expanded",
)

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import load_config
from src.core.state import get_clean_df, get_df
from src.core.styles import inject_custom_css, render_stat_card
from src.core.ui import instruction_block, page_navigation, sidebar_dataset_status
from src.core.ai_helper import ai_sidebar_assistant


def _plot(fig, key: str) -> None:
    st.plotly_chart(fig, use_container_width=True, key=key)


def _generate_financial_timeseries(days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Generate realistic financial time-series data with trends, volatility, and patterns."""
    rng = np.random.default_rng(seed)
    
    # Generate dates
    dates = pd.date_range(start="2023-01-01", periods=days, freq="D")
    
    # Price generation with realistic patterns
    base_price = 100.0
    drift = 0.0005  # Slight upward trend
    volatility = 0.02
    
    returns = rng.normal(drift, volatility, days)
    # Add occasional volatility spikes
    volatility_spikes = (rng.uniform(0, 1, days) < 0.05) * rng.normal(0, 0.05, days)
    returns += volatility_spikes
    
    # Cumulative returns to prices
    prices = base_price * np.exp(np.cumsum(returns))
    
    # OHLC data
    high = prices * (1 + np.abs(rng.normal(0, 0.01, days)))
    low = prices * (1 - np.abs(rng.normal(0, 0.01, days)))
    open_prices = prices + rng.normal(0, 0.5, days)
    close_prices = prices
    
    # Volume with trend correlation
    base_volume = 1_000_000
    volume = base_volume + rng.normal(0, 200_000, days) + np.abs(returns) * 5_000_000
    volume = np.maximum(volume, 100_000)
    
    # Create dataframe
    df = pd.DataFrame({
        "date": dates,
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close_prices,
        "volume": volume,
        "returns": returns,
    })
    
    # Add technical indicators
    df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["volatility"] = df["returns"].rolling(window=20, min_periods=1).std()
    df["volume_sma"] = df["volume"].rolling(window=20, min_periods=1).mean()
    
    return df


def _generate_forecast_data(df: pd.DataFrame, horizon: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic forecast data with uncertainty bands."""
    last_date = df["date"].iloc[-1]
    last_price = df["close"].iloc[-1]
    
    # Generate forecast dates
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=horizon, freq="D")
    
    # Simple forecast with uncertainty
    rng = np.random.default_rng(123)
    trend = np.linspace(0, 5, horizon)  # Slight upward trend
    noise = rng.normal(0, 2, horizon)
    forecast = last_price + trend + noise
    
    # Uncertainty bands (increasing with time)
    uncertainty = np.linspace(2, 8, horizon)
    upper_band = forecast + uncertainty
    lower_band = forecast - uncertainty
    
    forecast_df = pd.DataFrame({
        "date": forecast_dates,
        "forecast": forecast,
        "upper": upper_band,
        "lower": lower_band,
    })
    
    # Generate "actual" future data for comparison
    actual_noise = rng.normal(0, 3, horizon)
    actual = last_price + trend + actual_noise
    
    actual_df = pd.DataFrame({
        "date": forecast_dates,
        "actual": actual,
    })
    
    return forecast_df, actual_df


# ===========================
# Configuration & Setup
# ===========================
config = load_config()
inject_custom_css()
ai_sidebar_assistant()

# ===========================
# Header
# ===========================
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    ">
        <div style="font-size: 28px; font-weight: 800; margin-bottom: 8px;">📊 Visualization Journal & Practical Learning</div>
        <div style="font-size: 16px; opacity: 0.95;">Master financial data visualization through hands-on exploration — from beginner to expert</div>
    </div>
    """,
    unsafe_allow_html=True,
)

instruction_block(
    "How to use this Visual Journal",
    [
        "📚 **Learn by doing** — Every visualization is interactive with controls to explore",
        "🎯 **Progressive learning** — Start with basics, advance to expert-level techniques",
        "💡 **Visual literacy** — Build intuition through observation, not formulas",
        "🔍 **Focus on interpretation** — What to observe, common mistakes, practical tips",
        "🎨 **Beautiful charts** — Professional aesthetics with dark-mode design",
    ],
    expanded=True,
)

# Session datasets
raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)
sidebar_dataset_status(raw_df, clean_df)

st.divider()

# ===========================
# Generate Sample Data
# ===========================
st.subheader("📊 Practice Dataset")
st.caption("Synthetic financial time-series data for learning visualization techniques")

days_to_generate = st.slider("Days of historical data", 90, 730, 365, step=30, key="viz_journal_days")
financial_df = _generate_financial_timeseries(days=days_to_generate)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(render_stat_card("Records", f"{len(financial_df):,}", icon="📅"), unsafe_allow_html=True)
with c2:
    st.markdown(render_stat_card("Current Price", f"${financial_df['close'].iloc[-1]:.2f}", icon="💵"), unsafe_allow_html=True)
with c3:
    change_pct = ((financial_df['close'].iloc[-1] - financial_df['close'].iloc[0]) / financial_df['close'].iloc[0]) * 100
    st.markdown(render_stat_card("Total Return", f"{change_pct:+.2f}%", icon="📈"), unsafe_allow_html=True)
with c4:
    st.markdown(render_stat_card("Avg Volume", f"{financial_df['volume'].mean()/1e6:.2f}M", icon="📊"), unsafe_allow_html=True)

st.divider()

# ===========================
# Learning Path Tabs
# ===========================
learning_tabs = st.tabs([
    "🧭 Foundations & Chart Literacy",
    "🌱 Beginner: Fundamentals",
    "📈 Intermediate: Deep Exploration", 
    "🎓 Advanced: Expert Techniques",
    "🎯 Practice Exercises"
])

# Named tab handles for clarity
foundations_tab, beginner_tab, intermediate_tab, advanced_tab, practice_tab = learning_tabs

# ===========================
# FOUNDATIONS TAB (from Visualization Academy)
# ===========================
with foundations_tab:
    st.markdown("### 🧭 Foundations & Chart Literacy")
    st.caption("Quick-start lessons on axes, scales, and core chart reading — adapted from the Visualization Academy.")

    numeric_cols = financial_df.select_dtypes(include=["number"]).columns.tolist()

    with st.expander("Lesson: Axes & Scales (Linear vs Log)", expanded=True):
        st.markdown("Use log scales when values span orders of magnitude; keep linear when changes are small.")

        if numeric_cols:
            x = st.selectbox("X", options=numeric_cols, index=0, key="found_axes_x")
            y = st.selectbox("Y", options=numeric_cols, index=min(1, len(numeric_cols) - 1), key="found_axes_y")
            use_log_y = st.checkbox("Use log scale for Y", value=False, key="found_axes_log")

            fig = px.scatter(financial_df, x=x, y=y, title=f"Scatter: {x} vs {y}", template="plotly_dark")
            if use_log_y:
                fig.update_yaxes(type="log")
            fig.update_layout(height=420, hovermode="closest")
            _plot(fig, key="found_scatter")

            st.info("💡 Observe how scaling changes the apparent variability. Log scales compress large values and reveal relative differences.")
        else:
            st.info("Need numeric columns to demonstrate axis scaling.")

    with st.expander("Lesson: Core Price Chart with Layers"):
        st.markdown("Line chart + optional markers and a fast SMA overlay for trend context.")

        rng = st.slider("Days to show", 30, len(financial_df), 180, key="found_core_range")
        show_markers = st.checkbox("Show data points", False, key="found_core_markers")
        show_sma = st.checkbox("Show 20-day SMA", True, key="found_core_sma")

        subset = financial_df.tail(rng)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["close"],
            mode="lines+markers" if show_markers else "lines",
            name="Price",
            line=dict(color="#00D9FF", width=2),
            marker=dict(size=4),
        ))

        if show_sma:
            fig.add_trace(go.Scatter(
                x=subset["date"],
                y=subset["sma_20"],
                mode="lines",
                name="20d SMA",
                line=dict(color="#FFD700", width=2, dash="dash"),
            ))

        fig.update_layout(
            title="Core Price Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=420,
            hovermode="x unified",
        )
        _plot(fig, "found_core_line")

        st.info("💡 Exercise: Toggle the SMA. Does price stay above or below it most of the time? What does that say about trend direction?")

    with st.expander("Lesson: Color, Emphasis, and Clean Design"):
        st.markdown("""
        - Use color sparingly to group related series (bullish vs bearish, forecast vs actual).
        - Remove clutter: minimal gridlines, clear titles, concise annotations.
        - Highlight the takeaway with one accent color and short caption.
        """)
        st.success("Rule of thumb: One chart = one message. If you need two messages, use two charts.")

# ===========================
# BEGINNER TAB
# ===========================
with beginner_tab:
    st.markdown("### 🌱 Beginner Level: Understanding Basic Financial Visualizations")
    st.caption("Learn to read price movements, trends, and basic patterns")
    
    # Lesson 1: Basic Line Chart
    with st.expander("📊 Lesson 1: The Basic Line Chart — Your First Visualization", expanded=True):
        st.markdown("""
        **What you're seeing:** Price movement over time as a continuous line
        
        **What to observe:**
        - 📈 **Trend direction** — Is the price generally moving up, down, or sideways?
        - 🌊 **Volatility** — How smooth or jagged is the line? (Jagged = more volatile)
        - 🔄 **Patterns** — Do you see repeating cycles or sudden jumps?
        
        **Common mistakes:**
        - ❌ Looking at absolute price instead of percentage change
        - ❌ Ignoring the time scale (1 week vs 1 year tells different stories)
        - ❌ Seeing patterns where there's only noise
        
        **Pro tip:** Always check the Y-axis scale — a zoomed-in view can make small changes look dramatic
        """)
        
        # Interactive controls
        col1, col2 = st.columns([2, 1])
        with col1:
            date_range = st.slider(
                "Select time range (days)", 
                30, len(financial_df), 
                (0, len(financial_df)), 
                key="basic_line_range"
            )
        with col2:
            show_markers = st.checkbox("Show data points", False, key="basic_line_markers")
        
        # Plot
        subset = financial_df.iloc[date_range[0]:date_range[1]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["close"],
            mode="lines+markers" if show_markers else "lines",
            name="Price",
            line=dict(color="#00D9FF", width=2),
            marker=dict(size=4),
        ))
        
        fig.update_layout(
            title="Price Over Time — Basic Line Chart",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            hovermode="x unified",
            height=400,
        )
        _plot(fig, "beginner_line")
        
        st.info("💡 **Exercise:** Identify the 3 largest price drops in this chart. What do you notice about the spacing between them?")
    
    # Lesson 2: Candlestick Charts
    with st.expander("🕯️ Lesson 2: Candlestick Charts — Reading Market Psychology"):
        st.markdown("""
        **What you're seeing:** Each "candle" shows four prices for one day: Open, High, Low, Close (OHLC)
        
        **How to read candles:**
        - 🟢 **Green/White candle** — Price closed higher than it opened (buyers won)
        - 🔴 **Red/Black candle** — Price closed lower than it opened (sellers won)
        - 📏 **Body size** — Difference between open and close (bigger = stronger conviction)
        - 📍 **Wicks (shadows)** — High and low extremes during the period
        
        **What to observe:**
        - Long bodies = strong directional movement
        - Long wicks = price rejection (buyers/sellers fought back)
        - Small bodies = indecision
        - Consecutive same-color candles = sustained trend
        
        **Common mistakes:**
        - ❌ Ignoring the wicks (they show important price rejection)
        - ❌ Over-interpreting single candles (look for patterns)
        - ❌ Forgetting that green doesn't mean profit (depends on entry point)
        """)
        
        # Controls
        candle_range = st.slider("Days to display", 10, 90, 30, key="candle_range")
        
        # Plot candlestick
        subset = financial_df.tail(candle_range)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=subset["date"],
            open=subset["open"],
            high=subset["high"],
            low=subset["low"],
            close=subset["close"],
            name="OHLC",
            increasing_line_color="#00FF88",
            decreasing_line_color="#FF3366",
        ))
        
        fig.update_layout(
            title="Candlestick Chart — OHLC Price Action",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=450,
        )
        _plot(fig, "beginner_candle")
        
        st.info("💡 **Exercise:** Find a day with very long wicks but a small body. What does this tell you about that day's trading?")
    
    # Lesson 3: Volume and Price Relationship
    with st.expander("📊 Lesson 3: Volume — The Fuel Behind Price Moves"):
        st.markdown("""
        **What you're seeing:** Trading volume (number of shares/contracts traded) beneath the price chart
        
        **Why it matters:**
        - 🚀 **High volume + big price move** = Strong conviction (likely to continue)
        - 🤔 **Low volume + big price move** = Weak conviction (may reverse)
        - 💤 **Low volume + small moves** = Market is quiet/uncertain
        - ⚠️ **Volume spike** = Something significant happened (news, event)
        
        **What to observe:**
        - Does high volume confirm the price direction?
        - Do price breakouts happen on increasing volume?
        - Are reversals preceded by volume spikes?
        
        **Common mistakes:**
        - ❌ Ignoring volume entirely (price without volume is incomplete information)
        - ❌ Assuming high volume always means bullish (it just means activity)
        - ❌ Not comparing volume to its average (context matters)
        """)
        
        # Plot price and volume
        vol_range = st.slider("Days to display", 30, 180, 90, key="volume_range")
        subset = financial_df.tail(vol_range)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
            subplot_titles=("Price", "Volume")
        )
        
        # Price
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["close"],
                mode="lines",
                name="Price",
                line=dict(color="#00D9FF", width=2),
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ["#00FF88" if subset["close"].iloc[i] >= subset["open"].iloc[i] else "#FF3366" 
                  for i in range(len(subset))]
        
        fig.add_trace(
            go.Bar(
                x=subset["date"],
                y=subset["volume"],
                name="Volume",
                marker_color=colors,
            ),
            row=2, col=1
        )
        
        # Add volume average line
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["volume_sma"],
                mode="lines",
                name="Avg Volume",
                line=dict(color="#FFD700", width=1, dash="dash"),
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=550,
            hovermode="x unified",
            showlegend=True,
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        _plot(fig, "beginner_volume")
        
        st.info("💡 **Exercise:** Find a major price move. Was it accompanied by high or low volume? What might this suggest?")
    
    # Lesson 4: Simple Comparisons
    with st.expander("🔄 Lesson 4: Comparing Multiple Timeframes or Assets"):
        st.markdown("""
        **What you're seeing:** Multiple price series on the same chart for comparison
        
        **Why compare:**
        - 📊 See relative performance (which performed better?)
        - 🔍 Identify correlation (do they move together?)
        - ⚖️ Make informed decisions (diversification, hedging)
        
        **What to observe:**
        - Do the lines move together or independently?
        - Which has higher volatility (bigger swings)?
        - Are there periods where they diverge significantly?
        
        **Common mistakes:**
        - ❌ Comparing absolute prices instead of percentage returns
        - ❌ Not normalizing starting points (makes comparison difficult)
        - ❌ Ignoring different volatility levels
        
        **Pro tip:** When comparing different assets, normalize to 100 at the start to see relative performance
        """)
        
        # Generate two comparison series
        df2 = _generate_financial_timeseries(days=days_to_generate, seed=99)
        
        # Normalize to 100 at start
        compare_toggle = st.checkbox("Normalize to 100 (better for comparison)", True, key="compare_normalize")
        
        if compare_toggle:
            asset1 = (financial_df["close"] / financial_df["close"].iloc[0]) * 100
            asset2 = (df2["close"] / df2["close"].iloc[0]) * 100
            ylabel = "Normalized Price (Start = 100)"
        else:
            asset1 = financial_df["close"]
            asset2 = df2["close"]
            ylabel = "Price ($)"
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=financial_df["date"],
            y=asset1,
            mode="lines",
            name="Asset A",
            line=dict(color="#00D9FF", width=2),
        ))
        fig.add_trace(go.Scatter(
            x=df2["date"],
            y=asset2,
            mode="lines",
            name="Asset B",
            line=dict(color="#FF6B9D", width=2),
        ))
        
        fig.update_layout(
            title="Multi-Asset Comparison",
            xaxis_title="Date",
            yaxis_title=ylabel,
            template="plotly_dark",
            hovermode="x unified",
            height=400,
        )
        _plot(fig, "beginner_compare")
        
        st.info("💡 **Exercise:** Which asset was more volatile? Which would you prefer for long-term holding?")


# ===========================
# INTERMEDIATE TAB
# ===========================
with intermediate_tab:
    st.markdown("### 📈 Intermediate Level: Deep Exploration & Analysis")
    st.caption("Master zoom, pan, overlays, moving averages, and pattern recognition")
    
    # Lesson 1: Zooming and Panning
    with st.expander("🔍 Lesson 1: Zoom & Pan — Finding the Details", expanded=True):
        st.markdown("""
        **What you're learning:** How to navigate time-series data at different scales
        
        **Why it matters:**
        - 🔬 **Zoom in** — See intraday patterns, exact entry/exit points
        - 🌍 **Zoom out** — Understand long-term trends, avoid noise
        - 🎯 **Context switching** — Different timescales reveal different insights
        
        **What to observe:**
        - How does the trend look different at different scales?
        - Are there micro-patterns within macro-trends?
        - Do support/resistance levels become clearer when zoomed in?
        
        **Pro tip:** Use the range slider below to explore different time windows
        """)
        
        # Interactive zoom controls
        zoom_col1, zoom_col2 = st.columns([3, 1])
        with zoom_col1:
            zoom_range = st.slider(
                "Select time window to analyze",
                0, len(financial_df)-1,
                (len(financial_df)-90, len(financial_df)-1),
                key="zoom_range"
            )
        with zoom_col2:
            chart_type = st.radio("Chart type", ["Line", "Candles"], key="zoom_chart_type")
        
        subset = financial_df.iloc[zoom_range[0]:zoom_range[1]+1]
        
        fig = go.Figure()
        
        if chart_type == "Line":
            fig.add_trace(go.Scatter(
                x=subset["date"],
                y=subset["close"],
                mode="lines",
                name="Price",
                line=dict(color="#00D9FF", width=2),
            ))
        else:
            fig.add_trace(go.Candlestick(
                x=subset["date"],
                open=subset["open"],
                high=subset["high"],
                low=subset["low"],
                close=subset["close"],
                name="OHLC",
                increasing_line_color="#00FF88",
                decreasing_line_color="#FF3366",
            ))
        
        fig.update_layout(
            title=f"Focused View: {len(subset)} Days",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            height=450,
            xaxis_rangeslider_visible=True,
        )
        _plot(fig, "intermediate_zoom")
        
        st.info("💡 **Try this:** Zoom into a sharp price drop. Can you identify any pattern just before it happened?")
    
    # Lesson 2: Moving Averages
    with st.expander("📉 Lesson 2: Moving Averages — Smoothing the Noise"):
        st.markdown("""
        **What you're seeing:** Smoothed price lines that filter out short-term fluctuations
        
        **Types of moving averages:**
        - 📊 **20-day MA** (short-term) — Follows price closely, good for recent trends
        - 📈 **50-day MA** (medium-term) — Smoother, shows intermediate trends
        - 🎯 **Crossovers** — When short MA crosses above/below long MA (trend change signal)
        
        **What to observe:**
        - Is price above or below the moving averages? (trend direction)
        - How far apart are the MAs? (trend strength)
        - Do crossovers lead price changes or lag behind?
        
        **Common mistakes:**
        - ❌ Using MAs as absolute signals (they lag by design)
        - ❌ Ignoring the overall context
        - ❌ Over-relying on a single MA period
        
        **Pro tip:** MAs work best in trending markets, poorly in choppy sideways markets
        """)
        
        ma_period = st.slider("Days to display", 60, 300, 180, key="ma_period")
        subset = financial_df.tail(ma_period)
        
        show_sma20 = st.checkbox("Show 20-day MA", True, key="show_sma20")
        show_sma50 = st.checkbox("Show 50-day MA", True, key="show_sma50")
        
        fig = go.Figure()
        
        # Price
        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["close"],
            mode="lines",
            name="Price",
            line=dict(color="#FFFFFF", width=1.5),
            opacity=0.7,
        ))
        
        # Moving averages
        if show_sma20:
            fig.add_trace(go.Scatter(
                x=subset["date"],
                y=subset["sma_20"],
                mode="lines",
                name="20-day MA",
                line=dict(color="#00D9FF", width=2),
            ))
        
        if show_sma50:
            fig.add_trace(go.Scatter(
                x=subset["date"],
                y=subset["sma_50"],
                mode="lines",
                name="50-day MA",
                line=dict(color="#FF6B9D", width=2),
            ))
        
        fig.update_layout(
            title="Moving Averages — Trend Identification",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            hovermode="x unified",
            height=450,
        )
        _plot(fig, "intermediate_ma")
        
        st.info("💡 **Exercise:** Find where the 20-day MA crosses the 50-day MA. Did price follow the signal?")
    
    # Lesson 3: Volatility Visualization
    with st.expander("⚡ Lesson 3: Volatility — Measuring Market Uncertainty"):
        st.markdown("""
        **What you're seeing:** Rolling volatility (standard deviation of returns)
        
        **Why volatility matters:**
        - 📊 **High volatility** = Larger price swings, higher risk AND opportunity
        - 📉 **Low volatility** = Stable prices, lower risk but slower gains
        - ⚠️ **Volatility clusters** = Volatile periods tend to persist
        - 🔄 **Regime changes** = Sudden shifts from calm to chaotic
        
        **What to observe:**
        - Are there periods of expanding volatility (getting riskier)?
        - Does volatility spike during major price moves?
        - Is volatility mean-reverting (high vol → low vol → high vol)?
        
        **Common mistakes:**
        - ❌ Treating all price moves equally (ignore volatility context)
        - ❌ Not adjusting position size for volatility
        - ❌ Assuming past volatility predicts future moves
        
        **Pro tip:** Volatility expansion often precedes big directional moves
        """)
        
        vol_window = st.slider("Analysis window (days)", 90, 365, 180, key="vol_window")
        subset = financial_df.tail(vol_window)
        
        # Plot price and volatility
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.6, 0.4],
            subplot_titles=("Price", "Rolling Volatility (20-day)")
        )
        
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["close"],
                mode="lines",
                name="Price",
                line=dict(color="#00D9FF", width=2),
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["volatility"],
                mode="lines",
                name="Volatility",
                line=dict(color="#FF6B9D", width=2),
                fill="tozeroy",
                fillcolor="rgba(255, 107, 157, 0.3)",
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=550,
            hovermode="x unified",
        )
        fig.update_xaxes(title_text="Date", row=2, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=2, col=1)
        
        _plot(fig, "intermediate_volatility")
        
        st.info("💡 **Exercise:** Identify volatility expansion periods. Did they precede major price moves?")
    
    # Lesson 4: Annotated Patterns
    with st.expander("🎯 Lesson 4: Pattern Recognition — Marking Key Events"):
        st.markdown("""
        **What you're learning:** How to annotate and identify important patterns
        
        **Key patterns to recognize:**
        - 🔺 **Higher highs, higher lows** = Uptrend
        - 🔻 **Lower highs, lower lows** = Downtrend
        - ➡️ **Support levels** = Price bounces off a floor
        - ⬆️ **Resistance levels** = Price struggles to break through ceiling
        - ⚡ **Breakouts** = Price escapes a consolidation range
        
        **What to observe:**
        - How many times did price test a level before breaking?
        - Do breakouts happen on high volume?
        - After a breakout, does the old resistance become new support?
        
        **Pro tip:** The best patterns are validated by multiple tests and clear volume confirmation
        """)
        
        pattern_days = st.slider("Days to analyze", 60, 180, 90, key="pattern_days")
        subset = financial_df.tail(pattern_days)
        
        # Find local maxima and minima for annotations
        window = 10
        subset["local_max"] = subset["close"].rolling(window=window*2+1, center=True).max() == subset["close"]
        subset["local_min"] = subset["close"].rolling(window=window*2+1, center=True).min() == subset["close"]
        
        fig = go.Figure()
        
        # Price line
        fig.add_trace(go.Scatter(
            x=subset["date"],
            y=subset["close"],
            mode="lines",
            name="Price",
            line=dict(color="#00D9FF", width=2),
        ))
        
        # Annotate peaks
        peaks = subset[subset["local_max"] == True]
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=peaks["date"],
                y=peaks["close"],
                mode="markers",
                name="Local Peak",
                marker=dict(color="#FF3366", size=10, symbol="triangle-down"),
            ))
        
        # Annotate troughs
        troughs = subset[subset["local_min"] == True]
        if len(troughs) > 0:
            fig.add_trace(go.Scatter(
                x=troughs["date"],
                y=troughs["close"],
                mode="markers",
                name="Local Trough",
                marker=dict(color="#00FF88", size=10, symbol="triangle-up"),
            ))
        
        fig.update_layout(
            title="Pattern Recognition — Peaks & Troughs",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            hovermode="x unified",
            height=450,
        )
        _plot(fig, "intermediate_patterns")
        
        st.info("💡 **Exercise:** Connect the peaks with a line. Is there a clear resistance level? Do the same for troughs.")


# ===========================
# ADVANCED TAB
# ===========================
with advanced_tab:
    st.markdown("### 🎓 Advanced Level: Expert Techniques")
    st.caption("Animated visualizations, AI forecasts, uncertainty, distributions, and error analysis")
    
    # Lesson 1: Animated Time Replay
    with st.expander("🎬 Lesson 1: Animated Replay — See How Trends Develop", expanded=True):
        st.markdown("""
        **What you're learning:** Understanding how information unfolds over time
        
        **Why animation matters:**
        - 🎥 **Hindsight bias** — Charts look obvious in retrospect, but were uncertain in real-time
        - 📊 **Pattern formation** — See how trends emerge gradually
        - 🧠 **Decision points** — Where would you have entered/exited if trading live?
        
        **What to observe:**
        - At each moment, what information was available?
        - When did the trend become "obvious"?
        - How long did uncertainty persist?
        
        **Pro tip:** This exercise builds discipline — most patterns are unclear until after they complete
        """)
        
        replay_speed = st.slider("Replay up to day:", 30, min(180, len(financial_df)), 90, key="replay_speed")
        
        replay_subset = financial_df.iloc[:replay_speed]
        
        fig = go.Figure()
        
        # Historical data (shown in gray)
        fig.add_trace(go.Scatter(
            x=replay_subset["date"],
            y=replay_subset["close"],
            mode="lines",
            name="Price (Known)",
            line=dict(color="#00D9FF", width=2),
        ))
        
        # Add moving average up to current point
        fig.add_trace(go.Scatter(
            x=replay_subset["date"],
            y=replay_subset["sma_20"],
            mode="lines",
            name="20-day MA",
            line=dict(color="#FFD700", width=2, dash="dash"),
        ))
        
        # Future data (grayed out)
        future_subset = financial_df.iloc[replay_speed:]
        if len(future_subset) > 0:
            fig.add_trace(go.Scatter(
                x=future_subset["date"],
                y=future_subset["close"],
                mode="lines",
                name="Future (Unknown)",
                line=dict(color="#666666", width=1, dash="dot"),
                opacity=0.3,
            ))
        
        fig.update_layout(
            title=f"Time Replay — Day {replay_speed} of {len(financial_df)}",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            hovermode="x unified",
            height=450,
        )
        _plot(fig, "advanced_replay")
        
        st.info(f"💡 **Question:** Based on data up to day {replay_speed}, what would your prediction be? Now move the slider forward...")
    
    # Lesson 2: Forecast vs Actual Comparison
    with st.expander("🤖 Lesson 2: AI Forecast vs Reality — Measuring Accuracy"):
        st.markdown("""
        **What you're seeing:** Model predictions vs actual outcomes
        
        **Why this matters:**
        - 📊 **Model validation** — How well does the forecast perform?
        - 🎯 **Calibration** — Are confidence bands appropriate?
        - 🔍 **Error patterns** — Does the model miss certain events?
        - 📈 **Practical utility** — Would this forecast help decision-making?
        
        **What to observe:**
        - Does the actual price stay within the confidence bands?
        - Are errors random or systematic (always over/under-predicting)?
        - Does accuracy degrade over the forecast horizon?
        - Which periods have the largest forecast errors?
        
        **Common mistakes:**
        - ❌ Judging forecast quality on a single outcome
        - ❌ Ignoring uncertainty bands (point predictions are rarely perfect)
        - ❌ Not checking if errors are within expected ranges
        
        **Pro tip:** Good forecasts aren't always accurate — they're *calibrated* (50% confidence = right 50% of time)
        """)
        
        # Generate forecast
        forecast_horizon = st.slider("Forecast horizon (days ahead)", 10, 60, 30, key="forecast_horizon")
        
        # Use last portion of data for "historical" and rest for forecast comparison
        hist_cutoff = len(financial_df) - forecast_horizon
        hist_data = financial_df.iloc[:hist_cutoff]
        
        forecast_df, actual_df = _generate_forecast_data(hist_data, horizon=forecast_horizon)
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=hist_data["date"].tail(90),
            y=hist_data["close"].tail(90),
            mode="lines",
            name="Historical",
            line=dict(color="#00D9FF", width=2),
        ))
        
        # Forecast with uncertainty band
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["upper"],
            mode="lines",
            name="Upper Confidence",
            line=dict(width=0),
            showlegend=False,
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["lower"],
            mode="lines",
            name="Confidence Band",
            line=dict(width=0),
            fillcolor="rgba(255, 107, 157, 0.2)",
            fill="tonexty",
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast"],
            mode="lines",
            name="Forecast",
            line=dict(color="#FF6B9D", width=2, dash="dash"),
        ))
        
        # Actual future data
        fig.add_trace(go.Scatter(
            x=actual_df["date"],
            y=actual_df["actual"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#00FF88", width=2),
            marker=dict(size=4),
        ))
        
        fig.update_layout(
            title="Forecast vs Actual — Model Performance",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            hovermode="x unified",
            height=450,
        )
        _plot(fig, "advanced_forecast")
        
        # Calculate error metrics
        merged = pd.merge(forecast_df, actual_df, on="date")
        mae = np.abs(merged["forecast"] - merged["actual"]).mean()
        rmse = np.sqrt(((merged["forecast"] - merged["actual"]) ** 2).mean())
        within_bands = ((merged["actual"] >= merged["lower"]) & (merged["actual"] <= merged["upper"])).mean() * 100
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Absolute Error", f"${mae:.2f}")
        with col2:
            st.metric("RMSE", f"${rmse:.2f}")
        with col3:
            st.metric("Within Confidence Bands", f"{within_bands:.0f}%")
        
        st.info("💡 **Exercise:** Are the errors random or is there a pattern? Does the model consistently over or under-predict?")
    
    # Lesson 3: Distribution & Heatmap Analysis
    with st.expander("📊 Lesson 3: Distributions & Heatmaps — Understanding Data Density"):
        st.markdown("""
        **What you're seeing:** Return distributions and correlation heatmaps
        
        **Why distributions matter:**
        - 📊 **Return distribution** — Are returns normally distributed or fat-tailed?
        - 🎯 **Risk assessment** — How often do extreme events occur?
        - 📈 **Symmetry** — Are gains and losses symmetric?
        
        **What to observe:**
        - Shape of the distribution (bell curve vs fat tails)
        - Skewness (more extreme gains or losses?)
        - Outliers (how extreme are the extremes?)
        
        **Pro tip:** Financial returns often have "fat tails" (more extreme events than normal distribution predicts)
        """)
        
        # Return distribution
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Daily Returns Distribution", "Returns Over Time"),
            specs=[[{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=financial_df["returns"] * 100,  # Convert to percentage
                name="Returns",
                nbinsx=50,
                marker_color="#00D9FF",
                opacity=0.7,
            ),
            row=1, col=1
        )
        
        # Scatter of returns over time
        fig.add_trace(
            go.Scatter(
                x=financial_df["date"],
                y=financial_df["returns"] * 100,
                mode="markers",
                name="Daily Returns",
                marker=dict(
                    size=4,
                    color=financial_df["returns"],
                    colorscale="RdYlGn",
                    showscale=True,
                    colorbar=dict(title="Return %"),
                ),
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=400,
            showlegend=False,
        )
        fig.update_xaxes(title_text="Return (%)", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_yaxes(title_text="Return (%)", row=1, col=2)
        
        _plot(fig, "advanced_distribution")
        
        # Stats
        returns_pct = financial_df["returns"] * 100
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Return", f"{returns_pct.mean():.3f}%")
        with col2:
            st.metric("Std Dev", f"{returns_pct.std():.3f}%")
        with col3:
            st.metric("Skewness", f"{returns_pct.skew():.3f}")
        with col4:
            st.metric("Kurtosis", f"{returns_pct.kurtosis():.3f}")
        
        st.info("💡 **Exercise:** Is the distribution symmetric? Are there more extreme negative or positive returns?")
    
    # Lesson 4: Error Visualization Over Time
    with st.expander("📉 Lesson 4: Error Analysis — When Models Fail"):
        st.markdown("""
        **What you're seeing:** Forecast errors plotted over time
        
        **Why error analysis matters:**
        - 🎯 **Identify failure modes** — When does the model struggle?
        - 📊 **Temporal patterns** — Do errors cluster in certain periods?
        - 🔍 **Systematic bias** — Consistent over/under-prediction?
        - 📈 **Error evolution** — Does accuracy degrade over forecast horizon?
        
        **What to observe:**
        - Are large errors random or clustered?
        - Do errors increase with forecast horizon?
        - Are there specific market conditions where errors spike?
        
        **Pro tip:** Understanding when models fail is as important as knowing when they work
        """)
        
        # Generate forecast errors
        forecast_df, actual_df = _generate_forecast_data(financial_df.iloc[:-30], horizon=30)
        merged = pd.merge(forecast_df, actual_df, on="date")
        merged["error"] = merged["actual"] - merged["forecast"]
        merged["abs_error"] = np.abs(merged["error"])
        merged["error_pct"] = (merged["error"] / merged["actual"]) * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=("Forecast Error ($)", "Absolute Error ($)"),
        )
        
        # Error (positive = under-predicted, negative = over-predicted)
        colors = ["#00FF88" if e > 0 else "#FF3366" for e in merged["error"]]
        
        fig.add_trace(
            go.Bar(
                x=merged["date"],
                y=merged["error"],
                name="Error",
                marker_color=colors,
            ),
            row=1, col=1
        )
        
        # Absolute error trend
        fig.add_trace(
            go.Scatter(
                x=merged["date"],
                y=merged["abs_error"],
                mode="lines+markers",
                name="Absolute Error",
                line=dict(color="#FFD700", width=2),
                marker=dict(size=6),
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
            showlegend=False,
        )
        fig.update_xaxes(title_text="Forecast Date", row=2, col=1)
        fig.update_yaxes(title_text="Error ($)", row=1, col=1)
        fig.update_yaxes(title_text="Abs Error ($)", row=2, col=1)
        
        _plot(fig, "advanced_errors")
        
        # Error statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Error", f"${merged['error'].mean():.2f}")
        with col2:
            st.metric("Mean Absolute Error", f"${merged['abs_error'].mean():.2f}")
        with col3:
            st.metric("Max Error", f"${merged['abs_error'].max():.2f}")
        
        st.info("💡 **Exercise:** Does the error increase over the forecast horizon? Is there a systematic bias?")


# ===========================
# PRACTICE EXERCISES TAB
# ===========================
with practice_tab:
    st.markdown("### 🎯 Practice Exercises — Test Your Visual Literacy")
    st.caption("Apply what you've learned with interactive challenges")
    
    with st.expander("🎓 Exercise 1: Trend Identification Challenge", expanded=True):
        st.markdown("""
        **Your Task:** Analyze the chart below and answer:
        1. What is the overall trend direction?
        2. Identify at least 2 support or resistance levels
        3. Mark one period of high volatility
        4. Would you classify this as a trending or range-bound market?
        """)
        
        # Generate random segment
        ex1_start = np.random.randint(0, len(financial_df) - 120)
        ex1_subset = financial_df.iloc[ex1_start:ex1_start+120]
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=ex1_subset["date"],
            open=ex1_subset["open"],
            high=ex1_subset["high"],
            low=ex1_subset["low"],
            close=ex1_subset["close"],
            name="Price",
            increasing_line_color="#00FF88",
            decreasing_line_color="#FF3366",
        ))
        
        fig.update_layout(
            title="Exercise 1: What Do You See?",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False,
            height=400,
        )
        _plot(fig, "exercise1")
        
        with st.expander("💡 Reveal Analysis Tips"):
            st.markdown("""
            - **Trend:** Connect the highs and lows. Are they making higher highs/lows or lower highs/lows?
            - **Support/Resistance:** Look for price levels where the market bounced multiple times
            - **Volatility:** Look for periods with larger candle bodies and longer wicks
            - **Market Type:** Trending = clear direction, Range-bound = price bouncing between levels
            """)
    
    with st.expander("📊 Exercise 2: Volume Analysis Challenge"):
        st.markdown("""
        **Your Task:** Study price and volume, then answer:
        1. Find a major price move on high volume — is this a strong signal?
        2. Find a price move on low volume — what might this indicate?
        3. Identify a volume spike — what might have caused it?
        """)
        
        ex2_subset = financial_df.tail(100)
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
        )
        
        fig.add_trace(
            go.Scatter(
                x=ex2_subset["date"],
                y=ex2_subset["close"],
                mode="lines",
                name="Price",
                line=dict(color="#00D9FF", width=2),
            ),
            row=1, col=1
        )
        
        colors = ["#00FF88" if ex2_subset["close"].iloc[i] >= ex2_subset["open"].iloc[i] else "#FF3366" 
                  for i in range(len(ex2_subset))]
        
        fig.add_trace(
            go.Bar(
                x=ex2_subset["date"],
                y=ex2_subset["volume"],
                name="Volume",
                marker_color=colors,
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            template="plotly_dark",
            height=500,
        )
        _plot(fig, "exercise2")
        
        with st.expander("💡 Reveal Analysis Tips"):
            st.markdown("""
            - **High volume + big move:** Strong conviction, likely to continue
            - **Low volume + big move:** Weak conviction, prone to reversal
            - **Volume spikes:** Often indicate news, earnings, or significant events
            """)
    
    with st.expander("🔍 Exercise 3: Moving Average Crossover Strategy"):
        st.markdown("""
        **Your Task:** Backtest a simple strategy:
        - **Buy signal:** When 20-day MA crosses above 50-day MA
        - **Sell signal:** When 20-day MA crosses below 50-day MA
        
        Questions:
        1. How many buy/sell signals were generated?
        2. Were they profitable?
        3. What were the biggest weaknesses of this strategy?
        """)
        
        ex3_subset = financial_df.copy()
        
        # Detect crossovers
        ex3_subset["cross_up"] = (
            (ex3_subset["sma_20"] > ex3_subset["sma_50"]) & 
            (ex3_subset["sma_20"].shift(1) <= ex3_subset["sma_50"].shift(1))
        )
        ex3_subset["cross_down"] = (
            (ex3_subset["sma_20"] < ex3_subset["sma_50"]) & 
            (ex3_subset["sma_20"].shift(1) >= ex3_subset["sma_50"].shift(1))
        )
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=ex3_subset["date"],
            y=ex3_subset["close"],
            mode="lines",
            name="Price",
            line=dict(color="#FFFFFF", width=1),
            opacity=0.6,
        ))
        
        fig.add_trace(go.Scatter(
            x=ex3_subset["date"],
            y=ex3_subset["sma_20"],
            mode="lines",
            name="20-day MA",
            line=dict(color="#00D9FF", width=2),
        ))
        
        fig.add_trace(go.Scatter(
            x=ex3_subset["date"],
            y=ex3_subset["sma_50"],
            mode="lines",
            name="50-day MA",
            line=dict(color="#FF6B9D", width=2),
        ))
        
        # Mark crossover points
        buy_signals = ex3_subset[ex3_subset["cross_up"]]
        sell_signals = ex3_subset[ex3_subset["cross_down"]]
        
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals["date"],
                y=buy_signals["close"],
                mode="markers",
                name="Buy Signal",
                marker=dict(color="#00FF88", size=12, symbol="triangle-up"),
            ))
        
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals["date"],
                y=sell_signals["close"],
                mode="markers",
                name="Sell Signal",
                marker=dict(color="#FF3366", size=12, symbol="triangle-down"),
            ))
        
        fig.update_layout(
            title="Exercise 3: MA Crossover Strategy",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            template="plotly_dark",
            hovermode="x unified",
            height=450,
        )
        _plot(fig, "exercise3")
        
        st.markdown(f"**Buy signals:** {len(buy_signals)} | **Sell signals:** {len(sell_signals)}")
        
        with st.expander("💡 Reveal Analysis Tips"):
            st.markdown("""
            - **Lagging indicators:** MAs lag price, so signals come late
            - **Whipsaws:** In choppy markets, you get many false signals
            - **Trend followers:** Work well in strong trends, poorly in ranges
            - **Optimization needed:** Need to combine with volume, volatility filters
            """)

st.divider()

# ===========================
# Summary & Next Steps
# ===========================
st.markdown("### 🎯 Your Learning Journey")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🌱 Beginner Completed:**
    - ✅ Read line charts
    - ✅ Understand candlesticks
    - ✅ Interpret volume
    - ✅ Compare assets
    """)

with col2:
    st.markdown("""
    **📈 Intermediate Mastered:**
    - ✅ Navigate with zoom/pan
    - ✅ Use moving averages
    - ✅ Measure volatility
    - ✅ Recognize patterns
    """)

with col3:
    st.markdown("""
    **🎓 Advanced Skills:**
    - ✅ Understand forecasts
    - ✅ Analyze distributions
    - ✅ Evaluate model errors
    - ✅ Think probabilistically
    """)

st.divider()

instruction_block(
    "Next Steps: Apply What You've Learned",
    [
        "📊 **Practice regularly** — Use your own financial data to build visual intuition",
        "🎯 **Combine techniques** — Layer multiple visualization methods for deeper insights",
        "📈 **Stay skeptical** — Question patterns, validate with volume, check contexts",
        "🧠 **Think probabilistically** — No chart pattern is 100% reliable",
        "📚 **Keep learning** — Markets evolve, so should your analytical skills",
    ],
)

# Page navigation
page_navigation("8")
