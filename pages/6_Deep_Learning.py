from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DataScope Pro - Deep Learning", layout="wide", initial_sidebar_state="expanded")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import load_config
from src.core.logging_utils import log_event
from src.core.state import get_clean_df, get_df
from src.core.ui import sidebar_dataset_status, instruction_block, page_navigation
from src.core.styles import render_stat_card, inject_custom_css
from src.ml.forecast_transformer import (
    forecast_transformer,
    load_transformer_forecaster,
    train_simple_transformer_forecaster,
)
from src.ml.tft_optional import (
    explain_tft_requirements,
    forecast_tft,
    install_tft_dependencies,
    load_tft_forecaster,
    tft_available,
    train_tft_forecaster,
)
from src.storage.history import add_event
from src.storage.model_registry import register_model, get_model
from src.core.forecast_results import ForecastResult, estimate_confidence_intervals
from src.core.forecast_components import render_forecast_results, render_model_comparison


config = load_config()

# Apply custom CSS
inject_custom_css()

st.markdown(
    """
    <div style=\"background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;\">
        <div style=\"font-size: 24px; font-weight: 800;\">🧠 Deep Learning Forecasting</div>
        <div style=\"font-size: 15px; opacity: 0.95; margin-top: 6px;\">Train Transformer or TFT models for time-series forecasts with reusable model IDs.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
instruction_block(
    "How to use this page",
    [
        "Pick the time column and numeric target to forecast.",
        "Transformer works out of the box; TFT needs optional dependencies (the app will prompt if missing).",
        "Set horizon (future steps) and lookback (history per sample); more epochs take longer.",
        "Train, review backtest metrics, and compare forecasts against recent history.",
        "Grab the model ID to reuse on the Prediction page.",
    ],
)

st.info(
    "Forecast future values with neural models; both models log IDs you can deploy for inference.",
    icon="ℹ️",
)

# Beginner quick start and reading guide
st.success(
    "Quick start: (1) Choose time and target columns, (2) Horizon = how far ahead, Lookback = how much history, "
    "(3) Pick Transformer (always available) or TFT (advanced, needs deps), (4) Train, (5) In Results tab, pick the best model ID.",
    icon="✅",
)
st.caption("Reading metrics: Lower MAE/RMSE = better. Use the forecast chart to see if the line looks reasonable versus recent history.")

raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)

sidebar_dataset_status(raw_df, clean_df)

df = clean_df if clean_df is not None else raw_df
if df is None:
    st.info("Load data in Data Cleaning page first.")
    st.stop()
assert df is not None

# Create tabs for organization
tab1, tab2, tab3, tab4 = st.tabs(["🚀 Training", "📊 Comparison & Results", "📚 Learn", "⚙️ Advanced"])

with tab1:
    st.subheader("Time-Series Setup")
    st.caption("Configure the data source and model parameters for training")
    
    cols = list(df.columns)
    col1, col2 = st.columns(2)
    
    with col1:
        time_col = st.selectbox(
            "Time column",
            options=cols,
            help="Column with dates or time indices (must be sortable)"
        )
    with col2:
        target_col = st.selectbox(
            "Target column",
            options=cols,
            help="Column to forecast (must be numeric)"
        )
    
    st.divider()
    
    st.subheader("Data Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_stat_card("Rows", f"{df.shape[0]:,}", icon="🗂️"), unsafe_allow_html=True)
    with c2:
        st.markdown(render_stat_card("Columns", f"{df.shape[1]:,}", icon="📑"), unsafe_allow_html=True)
    with c3:
        target_valid = df[target_col].dropna().shape[0]
        st.markdown(render_stat_card("Valid samples", f"{target_valid:,}", icon="✅"), unsafe_allow_html=True)
    with c4:
        pct_missing = (1 - target_valid / len(df)) * 100
        st.markdown(render_stat_card("Missing %", f"{pct_missing:.1f}%", icon="⚠️"), unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Model Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_types = st.multiselect(
            "Models to train (for comparison)",
            options=["Transformer", "TFT"],
            default=["Transformer"],
            help="Transformer: lightweight, always available | TFT: advanced, requires optional deps"
        )
    
    if "TFT" in model_types and not tft_available():
        with col2:
            st.warning("⚠️ TFT dependencies not installed")

        with st.expander("📦 Install TFT Dependencies", expanded=True):
            st.code(explain_tft_requirements(), language="bash")
            pref = st.radio("Choose Lightning flavor", ["lightning", "pytorch-lightning"], horizontal=True)
            if st.button("Install now", type="primary"):
                with st.status("Installing TFT dependencies…", expanded=True) as s:
                    res = install_tft_dependencies(pref)
                    s.write("Attempted: " + ", ".join(res.get("tried", [])))
                    s.write("\n" + (res.get("logs", "")[-5000:]))
                    if res.get("success"):
                        s.update(label="✅ Install complete. Reloading page…", state="complete")
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()  # type: ignore[attr-defined]
                    else:
                        s.update(label="❌ Install failed. See logs above.", state="error")

        # If still unavailable after potential install, stop so the user can retry
        if not tft_available():
            st.stop()
    
    st.divider()
    
    st.subheader("Hyperparameters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        horizon = st.slider(
            "Forecast horizon",
            min_value=1,
            max_value=200,
            value=12,
            help="Number of future steps to forecast"
        )
    with col2:
        lookback = st.slider(
            "Lookback window",
            min_value=12,
            max_value=500,
            value=48,
            help="Historical context per training sample"
        )
    with col3:
        epochs = st.slider(
            "Training epochs",
            min_value=1,
            max_value=100,
            value=15,
            help="More epochs = better fit but slower training"
        )
    
    with st.expander("💡 Hyperparameter Guide", expanded=False):
        st.write("""
        **Horizon**: Number of steps into the future to predict
        - Small (1-12): Next hour/day/week forecasts
        - Medium (13-50): Monthly to quarterly forecasts
        - Large (51+): Long-term planning (use with caution)
        
        **Lookback Window**: How much historical data the model sees per sample
        - Smaller (12-24): Fast training, works for simple patterns
        - Medium (24-72): Good balance, captures seasonal effects
        - Larger (73+): Captures complex patterns but slow training
        
        **Epochs**: How many times to iterate through the training data
        - 5-10: Quick prototyping
        - 15-30: Typical training (good balance)
        - 50+: Intensive training (slow, better accuracy)
        """)
    
    st.divider()
    
    artifact_path = str(Path(config.artifacts_dir) / "forecast_transformer.pt")
    tft_artifact_path = str(Path(config.artifacts_dir) / "forecast_tft.ckpt")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            # Run training inline with live status so users see progress immediately
            try:
                if not model_types:
                    st.error("Select at least one model.")
                    st.stop()

                comparison_rows = []
                registered = []

                status_container = st.status("🚀 Training forecasting models...", expanded=True)
                with status_container:
                    if "Transformer" in model_types:
                        st.write("⏳ Training Transformer model...")
                        result = train_simple_transformer_forecaster(
                            df=df,
                            time_col=time_col,
                            target_col=target_col,
                            horizon=int(horizon),
                            lookback=int(lookback),
                            epochs=int(epochs),
                            artifact_path=artifact_path,
                        )
                        rec = register_model(
                            registry_dir=config.registry_dir,
                            artifacts_dir=config.artifacts_dir,
                            kind="forecast",
                            name="Transformer",
                            artifact_path=result.artifact_path,
                            meta=result.meta,
                        )
                        registered.append(rec)
                        comparison_rows.append({"model_id": rec.model_id, "model": "Transformer", **result.metrics})
                        add_event(config.history_db_path, "model_train", f"Trained forecasting model {rec.model_id}", json.dumps(rec.__dict__))
                        log_event(config.logging_dir, "model_train", rec.__dict__)
                        st.write("✅ Transformer model trained successfully")

                    if "TFT" in model_types:
                        st.write("⏳ Training TFT model (this may take longer)...")
                        tft_res = train_tft_forecaster(
                            df=df,
                            time_col=time_col,
                            target_col=target_col,
                            horizon=int(horizon),
                            lookback=int(lookback),
                            epochs=int(epochs),
                            artifact_path=tft_artifact_path,
                        )
                        rec = register_model(
                            registry_dir=config.registry_dir,
                            artifacts_dir=config.artifacts_dir,
                            kind="forecast",
                            name="TFT",
                            artifact_path=tft_res.artifact_path,
                            meta=tft_res.meta,
                        )
                        registered.append(rec)
                        comparison_rows.append({"model_id": rec.model_id, "model": "TFT", **tft_res.metrics})
                        add_event(config.history_db_path, "model_train", f"Trained forecasting model {rec.model_id}", json.dumps(rec.__dict__))
                        log_event(config.logging_dir, "model_train", rec.__dict__)
                        st.write("✅ TFT model trained successfully")

                status_container.update(label="✅ Training complete! See the Results tab for comparisons.", state="complete", expanded=False)
                st.success("✅ Training finished. Open the 'Comparison & Results' tab to explore metrics and plots.")

                # Store summary in session for the Results tab (avoid retraining)
                st.session_state["forecast_training_summary"] = {
                    "comparison_rows": comparison_rows,
                    "horizon": int(horizon),
                    "lookback": int(lookback),
                }
            except Exception as e:
                st.error(f"❌ Training failed: {str(e)}")
    with col2:
        st.caption("This will train and compare selected models")


with tab2:
    st.subheader("Training & Results")
    
    # If we already trained in the Training tab, just render the results from session
    if "forecast_training_summary" in st.session_state:
        try:
            comparison_rows = st.session_state["forecast_training_summary"].get("comparison_rows", [])
            horizon = st.session_state["forecast_training_summary"].get("horizon", horizon)
            lookback = st.session_state["forecast_training_summary"].get("lookback", lookback)
            registered = []  # Registered already added and stored in registry; we visualize from artifacts below
            st.success("✅ Showing results from the most recent training run.")
            
            # Results metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Trained", len(comparison_rows))
            with col2:
                st.metric("Forecast Horizon", f"{horizon} steps")
            with col3:
                st.metric("Lookback Window", f"{lookback} steps")

            st.subheader("Backtesting Comparison")
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True)

            # Add results explanation
            st.divider()
            st.markdown("### 📊 Understanding Your Results")
            
            with st.expander("💡 How to Interpret Forecast Metrics", expanded=True):
                st.markdown("""
                **What do these metrics mean?**
                
                - **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
                  - Measured in the same units as your target variable
                  - Lower is better — shows typical prediction error
                  - Example: MAE of 100 means predictions are off by ±100 on average
                
                - **RMSE (Root Mean Squared Error)**: Square root of average squared errors
                  - Also in same units as your target
                  - Lower is better — penalizes large errors more heavily than MAE
                  - More sensitive to outliers than MAE
                
                **Quick Quality Check:**
                """)
                
                if comparison_rows:
                    mae_val = comparison_rows[0].get("mae", 0)
                    rmse_val = comparison_rows[0].get("rmse", 0)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Your MAE", f"{mae_val:.2f}")
                        st.caption("Typical error magnitude")
                    with col2:
                        st.metric("Your RMSE", f"{rmse_val:.2f}")
                        st.caption("Error with outlier penalty")
                    
                    if rmse_val > mae_val * 1.5:
                        st.warning("RMSE is much larger than MAE — your data may have outliers or large occasional errors")
                    else:
                        st.success("RMSE and MAE are similar — predictions are consistent without major outliers")
                
                st.markdown("""
                **What makes a "good" score?**
                - Compare metrics to your target variable's range
                - If your target ranges from 0-1000, MAE of 50 means 5% average error
                - Lower percentages = better model performance
                - Always validate on completely unseen future data before deployment
                
                **Next Steps:**
                1. Use the best model ID for predictions on new data
                2. Check forecast plots below to visually verify predictions
                3. Monitor performance on real production data
                """)

            if comparison_rows:
                sample_keys = [k for k in comparison_rows[0].keys() if k not in {"model_id", "model"}]
                metric_key = next((k for k in sample_keys if k.lower() in {"rmse", "mae", "mape", "r2"}), None)
                if metric_key:
                    st.divider()
                    st.markdown("### 📈 Visual Performance Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_comp = px.bar(
                            comparison_rows,
                            x="model",
                            y=metric_key,
                            color="model",
                            text_auto=".3f",
                            title=f"Model Comparison ({metric_key.upper()})",
                        )
                        fig_comp.update_layout(showlegend=False)
                        st.plotly_chart(fig_comp, use_container_width=True)
                    with col2:
                        is_lower_better = metric_key.lower() in {"rmse", "mae", "mape"}
                        best = (min if is_lower_better else max)(comparison_rows, key=lambda r: r.get(metric_key, 0.0))
                        direction = "↓ lower" if is_lower_better else "↑ higher"
                        st.metric("Best model", best['model'])
                        st.metric(f"{metric_key.upper()}", f"{best.get(metric_key, 'n/a'):.3f}")
                        st.caption(f"{direction} is better")
                    
                    # Additional visualizations
                    if len(comparison_rows) > 1:
                        st.divider()
                        viz_col1, viz_col2 = st.columns(2)
                        
                        with viz_col1:
                            # Multi-metric radar chart
                            st.markdown("#### 🎯 Multi-Metric Comparison")
                            available_metrics = [k for k in comparison_rows[0].keys() if k not in {"model_id", "model"}]
                            
                            if len(available_metrics) >= 2:
                                # Normalize metrics for radar (0-1 scale)
                                radar_data = []
                                for row in comparison_rows:
                                    radar_data.append({
                                        "model": row["model"],
                                        **{m: row.get(m, 0) for m in available_metrics}
                                    })
                                
                                fig_radar = go.Figure()
                                for row in radar_data:
                                    values = [row.get(m, 0) for m in available_metrics]
                                    fig_radar.add_trace(go.Scatterpolar(
                                        r=values,
                                        theta=[m.upper() for m in available_metrics],
                                        fill='toself',
                                        name=row['model']
                                    ))
                                
                                fig_radar.update_layout(
                                    polar=dict(radialaxis=dict(visible=True)),
                                    showlegend=True,
                                    height=350,
                                    title="All Metrics at a Glance"
                                )
                                st.plotly_chart(fig_radar, use_container_width=True)
                            else:
                                st.info("Multiple metrics needed for radar chart")
                        
                        with viz_col2:
                            # Performance ranking table with visual bars
                            st.markdown("#### 🏆 Performance Ranking")
                            
                            rank_df = pd.DataFrame(comparison_rows)
                            rank_df = rank_df.sort_values(by=metric_key, ascending=is_lower_better)
                            rank_df['Rank'] = range(1, len(rank_df) + 1)
                            rank_df['Medal'] = ['🥇', '🥈', '🥉'] + [''] * (len(rank_df) - 3)
                            
                            display_cols = ['Medal', 'Rank', 'model', metric_key]
                            st.dataframe(
                                rank_df[display_cols].style.background_gradient(
                                    subset=[metric_key],
                                    cmap='RdYlGn_r' if is_lower_better else 'RdYlGn'
                                ),
                                use_container_width=True,
                                height=350
                            )
                    
                    # Error distribution visualization
                    st.divider()
                    st.markdown("#### 📊 Error Distribution Comparison")
                    
                    # Create side-by-side metric comparison
                    metrics_comparison = []
                    for row in comparison_rows:
                        for metric in ['mae', 'rmse']:
                            if metric in row:
                                metrics_comparison.append({
                                    'Model': row['model'],
                                    'Metric': metric.upper(),
                                    'Value': row[metric]
                                })
                    
                    if metrics_comparison:
                        fig_grouped = px.bar(
                            pd.DataFrame(metrics_comparison),
                            x='Model',
                            y='Value',
                            color='Metric',
                            barmode='group',
                            title='MAE vs RMSE by Model',
                            text_auto='.2f'
                        )
                        fig_grouped.update_layout(height=400)
                        st.plotly_chart(fig_grouped, use_container_width=True)
                        
                        st.caption("💡 If RMSE bars are much taller than MAE, the model struggles with occasional large errors (outliers)")

            st.info("Tip: Train again from the Training tab to refresh these results.")

            # --- Unified Forecast Visualization (Standardized) ---
            st.divider()
            st.subheader("🎯 Unified Forecast Visualization")
            st.caption("Consistent charts, confidence intervals, AI summary, and time aggregations")

            try:
                # Allow user to select which trained model to visualize
                model_options = [f"{row['model']} | {row['model_id']}" for row in comparison_rows]
                selected = st.selectbox("Select a model to visualize", options=model_options)
                selected_id = selected.split("|")[-1].strip()

                rec = get_model(config.registry_dir, selected_id)
                if not rec:
                    st.warning("Selected model record not found in registry")
                else:
                    meta = rec.meta or {}
                    target_col_vis = str(meta.get("target_col"))
                    time_col_vis = str(meta.get("time_col"))
                    horizon_vis = int(meta.get("horizon", st.session_state.get("forecast_training_summary", {}).get("horizon", 12)))
                    lookback_vis = int(meta.get("lookback", st.session_state.get("forecast_training_summary", {}).get("lookback", 48)))
                    model_type_raw = str(meta.get("type", "transformer_forecaster"))

                    # Prepare history from full dataframe
                    d_hist = df[[time_col_vis, target_col_vis]].copy().dropna()
                    if not pd.api.types.is_datetime64_any_dtype(d_hist[time_col_vis]):
                        d_hist[time_col_vis] = pd.to_datetime(d_hist[time_col_vis], errors="coerce")
                    d_hist = d_hist[d_hist[time_col_vis].notna()].sort_values(time_col_vis)
                    d_hist[target_col_vis] = pd.to_numeric(d_hist[target_col_vis], errors="coerce")
                    d_hist = d_hist[d_hist[target_col_vis].notna()]

                    if d_hist.empty:
                        st.error("❌ Not enough valid time/target data to visualize.")
                    else:
                        last_time = d_hist[time_col_vis].iloc[-1]
                        diffs = d_hist[time_col_vis].diff().dropna()
                        delta = diffs.median() if not diffs.empty else None
                        has_delta = isinstance(delta, pd.Timedelta) and delta > pd.Timedelta(0)

                        # Forecast via appropriate loader
                        if model_type_raw == "tft":
                            model = load_tft_forecaster(rec.artifact_path)
                            forecast_df, preds = forecast_tft(
                                model,
                                history_df=df,
                                time_col=time_col_vis,
                                target_col=target_col_vis,
                                horizon=horizon_vis,
                                lookback=lookback_vis,
                            )
                            if "forecast_time" in forecast_df.columns and forecast_df["forecast_time"].notna().any():
                                forecast_dates = forecast_df["forecast_time"].tolist()
                            else:
                                forecast_dates = (
                                    [last_time + (i + 1) * delta for i in range(horizon_vis)]
                                    if has_delta
                                    else list(range(1, len(forecast_df) + 1))
                                )
                            model_type_norm = "tft"
                        else:
                            model, ckpt = load_transformer_forecaster(rec.artifact_path)
                            series = d_hist[target_col_vis].to_numpy(dtype=np.float32)
                            preds = forecast_transformer(model, ckpt, history=series)
                            forecast_dates = (
                                [last_time + (i + 1) * delta for i in range(len(preds))]
                                if has_delta
                                else list(range(1, len(preds) + 1))
                            )
                            model_type_norm = "transformer"

                        # Historical tail for context
                        hist_tail = d_hist.tail(min(lookback_vis, len(d_hist)))
                        historical_values = hist_tail[target_col_vis].to_numpy()
                        historical_dates = hist_tail[time_col_vis].tolist()

                        # Confidence intervals
                        lower, upper = estimate_confidence_intervals(np.asarray(preds), model_type=model_type_norm)

                        # Build ForecastResult
                        fr = ForecastResult(
                            model_name=rec.name,
                            model_type=model_type_norm,
                            target_column=target_col_vis,
                            forecast_values=np.asarray(preds),
                            forecast_dates=forecast_dates,
                            historical_values=historical_values,
                            historical_dates=historical_dates,
                            lower_bound=lower,
                            upper_bound=upper,
                            confidence_level=0.95,
                            train_metrics=None,
                            test_metrics=(meta.get("metrics") or {}),
                            metadata={"model_id": rec.model_id, "horizon": horizon_vis, "lookback": lookback_vis},
                        )

                        render_forecast_results(fr)
            except Exception as e:
                st.warning(f"Unified forecast visualization failed: {e}")
        except Exception as e:
            st.error(f"❌ Failed to render results: {str(e)}")
    else:
        st.info("👈 Click **Train Models** in the Training tab to begin")

    
    # The original code path below performs training inside the Results tab when explicitly requested.
    # Keeping it for backwards compatibility if the session flag is set by other flows.
    if st.session_state.get("training_requested"):
        try:
            if not model_types:
                st.error("Select at least one model.")
                st.stop()

            comparison_rows = []
            registered = []

            # Status container for real-time updates
            status_container = st.status("🚀 Training forecasting models...", expanded=True)
            
            with status_container:
                if "Transformer" in model_types:
                    st.write("⏳ Training Transformer model...")
                    result = train_simple_transformer_forecaster(
                        df=df,
                        time_col=time_col,
                        target_col=target_col,
                        horizon=int(horizon),
                        lookback=int(lookback),
                        epochs=int(epochs),
                        artifact_path=artifact_path,
                    )
                    rec = register_model(
                        registry_dir=config.registry_dir,
                        artifacts_dir=config.artifacts_dir,
                        kind="forecast",
                        name="Transformer",
                        artifact_path=result.artifact_path,
                        meta=result.meta,
                    )
                    registered.append(rec)
                    comparison_rows.append({"model_id": rec.model_id, "model": "Transformer", **result.metrics})
                    add_event(config.history_db_path, "model_train", f"Trained forecasting model {rec.model_id}", json.dumps(rec.__dict__))
                    log_event(config.logging_dir, "model_train", rec.__dict__)
                    st.write("✅ Transformer model trained successfully")

                if "TFT" in model_types:
                    st.write("⏳ Training TFT model (this may take longer)...")
                    tft_res = train_tft_forecaster(
                        df=df,
                        time_col=time_col,
                        target_col=target_col,
                        horizon=int(horizon),
                        lookback=int(lookback),
                        epochs=int(epochs),
                        artifact_path=tft_artifact_path,
                    )
                    rec = register_model(
                        registry_dir=config.registry_dir,
                        artifacts_dir=config.artifacts_dir,
                        kind="forecast",
                        name="TFT",
                        artifact_path=tft_res.artifact_path,
                        meta=tft_res.meta,
                    )
                    registered.append(rec)
                    comparison_rows.append({"model_id": rec.model_id, "model": "TFT", **tft_res.metrics})
                    add_event(config.history_db_path, "model_train", f"Trained forecasting model {rec.model_id}", json.dumps(rec.__dict__))
                    log_event(config.logging_dir, "model_train", rec.__dict__)
                    st.write("✅ TFT model trained successfully")
            
            # Update status to complete
            status_container.update(label="✅ Training complete!", state="complete", expanded=False)

            st.success("✅ All models trained successfully!")
            
            # Results metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Trained", len(comparison_rows))
            with col2:
                st.metric("Forecast Horizon", f"{horizon} steps")
            with col3:
                st.metric("Lookback Window", f"{lookback} steps")
            
            st.subheader("Backtesting Comparison")
            comparison_df = pd.DataFrame(comparison_rows)
            st.dataframe(comparison_df, use_container_width=True)

            if comparison_rows:
                sample_keys = [k for k in comparison_rows[0].keys() if k not in {"model_id", "model"}]
                metric_key = next((k for k in sample_keys if k.lower() in {"rmse", "mae", "mape", "r2"}), None)
                
                if metric_key:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_comp = px.bar(
                            comparison_rows,
                            x="model",
                            y=metric_key,
                            color="model",
                            text_auto=".3f",
                            title=f"Model Comparison ({metric_key})",
                        )
                        fig_comp.update_layout(showlegend=False)
                        st.plotly_chart(fig_comp, use_container_width=True)
                    
                    with col2:
                        is_lower_better = metric_key.lower() in {"rmse", "mae", "mape"}
                        best = (min if is_lower_better else max)(comparison_rows, key=lambda r: r.get(metric_key, 0.0))
                        direction = "↓ lower" if is_lower_better else "↑ higher"
                        
                        st.metric("Best model", best['model'])
                        st.metric(f"{metric_key}", f"{best.get(metric_key, 'n/a'):.3f}")
                        st.caption(f"{direction} is better")

            st.divider()
            
            st.subheader("Registered Models")
            for rec in registered:
                with st.expander(f"📦 {rec.model_id} ({rec.name})"):
                    st.json(rec.meta if rec.meta else {"info": "No metadata"})

            st.divider()

            # --- Forecast visualization (latest window) ---
            st.subheader("Forecast Visualization")
            st.caption("Actual data (blue) vs model forecasts (orange/red)")

            d_plot = df[[time_col, target_col]].copy().dropna()
            if not pd.api.types.is_datetime64_any_dtype(d_plot[time_col]):
                d_plot[time_col] = pd.to_datetime(d_plot[time_col], errors="coerce")
            d_plot = d_plot[d_plot[time_col].notna()].sort_values(time_col)
            d_plot[target_col] = pd.to_numeric(d_plot[target_col], errors="coerce")
            d_plot = d_plot[d_plot[target_col].notna()]

            if d_plot.empty:
                st.info("Not enough valid time/target data to plot.")
            else:
                # Determine a best-effort time delta for the x-axis
                diffs = d_plot[time_col].diff().dropna()
                delta = diffs.median() if not diffs.empty else None
                has_delta = isinstance(delta, pd.Timedelta) and delta > pd.Timedelta(0)
                delta_td = delta if has_delta else None

                lookback_plot = min(int(lookback), max(10, len(d_plot)))
                actual_tail = d_plot.tail(lookback_plot)
                last_time = actual_tail[time_col].iloc[-1]

                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=actual_tail[time_col],
                        y=actual_tail[target_col],
                        mode="lines",
                        name="Actual",
                        line=dict(color="#1f77b4", width=2),
                    )
                )

                colors = ["#ff7f0e", "#d62728"]
                for idx, rec in enumerate(registered):
                    model_kind = rec.name
                    meta_type = str((rec.meta or {}).get("type", "transformer_forecaster"))
                    try:
                        if meta_type == "tft":
                            model = load_tft_forecaster(rec.artifact_path)
                            forecast_df, _ = forecast_tft(
                                model,
                                history_df=df,
                                time_col=time_col,
                                target_col=target_col,
                                horizon=int(horizon),
                                lookback=int(lookback),
                            )
                            if "forecast_time" in forecast_df.columns and forecast_df["forecast_time"].notna().any():
                                fx = forecast_df["forecast_time"]
                            else:
                                fx = (
                                    [last_time + (i + 1) * delta_td for i in range(int(horizon))]
                                    if delta_td is not None
                                    else list(range(1, len(forecast_df) + 1))
                                )
                            fy = forecast_df["forecast"]
                            fig.add_trace(go.Scatter(
                                x=fx, y=fy, mode="lines", 
                                name=f"Forecast ({model_kind})",
                                line=dict(color=colors[idx], dash="dash")
                            ))
                        else:
                            model, ckpt = load_transformer_forecaster(rec.artifact_path)
                            series = d_plot[target_col].to_numpy(dtype="float32")
                            pred = forecast_transformer(model, ckpt, history=series)
                            fx = (
                                [last_time + (i + 1) * delta_td for i in range(int(horizon))]
                                if delta_td is not None
                                else list(range(1, len(pred) + 1))
                            )
                            fig.add_trace(go.Scatter(
                                x=fx, y=pred, mode="lines", 
                                name=f"Forecast ({model_kind})",
                                line=dict(color=colors[idx], dash="dash")
                            ))
                    except Exception as e:
                        st.warning(f"Plot for {rec.model_id} failed: {e}")

                fig.update_layout(
                    title=f"Actual vs Forecast ({target_col})",
                    xaxis_title=time_col,
                    yaxis_title=target_col,
                    hovermode="x unified",
                    height=500,
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")


with tab3:
    st.subheader("Time-Series Forecasting Fundamentals")
    st.caption("Learn about time-series forecasting concepts and model selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("📚 What is Time-Series Forecasting?", expanded=True):
            st.write("""
            Time-series forecasting predicts future values based on historical patterns.
            
            **Key Characteristics:**
            - **Temporal Dependency**: Values depend on their past values
            - **Trend**: Long-term upward/downward movement
            - **Seasonality**: Repeating patterns (daily, weekly, yearly)
            - **Noise**: Random fluctuations
            
            **Common Use Cases:**
            - Stock price prediction
            - Weather forecasting
            - Energy demand prediction
            - Sales forecasting
            - IoT sensor readings
            """)
    
    with col2:
        with st.expander("🤖 Transformer vs TFT Models", expanded=True):
            st.write("""
            **Transformer Forecaster:**
            - ✅ Lightweight, always available
            - ✅ Good for simple to moderate patterns
            - ⚠️ May struggle with complex interactions
            
            **TFT (Temporal Fusion Transformer):**
            - ✅ State-of-the-art architecture
            - ✅ Handles multiple features and interactions
            - ✅ Interpretable attention weights
            - ⚠️ Requires optional dependencies
            - ⚠️ Slower training
            """)
    
    with st.expander("⏰ Best Practices for Time-Series", expanded=False):
        st.write("""
        1. **Data Preparation**
           - Ensure time column is sorted
           - Handle missing values appropriately
           - Remove or handle outliers
           
        2. **Feature Engineering**
           - Extract time features (hour, day, month)
           - Create lag features (previous values)
           - Include external features when relevant
           
        3. **Model Training**
           - Use appropriate lookback/horizon ratio
           - Split data properly (no leakage!)
           - Monitor overfitting with validation
           
        4. **Evaluation**
           - Use time-based split (not random)
           - Metric choice: RMSE for regression, MAPE for percentages
           - Compare multiple baselines
           
        5. **Deployment**
           - Monitor prediction accuracy over time
           - Retrain periodically with new data
           - Set up alerts for anomalies
        """)
    
    with st.expander("📊 Hyperparameter Tuning Tips", expanded=False):
        st.write("""
        **Horizon Selection:**
        - Short (1-12): Easier to predict, more frequent updates
        - Medium (13-50): Captures trend, seasonal patterns harder
        - Long (51+): Risk of diverging predictions
        
        **Lookback Selection:**
        - Rule of thumb: 2-4x the horizon
        - Minimum: At least 1 full seasonal cycle
        - Maximum: Available memory and speed
        
        **Epoch Selection:**
        - Start with 15-20 epochs
        - Watch for overfitting (validation loss plateaus)
        - Use early stopping if available
        
        **Practical Examples:**
        - Hourly data, 24h forecast: horizon=24, lookback=72-168
        - Daily data, monthly forecast: horizon=30, lookback=90-180
        - Monthly data, quarterly forecast: horizon=3, lookback=12-24
        """)
    
    with st.expander("⚠️ Common Pitfalls & Solutions", expanded=False):
        st.write("""
        **Problem**: High loss, poor predictions
        - Solution: Increase lookback window, more epochs, check data quality
        
        **Problem**: Training is very slow
        - Solution: Reduce lookback/horizon, fewer epochs, simpler model (Transformer)
        
        **Problem**: Great training metrics, bad real-world predictions
        - Solution: Check for data distribution shift, retrain with recent data
        
        **Problem**: Exploding or NaN predictions
        - Solution: Normalize/standardize data, reduce learning rate, check for outliers
        """)


with tab4:
    st.subheader("Advanced Configuration")
    
    with st.expander("🔧 Model Architecture", expanded=True):
        st.write("""
        **Transformer Architecture:**
        - Lightweight PyTorch implementation
        - Suitable for quick prototyping
        - Good for univariate and simple multivariate series
        - Faster training, lower memory footprint
        
        **TFT (Temporal Fusion Transformer):**
        - Advanced attention-based architecture
        - Better handling of multiple features
        - Provides feature importance interpretations
        - Requires PyTorch Lightning and additional dependencies
        """)
    
    with st.expander("🛠️ Troubleshooting", expanded=False):
        st.write("""
        **If TFT installation fails:**
        1. Check PyTorch is compatible with your system (CPU/GPU)
        2. Use: `pip install torch torchvision torchaudio`
        3. Then install: `pip install pytorch-forecasting pytorch-lightning`
        
        **For GPU acceleration (optional):**
        - CUDA support: `pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118`
        
        **Memory issues during training:**
        - Reduce batch size (automatic in current setup)
        - Use smaller lookback/horizon
        - Switch to Transformer if TFT is slow
        """)
    
    with st.expander("📝 Model Details (Technical)", expanded=False):
        st.json({
            "transformer": {
                "framework": "PyTorch",
                "input": "Time series (univariate or multivariate)",
                "output": "Forecast values for horizon steps",
                "optimized_for": "Quick prototyping, simplicity"
            },
            "tft": {
                "framework": "PyTorch Lightning",
                "input": "Time series + static/known future features",
                "output": "Forecast + attention visualization",
                "optimized_for": "Complex patterns, interpretability"
            }
        })

# Page navigation
page_navigation("6")
