from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
import subprocess

import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="DataScope Pro - Tabular ML", layout="wide", initial_sidebar_state="expanded")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import load_config
from src.core.logging_utils import log_event
from src.core.state import get_clean_df, get_df
from src.core.ui import app_header, sidebar_dataset_status, instruction_block, page_navigation
from src.core.standardized_ui import (
    standard_section_header,
    beginner_tip,
    concept_explainer,
    common_mistakes_panel,
)
from src.core.styles import inject_custom_css
from src.core.premium_styles import inject_premium_css, get_plotly_theme
from src.core.modern_components import smart_metric_card, comparison_table, success_message_with_next_steps, enhanced_chart
from src.ml.tabular import train_tabular
from src.storage.history import add_event
from src.storage.model_registry import register_model
from src.storage.model_registry import get_model, list_models
from src.core.forecast_results import ForecastResult, estimate_confidence_intervals
from src.core.forecast_components import render_model_comparison


def _model_available(name: str) -> bool:
    n = name.lower()
    if n == "xgboost":
        return importlib.util.find_spec("xgboost") is not None
    if n == "lightgbm":
        return importlib.util.find_spec("lightgbm") is not None
    return True


def _pip_install(packages: list[str]) -> tuple[bool, str]:
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True)
        logs = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return proc.returncode == 0, logs
    except Exception as e:  # pragma: no cover
        return False, f"Exception: {e}"


def _get_model_description(model_name: str) -> str:
    """Get educational description of each model."""
    descriptions = {
        "RandomForest": "🌲 Ensemble of decision trees. Great for handling non-linear patterns and feature interactions.",
        "XGBoost": "⚡ Optimized gradient boosting. Very powerful but can overfit; good for competitions.",
        "LightGBM": "💡 Fast gradient boosting. Handles large datasets efficiently with less memory.",
        "GradientBoosting": "📈 Sequential tree boosting. Balanced performance and interpretability.",
    }
    return descriptions.get(model_name, "Machine Learning Model")


def _get_metric_explanation(metric: str) -> str:
    """Get detailed explanation of each performance metric."""
    explanations = {
        "accuracy": "📊 **Accuracy**: Percentage of correct predictions. Use for balanced datasets.",
        "precision": "🎯 **Precision**: Of predicted positives, how many were actually positive? (Low false positives)",
        "recall": "🔍 **Recall**: Of actual positives, how many did we find? (Low false negatives)",
        "f1": "⚖️ **F1-Score**: Harmonic mean of precision & recall. Best for imbalanced data.",
        "f1_macro": "⚖️ **F1-Macro**: Average F1 score across all classes.",
        "r2": "📈 **R² Score**: Proportion of variance explained (0-1, higher is better)",
        "rmse": "📉 **RMSE**: Root Mean Squared Error. Penalizes large errors heavily.",
        "mae": "📊 **MAE**: Mean Absolute Error. Average absolute difference from actual values.",
        "mse": "🔢 **MSE**: Mean Squared Error. Average squared differences.",
    }
    return explanations.get(metric, f"📊 {metric}: Performance metric")


config = load_config()

# Apply custom CSS
inject_custom_css()

app_header(
    config,
    page_title="Tabular Machine Learning",
    subtitle="Train and compare models for classification or regression on row-and-column data",
    icon="🤖"
)

instruction_block(
    "How to use this page",
    [
        "Select the target column and choose one or more algorithms to compare.",
        "Leave Task on Auto unless you must force regression or classification.",
        "Pick a validation split that matches your data size; smaller data needs a smaller test size.",
        "Train runs save a model ID you can reuse on the Prediction page.",
        "Check the leaderboard to pick the best run, then continue to inference.",
    ],
)

st.info(
    "Train, compare, and save tabular models quickly—each run is logged and reusable for predictions.",
    icon="ℹ️",
)

# Beginner quick start and interpretation tips
st.success(
    "Quick start: (1) Pick your target column, (2) Leave Task on Auto unless you know better, (3) Select 1-2 models, "
    "(4) Click Train, (5) Use the leaderboard to pick the winner and note its Model ID.",
    icon="✅",
)
st.caption("Tip: For classification, higher Accuracy/F1 is better. For regression, higher R² and lower RMSE/MAE are better.")

# Add tabs for different sections
tab1, tab2, tab3 = st.tabs(["🚀 Training", "📚 Learn", "⚙️ Advanced"])

with tab3:
    st.subheader("Advanced Configuration")
    st.info("Configure advanced settings before training your models")
    
    adv_col1, adv_col2 = st.columns(2)
    with adv_col1:
        st.markdown("**Data Configuration**")
        random_seed = st.number_input("Random Seed (for reproducibility)", value=42, min_value=0)
    with adv_col2:
        st.markdown("**Model Configuration**")
        n_jobs = st.number_input("Parallel Jobs (-1 = all cores)", value=-1, min_value=-1, max_value=16)

with tab2:
    st.subheader("📖 Machine Learning Fundamentals")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Regression vs Classification
        
        **Classification (Discrete Output)**
        - Predicts categories: Yes/No, Cat/Dog, Spam/Ham
        - Metrics: Accuracy, Precision, Recall, F1
        - Example: Predict if email is spam
        
        **Regression (Continuous Output)**
        - Predicts numbers: Price, Temperature, Sales
        - Metrics: R², RMSE, MAE
        - Example: Predict house prices
        """)
    
    with col2:
        st.markdown("""
        ### 📊 Understanding Metrics
        
        **Key Performance Indicators**
        - **Train/Test Split**: How data is divided (typically 80/20)
        - **Overfitting**: Model memorizes training data (bad!)
        - **Underfitting**: Model too simple to learn patterns (bad!)
        - **Validation**: Uses unseen data to test model
        """)
    
    st.divider()
    
    st.markdown("### 🔬 Model Selection Guide")
    selection_df = pd.DataFrame({
        "Problem Type": ["Binary Classification", "Multi-class Classification", "Regression", "Large Dataset", "Imbalanced Data"],
        "Best Models": ["RandomForest, XGBoost", "LightGBM, GradientBoosting", "XGBoost, GradientBoosting", "LightGBM", "XGBoost w/ weights"],
        "Key Consideration": ["Simple & interpretable", "Handle multi-class well", "Minimize prediction error", "Speed & memory", "Weighted loss function"]
    })
    st.dataframe(selection_df, width="stretch")
    
    st.divider()
    
    st.markdown("### 📈 Why Ensemble Methods?")
    st.markdown("""
    All models on this page are **ensemble methods**, which combine multiple models:
    
    1. **Bagging (Random Forest)**: Train models in parallel, average predictions
    2. **Boosting (XGBoost, LightGBM)**: Train sequentially, each fixes previous mistakes
    3. **Stacking**: Combine multiple models' predictions as input to meta-model
    
    **Advantage**: More robust, fewer errors, better generalization
    """)

with tab1:
    raw_df = get_df(st.session_state)
    clean_df = get_clean_df(st.session_state)

    sidebar_dataset_status(raw_df, clean_df)

    df = clean_df if clean_df is not None else raw_df
    if df is None:
        st.info("Load data in Data Cleaning page first.")
        st.stop()
    assert df is not None

    # If a previous training summary exists, render it upfront for convenience
    prev_summary = st.session_state.get("tabular_training_summary")
    if isinstance(prev_summary, dict) and prev_summary.get("comparison_rows"):
        comparison_rows_cached = prev_summary.get("comparison_rows", [])
        metric_key_cached = prev_summary.get("metric_key")
        task_cached = prev_summary.get("task", "Unknown")
        target_cached = prev_summary.get("target_col", "—")

        if comparison_rows_cached:
            if not metric_key_cached:
                metric_key_cached = "accuracy" if task_cached == "classification" else "r2"
                if metric_key_cached not in comparison_rows_cached[0]:
                    metric_key_cached = next((k for k in ("f1_macro", "rmse", "mae") if k in comparison_rows_cached[0]), None)

            st.markdown("### 🔁 Last Training Results (from this session)")

            colm1, colm2, colm3 = st.columns(3)
            with colm1:
                smart_metric_card(
                    "Models Trained",
                    len(comparison_rows_cached),
                    help_text="Previously trained in this session",
                    interpretation="Cached results available"
                )
            with colm2:
                smart_metric_card(
                    "Target Column",
                    target_cached,
                    help_text="The variable predicted",
                    interpretation=f"Predicting: {target_cached}"
                )
            with colm3:
                smart_metric_card(
                    "Task Type",
                    task_cached.title() if isinstance(task_cached, str) else "—",
                    help_text="Classification or Regression",
                    interpretation="From previous run"
                )

            if metric_key_cached:
                comparison_table(
                    comparison_rows_cached,
                    highlight_best=metric_key_cached,
                    ascending=False if metric_key_cached in ["accuracy", "precision", "recall", "f1", "f1_macro", "r2"] else True
                )

                best_cached = max(comparison_rows_cached, key=lambda r: r.get(metric_key_cached, 0.0))
                best_score_cached = best_cached.get(metric_key_cached, "n/a")
                st.divider()
                with st.container(border=True):
                    st.markdown(f"### 🏆 Top Performer (cached): **{best_cached['model']}**")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.metric(f"{metric_key_cached.upper()} Score", f"{best_score_cached:.4f}" if isinstance(best_score_cached, (int, float)) else best_score_cached)
                    with c2:
                        st.metric("Rank", "🥇 1st Place")
                    with c3:
                        st.markdown(f"**Model ID:** `{best_cached['model_id']}`")
                cols_btn = st.columns([3, 1])
                with cols_btn[0]:
                    st.info("Showing results from the last training run in this session. Click 'Train model' below to run again.", icon="ℹ️")
                with cols_btn[1]:
                    if st.button("Clear cache", key="clear_tabular_cache"):
                        st.session_state.pop("tabular_training_summary", None)
                        st.toast("Cleared cached results.", icon="🧹")
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()  # type: ignore[attr-defined]

    st.subheader("Training")

    # Create columns for better layout
    col1, col2 = st.columns(2)

    with col1:
        target_col = st.selectbox("Target column", options=list(df.columns))
        st.caption("The variable you want to predict")

    with col2:
        task_override = st.selectbox("Task", options=["auto", "regression", "classification"], index=0)
        st.caption("Let the app decide or force a specific type")

    # Model selection with descriptions
    st.subheader("Algorithm Selection")
    st.caption("Click on cards below to learn about each algorithm")

    col1, col2, col3, col4 = st.columns(4)
    models = ["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"]
    available_status = {m: _model_available(m) for m in models}

    with col1:
        st.info(f"🌲 **RandomForest**\n{_get_model_description('RandomForest')}", icon="ℹ️")

    with col2:
        status = "✅" if available_status["XGBoost"] else "⚠️ (Optional)"
        st.warning(f"{status} **XGBoost**\n{_get_model_description('XGBoost')}", icon="⚡")

    with col3:
        status = "✅" if available_status["LightGBM"] else "⚠️ (Optional)"
        st.warning(f"{status} **LightGBM**\n{_get_model_description('LightGBM')}", icon="💡")

    with col4:
        st.info(f"📈 **GradientBoosting**\n{_get_model_description('GradientBoosting')}", icon="ℹ️")

    model_names = st.multiselect(
        "Models to train (for comparison)",
        options=["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"],
        default=["RandomForest"],
        help="Select multiple models to compare their performance on your data"
    )

    # Small availability badges inline
    with st.container():
        av = available_status
        badges = []
        badges.append(("RandomForest", "✅ Available"))
        badges.append(("GradientBoosting", "✅ Available"))
        badges.append(("XGBoost", "✅ Installed" if av["XGBoost"] else "🧩 Optional — not installed"))
        badges.append(("LightGBM", "✅ Installed" if av["LightGBM"] else "🧩 Optional — not installed"))
        left, right = st.columns([1, 3])
        with left:
            st.caption("Availability")
        with right:
            st.caption(
                " | ".join([f"**{name}**: {label}" for name, label in badges])
            )

    # Offer one-click installers for optional deps if selected but missing
    missing_selected = [m for m in model_names if m in ("XGBoost", "LightGBM") and not available_status.get(m, True)]
    if missing_selected:
        st.warning("⚠️ Optional dependencies missing: " + ", ".join(missing_selected))
        with st.expander("📦 Install Optional Dependencies", expanded=True):
            pip_map = {"XGBoost": "xgboost", "LightGBM": "lightgbm"}
            pkgs = [pip_map[m] for m in missing_selected]
            st.code("pip install --upgrade " + " ".join(pkgs), language="bash")
            if st.button("Install now", key="install_tabular_optional", type="primary"):
                with st.status("Installing selected dependencies…", expanded=True) as s:
                    ok, logs = _pip_install(pkgs)
                    s.write(logs[-5000:])
                    if ok:
                        s.update(label="✅ Install complete. Reloading…", state="complete")
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()  # type: ignore[attr-defined]
                    else:
                        s.update(label="❌ Install failed. See logs above.", state="error")

    # Also show a collapsed installer if optional deps are missing even when not selected
    missing_optional = [m for m in ("XGBoost", "LightGBM") if not available_status.get(m, True)]
    if missing_optional and not missing_selected:
        with st.expander("📦 Install Optional Dependencies (XGBoost/LightGBM)", expanded=False):
            pip_map = {"XGBoost": "xgboost", "LightGBM": "lightgbm"}
            pkgs = [pip_map[m] for m in missing_optional]
            st.info("Install to enable additional algorithms: " + ", ".join(missing_optional))
            st.code("pip install --upgrade " + " ".join(pkgs), language="bash")
            if st.button("Install now", key="install_tabular_optional_global", type="primary"):
                with st.status("Installing optional dependencies…", expanded=True) as s:
                    ok, logs = _pip_install(pkgs)
                    s.write(logs[-5000:])
                    if ok:
                        s.update(label="✅ Install complete. Reloading…", state="complete")
                        try:
                            st.rerun()
                        except Exception:
                            st.experimental_rerun()  # type: ignore[attr-defined]
                    else:
                        s.update(label="❌ Install failed. See logs above.", state="error")

    # Advanced options in expandable section
    with st.expander("⚙️ Advanced Settings", expanded=False):
        test_size = st.slider(
            "Test size (validation split)",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Larger test size = more validation data but less training data"
        )
        st.caption(f"📊 Train size: {1-test_size:.1%} | Test size: {test_size:.1%}")
    
    # Default test size if expander not used
    try:
        test_size
    except NameError:
        test_size = 0.2

    artifact_path = str(Path(config.artifacts_dir) / "tabular_model.joblib")

    if st.button("Train model", type="primary", width="stretch"):
        try:
            if not model_names:
                st.error("Select at least one model.")
                st.stop()

            jobs = []
            fallback_notes = []
            for requested in model_names:
                if _model_available(requested):
                    jobs.append({"display": requested, "actual": requested, "note": None})
                else:
                    fallback = "GradientBoosting" if requested.lower() in {"xgboost", "lightgbm"} else "RandomForest"
                    note = f"{requested} not installed; used {fallback} instead."
                    fallback_notes.append(note)
                    jobs.append({"display": f"{requested} (fallback: {fallback})", "actual": fallback, "note": note})

            if fallback_notes:
                st.warning("⚠️ Using fallbacks for unavailable models: " + " | ".join(fallback_notes))
            if not jobs:
                st.error("❌ No usable models selected. Install optional deps or pick a built-in model.")
                st.stop()

            comparison_rows = []
            registered = []
            stratify_warnings = []

            with st.status("⏳ Training tabular models...", expanded=True) as status:
                status.write("Preparing training jobs and configuration…")
                for job in jobs:
                    model_name = job["actual"]
                    display_name = job["display"]
                    per_model_artifact = str(Path(config.artifacts_dir) / f"tabular_{model_name.lower()}.joblib")

                    status.write(f"🚀 Training {display_name}…")
                    result = train_tabular(
                        df=df,
                        target_col=target_col,
                        model_name=model_name,
                        test_size=float(test_size),
                        explicit_task=None if task_override == "auto" else task_override,  # type: ignore[arg-type]
                        artifact_path=per_model_artifact,
                    )
                    status.write(f"✅ {display_name} trained. Registering model…")

                    # Check if stratification was skipped
                    if result.task == "classification" and not result.meta.get("stratified", True):
                        stratify_warnings.append(display_name)

                    rec = register_model(
                        registry_dir=config.registry_dir,
                        artifacts_dir=config.artifacts_dir,
                        kind="tabular",
                        name=display_name,
                        artifact_path=result.artifact_path,
                        meta=result.meta,
                    )
                    registered.append(rec)

                    result.meta["requested_model"] = display_name
                    result.meta["actual_model"] = model_name
                    result.meta["used_fallback"] = job["note"] is not None

                    row = {"model_id": rec.model_id, "model": display_name, "task": result.task, **result.metrics}
                    comparison_rows.append(row)

                    add_event(
                        config.history_db_path,
                        "model_train",
                        f"Trained tabular model {rec.model_id}",
                        json.dumps(rec.__dict__),
                    )
                    log_event(config.logging_dir, "model_train", rec.__dict__)

                status.update(label="✅ Training complete! Rendering results…", state="complete")

            # Show warning if stratification was skipped
            if stratify_warnings:
                st.warning(
                    f"Stratified train/test split was skipped for: {', '.join(stratify_warnings)}. "
                    f"Some classes have too few samples (minimum 2 required). Using random split instead.",
                    icon="⚠️"
                )

            st.toast("Training finished! Scroll to see results.", icon="✅")

            # Persist compact summary in the session for later reference
            try:
                if comparison_rows:
                    detected_task = comparison_rows[0].get("task", "Unknown")
                    # Prefer common keys for highlight metric
                    metric_key = "accuracy" if detected_task == "classification" else "r2"
                    if metric_key not in comparison_rows[0]:
                        metric_key = next((k for k in ("f1_macro", "rmse", "mae") if k in comparison_rows[0]), None)
                    st.session_state["tabular_training_summary"] = {
                        "comparison_rows": comparison_rows,
                        "target_col": target_col,
                        "test_size": float(test_size),
                        "task": detected_task,
                        "selected_models": [j["display"] for j in jobs],
                        "metric_key": metric_key,
                    }
            except Exception:
                # Non-fatal: ignore session persistence issues
                pass
            
            # Add AI interpretation helper
            explanations = create_ai_explanation_sections()
            task_type = comparison_rows[0].get("task", "Unknown") if comparison_rows else "Unknown"
            
            if task_type == "classification":
                ai_interpretation_box(
                    "Understanding Classification Results",
                    """
                    **Classification** predicts categories (classes). Your model was trained to assign data points to specific categories.
                    
                    **Key metrics to look at:**
                    - **Accuracy**: Overall correctness (good for balanced data)
                    - **F1-Score**: Balance of precision & recall (better for imbalanced data)
                    - **Precision**: How many predicted positives were correct?
                    - **Recall**: How many actual positives did we find?
                    """,
                    f"Explain my classification model results in detail. My best model achieved the following metrics: {comparison_rows[0] if comparison_rows else 'No results'}. Help me understand what these mean and if my model is good."
                )
            elif task_type == "regression":
                ai_interpretation_box(
                    "Understanding Regression Results",
                    """
                    **Regression** predicts continuous numerical values. Your model was trained to estimate numbers.
                    
                    **Key metrics to look at:**
                    - **R² Score**: Proportion of variance explained (closer to 1 is better)
                    - **RMSE**: Root Mean Squared Error (lower is better, in original units)
                    - **MAE**: Mean Absolute Error (average prediction error)
                    """,
                    f"Explain my regression model results in detail. My best model achieved the following metrics: {comparison_rows[0] if comparison_rows else 'No results'}. Help me understand what these mean and if my model is performing well."
                )
            
            # Main results section
            st.markdown("### 🎯 Training Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                smart_metric_card(
                    "Models Trained",
                    len(comparison_rows),
                    help_text="Number of models successfully trained",
                    interpretation="All selected models completed training"
                )
            with col2:
                smart_metric_card(
                    "Target Column",
                    target_col,
                    help_text="The variable being predicted",
                    interpretation=f"Predicting: {target_col}"
                )
            with col3:
                smart_metric_card(
                    "Task Type",
                    comparison_rows[0].get("task", "Unknown").title() if comparison_rows else "—",
                    help_text="Classification or Regression",
                    interpretation="Automatically detected based on target variable"
                )
            
            st.subheader("📊 Model Leaderboard")
            
            # Add ranking column
            metric_key = "accuracy" if comparison_rows[0].get("task") == "classification" else "r2"
            if metric_key not in comparison_rows[0]:
                metric_key = next((k for k in ("f1_macro", "rmse", "mae") if k in comparison_rows[0]), None)
            
            # Use modern comparison table
            comparison_table(
                comparison_rows,
                highlight_best=metric_key,
                ascending=False if metric_key in ["accuracy", "precision", "recall", "f1", "f1_macro", "r2"] else True
            )
            
            # Success message with next steps
            success_message_with_next_steps(
                "Models trained successfully!",
                [
                    f"Best model: {comparison_rows[0]['model']} (ID: {comparison_rows[0]['model_id']})",
                    "Review the leaderboard and metrics above",
                    "Visualize results in the charts below",
                    "Use the model ID for predictions on the Prediction page"
                ],
                {"label": "Go to Predictions", "page": "6_Prediction_Inference"}
            )
            
            # Old comparison df for backward compatibility
            comparison_df = pd.DataFrame(comparison_rows)
            
            if metric_key:
                comparison_df_sorted = comparison_df.sort_values(by=metric_key, ascending=False).reset_index(drop=True)
                comparison_df_sorted.index = comparison_df_sorted.index + 1
                comparison_df_sorted.index.name = "Rank"
            else:
                comparison_df_sorted = comparison_df

            if comparison_rows:
                # Create advanced visualizations
                st.divider()
                st.subheader("📈 Advanced Visualizations")
                
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    if metric_key:
                        # Create more detailed visualization
                        fig = make_subplots(
                            rows=1, cols=2,
                            subplot_titles=("Performance Leaderboard", "Model Comparison"),
                            specs=[[{"type": "bar"}, {"type": "table"}]]
                        )
                        
                        sorted_rows = sorted(comparison_rows, key=lambda r: r.get(metric_key, 0.0), reverse=True)
                        
                        fig.add_trace(
                            go.Bar(
                                x=[r["model"] for r in sorted_rows],
                                y=[r.get(metric_key, 0) for r in sorted_rows],
                                text=[f"{r.get(metric_key, 0):.4f}" for r in sorted_rows],
                                textposition="auto",
                                marker=dict(
                                    color=[r.get(metric_key, 0) for r in sorted_rows],
                                    colorscale="Viridis",
                                    showscale=True
                                ),
                                name="Score",
                                showlegend=False
                            ),
                            row=1, col=1
                        )
                        
                        fig.add_trace(
                            go.Table(
                                header=dict(
                                    values=["Model", metric_key.upper(), "Task"],
                                    fill_color="paleturquoise",
                                    align="left",
                                    font=dict(size=12, color="black")
                                ),
                                cells=dict(
                                    values=[
                                        [r["model"] for r in sorted_rows],
                                        [f"{r.get(metric_key, 0):.4f}" for r in sorted_rows],
                                        [r.get("task", "—") for r in sorted_rows]
                                    ],
                                    fill_color="lavender",
                                    align="left",
                                    font=dict(size=11),
                                    height=30
                                )
                            ),
                            row=1, col=2
                        )
                        
                        fig.update_layout(height=400, showlegend=False, title_text=f"Model Performance Comparison")
                        st.plotly_chart(fig, width="stretch")
                
                with viz_col2:
                    if metric_key and len(comparison_rows) > 1:
                        # Create radar/spider chart for multi-metric comparison
                        sorted_rows = sorted(comparison_rows, key=lambda r: r.get(metric_key, 0.0), reverse=True)[:3]  # Top 3
                        
                        # Get all available metrics
                        available_metrics = [k for k in sorted_rows[0].keys() if k not in ["model_id", "model", "task"]]
                        
                        if available_metrics:
                            fig_metrics = go.Figure()
                            
                            for row in sorted_rows:
                                metric_values = [row.get(m, 0) for m in available_metrics[:5]]  # Top 5 metrics
                                fig_metrics.add_trace(go.Scatterpolar(
                                    r=metric_values,
                                    theta=available_metrics[:5],
                                    fill='toself',
                                    name=row['model']
                                ))
                            
                            fig_metrics.update_layout(
                                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                                showlegend=True,
                                title="Top 3 Models - Multi-Metric Comparison",
                                height=400
                            )
                            st.plotly_chart(fig_metrics, width="stretch")

                # Highlight best model
                best = max(comparison_rows, key=lambda r: r.get(metric_key, 0.0)) if metric_key else comparison_rows[0]
                best_score = best.get(metric_key, "n/a")
                
                st.divider()
                
                with st.container(border=True):
                    st.markdown(f"### 🏆 Top Performer: **{best['model']}**")
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"{metric_key.upper()} Score", f"{best_score:.4f}" if isinstance(best_score, (int, float)) else best_score)
                    with col2:
                        st.metric("Rank", "🥇 1st Place")
                    with col3:
                        st.markdown(f"**Model ID:** `{best['model_id']}`")
                    
                    st.info(
                        f"✅ This model achieved the best {metric_key} score. "
                        f"Use Model ID `{best['model_id']}` on the **Prediction page** to make forecasts. "
                        f"\n\n**Recommendation:** Always validate on completely unseen data before production deployment.",
                        icon="🎯"
                    )
            
            # Detailed results section
            with st.expander("📋 Detailed Training Results", expanded=True):
                st.subheader("Registered Model Versions")
                for i, rec in enumerate(registered, 1):
                    with st.container(border=True):
                        st.markdown(f"**Model {i}: {rec.model_id}**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.caption(f"📁 Artifact: {Path(rec.artifact_path).name}")
                        with col2:
                            created = rec.created_utc[:10] if rec.created_utc else "Unknown"
                            st.caption(f"⏰ Created: {created}")
                        with col3:
                            st.caption(f"📊 Kind: {rec.kind}")
                    
                    if rec.meta:
                        st.json(rec.meta, expanded=False)
                
                # Performance metrics explanation
                with st.expander("📚 Understanding Performance Metrics", expanded=False):
                    st.subheader("Metric Definitions")
                    
                    if comparison_rows and comparison_rows[0].get("task") == "classification":
                        metrics_info = {
                            "accuracy": "Percentage of correct predictions overall",
                            "precision": "Of what we predicted positive, how many were actually positive?",
                            "recall": "Of actual positives in data, how many did we find?",
                            "f1": "Harmonic mean balancing precision and recall",
                            "f1_macro": "Average F1 across all classes (good for multi-class)"
                        }
                    else:
                        metrics_info = {
                            "r2": "Proportion of variance in target explained by features (0-1, higher better)",
                            "rmse": "Root Mean Squared Error (in target units, penalizes large errors)",
                            "mae": "Mean Absolute Error (average absolute prediction error)",
                            "mse": "Mean Squared Error (average squared errors)"
                        }
                    
                    for metric, description in metrics_info.items():
                        if comparison_rows and metric in comparison_rows[0]:
                            st.markdown(f"**{metric.upper()}**: {description}")
                
                # Model insights
                with st.expander("💡 Model Comparison Insights", expanded=False):
                    st.subheader("Key Findings")
                    
                    if len(comparison_rows) > 1:
                        # Find best and worst
                        metric_key = "accuracy" if comparison_rows[0].get("task") == "classification" else "r2"
                        if metric_key not in comparison_rows[0]:
                            metric_key = next((k for k in ("f1_macro", "rmse", "mae") if k in comparison_rows[0]), None)
                        
                        if metric_key:
                            sorted_rows = sorted(comparison_rows, key=lambda r: r.get(metric_key, 0.0), reverse=True)
                            best = sorted_rows[0]
                            worst = sorted_rows[-1]
                            
                            # Calculate performance gap
                            best_score = best.get(metric_key, 0.0)
                            worst_score = worst.get(metric_key, 0.0)
                            gap = abs(best_score - worst_score)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🏆 Best Model", best["model"])
                                st.caption(f"Score: {best_score:.4f}")
                            with col2:
                                st.metric("📊 Performance Gap", f"{gap:.4f}")
                                if gap > 0.1:
                                    st.caption("⚠️ Large difference - models vary significantly")
                                else:
                                    st.caption("✅ Similar performance - all models comparable")
                            with col3:
                                st.metric("Model Count", len(comparison_rows))
                                st.caption(f"Training metric: {metric_key}")
                        
                        st.info("""
                        **💭 How to interpret results:**
                        - **Large gap**: Some models much better than others - use the winner
                        - **Small gap**: All models similar - choose based on speed/complexity
                        - **Check test data**: Always validate on unseen data before deployment
                        """)

        except Exception as e:
            st.error(f"❌ Training failed: {str(e)}")
            st.exception(e)

st.subheader("📚 About These Algorithms")
with st.expander("Learn more about each model", expanded=False):
    st.markdown("""
    #### 🌲 Random Forest
    - **Best for:** Classification and regression on tabular data
    - **Pros:** Robust, handles non-linear patterns, built-in feature importance
    - **Cons:** Can be slow on very large datasets
    - **When to use:** Most general-purpose problems, great baseline
    - **Typical Accuracy:** 75-95% depending on problem complexity
    
    #### ⚡ XGBoost
    - **Best for:** High-performance scenarios (competitions, production)
    - **Pros:** Very powerful, excellent accuracy, handles missing values
    - **Cons:** Requires tuning, can overfit, slower training
    - **When to use:** When performance is critical
    - **Typical Accuracy:** 80-98% (often beats other models)
    
    #### 💡 LightGBM
    - **Best for:** Large datasets with many features
    - **Pros:** Fast training, memory-efficient, parallelizable
    - **Cons:** Needs careful hyperparameter tuning
    - **When to use:** Big data, production systems needing speed
    - **Typical Accuracy:** 80-98% (similar to XGBoost, faster)
    
    #### 📈 Gradient Boosting
    - **Best for:** Balanced performance and interpretability
    - **Pros:** Good accuracy, stable, well-understood
    - **Cons:** Slower than LightGBM, requires tuning
    - **When to use:** General purpose, when built-in models are preferred
    - **Typical Accuracy:** 75-95% (solid baseline performance)
    """)

st.subheader("💡 Tips & Best Practices")
with st.expander("Click to expand comprehensive tips", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🔧 Data Preparation
        1. **Feature Engineering:** Create meaningful features from raw data
        2. **Handle Missing Values:** Remove or impute NaN values
        3. **Encoding Categorical:** Convert text/categories to numbers
        4. **Scaling Numerical:** Normalize features to similar ranges
        5. **Feature Selection:** Remove irrelevant or redundant features
        
        ### 📊 Training Strategy
        1. **Start Simple:** Baseline with simple model first
        2. **Compare Models:** Always train multiple to find best
        3. **Cross-Validation:** Use k-fold for stable estimates
        4. **Hyperparameter Tuning:** Grid search or random search
        5. **Early Stopping:** Stop when validation performance plateaus
        """)
    
    with col2:
        st.markdown("""
        ### ⚠️ Common Pitfalls
        1. **Data Leakage:** Info from test data in training (deadly!)
        2. **Overfitting:** Model memorizes training data
        3. **Class Imbalance:** Unequal class distribution
        4. **Poor Feature Quality:** Garbage in → Garbage out
        5. **Wrong Metric:** Using inappropriate evaluation metric
        
        ### 🚀 Production Readiness
        1. **Monitor Performance:** Track metrics on production data
        2. **Retraining Schedule:** Retrain periodically with new data
        3. **Model Documentation:** Record hyperparameters & decisions
        4. **Validation Pipeline:** Automated testing before deployment
        5. **Fallback Plan:** Have alternative if model fails
        """)

st.divider()

# Add comparative analysis section
st.subheader("🔬 Algorithm Comparison Matrix")
comparison_matrix = pd.DataFrame({
    "Aspect": [
        "Speed (Training)",
        "Speed (Prediction)",
        "Memory Usage",
        "Interpretability",
        "Handles Non-linear",
        "Handles Missing Data",
        "Good for Big Data",
        "Hyperparameter Tuning",
        "Risk of Overfitting"
    ],
    "RandomForest": ["⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐", "⭐⭐", "⭐⭐"],
    "XGBoost": ["⭐⭐", "⭐⭐", "⭐⭐", "⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐"],
    "LightGBM": ["⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐⭐"],
    "GradientBoosting": ["⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐", "⭐⭐⭐⭐", "⭐⭐", "⭐⭐", "⭐⭐⭐", "⭐⭐"]
})
st.dataframe(comparison_matrix, width="stretch")
st.caption("⭐ = Rating (more stars = better). These are general guidelines and depend on data & parameters.")

st.divider()

st.subheader("📝 Documentation & Resources")

doc_col1, doc_col2, doc_col3 = st.columns(3)

with doc_col1:
    st.markdown("""
    ### 📚 Learn More
    - [Scikit-learn Docs](https://scikit-learn.org)
    - [XGBoost Tutorial](https://xgboost.readthedocs.io)
    - [ML Fundamentals](https://developers.google.com/machine-learning)
    """)

with doc_col2:
    st.markdown("""
    ### 🎯 Next Steps
    1. Train your first model
    2. Review the leaderboard
    3. Use top model for predictions
    4. Analyze feature importance
    """)

with doc_col3:
    st.markdown("""
    ### 🔧 Troubleshooting
    - **Low accuracy?** Try different features
    - **Slow training?** Reduce data or try LightGBM
    - **Overfitting?** Increase test size or regularization
    """)

st.caption(
    "**Optional Dependencies:** XGBoost and LightGBM require separate installation. "
    "See requirements.txt for instructions. The app automatically falls back to built-in models if unavailable."
)

# Page navigation
st.divider()
st.subheader("🔮 Unified Forecast Comparison (if available)")
st.caption("Compare saved forecasting models using the standardized visualization and metrics")

try:
    models = list_models(config.registry_dir)
    forecast_models = [m for m in models if m.kind == "forecast"]
    if not forecast_models:
        st.info("No forecasting models saved yet. Train on the Deep Learning page first.")
    else:
        # Let user select which forecast models to compare
        options_map = {f"{m.name} | {m.model_id}": m.model_id for m in forecast_models}
        selected_labels = st.multiselect(
            "Select forecast models to compare",
            options=list(options_map.keys()),
            default=list(options_map.keys())[:2]
        )

        if selected_labels:
            # Use current dataset if available for history context
            df_session = get_clean_df(st.session_state) or get_df(st.session_state)
            if df_session is None:
                st.warning("Load a dataset to provide historical context for the comparison.")
            else:
                results: list[ForecastResult] = []
                for label in selected_labels:
                    rid = options_map[label]
                    rec = get_model(config.registry_dir, rid)
                    if not rec:
                        st.warning(f"Model {label} not found in registry")
                        continue
                    meta = rec.meta or {}
                    target_col = str(meta.get("target_col"))
                    time_col = str(meta.get("time_col"))
                    horizon = int(meta.get("horizon", 12))
                    lookback = int(meta.get("lookback", 48))
                    model_type_raw = str(meta.get("type", "transformer_forecaster"))

                    # Prepare history
                    d_hist = df_session[[time_col, target_col]].copy().dropna()
                    if not pd.api.types.is_datetime64_any_dtype(d_hist[time_col]):
                        d_hist[time_col] = pd.to_datetime(d_hist[time_col], errors="coerce")
                    d_hist = d_hist[d_hist[time_col].notna()].sort_values(time_col)
                    d_hist[target_col] = pd.to_numeric(d_hist[target_col], errors="coerce")
                    d_hist = d_hist[d_hist[target_col].notna()]
                    if d_hist.empty:
                        st.warning(f"Model {label} lacks valid time/target columns in current data.")
                        continue

                    last_time = d_hist[time_col].iloc[-1]
                    diffs = d_hist[time_col].diff().dropna()
                    delta = diffs.median() if not diffs.empty else None
                    has_delta = isinstance(delta, pd.Timedelta) and delta > pd.Timedelta(0)

                    try:
                        if model_type_raw == "tft":
                            from src.ml.tft_optional import load_tft_forecaster, forecast_tft, tft_available
                            if not tft_available():
                                st.warning("TFT dependencies missing; skipping TFT model.")
                                continue
                            model = load_tft_forecaster(rec.artifact_path)
                            forecast_df, preds = forecast_tft(
                                model,
                                history_df=df_session,
                                time_col=time_col,
                                target_col=target_col,
                                horizon=horizon,
                                lookback=lookback,
                            )
                            if "forecast_time" in forecast_df.columns and forecast_df["forecast_time"].notna().any():
                                forecast_dates = pd.to_datetime(forecast_df["forecast_time"], errors="coerce")
                            elif has_delta:
                                forecast_dates = pd.to_datetime([last_time + (i + 1) * delta for i in range(horizon)])
                            else:
                                # Default to daily steps when cadence is unknown to keep dates datetime-like
                                forecast_dates = pd.date_range(start=last_time + pd.Timedelta(days=1), periods=len(forecast_df), freq="D")
                            model_type_norm = "tft"
                        else:
                            from src.ml.forecast_transformer import load_transformer_forecaster, forecast_transformer
                            model, ckpt = load_transformer_forecaster(rec.artifact_path)
                            series = d_hist[target_col].to_numpy(dtype=np.float32)
                            preds = forecast_transformer(model, ckpt, history=series)
                            if has_delta:
                                forecast_dates = pd.to_datetime([last_time + (i + 1) * delta for i in range(len(preds))])
                            else:
                                forecast_dates = pd.date_range(start=last_time + pd.Timedelta(days=1), periods=len(preds), freq="D")
                            model_type_norm = "transformer"

                        # Historical tail
                        hist_tail = d_hist.tail(min(lookback, len(d_hist)))
                        historical_values = hist_tail[target_col].to_numpy()
                        historical_dates = pd.DatetimeIndex(hist_tail[time_col])

                        # Ensure preds is 1D numpy array to avoid DataFrame truthiness issues
                        preds_array = np.asarray(preds).reshape(-1)

                        lower, upper = estimate_confidence_intervals(preds_array, model_type=model_type_norm)

                        # Ensure forecast_dates is DatetimeIndex, not list
                        if not isinstance(forecast_dates, pd.DatetimeIndex):
                            forecast_dates = pd.DatetimeIndex(forecast_dates)

                        results.append(
                            ForecastResult(
                                model_name=rec.name,
                                model_type=model_type_norm,
                                target_column=target_col,
                                forecast_values=preds_array,
                                forecast_dates=forecast_dates,
                                historical_values=historical_values,
                                historical_dates=historical_dates,
                                lower_bound=lower,
                                upper_bound=upper,
                                confidence_level=0.95,
                                train_metrics=None,
                                test_metrics=(meta.get("metrics") or {}),
                                metadata={"model_id": rec.model_id, "horizon": horizon, "lookback": lookback},
                            )
                        )
                    except Exception as e:
                        st.warning(f"Failed to prepare model {label}: {e}")

                if results:
                    render_model_comparison(results, title="Forecast Models Comparison")
except Exception as e:
    st.warning(f"Forecast comparison section failed: {e}")

standard_section_header("Learning Guide & Best Practices", "🎓")
concept_explainer(
    title="Tabular ML",
    explanation=(
        "Train supervised models on structured rows and columns. Compare algorithms, tune parameters, and select the best using appropriate metrics."
    ),
    real_world_example=(
        "Lead scoring: Predict conversion probability using demographics and interactions; compare Logistic Regression vs Random Forest and pick based on AUC and operational needs."
    ),
)
beginner_tip("Tip: Start with simpler models (Logistic/Linear) to establish a baseline and interpretability, then escalate to ensembles.")
common_mistakes_panel({
    "Wrong task selection": "Auto-detect or explicitly set classification vs regression.",
    "Ignoring class imbalance": "Use stratified splits and metrics beyond accuracy.",
    "Training on unclean data": "Run cleaning and feature engineering beforehand.",
    "No reproducibility": "Log seeds, configs, and dataset snapshots.",
    "Overfitting": "Use cross-validation and regularization; watch train vs test metrics.",
})

page_navigation("5")
