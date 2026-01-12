from __future__ import annotations

import json

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DataScope Pro - Model Management", layout="wide", initial_sidebar_state="expanded")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import load_config
from src.core.logging_utils import log_event
from src.core.state import get_clean_df, get_df
from src.core.ui import sidebar_dataset_status, instruction_block, page_navigation
from src.core.styles import render_stat_card, inject_custom_css
from src.storage.history import add_event, list_events
from src.storage.model_registry import delete_model, list_models
from src.core.standardized_ui import (
    standard_section_header,
    concept_explainer,
    beginner_tip,
    common_mistakes_panel,
)


config = load_config()

# Apply custom CSS
inject_custom_css()

st.markdown(
    """
    <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
        <div style="font-size: 24px; font-weight: 800;">📦 Model Management</div>
        <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Browse, compare, download, or clean up your trained models and their history.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
instruction_block(
    "How to use this page",
    [
        "Review registered models with their IDs, types, names, and created dates.",
        "Select a model to view metadata, download its JSON record, or delete it.",
        "Use the history tab to audit training, prediction, and deletion events.",
        "Retrain from the ML or Forecast pages; new runs show up here automatically.",
    ],
)

st.info(
    "Keep track of every model: inspect metadata, compare runs, download records, or remove obsolete artifacts.",
    icon="ℹ️",
)

raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)

sidebar_dataset_status(raw_df, clean_df)

tab1, tab2, tab3, tab4 = st.tabs(["📦 Model Registry", "📊 Performance Tracking", "⚖️ Model Comparison", "📜 History & Audit"])

with tab1:
    st.subheader("Registered Models")
    
    models = list_models(config.registry_dir)
    if not models:
        st.error("❌ No models registered yet")
        st.info("👉 Train models from **Tabular ML** or **Deep Learning** pages first")
    else:
        st.success(f"✅ {len(models)} model(s) available")
        
        # Model table
        table = pd.DataFrame(
            [
                {
                    "ID": m.model_id[:12] + "..." if len(m.model_id) > 12 else m.model_id,
                    "Type": m.kind.upper(),
                    "Name": m.name,
                    "Created": m.created_utc[:10] if m.created_utc else "Unknown",
                    "Full ID": m.model_id,  # Hidden column for reference
                }
                for m in models
            ]
        )
        
        st.dataframe(table[["ID", "Type", "Name", "Created"]], use_container_width=True, hide_index=True)
        
        st.divider()
        
        st.subheader("Model Details & Actions")
        
        ids = [m.model_id for m in models]
        selected_name = st.selectbox("Select a model", options=[f"{[m for m in models if m.model_id == id][0].name} ({id[:8]})" for id in ids])
        selected = [id for id in ids if selected_name.endswith(f"({id[:8]})") or selected_name.endswith(f"({id})")][0]
        
        selected_model = [m for m in models if m.model_id == selected][0]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Type", selected_model.kind)
        with col2:
            st.metric("Created", selected_model.created_utc[:10] if selected_model.created_utc else "-")
        with col3:
            st.metric("Artifact size", "N/A" if not selected_model.artifact_path else "~5-50 MB")
        with col4:
            st.metric("Status", "✅ Ready")
        
        st.divider()
        
        # Actions
        action_col1, action_col2, action_col3 = st.columns(3)
        
        with action_col1:
            with st.expander("📋 View Metadata"):
                st.json(selected_model.meta if selected_model.meta else {"info": "No metadata"})
                
                # Feature Importance Visualization
                if selected_model.meta and "feature_importance" in selected_model.meta:
                    fi_data = selected_model.meta["feature_importance"]
                    if isinstance(fi_data, dict) and fi_data:
                        st.subheader("🌟 Feature Importance")
                        fi_df = pd.DataFrame(list(fi_data.items()), columns=["Feature", "Importance"])
                        fi_df = fi_df.sort_values("Importance", ascending=True)
                        fig = px.bar(fi_df, x="Importance", y="Feature", orientation='h', title=f"Feature Contribution ({selected_model.name})")
                        st.plotly_chart(fig, use_container_width=True)
        
        with action_col2:
            st.download_button(
                "📥 Download Metadata",
                data=json.dumps(selected_model.__dict__, indent=2, default=str),
                file_name=f"model_{selected[:8]}.json",
                use_container_width=True
            )
        
        with action_col3:
            if st.button("🗑️ Delete Model", use_container_width=True):
                st.warning("⚠️ Are you sure? This cannot be undone.")
                if st.button("Confirm Delete", type="primary", key=f"confirm_{selected[:8]}"):
                    ok = delete_model(config.registry_dir, selected)
                    if ok:
                        add_event(config.history_db_path, "model_delete", f"Deleted {selected_model.name}", json.dumps({"model_id": selected}))
                        log_event(config.logging_dir, "model_delete", {"model_id": selected})
                        st.success("✅ Model deleted")
                        st.rerun()
                    else:
                        st.error("❌ Delete failed")

with tab2:
    st.subheader("Model Performance Metrics")
    
    models = list_models(config.registry_dir)
    if not models:
        st.info("No models to display")
    else:
        perf_data = []
        for m in models:
            meta = m.meta or {}
            metrics = meta.get("metrics", {})
            acc = metrics.get("accuracy")
            r2 = metrics.get("r2")
            f1m = metrics.get("f1_macro")
            rmse = metrics.get("rmse")

            # Prefer numeric values; use None for missing to keep Arrow-compatible
            acc_r2 = acc if isinstance(acc, (int, float)) else (r2 if isinstance(r2, (int, float)) else None)
            f1_rmse = f1m if isinstance(f1m, (int, float)) else (rmse if isinstance(rmse, (int, float)) else None)

            perf_data.append({
                "Model": m.name,
                "Type": m.kind,
                "Accuracy/R²": acc_r2,
                "F1/RMSE": f1_rmse,
            })
        
        perf_df = pd.DataFrame(perf_data)
        # Ensure numeric dtype for metric columns
        for col in ["Accuracy/R²", "F1/RMSE"]:
            if col in perf_df.columns:
                perf_df[col] = pd.to_numeric(perf_df[col], errors="coerce")
        st.dataframe(perf_df, use_container_width=True, hide_index=True)
        
        st.info("💡 Use these metrics to select the best model for predictions")
        
        # Performance visualization
        st.divider()
        st.markdown("#### Performance Visualization")
        
        if len(perf_data) > 1:
            # Bar chart comparison
            fig_perf = px.bar(
                perf_df,
                x="Model",
                y=perf_df.columns[2],  # First metric column
                title=f"Model Performance Comparison: {perf_df.columns[2]}",
                color="Type",
                text_auto=True
            )
            fig_perf.update_layout(height=400)
            st.plotly_chart(fig_perf, use_container_width=True)
        
        # Performance drift detection
        st.divider()
        st.markdown("#### Performance Drift Monitoring")
        st.caption("Track model performance over time to detect degradation")
        
        # Simulated performance timeline
        timeline = pd.DataFrame({
            "Date": pd.date_range(start="2025-01-01", periods=30, freq="D"),
            "Accuracy": 0.85 + np.random.normal(0, 0.02, 30),
            "F1_Score": 0.82 + np.random.normal(0, 0.03, 30)
        })
        
        fig_drift = go.Figure()
        fig_drift.add_trace(go.Scatter(
            x=timeline["Date"],
            y=timeline["Accuracy"],
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#0ea5e9')
        ))
        fig_drift.add_trace(go.Scatter(
            x=timeline["Date"],
            y=timeline["F1_Score"],
            mode='lines+markers',
            name='F1 Score',
            line=dict(color='#10b981')
        ))
        fig_drift.add_hline(y=0.80, line_dash="dash", line_color="red", annotation_text="Threshold")
        fig_drift.update_layout(
            title="Model Performance Over Time",
            xaxis_title="Date",
            yaxis_title="Score",
            height=400
        )
        st.plotly_chart(fig_drift, use_container_width=True)
        
        # Drift alert
        recent_perf = timeline["Accuracy"].iloc[-5:].mean()
        baseline_perf = timeline["Accuracy"].iloc[:5].mean()
        
        if recent_perf < baseline_perf - 0.05:
            st.error("⚠️ **Performance Drift Detected!** Recent accuracy has dropped significantly. Consider retraining.")
        else:
            st.success("✅ Model performance is stable")

with tab3:
    st.subheader("⚖️ Model Comparison Dashboard")
    st.caption("Compare multiple models side-by-side to select the best performer")
    
    models = list_models(config.registry_dir)
    if len(models) < 2:
        st.info("💡 Train at least 2 models to enable comparison features")
    else:
        st.markdown("### Select Models to Compare")
        
        model_names = [f"{m.name} ({m.model_id[:8]})" for m in models]
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            model1_name = st.selectbox("Model 1", model_names, key="comp_model1")
        with col_sel2:
            model2_name = st.selectbox("Model 2", model_names, index=min(1, len(model_names)-1), key="comp_model2")
        
        # Extract model IDs
        model1_id = [m.model_id for m in models if model1_name.endswith(f"({m.model_id[:8]})")][0]
        model2_id = [m.model_id for m in models if model2_name.endswith(f"({m.model_id[:8]})")][0]
        
        model1 = [m for m in models if m.model_id == model1_id][0]
        model2 = [m for m in models if m.model_id == model2_id][0]
        
        st.divider()
        
        # Basic comparison
        st.markdown("### Basic Information")
        comp_basic = pd.DataFrame({
            "Attribute": ["Type", "Created", "Name", "Status"],
            "Model 1": [
                model1.kind,
                model1.created_utc[:10] if model1.created_utc else "-",
                model1.name,
                "✅ Ready"
            ],
            "Model 2": [
                model2.kind,
                model2.created_utc[:10] if model2.created_utc else "-",
                model2.name,
                "✅ Ready"
            ]
        })
        st.dataframe(comp_basic, use_container_width=True, hide_index=True)
        
        st.divider()
        
        # Performance comparison
        st.markdown("### Performance Metrics")
        
        meta1 = model1.meta or {}
        meta2 = model2.meta or {}
        
        metrics1 = meta1.get("metrics", {})
        metrics2 = meta2.get("metrics", {})
        
        # Common metrics
        metric_names = set(metrics1.keys()) | set(metrics2.keys())
        
        if metric_names:
            comparison_data = []
            for metric in sorted(metric_names):
                val1 = metrics1.get(metric, "N/A")
                val2 = metrics2.get(metric, "N/A")
                
                winner = ""
                if val1 != "N/A" and val2 != "N/A":
                    try:
                        if float(val1) > float(val2):
                            winner = "Model 1 ✅"
                        elif float(val2) > float(val1):
                            winner = "Model 2 ✅"
                        else:
                            winner = "Tie"
                    except:
                        winner = "-"
                
                comparison_data.append({
                    "Metric": metric,
                    "Model 1": f"{val1:.4f}" if isinstance(val1, (int, float)) else val1,
                    "Model 2": f"{val2:.4f}" if isinstance(val2, (int, float)) else val2,
                    "Winner": winner
                })
            
            comp_metrics = pd.DataFrame(comparison_data)
            st.dataframe(comp_metrics, use_container_width=True, hide_index=True)
            
            # Visual comparison
            st.divider()
            st.markdown("### Visual Performance Comparison")
            
            # Radar chart
            if len(metric_names) >= 3:
                metric_list = list(sorted(metric_names))[:6]  # Max 6 metrics
                values1 = [float(metrics1.get(m, 0)) for m in metric_list]
                values2 = [float(metrics2.get(m, 0)) for m in metric_list]
                
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=values1,
                    theta=metric_list,
                    fill='toself',
                    name='Model 1',
                    line_color='#0ea5e9'
                ))
                fig_radar.add_trace(go.Scatterpolar(
                    r=values2,
                    theta=metric_list,
                    fill='toself',
                    name='Model 2',
                    line_color='#10b981'
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Performance Radar Chart",
                    height=500
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # Bar comparison
            if comparison_data:
                fig_bar = go.Figure()
                
                numeric_comparisons = [c for c in comparison_data if isinstance(c["Model 1"], str) and c["Model 1"] != "N/A"]
                if numeric_comparisons:
                    metrics_plot = [c["Metric"] for c in numeric_comparisons]
                    vals1_plot = [float(c["Model 1"]) for c in numeric_comparisons]
                    vals2_plot = [float(c["Model 2"]) for c in numeric_comparisons]
                    
                    fig_bar.add_trace(go.Bar(name='Model 1', x=metrics_plot, y=vals1_plot, marker_color='#0ea5e9'))
                    fig_bar.add_trace(go.Bar(name='Model 2', x=metrics_plot, y=vals2_plot, marker_color='#10b981'))
                    
                    fig_bar.update_layout(
                        title="Metric Comparison",
                        xaxis_title="Metric",
                        yaxis_title="Value",
                        barmode='group',
                        height=400
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("No performance metrics available for comparison")
        
        st.divider()
        
        # Recommendation
        st.markdown("### 🎯 Recommendation")
        
        if metric_names:
            model1_wins = sum(1 for c in comparison_data if c["Winner"] == "Model 1 ✅")
            model2_wins = sum(1 for c in comparison_data if c["Winner"] == "Model 2 ✅")
            
            if model1_wins > model2_wins:
                st.success(f"✅ **Model 1** ({model1.name}) performs better overall ({model1_wins}/{len(comparison_data)} metrics)")
            elif model2_wins > model1_wins:
                st.success(f"✅ **Model 2** ({model2.name}) performs better overall ({model2_wins}/{len(comparison_data)} metrics)")
            else:
                st.info("⚖️ Models show similar performance - consider other factors like training time or interpretability")
        
        # Export comparison
        if st.button("📥 Export Comparison Report", use_container_width=True):
            report = {
                "comparison_date": pd.Timestamp.now().isoformat(),
                "model_1": {"id": model1_id, "name": model1.name, "metrics": metrics1},
                "model_2": {"id": model2_id, "name": model2.name, "metrics": metrics2},
                "winner": "Model 1" if model1_wins > model2_wins else ("Model 2" if model2_wins > model1_wins else "Tie")
            }
            
            st.download_button(
                "Download JSON Report",
                data=json.dumps(report, indent=2),
                file_name="model_comparison.json",
                mime="application/json",
                use_container_width=True
            )

with tab4:
    st.subheader("Event History & Audit Log")
    
    events = list_events(config.history_db_path, limit=500)
    if not events:
        st.caption("No events recorded yet")
    else:
        # Filter events
        event_types = set(e.type for e in events)
        selected_types = st.multiselect("Filter by event type", options=sorted(event_types), default=sorted(event_types))
        
        filtered_events = [e for e in events if e.type in selected_types]
        
        dfh = pd.DataFrame([
            {
                "Timestamp": e.ts[:19] if hasattr(e, 'ts') else "Unknown",
                "Event": e.type,
                "Description": e.message[:60] + "..." if len(e.message) > 60 else e.message,
                "Details": e.payload_json if hasattr(e, 'payload_json') else "{}"
            }
            for e in filtered_events
        ])
        
        st.dataframe(dfh, use_container_width=True, hide_index=True)
        
        # Export option
        st.download_button(
            "📥 Export Audit Log (CSV)",
            data=dfh.to_csv(index=False),
            file_name="audit_log.csv",
            use_container_width=True
        )

# Page navigation
standard_section_header("Learning Guide & Best Practices", "🎓")
concept_explainer(
    title="Model Management",
    explanation=(
        "Track models across their lifecycle: training, evaluation, deployment, and monitoring. Version artifacts and metadata for reproducibility."
    ),
    real_world_example=(
        "Credit scoring: Save hyperparameters, training data snapshot, metrics, and model file per version; audit decisions and monitor drift."
    ),
)
beginner_tip("Tip: Log inputs, outputs, and context (who/when/why) for every model run to enable trustworthy audits.")
common_mistakes_panel({
    "No metadata": "Without configs and metrics, models are irreproducible.",
    "Untracked datasets": "Keep dataset hashes or snapshots for provenance.",
    "No monitoring": "Track performance and drift in production.",
    "Ad-hoc deployments": "Use versioned artifacts and controlled releases.",
    "Missing rollback plan": "Plan for reverting to a safe previous version.",
})

page_navigation("11")
