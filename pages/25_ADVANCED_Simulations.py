"""Advanced Enterprise-Grade ML Simulation Platform - 10M$ Feature Set"""

import streamlit as st

st.set_page_config(
    page_title="DataScope Pro - Advanced Simulations", 
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": "Advanced ML Simulation Platform with 15+ Enterprise Features"}
)

import pandas as pd
import numpy as np
from typing import Optional, List, Dict
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time

from src.core.config import load_config
from src.core.logging_utils import setup_logging
from src.core.ui import app_header
from src.core.styles import render_stat_card

# Advanced simulators
from src.simulation.advanced_simulators import (
    ClusteringSimulator,
    NeuralNetworkSimulator,
    AnomalyDetectionSimulator,
    EnsembleSimulator
)
from src.simulation.advanced_visualizations import AdvancedVisualizer
from src.simulation.statistical_tests import (
    StatisticalTestSimulator,
    ABTestSimulator,
    BayesianABTesting
)
from src.simulation.production_sim import (
    ProductionReadinessSimulator,
    DataDriftSimulator
)


def _init_session_state():
    """Initialize session state variables."""
    if 'advanced_sim_results' not in st.session_state:
        st.session_state.advanced_sim_results = None
    if 'advanced_sim_history' not in st.session_state:
        st.session_state.advanced_sim_history = []


def _load_uploaded_csv() -> pd.DataFrame | None:
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    if uploaded is None:
        return None
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} cols")
        _render_eda_preview(df)
        return df
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        return None


def _render_eda_preview(df: pd.DataFrame) -> None:
    """Display quick EDA after CSV upload."""
    st.sidebar.markdown("### 📊 Data Preview")
    
    # Basic stats
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    
    # Data types
    with st.sidebar.expander("Data Types", expanded=False):
        dtype_summary = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str)
        })
        st.dataframe(dtype_summary, use_container_width=True, hide_index=True)
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        with st.sidebar.expander("Missing Values", expanded=False):
            missing_pct = (missing / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing[missing > 0].index,
                'Count': missing[missing > 0].values,
                '% Missing': missing_pct[missing > 0].values
            })
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        with st.sidebar.expander("Numeric Summary", expanded=False):
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
    
    # Class distribution (for first non-numeric or last column)
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        first_cat = cat_cols[0]
        with st.sidebar.expander(f"Distribution: {first_cat}", expanded=False):
            dist = df[first_cat].value_counts()
            st.bar_chart(dist)
    elif len(numeric_cols) > 0:
        last_numeric = numeric_cols[-1]
        with st.sidebar.expander(f"Distribution: {last_numeric}", expanded=False):
            st.histogram(df[last_numeric].dropna(), use_container_width=True)


def _prepare_uploaded_xy(df: pd.DataFrame, problem_type: str, key_prefix: str):
    """Prepare (X, y) from an uploaded dataframe.

    - Uses a selected target column
    - Uses selected feature columns (or auto-select numeric)
    - Encodes classification targets when needed
    """

    cols = list(df.columns)
    target_col = st.sidebar.selectbox(
        "Target column",
        cols,
        index=len(cols) - 1,
        key=f"{key_prefix}::target",
    )

    candidate_features = [c for c in cols if c != target_col]
    feature_cols = st.sidebar.multiselect(
        "Feature columns (empty = auto numeric)",
        candidate_features,
        default=[],
        key=f"{key_prefix}::features",
    )
    if not feature_cols:
        feature_cols = [c for c in candidate_features if pd.api.types.is_numeric_dtype(df[c])]

    X_df = df[feature_cols].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    y_raw = df.loc[X_df.index, target_col]

    X_df = X_df.select_dtypes(include=["number"]).copy()
    X = X_df.to_numpy()

    if problem_type.lower().startswith("class"):
        if not pd.api.types.is_numeric_dtype(y_raw):
            from sklearn.preprocessing import LabelEncoder

            le = LabelEncoder()
            y = le.fit_transform(y_raw.astype(str))
        else:
            y = y_raw.to_numpy()
    else:
        y = pd.to_numeric(y_raw, errors="coerce").to_numpy()
        mask = ~np.isnan(y)
        X = X[mask]
        y = y[mask]

    return X, y, X_df.columns.tolist(), target_col


def _prepare_uploaded_X(df: pd.DataFrame, key_prefix: str) -> np.ndarray:
    cols = list(df.columns)
    numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    selected = st.sidebar.multiselect(
        "Feature columns", numeric_cols, default=numeric_cols[: min(10, len(numeric_cols))], key=f"{key_prefix}::X"
    )
    X_df = df[selected].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
    return X_df.to_numpy()


def main():
    """Main dashboard function."""
    config = load_config()
    logger = setup_logging(config.logging_dir, config.logging_level)
    _init_session_state()

    st.sidebar.markdown("### 📁 Upload Data (Optional)")
    uploaded_df = _load_uploaded_csv()
    
    # Header
    app_header(
        config=config,
        page_title="Advanced Simulations",
        subtitle="Enterprise-Grade ML Simulation Platform with 15+ Advanced Features",
        icon="🚀"
    )
    
    # Feature tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔮 Clustering",
        "🧠 Neural Networks",
        "📊 Ensemble Methods",
        "📉 Anomaly Detection",
        "🔬 A/B Testing & Statistics",
        "🚀 Production Readiness"
    ])
    
    with tab1:
        st.header("🔮 Clustering & Dimensionality Reduction")

        use_uploaded = st.checkbox(
            "Use uploaded CSV for clustering",
            value=False,
            disabled=uploaded_df is None,
            key="cluster_use_uploaded",
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Configuration")
            n_samples = st.slider("Number of Samples", 100, 2000, 500, key="cluster_samples")
            n_features = st.slider("Number of Features", 2, 50, 10, key="cluster_features")
            n_clusters = st.slider("True Number of Clusters", 2, 10, 3, key="cluster_n")
            data_pattern = st.selectbox(
                "Data Pattern",
                ["blobs", "moons", "circles", "elongated"],
                key="cluster_pattern"
            )
        
        with col2:
            st.subheader("Algorithm Selection")
            algorithm = st.selectbox(
                "Clustering Algorithm",
                ["kmeans", "dbscan", "hierarchical"],
                key="cluster_algo"
            )
            dim_reducer = st.selectbox(
                "Dimensionality Reducer",
                ["pca", "tsne", "isomap"],
                key="cluster_reducer"
            )
            
            if algorithm == "dbscan":
                eps = st.slider("DBSCAN Epsilon", 0.1, 2.0, 0.5, key="cluster_eps")
                min_samples = st.slider("Min Samples", 2, 20, 5, key="cluster_min")
            else:
                eps = min_samples = None
        
        if st.button("▶️ Run Clustering Analysis", key="cluster_run"):
            with st.spinner("Clustering..."):
                sim = ClusteringSimulator()
                if use_uploaded and uploaded_df is not None:
                    X = _prepare_uploaded_X(uploaded_df, key_prefix="cluster")
                    if X.shape[0] > n_samples:
                        X = X[:n_samples]
                else:
                    X = sim.generate_clustering_data(n_samples, n_features, n_clusters, data_pattern)
                
                kwargs = {}
                if algorithm == "dbscan":
                    kwargs['eps'] = eps
                    kwargs['min_samples'] = min_samples
                
                result = sim.run_clustering(X, algorithm, n_clusters, dim_reducer, **kwargs)
                
                # 2D Visualization
                col1, col2 = st.columns(2)
                with col1:
                    fig_2d = go.Figure(data=go.Scatter(
                        x=result.reduced_data[:, 0],
                        y=result.reduced_data[:, 1],
                        mode='markers',
                        marker=dict(
                            size=6,
                            color=result.labels,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Cluster")
                        ),
                        text=[f"Cluster {l}" for l in result.labels],
                        hoverinfo='text'
                    ))
                    fig_2d.update_layout(
                        title=f"{algorithm.title()} Clustering - {dim_reducer.upper()} Projection",
                        xaxis_title=f"{dim_reducer.upper()}-1",
                        yaxis_title=f"{dim_reducer.upper()}-2",
                        height=500
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
                
                with col2:
                    if result.original_data.shape[1] >= 3:
                        fig_3d = AdvancedVisualizer.plot_3d_clusters(result.original_data, result.labels)
                        st.plotly_chart(fig_3d, use_container_width=True)
                
                # Metrics
                st.subheader("📊 Clustering Quality Metrics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Silhouette Score",
                        f"{result.silhouette:.3f}",
                        delta="Higher is better" if result.silhouette > 0 else "Lower is worse",
                        delta_color="inverse"
                    )
                with col2:
                    st.metric(
                        "Davies-Bouldin Index",
                        f"{result.davies_bouldin:.3f}",
                        delta="Lower is better",
                        delta_color="inverse"
                    )
                with col3:
                    st.metric(
                        "Calinski-Harabasz",
                        f"{result.calinski_harabasz:.1f}",
                        delta="Higher is better"
                    )
    
    with tab2:
        st.header("🧠 Neural Network Architecture Search")
        st.write("Simulate AutoML Neural Architecture Search (NAS) to find optimal architectures.")

        use_uploaded = st.checkbox(
            "Use uploaded CSV dataset",
            value=False,
            disabled=uploaded_df is None,
            key="nas_use_uploaded",
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Dataset Configuration")
            problem_type = st.radio("Problem Type", ["Classification", "Regression"], key="nas_problem")
            n_samples = st.slider("Training Samples", 100, 2000, 500, key="nas_samples")
            n_features = st.slider("Number of Features", 5, 50, 20, key="nas_features")
            test_split = st.slider("Test Size %", 10, 50, 20, key="nas_test") / 100
        
        with col2:
            st.subheader("Architecture Search")
            max_layers = st.slider("Max Hidden Layers", 1, 10, 5, key="nas_layers")
            max_neurons = st.slider("Max Neurons per Layer", 32, 512, 128, key="nas_neurons")
            n_trials = st.slider("Number of Architectures to Try", 5, 50, 15, key="nas_trials")
        
        if st.button("🚀 Search for Best Architecture", key="nas_run"):
            with st.spinner("Running Neural Architecture Search..."):
                # Generate data
                if use_uploaded and uploaded_df is not None:
                    X, y, _, _ = _prepare_uploaded_xy(uploaded_df, problem_type, key_prefix="nas")
                    if X.shape[0] > n_samples:
                        X = X[:n_samples]
                        y = y[:n_samples]
                    rs = 42
                else:
                    if problem_type == "Classification":
                        X, y = make_classification(
                            n_samples=n_samples,
                            n_features=n_features,
                            n_informative=int(n_features * 0.7),
                            n_classes=3,
                            random_state=42
                        )
                    else:
                        X, y = make_regression(n_samples=n_samples, n_features=n_features, random_state=42)
                    rs = 42

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=rs)
                X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=rs)
                
                # Run NAS
                nas_sim = NeuralNetworkSimulator()
                results = nas_sim.simulate_architecture_search(
                    X_train, y_train, X_val, y_val,
                    problem_type.lower(),
                    max_layers, max_neurons, n_trials
                )
                
                if results:
                    best = results[0]
                    
                    st.subheader("🏆 Best Architecture Found")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Architecture", str(best.architecture))
                    with col2:
                        st.metric("Activation", best.activation)
                    with col3:
                        st.metric("Learning Rate", f"{best.learning_rate:.2e}")
                    with col4:
                        st.metric("Validation Score", f"{best.final_score:.4f}")
                    
                    # Architecture visualization
                    fig_arch = AdvancedVisualizer.plot_neural_architecture(best.architecture, best.activation)
                    st.plotly_chart(fig_arch, use_container_width=True)
                    
                    # Search history
                    st.subheader("📈 Architecture Search Progress")
                    progress_data = {
                        'Trial': list(range(1, len(results) + 1)),
                        'Score': [r.final_score for r in results],
                        'Layers': [len(r.architecture) for r in results]
                    }
                    progress_df = pd.DataFrame(progress_data)
                    
                    fig = px.scatter(
                        progress_df,
                        x='Trial',
                        y='Score',
                        size='Layers',
                        color='Score',
                        color_continuous_scale='Viridis',
                        title="NAS Search Progress",
                        hover_data=['Layers']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Top architectures table
                    st.subheader("Top 10 Architectures")
                    top_10 = pd.DataFrame({
                        'Rank': range(1, min(11, len(results) + 1)),
                        'Architecture': [str(r.architecture) for r in results[:10]],
                        'Activation': [r.activation for r in results[:10]],
                        'Learning Rate': [f"{r.learning_rate:.2e}" for r in results[:10]],
                        'Val Score': [f"{r.final_score:.4f}" for r in results[:10]],
                        'Iterations': [r.convergence_iteration for r in results[:10]]
                    })
                    st.dataframe(top_10, use_container_width=True, hide_index=True)
    
    with tab3:
        st.header("📊 Ensemble Methods Comparison")
        st.write("Compare Bagging, Boosting, and other ensemble strategies.")

        use_uploaded = st.checkbox(
            "Use uploaded CSV for ensemble comparison (classification)",
            value=False,
            disabled=uploaded_df is None,
            key="ens_use_uploaded",
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Dataset")
            n_samples = st.slider("Training Samples", 200, 2000, 500, key="ens_samples")
            n_features = st.slider("Number of Features", 5, 50, 20, key="ens_features")
            test_split = st.slider("Test Size %", 10, 50, 20, key="ens_test") / 100
        
        with col2:
            st.subheader("Configuration")
            n_estimators = st.slider("Number of Base Estimators", 10, 200, 50, key="ens_estimators")
        
        if st.button("▶️ Compare Ensemble Methods", key="ens_run"):
            with st.spinner("Comparing ensemble methods..."):
                # Generate data
                if use_uploaded and uploaded_df is not None:
                    X, y, _, _ = _prepare_uploaded_xy(uploaded_df, "Classification", key_prefix="ens")
                    if X.shape[0] > n_samples:
                        X = X[:n_samples]
                        y = y[:n_samples]
                    classes, counts = np.unique(y, return_counts=True)
                    if len(classes) < 2:
                        raise ValueError("Target column must have at least 2 classes")
                    if counts.min() < 2:
                        st.warning("Some classes have only 1 sample after filtering; disabling stratify and using a simple split.")
                        stratify_arg = None
                    else:
                        stratify_arg = y
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_split, random_state=42, stratify=stratify_arg
                    )
                else:
                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=n_features,
                        n_classes=2,
                        n_clusters_per_class=2,
                        random_state=42
                    )
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=42)
                
                # Run comparison
                ens_sim = EnsembleSimulator()
                results = ens_sim.compare_ensemble_methods(X_train, y_train, X_test, y_test, n_estimators)
                
                # Results dataframe
                results_df = pd.DataFrame(results).T
                results_df = results_df.round(4)
                
                st.subheader("📊 Detailed Results")
                st.dataframe(results_df, use_container_width=True)
                
                # Comparison visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_acc = px.bar(
                        x=results_df.index,
                        y=results_df['accuracy'],
                        title="Accuracy Comparison",
                        labels={'x': 'Method', 'y': 'Accuracy'},
                        color=results_df['accuracy'],
                        color_continuous_scale='Viridis'
                    )
                    st.plotly_chart(fig_acc, use_container_width=True)
                
                with col2:
                    fig_time = px.bar(
                        x=results_df.index,
                        y=results_df['train_time'],
                        title="Training Time Comparison",
                        labels={'x': 'Method', 'y': 'Time (seconds)'},
                        color=results_df['train_time'],
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig_time, use_container_width=True)
                
                # Best method
                best_idx = results_df['f1'].idxmax()
                st.success(f"🏆 Best Overall: **{best_idx}** (F1-Score: {results_df.loc[best_idx, 'f1']:.4f})")
    
    with tab4:
        st.header("📉 Anomaly Detection")
        st.write("Detect outliers using Isolation Forest, One-Class SVM, and Elliptic Envelope.")

        use_uploaded = st.checkbox(
            "Use uploaded CSV for anomaly detection",
            value=False,
            disabled=uploaded_df is None,
            key="anom_use_uploaded",
        )
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.subheader("Data Configuration")
            n_samples = st.slider("Number of Samples", 500, 2000, 1000, key="anom_samples")
            n_features = st.slider("Number of Features", 2, 50, 10, key="anom_features")
            contamination = st.slider("Contamination Rate", 0.01, 0.5, 0.10, key="anom_contam")
        
        with col2:
            st.subheader("Algorithm")
            algorithm = st.selectbox(
                "Detection Algorithm",
                ["isolation_forest", "one_class_svm", "elliptic_envelope"],
                format_func=lambda x: x.replace('_', ' ').title(),
                key="anom_algo"
            )
        
        if st.button("🔍 Detect Anomalies", key="anom_run"):
            with st.spinner("Detecting anomalies..."):
                sim = AnomalyDetectionSimulator()
                if use_uploaded and uploaded_df is not None:
                    X = _prepare_uploaded_X(uploaded_df, key_prefix="anom")
                    if X.shape[0] > n_samples:
                        X = X[:n_samples]
                    y_true = None
                else:
                    X, y_true = sim.generate_anomaly_data(n_samples, n_features, contamination)
                
                result = sim.detect_anomalies(X, algorithm, contamination)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Samples", n_samples)
                with col2:
                    st.metric("Anomalies Detected", result.n_anomalies)
                with col3:
                    st.metric("Anomaly Rate", f"{result.anomaly_rate:.2%}")
                with col4:
                    st.metric("Algorithm", algorithm.replace('_', ' ').title())
                
                # Visualization
                if n_features >= 2:
                    X_vis = X[:, :2] if n_features > 2 else X
                    
                    fig_2d = go.Figure()
                    
                    normal_mask = result.predictions == 0
                    fig_2d.add_trace(go.Scatter(
                        x=X_vis[normal_mask, 0],
                        y=X_vis[normal_mask, 1],
                        mode='markers',
                        name='Normal Points',
                        marker=dict(color='green', size=6, opacity=0.6)
                    ))
                    
                    anomaly_mask = result.predictions == 1
                    fig_2d.add_trace(go.Scatter(
                        x=X_vis[anomaly_mask, 0],
                        y=X_vis[anomaly_mask, 1],
                        mode='markers+text',
                        name='Anomalies',
                        marker=dict(color='red', size=12, symbol='x', line=dict(width=2)),
                        text=np.where(anomaly_mask)[0],
                        textposition='top center'
                    ))
                    
                    fig_2d.update_layout(
                        title=f"Anomaly Detection - {algorithm.replace('_', ' ').title()}",
                        xaxis_title="Feature 1",
                        yaxis_title="Feature 2",
                        height=500
                    )
                    st.plotly_chart(fig_2d, use_container_width=True)
                
                # Anomaly scores
                fig_scores = AdvancedVisualizer.plot_anomaly_scores(result.scores, result.predictions)
                st.plotly_chart(fig_scores, use_container_width=True)
    
    with tab5:
        st.header("🔬 A/B Testing & Statistical Analysis")
        st.write("Comprehensive A/B testing and statistical hypothesis testing toolkit.")

        use_uploaded_ab = st.checkbox(
            "Use uploaded CSV for A/B tests",
            value=False,
            disabled=uploaded_df is None,
            key="ab_use_uploaded",
        )
        
        test_type = st.radio(
            "Select Test Type",
            [
                "Continuous Metric (T-Test)",
                "Conversion Rate (Z-Test)",
                "Power Analysis",
                "Sequential Testing",
                "Bayesian A/B Test"
            ],
            key="ab_type"
        )
        
        if test_type == "Continuous Metric (T-Test)":
            if use_uploaded_ab and uploaded_df is not None:
                st.markdown("**Use uploaded data**")
                cols = list(uploaded_df.columns)
                group_col = st.selectbox("Group column (A/B)", cols, key="ab_ttest_group")
                metric_col = st.selectbox("Metric column (numeric)", cols, key="ab_ttest_metric")
                if st.button("▶️ Run A/B Test", key="ab_ct_run"):
                    df = uploaded_df[[group_col, metric_col]].dropna()
                    g1 = df[df[group_col] == df[group_col].unique()[0]][metric_col]
                    g2 = df[df[group_col] == df[group_col].unique()[1]][metric_col]
                    if len(g1) < 3 or len(g2) < 3:
                        st.error("Need at least 3 samples per group.")
                    else:
                        res = ABTestSimulator.t_test(g1.to_numpy(), g2.to_numpy())
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("P-Value", f"{res.p_value:.6f}")
                        with col2:
                            st.metric("Effect Size", f"{res.effect_size:.3f}")
                        with col3:
                            st.metric("Significant", "✅" if res.significant else "❌")
                        st.write(res.interpretation)
                        st.info(res.recommendation)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    control_mean = st.number_input("Control Group Mean", 100.0, 200.0, 100.0, key="ab_cm")
                    control_std = st.number_input("Control Group Std Dev", 1.0, 50.0, 10.0, key="ab_cs")
                with col2:
                    treatment_mean = st.number_input("Treatment Group Mean", 100.0, 150.0, 105.0, key="ab_tm")
                    treatment_std = st.number_input("Treatment Group Std Dev", 1.0, 50.0, 10.0, key="ab_ts")
                
                n_samples = st.slider("Samples per Group", 100, 5000, 1000, key="ab_n")
                
                if st.button("▶️ Run A/B Test", key="ab_ct_run"):
                    result = ABTestSimulator.simulate_ab_test(
                        control_mean, treatment_mean,
                        control_std, treatment_std,
                        n_samples
                    )
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Uplift", f"{result.uplift:+.2f}")
                    with col2:
                        st.metric("Uplift %", f"{result.uplift_pct:+.2f}%")
                    with col3:
                        st.metric("P-Value", f"{result.p_value:.6f}")
                    with col4:
                        status = "✅ SIGNIFICANT" if result.significant else "❌ NOT SIGNIFICANT"
                        st.metric("Result", status)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Confidence Interval (95%)**: [{result.confidence_interval[0]:.2f}, {result.confidence_interval[1]:.2f}]")
                        st.write(f"**Statistical Power**: {result.power:.2%}")
                    with col2:
                        st.write(f"**Minimum Detectable Effect**: {result.minimum_detectable_effect:.4f}")
        
        elif test_type == "Conversion Rate (Z-Test)":
            if use_uploaded_ab and uploaded_df is not None:
                st.markdown("**Use uploaded data**")
                cols = list(uploaded_df.columns)
                group_col = st.selectbox("Group column (A/B)", cols, key="ab_conv_group")
                outcome_col = st.selectbox("Binary outcome column (0/1)", cols, key="ab_conv_outcome")
                if st.button("▶️ Run Conversion Test", key="ab_conv_run"):
                    df = uploaded_df[[group_col, outcome_col]].dropna()
                    # enforce binary
                    vals = pd.to_numeric(df[outcome_col], errors="coerce")
                    df = df.loc[~vals.isna()].copy()
                    df[outcome_col] = (vals > 0).astype(int)
                    groups = df[group_col].unique()
                    if len(groups) < 2:
                        st.error("Need at least two groups in the group column.")
                    else:
                        gA = df[df[group_col] == groups[0]][outcome_col]
                        gB = df[df[group_col] == groups[1]][outcome_col]
                        res = ABTestSimulator.simulate_conversion_test(
                            control_rate=gA.mean(),
                            treatment_rate=gB.mean(),
                            n_samples=min(len(gA), len(gB))
                        )
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Uplift %", f"{res.uplift_pct:+.2f}%")
                        with col2:
                            st.metric("P-Value", f"{res.p_value:.6f}")
                        with col3:
                            st.metric("Power", f"{res.power:.2%}")
                        with col4:
                            st.metric("Result", "✅ SIG" if res.significant else "❌ NOT SIG")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    control_rate = st.slider("Control Conversion Rate", 0.01, 0.5, 0.10, key="ab_cr")
                    n_samples = st.slider("Samples per Group", 100, 10000, 2000, key="ab_crn")
                with col2:
                    treatment_rate = st.slider("Treatment Conversion Rate", 0.01, 0.5, 0.12, key="ab_tr")
                
                if st.button("▶️ Run Conversion Test", key="ab_conv_run"):
                    result = ABTestSimulator.simulate_conversion_test(control_rate, treatment_rate, n_samples)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Uplift %", f"{result.uplift_pct:+.2f}%")
                    with col2:
                        st.metric("P-Value", f"{result.p_value:.6f}")
                    with col3:
                        st.metric("Power", f"{result.power:.2%}")
                    with col4:
                        st.metric("Result", "✅ SIG" if result.significant else "❌ NOT SIG")
        
        elif test_type == "Power Analysis":
            if use_uploaded_ab and uploaded_df is not None:
                st.markdown("**Use uploaded data**")
                cols = list(uploaded_df.columns)
                group_col = st.selectbox("Group column (A/B)", cols, key="ab_pow_group")
                outcome_col = st.selectbox("Binary outcome column (0/1)", cols, key="ab_pow_outcome")
                min_n = st.slider("Min Sample Size", 50, 10000, 500, step=50, key="ab_pow_minn")
                max_n = st.slider("Max Sample Size", 200, 50000, 5000, step=100, key="ab_pow_maxn")
                max_n = max(max_n, min_n + 50)
                if st.button("▶️ Compute Power Matrix", key="ab_pow_run"):
                    df = uploaded_df[[group_col, outcome_col]].dropna()
                    vals = pd.to_numeric(df[outcome_col], errors="coerce")
                    df = df.loc[~vals.isna()].copy()
                    df[outcome_col] = (vals > 0).astype(int)
                    groups = df[group_col].unique()
                    if len(groups) < 2:
                        st.error("Need at least two groups in the group column.")
                    else:
                        gA = df[df[group_col] == groups[0]][outcome_col]
                        gB = df[df[group_col] == groups[1]][outcome_col]
                        baseline = gA.mean()
                        effect = max(gB.mean() - baseline, 0.0001)
                        effect_sizes = [effect * (0.5 + 0.1 * i) for i in range(1, 8)]
                        sample_sizes = list(range(int(min_n), int(max_n) + 1, max(50, (max_n - min_n) // 8)))
                        power_df = ABTestSimulator.power_analysis(baseline, effect_sizes, sample_sizes)
                        pivot_df = power_df.pivot(index='sample_size', columns='effect_pct', values='power')
                        pivot_df = pivot_df.sort_index().sort_index(axis=1)
                        if pivot_df.empty:
                            st.warning("Not enough data to compute power matrix.")
                        else:
                            fig = px.imshow(
                                z=pivot_df.to_numpy(),
                                x=[f"{c:.3f}" for c in pivot_df.columns],
                                y=pivot_df.index,
                                labels=dict(x="Effect Size (%)", y="Sample Size", color="Power"),
                                title="Power from uploaded data (baseline and uplift derived)",
                                color_continuous_scale='RdYlGn'
                            )
                            st.plotly_chart(fig, use_container_width=True)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    baseline = st.slider("Baseline Conversion Rate", 0.01, 0.5, 0.10, key="ab_base")
                    min_effect_pct = st.slider("Min Effect %", 1.0, 50.0, 5.0, key="ab_mineff") / 100
                with col2:
                    max_effect_pct = st.slider("Max Effect %", 1.0, 100.0, 50.0, key="ab_maxeff") / 100
                    min_n = st.slider("Min Sample Size", 100, 5000, 1000, step=100, key="ab_minn")
                    max_n = st.slider("Max Sample Size", 1000, 50000, 10000, step=100, key="ab_maxn")
                
                if st.button("▶️ Compute Power Matrix", key="ab_pow_run"):
                    effect_sizes = [min_effect_pct + (max_effect_pct - min_effect_pct) * i / 10 for i in range(11)]
                    sample_sizes = list(range(min_n, max_n + 1, max(1, (max_n - min_n) // 10)))
                    
                    power_df = ABTestSimulator.power_analysis(baseline, effect_sizes, sample_sizes)
                    pivot_df = power_df.pivot(index='sample_size', columns='effect_pct', values='power')
                    pivot_df = pivot_df.sort_index().sort_index(axis=1)
                    if pivot_df.empty:
                        st.warning("Not enough data to compute power matrix.")
                    else:
                        fig = px.imshow(
                            z=pivot_df.to_numpy(),
                            x=[f"{c:.3f}" for c in pivot_df.columns],
                            y=pivot_df.index,
                            labels=dict(x="Effect Size (%)", y="Sample Size", color="Power"),
                            title="Statistical Power Heatmap",
                            color_continuous_scale='RdYlGn'
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("🚀 Production Readiness Assessment")
        st.write("Complete production readiness evaluation for your ML models.")

        use_uploaded = st.checkbox(
            "Use uploaded CSV to build model + report",
            value=False,
            disabled=uploaded_df is None,
            key="prod_use_uploaded",
        )
        
        if st.button("📋 Generate Full Production Report", key="prod_run"):
            with st.spinner("Analyzing production readiness..."):
                # Create sample model
                if use_uploaded and uploaded_df is not None:
                    X, y, _, _ = _prepare_uploaded_xy(uploaded_df, "Classification", key_prefix="prod")
                    classes, counts = np.unique(y, return_counts=True)
                    if len(classes) < 2:
                        raise ValueError("Target column must have at least 2 classes")
                    if counts.min() < 2:
                        st.warning("Some classes have only 1 sample; disabling stratify and using a simple split.")
                        stratify_arg = None
                    else:
                        stratify_arg = y
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=stratify_arg
                    )
                    model = RandomForestClassifier(n_estimators=80, random_state=42)
                    model.fit(X_train, y_train)
                else:
                    X, y = make_classification(n_samples=500, n_features=20, random_state=42)
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                    model = RandomForestClassifier(n_estimators=50, random_state=42)
                    model.fit(X_train, y_train)
                
                # Generate report
                report = ProductionReadinessSimulator.generate_production_report(
                    model, X_train, y_train, X_test, y_test
                )
                
                # Overall status
                status_colors = {
                    'ready': '🟢',
                    'review_needed': '🟡',
                    'not_ready': '🔴'
                }
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Overall Status",
                        status_colors.get(report.overall_status, '⚪') + ' ' + report.overall_status.upper()
                    )
                with col2:
                    st.metric(
                        "Readiness Score",
                        f"{report.readiness_score:.1%}",
                        delta="Recommended: > 80%"
                    )
                with col3:
                    st.metric(
                        "Checks Passed",
                        f"{report.passed_checks}/{report.total_checks}",
                        delta=f"{(report.passed_checks/report.total_checks)*100:.0f}%"
                    )
                
                # Critical issues
                if report.critical_issues:
                    st.error("**🚨 Critical Issues (MUST FIX):**")
                    for issue in report.critical_issues:
                        st.write(f"- {issue}")
                
                # Warnings
                if report.warnings:
                    st.warning("**⚠️ Warnings (Should Review):**")
                    for warning in report.warnings:
                        st.write(f"- {warning}")
                
                # Detailed checks
                st.subheader("✅ Detailed Assessment")
                for check in report.checks:
                    status_icons = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}
                    
                    with st.expander(
                        f"{status_icons.get(check.status, '❓')} {check.check_name} - {check.status.upper()}",
                        expanded=(check.status == 'fail')
                    ):
                        st.write(f"**Status**: {check.message}")
                        st.info(f"**Recommendation**: {check.recommendation}")
                        if check.score is not None:
                            st.write(f"**Score**: {check.score:.3f}")
                
                # Deployment checklist
                st.subheader("📋 Deployment Checklist")
                checklist_items = [
                    ("Performance meets threshold", report.passed_checks > 0),
                    ("Model is stable across folds", any("stable" in c.message.lower() and c.status == 'pass' for c in report.checks)),
                    ("Data quality acceptable", any("quality" in c.check_name.lower() and c.status == 'pass' for c in report.checks)),
                    ("Latency within limits", any("latency" in c.check_name.lower() and c.status == 'pass' for c in report.checks)),
                    ("Model size reasonable", any("size" in c.check_name.lower() and c.status == 'pass' for c in report.checks)),
                    ("Monitoring configured", any("monitoring" in c.check_name.lower() for c in report.checks)),
                    ("Fairness assessed", any("fairness" in c.check_name.lower() for c in report.checks))
                ]
                
                for item, completed in checklist_items:
                    status = "✅" if completed else "⬜"
                    st.write(f"{status} {item}")


if __name__ == "__main__":
    main()
