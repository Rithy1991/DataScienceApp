"""
Comprehensive Unsupervised Learning Page
==========================================
Clustering, dimensionality reduction, and anomaly detection.
Discover patterns without labels: K-Means, DBSCAN, Hierarchical Clustering, PCA, t-SNE, UMAP
"""

import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="DataScope Pro - Unsupervised Learning",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.core.config import load_config
from src.core.ui import sidebar_dataset_status, instruction_block, page_navigation
from src.core.standardized_ui import (
    standard_section_header,
    beginner_tip,
    concept_explainer,
)
from src.core.styles import inject_custom_css
from src.core.ai_helper import ai_sidebar_assistant
from src.ml.unsupervised import (
    ClusteringModel,
    DimensionalityReducer,
    AnomalyDetector,
    find_optimal_clusters,
    profile_clusters,
)
from sklearn.preprocessing import StandardScaler

config = load_config()
inject_custom_css()
ai_sidebar_assistant()

warnings.filterwarnings("ignore")


# ==================== Header ====================
st.markdown(
    """
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: #f8fafc; 
                padding: 20px 24px; border-radius: 12px; margin-bottom: 20px;">
        <div style="font-size: 28px; font-weight: 900; margin-bottom: 8px;">🔍 Unsupervised Learning Explorer</div>
        <div style="font-size: 16px; opacity: 0.95;">
            Discover Hidden Patterns | Clustering • Dimensionality Reduction • Anomaly Detection
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ==================== Instructions ====================
instruction_block(
    "Unsupervised Learning Workflow",
    [
        "🎯 Choose task: Clustering (group similar items), Dimensionality Reduction (compress data), or Anomaly Detection",
        "📊 Load your dataset - no target/label needed!",
        "📈 Explore optimal parameters (number of clusters, dimensions, etc.)",
        "🤖 Apply algorithms: K-Means, DBSCAN, Hierarchical, PCA, t-SNE, UMAP",
        "📊 Visualize results and understand cluster characteristics",
        "💡 Interpret patterns and discover insights",
        "💾 Export clustering results or dimensionality-reduced data",
    ],
)

# ==================== Session State ====================
if "ul_task" not in st.session_state:
    st.session_state.ul_task = "clustering"
if "ul_clusterer" not in st.session_state:
    st.session_state.ul_clusterer = None
if "ul_reducer" not in st.session_state:
    st.session_state.ul_reducer = None
if "ul_labels" not in st.session_state:
    st.session_state.ul_labels = None

# ==================== Step 1: Task Selection ====================
standard_section_header("Step 1: Choose Your Unsupervised Task", "🎯")

task_col1, task_col2, task_col3 = st.columns(3)

with task_col1:
    st.markdown("### Clustering 📊")
    st.markdown("""
    **Group Similar Items**
    - Customer segmentation
    - Document clustering
    - Image segmentation
    
    **Methods:** K-Means, DBSCAN, Hierarchical
    """)
    if st.button("Use Clustering", key="btn_cluster", use_container_width=True):
        st.session_state.ul_task = "clustering"
        st.rerun()

with task_col2:
    st.markdown("### Dimensionality Reduction 📉")
    st.markdown("""
    **Compress High-Dimensional Data**
    - Visualization of complex data
    - Feature reduction
    - Noise removal
    
    **Methods:** PCA, t-SNE, UMAP
    """)
    if st.button("Use Dimensionality Reduction", key="btn_dimred", use_container_width=True):
        st.session_state.ul_task = "dimensionality_reduction"
        st.rerun()

with task_col3:
    st.markdown("### Anomaly Detection 🚨")
    st.markdown("""
    **Find Unusual Items**
    - Fraud detection
    - Outlier identification
    - Quality control
    
    **Methods:** Isolation Forest, Local Outlier Factor
    """)
    if st.button("Use Anomaly Detection", key="btn_anomaly", use_container_width=True):
        st.session_state.ul_task = "anomaly_detection"
        st.rerun()

task_type = st.session_state.ul_task

# ==================== Step 2: Data Loading ====================
standard_section_header("Step 2: Load Your Data", "📂")

sidebar_dataset_status(st.session_state.get("raw_df"), st.session_state.get("clean_df"))

tab_upload, tab_sample = st.tabs(["📁 Upload File", "📦 Sample Dataset"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv", "xlsx"],
        help="File with numeric data for clustering/reduction"
    )
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file)
            st.session_state.raw_df = df
            st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns")
            st.dataframe(df.head(10), use_container_width=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")

with tab_sample:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Load Iris (Clustering)", use_container_width=True):
            from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            st.session_state.raw_df = df
            st.rerun()
    
    with col2:
        if st.button("Load Digits (Dim Reduction)", use_container_width=True):
            from sklearn.datasets import load_digits
            digits = load_digits()
            df = pd.DataFrame(digits.data, columns=[f"pixel_{i}" for i in range(64)])
            st.session_state.raw_df = df
            st.session_state.ul_task = "dimensionality_reduction"
            st.rerun()
    
    with col3:
        if st.button("Load Blobs (Clustering)", use_container_width=True):
            from sklearn.datasets import make_blobs
            X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)
            df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
            st.session_state.raw_df = df
            st.rerun()

# Get data
df = st.session_state.get("raw_df")
if df is None:
    st.info("👆 Upload a dataset or select a sample to continue")
    st.stop()

# Select numeric features only
numeric_df = df.select_dtypes(include=[np.number])
if numeric_df.empty:
    st.error("No numeric columns found in dataset")
    st.stop()

# ==================== Step 3: Data Overview ====================
standard_section_header("Step 3: Data Overview", "🔍")

col_shape, col_numeric, col_missing = st.columns(3)

with col_shape:
    st.metric("Rows", f"{numeric_df.shape[0]:,}")

with col_numeric:
    st.metric("Numeric Features", f"{numeric_df.shape[1]}")

with col_missing:
    missing_pct = (numeric_df.isna().sum().sum() / (numeric_df.shape[0] * numeric_df.shape[1])) * 100
    st.metric("Missing %", f"{missing_pct:.2f}%")

# Handle missing values
if numeric_df.isna().sum().sum() > 0:
    numeric_df = numeric_df.fillna(numeric_df.mean())
    st.info("✅ Missing values handled with mean imputation")

# ==================== CLUSTERING WORKFLOW ====================
if task_type == "clustering":
    standard_section_header("Clustering Analysis", "📊")
    
    concept_explainer(
        "What is Clustering?",
        "Grouping similar data points together without pre-existing labels.",
        "Customer segments based on behavior, documents grouped by topic, images organized by content"
    )
    
    # Step 1: Find optimal clusters
    st.markdown("### Step 1: Find Optimal Number of Clusters")
    
    col_k1, col_k2 = st.columns(2)
    
    with col_k1:
        if st.button("🔍 Analyze Optimal Clusters", use_container_width=True, key="btn_optimal_k"):
            with st.spinner("Analyzing cluster numbers..."):
                k_analysis = find_optimal_clusters(numeric_df, k_range=(2, 10))
                st.session_state.k_analysis = k_analysis
                st.success("✅ Analysis complete")
    
    with col_k2:
        st.info(f"🎯 Suggested cluster count: **{st.session_state.get('k_analysis', {}).get('elbow_k', 'N/A')}**")
    
    if "k_analysis" in st.session_state:
        k_data = st.session_state.k_analysis
        
        # Elbow plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=k_data['k_range'],
            y=k_data['inertias'],
            mode='lines+markers',
            name='Inertia (Lower is Better)',
            line=dict(color='blue'),
            marker=dict(size=10)
        ))
        
        fig.add_vline(
            x=k_data['elbow_k'],
            line_dash="dash",
            line_color="red",
            annotation_text=f"Elbow: k={k_data['elbow_k']}"
        )
        
        fig.update_layout(
            title="Elbow Method for Optimal K",
            xaxis_title="Number of Clusters (K)",
            yaxis_title="Inertia",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Step 2: Choose clustering algorithm
    st.markdown("### Step 2: Choose Clustering Algorithm")
    
    col_algo1, col_algo2, col_algo3 = st.columns(3)
    
    with col_algo1:
        algorithm = st.selectbox(
            "Select algorithm:",
            ["kmeans", "hierarchical", "dbscan", "spectral"],
            help="kmeans: fast, needs k; dbscan: finds density-based clusters"
        )
    
    with col_algo2:
        if algorithm == "dbscan":
            eps = st.slider("EPS (neighborhood distance):", 0.1, 2.0, 0.5, 0.1)
            n_clusters_param = eps
        else:
            n_clusters_param = st.slider("Number of clusters:", 2, 10, 3)
    
    with col_algo3:
        if st.button("🚀 Run Clustering", use_container_width=True, type="primary"):
            with st.spinner("Clustering data..."):
                clusterer = ClusteringModel(
                    algorithm=algorithm,
                    n_clusters=n_clusters_param if algorithm != "dbscan" else 3,
                    random_state=42
                )
                
                clusterer.fit(numeric_df)
                st.session_state.ul_clusterer = clusterer
                st.session_state.ul_labels = clusterer.labels_
                
                eval_results = clusterer.evaluate(numeric_df)
                st.session_state.cluster_eval = eval_results
                
                st.success("✅ Clustering complete!")
    
    # Step 3: Results visualization
    if st.session_state.ul_clusterer:
        st.markdown("### Step 3: Clustering Results")
        
        clusterer = st.session_state.ul_clusterer
        eval_results = st.session_state.cluster_eval
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric("Number of Clusters", eval_results.get('n_clusters', 'N/A'))
        
        with col_m2:
            if 'silhouette_score' in eval_results:
                st.metric("Silhouette Score", f"{eval_results['silhouette_score']:.4f}")
        
        with col_m3:
            if 'davies_bouldin_score' in eval_results:
                st.metric("Davies-Bouldin Score", f"{eval_results['davies_bouldin_score']:.4f}")
        
        with col_m4:
            if 'calinski_harabasz_score' in eval_results:
                st.metric("Calinski-Harabasz Score", f"{eval_results['calinski_harabasz_score']:.4f}")
        
        # Visualization: 2D projection
        if numeric_df.shape[1] >= 2:
            # Ensure labels match the data
            if len(clusterer.labels_) == len(numeric_df):
                reducer = DimensionalityReducer(algorithm='pca', n_components=2)
                X_2d = reducer.fit(numeric_df).X_transformed_
                
                fig = px.scatter(
                    x=X_2d[:, 0],
                    y=X_2d[:, 1],
                    color=clusterer.labels_.astype(str),
                    title="2D Projection of Clusters (PCA)",
                    labels={'x': 'PC1', 'y': 'PC2', 'color': 'Cluster'},
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ Data mismatch: Current data has {len(numeric_df)} rows but clustering labels have {len(clusterer.labels_)} rows. Please re-run clustering.")
        
        # Cluster profiles
        if len(clusterer.labels_) == len(numeric_df):
            with st.expander("📊 Cluster Profiles", expanded=True):
                profiles = profile_clusters(numeric_df, clusterer.labels_)
                st.dataframe(profiles, use_container_width=True)
        else:
            st.info("ℹ️ Cluster profiles unavailable due to data mismatch. Please re-run clustering.")
        
        # Cluster sizes
        cluster_sizes = pd.Series(clusterer.labels_).value_counts().sort_index()
        
        fig = px.bar(
            x=cluster_sizes.index.astype(str),
            y=cluster_sizes.values,
            title="Cluster Sizes",
            labels={'x': 'Cluster', 'y': 'Size'},
            color=cluster_sizes.values,
            color_continuous_scale='Viridis'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Export results
        with st.expander("💾 Export Results"):
            export_df = numeric_df.copy()
            export_df['cluster'] = clusterer.labels_
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Data with Cluster Labels",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================== DIMENSIONALITY REDUCTION WORKFLOW ====================
elif task_type == "dimensionality_reduction":
    standard_section_header("Dimensionality Reduction Analysis", "📉")
    
    concept_explainer(
        "Why Dimensionality Reduction?",
        "Compress high-dimensional data while preserving important patterns.",
        "Visualize 64-pixel images in 2D, reduce 100 features to 10 while keeping 95% information"
    )
    
    st.markdown("### Choose Reduction Algorithm")
    
    col_dr1, col_dr2 = st.columns(2)
    
    with col_dr1:
        dr_algorithm = st.selectbox(
            "Select algorithm:",
            ["pca", "tsne", "mds", "lle"],
            help="PCA: linear, fast; t-SNE: non-linear, beautiful visualizations"
        )
    
    with col_dr2:
        n_components = st.slider("Number of output dimensions:", 2, min(10, numeric_df.shape[1]), 2)
    
    if st.button("🚀 Reduce Dimensions", use_container_width=True, type="primary"):
        with st.spinner("Reducing dimensions..."):
            reducer = DimensionalityReducer(
                algorithm=dr_algorithm,
                n_components=n_components,
                random_state=42
            )
            
            X_reduced = reducer.fit(numeric_df).X_transformed_
            st.session_state.ul_reducer = reducer
            st.session_state.X_reduced = X_reduced
            
            st.success("✅ Dimensionality reduction complete!")
    
    # Results
    if "X_reduced" in st.session_state:
        st.markdown("### Reduction Results")
        
        reducer = st.session_state.ul_reducer
        X_reduced = st.session_state.X_reduced
        
        # Variance explained (PCA only)
        var_info = reducer.get_variance_explained()
        if var_info:
            st.metric(
                "Variance Explained",
                f"{var_info['total_variance_explained']:.2%}",
                f"of {numeric_df.shape[1]} original dimensions"
            )
            
            # Cumulative variance plot
            if 'cumulative_variance_ratio' in var_info:
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    y=var_info['cumulative_variance_ratio'],
                    mode='lines+markers',
                    name='Cumulative Variance'
                ))
                
                fig.add_hline(y=0.95, line_dash="dash", line_color="red", annotation_text="95% threshold")
                
                fig.update_layout(
                    title="Cumulative Variance Explained",
                    xaxis_title="Number of Components",
                    yaxis_title="Cumulative Variance Ratio",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        # Visualization
        if n_components >= 2:
            # 2D or 3D plot
            if n_components == 2:
                fig = px.scatter(
                    x=X_reduced[:, 0],
                    y=X_reduced[:, 1],
                    title=f"2D {dr_algorithm.upper()} Projection",
                    labels={'x': f'{dr_algorithm.upper()} 1', 'y': f'{dr_algorithm.upper()} 2'}
                )
            elif n_components == 3:
                fig = px.scatter_3d(
                    x=X_reduced[:, 0],
                    y=X_reduced[:, 1],
                    z=X_reduced[:, 2],
                    title=f"3D {dr_algorithm.upper()} Projection",
                    labels={
                        'x': f'{dr_algorithm.upper()} 1',
                        'y': f'{dr_algorithm.upper()} 2',
                        'z': f'{dr_algorithm.upper()} 3'
                    }
                )
            else:
                # Show first 2 dimensions
                fig = px.scatter(
                    x=X_reduced[:, 0],
                    y=X_reduced[:, 1],
                    title=f"{dr_algorithm.upper()} Projection (First 2 Dimensions)",
                    labels={'x': f'{dr_algorithm.upper()} 1', 'y': f'{dr_algorithm.upper()} 2'}
                )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Export
        with st.expander("💾 Export Reduced Data"):
            reduced_df = pd.DataFrame(
                X_reduced,
                columns=[f"{dr_algorithm.upper()}_{i+1}" for i in range(n_components)]
            )
            
            csv = reduced_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Reduced Data",
                data=csv,
                file_name=f"dimensionality_reduced_{dr_algorithm}.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================== ANOMALY DETECTION WORKFLOW ====================
elif task_type == "anomaly_detection":
    standard_section_header("Anomaly Detection", "🚨")
    
    concept_explainer(
        "Detecting Outliers",
        "Identify unusual or suspicious data points that deviate from normal patterns.",
        "Credit card fraud detection, detecting sensor failures, quality control in manufacturing"
    )
    
    st.markdown("### Choose Anomaly Detection Method")
    
    col_ad1, col_ad2 = st.columns(2)
    
    with col_ad1:
        ad_method = st.selectbox(
            "Detection method:",
            ["isolation_forest", "local_outlier_factor", "elliptic_envelope"],
            help="Isolation Forest: fast, scalable; LOF: density-based"
        )
    
    with col_ad2:
        contamination = st.slider(
            "Contamination (% anomalies):",
            1, 50, 10,
            help="Expected percentage of anomalies in data"
        )
    
    if st.button("🚀 Detect Anomalies", use_container_width=True, type="primary"):
        with st.spinner("Detecting anomalies..."):
            detector = AnomalyDetector(method=ad_method, contamination=contamination / 100)
            detector.fit(numeric_df)
            
            predictions = detector.predict(numeric_df)
            scores = detector.score(numeric_df)
            
            st.session_state.anomaly_detector = detector
            st.session_state.anomaly_predictions = predictions
            st.session_state.anomaly_scores = scores
            
            st.success("✅ Anomaly detection complete!")
    
    # Results
    if "anomaly_predictions" in st.session_state:
        st.markdown("### Anomaly Detection Results")
        
        predictions = st.session_state.anomaly_predictions
        scores = st.session_state.anomaly_scores
        
        n_anomalies = (predictions == -1).sum()
        n_normal = (predictions == 1).sum()
        
        col_a1, col_a2, col_a3 = st.columns(3)
        
        with col_a1:
            st.metric("Normal Points", n_normal)
        
        with col_a2:
            st.metric("Anomalies", n_anomalies)
        
        with col_a3:
            st.metric("Anomaly %", f"{(n_anomalies / len(predictions) * 100):.2f}%")
        
        # Visualization
        if numeric_df.shape[1] >= 2:
            reducer = DimensionalityReducer(algorithm='pca', n_components=2)
            X_2d = reducer.fit(numeric_df).X_transformed_
            
            fig = px.scatter(
                x=X_2d[:, 0],
                y=X_2d[:, 1],
                color=['Anomaly' if pred == -1 else 'Normal' for pred in predictions],
                title="Anomalies Detected (2D PCA Projection)",
                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Type'},
                color_discrete_map={'Normal': 'blue', 'Anomaly': 'red'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly scores distribution
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=scores,
            nbinsx=50,
            name='Anomaly Scores',
            color_discrete_sequence=['blue']
        ))
        
        fig.update_layout(
            title="Anomaly Score Distribution",
            xaxis_title="Anomaly Score",
            yaxis_title="Frequency",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomalies detail
        with st.expander("🔍 Detected Anomalies", expanded=False):
            anomaly_indices = np.where(predictions == -1)[0]
            
            if len(anomaly_indices) > 0:
                anomalies_df = numeric_df.iloc[anomaly_indices].copy()
                anomalies_df['anomaly_score'] = scores[anomaly_indices]
                anomalies_df = anomalies_df.sort_values('anomaly_score')
                
                st.dataframe(anomalies_df, use_container_width=True)
            else:
                st.info("No anomalies detected with current settings")
        
        # Export
        with st.expander("💾 Export Results"):
            export_df = numeric_df.copy()
            export_df['anomaly_prediction'] = predictions
            export_df['anomaly_score'] = scores
            
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Data with Anomaly Labels",
                data=csv,
                file_name="anomaly_detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )

# ==================== Footer ====================
page_navigation("21")
