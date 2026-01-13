"""
Clustering Learning Module
Complete guide to unsupervised learning with elbow method, silhouette scores, and cluster interpretation.
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.cm as cm

from src.core.config import load_config
from src.core.ui import app_header, page_navigation, sidebar_dataset_status, instruction_block
from src.core.standardized_ui import (
    standard_section_header,
    beginner_tip,
    before_after_comparison,
    concept_explainer,
    common_mistakes_panel,
    metric_card,
)
from src.core.styles import inject_custom_css

st.set_page_config(page_title="Clustering Learning", layout="wide", initial_sidebar_state="expanded")

config = load_config()
inject_custom_css()

app_header(
    config,
    page_title="Clustering Learning",
    subtitle="Complete guide to unsupervised learning with elbow method, silhouette scores, and cluster interpretation",
    icon="🧑‍🎓"
)

sidebar_dataset_status(st.session_state.get("raw_df"), st.session_state.get("clean_df"))

instruction_block(
    title="Clustering Workflow",
    lines=[
        "**Load Data** → Upload CSV or use sample dataset (no target column needed)",
        "**Scale Features** → Normalize all features to same scale (critical for KMeans)",
        "**Find Optimal K** → Use elbow method to determine number of clusters",
        "**Run KMeans** → Partition data into K clusters",
        "**Evaluate Quality** → Silhouette score, inertia, cluster sizes",
        "**Interpret Clusters** → Analyze cluster characteristics and actionability"
    ]
)

# Step 1: Data Upload or Sample
standard_section_header("Step 1: Choose Your Data", "📂")
data_source = st.radio("Select data source:", ["Use sample dataset", "Upload CSV"], horizontal=True)
df = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
        except Exception as e:
            st.error(f"Failed to load CSV: {e}")
elif data_source == "Use sample dataset":
    # Three-cluster synthetic data
    np.random.seed(42)
    cluster1 = np.random.normal([0, 0], 1, (50, 2))
    cluster2 = np.random.normal([5, 5], 1.5, (50, 2))
    cluster3 = np.random.normal([2.5, -3], 1.2, (50, 2))
    data = np.vstack([cluster1, cluster2, cluster3])
    df = pd.DataFrame(data, columns=['Feature1', 'Feature2'])
    st.success("✅ Loaded sample clustering dataset (3 natural clusters).")

if df is not None:
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Rows", f"{len(df):,}", "Data points")
    with col2:
        metric_card("Columns", f"{len(df.columns)}", "Features")
    with col3:
        metric_card("Missing", f"{df.isnull().sum().sum()}", "Total NaN values")
    
    st.dataframe(df.head(10), width="stretch")
    beginner_tip("💡 **Note**: Clustering is *unsupervised* — no target column needed. It discovers groups automatically.")
else:
    st.info("Upload a CSV or use the sample dataset to begin.")
    st.stop()

# Step 2: Feature Selection
standard_section_header("Step 2: Select Features for Clustering", "🎯")
feature_cols = st.multiselect(
    "Feature columns (inputs):",
    options=list(df.columns),
    default=list(df.columns[:2]),
    help="Select 2+ numeric features. Non-numeric columns are skipped."
)
if not feature_cols:
    st.warning("Select at least 2 feature columns.")
    st.stop()

# Select only numeric features
numeric_features = df[feature_cols].select_dtypes(include=np.number).columns.tolist()
if len(numeric_features) < 2:
    st.error("Need at least 2 numeric features for clustering.")
    st.stop()

X = df[numeric_features].dropna()
if len(X) < 5:
    st.error(f"Need ≥5 valid rows for clustering, got {len(X)}.")
    st.stop()

st.info(f"Using **{len(numeric_features)}** features × **{len(X):,}** rows")

# Step 3: Feature Scaling
standard_section_header("Step 3: Feature Scaling", "📐")
st.write(
    "Scaling normalizes features to the same range. KMeans uses **distance**, so unscaled features with large ranges can dominate. "
    "For example, income (0-100,000) would overwhelm age (0-100) if not scaled."
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

before_after_comparison(
    before_df=X.head(5),
    after_df=pd.DataFrame(X_scaled[:5], columns=numeric_features),
    before_title="Raw Features (Unscaled)",
    after_title="Scaled Features (Mean=0, Std=1)"
)

concept_explainer(
    title="Why Scale Features?",
    explanation=(
        "StandardScaler transforms each feature to mean=0 and std=1. This ensures all features contribute equally to distance calculations. "
        "Without scaling, features with larger ranges (e.g., income) drowns out smaller ranges (e.g., age)."
    ),
    real_world_example=(
        "Customer segmentation: Age (18-80) and Income (20k-500k). Without scaling, income dominates. "
        "After scaling, both have equal influence on cluster formation."
    ),
)

# Step 4: Elbow Method
standard_section_header("Step 4: Elbow Method — Find Optimal K", "📉")
st.write(
    "The **elbow method** plots total within-cluster sum of squares (inertia) vs number of clusters (K). "
    "The 'elbow' point (where inertia stops dropping sharply) is often the optimal K."
)

col_e1, col_e2 = st.columns(2)
with col_e1:
    k_min = st.number_input("Min K", min_value=2, max_value=20, value=2, step=1)
with col_e2:
    k_max = st.number_input("Max K", min_value=k_min+1, max_value=30, value=10, step=1)

if st.button("🔍 Compute Elbow Curve", width="stretch", type="primary"):
    try:
        ks = list(range(int(k_min), int(k_max) + 1))
        inertias = []
        silhouette_scores = []
        
        progress_bar = st.progress(0)
        for i, kk in enumerate(ks):
            km = KMeans(n_clusters=kk, random_state=42, n_init=10)
            km.fit(X_scaled)
            inertias.append(km.inertia_)
            sil_score = silhouette_score(X_scaled, km.labels_)
            silhouette_scores.append(sil_score)
            progress_bar.progress((i + 1) / len(ks))
        
        # Plot both metrics
        col_plot1, col_plot2 = st.columns(2)
        
        with col_plot1:
            st.subheader("Inertia (Within-Cluster Sum of Squares)")
            fig_inertia = go.Figure()
            fig_inertia.add_trace(go.Scatter(
                x=ks, y=inertias,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='#0b5ed7', width=3),
                marker=dict(size=8)
            ))
            fig_inertia.update_layout(
                xaxis_title="K (Number of Clusters)",
                yaxis_title="Inertia (lower is better)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_inertia, width="stretch")
            st.caption("Look for the 'elbow' where inertia stops dropping sharply.")
        
        with col_plot2:
            st.subheader("Silhouette Score (Cluster Quality)")
            fig_silhouette = go.Figure()
            fig_silhouette.add_trace(go.Scatter(
                x=ks, y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette',
                line=dict(color='#28a745', width=3),
                marker=dict(size=8)
            ))
            fig_silhouette.update_layout(
                xaxis_title="K (Number of Clusters)",
                yaxis_title="Silhouette Score (-1 to +1)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig_silhouette, width="stretch")
            st.caption("Higher silhouette scores indicate better-defined clusters (+1 = perfect, 0 = overlapping).")
        
        # Find recommended K
        best_k_silhouette = ks[np.argmax(silhouette_scores)]
        st.success(f"✅ **Recommended K from Silhouette Score: {best_k_silhouette}** (score: {max(silhouette_scores):.3f})")
        
    except Exception as e:
        st.error(f"❌ Elbow computation failed: {e}")
else:
    st.info("Click 'Compute Elbow Curve' to analyze cluster quality across different K values.")

# Step 5: Choose K and Run KMeans
standard_section_header("Step 5: Run KMeans Clustering", "⚙️")

col_k1, col_k2 = st.columns([2, 1])
with col_k1:
    k = st.slider(
        "Number of clusters (K)",
        min_value=2,
        max_value=min(10, len(X) // 2),
        value=3,
        help="Recommended: Start with elbow method suggestion"
    )
with col_k2:
    random_state = st.number_input("Random seed (for reproducibility)", value=42, min_value=0)

if st.button("▶️ Run KMeans", type="primary", width="stretch"):
    with st.spinner("Computing clusters..."):
        kmeans = KMeans(n_clusters=k, random_state=int(random_state), n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        df_result = X.copy()
        df_result['Cluster'] = clusters
        
        st.success(f"✅ Clustering complete! Found **{k}** clusters.")
        
        # Metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            metric_card("Clusters", f"{k}", "groups")
        with col_m2:
            metric_card("Inertia", f"{kmeans.inertia_:.0f}", "within-cluster SS")
        with col_m3:
            sil_score = silhouette_score(X_scaled, clusters)
            metric_card("Silhouette", f"{sil_score:.3f}", "cluster quality")
        with col_m4:
            metric_card("Points", f"{len(X):,}", "total")
        
        # Cluster distribution
        cluster_counts = pd.Series(clusters).value_counts().sort_index()
        col_dist1, col_dist2 = st.columns([2, 1])
        
        with col_dist1:
            st.subheader("Cluster Sizes")
            fig_sizes = px.bar(
                x=cluster_counts.index.astype(str),
                y=cluster_counts.values,
                labels={'x': 'Cluster', 'y': 'Number of Points'},
                color=cluster_counts.index.astype(str)
            )
            st.plotly_chart(fig_sizes, width="stretch")
        
        with col_dist2:
            st.write("**Cluster Summary:**")
            for cluster_id, count in cluster_counts.items():
                pct = count / len(X) * 100
                st.write(f"Cluster {cluster_id}: {count:,} points ({pct:.1f}%)")
        
        # Cluster visualization (2D scatter)
        st.subheader("Cluster Visualization (2D)")
        if len(numeric_features) >= 2:
            fig_scatter = px.scatter(
                df_result,
                x=numeric_features[0],
                y=numeric_features[1],
                color='Cluster',
                title=f"KMeans Clustering ({k} clusters)",
                color_discrete_sequence=px.colors.qualitative.Set1,
                hover_data={numeric_features[0]: ':.2f', numeric_features[1]: ':.2f', 'Cluster': True}
            )
            st.plotly_chart(fig_scatter, width="stretch")
        
        # Silhouette plot
        st.subheader("Silhouette Analysis (Advanced)")
        st.write(
            "Silhouette coefficient for each point: +1 (well-clustered), 0 (on cluster boundary), -1 (misclassified). "
            "Negative values indicate points closer to other clusters than their own."
        )
        
        fig_silhouette, ax = plt.subplots(figsize=(10, 6))
        silhouette_vals = silhouette_samples(X_scaled, clusters)
        y_lower = 10
        colors = cm.nipy_spectral(clusters.astype(float) / k)
        
        for i in range(k):
            cluster_silhouette_vals = silhouette_vals[clusters == i]
            cluster_silhouette_vals.sort()
            size_cluster_i = cluster_silhouette_vals.shape[0]
            y_upper = y_lower + size_cluster_i
            ax.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                cluster_silhouette_vals,
                facecolor=colors[i],
                edgecolor=colors[i],
                alpha=0.7,
                label=f'Cluster {i}' if i == 0 else ""
            )
            y_lower = y_upper + 10
        
        ax.axvline(x=sil_score, color='red', linestyle='--', linewidth=2, label=f'Average Silhouette = {sil_score:.3f}')
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster")
        ax.set_title("Silhouette Plot for Each Cluster")
        st.pyplot(fig_silhouette)
        
        # Cluster profiles
        st.subheader("Cluster Profiles (Feature Statistics)")
        cluster_profiles = df_result.groupby('Cluster')[numeric_features].agg(['mean', 'std', 'min', 'max'])
        st.dataframe(cluster_profiles.round(2), width="stretch")
        
        st.dataframe(df_result.head(20), width="stretch")

else:
    st.info("👆 Adjust K and click 'Run KMeans' to generate clusters.")

st.divider()

# ============================================================================
# EDUCATIONAL CONTENT
# ============================================================================

st.markdown("## 🎓 Clustering Fundamentals")

concept_explainer(
    title="What is Clustering?",
    explanation=(
        "Clustering is *unsupervised learning*: grouping similar data points without a target column. "
        "KMeans is the simplest algorithm — it minimizes within-cluster distance by iteratively reassigning points to nearest cluster center."
    ),
    real_world_example=(
        "**Customer Segmentation**: Group customers by spending, frequency, product preference → tailor marketing for each segment. "
        "**Image Compression**: Cluster pixel colors → reduce palette size while keeping quality."
    ),
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### **Advantages of KMeans**")
    st.write(
        "✅ Fast and scalable (O(nkd) per iteration)\n"
        "✅ Easy to interpret (cluster centers)\n"
        "✅ Works on any numeric data\n"
        "✅ Elbow method guides K selection"
    )

with col2:
    st.markdown("### **Limitations**")
    st.write(
        "❌ Assumes spherical clusters\n"
        "❌ Sensitive to feature scaling\n"
        "❌ K must be chosen beforehand\n"
        "❌ Random initialization affects results"
    )

st.markdown("### **Elbow Method Interpretation**")
st.write(
    "The elbow method plots **inertia** (within-cluster sum of squares) vs K. "
    "As K increases, inertia always decreases. The 'elbow' is where the drop-off changes slope — "
    "increasing K beyond this point yields diminishing improvements."
)

st.markdown("### **Silhouette Score**")
st.write(
    "**Range: -1 to +1**\n"
    "- **+1**: Points well-separated and clustered\n"
    "- **0**: Points on cluster boundary\n"
    "- **-1**: Points in wrong cluster\n\n"
    "**Interpretation**: Average silhouette > 0.5 = good, > 0.7 = very good."
)

beginner_tip(
    "🎯 **Tip 1**: Always scale features before clustering. Otherwise, large-range features dominate."
)

beginner_tip(
    "🎯 **Tip 2**: Use elbow method + silhouette score together. Don't rely on just one metric."
)

beginner_tip(
    "🎯 **Tip 3**: Interpret clusters with domain knowledge. Statistics ≠ actionability."
)

common_mistakes_panel({
    "Forgetting to scale features": "KMeans uses distance; unscaled features with large ranges will dominate.",
    "Choosing K without guidance": "Use elbow method, silhouette scores, or domain knowledge—not guessing.",
    "Assuming clusters are meaningful": "Just because data splits doesn't mean clusters are actionable or interpretable.",
    "Ignoring negative silhouette values": "Points with negative silhouette may belong to different clusters. Review them.",
    "Re-running without random seed": "Set random_state for reproducibility; otherwise results vary between runs.",
})

page_navigation("7")
