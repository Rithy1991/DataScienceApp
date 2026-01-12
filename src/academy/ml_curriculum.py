"""
Enhanced Data Science Academy Module
====================================
Comprehensive ML curriculum with supervised and unsupervised learning.
"""

SUPERVISED_LEARNING_CURRICULUM = {
    "module_1": {
        "title": "Understanding Supervised Learning",
        "duration": "30 mins",
        "topics": [
            "What is supervised learning?",
            "Classification vs Regression",
            "Training/Testing paradigm",
            "Overfitting and underfitting",
            "Cross-validation basics"
        ],
        "key_concepts": [
            "Labeled data - each sample has a target label",
            "Binary vs Multi-class classification",
            "Continuous vs Discrete predictions",
            "Training set to test set ratio",
            "K-fold cross-validation for robust evaluation"
        ],
        "learning_outcomes": [
            "Understand when to use supervised learning",
            "Know the difference between classification and regression",
            "Understand bias-variance tradeoff",
            "Know how to split data properly",
            "Understand cross-validation"
        ],
        "code_example": """
# Basic supervised learning workflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Load data
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# 5. Feature importance
for feat, imp in zip(X.columns, model.feature_importances_):
    print(f'{feat}: {imp:.4f}')
"""
    },
    
    "module_2": {
        "title": "Exploratory Data Analysis for Supervised Learning",
        "duration": "45 mins",
        "topics": [
            "Data quality assessment",
            "Missing value handling",
            "Outlier detection and treatment",
            "Feature correlations",
            "Target distribution analysis",
            "Data imbalance issues"
        ],
        "key_concepts": [
            "Check data completeness and quality",
            "Missing values: drop, mean, median, forward-fill",
            "Outliers: IQR method, z-score, isolation forest",
            "High correlation = multicollinearity",
            "Imbalanced classes require special techniques",
            "Visualization reveals patterns humans miss"
        ],
        "learning_outcomes": [
            "Assess data quality comprehensively",
            "Handle missing values appropriately",
            "Detect and treat outliers",
            "Understand and address class imbalance",
            "Use visualizations for pattern discovery"
        ],
        "code_example": """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Check data quality
print(df.info())  # Data types and missing values
print(df.describe())  # Statistical summary

# Missing values
missing = df.isnull().sum() / len(df) * 100
print(missing[missing > 0])

# Handle missing values
df.fillna(df.mean(), inplace=True)  # Numeric
df['category'].fillna(df['category'].mode()[0], inplace=True)  # Categorical

# Outlier detection
Q1 = df['feature'].quantile(0.25)
Q3 = df['feature'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['feature'] < Q1 - 1.5*IQR) | (df['feature'] > Q3 + 1.5*IQR)]
print(f'Outliers found: {len(outliers)}')

# Target distribution
print(df['target'].value_counts())
print(df['target'].value_counts(normalize=True))
"""
    },
    
    "module_3": {
        "title": "Feature Engineering Mastery",
        "duration": "60 mins",
        "topics": [
            "Feature scaling and normalization",
            "Categorical encoding techniques",
            "Feature creation and interactions",
            "Polynomial features",
            "Feature selection methods",
            "Domain-specific feature engineering"
        ],
        "key_concepts": [
            "Scaling: StandardScaler, MinMaxScaler, RobustScaler",
            "Encoding: One-hot, Label, Target encoding",
            "Interactions: A × B, A / B, A + B",
            "Polynomial: X², X³ for non-linear patterns",
            "Selection: Correlation, importance, RFE, Mutual Information",
            "Domain knowledge matters - think about business context"
        ],
        "learning_outcomes": [
            "Scale features appropriately for different algorithms",
            "Encode categorical features correctly",
            "Create meaningful interaction features",
            "Select most important features",
            "Understand when to create complex vs simple features"
        ],
        "code_example": """
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.feature_selection import SelectKBest, f_classif

# 1. Scale numeric features
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X[numeric_cols])

# 2. Encode categorical features
encoder = OneHotEncoder(sparse_output=False)
X_cat = encoder.fit_transform(X[categorical_cols])

# 3. Create interaction features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_numeric)

# 4. Feature selection
selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)

# 5. Get feature importance
selector_mask = selector.get_support()
selected_features = X.columns[selector_mask]
print(f'Selected {len(selected_features)} features')
"""
    },
    
    "module_4": {
        "title": "Classification Models Deep Dive",
        "duration": "75 mins",
        "topics": [
            "Logistic Regression (linear classifier)",
            "Decision Trees (interpretable)",
            "Random Forest (ensemble)",
            "SVM (margin-based)",
            "Neural Networks (deep learning)",
            "Model selection criteria"
        ],
        "key_concepts": [
            "Logistic Regression: Linear decision boundary, interpretable coefficients",
            "Decision Trees: Easy to understand, prone to overfitting",
            "Random Forest: Ensemble of trees, reduces overfitting",
            "SVM: Finds optimal separating hyperplane",
            "Neural Networks: Non-linear transformations, powerful but complex",
            "Trade-off: Simplicity vs Accuracy"
        ],
        "learning_outcomes": [
            "Understand when to use each classification algorithm",
            "Know strengths and weaknesses of different models",
            "Train and tune classification models",
            "Interpret model predictions",
            "Handle binary and multi-class problems"
        ],
        "code_example": """
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100),
    'SVM': SVC(kernel='rbf')
}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    print(f'{name}:')
    print(f'  Accuracy: {accuracy_score(y_test, y_pred):.4f}')
    print(f'  Precision: {precision_score(y_test, y_pred, average=\"weighted\"):.4f}')
    print(f'  Recall: {recall_score(y_test, y_pred, average=\"weighted\"):.4f}')
    print(f'  F1: {f1_score(y_test, y_pred, average=\"weighted\"):.4f}')
"""
    },
    
    "module_5": {
        "title": "Regression Models and Techniques",
        "duration": "60 mins",
        "topics": [
            "Linear Regression (OLS)",
            "Ridge and Lasso (regularization)",
            "Ensemble Regressors",
            "Model validation for regression",
            "Residual analysis",
            "Handling multicollinearity"
        ],
        "key_concepts": [
            "Linear: Y = β₀ + β₁X₁ + ... + βₙXₙ",
            "Ridge (L2): Adds penalty to large coefficients, prevents overfitting",
            "Lasso (L1): Removes less important features, automatic selection",
            "Ensemble: Combine multiple models for better predictions",
            "Residuals: Differences between actual and predicted values",
            "Validate using R², RMSE, MAE, MAPE"
        ],
        "learning_outcomes": [
            "Understand and implement linear regression",
            "Apply regularization to prevent overfitting",
            "Use ensemble methods for regression",
            "Validate regression models properly",
            "Interpret regression results"
        ],
        "code_example": """
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Different regression models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100)
}

for name, model in models.items():
    # Train
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Evaluate
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f'{name}:')
    print(f'  R²: {r2:.4f} (1.0 is perfect)')
    print(f'  RMSE: {rmse:.4f} (lower is better)')
    print(f'  MAE: {mae:.4f} (average error)')
"""
    }
}

UNSUPERVISED_LEARNING_CURRICULUM = {
    "module_1": {
        "title": "Clustering Fundamentals",
        "duration": "45 mins",
        "topics": [
            "What is clustering?",
            "Use cases: segmentation, discovery",
            "Distance metrics: Euclidean, Manhattan, Cosine",
            "Clustering algorithm families",
            "Evaluating cluster quality"
        ],
        "key_concepts": [
            "Clustering groups similar unlabeled data",
            "No target variable needed",
            "Distance metrics define similarity",
            "Different algorithms work best for different shapes",
            "Silhouette, Calinski-Harabasz, Davies-Bouldin scores",
            "K-means: centroid-based; DBSCAN: density-based"
        ],
        "learning_outcomes": [
            "Understand clustering objectives",
            "Know different distance metrics",
            "Identify appropriate use cases",
            "Evaluate clustering results",
            "Understand algorithm strengths/weaknesses"
        ],
        "code_example": """
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score

# Scale data first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

# DBSCAN Clustering (density-based)
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_db = dbscan.fit_predict(X_scaled)

# Evaluate
silhouette_km = silhouette_score(X_scaled, labels_km)
davies_bouldin_km = davies_bouldin_score(X_scaled, labels_km)

print(f'K-Means:')
print(f'  Silhouette: {silhouette_km:.4f} (higher better, range [-1, 1])')
print(f'  Davies-Bouldin: {davies_bouldin_km:.4f} (lower better)')
print(f'  Clusters found: {len(set(labels_km))}')

# Cluster assignments
df['cluster'] = labels_km
print(df.groupby('cluster').size())
"""
    },
    
    "module_2": {
        "title": "K-Means Clustering Deep Dive",
        "duration": "50 mins",
        "topics": [
            "How K-Means works: centroid updates",
            "Choosing optimal K (elbow method)",
            "Initialization strategies",
            "Convergence and iterations",
            "Limitations and failure cases",
            "Real-world applications"
        ],
        "key_concepts": [
            "Algorithm: Initialize centroids → Assign points → Update centroids → Repeat",
            "Elbow method: Find knee in inertia vs K plot",
            "K-means++ initialization: smarter starting points",
            "Converges when centroids stop moving",
            "Works best for spherical, similar-sized clusters",
            "Fails with elongated or nested clusters"
        ],
        "learning_outcomes": [
            "Understand K-means algorithm steps",
            "Determine optimal number of clusters",
            "Implement and tune K-means",
            "Recognize when K-means fails",
            "Apply to real business problems"
        ],
        "code_example": """
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Find optimal K using elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

# Plot elbow
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'go-')
plt.xlabel('K (Number of Clusters)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by K')

plt.tight_layout()
plt.show()

# Train with optimal K
optimal_k = K_range[np.argmax(silhouette_scores)]
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans_final.fit_predict(X_scaled)
"""
    },
    
    "module_3": {
        "title": "Density-Based and Hierarchical Clustering",
        "duration": "55 mins",
        "topics": [
            "DBSCAN: density-based clustering",
            "Finding optimal EPS parameter",
            "Handling noise points",
            "Hierarchical clustering: agglomerative",
            "Dendrograms and linkage methods",
            "When to use each algorithm"
        ],
        "key_concepts": [
            "DBSCAN finds clusters of arbitrary shape",
            "Core points, border points, noise points",
            "EPS: neighborhood radius; MinPts: minimum points",
            "Hierarchical: builds tree of clusters",
            "Linkage: single, complete, average, ward",
            "Dendrograms visualize merging process"
        ],
        "learning_outcomes": [
            "Understand DBSCAN algorithm and parameters",
            "Find optimal EPS for DBSCAN",
            "Understand hierarchical clustering",
            "Interpret dendrograms",
            "Choose appropriate algorithm for data shape"
        ],
        "code_example": """
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# DBSCAN - requires tuning EPS
from sklearn.neighbors import NearestNeighbors

# Find optimal EPS using k-distance graph
k = 5
neighbors = NearestNeighbors(n_neighbors=k)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1], axis=0)

plt.plot(distances)
plt.ylabel('K-distance')
plt.title('K-distance Graph for EPS Selection')
plt.axhline(y=0.5, color='r', linestyle='--')
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_scaled)
n_clusters = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
print(f'Found {n_clusters} clusters, {sum(labels_dbscan == -1)} noise points')

# Hierarchical Clustering
linkage_matrix = linkage(X_scaled, method='ward')
dendrogram(linkage_matrix)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index or (Cluster Size)')
plt.ylabel('Distance')
plt.show()

# Cut dendrogram at threshold
hierarchical = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hier = hierarchical.fit_predict(X_scaled)
"""
    },
    
    "module_4": {
        "title": "Dimensionality Reduction Techniques",
        "duration": "60 mins",
        "topics": [
            "Why reduce dimensions?",
            "PCA: Principal Component Analysis",
            "Understanding explained variance",
            "t-SNE: visualization powerhouse",
            "UMAP: modern alternative",
            "Applications and trade-offs"
        ],
        "key_concepts": [
            "PCA: Linear transformation maximizing variance",
            "Explained variance: % of information in each component",
            "t-SNE: Non-linear, preserves local structure, slow",
            "UMAP: Non-linear, faster than t-SNE, global + local structure",
            "PCA for reduction; t-SNE/UMAP for visualization",
            "Choose dimensions based on explained variance threshold"
        ],
        "learning_outcomes": [
            "Implement PCA and interpret results",
            "Determine number of components needed",
            "Use t-SNE for beautiful visualizations",
            "Compare dimensionality reduction methods",
            "Apply reduction to solve real problems"
        ],
        "code_example": """
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# PCA: Linear, fast
pca = PCA(n_components=0.95)  # Keep 95% variance
X_pca = pca.fit_transform(X_scaled)

print(f'Original features: {X_scaled.shape[1]}')
print(f'PCA components: {X_pca.shape[1]}')
print(f'Explained variance ratio: {pca.explained_variance_ratio_}')
print(f'Cumulative variance: {np.cumsum(pca.explained_variance_ratio_)}')

# Plot explained variance
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
        pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Explained Variance')

plt.subplot(1, 2, 2)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.axhline(y=0.95, color='r', linestyle='--', label='95% threshold')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Cumulative Explained Variance')
plt.legend()

plt.tight_layout()
plt.show()

# t-SNE: Non-linear, beautiful visualization
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

# Visualize
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis')
plt.title('t-SNE Visualization')
plt.colorbar()
plt.show()
"""
    },
    
    "module_5": {
        "title": "Anomaly Detection & Pattern Discovery",
        "duration": "50 mins",
        "topics": [
            "Detecting outliers and anomalies",
            "Isolation Forest algorithm",
            "Local Outlier Factor (LOF)",
            "Statistical methods",
            "Business applications",
            "Interpreting anomalies"
        ],
        "key_concepts": [
            "Anomalies: rare, unusual, suspicious points",
            "Isolation Forest: isolates outliers quickly",
            "LOF: Local density-based detection",
            "Statistical: Z-score, IQR, Mahalanobis distance",
            "Contamination rate: expected % of anomalies",
            "Validate findings with domain experts"
        ],
        "learning_outcomes": [
            "Implement anomaly detection algorithms",
            "Choose appropriate contamination rate",
            "Evaluate anomaly detection results",
            "Interpret discovered anomalies",
            "Apply to fraud, quality control, security"
        ],
        "code_example": """
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

# Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies_if = iso_forest.fit_predict(X_scaled)
scores_if = iso_forest.score_samples(X_scaled)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
anomalies_lof = lof.fit_predict(X_scaled)
scores_lof = lof.negative_outlier_factor_

# Results
n_anomalies_if = sum(anomalies_if == -1)
n_anomalies_lof = sum(anomalies_lof == -1)

print(f'Isolation Forest: {n_anomalies_if} anomalies detected')
print(f'Local Outlier Factor: {n_anomalies_lof} anomalies detected')

# Visualize
df['anomaly_if'] = anomalies_if
df['anomaly_score'] = scores_if

# Find most anomalous points
most_anomalous = df.nsmallest(10, 'anomaly_score')
print('\\nMost anomalous points:')
print(most_anomalous)

# Visualize with PCA
pca_viz = PCA(n_components=2)
X_viz = pca_viz.fit_transform(X_scaled)

plt.scatter(X_viz[:, 0], X_viz[:, 1], c=anomalies_if, cmap='RdYlGn_r')
plt.title('Anomalies (Red)')
plt.colorbar()
plt.show()
"""
    }
}

def get_curriculum():
    """Get complete curriculum."""
    return {
        "supervised": SUPERVISED_LEARNING_CURRICULUM,
        "unsupervised": UNSUPERVISED_LEARNING_CURRICULUM
    }

def format_module(module_data):
    """Format module for display."""
    return f"""
## {module_data['title']}
⏱️ **Duration:** {module_data['duration']}

### Topics
{chr(10).join(f"- {topic}" for topic in module_data['topics'])}

### Key Concepts
{chr(10).join(f"- {concept}" for concept in module_data['key_concepts'])}

### Learning Outcomes
{chr(10).join(f"- {outcome}" for outcome in module_data['learning_outcomes'])}

### Code Example
```python
{module_data['code_example']}
```
"""
