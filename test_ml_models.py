"""Quick ML Model Test Script"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

print("=" * 60)
print("Testing Enhanced ML Models")
print("=" * 60)

# Test Supervised Models
print("\n1. Testing Supervised Learning Models...")
from src.ml.supervised import SupervisedLearningModel

# Test Classification
print("\n   Testing Classification Model...")
X_class, y_class = make_classification(n_samples=200, n_features=10, n_informative=5, random_state=42)
X_class_df = pd.DataFrame(X_class, columns=[f"feature_{i}" for i in range(10)])
y_class_series = pd.Series(y_class, name="target")

clf_model = SupervisedLearningModel(
    task_type="classification",
    model_type="random_forest",
    hyperparameters={"n_estimators": 50, "max_depth": 5},
    test_size=0.2,
    random_state=42
)

clf_model.train(X_class_df, y_class_series)
clf_results = clf_model.evaluation_results_
print(f"   ✓ Classification Model Trained")
print(f"     - Accuracy: {clf_results['accuracy']:.3f}")
print(f"     - F1 Score: {clf_results['f1']:.3f}")

# Test feature importance
fi = clf_model.get_feature_importance()
if fi is not None:
    print(f"     - Feature Importance: Top feature = {fi.iloc[0]['feature']}")

# Test Regression
print("\n   Testing Regression Model...")
X_reg, y_reg = make_regression(n_samples=200, n_features=10, n_informative=5, random_state=42)
X_reg_df = pd.DataFrame(X_reg, columns=[f"feature_{i}" for i in range(10)])
y_reg_series = pd.Series(y_reg, name="target")

reg_model = SupervisedLearningModel(
    task_type="regression",
    model_type="random_forest",
    hyperparameters={"n_estimators": 50, "max_depth": 5},
    test_size=0.2,
    random_state=42
)

reg_model.train(X_reg_df, y_reg_series)
reg_results = reg_model.evaluation_results_
print(f"   ✓ Regression Model Trained")
print(f"     - R² Score: {reg_results['r2']:.3f}")
print(f"     - RMSE: {reg_results['rmse']:.3f}")

# Test Unsupervised Models
print("\n2. Testing Unsupervised Learning Models...")
from src.ml.unsupervised import ClusteringModel, DimensionalityReducer

# Test Clustering
print("\n   Testing Clustering Model...")
X_cluster = pd.DataFrame(np.random.randn(100, 5), columns=[f"feature_{i}" for i in range(5)])

clustering_model = ClusteringModel(
    algorithm="kmeans",
    hyperparameters={"n_clusters": 3},
    random_state=42
)

clustering_model.fit(X_cluster)
cluster_results = clustering_model.evaluate()
print(f"   ✓ Clustering Model Fitted")
print(f"     - Number of Clusters: {cluster_results['n_clusters']}")
if cluster_results.get('silhouette_score'):
    print(f"     - Silhouette Score: {cluster_results['silhouette_score']:.3f}")

# Test Dimensionality Reduction
print("\n   Testing Dimensionality Reduction...")
dim_reducer = DimensionalityReducer(
    algorithm="pca",
    hyperparameters={"n_components": 2},
    random_state=42
)

dim_reducer.fit(X_cluster)
variance_info = dim_reducer.get_variance_explained()
print(f"   ✓ PCA Model Fitted")
if variance_info:
    print(f"     - Total Variance Explained: {variance_info['total_variance_explained']:.3f}")

print("\n" + "=" * 60)
print("✅ All ML Models Working Successfully!")
print("=" * 60)
