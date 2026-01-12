# 🚀 Supervised & Unsupervised Machine Learning Complete Guide

## Overview

This comprehensive guide covers **Supervised and Unsupervised Learning** from fundamentals to production deployment. Treat this as a **$10M-quality professional ML platform**.

---

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Supervised Learning](#supervised-learning)
3. [Unsupervised Learning](#unsupervised-learning)
4. [Feature Engineering](#feature-engineering)
5. [Model Evaluation](#model-evaluation)
6. [Best Practices](#best-practices)
7. [Complete Examples](#complete-examples)

---

## Getting Started

### Installation

```bash
# Install required packages
pip install -r requirements.txt

# Optional for advanced features
pip install xgboost lightgbm
pip install umap-learn  # For dimensionality reduction
```

### Import Core Modules

```python
from src.ml.supervised import SupervisedLearningModel, DataPreprocessor
from src.ml.unsupervised import ClusteringModel, DimensionalityReducer
from src.ml.feature_engineering import FeatureCreator, FeatureSelector
from src.ml.evaluation import ClassificationEvaluator, RegressionEvaluator
```

---

## Supervised Learning

### What is Supervised Learning?

Supervised learning uses **labeled data** (each example has a known target) to build models that can predict future targets.

**Examples:**
- **Classification**: Email spam filter, disease diagnosis, customer churn prediction
- **Regression**: House price prediction, stock forecasting, sales estimation

### Complete Workflow

#### Step 1: Prepare Data

```python
from src.ml.supervised import DataPreprocessor
import pandas as pd

# Create preprocessor
preprocessor = DataPreprocessor(
    scaler_type="standard",      # Scale to mean=0, std=1
    handle_missing="mean",        # Use mean for missing values
    encode_type="onehot",         # One-hot encode categories
    outlier_method="iqr"          # Remove outliers using IQR
)

# Fit and transform
df_processed = preprocessor.fit_transform(df)
```

#### Step 2: Split Features and Target

```python
X = df_processed[['feature1', 'feature2', 'feature3']]
y = df_processed['target_column']

# For classification - ensure target is categorical
# For regression - ensure target is numeric
```

#### Step 3: Train Model

```python
from src.ml.supervised import SupervisedLearningModel

# Classification example
model = SupervisedLearningModel(
    task_type="classification",
    model_type="random_forest",
    test_size=0.2,
    random_state=42
)

# Train the model
model.train(X, y)

# Get evaluation results
results = model.evaluate()
print(f"Accuracy: {results['accuracy']:.4f}")
print(f"F1-Score: {results['f1']:.4f}")
```

#### Step 4: Analyze Results

```python
# Get feature importance
importance_df = model.get_feature_importance(X)
print(importance_df.head(10))

# Get predictions
predictions = model.predict(X_new_data)

# Get prediction probabilities (classification)
probabilities = model.predict_proba(X_new_data)
```

#### Step 5: Cross-Validate

```python
# Robust evaluation with cross-validation
cv_results = model.cross_validate(X, y, cv=5)
print(f"Mean Accuracy: {cv_results['test_accuracy'].mean():.4f}")
print(f"Std: {cv_results['test_accuracy'].std():.4f}")
```

### Available Models

#### Classification Models
```python
# All these are supported:
models = [
    'logistic_regression',    # Fast, interpretable
    'random_forest',          # Powerful ensemble
    'gradient_boosting',      # Sequential boosting
    'xgboost',               # Optimized boosting
    'lightgbm',              # Fast boosting
    'svm',                   # Margin-based
    'knn',                   # Distance-based
    'naive_bayes',           # Probabilistic
    'mlp'                    # Neural network
]
```

#### Regression Models
```python
models = [
    'linear',                # Simple baseline
    'ridge',                 # L2 regularization
    'lasso',                 # L1 regularization  
    'elasticnet',            # Combined
    'random_forest',         # Ensemble
    'gradient_boosting',     # Sequential
    'xgboost',              # Optimized
    'svm',                  # Support Vector
    'mlp'                   # Neural network
]
```

### Hyperparameter Optimization

```python
from src.ml.supervised import HyperparameterOptimizer

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# Optimize
optimizer = HyperparameterOptimizer(
    model=model,
    search_type='grid',  # or 'random'
    cv=5
)

results = optimizer.optimize(X, y, param_grid)
print(f"Best parameters: {results['best_params']}")
print(f"Best score: {results['best_score']:.4f}")
```

---

## Unsupervised Learning

### What is Unsupervised Learning?

Unsupervised learning discovers patterns in **unlabeled data** without pre-existing targets.

**Types:**
- **Clustering**: Group similar items together
- **Dimensionality Reduction**: Compress high-dimensional data
- **Anomaly Detection**: Find unusual items

### Clustering

#### K-Means Clustering

```python
from src.ml.unsupervised import ClusteringModel, find_optimal_clusters
from sklearn.preprocessing import StandardScaler

# Always scale first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal number of clusters
k_analysis = find_optimal_clusters(X_scaled, k_range=(2, 10))
print(f"Suggested K: {k_analysis['elbow_k']}")

# Train clustering model
clusterer = ClusteringModel(
    algorithm='kmeans',
    n_clusters=3,
    random_state=42
)

clusterer.fit(X_scaled)

# Evaluate
results = clusterer.evaluate(X_scaled)
print(f"Silhouette Score: {results['silhouette_score']:.4f}")  # Higher is better
print(f"Davies-Bouldin Index: {results['davies_bouldin_score']:.4f}")  # Lower is better

# Get cluster labels
labels = clusterer.labels_

# Analyze clusters
cluster_profiles = profile_clusters(X, labels)
print(cluster_profiles)
```

#### DBSCAN (Density-Based)

```python
clusterer = ClusteringModel(
    algorithm='dbscan',
    n_clusters=0.5,  # This is EPS parameter
    random_state=42
)

clusterer.fit(X_scaled)

# DBSCAN marks noise as -1
n_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
n_noise = list(clusterer.labels_).count(-1)

print(f"Clusters found: {n_clusters}")
print(f"Noise points: {n_noise}")
```

### Dimensionality Reduction

#### PCA (Principal Component Analysis)

```python
from src.ml.unsupervised import DimensionalityReducer

# Reduce to 2 dimensions for visualization
reducer = DimensionalityReducer(
    algorithm='pca',
    n_components=2,
    random_state=42
)

X_2d = reducer.fit(X_scaled).X_transformed_

# Check variance explained
variance_info = reducer.get_variance_explained()
print(f"Total variance explained: {variance_info['total_variance_explained']:.2%}")
```

#### t-SNE (for Beautiful Visualizations)

```python
reducer = DimensionalityReducer(
    algorithm='tsne',
    n_components=2,
    random_state=42
)

X_tsne = reducer.fit(X_scaled).X_transformed_

# t-SNE is great for visualization but slow and has no 'transform' method
```

#### UMAP (Modern Alternative)

```python
reducer = DimensionalityReducer(
    algorithm='umap',
    n_components=2,
    random_state=42
)

X_umap = reducer.fit(X_scaled).X_transformed_

# UMAP is faster than t-SNE and has better global structure preservation
```

### Anomaly Detection

```python
from src.ml.unsupervised import AnomalyDetector

# Create detector
detector = AnomalyDetector(
    method='isolation_forest',  # or 'local_outlier_factor'
    contamination=0.1  # Expected % of anomalies
)

detector.fit(X_scaled)

# Get predictions: 1=normal, -1=anomaly
predictions = detector.predict(X_scaled)

# Get anomaly scores
scores = detector.score(X_scaled)

# Find anomalies
anomalies = X[predictions == -1]
print(f"Found {len(anomalies)} anomalies")
```

---

## Feature Engineering

### Scaling and Normalization

```python
from src.ml.feature_engineering import FeatureScaler

# StandardScaler: mean=0, std=1 (for algorithms like SVM, KNN)
X_standard = FeatureScaler.scale_numeric(X, method='standard')

# MinMaxScaler: range [0, 1] (for algorithms sensitive to scale)
X_minmax = FeatureScaler.scale_numeric(X, method='minmax')

# RobustScaler: resistant to outliers
X_robust = FeatureScaler.scale_numeric(X, method='robust')

# Log transformation (for skewed distributions)
X_log = FeatureScaler.log_transform(X)

# Box-Cox transformation (requires positive values)
X_boxcox = FeatureScaler.box_cox_transform(X)
```

### Categorical Encoding

```python
from src.ml.feature_engineering import FeatureEncoder

# One-Hot Encoding: expand categories to binary columns
X_onehot = FeatureEncoder.one_hot_encode(X, columns=['color', 'size'])

# Label Encoding: assign integers to categories
X_label = FeatureEncoder.label_encode(X, columns=['color', 'size'])

# Target Encoding: encode with target mean
X_target = FeatureEncoder.target_encode(X, y, columns=['category'])

# Frequency Encoding: encode with value frequency
X_freq = FeatureEncoder.frequency_encode(X, columns=['category'])
```

### Feature Creation

```python
from src.ml.feature_engineering import FeatureCreator

# Polynomial features: X², X³, interactions
X_poly = FeatureCreator.create_polynomial_features(X, degree=2)

# Interaction features: A×B, A/B
X_interact = FeatureCreator.create_interaction_features(X)

# Ratio features: A/B
X_ratio = FeatureCreator.create_ratio_features(X)

# Statistical features: mean, std, min, max per row
X_stat = FeatureCreator.create_statistical_features(X)

# Binned features: discretize continuous values
X_binned = FeatureCreator.create_binned_features(X, n_bins=5)

# Cyclic features: for periodic data (month, hour, day)
X_cyclic = FeatureCreator.create_cyclic_features(X, col='month', period=12)
```

### Feature Selection

```python
from src.ml.feature_engineering import FeatureSelector

# Remove low-variance features
important_features = FeatureSelector.select_by_variance(X)

# Select by correlation with target
important_features = FeatureSelector.select_by_importance(X, y, task_type='classification', n_features=20)

# Recursive Feature Elimination
important_features = FeatureSelector.select_by_rfe(X, y, task_type='classification', n_features=20)

# Tree-based importance
important_features = FeatureSelector.select_by_tree_importance(X, y, threshold=0.01)

# Keep only selected features
X_selected = X[important_features]
```

---

## Model Evaluation

### Classification Evaluation

```python
from src.ml.evaluation import ClassificationEvaluator

evaluator = ClassificationEvaluator(
    y_true=y_test,
    y_pred=predictions,
    y_pred_proba=probabilities  # Optional: for ROC curves
)

# Get all metrics
metrics = evaluator.evaluate()

# Print metrics summary
metrics_df = evaluator.get_metrics_summary()
print(metrics_df)

# Get confusion matrix
cm = evaluator.get_confusion_matrix()

# Get ROC curve (binary classification)
fpr, tpr, thresholds = evaluator.get_roc_curve()

# Get precision-recall curve
precision, recall, thresholds = evaluator.get_precision_recall_curve()
```

### Regression Evaluation

```python
from src.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(
    y_true=y_test,
    y_pred=predictions
)

# Get all metrics
metrics = evaluator.evaluate()
print(metrics)

# Key metrics:
# - r2: Proportion of variance explained (0-1, higher better)
# - rmse: Root mean squared error (lower better)
# - mae: Mean absolute error (lower better)
# - mape: Mean absolute percentage error (lower better)

# Get residuals
residuals = evaluator.get_residuals()
```

### Model Comparison

```python
from src.ml.evaluation import ModelComparator

comparator = ModelComparator()

# Train multiple models
models = {
    'Logistic Regression': lr_model,
    'Random Forest': rf_model,
    'SVM': svm_model
}

# Get predictions from all models
y_pred_dict = {name: model.predict(X_test) for name, model in models.items()}

# Compare
comparator.evaluate_classification(y_test, y_pred_dict)

# Get comparison table
comparison_table = comparator.get_all_metrics_comparison()
print(comparison_table)

# Find best model
best_model, best_score = comparator.get_best_model(metric='f1')
print(f"Best model: {best_model} (F1: {best_score:.4f})")
```

---

## Best Practices

### 1. Data Validation

```python
# Always check data before modeling
print(df.info())           # Data types and nulls
print(df.describe())       # Statistics
print(df.isna().sum())     # Missing values
print(df.duplicated().sum()) # Duplicates

# Check for class imbalance (classification)
print(y.value_counts())
```

### 2. Proper Train-Test Split

```python
# For classification: use stratified split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # Keep class distribution
)

# Never use test set for:
# - Hyperparameter tuning
# - Feature selection
# - Model selection
```

### 3. Feature Scaling

```python
# ALWAYS scale before clustering or distance-based models
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use training stats!
```

### 4. Handling Imbalanced Data

```python
# Check imbalance
print(y.value_counts(normalize=True))

# Techniques:
# 1. Resampling (oversampling, undersampling)
# 2. Adjusted class weights
# 3. SMOTE: Synthetic Minority Over-sampling

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
model = RandomForestClassifier(class_weight={0: class_weights[0], 1: class_weights[1]})
```

### 5. Cross-Validation

```python
# Always use cross-validation for robust evaluation
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='f1')
print(f"Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")

# cv=5 is standard, use cv=10 for small datasets
```

### 6. Save and Load Models

```python
# Save model
model.save('path/to/model.pkl')

# Load model
loaded_model = SupervisedLearningModel.load('path/to/model.pkl')
predictions = loaded_model.predict(X_new)
```

---

## Complete Examples

### Example 1: Binary Classification (Supervised)

```python
# Complete classification workflow
from src.ml.supervised import SupervisedLearningModel
from src.ml.feature_engineering import FeatureScaler, FeatureSelector
from sklearn.model_selection import train_test_split

# 1. Load and prepare data
df = pd.read_csv('data.csv')
df = df.dropna()  # Remove nulls

# 2. Feature engineering
X = df.drop('target', axis=1)
y = df['target']

# Scale features
X_scaled = FeatureScaler.scale_numeric(X)

# Select important features
important_cols = FeatureSelector.select_by_importance(X_scaled, y, n_features=20)
X_selected = X_scaled[important_cols]

# 3. Train multiple models
models = {}
for model_type in ['logistic_regression', 'random_forest', 'gradient_boosting']:
    model = SupervisedLearningModel(
        task_type='classification',
        model_type=model_type,
        test_size=0.2
    )
    model.train(X_selected, y)
    models[model_type] = model

# 4. Compare and select best
best_model = max(models.items(), 
                 key=lambda x: x[1].evaluation_results_.get('f1', 0))
print(f"Best model: {best_model[0]}")

# 5. Make predictions on new data
X_new = pd.read_csv('new_data.csv')
X_new_scaled = FeatureScaler.scale_numeric(X_new)
X_new_selected = X_new_scaled[important_cols]
predictions = best_model[1].predict(X_new_selected)
```

### Example 2: Customer Segmentation (Unsupervised)

```python
# Complete clustering workflow
from src.ml.unsupervised import ClusteringModel, DimensionalityReducer, find_optimal_clusters
from sklearn.preprocessing import StandardScaler

# 1. Load and prepare
df = pd.read_csv('customers.csv')
X = df[['spending', 'visits', 'age', 'tenure']].dropna()

# 2. Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Find optimal clusters
k_analysis = find_optimal_clusters(X_scaled, k_range=(2, 10))
optimal_k = k_analysis['elbow_k']

# 4. Train clustering
clusterer = ClusteringModel(algorithm='kmeans', n_clusters=optimal_k)
clusterer.fit(X_scaled)

# 5. Analyze segments
df['segment'] = clusterer.labels_
segment_profiles = df.groupby('segment').agg({
    'spending': 'mean',
    'visits': 'mean',
    'age': 'mean',
    'tenure': 'mean'
})

print(segment_profiles)

# 6. Visualize
reducer = DimensionalityReducer(algorithm='pca', n_components=2)
X_2d = reducer.fit(X_scaled).X_transformed_

import matplotlib.pyplot as plt
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=clusterer.labels_, cmap='viridis')
plt.title(f'Customer Segments (K={optimal_k})')
plt.show()
```

---

## Streamlit Pages

This system includes full Streamlit integration:

- **20_Supervised_Learning.py**: Complete supervised learning interface
- **21_Unsupervised_Learning.py**: Clustering and dimensionality reduction
- **22_ML_Academy.py**: Comprehensive ML curriculum

Run with:
```bash
streamlit run app.py
```

---

## Tips for $10M Quality

1. **Always validate**: Cross-validate, use proper metrics
2. **Document decisions**: Why this model, this parameter?
3. **Handle edge cases**: Missing values, outliers, imbalance
4. **Test thoroughly**: Unit tests for data pipelines
5. **Monitor production**: Track model drift and performance
6. **Be interpretable**: Use SHAP values, feature importance
7. **Automate**: Build reproducible, versioned pipelines
8. **Optimize**: Balance accuracy, speed, and resource usage

---

## Support

For issues, examples, or improvements, check the Streamlit interface or modify the code according to your needs!
