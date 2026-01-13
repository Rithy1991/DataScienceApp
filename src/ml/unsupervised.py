"""
Comprehensive Unsupervised Learning Module
============================================
Clustering, dimensionality reduction, and pattern discovery.

Author: Data Science Pro
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Clustering algorithms
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    AgglomerativeClustering,
    SpectralClustering,
    Birch,
    MeanShift,
    estimate_bandwidth,
)

# Dimensionality reduction
from sklearn.decomposition import (
    PCA,
    TruncatedSVD,
    NMF,
    FactorAnalysis,
)

# Manifold learning
from sklearn.manifold import (
    TSNE,
    LocallyLinearEmbedding,
    MDS,
)

# Preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    silhouette_samples,
)

# Optional imports
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

warnings.filterwarnings("ignore")


@dataclass
class ClusteringModel:
    """A standardized wrapper for common clustering algorithms."""

    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42

    def __post_init__(self):
        self.model = None
        self.labels_ = None
        self.cluster_centers_ = None
        self.inertia_ = None
        self.silhouette_score_ = None
        self.davies_bouldin_score_ = None
        self.calinski_harabasz_score_ = None
        self.n_clusters_ = None
        self.X_scaled_ = None
        self.scaler_ = None  # To store the fitted scaler

    def _get_model(self):
        """Get clustering model instance with standardized hyperparameters."""
        algo = self.algorithm.lower()
        params = self.hyperparameters.copy()

        # Standardize random_state for algorithms that support it
        if algo in ["kmeans", "spectral", "gmm"]:
            params["random_state"] = self.random_state

        if algo == "kmeans":
            # Use k-means++ for better initialization and set n_init
            params.setdefault("init", "k-means++")
            params.setdefault("n_init", 10)
            return KMeans(**params)

        elif algo == "dbscan":
            # min_samples is a critical parameter, provide a default
            params.setdefault("min_samples", 5)
            return DBSCAN(**params)

        elif algo == "hierarchical":
            return AgglomerativeClustering(**params)

        elif algo == "spectral":
            return SpectralClustering(**params)

        elif algo == "birch":
            return Birch(**params)

        elif algo == "meanshift":
            # Automatically estimate bandwidth if not provided
            if "bandwidth" not in params:
                try:
                    # Use a sample to speed up bandwidth estimation
                    sample_size = min(len(self.X_scaled_), 1000)
                    sample = self.X_scaled_[
                        np.random.choice(self.X_scaled_.shape[0], sample_size, replace=False)
                    ]
                    params["bandwidth"] = estimate_bandwidth(
                        sample, quantile=0.2, n_samples=500
                    )
                except Exception:
                    # Fallback if estimation fails
                    params["bandwidth"] = None
            return MeanShift(**params)
        
        elif algo == "gmm":
            from sklearn.mixture import GaussianMixture
            params.setdefault("n_init", 10)
            return GaussianMixture(**params)

        else:
            raise ValueError(f"Unsupported clustering algorithm: {self.algorithm}")

    def fit(self, X: pd.DataFrame) -> "ClusteringModel":
        """Fit clustering model to the data.

        Args:
            X: Input DataFrame with features.

        Returns:
            The fitted ClusteringModel instance.
        """
        if X.empty:
            raise ValueError("Input data cannot be empty.")

        # Scale features and store the scaler
        self.scaler_ = StandardScaler()
        self.X_scaled_ = self.scaler_.fit_transform(X)

        # Create and fit model
        self.model = self._get_model()

        # Handle algorithms that don't support fit_predict
        if hasattr(self.model, "fit_predict"):
            self.labels_ = self.model.fit_predict(self.X_scaled_)
        else:
            self.model.fit(self.X_scaled_)
            self.labels_ = self.model.labels_

        # Post-fit analysis
        self.n_clusters_ = len(np.unique(self.labels_[self.labels_ != -1]))  # Exclude noise points

        if hasattr(self.model, "cluster_centers_"):
            self.cluster_centers_ = self.model.cluster_centers_

        if hasattr(self.model, "inertia_"):
            self.inertia_ = self.model.inertia_

        # Auto-evaluate after fitting
        self.evaluate()

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict clusters for new data if the algorithm supports it."""
        if self.model is None or self.scaler_ is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

        if not hasattr(self.model, "predict"):
            raise NotImplementedError(
                f"The '{self.algorithm}' algorithm does not support predicting on new data."
            )

        # Use the stored scaler to transform new data
        X_scaled = self.scaler_.transform(X)
        return self.model.predict(X_scaled)

    def evaluate(self) -> Dict[str, float]:
        """Evaluate clustering quality and store scores in the instance.

        Returns:
            A dictionary containing the evaluation metrics.
        """
        if self.labels_ is None or self.X_scaled_ is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

        results = {"n_clusters": self.n_clusters_, "inertia": self.inertia_}

        # Metrics require at least 2 clusters and no more than n_samples-1
        if 1 < self.n_clusters_ < self.X_scaled_.shape[0]:
            try:
                self.silhouette_score_ = silhouette_score(self.X_scaled_, self.labels_)
                results["silhouette_score"] = self.silhouette_score_
            except ValueError:
                self.silhouette_score_ = None

            try:
                self.davies_bouldin_score_ = davies_bouldin_score(
                    self.X_scaled_, self.labels_
                )
                results["davies_bouldin_score"] = self.davies_bouldin_score_
            except ValueError:
                self.davies_bouldin_score_ = None

            try:
                self.calinski_harabasz_score_ = calinski_harabasz_score(
                    self.X_scaled_, self.labels_
                )
                results["calinski_harabasz_score"] = self.calinski_harabasz_score_
            except ValueError:
                self.calinski_harabasz_score_ = None
        
        return results
    
    def get_cluster_info(self, X: pd.DataFrame) -> pd.DataFrame:
        """Get detailed cluster information."""
        if self.labels_ is None:
            raise ValueError("Model not fitted yet.")
        
        info = pd.DataFrame({
            'cluster': self.labels_,
        })
        
        # Add cluster sizes
        cluster_sizes = pd.Series(self.labels_).value_counts().sort_index()
        info['cluster_size'] = info['cluster'].map(cluster_sizes)
        
        return info


@dataclass
class DimensionalityReducer:
    """A standardized wrapper for common dimensionality reduction techniques."""
    algorithm: str
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    random_state: int = 42

    def __post_init__(self):
        self.model = None
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.X_transformed_ = None
        self.scaler_ = None  # To store the fitted scaler
        self.model_ = None # Renamed to avoid conflict

    def _get_model(self):
        """Get dimensionality reduction model instance with standardized hyperparameters."""
        algo = self.algorithm.lower()
        params = self.hyperparameters.copy()

        # Standardize random_state for algorithms that support it
        if algo in ["pca", "tsne", "umap", "mds", "lle", "nmf", "fa", "svd"]:
            params["random_state"] = self.random_state

        if algo == "pca":
            return PCA(**params)
        elif algo == "tsne":
            # Default perplexity is often a good starting point
            params.setdefault("perplexity", 30)
            return TSNE(**params)
        elif algo == "umap":
            if not HAS_UMAP:
                raise ImportError(
                    "UMAP is not installed. Please install it with: pip install umap-learn"
                )
            return umap.UMAP(**params)
        elif algo == "mds":
            params.setdefault("n_init", 4) # Recommended default
            return MDS(**params)
        elif algo == "lle":
            params.setdefault("n_neighbors", 5) # Required parameter
            return LocallyLinearEmbedding(**params)
        elif algo == "nmf":
            params.setdefault("init", "nndsvda") # A robust default
            return NMF(**params)
        elif algo == "fa":
            return FactorAnalysis(**params)
        elif algo == "svd":
            return TruncatedSVD(**params)
        else:
            raise ValueError(f"Unsupported dimensionality reduction algorithm: {self.algorithm}")

    def fit(self, X: pd.DataFrame) -> "DimensionalityReducer":
        """Fit the dimensionality reducer to the data.

        Args:
            X: Input DataFrame with features.

        Returns:
            The fitted DimensionalityReducer instance.
        """
        if X.empty:
            raise ValueError("Input data cannot be empty.")

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        self.model_ = self._get_model()
        self.X_transformed_ = self.model_.fit_transform(X_scaled)

        if hasattr(self.model_, "explained_variance_ratio_"):
            self.explained_variance_ratio_ = self.model_.explained_variance_ratio_

        if hasattr(self.model_, "components_"):
            self.components_ = self.model_.components_

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted model."""
        if self.model_ is None or self.scaler_ is None:
            raise RuntimeError("Model is not fitted yet. Call .fit() first.")

        X_scaled = self.scaler_.transform(X)
        return self.model_.transform(X_scaled)

    def get_variance_explained(self) -> Optional[Dict[str, Any]]:
        """Get variance explanation details, if applicable (e.g., for PCA)."""
        if self.explained_variance_ratio_ is None:
            return None

        cumulative_variance = np.cumsum(self.explained_variance_ratio_)
        return {
            "explained_variance_ratio": self.explained_variance_ratio_.tolist(),
            "cumulative_variance_ratio": cumulative_variance.tolist(),
            "total_variance_explained": float(np.sum(self.explained_variance_ratio_)),
        }
    


class AnomalyDetector:
    """Detect anomalies using various methods."""
    
    def __init__(self, method: str = "isolation_forest", contamination: float = 0.1):
        """
        Initialize anomaly detector.
        
        Methods: isolation_forest, local_outlier_factor, elliptic_envelope, robust_covariance
        """
        self.method = method
        self.contamination = contamination
        self.model = None
        self.predictions_ = None
        self.anomaly_scores_ = None
        self._build_model()
    
    def _build_model(self):
        """Build anomaly detection model."""
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        from sklearn.covariance import EllipticEnvelope
        
        if self.method == "isolation_forest":
            self.model = IsolationForest(contamination=self.contamination, random_state=42)
        elif self.method == "local_outlier_factor":
            self.model = LocalOutlierFactor(contamination=self.contamination)
        elif self.method == "elliptic_envelope":
            self.model = EllipticEnvelope(contamination=self.contamination, random_state=42)
        else:
            self.model = IsolationForest(contamination=self.contamination, random_state=42)
    
    def fit(self, X: pd.DataFrame) -> AnomalyDetector:
        """Fit anomaly detector."""
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.model.fit(X_scaled)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict anomalies (-1 for anomalies, 1 for normal)."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return self.model.predict(X_scaled)
    
    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Get anomaly scores."""
        if self.model is None:
            raise ValueError("Model not fitted.")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return self.model.score_samples(X_scaled)


class RuleExtractor:
    """Extract association rules and patterns from data."""
    
    def __init__(self):
        self.frequent_items = None
        self.rules = None
    
    def find_frequent_items(self, X: pd.DataFrame, min_support: float = 0.3) -> Dict[str, float]:
        """Find frequently occurring item combinations."""
        from itertools import combinations
        
        n_rows = len(X)
        min_count = int(np.ceil(min_support * n_rows))
        
        frequent = {}
        
        # Single items
        for col in X.columns:
            for val in X[col].unique():
                mask = X[col] == val
                support = mask.sum() / n_rows
                if support >= min_support:
                    frequent[f"{col}={val}"] = support
        
        self.frequent_items = frequent
        return frequent
    
    @staticmethod
    def suggest_rules(data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Suggest interpretable patterns."""
        rules = []
        
        # Numeric feature patterns
        for col in data.select_dtypes(include=[np.number]).columns:
            mean_val = data[col].mean()
            std_val = data[col].std()
            
            high_group = data[data[col] > mean_val + std_val]
            if len(high_group) > 0:
                rules.append({
                    'feature': col,
                    'pattern': 'high_value',
                    'threshold': mean_val + std_val,
                    'frequency': len(high_group) / len(data)
                })
        
        return rules


def find_optimal_clusters(X: pd.DataFrame, k_range: Tuple[int, int] = (2, 10)) -> Dict[str, Any]:
    """Find optimal number of clusters using elbow method."""
    X_scaled = StandardScaler().fit_transform(X)
    
    inertias = []
    silhouette_scores = []
    K_range = range(k_range[0], k_range[1] + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))
    
    # Find elbow point
    diffs = np.diff(inertias)
    elbow_k = k_range[0] + np.argmin(diffs) + 1
    
    return {
        'k_range': list(K_range),
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'elbow_k': elbow_k,
        'best_k_silhouette': K_range[np.argmax(silhouette_scores)]
    }


def profile_clusters(X: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Create detailed cluster profiles."""
    X_copy = X.copy()
    X_copy['cluster'] = labels
    
    profiles = []
    for cluster_id in np.unique(labels):
        if cluster_id < 0:  # Skip noise in DBSCAN
            continue
        
        cluster_data = X_copy[X_copy['cluster'] == cluster_id]
        profile = {
            'cluster': cluster_id,
            'size': len(cluster_data),
            'percentage': len(cluster_data) / len(X_copy) * 100,
        }
        
        # Add mean values for numeric features
        for col in X.select_dtypes(include=[np.number]).columns:
            profile[f'{col}_mean'] = cluster_data[col].mean()
        
        profiles.append(profile)
    
    return pd.DataFrame(profiles)

# ============================================================================
# ADVANCED MODERN UNSUPERVISED TECHNIQUES
# ============================================================================

class AdvancedClusteringMethods:
    """Advanced clustering algorithms and analysis techniques."""
    
    @staticmethod
    def gaussian_mixture_model(
        X: np.ndarray,
        n_components: int = 3,
        covariance_type: str = 'full'
    ) -> Dict:
        """
        Gaussian Mixture Model for probabilistic clustering.
        Provides soft cluster assignments.
        """
        from sklearn.mixture import GaussianMixture
        
        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type=covariance_type,
            random_state=42,
            n_init=10
        )
        
        labels = gmm.fit_predict(X)
        probabilities = gmm.predict_proba(X)
        
        return {
            'labels': labels,
            'probabilities': probabilities,
            'bic': gmm.bic(X),
            'aic': gmm.aic(X),
            'converged': gmm.converged_
        }
    
    @staticmethod
    def density_based_clustering(
        X: np.ndarray,
        eps: float = 0.5,
        min_samples: int = 5
    ) -> Dict:
        """
        OPTICS clustering for density-based clustering
        with varying densities.
        """
        from sklearn.cluster import OPTICS
        
        clustering = OPTICS(eps=eps, min_samples=min_samples)
        labels = clustering.fit_predict(X)
        
        # Calculate silhouette for valid clusters
        valid_labels = labels[labels != -1]
        n_clusters = len(set(valid_labels))
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_outliers': len(labels[labels == -1]),
            'reachability': clustering.reachability_.tolist() if hasattr(clustering, 'reachability_') else None
        }
    
    @staticmethod
    def consensus_clustering(
        X: np.ndarray,
        algorithms: List[str] = ['kmeans', 'hierarchical', 'dbscan'],
        n_clusters: int = 3,
        n_iterations: int = 100
    ) -> Dict:
        """
        Ensemble clustering combining multiple algorithms
        for more stable and robust results.
        """
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.preprocessing import StandardScaler
        
        X_scaled = StandardScaler().fit_transform(X)
        consensus_matrix = np.zeros((len(X), len(X)))
        
        for _ in range(n_iterations):
            # Random sampling
            sample_idx = np.random.choice(len(X), size=int(0.8 * len(X)), replace=False)
            X_sample = X_scaled[sample_idx]
            
            # Run algorithms
            labels_list = []
            
            if 'kmeans' in algorithms:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels_list.append(kmeans.fit_predict(X_sample))
            
            if 'hierarchical' in algorithms:
                hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
                labels_list.append(hierarchical.fit_predict(X_sample))
            
            if 'dbscan' in algorithms:
                dbscan = DBSCAN(eps=0.5, min_samples=5)
                labels_list.append(dbscan.fit_predict(X_sample))
            
            # Update consensus matrix
            for i in range(len(sample_idx)):
                for j in range(i + 1, len(sample_idx)):
                    idx_i, idx_j = sample_idx[i], sample_idx[j]
                    
                    same_cluster = sum(
                        labels[i] == labels[j] for labels in labels_list
                    )
                    consensus_matrix[idx_i, idx_j] += same_cluster
                    consensus_matrix[idx_j, idx_i] += same_cluster
        
        # Normalize
        consensus_matrix /= n_iterations
        
        return {
            'consensus_matrix': consensus_matrix.tolist(),
            'mean_consensus': float(np.mean(consensus_matrix[np.triu_indices_from(consensus_matrix, k=1)])),
            'n_iterations': n_iterations
        }


class HierarchicalClustering:
    """Advanced hierarchical clustering analysis."""
    
    @staticmethod
    def create_dendrogram(
        X: np.ndarray,
        method: str = 'ward',
        metric: str = 'euclidean'
    ) -> Dict:
        """
        Create hierarchical clustering dendrogram.
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        
        Z = linkage(X, method=method, metric=metric)
        
        return {
            'linkage_matrix': Z.tolist(),
            'method': method,
            'metric': metric,
            'n_samples': len(X)
        }
    
    @staticmethod
    def find_optimal_cut_threshold(
        X: np.ndarray,
        method: str = 'ward',
        metric: str = 'euclidean'
    ) -> float:
        """
        Find optimal cutting threshold for dendrogram.
        Based on largest distances in linkage.
        """
        from scipy.cluster.hierarchy import linkage
        
        Z = linkage(X, method=method, metric=metric)
        
        # Find largest jumps in distances
        distances = Z[:, 2]
        last = distances[-10:]
        idxs = np.arange(1, len(last) + 1)
        
        acceleration = np.diff(last, 2)
        optimal_idx = acceleration.argmax() + 2
        
        return float(last[optimal_idx])


class AdvancedAnomalyDetection:
    """Advanced anomaly detection techniques."""
    
    @staticmethod
    def isolation_forest_advanced(
        X: np.ndarray,
        contamination: float = 0.1,
        n_estimators: int = 100
    ) -> Dict:
        """
        Advanced isolation forest with additional diagnostics.
        """
        from sklearn.ensemble import IsolationForest
        
        iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        labels = iso_forest.fit_predict(X)
        scores = iso_forest.score_samples(X)
        
        # Separate normal and anomalies
        normal_scores = scores[labels == 1]
        anomaly_scores = scores[labels == -1]
        
        return {
            'labels': labels,
            'anomaly_scores': scores.tolist(),
            'n_anomalies': int(np.sum(labels == -1)),
            'anomaly_threshold': float(iso_forest.offset_),
            'mean_normal_score': float(np.mean(normal_scores)) if len(normal_scores) > 0 else None,
            'mean_anomaly_score': float(np.mean(anomaly_scores)) if len(anomaly_scores) > 0 else None
        }
    
    @staticmethod
    def local_outlier_factor_advanced(
        X: np.ndarray,
        n_neighbors: int = 20,
        contamination: float = 0.1
    ) -> Dict:
        """
        Advanced LOF with neighborhood analysis.
        """
        from sklearn.neighbors import LocalOutlierFactor
        
        lof = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=False,
            n_jobs=-1
        )
        
        labels = lof.fit_predict(X)
        scores = lof.negative_outlier_factor_
        
        return {
            'labels': labels,
            'outlier_scores': (-scores).tolist(),  # Convert to positive scores
            'n_outliers': int(np.sum(labels == -1)),
            'mean_lof_normal': float(np.mean(-scores[labels == 1])) if np.sum(labels == 1) > 0 else None,
            'mean_lof_outlier': float(np.mean(-scores[labels == -1])) if np.sum(labels == -1) > 0 else None
        }
    
    @staticmethod
    def one_class_svm_anomaly_detection(
        X: np.ndarray,
        nu: float = 0.05,
        kernel: str = 'rbf'
    ) -> Dict:
        """
        One-Class SVM for anomaly detection.
        Effective for high-dimensional data.
        """
        from sklearn.svm import OneClassSVM
        
        oc_svm = OneClassSVM(nu=nu, kernel=kernel, gamma='auto')
        labels = oc_svm.fit_predict(X)
        scores = oc_svm.decision_function(X)
        
        return {
            'labels': labels,
            'decision_scores': scores.tolist(),
            'n_anomalies': int(np.sum(labels == -1)),
            'n_support_vectors': oc_svm.n_support_
        }


class TimeSeriesAnomalyDetection:
    """Specialized anomaly detection for time series."""
    
    @staticmethod
    def seasonal_decomposition_anomaly(
        data: np.ndarray,
        period: int = 7,
        threshold: float = 2.0
    ) -> Dict:
        """
        Detect anomalies using seasonal decomposition residuals.
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            decomposition = seasonal_decompose(
                pd.Series(data),
                model='additive',
                period=period
            )
            
            residuals = decomposition.resid.dropna()
            mean_residual = residuals.mean()
            std_residual = residuals.std()
            
            # Identify anomalies
            threshold_value = mean_residual + threshold * std_residual
            anomalies = residuals > threshold_value
            
            return {
                'anomalies': anomalies.tolist(),
                'residuals': residuals.tolist(),
                'trend': decomposition.trend.tolist(),
                'seasonal': decomposition.seasonal.tolist(),
                'n_anomalies': int(anomalies.sum()),
                'threshold': float(threshold_value)
            }
        except ImportError:
            return {'error': 'statsmodels not installed'}
    
    @staticmethod
    def moving_average_anomaly(
        data: np.ndarray,
        window_size: int = 7,
        threshold: float = 2.0
    ) -> Dict:
        """
        Detect anomalies as deviations from moving average.
        """
        moving_avg = pd.Series(data).rolling(window=window_size).mean()
        
        # Calculate deviation
        deviation = data - moving_avg
        mean_dev = np.nanmean(deviation)
        std_dev = np.nanstd(deviation)
        
        threshold_value = mean_dev + threshold * std_dev
        anomalies = deviation > threshold_value
        
        return {
            'anomalies': anomalies.tolist(),
            'moving_average': moving_avg.tolist(),
            'deviation': deviation.tolist(),
            'n_anomalies': int(np.nansum(anomalies)),
            'threshold': float(threshold_value)
        }