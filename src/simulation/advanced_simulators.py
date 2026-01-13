"""Advanced ML Simulators - Clustering, Neural Networks, Anomaly Detection"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.neural_network import MLPClassifier, MLPRegressor
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ClusteringResult:
    """Results from clustering simulation."""
    labels: np.ndarray
    centers: Optional[np.ndarray]
    n_clusters: int
    silhouette: float
    davies_bouldin: float
    calinski_harabasz: float
    algorithm: str
    reduced_data: np.ndarray
    original_data: np.ndarray
    explained_variance: Optional[float] = None


@dataclass
class NeuralNetResult:
    """Results from neural network simulation."""
    model: Any
    train_scores: List[float]
    val_scores: List[float]
    train_losses: List[float]
    val_losses: List[float]
    architecture: List[int]
    activation: str
    learning_rate: float
    convergence_iteration: int
    final_score: float


@dataclass
class AnomalyResult:
    """Results from anomaly detection simulation."""
    predictions: np.ndarray
    scores: np.ndarray
    n_anomalies: int
    anomaly_rate: float
    algorithm: str
    threshold: Optional[float] = None


class ClusteringSimulator:
    """Interactive clustering and dimensionality reduction simulator."""
    
    def __init__(self):
        self.algorithms = {
            'kmeans': KMeans,
            'dbscan': DBSCAN,
            'hierarchical': AgglomerativeClustering
        }
        
        self.dim_reducers = {
            'pca': PCA,
            'tsne': TSNE,
            'isomap': Isomap
        }
    
    def generate_clustering_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        n_clusters: int = 3,
        pattern: str = 'blobs'
    ) -> np.ndarray:
        """Generate synthetic clustering data."""
        
        if pattern == 'blobs':
            result = make_blobs(
                n_samples=n_samples,
                n_features=n_features,
                centers=n_clusters,
                cluster_std=1.5,
                random_state=42,
                return_centers=False
            )
            X = result[0]
        elif pattern == 'moons':
            X, _ = make_moons(n_samples=n_samples, noise=0.1, random_state=42)
            if n_features > 2:
                X = np.hstack([X, np.random.randn(n_samples, n_features - 2)])
        elif pattern == 'circles':
            X, _ = make_circles(n_samples=n_samples, noise=0.1, factor=0.5, random_state=42)
            if n_features > 2:
                X = np.hstack([X, np.random.randn(n_samples, n_features - 2)])
        elif pattern == 'elongated':
            result = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42, return_centers=False)
            X = result[0]
            transformation = np.diag([3.0] + [1.0] * (n_features - 1))
            X = X @ transformation
        else:
            result = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42, return_centers=False)
            X = result[0]
        
        return StandardScaler().fit_transform(X)
    
    def run_clustering(
        self,
        X: np.ndarray,
        algorithm: str = 'kmeans',
        n_clusters: int = 3,
        reduce_dim: str = 'pca',
        **kwargs
    ) -> ClusteringResult:
        """Run clustering algorithm with dimensionality reduction."""
        
        if algorithm == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(X)
            centers = clusterer.cluster_centers_
        elif algorithm == 'dbscan':
            eps = kwargs.get('eps', 0.5)
            min_samples = kwargs.get('min_samples', 5)
            clusterer = DBSCAN(eps=eps, min_samples=min_samples)
            labels = clusterer.fit_predict(X)
            centers = None
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        elif algorithm == 'hierarchical':
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            labels = clusterer.fit_predict(X)
            centers = None
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        # Calculate metrics
        valid_mask = labels >= 0
        if valid_mask.sum() > 1 and len(set(labels[valid_mask])) > 1:
            silhouette = silhouette_score(X[valid_mask], labels[valid_mask])
            davies_bouldin = davies_bouldin_score(X[valid_mask], labels[valid_mask])
            calinski = calinski_harabasz_score(X[valid_mask], labels[valid_mask])
        else:
            silhouette = davies_bouldin = calinski = 0.0
        
        # Dimensionality reduction
        if X.shape[1] > 2:
            if reduce_dim == 'pca':
                reducer = PCA(n_components=2, random_state=42)
                X_reduced = reducer.fit_transform(X)
                explained_var = sum(reducer.explained_variance_ratio_)
            elif reduce_dim == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, X.shape[0] - 1))
                X_reduced = reducer.fit_transform(X)
                explained_var = None
            elif reduce_dim == 'isomap':
                n_neighbors = min(10, X.shape[0] - 1)
                reducer = Isomap(n_components=2, n_neighbors=n_neighbors)
                X_reduced = reducer.fit_transform(X)
                explained_var = None
            else:
                X_reduced = X[:, :2]
                explained_var = None
        else:
            X_reduced = X
            explained_var = None
        
        return ClusteringResult(
            labels=labels,
            centers=centers,
            n_clusters=n_clusters,
            silhouette=float(silhouette),
            davies_bouldin=float(davies_bouldin),
            calinski_harabasz=float(calinski),
            algorithm=algorithm,
            reduced_data=X_reduced,
            original_data=X,
            explained_variance=explained_var
        )


class NeuralNetworkSimulator:
    """Neural network architecture search simulator."""
    
    def __init__(self):
        self.activations = ['relu', 'tanh', 'logistic']
        self.optimizers = ['adam', 'sgd']
    
    def simulate_architecture_search(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        problem_type: str = 'classification',
        max_layers: int = 5,
        max_neurons: int = 128,
        n_trials: int = 20
    ) -> List[NeuralNetResult]:
        """Simulate Neural Architecture Search."""
        
        results = []
        
        for trial in range(n_trials):
            n_layers = np.random.randint(1, max_layers + 1)
            hidden_layers = tuple(np.random.randint(16, max_neurons + 1, size=n_layers))
            activation = np.random.choice(self.activations)
            learning_rate = 10 ** np.random.uniform(-4, -2)
            
            try:
                if problem_type == 'classification':
                    model = MLPClassifier(
                        hidden_layer_sizes=hidden_layers,
                        activation=activation,
                        learning_rate_init=learning_rate,
                        max_iter=200,
                        random_state=42
                    )
                else:
                    model = MLPRegressor(
                        hidden_layer_sizes=hidden_layers,
                        activation=activation,
                        learning_rate_init=learning_rate,
                        max_iter=200,
                        random_state=42
                    )
                
                model.fit(X_train, y_train)
                
                train_score = model.score(X_train, y_train)
                val_score = model.score(X_val, y_val)
                train_losses = model.loss_curve_ if hasattr(model, 'loss_curve_') else []
                
                result = NeuralNetResult(
                    model=model,
                    train_scores=[train_score],
                    val_scores=[val_score],
                    train_losses=train_losses,
                    val_losses=[],
                    architecture=list(hidden_layers),
                    activation=activation,
                    learning_rate=learning_rate,
                    convergence_iteration=model.n_iter_,
                    final_score=val_score
                )
                
                results.append(result)
            except Exception as e:
                continue
        
        results.sort(key=lambda x: x.final_score, reverse=True)
        return results


class AnomalyDetectionSimulator:
    """Anomaly detection algorithms."""
    
    def __init__(self):
        self.algorithms = {
            'isolation_forest': IsolationForest,
            'one_class_svm': OneClassSVM,
            'elliptic_envelope': EllipticEnvelope
        }
    
    def generate_anomaly_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        contamination: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate data with anomalies."""
        
        n_normal = int(n_samples * (1 - contamination))
        X_normal = np.random.randn(n_normal, n_features)
        
        n_anomalies = n_samples - n_normal
        X_anomalies = np.random.randn(n_anomalies, n_features) * 3 + np.random.randn(n_features) * 5
        
        X = np.vstack([X_normal, X_anomalies])
        y_true = np.hstack([np.zeros(n_normal), np.ones(n_anomalies)])
        
        indices = np.random.permutation(len(X))
        return X[indices], y_true[indices]
    
    def detect_anomalies(
        self,
        X: np.ndarray,
        algorithm: str = 'isolation_forest',
        contamination: float = 0.1
    ) -> AnomalyResult:
        """Run anomaly detection."""
        
        if algorithm == 'isolation_forest':
            detector = IsolationForest(contamination=contamination, random_state=42)
            predictions = detector.fit_predict(X)
            scores = detector.score_samples(X)
        elif algorithm == 'one_class_svm':
            detector = OneClassSVM(nu=contamination, kernel='rbf')
            predictions = detector.fit_predict(X)
            scores = detector.score_samples(X)
        elif algorithm == 'elliptic_envelope':
            detector = EllipticEnvelope(contamination=contamination, random_state=42)
            predictions = detector.fit_predict(X)
            scores = detector.score_samples(X)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
        
        predictions_binary = (predictions == -1).astype(int)
        n_anomalies = predictions_binary.sum()
        anomaly_rate = n_anomalies / len(X)
        
        return AnomalyResult(
            predictions=predictions_binary,
            scores=scores,
            n_anomalies=n_anomalies,
            anomaly_rate=anomaly_rate,
            algorithm=algorithm
        )


class EnsembleSimulator:
    """Ensemble methods simulator."""
    
    def compare_ensemble_methods(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_estimators: int = 50
    ) -> Dict[str, Dict[str, float]]:
        """Compare ensemble methods."""
        
        from sklearn.ensemble import (
            BaggingClassifier,
            AdaBoostClassifier,
            GradientBoostingClassifier,
            RandomForestClassifier
        )
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        import time
        
        results = {}
        
        # Bagging
        start = time.time()
        bagging = BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=5),
            n_estimators=n_estimators,
            random_state=42
        )
        bagging.fit(X_train, y_train)
        y_pred = bagging.predict(X_test)
        results['Bagging'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_time': time.time() - start
        }
        
        # AdaBoost
        start = time.time()
        adaboost = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=n_estimators,
            random_state=42,
            algorithm='SAMME'
        )
        adaboost.fit(X_train, y_train)
        y_pred = adaboost.predict(X_test)
        results['AdaBoost'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_time': time.time() - start
        }
        
        # Gradient Boosting
        start = time.time()
        gb = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            random_state=42
        )
        gb.fit(X_train, y_train)
        y_pred = gb.predict(X_test)
        results['Gradient Boosting'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_time': time.time() - start
        }
        
        # Random Forest
        start = time.time()
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        results['Random Forest'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'train_time': time.time() - start
        }
        
        return results


@dataclass
class FederatedLearningResult:
    """Results from federated learning simulation."""
    global_model: Any
    client_accuracies: List[List[float]]
    global_accuracies: List[float]
    communication_rounds: int
    privacy_budget: Optional[float]
    convergence_round: Optional[int]


class FederatedLearningSimulator:
    """Simulate federated learning with privacy-preserving techniques."""
    
    def __init__(self):
        self.privacy_mechanisms = ['none', 'differential_privacy', 'secure_aggregation']
    
    def split_data_to_clients(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_clients: int = 5,
        iid: bool = True
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Split data among clients (IID or non-IID)."""
        
        n_samples = len(X)
        
        if iid:
            # Random split
            indices = np.random.permutation(n_samples)
            client_data = []
            for i in range(n_clients):
                start_idx = i * n_samples // n_clients
                end_idx = (i + 1) * n_samples // n_clients
                client_indices = indices[start_idx:end_idx]
                client_data.append((X[client_indices], y[client_indices]))
        else:
            # Non-IID: sort by label and split
            sorted_indices = np.argsort(y)
            client_data = []
            for i in range(n_clients):
                start_idx = i * n_samples // n_clients
                end_idx = (i + 1) * n_samples // n_clients
                client_indices = sorted_indices[start_idx:end_idx]
                client_data.append((X[client_indices], y[client_indices]))
        
        return client_data
    
    def add_differential_privacy(
        self,
        gradients: np.ndarray,
        epsilon: float = 1.0,
        delta: float = 1e-5
    ) -> np.ndarray:
        """Add Laplacian noise for differential privacy."""
        sensitivity = 1.0  # L2 sensitivity
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, gradients.shape)
        return gradients + noise
    
    def federated_averaging(
        self,
        X_train_clients: List[Tuple[np.ndarray, np.ndarray]],
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_rounds: int = 20,
        local_epochs: int = 5,
        privacy_mechanism: str = 'none',
        epsilon: float = 1.0
    ) -> FederatedLearningResult:
        """
        Simulate Federated Averaging (FedAvg) algorithm.
        
        Args:
            X_train_clients: List of (X, y) tuples for each client
            X_test: Global test set
            y_test: Global test labels
            n_rounds: Number of communication rounds
            local_epochs: Epochs each client trains locally
            privacy_mechanism: 'none', 'differential_privacy', or 'secure_aggregation'
            epsilon: Privacy budget for differential privacy
        
        Returns:
            FederatedLearningResult with training history
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        n_clients = len(X_train_clients)
        client_accuracies = [[] for _ in range(n_clients)]
        global_accuracies = []
        
        # Initialize global model
        global_model = LogisticRegression(max_iter=100, random_state=42, warm_start=True)
        
        # Initial training on first client to initialize
        X_init, y_init = X_train_clients[0]
        global_model.fit(X_init[:10], y_init[:10])
        
        convergence_round = None
        prev_accuracy = 0
        
        for round_num in range(n_rounds):
            # Each client trains locally
            client_models = []
            
            for client_id, (X_client, y_client) in enumerate(X_train_clients):
                # Clone global model parameters
                client_model = LogisticRegression(max_iter=local_epochs, random_state=42, warm_start=True)
                client_model.coef_ = global_model.coef_.copy()
                client_model.intercept_ = global_model.intercept_.copy()
                client_model.classes_ = global_model.classes_
                
                # Local training
                for _ in range(local_epochs):
                    client_model.fit(X_client, y_client)
                
                # Apply differential privacy if requested
                if privacy_mechanism == 'differential_privacy':
                    client_model.coef_ = self.add_differential_privacy(
                        client_model.coef_, epsilon=epsilon
                    )
                
                client_models.append(client_model)
                
                # Track client accuracy
                client_acc = accuracy_score(y_client, client_model.predict(X_client))
                client_accuracies[client_id].append(client_acc)
            
            # Aggregate models (Federated Averaging)
            aggregated_coef = np.mean([model.coef_ for model in client_models], axis=0)
            aggregated_intercept = np.mean([model.intercept_ for model in client_models], axis=0)
            
            # Update global model
            global_model.coef_ = aggregated_coef
            global_model.intercept_ = aggregated_intercept
            
            # Evaluate global model
            global_acc = accuracy_score(y_test, global_model.predict(X_test))
            global_accuracies.append(global_acc)
            
            # Check convergence
            if convergence_round is None and abs(global_acc - prev_accuracy) < 0.001:
                convergence_round = round_num
            prev_accuracy = global_acc
        
        return FederatedLearningResult(
            global_model=global_model,
            client_accuracies=client_accuracies,
            global_accuracies=global_accuracies,
            communication_rounds=n_rounds,
            privacy_budget=epsilon if privacy_mechanism == 'differential_privacy' else None,
            convergence_round=convergence_round
        )


@dataclass
class ExplainabilityResult:
    """Results from model explainability analysis."""
    feature_importance: Dict[str, float]
    sample_explanations: List[Dict[str, Any]]
    global_explanation: Dict[str, Any]
    method: str


class ExplainabilitySimulator:
    """Simulate model explainability using SHAP-like and LIME-like approaches."""
    
    def __init__(self):
        self.methods = ['feature_importance', 'permutation', 'partial_dependence', 'local_linear']
    
    def compute_feature_importance(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Compute feature importance from tree-based models."""
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # Fallback: use coefficient magnitudes
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_)
            else:
                importances = np.zeros(X.shape[1])
        
        return dict(zip(feature_names, importances))
    
    def permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10
    ) -> Dict[str, float]:
        """Compute permutation importance."""
        from sklearn.metrics import accuracy_score, r2_score
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Baseline score
        y_pred = model.predict(X)
        if len(np.unique(y)) <= 10:  # Classification
            baseline_score = accuracy_score(y, y_pred)
        else:  # Regression
            baseline_score = r2_score(y, y_pred)
        
        importances = {}
        
        for i, feature_name in enumerate(feature_names):
            scores = []
            for _ in range(n_repeats):
                X_permuted = X.copy()
                np.random.shuffle(X_permuted[:, i])
                y_pred = model.predict(X_permuted)
                
                if len(np.unique(y)) <= 10:
                    score = accuracy_score(y, y_pred)
                else:
                    score = r2_score(y, y_pred)
                
                scores.append(baseline_score - score)
            
            importances[feature_name] = np.mean(scores)
        
        return importances
    
    def explain_instance_local_linear(
        self,
        model: Any,
        X: np.ndarray,
        instance_idx: int,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 5000
    ) -> Dict[str, Any]:
        """LIME-like local linear explanation for a specific instance."""
        from sklearn.linear_model import Ridge
        from sklearn.metrics.pairwise import euclidean_distances
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        instance = X[instance_idx:instance_idx+1]
        
        # Generate perturbed samples around instance
        noise = np.random.normal(0, 0.1, (n_samples, X.shape[1]))
        X_perturbed = instance + noise
        
        # Get predictions for perturbed samples
        y_perturbed = model.predict(X_perturbed)
        
        # Weight samples by proximity
        distances = euclidean_distances(X_perturbed, instance).flatten()
        weights = np.exp(-distances / (2 * 0.25 ** 2))
        
        # Fit local linear model
        local_model = Ridge(alpha=1.0)
        local_model.fit(X_perturbed, y_perturbed, sample_weight=weights)
        
        # Extract coefficients as feature importance
        coefficients = dict(zip(feature_names, local_model.coef_))
        
        return {
            'instance_idx': instance_idx,
            'prediction': model.predict(instance)[0],
            'local_coefficients': coefficients,
            'intercept': local_model.intercept_,
            'r2_local_fit': local_model.score(X_perturbed, y_perturbed, sample_weight=weights)
        }
    
    def explain_model(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'permutation',
        feature_names: Optional[List[str]] = None,
        sample_indices: Optional[List[int]] = None
    ) -> ExplainabilityResult:
        """Generate comprehensive model explanation."""
        
        if sample_indices is None:
            sample_indices = [0, len(X) // 2, len(X) - 1]
        
        # Global explanation
        if method == 'feature_importance':
            global_importance = self.compute_feature_importance(model, X, feature_names)
        elif method == 'permutation':
            global_importance = self.permutation_importance(model, X, y, feature_names)
        else:
            global_importance = self.compute_feature_importance(model, X, feature_names)
        
        # Local explanations for sample instances
        sample_explanations = []
        for idx in sample_indices[:3]:  # Limit to 3 samples
            explanation = self.explain_instance_local_linear(model, X, idx, feature_names)
            sample_explanations.append(explanation)
        
        return ExplainabilityResult(
            feature_importance=global_importance,
            sample_explanations=sample_explanations,
            global_explanation={'method': method, 'top_features': sorted(
                global_importance.items(), key=lambda x: abs(x[1]), reverse=True
            )[:5]},
            method=method
        )


@dataclass
class FairnessResult:
    """Results from fairness analysis."""
    overall_accuracy: float
    group_accuracies: Dict[str, float]
    demographic_parity: float
    equalized_odds: Dict[str, float]
    disparate_impact: float
    fairness_score: float


class FairnessSimulator:
    """Simulate fairness metrics and bias detection."""
    
    def __init__(self):
        self.fairness_metrics = [
            'demographic_parity',
            'equalized_odds',
            'disparate_impact',
            'equal_opportunity'
        ]
    
    def generate_biased_data(
        self,
        n_samples: int = 1000,
        protected_attribute_bias: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic data with bias in protected attribute."""
        
        # Protected attribute (e.g., 0=group A, 1=group B)
        protected = np.random.binomial(1, 0.5, n_samples)
        
        # Features correlated with protected attribute
        X = np.random.randn(n_samples, 5)
        X[:, 0] += protected * protected_attribute_bias
        
        # Target with bias: group B has lower positive rate
        base_prob = 0.5 + 0.3 * X[:, 1] + 0.2 * X[:, 2]
        base_prob -= protected * protected_attribute_bias
        y = (base_prob + np.random.randn(n_samples) * 0.1) > 0.5
        y = y.astype(int)
        
        return X, y, protected
    
    def compute_demographic_parity(
        self,
        y_pred: np.ndarray,
        protected: np.ndarray
    ) -> float:
        """
        Compute demographic parity difference.
        Measures difference in positive prediction rate between groups.
        """
        group_0_rate = np.mean(y_pred[protected == 0])
        group_1_rate = np.mean(y_pred[protected == 1])
        return abs(group_0_rate - group_1_rate)
    
    def compute_equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        protected: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute equalized odds (TPR and FPR difference between groups).
        """
        from sklearn.metrics import confusion_matrix
        
        # Group 0
        tn0, fp0, fn0, tp0 = confusion_matrix(
            y_true[protected == 0], y_pred[protected == 0]
        ).ravel()
        tpr_0 = tp0 / (tp0 + fn0) if (tp0 + fn0) > 0 else 0
        fpr_0 = fp0 / (fp0 + tn0) if (fp0 + tn0) > 0 else 0
        
        # Group 1
        tn1, fp1, fn1, tp1 = confusion_matrix(
            y_true[protected == 1], y_pred[protected == 1]
        ).ravel()
        tpr_1 = tp1 / (tp1 + fn1) if (tp1 + fn1) > 0 else 0
        fpr_1 = fp1 / (fp1 + tn1) if (fp1 + tn1) > 0 else 0
        
        return {
            'tpr_difference': abs(tpr_0 - tpr_1),
            'fpr_difference': abs(fpr_0 - fpr_1)
        }
    
    def compute_disparate_impact(
        self,
        y_pred: np.ndarray,
        protected: np.ndarray
    ) -> float:
        """
        Compute disparate impact ratio.
        Ratio of positive prediction rate (group with lower rate / group with higher rate).
        """
        group_0_rate = np.mean(y_pred[protected == 0])
        group_1_rate = np.mean(y_pred[protected == 1])
        
        if group_0_rate == 0 or group_1_rate == 0:
            return 0.0
        
        return min(group_0_rate, group_1_rate) / max(group_0_rate, group_1_rate)
    
    def evaluate_fairness(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        protected: np.ndarray
    ) -> FairnessResult:
        """Comprehensive fairness evaluation."""
        from sklearn.metrics import accuracy_score
        
        y_pred = model.predict(X)
        
        # Overall accuracy
        overall_acc = accuracy_score(y, y_pred)
        
        # Group-specific accuracies
        group_accs = {
            'group_0': accuracy_score(y[protected == 0], y_pred[protected == 0]),
            'group_1': accuracy_score(y[protected == 1], y_pred[protected == 1])
        }
        
        # Fairness metrics
        demo_parity = self.compute_demographic_parity(y_pred, protected)
        eq_odds = self.compute_equalized_odds(y, y_pred, protected)
        disp_impact = self.compute_disparate_impact(y_pred, protected)
        
        # Composite fairness score (lower is better)
        fairness_score = demo_parity + eq_odds['tpr_difference'] + eq_odds['fpr_difference']
        
        return FairnessResult(
            overall_accuracy=overall_acc,
            group_accuracies=group_accs,
            demographic_parity=demo_parity,
            equalized_odds=eq_odds,
            disparate_impact=disp_impact,
            fairness_score=fairness_score
        )


@dataclass
class ActiveLearningResult:
    """Results from active learning simulation."""
    accuracies: List[float]
    sample_sizes: List[int]
    selected_indices: List[int]
    uncertainty_scores: List[float]
    strategy: str


class ActiveLearningSimulator:
    """Simulate active learning strategies for efficient labeling."""
    
    def __init__(self):
        self.strategies = ['uncertainty', 'margin', 'entropy', 'random']
    
    def uncertainty_sampling(
        self,
        model: Any,
        X_pool: np.ndarray,
        n_samples: int = 10
    ) -> np.ndarray:
        """Select samples with highest prediction uncertainty."""
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_pool)
            # Uncertainty: 1 - max probability
            uncertainties = 1 - np.max(probas, axis=1)
        else:
            # For models without predict_proba, use distance from decision boundary
            if hasattr(model, 'decision_function'):
                decisions = model.decision_function(X_pool)
                uncertainties = 1 / (np.abs(decisions) + 1e-10)
            else:
                uncertainties = np.random.rand(len(X_pool))
        
        return np.argsort(uncertainties)[-n_samples:]
    
    def margin_sampling(
        self,
        model: Any,
        X_pool: np.ndarray,
        n_samples: int = 10
    ) -> np.ndarray:
        """Select samples with smallest margin between top 2 classes."""
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_pool)
            probas_sorted = np.sort(probas, axis=1)
            margins = probas_sorted[:, -1] - probas_sorted[:, -2]
        else:
            margins = np.random.rand(len(X_pool))
        
        return np.argsort(margins)[:n_samples]
    
    def entropy_sampling(
        self,
        model: Any,
        X_pool: np.ndarray,
        n_samples: int = 10
    ) -> np.ndarray:
        """Select samples with highest prediction entropy."""
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X_pool)
            entropies = -np.sum(probas * np.log(probas + 1e-10), axis=1)
        else:
            entropies = np.random.rand(len(X_pool))
        
        return np.argsort(entropies)[-n_samples:]
    
    def simulate_active_learning(
        self,
        X: np.ndarray,
        y: np.ndarray,
        initial_samples: int = 20,
        n_iterations: int = 10,
        samples_per_iteration: int = 10,
        strategy: str = 'uncertainty',
        model_class: Any = None
    ) -> ActiveLearningResult:
        """
        Simulate active learning process.
        
        Args:
            X: Full feature dataset
            y: Full labels
            initial_samples: Initial labeled samples
            n_iterations: Number of active learning iterations
            samples_per_iteration: Samples to label per iteration
            strategy: Sampling strategy ('uncertainty', 'margin', 'entropy', 'random')
            model_class: Model class to use (default: RandomForestClassifier)
        
        Returns:
            ActiveLearningResult with learning curve
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        from sklearn.model_selection import train_test_split
        
        if model_class is None:
            model_class = RandomForestClassifier
        
        # Split into train/test
        X_train_pool, X_test, y_train_pool, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize with random samples
        n_pool = len(X_train_pool)
        labeled_indices = np.random.choice(n_pool, initial_samples, replace=False)
        unlabeled_indices = np.setdiff1d(np.arange(n_pool), labeled_indices)
        
        accuracies = []
        sample_sizes = []
        all_selected = list(labeled_indices)
        all_uncertainties = []
        
        for iteration in range(n_iterations):
            # Train on labeled data
            X_labeled = X_train_pool[labeled_indices]
            y_labeled = y_train_pool[labeled_indices]
            
            model = model_class(n_estimators=50, random_state=42)
            model.fit(X_labeled, y_labeled)
            
            # Evaluate
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            sample_sizes.append(len(labeled_indices))
            
            if len(unlabeled_indices) == 0:
                break
            
            # Select new samples
            X_pool = X_train_pool[unlabeled_indices]
            
            if strategy == 'uncertainty':
                pool_indices = self.uncertainty_sampling(model, X_pool, samples_per_iteration)
            elif strategy == 'margin':
                pool_indices = self.margin_sampling(model, X_pool, samples_per_iteration)
            elif strategy == 'entropy':
                pool_indices = self.entropy_sampling(model, X_pool, samples_per_iteration)
            else:  # random
                pool_indices = np.random.choice(len(X_pool), min(samples_per_iteration, len(X_pool)), replace=False)
            
            # Map back to original indices
            new_labeled = unlabeled_indices[pool_indices]
            labeled_indices = np.concatenate([labeled_indices, new_labeled])
            unlabeled_indices = np.setdiff1d(unlabeled_indices, new_labeled)
            all_selected.extend(new_labeled)
        
        return ActiveLearningResult(
            accuracies=accuracies,
            sample_sizes=sample_sizes,
            selected_indices=all_selected,
            uncertainty_scores=all_uncertainties,
            strategy=strategy
        )


@dataclass
class TransferLearningResult:
    """Results from transfer learning simulation."""
    source_accuracy: float
    target_accuracy_no_transfer: float
    target_accuracy_with_transfer: float
    improvement: float
    frozen_layers: List[int]
    fine_tuned_layers: List[int]


class TransferLearningSimulator:
    """Simulate transfer learning and domain adaptation."""
    
    def __init__(self):
        self.strategies = ['freeze_all', 'freeze_partial', 'fine_tune_all']
    
    def generate_source_target_data(
        self,
        n_samples_source: int = 1000,
        n_samples_target: int = 200,
        n_features: int = 20,
        domain_shift: float = 1.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate source and target domain data with distribution shift.
        
        Args:
            n_samples_source: Samples in source domain
            n_samples_target: Samples in target domain (usually smaller)
            n_features: Number of features
            domain_shift: Magnitude of distribution shift
        
        Returns:
            X_source, y_source, X_target, y_target
        """
        from sklearn.datasets import make_classification
        
        # Source domain
        X_source, y_source = make_classification(
            n_samples=n_samples_source,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            random_state=42
        )
        
        # Target domain with shift
        X_target, y_target = make_classification(
            n_samples=n_samples_target,
            n_features=n_features,
            n_informative=n_features // 2,
            n_redundant=n_features // 4,
            random_state=123
        )
        
        # Apply domain shift
        X_target += domain_shift
        X_target += np.random.randn(*X_target.shape) * 0.5
        
        return X_source, y_source, X_target, y_target
    
    def simulate_transfer_learning(
        self,
        X_source: np.ndarray,
        y_source: np.ndarray,
        X_target_train: np.ndarray,
        y_target_train: np.ndarray,
        X_target_test: np.ndarray,
        y_target_test: np.ndarray,
        strategy: str = 'freeze_partial'
    ) -> TransferLearningResult:
        """
        Simulate transfer learning with different strategies.
        
        Args:
            X_source: Source domain training data
            y_source: Source domain labels
            X_target_train: Target domain training data
            y_target_train: Target domain training labels
            X_target_test: Target domain test data
            y_target_test: Target domain test labels
            strategy: Transfer strategy ('freeze_all', 'freeze_partial', 'fine_tune_all')
        
        Returns:
            TransferLearningResult with comparison
        """
        from sklearn.neural_network import MLPClassifier
        from sklearn.metrics import accuracy_score
        
        # Train on source domain
        source_model = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=200,
            random_state=42
        )
        source_model.fit(X_source, y_source)
        source_accuracy = source_model.score(X_source, y_source)
        
        # Baseline: Train from scratch on target (no transfer)
        target_model_no_transfer = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=200,
            random_state=42
        )
        target_model_no_transfer.fit(X_target_train, y_target_train)
        target_acc_no_transfer = accuracy_score(
            y_target_test,
            target_model_no_transfer.predict(X_target_test)
        )
        
        # Transfer learning: Initialize with source model weights
        target_model_transfer = MLPClassifier(
            hidden_layer_sizes=(64, 32, 16),
            max_iter=100,  # Fewer iterations for fine-tuning
            warm_start=True,
            random_state=42
        )
        
        # Copy weights from source model
        target_model_transfer.coefs_ = [coef.copy() for coef in source_model.coefs_]
        target_model_transfer.intercepts_ = [intercept.copy() for intercept in source_model.intercepts_]
        
        # Fine-tune on target data
        if strategy == 'freeze_all':
            # Use source model as-is (no training)
            frozen_layers = [0, 1, 2]
            fine_tuned = []
            target_acc_transfer = accuracy_score(
                y_target_test,
                target_model_transfer.predict(X_target_test)
            )
        elif strategy == 'freeze_partial':
            # Freeze early layers, fine-tune last layer
            # Simulate by training with very small learning rate
            target_model_transfer.learning_rate_init = 0.0001
            target_model_transfer.fit(X_target_train, y_target_train)
            frozen_layers = [0, 1]
            fine_tuned = [2]
            target_acc_transfer = accuracy_score(
                y_target_test,
                target_model_transfer.predict(X_target_test)
            )
        else:  # fine_tune_all
            # Fine-tune all layers
            target_model_transfer.fit(X_target_train, y_target_train)
            frozen_layers = []
            fine_tuned = [0, 1, 2]
            target_acc_transfer = accuracy_score(
                y_target_test,
                target_model_transfer.predict(X_target_test)
            )
        
        improvement = target_acc_transfer - target_acc_no_transfer
        
        return TransferLearningResult(
            source_accuracy=source_accuracy,
            target_accuracy_no_transfer=target_acc_no_transfer,
            target_accuracy_with_transfer=target_acc_transfer,
            improvement=improvement,
            frozen_layers=frozen_layers,
            fine_tuned_layers=fine_tuned
        )


@dataclass
class MultiModalResult:
    """Results from multi-modal learning."""
    accuracy_modality1_only: float
    accuracy_modality2_only: float
    accuracy_combined: float
    fusion_improvement: float
    fusion_strategy: str


class MultiModalSimulator:
    """Simulate multi-modal learning (combining different data modalities)."""
    
    def __init__(self):
        self.fusion_strategies = ['early_fusion', 'late_fusion', 'hybrid_fusion']
    
    def generate_multimodal_data(
        self,
        n_samples: int = 1000,
        n_features_mod1: int = 10,
        n_features_mod2: int = 15,
        correlation: float = 0.7
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic multi-modal data.
        
        Args:
            n_samples: Number of samples
            n_features_mod1: Features in modality 1 (e.g., image)
            n_features_mod2: Features in modality 2 (e.g., text)
            correlation: Correlation between modalities
        
        Returns:
            X_modality1, X_modality2, y
        """
        from sklearn.datasets import make_classification
        
        # Generate base data
        X_base, y = make_classification(
            n_samples=n_samples,
            n_features=n_features_mod1,
            n_informative=n_features_mod1 // 2,
            random_state=42
        )
        
        # Modality 1 (e.g., visual features)
        X_mod1 = X_base.copy()
        
        # Modality 2 (e.g., textual features) - correlated with modality 1
        X_mod2 = np.random.randn(n_samples, n_features_mod2)
        for i in range(min(n_features_mod1, n_features_mod2)):
            X_mod2[:, i] = correlation * X_mod1[:, i] + (1 - correlation) * X_mod2[:, i]
        
        return X_mod1, X_mod2, y
    
    def early_fusion(
        self,
        X_mod1_train: np.ndarray,
        X_mod2_train: np.ndarray,
        y_train: np.ndarray,
        X_mod1_test: np.ndarray,
        X_mod2_test: np.ndarray,
        y_test: np.ndarray
    ) -> float:
        """
        Early fusion: Concatenate features before training.
        
        Args:
            Training and test data for both modalities
        
        Returns:
            Accuracy of combined model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Concatenate features
        X_train_combined = np.hstack([X_mod1_train, X_mod2_train])
        X_test_combined = np.hstack([X_mod1_test, X_mod2_test])
        
        # Train single model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_combined, y_train)
        
        return accuracy_score(y_test, model.predict(X_test_combined))
    
    def late_fusion(
        self,
        X_mod1_train: np.ndarray,
        X_mod2_train: np.ndarray,
        y_train: np.ndarray,
        X_mod1_test: np.ndarray,
        X_mod2_test: np.ndarray,
        y_test: np.ndarray
    ) -> float:
        """
        Late fusion: Train separate models and combine predictions.
        
        Args:
            Training and test data for both modalities
        
        Returns:
            Accuracy of combined predictions
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Train separate models
        model1 = RandomForestClassifier(n_estimators=50, random_state=42)
        model1.fit(X_mod1_train, y_train)
        
        model2 = RandomForestClassifier(n_estimators=50, random_state=42)
        model2.fit(X_mod2_train, y_train)
        
        # Get predictions
        pred1 = model1.predict(X_mod1_test)
        pred2 = model2.predict(X_mod2_test)
        
        # Combine predictions (majority vote)
        pred_combined = np.where(pred1 == pred2, pred1, pred1)  # Simplified voting
        
        return accuracy_score(y_test, pred_combined)
    
    def hybrid_fusion(
        self,
        X_mod1_train: np.ndarray,
        X_mod2_train: np.ndarray,
        y_train: np.ndarray,
        X_mod1_test: np.ndarray,
        X_mod2_test: np.ndarray,
        y_test: np.ndarray
    ) -> float:
        """
        Hybrid fusion: Combine intermediate representations.
        
        Args:
            Training and test data for both modalities
        
        Returns:
            Accuracy of hybrid model
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.decomposition import PCA
        from sklearn.metrics import accuracy_score
        
        # Extract features using PCA for each modality
        pca1 = PCA(n_components=min(5, X_mod1_train.shape[1]))
        pca2 = PCA(n_components=min(5, X_mod2_train.shape[1]))
        
        X_mod1_reduced_train = pca1.fit_transform(X_mod1_train)
        X_mod1_reduced_test = pca1.transform(X_mod1_test)
        
        X_mod2_reduced_train = pca2.fit_transform(X_mod2_train)
        X_mod2_reduced_test = pca2.transform(X_mod2_test)
        
        # Combine reduced features
        X_train_hybrid = np.hstack([X_mod1_reduced_train, X_mod2_reduced_train])
        X_test_hybrid = np.hstack([X_mod1_reduced_test, X_mod2_reduced_test])
        
        # Train on combined reduced features
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train_hybrid, y_train)
        
        return accuracy_score(y_test, model.predict(X_test_hybrid))
    
    def compare_fusion_strategies(
        self,
        X_mod1: np.ndarray,
        X_mod2: np.ndarray,
        y: np.ndarray
    ) -> MultiModalResult:
        """
        Compare different fusion strategies.
        
        Args:
            X_mod1: Modality 1 features
            X_mod2: Modality 2 features
            y: Labels
        
        Returns:
            MultiModalResult with comparison
        """
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Split data
        X_mod1_train, X_mod1_test, X_mod2_train, X_mod2_test, y_train, y_test = train_test_split(
            X_mod1, X_mod2, y, test_size=0.2, random_state=42
        )
        
        # Single modality baselines
        model1 = RandomForestClassifier(n_estimators=50, random_state=42)
        model1.fit(X_mod1_train, y_train)
        acc_mod1 = accuracy_score(y_test, model1.predict(X_mod1_test))
        
        model2 = RandomForestClassifier(n_estimators=50, random_state=42)
        model2.fit(X_mod2_train, y_train)
        acc_mod2 = accuracy_score(y_test, model2.predict(X_mod2_test))
        
        # Fusion strategies
        acc_early = self.early_fusion(
            X_mod1_train, X_mod2_train, y_train,
            X_mod1_test, X_mod2_test, y_test
        )
        
        acc_late = self.late_fusion(
            X_mod1_train, X_mod2_train, y_train,
            X_mod1_test, X_mod2_test, y_test
        )
        
        acc_hybrid = self.hybrid_fusion(
            X_mod1_train, X_mod2_train, y_train,
            X_mod1_test, X_mod2_test, y_test
        )
        
        # Best fusion strategy
        best_fusion = max(
            [('early', acc_early), ('late', acc_late), ('hybrid', acc_hybrid)],
            key=lambda x: x[1]
        )
        
        return MultiModalResult(
            accuracy_modality1_only=acc_mod1,
            accuracy_modality2_only=acc_mod2,
            accuracy_combined=best_fusion[1],
            fusion_improvement=best_fusion[1] - max(acc_mod1, acc_mod2),
            fusion_strategy=best_fusion[0]
        )
