"""Core Simulation Engine with Synthetic Data Generation and Parameter Control."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_blobs,
    make_moons,
    make_circles,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@dataclass
class SimulationParameters:
    """Container for simulation configuration parameters."""
    
    # Data generation
    n_samples: int = 1000
    n_features: int = 10
    n_informative: int = 8
    n_redundant: int = 2
    n_classes: int = 2
    random_state: int = 42
    
    # Data quality
    noise_level: float = 0.1
    missing_rate: float = 0.0
    outlier_rate: float = 0.0
    class_imbalance: Optional[List[float]] = None
    feature_correlation: float = 0.0
    
    # Model complexity
    max_depth: Optional[int] = 10
    n_estimators: int = 100
    learning_rate: float = 0.1
    epochs: int = 100
    
    # Training
    test_size: float = 0.2
    use_gpu: bool = False
    
    # Simulation metadata
    simulation_id: str = ""
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


class DataGenerator:
    """Advanced synthetic data generator with realistic scenarios."""
    
    @staticmethod
    def generate_classification(
        params: SimulationParameters,
        dataset_type: str = "standard"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate classification datasets with various patterns.
        
        Args:
            params: Simulation parameters
            dataset_type: Type of dataset pattern
                - standard: Linear separable
                - moons: Non-linear crescent patterns
                - circles: Concentric circles
                - imbalanced: Class imbalance
                - noisy: High noise
        
        Returns:
            X, y arrays
        """
        np.random.seed(params.random_state)
        
        if dataset_type == "moons":
            X, y = make_moons(
                n_samples=params.n_samples,
                noise=params.noise_level,
                random_state=params.random_state
            )
        elif dataset_type == "circles":
            X, y = make_circles(
                n_samples=params.n_samples,
                noise=params.noise_level,
                factor=0.5,
                random_state=params.random_state
            )
        elif dataset_type == "blobs":
            X, y = make_blobs(
                n_samples=params.n_samples,
                n_features=params.n_features,
                centers=params.n_classes,
                cluster_std=params.noise_level * 10,
                random_state=params.random_state
            )
        else:  # standard
            weights = params.class_imbalance if params.class_imbalance else None
            
            # Ensure n_informative + n_redundant <= n_features
            # Also ensure n_classes * n_clusters_per_class <= 2^n_informative
            # Reserve at least 1 feature for noise if possible
            max_informative_redundant = max(1, params.n_features - 1)
            n_informative = min(params.n_informative, max_informative_redundant)
            n_redundant = min(params.n_redundant, max_informative_redundant - n_informative)
            
            # Calculate required n_informative for the constraint:
            # n_classes * n_clusters_per_class <= 2^n_informative
            # For default n_clusters_per_class=2: n_classes * 2 <= 2^n_informative
            # So n_informative >= log2(n_classes * 2)
            import math
            min_n_informative = math.ceil(math.log2(params.n_classes * 2))
            
            # Ensure n_informative meets the constraint
            n_informative = max(n_informative, min(min_n_informative, params.n_features))
            
            # Adjust n_redundant if needed after increasing n_informative
            n_redundant = min(n_redundant, max(0, params.n_features - n_informative - 1))
            
            X, y = make_classification(
                n_samples=params.n_samples,
                n_features=params.n_features,
                n_informative=n_informative,
                n_redundant=n_redundant,
                n_classes=params.n_classes,
                weights=weights,
                flip_y=params.noise_level,
                random_state=params.random_state
            )
        
        # Add correlation between features
        if params.feature_correlation > 0 and X.shape[1] > 1:
            X = DataGenerator._add_feature_correlation(X, params.feature_correlation)
        
        # Add missing values
        if params.missing_rate > 0:
            X = DataGenerator._add_missing_values(X, params.missing_rate)
        
        # Add outliers
        if params.outlier_rate > 0:
            X = DataGenerator._add_outliers(X, params.outlier_rate)
        
        return X, y
    
    @staticmethod
    def generate_regression(
        params: SimulationParameters,
        pattern: str = "linear"
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate regression datasets with various patterns.
        
        Args:
            params: Simulation parameters
            pattern: Data pattern type
                - linear: Linear relationship
                - polynomial: Polynomial relationship
                - exponential: Exponential growth
                - sinusoidal: Periodic pattern
                - step: Step function
        
        Returns:
            X, y arrays
        """
        np.random.seed(params.random_state)
        
        if pattern == "linear":
            # Ensure n_informative <= n_features
            n_informative = min(params.n_informative, params.n_features)
            
            X, y = make_regression(
                n_samples=params.n_samples,
                n_features=params.n_features,
                n_informative=n_informative,
                noise=params.noise_level * 100,
                random_state=params.random_state
            )
        
        elif pattern == "polynomial":
            X = np.random.randn(params.n_samples, params.n_features)
            y = (X[:, 0] ** 2 + 
                 2 * X[:, 1] ** 3 + 
                 np.random.randn(params.n_samples) * params.noise_level * 10)
        
        elif pattern == "exponential":
            X = np.random.randn(params.n_samples, params.n_features)
            y = (np.exp(X[:, 0] * 0.5) + 
                 np.random.randn(params.n_samples) * params.noise_level * 10)
        
        elif pattern == "sinusoidal":
            X = np.random.randn(params.n_samples, params.n_features)
            X[:, 0] = np.sort(X[:, 0])
            y = (np.sin(X[:, 0] * 2) * 10 + 
                 np.random.randn(params.n_samples) * params.noise_level * 5)
        
        elif pattern == "step":
            X = np.random.randn(params.n_samples, params.n_features)
            y = (np.where(X[:, 0] > 0, 10, -10) + 
                 np.random.randn(params.n_samples) * params.noise_level * 5)
        
        else:
            X, y = make_regression(
                n_samples=params.n_samples,
                n_features=params.n_features,
                noise=params.noise_level * 100,
                random_state=params.random_state
            )
        
        # Add missing values and outliers
        if params.missing_rate > 0:
            X = DataGenerator._add_missing_values(X, params.missing_rate)
        
        if params.outlier_rate > 0:
            X = DataGenerator._add_outliers(X, params.outlier_rate)
            y = DataGenerator._add_outliers(y.reshape(-1, 1), params.outlier_rate).ravel()
        
        return X, y
    
    @staticmethod
    def generate_time_series(
        params: SimulationParameters,
        pattern: str = "trend_seasonal"
    ) -> pd.DataFrame:
        """
        Generate time series data with various patterns.
        
        Args:
            params: Simulation parameters
            pattern: Time series pattern
                - trend_seasonal: Trend + seasonality
                - random_walk: Random walk
                - autoregressive: AR process
                - cyclical: Business cycles
                - anomaly: With anomalies
        
        Returns:
            DataFrame with datetime index and value column
        """
        np.random.seed(params.random_state)
        
        dates = pd.date_range('2020-01-01', periods=params.n_samples, freq='D')
        t = np.arange(params.n_samples)
        
        if pattern == "trend_seasonal":
            trend = 0.1 * t
            seasonal = 10 * np.sin(2 * np.pi * t / 365)
            noise = np.random.randn(params.n_samples) * params.noise_level * 5
            values = trend + seasonal + noise
        
        elif pattern == "random_walk":
            values = np.cumsum(np.random.randn(params.n_samples) * params.noise_level * 5)
        
        elif pattern == "autoregressive":
            values = np.zeros(params.n_samples)
            values[0] = np.random.randn()
            for i in range(1, params.n_samples):
                values[i] = 0.8 * values[i-1] + np.random.randn() * params.noise_level * 5
        
        elif pattern == "cyclical":
            cycle1 = 20 * np.sin(2 * np.pi * t / 90)  # Quarterly
            cycle2 = 10 * np.sin(2 * np.pi * t / 30)  # Monthly
            noise = np.random.randn(params.n_samples) * params.noise_level * 5
            values = cycle1 + cycle2 + noise + 50
        
        elif pattern == "anomaly":
            trend = 0.05 * t
            noise = np.random.randn(params.n_samples) * params.noise_level * 2
            values = trend + noise + 50
            # Add anomalies
            anomaly_idx = np.random.choice(
                params.n_samples, 
                size=int(params.n_samples * 0.05), 
                replace=False
            )
            values[anomaly_idx] += np.random.choice([-30, 30], size=len(anomaly_idx))
        
        else:
            values = np.cumsum(np.random.randn(params.n_samples))
        
        return pd.DataFrame({
            'date': dates,
            'value': values
        }).set_index('date')
    
    @staticmethod
    def _add_feature_correlation(X: np.ndarray, correlation: float) -> np.ndarray:
        """Add correlation between features."""
        X_corr = X.copy()
        n_features = X.shape[1]
        
        for i in range(1, n_features):
            X_corr[:, i] = (correlation * X_corr[:, 0] + 
                           (1 - correlation) * X_corr[:, i])
        
        return X_corr
    
    @staticmethod
    def _add_missing_values(X: np.ndarray, missing_rate: float) -> np.ndarray:
        """Add missing values randomly."""
        X_missing = X.copy()
        mask = np.random.rand(*X.shape) < missing_rate
        X_missing[mask] = np.nan
        return X_missing
    
    @staticmethod
    def _add_outliers(X: np.ndarray, outlier_rate: float) -> np.ndarray:
        """Add outliers to data."""
        X_outliers = X.copy()
        n_outliers = int(X.shape[0] * outlier_rate)
        
        for _ in range(n_outliers):
            idx = np.random.randint(0, X.shape[0])
            col = np.random.randint(0, X.shape[1])
            # Add extreme value (5-10 std deviations)
            X_outliers[idx, col] += np.random.choice([-1, 1]) * np.std(X[:, col]) * np.random.uniform(5, 10)
        
        return X_outliers


class SimulationEngine:
    """Orchestrates interactive ML simulations with real-time parameter updates."""
    
    def __init__(self):
        self.generator = DataGenerator()
        self.history: List[Dict[str, Any]] = []
        self.current_params: Optional[SimulationParameters] = None
        self.current_results: Dict[str, Any] = {}
    
    def run_simulation(
        self,
        params: SimulationParameters,
        simulation_type: str,
        model_fn: Callable,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run a complete simulation cycle.
        
        Args:
            params: Simulation parameters
            simulation_type: Type of simulation (classification, regression, etc.)
            model_fn: Function that trains and evaluates model
            **kwargs: Additional arguments for model_fn
        
        Returns:
            Dictionary with simulation results
        """
        start_time = time.time()
        
        # Generate data based on simulation type
        if simulation_type == "classification":
            X, y = self.generator.generate_classification(
                params, 
                dataset_type=kwargs.get('dataset_type', 'standard')
            )
        elif simulation_type == "regression":
            X, y = self.generator.generate_regression(
                params,
                pattern=kwargs.get('pattern', 'linear')
            )
        elif simulation_type == "time_series":
            data = self.generator.generate_time_series(
                params,
                pattern=kwargs.get('pattern', 'trend_seasonal')
            )
            results = {
                'data': data,
                'params': params.to_dict(),
                'execution_time': time.time() - start_time
            }
            self._record_simulation(results)
            return results
        else:
            raise ValueError(f"Unknown simulation type: {simulation_type}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=params.test_size,
            random_state=params.random_state
        )
        
        # Handle missing values for training
        if np.isnan(X_train).any():
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='mean')
            X_train = imputer.fit_transform(X_train)
            X_test = imputer.transform(X_test)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate model
        model_results = model_fn(
            X_train_scaled, X_test_scaled,
            y_train, y_test,
            params,
            **kwargs
        )
        
        # Compile results
        results = {
            'data': {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'X_train_scaled': X_train_scaled,
                'X_test_scaled': X_test_scaled,
            },
            'model_results': model_results,
            'params': params.to_dict(),
            'execution_time': time.time() - start_time,
            'data_quality': self._assess_data_quality(X, y),
        }
        
        self._record_simulation(results)
        return results
    
    def _assess_data_quality(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Assess data quality metrics."""
        return {
            'missing_rate': np.isnan(X).sum() / X.size if X.size > 0 else 0,
            'outlier_rate': self._detect_outliers(X),
            'class_balance': self._calculate_class_balance(y),
            'feature_variance': np.var(X, axis=0).tolist() if not np.isnan(X).any() else [],
        }
    
    @staticmethod
    def _detect_outliers(X: np.ndarray) -> float:
        """Detect outliers using IQR method."""
        if np.isnan(X).any():
            X_clean = X[~np.isnan(X).any(axis=1)]
        else:
            X_clean = X
        
        if len(X_clean) == 0:
            return 0.0
        
        Q1 = np.percentile(X_clean, 25, axis=0)
        Q3 = np.percentile(X_clean, 75, axis=0)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = ((X_clean < lower_bound) | (X_clean > upper_bound)).any(axis=1)
        return outliers.sum() / len(X_clean)
    
    @staticmethod
    def _calculate_class_balance(y: np.ndarray) -> Dict[int, float]:
        """Calculate class distribution."""
        unique, counts = np.unique(y, return_counts=True)
        total = len(y)
        return {int(cls): count / total for cls, count in zip(unique, counts)}
    
    def _record_simulation(self, results: Dict[str, Any]):
        """Record simulation in history."""
        self.history.append({
            'timestamp': time.time(),
            'params': results['params'],
            'execution_time': results.get('execution_time', 0),
            'results_summary': self._summarize_results(results)
        })
        self.current_results = results
    
    @staticmethod
    def _summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of results."""
        summary = {}
        
        if 'model_results' in results:
            model_results = results['model_results']
            if isinstance(model_results, dict):
                for key in ['accuracy', 'f1_score', 'r2_score', 'mse', 'mae']:
                    if key in model_results:
                        summary[key] = model_results[key]
        
        if 'data_quality' in results:
            summary['data_quality'] = results['data_quality']
        
        return summary
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get simulation history."""
        return self.history
    
    def clear_history(self):
        """Clear simulation history."""
        self.history = []
        self.current_results = {}
