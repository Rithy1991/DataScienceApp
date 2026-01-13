"""Uncertainty Quantification: Monte Carlo Simulations and Bootstrap Analysis."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import resample
from sklearn.model_selection import cross_val_score
from concurrent.futures import ThreadPoolExecutor, as_completed


class UncertaintyAnalyzer:
    """Analyze model uncertainty and stability using statistical methods."""
    
    def __init__(self, n_iterations: int = 100, confidence_level: float = 0.95):
        """
        Initialize uncertainty analyzer.
        
        Args:
            n_iterations: Number of Monte Carlo/bootstrap iterations
            confidence_level: Confidence level for intervals (default 95%)
        """
        self.n_iterations = n_iterations
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def monte_carlo_simulation(
        self,
        model_fn: Callable,
        X: np.ndarray,
        y: np.ndarray,
        metric_fn: Callable,
        params: Dict[str, Any],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation with random train/test splits.
        
        Args:
            model_fn: Function to train and predict with model
            X: Feature matrix
            y: Target vector
            metric_fn: Function to calculate performance metric
            params: Model parameters
            parallel: Whether to run in parallel
        
        Returns:
            Dictionary with statistics and confidence intervals
        """
        
        def single_iteration(seed):
            """Run a single Monte Carlo iteration."""
            np.random.seed(seed)
            
            # Random train/test split
            indices = np.random.permutation(len(X))
            split_point = int(0.8 * len(X))
            train_idx, test_idx = indices[:split_point], indices[split_point:]
            
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train and evaluate
            y_pred = model_fn(X_train, y_train, X_test, params)
            score = metric_fn(y_test, y_pred)
            
            return score
        
        # Run iterations
        scores = []
        
        if parallel and self.n_iterations > 10:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(single_iteration, i) 
                    for i in range(self.n_iterations)
                ]
                
                for future in as_completed(futures):
                    try:
                        scores.append(future.result())
                    except Exception as e:
                        print(f"Iteration failed: {e}")
        else:
            for i in range(self.n_iterations):
                try:
                    scores.append(single_iteration(i))
                except Exception as e:
                    print(f"Iteration {i} failed: {e}")
        
        scores = np.array(scores)
        
        # Calculate statistics
        return self._calculate_statistics(scores, "Monte Carlo Simulation")
    
    def bootstrap_analysis(
        self,
        model_fn: Callable,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        metric_fn: Callable,
        params: Dict[str, Any],
        parallel: bool = True
    ) -> Dict[str, Any]:
        """
        Run bootstrap analysis to estimate model stability.
        
        Args:
            model_fn: Function to train and predict with model
            X_train, y_train: Training data
            X_test, y_test: Test data
            metric_fn: Function to calculate performance metric
            params: Model parameters
            parallel: Whether to run in parallel
        
        Returns:
            Dictionary with bootstrap statistics
        """
        
        def single_bootstrap(seed):
            """Run a single bootstrap iteration."""
            # Bootstrap resample training data
            X_boot, y_boot = resample(
                X_train, y_train,
                random_state=seed,
                n_samples=len(X_train)
            )
            
            # Train and evaluate
            y_pred = model_fn(X_boot, y_boot, X_test, params)
            score = metric_fn(y_test, y_pred)
            
            return score
        
        # Run bootstrap iterations
        scores = []
        
        if parallel and self.n_iterations > 10:
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(single_bootstrap, i) 
                    for i in range(self.n_iterations)
                ]
                
                for future in as_completed(futures):
                    try:
                        scores.append(future.result())
                    except Exception as e:
                        print(f"Bootstrap iteration failed: {e}")
        else:
            for i in range(self.n_iterations):
                try:
                    scores.append(single_bootstrap(i))
                except Exception as e:
                    print(f"Bootstrap iteration {i} failed: {e}")
        
        scores = np.array(scores)
        
        return self._calculate_statistics(scores, "Bootstrap Analysis")
    
    def cross_validation_uncertainty(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """
        Estimate uncertainty using cross-validation.
        
        Args:
            model: Scikit-learn compatible model
            X: Feature matrix
            y: Target vector
            cv: Number of cross-validation folds
            scoring: Scoring metric
        
        Returns:
            Dictionary with CV statistics
        """
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        
        return self._calculate_statistics(scores, "Cross-Validation")
    
    def prediction_intervals(
        self,
        predictions: List[np.ndarray],
        method: str = 'percentile'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate prediction intervals from multiple predictions.
        
        Args:
            predictions: List of prediction arrays
            method: Method to calculate intervals ('percentile' or 'std')
        
        Returns:
            Lower and upper bounds of prediction intervals
        """
        
        predictions_array = np.array(predictions)
        
        if method == 'percentile':
            lower_percentile = (self.alpha / 2) * 100
            upper_percentile = (1 - self.alpha / 2) * 100
            
            lower_bound = np.percentile(predictions_array, lower_percentile, axis=0)
            upper_bound = np.percentile(predictions_array, upper_percentile, axis=0)
        
        elif method == 'std':
            mean_pred = np.mean(predictions_array, axis=0)
            std_pred = np.std(predictions_array, axis=0)
            
            # Use normal approximation
            z_score = 1.96  # 95% confidence
            lower_bound = mean_pred - z_score * std_pred
            upper_bound = mean_pred + z_score * std_pred
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return lower_bound, upper_bound
    
    def model_stability_score(
        self,
        scores: np.ndarray,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Calculate model stability score.
        
        Args:
            scores: Array of performance scores from multiple runs
            threshold: Acceptable variation threshold
        
        Returns:
            Dictionary with stability metrics
        """
        
        coefficient_of_variation = np.std(scores) / (np.mean(scores) + 1e-10)
        
        stability_score = 1 - min(coefficient_of_variation, 1.0)
        
        is_stable = coefficient_of_variation < threshold
        
        return {
            'stability_score': stability_score,
            'coefficient_of_variation': coefficient_of_variation,
            'is_stable': is_stable,
            'threshold': threshold,
            'interpretation': self._interpret_stability(stability_score)
        }
    
    def _calculate_statistics(
        self,
        scores: np.ndarray,
        method_name: str
    ) -> Dict[str, Any]:
        """Calculate comprehensive statistics from score distribution."""
        
        # Basic statistics
        mean = np.mean(scores)
        std = np.std(scores)
        median = np.median(scores)
        
        # Confidence interval
        lower_percentile = (self.alpha / 2) * 100
        upper_percentile = (1 - self.alpha / 2) * 100
        ci_lower = np.percentile(scores, lower_percentile)
        ci_upper = np.percentile(scores, upper_percentile)
        
        # Additional metrics
        min_score = np.min(scores)
        max_score = np.max(scores)
        range_score = max_score - min_score
        
        # Stability
        stability = self.model_stability_score(scores)
        
        return {
            'method': method_name,
            'n_iterations': len(scores),
            'mean': float(mean),
            'std': float(std),
            'median': float(median),
            'min': float(min_score),
            'max': float(max_score),
            'range': float(range_score),
            'confidence_interval': {
                'lower': float(ci_lower),
                'upper': float(ci_upper),
                'level': self.confidence_level
            },
            'stability': stability,
            'scores': scores.tolist(),
        }
    
    @staticmethod
    def _interpret_stability(stability_score: float) -> str:
        """Interpret stability score."""
        if stability_score > 0.95:
            return "Highly Stable - Model performance is very consistent"
        elif stability_score > 0.85:
            return "Stable - Model performance is reasonably consistent"
        elif stability_score > 0.70:
            return "Moderately Stable - Some variation in performance"
        elif stability_score > 0.50:
            return "Unstable - Significant variation in performance"
        else:
            return "Very Unstable - Model performance is highly inconsistent"


class EnsembleUncertainty:
    """Estimate uncertainty using ensemble methods."""
    
    @staticmethod
    def predict_with_uncertainty(
        models: List[Any],
        X: np.ndarray,
        problem_type: str = 'classification'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get predictions with uncertainty estimates from ensemble.
        
        Args:
            models: List of trained models
            X: Feature matrix
            problem_type: 'classification' or 'regression'
        
        Returns:
            Predictions and uncertainty estimates
        """
        
        predictions = np.array([model.predict(X) for model in models])
        
        if problem_type == 'classification':
            # Use vote entropy as uncertainty
            from scipy.stats import entropy
            
            # Get probability distributions if available
            if hasattr(models[0], 'predict_proba'):
                probas = np.array([model.predict_proba(X) for model in models])
                mean_proba = np.mean(probas, axis=0)
                uncertainty = entropy(mean_proba.T)
            else:
                # Use prediction variance
                mean_pred = np.mean(predictions, axis=0)
                uncertainty = np.std(predictions, axis=0)
        
        else:  # regression
            mean_pred = np.mean(predictions, axis=0)
            uncertainty = np.std(predictions, axis=0)
        
        return mean_pred, uncertainty
    
    @staticmethod
    def calculate_prediction_confidence(
        predictions: np.ndarray,
        method: str = 'std'
    ) -> np.ndarray:
        """
        Calculate confidence scores for predictions.
        
        Args:
            predictions: Array of shape (n_models, n_samples)
            method: Method to calculate confidence
        
        Returns:
            Confidence scores (0-1, higher is more confident)
        """
        
        if method == 'std':
            # Lower std = higher confidence
            std = np.std(predictions, axis=0)
            max_std = np.max(std) + 1e-10
            confidence = 1 - (std / max_std)
        
        elif method == 'agreement':
            # Higher agreement = higher confidence
            mode = np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(),
                axis=0,
                arr=predictions
            )
            agreement = np.mean(predictions == mode, axis=0)
            confidence = agreement
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return confidence
