"""Scenario-Specific Simulators for Classification, Regression, Time-Series, and What-If Analysis."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)

from src.simulation.engine import SimulationEngine, SimulationParameters


class ClassificationSimulator:
    """Interactive classification simulation with multiple algorithms."""
    
    ALGORITHMS = {
        'Logistic Regression': LogisticRegression,
        'Decision Tree': DecisionTreeClassifier,
        'Random Forest': RandomForestClassifier,
        'Gradient Boosting': GradientBoostingClassifier,
        'SVM': SVC,
    }
    
    def __init__(self):
        self.engine = SimulationEngine()
        self.comparison_results: Dict[str, Any] = {}
    
    def run_simulation(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        algorithm: str = 'Random Forest'
    ) -> Dict[str, Any]:
        """
        Run classification simulation with pre-split data.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training labels
            y_test: Testing labels
            algorithm: Algorithm name
            
        Returns:
            Dictionary with results
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Select and configure model
        model_class = self.ALGORITHMS.get(algorithm, RandomForestClassifier)
        
        if algorithm == 'Decision Tree':
            model = model_class(max_depth=10, random_state=42)
        elif algorithm == 'Random Forest':
            model = model_class(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        elif algorithm == 'Gradient Boosting':
            model = model_class(n_estimators=100, learning_rate=0.1, max_depth=10, random_state=42)
        elif algorithm == 'SVM':
            model = model_class(kernel='rbf', random_state=42)
        else:  # Logistic Regression
            model = model_class(max_iter=1000, random_state=42)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'predictions': y_pred,
            'model': model
        }
    
    def run_single_model(
        self,
        params: SimulationParameters,
        algorithm: str = 'Random Forest',
        dataset_type: str = 'standard'
    ) -> Dict[str, Any]:
        """Run simulation with a single classification model."""
        
        def model_fn(X_train, X_test, y_train, y_test, params, **kwargs):
            # Select and configure model
            model_class = self.ALGORITHMS.get(algorithm, RandomForestClassifier)
            
            if algorithm == 'Decision Tree':
                model = model_class(
                    max_depth=params.max_depth,
                    random_state=params.random_state
                )
            elif algorithm == 'Random Forest':
                model = model_class(
                    n_estimators=params.n_estimators,
                    max_depth=params.max_depth,
                    random_state=params.random_state,
                    n_jobs=-1
                )
            elif algorithm == 'Gradient Boosting':
                model = model_class(
                    n_estimators=params.n_estimators,
                    learning_rate=params.learning_rate,
                    max_depth=params.max_depth,
                    random_state=params.random_state
                )
            elif algorithm == 'SVM':
                model = model_class(
                    kernel='rbf',
                    random_state=params.random_state
                )
            else:  # Logistic Regression
                model = model_class(
                    max_iter=1000,
                    random_state=params.random_state
                )
            
            # Train
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train
            
            # Predict
            start_pred = time.time()
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            pred_time = time.time() - start_pred
            
            # Evaluate
            results = {
                'algorithm': algorithm,
                'model': model,
                'train_accuracy': accuracy_score(y_train, y_pred_train),
                'test_accuracy': accuracy_score(y_test, y_pred_test),
                'precision': precision_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'recall': recall_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'f1_score': f1_score(y_test, y_pred_test, average='weighted', zero_division=0),
                'confusion_matrix': confusion_matrix(y_test, y_pred_test),
                'classification_report': classification_report(y_test, y_pred_test, zero_division=0),
                'predictions': {'train': y_pred_train, 'test': y_pred_test},
                'train_time': train_time,
                'pred_time': pred_time,
            }
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = model.feature_importances_
            
            # Calculate bias-variance indicators
            results['overfitting_score'] = results['train_accuracy'] - results['test_accuracy']
            
            return results
        
        return self.engine.run_simulation(
            params,
            'classification',
            model_fn,
            dataset_type=dataset_type
        )
    
    def compare_models(
        self,
        params: SimulationParameters,
        algorithms: Optional[List[str]] = None,
        dataset_type: str = 'standard'
    ) -> Dict[str, Any]:
        """Run side-by-side comparison of multiple classification algorithms."""
        
        if algorithms is None:
            algorithms = list(self.ALGORITHMS.keys())
        
        comparison = {
            'algorithms': algorithms,
            'results': {},
            'summary': {},
        }
        
        for algo in algorithms:
            result = self.run_single_model(params, algo, dataset_type)
            comparison['results'][algo] = result
            
            # Extract key metrics for summary
            model_res = result['model_results']
            comparison['summary'][algo] = {
                'test_accuracy': model_res['test_accuracy'],
                'f1_score': model_res['f1_score'],
                'train_time': model_res['train_time'],
                'overfitting': model_res['overfitting_score'],
            }
        
        self.comparison_results = comparison
        return comparison


class RegressionSimulator:
    """Interactive regression simulation with multiple algorithms."""
    
    ALGORITHMS = {
        'Linear Regression': LinearRegression,
        'Ridge Regression': Ridge,
        'Lasso Regression': Lasso,
        'Decision Tree': DecisionTreeRegressor,
        'Random Forest': RandomForestRegressor,
        'SVR': SVR,
    }
    
    def __init__(self):
        self.engine = SimulationEngine()
        self.comparison_results: Dict[str, Any] = {}
    
    def run_simulation(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        algorithm: str = 'Random Forest'
    ) -> Dict[str, Any]:
        """
        Run regression simulation with pre-split data.
        
        Args:
            X_train: Training features
            X_test: Testing features
            y_train: Training targets
            y_test: Testing targets
            algorithm: Algorithm name
            
        Returns:
            Dictionary with results
        """
        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
        
        # Select and configure model
        model_class = self.ALGORITHMS.get(algorithm, RandomForestRegressor)
        
        if algorithm == 'Decision Tree':
            model = model_class(max_depth=10, random_state=42)
        elif algorithm == 'Random Forest':
            model = model_class(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        elif algorithm in ['Ridge Regression', 'Lasso Regression']:
            model = model_class(alpha=0.1, random_state=42)
        else:
            model = model_class(random_state=42)
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        return {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'predictions': y_pred,
            'residuals': y_test - y_pred,
            'model': model
        }
    
    def run_single_model(
        self,
        params: SimulationParameters,
        algorithm: str = 'Random Forest',
        pattern: str = 'linear'
    ) -> Dict[str, Any]:
        """Run simulation with a single regression model."""
        
        def model_fn(X_train, X_test, y_train, y_test, params, **kwargs):
            # Select and configure model
            model_class = self.ALGORITHMS.get(algorithm, RandomForestRegressor)
            
            if algorithm == 'Decision Tree':
                model = model_class(
                    max_depth=params.max_depth,
                    random_state=params.random_state
                )
            elif algorithm == 'Random Forest':
                model = model_class(
                    n_estimators=params.n_estimators,
                    max_depth=params.max_depth,
                    random_state=params.random_state,
                    n_jobs=-1
                )
            elif algorithm in ['Ridge Regression', 'Lasso Regression']:
                model = model_class(
                    alpha=params.learning_rate,
                    random_state=params.random_state
                )
            else:
                model = model_class(random_state=params.random_state)
            
            # Train
            start_train = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_train
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            results = {
                'algorithm': algorithm,
                'model': model,
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mse': mean_squared_error(y_train, y_pred_train),
                'test_mse': mean_squared_error(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'predictions': {'train': y_pred_train, 'test': y_pred_test},
                'residuals': {'train': y_train - y_pred_train, 'test': y_test - y_pred_test},
                'train_time': train_time,
            }
            
            # Add feature importance if available
            if hasattr(model, 'feature_importances_'):
                results['feature_importance'] = model.feature_importances_
            elif hasattr(model, 'coef_'):
                results['coefficients'] = model.coef_
            
            # Calculate overfitting score
            results['overfitting_score'] = results['train_r2'] - results['test_r2']
            
            return results
        
        return self.engine.run_simulation(
            params,
            'regression',
            model_fn,
            pattern=pattern
        )
    
    def compare_models(
        self,
        params: SimulationParameters,
        algorithms: Optional[List[str]] = None,
        pattern: str = 'linear'
    ) -> Dict[str, Any]:
        """Run side-by-side comparison of multiple regression algorithms."""
        
        if algorithms is None:
            algorithms = list(self.ALGORITHMS.keys())
        
        comparison = {
            'algorithms': algorithms,
            'results': {},
            'summary': {},
        }
        
        for algo in algorithms:
            result = self.run_single_model(params, algo, pattern)
            comparison['results'][algo] = result
            
            model_res = result['model_results']
            comparison['summary'][algo] = {
                'test_r2': model_res['test_r2'],
                'test_rmse': model_res['test_rmse'],
                'train_time': model_res['train_time'],
                'overfitting': model_res['overfitting_score'],
            }
        
        self.comparison_results = comparison
        return comparison


class TimeSeriesSimulator:
    """Interactive time series simulation and forecasting."""
    
    def __init__(self):
        self.engine = SimulationEngine()
    
    def generate_and_analyze(
        self,
        params: SimulationParameters,
        pattern: str = 'trend_seasonal'
    ) -> Dict[str, Any]:
        """Generate time series data and perform analysis."""
        
        result = self.engine.run_simulation(
            params,
            'time_series',
            lambda *args, **kwargs: {},  # Dummy function
            pattern=pattern
        )
        
        data = result['data']
        
        # Calculate statistics
        analysis = {
            'mean': data['value'].mean(),
            'std': data['value'].std(),
            'min': data['value'].min(),
            'max': data['value'].max(),
            'trend': self._calculate_trend(data['value'].values),
            'seasonality_detected': self._detect_seasonality(data['value'].values),
            'stationarity': self._test_stationarity(data['value'].values),
        }
        
        result['analysis'] = analysis
        return result
    
    @staticmethod
    def _calculate_trend(values: np.ndarray) -> float:
        """Calculate trend using linear regression."""
        x = np.arange(len(values))
        coeffs = np.polyfit(x, values, 1)
        return coeffs[0]
    
    @staticmethod
    def _detect_seasonality(values: np.ndarray) -> bool:
        """Simple seasonality detection using autocorrelation."""
        if len(values) < 30:
            return False
        
        # Check autocorrelation at various lags
        from scipy.stats import pearsonr
        
        for lag in [7, 30, 90, 365]:
            if len(values) > lag:
                corr, _ = pearsonr(values[:-lag], values[lag:])
                if abs(corr) > 0.5:
                    return True
        return False
    
    @staticmethod
    def _test_stationarity(values: np.ndarray) -> str:
        """Test stationarity using rolling statistics."""
        if len(values) < 50:
            return "insufficient_data"
        
        # Rolling mean and std
        window = min(30, len(values) // 3)
        rolling_mean = pd.Series(values).rolling(window=window).mean()
        rolling_std = pd.Series(values).rolling(window=window).std()
        
        # Check if mean and std are relatively constant
        mean_var = rolling_mean.var()
        std_var = rolling_std.var()
        
        if mean_var < 1 and std_var < 1:
            return "stationary"
        else:
            return "non_stationary"


class OverfittingSimulator:
    """Simulate and visualize overfitting vs underfitting scenarios."""
    
    def __init__(self):
        self.engine = SimulationEngine()
    
    def run_complexity_analysis(
        self,
        params: SimulationParameters,
        complexity_range: Dict[str, List[Any]],
        problem_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Analyze model performance across complexity spectrum.
        
        Args:
            params: Base simulation parameters
            complexity_range: Dict with complexity parameter ranges
                e.g., {'max_depth': [1, 2, 5, 10, 20, None]}
            problem_type: 'classification' or 'regression'
        
        Returns:
            Results showing bias-variance tradeoff
        """
        
        results = {
            'complexity_param': list(complexity_range.keys())[0],
            'complexity_values': [],
            'train_scores': [],
            'test_scores': [],
            'overfitting_scores': [],
            'train_times': [],
        }
        
        param_name = results['complexity_param']
        param_values = complexity_range[param_name]
        
        for value in param_values:
            # Update parameter
            test_params = SimulationParameters(**params.to_dict())
            setattr(test_params, param_name, value)
            
            # Run simulation
            if problem_type == 'classification':
                sim = ClassificationSimulator()
                result = sim.run_single_model(test_params)
                train_score = result['model_results']['train_accuracy']
                test_score = result['model_results']['test_accuracy']
            else:
                sim = RegressionSimulator()
                result = sim.run_single_model(test_params)
                train_score = result['model_results']['train_r2']
                test_score = result['model_results']['test_r2']
            
            results['complexity_values'].append(str(value))
            results['train_scores'].append(train_score)
            results['test_scores'].append(test_score)
            results['overfitting_scores'].append(train_score - test_score)
            results['train_times'].append(result['model_results']['train_time'])
        
        # Identify sweet spot (minimum overfitting with good performance)
        overfitting_array = np.array(results['overfitting_scores'])
        test_array = np.array(results['test_scores'])
        
        # Find index with good test score and low overfitting
        if problem_type == 'classification':
            good_performance = test_array > 0.7
        else:
            good_performance = test_array > 0.5
        
        if good_performance.any():
            candidate_indices = np.where(good_performance)[0]
            best_idx = candidate_indices[np.argmin(overfitting_array[candidate_indices])]
        else:
            best_idx = np.argmax(test_array)
        
        results['optimal_complexity'] = {
            'value': param_values[best_idx],
            'index': int(best_idx),
            'train_score': results['train_scores'][best_idx],
            'test_score': results['test_scores'][best_idx],
            'overfitting': results['overfitting_scores'][best_idx],
        }
        
        return results


class WhatIfSimulator:
    """Run what-if scenario simulations to understand data quality impact."""
    
    def __init__(self):
        self.engine = SimulationEngine()
    
    def run_scenario(
        self,
        base_params: SimulationParameters,
        scenario: str,
        problem_type: str = 'classification'
    ) -> Dict[str, Any]:
        """
        Run specific what-if scenario.
        
        Args:
            base_params: Baseline parameters
            scenario: Scenario type
                - 'data_quality_degradation': Increase noise and missing values
                - 'sample_size_impact': Vary sample size
                - 'feature_reduction': Remove features
                - 'class_imbalance': Introduce imbalance
                - 'outlier_injection': Add outliers
            problem_type: 'classification' or 'regression'
        
        Returns:
            Comparison of baseline vs scenario results
        """
        
        scenarios_config = {
            'data_quality_degradation': [
                {'noise_level': 0.0, 'missing_rate': 0.0},
                {'noise_level': 0.1, 'missing_rate': 0.05},
                {'noise_level': 0.2, 'missing_rate': 0.1},
                {'noise_level': 0.3, 'missing_rate': 0.2},
            ],
            'sample_size_impact': [
                {'n_samples': 100},
                {'n_samples': 500},
                {'n_samples': 1000},
                {'n_samples': 5000},
            ],
            'feature_reduction': [
                {'n_features': 20, 'n_informative': 18},
                {'n_features': 10, 'n_informative': 8},
                {'n_features': 5, 'n_informative': 4},
                {'n_features': 2, 'n_informative': 2},
            ],
            'class_imbalance': [
                {'class_imbalance': [0.5, 0.5]},
                {'class_imbalance': [0.7, 0.3]},
                {'class_imbalance': [0.9, 0.1]},
                {'class_imbalance': [0.95, 0.05]},
            ],
            'outlier_injection': [
                {'outlier_rate': 0.0},
                {'outlier_rate': 0.05},
                {'outlier_rate': 0.1},
                {'outlier_rate': 0.2},
            ],
        }
        
        if scenario not in scenarios_config:
            raise ValueError(f"Unknown scenario: {scenario}")
        
        results = {
            'scenario': scenario,
            'variations': [],
            'metrics': [],
        }
        
        for variation in scenarios_config[scenario]:
            # Create modified parameters
            test_params = SimulationParameters(**base_params.to_dict())
            for key, value in variation.items():
                setattr(test_params, key, value)
            
            # Run simulation
            if problem_type == 'classification':
                sim = ClassificationSimulator()
                result = sim.run_single_model(test_params)
                metric = result['model_results']['test_accuracy']
            else:
                sim = RegressionSimulator()
                result = sim.run_single_model(test_params)
                metric = result['model_results']['test_r2']
            
            results['variations'].append(variation)
            results['metrics'].append(metric)
        
        return results


class AdversarialMLSimulator:
    """Simulate adversarial attacks on ML models."""
    
    def __init__(self):
        self.attack_types = ['evasion', 'poisoning', 'model_inversion', 'membership_inference']
        self.engine = SimulationEngine()
    
    def evasion_attack(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        epsilon: float = 0.1,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Simulate evasion attack (adversarial examples at test time).
        
        Args:
            model: Trained model
            X: Test samples
            y: True labels
            epsilon: Perturbation magnitude
            n_samples: Number of samples to attack
        
        Returns:
            Attack results with success rate
        """
        from sklearn.metrics import accuracy_score
        
        # Select random samples
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        X_attack = X[indices].copy()
        y_attack = y[indices]
        
        # Original predictions
        y_pred_original = model.predict(X_attack)
        original_accuracy = accuracy_score(y_attack, y_pred_original)
        
        # Generate adversarial examples (FGSM-like)
        adversarial_examples = []
        successful_attacks = 0
        
        for i in range(len(X_attack)):
            # Add random perturbation
            perturbation = np.random.randn(*X_attack[i].shape) * epsilon
            x_adv = X_attack[i] + perturbation
            
            # Clip to valid range if needed
            x_adv = np.clip(x_adv, X.min(), X.max())
            
            adversarial_examples.append(x_adv)
            
            # Check if attack succeeded (prediction changed)
            y_pred_adv = model.predict(x_adv.reshape(1, -1))[0]
            if y_pred_adv != y_pred_original[i]:
                successful_attacks += 1
        
        X_adversarial = np.array(adversarial_examples)
        y_pred_adversarial = model.predict(X_adversarial)
        adversarial_accuracy = accuracy_score(y_attack, y_pred_adversarial)
        
        return {
            'attack_type': 'evasion',
            'epsilon': epsilon,
            'original_accuracy': original_accuracy,
            'adversarial_accuracy': adversarial_accuracy,
            'attack_success_rate': successful_attacks / len(X_attack),
            'accuracy_drop': original_accuracy - adversarial_accuracy,
            'n_samples': len(X_attack)
        }
    
    def poisoning_attack(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        poison_rate: float = 0.1,
        model_class: Any = None
    ) -> Dict[str, Any]:
        """
        Simulate data poisoning attack (corrupt training data).
        
        Args:
            X_train: Clean training features
            y_train: Clean training labels
            X_test: Test features
            y_test: Test labels
            poison_rate: Fraction of training data to poison
            model_class: Model class to train
        
        Returns:
            Comparison of clean vs poisoned model performance
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        if model_class is None:
            model_class = RandomForestClassifier
        
        # Train clean model
        clean_model = model_class(n_estimators=50, random_state=42)
        clean_model.fit(X_train, y_train)
        clean_accuracy = accuracy_score(y_test, clean_model.predict(X_test))
        
        # Poison training data
        n_poison = int(len(X_train) * poison_rate)
        poison_indices = np.random.choice(len(X_train), n_poison, replace=False)
        
        X_poisoned = X_train.copy()
        y_poisoned = y_train.copy()
        
        # Flip labels for poisoned samples
        y_poisoned[poison_indices] = 1 - y_poisoned[poison_indices]
        
        # Add noise to features
        X_poisoned[poison_indices] += np.random.randn(n_poison, X_train.shape[1]) * 0.5
        
        # Train poisoned model
        poisoned_model = model_class(n_estimators=50, random_state=42)
        poisoned_model.fit(X_poisoned, y_poisoned)
        poisoned_accuracy = accuracy_score(y_test, poisoned_model.predict(X_test))
        
        return {
            'attack_type': 'poisoning',
            'poison_rate': poison_rate,
            'n_poisoned_samples': n_poison,
            'clean_accuracy': clean_accuracy,
            'poisoned_accuracy': poisoned_accuracy,
            'accuracy_drop': clean_accuracy - poisoned_accuracy,
            'attack_effectiveness': (clean_accuracy - poisoned_accuracy) / clean_accuracy
        }
    
    def backdoor_attack(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        backdoor_rate: float = 0.05,
        target_class: int = 1
    ) -> Dict[str, Any]:
        """
        Simulate backdoor attack (inject trigger pattern).
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            backdoor_rate: Fraction of training data with backdoor
            target_class: Class to force backdoor samples to
        
        Returns:
            Backdoor attack analysis
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        # Define backdoor trigger (specific pattern in features)
        trigger = np.ones(X_train.shape[1]) * 2.0
        
        # Insert backdoor into training data
        n_backdoor = int(len(X_train) * backdoor_rate)
        backdoor_indices = np.random.choice(len(X_train), n_backdoor, replace=False)
        
        X_backdoored = X_train.copy()
        y_backdoored = y_train.copy()
        
        # Add trigger to backdoored samples
        X_backdoored[backdoor_indices, :3] = trigger[:3]
        y_backdoored[backdoor_indices] = target_class
        
        # Train backdoored model
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_backdoored, y_backdoored)
        
        # Test on clean data
        clean_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Test on backdoored test data
        X_test_backdoor = X_test.copy()
        X_test_backdoor[:, :3] = trigger[:3]
        y_test_backdoor_pred = model.predict(X_test_backdoor)
        backdoor_success_rate = np.mean(y_test_backdoor_pred == target_class)
        
        return {
            'attack_type': 'backdoor',
            'backdoor_rate': backdoor_rate,
            'target_class': target_class,
            'clean_accuracy': clean_accuracy,
            'backdoor_success_rate': backdoor_success_rate,
            'n_backdoor_samples': n_backdoor
        }


class DriftDetectionSimulator:
    """Simulate and detect concept drift and data drift."""
    
    def __init__(self):
        self.drift_types = ['sudden', 'gradual', 'recurring', 'incremental']
        self.engine = SimulationEngine()
    
    def generate_drift_data(
        self,
        n_samples: int = 2000,
        n_features: int = 10,
        drift_type: str = 'sudden',
        drift_position: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate data with concept drift.
        
        Args:
            n_samples: Total samples
            n_features: Number of features
            drift_type: Type of drift ('sudden', 'gradual', 'recurring')
            drift_position: Position of drift (0-1)
        
        Returns:
            X, y, drift_indicator
        """
        from sklearn.datasets import make_classification
        
        drift_point = int(n_samples * drift_position)
        
        if drift_type == 'sudden':
            # Generate two different distributions
            X1, y1 = make_classification(
                n_samples=drift_point,
                n_features=n_features,
                n_informative=n_features // 2,
                random_state=42
            )
            X2, y2 = make_classification(
                n_samples=n_samples - drift_point,
                n_features=n_features,
                n_informative=n_features // 2,
                random_state=123
            )
            # Shift distribution
            X2 += 2.0
            
            X = np.vstack([X1, X2])
            y = np.hstack([y1, y2])
            drift_indicator = np.hstack([
                np.zeros(drift_point),
                np.ones(n_samples - drift_point)
            ])
        
        elif drift_type == 'gradual':
            # Gradually shift distribution
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features // 2,
                random_state=42
            )
            
            # Apply gradual shift
            for i in range(n_samples):
                if i > drift_point:
                    shift_amount = (i - drift_point) / (n_samples - drift_point) * 2.0
                    X[i] += shift_amount
            
            drift_indicator = np.where(
                np.arange(n_samples) > drift_point,
                (np.arange(n_samples) - drift_point) / (n_samples - drift_point),
                0
            )
        
        elif drift_type == 'recurring':
            # Cyclical drift
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features // 2,
                random_state=42
            )
            
            # Apply cyclical shift
            cycle_length = n_samples // 4
            for i in range(n_samples):
                cycle_pos = (i % cycle_length) / cycle_length
                X[i] += np.sin(cycle_pos * 2 * np.pi) * 1.5
            
            drift_indicator = np.sin(np.arange(n_samples) / cycle_length * 2 * np.pi)
        
        else:  # No drift
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=n_features // 2,
                random_state=42
            )
            drift_indicator = np.zeros(n_samples)
        
        return X, y, drift_indicator
    
    def detect_drift_ddm(
        self,
        errors: np.ndarray,
        warning_level: float = 2.0,
        drift_level: float = 3.0
    ) -> Dict[str, Any]:
        """
        Drift Detection Method (DDM) - monitors error rate.
        
        Args:
            errors: Binary array of prediction errors (1=error, 0=correct)
            warning_level: Standard deviations for warning
            drift_level: Standard deviations for drift detection
        
        Returns:
            Drift detection results
        """
        n = len(errors)
        drift_points = []
        warning_points = []
        
        error_rate = []
        std = []
        
        # Running statistics
        p_min = 1.0
        s_min = 0.0
        
        for i in range(1, n + 1):
            # Calculate error rate and std
            p_i = np.mean(errors[:i])
            s_i = np.sqrt(p_i * (1 - p_i) / i)
            
            error_rate.append(p_i)
            std.append(s_i)
            
            # Update minimum
            if p_i + s_i < p_min + s_min:
                p_min = p_i
                s_min = s_i
            
            # Check for drift
            if p_i + s_i >= p_min + drift_level * s_min:
                drift_points.append(i)
            elif p_i + s_i >= p_min + warning_level * s_min:
                warning_points.append(i)
        
        return {
            'method': 'DDM',
            'drift_points': drift_points,
            'warning_points': warning_points,
            'n_drifts': len(drift_points),
            'error_rate': error_rate,
            'std': std
        }
    
    def detect_drift_adwin(
        self,
        errors: np.ndarray,
        delta: float = 0.002
    ) -> Dict[str, Any]:
        """
        ADWIN (Adaptive Windowing) drift detection.
        
        Args:
            errors: Binary array of prediction errors
            delta: Confidence parameter
        
        Returns:
            Drift detection results
        """
        drift_points = []
        window_sizes = []
        
        window = []
        
        for i, error in enumerate(errors):
            window.append(error)
            
            if len(window) > 10:  # Minimum window size
                # Split window and compare
                mid = len(window) // 2
                mean_1 = np.mean(window[:mid])
                mean_2 = np.mean(window[mid:])
                
                # Simple drift detection: significant difference in means
                diff = abs(mean_1 - mean_2)
                threshold = np.sqrt(2 * np.log(2 / delta) / len(window))
                
                if diff > threshold:
                    drift_points.append(i)
                    window = []  # Reset window
            
            window_sizes.append(len(window))
        
        return {
            'method': 'ADWIN',
            'drift_points': drift_points,
            'n_drifts': len(drift_points),
            'window_sizes': window_sizes
        }
    
    def simulate_drift_monitoring(
        self,
        X: np.ndarray,
        y: np.ndarray,
        drift_indicator: np.ndarray,
        model_class: Any = None,
        retrain_on_drift: bool = True
    ) -> Dict[str, Any]:
        """
        Simulate continuous monitoring with drift detection and retraining.
        
        Args:
            X: Feature data (time-ordered)
            y: Labels (time-ordered)
            drift_indicator: True drift locations
            model_class: Model to use
            retrain_on_drift: Whether to retrain when drift detected
        
        Returns:
            Monitoring results
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        
        if model_class is None:
            model_class = RandomForestClassifier
        
        # Initial training
        initial_size = len(X) // 5
        model = model_class(n_estimators=30, random_state=42)
        model.fit(X[:initial_size], y[:initial_size])
        
        accuracies = []
        predictions = []
        errors = []
        retrain_points = []
        
        # Sliding window monitoring
        for i in range(initial_size, len(X)):
            # Predict
            y_pred = model.predict(X[i:i+1])[0]
            predictions.append(y_pred)
            
            # Record error
            error = 1 if y_pred != y[i] else 0
            errors.append(error)
            
            # Calculate running accuracy
            if i > initial_size + 10:
                recent_acc = accuracy_score(
                    y[i-10:i],
                    predictions[-10:]
                )
                accuracies.append(recent_acc)
                
                # Simple drift detection: accuracy drop
                if recent_acc < 0.7 and retrain_on_drift:
                    # Retrain on recent data
                    retrain_start = max(0, i - 100)
                    model.fit(X[retrain_start:i], y[retrain_start:i])
                    retrain_points.append(i)
        
        # Detect drift using DDM
        ddm_results = self.detect_drift_ddm(np.array(errors))
        
        return {
            'initial_training_size': initial_size,
            'accuracies': accuracies,
            'errors': errors,
            'retrain_points': retrain_points,
            'n_retrains': len(retrain_points),
            'drift_detection': ddm_results,
            'true_drift_points': np.where(np.diff(drift_indicator) > 0)[0].tolist()
        }
