"""
Model Evaluation and Analysis Module
=====================================
Comprehensive model evaluation, comparison, and interpretation tools.

Author: Data Science Pro
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    # Classification
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    classification_report,
    # Regression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)
from sklearn.model_selection import learning_curve, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore")


@dataclass
class ClassificationEvaluator:
    """Evaluate classification models comprehensively."""
    
    y_true: np.ndarray
    y_pred: np.ndarray
    y_pred_proba: Optional[np.ndarray] = None
    class_names: Optional[List[str]] = None
    
    def __post_init__(self):
        self.metrics_ = {}
        self.confusion_matrix_ = None
    
    def evaluate(self) -> Dict[str, Any]:
        """Calculate all classification metrics."""
        self.metrics_ = {}
        
        # Basic metrics
        self.metrics_['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        
        # Handle binary vs multi-class
        is_binary = len(np.unique(self.y_true)) == 2
        
        if is_binary:
            self.metrics_['precision'] = precision_score(self.y_true, self.y_pred, zero_division=0)
            self.metrics_['recall'] = recall_score(self.y_true, self.y_pred, zero_division=0)
            self.metrics_['f1'] = f1_score(self.y_true, self.y_pred, zero_division=0)
            
            if self.y_pred_proba is not None:
                self.metrics_['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba[:, 1])
        else:
            self.metrics_['precision'] = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
            self.metrics_['recall'] = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
            self.metrics_['f1'] = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
            
            if self.y_pred_proba is not None:
                try:
                    self.metrics_['roc_auc'] = roc_auc_score(self.y_true, self.y_pred_proba, multi_class='ovr')
                except:
                    pass
        
        # Confusion matrix
        self.confusion_matrix_ = confusion_matrix(self.y_true, self.y_pred)
        self.metrics_['confusion_matrix'] = self.confusion_matrix_
        
        # Classification report
        self.metrics_['classification_report'] = classification_report(
            self.y_true,
            self.y_pred,
            output_dict=True,
            zero_division=0
        )
        
        return self.metrics_
    
    def get_confusion_matrix(self) -> np.ndarray:
        """Get confusion matrix."""
        if self.confusion_matrix_ is None:
            self.evaluate()
        return self.confusion_matrix_
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve (binary classification only)."""
        if self.y_pred_proba is None:
            raise ValueError("Probability predictions required for ROC curve.")
        
        if len(np.unique(self.y_true)) != 2:
            raise ValueError("ROC curve only for binary classification.")
        
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba[:, 1])
        return fpr, tpr, thresholds
    
    def get_precision_recall_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get precision-recall curve (binary classification only)."""
        if self.y_pred_proba is None:
            raise ValueError("Probability predictions required.")
        
        if len(np.unique(self.y_true)) != 2:
            raise ValueError("Precision-recall curve only for binary classification.")
        
        precision, recall, thresholds = precision_recall_curve(self.y_true, self.y_pred_proba[:, 1])
        return precision, recall, thresholds
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of all metrics."""
        if not self.metrics_:
            self.evaluate()
        
        metrics_dict = {k: v for k, v in self.metrics_.items() 
                       if k not in ['confusion_matrix', 'classification_report']}
        
        return pd.DataFrame([metrics_dict])


@dataclass
class RegressionEvaluator:
    """Evaluate regression models comprehensively."""
    
    y_true: np.ndarray
    y_pred: np.ndarray
    
    def __post_init__(self):
        self.metrics_ = {}
    
    def evaluate(self) -> Dict[str, Any]:
        """Calculate all regression metrics."""
        self.metrics_ = {}
        
        # Error-based metrics
        self.metrics_['mae'] = mean_absolute_error(self.y_true, self.y_pred)
        self.metrics_['mse'] = mean_squared_error(self.y_true, self.y_pred)
        self.metrics_['rmse'] = np.sqrt(self.metrics_['mse'])
        self.metrics_['median_ae'] = median_absolute_error(self.y_true, self.y_pred)
        
        # Percentage metrics
        try:
            self.metrics_['mape'] = mean_absolute_percentage_error(self.y_true, self.y_pred)
        except:
            pass
        
        # R² and adjusted R²
        self.metrics_['r2'] = r2_score(self.y_true, self.y_pred)
        
        # Directional accuracy
        direction_correct = np.sum(np.sign(self.y_pred - self.y_true[:-1]) == np.sign(self.y_true[1:] - self.y_true[:-1]))
        self.metrics_['directional_accuracy'] = direction_correct / (len(self.y_true) - 1) * 100
        
        return self.metrics_
    
    def get_residuals(self) -> np.ndarray:
        """Get prediction residuals."""
        return self.y_true - self.y_pred
    
    def get_metrics_summary(self) -> pd.DataFrame:
        """Get summary of all metrics."""
        if not self.metrics_:
            self.evaluate()
        
        return pd.DataFrame([self.metrics_])


@dataclass
class ModelComparator:
    """Compare multiple models systematically."""
    
    models: Dict[str, Any] = field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def add_model(self, name: str, model: Any) -> ModelComparator:
        """Add a model to compare."""
        self.models[name] = model
        return self
    
    def evaluate_classification(
        self,
        y_true: np.ndarray,
        y_pred_dict: Dict[str, np.ndarray],
        y_pred_proba_dict: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple classification models."""
        for model_name, y_pred in y_pred_dict.items():
            y_proba = y_pred_proba_dict.get(model_name) if y_pred_proba_dict else None
            
            evaluator = ClassificationEvaluator(y_true, y_pred, y_proba)
            self.results[model_name] = evaluator.evaluate()
        
        return self.results
    
    def evaluate_regression(
        self,
        y_true: np.ndarray,
        y_pred_dict: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, Any]]:
        """Evaluate multiple regression models."""
        for model_name, y_pred in y_pred_dict.items():
            evaluator = RegressionEvaluator(y_true, y_pred)
            self.results[model_name] = evaluator.evaluate()
        
        return self.results
    
    def get_comparison_table(self, metric: str = 'f1') -> pd.DataFrame:
        """Get comparison table for specific metric."""
        comparison = {}
        for model_name, metrics in self.results.items():
            comparison[model_name] = metrics.get(metric, np.nan)
        
        df = pd.DataFrame.from_dict(comparison, orient='index', columns=[metric])
        return df.sort_values(metric, ascending=False)
    
    def get_all_metrics_comparison(self) -> pd.DataFrame:
        """Get all metrics comparison table."""
        comparison_data = {}
        
        for model_name, metrics in self.results.items():
            clean_metrics = {k: v for k, v in metrics.items() 
                           if not isinstance(v, (dict, np.ndarray))}
            comparison_data[model_name] = clean_metrics
        
        df = pd.DataFrame.from_dict(comparison_data, orient='index')
        return df
    
    def get_best_model(self, metric: str = 'f1') -> Tuple[str, float]:
        """Get best model for specific metric."""
        scores = {}
        for model_name, metrics in self.results.items():
            if metric in metrics:
                scores[model_name] = metrics[metric]
        
        if not scores:
            return None, None
        
        best_name = max(scores, key=scores.get)
        return best_name, scores[best_name]


class CrossValidationAnalyzer:
    """Analyze cross-validation results."""
    
    @staticmethod
    def analyze_cv_scores(
        cv_scores: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Analyze cross-validation scores."""
        return {
            'model': model_name,
            'mean': cv_scores.mean(),
            'std': cv_scores.std(),
            'min': cv_scores.min(),
            'max': cv_scores.max(),
            'cv_scores': cv_scores.tolist()
        }
    
    @staticmethod
    def compare_cv_results(
        cv_results_dict: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Compare CV results across models."""
        comparison = []
        
        for model_name, cv_scores in cv_results_dict.items():
            comparison.append({
                'model': model_name,
                'mean': cv_scores.mean(),
                'std': cv_scores.std(),
                'min': cv_scores.min(),
                'max': cv_scores.max()
            })
        
        df = pd.DataFrame(comparison)
        return df.sort_values('mean', ascending=False)


class LearningCurveAnalyzer:
    """Analyze learning curves to detect bias and variance."""
    
    @staticmethod
    def plot_learning_curve(
        estimator: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        train_sizes: np.ndarray = np.linspace(0.1, 1.0, 10),
        scoring: str = 'accuracy'
    ) -> Dict[str, Any]:
        """Generate learning curve data."""
        train_sizes_abs, train_scores, val_scores = learning_curve(
            estimator,
            X, y,
            cv=cv,
            train_sizes=train_sizes,
            scoring=scoring,
            n_jobs=-1
        )
        
        return {
            'train_sizes': train_sizes_abs,
            'train_scores': train_scores,
            'val_scores': val_scores,
            'train_mean': train_scores.mean(axis=1),
            'train_std': train_scores.std(axis=1),
            'val_mean': val_scores.mean(axis=1),
            'val_std': val_scores.std(axis=1)
        }
    
    @staticmethod
    def diagnose_bias_variance(curve_data: Dict[str, Any]) -> str:
        """Diagnose bias-variance issue from learning curve."""
        train_mean = curve_data['train_mean']
        val_mean = curve_data['val_mean']
        
        final_train = train_mean[-1]
        final_val = val_mean[-1]
        gap = final_train - final_val
        
        if final_val < 0.7:
            if gap < 0.05:
                return "High Bias: Model underfitting. Needs more complexity."
            else:
                return "High Variance: Model overfitting. Needs regularization."
        else:
            if gap < 0.05:
                return "Good Fit: Model performing well."
            else:
                return "Some Variance: Consider regularization or more data."


class FeatureImportanceAnalyzer:
    """Analyze and interpret feature importance."""
    
    @staticmethod
    def analyze_importance(
        importance_scores: np.ndarray,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """Analyze feature importance scores."""
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance_scores
        })
        
        df['importance_pct'] = (df['importance'] / df['importance'].sum()) * 100
        df['cumulative_importance'] = df['importance_pct'].cumsum()
        
        return df.sort_values('importance', ascending=False).head(top_n)
    
    @staticmethod
    def find_critical_features(
        importance_df: pd.DataFrame,
        cumulative_threshold: float = 0.8
    ) -> List[str]:
        """Find critical features that explain threshold % of importance."""
        critical = importance_df[importance_df['cumulative_importance'] <= cumulative_threshold]
        return critical['feature'].tolist()


class ModelInterpretability:
    """Tools for interpreting model predictions."""
    
    @staticmethod
    def permutation_importance(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10
    ) -> pd.DataFrame:
        """Calculate permutation importance."""
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model, X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': result.importances_mean,
            'std': result.importances_std
        })
        
        return importances.sort_values('importance', ascending=False)
    
    @staticmethod
    def analyze_predictions(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        predictions_df: pd.DataFrame,
        top_n: int = 10
    ) -> Dict[str, Any]:
        """Analyze model predictions by performance."""
        residuals = y_true - y_pred
        abs_residuals = np.abs(residuals)
        
        worst_predictions_idx = np.argsort(abs_residuals)[-top_n:]
        best_predictions_idx = np.argsort(abs_residuals)[:top_n]
        
        return {
            'worst_predictions': predictions_df.iloc[worst_predictions_idx],
            'best_predictions': predictions_df.iloc[best_predictions_idx],
            'mean_residual': residuals.mean(),
            'std_residual': residuals.std(),
            'max_error': abs_residuals.max(),
            'median_error': np.median(abs_residuals)
        }

# ============================================================================
# ADVANCED MODERN EVALUATION METRICS & TECHNIQUES
# ============================================================================

@dataclass
class AdvancedMetricsCalculator:
    """Modern metrics for comprehensive model evaluation."""
    
    @staticmethod
    def calculate_auc_pr(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate Area Under Precision-Recall Curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        auc_pr = auc(recall, precision)
        return float(auc_pr)
    
    @staticmethod
    def calculate_specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate specificity (true negative rate)."""
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        return float(specificity)
    
    @staticmethod
    def calculate_sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate sensitivity (true positive rate)."""
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        return float(sensitivity)
    
    @staticmethod
    def calculate_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate balanced accuracy for imbalanced datasets."""
        from sklearn.metrics import balanced_accuracy_score
        return float(balanced_accuracy_score(y_true, y_pred))
    
    @staticmethod
    def calculate_matthews_corr_coeff(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Matthews Correlation Coefficient (robust to class imbalance)."""
        from sklearn.metrics import matthews_corrcoef
        return float(matthews_corrcoef(y_true, y_pred))
    
    @staticmethod
    def calculate_cohens_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Cohen's Kappa (accounts for chance agreement)."""
        from sklearn.metrics import cohen_kappa_score
        return float(cohen_kappa_score(y_true, y_pred))


@dataclass
class RobustnessAnalyzer:
    """Analyze model robustness across different conditions."""
    
    @staticmethod
    def cross_validation_stability(
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        cv: int = 5,
        metric: str = 'accuracy'
    ) -> Dict:
        """
        Analyze cross-validation stability and variance.
        """
        from sklearn.model_selection import cross_val_score
        
        scores = cross_val_score(model, X, y, cv=cv, scoring=metric)
        
        return {
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'cv_scores': scores.tolist(),
            'coefficient_of_variation': float(scores.std() / scores.mean()) if scores.mean() != 0 else 0
        }
    
    @staticmethod
    def perturbation_sensitivity(
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        metric_fn: Callable,
        perturbation_level: float = 0.1
    ) -> Dict:
        """
        Test model sensitivity to data perturbations.
        """
        baseline_pred = model.predict(X)
        baseline_score = metric_fn(y, baseline_pred)
        
        perturbation_scores = []
        
        for col in X.select_dtypes(include=[np.number]).columns:
            X_perturbed = X.copy()
            std = X_perturbed[col].std()
            X_perturbed[col] += np.random.normal(0, std * perturbation_level, len(X))
            
            perturbed_pred = model.predict(X_perturbed)
            perturbed_score = metric_fn(y, perturbed_pred)
            
            sensitivity = abs(baseline_score - perturbed_score) / baseline_score
            perturbation_scores.append({
                'feature': col,
                'sensitivity': float(sensitivity)
            })
        
        return {
            'baseline_score': float(baseline_score),
            'perturbation_level': perturbation_level,
            'sensitivities': perturbation_scores,
            'mean_sensitivity': float(np.mean([s['sensitivity'] for s in perturbation_scores]))
        }


@dataclass
class FairnessAnalyzer:
    """Analyze model fairness across demographic groups."""
    
    @staticmethod
    def demographic_parity(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_labels: np.ndarray
    ) -> Dict:
        """
        Calculate demographic parity (equal selection rates across groups).
        """
        unique_groups = np.unique(group_labels)
        results = {}
        
        for group in unique_groups:
            group_mask = group_labels == group
            group_pred = y_pred[group_mask]
            
            selection_rate = np.mean(group_pred)
            results[f'group_{group}'] = float(selection_rate)
        
        # Calculate parity ratio (min/max)
        if results.values():
            parity_ratio = min(results.values()) / max(results.values())
        else:
            parity_ratio = 1.0
        
        return {
            'selection_rates': results,
            'parity_ratio': float(parity_ratio),
            'is_fair': float(parity_ratio) >= 0.8  # Typically 80/20 rule
        }
    
    @staticmethod
    def equalized_odds(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        group_labels: np.ndarray
    ) -> Dict:
        """
        Calculate equalized odds (equal TPR and FPR across groups).
        """
        unique_groups = np.unique(group_labels)
        tpr_by_group = {}
        fpr_by_group = {}
        
        for group in unique_groups:
            group_mask = group_labels == group
            y_true_group = y_true[group_mask]
            y_pred_group = y_pred[group_mask]
            
            tp = np.sum((y_pred_group == 1) & (y_true_group == 1))
            fn = np.sum((y_pred_group == 0) & (y_true_group == 1))
            fp = np.sum((y_pred_group == 1) & (y_true_group == 0))
            tn = np.sum((y_pred_group == 0) & (y_true_group == 0))
            
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            
            tpr_by_group[f'group_{group}'] = float(tpr)
            fpr_by_group[f'group_{group}'] = float(fpr)
        
        return {
            'tpr_by_group': tpr_by_group,
            'fpr_by_group': fpr_by_group,
            'tpr_disparity': float(max(tpr_by_group.values()) - min(tpr_by_group.values())),
            'fpr_disparity': float(max(fpr_by_group.values()) - min(fpr_by_group.values()))
        }