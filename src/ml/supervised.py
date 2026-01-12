"""
Comprehensive Supervised Learning Module
==========================================
Complete end-to-end supervised learning with classification and regression.
Includes data preprocessing, model training, evaluation, and deployment.

Author: Data Science Pro
"""

from __future__ import annotations

import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import pickle
import json

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV,
    learning_curve,
    StratifiedKFold,
    KFold,
)
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    LabelEncoder,
    OneHotEncoder,
    PolynomialFeatures,
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
)
from sklearn.metrics import (
    # Classification
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    log_loss,
    hamming_loss,
    # Regression
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error,
    median_absolute_error,
)

# Model imports
from sklearn.linear_model import (
    LogisticRegression,
    Ridge,
    Lasso,
    ElasticNet,
    LinearRegression,
)
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    VotingClassifier,
    VotingRegressor,
    BaggingClassifier,
    BaggingRegressor,
    StackingClassifier,
    StackingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Optional imports
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except (ImportError, Exception):
    # Catch both ImportError and library loading errors (like OpenMP missing)
    HAS_XGBOOST = False

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except (ImportError, Exception):
    HAS_LIGHTGBM = False

warnings.filterwarnings("ignore")


@dataclass
class DataPreprocessor:
    """Handle data preprocessing, feature scaling, and encoding."""
    
    scaler_type: str = "standard"  # standard, minmax, robust
    handle_missing: str = "mean"  # mean, median, drop
    encode_type: str = "onehot"  # onehot, label
    outlier_method: str = "iqr"  # iqr, zscore, none
    outlier_threshold: float = 3.0
    
    def __post_init__(self):
        self.scaler = self._get_scaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.feature_names_: Optional[List[str]] = None
        self.numeric_features_: List[str] = []
        self.categorical_features_: List[str] = []
    
    def _get_scaler(self):
        """Get appropriate scaler."""
        if self.scaler_type == "minmax":
            return MinMaxScaler()
        elif self.scaler_type == "robust":
            return RobustScaler()
        else:
            return StandardScaler()
    
    def fit(self, X: pd.DataFrame) -> DataPreprocessor:
        """Fit preprocessor to data."""
        self.feature_names_ = X.columns.tolist()
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Fit scaler on numeric features
        if self.numeric_features_:
            self.scaler.fit(X[self.numeric_features_])
        
        # Fit label encoders for categorical features
        for col in self.categorical_features_:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.label_encoders[col] = le
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data."""
        X_copy = X.copy()
        
        # Handle missing values
        if self.handle_missing == "mean":
            X_copy[self.numeric_features_] = X_copy[self.numeric_features_].fillna(
                X_copy[self.numeric_features_].mean()
            )
        elif self.handle_missing == "median":
            X_copy[self.numeric_features_] = X_copy[self.numeric_features_].fillna(
                X_copy[self.numeric_features_].median()
            )
        
        # Fill categorical with mode
        for col in self.categorical_features_:
            X_copy[col] = X_copy[col].fillna(X_copy[col].mode()[0] if not X_copy[col].mode().empty else "Unknown")
        
        # Handle outliers
        if self.outlier_method == "iqr":
            for col in self.numeric_features_:
                Q1 = X_copy[col].quantile(0.25)
                Q3 = X_copy[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - self.outlier_threshold * IQR
                upper_bound = Q3 + self.outlier_threshold * IQR
                X_copy[col] = X_copy[col].clip(lower_bound, upper_bound)
        
        # Scale numeric features
        if self.numeric_features_:
            X_copy[self.numeric_features_] = self.scaler.transform(X_copy[self.numeric_features_])
        
        # Encode categorical features
        for col in self.categorical_features_:
            if col in self.label_encoders:
                X_copy[col] = self.label_encoders[col].transform(X_copy[col].astype(str))
        
        return X_copy
    
    def fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X).transform(X)


@dataclass
class SupervisedLearningModel:
    """Comprehensive supervised learning model wrapper."""
    
    task_type: str  # 'classification' or 'regression'
    model_type: str  # 'logistic', 'random_forest', 'xgboost', etc.
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    preprocessor: Optional[DataPreprocessor] = None
    
    def __post_init__(self):
        self.model = None
        self.preprocessor = self.preprocessor or DataPreprocessor()
        self.X_train_ = None
        self.X_test_ = None
        self.y_train_ = None
        self.y_test_ = None
        self.train_history_: Dict[str, Any] = {}
        self.evaluation_results_: Dict[str, Any] = {}
        self.feature_importance_: Optional[pd.DataFrame] = None
        self.predictions_ = None
        self.feature_engineering_: Dict[str, Any] = {}
    
    def _get_model(self):
        """Get model instance based on type."""
        task = self.task_type.lower()
        mtype = self.model_type.lower()
        
        if task == "classification":
            if mtype == "logistic":
                return LogisticRegression(max_iter=1000, random_state=self.random_state)
            elif mtype == "random_forest":
                return RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            elif mtype == "gradient_boosting":
                return GradientBoostingClassifier(random_state=self.random_state)
            elif mtype == "svm":
                return SVC(probability=True, random_state=self.random_state)
            elif mtype == "knn":
                return KNeighborsClassifier(n_neighbors=5)
            elif mtype == "decision_tree":
                return DecisionTreeClassifier(random_state=self.random_state)
            elif mtype == "naive_bayes":
                return GaussianNB()
            elif mtype == "mlp":
                return MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.random_state)
            elif mtype == "adaboost":
                return AdaBoostClassifier(random_state=self.random_state)
            elif mtype == "xgboost" and HAS_XGBOOST:
                return xgb.XGBClassifier(random_state=self.random_state, use_label_encoder=False, eval_metric='logloss')
            elif mtype == "lightgbm" and HAS_LIGHTGBM:
                return lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
            else:
                return RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        
        else:  # regression
            if mtype == "linear":
                return LinearRegression()
            elif mtype == "ridge":
                return Ridge(alpha=1.0)
            elif mtype == "lasso":
                return Lasso(alpha=0.1)
            elif mtype == "elasticnet":
                return ElasticNet(alpha=0.1)
            elif mtype == "random_forest":
                return RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
            elif mtype == "gradient_boosting":
                return GradientBoostingRegressor(random_state=self.random_state)
            elif mtype == "svm":
                return SVR(kernel='rbf')
            elif mtype == "knn":
                return KNeighborsRegressor(n_neighbors=5)
            elif mtype == "mlp":
                return MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=self.random_state)
            elif mtype == "xgboost" and HAS_XGBOOST:
                return xgb.XGBRegressor(random_state=self.random_state)
            elif mtype == "lightgbm" and HAS_LIGHTGBM:
                return lgb.LGBMRegressor(random_state=self.random_state, verbose=-1)
            else:
                return RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
    
    def prepare_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare and split data."""
        # Preprocess features
        X_processed = self.preprocessor.fit_transform(X)
        
        # Split data
        stratify = y if self.stratify and self.task_type == "classification" else None
        self.X_train_, self.X_test_, self.y_train_, self.y_test_ = train_test_split(
            X_processed,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=stratify
        )
        
        return self.X_train_, self.X_test_, self.y_train_, self.y_test_
    
    def train(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> SupervisedLearningModel:
        """Train the model."""
        # Prepare data
        self.prepare_data(X, y)
        
        # Create and train model
        self.model = self._get_model()
        self.model.fit(self.X_train_, self.y_train_, **kwargs)
        
        # Store training info
        self.train_history_["model_type"] = self.model_type
        self.train_history_["task_type"] = self.task_type
        self.train_history_["training_samples"] = len(self.y_train_)
        self.train_history_["test_samples"] = len(self.y_test_)
        
        return self
    
    def evaluate(self, threshold: float = 0.5) -> Dict[str, Any]:
        """Evaluate model on test set."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        self.evaluation_results_ = {}
        
        # Get predictions
        self.predictions_ = self.model.predict(self.X_test_)
        
        if self.task_type == "classification":
            # Classification metrics
            self.evaluation_results_["accuracy"] = accuracy_score(self.y_test_, self.predictions_)
            
            # Handle binary vs multi-class
            if len(np.unique(self.y_test_)) == 2:
                self.evaluation_results_["precision"] = precision_score(self.y_test_, self.predictions_, zero_division=0)
                self.evaluation_results_["recall"] = recall_score(self.y_test_, self.predictions_, zero_division=0)
                self.evaluation_results_["f1"] = f1_score(self.y_test_, self.predictions_, zero_division=0)
                
                # ROC-AUC
                try:
                    y_pred_proba = self.model.predict_proba(self.X_test_)[:, 1]
                    self.evaluation_results_["roc_auc"] = roc_auc_score(self.y_test_, y_pred_proba)
                except:
                    pass
            else:
                self.evaluation_results_["precision"] = precision_score(self.y_test_, self.predictions_, average='weighted', zero_division=0)
                self.evaluation_results_["recall"] = recall_score(self.y_test_, self.predictions_, average='weighted', zero_division=0)
                self.evaluation_results_["f1"] = f1_score(self.y_test_, self.predictions_, average='weighted', zero_division=0)
            
            self.evaluation_results_["confusion_matrix"] = confusion_matrix(self.y_test_, self.predictions_)
            self.evaluation_results_["classification_report"] = classification_report(self.y_test_, self.predictions_, output_dict=True)
        
        else:  # regression
            self.evaluation_results_["mse"] = mean_squared_error(self.y_test_, self.predictions_)
            self.evaluation_results_["rmse"] = np.sqrt(self.evaluation_results_["mse"])
            self.evaluation_results_["mae"] = mean_absolute_error(self.y_test_, self.predictions_)
            self.evaluation_results_["r2"] = r2_score(self.y_test_, self.predictions_)
            
            try:
                self.evaluation_results_["mape"] = mean_absolute_percentage_error(self.y_test_, self.predictions_)
            except:
                pass
        
        return self.evaluation_results_
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, List[float]]:
        """Perform cross-validation."""
        X_processed = self.preprocessor.fit_transform(X)
        
        if self.task_type == "classification":
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
            scoring = ['r2', 'neg_mean_squared_error', 'neg_mean_absolute_error']
        
        model = self._get_model()
        results = cross_validate(model, X_processed, y, cv=cv_splitter, scoring=scoring, return_train_score=True)
        
        return results
    
    def get_feature_importance(self, X: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Get feature importance scores."""
        if self.model is None:
            return None
        
        try:
            # For tree-based models
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = self.preprocessor.feature_names_ or [f"Feature_{i}" for i in range(len(importances))]
                
                self.feature_importance_ = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                return self.feature_importance_
            
            # For linear models
            elif hasattr(self.model, 'coef_'):
                coef = np.abs(self.model.coef_.flatten())
                feature_names = self.preprocessor.feature_names_ or [f"Feature_{i}" for i in range(len(coef))]
                
                self.feature_importance_ = pd.DataFrame({
                    'feature': feature_names,
                    'importance': coef
                }).sort_values('importance', ascending=False)
                
                return self.feature_importance_
        except Exception as e:
            print(f"Could not compute feature importance: {e}")
        
        return None
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict(X_processed)
    
    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Get prediction probabilities (classification only)."""
        if self.model is None or self.task_type != "classification":
            return None
        
        if not hasattr(self.model, 'predict_proba'):
            return None
        
        X_processed = self.preprocessor.transform(X)
        return self.model.predict_proba(X_processed)
    
    def save(self, filepath: Union[str, Path]) -> None:
        """Save model to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'model': self.model,
            'preprocessor': self.preprocessor,
            'task_type': self.task_type,
            'model_type': self.model_type,
            'evaluation_results': self.evaluation_results_,
            'feature_importance': self.feature_importance_,
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @staticmethod
    def load(filepath: Union[str, Path]) -> SupervisedLearningModel:
        """Load model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        instance = SupervisedLearningModel(
            task_type=data['task_type'],
            model_type=data['model_type'],
            preprocessor=data['preprocessor']
        )
        instance.model = data['model']
        instance.evaluation_results_ = data['evaluation_results']
        instance.feature_importance_ = data['feature_importance']
        
        return instance


@dataclass
class HyperparameterOptimizer:
    """Grid and Random search for hyperparameter optimization."""
    
    model: SupervisedLearningModel
    search_type: str = "grid"  # 'grid', 'random', 'bayesian'
    cv: int = 5
    n_iter: int = 20
    n_jobs: int = -1
    
    def optimize(self, X: pd.DataFrame, y: pd.Series, param_grid: Dict[str, Any]) -> Dict[str, Any]:
        """Run hyperparameter search."""
        X_processed = model.preprocessor.fit_transform(X)
        
        if self.search_type == "grid":
            searcher = GridSearchCV(
                model.model,
                param_grid,
                cv=self.cv,
                scoring='accuracy' if model.task_type == 'classification' else 'r2',
                n_jobs=self.n_jobs
            )
        else:  # random
            searcher = RandomizedSearchCV(
                model.model,
                param_grid,
                n_iter=self.n_iter,
                cv=self.cv,
                random_state=model.random_state,
                n_jobs=self.n_jobs
            )
        
        searcher.fit(X_processed, y)
        
        return {
            'best_params': searcher.best_params_,
            'best_score': searcher.best_score_,
            'cv_results': searcher.cv_results_
        }


def create_ensemble(
    task_type: str,
    models: List[Tuple[str, Any]],
    ensemble_type: str = "voting"
) -> Union[VotingClassifier, VotingRegressor]:
    """Create ensemble model."""
    if task_type == "classification":
        return VotingClassifier(estimators=models, voting='soft')
    else:
        return VotingRegressor(estimators=models)

# ============================================================================
# ADVANCED MODERN FEATURES - Model Calibration & Diagnostics
# ============================================================================

class ModelCalibration:
    """Advanced model calibration for better probability predictions."""
    
    @staticmethod
    def calibrate_classifier(
        model: Any,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        method: str = "sigmoid"
    ) -> Any:
        """
        Calibrate classifier using modern methods.
        
        Methods:
        - sigmoid: Platt scaling (default)
        - isotonic: Non-parametric isotonic regression
        """
        from sklearn.calibration import CalibratedClassifierCV
        
        calibrated = CalibratedClassifierCV(
            estimator=model,
            method=method,
            cv='prefit'
        )
        
        calibrated.fit(X_cal, y_cal)
        return calibrated
    
    @staticmethod
    def get_calibration_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict:
        """
        Calculate calibration quality metrics.
        """
        from sklearn.calibration import calibration_curve
        
        prob_true, prob_pred = calibration_curve(
            y_true,
            y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba,
            n_bins=10,
            strategy='uniform'
        )
        
        calibration_error = np.mean(np.abs(prob_true - prob_pred))
        
        return {
            'calibration_error': float(calibration_error),
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist(),
            'brier_score': float(np.mean((y_pred_proba.max(axis=1) - y_true) ** 2))
        }


class ClassImbalanceHandler:
    """Handle class imbalance with modern techniques."""
    
    @staticmethod
    def get_class_weights(y: np.ndarray, strategy: str = "balanced") -> Dict:
        """
        Calculate class weights for imbalanced datasets.
        
        Strategies:
        - balanced: Inverse of class frequency
        - balanced_subsample: For tree-based models
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight(
            class_weight=strategy,
            classes=classes,
            y=y
        )
        
        return {cls: weight for cls, weight in zip(classes, weights)}
    
    @staticmethod
    def apply_smote(X: np.ndarray, y: np.ndarray, sampling_strategy: float = 0.5) -> Tuple:
        """
        Apply SMOTE for minority class oversampling.
        Falls back to simple oversampling if imbalanced-learn unavailable.
        """
        try:
            from imblearn.over_sampling import SMOTE
            
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=42,
                n_jobs=-1
            )
            return smote.fit_resample(X, y)
        except ImportError:
            # Fallback: simple random oversampling
            from sklearn.utils import resample
            
            X_resampled = X.copy()
            y_resampled = y.copy()
            
            for cls in np.unique(y):
                X_minority = X[y == cls]
                y_minority = y[y == cls]
                
                target_size = max(len(y_resampled[y_resampled == cls]), 
                                int(len(X) * sampling_strategy))
                
                if len(X_minority) < target_size:
                    X_oversampled, y_oversampled = resample(
                        X_minority,
                        y_minority,
                        n_samples=target_size,
                        replace=True,
                        random_state=42
                    )
                    X_resampled = np.vstack([X_resampled, X_oversampled[len(X_minority):]])
                    y_resampled = np.hstack([y_resampled, y_oversampled[len(y_minority):]])
            
            return X_resampled, y_resampled


class AdvancedDiagnostics:
    """Modern diagnostic tools for model analysis."""
    
    @staticmethod
    def get_residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        Analyze residuals for regression models.
        """
        residuals = y_true - y_pred
        
        return {
            'mean_residual': float(np.mean(residuals)),
            'std_residual': float(np.std(residuals)),
            'skewness': float(pd.Series(residuals).skew()),
            'kurtosis': float(pd.Series(residuals).kurtosis()),
            'normality_test': float(np.mean((residuals - np.mean(residuals)) ** 3) / (np.std(residuals) ** 3)),
            'residuals': residuals.tolist()
        }
    
    @staticmethod
    def get_prediction_intervals(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        confidence: float = 0.95
    ) -> Dict:
        """
        Calculate prediction intervals (95% by default).
        """
        residuals = y_true - y_pred
        residual_std = np.std(residuals)
        
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        margin = z_score * residual_std
        
        return {
            'lower_bound': (y_pred - margin).tolist(),
            'upper_bound': (y_pred + margin).tolist(),
            'margin': float(margin)
        }
    
    @staticmethod
    def feature_stability_analysis(
        X: pd.DataFrame,
        y: np.ndarray,
        model: Any,
        n_splits: int = 5
    ) -> Dict:
        """
        Analyze feature importance stability across CV folds.
        """
        from sklearn.model_selection import KFold
        
        feature_importances = []
        skf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf.split(X):
            X_train, y_train = X.iloc[train_idx], y[train_idx]
            
            model_clone = type(model)(**model.get_params())
            model_clone.fit(X_train, y_train)
            
            if hasattr(model_clone, 'feature_importances_'):
                feature_importances.append(model_clone.feature_importances_)
        
        if feature_importances:
            fi_array = np.array(feature_importances)
            
            return {
                'mean_importance': fi_array.mean(axis=0).tolist(),
                'std_importance': fi_array.std(axis=0).tolist(),
                'cv_folds': n_splits,
                'stability_score': float(1 - fi_array.std(axis=0).mean())
            }
        
        return {'error': 'Model does not support feature importance'}


class PerformanceOptimizer:
    """Optimize model performance with modern techniques."""
    
    @staticmethod
    def threshold_optimization(
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        metric: str = 'f1'
    ) -> Dict:
        """
        Find optimal classification threshold for custom metrics.
        """
        from sklearn.metrics import precision_recall_curve, f1_score, roc_curve
        
        if metric == 'f1':
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        elif metric == 'roc':
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        else:
            optimal_threshold = 0.5
        
        return {
            'optimal_threshold': float(optimal_threshold),
            'metric': metric
        }
    
    @staticmethod
    def get_learning_efficiency(
        train_sizes: np.ndarray,
        train_scores: np.ndarray,
        val_scores: np.ndarray
    ) -> Dict:
        """
        Analyze learning efficiency and convergence.
        """
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        
        # Gap between training and validation
        generalization_gap = train_mean - val_mean
        
        # Convergence rate (improvement per doubling of samples)
        improvement_rate = None
        if len(train_sizes) >= 2:
            improvement_rate = (val_mean[-1] - val_mean[0]) / np.log2(train_sizes[-1] / train_sizes[0])
        
        return {
            'train_scores': train_mean.tolist(),
            'val_scores': val_mean.tolist(),
            'generalization_gap': generalization_gap.tolist(),
            'convergence_rate': float(improvement_rate) if improvement_rate else None,
            'final_gap': float(generalization_gap[-1])
        }