"""
Advanced Feature Engineering Module
====================================
Feature selection, transformation, interaction creation, and encoding strategies.

Author: Data Science Pro
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, entropy

# Feature engineering
from sklearn.preprocessing import (
    PolynomialFeatures,
    KBinsDiscretizer,
    FunctionTransformer,
)
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
    RFE,
    chi2,
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

warnings.filterwarnings("ignore")


class FeatureScaler:
    """Normalize and scale features appropriately."""
    
    @staticmethod
    def scale_numeric(X: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
        """Scale numeric features."""
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        X_numeric = X.select_dtypes(include=[np.number])
        
        if method == "standard":
            scaler = StandardScaler()
        elif method == "minmax":
            scaler = MinMaxScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        X_scaled = scaler.fit_transform(X_numeric)
        result = X.copy()
        result[X_numeric.columns] = X_scaled
        
        return result
    
    @staticmethod
    def log_transform(X: pd.DataFrame, offset: float = 1) -> pd.DataFrame:
        """Apply log transformation to reduce skewness."""
        X_numeric = X.select_dtypes(include=[np.number])
        result = X.copy()
        
        for col in X_numeric.columns:
            if (X_numeric[col] > 0).all():
                result[col] = np.log1p(X_numeric[col] + offset)
        
        return result
    
    @staticmethod
    def box_cox_transform(X: pd.DataFrame) -> pd.DataFrame:
        """Apply Box-Cox transformation."""
        from scipy.stats import boxcox
        
        X_numeric = X.select_dtypes(include=[np.number])
        result = X.copy()
        
        for col in X_numeric.columns:
            if (X_numeric[col] > 0).all():
                result[col], _ = boxcox(X_numeric[col] + 1)
        
        return result
    
    @staticmethod
    def yeo_johnson_transform(X: pd.DataFrame) -> pd.DataFrame:
        """Apply Yeo-Johnson transformation (handles negative values)."""
        from scipy.stats import yeojohnson
        
        X_numeric = X.select_dtypes(include=[np.number])
        result = X.copy()
        
        for col in X_numeric.columns:
            result[col], _ = yeojohnson(X_numeric[col])
        
        return result


class FeatureCreator:
    """Create new features through transformations and interactions."""
    
    @staticmethod
    def create_polynomial_features(
        X: pd.DataFrame,
        degree: int = 2,
        include_bias: bool = False
    ) -> pd.DataFrame:
        """Create polynomial features."""
        X_numeric = X.select_dtypes(include=[np.number])
        
        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        X_poly = poly.fit_transform(X_numeric)
        
        feature_names = poly.get_feature_names_out(X_numeric.columns)
        
        result = X.copy()
        for col in X_numeric.columns:
            result = result.drop(columns=col)
        
        for fname in feature_names:
            if fname != '1':
                result[fname] = X_poly[:, list(feature_names).index(fname)]
        
        return result
    
    @staticmethod
    def create_interaction_features(X: pd.DataFrame, max_degree: int = 2) -> pd.DataFrame:
        """Create interaction features between numeric columns."""
        X_numeric = X.select_dtypes(include=[np.number])
        result = X.copy()
        
        numeric_cols = X_numeric.columns.tolist()
        
        # Create pairwise interactions
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                result[f"{col1}_x_{col2}"] = X_numeric[col1] * X_numeric[col2]
                result[f"{col1}_div_{col2}"] = np.where(
                    X_numeric[col2] != 0,
                    X_numeric[col1] / X_numeric[col2],
                    0
                )
        
        return result
    
    @staticmethod
    def create_ratio_features(X: pd.DataFrame) -> pd.DataFrame:
        """Create ratio features from numeric columns."""
        X_numeric = X.select_dtypes(include=[np.number])
        result = X.copy()
        
        numeric_cols = X_numeric.columns.tolist()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                result[f"{col1}_ratio_{col2}"] = np.where(
                    X_numeric[col2] != 0,
                    X_numeric[col1] / (X_numeric[col2] + 1e-8),
                    0
                )
        
        return result
    
    @staticmethod
    def create_statistical_features(X: pd.DataFrame) -> pd.DataFrame:
        """Create statistical aggregate features."""
        X_numeric = X.select_dtypes(include=[np.number])
        result = X.copy()
        
        # Row-wise statistics
        result['row_mean'] = X_numeric.mean(axis=1)
        result['row_std'] = X_numeric.std(axis=1)
        result['row_min'] = X_numeric.min(axis=1)
        result['row_max'] = X_numeric.max(axis=1)
        result['row_range'] = result['row_max'] - result['row_min']
        result['row_sum'] = X_numeric.sum(axis=1)
        
        return result
    
    @staticmethod
    def create_binned_features(X: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
        """Bin continuous features into categories."""
        X_numeric = X.select_dtypes(include=[np.number])
        result = X.copy()
        
        binizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        X_binned = binizer.fit_transform(X_numeric)
        
        for col, binned_col in zip(X_numeric.columns, X_binned.T):
            result[f"{col}_binned"] = binned_col.astype(int)
        
        return result
    
    @staticmethod
    def create_cyclic_features(X: pd.DataFrame, col: str, period: int) -> pd.DataFrame:
        """Create cyclic features from periodic data (e.g., month, day)."""
        result = X.copy()
        
        if col in result.columns:
            result[f"{col}_sin"] = np.sin(2 * np.pi * result[col] / period)
            result[f"{col}_cos"] = np.cos(2 * np.pi * result[col] / period)
        
        return result


class FeatureSelector:
    """Select most important features."""
    
    @staticmethod
    def select_by_variance(X: pd.DataFrame, threshold: float = 0.01) -> List[str]:
        """Remove low-variance features."""
        from sklearn.feature_selection import VarianceThreshold
        
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X.select_dtypes(include=[np.number]))
        
        mask = vt.get_support()
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        
        return numeric_cols[mask].tolist()
    
    @staticmethod
    def select_by_correlation(X: pd.DataFrame, target: pd.Series = None, threshold: float = 0.9) -> List[str]:
        """Remove highly correlated features."""
        if target is not None:
            # Select features correlated with target
            corr = X.select_dtypes(include=[np.number]).corrwith(target).abs()
            return corr[corr > threshold].index.tolist()
        else:
            # Remove multicollinearity
            X_numeric = X.select_dtypes(include=[np.number])
            corr_matrix = X_numeric.corr().abs()
            
            # Select upper triangle
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            
            # Find columns with correlation > threshold
            to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
            
            return [col for col in X_numeric.columns if col not in to_drop]
    
    @staticmethod
    def select_by_importance(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        n_features: int = 20,
        method: str = "mutual_info"
    ) -> List[str]:
        """Select features by importance score."""
        X_numeric = X.select_dtypes(include=[np.number])
        
        if task_type == "classification":
            if method == "mutual_info":
                selector = SelectKBest(mutual_info_classif, k=min(n_features, X_numeric.shape[1]))
            else:
                selector = SelectKBest(f_classif, k=min(n_features, X_numeric.shape[1]))
        else:
            if method == "mutual_info":
                selector = SelectKBest(mutual_info_regression, k=min(n_features, X_numeric.shape[1]))
            else:
                selector = SelectKBest(f_regression, k=min(n_features, X_numeric.shape[1]))
        
        selector.fit(X_numeric, y)
        mask = selector.get_support()
        
        return X_numeric.columns[mask].tolist()
    
    @staticmethod
    def select_by_rfe(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        n_features: int = 20
    ) -> List[str]:
        """Select features using recursive feature elimination."""
        X_numeric = X.select_dtypes(include=[np.number])
        
        if task_type == "classification":
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        rfe = RFE(estimator, n_features_to_select=min(n_features, X_numeric.shape[1]))
        rfe.fit(X_numeric, y)
        
        mask = rfe.get_support()
        return X_numeric.columns[mask].tolist()
    
    @staticmethod
    def select_by_tree_importance(
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str = "classification",
        threshold: float = 0.01
    ) -> List[str]:
        """Select features by tree-based importance."""
        X_numeric = X.select_dtypes(include=[np.number])
        
        if task_type == "classification":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        model.fit(X_numeric, y)
        
        importances = model.feature_importances_
        mask = importances > threshold
        
        return X_numeric.columns[mask].tolist()


class FeatureEncoder:
    """Encode categorical features properly."""
    
    @staticmethod
    def one_hot_encode(X: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """One-hot encode categorical columns."""
        if columns is None:
            columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        result = X.copy()
        
        for col in columns:
            if col in result.columns:
                dummies = pd.get_dummies(result[col], prefix=col, drop_first=True)
                result = result.drop(columns=col)
                result = pd.concat([result, dummies], axis=1)
        
        return result
    
    @staticmethod
    def label_encode(X: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Label encode categorical columns."""
        from sklearn.preprocessing import LabelEncoder
        
        if columns is None:
            columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        result = X.copy()
        
        for col in columns:
            if col in result.columns:
                le = LabelEncoder()
                result[col] = le.fit_transform(result[col].astype(str))
        
        return result
    
    @staticmethod
    def target_encode(X: pd.DataFrame, y: pd.Series, columns: List[str] = None) -> pd.DataFrame:
        """Target encode categorical columns (mean encoding)."""
        if columns is None:
            columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        result = X.copy()
        
        for col in columns:
            if col in result.columns:
                target_mean = X.groupby(col)[y].mean()
                result[col] = result[col].map(target_mean)
        
        return result
    
    @staticmethod
    def frequency_encode(X: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """Frequency encode categorical columns."""
        if columns is None:
            columns = X.select_dtypes(include=['object', 'category']).columns.tolist()
        
        result = X.copy()
        
        for col in columns:
            if col in result.columns:
                freq = X[col].value_counts(normalize=True)
                result[col] = result[col].map(freq)
        
        return result


class FeatureAnalyzer:
    """Analyze feature distributions and relationships."""
    
    @staticmethod
    def feature_statistics(X: pd.DataFrame) -> pd.DataFrame:
        """Get detailed statistics for all features."""
        stats = X.describe(include='all').T
        
        # Add additional stats
        additional_stats = {}
        for col in X.columns:
            additional_stats[col] = {
                'missing_count': X[col].isna().sum(),
                'missing_percent': X[col].isna().sum() / len(X) * 100,
                'dtype': str(X[col].dtype),
                'unique': X[col].nunique(),
            }
        
        additional = pd.DataFrame(additional_stats).T
        result = pd.concat([stats, additional], axis=1)
        
        return result
    
    @staticmethod
    def correlation_analysis(X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature correlations."""
        X_numeric = X.select_dtypes(include=[np.number])
        
        corr_pearson = X_numeric.corr(method='pearson')
        corr_spearman = X_numeric.corr(method='spearman')
        
        # Find highly correlated pairs
        high_corr = []
        for i in range(len(corr_pearson.columns)):
            for j in range(i+1, len(corr_pearson.columns)):
                if abs(corr_pearson.iloc[i, j]) > 0.8:
                    high_corr.append({
                        'feature1': corr_pearson.columns[i],
                        'feature2': corr_pearson.columns[j],
                        'pearson': corr_pearson.iloc[i, j],
                        'spearman': corr_spearman.iloc[i, j]
                    })
        
        return {
            'pearson_correlation': corr_pearson,
            'spearman_correlation': corr_spearman,
            'high_correlations': high_corr
        }
    
    @staticmethod
    def distribution_analysis(X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature distributions."""
        from scipy.stats import skew, kurtosis
        
        X_numeric = X.select_dtypes(include=[np.number])
        
        distributions = {}
        for col in X_numeric.columns:
            distributions[col] = {
                'mean': X_numeric[col].mean(),
                'median': X_numeric[col].median(),
                'std': X_numeric[col].std(),
                'skewness': skew(X_numeric[col].dropna()),
                'kurtosis': kurtosis(X_numeric[col].dropna()),
                'cv': X_numeric[col].std() / (X_numeric[col].mean() + 1e-8),  # Coefficient of variation
            }
        
        return distributions


class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline."""
    
    def __init__(self):
        self.steps = []
        self.transformed_data_ = None
    
    def add_step(self, name: str, func: Callable) -> FeatureEngineeringPipeline:
        """Add a transformation step."""
        self.steps.append((name, func))
        return self
    
    def execute(self, X: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Execute all transformation steps."""
        result = X.copy()
        
        for name, func in self.steps:
            if y is not None and 'y' in func.__code__.co_varnames:
                result = func(result, y)
            else:
                result = func(result)
        
        self.transformed_data_ = result
        return result
    
    def get_steps(self) -> List[Tuple[str, Callable]]:
        """Get all pipeline steps."""
        return self.steps

# ============================================================================
# ADVANCED MODERN FEATURE ENGINEERING TECHNIQUES
# ============================================================================

class AdvancedFeatureSelection:
    """Modern feature selection techniques for high-dimensional data."""
    
    @staticmethod
    def select_by_shap_importance(
        model: Any,
        X: pd.DataFrame,
        n_features: int = 20
    ) -> List[str]:
        """
        Select features using SHAP (SHapley Additive exPlanations) importance.
        Requires: pip install shap
        """
        try:
            import shap
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For multi-class
            
            feature_importance = np.abs(shap_values).mean(axis=0)
            top_features_idx = np.argsort(feature_importance)[-n_features:]
            
            return X.columns[top_features_idx].tolist()
        except ImportError:
            return X.columns[:n_features].tolist()
    
    @staticmethod
    def select_by_permutation_importance(
        model: Any,
        X: pd.DataFrame,
        y: np.ndarray,
        n_features: int = 20,
        n_repeats: int = 10
    ) -> List[str]:
        """
        Select features based on permutation importance.
        Model-agnostic approach.
        """
        from sklearn.inspection import permutation_importance
        
        result = permutation_importance(
            model,
            X,
            y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        top_features_idx = np.argsort(result.importances_mean)[-n_features:]
        return X.columns[top_features_idx].tolist()
    
    @staticmethod
    def select_by_mutual_information_with_target(
        X: pd.DataFrame,
        y: np.ndarray,
        n_features: int = 20,
        task_type: str = 'classification'
    ) -> List[str]:
        """
        Select features by mutual information with target.
        Captures non-linear relationships.
        """
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        
        if task_type == 'classification':
            scores = mutual_info_classif(X, y, random_state=42)
        else:
            scores = mutual_info_regression(X, y, random_state=42)
        
        top_features_idx = np.argsort(scores)[-n_features:]
        return X.columns[top_features_idx].tolist()
    
    @staticmethod
    def select_low_variance_features(
        X: pd.DataFrame,
        variance_threshold: float = 0.01
    ) -> List[str]:
        """
        Remove low-variance features that carry little information.
        """
        from sklearn.feature_selection import VarianceThreshold
        
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(X)
        
        selected_features = X.columns[selector.get_support()].tolist()
        return selected_features


class TimeSeriesFeatureEngineer:
    """Modern time series feature engineering."""
    
    @staticmethod
    def create_lag_features(
        data: pd.Series,
        lags: List[int] = [1, 2, 7, 30]
    ) -> pd.DataFrame:
        """Create lag features for time series."""
        df = pd.DataFrame({'value': data})
        
        for lag in lags:
            df[f'lag_{lag}'] = df['value'].shift(lag)
        
        return df.dropna()
    
    @staticmethod
    def create_rolling_features(
        data: pd.Series,
        windows: List[int] = [7, 30, 90]
    ) -> pd.DataFrame:
        """Create rolling statistics features."""
        df = pd.DataFrame({'value': data})
        
        for window in windows:
            df[f'rolling_mean_{window}'] = data.rolling(window).mean()
            df[f'rolling_std_{window}'] = data.rolling(window).std()
            df[f'rolling_min_{window}'] = data.rolling(window).min()
            df[f'rolling_max_{window}'] = data.rolling(window).max()
        
        return df.dropna()
    
    @staticmethod
    def create_seasonal_features(data: pd.Series) -> pd.DataFrame:
        """Extract seasonal patterns from time series."""
        df = pd.DataFrame({'value': data})
        df['hour'] = df.index.hour if hasattr(df.index, 'hour') else 0
        df['day_of_week'] = df.index.dayofweek if hasattr(df.index, 'dayofweek') else 0
        df['month'] = df.index.month if hasattr(df.index, 'month') else 0
        df['quarter'] = df.index.quarter if hasattr(df.index, 'quarter') else 0
        
        return df


class TextFeatureEngineer:
    """Modern text feature engineering."""
    
    @staticmethod
    def extract_text_statistics(texts: List[str]) -> pd.DataFrame:
        """Extract statistical features from text."""
        features = {
            'length': [len(text) for text in texts],
            'word_count': [len(text.split()) for text in texts],
            'avg_word_length': [np.mean([len(w) for w in text.split()]) if len(text.split()) > 0 else 0 for text in texts],
            'unique_words': [len(set(text.lower().split())) for text in texts],
            'uppercase_ratio': [sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0 for text in texts],
            'digit_ratio': [sum(1 for c in text if c.isdigit()) / len(text) if len(text) > 0 else 0 for text in texts],
        }
        
        return pd.DataFrame(features)
    
    @staticmethod
    def extract_ngram_features(
        texts: List[str],
        n: int = 2,
        max_features: int = 100
    ) -> pd.DataFrame:
        """Extract n-gram features from text."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            ngram_range=(n, n),
            max_features=max_features
        )
        
        X = vectorizer.fit_transform(texts)
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())


class AutomatedFeatureInteractionDetector:
    """Automatically detect and create important feature interactions."""
    
    @staticmethod
    def detect_interactions(
        X: pd.DataFrame,
        y: np.ndarray,
        n_interactions: int = 10,
        correlation_threshold: float = 0.3
    ) -> List[Tuple[str, str]]:
        """
        Detect feature interactions that improve prediction.
        """
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        interactions = []
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                # Create interaction
                interaction_feature = X[col1] * X[col2]
                
                # Check correlation with target
                if y.dtype in [np.int32, np.int64]:
                    # Classification: use mutual information
                    from sklearn.feature_selection import mutual_info_classif
                    score = mutual_info_classif(
                        interaction_feature.values.reshape(-1, 1),
                        y,
                        random_state=42
                    )[0]
                else:
                    # Regression: use correlation
                    score = abs(np.corrcoef(interaction_feature, y)[0, 1])
                
                if score > correlation_threshold:
                    interactions.append((col1, col2, float(score)))
        
        # Return top n interactions sorted by importance
        interactions.sort(key=lambda x: x[2], reverse=True)
        return [(i[0], i[1]) for i in interactions[:n_interactions]]
    
    @staticmethod
    def create_detected_interactions(
        X: pd.DataFrame,
        interactions: List[Tuple[str, str]]
    ) -> pd.DataFrame:
        """Create interaction features."""
        X_interactions = X.copy()
        
        for col1, col2 in interactions:
            interaction_name = f'{col1}_x_{col2}'
            X_interactions[interaction_name] = X_interactions[col1] * X_interactions[col2]
        
        return X_interactions