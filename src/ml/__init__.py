"""
Machine Learning Module
=======================
Complete supervised and unsupervised learning toolkit.
"""

from .supervised import (
    DataPreprocessor,
    SupervisedLearningModel,
    HyperparameterOptimizer,
    create_ensemble,
)

from .unsupervised import (
    ClusteringModel,
    DimensionalityReducer,
    AnomalyDetector,
    RuleExtractor,
    find_optimal_clusters,
    profile_clusters,
)

from .feature_engineering import (
    FeatureScaler,
    FeatureCreator,
    FeatureSelector,
    FeatureEncoder,
    FeatureAnalyzer,
    FeatureEngineeringPipeline,
)

from .evaluation import (
    ClassificationEvaluator,
    RegressionEvaluator,
    ModelComparator,
    CrossValidationAnalyzer,
    LearningCurveAnalyzer,
    FeatureImportanceAnalyzer,
    ModelInterpretability,
)

__all__ = [
    # Supervised
    'DataPreprocessor',
    'SupervisedLearningModel',
    'HyperparameterOptimizer',
    'create_ensemble',
    # Unsupervised
    'ClusteringModel',
    'DimensionalityReducer',
    'AnomalyDetector',
    'RuleExtractor',
    'find_optimal_clusters',
    'profile_clusters',
    # Feature Engineering
    'FeatureScaler',
    'FeatureCreator',
    'FeatureSelector',
    'FeatureEncoder',
    'FeatureAnalyzer',
    'FeatureEngineeringPipeline',
    # Evaluation
    'ClassificationEvaluator',
    'RegressionEvaluator',
    'ModelComparator',
    'CrossValidationAnalyzer',
    'LearningCurveAnalyzer',
    'FeatureImportanceAnalyzer',
    'ModelInterpretability',
]
