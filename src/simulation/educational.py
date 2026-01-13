"""Educational Explanations and AI-Powered Guidance for ML Simulations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class EducationalExplainer:
    """Generate beginner-friendly explanations for ML concepts and results."""
    
    # Educational content library
    CONCEPTS = {
        'overfitting': {
            'definition': "Overfitting occurs when a model learns the training data too well, including noise and outliers, making it perform poorly on new data.",
            'signs': [
                "High training accuracy but low test accuracy",
                "Large gap between training and test performance",
                "Model memorizes rather than generalizes patterns"
            ],
            'solutions': [
                "Reduce model complexity (lower max_depth, fewer estimators)",
                "Add more training data",
                "Use regularization techniques",
                "Apply cross-validation",
                "Feature selection to remove noise"
            ],
            'visual_cues': "Training curve goes to 100% while test curve plateaus or decreases"
        },
        
        'underfitting': {
            'definition': "Underfitting occurs when a model is too simple to capture the underlying patterns in the data.",
            'signs': [
                "Low training and test accuracy",
                "Model predictions are close to random",
                "Cannot learn from increased training data"
            ],
            'solutions': [
                "Increase model complexity (higher max_depth, more estimators)",
                "Add more relevant features",
                "Reduce regularization",
                "Train for more epochs",
                "Try more sophisticated algorithms"
            ],
            'visual_cues': "Both training and test curves remain flat and low"
        },
        
        'bias_variance': {
            'definition': "Bias-variance tradeoff: High bias = underfitting (too simple), high variance = overfitting (too complex)",
            'explanation': "The goal is to find the sweet spot where the model is complex enough to capture patterns but simple enough to generalize.",
            'practical_advice': "Start with moderate complexity and adjust based on the gap between training and test performance."
        },
        
        'class_imbalance': {
            'definition': "Class imbalance occurs when one class has significantly more samples than others.",
            'impact': "Models tend to be biased toward the majority class, achieving high accuracy by simply predicting the majority class.",
            'solutions': [
                "Use stratified sampling",
                "Apply class weights",
                "Oversample minority class (SMOTE)",
                "Undersample majority class",
                "Use appropriate metrics (F1, Precision, Recall) instead of accuracy"
            ]
        },
        
        'data_quality': {
            'definition': "Data quality refers to accuracy, completeness, consistency, and reliability of your dataset.",
            'issues': {
                'missing_values': "Missing data can bias results and reduce model performance",
                'outliers': "Extreme values can distort patterns and mislead models",
                'noise': "Random errors in data make it harder to learn true patterns",
                'duplicates': "Duplicate records can cause data leakage and overfitting"
            },
            'best_practices': [
                "Always explore data before modeling",
                "Handle missing values appropriately",
                "Detect and handle outliers",
                "Validate data consistency",
                "Document data cleaning decisions"
            ]
        }
    }
    
    @classmethod
    def explain_simulation_result(
        cls,
        result: Dict[str, Any],
        problem_type: str = 'classification'
    ) -> Dict[str, str]:
        """
        Generate comprehensive explanation of simulation results.
        
        Args:
            result: Simulation result dictionary
            problem_type: Type of ML problem
        
        Returns:
            Dictionary with explanations
        """
        
        explanations = {}
        model_results = result.get('model_results', {})
        
        # Performance explanation
        if problem_type == 'classification':
            train_acc = model_results.get('train_accuracy', 0)
            test_acc = model_results.get('test_accuracy', 0)
            
            explanations['performance'] = cls._explain_classification_performance(
                train_acc, test_acc
            )
            
            # Overfitting/underfitting analysis
            overfitting_score = model_results.get('overfitting_score', 0)
            explanations['model_fit'] = cls._explain_model_fit(
                train_acc, test_acc, overfitting_score
            )
        
        elif problem_type == 'regression':
            train_r2 = model_results.get('train_r2', 0)
            test_r2 = model_results.get('test_r2', 0)
            
            explanations['performance'] = cls._explain_regression_performance(
                train_r2, test_r2
            )
            
            overfitting_score = model_results.get('overfitting_score', 0)
            explanations['model_fit'] = cls._explain_model_fit(
                train_r2, test_r2, overfitting_score, is_regression=True
            )
        
        # Data quality explanation
        if 'data_quality' in result:
            explanations['data_quality'] = cls._explain_data_quality(
                result['data_quality']
            )
        
        # Algorithm explanation
        algorithm = model_results.get('algorithm', 'Unknown')
        explanations['algorithm'] = cls.explain_algorithm(algorithm)
        
        return explanations
    
    @classmethod
    def _explain_classification_performance(
        cls,
        train_acc: float,
        test_acc: float
    ) -> str:
        """Explain classification performance."""
        
        if test_acc > 0.9:
            performance = "Excellent"
            detail = "Your model is performing very well on unseen data."
        elif test_acc > 0.8:
            performance = "Good"
            detail = "Your model has good predictive performance."
        elif test_acc > 0.7:
            performance = "Fair"
            detail = "Your model has moderate performance. Consider improving features or trying different algorithms."
        elif test_acc > 0.6:
            performance = "Poor"
            detail = "Your model needs improvement. Try increasing model complexity or getting better data."
        else:
            performance = "Very Poor"
            detail = "Your model is barely better than random guessing. Revisit your data and feature engineering."
        
        return f"**{performance} Performance** ({test_acc:.1%} accuracy)\n\n{detail}\n\nTraining accuracy: {train_acc:.1%}"
    
    @classmethod
    def _explain_regression_performance(
        cls,
        train_r2: float,
        test_r2: float
    ) -> str:
        """Explain regression performance."""
        
        if test_r2 > 0.8:
            performance = "Excellent"
            detail = "Your model explains most of the variance in the target variable."
        elif test_r2 > 0.6:
            performance = "Good"
            detail = "Your model captures a good amount of the underlying pattern."
        elif test_r2 > 0.4:
            performance = "Fair"
            detail = "Your model captures some patterns but there's room for improvement."
        elif test_r2 > 0.2:
            performance = "Poor"
            detail = "Your model struggles to predict the target variable accurately."
        else:
            performance = "Very Poor"
            detail = "Your model may not be suitable for this data. Consider different approaches."
        
        return f"**{performance} Performance** (R² = {test_r2:.3f})\n\n{detail}\n\nTraining R²: {train_r2:.3f}"
    
    @classmethod
    def _explain_model_fit(
        cls,
        train_score: float,
        test_score: float,
        overfitting_score: float,
        is_regression: bool = False
    ) -> str:
        """Explain model fit (overfitting/underfitting)."""
        
        threshold = 0.7 if not is_regression else 0.5
        
        if train_score < threshold and test_score < threshold:
            # Underfitting
            return cls._format_concept_explanation('underfitting')
        
        elif overfitting_score > 0.15:
            # Overfitting
            return cls._format_concept_explanation('overfitting')
        
        else:
            # Good fit
            return """**✅ Good Model Fit**

Your model has found the right balance between complexity and generalization:
- Training and test scores are both good
- Small gap between training and test performance
- Model generalizes well to new data

This is the ideal scenario! Your model complexity is appropriate for your data."""
    
    @classmethod
    def _explain_data_quality(cls, quality_metrics: Dict[str, Any]) -> str:
        """Explain data quality metrics."""
        
        issues = []
        recommendations = []
        
        missing_rate = quality_metrics.get('missing_rate', 0)
        if missing_rate > 0.2:
            issues.append(f"High missing value rate ({missing_rate:.1%})")
            recommendations.append("Investigate why data is missing and consider imputation or collection of more complete data")
        elif missing_rate > 0.05:
            issues.append(f"Moderate missing values ({missing_rate:.1%})")
            recommendations.append("Apply appropriate imputation methods")
        
        outlier_rate = quality_metrics.get('outlier_rate', 0)
        if outlier_rate > 0.1:
            issues.append(f"High outlier rate ({outlier_rate:.1%})")
            recommendations.append("Investigate outliers - are they errors or valid extreme values?")
        
        class_balance = quality_metrics.get('class_balance', {})
        if class_balance:
            min_ratio = min(class_balance.values())
            if min_ratio < 0.2:
                issues.append(f"Severe class imbalance (minority class: {min_ratio:.1%})")
                recommendations.append("Use techniques to address class imbalance (SMOTE, class weights)")
        
        if not issues:
            return "**✅ Good Data Quality**\n\nYour data appears clean with minimal issues."
        
        explanation = "**⚠️ Data Quality Issues Detected**\n\n**Issues:**\n"
        explanation += "\n".join(f"- {issue}" for issue in issues)
        explanation += "\n\n**Recommendations:**\n"
        explanation += "\n".join(f"- {rec}" for rec in recommendations)
        
        return explanation
    
    @classmethod
    def explain_algorithm(cls, algorithm: str) -> str:
        """Explain how an algorithm works."""
        
        algorithms = {
            'Logistic Regression': {
                'how_it_works': "Uses a mathematical function (sigmoid) to model probability of class membership based on input features.",
                'strengths': ["Fast training", "Interpretable coefficients", "Works well with linear relationships"],
                'weaknesses': ["Cannot capture non-linear patterns", "Assumes feature independence"],
                'best_for': "Binary classification with linear decision boundaries"
            },
            
            'Decision Tree': {
                'how_it_works': "Creates a tree of if-then rules by repeatedly splitting data based on feature values that best separate classes.",
                'strengths': ["Easy to understand", "Handles non-linear relationships", "No data scaling needed"],
                'weaknesses': ["Prone to overfitting", "Unstable (small data changes affect tree)"],
                'best_for': "Problems requiring interpretability and rule-based decisions"
            },
            
            'Random Forest': {
                'how_it_works': "Builds many decision trees on random subsets of data and features, then averages their predictions.",
                'strengths': ["Reduces overfitting", "Handles non-linearity", "Feature importance", "Robust"],
                'weaknesses': ["Less interpretable", "Slower than single trees", "Memory intensive"],
                'best_for': "General-purpose classification and regression tasks"
            },
            
            'Gradient Boosting': {
                'how_it_works': "Builds trees sequentially, where each tree corrects errors of the previous trees by focusing on misclassified samples.",
                'strengths': ["Very accurate", "Handles complex patterns", "Feature importance"],
                'weaknesses': ["Prone to overfitting if not tuned", "Slower training", "Sensitive to hyperparameters"],
                'best_for': "Competitions and tasks requiring maximum accuracy"
            },
            
            'SVM': {
                'how_it_works': "Finds the optimal boundary (hyperplane) that maximally separates classes in high-dimensional space.",
                'strengths': ["Effective in high dimensions", "Memory efficient", "Versatile kernels"],
                'weaknesses': ["Slow on large datasets", "Requires feature scaling", "Hard to interpret"],
                'best_for': "High-dimensional data with clear margins between classes"
            },
            
            'Linear Regression': {
                'how_it_works': "Fits a straight line (or hyperplane) that minimizes the distance to all data points.",
                'strengths': ["Simple and interpretable", "Fast training", "Coefficients show feature impact"],
                'weaknesses': ["Only models linear relationships", "Sensitive to outliers"],
                'best_for': "Understanding linear relationships between variables"
            },
        }
        
        if algorithm not in algorithms:
            return f"**{algorithm}**\n\nA machine learning algorithm used for prediction tasks."
        
        info = algorithms[algorithm]
        
        explanation = f"""**{algorithm}**

**How it works:**
{info['how_it_works']}

**Strengths:**
{chr(10).join(f'- {s}' for s in info['strengths'])}

**Weaknesses:**
{chr(10).join(f'- {w}' for w in info['weaknesses'])}

**Best for:** {info['best_for']}
"""
        
        return explanation
    
    @classmethod
    def _format_concept_explanation(cls, concept: str) -> str:
        """Format a concept explanation."""
        
        if concept not in cls.CONCEPTS:
            return f"Explanation for '{concept}' not available."
        
        info = cls.CONCEPTS[concept]
        
        explanation = f"**{concept.replace('_', ' ').title()}**\n\n"
        
        if 'definition' in info:
            explanation += f"{info['definition']}\n\n"
        
        if 'signs' in info:
            explanation += "**Warning Signs:**\n"
            explanation += "\n".join(f"- {sign}" for sign in info['signs'])
            explanation += "\n\n"
        
        if 'solutions' in info:
            explanation += "**How to Fix:**\n"
            explanation += "\n".join(f"- {sol}" for sol in info['solutions'])
            explanation += "\n\n"
        
        if 'visual_cues' in info:
            explanation += f"**Visual Indicator:** {info['visual_cues']}\n"
        
        return explanation
    
    @classmethod
    def generate_step_by_step_guide(
        cls,
        simulation_type: str
    ) -> List[Dict[str, str]]:
        """Generate step-by-step guide for running a simulation."""
        
        guides = {
            'classification': [
                {
                    'step': 1,
                    'title': 'Configure Data Generation',
                    'description': 'Set sample size, number of features, and data quality parameters',
                    'tips': ['Start with 1000 samples', 'Use 10-20 features', 'Begin with no noise']
                },
                {
                    'step': 2,
                    'title': 'Choose Algorithm',
                    'description': 'Select a classification algorithm based on your goals',
                    'tips': ['Random Forest is a good all-around choice', 'Logistic Regression for interpretability']
                },
                {
                    'step': 3,
                    'title': 'Set Model Complexity',
                    'description': 'Configure hyperparameters that control model complexity',
                    'tips': ['Start with moderate values', 'max_depth=10, n_estimators=100']
                },
                {
                    'step': 4,
                    'title': 'Run Simulation',
                    'description': 'Execute the simulation and observe training',
                    'tips': ['Watch training time', 'Compare train vs test scores']
                },
                {
                    'step': 5,
                    'title': 'Analyze Results',
                    'description': 'Review performance metrics and visualizations',
                    'tips': ['Check confusion matrix', 'Look for overfitting signs', 'Examine feature importance']
                },
                {
                    'step': 6,
                    'title': 'Experiment',
                    'description': 'Adjust parameters and see how results change',
                    'tips': ['Try different complexity levels', 'Add noise to see impact', 'Compare multiple algorithms']
                }
            ],
            
            'overfitting': [
                {
                    'step': 1,
                    'title': 'Set Baseline',
                    'description': 'Start with a moderate complexity model',
                    'tips': ['Use default parameters first']
                },
                {
                    'step': 2,
                    'title': 'Vary Complexity',
                    'description': 'Test a range of complexity values',
                    'tips': ['Try max_depth from 1 to 20', 'Observe when overfitting starts']
                },
                {
                    'step': 3,
                    'title': 'Identify Sweet Spot',
                    'description': 'Find the complexity that balances bias and variance',
                    'tips': ['Look for minimal gap between train and test', 'Good test performance']
                },
                {
                    'step': 4,
                    'title': 'Understand Tradeoffs',
                    'description': 'Learn how complexity affects generalization',
                    'tips': ['Too simple = underfitting', 'Too complex = overfitting']
                }
            ],
            
            'what_if': [
                {
                    'step': 1,
                    'title': 'Choose Scenario',
                    'description': 'Select which aspect to investigate',
                    'tips': ['Data quality degradation', 'Sample size impact', 'Feature reduction']
                },
                {
                    'step': 2,
                    'title': 'Run Baseline',
                    'description': 'Establish baseline performance',
                    'tips': ['Use clean data first']
                },
                {
                    'step': 3,
                    'title': 'Apply Variations',
                    'description': 'Systematically change one aspect',
                    'tips': ['Change one thing at a time', 'Use multiple levels']
                },
                {
                    'step': 4,
                    'title': 'Compare Results',
                    'description': 'Analyze how performance changes',
                    'tips': ['Look for trends', 'Identify critical thresholds']
                }
            ]
        }
        
        return guides.get(simulation_type, [])
    
    @classmethod
    def explain_metric(cls, metric_name: str, value: float) -> str:
        """Explain what a metric means."""
        
        metrics = {
            'accuracy': {
                'description': "Percentage of correct predictions",
                'range': "0-100% (higher is better)",
                'interpretation': cls._interpret_percentage(value, 'accuracy')
            },
            'precision': {
                'description': "Of all positive predictions, how many were actually positive?",
                'range': "0-100% (higher is better)",
                'interpretation': cls._interpret_percentage(value, 'precision'),
                'use_case': "Important when false positives are costly"
            },
            'recall': {
                'description': "Of all actual positives, how many did we identify?",
                'range': "0-100% (higher is better)",
                'interpretation': cls._interpret_percentage(value, 'recall'),
                'use_case': "Important when false negatives are costly"
            },
            'f1_score': {
                'description': "Balanced measure of precision and recall",
                'range': "0-100% (higher is better)",
                'interpretation': cls._interpret_percentage(value, 'f1'),
                'use_case': "Good general metric when classes are imbalanced"
            },
            'r2_score': {
                'description': "How much variance in target is explained by features",
                'range': "-∞ to 1 (1 is perfect, 0 is baseline, negative is worse than baseline)",
                'interpretation': cls._interpret_r2(value)
            },
            'mse': {
                'description': "Average squared difference between predictions and actual values",
                'range': "0 to ∞ (lower is better)",
                'interpretation': f"MSE of {value:.2f} - lower is better"
            },
            'mae': {
                'description': "Average absolute difference between predictions and actual",
                'range': "0 to ∞ (lower is better)",
                'interpretation': f"On average, predictions are off by {value:.2f} units"
            }
        }
        
        if metric_name.lower() not in metrics:
            return f"{metric_name}: {value:.3f}"
        
        info = metrics[metric_name.lower()]
        
        explanation = f"**{metric_name.upper()}**: {value:.3f}\n\n"
        explanation += f"{info['description']}\n\n"
        explanation += f"**Range:** {info['range']}\n\n"
        explanation += f"**Your Result:** {info['interpretation']}"
        
        if 'use_case' in info:
            explanation += f"\n\n**When to use:** {info['use_case']}"
        
        return explanation
    
    @staticmethod
    def _interpret_percentage(value: float, metric_type: str) -> str:
        """Interpret percentage-based metrics."""
        pct = value * 100
        
        if pct > 90:
            return f"Excellent - {pct:.1f}% is very high"
        elif pct > 80:
            return f"Good - {pct:.1f}% is above average"
        elif pct > 70:
            return f"Fair - {pct:.1f}% is acceptable"
        elif pct > 60:
            return f"Poor - {pct:.1f}% needs improvement"
        else:
            return f"Very Poor - {pct:.1f}% is low"
    
    @staticmethod
    def _interpret_r2(value: float) -> str:
        """Interpret R² score."""
        if value > 0.9:
            return f"Excellent fit - {value:.3f} means model explains {value*100:.1f}% of variance"
        elif value > 0.7:
            return f"Good fit - {value:.3f} means model explains {value*100:.1f}% of variance"
        elif value > 0.5:
            return f"Moderate fit - {value:.3f} means model explains {value*100:.1f}% of variance"
        elif value > 0.3:
            return f"Poor fit - {value:.3f} means model only explains {value*100:.1f}% of variance"
        elif value > 0:
            return f"Very poor fit - {value:.3f} means model barely explains {value*100:.1f}% of variance"
        else:
            return f"Worse than baseline - {value:.3f} means model performs worse than simply predicting the mean"
