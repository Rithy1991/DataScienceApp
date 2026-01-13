"""
Machine Learning Best Practices & Educational Content
======================================================
Comprehensive guidance for building production-ready ML models.
"""

from typing import Dict, List, Any
import streamlit as st


ML_BEST_PRACTICES = {
    "data_quality": {
        "title": "🎯 Data Quality Checklist",
        "icon": "✓",
        "practices": [
            {
                "name": "Check for Class Imbalance",
                "description": "For classification: ensure classes are reasonably balanced. Use SMOTE, class weights, or stratified sampling if needed.",
                "code_example": """
# Check class distribution
print(y.value_counts(normalize=True))

# Apply class weights in scikit-learn
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
"""
            },
            {
                "name": "Handle Missing Values Appropriately",
                "description": "Don't just drop rows! Consider imputation strategies based on data type and missingness pattern.",
                "code_example": """
# For numerical: mean/median imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_numeric)

# For categorical: mode or constant
cat_imputer = SimpleImputer(strategy='most_frequent')
"""
            },
            {
                "name": "Detect and Handle Outliers",
                "description": "Identify outliers using IQR or Z-score. Decide whether to remove, cap, or transform.",
                "code_example": """
# IQR method
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
outliers = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
"""
            },
            {
                "name": "Feature Scaling",
                "description": "Scale features for algorithms sensitive to magnitude (SVM, KNN, Neural Networks). Use StandardScaler or MinMaxScaler.",
                "code_example": """
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
# Remember to use same scaler on test set!
X_test_scaled = scaler.transform(X_test)
"""
            }
        ]
    },
    "model_selection": {
        "title": "🤖 Model Selection Guidelines",
        "icon": "🎯",
        "practices": [
            {
                "name": "Start Simple, Then Iterate",
                "description": "Begin with simple models (Logistic Regression, Decision Tree) as baselines before trying complex ones.",
                "tip": "A simple model that's well-understood beats a complex black box!"
            },
            {
                "name": "Match Algorithm to Problem Type",
                "description": "Classification vs Regression | Linear vs Non-linear | Interpretable vs Accurate",
                "guide": {
                    "Linear problems": ["Linear Regression", "Logistic Regression", "SVM"],
                    "Non-linear patterns": ["Random Forest", "Gradient Boosting", "Neural Networks"],
                    "Interpretability needed": ["Decision Trees", "Linear Models", "Rule-based"],
                    "High accuracy priority": ["XGBoost", "LightGBM", "Deep Learning"]
                }
            },
            {
                "name": "Consider Data Size",
                "description": "Small datasets (<1000 samples): Simple models | Large datasets: Complex models with regularization",
                "tip": "Deep Learning needs 1000s-millions of samples to shine!"
            }
        ]
    },
    "training": {
        "title": "🏋️ Training Best Practices",
        "icon": "💪",
        "practices": [
            {
                "name": "Always Use Train-Test Split",
                "description": "Never evaluate on training data! Use 70-80% train, 20-30% test. For small datasets, use cross-validation.",
                "code_example": """
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
"""
            },
            {
                "name": "Use Cross-Validation",
                "description": "K-fold CV gives more reliable estimates of model performance, especially for small datasets.",
                "code_example": """
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_weighted')
print(f"CV F1: {scores.mean():.3f} (+/- {scores.std():.3f})")
"""
            },
            {
                "name": "Hyperparameter Tuning",
                "description": "Use GridSearchCV or RandomizedSearchCV to find optimal hyperparameters systematically.",
                "code_example": """
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
"""
            },
            {
                "name": "Monitor for Overfitting",
                "description": "Compare train vs test performance. Large gap = overfitting. Use regularization, simpler models, or more data.",
                "signs": [
                    "Training accuracy >> Test accuracy",
                    "Training loss << Validation loss",
                    "Model performs poorly on new data"
                ]
            }
        ]
    },
    "evaluation": {
        "title": "📊 Evaluation Metrics Guide",
        "icon": "📈",
        "practices": [
            {
                "name": "Classification Metrics",
                "description": "Choose metrics based on your problem and class balance.",
                "metrics": {
                    "Accuracy": "Overall correctness. Don't use for imbalanced data!",
                    "Precision": "Of all positive predictions, how many were correct? (Minimize false positives)",
                    "Recall": "Of all actual positives, how many did we find? (Minimize false negatives)",
                    "F1-Score": "Harmonic mean of precision and recall. Good for imbalanced data.",
                    "ROC-AUC": "Trade-off between true positive rate and false positive rate. Good for binary classification.",
                    "Confusion Matrix": "See exactly where your model makes mistakes."
                }
            },
            {
                "name": "Regression Metrics",
                "description": "Measure prediction error in different ways.",
                "metrics": {
                    "MAE": "Mean Absolute Error - average magnitude of errors (robust to outliers)",
                    "MSE": "Mean Squared Error - penalizes large errors more heavily",
                    "RMSE": "Root MSE - in same units as target variable",
                    "R² Score": "Proportion of variance explained (0-1, higher is better)",
                    "MAPE": "Mean Absolute Percentage Error - relative error as %"
                }
            },
            {
                "name": "Business Metrics Matter Most",
                "description": "Always connect ML metrics to business outcomes!",
                "examples": [
                    "Medical diagnosis: Maximize recall (catch all diseases) even if precision is lower",
                    "Spam detection: Balance precision and recall (F1) to avoid filtering legitimate emails",
                    "Credit default: Minimize false negatives (missed defaults are costly)"
                ]
            }
        ]
    },
    "deployment": {
        "title": "🚀 Deployment & Production",
        "icon": "⚡",
        "practices": [
            {
                "name": "Model Versioning",
                "description": "Track model versions, training data, hyperparameters, and performance metrics.",
                "tip": "Use tools like MLflow, Weights & Biases, or simple naming conventions with dates."
            },
            {
                "name": "Monitor Model Performance",
                "description": "Set up monitoring to detect model drift and data distribution changes over time.",
                "warning": "Models degrade over time as real-world data evolves!"
            },
            {
                "name": "A/B Testing",
                "description": "Test new models against current production models before full deployment.",
                "tip": "Start with 10% traffic, monitor metrics, gradually increase if performance improves."
            },
            {
                "name": "Explainability & Interpretability",
                "description": "Provide explanations for model predictions, especially in regulated industries.",
                "tools": ["SHAP values", "LIME", "Feature importance", "Partial dependence plots"]
            }
        ]
    },
    "common_mistakes": {
        "title": "⚠️ Common Mistakes to Avoid",
        "icon": "🚫",
        "mistakes": [
            {
                "mistake": "Data Leakage",
                "description": "Using information in training that wouldn't be available at prediction time.",
                "example": "Including the target variable or future information in features.",
                "solution": "Carefully review your features and preprocessing pipeline."
            },
            {
                "mistake": "Not Scaling Test Data Properly",
                "description": "Fitting scaler on test data instead of using training scaler.",
                "solution": "Always fit preprocessing on training data only, then transform test data."
            },
            {
                "mistake": "Ignoring Class Imbalance",
                "description": "Training on imbalanced data without addressing it leads to biased models.",
                "solution": "Use SMOTE, class weights, stratified sampling, or choose appropriate metrics."
            },
            {
                "mistake": "Overfitting to Validation Set",
                "description": "Tuning hyperparameters too much based on validation performance.",
                "solution": "Use a separate test set that you only evaluate on at the very end."
            },
            {
                "mistake": "Not Validating Assumptions",
                "description": "Assuming your data meets algorithm requirements without checking.",
                "solution": "Check for normality, linearity, independence, homoscedasticity as needed."
            }
        ]
    }
}


def render_best_practices_section(section_key: str) -> None:
    """Render a specific best practices section."""
    if section_key not in ML_BEST_PRACTICES:
        return
    
    section = ML_BEST_PRACTICES[section_key]
    
    with st.expander(f"📚 {section['title']}", expanded=False):
        if "practices" in section:
            for practice in section["practices"]:
                st.markdown(f"### {section.get('icon', '•')} {practice['name']}")
                st.write(practice["description"])
                
                if "code_example" in practice:
                    st.code(practice["code_example"], language="python")
                
                if "tip" in practice:
                    st.info(f"💡 **Pro Tip:** {practice['tip']}")
                
                if "guide" in practice:
                    for category, algorithms in practice["guide"].items():
                        st.markdown(f"**{category}:** {', '.join(algorithms)}")
                
                if "metrics" in practice:
                    for metric, description in practice["metrics"].items():
                        st.markdown(f"- **{metric}:** {description}")
                
                if "examples" in practice:
                    st.markdown("**Examples:**")
                    for example in practice["examples"]:
                        st.markdown(f"- {example}")
                
                if "signs" in practice:
                    st.markdown("**Warning Signs:**")
                    for sign in practice["signs"]:
                        st.markdown(f"- ⚠️ {sign}")
                
                if "tools" in practice:
                    st.markdown(f"**Tools:** {', '.join(practice['tools'])}")
                
                st.divider()
        
        elif "mistakes" in section:
            for mistake in section["mistakes"]:
                st.markdown(f"### ❌ {mistake['mistake']}")
                st.write(mistake["description"])
                
                if "example" in mistake:
                    st.warning(f"**Example:** {mistake['example']}")
                
                st.success(f"✅ **Solution:** {mistake['solution']}")
                st.divider()


def render_all_best_practices() -> None:
    """Render all best practices sections."""
    st.markdown("# 🎓 Machine Learning Best Practices")
    st.markdown("Learn professional techniques for building production-ready models.")
    
    st.divider()
    
    # Render each section
    for section_key in ML_BEST_PRACTICES.keys():
        render_best_practices_section(section_key)


def get_contextual_tips(page_type: str) -> List[str]:
    """Get contextual tips based on the current page."""
    tips_by_page = {
        "data_loading": [
            "Always inspect your data first! Use df.info(), df.describe(), and df.head()",
            "Check for missing values early: df.isnull().sum()",
            "Verify data types match your expectations"
        ],
        "eda": [
            "Visualize distributions before making decisions about transformations",
            "Look for correlations between features and target variable",
            "Check for outliers using box plots and statistical methods"
        ],
        "cleaning": [
            "Document every cleaning decision you make - reproducibility matters!",
            "Never drop rows without understanding why data is missing",
            "Consider if outliers are errors or legitimate extreme values"
        ],
        "feature_engineering": [
            "Create features that capture domain knowledge",
            "Use feature selection to reduce overfitting",
            "Try polynomial features for capturing non-linear relationships"
        ],
        "model_training": [
            "Start with simple models before complex ones",
            "Always use cross-validation for robust evaluation",
            "Monitor training time - complex models may not be worth the compute cost"
        ],
        "evaluation": [
            "Don't rely on a single metric - use multiple perspectives",
            "Analyze confusion matrix to understand error types",
            "Test on truly unseen data before claiming success"
        ]
    }
    
    return tips_by_page.get(page_type, ["Keep learning and experimenting!"])


def render_quick_tip(page_type: str) -> None:
    """Render a quick tip box for the current page."""
    tips = get_contextual_tips(page_type)
    
    import random
    tip = random.choice(tips)
    
    st.info(f"💡 **Quick Tip:** {tip}")
