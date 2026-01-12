"""
Data Science Personal Assistant
Interactive guide and helper for your data science projects
"""

from __future__ import annotations

import json
import streamlit as st

st.set_page_config(page_title="DataScope Pro - DS Assistant", layout="wide", initial_sidebar_state="expanded")

import pandas as pd
from pathlib import Path

from src.core.config import load_config
from src.core.logging_utils import setup_logging, log_event
from src.core.state import get_clean_df, get_df
from src.core.ui import app_header, sidebar_dataset_status, instruction_block, page_navigation
from src.core.standardized_ui import (
    standard_section_header,
    beginner_tip,
    concept_explainer,
    common_mistakes_panel,
)
from src.core.styles import inject_custom_css
from src.core.ai_helper import ai_help_button, ai_interpretation_box, ai_sidebar_assistant
from src.storage.history import add_event


def _get_dataset_recommendations(df: pd.DataFrame) -> dict:
    """Analyze dataset and provide recommendations."""
    if df is None or df.empty:
        return {}
    
    recommendations = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_pct": (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
        "numeric_cols": df.select_dtypes(include=['number']).shape[1],
        "categorical_cols": df.select_dtypes(include=['object']).shape[1],
        "suggested_tasks": []
    }
    
    # Suggest tasks based on dataset characteristics
    if recommendations["rows"] > 10000:
        recommendations["suggested_tasks"].append("✅ Large dataset - Consider LightGBM or XGBoost")
    
    if recommendations["missing_pct"] > 20:
        recommendations["suggested_tasks"].append("⚠️ High missing data (>20%) - Clean data first")
    elif recommendations["missing_pct"] > 5:
        recommendations["suggested_tasks"].append("📊 Some missing data detected - May need imputation")
    
    if recommendations["categorical_cols"] > recommendations["numeric_cols"]:
        recommendations["suggested_tasks"].append("🏷️ More categorical than numeric - Encoding recommended")
    
    if recommendations["numeric_cols"] > 20:
        recommendations["suggested_tasks"].append("🔍 Many features - Consider feature selection or dimensionality reduction")
    
    return recommendations


def _workflow_guide() -> None:
    """Display interactive workflow guide."""
    st.subheader("📋 Data Science Workflow Guide")
    
    workflow_steps = {
        "1️⃣ Data Exploration": {
            "description": "Understand your data",
            "tasks": [
                "📊 Load and inspect your dataset",
                "🔍 Check data types and missing values",
                "📈 View basic statistics and distributions",
                "🎯 Identify target variable for prediction"
            ],
            "page": "2_Data_Analysis_EDA"
        },
        "2️⃣ Data Cleaning": {
            "description": "Prepare data for modeling",
            "tasks": [
                "🧹 Remove or impute missing values",
                "🔄 Handle outliers and anomalies",
                "📅 Parse dates and timestamps",
                "⚖️ Handle class imbalance if needed"
            ],
            "page": "13_Data_Cleaning"
        },
        "3️⃣ Feature Engineering": {
            "description": "Create meaningful features",
            "tasks": [
                "🔨 Create derived features",
                "🏷️ Encode categorical variables",
                "📊 Scale/normalize numeric features",
                "🎯 Select most important features"
            ],
            "page": "4_Feature_Engineering"
        },
        "4️⃣ Model Training": {
            "description": "Build ML models",
            "tasks": [
                "🤖 Train classification or regression models",
                "⚖️ Compare different algorithms",
                "🔧 Tune hyperparameters",
                "📈 Evaluate performance metrics"
            ],
            "page": "5_Tabular_Machine_Learning"
        },
        "5️⃣ Predictions": {
            "description": "Generate predictions",
            "tasks": [
                "🎯 Make predictions on new data",
                "📊 Visualize prediction results",
                "🔍 Analyze prediction confidence",
                "💾 Export predictions"
            ],
            "page": "9_Prediction"
        },
        "6️⃣ Visualization": {
            "description": "Tell the story",
            "tasks": [
                "📊 Create insightful charts",
                "🎨 Build interactive dashboards",
                "📉 Show trends and patterns",
                "🎯 Communicate findings"
            ],
            "page": "7_Visualization"
        }
    }
    
    selected_step = st.selectbox("Select a workflow step:", list(workflow_steps.keys()))
    
    if selected_step:
        step_info = workflow_steps[selected_step]
        st.markdown(f"### {step_info['description']}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Tasks in this step:**")
            for task in step_info['tasks']:
                st.markdown(f"- {task}")
        
        with col2:
            if st.button(f"Go to {step_info['page']} →", use_container_width=True):
                st.switch_page(f"pages/{step_info['page']}.py" if step_info['page'] != "app" else "app.py")


def _problem_selector() -> None:
    """Help user identify their problem type."""
    st.subheader("🎯 What's Your Problem?")
    
    problem_types = {
        "Classification": {
            "description": "Predict categories (e.g., spam/not spam, churn yes/no)",
            "examples": ["Binary Classification", "Multi-class Classification"],
            "models": ["Random Forest", "XGBoost", "LightGBM"],
            "metrics": ["Accuracy", "Precision", "Recall", "F1-Score"]
        },
        "Regression": {
            "description": "Predict continuous values (e.g., price, temperature, sales)",
            "examples": ["Linear Regression", "Time Series Forecasting"],
            "models": ["Random Forest", "Gradient Boosting", "XGBoost"],
            "metrics": ["R² Score", "RMSE", "MAE"]
        },
        "Clustering": {
            "description": "Group similar items together",
            "examples": ["Customer Segmentation", "Anomaly Detection"],
            "models": ["K-Means", "DBSCAN", "Hierarchical Clustering"],
            "metrics": ["Silhouette Score", "Davies-Bouldin Index"]
        },
        "Time Series": {
            "description": "Predict future values based on past data",
            "examples": ["Stock Prediction", "Weather Forecasting"],
            "models": ["ARIMA", "Prophet", "Transformer", "LSTM"],
            "metrics": ["MAE", "RMSE", "MAPE"]
        }
    }
    
    col1, col2 = st.columns(2)
    with col1:
        selected_problem = st.radio("Problem Type:", list(problem_types.keys()))
    
    with col2:
        if selected_problem:
            problem_info = problem_types[selected_problem]
            st.markdown(f"**{problem_info['description']}**")
            st.markdown("**Examples:**")
            for ex in problem_info['examples']:
                st.markdown(f"- {ex}")
            
            st.markdown("**Recommended Models:**")
            for model in problem_info['models']:
                st.markdown(f"- {model}")
            
            st.markdown("**Key Metrics:**")
            for metric in problem_info['metrics']:
                st.markdown(f"- {metric}")


def _quick_tips() -> None:
    """Display quick tips and best practices."""
    st.subheader("💡 Quick Tips & Best Practices")
    
    tips = {
        "🧹 Data Cleaning": [
            "Always check for missing values before modeling",
            "Outliers can significantly impact model performance",
            "Ensure data types are correct (dates as datetime, not strings)",
            "Remove or impute rows/columns with >50% missing data"
        ],
        "🎯 Feature Selection": [
            "More features ≠ better model (curse of dimensionality)",
            "Use correlation analysis to find related features",
            "Remove features with near-zero variance",
            "Consider domain knowledge when selecting features"
        ],
        "🤖 Model Training": [
            "Always split data: 70% train, 15% validation, 15% test",
            "Use cross-validation for more reliable estimates",
            "Start simple (Linear models) before complex ones (Ensembles)",
            "Monitor for overfitting: training vs validation performance"
        ],
        "📊 Evaluation": [
            "Use appropriate metrics for your problem type",
            "Consider business impact, not just statistical metrics",
            "Class imbalance? Use F1-Score, not Accuracy",
            "Confusion matrix shows where your model struggles"
        ],
        "🚀 Deployment": [
            "Document your preprocessing steps carefully",
            "Save feature names and scaling parameters",
            "Test on new data different from training set",
            "Monitor model performance in production"
        ]
    }
    
    tab_names = list(tips.keys())
    tabs = st.tabs(tab_names)
    
    for tab, category in zip(tabs, tab_names):
        with tab:
            for tip in tips[category]:
                st.info(tip)


def _model_comparison_helper() -> None:
    """Help user understand model differences."""
    st.subheader("🤖 Model Comparison Helper")
    
    models_db = {
        "Random Forest": {
            "pros": ["✅ Handles both regression & classification", "✅ Handles missing data well", "✅ Feature importance"],
            "cons": ["⚠️ Can overfit on small datasets", "⚠️ Slower predictions on large data"],
            "best_for": "Medium datasets, mixed feature types",
            "hyperparams": ["n_estimators", "max_depth", "min_samples_split"]
        },
        "XGBoost": {
            "pros": ["✅ State-of-the-art performance", "✅ Handles feature interactions", "✅ Built-in regularization"],
            "cons": ["⚠️ Complex to tune", "⚠️ Black box model"],
            "best_for": "Competitions, when accuracy matters most",
            "hyperparams": ["learning_rate", "max_depth", "n_estimators", "subsample"]
        },
        "LightGBM": {
            "pros": ["✅ Fast training", "✅ Low memory usage", "✅ Handles large datasets"],
            "cons": ["⚠️ Can overfit with small data", "⚠️ Less interpretable"],
            "best_for": "Large datasets (>100K rows)",
            "hyperparams": ["learning_rate", "num_leaves", "n_estimators"]
        },
        "Gradient Boosting": {
            "pros": ["✅ Good for regression", "✅ Interpretable", "✅ Balanced performance"],
            "cons": ["⚠️ Slower than XGBoost/LightGBM", "⚠️ Prone to overfitting"],
            "best_for": "When interpretability matters",
            "hyperparams": ["learning_rate", "max_depth", "n_estimators", "subsample"]
        }
    }
    
    col1, col2 = st.columns(2)
    with col1:
        model1 = st.selectbox("Compare Model 1:", list(models_db.keys()), key="model1")
    with col2:
        model2 = st.selectbox("Compare Model 2:", list(models_db.keys()), key="model2", index=1)
    
    comp_col1, comp_col2 = st.columns(2)
    
    with comp_col1:
        st.markdown(f"### {model1}")
        m1 = models_db[model1]
        st.markdown("**Pros:**")
        for pro in m1['pros']:
            st.markdown(pro)
        st.markdown("**Cons:**")
        for con in m1['cons']:
            st.markdown(con)
        st.markdown(f"**Best for:** {m1['best_for']}")
    
    with comp_col2:
        st.markdown(f"### {model2}")
        m2 = models_db[model2]
        st.markdown("**Pros:**")
        for pro in m2['pros']:
            st.markdown(pro)
        st.markdown("**Cons:**")
        for con in m2['cons']:
            st.markdown(con)
        st.markdown(f"**Best for:** {m2['best_for']}")


def main() -> None:
    config = load_config()
    logger = setup_logging(config.logging_dir, config.logging_level)
    
    inject_custom_css()
    
    # Add AI sidebar assistant
    ai_sidebar_assistant()
    
    st.markdown(
        """
        <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
            <div style="font-size: 24px; font-weight: 800;">🤖 DS Assistant</div>
            <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Quick guidance, workflow steps, and tips tailored to your data.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    instruction_block(
        "How to use this page",
        [
            "Follow the workflow tab for an end-to-end checklist.",
            "Use Problem to pick your task type.",
            "Check Tips for quick, practical advice.",
            "Compare Models for a short list of good defaults.",
            "Open Data Insights to see quick stats on your loaded data.",
        ],
    )
    
    raw_df = get_df(st.session_state)
    clean_df = get_clean_df(st.session_state)
    sidebar_dataset_status(raw_df, clean_df)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📋 Workflow", "🎯 Problem", "💡 Tips", "🤖 Models", "📊 Data Insights"]
    )
    
    with tab1:
        _workflow_guide()
    
    with tab2:
        _problem_selector()
    
    with tab3:
        _quick_tips()
    
    with tab4:
        _model_comparison_helper()
    
    with tab5:
        st.subheader("📊 Your Dataset Insights")
        if clean_df is not None and not clean_df.empty:
            recommendations = _get_dataset_recommendations(clean_df)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rows", f"{recommendations['rows']:,}")
            with col2:
                st.metric("Columns", recommendations['columns'])
            with col3:
                st.metric("Missing Data", f"{recommendations['missing_pct']:.1f}%")
            with col4:
                st.metric("Numeric Features", recommendations['numeric_cols'])
            
            st.markdown("### 🎯 AI-Powered Recommendations")
            if recommendations['suggested_tasks']:
                for task in recommendations['suggested_tasks']:
                    st.info(task)
            else:
                st.success("✅ Your data looks good! Ready for modeling.")
        else:
            st.warning("📊 Load a dataset first to get insights")
    
    # Navigation
    st.divider()
    standard_section_header("Learning Guide & Best Practices", "🎓")
    concept_explainer(
        title="How to Use DS Assistant",
        explanation=(
            "This assistant helps you plan and execute an end-to-end data science project: explore data, clean, engineer features, train models, and deploy predictions."
        ),
        real_world_example=(
            "Retail analytics: Start with EDA to understand sales drivers, clean missing SKU attributes, engineer promotion flags, train a regression model, and deploy weekly forecasts."
        ),
    )
    beginner_tip("Tip: Start simple — validate a baseline model before adding complexity.")
    common_mistakes_panel({
        "Skipping EDA": "Without exploring data, you risk modeling noise or bias.",
        "No clear target": "Define what you want to predict and why.",
        "Over-complex pipelines": "Complexity increases maintenance cost — iterate in small steps.",
        "Ignoring monitoring": "Plan metrics to track in production from day one.",
    })

    page_navigation("1")


if __name__ == "__main__":
    main()
