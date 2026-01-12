"""
Beginner Regression Learning Module
Simple, step-by-step guide for learning regression with visual explanations and real-world examples.
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from src.core.ui import page_navigation, sidebar_dataset_status
from src.core.standardized_ui import (
    standard_page_header,
    standard_section_header,
    beginner_tip,
    before_after_comparison,
    metric_card,
)

st.set_page_config(page_title="Regression Learning", layout="wide", initial_sidebar_state="expanded")

standard_page_header(
    title="Regression Learning (Beginner)",
    subtitle="Predict numbers and understand model performance.",
    icon="🧑‍🎓",
)

st.markdown("""
### 🎯 Complete Regression Workflow

**Follow these steps in order for best results:**
1. 📚 **Understand Regression** - Learn the fundamentals
2. 📂 **Load Your Data** - Upload or use sample dataset
3. 🎯 **Choose Features & Target** - Select inputs and output
4. 🧪 **Split Data** - Separate training and testing sets
5. ⚙️ **Choose Model** - Pick algorithm
6. 🚀 **Train & Evaluate** - Build model and check performance
7. 📊 **Understand Metrics** - Interpret results
8. 🔍 **Feature Analysis** - See what matters most

**📚 What is Regression?**

Regression is supervised learning that predicts **continuous numeric values** (not categories).

**Real-world examples:**
- 🏠 **House Prices**: Predict $320,000 based on size, location, age
- 📈 **Stock Prices**: Forecast tomorrow's price at $152.30
- 🌡️ **Temperature**: Predict 72.5°F for tomorrow
- 💰 **Sales**: Estimate revenue of $45,230 next month
- ⏱️ **Delivery Time**: Predict 23.5 minutes

**How it works:**
1. **Training**: Model learns relationship between features (inputs) and target (output)
2. **Prediction**: Given new inputs, model estimates the numeric output
3. **Evaluation**: Measure how close predictions are to actual values

**Key difference from classification:**
- **Regression**: Predicts numbers (23.5, $45,000, 72°F)
- **Classification**: Predicts categories (Yes/No, Cat/Dog, High/Low)

**Types of regression problems:**
- **Simple**: One input feature (e.g., predict price from just square footage)
- **Multiple**: Many input features (predict price from size, location, age, bedrooms, etc.)
- **Time series**: Predict future values based on past (stock prices, sales forecasting)
""")

beginner_tip(
    "💡 **New to regression?** Think of it as 'connect the dots' - the model finds the best line/curve "
    "through your data points to make predictions. More data = better line = better predictions!"
)

sidebar_dataset_status(st.session_state.get("raw_df"), st.session_state.get("clean_df"))

# Step 1: Data Upload or Sample
standard_section_header("Step 1: Choose Your Data", "📂")
st.markdown("""
**Data requirements for regression:**
- ✅ Numeric target column (the value you want to predict)
- ✅ At least one feature column (can be numeric or categorical)
- ✅ Enough rows (50+ minimum, 100+ recommended, 1000+ ideal)
- ✅ No extreme missing values (< 20% missing ideal)

**Sample dataset:**
Simple synthetic data showing linear relationship between a feature and target.
Perfect for learning how regression works!
""")

data_source = st.radio("Select data source:", ["Upload CSV", "Use sample dataset"], horizontal=True)
df = None
if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {df.shape[0]} rows × {df.shape[1]} columns.")
elif data_source == "Use sample dataset":
    np.random.seed(42)
    X = np.linspace(0, 10, 150)
    y = 2 * X + np.random.normal(0, 2, 150)
    df = pd.DataFrame({'Feature': X, 'Target': y})
    st.success("Loaded sample regression dataset.")

if df is not None:
    st.dataframe(df.head(10), use_container_width=True)
    st.caption("Preview: First 10 rows of your data.")
    st.markdown("""
    **✅ Data loaded successfully!**
    
    **What to check:**
    - Do you see your target column (what you want to predict)?
    - Are feature columns present (inputs for prediction)?
    - Any obvious errors or missing values?
    - Do numbers look reasonable (no typos like 99999999)?
    """)
else:
    st.info("Upload a CSV or use the sample dataset to begin.")
    st.stop()

# Step 2: Select Features and Target
standard_section_header("Step 2: Select Features and Target", "🎯")
st.markdown("""
**📚 Understanding Features and Target:**

**Target (Output):**
- The number you want to predict
- Examples: price, temperature, sales amount
- Must be a single numeric column
- Should have meaningful variation (not all the same value)

**Features (Inputs):**
- Columns used to make predictions
- Can be numeric (age, income) or categorical (color, region)
- More relevant features = better predictions
- Quality > quantity (10 good features beats 100 random ones)

**Feature selection tips:**
- ✅ Include features you believe influence the target
- ✅ Start with 3-10 features for simplicity
- ❌ Avoid including target-derived columns (data leakage!)
- ❌ Avoid ID columns or random identifiers

**Example:**
- **Target**: House Price
- **Good features**: Square footage, bedrooms, location, age
- **Bad features**: Listing ID, sale date (might leak information)
""")
feature_cols = st.multiselect("Feature columns (inputs):", options=[c for c in df.columns if c != 'Target'], default=[c for c in df.columns if c != 'Target'])
target_col = st.selectbox("Target column (output):", options=[c for c in df.columns if c not in feature_cols])
if not feature_cols or not target_col:
    st.warning("Select at least one feature and a target column.")
    st.stop()

# Properly align X and y - drop rows with NaN in either X or y
df_clean = df[feature_cols + [target_col]].dropna()
if len(df_clean) < 10:
    st.error(f"Not enough valid rows after removing NaN values. You have {len(df_clean)} rows, need at least 10.")
    st.stop()

X = df_clean[feature_cols]
y = df_clean[target_col]

# Step 3: Train/Test Split
standard_section_header("Step 3: Split Data for Training and Testing", "🧪")
st.markdown("""
**📚 Why Split Data?**

Just like students need practice problems AND a final exam:
- **Training set (80%)**: Model learns patterns from this data
- **Test set (20%)**: Evaluates model on new, unseen data

**Why this matters:**
- Testing on training data gives falsely optimistic scores (like memorizing answers)
- Test set simulates real-world performance on future data
- Helps detect overfitting (model memorized training data but can't generalize)

**Rule of thumb:**
- 80/20 split is standard for medium datasets (hundreds to thousands of rows)
- 70/30 for smaller datasets (< 500 rows)
- 90/10 for very large datasets (> 10,000 rows)
""")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
col1, col2 = st.columns(2)
with col1:
    st.metric("Training Rows", f"{X_train.shape[0]}", help="Model learns from this data")
with col2:
    st.metric("Testing Rows", f"{X_test.shape[0]}", help="Model is evaluated on this unseen data")

# Step 4: Model Selection
standard_section_header("Step 4: Choose a Regression Model", "⚙️")
st.markdown("""
**🤖 Understanding Regression Models:**

📈 **Linear Regression** (Simple & Fast)
- **How it works**: Fits the best straight line (y = mx + b) through your data
- **Best for**: When relationship between features and target is roughly linear
- **Strengths**: Fast, interpretable, gives coefficient weights
- **Weaknesses**: Assumes linear relationships, sensitive to outliers
- **Use when**: You need quick results and explainability

📈 **Ridge Regression** (Regularized Linear)
- **How it works**: Like linear regression but penalizes large coefficients
- **Best for**: When you have many correlated features
- **Strengths**: Prevents overfitting, handles multicollinearity well
- **Weaknesses**: Still assumes linearity
- **Use when**: Linear regression overfits or features are correlated

📈 **Lasso Regression** (Feature Selection)
- **How it works**: Like Ridge but can shrink coefficients to exactly zero
- **Best for**: When you suspect many features are irrelevant
- **Strengths**: Automatic feature selection, interpretable
- **Weaknesses**: May not perform well with highly correlated features
- **Use when**: You want automatic feature elimination

🌲 **Random Forest Regression** (Powerful & Flexible)
- **How it works**: Averages predictions from many decision trees
- **Best for**: Complex non-linear relationships
- **Strengths**: Handles non-linearity, robust to outliers, no scaling needed
- **Weaknesses**: Slower, less interpretable, can overfit with default settings
- **Use when**: Linear models fail or relationships are complex

**🎯 Quick Selection Guide:**

| Your Situation | Recommended Model |
|----------------|------------------|
| Simple linear relationship | Linear Regression |
| Many correlated features | Ridge |
| Want automatic feature selection | Lasso |
| Complex/non-linear patterns | Random Forest |
| Need interpretability | Linear, Ridge, or Lasso |
| Have outliers | Random Forest |

**💡 Pro tip:** Always start with Linear Regression as baseline, then try others if performance is poor.
""")

model_name = st.selectbox("Model:", ["Linear Regression", "Ridge", "Lasso", "Random Forest Regression"])
if model_name == "Linear Regression":
    model = LinearRegression()
    st.info("✅ Simple and interpretable. Best for linear relationships.")
elif model_name == "Ridge":
    model = Ridge()
    st.info("✅ Handles correlated features well. Good for preventing overfitting.")
elif model_name == "Lasso":
    model = Lasso()
    st.info("✅ Automatic feature selection. Sets unimportant features to zero.")
elif model_name == "Random Forest Regression":
    model = RandomForestRegressor()
    st.info("✅ Powerful for complex patterns. Handles non-linearity naturally.")
else:
    st.error("Unknown model selected.")
    st.stop()

# Step 5: Train the Model
standard_section_header("Step 5: Train Your Model & Evaluate Results", "🚀")
if st.button("Train Model", type="primary"):
    with st.spinner("Training model... ⏳"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
    
    st.success(f"✅ Model trained successfully!")
    
    st.markdown("""
    ---
    ### 📊 Understanding Regression Metrics
    
    Different metrics tell you different things about prediction quality:
    
    🎯 **MAE (Mean Absolute Error)**
    - **Formula**: Average of |actual - predicted|
    - **What it means**: Average error size in original units
    - **Example**: MAE of $5,000 means predictions are off by $5k on average
    - **When to use**: When all errors matter equally, easy to explain
    - **Good value**: Depends on scale; compare to target's standard deviation
    - **Pros**: Easy to understand, robust to outliers
    - **Cons**: Doesn't heavily penalize large errors
    
    🎯 **MSE (Mean Squared Error)**
    - **Formula**: Average of (actual - predicted)²
    - **What it means**: Average squared error (penalizes large mistakes more)
    - **Why squared**: Amplifies large errors (error of 10 counts as 100)
    - **When to use**: When large errors are particularly bad
    - **Challenge**: Units are squared (hard to interpret directly)
    - **Pros**: Differentiable (good for optimization), penalizes outliers
    - **Cons**: Sensitive to outliers, units are squared
    
    🎯 **RMSE (Root Mean Squared Error)**
    - **Formula**: √MSE
    - **What it means**: Like MSE but back in original units
    - **Example**: RMSE of $7,000 means typical error is $7k (more weight on large errors than MAE)
    - **When to use**: When you want to penalize large errors but keep interpretable units
    - **Interpretation**: Usually larger than MAE (due to squaring large errors)
    - **Pros**: Same units as target, penalizes large errors
    - **Cons**: Still sensitive to outliers
    
    🎯 **R² Score (Coefficient of Determination)**
    - **Formula**: 1 - (Sum of Squared Errors / Total Variance)
    - **Range**: -∞ to 1.0 (usually 0 to 1)
    - **What it means**: % of variance in target explained by model
    - **Interpretation**:
      - **1.0** = Perfect predictions ✅
      - **0.8-0.9** = Excellent predictions
      - **0.7-0.8** = Good predictions
      - **0.5-0.7** = Moderate predictions
      - **< 0.5** = Poor predictions
      - **0.0** = Model as good as predicting the mean
      - **< 0.0** = Model worse than predicting the mean! ⚠️
    - **When to use**: Comparing models on same data, scale-independent metric
    - **Pros**: Easy to interpret (0-1 scale), scale-independent
    - **Cons**: Can be misleading with non-linear relationships
    
    **📈 Visual interpretation aids:**
    - **Actual vs Predicted plot**: Points should align with diagonal line
    - **Error distribution**: Should be centered around zero with symmetrical spread
    - **Residuals plot**: Random scatter = good; patterns = model missing something
    
    **🎯 Which metric to prioritize?**
    
    | Scenario | Best Metric | Why |
    |----------|------------|-----|
    | Easy interpretation | MAE or R² | Simple to explain to stakeholders |
    | Outliers present | MAE | Less sensitive to extreme values |
    | Large errors very bad | RMSE or MSE | Penalizes big mistakes heavily |
    | Comparing models | R² | Scale-independent comparison |
    | General purpose | RMSE + R² | RMSE for magnitude, R² for fit quality |
    
    **💡 Rules of thumb:**
    - MAE < 10% of target range = Good model
    - R² > 0.7 = Good explanatory power
    - RMSE should be compared to target's std deviation (RMSE < 0.5×std is good)
    - If R² < 0: Model is broken or data has issues!
    """)
    
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("MAE", f"{mae:.3f}", help="Average absolute error")
        st.caption(f"Lower is better")
    with m2:
        st.metric("RMSE", f"{rmse:.3f}", help="Root mean squared error")
        st.caption(f"Penalizes large errors")
    with m3:
        st.metric("R² Score", f"{r2:.3f}", help="Variance explained (0-1)")
        if r2 >= 0.8:
            st.caption("✅ Excellent!")
        elif r2 >= 0.7:
            st.caption("✅ Good")
        elif r2 >= 0.5:
            st.caption("⚠️ Moderate")
        else:
            st.caption("❌ Needs improvement")
    with m4:
        # Calculate MAPE if no zeros in actual values
        if (y_test == 0).any():
            st.metric("MAPE", "N/A", help="Not available (zero values present)")
        else:
            mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
            st.metric("MAPE", f"{mape:.1f}%", help="Mean Absolute Percentage Error")
            st.caption(f"Avg error %")

    st.markdown("---")
    st.markdown("#### 📊 Visual Results Analysis")
    
    # Actual vs Predicted
    col_vis1, col_vis2 = st.columns(2)
    
    # Actual vs Predicted
    col_vis1, col_vis2 = st.columns(2)
    
    with col_vis1:
        st.markdown("**Actual vs Predicted**")
        st.markdown("Points should cluster near the diagonal line for good predictions")
        actual_pred_df = pd.DataFrame({
            'Actual': y_test.values,
            'Predicted': y_pred,
            'Index': range(len(y_test))
        })
        
        # Use Plotly for better visualization
        import plotly.graph_objects as go
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_test.values,
            y=y_pred,
            mode='markers',
            name='Predictions',
            marker=dict(size=8, opacity=0.6, color='blue')
        ))
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash', width=2)
        ))
        
        fig.update_layout(
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            height=350,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("✅ Good: Points near red line | ❌ Bad: Points scattered far from line")
    
    with col_vis2:
        st.markdown("**Error Distribution (Residuals)**")
        st.markdown("Should be centered around zero with symmetric spread")
        errors = y_test.values - y_pred
        
        fig_err = go.Figure()
        fig_err.add_trace(go.Histogram(
            x=errors,
            nbinsx=30,
            name='Errors',
            marker=dict(color='steelblue')
        ))
        
        fig_err.update_layout(
            xaxis_title="Prediction Error",
            yaxis_title="Frequency",
            height=350,
            showlegend=False
        )
        st.plotly_chart(fig_err, use_container_width=True)
        st.caption("✅ Good: Bell curve centered at 0 | ❌ Bad: Skewed or multiple peaks")

    st.markdown("""
    ---
    ### 💡 Interpreting Your Results
    
    **If R² is high (> 0.7):**
    - ✅ Model captures patterns well
    - ✅ Features are predictive
    - Next: Deploy or try advanced models for small improvements
    
    **If R² is moderate (0.5-0.7):**
    - ⚠️ Model explains some variance but missing patterns
    - Try: Add more relevant features, engineer interactions, try non-linear models
    
    **If R² is low (< 0.5):**
    - ❌ Model doesn't capture relationship well
    - Check: Are features actually related to target? Is there enough data?
    - Try: Collect more data, add better features, check for data quality issues
    
    **If R² is negative:**
    - 🚨 Model is worse than predicting the mean!
    - Problem: Usually indicates serious issues (wrong model, data leakage, bugs)
    - Action: Check data, revisit feature selection, try simpler model
    
    **Error pattern analysis:**
    - **Random scatter in residuals**: Good! ✅
    - **Funnel shape** (errors increase with value): Consider log transformation
    - **Patterns or curves**: Model missing non-linear relationships
    - **Outliers**: Consider removing or using robust methods
    """)

    st.markdown("---")
    # Explainability: Permutation Importance
    try:
        standard_section_header("Step 6: Feature Importance Analysis", "🧪")
        st.markdown("""
        **📚 Understanding Feature Importance:**
        
        **What is it?**
        - Shows which features most influence predictions
        - Helps identify key drivers and redundant features
        
        **How it's calculated (Permutation Importance):**
        1. Measure baseline model performance
        2. Randomly shuffle one feature's values
        3. Measure new performance
        4. Importance = drop in performance
        5. Repeat for all features
        
        **How to interpret:**
        - **High importance**: Feature strongly influences predictions (keep!)
        - **Near-zero**: Feature doesn't help much (consider removing)
        - **Negative**: Rare, but means shuffling improved model (red flag - may indicate overfitting)
        
        **What to do with results:**
        - Focus engineering efforts on high-importance features
        - Remove low-importance features to simplify model
        - Investigate unexpected importances (domain validation)
        - Look for feature groups (related features may share importance)
        
        **💡 Pro tips:**
        - For Linear/Ridge/Lasso: Can also check coefficients directly
        - Random Forest: Also has built-in importance (faster but less reliable)
        - Multiple runs give confidence intervals (importance ± std deviation)
        """)
        
        result = permutation_importance(model, X_test, y_test, scoring="r2", n_repeats=5, random_state=42)
        importances = pd.DataFrame({
            "Feature": X.columns,
            "Importance": result.importances_mean,
            "Std Dev": result.importances_std,
        }).sort_values("Importance", ascending=False)
        
        # Visualize importance
        fig_imp = go.Figure()
        fig_imp.add_trace(go.Bar(
            x=importances["Importance"],
            y=importances["Feature"],
            orientation='h',
            error_x=dict(type='data', array=importances["Std Dev"]),
            marker=dict(color='skyblue')
        ))
        fig_imp.update_layout(
            title="Feature Importance (with std deviation)",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=max(300, len(importances) * 30)
        )
        st.plotly_chart(fig_imp, use_container_width=True)
        
        st.dataframe(importances, use_container_width=True)
        
        # Interpretation help
        top_feature = importances.iloc[0]["Feature"]
        top_importance = importances.iloc[0]["Importance"]
        st.success(f"🎯 **Key Finding**: '{top_feature}' is the most important feature (importance: {top_importance:.4f})")
        
        low_importance_features = importances[importances["Importance"] < 0.01]["Feature"].tolist()
        if low_importance_features:
            st.info(f"💡 **Optimization opportunity**: Consider removing low-importance features: {', '.join(low_importance_features)}")
        
    except Exception as e:
        st.caption(f"Feature importance unavailable: {e}")

    st.markdown("---")
    # Advanced Explainability: SHAP (if installed)
    if st.session_state.get("ui_mode") == "Advanced":
        try:
            import shap
            standard_section_header("Advanced Explainability (SHAP)", "🔍")
            st.markdown("""
            **SHAP (SHapley Additive exPlanations):**
            - More sophisticated than permutation importance
            - Shows both magnitude AND direction of impact
            - Can explain individual predictions
            - Based on game theory (fair credit allocation)
            """)
            
            sample_X = X_test.copy()
            if len(sample_X) > 200:
                sample_X = sample_X.sample(200, random_state=42)
            explainer = shap.Explainer(model, X_train)
            explanation = explainer(sample_X)
            vals = explanation.values
            mean_abs = np.mean(np.abs(vals), axis=0)
            shap_df = pd.DataFrame({
                "Feature": X.columns,
                "Mean |SHAP|": mean_abs,
            }).sort_values("Mean |SHAP|", ascending=False)
            st.dataframe(shap_df.head(10), use_container_width=True)
            st.caption("Mean absolute SHAP values indicate average feature impact on predictions. Requires 'shap' package.")
        except Exception as e:
            st.caption(f"SHAP explainability unavailable: {e}. Tip: Install with 'pip install shap'")
else:
    st.info("👆 Click 'Train Model' above to see results, visualizations, and feature importance analysis.")

page_navigation("6")
