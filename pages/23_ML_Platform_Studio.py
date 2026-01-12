"""
ML Platform Studio
------------------
A consolidated end-to-end Data Science & Advanced ML experience
covering Beginner, Intermediate, and Advanced flows with education-first UI.
"""

import datetime as dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.core.ui import page_navigation
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    LinearRegression,
    LogisticRegression,
    Lasso,
    PoissonRegressor,
    QuantileRegressor,
    Ridge,
)
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

# Optional advanced models
HAS_XGB = False
HAS_CAT = False
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    pass
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    HAS_CAT = True
except Exception:
    pass

MAX_ROWS = 50000  # guardrail for very large uploads
MAX_SHAP_ROWS = 1000
MAX_PI_ROWS = 5000
MAX_CAL_ROWS = 2000

st.set_page_config(
    page_title="ML Platform Studio",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sample_dataset(name: str) -> Tuple[pd.DataFrame, pd.Series, str]:
    """Return (X, y, task) for built-in samples."""
    if name == "iris":
        from sklearn.datasets import load_iris
        data = load_iris(as_frame=True)
        return data.data, data.target, "classification"
    if name == "titanic":
        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
        target = (df["Survived"]).astype(int)
        X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
        return X, target, "classification"
    if name == "diabetes":
        from sklearn.datasets import load_diabetes
        data = load_diabetes(as_frame=True)
        return data.data, data.target, "regression"
    if name == "california":
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing(as_frame=True)
        return data.data.sample(4000, random_state=42), data.target.sample(4000, random_state=42), "regression"
    if name == "airline":
        rng = pd.date_range("2015-01-01", periods=180, freq="D")
        values = 200 + 0.5 * np.arange(len(rng)) + 20 * np.sin(np.arange(len(rng)) / 14) + np.random.normal(0, 5, len(rng))
        df = pd.DataFrame({"date": rng, "value": values})
        df["month"] = df["date"].dt.month
        df["dayofweek"] = df["date"].dt.dayofweek
        df["day"] = df["date"].dt.day
        target = df.pop("value")
        df = df.drop(columns=["date"])
        return df, target, "regression"
    raise ValueError("Unknown sample dataset")


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cats = X.select_dtypes(include=["object", "category"]).columns.tolist()
    nums = [c for c in X.columns if c not in cats]
    cat_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]
    )
    num_pipe = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    return ColumnTransformer(
        transformers=[("cat", cat_pipe, cats), ("num", num_pipe, nums)], remainder="drop"
    )


def summarize_classification(y_true, y_pred, y_proba=None) -> Dict:
    classes = np.unique(y_true)
    is_multiclass = len(classes) > 2
    avg = "macro" if is_multiclass else "binary"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
        "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }
    if y_proba is not None:
        try:
            if is_multiclass and hasattr(y_proba, "shape") and len(getattr(y_proba, "shape", [])) == 2 and y_proba.shape[1] > 2:
                metrics["roc_auc_macro"] = roc_auc_score(y_true, y_proba, multi_class="ovo", average="macro")
            else:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        except Exception:
            pass
        try:
            if is_multiclass and hasattr(y_proba, "shape") and len(getattr(y_proba, "shape", [])) == 2 and y_proba.shape[1] > 2:
                metrics["pr_auc_macro"] = average_precision_score(y_true, y_proba, average="macro")
            else:
                metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        except Exception:
            pass
    return metrics


def summarize_regression(y_true, y_pred) -> Dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "medae": median_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

def section_header(title: str, subtitle: str = ""):
    st.markdown(f"### {title}")
    if subtitle:
        st.caption(subtitle)


def info_box(text: str, icon: str = "ℹ️"):
    st.info(f"{icon} {text}")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.title("💎 ML Platform Studio")
st.caption("Complete end-to-end ML workflow: Data → EDA → Train → Evaluate → Explain → Predict")

st.markdown("""
### 📋 Workflow Guide
Follow these steps in order for best results:
1. **Data Loading** - Select or upload your dataset
2. **Data Profiling** - Understand your data structure and quality
3. **EDA & Visualization** - Explore patterns and relationships
4. **Data Preparation** - Configure cleaning and encoding
5. **Model Training** - Train and compare multiple models
6. **Results Interpretation** - Understand metrics and performance
7. **Model Explainability** - Discover feature importance
8. **Predictions** - Use your trained model on new data
9. **Model Comparison** - Compare all models side-by-side
""")

with st.container():
    cols = st.columns(6)
    chips = ["📊 Data", "🔍 EDA", "🎯 Train", "📈 Evaluate", "🔬 Explain", "🚀 Predict"]
    for i, chip in enumerate(chips):
        cols[i].button(chip, key=f"chip_{chip}")

# Sidebar: dataset selection
with st.sidebar:
    st.header("📂 Dataset")
    sample = st.selectbox(
        "Choose sample or upload",
        ["iris (class)", "titanic (class)", "diabetes (regr)", "california (regr)", "airline (ts)", "Upload CSV"],
    )
    uploaded = None
    if sample == "Upload CSV":
        uploaded = st.file_uploader("Upload CSV", type=["csv", "txt"])
    target_col = st.text_input("Target column (for upload)", value="")

# Load data
if "platform_data" not in st.session_state:
    st.session_state.platform_data = None
    st.session_state.platform_target = None
    st.session_state.platform_task = None
    st.session_state.platform_thresholds = {}

if sample != "Upload CSV":
    X_raw, y_raw, task_type = get_sample_dataset(sample.split()[0])
    st.session_state.platform_data = X_raw
    st.session_state.platform_target = y_raw
    st.session_state.platform_task = task_type
else:
    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        if target_col and target_col in df_up.columns:
            y_raw = df_up[target_col]
            X_raw = df_up.drop(columns=[target_col])
            st.session_state.platform_data = X_raw
            st.session_state.platform_target = y_raw
            st.session_state.platform_task = "classification" if y_raw.nunique() <= 15 else "regression"
        else:
            st.warning("Please provide a valid target column name")

X = st.session_state.platform_data
y = st.session_state.platform_target
current_task = st.session_state.platform_task

if X is None or y is None:
    st.stop()

if not isinstance(X, pd.DataFrame):
    X = pd.DataFrame(X)
if not isinstance(y, pd.Series):
    y = pd.Series(y)
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

if len(X) > MAX_ROWS:
    st.warning(f"Sampling to {MAX_ROWS} rows for speed; upload filters not affected.")
    X = X.sample(MAX_ROWS, random_state=42)
    y = y.loc[X.index]

if X is None or y is None:
    st.stop()

# ---------------------------------------------------------------------------
# STEP 1: Data Loading & Profiling
# ---------------------------------------------------------------------------
section_header("📊 Step 1: Data Profiling", "Understand your dataset structure and quality")

with st.expander("📋 Data Overview", expanded=True):
    st.markdown("""
    **What to look for:**
    - **Rows**: Sample size affects model reliability (more is generally better)
    - **Columns**: Number of features; too many may cause overfitting
    - **Missing %**: High percentages may require imputation or removal
    """)
    colA, colB, colC = st.columns(3)
    colA.metric("Rows", len(X))
    colB.metric("Columns", X.shape[1])
    colC.metric("Missing (%)", round(X.isna().mean().mean() * 100, 2))
    
    st.markdown("**Sample of your data:**")
    st.dataframe(X.head(), use_container_width=True)
    
    st.markdown("""
    💡 **Data Quality Tips:**
    - Missing < 5%: Usually safe to proceed
    - Missing 5-20%: Consider imputation strategies
    - Missing > 20%: May need to drop columns or investigate data collection
    """)

# ---------------------------------------------------------------------------
# STEP 2: EDA & Visualization
# ---------------------------------------------------------------------------
section_header("🔍 Step 2: Exploratory Data Analysis", "Visualize distributions and relationships")

with st.expander("📊 Feature Distributions", expanded=False):
    st.markdown("""
    **How to interpret:**
    - **Histogram**: Shows value distribution; look for skewness or outliers
    - **Box plot** (top): Reveals median, quartiles, and outliers
    - **Skewed data**: May benefit from log transformation
    - **Outliers**: May indicate errors or important edge cases
    """)
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    viz_df = X.sample(min(2000, len(X)), random_state=42)
    if num_cols:
        feature = st.selectbox("Numeric feature", num_cols)
        fig = px.histogram(viz_df, x=feature, nbins=25, marginal="box")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric features found for visualization")

with st.expander("🔗 Feature Relationships", expanded=False):
    st.markdown("""
    **Scatter plot interpretation:**
    - **Linear patterns**: Suggest strong relationships
    - **Clusters**: May indicate natural groupings
    - **Color patterns**: Show how target relates to features
    """)
    if len(num_cols) >= 2:
        f1, f2 = st.selectbox("X-axis", num_cols, key="scatter_x"), st.selectbox("Y-axis", num_cols, key="scatter_y")
        fig2 = px.scatter(viz_df, x=f1, y=f2, color=y.loc[viz_df.index] if len(np.unique(y))<15 else None)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Need at least 2 numeric features for scatter plots")

with st.expander("🔥 Correlation Heatmap", expanded=False):
    st.markdown("""
    **Correlation interpretation:**
    - **+1.0**: Perfect positive correlation (both increase together)
    - **0.0**: No linear relationship
    - **-1.0**: Perfect negative correlation (one increases, other decreases)
    - **> 0.7 or < -0.7**: Strong correlation; may indicate redundant features
    """)
    if len(num_cols) >= 2:
        corr_df = X[num_cols].sample(min(5000, len(X)), random_state=42)
        corr = corr_df.corr()
        fig = px.imshow(corr, color_continuous_scale="RdBu", origin="lower", 
                       zmin=-1, zmax=1, text_auto=".2f")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 2 numeric features to compute correlation")

# ---------------------------------------------------------------------------
# STEP 3: Data Preparation
# ---------------------------------------------------------------------------
section_header("🛠️ Step 3: Data Preparation", "Configure data cleaning and encoding")

with st.expander("⚙️ Preprocessing Options", expanded=False):
    st.markdown("""
    **Strategy guide:**
    - **Missing values**: Choose strategy based on data type and % missing
    - **Scaling**: StandardScaler recommended for most algorithms
    - **Encoding**: One-hot encoding works well for categorical features
    """)
    miss_strategy = st.selectbox("Missing values", ["median", "most_frequent", "drop"], index=0)
    scale_choice = st.selectbox("Scaling", ["standard", "none"], index=0)
    encode_choice = st.selectbox("Categorical encoding", ["one-hot"], index=0)
    st.info("💡 These settings are applied automatically during model training")

# ---------------------------------------------------------------------------
# STEP 4: Model Training
# ---------------------------------------------------------------------------
section_header("🎯 Step 4: Model Training", "Train and compare multiple ML models")

with st.expander("🚀 Train Models", expanded=True):
    st.markdown("""
    **Training options:**
    - **Test size**: 20% is standard; larger for small datasets
    - **Class weights**: Use for imbalanced classification problems
    - **Calibration**: Improves probability estimates
    - **Models**: Start with RandomForest & ExtraTrees for robust baselines
    """)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", 0, 9999, 42)
    class_weight = st.checkbox("Class weights (balanced)", value=False)
    calibrate = st.checkbox("Calibrate probabilities (classification)", value=False)

    model_choice = st.multiselect(
        "Models",
        [
            "LogisticRegression",
            "SVC",
            "KNN",
            "NaiveBayes",
            "DecisionTree",
            "RandomForest",
            "ExtraTrees",
            "HistGradientBoosting",
            "GradientBoosting",
            "Linear/SVR",
            "ElasticNet/Ridge/Lasso",
            "Huber/Quantile/Poisson",
            "MLP",
            "QuantileRegressor",
            "PoissonRegressor",
        ],
        default=["RandomForest", "ExtraTrees", "LogisticRegression"],
    )

    if st.button("Train selected models", type="primary"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y if current_task=="classification" else None, random_state=random_state)
        pre = get_preprocessor(X)
        proba_cache = {}

        def make_model(name: str):
            if name == "LogisticRegression":
                base = LogisticRegression(max_iter=1000, class_weight="balanced" if class_weight else None)
                return CalibratedClassifierCV(base, cv=3, method="isotonic") if calibrate and current_task=="classification" else base
            if name == "SVC":
                base = SVC(probability=True, class_weight="balanced" if class_weight else None)
                return CalibratedClassifierCV(base, cv=3, method="sigmoid") if calibrate and current_task=="classification" else base
            if name == "KNN":
                return KNeighborsClassifier() if current_task == "classification" else KNeighborsRegressor()
            if name == "NaiveBayes":
                return GaussianNB()
            if name == "DecisionTree":
                return DecisionTreeClassifier(class_weight="balanced" if class_weight else None) if current_task == "classification" else DecisionTreeRegressor()
            if name == "RandomForest":
                return RandomForestClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced" if class_weight else None) if current_task == "classification" else RandomForestRegressor(n_estimators=400, n_jobs=-1)
            if name == "ExtraTrees":
                return ExtraTreesClassifier(n_estimators=400, n_jobs=-1, class_weight="balanced" if class_weight else None) if current_task == "classification" else ExtraTreesRegressor(n_estimators=400, n_jobs=-1)
            if name == "HistGradientBoosting":
                return HistGradientBoostingClassifier() if current_task == "classification" else HistGradientBoostingRegressor()
            if name == "GradientBoosting":
                return GradientBoostingClassifier() if current_task == "classification" else GradientBoostingRegressor()
            if name == "Linear/SVR":
                return LogisticRegression(max_iter=1000, class_weight="balanced" if class_weight else None) if current_task == "classification" else SVR()
            if name == "ElasticNet/Ridge/Lasso" and current_task == "regression":
                return ElasticNet(max_iter=2000)
            if name == "Huber/Quantile/Poisson" and current_task == "regression":
                return HuberRegressor()
            if name == "ElasticNet/Ridge/Lasso" and current_task == "classification":
                return LogisticRegression(max_iter=1500, penalty="l2", class_weight="balanced" if class_weight else None)
            if name == "Huber/Quantile/Poisson" and current_task == "classification":
                return LogisticRegression(max_iter=1500, class_weight="balanced" if class_weight else None)
            if name == "QuantileRegressor" and current_task == "regression":
                return QuantileRegressor(quantile=0.5, alpha=0.0, solver="highs")
            if name == "PoissonRegressor" and current_task == "regression":
                return PoissonRegressor(max_iter=1000)
            if name == "MLP":
                return MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300) if current_task == "classification" else MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=300)
            return RandomForestClassifier() if current_task == "classification" else RandomForestRegressor()

        results = []
        st.session_state.platform_models = {}
        for name in model_choice:
            model = make_model(name)
            pipe = Pipeline(steps=[("prep", pre), ("model", model)])
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            y_proba = None
            if current_task == "classification" and hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_test)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    y_proba = proba[:, 1]
                else:
                    y_proba = proba
            proba_cache[name] = y_proba
            if current_task == "classification":
                m = summarize_classification(y_test, y_pred, y_proba)
                if y_proba is not None and isinstance(y_proba, np.ndarray) and y_proba.ndim == 1:
                    thresholds = np.linspace(0.1, 0.9, 17)
                    f1s = [f1_score(y_test, (y_proba >= t).astype(int), zero_division=0) for t in thresholds]
                    best_t = thresholds[int(np.argmax(f1s))]
                    m["best_threshold_f1"] = float(best_t)
                    st.session_state.platform_thresholds[name] = best_t
            else:
                m = summarize_regression(y_test, y_pred)
            results.append({"model": name, **m})
            st.session_state.platform_models[name] = pipe
        st.success("✅ Training complete!")
        
        # ---------------------------------------------------------------------------
        # STEP 5: Results Interpretation
        # ---------------------------------------------------------------------------
        st.markdown("---")
        st.markdown("### 📈 Step 5: Results Interpretation")
        
        if current_task == "classification":
            st.markdown("""
            **Classification Metrics Guide:**
            - **Accuracy**: % of correct predictions (use cautiously with imbalanced data)
            - **Balanced Accuracy**: Better for imbalanced classes (average of sensitivity/specificity)
            - **Precision**: Of predicted positives, how many are actually positive? (minimize false positives)
            - **Recall**: Of actual positives, how many did we catch? (minimize false negatives)
            - **F1 Score**: Harmonic mean of precision & recall (good overall metric)
            - **ROC-AUC**: Model's ability to distinguish classes (0.5=random, 1.0=perfect)
            - **PR-AUC**: Better than ROC-AUC for highly imbalanced data
            - **MCC**: Matthews correlation (-1 to +1); robust metric for all class sizes
            - **Kappa**: Agreement beyond chance; good for imbalanced data
            
            **Which metric to prioritize?**
            - Balanced dataset: Accuracy or F1
            - Imbalanced dataset: PR-AUC, Balanced Accuracy, or MCC
            - Cost-sensitive: Precision (costly false positives) or Recall (costly false negatives)
            """)
        else:
            st.markdown("""
            **Regression Metrics Guide:**
            - **RMSE** (Root Mean Squared Error): Average prediction error in original units (penalizes large errors)
            - **MAE** (Mean Absolute Error): Average absolute error (less sensitive to outliers)
            - **MAPE** (Mean Absolute % Error): Error as percentage (good for comparing across scales)
            - **Median AE**: Median absolute error (most robust to outliers)
            - **R²**: % of variance explained (0-1; higher is better; can be negative if model is worse than mean)
            
            **Which metric to prioritize?**
            - General use: RMSE or MAE
            - With outliers: Median AE
            - Comparing across datasets: MAPE or R²
            - Lower is better for all except R² (higher is better)
            """)
        
        results_df = pd.DataFrame(results)
        st.dataframe(results_df, use_container_width=True)
        
        st.markdown("""
        💡 **Model Selection Tips:**
        - Look for best metric, but also consider model complexity
        - Check multiple metrics; don't rely on just one
        - Simpler models (LogisticRegression, DecisionTree) are easier to explain
        - Ensemble models (RandomForest, GradientBoosting) usually perform better
        """)

        # Visuals
        st.markdown("### 📊 Visual Comparison")
        if current_task == "classification" and len(model_choice) > 0:
            st.markdown("**F1 Score Comparison** - Higher bars indicate better balance of precision & recall")
            figm = px.bar(results_df, x="model", y="f1", title="F1 scores")
            st.plotly_chart(figm, use_container_width=True)
            if "pr_auc" in results_df.columns:
                st.markdown("**PR-AUC Comparison** - Especially useful for imbalanced datasets")
                fig_pr = px.bar(results_df, x="model", y="pr_auc", title="PR-AUC (imbalance friendly)")
                st.plotly_chart(fig_pr, use_container_width=True)
            calib_source = next(((n, proba_cache[n]) for n in model_choice if isinstance(proba_cache.get(n), np.ndarray) and proba_cache[n].ndim == 1), None)
            if calib_source:
                name, probs = calib_source
                sample_idx = np.random.RandomState(42).choice(len(y_test), size=min(len(y_test), MAX_CAL_ROWS), replace=False)
                frac_y = np.array(y_test)[sample_idx]
                frac_p = np.array(probs)[sample_idx]
                try:
                    prob_true, prob_pred = calibration_curve(frac_y, frac_p, n_bins=8, strategy="quantile")
                    fig_cal = go.Figure()
                    fig_cal.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Perfect", line=dict(color="gray", dash="dash")))
                    fig_cal.add_trace(go.Scatter(x=prob_pred, y=prob_true, mode="lines+markers", name=f"{name} (reliability)"))
                    fig_cal.update_layout(title="Calibration curve (reliability diagram)", 
                                         xaxis_title="Mean predicted probability", 
                                         yaxis_title="Fraction of positives")
                    st.plotly_chart(fig_cal, use_container_width=True)
                    st.markdown("""
                    **Calibration Curve Interpretation:**
                    - **Diagonal line**: Perfect calibration (predicted probabilities match reality)
                    - **Above diagonal**: Model underestimates probabilities
                    - **Below diagonal**: Model overestimates probabilities
                    - **Use case**: Important when probability values matter (not just class predictions)
                    """)
                except Exception:
                    st.info("Calibration plot not available for this model")
        if current_task == "regression" and len(model_choice) > 0:
            st.markdown("**RMSE Comparison** - Lower bars are better; shows average prediction error")
            figm = px.bar(results_df, x="model", y="rmse", title="RMSE (lower is better)")
            st.plotly_chart(figm, use_container_width=True)
            if "mape" in results_df.columns:
                st.markdown("**MAPE Comparison** - Percentage error; easier to interpret across scales")
                fig_mape = px.bar(results_df, x="model", y="mape", title="MAPE (lower is better)")
                st.plotly_chart(fig_mape, use_container_width=True)
            if "medae" in results_df.columns:
                st.markdown("**Median AE** - Most robust to outliers; reliable error estimate")
                fig_medae = px.bar(results_df, x="model", y="medae", title="Median AE (robust)")
                st.plotly_chart(fig_medae, use_container_width=True)

# ---------------------------------------------------------------------------
# STEP 6: Advanced Tuning (Optional)
# ---------------------------------------------------------------------------
section_header("⚡ Step 6: Advanced Model Tuning", "Hyperparameter optimization for better performance")

with st.expander("🔧 Hyperparameter Search", expanded=False):
    st.markdown("""
    **When to use:**
    - After finding a good baseline model
    - When you need to squeeze out extra performance
    - More iterations = better optimization (but slower)
    
    **Trade-off**: More thorough search takes longer but may improve results by 1-5%
    """)
    adv_models = ["RandomForest", "ExtraTrees", "HistGradientBoosting", "GradientBoosting"]
    if HAS_XGB:
        adv_models.append("XGBoost")
    if HAS_CAT:
        adv_models.append("CatBoost")
    adv_choice = st.selectbox("Advanced model", adv_models)
    n_iter = st.slider("Randomized search iters", 5, 50, 10, 1)
    if st.button("Run advanced search"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y if current_task=="classification" else None, random_state=42)
        pre = get_preprocessor(X)
        if adv_choice == "RandomForest":
            base = RandomForestClassifier() if current_task=="classification" else RandomForestRegressor()
            param = {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 12]}
        elif adv_choice == "ExtraTrees":
            base = ExtraTreesClassifier() if current_task=="classification" else ExtraTreesRegressor()
            param = {"model__n_estimators": [300, 500], "model__max_depth": [None, 10, 16]}
        elif adv_choice == "HistGradientBoosting":
            base = HistGradientBoostingClassifier() if current_task=="classification" else HistGradientBoostingRegressor()
            param = {"model__learning_rate": [0.05, 0.1], "model__max_depth": [None, 6, 10]}
        elif adv_choice == "GradientBoosting":
            base = GradientBoostingClassifier() if current_task=="classification" else GradientBoostingRegressor()
            param = {"model__n_estimators": [200, 400], "model__learning_rate": [0.05, 0.1]}
        elif adv_choice == "XGBoost" and HAS_XGB:
            base = XGBClassifier(eval_metric="logloss") if current_task=="classification" else XGBRegressor()
            param = {"model__n_estimators": [200, 400], "model__max_depth": [3, 6], "model__learning_rate": [0.05, 0.1]}
        elif adv_choice == "CatBoost" and HAS_CAT:
            base = CatBoostClassifier(verbose=False) if current_task=="classification" else CatBoostRegressor(verbose=False)
            param = {"model__depth": [4, 6, 8], "model__learning_rate": [0.05, 0.1]}
        else:
            base = RandomForestClassifier() if current_task=="classification" else RandomForestRegressor()
            param = {}
        pipe = Pipeline(steps=[("prep", pre), ("model", base)])
        search = RandomizedSearchCV(pipe, param, n_iter=n_iter, cv=3, n_jobs=-1, random_state=42)
        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        y_proba = None
        if current_task=="classification" and hasattr(search, "predict_proba"):
            proba = search.predict_proba(X_test)
            if proba.ndim == 2 and proba.shape[1] == 2:
                y_proba = proba[:,1]
            else:
                y_proba = proba
        metrics = summarize_classification(y_test, y_pred, y_proba) if current_task=="classification" else summarize_regression(y_test, y_pred)
        st.success(f"Best params: {search.best_params_}")
        st.json(metrics)
        st.session_state.platform_models[f"adv_{adv_choice}"] = search.best_estimator_

# ---------------------------------------------------------------------------
# STEP 7: Model Explainability
# ---------------------------------------------------------------------------
section_header("🔬 Step 7: Model Explainability", "Understand which features drive predictions")

with st.expander("🎯 Feature Importance", expanded=False):
    st.markdown("""
    **Feature importance interpretation:**
    - **High importance**: Feature strongly influences predictions
    - **Zero/low importance**: Feature can likely be removed
    - **Permutation importance**: Shows impact when feature is shuffled (more reliable than built-in importance)
    - **SHAP values**: Gold standard; shows both magnitude and direction of impact
    
    **Use cases:**
    - Feature selection: Remove low-importance features
    - Domain validation: Check if important features make business sense
    - Model debugging: Unexpected important features may indicate data leakage
    """)
    if st.session_state.get("platform_models"):
        model_name = st.selectbox("Select trained model", list(st.session_state.platform_models.keys()))
        model = st.session_state.platform_models[model_name]
        try:
            import shap
            shap_df = X.sample(min(MAX_SHAP_ROWS, len(X)), random_state=42)
            explainer = shap.Explainer(model.predict, model[:-1].transform(shap_df) if hasattr(model, "__getitem__") else shap_df)
            shap_vals = explainer(shap_df)
            shap_fig = shap.plots.bar(shap_vals, show=False)
            st.pyplot(shap_fig)
        except Exception:
            st.info("SHAP not available; showing permutation importance instead")
            pi_df = X.sample(min(MAX_PI_ROWS, len(X)), random_state=42)
            pi_target = y.loc[pi_df.index]
            pi = permutation_importance(model, pi_df, pi_target, n_repeats=8, random_state=42, n_jobs=-1)
            imp_df = pd.DataFrame({"feature": X.columns, "importance": pi.importances_mean})
            fig = px.bar(imp_df.sort_values("importance"), x="importance", y="feature", orientation="h")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train a model first")

# ---------------------------------------------------------------------------
# STEP 8: Making Predictions
# ---------------------------------------------------------------------------
section_header("🚀 Step 8: Making Predictions", "Use your trained model on new data")

with st.expander("🎯 Single Prediction", expanded=False):
    st.markdown("""
    **How to use:**
    1. Select a trained model
    2. Enter feature values for one data point
    3. Click Predict to see the result
    
    **For classification**: You'll see the predicted class and confidence (probability)
    """)
    if st.session_state.get("platform_models"):
        model_name = st.selectbox("Model", list(st.session_state.platform_models.keys()), key="pred_model")
        model = st.session_state.platform_models[model_name]
        inputs = {}
        for col in X.columns:
            if col in X.select_dtypes(include=np.number).columns:
                inputs[col] = st.number_input(col, value=float(X[col].median()))
            else:
                inputs[col] = st.selectbox(col, options=sorted(X[col].dropna().unique()), index=0)
        if st.button("Predict"):
            row = pd.DataFrame([inputs])
            pred = model.predict(row)[0]
            proba = None
            if current_task == "classification" and hasattr(model, "predict_proba"):
                try:
                    proba_full = model.predict_proba(row)[0]
                    proba = proba_full[1] if len(proba_full) == 2 else None
                except Exception:
                    pass
            st.success(f"Prediction: {pred}")
            if proba is not None:
                st.info(f"Confidence: {proba:.3f}")
                thr = st.session_state.platform_thresholds.get(model_name)
                if thr is not None:
                    decision = "Positive" if proba >= thr else "Negative"
                    st.caption(f"Suggested threshold {thr:.2f} → decision: {decision}")
    else:
        st.info("Train a model first")

with st.expander("📦 Batch Predictions", expanded=False):
    st.markdown("""
    **Batch prediction workflow:**
    1. Prepare CSV file with same columns as training data (except target)
    2. Upload the file
    3. Select trained model
    4. Download predictions as CSV
    
    **Use cases**: Scoring new customers, forecasting multiple periods, production deployment
    """)
    if st.session_state.get("platform_models"):
        uploaded_pred = st.file_uploader("Upload batch CSV", type=["csv"], key="batch_pred")
        if uploaded_pred is not None:
            df_batch = pd.read_csv(uploaded_pred)
            model_name = st.selectbox("Model", list(st.session_state.platform_models.keys()), key="pred_model_batch")
            model = st.session_state.platform_models[model_name]
            # Improved feature validation before prediction
            trained_features = getattr(model, "feature_names_in_", None)
            if trained_features is None and hasattr(model, "meta"):
                trained_features = model.meta.get("features", None)
            if trained_features is not None:
                missing_cols = set(trained_features) - set(df_batch.columns)
                if missing_cols:
                    st.error(f"\u274c Missing columns in uploaded data: {missing_cols}")
                    st.info(f"Expected columns: {list(trained_features)}")
                    st.stop()
            preds = model.predict(df_batch)
            df_out = df_batch.copy()
            df_out["prediction"] = preds
            st.dataframe(df_out.head(), use_container_width=True)
            st.download_button("Download predictions", df_out.to_csv(index=False), file_name="predictions.csv", mime="text/csv")
    else:
        st.info("Train a model first")

# ---------------------------------------------------------------------------
# STEP 9: Final Model Comparison
# ---------------------------------------------------------------------------
section_header("📊 Step 9: Final Model Comparison", "Compare all trained models side-by-side")

st.markdown("""
**Choosing the best model:**
1. **Check multiple metrics** - Don't rely on just one
2. **Consider complexity** - Simpler models are easier to deploy and explain
3. **Validate assumptions** - Ensure model makes sense for your domain
4. **Test stability** - Run multiple times with different random states
5. **Production readiness** - Consider inference speed and resource requirements
""")

if st.session_state.get("platform_models"):
    entries = []
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y if current_task=="classification" else None, random_state=42)
    group_col = None
    if current_task == "classification":
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if cat_cols:
            group_col = st.selectbox("Optional fairness group column", ["(none)"] + cat_cols)
            if group_col == "(none)":
                group_col = None
    for name, model in st.session_state.platform_models.items():
        y_pred = model.predict(X_test)
        y_proba = None
        if current_task == "classification" and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
                if proba.ndim == 2 and proba.shape[1] == 2:
                    y_proba = proba[:,1]
                else:
                    y_proba = proba
            except Exception:
                y_proba = None
        metrics = summarize_classification(y_test, y_pred, y_proba) if current_task=="classification" else summarize_regression(y_test, y_pred)
        if current_task == "classification" and group_col:
            preds_series = pd.Series(y_pred, index=X_test.index)
            rates = preds_series.groupby(X_test[group_col]).mean()
            if len(rates) > 1:
                metrics["group_gap"] = float(rates.max() - rates.min())
                metrics["group_min"] = float(rates.min())
                metrics["group_max"] = float(rates.max())
        entries.append({"model": name, **metrics})
    cmp_df = pd.DataFrame(entries)
    st.dataframe(cmp_df, use_container_width=True)
    if current_task == "classification":
        fig = px.bar(cmp_df, x="model", y="f1", title="F1 by model")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = px.bar(cmp_df, x="model", y="rmse", title="RMSE by model (lower is better)")
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Train at least one model to compare.")

# ---------------------------------------------------------------------------
# Final Tips & Next Steps
# ---------------------------------------------------------------------------
st.markdown("---")
section_header("✨ Next Steps & Best Practices", "Take your models to production")

st.markdown("""
### 🎯 Production Deployment Checklist
- [ ] **Validate on holdout data** - Test on completely unseen data
- [ ] **Document model assumptions** - Note data requirements and limitations
- [ ] **Monitor performance** - Track metrics over time for drift
- [ ] **Set up retraining** - Schedule periodic model updates
- [ ] **Create fallback logic** - Handle edge cases and errors gracefully

### 📚 Learning Resources
- **Feature Engineering**: Try polynomial features, interactions, domain transforms
- **Cross-Validation**: Use k-fold CV for more robust evaluation
- **Ensemble Methods**: Combine multiple models for better predictions
- **Model Interpretability**: Deep dive into SHAP, LIME, PDP
- **MLOps**: Learn versioning, monitoring, and deployment best practices
""")

# ---------------------------------------------------------------------------
# AI Tutor / Suggestions
# ---------------------------------------------------------------------------
st.markdown("---")
section_header("🤖 AI Tutor", "Smart suggestions based on your workflow")

suggestions = []
if current_task == "classification":
    suggestions.append("💡 **Imbalanced classes?** Enable class weights or explore SMOTE techniques")
    suggestions.append("📊 **Metric choice**: Compare ROC-AUC and PR-AUC to balance precision/recall")
    suggestions.append("🎯 **Threshold tuning**: Check the best_threshold_f1 values in results")
else:
    suggestions.append("📈 **Residual patterns?** Try non-linear models like GradientBoosting")
    suggestions.append("🎯 **Reduce error**: Consider feature engineering or polynomial features")
    suggestions.append("📊 **Outliers present?** Median AE is your most reliable metric")

suggestions.append("🔍 **Feature importance**: Always validate that top features make domain sense")
suggestions.append("📝 **Experiment tracking**: Log runs with timestamps to track improvements")
suggestions.append("🚀 **Next level**: Try stacking/blending models for 1-3% performance gain")

for s in suggestions:
    st.markdown(f"- {s}")

st.caption(f"Session: {dt.datetime.utcnow().isoformat()} | Dataset size: {len(X)} rows | Task: {current_task}")

# Footer navigation
page_navigation("23")
