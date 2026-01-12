import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from src.core.ui import page_navigation, sidebar_dataset_status
from src.core.standardized_ui import (
    standard_page_header,
    standard_section_header,
    beginner_tip,
    concept_explainer,
    before_after_comparison,
    model_explanation_panel,
    common_mistakes_panel,
    metric_card,
)


st.set_page_config(layout="wide")

# Page Header
standard_page_header(
    title="Classification Learning",
    subtitle="Learn to predict categories using simple models with step-by-step guidance.",
    icon="🧑‍🎓",
)

# Session dataset status in sidebar
df = st.session_state.get("raw_df")
clean_df = st.session_state.get("clean_df")
sidebar_dataset_status(df, clean_df)


def load_sample_iris() -> pd.DataFrame:
    import seaborn as sns
    return sns.load_dataset("iris")


standard_section_header("Step 1: Understand Classification", "📘")
concept_explainer(
    title="What is Classification?",
    explanation="Classification predicts categories like 'spam' vs 'not spam' or 'will churn' vs 'will stay'.",
    real_world_example="Email spam filter — the model learns patterns that distinguish spam from non-spam emails.",
)

beginner_tip("Tip: Start with clean data and a clear target column (the thing you want to predict).")

# Dataset selection / sample fallback
standard_section_header("Step 2: Load or Select Your Dataset", "📂")
if df is None:
    st.info("No dataset found in session. Load a sample dataset to get started.")
    col_s1, col_s2 = st.columns([1, 1])
    with col_s1:
        if st.button("Load Iris Sample Dataset"):
            st.session_state.raw_df = load_sample_iris()
            df = st.session_state.raw_df
            st.success("Loaded Iris sample dataset.")
    with col_s2:
        st.markdown("**Or upload a dataset (CSV/Excel)**")
        uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls"], accept_multiple_files=False)
        if uploaded is not None:
            try:
                if uploaded.name.lower().endswith((".xlsx", ".xls")):
                    df_up = pd.read_excel(uploaded)
                else:
                    df_up = pd.read_csv(uploaded)
                st.session_state.raw_df = df_up
                df = df_up
                st.success(f"Uploaded {df_up.shape[0]:,} rows × {df_up.shape[1]:,} cols")
            except Exception as e:
                st.error(f"Failed to read file: {e}")

if df is None:
    st.stop()

st.dataframe(df.head(10), use_container_width=True)
st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

# Target selection
standard_section_header("Step 3: Choose Target (What to Predict)", "🎯")
target_col = st.selectbox(
    "Select the target column (category to predict)",
    options=list(df.columns),
)

if target_col is None:
    st.warning("Please select a target column to continue.")
    st.stop()

# Validate target type (must be categorical/discrete)
unique_target_values = df[target_col].dropna().unique()
is_target_categorical = (
    df[target_col].dtype == "object" or df[target_col].dtype.name == "category" or len(unique_target_values) <= 50
)

if len(unique_target_values) < 2:
    st.error("Target column has only one class. Please choose a target with at least two classes for classification.")
    st.stop()
if not is_target_categorical:
    st.error("Selected target appears continuous (many unique numeric values). For classification, choose a categorical target.")
    st.stop()

# Class balance check and auto-merge rare classes
standard_section_header("Step 3b: Class Distribution & Balance Check", "📊")
temp_y = df[target_col].dropna()
class_counts = pd.Series(temp_y).value_counts().sort_values(ascending=False)
class_pct = (class_counts / len(temp_y) * 100).round(2)
dist_df = pd.DataFrame({
    "Class": class_counts.index,
    "Count": class_counts.values,
    "Percentage": class_pct.values,
})
st.dataframe(dist_df, use_container_width=True)

# Auto-detect and offer to merge rare classes (< 2 samples)
rare_classes = class_counts[class_counts < 2].index.tolist()
if rare_classes:
    st.warning(f"⚠️ Found class(es) with only 1 sample: {rare_classes}. Stratified split will be disabled.")
    if st.checkbox(f"Merge rare classes into 'Other'?", value=True, key="merge_rare_classes"):
        df[target_col] = df[target_col].astype(str)
        for rare_class in rare_classes:
            df[target_col] = df[target_col].replace(str(rare_class), "Other")
        st.success("✅ Merged rare classes. Updated distribution:")
        class_counts_new = pd.Series(df[target_col]).value_counts()
        class_pct_new = (class_counts_new / len(df[target_col]) * 100).round(2)
        dist_df_new = pd.DataFrame({
            "Class": class_counts_new.index,
            "Count": class_counts_new.values,
            "Percentage": class_pct_new.values,
        }).sort_values("Count", ascending=False)
        st.dataframe(dist_df_new, use_container_width=True)

# Feature selection
feature_cols = [c for c in df.columns if c != target_col]
selected_features = st.multiselect(
    "Select feature columns (inputs)",
    options=feature_cols,
    default=feature_cols,
)

if not selected_features:
    st.warning("Select at least one feature column.")
    st.stop()

X = df[selected_features].copy()
y = df[target_col].copy()

# Drop rows with NaN values in features or target
valid_idx = (~X.isna().any(axis=1)) & (~y.isna())
X = X[valid_idx].reset_index(drop=True)
y = y[valid_idx].reset_index(drop=True)

if len(X) == 0:
    st.error("No valid rows after removing NaN values. Please check your data.")
    st.stop()

if len(X) < 10:
    st.warning(f"⚠️ Only {len(X)} valid rows after NaN removal. Consider reviewing your data.")

# Encode categorical features automatically
def _auto_encode_features(X_df: pd.DataFrame) -> pd.DataFrame:
    X_enc = X_df.copy()
    for col in X_enc.columns:
        if X_enc[col].dtype == "object" or str(X_enc[col].dtype).startswith("category"):
            X_enc[col] = pd.factorize(X_enc[col].astype(str))[0]
        elif X_enc[col].dtype == "bool":
            X_enc[col] = X_enc[col].astype(int)
    return X_enc

# Encode target labels to integers if needed
target_encoder = None
if y.dtype == "object" or str(y.dtype).startswith("category"):
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y.astype(str))

X_encoded = _auto_encode_features(X)

before_after_comparison(df[selected_features], X_encoded, before_title="Raw Features", after_title="Encoded Features")

# Train/validation split controls
standard_section_header("Step 4: Train/Test Split", "🧪")
st.markdown("""
**📚 Why Split Data?**

You never test a student on the same questions they studied from! Same principle applies to ML:

**Training Set** (📚 Learn):
- Model learns patterns from this data
- Typically 70-80% of total data
- Larger = more learning, but less testing data

**Test Set** (🎯 Evaluate):
- Model is evaluated on this UNSEEN data
- Typically 20-30% of total data
- Simulates real-world performance
- NEVER shown to model during training

**Why this matters:**
- **Prevent overfitting**: If we test on training data, we'd get falsely high scores
- **Honest evaluation**: Test set shows how model performs on new data
- **Detect problems**: Large gap between train/test performance indicates overfitting

**Stratified Split:**
- Maintains class proportions in both sets
- Example: If 30% of data is "Spam", both train/test will have ~30% "Spam"
- Important for imbalanced datasets
- Automatically used when possible

**Random State:**
- Controls random split - same number = same split every time
- Use for reproducibility (comparing experiments)
- Change to test model stability across different splits
""")
col_t1, col_t2 = st.columns(2)
with col_t1:
    test_size = st.slider("Test size (fraction for validation)", min_value=0.1, max_value=0.5, value=0.2)
with col_t2:
    random_state = st.number_input("Random state (reproducibility)", min_value=0, value=42, step=1)
# Safe stratified split: fallback if any class has < 2 samples
class_counts = pd.Series(y).value_counts()
use_stratify = class_counts.min() >= 2
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=(y if use_stratify else None)
    )
    if not use_stratify:
        st.warning("Stratified split disabled because at least one class has < 2 samples. Consider collecting more data or merging rare classes.")
except Exception as e:
    st.error(f"Train/Test split failed: {e}")
    st.stop()

# Model selection
standard_section_header("Step 5: Choose and Train Model", "⚙️")
st.markdown("""
**🤖 Understanding Classification Models:**

Each algorithm has different strengths - here's how to choose:

🔹 **Logistic Regression** (Simple & Fast)
- **How it works**: Finds linear decision boundary between classes
- **Best for**: Binary classification, when you need probability scores
- **Strengths**: Fast, interpretable, gives confidence scores
- **Weaknesses**: Only works well if classes are linearly separable
- **Use when**: You need a quick baseline or want to understand feature impacts
- **Parameters**: C (regularization) - lower = simpler model, higher = more complex

🌳 **Decision Tree** (Visual & Interpretable)
- **How it works**: Creates a flowchart of yes/no questions
- **Best for**: When you need to explain decisions to non-technical stakeholders
- **Strengths**: Easy to visualize, handles non-linear patterns, no scaling needed
- **Weaknesses**: Can overfit easily, unstable (small data changes = different tree)
- **Use when**: You need interpretability and don't mind tuning
- **Parameters**: Max depth (tree size), min samples split (when to stop splitting)

🌲 **Random Forest** (Powerful & Robust)
- **How it works**: Trains many decision trees and averages their predictions
- **Best for**: Most real-world problems - great general-purpose algorithm
- **Strengths**: High accuracy, handles complex patterns, reduces overfitting
- **Weaknesses**: Slower than single tree, less interpretable, needs more memory
- **Use when**: You want best accuracy without much tuning
- **Parameters**: n_estimators (number of trees - more is better but slower)

🔵 **Support Vector Machine (SVM)** (Sophisticated)
- **How it works**: Finds the widest margin between classes (maximum separation)
- **Best for**: High-dimensional data, when you have clear margins between classes
- **Strengths**: Powerful with right kernel, works well with clear decision boundaries
- **Weaknesses**: Slow on large datasets, requires feature scaling, hard to interpret
- **Use when**: You have complex boundaries and can afford longer training time
- **Parameters**: C (regularization), kernel (linear for simple, rbf for complex)

**🎯 Model Selection Guide:**

| Need | Recommended Model |
|------|------------------|
| Speed & Simplicity | Logistic Regression |
| Interpretability | Decision Tree |
| Best Accuracy | Random Forest |
| Complex Boundaries | SVM (RBF kernel) |
| Probability Scores | Logistic Regression |
| No Scaling Required | Decision Tree / Random Forest |

**💡 Pro Tips:**
- Start with Random Forest - it's a great default
- Try Logistic Regression first for baseline speed
- Use Decision Tree when stakeholders need visual explanations
- Reserve SVM for when simpler methods fail
""")

model_name = st.selectbox(
    "Model",
    options=["Logistic Regression", "Decision Tree", "Random Forest", "Support Vector Machine (SVM)"],
    index=0,
)

params_col1, params_col2, params_col3 = st.columns(3)
model = None

if model_name == "Logistic Regression":
    with params_col1:
        C = st.slider("Regularization (C)", min_value=0.01, max_value=10.0, value=1.0)
    with params_col2:
        max_iter = st.number_input("Max Iterations", min_value=100, value=500, step=50)
    model = LogisticRegression(C=C, max_iter=max_iter)

elif model_name == "Decision Tree":
    with params_col1:
        max_depth = st.slider("Max Depth", min_value=1, max_value=50, value=5)
    with params_col2:
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2)
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)

elif model_name == "Random Forest":
    with params_col1:
        n_estimators = st.slider("Number of Trees", min_value=10, max_value=200, value=100, step=10)
    with params_col2:
        max_depth_slider = st.slider("Max Depth", min_value=1, max_value=50, value=15)
    with params_col3:
        min_samples_split = st.slider("Min Samples Split", min_value=2, max_value=20, value=2)
    # Don't set to None, use a reasonable default
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth_slider, min_samples_split=min_samples_split, random_state=random_state)

else:  # Support Vector Machine (SVM)
    with params_col1:
        kernel = st.selectbox("Kernel Type", options=["linear", "rbf", "poly"], index=1)
    with params_col2:
        C = st.slider("Regularization (C)", min_value=0.01, max_value=10.0, value=1.0)
    with params_col3:
        gamma = st.selectbox("Gamma (for rbf/poly)", options=["scale", "auto"], index=0)
    model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=random_state, probability=True)

train_btn = st.button("Train Model", type="primary")

if train_btn:
    with st.spinner("Training model..."):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))

    # Confusion matrix visualization
    standard_section_header("Step 6: Understanding Results - Confusion Matrix & Metrics", "📈")
    st.markdown("""
    **📚 Understanding the Confusion Matrix:**
    
    The confusion matrix shows where your model gets confused:
    
    |                  | Predicted Positive | Predicted Negative |
    |------------------|-------------------|-------------------|
    | **Actually Positive** | ✅ True Positive (TP) | ❌ False Negative (FN) |
    | **Actually Negative** | ❌ False Positive (FP) | ✅ True Negative (TN) |
    
    **How to read it:**
    - **Diagonal (top-left to bottom-right)**: Correct predictions ✅
    - **Off-diagonal**: Mistakes ❌
    - **Darker colors**: More instances
    
    **What to look for:**
    - **Strong diagonal**: Good! Model is mostly correct
    - **Scattered errors**: Model is confused about classes
    - **Systematic errors**: Model consistently confuses specific classes
    
    **Example interpretation:**
    - If predicting "Spam" and confusion matrix shows high FN → model misses spam emails
    - High FP → model flags legitimate emails as spam
    """)
    
    class_labels = (
        list(target_encoder.classes_) if target_encoder is not None else [str(c) for c in np.unique(y)]
    )
    cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(class_labels)))
    fig = go.Figure(data=go.Heatmap(z=cm, x=class_labels, y=class_labels, colorscale="Blues", showscale=True, text=cm, texttemplate="%{text}", textfont={"size":14}))
    fig.update_layout(title="Confusion Matrix - Darker = More Predictions", xaxis_title="Predicted Class", yaxis_title="Actual Class", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    ---
    **📊 Understanding Classification Metrics:**
    
    Different metrics tell different stories about your model:
    
    🎯 **Accuracy** = (TP + TN) / Total
    - **What it means**: Overall % of correct predictions
    - **When to use**: Balanced datasets with equal class importance
    - **Warning**: Can be misleading with imbalanced data (99% accuracy predicting "not fraud" if only 1% is fraud)
    - **Good score**: > 0.9 excellent, > 0.8 good, > 0.7 acceptable
    
    🎯 **Precision** = TP / (TP + FP)
    - **What it means**: Of predictions marked as positive, how many were actually positive?
    - **The question**: "When the model says YES, is it usually right?"
    - **When to prioritize**: When false positives are costly (spam filter blocking important email)
    - **Trade-off**: High precision often means lower recall
    - **Good score**: Depends on domain; > 0.8 generally good
    
    🎯 **Recall (Sensitivity)** = TP / (TP + FN)
    - **What it means**: Of all actual positives, how many did we catch?
    - **The question**: "Are we finding all the positive cases?"
    - **When to prioritize**: When false negatives are costly (missing cancer diagnosis)
    - **Trade-off**: High recall often means lower precision
    - **Good score**: Depends on domain; > 0.8 generally good
    
    🎯 **F1-Score** = 2 × (Precision × Recall) / (Precision + Recall)
    - **What it means**: Harmonic mean of precision and recall
    - **When to use**: When you need balance between precision and recall
    - **Why it's useful**: Single metric that considers both false positives and false negatives
    - **Good score**: > 0.8 excellent, > 0.7 good, > 0.6 acceptable
    
    **💡 Which metric to prioritize?**
    
    | Scenario | Prioritize | Why |
    |----------|-----------|-----|
    | Spam Detection | Precision | Blocking important emails (FP) is worse than letting spam through |
    | Cancer Diagnosis | Recall | Missing cancer (FN) is worse than false alarm |
    | Fraud Detection | F1 or Recall | Need to catch fraud (high recall) but not overwhelm reviewers (reasonable precision) |
    | Balanced Classes | Accuracy or F1 | All errors equally costly |
    | Imbalanced Classes | F1, Precision, or Recall | Accuracy is misleading |
    
    **🎭 Understanding "Macro" Average:**
    - Calculates metric for each class separately, then averages
    - Treats all classes equally important (good for imbalanced data)
    - Alternative: "weighted" (weights by class size) or "micro" (global calculation)
    """)
    
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.metric("Accuracy", f"{acc:.3f}", help="Overall correctness - use for balanced classes")
        if acc >= 0.9:
            st.caption("✅ Excellent")
        elif acc >= 0.8:
            st.caption("✅ Good")
        elif acc >= 0.7:
            st.caption("⚠️ Acceptable")
        else:
            st.caption("❌ Needs improvement")
    with mcol2:
        st.metric("Precision", f"{prec:.3f}", help="Of predicted positives, how many were correct")
        st.caption("High = Few false alarms")
    with mcol3:
        st.metric("Recall", f"{rec:.3f}", help="Of actual positives, how many were found")
        st.caption("High = Few missed cases")
    with mcol4:
        st.metric("F1-Score", f"{f1:.3f}", help="Balance of precision & recall")
        if f1 >= 0.8:
            st.caption("✅ Excellent balance")
        elif f1 >= 0.7:
            st.caption("✅ Good balance")
        elif f1 >= 0.6:
            st.caption("⚠️ Acceptable")
        else:
            st.caption("❌ Needs improvement")

    # Model explanation panel
    explanation = {
        "Logistic Regression": {
            "how": "Great for simple, binary choices (Yes/No). Despite the name, it's a classifier, not a regressor.",
            "when": "Use as a baseline for linear separable problems or when you need fast, interpretable results.",
            "pros": ["Fast", "Interpretable", "Works well on many datasets", "Gives probability scores"],
            "cons": ["May struggle with non-linear boundaries", "Requires feature scaling for best results"],
        },
        "Decision Tree": {
            "how": "Uses a flowchart-like structure of questions (e.g., 'Does the email contain an attachment?' → 'Yes' → 'Is it from a known contact?').",
            "when": "Use when non-linear relationships exist and you want to explain your model like a flowchart.",
            "pros": ["Interpretable", "Handles non-linearity", "No need to scale features", "Easy to explain"],
            "cons": ["Can overfit", "May be unstable without pruning", "Tends to be biased toward dominant classes"],
        },
        "Random Forest": {
            "how": "A 'forest' of many decision trees that vote on the final answer, making it much more accurate and stable.",
            "when": "Use when you want better accuracy than a single tree and can afford slightly longer training time.",
            "pros": ["High accuracy", "Handles non-linearity well", "Robust to outliers", "Reduces overfitting", "Gives feature importance"],
            "cons": ["Less interpretable than single tree", "Slower than tree or logistic regression", "Needs tuning for best results"],
        },
        "Support Vector Machine (SVM)": {
            "how": "Finds the widest possible 'road' or gap between two classes to ensure they stay separated.",
            "when": "Use when you have complex, non-linear decision boundaries and want a powerful classifier.",
            "pros": ["High accuracy on complex data", "Works well with non-linear kernels", "Memory efficient", "Versatile"],
            "cons": ["Requires feature scaling", "Slower to train on large datasets", "Hard to interpret (black box)", "Need to tune kernel and C carefully"],
        },
    }
    info = explanation.get(model_name, explanation["Logistic Regression"])  # fallback
    model_explanation_panel(
        model_name,
        how_it_works=info["how"],
        when_to_use=info["when"],
        pros=info["pros"],
        cons=info["cons"],
    )

    common_mistakes_panel({
        "Using test data to tune model parameters (data leakage)": "Split your data into train/validation/test and only evaluate final performance on test.",
        "Forgetting to encode categorical features": "Apply label/factor encoding before training; one-hot for tree-based models is often best.",
        "Not scaling features when using KNN": "Scale inputs (e.g., StandardScaler) so distances are meaningful across features.",
        "Ignoring class imbalance": "Use F1-score/Recall and consider class weights or resampling techniques.",
    })

    # Simple summary
    st.success("Model training complete.")
    st.markdown(
        f"**Summary:** The {model_name} achieved accuracy of {acc:.3f} with macro F1 of {f1:.3f}."
    )

    # ROC Curves and Macro AUC (if probabilities available)
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X_test))
            y_bin = np.asarray(label_binarize(y_test, classes=np.arange(len(class_labels))))
            auc_macro = roc_auc_score(y_bin, proba, average="macro", multi_class="ovr")

            standard_section_header("ROC Curves & Macro AUC", "📉")
            fig_roc = go.Figure()
            for i, label in enumerate(class_labels):
                fpr, tpr, _ = roc_curve(y_bin[:, i], proba[:, i])
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Class {label}"))
            fig_roc.add_trace(
                go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random", line=dict(dash="dash", color="#999"))
            )
            fig_roc.update_layout(title=f"ROC Curves (Macro AUC = {auc_macro:.3f})", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(fig_roc, use_container_width=True)

            # Show AUC card
            metric_card("Macro AUC", f"{auc_macro:.3f}", "Area under ROC across classes (one-vs-rest). Higher is better.")
        except Exception as e:
            st.caption(f"ROC/AUC not available: {e}")

    # Explainability: Permutation Importance (works for most models)
    try:
        standard_section_header("Feature Importance (Permutation)", "🧪")
        scoring = "f1_macro"
        result = permutation_importance(model, X_test, y_test, scoring=scoring, n_repeats=5, random_state=random_state)
        importances = pd.DataFrame({
            "Feature": X_encoded.columns,
            "Importance": result.importances_mean,
        }).sort_values("Importance", ascending=False)
        st.dataframe(importances.head(10), use_container_width=True)
        st.caption("Higher importance means the feature contributes more to model performance. Computed via shuffling feature values and measuring drop in F1-macro.")
    except Exception as e:
        st.caption(f"Feature importance unavailable: {e}")

    # Advanced Explainability: SHAP (if installed)
    if st.session_state.get("ui_mode") == "Advanced":
        try:
            import shap
            standard_section_header("Explainability (SHAP)", "🔍")
            sample_X = pd.DataFrame(X_test, columns=X_encoded.columns)
            # Limit to 200 samples for speed
            if len(sample_X) > 200:
                sample_X = sample_X.sample(200, random_state=random_state)
            explainer = shap.Explainer(model, pd.DataFrame(X_train, columns=X_encoded.columns))
            explanation = explainer(sample_X)
            vals = explanation.values
            # Aggregate for multiclass if needed
            if vals.ndim == 3:
                vals = vals.mean(axis=1)
            mean_abs = np.mean(np.abs(vals), axis=0)
            shap_df = pd.DataFrame({
                "Feature": X_encoded.columns,
                "Mean |SHAP|": mean_abs,
            }).sort_values("Mean |SHAP|", ascending=False)
            st.dataframe(shap_df.head(10), use_container_width=True)
            st.caption("Mean absolute SHAP values indicate average feature impact on predictions. Requires 'shap' package.")
        except Exception as e:
            st.caption(f"SHAP explainability unavailable: {e}. Tip: pip install shap")

# Navigation
page_navigation("5")
