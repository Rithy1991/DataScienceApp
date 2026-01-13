from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import re

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.core.config import load_config
from src.core.ui import app_header, page_navigation, sidebar_dataset_status
from src.core.standardized_ui import (
    standard_section_header,
    beginner_tip,
    concept_explainer,
    model_explanation_panel,
    common_mistakes_panel,
    metric_card,
    before_after_comparison,
)

st.set_page_config(page_title="NLP: TF-IDF & Sentiment", layout="wide", initial_sidebar_state="expanded")

config = load_config()

# Page Header
app_header(
    config,
    page_title="NLP: TF-IDF & Sentiment",
    subtitle="Vectorize text with TF-IDF and train a simple sentiment classifier",
    icon="🗣️"
)

# Session dataset status in sidebar
raw_df = st.session_state.get("raw_df")
clean_df = st.session_state.get("clean_df")
sidebar_dataset_status(raw_df, clean_df)

# Step 1: Learn the Concept
standard_section_header("Step 1: Understand TF-IDF", "📘")
concept_explainer(
    title="What is TF-IDF?",
    explanation=(
        "TF-IDF (Term Frequency–Inverse Document Frequency) converts text into numeric features. "
        "Words that appear often in a document but rarely across all documents get higher weights."
    ),
    real_world_example=(
        "In reviews, words like 'great', 'love', 'terrible' can carry strong sentiment. TF-IDF emphasizes these informative words."
    ),
)
beginner_tip(
    "Tip: Keep preprocessing simple at first. TF-IDF with unigrams/bigrams and stopword removal works well as a baseline."
)

# Step 2: Load or Create a Dataset
standard_section_header("Step 2: Load or Select Your Dataset", "📂")
source = st.radio("Data source:", ["Use sample dataset", "Upload CSV"], horizontal=True)

df = None
if source == "Use sample dataset":
    df = pd.DataFrame(
        {
            "text": [
                "I absolutely love this product!",
                "This is the worst experience ever.",
                "Amazing quality and great support.",
                "Terrible build, broke in a week.",
                "Works as expected, very satisfied.",
                "Not worth the money, disappointing.",
                "Five stars! Highly recommend.",
                "Awful service, will not buy again.",
                "Pretty good overall.",
                "Horrible and frustrating to use.",
                "Great value for the price.",
                "Bad quality control.",
                "Superb performance and features.",
                "Extremely poor design.",
                "Happy with my purchase.",
                "Regret buying this.",
                "Delightful experience.",
                "Terrible customer support.",
                "Solid product.",
                "Disappointing results.",
            ],
            "label": [
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
                "positive",
                "negative",
            ],
        }
    )
    st.success("Loaded sample sentiment dataset (20 rows).")
    # Allow users to download the sample as CSV
    sample_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download sample sentiment CSV",
        data=sample_csv,
        file_name="sample_sentiment.csv",
        mime="text/csv",
        width="stretch",
    )
else:
    uploaded = st.file_uploader("Upload CSV with text and label columns", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded {df.shape[0]:,} rows × {df.shape[1]:,} columns.")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

if df is None:
    st.stop()

st.dataframe(df.head(10), width="stretch")
st.caption(f"{df.shape[0]} rows × {df.shape[1]} columns")

# Step 3: Select Columns
standard_section_header("Step 3: Select Text and Target Columns", "🎯")
text_col = st.selectbox("Text column", options=list(df.columns))
label_col = st.selectbox("Target (label) column", options=[c for c in df.columns if c != text_col])

if text_col is None or label_col is None:
    st.warning("Please select both text and target columns.")
    st.stop()

# Basic validation
labels = df[label_col].dropna().unique()
if len(labels) < 2:
    st.error("Target column must have at least two classes.")
    st.stop()

# Step 4: TF-IDF Vectorization Settings
standard_section_header("Step 4: Configure TF-IDF", "🔧")
col_v1, col_v2, col_v3 = st.columns(3)
with col_v1:
    use_bigrams = st.checkbox("Include bigrams (n-gram range (1,2))", value=True)
with col_v2:
    max_features = st.number_input("Max features", min_value=500, max_value=50000, value=5000, step=500)
with col_v3:
    remove_stopwords = st.checkbox("Remove English stopwords", value=True)

ngram_range = (1, 2) if use_bigrams else (1, 1)
stop_words = "english" if remove_stopwords else None

X_text = df[text_col].astype(str).fillna("")
y = df[label_col].astype(str).fillna("")

# Optional: Text Preprocessing
standard_section_header("Optional: Text Preprocessing", "🧼")
pp_col1, pp_col2, pp_col3, pp_col4 = st.columns(4)
with pp_col1:
    do_lowercase = st.checkbox("Lowercase text", value=True)
with pp_col2:
    remove_punct = st.checkbox("Remove punctuation", value=True)
with pp_col3:
    remove_numbers = st.checkbox("Remove numbers", value=False)
with pp_col4:
    collapse_ws = st.checkbox("Collapse whitespace", value=True)

def _preprocess_series(s: pd.Series) -> pd.Series:
    def _proc(t: str) -> str:
        if remove_punct:
            # Replace common punctuation with spaces; fallback for environments without Unicode classes
            t = re.sub(r"[^\w\s]", " ", t)
        if remove_numbers:
            t = re.sub(r"\d+", " ", t)
        if do_lowercase:
            t = t.lower()
        if collapse_ws:
            t = re.sub(r"\s+", " ", t).strip()
        return t
    return s.apply(_proc)

X_text_proc = _preprocess_series(X_text)

# Preview preprocessing results
standard_section_header("Preprocessing Preview", "🔎")
before_after_comparison(
    pd.DataFrame({"raw_text": X_text.head(5)}),
    pd.DataFrame({"processed_text": X_text_proc.head(5)}),
    before_title="Raw Text (sample)",
    after_title="Processed Text (sample)",
)

st.markdown(
    """
    - **Lowercase**: Convert all letters to lowercase to normalize tokens.
    - **Remove punctuation**: Strip symbols like ! ? . , which rarely add meaning for TF-IDF.
    - **Remove numbers**: Drop digits to avoid IDs or counts dominating features.
    - **Collapse whitespace**: Replace multiple spaces with a single space and trim ends.
    """
)

# Mini examples per toggle
standard_section_header("Mini Examples", "🧪")
example_text = st.text_input("Example sentence", value="Wow!!! I spent $123 on this—Amazing product!!  ")
ex_lower = example_text.lower()
ex_nopunct = re.sub(r"[^\w\s]", " ", example_text)
ex_nonumbers = re.sub(r"\d+", " ", example_text)
ex_ws = re.sub(r"\s+", " ", example_text).strip()
ex_combined = _preprocess_series(pd.Series([example_text]))[0]

st.dataframe(
    pd.DataFrame(
        {
            "Raw": [example_text],
            "Lowercase": [ex_lower],
            "No punctuation": [ex_nopunct],
            "No numbers": [ex_nonumbers],
            "Collapse whitespace": [ex_ws],
            "Combined (current toggles)": [ex_combined],
        }
    ),
    width="stretch",
)

# Save processed dataset to session
standard_section_header("Save Processed Dataset", "💾")
new_df = df.copy()
new_df["processed_text"] = X_text_proc
save_target = st.selectbox(
    "Save target in session",
    options=["nlp_df", "clean_df"],
    index=0,
    help="Choose where to store. 'clean_df' will replace the cleaned dataset used across pages.",
)
if st.button("Save to session", width="stretch"):
    st.session_state[save_target] = new_df
    st.success(f"Saved processed dataset to session as {save_target}.")

# Download processed CSV
processed_csv = new_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download processed CSV",
    data=processed_csv,
    file_name="processed_sentiment.csv",
    mime="text/csv",
    width="stretch",
)

# Preview TF-IDF transformation on a small sample
vectorizer_preview = TfidfVectorizer(
    ngram_range=ngram_range,
    stop_words=stop_words,
    max_features=min(1000, max_features),
    lowercase=False,  # respect the explicit lowercase toggle applied in preprocessing
)
X_preview = vectorizer_preview.fit_transform(X_text_proc.head(50))
preview_features = vectorizer_preview.get_feature_names_out().tolist()

before_after_comparison(
    pd.DataFrame({"sample_text": X_text.head(5)}),
    pd.DataFrame(X_preview.toarray(), columns=preview_features).head(5),
    before_title="Raw Text (sample)",
    after_title="TF-IDF Features (sample)",
)

# Step 5: Train/Test Split
standard_section_header("Step 5: Train/Test Split", "🧪")
col_s1, col_s2 = st.columns(2)
with col_s1:
    test_size = st.slider("Test size (fraction)", min_value=0.1, max_value=0.5, value=0.2)
with col_s2:
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

# Stratify when possible (each class >= 2)
class_counts = y.value_counts()
stratify_arg = y if class_counts.min() >= 2 else None
if stratify_arg is None:
    st.warning("Stratified split disabled: at least one class has < 2 samples.")

X_train_text, X_test_text, y_train, y_test = train_test_split(
    X_text_proc, y, test_size=test_size, random_state=random_state, stratify=stratify_arg
)

# Fit full TF-IDF on training text only (avoid leakage)
vectorizer = TfidfVectorizer(
    ngram_range=ngram_range,
    stop_words=stop_words,
    max_features=max_features,
    lowercase=False,  # we already handled casing in preprocessing when selected
)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Step 6: Choose and Train Model
standard_section_header("Step 6: Choose and Train Model", "⚙️")
model_name = st.selectbox("Model", options=["Logistic Regression", "Linear SVM (LinearSVC)"])

params_col1, params_col2 = st.columns(2)
model = None
proba_supported = False

if model_name == "Logistic Regression":
    with params_col1:
        C = st.slider("Regularization (C)", min_value=0.01, max_value=10.0, value=1.0)
    with params_col2:
        max_iter = st.number_input("Max Iterations", min_value=100, value=500, step=50)
    model = LogisticRegression(C=C, max_iter=max_iter)
    proba_supported = True
else:
    with params_col1:
        C = st.slider("Regularization (C)", min_value=0.01, max_value=10.0, value=1.0)
    model = LinearSVC(C=C)
    proba_supported = False

train_btn = st.button("Train Sentiment Model", type="primary")

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
    standard_section_header("Step 7: Confusion Matrix & Metrics", "📈")
    class_labels = [str(c) for c in np.unique(y)]
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    fig = go.Figure(data=go.Heatmap(z=cm, x=class_labels, y=class_labels, colorscale="Blues", showscale=True))
    fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
    st.plotly_chart(fig, width="stretch")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        metric_card("Accuracy", f"{acc:.3f}", "Overall correctness of predictions.")
    with mcol2:
        metric_card("Precision (Macro)", f"{prec:.3f}", "Of predicted positives, how many were correct.")
    with mcol3:
        metric_card("Recall (Macro)", f"{rec:.3f}", "Of actual positives, how many were found.")
    with mcol4:
        metric_card("F1-Score (Macro)", f"{f1:.3f}", "Balance between precision and recall.")

    # Feature importance: top weighted terms per class (LogReg only)
    standard_section_header("Step 8: Top Informative Terms", "🔍")
    try:
        feature_names = vectorizer.get_feature_names_out()
        if hasattr(model, "coef_"):
            coefs = model.coef_
            # handle binary vs multiclass
            if coefs.ndim == 1:
                coefs = coefs.reshape(1, -1)
            classes = model.classes_ if hasattr(model, "classes_") else np.array(sorted(pd.Series(y_train).unique()))
            top_k = st.slider("Top terms per class", min_value=5, max_value=30, value=10)

            for i, cls in enumerate(classes):
                weights = coefs[i]
                idx_top = np.argsort(weights)[-top_k:][::-1]
                terms = [feature_names[j] for j in idx_top]
                values = [float(weights[j]) for j in idx_top]
                st.markdown(f"**Class: {cls}**")
                st.dataframe(pd.DataFrame({"term": terms, "weight": values}), width="stretch")
        else:
            st.caption("Term importance not available for this model.")
    except Exception as e:
        st.caption(f"Could not compute term importance: {e}")

    # Quick prediction demo
    standard_section_header("Step 9: Try a Custom Prediction", "📝")
    user_text = st.text_area("Enter text to classify", value="I love this!")
    if st.button("Predict Sentiment", width="stretch"):
        try:
            X_user = vectorizer.transform([user_text])
            pred = model.predict(X_user)[0]
            st.success(f"Predicted: {pred}")
            if proba_supported and hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_user)[0]
                st.caption("Class probabilities:")
                st.write({cls: float(p) for cls, p in zip(model.classes_, probs)})
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    # Educational model panel
    explanation = {
        "Logistic Regression": {
            "how": "Learns weighted combinations of TF-IDF features to separate sentiment classes.",
            "when": "Use as a baseline for fast, interpretable sentiment models.",
            "pros": ["Fast", "Interpretable", "Outputs probabilities"],
            "cons": ["May struggle with non-linear cues", "Needs good preprocessing"],
        },
        "Linear SVM (LinearSVC)": {
            "how": "Finds a separating hyperplane with maximum margin using TF-IDF features.",
            "when": "Use when you want strong performance without probabilities.",
            "pros": ["High accuracy", "Robust to high-dimensional text"],
            "cons": ["No probabilities", "Less interpretable"],
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
        "Not cleaning text": "Remove stopwords and consider lowercasing; noisy tokens hurt performance.",
        "Class imbalance": "Ensure enough samples per class or use stratified split.",
        "Feature explosion": "Too many n-grams can overfit; cap max_features.",
        "Train/test leakage": "Fit TF-IDF on training data only, then transform test.",
        "Ignoring baseline": "Start with TF-IDF + Logistic Regression before complex models.",
    })

# Page navigation (registered as page 16 in UI map)
page_navigation("16")
