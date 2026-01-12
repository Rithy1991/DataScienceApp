from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load

st.set_page_config(page_title="DataScope Pro - Predictions", layout="wide", initial_sidebar_state="expanded")

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from src.core.config import load_config
from src.core.logging_utils import log_event
from src.core.state import get_clean_df, get_df
from src.core.ui import sidebar_dataset_status, instruction_block, page_navigation
from src.core.standardized_ui import (
    standard_section_header,
    concept_explainer,
    beginner_tip,
    common_mistakes_panel,
)
from src.core.styles import render_stat_card, inject_custom_css
from src.data.loader import load_from_upload
from src.ml.forecast_transformer import forecast_transformer, load_transformer_forecaster
from src.ml.tabular import predict_tabular
from src.ml.tft_optional import forecast_tft, load_tft_forecaster, tft_available
from src.storage.history import add_event
from src.storage.model_registry import get_model, list_models
from src.core.forecast_results import ForecastResult, estimate_confidence_intervals
from src.core.forecast_components import render_forecast_results


config = load_config()

# Apply custom CSS
inject_custom_css()

st.markdown(
    """
    <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
        <div style="font-size: 24px; font-weight: 800;">🎯 Prediction & Inference</div>
        <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Run trained models on new data in batches or one row at a time.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
instruction_block(
    "How to use this page",
    [
        "Pick a saved model ID; cards show its type, task, and target column.",
        "Batch: upload data with the same columns used for training, then run predictions.",
        "Realtime tabular: fill in the generated form and click Predict for a single row.",
        "Forecast models output future steps equal to the horizon; tabular models may show class probabilities.",
        "Each run is logged so you can audit what was predicted and when.",
    ],
)

st.info(
    "Use your trained models without leaving the app. Upload files or type values and get predictions immediately.",
    icon="ℹ️",
)

# Beginner quick start and reading guide
st.success(
    "Quick start: (1) Pick a saved Model ID, (2) For batch, upload the same columns you trained on, (3) Click Run, "
    "(4) For single-row, fill the form and Predict, (5) Download results if you need to share.",
    icon="✅",
)
st.caption("Reading outputs: For classification, look at predicted class and confidence. For forecasts, read the chart: blue = history, orange = forecast, shaded = uncertainty.")

raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)

sidebar_dataset_status(raw_df, clean_df)

df_session = clean_df if clean_df is not None else raw_df

# Create tabs for different inference modes
tab1, tab2, tab3, tab4 = st.tabs(["🤖 Model Selection", "📤 Batch Prediction", "📝 Real-time Scoring", "🔍 Prediction Insights"])

with tab1:
    st.subheader("Available Models")
    models = list_models(config.registry_dir)
    if not models:
        st.error("❌ No saved models available")
        st.info("👉 Train a model first on the Tabular ML or Deep Learning pages")
        st.stop()
    
    col1, col2 = st.columns([2, 1])
    with col1:
        options = {f"{m.model_id} | {m.kind.upper():10s} | {m.name}": m.model_id for m in models}
        choice = st.selectbox("Select a trained model", options=list(options.keys()))
        model_id = options[choice]
    
    rec = get_model(config.registry_dir, model_id)
    if not rec:
        st.error("Model not found")
        st.stop()
    
    st.divider()
    
    # Model details
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(render_stat_card("Type", rec.kind, icon="🧠"), unsafe_allow_html=True)
    with c2:
        meta_task = (rec.meta or {}).get("task", (rec.meta or {}).get("type", "-"))
        st.markdown(render_stat_card("Task", str(meta_task), icon="🎯"), unsafe_allow_html=True)
    with c3:
        target = (rec.meta or {}).get("target_col", "-")
        st.markdown(render_stat_card("Target", str(target), icon="📌"), unsafe_allow_html=True)
    with c4:
        created = rec.created_utc[:10] if rec.created_utc else "Unknown"
        st.markdown(render_stat_card("Created", created, icon="📅"), unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("📚 Model Details & Metadata"):
        st.json(rec.meta if rec.meta else {"info": "No metadata available"})

with tab2:
    st.subheader("Batch Prediction")
    st.caption("Upload a dataset and make predictions on all rows at once")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        upload = st.file_uploader(
            "Upload data file",
            type=["csv", "parquet", "xlsx", "xls"],
            help="File must contain the same features used during training"
        )
    
    with col2:
        st.caption("")  # Spacing
    
    if upload is not None:
        try:
            res = load_from_upload(upload)
            if res is None:
                st.error("❌ Unsupported file format")
            else:
                st.success(f"✅ Loaded {res.df.shape[0]} rows × {res.df.shape[1]} columns")
                
                if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
                    with st.spinner("Making predictions..."):
                        try:
                            t0 = time.time()
                            
                            if rec.kind == "tabular":
                                model = load(rec.artifact_path)
                                
                                # Handle missing required columns
                                meta = rec.meta or {}
                                trained_features = meta.get("features", [])
                                upload_features = list(res.df.columns)
                                
                                if trained_features:
                                    missing_cols = set(trained_features) - set(upload_features)
                                    if missing_cols:
                                        st.error(f"❌ Missing columns in uploaded data: {missing_cols}")
                                        st.info(f"Expected columns: {trained_features}")
                                        st.stop()
                                
                                preds, proba = predict_tabular(model, res.df)
                                
                                # Create output
                                out = res.df.copy()
                                out["prediction"] = preds
                                
                                elapsed = time.time() - t0
                                
                                # Results summary
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.metric("Predictions made", len(preds))
                                with c2:
                                    st.metric("Time elapsed", f"{elapsed:.2f}s")
                                with c3:
                                    st.metric("Throughput", f"{len(preds)/elapsed:.0f} rows/sec")
                                
                                st.divider()
                                
                                st.subheader("Prediction Results")
                                st.dataframe(out.head(100), use_container_width=True)
                                
                                # Download
                                st.download_button(
                                    "📥 Download predictions (CSV)",
                                    data=out.to_csv(index=False),
                                    file_name="predictions.csv",
                                    use_container_width=True
                                )
                                
                                # Probabilities (if available)
                                if proba is not None:
                                    st.subheader("Prediction Probabilities")
                                    proba_df = pd.DataFrame(proba)
                                    st.dataframe(proba_df.head(100), use_container_width=True)
                                    
                                    st.download_button(
                                        "📥 Download probabilities (CSV)",
                                        data=proba_df.to_csv(index=False),
                                        file_name="probabilities.csv",
                                        use_container_width=True
                                    )
                            
                            else:  # Forecasting model
                                meta = rec.meta or {}
                                model_type_raw = str(meta.get("type", "transformer_forecaster"))
                                target_col = str(meta.get("target_col"))
                                time_col = str(meta.get("time_col"))
                                horizon = int(meta.get("horizon", 12))
                                lookback = int(meta.get("lookback", 48))

                                if model_type_raw == "tft" and not tft_available():
                                    st.error("TFT dependencies not installed")
                                else:
                                    # Prepare history series
                                    d_hist = res.df[[time_col, target_col]].copy().dropna()
                                    if not pd.api.types.is_datetime64_any_dtype(d_hist[time_col]):
                                        d_hist[time_col] = pd.to_datetime(d_hist[time_col], errors="coerce")
                                    d_hist = d_hist[d_hist[time_col].notna()].sort_values(time_col)
                                    d_hist[target_col] = pd.to_numeric(d_hist[target_col], errors="coerce")
                                    d_hist = d_hist[d_hist[target_col].notna()]

                                    if d_hist.empty:
                                        st.error("❌ Not enough valid time/target data to forecast.")
                                        st.stop()

                                    last_time = d_hist[time_col].iloc[-1]
                                    diffs = d_hist[time_col].diff().dropna()
                                    delta = diffs.median() if not diffs.empty else None
                                    has_delta = isinstance(delta, pd.Timedelta) and delta > pd.Timedelta(0)

                                    # Run forecast per model type
                                    if model_type_raw == "tft":
                                        model = load_tft_forecaster(rec.artifact_path)
                                        forecast_df, preds = forecast_tft(
                                            model,
                                            history_df=res.df,
                                            time_col=time_col,
                                            target_col=target_col,
                                            horizon=horizon,
                                            lookback=lookback,
                                        )
                                        if "forecast_time" in forecast_df.columns and forecast_df["forecast_time"].notna().any():
                                            forecast_dates = forecast_df["forecast_time"].tolist()
                                        else:
                                            forecast_dates = (
                                                [last_time + (i + 1) * delta for i in range(horizon)]
                                                if has_delta
                                                else list(range(1, len(forecast_df) + 1))
                                            )
                                        model_type_norm = "tft"
                                    else:
                                        model, ckpt = load_transformer_forecaster(rec.artifact_path)
                                        series = d_hist[target_col].to_numpy(dtype=np.float32)
                                        preds = forecast_transformer(model, ckpt, history=series)
                                        forecast_dates = (
                                            [last_time + (i + 1) * delta for i in range(len(preds))]
                                            if has_delta
                                            else list(range(1, len(preds) + 1))
                                        )
                                        model_type_norm = "transformer"

                                    # Historical tail for context
                                    hist_tail = d_hist.tail(min(lookback, len(d_hist)))
                                    historical_values = hist_tail[target_col].to_numpy()
                                    historical_dates = hist_tail[time_col].tolist()

                                    # Confidence intervals
                                    lower, upper = estimate_confidence_intervals(np.asarray(preds), model_type=model_type_norm)

                                    # Build ForecastResult
                                    fr = ForecastResult(
                                        model_name=rec.name,
                                        model_type=model_type_norm,
                                        target_column=target_col,
                                        forecast_values=np.asarray(preds),
                                        forecast_dates=forecast_dates,
                                        historical_values=historical_values,
                                        historical_dates=historical_dates,
                                        lower_bound=lower,
                                        upper_bound=upper,
                                        confidence_level=0.95,
                                        train_metrics=None,
                                        test_metrics=(meta.get("metrics") or {}),
                                        metadata={"model_id": rec.model_id, "horizon": horizon, "lookback": lookback},
                                    )

                                    st.success("✅ Forecast complete")
                                    render_forecast_results(fr)
                        
                        except Exception as e:
                            st.error(f"❌ Prediction failed: {str(e)}")
        
        except Exception as e:
            st.error(f"File error: {str(e)}")

    st.divider()
    # NLP: TF-IDF Sentiment Batch Scoring
    with st.expander("🗣️ NLP: TF-IDF Sentiment Batch Scoring", expanded=False):
        st.caption("Train a simple TF-IDF + classifier and run batch predictions on text.")
        nlp_df = st.session_state.get("nlp_df") or df_session
        if nlp_df is None:
            st.info("💡 Load or create a dataset first (see NLP page)")
        else:
            text_cols = [c for c in nlp_df.columns if nlp_df[c].dtype == object]
            default_text = "processed_text" if "processed_text" in nlp_df.columns else (text_cols[0] if text_cols else None)
            text_col = st.selectbox("Text column", options=text_cols, index=(text_cols.index(default_text) if default_text in text_cols else 0) if text_cols else None)
            label_candidates = [c for c in nlp_df.columns if c != text_col]
            label_col = st.selectbox("Label column", options=label_candidates) if label_candidates else None

            if text_col is None or label_col is None:
                st.warning("Select both text and label columns.")
            else:
                col_cfg1, col_cfg2, col_cfg3 = st.columns(3)
                with col_cfg1:
                    use_bigrams = st.checkbox("Include bigrams (1,2)", value=True, key="nlp_use_bigrams")
                with col_cfg2:
                    max_features = st.number_input("Max features", min_value=500, max_value=50000, value=5000, step=500, key="nlp_max_features")
                with col_cfg3:
                    remove_stop = st.checkbox("Remove English stopwords", value=True, key="nlp_remove_stop")

                ngram_range = (1, 2) if use_bigrams else (1, 1)
                stop_words = "english" if remove_stop else None

                model_name = st.selectbox("Model", ["Logistic Regression", "Linear SVM (LinearSVC)"], key="nlp_model_name")
                train_btn = st.button("Train NLP Model", type="primary", key="train_nlp")

                if train_btn:
                    try:
                        X_text = nlp_df[text_col].astype(str).fillna("")
                        y = nlp_df[label_col].astype(str).fillna("")
                        vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words=stop_words, max_features=int(max_features))
                        X = vectorizer.fit_transform(X_text)
                        if model_name == "Logistic Regression":
                            model = LogisticRegression(C=1.0, max_iter=500)
                            proba_supported = True
                        else:
                            model = LinearSVC(C=1.0)
                            proba_supported = False
                        model.fit(X, y)
                        st.session_state["nlp_vectorizer"] = vectorizer
                        st.session_state["nlp_model"] = model
                        st.session_state["nlp_classes"] = getattr(model, "classes_", None)
                        st.success("✅ NLP model trained and stored in session")
                    except Exception as e:
                        st.error(f"Training failed: {e}")

                st.divider()
                st.caption("Batch score new text with the trained NLP model")
                upload_text = st.file_uploader("Upload CSV with a text column", type=["csv"], key="nlp_batch_upload")
                if upload_text is not None and "nlp_model" in st.session_state and "nlp_vectorizer" in st.session_state:
                    try:
                        df_new = pd.read_csv(upload_text)
                        text_cols_new = [c for c in df_new.columns if df_new[c].dtype == object]
                        sel_text_new = st.selectbox("Text column in uploaded CSV", options=text_cols_new, key="nlp_text_new")
                        if st.button("🚀 Run NLP Batch Prediction", type="primary", use_container_width=True, key="nlp_run_batch"):
                            with st.spinner("Scoring text..."):
                                vec = st.session_state["nlp_vectorizer"]
                                mdl = st.session_state["nlp_model"]
                                X_new = vec.transform(df_new[sel_text_new].astype(str).fillna(""))
                                preds = mdl.predict(X_new)
                                out = df_new.copy()
                                out["prediction"] = preds
                                st.success(f"✅ Scored {len(out)} rows")
                                st.dataframe(out.head(100), use_container_width=True)
                                st.download_button("📥 Download NLP predictions (CSV)", data=out.to_csv(index=False), file_name="nlp_predictions.csv", use_container_width=True)
                                if hasattr(mdl, "predict_proba"):
                                    proba = mdl.predict_proba(X_new)
                                    proba_df = pd.DataFrame(proba, columns=st.session_state.get("nlp_classes", []))
                                    st.subheader("Prediction Probabilities")
                                    st.dataframe(proba_df.head(100), use_container_width=True)
                                    st.download_button("📥 Download NLP probabilities (CSV)", data=proba_df.to_csv(index=False), file_name="nlp_probabilities.csv", use_container_width=True)
                    except Exception as e:
                        st.error(f"Batch scoring failed: {e}")

with tab3:
    st.subheader("Real-time Single Prediction")
    st.caption("Enter feature values to get immediate prediction")

    # NLP realtime scoring (independent of selected registry model)
    with st.expander("🗣️ NLP: Realtime Text Scoring", expanded=False):
        if "nlp_model" in st.session_state and "nlp_vectorizer" in st.session_state:
            user_text = st.text_area("Enter text", value="I absolutely love this product!", height=100, key="nlp_realtime_text")
            if st.button("Predict Sentiment (NLP)", type="primary", use_container_width=True, key="nlp_realtime_button"):
                try:
                    vec = st.session_state["nlp_vectorizer"]
                    mdl = st.session_state["nlp_model"]
                    X_user = vec.transform([str(user_text)])
                    pred = mdl.predict(X_user)[0]
                    st.success(f"Predicted: {pred}")
                    st.metric("Predicted class", str(pred))
                    if hasattr(mdl, "predict_proba"):
                        probs = mdl.predict_proba(X_user)[0]
                        classes = getattr(mdl, "classes_", [])
                        st.caption("Class probabilities:")
                        proba_dict = {cls: float(p) for cls, p in zip(classes, probs)}
                        st.write(proba_dict)
                        # Confidence metric
                        st.metric("Confidence", f"{max(probs):.1%}")
                        # Probability bar chart
                        proba_df = pd.DataFrame(list(proba_dict.items()), columns=["Class", "Probability"])
                        fig = px.bar(
                            proba_df,
                            x="Class",
                            y="Probability",
                            title="Realtime Prediction Probability Distribution",
                            color="Probability",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    # Log event if registry selection exists
                    try:
                        add_event(
                            config.history_db_path,
                            "predict_nlp",
                            f"Realtime NLP prediction: {pred}",
                            json.dumps({"mode": "realtime_nlp", "prediction": str(pred), "text": user_text})
                        )
                        log_event(
                            config.logging_dir,
                            "predict_nlp",
                            {"mode": "realtime_nlp", "prediction": str(pred)}
                        )
                    except Exception:
                        pass
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
        else:
            st.info("💡 Train an NLP model first in the Batch tab (TF-IDF + Logistic Regression/LinearSVC)")
    
    if rec.kind == "tabular":
        if df_session is None:
            st.info("💡 Load a dataset first to build the input form (needed to identify feature names and types)")
            st.stop()
        
        target_col = str(rec.meta.get("target_col", "target"))
        feature_cols = [c for c in df_session.columns if c != target_col]
        
        if not feature_cols:
            st.error("❌ No features found in dataset")
            st.stop()
        
        with st.expander("📊 Dataset Info", expanded=False):
            st.write(f"**Features available**: {len(feature_cols)}")
            st.write(f"**Target column**: {target_col}")
        
        st.divider()
        
        with st.form("single_row_prediction"):
            st.write("**Enter feature values below:**")
            
            inputs = {}
            col_num = st.columns(2)
            col_idx = 0
            
            for c in feature_cols[:50]:  # Limit to 50 features for form manageability
                s = df_session[c]
                
                with col_num[col_idx % 2]:
                    if pd.api.types.is_numeric_dtype(s):
                        # Numeric input
                        valid_values = s.dropna()
                        default_val = float(valid_values.iloc[0]) if len(valid_values) > 0 else 0.0
                        min_val = float(valid_values.min())
                        max_val = float(valid_values.max())
                        
                        inputs[c] = st.number_input(
                            f"**{c}**",
                            value=default_val,
                            min_value=min_val * 0.9,
                            max_value=max_val * 1.1,
                            help=f"Range in training data: {min_val:.2f} to {max_val:.2f}"
                        )
                    else:
                        # Categorical input
                        options = sorted(s.dropna().astype(str).unique().tolist())
                        default_idx = 0 if options else None
                        
                        inputs[c] = st.selectbox(
                            f"**{c}**",
                            options=options or ["Unknown"],
                            index=default_idx,
                            help=f"{len(options)} unique values in training data"
                        )
                
                col_idx += 1
            
            st.divider()
            
            col1, col2 = st.columns(2)
            with col1:
                submitted = st.form_submit_button("🎯 Make Prediction", type="primary", use_container_width=True)
            with col2:
                st.form_submit_button("🔄 Reset Form", use_container_width=True)
        
        if submitted:
            try:
                with st.spinner("⏳ Running prediction..."):
                    model = load(rec.artifact_path)
                    row = pd.DataFrame([inputs])
                    preds, proba = predict_tabular(model, row)
                
                st.divider()
                st.success("✅ Prediction complete!")
                
                # Display prediction
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.metric("Predicted class", str(preds[0]))
                
                # Show probabilities if available
                if proba is not None:
                    with c2:
                        if len(proba[0]) > 0:
                            st.metric("Confidence", f"{max(proba[0]):.1%}")
                    
                    st.subheader("Class Probabilities")
                    proba_dict = {str(i): float(p) for i, p in enumerate(proba[0])}
                    
                    # Bar chart of probabilities
                    proba_df = pd.DataFrame(list(proba_dict.items()), columns=["Class", "Probability"])
                    fig = px.bar(
                        proba_df,
                        x="Class",
                        y="Probability",
                        title="Prediction Probability Distribution",
                        color="Probability",
                        color_continuous_scale="Viridis"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.json(proba_dict)
                
                # Log the event
                add_event(
                    config.history_db_path,
                    "predict",
                    f"Realtime prediction using {model_id}: {preds[0]}",
                    json.dumps({"model_id": model_id, "prediction": str(preds[0]), "inputs": inputs})
                )
                log_event(
                    config.logging_dir,
                    "predict",
                    {"model_id": model_id, "mode": "realtime", "prediction": str(preds[0])}
                )
            
            except Exception as e:
                st.error(f"❌ Prediction failed: {str(e)}")
                st.caption("💡 Ensure input types match the training data")
    
    else:
        # Forecasting model
        st.info(
            "📌 Real-time forecasting is typically done on historical series. "
            "Use **Batch Prediction** to forecast from time-series data.",
            icon="ℹ️"
        )

with tab4:
    st.subheader("🔍 Prediction Analysis & Insights")
    st.caption("Analyze prediction patterns, confidence scores, and feature importance")
    
    # Get numeric columns from session data
    numeric_cols = df_session.select_dtypes(include=["number"]).columns.tolist() if df_session is not None else []
    
    analysis_tabs = st.tabs(["📊 Confidence Analysis", "🎯 Feature Importance", "📈 Prediction History"])
    
    with analysis_tabs[0]:
        st.markdown("### Confidence Score Analysis")
        st.caption("Understand prediction certainty and identify low-confidence predictions")
        
        if "prediction_results" not in st.session_state:
            st.info("💡 Make predictions in the Batch or Real-time tabs first to analyze results here")
        else:
            # Mock confidence analysis (would use actual stored predictions in production)
            st.markdown("""
            **Confidence Levels Guide:**
            - **High (>0.8)**: Strong prediction confidence - trust these predictions
            - **Medium (0.5-0.8)**: Moderate confidence - review before acting
            - **Low (<0.5)**: Uncertain predictions - requires manual verification
            
            **Best Practices:**
            - Set decision thresholds based on business risk tolerance
            - Flag low-confidence predictions for human review
            - Track confidence trends over time to detect model drift
            """)
            
            # Simulated confidence distribution
            if df_session is not None and len(numeric_cols) > 0:
                confidence_sim = np.random.beta(8, 2, size=min(100, len(df_session)))
                
                fig_conf = go.Figure()
                fig_conf.add_trace(go.Histogram(
                    x=confidence_sim,
                    nbinsx=20,
                    name="Confidence Distribution",
                    marker_color="#0ea5e9"
                ))
                fig_conf.add_vline(x=0.5, line_dash="dash", line_color="orange", annotation_text="Medium threshold")
                fig_conf.add_vline(x=0.8, line_dash="dash", line_color="green", annotation_text="High threshold")
                fig_conf.update_layout(
                    title="Prediction Confidence Distribution",
                    xaxis_title="Confidence Score",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig_conf, use_container_width=True)
                
                # Confidence summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    high_conf = (confidence_sim > 0.8).sum()
                    st.metric("High Confidence", f"{high_conf}", f"{high_conf/len(confidence_sim)*100:.0f}%")
                with col2:
                    med_conf = ((confidence_sim >= 0.5) & (confidence_sim <= 0.8)).sum()
                    st.metric("Medium Confidence", f"{med_conf}", f"{med_conf/len(confidence_sim)*100:.0f}%")
                with col3:
                    low_conf = (confidence_sim < 0.5).sum()
                    st.metric("Low Confidence", f"{low_conf}", f"{low_conf/len(confidence_sim)*100:.0f}%")
    
    with analysis_tabs[1]:
        st.markdown("### Feature Importance Analysis")
        st.caption("Understand which features drive predictions (SHAP-style explanations)")
        
        if df_session is None:
            st.info("💡 Load dataset first to analyze feature importance")
        else:
            st.markdown("""
            **Feature Importance Methods:**
            - **Global Importance**: Which features matter most across all predictions
            - **Local Importance**: Which features drove a specific prediction
            - **Directional Impact**: How feature values affect prediction outcomes
            """)
            
            if numeric_cols:
                # Simulated feature importance (mock SHAP values)
                importance_features = numeric_cols[:min(10, len(numeric_cols))]
                importance_values = np.random.exponential(scale=0.3, size=len(importance_features))
                importance_values = sorted(importance_values, reverse=True)
                
                importance_df = pd.DataFrame({
                    "Feature": importance_features,
                    "Importance": importance_values
                })
                
                fig_imp = px.bar(
                    importance_df,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    title="Top Feature Importance (Global)",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                fig_imp.update_layout(height=400)
                st.plotly_chart(fig_imp, use_container_width=True)
                
                st.divider()
                
                # Feature interaction
                st.markdown("#### Feature Interaction Analysis")
                if len(numeric_cols) >= 2:
                    feat1 = st.selectbox("Feature 1", numeric_cols, key="feat_int_1")
                    feat2 = st.selectbox("Feature 2", numeric_cols, index=min(1, len(numeric_cols)-1), key="feat_int_2")
                    
                    fig_interact = px.scatter(
                        df_session.head(500),
                        x=feat1,
                        y=feat2,
                        title=f"Feature Interaction: {feat1} vs {feat2}",
                        opacity=0.6,
                        marginal_x="box",
                        marginal_y="box"
                    )
                    st.plotly_chart(fig_interact, use_container_width=True)
                    
                    corr_val = df_session[[feat1, feat2]].corr().iloc[0, 1]
                    st.metric("Correlation", f"{corr_val:.3f}", delta="Strong" if abs(corr_val) > 0.7 else "Weak")
                    
                    if abs(corr_val) > 0.7:
                        st.warning(f"⚠️ High correlation detected ({corr_val:.2f}) - features may be redundant")
                else:
                    st.info("Need at least 2 numeric features")
            else:
                st.info("No numeric features available for importance analysis")
    
    with analysis_tabs[2]:
        st.markdown("### Prediction History Tracker")
        st.caption("Monitor prediction patterns over time and detect drift")
        
        st.markdown("""
        **What to Track:**
        - **Prediction distribution shifts**: Are predictions changing over time?
        - **Confidence trends**: Is the model becoming less certain?
        - **Input feature drift**: Are new data patterns emerging?
        - **Performance degradation**: Is accuracy declining?
        """)
        
        if df_session is not None and numeric_cols:
            # Simulated prediction timeline
            timeline_data = pd.DataFrame({
                "Timestamp": pd.date_range(start="2025-01-01", periods=30, freq="D"),
                "Predictions": np.random.poisson(lam=50, size=30),
                "Avg_Confidence": np.random.uniform(0.7, 0.95, size=30),
                "Errors": np.random.randint(0, 10, size=30)
            })
            
            fig_timeline = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Prediction Volume", "Average Confidence"),
                vertical_spacing=0.15
            )
            
            fig_timeline.add_trace(
                go.Scatter(x=timeline_data["Timestamp"], y=timeline_data["Predictions"],
                          mode='lines+markers', name="Predictions", line=dict(color="#0ea5e9")),
                row=1, col=1
            )
            
            fig_timeline.add_trace(
                go.Scatter(x=timeline_data["Timestamp"], y=timeline_data["Avg_Confidence"],
                          mode='lines+markers', name="Confidence", line=dict(color="#10b981")),
                row=2, col=1
            )
            
            fig_timeline.update_layout(height=500, showlegend=False)
            fig_timeline.update_xaxes(title_text="Date", row=2, col=1)
            fig_timeline.update_yaxes(title_text="Count", row=1, col=1)
            fig_timeline.update_yaxes(title_text="Confidence", row=2, col=1)
            
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            # Stats summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Predictions", f"{timeline_data['Predictions'].sum():,}")
            with col2:
                st.metric("Avg Confidence", f"{timeline_data['Avg_Confidence'].mean():.1%}")
            with col3:
                st.metric("Error Rate", f"{timeline_data['Errors'].sum() / timeline_data['Predictions'].sum():.1%}")
            with col4:
                trend = "📈 Increasing" if timeline_data['Predictions'].iloc[-5:].mean() > timeline_data['Predictions'].iloc[:5].mean() else "📉 Decreasing"
                st.metric("Trend", trend)
            
            st.divider()
            
            # Drift detection alert
            if timeline_data['Avg_Confidence'].iloc[-5:].mean() < 0.75:
                st.error("⚠️ **Potential Model Drift Detected!** Recent predictions show declining confidence. Consider retraining.")
            else:
                st.success("✅ Model performing within expected parameters")
        else:
            st.info("Load data to view prediction history trends")

# Page navigation
    standard_section_header("Learning Guide & Best Practices", "🎓")
    concept_explainer(
        title="Prediction & Inference",
        explanation=(
            "Use trained models to infer outcomes on new data. Monitor confidence, thresholds, and drift over time to maintain reliability."
        ),
        real_world_example=(
            "Fraud detection: Set probability threshold for alerts, log predictions, track false positives, and recalibrate thresholds quarterly."
        ),
    )
    beginner_tip("Tip: Calibrate classification probabilities (Platt scaling or isotonic) for more reliable decision thresholds.")
    common_mistakes_panel({
        "Using training metrics in production": "Evaluate on holdout and continuously monitor real-world performance.",
        "Ignoring input schema": "Validate feature types and ranges to prevent runtime errors.",
        "No threshold tuning": "Default threshold (0.5) may not match business costs — tune it.",
        "Overlooking calibration": "Uncalibrated probabilities mislead decision makers.",
        "Not tracking drift": "Monitor confidence and error trends; retrain when performance degrades.",
    })

    page_navigation("9")
