"""Interactive Simulation Dashboard - Modern UI with Real-Time Parameter Updates."""

import streamlit as st
import pandas as pd
import numpy as np

from src.core.config import load_config
from src.core.logging_utils import setup_logging
from src.core.ui import app_header
from src.core.styles import render_stat_card
from src.simulation.engine import SimulationParameters, DataGenerator

st.set_page_config(
    page_title="DataScope Pro - Interactive Simulations",
    layout="wide",
    initial_sidebar_state="expanded"
)


def _task_key(sim_key: str, task_id: int) -> str:
    return f"interactive_sim::{sim_key}::task::{task_id}"


def render_tasks(sim_key: str, title: str, tasks: list[str]) -> None:
    st.markdown(f"### ✅ {title}")
    cols = st.columns(2) if len(tasks) >= 6 else None
    for i, task in enumerate(tasks, start=1):
        key = _task_key(sim_key, i)
        if cols is None:
            st.checkbox(task, key=key)
        else:
            with cols[0] if (i % 2 == 1) else cols[1]:
                st.checkbox(task, key=key)


def _render_eda_preview(df: pd.DataFrame) -> None:
    """Display quick EDA after CSV upload."""
    st.sidebar.markdown("### 📊 Data Preview")
    
    # Basic stats
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Rows", df.shape[0])
    with col2:
        st.metric("Columns", df.shape[1])
    
    # Data types
    with st.sidebar.expander("Data Types", expanded=False):
        dtype_summary = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str)
        })
        st.dataframe(dtype_summary, use_container_width=True, hide_index=True)
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        with st.sidebar.expander("Missing Values", expanded=False):
            missing_pct = (missing / len(df) * 100).round(2)
            missing_df = pd.DataFrame({
                'Column': missing[missing > 0].index,
                'Count': missing[missing > 0].values,
                '% Missing': missing_pct[missing > 0].values
            })
            st.dataframe(missing_df, use_container_width=True, hide_index=True)
    
    # Numeric summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        with st.sidebar.expander("Numeric Summary", expanded=False):
            st.dataframe(df[numeric_cols].describe().round(3), use_container_width=True)
    
    # Class distribution (for first non-numeric or last column)
    cat_cols = df.select_dtypes(include=['object']).columns
    if len(cat_cols) > 0:
        first_cat = cat_cols[0]
        with st.sidebar.expander(f"Distribution: {first_cat}", expanded=False):
            dist = df[first_cat].value_counts()
            st.bar_chart(dist)
    elif len(numeric_cols) > 0:
        last_numeric = numeric_cols[-1]
        with st.sidebar.expander(f"Distribution: {last_numeric}", expanded=False):
            st.histogram(df[last_numeric].dropna(), use_container_width=True)


def _make_2d_dataset(kind: str, n_samples: int, noise: float, random_state: int):
    from sklearn.datasets import make_moons, make_circles, make_classification

    if kind == "Moons":
        return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    if kind == "Circles":
        return make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    # Linear/Blobs-like
    X, y = make_classification(
        n_samples=n_samples,
        n_features=2,
        n_redundant=0,
        n_informative=2,
        n_clusters_per_class=1,
        class_sep=max(0.5, 2.0 - 1.5 * noise),
        flip_y=min(0.3, noise),
        random_state=random_state,
    )
    return X, y


def _plot_decision_boundary(model, X: np.ndarray, y: np.ndarray, title: str = "Decision Boundary") -> None:
    import plotly.graph_objects as go

    x_min, x_max = X[:, 0].min() - 0.8, X[:, 0].max() + 0.8
    y_min, y_max = X[:, 1].min() - 0.8, X[:, 1].max() + 0.8

    grid_size = 220
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, grid_size),
        np.linspace(y_min, y_max, grid_size),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    try:
        zz = model.predict(grid)
    except Exception:
        zz = np.zeros((grid.shape[0],), dtype=int)
    zz = zz.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, grid_size),
            y=np.linspace(y_min, y_max, grid_size),
            z=zz,
            showscale=False,
            opacity=0.35,
            colorscale=[[0, "#f093fb"], [1, "#f5576c"]],
            contours=dict(showlines=False),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
                size=7,
                color=y,
                colorscale="Viridis",
                line=dict(width=0.5, color="white"),
            ),
            name="Samples",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Feature 1",
        yaxis_title="Feature 2",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function for interactive simulations page."""
    config = load_config()
    logger = setup_logging(log_dir=config.logging_dir)
    
    # Page header
    app_header(config, page_title="🔬 Interactive Simulations")
    
    st.markdown("""
    ## 🎯 Welcome to Interactive ML Simulations
    
    **Master machine learning through hands-on experimentation!** This interactive lab allows you to:
    - 🔬 Experiment with different ML algorithms in real-time
    - 📊 Visualize how parameters affect model performance
    - 🎓 Learn core ML concepts through practical examples
    - 🚀 Compare algorithms side-by-side
    - 💡 Understand trade-offs between bias and variance
    
    ### 🌟 What's New
    - **Advanced Algorithms**: Neural Networks, Ensemble Methods, SVM
    - **Real-World Scenarios**: Imbalanced Data, High-Dimensional Data
    - **Interactive Visualizations**: Decision Boundaries, Learning Curves
    - **Performance Analysis**: Confusion Matrices, ROC Curves
    """)

    with st.expander("📌 Quick start (recommended)", expanded=False):
        st.markdown(
            """
            If you're not sure where to begin, use this order:
            1) 🌳 Decision Boundaries → understand model geometry
            2) 📉 Learning Curves → diagnose data vs model capacity
            3) ⚖️ Bias–Variance → pick the right complexity
            4) 🎲 Imbalanced Data → avoid misleading accuracy
            5) 🔄 Cross-Validation → get stable estimates
            """
        )
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Simulation Settings")

    st.sidebar.markdown("### 📁 Use Your Own Data (CSV)")
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"], accept_multiple_files=False)
    uploaded_df: pd.DataFrame | None = None
    if uploaded is not None:
        try:
            uploaded_df = pd.read_csv(uploaded)
            st.sidebar.success(f"Loaded {uploaded_df.shape[0]} rows × {uploaded_df.shape[1]} cols")
            _render_eda_preview(uploaded_df)
        except Exception as e:
            st.sidebar.error(f"Could not read CSV: {e}")
            uploaded_df = None

    use_uploaded = st.sidebar.checkbox(
        "Use uploaded data for Classification/Regression/Time Series",
        value=False,
        disabled=uploaded_df is None,
    )
    
    sim_type = st.sidebar.selectbox(
        "Select Simulation Type",
        [
            "🎯 Classification - Multi-Algorithm",
            "📈 Regression - Advanced Models",
            "📊 Time Series - Forecasting",
            "🌳 Decision Boundaries Visualization",
            "📉 Learning Curves Analysis",
            "⚖️ Bias-Variance Trade-off",
            "🎲 Imbalanced Data Handling",
            "🔄 Cross-Validation Demo",
            "🧠 Neural Network Playground",
            "🎪 Ensemble Methods"
        ]
    )
    
    # Common parameters
    dataset_size = st.sidebar.slider(
        "Dataset Size",
        min_value=50,
        max_value=1000,
        value=200,
        step=50
    )
    
    noise_level = st.sidebar.slider(
        "Noise Level",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1
    )
    
    random_seed = st.sidebar.number_input(
        "Random Seed",
        value=42,
        step=1
    )
    
    # Create simulation parameters (used by the built-in generator helpers)
    params = SimulationParameters(n_samples=dataset_size, noise_level=noise_level, random_state=int(random_seed))

    def _prepare_uploaded_xy(task: str):
        assert uploaded_df is not None
        df = uploaded_df.copy()

        st.sidebar.markdown("#### 🎯 Target / Columns")
        cols = list(df.columns)
        target_col = st.sidebar.selectbox("Target column", cols, index=len(cols) - 1, key=f"upl_target::{task}")
        feature_cols = st.sidebar.multiselect(
            "Feature columns (leave empty = auto)",
            [c for c in cols if c != target_col],
            default=[],
            key=f"upl_features::{task}",
        )

        if not feature_cols:
            # Prefer numeric columns automatically
            feature_cols = [c for c in cols if c != target_col and pd.api.types.is_numeric_dtype(df[c])]

        X_df = df[feature_cols].copy()
        y_raw = df[target_col].copy()

        # Basic cleanup
        X_df = X_df.replace([np.inf, -np.inf], np.nan)
        X_df = X_df.dropna(axis=0)
        y_raw = y_raw.loc[X_df.index]

        # Encode target if needed
        if task == "classification":
            if not pd.api.types.is_numeric_dtype(y_raw):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y_raw.astype(str))
            else:
                y = y_raw.to_numpy()
        else:
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy()
            mask = ~np.isnan(y)
            X_df = X_df.loc[mask]
            y = y[mask]

        # Keep numeric features only
        X_df = X_df.select_dtypes(include=["number"]).copy()
        X = X_df.to_numpy()

        return X, y, X_df.columns.tolist(), target_col
    
    # Simulation dispatch
    if sim_type == "🎯 Classification - Multi-Algorithm":
        st.subheader("🎯 Classification Simulation - Compare Multiple Algorithms")
        
        st.info("""
        **Learning Objective**: Compare how different classification algorithms perform on the same dataset.
        Experiment with parameters to understand when each algorithm excels.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_features = st.slider("Number of Features", 2, 10, 2)
            n_classes = st.slider("Number of Classes", 2, 5, 2)
            algorithm = st.selectbox("Algorithm", [
                "Logistic Regression",
                "Random Forest",
                "Support Vector Machine",
                "K-Nearest Neighbors",
                "Naive Bayes",
                "Gradient Boosting"
            ])
        
        with col2:
            test_size = st.slider("Test Split", 0.1, 0.5, 0.2)
            feature_scaling = st.checkbox("Scale Features", value=True)
            class_weight = st.selectbox("Class Weight", ["balanced", "None"])
        
        with col3:
            show_details = st.checkbox("Show Detailed Metrics", value=True)
            show_confusion = st.checkbox("Show Confusion Matrix", value=True)
            show_roc = st.checkbox("Show ROC Curve (Binary)", value=False)
        
        if st.button("🚀 Run Classification Simulation", key="clf_sim", type="primary"):
            try:
                with st.spinner("Running classification simulation..."):
                    if use_uploaded and uploaded_df is not None:
                        X, y, feature_names, target_col = _prepare_uploaded_xy("classification")
                    else:
                        # Create parameters
                        params = SimulationParameters(
                            n_samples=dataset_size,
                            n_features=n_features,
                            n_classes=n_classes,
                            noise_level=noise_level,
                            random_state=int(random_seed),
                            test_size=test_size,
                        )
                        # Generate data
                        X, y = DataGenerator.generate_classification(params)
                        feature_names, target_col = None, None
                    
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=int(random_seed)
                    )
                    
                    if feature_scaling:
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)
                    
                    # Select and train algorithm
                    cw = class_weight if class_weight != "None" else None
                    
                    if algorithm == "Logistic Regression":
                        from sklearn.linear_model import LogisticRegression
                        model = LogisticRegression(class_weight=cw, max_iter=1000, random_state=int(random_seed))
                    elif algorithm == "Random Forest":
                        from sklearn.ensemble import RandomForestClassifier
                        model = RandomForestClassifier(n_estimators=100, class_weight=cw, random_state=int(random_seed))
                    elif algorithm == "Support Vector Machine":
                        from sklearn.svm import SVC
                        model = SVC(class_weight=cw, probability=True, random_state=int(random_seed))
                    elif algorithm == "K-Nearest Neighbors":
                        from sklearn.neighbors import KNeighborsClassifier
                        model = KNeighborsClassifier()
                    elif algorithm == "Naive Bayes":
                        from sklearn.naive_bayes import GaussianNB
                        model = GaussianNB()
                    else:  # Gradient Boosting
                        from sklearn.ensemble import GradientBoostingClassifier
                        model = GradientBoostingClassifier(random_state=int(random_seed))
                    
                    # Train and evaluate
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                    results = {
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                        'predictions': y_pred,
                        'model': model
                    }
                    
                    # Display results
                    st.success(f"✅ Simulation completed with {algorithm}!")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(render_stat_card(
                            label="Accuracy",
                            value=f"{results['accuracy']:.3f}",
                            icon="🎯"
                        ), unsafe_allow_html=True)
                    with col2:
                        st.markdown(render_stat_card(
                            label="Precision",
                            value=f"{results['precision']:.3f}",
                            icon="📊"
                        ), unsafe_allow_html=True)
                    with col3:
                        st.markdown(render_stat_card(
                            label="Recall",
                            value=f"{results['recall']:.3f}",
                            icon="🔍"
                        ), unsafe_allow_html=True)
                    with col4:
                        st.markdown(render_stat_card(
                            label="F1 Score",
                            value=f"{results['f1']:.3f}",
                            icon="⭐"
                        ), unsafe_allow_html=True)
                    
                    if show_confusion and n_classes <= 5:
                        st.write("### 📊 Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        
                        import plotly.figure_factory as ff
                        fig = ff.create_annotated_heatmap(
                            cm, 
                            x=[f"Pred {i}" for i in range(n_classes)],
                            y=[f"True {i}" for i in range(n_classes)],
                            colorscale='Blues'
                        )
                        fig.update_layout(title="Confusion Matrix", width=500, height=500)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if show_roc and n_classes == 2:
                        st.write("### 📈 ROC Curve (Binary Classification)")
                        from sklearn.metrics import roc_curve, auc
                        y_proba = model.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, y_proba)
                        roc_auc = auc(fpr, tpr)
                        
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', 
                                                name=f'ROC (AUC={roc_auc:.3f})'))
                        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                                                line=dict(dash='dash'), name='Random'))
                        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate",
                                        yaxis_title="True Positive Rate")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if show_details:
                        st.write("### 📋 Detailed Classification Report")
                        from sklearn.metrics import classification_report
                        report = classification_report(y_test, y_pred, output_dict=True)
                        st.dataframe(pd.DataFrame(report).transpose())

                    render_tasks(
                        "classification",
                        "Try these tasks",
                        [
                            "Switch between Logistic Regression and Random Forest and compare Precision/Recall.",
                            "Turn feature scaling off and see which algorithms degrade.",
                            "Set classes to 3–5 and inspect confusion patterns.",
                            "Enable ROC (binary) and increase noise to see AUC drop.",
                            "Use class_weight=balanced and compare minority recall.",
                        ],
                    )
                        
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                logger.error(f"Classification simulation error: {str(e)}")
    
    elif sim_type == "📈 Regression - Advanced Models":
        st.subheader("📈 Regression Simulation - Advanced Modeling")
        
        st.info("""
        **Learning Objective**: Explore different regression algorithms and understand their strengths.
        Compare linear vs non-linear models on various data patterns.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_features = st.slider("Number of Features", 1, 10, 1)
            n_samples = st.slider("Samples", 50, 1000, dataset_size)
            algorithm = st.selectbox("Algorithm", [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "Random Forest",
                "Gradient Boosting",
                "Support Vector Regression"
            ])
        
        with col2:
            test_size = st.slider("Test Split", 0.1, 0.5, 0.2)
            show_plot = st.checkbox("Show Predictions Plot", value=True)
            alpha = st.slider("Regularization (α)", 0.01, 10.0, 1.0) if "Ridge" in algorithm or "Lasso" in algorithm else None
        
        with col3:
            show_details = st.checkbox("Show Residuals", value=True)
            show_feature_importance = st.checkbox("Show Feature Importance", value=False)
        
        if st.button("🚀 Run Regression Simulation", key="reg_sim", type="primary"):
            try:
                with st.spinner("Running regression simulation..."):
                    if use_uploaded and uploaded_df is not None:
                        X, y, feature_names, target_col = _prepare_uploaded_xy("regression")
                    else:
                        # Create parameters
                        params = SimulationParameters(
                            n_samples=n_samples,
                            n_features=n_features,
                            noise_level=noise_level,
                            random_state=int(random_seed),
                            test_size=test_size,
                        )
                        # Generate data
                        X, y = DataGenerator.generate_regression(params)
                        feature_names, target_col = None, None
                    
                    from sklearn.model_selection import train_test_split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=int(random_seed)
                    )
                    
                    # Select algorithm
                    if algorithm == "Linear Regression":
                        from sklearn.linear_model import LinearRegression
                        model = LinearRegression()
                    elif algorithm == "Ridge Regression":
                        from sklearn.linear_model import Ridge
                        model = Ridge(alpha=alpha)
                    elif algorithm == "Lasso Regression":
                        from sklearn.linear_model import Lasso
                        model = Lasso(alpha=alpha)
                    elif algorithm == "Random Forest":
                        from sklearn.ensemble import RandomForestRegressor
                        model = RandomForestRegressor(n_estimators=100, random_state=int(random_seed))
                    elif algorithm == "Gradient Boosting":
                        from sklearn.ensemble import GradientBoostingRegressor
                        model = GradientBoostingRegressor(random_state=int(random_seed))
                    else:  # SVR
                        from sklearn.svm import SVR
                        model = SVR()
                    
                    # Train
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    results = {
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'predictions': y_pred,
                        'model': model
                    }
                    
                    st.success(f"✅ Simulation completed with {algorithm}!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(render_stat_card(
                            label="R² Score",
                            value=f"{results['r2']:.3f}",
                            icon="📈"
                        ), unsafe_allow_html=True)
                    with col2:
                        st.markdown(render_stat_card(
                            label="RMSE",
                            value=f"{results['rmse']:.3f}",
                            icon="📉"
                        ), unsafe_allow_html=True)
                    with col3:
                        st.markdown(render_stat_card(
                            label="MAE",
                            value=f"{results['mae']:.3f}",
                            icon="📊"
                        ), unsafe_allow_html=True)
                    
                    if show_plot:
                        st.write("### 📊 Predictions vs Actual")
                        import plotly.graph_objects as go
                        fig = go.Figure()
                        
                        # Add perfect prediction line
                        min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
                        fig.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Perfect Prediction',
                            line=dict(dash='dash', color='gray')
                        ))
                        
                        fig.add_trace(go.Scatter(
                            x=y_test[:100],
                            y=y_pred[:100],
                            mode='markers',
                            name='Predictions',
                            marker=dict(size=8, color='#f5576c')
                        ))
                        fig.update_layout(
                            xaxis_title="Actual Values",
                            yaxis_title="Predicted Values",
                            title="Prediction Accuracy"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    if show_details:
                        st.write("### 📉 Residuals Analysis")
                        residuals = y_test - y_pred
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=y_pred,
                                y=residuals,
                                mode='markers',
                                marker=dict(size=6)
                            ))
                            fig.add_hline(y=0, line_dash="dash", line_color="red")
                            fig.update_layout(
                                title="Residuals Plot",
                                xaxis_title="Predicted Values",
                                yaxis_title="Residuals"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            fig = go.Figure()
                            fig.add_trace(go.Histogram(x=residuals, nbinsx=30))
                            fig.update_layout(
                                title="Residuals Distribution",
                                xaxis_title="Residuals",
                                yaxis_title="Frequency"
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    if show_feature_importance and hasattr(model, 'feature_importances_'):
                        st.write("### 🎯 Feature Importance")
                        importance = pd.DataFrame({
                            'Feature': [f'Feature {i+1}' for i in range(n_features)],
                            'Importance': model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        import plotly.express as px
                        fig = px.bar(importance, x='Feature', y='Importance', 
                                    title="Feature Importance Ranking")
                        st.plotly_chart(fig, use_container_width=True)

                    render_tasks(
                        "regression",
                        "Try these tasks",
                        [
                            "Compare Linear vs Random Forest when noise is low vs high.",
                            "Increase α for Ridge/Lasso and watch RMSE change.",
                            "Enable residual plots and look for non-random structure.",
                            "Turn on feature importance for Random Forest and interpret the ranking.",
                            "Increase dataset size and observe stability of metrics.",
                        ],
                    )
                        
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                logger.error(f"Regression simulation error: {str(e)}")
    
    elif sim_type == "📊 Time Series - Forecasting":
        st.subheader("📊 Time Series Simulation & Forecasting")
        
        st.info("""
        **Learning Objective**: Generate realistic time series with trend, seasonality, and noise.
        Understand components of time series data.
        """)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_steps = st.slider("Time Steps", 50, 500, 100)
            trend = st.checkbox("Add Trend", value=True)
        
        with col2:
            seasonality = st.checkbox("Add Seasonality", value=True)
            season_period = st.slider("Season Period", 5, 50, 12)
        
        with col3:
            volatility = st.slider("Volatility", 0.0, 1.0, 0.1)
            show_components = st.checkbox("Show Components")
        
        if st.button("Generate Time Series", key="ts_sim"):
            try:
                with st.spinner("Running time series simulation..."):
                    if use_uploaded and uploaded_df is not None:
                        df = uploaded_df.copy()
                        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                        if not numeric_cols:
                            raise ValueError("Uploaded CSV has no numeric columns for time series")
                        ts_col = st.sidebar.selectbox("Time series column", numeric_cols, key="upl_ts_col")
                        series = pd.to_numeric(df[ts_col], errors="coerce").dropna().to_numpy()
                        if len(series) < 10:
                            raise ValueError("Time series column has too few numeric values")
                        ts_data = series[: int(n_steps)]
                    else:
                        # Create parameters
                        params = SimulationParameters(
                            n_samples=n_steps,
                            noise_level=volatility,
                            random_state=int(random_seed),
                        )
                        # Generate time series data
                        ts_data = DataGenerator.generate_time_series(
                            params,
                            trend=trend,
                            seasonality=seasonality,
                            season_period=season_period,
                        )
                    
                    st.success("✅ Time series generated!")
                    
                    # Plot time series
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=ts_data,
                        mode='lines',
                        name='Time Series',
                        line=dict(color='#1f77b4')
                    ))
                    fig.update_layout(
                        title="Generated Time Series",
                        xaxis_title="Time Steps",
                        yaxis_title="Value",
                        hovermode='x unified'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    if show_components:
                        st.write("### Series Statistics")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(render_stat_card(
                                label="Mean",
                                value=f"{np.mean(ts_data):.3f}",
                                icon="📊"
                            ), unsafe_allow_html=True)
                        with col2:
                            st.markdown(render_stat_card(
                                label="Std Dev",
                                value=f"{np.std(ts_data):.3f}",
                                icon="📈"
                            ), unsafe_allow_html=True)
                        with col3:
                            st.markdown(render_stat_card(
                                label="Range",
                                value=f"{np.max(ts_data) - np.min(ts_data):.3f}",
                                icon="📏"
                            ), unsafe_allow_html=True)

                    render_tasks(
                        "time_series",
                        "Try these tasks",
                        [
                            "Turn off seasonality, keep trend, and compare variability.",
                            "Increase season period and see how the pattern changes.",
                            "Increase volatility and observe how summary stats respond.",
                            "Change the random seed to explore different realizations.",
                            "Use a small vs large number of steps and compare smoothness.",
                        ],
                    )
                        
            except Exception as e:
                st.error(f"Simulation failed: {str(e)}")
                logger.error(f"Time series error: {str(e)}")

    elif sim_type == "🌳 Decision Boundaries Visualization":
        st.subheader("🌳 Decision Boundaries Visualization")
        st.info(
            "Learn how model choice and hyperparameters change the *shape* of the decision boundary. "
            "This is the fastest way to build intuition about linear vs non-linear classifiers."
        )

        left, mid, right = st.columns(3)
        with left:
            dataset_kind = st.selectbox("2D Dataset", ["Moons", "Circles", "Linear"], index=0)
            n_samples = st.slider("Samples", 100, 1000, min(400, int(dataset_size) * 2), step=50)
        with mid:
            model_kind = st.selectbox(
                "Model",
                ["Logistic Regression", "SVM (RBF)", "KNN", "Decision Tree", "Random Forest"],
                index=1,
            )
            scale = st.checkbox("Standardize features", value=True)
        with right:
            if model_kind == "SVM (RBF)":
                C = st.slider("C", 0.1, 30.0, 3.0)
                gamma = st.slider("gamma", 0.001, 5.0, 0.5)
            elif model_kind == "KNN":
                k = st.slider("k", 1, 31, 9, step=2)
            elif model_kind == "Decision Tree":
                depth = st.slider("max_depth", 1, 20, 4)
            elif model_kind == "Random Forest":
                trees = st.slider("n_estimators", 20, 400, 150, step=10)
                depth = st.slider("max_depth", 1, 20, 8)

        if st.button("Render Decision Boundary", type="primary", key="db_run"):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.svm import SVC
                from sklearn.neighbors import KNeighborsClassifier
                from sklearn.tree import DecisionTreeClassifier
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import accuracy_score

                X, y = _make_2d_dataset(dataset_kind, n_samples, noise_level, int(random_seed))
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=int(random_seed), stratify=y
                )

                if model_kind == "Logistic Regression":
                    base = LogisticRegression(max_iter=2000, random_state=int(random_seed))
                elif model_kind == "SVM (RBF)":
                    base = SVC(C=C, gamma=gamma, probability=False, random_state=int(random_seed))
                elif model_kind == "KNN":
                    base = KNeighborsClassifier(n_neighbors=k)
                elif model_kind == "Decision Tree":
                    base = DecisionTreeClassifier(max_depth=depth, random_state=int(random_seed))
                else:
                    base = RandomForestClassifier(
                        n_estimators=trees, max_depth=depth, random_state=int(random_seed)
                    )

                model = Pipeline([("scaler", StandardScaler()), ("model", base)]) if scale else base
                model.fit(X_train, y_train)
                acc = accuracy_score(y_test, model.predict(X_test))

                col1, col2 = st.columns([2, 1])
                with col2:
                    st.markdown(
                        render_stat_card(label="Test accuracy", value=f"{acc:.3f}", icon="🎯"),
                        unsafe_allow_html=True,
                    )
                with col1:
                    _plot_decision_boundary(model, X, y, title=f"{dataset_kind} — {model_kind}")

                render_tasks(
                    "decision_boundaries",
                    "Try these tasks",
                    [
                        "Use Moons and compare Logistic Regression vs SVM (RBF).",
                        "Increase gamma for SVM and watch overfitting appear.",
                        "Try KNN with k=1 vs k=31 and observe boundary smoothness.",
                        "Disable standardization and see which models are sensitive.",
                    ],
                )
            except Exception as e:
                st.error(f"Decision boundary demo failed: {e}")
                logger.error(f"Decision boundary error: {e}")

    elif sim_type == "📉 Learning Curves Analysis":
        st.subheader("📉 Learning Curves Analysis")
        st.info(
            "Use learning curves to diagnose whether you need more data, a stronger model, or regularization."
        )

        left, right = st.columns(2)
        with left:
            task_type = st.selectbox("Task", ["Classification", "Regression"], index=0)
            model_kind = st.selectbox(
                "Model",
                ["Logistic/Ridge", "Random Forest", "Gradient Boosting", "SVM/SVR"],
                index=0,
            )
        with right:
            n_samples = st.slider("Samples", 200, 3000, 800, step=100)
            cv = st.slider("CV folds", 3, 10, 5)
            train_sizes = st.slider("# Train sizes", 5, 15, 8)

        if st.button("Compute Learning Curve", type="primary", key="lc_run"):
            try:
                from sklearn.model_selection import learning_curve
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import make_scorer
                from sklearn.metrics import accuracy_score, r2_score
                from sklearn.linear_model import LogisticRegression, Ridge
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
                from sklearn.svm import SVC, SVR
                from sklearn.datasets import make_classification, make_regression
                import plotly.graph_objects as go

                rs = int(random_seed)

                if task_type == "Classification":
                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=12,
                        n_informative=8,
                        n_redundant=2,
                        flip_y=min(0.25, noise_level),
                        class_sep=max(0.6, 1.8 - 1.2 * noise_level),
                        random_state=rs,
                    )
                    scorer = make_scorer(accuracy_score)
                    if model_kind == "Logistic/Ridge":
                        base = LogisticRegression(max_iter=2000, random_state=rs)
                        model = Pipeline([("scaler", StandardScaler()), ("model", base)])
                    elif model_kind == "Random Forest":
                        model = RandomForestClassifier(n_estimators=250, random_state=rs)
                    elif model_kind == "Gradient Boosting":
                        model = GradientBoostingClassifier(random_state=rs)
                    else:
                        base = SVC(C=2.0, gamma="scale", random_state=rs)
                        model = Pipeline([("scaler", StandardScaler()), ("model", base)])
                else:
                    X, y = make_regression(
                        n_samples=n_samples,
                        n_features=14,
                        n_informative=10,
                        noise=20.0 * noise_level,
                        random_state=rs,
                    )
                    scorer = make_scorer(r2_score)
                    if model_kind == "Logistic/Ridge":
                        model = Ridge(alpha=1.0)
                    elif model_kind == "Random Forest":
                        model = RandomForestRegressor(n_estimators=250, random_state=rs)
                    elif model_kind == "Gradient Boosting":
                        model = GradientBoostingRegressor(random_state=rs)
                    else:
                        base = SVR(C=3.0, gamma="scale")
                        model = Pipeline([("scaler", StandardScaler()), ("model", base)])

                sizes = np.linspace(0.1, 1.0, train_sizes)
                train_sizes_abs, train_scores, test_scores = learning_curve(
                    model,
                    X,
                    y,
                    cv=int(cv),
                    scoring=scorer,
                    train_sizes=sizes,
                    n_jobs=None,
                )

                train_mean = train_scores.mean(axis=1)
                train_std = train_scores.std(axis=1)
                test_mean = test_scores.mean(axis=1)
                test_std = test_scores.std(axis=1)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=train_sizes_abs, y=train_mean, mode="lines+markers", name="Train"))
                fig.add_trace(go.Scatter(x=train_sizes_abs, y=test_mean, mode="lines+markers", name="CV"))
                fig.update_layout(
                    title=f"Learning Curve — {task_type} / {model_kind}",
                    xaxis_title="Training samples",
                    yaxis_title="Score",
                    height=480,
                )
                st.plotly_chart(fig, use_container_width=True)

                df = pd.DataFrame(
                    {
                        "train_size": train_sizes_abs,
                        "train_mean": train_mean,
                        "train_std": train_std,
                        "cv_mean": test_mean,
                        "cv_std": test_std,
                    }
                )
                st.dataframe(df, use_container_width=True)

                render_tasks(
                    "learning_curves",
                    "Try these tasks",
                    [
                        "Run with Logistic/Ridge and notice train vs CV gap.",
                        "Increase noise and see the score ceiling drop.",
                        "Compare Random Forest vs linear model when noise is moderate.",
                        "Increase CV folds and observe variance in estimates.",
                    ],
                )
            except Exception as e:
                st.error(f"Learning curve demo failed: {e}")
                logger.error(f"Learning curve error: {e}")

    elif sim_type == "⚖️ Bias-Variance Trade-off":
        st.subheader("⚖️ Bias–Variance Trade-off (Regression)")
        st.info(
            "Increase model complexity to reduce bias, but too much complexity increases variance. "
            "This experiment shows the classic U-shaped test error curve."
        )

        left, right = st.columns(2)
        with left:
            n_samples = st.slider("Samples", 100, 2000, 400, step=50)
            max_degree = st.slider("Max polynomial degree", 1, 20, 12)
        with right:
            test_size = st.slider("Test split", 0.1, 0.5, 0.25)
            ridge_alpha = st.slider("Ridge α", 0.0, 10.0, 0.0)

        if st.button("Run Bias–Variance Experiment", type="primary", key="bv_run"):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import PolynomialFeatures
                from sklearn.linear_model import LinearRegression, Ridge
                from sklearn.metrics import mean_squared_error
                import plotly.graph_objects as go

                rs = int(random_seed)
                rng = np.random.default_rng(rs)
                X = np.linspace(-3, 3, n_samples)
                y_true = np.sin(X) + 0.25 * np.cos(2 * X)
                y = y_true + rng.normal(0, 0.2 + 0.9 * noise_level, size=n_samples)
                X = X.reshape(-1, 1)

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=float(test_size), random_state=rs
                )

                degrees = list(range(1, int(max_degree) + 1))
                train_mse = []
                test_mse = []

                for d in degrees:
                    reg = Ridge(alpha=ridge_alpha) if ridge_alpha > 0 else LinearRegression()
                    model = Pipeline([("poly", PolynomialFeatures(degree=d, include_bias=False)), ("reg", reg)])
                    model.fit(X_train, y_train)
                    train_mse.append(mean_squared_error(y_train, model.predict(X_train)))
                    test_mse.append(mean_squared_error(y_test, model.predict(X_test)))

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=degrees, y=train_mse, mode="lines+markers", name="Train MSE"))
                fig.add_trace(go.Scatter(x=degrees, y=test_mse, mode="lines+markers", name="Test MSE"))
                best_degree = degrees[int(np.argmin(test_mse))]
                fig.add_vline(x=best_degree, line_dash="dash", line_color="#f5576c")
                fig.update_layout(
                    title=f"Bias–Variance Curve (best degree ≈ {best_degree})",
                    xaxis_title="Polynomial degree",
                    yaxis_title="MSE",
                    height=480,
                )
                st.plotly_chart(fig, use_container_width=True)

                render_tasks(
                    "bias_variance",
                    "Try these tasks",
                    [
                        "Increase noise and observe the test MSE floor rise.",
                        "Increase max degree and see overfitting (test MSE increases).",
                        "Add Ridge regularization (α>0) and compare the curve shape.",
                        "Change the seed and see variability in best degree.",
                    ],
                )
            except Exception as e:
                st.error(f"Bias–variance demo failed: {e}")
                logger.error(f"Bias–variance error: {e}")

    elif sim_type == "🎲 Imbalanced Data Handling":
        st.subheader("🎲 Imbalanced Data Handling")
        st.info(
            "Accuracy can be misleading with severe class imbalance. This lab compares metrics and remedies."
        )

        left, mid, right = st.columns(3)
        with left:
            n_samples = st.slider("Samples", 200, 6000, 2000, step=200)
            minority = st.slider("Minority fraction", 0.01, 0.40, 0.08)
        with mid:
            model_kind = st.selectbox("Model", ["Logistic Regression", "Random Forest"], index=0)
            use_balanced = st.checkbox("Use class_weight=balanced", value=True)
        with right:
            threshold = st.slider("Decision threshold (prob>t)", 0.05, 0.95, 0.50)
            show_pr = st.checkbox("Show Precision–Recall curve", value=True)

        if st.button("Run Imbalance Lab", type="primary", key="imb_run"):
            try:
                from sklearn.datasets import make_classification
                from sklearn.model_selection import train_test_split
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                from sklearn.ensemble import RandomForestClassifier
                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    balanced_accuracy_score,
                    confusion_matrix,
                    precision_recall_curve,
                    average_precision_score,
                )
                import plotly.figure_factory as ff
                import plotly.graph_objects as go

                rs = int(random_seed)
                weights = [1.0 - float(minority), float(minority)]
                X, y = make_classification(
                    n_samples=n_samples,
                    n_features=20,
                    n_informative=10,
                    n_redundant=4,
                    weights=weights,
                    flip_y=min(0.08, noise_level),
                    class_sep=max(0.5, 1.8 - 1.2 * noise_level),
                    random_state=rs,
                )
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=rs, stratify=y
                )

                cw = "balanced" if use_balanced else None
                if model_kind == "Logistic Regression":
                    base = LogisticRegression(max_iter=3000, class_weight=cw, random_state=rs)
                    model = Pipeline([("scaler", StandardScaler()), ("model", base)])
                else:
                    model = RandomForestClassifier(
                        n_estimators=350, class_weight=cw, random_state=rs
                    )

                model.fit(X_train, y_train)

                try:
                    proba = model.predict_proba(X_test)[:, 1]
                    y_pred = (proba >= float(threshold)).astype(int)
                except Exception:
                    proba = None
                    y_pred = model.predict(X_test)

                acc = accuracy_score(y_test, y_pred)
                bacc = balanced_accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, zero_division=0)
                rec = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)

                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.markdown(render_stat_card("Accuracy", f"{acc:.3f}", "✅"), unsafe_allow_html=True)
                with c2:
                    st.markdown(render_stat_card("Balanced Acc", f"{bacc:.3f}", "⚖️"), unsafe_allow_html=True)
                with c3:
                    st.markdown(render_stat_card("Precision", f"{prec:.3f}", "🎯"), unsafe_allow_html=True)
                with c4:
                    st.markdown(render_stat_card("Recall", f"{rec:.3f}", "🔍"), unsafe_allow_html=True)
                with c5:
                    st.markdown(render_stat_card("F1", f"{f1:.3f}", "⭐"), unsafe_allow_html=True)

                st.write("### 📊 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig = ff.create_annotated_heatmap(
                    cm,
                    x=["Pred 0", "Pred 1"],
                    y=["True 0", "True 1"],
                    colorscale="Blues",
                )
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

                if show_pr and proba is not None:
                    st.write("### 📈 Precision–Recall Curve")
                    p, r, _ = precision_recall_curve(y_test, proba)
                    ap = average_precision_score(y_test, proba)
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=r, y=p, mode="lines", name=f"AP={ap:.3f}"))
                    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", height=420)
                    st.plotly_chart(fig, use_container_width=True)

                render_tasks(
                    "imbalanced",
                    "Try these tasks",
                    [
                        "Set minority fraction to ~0.02 and compare Accuracy vs Balanced Acc.",
                        "Toggle class_weight=balanced and watch recall change.",
                        "Lower the threshold (e.g., 0.2) and observe precision/recall tradeoff.",
                        "Use Random Forest and compare PR curve vs Logistic Regression.",
                    ],
                )
            except Exception as e:
                st.error(f"Imbalanced data lab failed: {e}")
                logger.error(f"Imbalance error: {e}")

    elif sim_type == "🔄 Cross-Validation Demo":
        st.subheader("🔄 Cross-Validation Demo")
        st.info(
            "Cross-validation gives a more reliable estimate than a single train/test split."
        )

        left, right = st.columns(2)
        with left:
            task_type = st.selectbox("Task type", ["Classification", "Regression"], index=0, key="cv_task")
            model_kind = st.selectbox(
                "Model",
                ["Linear", "Random Forest", "Gradient Boosting"],
                index=1,
                key="cv_model",
            )
        with right:
            n_samples = st.slider("Samples", 200, 6000, 1500, step=100, key="cv_samples")
            folds = st.slider("Folds", 3, 12, 5, key="cv_folds")

        if st.button("Run CV", type="primary", key="cv_run"):
            try:
                from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
                from sklearn.datasets import make_classification, make_regression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression, Ridge
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
                import plotly.express as px

                rs = int(random_seed)
                if task_type == "Classification":
                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=18,
                        n_informative=10,
                        n_redundant=4,
                        flip_y=min(0.2, noise_level),
                        random_state=rs,
                    )
                    if model_kind == "Linear":
                        model = Pipeline(
                            [
                                ("scaler", StandardScaler()),
                                ("model", LogisticRegression(max_iter=2500, random_state=rs)),
                            ]
                        )
                    elif model_kind == "Random Forest":
                        model = RandomForestClassifier(n_estimators=300, random_state=rs)
                    else:
                        model = GradientBoostingClassifier(random_state=rs)
                    cv_obj = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=rs)
                    scores = cross_val_score(model, X, y, cv=cv_obj, scoring="accuracy")
                    score_name = "accuracy"
                else:
                    X, y = make_regression(
                        n_samples=n_samples,
                        n_features=18,
                        n_informative=12,
                        noise=25.0 * noise_level,
                        random_state=rs,
                    )
                    if model_kind == "Linear":
                        model = Ridge(alpha=1.0)
                    elif model_kind == "Random Forest":
                        model = RandomForestRegressor(n_estimators=300, random_state=rs)
                    else:
                        model = GradientBoostingRegressor(random_state=rs)
                    cv_obj = KFold(n_splits=int(folds), shuffle=True, random_state=rs)
                    scores = cross_val_score(model, X, y, cv=cv_obj, scoring="r2")
                    score_name = "r2"

                st.write("### 📦 Fold scores")
                st.dataframe(pd.DataFrame({"fold": np.arange(1, len(scores) + 1), score_name: scores}))

                fig = px.box(scores, points="all", title=f"CV score distribution ({score_name})")
                st.plotly_chart(fig, use_container_width=True)

                st.markdown(
                    render_stat_card("Mean", f"{scores.mean():.3f}", "📌"), unsafe_allow_html=True
                )
                st.markdown(
                    render_stat_card("Std", f"{scores.std():.3f}", "📏"), unsafe_allow_html=True
                )

                render_tasks(
                    "cross_validation",
                    "Try these tasks",
                    [
                        "Increase folds and see score variance stabilize.",
                        "Switch Linear ↔ Random Forest and compare mean + spread.",
                        "Increase noise and watch the distribution shift.",
                        "Increase sample size and observe reduced variance.",
                    ],
                )
            except Exception as e:
                st.error(f"Cross-validation demo failed: {e}")
                logger.error(f"CV error: {e}")

    elif sim_type == "🧠 Neural Network Playground":
        st.subheader("🧠 Neural Network Playground")
        st.info(
            "Explore how neural network architecture and regularization affect decision boundaries."
        )

        left, mid, right = st.columns(3)
        with left:
            dataset_kind = st.selectbox("2D Dataset", ["Moons", "Circles", "Linear"], key="nn_ds")
            n_samples = st.slider("Samples", 200, 2000, 800, step=50, key="nn_n")
        with mid:
            layers = st.slider("Hidden layers", 1, 3, 2)
            units = st.slider("Units per layer", 4, 128, 32, step=4)
        with right:
            activation = st.selectbox("Activation", ["relu", "tanh", "logistic"], index=0)
            alpha = st.slider("L2 regularization (alpha)", 0.00001, 0.01, 0.001, format="%.5f")
            max_iter = st.slider("Max iterations", 200, 3000, 1200, step=100)

        if st.button("Train Neural Net", type="primary", key="nn_run"):
            try:
                from sklearn.neural_network import MLPClassifier
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score
                import plotly.graph_objects as go

                rs = int(random_seed)
                X, y = _make_2d_dataset(dataset_kind, int(n_samples), noise_level, rs)
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.25, random_state=rs, stratify=y
                )

                hidden = tuple([int(units)] * int(layers))
                base = MLPClassifier(
                    hidden_layer_sizes=hidden,
                    activation=activation,
                    alpha=float(alpha),
                    max_iter=int(max_iter),
                    random_state=rs,
                )
                model = Pipeline([("scaler", StandardScaler()), ("model", base)])
                model.fit(X_train, y_train)

                acc = accuracy_score(y_test, model.predict(X_test))
                st.markdown(render_stat_card("Test accuracy", f"{acc:.3f}", "🧠"), unsafe_allow_html=True)
                _plot_decision_boundary(model, X, y, title=f"NN {hidden} — {dataset_kind}")

                try:
                    loss_curve = model.named_steps["model"].loss_curve_
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=loss_curve, mode="lines", name="loss"))
                    fig.update_layout(title="Training loss curve", xaxis_title="Iteration", yaxis_title="Loss")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception:
                    pass

                render_tasks(
                    "neural_network",
                    "Try these tasks",
                    [
                        "Increase units and observe more flexible boundaries.",
                        "Switch activation relu ↔ tanh and compare convergence.",
                        "Increase alpha and see smoother (less overfit) boundaries.",
                        "Increase noise and watch accuracy drop.",
                    ],
                )
            except Exception as e:
                st.error(f"Neural network playground failed: {e}")
                logger.error(f"NN error: {e}")

    elif sim_type == "🎪 Ensemble Methods":
        st.subheader("🎪 Ensemble Methods Comparison")
        st.info(
            "Compare ensemble learners on the same dataset to understand bagging vs boosting vs forests."
        )

        left, right = st.columns(2)
        with left:
            dataset_kind = st.selectbox("Dataset", ["Classification", "Regression"], index=0)
            n_samples = st.slider("Samples", 300, 6000, 1500, step=100)
        with right:
            test_size = st.slider("Test split", 0.1, 0.5, 0.25, key="ens_test")
            scale = st.checkbox("Standardize (for linear baselines)", value=True)

        if st.button("Run Ensemble Comparison", type="primary", key="ens_run"):
            try:
                from sklearn.model_selection import train_test_split
                from sklearn.datasets import make_classification, make_regression
                from sklearn.pipeline import Pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression, Ridge
                from sklearn.ensemble import (
                    RandomForestClassifier,
                    RandomForestRegressor,
                    GradientBoostingClassifier,
                    GradientBoostingRegressor,
                    AdaBoostClassifier,
                    AdaBoostRegressor,
                    BaggingClassifier,
                    BaggingRegressor,
                    ExtraTreesClassifier,
                    ExtraTreesRegressor,
                )
                from sklearn.metrics import accuracy_score, r2_score
                import plotly.express as px

                rs = int(random_seed)
                if dataset_kind == "Classification":
                    X, y = make_classification(
                        n_samples=n_samples,
                        n_features=24,
                        n_informative=14,
                        n_redundant=6,
                        flip_y=min(0.2, noise_level),
                        class_sep=max(0.6, 1.7 - 1.1 * noise_level),
                        random_state=rs,
                    )
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=float(test_size), random_state=rs, stratify=y
                    )

                    models = {
                        "Logistic": Pipeline(
                            [("scaler", StandardScaler()), ("m", LogisticRegression(max_iter=2500, random_state=rs))]
                        )
                        if scale
                        else LogisticRegression(max_iter=2500, random_state=rs),
                        "RandomForest": RandomForestClassifier(n_estimators=350, random_state=rs),
                        "ExtraTrees": ExtraTreesClassifier(n_estimators=350, random_state=rs),
                        "Bagging": BaggingClassifier(n_estimators=250, random_state=rs),
                        "AdaBoost": AdaBoostClassifier(n_estimators=300, random_state=rs),
                        "GradBoost": GradientBoostingClassifier(random_state=rs),
                    }
                    metric_name = "accuracy"
                    scorer = lambda yt, yp: accuracy_score(yt, yp)
                else:
                    X, y = make_regression(
                        n_samples=n_samples,
                        n_features=26,
                        n_informative=16,
                        noise=30.0 * noise_level,
                        random_state=rs,
                    )
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=float(test_size), random_state=rs
                    )
                    models = {
                        "Ridge": Ridge(alpha=1.0),
                        "RandomForest": RandomForestRegressor(n_estimators=350, random_state=rs),
                        "ExtraTrees": ExtraTreesRegressor(n_estimators=350, random_state=rs),
                        "Bagging": BaggingRegressor(n_estimators=250, random_state=rs),
                        "AdaBoost": AdaBoostRegressor(n_estimators=300, random_state=rs),
                        "GradBoost": GradientBoostingRegressor(random_state=rs),
                    }
                    metric_name = "r2"
                    scorer = lambda yt, yp: r2_score(yt, yp)

                scores = []
                for name, model in models.items():
                    model.fit(X_train, y_train)
                    pred = model.predict(X_test)
                    scores.append({"model": name, metric_name: float(scorer(y_test, pred))})

                df = pd.DataFrame(scores).sort_values(metric_name, ascending=False)
                st.dataframe(df, use_container_width=True)
                fig = px.bar(df, x="model", y=metric_name, title=f"Model comparison ({metric_name})")
                st.plotly_chart(fig, use_container_width=True)

                render_tasks(
                    "ensemble",
                    "Try these tasks",
                    [
                        "Increase noise and see which ensemble is most robust.",
                        "Increase sample size and observe ranking stability.",
                        "Switch Classification ↔ Regression and compare behavior.",
                        "Compare RandomForest vs GradientBoosting and interpret why one wins.",
                    ],
                )
            except Exception as e:
                st.error(f"Ensemble comparison failed: {e}")
                logger.error(f"Ensemble error: {e}")
    
    # Footer
    st.markdown(
        """
        ---
        ### 💡 Tips for Using Simulations
        - If results look unstable, increase dataset size or use cross-validation.
        - If Train score ≫ CV score, you're likely overfitting (reduce complexity / add regularization).
        - If both Train and CV are low, you're underfitting (increase model capacity).
        - For imbalanced data, track **recall** / **balanced accuracy**, not only accuracy.
        - Use random seed for reproducible experiments.
        """
    )


if __name__ == "__main__":
    main()
