import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.core.config import load_config
from src.core.ui import app_header, instruction_block, sidebar_dataset_status, page_navigation
from src.core.standardized_ui import (
    standard_section_header,
    beginner_tip,
    concept_explainer,
    common_mistakes_panel,
)
from src.data.samples import describe_sample_dataset, get_sample_datasets
from src.core.styles import inject_custom_css
from src.academy.real_world_labs import LABS, get_lab

# --- Page Setup ---
st.set_page_config(page_title="DataScope Pro - Academy", layout="wide", initial_sidebar_state="expanded")
config = load_config()
inject_custom_css()

# --- Header ---
app_header(
    config,
    page_title="Data Science Academy",
    subtitle="A complete interactive curriculum: From Python basics to production-grade ML. Learn by doing with real-world labs and copy-pasteable best practices",
    icon="🎓"
)

# --- Outcomes ---
st.markdown("### 🎯 Learning Outcomes")
# --- Learning Guide ---
standard_section_header("Academy Guide & Tips", "🎓")
concept_explainer(
    title="Learn by Doing",
    explanation=(
        "Follow the curriculum from EDA to deployment. Each lab shows real-world tasks with datasets, code, and explanations to build practical skills."
    ),
    real_world_example=(
        "Churn analysis: Clean customer data, engineer tenure features, compare classifiers, explain results to stakeholders, and deploy prediction."
    ),
)
beginner_tip("Tip: Save your cleaned data and reuse it across modules to keep consistency.")
common_mistakes_panel({
    "Skipping practice": "Apply concepts in labs; repetition builds confidence.",
    "Unorganized notes": "Document decisions, parameters, and outcomes.",
    "Ignoring metrics": "Tie learning to measurable improvements.",
    "No project goal": "Define success criteria for each lab.",
})
outcomes_col1, outcomes_col2, outcomes_col3 = st.columns(3)
with outcomes_col1:
    st.markdown(
        """
        <div style="background: #f0f9ff; padding: 16px; border-radius: 10px; height: 100%;">
            <div style="font-size: 32px; text-align: center; margin-bottom: 8px;">📊</div>
            <div style="font-weight: 700; color: #0369a1; text-align: center; margin-bottom: 8px;">Data Mastery</div>
            <ul style="font-size: 14px; color: #0c4a6e; margin: 0; padding-left: 20px;">
                <li>Load & clean messy datasets</li>
                <li>Handle missing values & outliers</li>
                <li>Feature engineering patterns</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with outcomes_col2:
    st.markdown(
        """
        <div style="background: #f0fdf4; padding: 16px; border-radius: 10px; height: 100%;">
            <div style="font-size: 32px; text-align: center; margin-bottom: 8px;">📈</div>
            <div style="font-weight: 700; color: #15803d; text-align: center; margin-bottom: 8px;">Visualization Pro</div>
            <ul style="font-size: 14px; color: #14532d; margin: 0; padding-left: 20px;">
                <li>Interactive Plotly charts</li>
                <li>Statistical story-telling</li>
                <li>Faceted & 3D analysis</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )
with outcomes_col3:
    st.markdown(
        """
        <div style="background: #faf5ff; padding: 16px; border-radius: 10px; height: 100%;">
            <div style="font-size: 32px; text-align: center; margin-bottom: 8px;">🤖</div>
            <div style="font-weight: 700; color: #7c3aed; text-align: center; margin-bottom: 8px;">Production AI</div>
            <ul style="font-size: 14px; color: #581c87; margin: 0; padding-left: 20px;">
                <li>End-to-end ML pipelines</li>
                <li>Model evaluation & tuning</li>
                <li>Drift detection & deployment</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- Python Fundamentals ---
st.subheader("🐍 Python Fundamentals: The Basics")
fund_tabs = st.tabs(["Syntax & Variables", "Data Types", "Loops & Logic", "Functions"])
with fund_tabs[0]:
    st.code("""
# Variables are containers for storing data values
message = "Hello Data Science"  # String
count = 3                       # Integer
price = 19.99                   # Float

print(message)
print(f"Total cost: {count * price}")
    """, language="python")
with fund_tabs[1]:
    st.code("""
# Common Data Structures
numbers = [1, 2, 3, 4, 5]               # List (mutable)
coordinates = (10.0, 20.0)              # Tuple (immutable)
user = {"id": 1, "name": "Alice"}       # Dictionary (key-value)
unique_ids = {101, 102, 103}            # Set (unique values)

print(user["name"])
    """, language="python")
with fund_tabs[2]:
    st.code("""
# Control Flow
sales = [120, 80, 0, 150]

for s in sales:
    if s == 0:
        print("Warning: No sales!")
    elif s > 100:
        print(f"Great day: {s}")
    else:
        print(f"Average day: {s}")
    """, language="python")
with fund_tabs[3]:
    st.code("""
def calculate_growth(current, previous):
    if previous == 0:
        return 0.0
    return (current - previous) / previous

growth = calculate_growth(150, 100)
print(f"Growth: {growth:.1%}")
    """, language="python")

# --- Detailed Code Library ---
st.subheader("💻 The 10M$ Code Library: Production Patterns")
st.caption("Copy-pasteable, production-ready code snippets for every stage of the data pipeline.")

code_tabs = st.tabs([
    "🧹 Data Cleaning",
    "📊 Transformation",
    "🔍 EDA Patterns",
    "🎯 Feature Eng.",
    "📈 Visualization",
    "🤖 ML Workflows",
    "🛡️ Advanced"
])

with code_tabs[0]:  # Cleaning
    st.markdown("### 1. Robust Data Cleaning")
    with st.expander("Handling Missing Data Strategies", expanded=True):
        st.code("""
# Strategy 1: Smart Filling
df['numeric_col'] = df['numeric_col'].fillna(df['numeric_col'].median())
df['category_col'] = df['category_col'].fillna(df['category_col'].mode()[0])

# Strategy 2: Grouped Imputation (Fill age by title/class)
df['age'] = df.groupby('pclass')['age'].transform(lambda x: x.fillna(x.median()))

# Strategy 3: Forward/Backward Fill (Time Series)
df['stock_price'] = df['stock_price'].ffill()

# Strategy 4: Drop if Critical
df.dropna(subset=['target_variable'], inplace=True)
        """, language="python")

    with st.expander("Outlier Detection & Handling", expanded=False):
        st.code("""
# IQR Method (Robust)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers instead of removing
df['value_capped'] = df['value'].clip(lower_bound, upper_bound)

# Mark outliers
df['is_outlier'] = ~df['value'].between(lower_bound, upper_bound)
        """, language="python")

with code_tabs[1]:  # Transformation
    st.markdown("### 2. Data Transformation")
    st.code("""
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Scaling (Crucial for KNN, SVM, Neural Nets)
scaler = StandardScaler()
df['scaled_value'] = scaler.fit_transform(df[['raw_value']])

# Log Transform (Fix Skewness)
df['log_revenue'] = np.log1p(df['revenue'])

# Binning (Continuous to Categorical)
df['age_group'] = pd.cut(df['age'], bins=[0, 18, 35, 65, 100], labels=['Child', 'Young Adult', 'Adult', 'Senior'])
    """, language="python")

with code_tabs[2]:  # EDA
    st.markdown("### 3. Exploratory Data Analysis Patterns")
    st.code("""
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Distribution Check
sns.histplot(data=df, x='price', kde=True)
plt.title('Price Distribution with Density')

# 2. Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

# 3. Categorical Boxplot (Price by Category)
sns.boxplot(data=df, x='category', y='price')

# 4. Pairplot (Quick overview)
sns.pairplot(df[['price', 'age', 'rating']], diag_kind='kde')
    """, language="python")

with code_tabs[3]:  # Feature Eng
    st.markdown("### 4. Feature Engineering")
    st.code("""
# 1. Date Features
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['is_weekend'] = df['date'].dt.dayofweek >= 5

# 2. Interaction Features
df['price_per_sqft'] = df['price'] / df['sqft']
df['quality_score'] = df['condition'] * df['grade']

# 3. Text Features (Simple)
df['desc_len'] = df['description'].str.len()
df['has_keyword'] = df['description'].str.contains('luxury', case=False).astype(int)

# 4. Lag Features (Time Series)
df['prev_day_sales'] = df['sales'].shift(1)
df['sales_7d_avg'] = df['sales'].rolling(7).mean()
    """, language="python")

with code_tabs[4]:  # Viz
    st.markdown("### 5. Advanced Visualization (Plotly)")
    st.code("""
import plotly.express as px

# Interactive Scatter with Trendline
fig = px.scatter(df, x="gdp_per_capita", y="life_exp", 
                 color="continent", size="pop", hover_name="country",
                 log_x=True, size_max=60, trendline="ols",
                 title="Wealth vs Health by Country")
fig.show()

# Animated Time Series
fig = px.scatter(df, x="gdp", y="life_exp", animation_frame="year", 
                 animation_group="country", size="pop", color="continent", 
                 hover_name="country", log_x=True, size_max=55, 
                 range_x=[100,100000], range_y=[25,90])
fig.show()

# 3D Plot
fig = px.scatter_3d(df, x='sepal_length', y='sepal_width', z='petal_width',
                    color='species')
fig.show()
    """, language="python")

with code_tabs[5]:  # ML
    st.markdown("### 6. Machine Learning Workflows")
    with st.expander("Full Classification Pipeline", expanded=True):
        st.code("""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 1. Split
X = df.drop('churn', axis=1)
y = df['churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# 2. Pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('model', RandomForestClassifier(n_estimators=100))
])

# 3. Train
pipeline.fit(X_train, y_train)

# 4. Evaluate
preds = pipeline.predict(X_test)
print(classification_report(y_test, preds))
        """, language="python")

with code_tabs[6]:  # Advanced
    st.markdown("### 7. Advanced / Production")
    st.code("""
# 1. Pipeline Feature Importance
model = pipeline.named_steps['model']
importances = pd.Series(model.feature_importances_, index=X.columns)
importances.nlargest(10).plot(kind='barh')

# 2. Save Model (Pickle/Joblib)
import joblib
joblib.dump(pipeline, 'model_v1.joblib')

# 3. Load & Predict
loaded_model = joblib.load('model_v1.joblib')
new_preds = loaded_model.predict(new_data)
    """, language="python")

st.markdown("---")

# --- Interactive Labs Section ---
st.subheader("🏆 Real-World End-to-End Labs")
st.markdown("Select a scenario to generate a realistic, messy dataset and solve a business problem.")

selected_lab = st.selectbox(
    "Choose a Lab Scenario:",
    list(LABS.keys()),
    format_func=lambda x: f"🧪 {x}: {LABS[x]['description']}"
)

# Lab Controls
col_lab_1, col_lab_2 = st.columns([1, 3])
with col_lab_1:
    if st.button("🚀 Launch Lab", type="primary", width="stretch"):
        st.session_state['current_lab'] = selected_lab
        with st.spinner(f"Generating realistic {selected_lab} data..."):
            st.session_state['lab_df'] = get_lab(selected_lab)
        st.success("Data Generated!")

with col_lab_2:
    if 'current_lab' in st.session_state:
        st.info(f"**Active Lab:** {st.session_state['current_lab']} | Rows: {len(st.session_state['lab_df'])}")

if 'lab_df' in st.session_state and 'current_lab' in st.session_state:
    df_lab = st.session_state['lab_df']
    lab_meta = LABS[st.session_state['current_lab']]
    
    st.divider()
    
    # Lab Workspace
    lab_tabs = st.tabs(["1. Inspect & Clean", "2. Explore (EDA)", "3. Model & Solve", "4. Business Insights"])
    
    with lab_tabs[0]:
        st.markdown(f"### 🧐 Inspect: {st.session_state['current_lab']}")
        st.markdown(f"**Key Issues to Fix:** {', '.join(lab_meta['key_issues'])}")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**First 5 rows:**")
            st.dataframe(df_lab.head(), width="stretch")
        with c2:
            st.markdown("**Data Types & Missing:**")
            info_df = pd.DataFrame({
                'Type': df_lab.dtypes.astype(str),
                'Missing': df_lab.isna().sum(),
                'Missing %': (df_lab.isna().mean() * 100).round(1)
            })
            st.dataframe(info_df, width="stretch")
            
        st.warning("⚠️ Action Required: Look for missing values in critical columns and outliers in numeric fields.")

    with lab_tabs[1]:
        st.markdown("### 📊 Explore Patterns")
        numeric_cols = df_lab.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df_lab.select_dtypes(include='object').columns.tolist()
        
        col_eda_1, col_eda_2 = st.columns(2)
        
        with col_eda_1:
            if numeric_cols:
                y_axis = st.selectbox("Select Numeric Variable", numeric_cols, key="lab_num_y")
                fig_hist = px.histogram(df_lab, x=y_axis, title=f"Distribution of {y_axis}", marginal="box")
                st.plotly_chart(fig_hist, width="stretch")
                
        with col_eda_2:
            if len(numeric_cols) > 1:
                x_axis = st.selectbox("Select X Axis", numeric_cols, key="lab_num_x")
                fig_scat = px.scatter(df_lab, x=x_axis, y=y_axis, title=f"{x_axis} vs {y_axis}")
                st.plotly_chart(fig_scat, width="stretch")
                
        if len(numeric_cols) > 1:
            st.markdown("**Correlation Heatmap**")
            corr = df_lab[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Matrix")
            st.plotly_chart(fig_corr, width="stretch")

    with lab_tabs[2]:
        st.markdown("### 🤖 Modeling Sandbox")
        st.info("This is a simplified modeling playground effectively demonstrating the workflow.")
        
        target = st.selectbox("Select Target Variable", df_lab.columns)
        predictors = st.multiselect("Select Predictors", [c for c in df_lab.columns if c != target], default=[c for c in numeric_cols if c != target][:3])
        
        if st.button("Train Model"):
            if not predictors:
                st.error("Please select at least one predictor.")
            else:
                try:
                    # Very simple prep for demo
                    X = df_lab[predictors].dropna()
                    y = df_lab.loc[X.index, target]
                    
                    # Handle categorization blindly for demo
                    X = pd.get_dummies(X, drop_first=True)
                    
                    if pd.api.types.is_numeric_dtype(y):
                        from sklearn.linear_model import LinearRegression
                        from sklearn.metrics import r2_score, mean_absolute_error
                        
                        model = LinearRegression()
                        model.fit(X, y)
                        preds = model.predict(X)
                        
                        st.success("Regression Model Trained!")
                        c1, c2 = st.columns(2)
                        c1.metric("R² Score", f"{r2_score(y, preds):.3f}")
                        c2.metric("MAE", f"{mean_absolute_error(y, preds):.2f}")
                        
                        fig = px.scatter(x=y, y=preds, labels={'x': 'Actual', 'y': 'Predicted'}, title="Actual vs Predicted")
                        fig.add_shape(type="line", x0=y.min(), y0=y.max(), x1=y.min(), y1=y.max(), line=dict(color="Red", dash="dash"))
                        st.plotly_chart(fig, width="stretch")
                        
                    else:
                        from sklearn.ensemble import RandomForestClassifier
                        from sklearn.metrics import accuracy_score
                        
                        model = RandomForestClassifier(n_estimators=10)
                        y = y.astype(str) # Force cat
                        model.fit(X, y)
                        preds = model.predict(X)
                        
                        st.success("Classification Model Trained!")
                        st.metric("Accuracy", f"{accuracy_score(y, preds):.1%}")
                        
                except Exception as e:
                    st.error(f"Modeling Error (Auto-prep failed): {str(e)}")

    with lab_tabs[3]:
        st.markdown("### 💡 Strategic Insights")
        st.markdown(
            f"""
            **Based on the {st.session_state['current_lab']} scenario, answer these questions:**
            
            1. **Data Quality:** Did you find the injected outliers or missing values? How did they affect the stats?
            2. **Key Drivers:** Which features had the strongest correlation with the target?
            3. **Business Action:** If the model predicts X, what should the business *do* differently?
            """
        )
        st.text_area("Write your analysis here:", height=150)

st.markdown("---")
page_navigation("12")
