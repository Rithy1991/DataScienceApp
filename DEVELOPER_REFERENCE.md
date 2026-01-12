# DataScope Pro - Developer Quick Reference Guide

## 🏗️ PROJECT ARCHITECTURE

### **Directory Structure**
```
/
├── app.py                           # Home page - data upload & preview
├── config.yaml                      # App configuration (title, theme, etc.)
├── requirements.txt                 # Python dependencies
├── pages/                           # Streamlit pages (auto-converted to menu items)
│   ├── 1_DS_Assistant.py           # Workflow guide & problem selector
│   ├── 2_Data_Analysis_EDA.py      # Exploratory data analysis
│   ├── 3_Data_Cleaning.py          # Data cleaning & preprocessing
│   ├── 4_Feature_Engineering.py    # Feature scaling, encoding, selection
│   ├── 5_Tabular_Machine_Learning.py # Advanced ML (XGBoost, LightGBM)
│   ├── 7_Visualization.py          # Custom chart builder
│   ├── 9_Prediction.py             # Batch/real-time prediction
│   ├── 10_AI_Insights.py           # AI explanation panel
│   ├── 12_DS_Academy.py            # Learning tutorials
│   ├── 13_Settings.py              # Configuration
│   ├── 14_Classification_Learning.py # Beginner classification
│   ├── 15_Clustering_Learning.py   # KMeans clustering
│   ├── 16_Regression_Learning.py   # Beginner regression
│   ├── 17_Demo_Workflow.py         # Stakeholder demo
│   └── 18_Sample_Report.py         # Export & reporting
├── src/
│   ├── core/
│   │   ├── config.py               # Configuration management
│   │   ├── ui.py                   # Navigation, headers, UI utilities
│   │   ├── standardized_ui.py      # Reusable UI components (NEW)
│   │   └── state.py                # Session state helpers
│   ├── data/
│   │   ├── loader.py               # Data loading (CSV, Excel, samples)
│   │   ├── preprocessor.py         # Cleaning, encoding, scaling
│   │   └── sampler.py              # Sample dataset generation
│   ├── ml/
│   │   ├── models.py               # Model wrappers
│   │   ├── metrics.py              # Evaluation metrics
│   │   ├── explainer.py            # AI explanations
│   │   └── registry.py             # Model registry
│   └── storage/
│       ├── local.py                # Local file storage
│       └── models.json             # Model metadata
└── artifacts/                       # Trained models & checkpoints
```

---

## 🎯 PAGE NUMBERING CONVENTION

| Number | File | Purpose | Order |
|--------|------|---------|-------|
| 0 | app.py | Home | 0 |
| 1 | 1_DS_Assistant.py | Workflow | 1 |
| 2 | 2_Data_Analysis_EDA.py | EDA | 2 |
| 3 | 3_Data_Cleaning.py | Cleaning | 3 |
| 4 | 4_Feature_Engineering.py | Features | 4 |
| 5 | **14_Classification** | Beginner Classification | **5** |
| 6 | **16_Regression** | Beginner Regression | **6** |
| 7 | **15_Clustering** | Beginner Clustering | **7** |
| 8 | 5_Tabular_ML | Advanced ML | 8 |
| 9 | 10_AI_Insights | Explanations | 9 |
| 10 | 9_Prediction | Prediction | 10 |
| 11 | 7_Visualization | Charts | 11 |
| 12 | 18_Sample_Report | Export | 12 |
| 13 | 17_Demo_Workflow | Demo | 13 |
| 14 | 12_DS_Academy | Academy | 14 |
| 15 | 13_Settings | Settings | 15 |

**Note**: File names don't reflect order. Navigation order is defined in [src/core/ui.py](src/core/ui.py) `page_navigation()` function.

---

## 📦 KEY DEPENDENCIES

### **Core Framework**
```
streamlit==1.28.0          # Web framework
pandas==2.0.3              # Data manipulation
numpy==1.24.3              # Numerical computing
scipy==1.11.0              # Statistical functions
```

### **Machine Learning**
```
scikit-learn==1.3.0        # Core ML algorithms
xgboost==2.0.0             # XGBoost
lightgbm==4.0.0            # LightGBM
joblib==1.3.0              # Model serialization
```

### **Visualization**
```
plotly==5.16.1             # Interactive charts
matplotlib==3.8.0          # Static plots
seaborn==0.12.0            # Statistical plots
```

### **AI/NLP**
```
transformers==4.30.0       # HuggingFace models (optional)
torch==2.0.0               # PyTorch (optional)
openai==0.28.0             # OpenAI API (optional)
```

---

## 🚀 QUICK START FOR DEVELOPERS

### **1. Setup Development Environment**
```bash
# Clone repository
git clone <repo-url>
cd datascope-pro

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

### **2. Add New Page**
```python
# Create pages/20_New_Feature.py
import streamlit as st
from src.core.ui import app_header

st.set_page_config(layout="wide")
app_header(st.session_state.config, "New Feature", "Description")

# Your content here
```

Then update [src/core/ui.py](src/core/ui.py) `page_navigation()` to add to menu.

### **3. Use Session State**
```python
# Store raw dataframe
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None

# Access from any page
df = st.session_state.raw_df
```

### **4. Add New ML Model**
Edit [pages/14_Classification_Learning.py](pages/14_Classification_Learning.py):
```python
from sklearn.ensemble import GradientBoostingClassifier

models = {
    'Logistic Regression': LogisticRegression(),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Gradient Boosting': GradientBoostingClassifier()  # ADD THIS
}
```

### **5. Add New Visualization**
```python
import plotly.graph_objects as go

fig = go.Figure(data=[go.Bar(x=['A', 'B'], y=[1, 2])])
fig.update_layout(title="My Chart")
st.plotly_chart(fig, use_container_width=True)
```

---

## 💡 COMMON PATTERNS & BEST PRACTICES

### **1. Session State for Data Persistence**
```python
# Initialize
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None

# Set (in upload page)
st.session_state.raw_df = pd.read_csv(uploaded_file)

# Use (in other pages)
df = st.session_state.raw_df
if df is not None:
    st.write(df.shape)
```

### **2. Caching for Performance**
```python
# Cache data loading
@st.cache_data
def load_sample_data(name):
    return pd.read_csv(f"data/{name}.csv")

# Cache ML computations
@st.cache_resource
def train_model(X, y, model_type):
    model = get_model(model_type)
    return model.fit(X, y)
```

### **3. Error Handling**
```python
try:
    result = some_operation()
except ValueError as e:
    st.error(f"Invalid input: {str(e)}")
except Exception as e:
    st.error(f"Unexpected error: {str(e)}")
    st.caption("This shouldn't happen. Please contact support.")
```

### **4. Expandable Sections**
```python
with st.expander("Advanced Options", expanded=False):
    param1 = st.slider("Param 1", 0, 100, 50)
    param2 = st.slider("Param 2", 0.0, 1.0, 0.5)
```

### **5. Columns for Layout**
```python
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", 0.92)
with col2:
    st.metric("F1-Score", 0.89)
```

### **6. Display DataFrames Safely**
```python
# Always check before displaying
if df is not None and not df.empty:
    st.dataframe(df.head(10), use_container_width=True)
else:
    st.info("No data available")
```

### **7. Progress Indicators**
```python
# Long operations
with st.spinner("Training model..."):
    model = train_large_model(X, y)

st.success("Model trained successfully!")
```

### **8. Before/After Comparison**
```python
col1, col2 = st.columns(2)
with col1:
    st.subheader("Before Cleaning")
    st.dataframe(raw_df.head())
with col2:
    st.subheader("After Cleaning")
    st.dataframe(clean_df.head())
```

---

## 📊 DATA FLOW DIAGRAM

```
┌─────────────────┐
│   Upload Data   │ (app.py)
│  (CSV/Excel)    │
└────────┬────────┘
         │ st.session_state['raw_df']
         ▼
┌─────────────────┐
│  Data Preview   │ (app.py)
│ + Quick Stats   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Data EDA      │ (pages/2_Data_Analysis_EDA.py)
│  Distributions  │
│  Correlations   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Cleaning  │ (pages/3_Data_Cleaning.py)
│  + Preprocessing│ st.session_state['clean_df']
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Feature        │ (pages/4_Feature_Engineering.py)
│  Engineering    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Model Task     │ (pages/1_DS_Assistant.py)
│  Selection      │ Classification/Regression/Clustering
└────────┬────────┘
         │
    ┌────┴────────────────────┐
    │                         │
    ▼                         ▼
┌──────────────┐       ┌──────────────┐
│Classification│       │ Regression/  │
│ Learning     │       │ Clustering   │
│ (Beginner)   │       │ (Beginner)   │
└──────┬───────┘       └──────┬───────┘
       │                      │
       └──────────┬───────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Train Model    │ (pages/5_Tabular_ML.py)
         │  (Advanced)     │ (or learning pages)
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │  Evaluate Model │
         │  (Metrics,      │
         │   Confusion     │
         │   Matrix)       │
         └────────┬────────┘
                  │
                  ▼
         ┌─────────────────┐
         │ AI Explanations │ (pages/10_AI_Insights.py)
         │ (Plain English) │
         └────────┬────────┘
                  │
         ┌────────┴────────┐
         │                 │
         ▼                 ▼
    ┌─────────┐       ┌──────────┐
    │Prediction│      │Visualization│
    │         │       │& Export  │
    └─────────┘       └──────────┘
```

---

## 🔧 DEBUGGING COMMON ISSUES

### **Issue: "Streamlit app crashed"**
```python
# Check session state initialization
if 'raw_df' not in st.session_state:
    st.session_state.raw_df = None

# Always validate before using
if st.session_state.raw_df is None:
    st.error("Please upload data first")
    st.stop()
```

### **Issue: "Unbound variable" error**
```python
# Initialize all variables at start of page
df = None
model = None
predictions = None

# Later...
if df is not None:  # Always check
    predictions = model.predict(df)
```

### **Issue: "Could not convert string to float"**
```python
# Encode categorical columns before ML
for col in df.select_dtypes(include=['object']).columns:
    df[col] = pd.factorize(df[col])[0]

# Or use sklearn
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df[col] = le.fit_transform(df[col])
```

### **Issue: "Cache key collision"**
```python
# Use explicit hash_funcs for unhashable types
@st.cache_data(hash_funcs={pd.DataFrame: lambda x: x.shape})
def process_data(df):
    return df.copy()
```

---

## 📈 TESTING CHECKLIST

- [ ] Data upload (CSV, Excel)
- [ ] Sample data loading
- [ ] Data preview showing correct types
- [ ] Missing value detection
- [ ] Data cleaning operations
- [ ] Feature engineering (encoding, scaling)
- [ ] Classification model training
- [ ] Regression model training
- [ ] Clustering model training
- [ ] Metrics display
- [ ] Confusion matrix visualization
- [ ] Prediction on new data
- [ ] Chart generation
- [ ] Export to CSV
- [ ] Navigation between pages
- [ ] Session state persistence
- [ ] Mobile responsiveness
- [ ] Error handling

---

## 🎨 STYLING & THEMES

### **Use Standardized Components** (NEW)
```python
from src.core.standardized_ui import (
    standard_page_header,
    standard_section_header,
    beginner_tip,
    concept_explainer,
    before_after_comparison,
    model_explanation_panel,
    common_mistakes_panel,
    metric_card
)

# Page header
standard_page_header(
    title="Classification Learning",
    subtitle="Learn to classify with simple models",
    icon="🧑‍🎓"
)

# Section
standard_section_header("Step 1: Prepare Data", "📊")

# Tip
beginner_tip("Always split your data before training!")

# Comparison
before_after_comparison(raw_df, clean_df)

# Mistakes
common_mistakes_panel([
    "Using test data to train the model",
    "Not scaling features when needed",
    "Ignoring class imbalance"
])
```

### **Color Scheme**
- Primary: `#f093fb` → `#f5576c` (gradient)
- Success: `#2ecc71`
- Error: `#e74c3c`
- Info: `#3498db`
- Warning: `#f39c12`

---

## 📋 CODE QUALITY STANDARDS

- Use type hints: `def load_data(path: str) -> pd.DataFrame:`
- Document functions with docstrings
- Keep functions small and focused
- Use meaningful variable names
- Add comments for complex logic
- Handle errors gracefully
- Cache expensive operations
- Use session state for persistence

---

## 🚢 DEPLOYMENT CHECKLIST

- [ ] All pages load without errors
- [ ] Data flow works end-to-end
- [ ] Session state persists correctly
- [ ] Caching reduces redundant computations
- [ ] Error messages are user-friendly
- [ ] Mobile responsiveness verified
- [ ] Performance metrics good (<2s per action)
- [ ] Models save/load correctly
- [ ] Export functions work
- [ ] Navigation is intuitive
- [ ] Security checks passed (file validation)
- [ ] Documentation updated

---

## 📚 USEFUL RESOURCES

- **Streamlit Docs**: https://docs.streamlit.io
- **scikit-learn**: https://scikit-learn.org/stable/
- **Plotly**: https://plotly.com/python/
- **Pandas**: https://pandas.pydata.org/docs/
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/

---

## 🤝 CONTRIBUTING

1. Create a feature branch: `git checkout -b feature/new-feature`
2. Make changes with clear commit messages
3. Test thoroughly on multiple datasets
4. Update documentation
5. Submit PR with description

---

## 📞 SUPPORT & CONTACT

For questions or issues:
1. Check existing issues on GitHub
2. Review this documentation
3. Contact dev team: dev@datascope.pro

---

*Last Updated: January 10, 2026*
*Version: 1.0 | Status: PRODUCTION READY*
