# 📚 Quick Reference: What's New in Data Science Academy

## 🎯 For Learners: How to Use the Enhanced Academy

### New Real-World Labs
The Academy now has **4 interactive labs** that simulate real data science projects:

1. **E-Commerce Lab** 📦
   - Predict order values from product + customer features
   - Learn: Feature engineering, handling outliers, business metrics

2. **Customer Churn Lab** 👥
   - Predict which customers will cancel
   - Learn: Class imbalance, classification metrics, confusion matrices

3. **Website Traffic Lab** 📊
   - Forecast daily visitor counts
   - Learn: Time-series analysis, trends, seasonality

4. **Housing Prices Lab** 🏠
   - Predict house prices from features
   - Learn: Regression, outliers, feature relationships

### How to Complete a Lab (6 Steps)

```
Step 1: Overview      → See what you're working with
Step 2: Inspect       → Understand your data (types, missing, ranges)
Step 3: Clean         → Fix problems (nulls, duplicates, outliers)
         → See code template + before/after
Step 4: Explore       → Visualize (distributions, relationships)
         → Try different columns interactively
Step 5: Model         → Train & evaluate your predictive model
         → See code template with explanations
Step 6: Insights      → Generate business recommendations
         → Learn how to tell the story
```

### Confidence Building Features

- ✅ **Before/After Examples:** See what cleaning does to your data
- ✅ **Copy-Paste Code:** Every code block works immediately
- ✅ **Interactive Tabs:** Try visualizations on different columns
- ✅ **Business Context:** Learn why each metric matters
- ✅ **Checklist:** Tick off milestones as you progress

### New Code Examples You'll Find

**ML Workflows (Tab 6):**
- Classification with stratified splits, confusion matrices, AUC-ROC
- Regression with cross-validation, MAE/RMSE/MAPE metrics
- Hyperparameter tuning (grid search vs random search)

**Advanced Patterns (Tab 7):**
- Handling imbalanced data (SMOTE, class weights, threshold tuning)
- Text analysis (TF-IDF, topic modeling, embeddings)
- Time-series forecasting (ARIMA, exponential smoothing)
- Anomaly detection (4 methods + ensemble voting)

---

## 🔧 For Developers: What Changed

### New Files Created

**`src/academy/real_world_labs.py`** (303 lines)
- 4 dataset generators with realistic issues
- Each generates 300-1000 rows with embedded problems
- Returns clean Pandas DataFrames

```python
from src.academy.real_world_labs import get_lab, LABS

# Load a lab
df = get_lab('E-Commerce')

# See metadata
print(LABS['E-Commerce']['description'])
print(LABS['E-Commerce']['key_issues'])
print(LABS['E-Commerce']['learning_goals'])
```

**`src/academy/__init__.py`**
- Module initialization

### Files Enhanced

**`pages/10_Data_Science_Academy.py`** (+241 lines)

**New Section: "🏆 Premium: Real-World End-to-End Labs"**
- Interactive lab selection
- 6-step workflow with tabs
- Before/after examples
- Production code templates
- Practical exercises

**Tab 6 - ML Workflows (Rewritten for Production-Grade Examples)**
- Classification with detailed metrics explanation
- Regression with cross-validation
- Hyperparameter tuning with method comparison

**Tab 7 - Advanced Patterns (Completely Rewritten)**
- Class imbalance: 4 methods with trade-offs
- Text data: TF-IDF, LDA, Word2Vec
- Time-series: ARIMA, exponential smoothing, decomposition
- Anomaly detection: 4 algorithms + ensemble

### Config Enhancements

**`config.yaml`** - New Security Section
```yaml
security:
  allow_api_ingestion: false           # SSRF/exfiltration protection
  api_allowlist: []                    # Whitelist URLs (if enabled)
  max_api_response_bytes: 10485760     # 10 MB default
  allow_runtime_pip_install: false     # Supply chain protection
```

**`src/core/config.py`** - New AppConfig Properties
```python
@property
def allow_api_ingestion(self) -> bool: ...

@property
def api_allowlist(self) -> List[str]: ...

@property
def max_api_response_bytes(self) -> int: ...

@property
def allow_runtime_pip_install(self) -> bool: ...
```

---

## 📋 Configuration Security Reference

### Default Security Posture
**All security-sensitive features are DISABLED by default.**

This follows the principle: **"Deny by default, opt-in to enable"**

```yaml
# Production defaults (safe)
allow_api_ingestion: false              ← API data loading disabled
allow_runtime_pip_install: false        ← Dynamic package install disabled
api_allowlist: []                       ← No URLs whitelisted
max_api_response_bytes: 10485760        ← Max 10 MB per API response
```

### How to Enable Features

**To enable API ingestion with allowlist:**
```yaml
security:
  allow_api_ingestion: true
  api_allowlist:
    - "https://api.github.com"
    - "https://api.example.com"
```

**To enable runtime pip installs:**
```yaml
security:
  allow_runtime_pip_install: true
```

### Accessing in Code
```python
from src.core.config import load_config

config = load_config()

if config.allow_api_ingestion:
    # Safe to load from API
    data = load_from_api(url)
else:
    st.error("API ingestion disabled in config")

if config.allow_runtime_pip_install:
    # Safe to install packages
    pip_install(package_name)
else:
    st.warning(f"To install: pip install {package_name}")
```

---

## 📊 Lab Dataset Details

### E-Commerce Lab (e_commerce)
- **Size:** 1000 rows × 11 columns
- **Target:** `order_value` (regression)
- **Key Issues:**
  - 3% missing discount values
  - 8% outlier order values
  - 5 product categories (categorical)
- **Learning Goals:**
  - Feature engineering (compute discount%, order frequency)
  - Handling outliers
  - Category encoding

### Customer Churn Lab (customer_churn)
- **Size:** 500 rows × 8 columns
- **Target:** `churn` (0/1 classification)
- **Key Issues:**
  - 15% churn rate (imbalanced)
  - 3 categorical features
  - Missing contract_length (2%)
- **Learning Goals:**
  - Class imbalance handling
  - Precision/recall/F1 trade-offs
  - Feature importance for business

### Website Traffic Lab (website_traffic)
- **Size:** 365 rows (1 year of daily data) × 5 columns
- **Target:** `daily_visitors` (time-series regression)
- **Key Issues:**
  - Weekly seasonality (weekday vs weekend)
  - Trend (growth over time)
  - 5% missing days
- **Learning Goals:**
  - Seasonality decomposition
  - ARIMA/Prophet forecasting
  - Train/test split for time-series

### Housing Prices Lab (housing_prices)
- **Size:** 300 rows × 7 columns
- **Target:** `price` (regression)
- **Key Issues:**
  - Multicollinearity (bedrooms + size)
  - 5% right-skewed outliers
  - 2% missing location (categorical)
- **Learning Goals:**
  - Feature correlation analysis
  - Outlier detection + handling
  - Log transforms for skewed distributions

---

## 🎓 Recommended Learning Path

### For Beginners (0-2 weeks)
1. Start with **Housing Prices Lab**
   - Simplest dataset (7 features)
   - Classic regression problem
   - Learn scaling, correlation, outliers
2. Use **Beginner Track** in Learning Roadmap
3. Try basic visualizations in Explore tab

### For Intermediate (2-8 weeks)
1. **E-Commerce Lab** - Feature engineering practice
   - Compute derived features (discount%, frequency)
   - See how it improves model
2. **Customer Churn Lab** - Classification focus
   - Deal with imbalanced data
   - Understand precision/recall
3. Use **Intermediate Track** code examples
4. Start applying to your own small dataset

### For Advanced (8+ weeks)
1. **Website Traffic Lab** - Time-series mastery
   - ARIMA, exponential smoothing
   - Walk-forward validation
   - Forecast multiple periods ahead
2. Combine techniques from Advanced Patterns
   - Anomaly detection on your data
   - Text analysis if applicable
   - Ensemble methods
3. Use **Advanced Track** projects

---

## ✅ Quality Checklist for Your Project

Use this when building your own project:

**Data Preparation**
- [ ] Explored all columns (types, distributions, missing)
- [ ] Checked for duplicates
- [ ] Handled missing values (removal or imputation)
- [ ] Fixed outliers (removed or capped)
- [ ] Encoded categorical variables

**Splitting & Scaling**
- [ ] Train/test split (80/20 or appropriate ratio)
- [ ] Used stratify=y for classification
- [ ] Scaled training data, applied same scaling to test
- [ ] For time-series: used walk-forward not random split

**Model Training**
- [ ] Selected 2+ baseline models to compare
- [ ] Used cross-validation (5 or 10-fold)
- [ ] Tuned hyperparameters on training data only
- [ ] Evaluated on test data (held out from training)

**Evaluation**
- [ ] Used appropriate metrics (not just accuracy)
- [ ] Checked confusion matrix or residuals
- [ ] Compared vs baseline/benchmark
- [ ] Analyzed feature importance

**Interpretation**
- [ ] Explained top 3 most important features
- [ ] Identified business implications
- [ ] Listed model limitations
- [ ] Recommended next steps

---

## 🚀 Using Labs in Your Own Workflow

### Save Lab Code to Your Project
1. Find the code example you like in a lab
2. Copy entire code block
3. Create new Python file: `my_project.py`
4. Paste code
5. Replace sample data with your CSV:
   ```python
   # Instead of:
   df = get_lab('E-Commerce')
   
   # Use your data:
   df = pd.read_csv('my_data.csv')
   ```
6. Update column names to match your data
7. Run and adapt as needed

### Example: Using Housing Lab Template

```python
# Original (from lab)
X = df[['bedrooms', 'size', 'age', 'location']]
y = df['price']

# Modified (your data)
X = df[['num_rooms', 'square_feet', 'years_old', 'city']]
y = df['house_price']

# Rest of code stays the same
```

---

## 📞 Support & Resources

### Within the App
- **Academy Page:** All fundamentals, 7 code libraries, new labs
- **Model Management:** Save/load/version your models
- **Prediction:** Score new data with trained models
- **AI Insights:** Generate analysis with LLM

### External Resources
- **Scikit-learn Docs:** https://scikit-learn.org/stable/
- **Pandas Docs:** https://pandas.pydata.org/
- **Statsmodels (Time-Series):** https://www.statsmodels.org/
- **Plotly Docs:** https://plotly.com/python/

### Common Issues

**Lab won't load?**
- Check `src/academy/real_world_labs.py` exists
- Verify import in Academy page: `from src.academy.real_world_labs import LABS, get_lab`

**Code examples not working?**
- Copy entire block including imports
- Check column names match your data
- Verify data types (numeric vs categorical)

**Models not saving?**
- Check `artifacts/` folder exists
- Verify write permissions
- Use Model Management page to save

---

## 💡 Pro Tips

1. **Always stratify classification:** `stratify=y` prevents imbalance issues
2. **Scale before modeling:** StandardScaler on numeric features
3. **Use random_state:** For reproducibility across runs
4. **Check cross-val first:** Before spending time tuning
5. **Visualize predictions:** Plots often reveal patterns metrics miss
6. **Document assumptions:** What did you assume about your data?
7. **Test on holdout data:** Never evaluate on data used for training
8. **Monitor in production:** Models drift over time; retrain regularly

---

**Happy learning! 🎓**
