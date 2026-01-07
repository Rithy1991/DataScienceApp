# 🚀 Quick Start Guide - DataScope Pro

## Getting Started in 3 Steps

### Step 1: Load Your Data
**Option A - Use Sample Data** (Recommended for first-time users)
1. Go to the main page
2. Scroll to "Quick Start with Sample Data"
3. Select a dataset (e.g., "Sales Forecasting")
4. Click "Load Sample Dataset"

**Option B - Upload Your Own Data**
1. Click "Browse files" button
2. Select a CSV or Excel file
3. Wait for automatic validation
4. Your data is ready!

### Step 2: Explore Your Data
1. Navigate to **📊 Data Analysis & EDA** page (sidebar)
2. Review the automatic analysis:
   - Basic statistics
   - Data quality metrics
   - Missing value summary
3. **Pro tip**: Use the manual editing feature to fix any issues

### Step 3: Build a Model
1. Go to **🤖 Tabular Machine Learning** page
2. Select your target column (what you want to predict)
3. Choose a model (RandomForest is a good start)
4. Click "Train Model"
5. View your predictions!

---

## Page-by-Page Guide

### 🏠 Home - Data Upload & Preview
**What it does**: Load data from files, APIs, or samples  
**When to use**: Starting a new project  
**Key features**: Auto-validation, data preview, sample datasets

### 📊 Data Analysis & EDA
**What it does**: Understand your data before modeling  
**When to use**: After loading data, before training models  
**Key features**: Statistics, correlations, distributions, manual editing

### 🤖 Tabular Machine Learning
**What it does**: Train predictive models (RandomForest, XGBoost, LightGBM)  
**When to use**: For classification or regression tasks  
**Key features**: Auto-tuning, cross-validation, forecast comparison

### 🔮 Deep Learning (TFT Transformer)
**What it does**: Advanced time series forecasting  
**When to use**: Complex patterns, multiple variables, long-term predictions  
**Key features**: Attention mechanisms, confidence intervals, interpretability

### 📈 Visualization
**What it does**: Create presentation-ready charts  
**When to use**: Sharing insights with stakeholders  
**Key features**: Interactive plots, export options, customization

### 🎯 Prediction & Inference
**What it does**: Make predictions with trained models  
**When to use**: After training a model  
**Key features**: Batch predictions, single predictions, result export

### 🧠 AI Insights (SLM-Powered)
**What it does**: Get AI-generated analysis and recommendations  
**When to use**: Need quick insights or explanations  
**Key features**: Auto-analysis, plain-language summaries, actionable tips

### 💾 Model Management
**What it does**: Track and manage all your models  
**When to use**: Managing multiple experiments  
**Key features**: Version control, performance tracking, model comparison

### ⚙️ Settings & Configuration
**What it does**: Customize app behavior  
**When to use**: First-time setup or changing defaults  
**Key features**: Model parameters, display options, API settings

### 🎓 Data Science Academy
**What it does**: Learn data science concepts  
**When to use**: Understanding terminology and methods  
**Key features**: Tutorials, examples, best practices

---

## Common Tasks

### Task: Forecast Sales for Next Month
1. Load your sales data (needs date + sales columns)
2. Go to **Tabular Machine Learning** or **Deep Learning** page
3. Select "sales" as target column
4. Set forecast horizon to 30 days
5. Train model
6. View forecast chart and download predictions

### Task: Predict Customer Churn
1. Load customer data (needs outcome column like "churned")
2. Go to **Data Analysis** - check data quality
3. Go to **Tabular Machine Learning**
4. Select "churned" as target
5. Choose RandomForest or XGBoost
6. Train and evaluate model
7. Use **Prediction** page for new customers

### Task: Compare Multiple Models
1. Train first model (e.g., RandomForest)
2. Train second model (e.g., XGBoost)
3. Scroll to "Forecast Comparison" section
4. Compare metrics (RMSE, MAE) and visual plots
5. Choose best performer

### Task: Export Results
1. After training, go to **Prediction** page
2. Make predictions on new data
3. Results appear in table format
4. Use Streamlit's built-in download button (top-right of tables)
5. Save as CSV

---

## Understanding Results

### Forecast Charts
- **Blue line**: Your predictions
- **Gray shaded area**: Confidence interval (95% likely range)
- **Red dots**: Historical actual values
- **Horizontal axis**: Time/date
- **Vertical axis**: Predicted value

### Metrics Explained
- **RMSE** (Root Mean Square Error): Average prediction error (lower is better)
- **MAE** (Mean Absolute Error): Average absolute error (lower is better)
- **R² Score**: How well model fits data (closer to 1.0 is better)
- **MAPE**: Percentage error (lower is better)

### Confidence Intervals
The shaded area shows where we expect the real value to fall:
- **Narrow interval**: High confidence, less uncertainty
- **Wide interval**: Lower confidence, more uncertainty
- **95% confidence**: We're 95% sure the true value is in this range

---

## Troubleshooting

### "No data loaded"
**Solution**: Go to Home page and load data first

### "Target column not found"
**Solution**: Make sure your dataset has the column you're trying to predict

### "Not enough data to train"
**Solution**: Need at least 20-30 rows for basic models, 100+ for best results

### "Model training failed"
**Solution**: Check for:
- Missing values (use Data Cleaning)
- Non-numeric target for regression
- Too few samples per class

### App is slow
**Solution**: 
- Use smaller datasets for initial exploration
- Reduce forecast horizon
- Use RandomForest instead of XGBoost for speed

---

## Best Practices

### Data Preparation
1. ✅ Always start with EDA before modeling
2. ✅ Handle missing values explicitly
3. ✅ Check for outliers and anomalies
4. ✅ Ensure date columns are properly formatted
5. ✅ Use at least 50-100 rows for reliable models

### Model Selection
- **RandomForest**: Fast, reliable, good default choice
- **XGBoost**: Better accuracy, slower training
- **LightGBM**: Fast training, large datasets
- **TFT Transformer**: Complex time series, multiple variables

### Validation
1. ✅ Always check test metrics (not just training)
2. ✅ Compare multiple models
3. ✅ Validate predictions make business sense
4. ✅ Test on new data before deployment

---

## Getting Help

### In-App Help
- Each page has a **quick-start guide** at the top
- Look for 💡 info boxes with tips
- Check **Data Science Academy** for concepts

### Common Resources
- Sample datasets demonstrate proper format
- AI Insights can explain your results
- Model Management shows training history

---

## Keyboard Shortcuts (Streamlit)
- `R`: Rerun app
- `C`: Clear cache
- `S`: Screenshot current view

---

*Last updated: January 7, 2026*
*App version: DataScope Pro v1.0*
