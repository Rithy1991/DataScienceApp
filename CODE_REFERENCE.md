# Code Reference: Implementation Details

## Quick Reference for Using the New Modules

### 1. Cleaning Report State Management

**Initialize Report (Automatic):**
```python
from src.core.cleaning_report_state import initialize_cleaning_report

# Called automatically by other functions, but you can call it explicitly:
initialize_cleaning_report(st.session_state)
```

**Log an Action:**
```python
from src.core.cleaning_report_state import add_cleaning_action

add_cleaning_action(
    session_state=st.session_state,
    action_name="missing_value_imputation",  # Machine-readable
    action_description="Filled 150 missing values in 'income' using median",  # User-friendly
    metrics={  # Any relevant metadata
        "column": "income",
        "method": "median",
        "missing_before": 150,
        "missing_after": 0,
        "rows_affected": 150,
    }
)
```

**Get Summary of Changes:**
```python
from src.core.cleaning_report_state import get_report_summary

summary = get_report_summary(st.session_state)
print(summary)
# Output:
# {
#   'rows_before': 1000,
#   'rows_after': 990,
#   'rows_removed': 10,
#   'missing_before': 247,
#   'missing_after': 0,
#   'missing_fixed': 247,
#   'duplicates_before': 8,
#   'duplicates_after': 0,
#   'duplicates_removed': 8,
#   'actions_count': 5
# }
```

**Record Before/After Metrics:**
```python
from src.core.cleaning_report_state import set_before_metrics, set_after_metrics

raw_df = get_df(st.session_state)  # Your raw dataframe
clean_df = get_clean_df(st.session_state)  # Your cleaned dataframe

# Usually done automatically, but can be called explicitly:
set_before_metrics(st.session_state, raw_df)
# ... do cleaning ...
set_after_metrics(st.session_state, clean_df)
```

**Export Formats:**
```python
from src.core.cleaning_report_state import (
    export_report_json,
    export_report_csv,
    export_report_markdown,
    get_cleaning_report
)

# Get the raw report dict
report = get_cleaning_report(st.session_state)
print(report)

# Export as JSON string
json_str = export_report_json(st.session_state)
st.download_button("Download JSON", json_str, "report.json", "application/json")

# Export as CSV DataFrame
csv_df = export_report_csv(st.session_state)
st.download_button("Download CSV", csv_df.to_csv(index=False), "actions.csv", "text/csv")

# Export as Markdown string
md_str = export_report_markdown(st.session_state)
st.download_button("Download Markdown", md_str, "report.md", "text/markdown")
```

---

### 2. Flow Guidance System

**Get User's Current Position:**
```python
from src.core.flow_guidance import get_current_pipeline_step, is_step_completed

current_step = get_current_pipeline_step(st.session_state)  # Returns 0-7 or None
is_done = is_step_completed(st.session_state, step_id=2)  # True if step 2 is complete
```

**Show "Where Am I?" Guidance:**
```python
from src.core.flow_guidance import render_step_guidance

render_step_guidance(
    current_step_id=2,
    current_step_name="Data Cleaning",
    current_step_description="Remove missing values, duplicates, and outliers"
)

# Renders two side-by-side cards:
# [Current Step: 🧼 Data Cleaning]  [Next Step: 🔨 Feature Engineering]
```

**Show Full Pipeline Roadmap:**
```python
from src.core.flow_guidance import render_pipeline_roadmap

render_pipeline_roadmap()

# Renders 8 colored boxes in a 4x2 grid showing all steps
```

**Continue Button:**
```python
from src.core.flow_guidance import render_next_step_button

# At the bottom of your page:
render_next_step_button(next_step_id=3)

# Renders: [➡️ Continue to Feature Engineering]
# When clicked, navigates to that page
```

**Progress Checklist:**
```python
from src.core.flow_guidance import render_completion_checklist

render_completion_checklist(st.session_state)

# Shows checkmarks for completed steps, current step indicator for current,
# and circles for incomplete steps
```

**Sidebar Progress:**
```python
from src.core.flow_guidance import render_pipeline_progress_sidebar

render_pipeline_progress_sidebar()

# Shows pipeline in sidebar with visual progress indicators
# (Ready to use but not yet integrated into ui.py)
```

---

### 3. Pipeline Steps Reference

The 8 steps defined in `flow_guidance.py`:

```python
PIPELINE_STEPS = [
    {
        "id": 0,
        "emoji": "🏠",
        "name": "Load Data",
        "description": "Upload or select a dataset",
        "file": "app.py",
    },
    {
        "id": 1,
        "emoji": "📊",
        "name": "Explore Data",
        "description": "Understand structure, types, and distributions",
        "file": "pages/2_Data_Analysis_EDA.py",
    },
    {
        "id": 2,
        "emoji": "🧼",
        "name": "Clean Data",
        "description": "Handle missing, duplicates, and outliers",
        "file": "pages/3_Data_Cleaning.py",
    },
    {
        "id": 3,
        "emoji": "🔨",
        "name": "Engineer Features",
        "description": "Create and select features for models",
        "file": "pages/4_Feature_Engineering.py",
    },
    {
        "id": 4,
        "emoji": "🎯",
        "name": "Train Model",
        "description": "Build and optimize machine learning models",
        "file": "pages/5_Tabular_Machine_Learning.py",
    },
    {
        "id": 5,
        "emoji": "📈",
        "name": "Evaluate Model",
        "description": "Assess performance with metrics and visualizations",
        "file": "pages/5_Tabular_Machine_Learning.py",
    },
    {
        "id": 6,
        "emoji": "🎯",
        "name": "Predict & Infer",
        "description": "Apply model to new data",
        "file": "pages/9_Prediction.py",
    },
    {
        "id": 7,
        "emoji": "📄",
        "name": "Report & Export",
        "description": "Generate findings and export results",
        "file": "pages/18_Sample_Report.py",
    },
]
```

---

### 4. Example: Implementing Cleaning Report in Your Own Page

```python
# pages/4_Feature_Engineering.py

from src.core.cleaning_report_state import add_cleaning_action
from src.core.flow_guidance import render_step_guidance, render_next_step_button

st.set_page_config(page_title="Feature Engineering", layout="wide")

# At the top of your main() function:
render_step_guidance(
    current_step_id=3,
    current_step_name="Feature Engineering",
    current_step_description="Create new features and select important ones"
)

# When user creates a new feature:
if st.button("Create Feature"):
    new_feature = create_polynomial_features(df)
    
    # Log the action
    add_cleaning_action(
        session_state=st.session_state,
        action_name="feature_creation",
        action_description=f"Created polynomial features (degree=2)",
        metrics={
            "new_features_created": 5,
            "total_features": len(new_feature.columns),
        }
    )
    
    st.success("Feature created!")

# At the bottom of your page:
render_next_step_button(next_step_id=4)
```

---

### 5. Complete Example: Data Cleaning with Report

```python
# Simplified example from pages/3_Data_Cleaning.py

from src.core.cleaning_report_state import (
    initialize_cleaning_report,
    add_cleaning_action,
    set_before_metrics,
    set_after_metrics,
    get_report_summary,
    export_report_json,
)
from src.core.state import get_df, get_clean_df, set_clean_df
from src.core.flow_guidance import render_step_guidance, render_next_step_button

def main():
    st.set_page_config(page_title="Data Cleaning", layout="wide")
    
    # Show guidance
    render_step_guidance(
        current_step_id=2,
        current_step_name="Data Cleaning",
        current_step_description="Handle missing values, duplicates, and outliers"
    )
    
    # Initialize report
    initialize_cleaning_report(st.session_state)
    
    # Record original data
    raw_df = get_df(st.session_state)
    if raw_df is not None:
        set_before_metrics(st.session_state, raw_df)
    
    # Cleaning operations
    df = get_clean_df(st.session_state) or raw_df
    
    if st.button("Fill missing values"):
        # Perform cleaning
        df_cleaned = df.fillna(df.mean())
        
        # Save to session
        set_clean_df(st.session_state, df_cleaned)
        
        # Log the action
        missing_before = df.isnull().sum().sum()
        missing_after = df_cleaned.isnull().sum().sum()
        add_cleaning_action(
            session_state=st.session_state,
            action_name="missing_value_imputation",
            action_description="Filled all missing values with column means",
            metrics={
                "missing_before": int(missing_before),
                "missing_after": int(missing_after),
                "fixed": int(missing_before - missing_after),
            }
        )
        
        st.success("Cleaning complete!")
    
    # Summary
    st.markdown("## Report")
    summary = get_report_summary(st.session_state)
    st.json(summary)
    
    # Export
    report_json = export_report_json(st.session_state)
    st.download_button(
        "Download Report",
        report_json,
        "cleaning_report.json",
        "application/json"
    )
    
    # Record final data
    final_df = get_clean_df(st.session_state)
    if final_df is not None:
        set_after_metrics(st.session_state, final_df)
    
    # Next step
    render_next_step_button(next_step_id=3)

if __name__ == "__main__":
    main()
```

---

### 6. Session State Keys Used

These are the session_state keys used by the new modules:

```python
# From src/core/state.py (existing)
st.session_state["dsai_df"]        # Raw dataset
st.session_state["dsai_df_clean"]  # Cleaned dataset

# From src/core/cleaning_report_state.py (new)
st.session_state["dsai_cleaning_report"]  # Entire report dict with:
#   - timestamp_started
#   - timestamp_completed
#   - actions (list)
#   - before (dict with rows, cols, missing, duplicates)
#   - after (dict with rows, cols, missing, duplicates)
```

---

### 7. Common Patterns

**Pattern 1: Check if data is loaded**
```python
from src.core.state import get_df, get_clean_df

raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)

if raw_df is None:
    st.warning("Please load data first")
    st.stop()
```

**Pattern 2: Working with cleaned data**
```python
df = clean_df if clean_df is not None else raw_df
if df is None:
    st.error("No data available")
    st.stop()
```

**Pattern 3: Log and save changes**
```python
from src.core.cleaning_report_state import add_cleaning_action
from src.core.state import set_clean_df

# Perform operation
new_df = df.drop_duplicates()

# Save
set_clean_df(st.session_state, new_df)

# Log
add_cleaning_action(
    st.session_state,
    action_name="duplicate_removal",
    action_description=f"Removed {len(df) - len(new_df)} duplicates",
    metrics={"duplicates_removed": len(df) - len(new_df)}
)

st.success("Done!")
```

**Pattern 4: Before/after metrics**
```python
from src.core.cleaning_report_state import set_before_metrics, set_after_metrics

raw = get_df(st.session_state)
clean = get_clean_df(st.session_state)

if raw:
    set_before_metrics(st.session_state, raw)
if clean:
    set_after_metrics(st.session_state, clean)

summary = get_report_summary(st.session_state)
st.metric("Rows Removed", summary["rows_removed"])
st.metric("Missing Fixed", summary["missing_fixed"])
```

---

### 8. Debugging

**To see the entire report dict:**
```python
from src.core.cleaning_report_state import get_cleaning_report
import json

report = get_cleaning_report(st.session_state)
st.json(report)  # Pretty-print the report
```

**To see what actions have been logged:**
```python
from src.core.cleaning_report_state import get_cleaning_report

report = get_cleaning_report(st.session_state)
for i, action in enumerate(report["actions"], 1):
    print(f"{i}. {action['action_description']}")
    print(f"   Metrics: {action['metrics']}")
```

**To manually reset the report:**
```python
from src.core.cleaning_report_state import reset_cleaning_report

reset_cleaning_report(st.session_state)
st.success("Report reset!")
```

---

## Architecture Overview

```
User Loads Data (home/app.py)
    ↓
Sees flow guidance (src/core/flow_guidance.py)
    ↓
Goes to Data Cleaning (pages/3_Data_Cleaning.py)
    ↓
Performs operations (missing, duplicates, outliers)
    ↓
Each operation logs to report (src/core/cleaning_report_state.py)
    ↓
Summary tab shows all actions and metrics
    ↓
User can export JSON/CSV/Markdown
    ↓
User sees "Next: Feature Engineering" with button
    ↓
Continues to next step...
```

---

## Testing Code Snippets

**Test Report Creation:**
```python
import streamlit as st
from src.core.cleaning_report_state import (
    initialize_cleaning_report,
    add_cleaning_action,
    get_cleaning_report,
    export_report_json,
)

# Initialize
initialize_cleaning_report(st.session_state)

# Add actions
add_cleaning_action(
    st.session_state,
    "test_action_1",
    "First test action",
    {"metric": "value1"}
)

add_cleaning_action(
    st.session_state,
    "test_action_2",
    "Second test action",
    {"metric": "value2"}
)

# View report
report = get_cleaning_report(st.session_state)
st.json(report)

# Export
json_str = export_report_json(st.session_state)
st.text(json_str)
```

**Test Flow Detection:**
```python
import streamlit as st
from src.core.state import get_df, get_clean_df, set_df, set_clean_df
from src.core.flow_guidance import get_current_pipeline_step
import pandas as pd

# Simulate loaded data
df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
set_df(st.session_state, df, "Test")

current = get_current_pipeline_step(st.session_state)
st.write(f"Current step (should be 1): {current}")

# Simulate cleaned data
set_clean_df(st.session_state, df)
current = get_current_pipeline_step(st.session_state)
st.write(f"Current step (should be 2+): {current}")
```

---

That's it! You now have complete documentation of the code and how to use it. 🚀
