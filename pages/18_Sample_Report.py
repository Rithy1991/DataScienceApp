"""
Sample Report Export Page
Generate and download a sample report for stakeholders
"""
import streamlit as st
from src.core.ui import app_header, render_footer
from src.core.config import load_config
from src.core.standardized_ui import (
    standard_section_header,
    concept_explainer,
    beginner_tip,
    common_mistakes_panel,
)
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Sample Report Export", layout="wide", initial_sidebar_state="expanded")

config = load_config()

app_header(
    config,
    page_title="Sample Report Export",
    subtitle="Generate downloadable reports showing cleaned data, model results, charts, and AI explanations",
    icon="📄"
)
st.markdown("""
Generate a downloadable sample report (CSV) showing cleaned data, model results, charts, and AI explanations.
""")

# Simulate cleaned data and model results
np.random.seed(42)
df = pd.DataFrame({
    'Feature1': np.random.normal(0, 1, 50),
    'Feature2': np.random.normal(5, 2, 50),
    'Target': np.random.choice(['A', 'B'], 50)
})
model_results = pd.DataFrame({
    'Model': ['Logistic Regression', 'KNN', 'Decision Tree'],
    'Accuracy': [0.92, 0.89, 0.87],
    'Precision': [0.91, 0.88, 0.86],
    'Recall': [0.93, 0.90, 0.88]
})
ai_explanation = "This model predicts the target class based on the features. High accuracy means reliable predictions. For best results, ensure data is clean and features are relevant."

st.subheader("Preview: Cleaned Data")
st.dataframe(df.head(10), width="stretch")

st.subheader("Preview: Model Results")
st.dataframe(model_results, width="stretch")

st.subheader("AI-Generated Explanation")
st.info(ai_explanation)

# Export as CSV
st.subheader("Download Sample Report (CSV)")
output = BytesIO()
df.to_csv(output, index=False)
output.seek(0)
st.download_button("Download Cleaned Data CSV", data=output, file_name="cleaned_data.csv", mime="text/csv")

output2 = BytesIO()
model_results.to_csv(output2, index=False)
output2.seek(0)
st.download_button("Download Model Results CSV", data=output2, file_name="model_results.csv", mime="text/csv")

st.markdown("---")
st.markdown("**For a full report, export results from each module and combine with AI explanations and charts.**")

standard_section_header("Learning Guide & Best Practices", "🎓")
concept_explainer(
    title="Export & Reporting",
    explanation=(
        "Assemble cleaned data, model metrics, charts, and AI explanations into shareable artifacts. Ensure reproducibility and clarity for stakeholders."
    ),
    real_world_example=(
        "Weekly KPI report: Include data source notes, preprocessing summary, top model metrics, trend charts, and action recommendations."
    ),
)
beginner_tip("Tip: Keep reports concise and decision-oriented — highlight insights, not just numbers.")
common_mistakes_panel({
    "No data lineage": "Specify sources, versions, and timestamps.",
    "Overloaded charts": "Use clean visuals with annotations.",
    "Missing reproducibility": "Store code, configs, and dataset snapshots.",
    "Unclear metrics": "Define metric meanings and business thresholds.",
})

render_footer()
