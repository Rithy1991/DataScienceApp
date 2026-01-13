"""
Demo Workflow: End-to-End Data Science Project Walkthrough
Showcase for stakeholders and students
"""
import streamlit as st
from src.core.ui import app_header, render_footer
from src.core.config import load_config

st.set_page_config(page_title="Demo Workflow", layout="wide", initial_sidebar_state="expanded")

config = load_config()

app_header(
    config,
    page_title="Demo Workflow",
    subtitle="End-to-end data science project walkthrough for stakeholders and students",
    icon="🚀"
)
st.markdown("""
This demo walks through a complete data science project using the platform. Each step is explained for beginners and stakeholders.
""")

steps = [
    "1. Data Input & Preview",
    "2. Data Cleaning & Preprocessing",
    "3. Exploratory Data Analysis (EDA)",
    "4. Machine Learning Task Selection",
    "5. Classification/Regression/Clustering",
    "6. Model Training & Evaluation",
    "7. AI-Generated Explanations",
    "8. Prediction & Inference",
    "9. Export & Reporting"
]

for step in steps:
    st.header(step)
    st.markdown(f"**{step}**: See the corresponding module for hands-on experience. Each page provides step-by-step guidance, visualizations, and explanations.")
    st.info("For a real demo, use a sample dataset and follow the workflow sidebar.")

st.markdown("---")
st.markdown("**Ready to show stakeholders? Walk through each page, export results, and highlight educational features, AI explanations, and business impact.**")

render_footer()
