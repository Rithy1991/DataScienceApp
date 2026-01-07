"""
AI Helper Component - Contextual AI assistance throughout the app
Provides quick interpretation help and links to full AI Insights page
"""

from __future__ import annotations

import streamlit as st
from typing import Optional


def ai_help_button(
    context: str,
    button_text: str = "🤖 Ask AI for Help",
    help_text: str = "Get AI-powered interpretation and insights",
    key: Optional[str] = None
) -> None:
    """
    Display an AI help button that provides contextual assistance.
    
    Args:
        context: The context or question to ask the AI
        button_text: Text to display on the button
        help_text: Tooltip text
        key: Unique key for the button
    """
    if st.button(button_text, help=help_text, key=key, type="secondary"):
        st.session_state["ai_context"] = context
        st.switch_page("pages/7_AI_Insights_SLM_Powered.py")


def ai_interpretation_box(
    title: str,
    interpretation: str,
    context: str = "",
    show_learn_more: bool = True
) -> None:
    """
    Display an interpretation box with AI assistance option.
    
    Args:
        title: Title of the interpretation box
        interpretation: The interpretation text to display
        context: Context for AI if user wants deeper analysis
        show_learn_more: Whether to show "Ask AI for more" button
    """
    with st.expander(f"💡 {title}", expanded=False):
        st.markdown(interpretation)
        
        if show_learn_more and context:
            col1, col2 = st.columns([3, 1])
            with col2:
                if st.button("Ask AI →", key=f"ai_more_{hash(context)}", use_container_width=True):
                    st.session_state["ai_context"] = context
                    st.switch_page("pages/7_AI_Insights_SLM_Powered.py")


def inline_ai_help(
    metric_name: str,
    explanation: str,
    ai_context: Optional[str] = None
) -> None:
    """
    Display inline help icon with explanation and optional AI deep dive.
    
    Args:
        metric_name: Name of the metric/concept
        explanation: Quick explanation text
        ai_context: Optional context for AI deep dive
    """
    with st.popover(f"ℹ️", help=f"Learn about {metric_name}"):
        st.markdown(f"**{metric_name}**")
        st.markdown(explanation)
        
        if ai_context:
            st.divider()
            if st.button("🤖 Ask AI for detailed analysis", key=f"inline_ai_{hash(metric_name)}"):
                st.session_state["ai_context"] = ai_context
                st.switch_page("pages/7_AI_Insights_SLM_Powered.py")


def ai_quick_tip(
    tip_text: str,
    tip_type: str = "info",
    show_ai_button: bool = False,
    ai_context: str = ""
) -> None:
    """
    Display a quick tip with optional AI assistance.
    
    Args:
        tip_text: The tip text to display
        tip_type: Type of alert (info, success, warning, error)
        show_ai_button: Whether to show AI help button
        ai_context: Context for AI if button is clicked
    """
    if tip_type == "info":
        st.info(tip_text, icon="💡")
    elif tip_type == "success":
        st.success(tip_text, icon="✅")
    elif tip_type == "warning":
        st.warning(tip_text, icon="⚠️")
    elif tip_type == "error":
        st.error(tip_text, icon="❌")
    
    if show_ai_button and ai_context:
        if st.button("🤖 Learn more with AI", key=f"tip_ai_{hash(tip_text)}"):
            st.session_state["ai_context"] = ai_context
            st.switch_page("pages/7_AI_Insights_SLM_Powered.py")


def ai_sidebar_assistant() -> None:
    """
    Display a persistent AI assistant in the sidebar.
    """
    with st.sidebar:
        st.divider()
        st.markdown("### 🤖 AI Assistant")
        st.caption("Need help interpreting your results?")
        
        quick_question = st.text_input(
            "Quick question:",
            placeholder="e.g., What does R² mean?",
            key="sidebar_ai_question",
            label_visibility="collapsed"
        )
        
        if st.button("Ask AI", key="sidebar_ai_ask", use_container_width=True):
            if quick_question:
                st.session_state["ai_context"] = quick_question
                st.switch_page("pages/7_AI_Insights_SLM_Powered.py")
            else:
                st.warning("Please enter a question")


def create_ai_explanation_sections() -> dict:
    """
    Return a dictionary of common AI explanations for various metrics.
    """
    return {
        "accuracy": {
            "quick": "Percentage of correct predictions out of all predictions.",
            "detailed_context": "Explain accuracy metric in machine learning, when to use it, and its limitations especially with imbalanced datasets."
        },
        "precision": {
            "quick": "Of all positive predictions, how many were actually correct?",
            "detailed_context": "Explain precision in machine learning classification, including real-world examples and when high precision is critical."
        },
        "recall": {
            "quick": "Of all actual positives, how many did we correctly identify?",
            "detailed_context": "Explain recall (sensitivity) in machine learning, use cases where high recall is crucial, and trade-offs with precision."
        },
        "f1_score": {
            "quick": "Harmonic mean of precision and recall. Good for imbalanced data.",
            "detailed_context": "Explain F1 score in machine learning, why it's better than accuracy for imbalanced datasets, and how to interpret it."
        },
        "r2_score": {
            "quick": "Proportion of variance in the target explained by the model (0-1, higher is better).",
            "detailed_context": "Explain R-squared (coefficient of determination) in regression, how to interpret values, and what negative R² means."
        },
        "rmse": {
            "quick": "Root Mean Squared Error - average prediction error in original units.",
            "detailed_context": "Explain RMSE in regression, how it differs from MAE, why it penalizes large errors more, and how to interpret it."
        },
        "mae": {
            "quick": "Mean Absolute Error - average absolute difference between predictions and actual values.",
            "detailed_context": "Explain MAE in regression, how it differs from RMSE, when to prefer MAE over RMSE, and interpretation."
        },
        "confusion_matrix": {
            "quick": "Shows True Positives, False Positives, True Negatives, False Negatives.",
            "detailed_context": "Explain confusion matrix in detail, how to read it, and what each quadrant tells you about model performance."
        },
        "roc_auc": {
            "quick": "Area Under ROC Curve - measures ability to distinguish between classes (0.5-1, higher is better).",
            "detailed_context": "Explain ROC AUC score, how ROC curve is created, what AUC represents, and how to interpret different AUC values."
        },
        "overfitting": {
            "quick": "Model performs well on training data but poorly on new data.",
            "detailed_context": "Explain overfitting in machine learning, causes, how to detect it, and strategies to prevent it."
        },
        "underfitting": {
            "quick": "Model is too simple and performs poorly even on training data.",
            "detailed_context": "Explain underfitting in machine learning, causes, how to detect it, and how to address it."
        },
        "cross_validation": {
            "quick": "Technique to evaluate model on multiple train/test splits for more reliable performance estimate.",
            "detailed_context": "Explain cross-validation in machine learning, different types (k-fold, stratified), and why it's important."
        },
        "feature_importance": {
            "quick": "Shows which features contribute most to predictions.",
            "detailed_context": "Explain feature importance in machine learning, how it's calculated in different models, and how to use it for feature selection."
        }
    }
