"""
Professional Platform UI Components
Enhanced styling and branding for a unified data science platform
"""

from __future__ import annotations

import streamlit as st
from typing import Optional, List


def platform_header(
    title: str = "DataScope Pro",
    tagline: str = "Your Complete Data Science Workspace"
) -> None:
    """Display professional platform header with branding."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; margin-bottom: 25px;
                    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);">
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="font-size: 48px;">🔬</div>
                <div>
                    <h1 style="color: white; margin: 0; font-size: 36px; font-weight: 800;">
                        {title}
                    </h1>
                    <p style="color: rgba(255,255,255,0.9); margin: 5px 0 0 0; font-size: 18px;">
                        {tagline}
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def feature_card(
    icon: str,
    title: str,
    description: str,
    color: str = "#667eea"
) -> None:
    """Display a feature card with icon, title and description."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, {color}15, {color}05);
                    border-left: 4px solid {color};
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;
                    transition: transform 0.2s;">
            <div style="font-size: 32px; margin-bottom: 10px;">{icon}</div>
            <h3 style="color: {color}; margin: 10px 0; font-size: 20px; font-weight: 700;">
                {title}
            </h3>
            <p style="color: #475569; font-size: 15px; line-height: 1.6; margin: 0;">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


def module_section(
    title: str,
    subtitle: str,
    icon: str = "📚"
) -> None:
    """Display a module section header."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(90deg, #f8fafc 0%, #e2e8f0 100%);
                    padding: 20px 25px;
                    border-radius: 12px;
                    margin: 25px 0 20px 0;
                    border: 2px solid #cbd5e1;">
            <div style="display: flex; align-items: center; gap: 15px;">
                <div style="font-size: 40px;">{icon}</div>
                <div>
                    <h2 style="color: #1e293b; margin: 0; font-size: 28px; font-weight: 800;">
                        {title}
                    </h2>
                    <p style="color: #64748b; margin: 5px 0 0 0; font-size: 16px;">
                        {subtitle}
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def learning_objective_box(
    objectives: List[str],
    title: str = "Learning Objectives"
) -> None:
    """Display learning objectives in a styled box."""
    objectives_html = "".join([f"<li style='margin: 8px 0;'>{obj}</li>" for obj in objectives])
    
    st.markdown(
        f"""
        <div style="background: #f0f9ff; 
                    border: 2px solid #0ea5e9;
                    padding: 20px;
                    border-radius: 10px;
                    margin: 15px 0;">
            <h4 style="color: #0369a1; margin: 0 0 15px 0; font-size: 18px; font-weight: 700;">
                🎯 {title}
            </h4>
            <ul style="color: #0c4a6e; margin: 0; padding-left: 25px; line-height: 1.8;">
                {objectives_html}
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )


def code_example(
    code: str,
    language: str = "python",
    title: Optional[str] = None,
    explanation: Optional[str] = None
) -> None:
    """Display a code example with optional title and explanation."""
    if title:
        st.markdown(f"**{title}**")
    
    st.code(code, language=language)
    
    if explanation:
        st.info(explanation, icon="💡")


def interactive_demo_box(title: str) -> None:
    """Display a box indicating interactive demo section."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%);
                    color: white;
                    padding: 15px 20px;
                    border-radius: 10px;
                    margin: 20px 0 10px 0;
                    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="font-size: 28px;">🎮</div>
                <div style="font-size: 18px; font-weight: 700;">
                    {title}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def key_concept_highlight(
    concept: str,
    definition: str,
    example: Optional[str] = None
) -> None:
    """Highlight a key concept with definition."""
    st.markdown(
        f"""
        <div style="background: #fef3c7;
                    border-left: 5px solid #f59e0b;
                    padding: 18px;
                    border-radius: 8px;
                    margin: 15px 0;">
            <h4 style="color: #92400e; margin: 0 0 10px 0; font-size: 17px; font-weight: 700;">
                💡 {concept}
            </h4>
            <p style="color: #78350f; margin: 0; line-height: 1.7; font-size: 15px;">
                {definition}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if example:
        st.markdown(f"**Example:** {example}")


def progress_tracker(
    completed: int,
    total: int,
    label: str = "Progress"
) -> None:
    """Display a progress tracker."""
    percentage = (completed / total * 100) if total > 0 else 0
    
    st.markdown(
        f"""
        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-weight: 600; color: #1e293b;">{label}</span>
                <span style="font-weight: 600; color: #64748b;">{completed}/{total} ({percentage:.0f}%)</span>
            </div>
            <div style="background: #e2e8f0; height: 12px; border-radius: 999px; overflow: hidden;">
                <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
                            height: 100%;
                            width: {percentage}%;
                            transition: width 0.5s ease;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def workflow_step(
    step_num: int,
    title: str,
    description: str,
    is_current: bool = False
) -> None:
    """Display a workflow step."""
    bg_color = "#667eea" if is_current else "#cbd5e1"
    text_color = "white" if is_current else "#475569"
    
    st.markdown(
        f"""
        <div style="display: flex; align-items: start; gap: 15px; margin: 15px 0;">
            <div style="background: {bg_color};
                        color: {text_color};
                        width: 40px;
                        height: 40px;
                        border-radius: 50%;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        font-weight: 700;
                        font-size: 18px;
                        flex-shrink: 0;">
                {step_num}
            </div>
            <div style="flex: 1;">
                <h4 style="color: #1e293b; margin: 0 0 5px 0; font-size: 17px; font-weight: 700;">
                    {title}
                </h4>
                <p style="color: #64748b; margin: 0; font-size: 14px; line-height: 1.6;">
                    {description}
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def quiz_question(
    question: str,
    options: List[str],
    correct_answer: int,
    explanation: str,
    key: str
) -> bool:
    """Display an interactive quiz question."""
    st.markdown(f"**❓ {question}**")
    
    user_answer = st.radio(
        "Select your answer:",
        options,
        key=key,
        label_visibility="collapsed"
    )
    
    if st.button("Check Answer", key=f"{key}_check"):
        if options.index(user_answer) == correct_answer:
            st.success(f"✅ Correct! {explanation}")
            return True
        else:
            st.error(f"❌ Not quite. {explanation}")
            return False
    
    return False


def platform_footer() -> None:
    """Display platform footer."""
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #94a3b8; padding: 20px; font-size: 14px;">
            <p style="margin: 0;">Built with ❤️ using Streamlit • DataScope Pro © 2026</p>
            <p style="margin: 5px 0 0 0;">Your Complete Data Science Workspace</p>
        </div>
        """,
        unsafe_allow_html=True
    )
