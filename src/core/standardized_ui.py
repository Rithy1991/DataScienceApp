"""
Standardized UI components for consistent look and feel across all pages.
"""
import streamlit as st

def standard_page_header(title: str, subtitle: str, icon: str = "📊") -> None:
    """Create a standardized page header for all pages."""
    st.markdown(
        f"""
        <div style="background: linear-gradient(135deg, #0b5ed7 0%, #0a58ca 100%); 
                    color: #f8fafc; padding: 20px 24px; border-radius: 12px; 
                    margin-bottom: 20px; box-shadow: 0 4px 12px rgba(11, 94, 215, 0.2);">
            <div style="font-size: 28px; font-weight: 800; margin-bottom: 6px;">
                {icon} {title}
            </div>
            <div style="font-size: 14px; opacity: 0.95;">
                {subtitle}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def standard_section_header(title: str, icon: str = "📌") -> None:
    """Create a standardized section header."""
    st.markdown(f"### {icon} {title}")

def beginner_tip(text: str, icon: str = "💡") -> None:
    """Display a beginner-friendly tip."""
    st.info(f"{icon} **Beginner Tip**: {text}")

def concept_explainer(title: str, explanation: str, real_world_example: str) -> None:
    """Display a concept with plain-language explanation and real-world example."""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{title}**")
        st.write(explanation)
    with col2:
        st.markdown("**Real-World Example**")
        st.write(real_world_example)

def metric_card(label: str, value: str, explanation: str = "", help_text: str = "") -> None:
    """Display a metric card with explanation."""
    col1, col2 = st.columns([1, 2])
    with col1:
        st.metric(label, value)
    with col2:
        if explanation:
            st.caption(explanation)
        if help_text:
            st.caption(f"ℹ️ {help_text}")

def before_after_comparison(before_df, after_df, before_title="Raw Data", after_title="Processed Data"):
    """Display before/after data comparison."""
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(before_title)
        st.dataframe(before_df.head(10), use_container_width=True)
        st.caption(f"{before_df.shape[0]} rows × {before_df.shape[1]} columns")
    with col2:
        st.subheader(after_title)
        st.dataframe(after_df.head(10), use_container_width=True)
        st.caption(f"{after_df.shape[0]} rows × {after_df.shape[1]} columns")

def model_explanation_panel(model_name: str, how_it_works: str, when_to_use: str, pros: list, cons: list):
    """Display model explanation with comprehensive details."""
    with st.container(border=True):
        st.markdown(f"### 🤖 {model_name}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**How it works:**")
            st.write(how_it_works)
            st.markdown("**When to use:**")
            st.write(when_to_use)
        
        with col2:
            st.markdown("**Pros:**")
            for pro in pros:
                st.markdown(f"✅ {pro}")
            st.markdown("**Cons:**")
            for con in cons:
                st.markdown(f"⚠️ {con}")

def common_mistakes_panel(mistakes: dict):
    """Display common beginner mistakes and how to fix them."""
    st.markdown("### ⚠️ Common Beginner Mistakes & How to Fix Them")
    for mistake, fix in mistakes.items():
        with st.expander(f"❌ {mistake}"):
            st.markdown(f"**✅ Fix**: {fix}")
