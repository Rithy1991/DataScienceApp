"""
Custom CSS and styling utilities for the Streamlit app.
Provides Bootstrap-inspired styling and visual enhancements.
"""

from __future__ import annotations


def inject_custom_css() -> None:
    """Inject custom CSS for better visual styling and Bootstrap-like appearance."""
    import streamlit as st

    css = """
    <style>
    /* Root color scheme */
    :root {
        --primary: #0ea5e9;
        --primary-strong: #0b8ed1;
        --accent: #6366f1;
        --success: #16a34a;
        --warning: #f59e0b;
        --danger: #dc2626;
        --info: #0ea5e9;
        --light: #f8f9fa;
        --dark: #212529;
        --border-color: #dee2e6;
    }

    /* Streamlit container styling */
    .main {
        background-color: #ffffff;
    }

    /* Header styling */
    h1 {
        color: var(--dark) !important;
        font-weight: 600 !important;
        padding-bottom: 0.5rem !important;
        border-bottom: 3px solid var(--primary) !important;
        margin-bottom: 1.5rem !important;
    }

    h2 {
        color: var(--dark) !important;
        font-weight: 600 !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
    }

    h3 {
        color: #495057 !important;
        font-weight: 500 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }

    /* Card-like containers for sections */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] {
        background-color: #f8f9fa;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1rem;
        border: 1px solid var(--border-color);
    }

    /* Button styling */
    button[data-testid="baseButton-primary"] {
        background-color: var(--primary) !important;
        border-color: var(--primary) !important;
        font-weight: 500 !important;
        padding: 0.5rem 1rem !important;
        border-radius: 0.375rem !important;
    }

    button[data-testid="baseButton-primary"]:hover {
        background-color: #0b5ed7 !important;
        border-color: #0b5ed7 !important;
    }

    button[data-testid="baseButton-secondary"] {
        border-color: var(--border-color) !important;
        color: #495057 !important;
        font-weight: 500 !important;
    }

    /* Info/alert boxes */
    .stAlert {
        border-radius: 0.375rem !important;
        border-left: 4px solid var(--primary) !important;
        padding: 1rem !important;
    }

    .stAlert[data-testid="stInfo"] {
        background-color: #cfe2ff;
        border-left-color: var(--info) !important;
    }

    .stAlert[data-testid="stSuccess"] {
        background-color: #d1e7dd;
        border-left-color: var(--success) !important;
    }

    .stAlert[data-testid="stWarning"] {
        background-color: #fff3cd;
        border-left-color: var(--warning) !important;
    }

    .stAlert[data-testid="stError"] {
        background-color: #f8d7da;
        border-left-color: var(--danger) !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid var(--border-color);
    }

    [data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }

    /* Divider styling */
    hr {
        border: none;
        border-top: 1px solid var(--border-color) !important;
        margin: 1rem 0 !important;
    }

    /* Tabs styling */
    button[data-testid="stTab"] {
        color: #6c757d !important;
        border-bottom: 2px solid transparent !important;
        font-weight: 500 !important;
    }

    button[data-testid="stTab"][aria-selected="true"] {
        color: var(--primary) !important;
        border-bottom: 2px solid var(--primary) !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        border: 1px solid var(--border-color) !important;
        border-radius: 0.375rem !important;
    }

    /* Label and caption styling */
    label {
        font-weight: 500 !important;
        color: #212529 !important;
        margin-bottom: 0.5rem !important;
    }

    .stCaption {
        color: #6c757d !important;
        font-size: 0.875rem !important;
    }

    /* Text input styling */
    input, textarea, select {
        border: 1px solid var(--border-color) !important;
        border-radius: 0.375rem !important;
        padding: 0.5rem 0.75rem !important;
        font-family: inherit !important;
    }

    input:focus, textarea:focus, select:focus {
        border-color: var(--primary) !important;
        box-shadow: 0 0 0 0.2rem rgba(13, 110, 253, 0.25) !important;
    }

    /* Metric styling */
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #6c757d !important;
        font-size: 0.875rem !important;
    }

    /* Spacing utilities */
    .mt-3 { margin-top: 1rem !important; }
    .mb-3 { margin-bottom: 1rem !important; }
    .p-3 { padding: 1rem !important; }

    /* Login form polish */
    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.9) !important;
    }

    button[kind="primary"], button[data-testid="baseButton-primary"] {
        box-shadow: 0 10px 30px rgba(13, 110, 253, 0.25) !important;
    }

    /* Premium card gradient */
    .premium-card {
        background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
        color: #f8fafc;
        border-radius: 0.75rem;
        padding: 1.25rem;
        box-shadow: 0 14px 35px rgba(0,0,0,0.15);
    }

    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero_banner(title: str, subtitle: str = "", icon: str = "📊") -> None:
    """Render a Bootstrap-style hero banner section."""
    import streamlit as st

    banner_html = f"""
    <div style="
        background: linear-gradient(135deg, #0d6efd 0%, #0dcaf0 100%);
        color: white;
        padding: 2rem;
        border-radius: 0.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    ">
        <h1 style="
            color: white !important;
            border-bottom: none !important;
            margin: 0 0 0.5rem 0 !important;
            font-size: 2.5rem !important;
        ">{icon} {title}</h1>
        <p style="
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.1rem;
            margin: 0;
        ">{subtitle}</p>
    </div>
    """
    st.markdown(banner_html, unsafe_allow_html=True)


def render_stat_card(label: str, value: str, icon: str = "📈") -> str:
    """Generate HTML for a Bootstrap-style stat card."""
    return f"""
    <div style="
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0.125rem 0.25rem rgba(0, 0, 0, 0.075);
    ">
        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</div>
        <div style="font-weight: 600; font-size: 1.5rem; color: #0d6efd;">{value}</div>
        <div style="color: #6c757d; font-size: 0.875rem; margin-top: 0.5rem;">{label}</div>
    </div>
    """


def render_alert(message: str, alert_type: str = "info", title: str = "") -> None:
    """Render a Bootstrap-style alert box."""
    import streamlit as st

    colors = {
        "info": ("#0dcaf0", "#cff4fc"),
        "success": ("#198754", "#d1e7dd"),
        "warning": ("#ffc107", "#fff3cd"),
        "danger": ("#dc3545", "#f8d7da"),
    }
    border_color, bg_color = colors.get(alert_type, colors["info"])

    alert_html = f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    ">
        {f'<strong style="color: #212529;">{title}</strong><br>' if title else ''}
        <div style="color: #212529;">{message}</div>
    </div>
    """
    st.markdown(alert_html, unsafe_allow_html=True)
