"""
Enhanced Modern CSS Styling System
Premium SaaS design with professional aesthetics
"""

from __future__ import annotations

import streamlit as st


def inject_premium_css() -> None:
    """Inject premium CSS for modern, professional UI."""
    st.markdown("""
    <style>
    /* ===== GLOBAL THEME & TYPOGRAPHY ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Main content area */
    .main {
        padding: 2rem 3rem !important;
        background: linear-gradient(135deg, #f5f7fa 0%, #f8f9fb 100%) !important;
        max-width: 100% !important;
    }
    
    /* Block container spacing to prevent overlaps */
    .block-container {
        padding: 2rem 1rem !important;
        max-width: 100% !important;
    }
    
    /* Fix overlapping elements */
    [data-testid="stVerticalBlock"] {
        gap: 1rem !important;
    }
    
    [data-testid="stHorizontalBlock"] {
        gap: 1rem !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%) !important;
        padding: 1.5rem 1rem !important;
        z-index: 999 !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
        font-weight: 700 !important;
        margin-top: 1rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* ===== HEADERS & TEXT ===== */
    h1 {
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-top: 0 !important;
        margin-bottom: 1.5rem !important;
        padding-bottom: 0.5rem !important;
        letter-spacing: -0.02em;
        line-height: 1.2 !important;
    }
    
    h2 {
        font-weight: 700 !important;
        color: #1e293b !important;
        font-size: 1.875rem !important;
        margin-top: 2.5rem !important;
        margin-bottom: 1.25rem !important;
        letter-spacing: -0.01em;
        line-height: 1.3 !important;
        clear: both !important;
    }
    
    h3 {
        font-weight: 600 !important;
        color: #334155 !important;
        font-size: 1.5rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
        letter-spacing: -0.01em;
        line-height: 1.4 !important;
        clear: both !important;
    }
    
    h4, h5, h6 {
        margin-top: 1.5rem !important;
        margin-bottom: 0.875rem !important;
        line-height: 1.5 !important;
        clear: both !important;
    }
    
    p, li, span {
        color: #475569 !important;
        line-height: 1.7 !important;
        font-size: 0.95rem !important;
        margin-bottom: 0.75rem !important;
    }
    
    /* Ensure text doesn't overlap with other elements */
    p, div, span {
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* ===== BUTTONS ===== */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.625rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.01em !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    
    /* Secondary buttons */
    .stButton > button[kind="secondary"] {
        background: white !important;
        color: #667eea !important;
        border: 2px solid #667eea !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15) !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background: #f8f9fb !important;
        border-color: #764ba2 !important;
    }
    
    /* ===== INPUT FIELDS ===== */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        border: 2px solid #e2e8f0 !important;
        border-radius: 8px !important;
        padding: 0.625rem 1rem !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        background: white !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* ===== CARDS & CONTAINERS ===== */
    [data-testid="stExpander"] {
        background: white !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        margin: 1.5rem 0 !important;
        padding: 0 !important;
        transition: all 0.2s ease !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    [data-testid="stExpander"]:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
        border-color: #cbd5e1 !important;
    }
    
    [data-testid="stExpander"] summary {
        font-weight: 600 !important;
        color: #1e293b !important;
        font-size: 1.05rem !important;
        padding: 1rem 1.25rem !important;
        margin: 0 !important;
    }
    
    [data-testid="stExpander"] > div {
        padding: 1rem 1.25rem !important;
        margin: 0 !important;
    }
    
    /* Markdown containers spacing */
    [data-testid="stMarkdownContainer"] {
        margin-bottom: 1rem !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    /* Ensure proper stacking context */
    [data-testid="stMarkdownContainer"] > * {
        margin-bottom: 0.5rem !important;
    }
    
    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: white !important;
        padding: 1.25rem !important;
        border-radius: 12px !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
        margin: 0.75rem 0 !important;
        position: relative !important;
        z-index: 1 !important;
    }
    
    [data-testid="stMetric"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    }
    
    [data-testid="stMetric"] label {
        font-weight: 600 !important;
        color: #64748b !important;
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
        margin-bottom: 0.5rem !important;
    }
    
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-weight: 700 !important;
        font-size: 2rem !important;
        color: #1e293b !important;
        line-height: 1.2 !important;
        margin: 0.5rem 0 !important;
    }
    
    /* ===== DATAFRAMES ===== */
    [data-testid="stDataFrame"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
    }
    
    [data-testid="stDataFrame"] table {
        font-size: 0.875rem !important;
    }
    
    [data-testid="stDataFrame"] th {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
        color: #1e293b !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em !important;
        padding: 0.875rem 1rem !important;
        border-bottom: 2px solid #cbd5e1 !important;
    }
    
    [data-testid="stDataFrame"] td {
        padding: 0.75rem 1rem !important;
        border-bottom: 1px solid #f1f5f9 !important;
    }
    
    /* ===== TABS ===== */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem !important;
        background: white !important;
        padding: 0.5rem !important;
        border-radius: 10px !important;
        border: 1px solid #e2e8f0 !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 3rem !important;
        padding: 0 1.5rem !important;
        background: transparent !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        color: #64748b !important;
        transition: all 0.2s ease !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: #f8f9fb !important;
        color: #334155 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
    }
    
    /* ===== ALERTS & MESSAGES ===== */
    .stAlert {
        border-radius: 10px !important;
        border: none !important;
        padding: 1rem 1.25rem !important;
        font-size: 0.9rem !important;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06) !important;
        margin: 1rem 0 !important;
        position: relative !important;
        z-index: 1 !important;
        line-height: 1.6 !important;
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%) !important;
        color: #065f46 !important;
        border-left: 4px solid #10b981 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%) !important;
        color: #1e40af !important;
        border-left: 4px solid #3b82f6 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%) !important;
        color: #92400e !important;
        border-left: 4px solid #f59e0b !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%) !important;
        color: #991b1b !important;
        border-left: 4px solid #ef4444 !important;
    }
    
    /* Alert icons and text should not overlap */
    .stAlert svg {
        margin-right: 0.75rem !important;
    }
    
    .stAlert p {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* ===== SLIDERS ===== */
    .stSlider > div > div > div {
        background: #e2e8f0 !important;
    }
    
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* ===== FILE UPLOADER ===== */
    [data-testid="stFileUploader"] {
        background: white !important;
        border: 2px dashed #cbd5e1 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea !important;
        background: #f8f9fb !important;
    }
    
    /* ===== PLOTLY CHARTS ===== */
    .js-plotly-plot {
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
        background: white !important;
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    
    /* ===== SCROLLBAR ===== */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* ===== DIVIDERS ===== */
    hr {
        margin: 2rem 0 !important;
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent 0%, #cbd5e1 50%, transparent 100%) !important;
    }
    
    /* ===== TOOLTIPS ===== */
    [data-testid="stTooltipIcon"] {
        color: #667eea !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stTooltipIcon"]:hover {
        color: #764ba2 !important;
        transform: scale(1.1) !important;
    }
    
    /* ===== LOADING SPINNER ===== */
    .stSpinner > div {
        border-top-color: #667eea !important;
        border-right-color: #764ba2 !important;
    }
    
    /* ===== PROGRESS BAR ===== */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* ===== RADIO BUTTONS & CHECKBOXES ===== */
    .stRadio > label, .stCheckbox > label {
        font-weight: 500 !important;
        color: #334155 !important;
    }
    
    /* ===== ANIMATIONS ===== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    [data-testid="stMarkdownContainer"],
    [data-testid="stMetric"],
    [data-testid="stExpander"] {
        animation: fadeIn 0.3s ease-out;
    }
    
    /* ===== RESPONSIVE IMPROVEMENTS ===== */
    @media (max-width: 768px) {
        .main {
            padding: 1rem !important;
        }
        
        h1 {
            font-size: 2rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
    }
    
    /* ===== CODE BLOCKS ===== */
    code {
        background: #f1f5f9 !important;
        padding: 0.2rem 0.4rem !important;
        border-radius: 4px !important;
        font-size: 0.875rem !important;
        color: #be185d !important;
        font-weight: 500 !important;
    }
    
    pre {
        background: #1e293b !important;
        border-radius: 10px !important;
        padding: 1.25rem !important;
        border: 1px solid #334155 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    pre code {
        background: transparent !important;
        color: #e2e8f0 !important;
    }
    
    /* ===== COLUMNS ===== */
    [data-testid="column"] {
        padding: 0 0.75rem !important;
        position: relative !important;
    }
    
    /* First column - remove left padding */
    [data-testid="column"]:first-child {
        padding-left: 0 !important;
    }
    
    /* Last column - remove right padding */
    [data-testid="column"]:last-child {
        padding-right: 0 !important;
    }
    
    /* Prevent column content overflow */
    [data-testid="column"] > * {
        max-width: 100% !important;
        overflow-wrap: break-word !important;
    }
    
    /* ===== DOWNLOAD BUTTONS ===== */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.625rem 1.5rem !important;
        font-weight: 600 !important;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.4) !important;
    }
    
    /* ===== CRITICAL SPACING & OVERLAP FIXES ===== */
    
    /* Ensure all major containers have proper spacing */
    .element-container {
        margin-bottom: 1rem !important;
    }
    
    /* Fix text overlapping in forms */
    .stForm {
        padding: 1rem !important;
        margin-bottom: 1.5rem !important;
    }
    
    /* Label spacing to prevent overlap with inputs */
    .stTextInput label,
    .stSelectbox label,
    .stNumberInput label,
    .stTextArea label {
        display: block !important;
        margin-bottom: 0.5rem !important;
        line-height: 1.5 !important;
    }
    
    /* Input field spacing */
    .stTextInput,
    .stSelectbox,
    .stNumberInput,
    .stTextArea {
        margin-bottom: 1.25rem !important;
    }
    
    /* Radio and checkbox spacing */
    .stRadio,
    .stCheckbox {
        margin: 1rem 0 !important;
    }
    
    .stRadio > div,
    .stCheckbox > div {
        gap: 0.75rem !important;
    }
    
    /* Prevent button text overlap */
    .stButton {
        margin: 0.5rem 0 !important;
    }
    
    .stButton button {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Table header spacing */
    thead th {
        padding: 1rem !important;
        white-space: nowrap !important;
    }
    
    /* Fix navigation button overlaps */
    .stButton[data-testid*="page"] button {
        min-height: 2.5rem !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Ensure dividers have proper spacing */
    hr, .stDivider {
        margin: 2rem 0 !important;
        clear: both !important;
    }
    
    /* Fix plotly chart container spacing */
    .js-plotly-plot {
        margin: 1.5rem 0 !important;
        clear: both !important;
    }
    
    /* Prevent sidebar content overlap */
    [data-testid="stSidebar"] > div > div {
        gap: 1rem !important;
    }
    
    /* Widget spacing in sidebar */
    [data-testid="stSidebar"] .stButton,
    [data-testid="stSidebar"] .stSelectbox,
    [data-testid="stSidebar"] .stTextInput {
        margin-bottom: 1rem !important;
    }
    
    /* Fix multiselect dropdown spacing */
    .stMultiSelect {
        margin-bottom: 1.25rem !important;
    }
    
    /* Ensure proper line height for all text elements */
    * {
        line-height: inherit !important;
    }
    
    /* Container queries for better responsive layout */
    @container (max-width: 640px) {
        [data-testid="column"] {
            margin-bottom: 1rem !important;
        }
    }
    
    /* Fix overlapping when using st.columns */
    .row-widget {
        gap: 1rem !important;
    }
    
    /* Ensure st.container has proper spacing */
    .stContainer {
        padding: 1rem !important;
        margin: 1rem 0 !important;
    }
    
    /* Fix caption text not overlapping with content below */
    .stCaption {
        display: block !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Clear floats to prevent overlaps */
    .clearfix::after {
        content: "" !important;
        display: table !important;
        clear: both !important;
    }
    </style>
    """, unsafe_allow_html=True)


def get_plotly_theme() -> dict:
    """Return modern Plotly theme configuration."""
    return {
        'layout': {
            'font': {
                'family': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
                'size': 12,
                'color': '#1e293b'
            },
            'title': {
                'font': {
                    'size': 18,
                    'color': '#1e293b',
                    'family': 'Inter, sans-serif'
                },
                'x': 0.5,
                'xanchor': 'center'
            },
            'paper_bgcolor': 'white',
            'plot_bgcolor': '#f8fafc',
            'colorway': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#10b981', '#f59e0b', '#3b82f6', '#8b5cf6'],
            'hovermode': 'closest',
            'hoverlabel': {
                'bgcolor': 'white',
                'font': {'size': 12, 'color': '#1e293b', 'family': 'Inter'},
                'bordercolor': '#e2e8f0'
            },
            'margin': {'l': 60, 'r': 40, 't': 60, 'b': 60},
            'xaxis': {
                'showgrid': True,
                'gridcolor': '#e2e8f0',
                'showline': True,
                'linecolor': '#cbd5e1',
                'tickfont': {'size': 11, 'color': '#64748b'}
            },
            'yaxis': {
                'showgrid': True,
                'gridcolor': '#e2e8f0',
                'showline': True,
                'linecolor': '#cbd5e1',
                'tickfont': {'size': 11, 'color': '#64748b'}
            }
        }
    }
