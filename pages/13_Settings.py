from __future__ import annotations

import os
import sys

import streamlit as st

st.set_page_config(page_title="DataScope Pro - Settings", layout="wide", initial_sidebar_state="expanded")

from src.core.config import load_config, save_config
from src.core.state import get_clean_df, get_df
from src.core.ui import sidebar_dataset_status, instruction_block, page_navigation
from src.core.styles import inject_custom_css


config_obj = load_config()

# Apply custom CSS
inject_custom_css()

st.markdown(
    """
    <div style="background: #0b5ed7; color: #f8fafc; padding: 18px 20px; border-radius: 12px; margin-bottom: 16px;">
        <div style="font-size: 24px; font-weight: 800;">⚙️ Settings</div>
        <div style="font-size: 15px; opacity: 0.95; margin-top: 6px;">Tweak defaults, save config.yaml, and keep secrets out of the codebase.</div>
    </div>
    """,
    unsafe_allow_html=True,
)
instruction_block(
    "How to use this page",
    [
        "Set the app title, refresh rate, and preview row limit to match your hardware.",
        "Pick default tabular/forecast models and AI settings, then save to config.yaml.",
        "Keep secrets in environment variables or Streamlit secrets, not in config.yaml.",
        "Use the dependencies tab for the commands to enable extras like XGBoost or transformers.",
    ],
)

st.info(
    "Configure the app safely: adjust defaults, keep keys out of the repo, and document optional extras.",
    icon="ℹ️",
)

raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)

sidebar_dataset_status(raw_df, clean_df)

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["⚙️ Application Settings", "🔐 Secrets Management", "📦 Dependencies", "🌍 Environment"])

raw = dict(config_obj.raw)

# ===== TAB 1: APPLICATION SETTINGS =====
with tab1:
    st.subheader("Application Settings")
    st.caption("These settings are stored in config.yaml (non-secrets).")
    
    col1, col2 = st.columns(2)
    
    with col1:
        app_title = st.text_input(
            "App title",
            value=str(raw.get("app", {}).get("title", config_obj.title)),
            help="Display name for the application"
        )
        refresh = st.number_input(
            "Refresh rate (seconds)",
            min_value=5,
            max_value=3600,
            value=int(config_obj.refresh_rate_seconds),
            step=5,
            help="How often to check for data updates"
        )
    
    with col2:
        max_rows_preview = st.number_input(
            "Max preview rows",
            min_value=100,
            max_value=100000,
            value=int(config_obj.max_rows_preview),
            step=100,
            help="Limit rows displayed in data previews"
        )
    
    st.divider()
    
    st.subheader("Default Models")
    col1, col2 = st.columns(2)
    
    with col1:
        def_tabular = st.selectbox(
            "Default tabular model",
            options=["RandomForest", "XGBoost", "LightGBM", "GradientBoosting"],
            index=0,
            help="Default algorithm for structured data"
        )
    
    with col2:
        def_forecast = st.selectbox(
            "Default forecast model",
            options=["Transformer"],
            index=0,
            help="Default model for time-series forecasting"
        )
    
    st.divider()
    
    st.subheader("AI Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        ai_provider = st.selectbox(
            "AI provider",
            options=["local", "openai_compatible"],
            index=0,
            help="Where to run language models"
        )
        ai_model = st.text_input(
            "AI model",
            value=str(raw.get("ai", {}).get("model", config_obj.ai_model)),
            help="Model identifier (e.g., 'gpt-3.5-turbo' or 'mistral-7b')"
        )
    
    with col2:
        ai_tokens = st.number_input(
            "AI max new tokens",
            min_value=64,
            max_value=4096,
            value=int(config_obj.ai_max_new_tokens),
            step=64,
            help="Maximum tokens to generate in AI responses"
        )
    
    st.divider()
    
    # Save button
    if st.button("💾 Save All Settings", type="primary", use_container_width=True):
        new_raw = {
            **raw,
            "app": {"title": app_title, "refresh_rate_seconds": int(refresh), "default_task": raw.get("app", {}).get("default_task", "auto")},
            "data": {"max_rows_preview": int(max_rows_preview)},
            "models": {
                "registry_dir": raw.get("models", {}).get("registry_dir", "models"),
                "artifacts_dir": raw.get("models", {}).get("artifacts_dir", "artifacts"),
                "default_tabular_model": def_tabular,
                "default_forecast_model": def_forecast,
            },
            "ai": {"provider": ai_provider, "model": ai_model, "max_new_tokens": int(ai_tokens)},
            "logging": raw.get("logging", {}),
        }
        save_config(new_raw)
        st.success("✅ Configuration saved to config.yaml")
    
    # Info box
    with st.expander("💡 Best Practices", expanded=False):
        st.write("""
        - **App Title**: Use a descriptive name that reflects your analysis focus
        - **Refresh Rate**: Lower values = more frequent updates (requires more compute)
        - **Preview Rows**: Higher values help but may slow down the interface
        - **Default Models**: Choose based on your typical use case to save configuration time
        - **AI Provider**: "local" is free but slower; "openai_compatible" requires API key
        """)


# ===== TAB 2: SECRETS MANAGEMENT =====
with tab2:
    st.subheader("🔐 Secrets Management")
    st.caption("Keep API keys and sensitive data secure—never commit them to version control.")
    
    st.warning(
        "⚠️ **IMPORTANT**: Never hardcode secrets in config.yaml. Always use environment variables or Streamlit secrets.",
        icon="🔒"
    )
    
    st.subheader("Setup Methods")
    
    with st.expander("📋 Method 1: Environment Variables (Recommended)", expanded=True):
        st.write("Set these in your terminal or system environment:")
        st.code(
            'export OPENAI_BASE_URL="https://api.openai.com/v1"\n'
            'export OPENAI_API_KEY="sk-..."\n',
            language="bash"
        )
        st.caption("On macOS/Linux, add to ~/.zshrc or ~/.bashrc for persistence.")
    
    with st.expander("🔑 Method 2: Streamlit Secrets (For Deployment)", expanded=False):
        st.write("Create `.streamlit/secrets.toml` in your project:")
        st.code(
            'OPENAI_BASE_URL = "https://api.openai.com/v1"\n'
            'OPENAI_API_KEY = "sk-..."\n',
            language="toml"
        )
        st.caption("Streamlit handles these securely during deployment.")
    
    with st.expander("🛡️ Security Best Practices", expanded=False):
        st.write("""
        1. **Never commit secrets**: Add .env and secrets.toml to .gitignore
        2. **Use environment variables**: Preferred for local development
        3. **Rotate keys regularly**: Update API keys if compromised
        4. **Limit permissions**: Use API tokens with minimal required scopes
        5. **Monitor usage**: Check your API provider's dashboard for unusual activity
        6. **Use secrets manager**: For production, consider AWS Secrets Manager, HashiCorp Vault, etc.
        """)
    
    st.subheader("Current Secrets Status")
    secrets_dict = {
        "OPENAI_BASE_URL": "✅ Set" if os.getenv("OPENAI_BASE_URL") else "❌ Not found",
        "OPENAI_API_KEY": "✅ Set" if os.getenv("OPENAI_API_KEY") else "❌ Not found",
    }
    
    col1, col2 = st.columns(2)
    for i, (key, status) in enumerate(secrets_dict.items()):
        with col1 if i % 2 == 0 else col2:
            st.metric(f"{key}", status)


# ===== TAB 3: DEPENDENCIES =====
with tab3:
    st.subheader("📦 Optional Dependencies")
    st.caption("Enable advanced models by installing optional packages.")
    
    dependencies = {
        "XGBoost": {
            "command": "pip install xgboost",
            "description": "Gradient boosting for tabular data—often best for structured datasets",
            "when_to_use": "Competing in ML competitions or need maximum accuracy"
        },
        "LightGBM": {
            "command": "pip install lightgbm",
            "description": "Fast gradient boosting—uses less memory, trains faster than XGBoost",
            "when_to_use": "Large datasets or limited compute resources"
        },
        "Transformers & PyTorch": {
            "command": "pip install transformers torch",
            "description": "Deep learning models for NLP and advanced analysis",
            "when_to_use": "Text analysis, embeddings, or transfer learning"
        },
        "TFT (Temporal Fusion)": {
            "command": "pip install pytorch-forecasting pytorch-lightning torch",
            "description": "Attention-based time-series forecasting—state-of-the-art for time-series",
            "when_to_use": "Complex time-series patterns with multiple features"
        },
    }
    
    for i, (name, info) in enumerate(dependencies.items()):
        with st.expander(f"📦 {name}", expanded=(i < 2)):
            st.write(f"**Description**: {info['description']}")
            st.write(f"**When to use**: {info['when_to_use']}")
            st.code(info['command'], language="bash")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                st.caption("Copy the command above and run in your terminal")
            with col2:
                if st.button("Copy", key=f"copy_{name}"):
                    st.success("Copied!")
    
    st.divider()
    
    with st.expander("🔗 All Dependencies at Once", expanded=False):
        all_commands = "\n".join([info['command'] for info in dependencies.values()])
        st.code(all_commands, language="bash")
        st.caption("Run all optional packages at once (not recommended for first-time setup)")


# ===== TAB 4: ENVIRONMENT =====
with tab4:
    st.subheader("🌍 Environment & Diagnostics")
    st.caption("View your system configuration and check what's available.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Python Version", f"{sys.version.split()[0]}")
    with col2:
        st.metric("Streamlit Version", st.__version__)
    with col3:
        st.metric("OS Platform", sys.platform.capitalize())
    
    st.divider()
    
    st.subheader("Environment Variables")
    st.caption("Variables available to the application:")
    
    env_display = {
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "Not set"),
        "OPENAI_API_KEY": "***HIDDEN***" if os.getenv("OPENAI_API_KEY") else "Not set",
        "PATH (length)": f"{len(os.getenv('PATH', ''))} chars",
        "HOME": os.getenv("HOME", "Not set"),
    }
    
    st.json(env_display)
    
    st.divider()
    
    st.subheader("Available ML Libraries")
    st.caption("Check which optional libraries are installed:")
    
    libraries = {
        "XGBoost": "xgboost",
        "LightGBM": "lightgbm",
        "PyTorch": "torch",
        "Transformers": "transformers",
        "PyTorch Lightning": "pytorch_lightning",
    }
    
    col1, col2, col3 = st.columns(3)
    for i, (display_name, lib_name) in enumerate(libraries.items()):
        try:
            __import__(lib_name)
            status = "✅ Installed"
            status_color = "green"
        except Exception:
            status = "❌ Not installed"
            status_color = "red"
        
        with [col1, col2, col3][i % 3]:
            st.write(f"**{display_name}**: {status}")

# Page navigation
page_navigation("13")
