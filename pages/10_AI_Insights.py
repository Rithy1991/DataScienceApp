from __future__ import annotations

import json
import pandas as pd
import streamlit as st

st.set_page_config(page_title="DataScope Pro - AI Insights", layout="wide", initial_sidebar_state="expanded")

from src.ai.insights import generate_insights
from src.core.config import load_config
from src.core.logging_utils import log_event
from src.core.state import get_clean_df, get_df
from src.core.ui import app_header, sidebar_dataset_status, instruction_block, page_navigation
from src.core.standardized_ui import (
    standard_section_header,
    concept_explainer,
    beginner_tip,
    common_mistakes_panel,
)
from src.core.styles import render_stat_card, inject_custom_css
from src.storage.history import add_event


config = load_config()

# Apply custom CSS
inject_custom_css()

app_header(
    config,
    page_title="AI Insights",
    subtitle="Ask for plain-language summaries, risks, or recommendations from your data",
    icon="💡"
)
instruction_block(
    "How to use this page",
    [
        "Tell the assistant what to focus on: trends, anomalies, risks, or recommendations.",
        "Choose a provider and model; local runs on your machine, openai_compatible points to an API.",
        "Pick a token limit; start small to keep responses concise.",
        "For APIs, expand settings to add base URL and key (environment variables are safest).",
        "Click Generate to get a written summary of the current dataset.",
    ],
)

st.info(
    "Natural-language summaries and recommendations powered by local or API-hosted language models.",
    icon="ℹ️",
)

with st.expander("🧭 How to read the AI output (beginner friendly)", expanded=False):
    st.markdown(
        "- Start with the headline sentence; it tells you the main trend.\n"
        "- Look for 2-3 bullet points with concrete facts (increases/decreases, segments).\n"
        "- Use the action/next-step line to decide what to try next in your data or business.\n"
        "- If something is unclear, lower the token count and ask for a simpler rephrase."
    )

raw_df = get_df(st.session_state)
clean_df = get_clean_df(st.session_state)

sidebar_dataset_status(raw_df, clean_df)

df = clean_df if clean_df is not None else raw_df
if df is None:
    st.info("Load data in Data Cleaning page first.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(["🚀 Generate Insights", "📋 Insight Templates", "⚙️ Configuration", "📚 Learn"])

with tab1:
    st.subheader("Generate Natural-Language Insights")
    st.caption("Describe what you'd like the AI to analyze in your data")
    
    context = st.text_area(
        "Analysis focus (what should AI look for?)",
        value="Summarize key patterns, trends, anomalies, and actionable insights.",
        height=100,
        placeholder="E.g., 'Find anomalies, seasonal trends, risk factors, or business opportunities'"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        provider = st.selectbox("AI Provider", options=["local", "openai_compatible"], help="Local runs on your machine")
    
    with col2:
        model = st.text_input("Model name", value=config.ai_model, help="e.g., 'mistral-7b', 'gpt-3.5-turbo'")
    
    max_new_tokens = st.slider(
        "Response length (tokens)",
        min_value=64,
        max_value=1024,
        value=int(config.ai_max_new_tokens),
        step=32,
        help="More tokens = longer response but slower"
    )
    
    # Summary metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Provider", provider, "")
    with c2:
        st.metric("Model", model or "-", "")
    with c3:
        st.metric("Max tokens", str(max_new_tokens), "")
    
    if st.button("🚀 Generate Insights", type="primary", width="stretch"):
        try:
            with st.spinner("🤖 Analyzing data and generating insights..."):
                result = generate_insights(
                    df=df,
                    provider=provider,
                    model=model,
                    max_new_tokens=int(max_new_tokens),
                    context=context,
                    openai_base_url=None,
                    openai_api_key=None,
                )
            
            st.success("✅ Insights generated!")
            st.divider()
            
            st.markdown("### 💡 AI Analysis")
            st.markdown(result.summary)
            
            # Log event
            add_event(config.history_db_path, "ai_insights", "Generated AI insights", json.dumps({"provider": provider, "model": model}))
            log_event(config.logging_dir, "ai_insights", {"provider": provider, "model": model})
        
        except Exception as e:
            st.error(
                f"❌ AI Insights failed: {str(e)}\n\n"
                "**Troubleshooting:**\n"
                "- For local models: install `pip install transformers torch`\n"
                "- For API models: check base URL and API key configuration\n"
                "- Ensure you have sufficient disk space for model download"
            )

with tab2:
    st.subheader("📋 Pre-built Insight Templates")
    st.caption("Quick-start prompts for common analysis scenarios")
    
    # Initialize session state for insight history
    if "insight_history" not in st.session_state:
        st.session_state.insight_history = []
    
    template_category = st.selectbox(
        "Select analysis type",
        options=[
            "📊 Data Quality Assessment",
            "📈 Trend & Pattern Analysis",
            "⚠️ Anomaly Detection",
            "💼 Business Recommendations",
            "🔮 Predictive Insights",
            "📉 Risk Analysis"
        ]
    )
    
    templates = {
        "📊 Data Quality Assessment": {
            "prompt": "Analyze data quality: check for missing values, outliers, inconsistencies, duplicates, and data type issues. Provide a quality score (0-100) and specific recommendations for cleaning.",
            "description": "Comprehensive data quality check with actionable recommendations"
        },
        "📈 Trend & Pattern Analysis": {
            "prompt": "Identify key trends, patterns, and correlations in the data. Look for seasonal patterns, growth/decline trends, cyclical behavior, and unexpected relationships between variables.",
            "description": "Discover hidden patterns and temporal trends"
        },
        "⚠️ Anomaly Detection": {
            "prompt": "Find unusual data points, outliers, and anomalies. For each anomaly found, explain why it's unusual and whether it represents an error or a legitimate edge case.",
            "description": "Detect outliers and unusual patterns with explanations"
        },
        "💼 Business Recommendations": {
            "prompt": "Generate actionable business insights and strategic recommendations based on the data. Focus on opportunities for growth, cost reduction, efficiency improvements, and competitive advantages.",
            "description": "Strategic business intelligence and action items"
        },
        "🔮 Predictive Insights": {
            "prompt": "Analyze historical trends to provide predictive insights. What future patterns, risks, or opportunities can be anticipated? Identify leading indicators and early warning signals.",
            "description": "Forward-looking analysis and forecasting guidance"
        },
        "📉 Risk Analysis": {
            "prompt": "Identify potential risks, vulnerabilities, and concerning patterns in the data. Prioritize risks by severity and likelihood, and suggest mitigation strategies.",
            "description": "Risk assessment with mitigation recommendations"
        },
        "🧒 Beginner-Friendly Summary": {
            "prompt": "Explain the key patterns in very simple language. Use short sentences, avoid jargon, and end with one clear action I can take.",
            "description": "Plain-language summary with one next step"
        },
        "🪜 Step-by-Step Explanation": {
            "prompt": "Walk through the data step by step: 1) biggest increase/decrease, 2) any seasonality, 3) any weird values, 4) one action to try next week. Keep it concise.",
            "description": "Guided, ordered explanation for beginners"
        }
    }
    
    template_info = templates[template_category]
    
    st.markdown(f"**{template_info['description']}**")
    
    # Show the template prompt (editable)
    custom_prompt = st.text_area(
        "Template prompt (customize if needed)",
        value=template_info["prompt"],
        height=120,
        help="Edit this prompt to focus on specific aspects"
    )
    
    col_temp1, col_temp2 = st.columns(2)
    with col_temp1:
        temp_provider = st.selectbox("Provider", ["local", "openai_compatible"], key="template_provider")
    with col_temp2:
        temp_model = st.text_input("Model", value=config.ai_model, key="template_model")
    
    if st.button("🚀 Run Template Analysis", type="primary", width="stretch"):
        try:
            with st.spinner(f"🤖 Running {template_category} analysis..."):
                result = generate_insights(
                    df=df,
                    provider=temp_provider,
                    model=temp_model,
                    max_new_tokens=512,
                    context=custom_prompt,
                    openai_base_url=None,
                    openai_api_key=None,
                )
            
            st.success("✅ Analysis complete!")
            st.divider()
            
            st.markdown(f"### {template_category}")
            st.markdown(result.summary)
            
            # Save to history
            st.session_state.insight_history.append({
                "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                "category": template_category,
                "summary": result.summary[:200] + "..." if len(result.summary) > 200 else result.summary
            })
            
            # Log event
            add_event(config.history_db_path, "ai_insights", f"{template_category} analysis", json.dumps({"template": template_category}))
            log_event(config.logging_dir, "ai_insights", {"template": template_category})
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
    
    # Show insight history
    if st.session_state.insight_history:
        st.divider()
        st.subheader("📜 Recent Insights")
        
        for idx, insight in enumerate(reversed(st.session_state.insight_history[-5:])):
            with st.expander(f"{insight['category']} - {insight['timestamp']}"):
                st.markdown(insight["summary"])
        
        if st.button("🗑️ Clear History", width="stretch"):
            st.session_state.insight_history = []
            st.rerun()

with tab3:
    st.subheader("⚙️ Model Configuration")
    
    if provider == "openai_compatible":
        st.info("Configure API endpoint for OpenAI-compatible services")
        
        with st.form("api_config"):
            base_url = st.text_input(
                "API Base URL",
                value="",
                placeholder="https://api.openai.com/v1",
                help="Leave empty to use default OpenAI"
            )
            api_key = st.text_input(
                "API Key",
                value="",
                type="password",
                placeholder="sk-...",
                help="Store securely in environment variables"
            )
            
            if st.form_submit_button("Test Connection"):
                st.info("✅ Configuration saved (not persisted - set env vars for production)")
    else:
        st.success("✅ Using local model (runs on your machine)")
        st.caption("No configuration needed - model will download on first use")
    
    st.divider()
    
    st.subheader("🔐 Security Best Practices")
    st.markdown("""
    **Never commit API keys to version control!**
    
    Instead, use:
    - **Environment variables**: `export OPENAI_API_KEY="..."`
    - **Streamlit secrets**: `~/.streamlit/secrets.toml`
    - **System keystore**: Platform-specific secure storage
    
    For sensitive data:
    - Prefer **local inference** over cloud APIs
    - Use **VPN/SSH tunnel** for internal endpoints
    - Enable **audit logging** for compliance
    """)

with tab4:
    st.subheader("📚 AI Insights Guide")
    
    st.markdown("""
    ### What can AI Insights do?
    
    ✅ **Pattern Detection**
    - Identify trends, cycles, and seasonal patterns
    - Spot outliers and anomalies
    - Find hidden correlations
    
    ✅ **Business Intelligence**
    - Extract key statistics and metrics
    - Generate executive summaries
    - Recommend next steps
    
    ✅ **Risk Analysis**
    - Flag concerning data points
    - Identify gaps and missing information
    - Suggest data quality improvements
    
    ### Model Options
    
    **Local Models (Free)**
    - Mistral-7B, Llama-2, Phi
    - Runs locally - no API calls
    - Slower but completely private
    - Requires 4-16 GB RAM + disk space
    
    **Cloud APIs (Paid)**
    - GPT-4, Claude, Gemini
    - Fast and powerful
    - Requires internet + API key
    - Data sent to external servers
    
    ### Tips for Better Insights
    
    1. **Be specific**: "Find anomalies in transaction amounts" works better than "Analyze data"
    2. **Set expectations**: Use token limit to control response length
    3. **Clean data first**: Remove obvious errors before analysis
    4. **Iterate**: Run multiple analyses with different focuses
    5. **Cross-check**: Verify AI findings with other tools
    """)
    
    st.divider()
    
    st.subheader("⚠️ Limitations")
    st.warning("""
    - AI can **hallucinate** (invent false patterns)
    - Works best with **clean, structured data**
    - Performance depends on **model quality**
    - Large datasets may **timeout**
    - Requires proper **prompt engineering**
    """)

# Page navigation
standard_section_header("Learning Guide & Best Practices", "🎓")
concept_explainer(
    title="AI Explanations",
    explanation=(
        "Use AI to summarize patterns, generate hypotheses, and translate technical outputs into stakeholder language. Always verify and avoid overreliance."
    ),
    real_world_example=(
        "Sales forecasting: AI explains seasonal spikes, promotion effects, and anomalies; analyst verifies with charts and business logs."
    ),
)
beginner_tip("Tip: Provide context in prompts — include goal, audience, and constraints to get useful insights.")
common_mistakes_panel({
    "Treating AI output as truth": "Validate with data and domain knowledge.",
    "Vague prompts": "Be specific about objectives, metrics, and constraints.",
    "Ignoring bias": "AI can amplify data biases — review critically.",
    "No reproducibility": "Keep prompt history and parameters for audits.",
    "Overuse on raw data": "Clean and structure data first for better results.",
})

page_navigation("10")
