"""
Platform Welcome Guide
First-time user orientation and quick tour
"""

import streamlit as st


def show_welcome_guide():
    """Display welcome guide for new users."""
    
    if 'welcome_shown' not in st.session_state:
        st.session_state.welcome_shown = False
    
    if not st.session_state.welcome_shown:
        with st.expander("👋 Welcome to DataScope Pro! (Click to view platform guide)", expanded=True):
            st.markdown("""
            # 🔬 Welcome to DataScope Pro
            
            **Your Complete Data Science Workspace** - Everything you need for data analysis, 
            machine learning, and insights in one integrated platform.
            
            ## 🚀 Quick Start Guide
            
            ### First-Time Users
            1. **Explore Sample Data** - Go to **Data Exploration** → Sample Data tab
            2. **Try the Academy** - Visit **Data Science Academy** for tutorials
            3. **Use DS Assistant** - Get personalized guidance for your workflow
            
            ### For Your Own Data
            1. **Upload** - Data Exploration → Upload your CSV/Excel/Parquet
            2. **Clean** - Handle missing values, outliers, and data types
            3. **Analyze** - Data Analysis (EDA) → Explore distributions and patterns
            4. **Engineer** - Feature Engineering → Transform and select features
            5. **Model** - Tabular ML → Train and compare models
            6. **Visualize** - Create publication-ready charts
            7. **Predict** - Deploy models for inference
            
            ## 📚 Platform Modules
            
            | Module | Purpose | Best For |
            |--------|---------|----------|
            | 🤖 **DS Assistant** | Guided workflows & help | Getting started, decision support |
            | 🏠 **Data Exploration** | Load & clean data | Data preparation |
            | 📊 **Data Analysis** | Statistical exploration | Understanding patterns |
            | 🔨 **Feature Engineering** | Create better features | Improving model performance |
            | 🎯 **Tabular ML** | Train ML models | Classification & regression |
            | ⏰ **Deep Learning** | Time series & neural nets | Forecasting, sequences |
            | 🎨 **Visualization** | Interactive charts | Presentations, insights |
            | 🔮 **Predictions** | Model deployment | Production inference |
            | 💡 **AI Insights** | Natural language help | Interpretation, explanations |
            | 🎓 **Academy** | Learn data science | Education, skill building |
            
            ## 💡 Pro Tips
            
            - **AI Help Everywhere**: Look for the AI assistant in the sidebar on every page
            - **Sample Data**: Use built-in datasets to learn without uploading files
            - **Save Progress**: Clean data is automatically saved across pages
            - **Export Results**: Download predictions, models, and visualizations
            - **Interactive Charts**: Click, zoom, pan, and hover on all visualizations
            
            ## 🎯 Common Workflows
            
            ### Classification (Predict Categories)
            Data Exploration → Clean → EDA → Feature Engineering → Tabular ML (Classification) → Predictions
            
            ### Regression (Predict Numbers)  
            Data Exploration → Clean → EDA → Feature Engineering → Tabular ML (Regression) → Predictions
            
            ### Time Series Forecasting
            Data Exploration → Clean → EDA → Deep Learning (Transformer) → Predictions
            
            ### Exploratory Analysis
            Data Exploration → EDA → Visualization → AI Insights
            
            ## 🆘 Need Help?
            
            - **AI Sidebar**: Ask questions directly on any page
            - **Tooltips**: Hover over icons and buttons
            - **Academy**: Comprehensive tutorials with examples
            - **Interpretation Boxes**: Click "Ask AI for more" for deep dives
            
            ---
            
            **Ready to start?** Close this guide and explore the platform!
            """)
            
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🚀 Let's Get Started!", width="stretch", type="primary"):
                    st.session_state.welcome_shown = True
                    st.rerun()


def show_page_tips(page_name: str):
    """Show contextual tips for specific pages."""
    
    tips = {
        "data_exploration": {
            "icon": "🏠",
            "tip": "Start here! Load sample data or upload your own CSV/Excel file to begin.",
            "next_step": "After cleaning, move to Data Analysis (EDA) to explore patterns."
        },
        "eda": {
            "icon": "📊",
            "tip": "Look for patterns in distributions, correlations, and outliers before modeling.",
            "next_step": "Use Feature Engineering to create better predictive features."
        },
        "feature_engineering": {
            "icon": "🔨",
            "tip": "Transform categorical variables and scale numeric features for better models.",
            "next_step": "Now train ML models in Tabular Machine Learning."
        },
        "tabular_ml": {
            "icon": "🎯",
            "tip": "Compare multiple algorithms. The leaderboard shows which performs best.",
            "next_step": "Use your trained model in Predictions for inference."
        },
        "visualization": {
            "icon": "🎨",
            "tip": "All charts are interactive! Zoom, pan, and hover for details.",
            "next_step": "Get AI explanations in AI Insights page."
        },
        "academy": {
            "icon": "🎓",
            "tip": "Learn by doing! Each tutorial includes live code you can modify and run.",
            "next_step": "Apply what you learn on real data in other modules."
        }
    }
    
    if page_name in tips:
        tip_data = tips[page_name]
        st.info(f"{tip_data['icon']} **Tip:** {tip_data['tip']}\n\n➡️ {tip_data['next_step']}", icon="💡")
