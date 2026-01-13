"""
Enhanced Data Science Academy Page
===================================
Comprehensive ML curriculum with supervised and unsupervised learning modules.
"""

import streamlit as st
import pandas as pd
import numpy as np
from src.core.config import load_config
from src.core.ui import app_header, instruction_block, sidebar_dataset_status, page_navigation
from src.core.standardized_ui import standard_section_header, beginner_tip, concept_explainer
from src.core.styles import inject_custom_css
from src.core.ai_helper import ai_sidebar_assistant
from src.academy.ml_curriculum import get_curriculum, format_module

st.set_page_config(page_title="DataScope Pro - ML Academy", layout="wide", initial_sidebar_state="expanded")

config = load_config()
inject_custom_css()
ai_sidebar_assistant()

# ==================== Header ====================
app_header(
    config,
    page_title="ML Academy 2.0",
    subtitle="Master Supervised & Unsupervised Learning | 10 Complete Modules | Code Examples Included",
    icon="🎓"
)

instruction_block(
    "Academy Structure",
    [
        "5️⃣ Supervised Learning Modules: Classification, Regression, Evaluation",
        "5️⃣ Unsupervised Learning Modules: Clustering, Dimensionality Reduction, Anomaly Detection",
        "📚 Each module includes: concepts, learning outcomes, and copy-pasteable code",
        "🎯 Progressive difficulty: Fundamentals → Advanced Techniques → Real-World Applications",
        "💡 Learn by doing: Practice with sample code and your own data",
    ],
)

# ==================== Module Selection ====================
standard_section_header("Choose Your Learning Path", "🛤️")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Supervised Learning")
    st.markdown("""
    **Learn to predict labels with data that has target values**
    
    - Classification: Predict categories
    - Regression: Predict numbers
    - Model comparison and evaluation
    - Feature importance analysis
    
    **Best for:** Prediction tasks with labeled data
    """)
    learn_supervised = st.checkbox("Learn Supervised Learning", value=True, key="learn_sup")

with col2:
    st.markdown("### 🔍 Unsupervised Learning")
    st.markdown("""
    **Discover patterns without labels**
    
    - Clustering: Group similar items
    - Dimensionality reduction: Compress data
    - Anomaly detection: Find outliers
    - Pattern discovery
    
    **Best for:** Exploration and pattern discovery
    """)
    learn_unsupervised = st.checkbox("Learn Unsupervised Learning", value=True, key="learn_unsup")

curriculum = get_curriculum()

# ==================== SUPERVISED LEARNING ====================
if learn_supervised:
    st.markdown("---")
    st.markdown("## 🎯 Supervised Learning Curriculum")
    
    st.markdown("""
    ### Why Supervised Learning?
    When you have historical data with known outcomes (labels), supervised learning helps you build 
    models to predict future outcomes. This is the foundation of most business ML applications.
    
    **Real-world examples:**
    - Email: Predict if message is spam
    - Finance: Predict if loan applicant will default
    - Healthcare: Predict disease diagnosis from symptoms
    - Retail: Predict customer churn likelihood
    """)
    
    beginner_tip("""
    💡 **Pro Tip:** Always start with simple models (Logistic Regression, Linear Regression) before 
    moving to complex ones (Neural Networks). Simpler models are faster to train, easier to interpret, 
    and often perform just as well!
    """)
    
    # Module selection
    sup_modules = st.multiselect(
        "Select modules to study:",
        ["Module 1: Fundamentals", "Module 2: EDA", "Module 3: Feature Engineering", 
         "Module 4: Classification", "Module 5: Regression"],
        default=["Module 1: Fundamentals"]
    )
    
    # Display selected modules
    module_map = {
        "Module 1: Fundamentals": "module_1",
        "Module 2: EDA": "module_2",
        "Module 3: Feature Engineering": "module_3",
        "Module 4: Classification": "module_4",
        "Module 5: Regression": "module_5"
    }
    
    for module_name in sup_modules:
        module_key = module_map[module_name]
        module_data = curriculum['supervised'][module_key]
        
        with st.expander(f"📖 {module_data['title']}", expanded=False):
            st.markdown(format_module(module_data))
            
            # Practice section
            st.markdown("### 🏋️ Practice Questions")
            if module_key == "module_1":
                st.markdown("""
                1. What's the difference between classification and regression?
                2. Why is train-test split important?
                3. What's the difference between overfitting and underfitting?
                4. How does cross-validation help model evaluation?
                5. When would you use 80-20 split vs 70-30 split?
                """)
            elif module_key == "module_4":
                st.markdown("""
                1. When would you use SVM vs Random Forest for classification?
                2. How do you handle class imbalance in classification?
                3. What's the difference between precision and recall?
                4. How do you interpret confusion matrix values?
                5. When would you optimize for precision vs recall?
                """)

# ==================== UNSUPERVISED LEARNING ====================
if learn_unsupervised:
    st.markdown("---")
    st.markdown("## 🔍 Unsupervised Learning Curriculum")
    
    st.markdown("""
    ### Why Unsupervised Learning?
    When you don't have labels, unsupervised learning helps you discover hidden patterns, 
    segment data, reduce complexity, or detect anomalies. Essential for exploratory analysis.
    
    **Real-world examples:**
    - Customer segmentation by behavior
    - Document clustering by topic
    - Network anomaly detection
    - Gene sequence clustering
    - Visualizing high-dimensional data
    """)
    
    beginner_tip("""
    💡 **Pro Tip:** Always **normalize/scale** your features before clustering! 
    Different scales can bias your results. Use StandardScaler before fitting.
    """)
    
    # Module selection
    unsup_modules = st.multiselect(
        "Select modules to study:",
        ["Module 1: Clustering Fundamentals", "Module 2: K-Means", 
         "Module 3: Advanced Clustering", "Module 4: Dimensionality Reduction", 
         "Module 5: Anomaly Detection"],
        default=["Module 1: Clustering Fundamentals"]
    )
    
    # Display selected modules
    module_map_unsup = {
        "Module 1: Clustering Fundamentals": "module_1",
        "Module 2: K-Means": "module_2",
        "Module 3: Advanced Clustering": "module_3",
        "Module 4: Dimensionality Reduction": "module_4",
        "Module 5: Anomaly Detection": "module_5"
    }
    
    for module_name in unsup_modules:
        module_key = module_map_unsup[module_name]
        module_data = curriculum['unsupervised'][module_key]
        
        with st.expander(f"📖 {module_data['title']}", expanded=False):
            st.markdown(format_module(module_data))
            
            # Practice section
            st.markdown("### 🏋️ Practice Questions")
            if module_key == "module_1":
                st.markdown("""
                1. What's the main difference between clustering and classification?
                2. Name three distance metrics and when to use each.
                3. What does it mean for a clustering algorithm to scale poorly?
                4. How would you choose between K-Means and DBSCAN?
                5. What metrics would you use to evaluate cluster quality?
                """)
            elif module_key == "module_4":
                st.markdown("""
                1. What's the goal of dimensionality reduction?
                2. What does "explained variance ratio" tell you in PCA?
                3. When would you use t-SNE vs PCA?
                4. Why is UMAP becoming popular for visualization?
                5. How do you decide how many dimensions to keep?
                """)

# ==================== Quick Reference ====================
st.markdown("---")
standard_section_header("Quick Reference: Essential Commands", "⚡")

st.markdown("### Supervised Learning")
st.code("""
# Complete supervised learning pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Prepare data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 4. Evaluate
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
""", language="python")

st.markdown("### Unsupervised Learning")
st.code("""
# Complete clustering + dimensionality reduction pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# 1. Prepare data
X = df.select_dtypes(include=[np.number])
X_scaled = StandardScaler().fit_transform(X)

# 2. Find optimal clusters
inertias = []
silhouette_scores = []
for k in range(2, 11):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, km.labels_))

# 3. Train final model
optimal_k = np.argmax(silhouette_scores) + 2
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 4. Reduce dimensions for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)
""", language="python")

# ==================== Learning Tips ====================
st.markdown("---")
standard_section_header("🎯 Academy Success Tips", "💡")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 1. Code Along
    Don't just read - write the code yourself. Make mistakes, debug, learn.
    """)

with col2:
    st.markdown("""
    ### 2. Use Your Data
    Apply concepts to your own datasets. Real data teaches more than samples.
    """)

with col3:
    st.markdown("""
    ### 3. Experiment
    Try different parameters, models, techniques. Build intuition through practice.
    """)

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("""
    ### 4. Document
    Write down what you learned, challenges faced, solutions found.
    """)

with col5:
    st.markdown("""
    ### 5. Compare
    Train multiple models and compare results. Understand trade-offs.
    """)

with col6:
    st.markdown("""
    ### 6. Deep Dive
    When confused, research thoroughly. Read papers, watch videos, ask experts.
    """)

# ==================== Resources ====================
st.markdown("---")
standard_section_header("📚 Additional Resources", "📖")

st.markdown("""
### Official Documentation
- **Scikit-learn:** https://scikit-learn.org/stable/
- **TensorFlow/Keras:** https://www.tensorflow.org/
- **XGBoost:** https://xgboost.readthedocs.io/
- **Pandas:** https://pandas.pydata.org/

### Books
- "Hands-On Machine Learning" by Aurélien Géron
- "Introduction to Statistical Learning" (ISLR)
- "Pattern Recognition and Machine Learning" (PRML)

### Online Courses
- Coursera: Machine Learning Specialization
- Fast.ai: Practical Deep Learning for Coders
- Kaggle Learn: Free micro-courses

### Communities
- Kaggle: Competitions and datasets
- Stack Overflow: Q&A for coding issues
- Reddit: r/MachineLearning, r/datascience
- Towards Data Science: Medium articles
""")

page_navigation("22")
