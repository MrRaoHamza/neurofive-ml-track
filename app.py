import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------
# Required for Joblib Unpickling & Custom Feature Engineering
# Must be defined in top-level module scope
# ---------------------------------------------------------
def engineer_titanic_features(X):
    """
    Feature engineering function matching pipeline definition.
    Computes FamilySize, IsAlone, and FarePerPerson dynamically.
    """
    X_out = X.copy()
    X_out['FamilySize'] = X_out['SibSp'] + X_out['Parch'] + 1
    X_out['IsAlone'] = (X_out['FamilySize'] == 1).astype(int)
    X_out['FarePerPerson'] = X_out['Fare'] / X_out['FamilySize']
    return X_out

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Inference Engine",
    page_icon="⚓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Professional Modern Styling (Clean Enterprise UI)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    /* Header Container */
    .app-header {
        background: linear-gradient(180deg, #131b2e 0%, #0b0f19 100%);
        border-bottom: 1px solid #1e293b;
        padding: 2.5rem 0 1.5rem 0;
        margin-bottom: 2rem;
    }
    
    .app-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #f8fafc;
        margin: 0;
    }
    
    .app-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 0.3rem;
        font-weight: 400;
    }

    /* Cards & Containers */
    .panel-card {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
    }
    
    .panel-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #f3f4f6;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #1f2937;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Feature Value Pills */
    .feature-pill {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.6rem 0.8rem;
        background-color: #1f2937;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.875rem;
    }
    
    .pill-label {
        color: #9ca3af;
        font-weight: 500;
    }
    
    .pill-value {
        color: #f3f4f6;
        font-weight: 600;
    }

    /* Verdict Badges */
    .verdict-box-survived {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .verdict-title-survived {
        color: #34d399;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    .verdict-box-perished {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    
    .verdict-title-perished {
        color: #f87171;
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
    }

    /* Streamlit Input Styling */
    div[data-baseweb="select"] > div {
        background-color: #1f2937 !important;
        border-color: #374151 !important;
        border-radius: 8px !important;
        color: #f9fafb !important;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #4338ca 0%, #2563eb 100%);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Fallback Model Builder (Guarantees Zero Error on Version Mismatch)
# ---------------------------------------------------------
def build_and_fit_pipeline():
    possible_data_paths = [
        "data/titanic.csv",
        "../../data/titanic.csv"
    ]
    data_file = None
    for d in possible_data_paths:
        if os.path.exists(d):
            data_file = d
            break
            
    if data_file is None:
        raise FileNotFoundError("Could not find 'data/titanic.csv' to fit pipeline.")
        
    df = pd.read_csv(data_file)
    X_raw = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
    y = df['Survived']
    
    num_features_eng = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'FarePerPerson']
    cat_features_eng = ['Sex', 'Embarked', 'Pclass', 'IsAlone']
    
    num_pipe_eng = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipe_eng = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])

    preprocessor_eng = ColumnTransformer([
        ('num', num_pipe_eng, num_features_eng),
        ('cat', cat_pipe_eng, cat_features_eng)
    ])

    pipe = Pipeline([
        ('feature_engineer', FunctionTransformer(engineer_titanic_features)),
        ('preprocessor', preprocessor_eng),
        ('classifier', LogisticRegression(C=1.5, max_iter=1000, random_state=42))
    ])
    
    pipe.fit(X_raw, y)
    return pipe

# ---------------------------------------------------------
# Cached Pipeline Loader with Version Fallback
# ---------------------------------------------------------
@st.cache_resource
def get_pipeline():
    possible_model_paths = [
        "models/titanic_pipeline.joblib",
        "tasks/Task-07-Scikit-Learn-Pipeline/titanic_pipeline.joblib",
        "../../models/titanic_pipeline.joblib"
    ]
    
    # 1. Try loading pre-trained joblib model
    for p in possible_model_paths:
        if os.path.exists(p):
            try:
                model = joblib.load(p)
                return model
            except Exception:
                pass
                
    # 2. Fallback: Fit fresh pipeline on-the-fly (takes < 0.1 seconds)
    return build_and_fit_pipeline()

try:
    pipeline = get_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Unable to load predictive engine: {e}")

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">Titanic Survival Decision Engine</div>
    <div class="app-subtitle">Production Machine Learning Inference System — Neurofive ML Track (Task 10)</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Preset Profiles Selector
# ---------------------------------------------------------
st.markdown("##### ⚡ Quick Preset Profiles")

default_pclass = 3
default_sex = "male"
default_age = 22.0
default_sibsp = 0
default_parch = 0
default_fare = 7.25
default_embarked = "S"

preset = st.radio(
    "Select a pre-configured profile or customize inputs below:",
    options=["Custom Input", "First-Class Luxury Female (Margaret Brown)", "Third-Class Single Male (Steerage Passenger)", "Second-Class Family Child (8 yo)"],
    horizontal=True,
    label_visibility="collapsed"
)

if preset == "First-Class Luxury Female (Margaret Brown)":
    default_pclass, default_sex, default_age, default_sibsp, default_parch, default_fare, default_embarked = 1, "female", 38.0, 0, 0, 110.0, "C"
elif preset == "Third-Class Single Male (Steerage Passenger)":
    default_pclass, default_sex, default_age, default_sibsp, default_parch, default_fare, default_embarked = 3, "male", 22.0, 0, 0, 7.25, "S"
elif preset == "Second-Class Family Child (8 yo)":
    default_pclass, default_sex, default_age, default_sibsp, default_parch, default_fare, default_embarked = 2, "female", 8.0, 1, 2, 26.25, "S"

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Interface Layout
# ---------------------------------------------------------
col_left, col_right = st.columns([1.1, 1], gap="large")

with col_left:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">Passenger Attributes & Ticket Parameters</div>', unsafe_allow_html=True)
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        pclass = st.selectbox(
            "Ticket Class (Pclass)",
            options=[1, 2, 3],
            index=[1, 2, 3].index(default_pclass),
            format_func=lambda x: {1: "1st Class (Upper)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower)"}[x]
        )
        
        sex = st.selectbox(
            "Passenger Gender",
            options=["female", "male"],
            index=["female", "male"].index(default_sex),
            format_func=lambda x: "Female" if x == "female" else "Male"
        )
        
        embarked = st.selectbox(
            "Port of Embarkation",
            options=["S", "C", "Q"],
            index=["S", "C", "Q"].index(default_embarked),
            format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x]
        )

    with col_in2:
        age = st.slider(
            "Age (Years)",
            min_value=0.42,
            max_value=80.0,
            value=float(default_age),
            step=0.5
        )
        
        fare = st.number_input(
            "Ticket Fare ($)",
            min_value=0.0,
            max_value=512.0,
            value=float(default_fare),
            step=1.0
        )
        
    st.markdown("<hr style='border-color: #1f2937; margin: 1rem 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.9rem; font-weight: 600; color: #9ca3af; margin-bottom: 0.5rem;'>Family Members Onboard</div>", unsafe_allow_html=True)
    
    col_fam1, col_fam2 = st.columns(2)
    with col_fam1:
        sibsp = st.number_input("Siblings / Spouses (SibSp)", min_value=0, max_value=8, value=int(default_sibsp))
    with col_fam2:
        parch = st.number_input("Parents / Children (Parch)", min_value=0, max_value=6, value=int(default_parch))

    st.markdown("</div>", unsafe_allow_html=True)
    
    run_inference = st.button("Run Survival Inference Pipeline", use_container_width=True)

with col_right:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="panel-header">Pipeline Feature Extraction & Diagnostics</div>', unsafe_allow_html=True)
    
    # Compute Real-Time Dynamic Feature Pipeline Values
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = fare / family_size if family_size > 0 else fare
    
    st.markdown(f"""
    <div class="feature-pill">
        <span class="pill-label">Total Family Size</span>
        <span class="pill-value">{family_size} person(s)</span>
    </div>
    <div class="feature-pill">
        <span class="pill-label">Traveling Solo</span>
        <span class="pill-value">{"Yes (Single)" if is_alone else "No (Group)"}</span>
    </div>
    <div class="feature-pill">
        <span class="pill-label">Adjusted Fare Per Person</span>
        <span class="pill-value">${fare_per_person:.2f}</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)

    # Perform Model Inference
    if model_loaded:
        input_df = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked
        }])
        
        try:
            pred_class = pipeline.predict(input_df)[0]
            pred_probs = pipeline.predict_proba(input_df)[0]
            surv_prob = pred_probs[1]
            perish_prob = pred_probs[0]
            
            st.markdown('<div class="panel-card">', unsafe_allow_html=True)
            st.markdown('<div class="panel-header">Model Prediction Outcome</div>', unsafe_allow_html=True)
            
            if pred_class == 1:
                st.markdown(f"""
                <div class="verdict-box-survived">
                    <div class="verdict-title-survived">Predicted: Survived</div>
                    <div style="color: #94a3b8; font-size: 0.95rem;">Model Probability: <b>{surv_prob*100:.1f}%</b> Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="verdict-box-perished">
                    <div class="verdict-title-perished">Predicted: Did Not Survive</div>
                    <div style="color: #94a3b8; font-size: 0.95rem;">Model Probability: <b>{perish_prob*100:.1f}%</b> Mortality Risk</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 0.85rem; color: #94a3b8; margin-bottom: 0.3rem;'>Survival Probability Distribution</div>", unsafe_allow_html=True)
            st.progress(float(surv_prob))
            
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Survival Likelihood", f"{surv_prob*100:.1f}%")
            with m_col2:
                st.metric("Mortality Likelihood", f"{perish_prob*100:.1f}%")

            st.markdown("</div>", unsafe_allow_html=True)
            
        except Exception as ex:
            st.error(f"Inference pipeline execution error: {ex}")

# Footer Documentation
st.markdown("<br><hr style='border-color: #1e2937;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding-bottom: 1.5rem;">
    Scikit-Learn Production Pipeline Encapsulation • Built by <b>Rao Hamza Irshad</b> • Neurofive ML Track Task 10
</div>
""", unsafe_allow_html=True)
