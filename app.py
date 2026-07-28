import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin

# Page Configuration
st.set_page_config(
    page_title="Titanic Survival Predictor | Neurofive ML Track",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Glassmorphic Theme)
st.markdown("""
<style>
    .main {
        background-color: #0f172a;
        color: #f8fafc;
    }
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        backdrop-filter: blur(10px);
        margin-bottom: 15px;
    }
    .survived-badge {
        background-color: #10b981;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
    .perished-badge {
        background-color: #ef4444;
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# Load Model Pipeline with Cache
@st.cache_resource
def load_pipeline():
    model_path = "models/titanic_pipeline.joblib"
    if not os.path.exists(model_path):
        model_path = "tasks/Task-07-Scikit-Learn-Pipeline/titanic_pipeline.joblib"
    return joblib.load(model_path)

try:
    pipeline = load_pipeline()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model pipeline: {e}")

# Header Banner
st.title("🚢 Titanic Survival Prediction Web App")
st.caption("Neurofive Machine Learning Track — Task 10: Production Model Deployment with Streamlit")
st.markdown("---")

# Main Layout
col_inputs, col_results = st.columns([1.1, 1], gap="large")

with col_inputs:
    st.subheader("📋 Passenger Demographic & Ticket Details")
    
    with st.container():
        pclass = st.selectbox(
            "Passenger Ticket Class (Pclass)",
            options=[1, 2, 3],
            format_func=lambda x: {1: "1st Class (Upper / Executive)", 2: "2nd Class (Middle)", 3: "3rd Class (Lower / Economy)"}[x],
            index=2,
            help="Ticket class proxy for socio-economic status"
        )
        
        sex = st.selectbox(
            "Gender (Sex)",
            options=["female", "male"],
            format_func=lambda x: "Female 👩" if x == "female" else "Male 👨",
            index=0
        )
        
        age = st.slider(
            "Age (Years)",
            min_value=0.5,
            max_value=80.0,
            value=28.0,
            step=0.5
        )
        
        c_fam1, c_fam2 = st.columns(2)
        with c_fam1:
            sibsp = st.number_input("Siblings / Spouses Onboard (SibSp)", min_value=0, max_value=8, value=0)
        with c_fam2:
            parch = st.number_input("Parents / Children Onboard (Parch)", min_value=0, max_value=6, value=0)
            
        c_fare1, c_fare2 = st.columns(2)
        with c_fare1:
            fare = st.number_input("Ticket Fare ($)", min_value=0.0, max_value=512.0, value=32.2, step=1.0)
        with c_fare2:
            embarked = st.selectbox(
                "Port of Embarkation",
                options=["S", "C", "Q"],
                format_func=lambda x: {"S": "Southampton (S)", "C": "Cherbourg (C)", "Q": "Queenstown (Q)"}[x]
            )

    predict_btn = st.button("🔮 Predict Survival Outcome", use_container_width=True, type="primary")

with col_results:
    st.subheader("📊 Survival Analysis & Inference Results")
    
    # Calculate Live Dynamic Features
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = fare / family_size if family_size > 0 else fare
    
    st.markdown(f"""
    <div class="metric-card">
        <h4>💡 Engineered Feature Calculation (Live Pipeline Preview)</h4>
        <ul>
            <li><b>Total Family Size:</b> {family_size} person(s)</li>
            <li><b>Traveling Alone:</b> {"Yes 🧍" if is_alone else "No 👥"}</li>
            <li><b>Adjusted Fare Per Person:</b> ${fare_per_person:.2f}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if predict_btn and model_loaded:
        # Create raw input DataFrame matching training feature schema
        raw_input = pd.DataFrame([{
            'Pclass': pclass,
            'Sex': sex,
            'Age': age,
            'SibSp': sibsp,
            'Parch': parch,
            'Fare': fare,
            'Embarked': embarked
        }])
        
        with st.spinner("Running End-to-End Scikit-Learn Pipeline..."):
            pred_class = pipeline.predict(raw_input)[0]
            pred_probs = pipeline.predict_proba(raw_input)[0]
            surv_prob = pred_probs[1] * 100
            perish_prob = pred_probs[0] * 100
            
        st.markdown("### Model Prediction Outcome:")
        if pred_class == 1:
            st.markdown('<div class="survived-badge">🟢 SURVIVED PASSENGER</div>', unsafe_allow_html=True)
            st.success(f"The model estimates a **{surv_prob:.2f}% probability of survival** for this passenger profile.")
        else:
            st.markdown('<div class="perished-badge">🔴 DID NOT SURVIVE</div>', unsafe_allow_html=True)
            st.error(f"The model estimates a **{perish_prob:.2f}% probability of mortality** ({surv_prob:.2f}% survival probability).")
            
        st.markdown("#### Probability Distribution Gauge:")
        st.progress(int(surv_prob))
        
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.metric("Survival Probability", f"{surv_prob:.2f}%")
        with col_m2:
            st.metric("Perish Probability", f"{perish_prob:.2f}%")
            
    elif not predict_btn:
        st.info("👈 Adjust passenger parameters on the left panel and click **'Predict Survival Outcome'** to generate live inference.")

# Footer
st.markdown("---")
st.caption("Developed by **Rao Hamza Irshad** | Neurofive Machine Learning Track | Powered by Scikit-Learn Pipelines & Streamlit")
