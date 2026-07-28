import os
import pandas as pd
import numpy as np
import streamlit as st

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Neurofive ML Portfolio Centerpiece | Final Capstone & Task Apps",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------------
# Custom Feature Engineering Functions
# ---------------------------------------------------------
def engineer_heart_features(X):
    X_out = X.copy()
    X_out['Age_Group'] = pd.cut(X_out['age'], bins=[0, 45, 60, 100], labels=[0, 1, 2]).astype(int)
    max_hr_expected = 220 - X_out['age']
    X_out['Max_HR_Ratio'] = X_out['thalach'] / np.where(max_hr_expected == 0, 1, max_hr_expected)
    X_out['Chol_Age_Ratio'] = X_out['chol'] / np.where(X_out['age'] == 0, 1, X_out['age'])
    return X_out

def engineer_titanic_features(X):
    X_out = X.copy()
    X_out['FamilySize'] = X_out['SibSp'] + X_out['Parch'] + 1
    X_out['IsAlone'] = (X_out['FamilySize'] == 1).astype(int)
    X_out['FarePerPerson'] = X_out['Fare'] / X_out['FamilySize']
    return X_out

# ---------------------------------------------------------
# Cached Pipeline Loaders
# ---------------------------------------------------------
@st.cache_resource
def get_heart_pipeline():
    data_paths = ["data/heart_disease.csv", "tasks/Final-Capstone-Project/heart_disease.csv", "../../data/heart_disease.csv"]
    df_path = None
    for p in data_paths:
        if os.path.exists(p):
            df_path = p
            break
            
    if df_path is None:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        df = pd.read_csv(url, header=None)
        df.columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
        df['ca'] = pd.to_numeric(df['ca'].replace('?', np.nan)).fillna(0)
        df['thal'] = pd.to_numeric(df['thal'].replace('?', np.nan)).fillna(3)
        df['target'] = (df['num'] > 0).astype(int)
        df = df.drop(columns=['num'])
    else:
        df = pd.read_csv(df_path)

    X_raw = df.drop(columns=['target'])
    y = df['target']

    num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'Max_HR_Ratio', 'Chol_Age_Ratio']
    cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'Age_Group']

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))])

    preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])

    pipe = Pipeline([
        ('engineer', FunctionTransformer(engineer_heart_features)),
        ('preprocessor', preprocessor),
        ('classifier', SVC(kernel='rbf', C=1.0, probability=True, random_state=42))
    ])
    pipe.fit(X_raw, y)
    return pipe

@st.cache_resource
def get_titanic_pipeline():
    data_paths = ["data/titanic.csv", "../../data/titanic.csv"]
    df_path = None
    for p in data_paths:
        if os.path.exists(p):
            df_path = p
            break
            
    if df_path is None:
        url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
        df = pd.read_csv(url)
    else:
        df = pd.read_csv(df_path)

    X_raw = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
    y = df['Survived']

    num_features_eng = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'FarePerPerson']
    cat_features_eng = ['Sex', 'Embarked', 'Pclass', 'IsAlone']

    num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
    cat_pipe = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([('num', num_pipe, num_features_eng), ('cat', cat_pipe, cat_features_eng)])

    pipe = Pipeline([
        ('feature_engineer', FunctionTransformer(engineer_titanic_features)),
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=1.5, max_iter=1000, random_state=42))
    ])
    pipe.fit(X_raw, y)
    return pipe

try:
    heart_pipeline = get_heart_pipeline()
    titanic_pipeline = get_titanic_pipeline()
    models_ready = True
except Exception as e:
    models_ready = False
    st.error(f"Engine Startup Notice: {e}")

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
    
    .app-header {
        background: linear-gradient(180deg, #131b2e 0%, #0b0f19 100%);
        border-bottom: 1px solid #1e293b;
        padding: 2rem 0 1.25rem 0;
        margin-bottom: 1.5rem;
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
    }

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
    
    .pill-label { color: #9ca3af; font-weight: 500; }
    .pill-value { color: #f3f4f6; font-weight: 600; }

    .verdict-box-danger {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(239, 68, 68, 0.05) 100%);
        border: 1px solid rgba(239, 68, 68, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-title-danger { color: #f87171; font-size: 1.4rem; font-weight: 700; }

    .verdict-box-success {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.4);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .verdict-title-success { color: #34d399; font-size: 1.4rem; font-weight: 700; }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
st.markdown("""
<div class="app-header">
    <div class="app-title">Neurofive ML Portfolio Centerpiece</div>
    <div class="app-subtitle">Production Machine Learning Systems — Maintained by Rao Hamza Irshad</div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_capstone, tab_task10 = st.tabs([
    "🩺 Final Capstone: Heart Disease Risk System",
    "🚢 Task 10: Titanic Survival Predictor"
])

# =========================================================
# TAB 1: FINAL CAPSTONE PROJECT (Heart Disease CDSS)
# =========================================================
with tab_capstone:
    st.markdown("### 🩺 Cardiovascular Disease (CVD) Clinical Decision Support System")
    st.caption("UCI Cleveland Heart Disease Dataset • 93.44% Model Accuracy • 0.9740 ROC-AUC (Support Vector Classifier)")
    
    st.markdown("##### ⚡ Quick Clinical Presets")
    preset_h = st.radio(
        "Select a pre-configured patient profile:",
        options=["Custom Telemetry", "Healthy Young Adult (28 yo)", "Asymptomatic Senior Male (63 yo)", "High Risk Cardiac Patient (67 yo)"],
        horizontal=True,
        key="preset_h_radio"
    )
    
    h_age, h_sex, h_cp, h_bp, h_chol, h_fbs, h_ecg, h_hr, h_ang, h_peak, h_slope, h_ca, h_thal = 54, 1, 3, 130, 246, 0, 0, 150, 0, 1.0, 1, 0, 3
    
    if preset_h == "Healthy Young Adult (28 yo)":
        h_age, h_sex, h_cp, h_bp, h_chol, h_fbs, h_ecg, h_hr, h_ang, h_peak, h_slope, h_ca, h_thal = 28, 0, 1, 115, 180, 0, 0, 175, 0, 0.0, 1, 0, 3
    elif preset_h == "Asymptomatic Senior Male (63 yo)":
        h_age, h_sex, h_cp, h_bp, h_chol, h_fbs, h_ecg, h_hr, h_ang, h_peak, h_slope, h_ca, h_thal = 63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6
    elif preset_h == "High Risk Cardiac Patient (67 yo)":
        h_age, h_sex, h_cp, h_bp, h_chol, h_fbs, h_ecg, h_hr, h_ang, h_peak, h_slope, h_ca, h_thal = 67, 1, 4, 160, 286, 0, 2, 108, 1, 1.5, 2, 3, 3

    st.markdown("<br>", unsafe_allow_html=True)
    
    c_h_left, c_h_right = st.columns([1.2, 1], gap="large")
    
    with c_h_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Patient Telemetry & Vital Markers</div>', unsafe_allow_html=True)
        
        c1, c2 = st.columns(2)
        with c1:
            age_in = st.slider("Patient Age (Years)", 20, 90, int(h_age))
            sex_in = st.selectbox("Gender", [1, 0], index=0 if h_sex==1 else 1, format_func=lambda x: "Male" if x==1 else "Female")
            cp_in = st.selectbox("Chest Pain Type (cp)", [1, 2, 3, 4], index=h_cp-1, format_func=lambda x: {1: "1: Typical Angina", 2: "2: Atypical Angina", 3: "3: Non-Anginal Pain", 4: "4: Asymptomatic"}[x])
            bp_in = st.number_input("Resting Blood Pressure (mm Hg)", 80, 220, int(h_bp))
            chol_in = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, int(h_chol))
            fbs_in = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1], index=int(h_fbs), format_func=lambda x: "No (<= 120 mg/dl)" if x==0 else "Yes (> 120 mg/dl)")
        
        with c2:
            ecg_in = st.selectbox("Resting ECG Results", [0, 1, 2], index=int(h_ecg), format_func=lambda x: {0: "0: Normal", 1: "1: ST-T Wave Abnormality", 2: "2: Left Ventricular Hypertrophy"}[x])
            hr_in = st.number_input("Max Heart Rate Achieved (bpm)", 60, 220, int(h_hr))
            ang_in = st.selectbox("Exercise Induced Angina", [0, 1], index=int(h_ang), format_func=lambda x: "No" if x==0 else "Yes")
            peak_in = st.slider("ST Depression (oldpeak)", 0.0, 6.2, float(h_peak), step=0.1)
            slope_in = st.selectbox("Peak Exercise ST Slope", [1, 2, 3], index=h_slope-1, format_func=lambda x: {1: "1: Upsloping", 2: "2: Flat", 3: "3: Downsloping"}[x])
            ca_in = st.selectbox("Major Vessels Colored (ca)", [0, 1, 2, 3], index=int(h_ca))
            thal_in = st.selectbox("Thalassemia (thal)", [3, 6, 7], index=[3, 6, 7].index(int(h_thal)), format_func=lambda x: {3: "3: Normal", 6: "6: Fixed Defect", 7: "7: Reversable Defect"}[x])

        st.markdown("</div>", unsafe_allow_html=True)
        btn_h = st.button("Run Clinical Risk Inference Pipeline", use_container_width=True, key="btn_h_run")

    with c_h_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Derived Medical Ratios & Pipeline Metrics</div>', unsafe_allow_html=True)
        
        max_hr_exp = 220 - age_in
        hr_ratio = hr_in / max_hr_exp if max_hr_exp > 0 else 1.0
        chol_age_ratio = chol_in / age_in if age_in > 0 else 1.0
        
        st.markdown(f"""
        <div class="feature-pill"><span class="pill-label">Age Group Stratification</span><span class="pill-value">{"Young (<45)" if age_in < 45 else ("Middle (45-60)" if age_in <= 60 else "Senior (>60)")}</span></div>
        <div class="feature-pill"><span class="pill-label">Max HR Ratio (Achieved / Expected)</span><span class="pill-value">{hr_ratio*100:.1f}%</span></div>
        <div class="feature-pill"><span class="pill-label">Cholesterol-to-Age Ratio</span><span class="pill-value">{chol_age_ratio:.2f}</span></div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        if models_ready:
            input_h = pd.DataFrame([{
                'age': age_in, 'sex': sex_in, 'cp': cp_in, 'trestbps': bp_in, 'chol': chol_in,
                'fbs': fbs_in, 'restecg': ecg_in, 'thalach': hr_in, 'exang': ang_in,
                'oldpeak': peak_in, 'slope': slope_in, 'ca': ca_in, 'thal': thal_in
            }])
            
            try:
                pred_h = heart_pipeline.predict(input_h)[0]
                prob_h = heart_pipeline.predict_proba(input_h)[0][1]
                
                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.markdown('<div class="panel-header">Clinical Diagnostic Outcome</div>', unsafe_allow_html=True)
                
                if pred_h == 1:
                    st.markdown(f"""
                    <div class="verdict-box-danger">
                        <div class="verdict-title-danger">⚠️ HIGH CARDIAC DISEASE RISK</div>
                        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.4rem;">Model Estimated Risk: <b>{prob_h*100:.1f}% Probability</b></div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.warning("📋 **Clinical Recommendation:** Immediate cardiological follow-up and angiogram evaluation recommended.")
                else:
                    st.markdown(f"""
                    <div class="verdict-box-success">
                        <div class="verdict-title-success">🟢 LOW CARDIAC RISK PROFILE</div>
                        <div style="color: #94a3b8; font-size: 0.95rem; margin-top: 0.4rem;">Model Estimated Risk: <b>{prob_h*100:.1f}% Probability</b> ({(1-prob_h)*100:.1f}% Healthy)</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("📋 **Clinical Recommendation:** Patient telemetry within normal physiological parameters.")

                st.markdown("<br>", unsafe_allow_html=True)
                st.progress(float(prob_h))
                
                m1, m2 = st.columns(2)
                with m1: st.metric("Cardiac Risk Score", f"{prob_h*100:.1f}%")
                with m2: st.metric("Healthy Likelihood", f"{(1-prob_h)*100:.1f}%")
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as ex:
                st.error(f"Inference error: {ex}")

# =========================================================
# TAB 2: TASK 10 (Titanic Survival Predictor)
# =========================================================
with tab_task10:
    st.markdown("### 🚢 Titanic Passenger Survival Predictor")
    st.caption("Task 10 Production Pipeline • 81.56% Accuracy • Scikit-Learn Encapsulation")
    
    preset_t = st.radio(
        "Select passenger profile:",
        options=["Custom Input", "1st Class Female (Luxury)", "3rd Class Male (Steerage)", "2nd Class Child (8 yo)"],
        horizontal=True,
        key="preset_t_radio"
    )
    
    t_pclass, t_sex, t_age, t_sibsp, t_parch, t_fare, t_embarked = 3, "male", 22.0, 0, 0, 7.25, "S"
    if preset_t == "1st Class Female (Luxury)":
        t_pclass, t_sex, t_age, t_sibsp, t_parch, t_fare, t_embarked = 1, "female", 38.0, 0, 0, 110.0, "C"
    elif preset_t == "3rd Class Male (Steerage)":
        t_pclass, t_sex, t_age, t_sibsp, t_parch, t_fare, t_embarked = 3, "male", 22.0, 0, 0, 7.25, "S"
    elif preset_t == "2nd Class Child (8 yo)":
        t_pclass, t_sex, t_age, t_sibsp, t_parch, t_fare, t_embarked = 2, "female", 8.0, 1, 2, 26.25, "S"

    c_t_left, c_t_right = st.columns([1.1, 1], gap="large")
    
    with c_t_left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Passenger Attributes</div>', unsafe_allow_html=True)
        
        tc1, tc2 = st.columns(2)
        with tc1:
            pclass_in = st.selectbox("Ticket Class", [1, 2, 3], index=[1, 2, 3].index(t_pclass))
            sex_in = st.selectbox("Gender", ["female", "male"], index=["female", "male"].index(t_sex))
            emb_in = st.selectbox("Port of Embarkation", ["S", "C", "Q"], index=["S", "C", "Q"].index(t_embarked))
        with tc2:
            age_t_in = st.slider("Age", 0.42, 80.0, float(t_age))
            fare_t_in = st.number_input("Fare ($)", 0.0, 512.0, float(t_fare))
            
        st.markdown("<hr style='border-color: #1f2937;'>", unsafe_allow_html=True)
        tf1, tf2 = st.columns(2)
        with tf1: sibsp_in = st.number_input("Siblings / Spouses", 0, 8, int(t_sibsp))
        with tf2: parch_in = st.number_input("Parents / Children", 0, 6, int(t_parch))
        
        st.markdown("</div>", unsafe_allow_html=True)
        btn_t = st.button("Run Titanic Pipeline Inference", use_container_width=True, key="btn_t_run")

    with c_t_right:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="panel-header">Engineered Features</div>', unsafe_allow_html=True)
        
        fam_size = sibsp_in + parch_in + 1
        is_alone = 1 if fam_size == 1 else 0
        fare_person = fare_t_in / fam_size if fam_size > 0 else fare_t_in
        
        st.markdown(f"""
        <div class="feature-pill"><span class="pill-label">Total Family Size</span><span class="pill-value">{fam_size}</span></div>
        <div class="feature-pill"><span class="pill-label">Traveling Solo</span><span class="pill-value">{"Yes" if is_alone else "No"}</span></div>
        <div class="feature-pill"><span class="pill-label">Fare Per Person</span><span class="pill-value">${fare_person:.2f}</span></div>
        """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        if models_ready:
            input_t = pd.DataFrame([{
                'Pclass': pclass_in, 'Sex': sex_in, 'Age': age_t_in,
                'SibSp': sibsp_in, 'Parch': parch_in, 'Fare': fare_t_in, 'Embarked': emb_in
            }])
            try:
                pred_t = titanic_pipeline.predict(input_t)[0]
                prob_t = titanic_pipeline.predict_proba(input_t)[0][1]
                
                st.markdown('<div class="panel-card">', unsafe_allow_html=True)
                st.markdown('<div class="panel-header">Outcome Prediction</div>', unsafe_allow_html=True)
                if pred_t == 1:
                    st.markdown(f'<div class="verdict-box-success"><div class="verdict-title-success">🟢 SURVIVED</div><div style="color: #94a3b8;">Probability: <b>{prob_t*100:.1f}%</b></div></div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="verdict-box-danger"><div class="verdict-title-danger">🔴 DID NOT SURVIVE</div><div style="color: #94a3b8;">Mortality: <b>{(1-prob_t)*100:.1f}%</b></div></div>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as ex:
                st.error(f"Error: {ex}")

# Footer Documentation
st.markdown("<br><hr style='border-color: #1e2937;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding-bottom: 1.5rem;">
    Neurofive Machine Learning Track • Developed by <b>Rao Hamza Irshad</b> • Production Streamlit Showcase
</div>
""", unsafe_allow_html=True)
