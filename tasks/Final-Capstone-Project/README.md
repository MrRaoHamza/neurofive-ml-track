# 🩺 Final Capstone Project: Heart Disease Risk Prediction & Clinical Decision Support System

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Final Capstone  
**Dataset:** UCI Cleveland Heart Disease Dataset (`data/heart_disease.csv` - 303 patient records, 13 clinical predictors)  
**Notebook:** [`Final_Capstone_Heart_Disease_Prediction.ipynb`](Final_Capstone_Heart_Disease_Prediction.ipynb)  
**Live Web Application:** 🌐 [**neurofive-ml-track-titanic.streamlit.app**](https://neurofive-ml-track-titanic.streamlit.app)  
**Standalone Capstone App:** [`app_capstone.py`](../../app_capstone.py)  
**Model Artifact:** `heart_disease_pipeline.joblib`  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Overview & Problem Context
Cardiovascular Diseases (CVDs) remain the leading cause of global mortality (~17.9M deaths annually). Diagnostic sensitivity (**Recall**) is paramount — failing to identify a high-risk patient carries severe medical consequences. This Capstone builds an end-to-end **Clinical Decision Support System (CDSS)** to predict early cardiac risk and assist healthcare providers in preventative medical intervention.

---

## 📊 Multi-Model Benchmark Results

Evaluated 5 distinct machine learning model architectures on an 80/20 stratified holdout test split (61 patient test samples):

| Model Architecture | Model Category | Accuracy | Precision (Risk) | Recall (Sensitivity) | F1-Score | ROC-AUC Score | Performance Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Logistic Regression (Class-Balanced)** | Single Linear | 86.89% | 83.33% | 89.29% | 0.8621 | 0.9643 | Strong linear baseline |
| **Decision Tree Classifier (`max_depth=4`)** | Single Tree | 72.13% | 76.19% | 57.14% | 0.6531 | 0.7965 | Underfits complex non-linear boundaries |
| **Random Forest Classifier (Bagging)** | Ensemble | 90.16% | 86.67% | 92.86% | 0.8966 | 0.9610 | High precision & stability |
| **XGBoost Classifier (Gradient Boosting)** | Ensemble | 88.52% | 88.89% | 85.71% | 0.8727 | 0.9491 | Excellent gradient minimization |
| **Support Vector Classifier (SVC RBF)** | **Kernel Machine** | **93.44%** | **90.00%** | **96.43%** | **0.9310** | **0.9740** | 🏆 **Winning Capstone Model** |

![Capstone Benchmark](../../assets/capstone_model_benchmark_comparison.png)

![Capstone Confusion Matrix](../../assets/capstone_confusion_matrix.png)

---

## 📄 Half-Page Case Study Writeup

> ### 🩺 Business & Clinical Impact
> In emergency triage and cardiological outpatient clinics, physicians face time constraints when evaluating chest pain. A machine learning-powered CDSS provides rapid, evidence-based risk stratification in seconds, prioritizing high-risk patients for urgent angiograms and cardiological consults. Our winning **Support Vector Classifier** achieved **93.44% Accuracy**, **96.43% Recall**, and **0.9740 ROC-AUC**, ensuring 9 out of 10 cardiac risk patients are identified proactively without unnecessary invasive testing.

---

## 📂 Capstone Artifacts
- **Capstone Notebook:** [`Final_Capstone_Heart_Disease_Prediction.ipynb`](Final_Capstone_Heart_Disease_Prediction.ipynb)
- **Serialized Model Artifact:** [`heart_disease_pipeline.joblib`](heart_disease_pipeline.joblib)
- **Diagnostic Visualizations:** [`../../assets/capstone_*.png`](../../assets/)
