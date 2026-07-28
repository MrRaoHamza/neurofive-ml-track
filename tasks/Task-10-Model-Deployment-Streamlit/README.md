# 🚀 Task 10: Production Model Deployment with Streamlit

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Task 10  
**Application Code:** [`app.py`](../../app.py)  
**Notebook:** [`Task_10_Model_Deployment_Streamlit.ipynb`](Task_10_Model_Deployment_Streamlit.ipynb)  
**Live Web App:** 🌐 [**neurofive-ml-track-titanic.streamlit.app**](https://neurofive-ml-track-titanic.streamlit.app)  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Summary
Task 10 turns our fitted **Scikit-Learn Production Pipeline** (`models/titanic_pipeline.joblib`) into an interactive, dark-mode, shareable **Streamlit Web Application** (`app.py`), bridging the gap between model development and production product deployment.

---

## 🛠️ Web App Features & Architecture

- **Single-Line Inference:** Raw user inputs (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`) pass directly into `pipeline.predict(df)` and `pipeline.predict_proba(df)`.
- **Live Engineered Feature Preview:** Dynamically calculates `FamilySize`, `IsAlone`, and `FarePerPerson`.
- **UI Feedback:** Dark glassmorphic theme, status badges (🟢 **SURVIVED** / 🔴 **DID NOT SURVIVE**), and confidence gauges.

---

## ⚙️ Quickstart Command

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📂 Task Artifacts
- **Streamlit Web App File:** [`../../app.py`](../../app.py)
- **Jupyter Notebook:** [`Task_10_Model_Deployment_Streamlit.ipynb`](Task_10_Model_Deployment_Streamlit.ipynb)
- **Model Pipeline Artifact:** [`../../models/titanic_pipeline.joblib`](../../models/titanic_pipeline.joblib)
