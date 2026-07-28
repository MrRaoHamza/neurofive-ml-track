# ⚙️ Task 07: Scikit-Learn Production Pipelines & Leak-Free ML

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Task 07  
**Dataset:** Titanic Dataset (`data/titanic.csv`)  
**Notebook:** [`Task_07_Scikit_Learn_Pipeline.ipynb`](Task_07_Scikit_Learn_Pipeline.ipynb)  
**Model Artifact:** `titanic_pipeline.joblib`  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Summary
Task 07 constructs a clean, end-to-end **Scikit-Learn Production Pipeline** using `ColumnTransformer` and custom feature engineering (`TitanicFeatureEngineer`). Chaining imputation, scaling, encoding, and model inference into a single object eliminates **data leakage** between train/test folds and enables single-line inference.

---

## 🛠️ Pipeline Architecture & Feature Engineering

1. **Custom Transformer (`TitanicFeatureEngineer`):**
   - `FamilySize` = `SibSp` + `Parch` + 1
   - `IsAlone` = 1 if `FamilySize == 1` else 0
   - `FarePerPerson` = `Fare` / `FamilySize`
2. **ColumnTransformer Preprocessing:**
   - Numerical (`SimpleImputer(median)` + `StandardScaler`)
   - Categorical (`SimpleImputer(most_frequent)` + `OneHotEncoder(drop='first')`)
3. **Classifier:** `LogisticRegression(C=1.5, max_iter=1000)`

![Pipeline Architecture](../../assets/pipeline_architecture_diagram.png)

---

## 📊 Benchmark Results: Manual Baseline vs Production Pipeline

| Evaluation Metric | Manual Baseline | Production Pipeline | Improvement |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 79.89% | **81.56%** | **+1.67%** |
| **Precision (Survivors)** | 77.97% | **81.03%** | **+3.06%** |
| **Recall (Survivors)** | 66.67% | **68.12%** | **+1.45%** |
| **ROC-AUC Score** | 0.8436 | **0.8509** | **+0.0073** |

![Pipeline Performance Comparison](../../assets/pipeline_performance_comparison.png)

---

## 📂 Task Artifacts
- **Jupyter Notebook:** [`Task_07_Scikit_Learn_Pipeline.ipynb`](Task_07_Scikit_Learn_Pipeline.ipynb)
- **Serialized Model:** [`titanic_pipeline.joblib`](titanic_pipeline.joblib)
