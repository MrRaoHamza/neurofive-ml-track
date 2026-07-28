# 🌲 Task 08: Ensemble Methods — Random Forest vs. XGBoost

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Task 08  
**Dataset:** Telco Customer Churn (`data/telco_customer_churn.csv`)  
**Notebook:** [`Task_08_Ensemble_Methods.ipynb`](Task_08_Ensemble_Methods.ipynb)  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Summary
Task 08 benchmarks ensemble learning architectures — **Random Forest (Bagging)** and **XGBoost (Gradient Boosting)** — against single models (Logistic Regression, Decision Trees) on the Telco Churn dataset.

---

## 📊 Model Benchmark Comparison Table

| Model Architecture | Category | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Class-Balanced)** | Single Linear | 73.88% | 50.52% | **78.07%** | 0.6134 | 0.8412 |
| **Decision Tree (`max_depth=5`)** | Single Tree | 79.42% | 62.96% | 54.55% | 0.5845 | 0.8284 |
| **Random Forest (Bagging)** | **Ensemble** | **81.05%** | **68.90%** | 52.14% | **0.5936** | 0.8432 |
| **XGBoost (Boosting)** | **Ensemble** | 80.06% | 65.25% | 53.21% | 0.5862 | **0.8453** |

![Ensemble Model Comparison](../../assets/ensemble_model_comparison.png)

---

## 🔍 Bagging vs. Boosting Theoretical Breakdown

- **Random Forest (Bagging):** Trains independent decision trees in parallel on bootstrap samples. Reduces overall variance by averaging uncorrelated trees. Feature importances are distributed smoothly across numerical variables (`tenure`, `TotalCharges`).
- **XGBoost (Gradient Boosting):** Trains decision trees sequentially in series, minimizing residual gradients of previous trees. Feature importances concentrate on decisive binary split nodes (`InternetService_Fiber optic`, `Contract_One year`).

![Feature Importance Comparison](../../assets/ensemble_feature_importances.png)

---

## 📂 Task Artifacts
- **Jupyter Notebook:** [`Task_08_Ensemble_Methods.ipynb`](Task_08_Ensemble_Methods.ipynb)
- **Visual Assets:** [`../../assets/ensemble_model_comparison.png`](../../assets/ensemble_model_comparison.png), [`../../assets/ensemble_feature_importances.png`](../../assets/ensemble_feature_importances.png)
