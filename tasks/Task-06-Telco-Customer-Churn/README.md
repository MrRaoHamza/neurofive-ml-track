# 📉 Task 06: Telco Customer Churn Prediction — Decision Trees

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Task 06  
**Dataset:** Telco Customer Churn (`data/telco_customer_churn.csv` - 7,043 records)  
**Notebook:** [`Task_06_Telco_Customer_Churn.ipynb`](Task_06_Telco_Customer_Churn.ipynb)  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Summary
Task 06 analyzes real-world customer attrition on IBM's Telco Churn dataset using a **Decision Tree Classifier (`max_depth=5`)** vs **Logistic Regression**. We identify the top 3 drivers of customer loss and formulate a 5-sentence executive business pitch.

---

## 📊 Decision Tree vs. Logistic Regression Comparison

| Evaluation Metric | Decision Tree (`max_depth=5`) | Logistic Regression (`class_weight='balanced'`) | Winning Model |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **79.42%** | 73.88% | **Decision Tree (+5.54%)** |
| **Precision (Churners)** | **62.96%** | 50.52% | **Decision Tree (+12.44%)** |
| **Recall (Churners)** | 54.55% | **78.07%** | **Logistic Regression (+23.52%)** |
| **ROC-AUC Score** | 0.8284 | **0.8412** | **Logistic Regression (+0.0128)** |

---

## 🔍 Top 3 Feature Importances (`.feature_importances_`)

1. **`tenure` (42.14% Weight):** Customer tenure in months is the single strongest predictor of loyalty.
2. **`InternetService_Fiber optic` (35.75% Weight):** Fiber Optic subscribers churn at elevated rates due to high monthly billing.
3. **`TotalCharges` (4.71% Weight):** Accumulated spending threshold.

![Telco Churn Feature Importances](../../assets/churn_feature_importances.png)

---

## 💼 Executive Business Summary

> Our predictive analysis of **7,043 telecom customers** reveals an annual churn rate of **26.54%**, representing substantial recurring revenue loss. Using a Decision Tree classification model (79.42% accuracy), we identified the **top 3 primary drivers of customer departure** as: **(1) Customer Tenure** (42.1% of predictive weight), **(2) Fiber Optic Internet Service** (35.8% of weight), and **(3) Accumulated Total Charges**. Crucially, customers on month-to-month contracts churn at an alarming **42.7% rate**, compared to just **2.8%** for two-year contract holders. To immediately reduce churn, executive leadership should implement targeted multi-year contract upgrade discounts and bundled technical support incentives during a customer's first 12 months.

---

## 📂 Task Artifacts
- **Jupyter Notebook:** [`Task_06_Telco_Customer_Churn.ipynb`](Task_06_Telco_Customer_Churn.ipynb)
- **Visual Assets:** [`../../assets/telco_churn_eda.png`](../../assets/telco_churn_eda.png), [`../../assets/churn_feature_importances.png`](../../assets/churn_feature_importances.png)
