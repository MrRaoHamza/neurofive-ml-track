# ⚖️ Task 09: Handling Imbalanced Data & Evaluation Beyond Accuracy

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Task 09  
**Dataset:** Telco Customer Churn (`data/telco_customer_churn.csv` - 73.46% vs 26.54%)  
**Notebook:** [`Task_09_Handling_Imbalanced_Data.ipynb`](Task_09_Handling_Imbalanced_Data.ipynb)  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Summary
Task 09 addresses target class imbalance using **SMOTE** (`imbalanced-learn`), **Class Weighting**, and **Random Undersampling**. We demonstrate why raw Accuracy is deceptive on imbalanced data and evaluate models using Precision, Recall, F1-Score, and ROC-AUC.

---

## 📊 Before & After Rebalancing Benchmark Table

| Rebalancing Strategy | Accuracy | Precision (Churn) | Recall (Churn) | F1-Score | ROC-AUC | Recall Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unbalanced Baseline** | **79.63%** | **67.90%** | 44.12% | 0.5348 | 0.8429 | Baseline |
| **Class Weighting (`balanced`)** | 74.31% | 51.02% | **80.21%** | **0.6237** | 0.8415 | **+36.09%** |
| **SMOTE Oversampling** | 74.80% | 51.82% | **72.19%** | 0.6034 | 0.8322 | **+28.07%** |
| **Random Undersampling** | 74.17% | 50.85% | **79.68%** | 0.6208 | **0.8444** | **+35.56%** |

![Imbalance Comparison](../../assets/imbalance_techniques_comparison.png)

---

## 🧠 The Accuracy Paradox Explained

> Raw Accuracy measures total correct predictions across all samples. On an imbalanced dataset where 95% of samples are negative and 5% are positive, a naive classifier predicting "Negative" for every input scores 95% accuracy while catching 0% of minority events. In business-critical tasks (churn, fraud, medical diagnosis), **Recall** is far more critical than raw accuracy because a False Negative carries heavy business loss.

---

## 📂 Task Artifacts
- **Jupyter Notebook:** [`Task_09_Handling_Imbalanced_Data.ipynb`](Task_09_Handling_Imbalanced_Data.ipynb)
- **Visual Assets:** [`../../assets/class_imbalance_distribution.png`](../../assets/class_imbalance_distribution.png), [`../../assets/imbalance_techniques_comparison.png`](../../assets/imbalance_techniques_comparison.png), [`../../assets/smote_confusion_matrix.png`](../../assets/smote_confusion_matrix.png)
