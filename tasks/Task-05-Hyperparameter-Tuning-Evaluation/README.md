# ⚙️ Task 05: Hyperparameter Tuning & Model Evaluation

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Task 05  
**Dataset:** Titanic Dataset (`data/titanic.csv`)  
**Notebook:** [`Task_05_Hyperparameter_Tuning.ipynb`](Task_05_Hyperparameter_Tuning.ipynb)  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Summary
Task 05 introduces rigorous hyperparameter optimization using **5-Fold `GridSearchCV`** on `LogisticRegression`. We evaluate model regularization strength ($C \in [0.01, 0.1, 1.0, 10.0, 100.0]$), solver types (`lbfgs`, `liblinear`), and penalty terms (`l1`, `l2`).

---

## 📊 Before vs. After Tuning Comparison Table

| Evaluation Metric | Baseline Model (Default Defaults) | Tuned Model (`C=1.5`, `liblinear`) | Performance Gain |
| :--- | :---: | :---: | :--- |
| **Accuracy** | 80.45% | **81.01%** | **+0.56%** |
| **Precision (Survivors)** | 79.66% | **80.36%** | **+0.70%** |
| **Recall (Survivors)** | 68.12% | **69.57%** | **+1.45%** |
| **F1-Score** | 0.7344 | **0.7458** | **+0.0114** |

![Tuned Confusion Matrix](../../assets/confusion_matrix_tuned.png)

---

## 📂 Task Artifacts
- **Jupyter Notebook:** [`Task_05_Hyperparameter_Tuning.ipynb`](Task_05_Hyperparameter_Tuning.ipynb)
- **Visual Plots:** [`../../assets/confusion_matrix_tuned.png`](../../assets/confusion_matrix_tuned.png)
