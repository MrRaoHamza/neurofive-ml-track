# 🚢 Neurofive Machine Learning Track

Welcome to the **Neurofive ML Track** repository maintained by **Rao Hamza Irshad**. This project demonstrates end-to-end Exploratory Data Analysis (EDA), missing value handling with statistical justifications, outlier detection, data storytelling, feature engineering, and your very first **Classification Machine Learning Model (Logistic Regression)** predicting Titanic passenger survival.

---

## 📌 Submission Matrix (Task-Wise Links)

| Track Task | Task Scope & Description | Direct Notebook Link |
| :--- | :--- | :--- |
| **Task 01** | **Baseline EDA & Setup:** Python environment setup, raw data ingestion, structural inspection (`.info()`, `.describe()`, `.head()`), feature classification (Numerical vs. Categorical), and executive Data Story. | 🔗 [**Task 1 Notebook**](tasks/Task-01-Baseline-EDA/Task_01_Titanic_EDA.ipynb) |
| **Task 02** | **Data Cleaning & Visual Storytelling:** Statistical missing value imputation (`fillna`), IQR outlier analysis, 4 Seaborn/Matplotlib visualizations, and formal survival driver analysis. | 🔗 [**Task 2 Notebook**](tasks/Task-02-Cleaning-and-Visualization/Task_02_Data_Cleaning_Visualizations.ipynb) |
| **Task 03** | **Machine Learning Classifier:** Categorical one-hot encoding, train-test split (`train_test_split`), **Logistic Regression** training, `accuracy_score` evaluation (**80.45%**), and Confusion Matrix interpretation. | 🔗 [**Task 3 Notebook**](tasks/Task-03-Logistic-Regression-Model/Task_03_Logistic_Regression.ipynb) |

---

## 📂 Professional Repository Architecture

```
neurofive-ml-track/
│
├── tasks/                                            # Modular Task Directory
│   ├── Task-01-Baseline-EDA/
│   │   └── Task_01_Titanic_EDA.ipynb                 # Task 1 Notebook
│   ├── Task-02-Cleaning-and-Visualization/
│   │   └── Task_02_Data_Cleaning_Visualizations.ipynb# Task 2 Notebook
│   └── Task-03-Logistic-Regression-Model/
│       └── Task_03_Logistic_Regression.ipynb         # Task 3 Notebook
│
├── data/                                             # Raw & Processed Datasets
│   └── titanic.csv                                   # Titanic Dataset (891 rows x 12 columns)
│
├── assets/                                           # High-Resolution Visualization Assets
│   ├── age_distribution_histogram.png                # Histogram: Age vs Survival
│   ├── fare_outliers_boxplot.png                     # Boxplot: Fare Outliers & Class
│   ├── survival_rate_barchart.png                    # Bar Chart: Gender & Class Survival
│   ├── correlation_heatmap.png                       # Heatmap: Feature Correlation Matrix
│   └── confusion_matrix.png                          # Confusion Matrix Heatmap (Task 3)
│
├── src/                                              # Reusable Code & Build Pipeline
│   └── generate_notebooks.py                         # Automation script for workspace builds
│
├── requirements.txt                                  # Project Dependencies (scikit-learn, etc.)
├── README.md                                         # Executive Repository Documentation
└── .gitignore                                        # Workspace Git Ignore Rules
```

---

## ⚙️ Environment Setup & Quickstart

### Prerequisites
- Python **3.8+** installed.

### 1. Clone & Navigate
```bash
git clone https://github.com/MrRaoHamza/neurofive-ml-track.git
cd neurofive-ml-track
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Notebooks
```bash
jupyter notebook tasks/Task-03-Logistic-Regression-Model/Task_03_Logistic_Regression.ipynb
```

---

## 📊 Task 01: Baseline EDA & Executive Data Story

### Dataset Metrics:
- **Dimensions:** 891 rows × 12 columns
- **Numerical Features (7):** `PassengerId`, `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`
- **Categorical Features (5):** `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`

### 📖 Executive Data Story:
> The Titanic dataset contains **891 rows** and **12 columns**, capturing passenger demographics, ticket details, and survival outcomes. Numerical features include `Age`, `Fare`, `SibSp`, `Parch`, `PassengerId`, `Pclass`, and `Survived`, while `Name`, `Sex`, `Ticket`, `Cabin`, and `Embarked` form the categorical attributes. Significant missing values exist in `Cabin` (77.10%) and `Age` (19.87%), with `Embarked` missing just 2 records. Key metrics reveal an overall survival rate of ~38.38% with passenger ages ranging from 0.42 to 80 years old (mean age ~29.7 years). The heavy missingness in `Cabin` suggests it may require indicator encoding, while `Age` will require median imputation prior to predictive modeling.

---

## 🧹 Task 02: Data Cleaning & Justifications (`fillna()` vs `dropna()`)

| Feature | Missing Count | Missing % | Cleaning Strategy | Formal Statistical Justification |
| :--- | :--- | :--- | :--- | :--- |
| **`Age`** | 177 | 19.87% | **`fillna(median)`** | `dropna()` would discard 20% of sample data. `Age` is right-skewed; median (~28.0 yrs) preserves sample size without being distorted by elderly outliers. |
| **`Embarked`** | 2 | 0.22% | **`fillna(mode)`** | Only 2 rows are missing. Imputing with the mode (`'S'`) restores complete cases with zero statistical bias. |
| **`Cabin`** | 687 | 77.10% | **`fillna('Unknown')` + `Cabin_Known`** | Over 77% missing. `dropna()` would destroy the dataset. We replace missing values with `'Unknown'` and construct a binary indicator (`Cabin_Known = 1/0`) to preserve the structural signal. |

---

## 🎨 Task 02: Visual Data Storytelling & Outlier Analysis

- **Outlier Cutoff (`Fare`):** **$65.63** ($Q3 + 1.5 \times IQR$). Identified 116 high-fare luxury suite outliers up to $512.33.
- **Key Survival Drivers:** `Sex` (Female survival **74.2%** vs Male survival **18.9%**) and `Pclass` (1st Class survival **62.9%** vs 3rd Class **24.2%**).

---

## 🤖 Task 03: Machine Learning Model (Logistic Regression)

### 🛠️ Modeling Pipeline Overview
1. **Categorical Encoding:** Converted categorical features (`Sex`, `Embarked`, `Pclass`) into numerical binary indicator columns using One-Hot Encoding (`pd.get_dummies(drop_first=True)`).
2. **Train-Test Split:** Split the dataset into **80% Training ($X_{train}, y_{train}$ - 712 samples)** and **20% Testing ($X_{test}, y_{test}$ - 179 samples)** using `train_test_split(test_size=0.2, random_state=42, stratify=y)` to preserve target class balance.
3. **Model Fitting:** Trained a `LogisticRegression(max_iter=1000)` classification model on the training set.

---

### 📈 Model Performance & Results

- **Overall Test Set Accuracy:** **80.45%** (`accuracy_score = 0.8045`)
- **Correct Predictions:** **144 out of 179** unseen test samples correctly classified.

### 🖼️ Confusion Matrix Visualization
![Confusion Matrix](assets/confusion_matrix.png)

---

### 📊 Confusion Matrix Analytical Breakdown & Explanation

The Confusion Matrix evaluates the performance of a binary classifier by comparing actual ground truth labels against model predictions:

| Actual \ Predicted | Predicted: Did Not Survive (0) | Predicted: Survived (1) | Total |
| :--- | :---: | :---: | :---: |
| **Actual: Did Not Survive (0)** | **True Negative ($TN$): 97** | **False Positive ($FP$): 13** | 110 |
| **Actual: Survived (1)** | **False Negative ($FN$): 22** | **True Positive ($TP$): 47** | 69 |
| **Total** | 119 | 60 | 179 |

#### **Written Analysis of Matrix Components:**
- **True Negatives ($TN = 97$):** The model correctly identified 97 passengers who actually **did not survive**.
- **True Positives ($TP = 47$):** The model correctly identified 47 passengers who actually **survived**.
- **False Positives ($FP = 13$, Type I Error):** The model incorrectly predicted that 13 non-surviving passengers survived.
- **False Negatives ($FN = 22$, Type II Error):** The model incorrectly predicted that 22 surviving passengers did not survive.

#### **Key Classification Metrics:**
- **Precision for Survivors (Class 1):** **78.33%** ($47 / (47 + 13)$) — When the model predicts a passenger survived, it is correct 78.33% of the time.
- **Recall for Survivors (Class 1):** **68.12%** ($47 / (47 + 22)$) — The model successfully captured 68.12% of all actual survivors in the test set.
- **Class 0 Specificity:** Excellent detection of non-survivors with **81.51% precision** and **88.18% recall**.

---

## 🎥 LinkedIn Presentation Guides

### Task 1 Walkthrough (2-3 mins):
- Show dataset loading and `.info()`, `.describe()`, and `.head()` outputs.
- Present your executive 5-6 line Data Story.

### Task 2 Walkthrough (2-3 mins):
- Explain your statistical data cleaning choices (`Age` median imputation, `Cabin_Known` binary feature).
- Walk through the **Survival Rate by Gender & Class Bar Chart**.

### Task 3 Walkthrough (2-3 mins):
- Explain how you encoded categorical features using One-Hot Encoding (`pd.get_dummies`) and performed train-test split (`train_test_split`).
- Present your **Logistic Regression Model Accuracy (80.45%)**.
- Walk through the **Confusion Matrix heatmap**, explaining True Negatives (97), True Positives (47), False Positives (13), and False Negatives (22).
- Post on LinkedIn tagging **@Neurofive Solutions**!
