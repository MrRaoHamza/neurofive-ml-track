# Neurofive ML Track - Titanic EDA & Data Cleaning

Welcome to the **Neurofive ML Track** repository by **Rao Hamza Irshad**! This repository contains task-wise notebooks and code for Task 1 and Task 2.

---

## 🔗 Task Submission Links for Evaluators

| Task | Description | Direct Notebook Link |
| :--- | :--- | :--- |
| **Task 1** | Python Environment Setup, Dataset Ingestion, Inspection (`.info()`, `.describe()`, `.head()`), Feature Classification & Data Story | 🔗 [**Task 1 Notebook**](https://github.com/MrRaoHamza/neurofive-ml-track/blob/main/Task1_Titanic_EDA.ipynb) |
| **Task 2** | Missing Value Handling & Justifications (`fillna`), Outlier Detection (Boxplot/IQR), 4 Visualizations, Feature Survival Analysis | 🔗 [**Task 2 Notebook**](https://github.com/MrRaoHamza/neurofive-ml-track/blob/main/Task2_Data_Cleaning_Visualizations.ipynb) |
| **Combined** | Full Master Notebook containing both Task 1 and Task 2 | 🔗 [**Combined Notebook**](https://github.com/MrRaoHamza/neurofive-ml-track/blob/main/eda_titanic.ipynb) |

---

## 📁 Repository Structure
```
neurofive-ml-track/
├── Task1_Titanic_EDA.ipynb                   # Dedicated Notebook for Task 1 Submission
├── Task2_Data_Cleaning_Visualizations.ipynb  # Dedicated Notebook for Task 2 Submission
├── eda_titanic.ipynb                         # Combined Master Notebook (Tasks 1 & 2)
├── titanic.csv                               # Kaggle Titanic Dataset (891 rows x 12 columns)
├── build_eda.py                              # Python script generating master notebook
├── create_task_notebooks.py                  # Script generating dedicated task notebooks
├── visualizations/                           # Generated high-resolution plots (.png)
│   ├── plot1_age_histogram.png
│   ├── plot2_fare_boxplot.png
│   ├── plot3_survival_barchart.png
│   └── plot4_correlation_heatmap.png
├── README.md                                 # Complete project documentation
└── .gitignore                                # Git ignore rules
```

---

## 🧹 Data Cleaning Strategy & Justifications (`fillna()` vs `dropna()`)

| Feature | Missing Count | Missing % | Cleaning Strategy | Justification |
| :--- | :--- | :--- | :--- | :--- |
| **`Age`** | 177 | 19.87% | **`fillna(median)`** | `dropna()` would discard 20% of sample data. `Age` is right-skewed; median (~28.0 yrs) preserves sample size without being pulled by elderly outliers. |
| **`Embarked`** | 2 | 0.22% | **`fillna(mode)`** | Only 2 rows are missing. Imputing with the mode (`'S'`) restores complete cases with zero statistical bias. |
| **`Cabin`** | 687 | 77.10% | **`fillna('Unknown')` + `Cabin_Known`** | Over 77% missing. `dropna()` would destroy the dataset. We replace missing values with `'Unknown'` and construct a binary indicator (`Cabin_Known = 1/0`) to capture the structural signal of recording cabin data. |

---

## 🔍 Outlier Detection (Interquartile Range - IQR Analysis)

Using `sns.boxplot` on `Fare` across `Pclass`, we identified significant extreme value outliers:
- **First Quartile (Q1):** $7.91 | **Third Quartile (Q3):** $31.00 | **IQR:** $23.09
- **Upper Outlier Cutoff ($Q3 + 1.5 \times IQR$):** **$65.63**
- **Total Fare Outliers Detected:** **116 passengers** (13.02% of sample)
- **Maximum Fare Recorded:** **$512.33** (Paid by elite 1st Class passengers in luxury suites)

---

## 🎨 4 Key Visualizations (`matplotlib` & `seaborn`)

### 1. Histogram: Age Distribution by Survival Outcome
![Age Histogram](visualizations/plot1_age_histogram.png)

---

### 2. Boxplot: Fare Distribution & Outlier Detection
![Fare Boxplot](visualizations/plot2_fare_boxplot.png)

---

### 3. Bar Chart: Survival Rate by Gender & Passenger Class
![Survival Rate Bar Chart](visualizations/plot3_survival_barchart.png)

---

### 4. Correlation Matrix Heatmap
![Correlation Heatmap](visualizations/plot4_correlation_heatmap.png)

---

## ❓ Feature Importance Analysis

### **Question: Which feature do you think most affects survival, and why?**

### 💡 **Answer:**
**`Sex` (Gender)** is the single most decisive feature affecting survival, followed closely by **`Pclass` (Passenger Class)**.

1. **Gender Priority (`Sex`):** Females achieved a **74.2%** overall survival rate, whereas males recorded only **18.9%** (correlation **+0.54**). This was driven by the strict enforcement of the historical maritime evacuation protocol: **"Women and children first"**.
2. **Socioeconomic Advantage (`Pclass`):** First-class passengers achieved a **62.9%** survival rate versus **24.2%** for 3rd class. First-class cabins were situated on the upper decks adjacent to lifeboat launch stations.
3. **Compound Effect:** 1st-class females achieved a **96.8%** survival rate, while 3rd-class males suffered an **86.5% mortality rate**.
