# 🚢 Task 01: Baseline Exploratory Data Analysis & Workspace Setup

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Task 01  
**Dataset:** Titanic Survival Dataset (`data/titanic.csv` - 891 records, 12 features)  
**Notebook:** [`Task_01_Titanic_EDA.ipynb`](Task_01_Titanic_EDA.ipynb)  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Summary
Task 01 establishes the foundational data ingestion and exploratory workflow for the **Neurofive ML Track**. It performs structural inspection (`.info()`, `.describe()`, `.head()`), identifies categorical vs. numerical features, audits missing value distributions, and formulates a 6-line executive Data Story.

---

## 📊 Dataset Structure & Data Types

- **Total Passenger Records:** 891 rows
- **Total Attributes:** 12 columns (1 Target: `Survived`, 11 Predictors)
- **Numerical Attributes (5):** `Age`, `Fare`, `SibSp`, `Parch`, `PassengerId`
- **Categorical Attributes (5):** `Sex`, `Embarked`, `Pclass`, `Ticket`, `Cabin`
- **Target Label:** `Survived` (Binary: 0 = Perished, 1 = Survived)

---

## 🔍 Data Quality Audit & Missing Value Summary

| Feature Column | Total Non-Null Rows | Missing Count | Missing Percentage | Action Strategy |
| :--- | :---: | :---: | :---: | :--- |
| **`Age`** | 714 | 177 | 19.87% | Median imputation grouped by `Pclass` & `Sex` |
| **`Cabin`** | 204 | 687 | 77.10% | Drop column or engineer `HasCabin` indicator |
| **`Embarked`** | 889 | 2 | 0.22% | Mode imputation ('S' = Southampton) |
| **All Other Columns** | 891 | 0 | 0.00% | Clean & ready for feature processing |

---

## 📖 6-Line Executive Data Story

> 1. The Titanic dataset contains 891 passenger records across 12 features, with 342 survivors representing a 38.38% overall survival rate.
> 2. Gender exhibited the strongest bivariate correlation with survival: 74.20% of female passengers survived compared to only 18.89% of male passengers.
> 3. Passenger ticket class (`Pclass`) revealed stark socio-economic disparities, with 1st Class passengers achieving a 62.96% survival rate versus 24.24% for 3rd Class.
> 4. Data quality inspection identified 177 missing values in `Age` (19.87%) and 687 missing values in `Cabin` (77.10%).
> 5. Outliers were detected in ticket fares (`Fare`), with top-tier luxury tickets reaching up to $512.33 compared to a median fare of $14.45.
> 6. This initial EDA establishes the baseline feature set required for statistical cleaning and classification modeling in subsequent tasks.

---

## 📂 Task Artifacts
- **Jupyter Notebook:** [`Task_01_Titanic_EDA.ipynb`](Task_01_Titanic_EDA.ipynb)
- **Dataset File:** [`../../data/titanic.csv`](../../data/titanic.csv)
