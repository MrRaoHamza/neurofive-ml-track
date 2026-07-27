# Neurofive ML Track - Task 1: Titanic EDA & Environment Setup

Welcome to **Task 1** of the **Neurofive ML Track**! Before jumping into machine learning algorithms, this project establishes a clean Python data science environment and performs a detailed Exploratory Data Analysis (EDA) on the classic Kaggle Titanic dataset.

---

## 📌 Project Overview
The primary goal of this initial task is to practice "listening" to a raw dataset before performing any modeling. We inspect the Titanic dataset's structure, statistical distributions, data types, and missing values to craft a baseline **Data Story**.

### Key Deliverables:
1. Python environment setup with `pandas`, `numpy`, and `jupyter`.
2. Titanic dataset download and ingestion.
3. Notebook (`eda_titanic.ipynb`) inspection using `df.head()`, `df.info()`, and `df.describe()`.
4. Analysis of missing values and categorization of feature types (Numerical vs. Categorical).
5. A concise **Data Story** summarizing initial data insights.
6. GitHub submission structure & video walkthrough guide for LinkedIn.

---

## 📁 Repository Structure
```
neurofive-ml-track/
├── eda_titanic.ipynb     # Main Jupyter Notebook containing EDA & Data Story
├── titanic.csv           # Titanic dataset (891 rows x 12 columns)
├── build_eda.py          # Python script to programmatically build notebook
├── README.md             # Project documentation & track guide
└── .gitignore            # Git ignore rules for Python/Jupyter workspace
```

---

## 🛠️ Environment Setup & Quickstart

### Prerequisites
- Python 3.8 or higher installed on your system.

### 1. Clone & Navigate
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/neurofive-ml-track.git
cd neurofive-ml-track
```

### 2. Install Required Libraries
Install the core data science toolkit:
```bash
pip install pandas numpy jupyter matplotlib seaborn
```

### 3. Launch Jupyter Notebook
```bash
jupyter notebook eda_titanic.ipynb
```
*(Alternatively, open `eda_titanic.ipynb` in VS Code or Google Colab).*

---

## 📊 Exploratory Data Analysis (EDA) Summary

### Dataset Metrics:
- **Total Rows (Observations):** 891
- **Total Columns (Features):** 12

### Feature Classification:
- **Numerical Features (7):** `PassengerId`, `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`
- **Categorical Features (5):** `Name`, `Sex`, `Ticket`, `Cabin`, `Embarked`

### Missing Values Identified:
- `Cabin`: **687 missing** (~77.10% missing rate)
- `Age`: **177 missing** (~19.87% missing rate)
- `Embarked`: **2 missing** (~0.22% missing rate)

---

## 📖 Data Story (5-6 Line Summary)

> The Titanic dataset contains **891 rows** and **12 columns**, capturing passenger demographics, ticket details, and survival outcomes. Numerical features include `Age`, `Fare`, `SibSp`, `Parch`, `PassengerId`, `Pclass`, and `Survived`, while `Name`, `Sex`, `Ticket`, `Cabin`, and `Embarked` form the categorical attributes. Significant data missingness is observed in `Cabin` (77.10%) and `Age` (19.87%), alongside 2 missing records in `Embarked`. Overall passenger survival rate stands at ~38.38%, with passenger ages ranging from 0.42 to 80 years old (mean age ~29.7 years). Due to high missingness, `Cabin` will likely require indicator encoding or removal, whereas `Age` will require feature imputation prior to predictive modeling.

---

## 🚀 How to Push to GitHub (`neurofive-ml-track`)

Follow these commands to create and push your public repository:

```bash
# 1. Initialize git inside this folder
git init

# 2. Add files and make initial commit
git add .
git commit -m "feat: complete Titanic EDA and data story for Task 1"

# 3. Rename branch to main
git branch -M main

# 4. Add your public GitHub remote repository
git remote add origin https://github.com/YOUR_GITHUB_USERNAME/neurofive-ml-track.git

# 5. Push code
git push -u origin main
```

---

## 🎥 LinkedIn Presentation Video Walkthrough Guide

As part of the Neurofive ML Track requirement, record a **2-3 minute screen recording** explaining your notebook:

1. **Introduction (15s):** Introduce yourself, state your participation in the **Neurofive ML Track**, and mention Task 1.
2. **Environment & Dataset Ingestion (30s):** Show `pandas` and `numpy` imports and loading `titanic.csv` into a dataframe.
3. **Inspection (`.head()`, `.info()`, `.describe()`) (45s):** Scroll through the output of these functions, pointing out the 891 rows and 12 columns.
4. **Missing Values & Feature Types (45s):** Highlight the missing values in `Cabin` (77.1%) and `Age` (19.87%) and distinguish categorical vs numerical columns.
5. **Data Story & Conclusion (30s):** Read through your 5-6 line Data Story summary cell and outline next steps for feature engineering.
6. **Posting:** Post your video on LinkedIn and tag **@Neurofive Solutions**!
