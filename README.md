# Neurofive ML Track - Titanic EDA & Visual Data Storytelling (Task 1 & Task 2)

Welcome to the **Neurofive ML Track** repository! This project covers **Task 1** (Environment Setup & Baseline EDA) and **Task 2** (Data Cleaning, Outlier Detection, Visual Data Storytelling, and Feature Importance Analysis) using the classic Titanic dataset.

---

## 📌 Project Overview
- **Task 1:** Inspect raw dataset structure (`df.head()`, `df.info()`, `df.describe()`), missingness overview, and feature categorization.
- **Task 2:** Clean missing values with formal statistical justifications, detect numerical outliers via IQR and boxplots, build 4 distinct visualizations using `seaborn` and `matplotlib`, answer the core survival driver question, and prepare a public project release.

---

## 📁 Repository Structure
```
neurofive-ml-track/
├── eda_titanic.ipynb              # Main Jupyter Notebook (Tasks 1 & 2 combined)
├── titanic.csv                    # Kaggle Titanic Dataset (891 rows x 12 columns)
├── build_eda.py                   # Python script generating notebook & visualizations
├── visualizations/                # Generated high-resolution plots (.png)
│   ├── plot1_age_histogram.png
│   ├── plot2_fare_boxplot.png
│   ├── plot3_survival_barchart.png
│   └── plot4_correlation_heatmap.png
├── README.md                      # Complete project documentation
└── .gitignore                     # Git ignore rules for Python/Jupyter
```

---

## 🧹 Data Cleaning Strategy & Justifications (`fillna()` vs `dropna()`)

Real-world datasets contain missing entries. Dropping rows blindly distorts sample sizes, while improper filling introduces bias.

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

*Insight:* Extreme `Fare` outliers reflect genuine luxury accommodations (e.g. Cardeza & Widener suites) rather than measurement errors.

---

## 🎨 4 Key Visualizations (`matplotlib` & `seaborn`)

### 1. Histogram: Age Distribution by Survival Outcome
![Age Histogram](visualizations/plot1_age_histogram.png)
*Highlights child survival priority (<10 years old) and high mortality among young adults (20-30 years).*

---

### 2. Boxplot: Fare Distribution & Outlier Detection
![Fare Boxplot](visualizations/plot2_fare_boxplot.png)
*Illustrates ticket price variance across classes and extreme fare outliers in 1st Class.*

---

### 3. Bar Chart: Survival Rate by Gender & Passenger Class
![Survival Rate Bar Chart](visualizations/plot3_survival_barchart.png)
*Shows near-certain survival for 1st/2nd Class females (96.8% & 92.1%) vs stark 3rd Class male mortality (13.5%).*

---

### 4. Correlation Matrix Heatmap
![Correlation Heatmap](visualizations/plot4_correlation_heatmap.png)
*Quantifies relationship strengths: `Sex_Numeric` (+0.54) and `Pclass` (-0.34) exhibit strongest correlations with `Survived`.*

---

## ❓ Feature Importance Analysis

### **Question: Which feature do you think most affects survival, and why?**

### 💡 **Answer:**
**`Sex` (Gender)** is the single most decisive feature affecting survival, followed closely by **`Pclass` (Passenger Class)**.

#### **Why? (Empirical & Historical Evidence):**
1. **Gender Priority (`Sex`):** Females achieved a **74.2%** overall survival rate, whereas males recorded only **18.9%**. This was driven by the strict enforcement of the historical maritime evacuation protocol: **"Women and children first"**.
2. **Socioeconomic Advantage (`Pclass`):** First-class passengers achieved a **62.9%** survival rate versus **24.2%** for 3rd class. First-class cabins were situated on the upper decks adjacent to lifeboat launch stations, giving them physical proximity and priority access.
3. **Compound Effect:** 1st-class females achieved a **96.8%** survival rate, while 3rd-class males suffered an **86.5% mortality rate**.

---

## 🚀 How to Commit & Push to GitHub

To push Task 2 updates to your repository:

```bash
# 1. Check workspace status
git status

# 2. Stage updated notebook, script, README, and visualizations
git add .

# 3. Commit with a descriptive commit message
git commit -m "feat: complete data cleaning, outlier detection, 4 visualizations, and feature analysis for Task 2"

# 4. Push to remote main branch
git push origin main
```

---

## 🎥 LinkedIn Video Walkthrough Guide (Task 2)

Record a **2–3 minute video** highlighting your findings:

1. **Introduction (15s):** Introduce yourself and announce **Task 2** of the **Neurofive ML Track**.
2. **Data Cleaning Choices (30s):** Explain why median imputation was selected for `Age` (avoiding data loss) and how `Cabin` missingness was turned into a binary feature.
3. **Surprising Visualization (60s):** Feature the **Survival Rate by Gender & Class Bar Chart** or **Fare Boxplot Outliers**. Explain what surprised you (e.g. 96.8% survival for 1st-class females vs 13.5% for 3rd-class males).
4. **Feature Importance Answer (30s):** Summarize why `Sex` and `Pclass` governed survival rates ("women and children first" protocol + upper deck access).
5. **Call to Action:** Post on LinkedIn tagging **@Neurofive Solutions**!
