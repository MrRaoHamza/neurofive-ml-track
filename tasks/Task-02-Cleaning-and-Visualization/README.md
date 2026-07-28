# 🧹 Task 02: Data Cleaning, Outlier Handling & Visual Data Storytelling

**Author:** Rao Hamza Irshad  
**Track:** Neurofive Machine Learning Track — Task 02  
**Dataset:** Titanic Dataset (`data/titanic.csv`)  
**Notebook:** [`Task_02_Data_Cleaning_Visualizations.ipynb`](Task_02_Data_Cleaning_Visualizations.ipynb)  
**Master Repository:** [← Back to Master README](../../README.md)

---

## 📌 Executive Summary
Task 02 performs statistical data cleaning, handles missing values using domain-justified strategies, executes Interquartile Range (IQR) outlier detection, and creates 4 high-resolution diagnostic Seaborn visualizations to uncover survival patterns.

---

## 🛠️ Data Cleaning & Imputation Strategy

1. **`Age` Imputation:** Imputed 177 missing values using the **median age grouped by `Pclass` and `Sex`** rather than global mean, preserving demographic variance (e.g., 1st Class males median age = 40 vs 3rd Class males median age = 25).
2. **`Embarked` Imputation:** Filled 2 missing values with the **mode ('S' = Southampton)**.
3. **`Cabin` Transformation:** Converted high-cardinality `Cabin` into binary feature `HasCabin` (1 = Cabin recorded, 0 = Missing).

---

## 📊 IQR Outlier Detection Analysis

Using the Interquartile Range formula ($IQR = Q3 - Q1$), we identified upper bound outliers ($Q3 + 1.5 \times IQR$):
- **`Fare` Outliers:** 116 passenger records exceeded the upper boundary of **$66.30**, representing luxury suites (up to $512.33). These were retained for log-transformation rather than deleted, as high fares correlate strongly with 1st Class survival.
- **`Age` Outliers:** 11 passenger records exceeded the upper boundary of **64.8 years**.

---

## 🖼️ Diagnostic Visualizations Generated

- **Histogram (`age_distribution_histogram.png`):** Age distribution by survival status demonstrating pediatric priority ("women and children first").
- **Boxplot (`fare_outliers_boxplot.png`):** Ticket fare outliers across passenger classes.
- **Bar Chart (`survival_rate_barchart.png`):** Cross-categorical survival rate comparing `Sex` and `Pclass`.
- **Correlation Heatmap (`correlation_heatmap.png`):** Matrix of numerical feature correlations.

![Survival Rate Bar Chart](../../assets/survival_rate_barchart.png)

---

## 📂 Task Artifacts
- **Jupyter Notebook:** [`Task_02_Data_Cleaning_Visualizations.ipynb`](Task_02_Data_Cleaning_Visualizations.ipynb)
- **Visual Plots:** [`../../assets/`](../../assets/)
