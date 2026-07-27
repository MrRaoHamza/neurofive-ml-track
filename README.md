# 🚢 Neurofive Machine Learning Track

Welcome to the official **Neurofive ML Track** repository maintained by **Rao Hamza Irshad**. This repository showcases complete, modular tasks covering Exploratory Data Analysis (EDA), Statistical Data Cleaning, Data Storytelling, **Linear Regression (Continuous Price Prediction)**, and **Logistic Regression (Binary Classification)**.

---

## 📌 Task Submission Matrix (Task-Wise Links)

| Track Task | Task Scope & Model Type | Key Deliverables & Performance | Direct Notebook Link |
| :--- | :--- | :--- | :--- |
| **Task 01** | **Baseline EDA & Setup** *(Exploratory Data Analysis)* | Structural inspection (`.info()`, `.describe()`, `.head()`), feature classification, 6-line executive Data Story. | 🔗 [**Task 1 Notebook**](tasks/Task-01-Baseline-EDA/Task_01_Titanic_EDA.ipynb) |
| **Task 02** | **Data Cleaning & Visual Storytelling** *(EDA & Visualization)* | Statistical `fillna()` justifications, IQR Outlier Detection, 4 Seaborn plots, written feature survival analysis. | 🔗 [**Task 2 Notebook**](tasks/Task-02-Cleaning-and-Visualization/Task_02_Data_Cleaning_Visualizations.ipynb) |
| **Task 03** | **Linear Regression Model** *(Regression - Housing Prices)* | Predict house values using 4 features (`MedInc`, `AveRooms`, `Latitude`, `Longitude`). **RMSE = $74,858.88**, **$R^2$ = 57.24%**, Scatter Plot, Plain English $R^2$ note. | 🔗 [**Task 3 Notebook**](tasks/Task-03-Linear-Regression-House-Prices/Task_03_Linear_Regression.ipynb) |
| **Task 04** | **Logistic Regression Model** *(Classification - Survival)* | Categorical One-Hot Encoding (`pd.get_dummies`), 80/20 train-test split, **Accuracy = 80.45%**, Confusion Matrix breakdown. | 🔗 [**Task 4 Notebook**](tasks/Task-04-Logistic-Regression-Titanic/Task_04_Logistic_Regression.ipynb) |

---

## 📂 Professional Repository Architecture

```
neurofive-ml-track/
│
├── tasks/                                            # Modular Task Directory
│   ├── Task-01-Baseline-EDA/
│   │   └── Task_01_Titanic_EDA.ipynb                 # Task 1 Notebook (Titanic EDA)
│   ├── Task-02-Cleaning-and-Visualization/
│   │   └── Task_02_Data_Cleaning_Visualizations.ipynb# Task 2 Notebook (Titanic Cleaning)
│   ├── Task-03-Linear-Regression-House-Prices/
│   │   └── Task_03_Linear_Regression.ipynb           # Task 3 Notebook (Linear Regression)
│   └── Task-04-Logistic-Regression-Titanic/
│       └── Task_04_Logistic_Regression.ipynb         # Task 4 Notebook (Logistic Regression)
│
├── data/                                             # Raw Datasets
│   ├── titanic.csv                                   # Titanic Dataset (891 rows x 12 columns)
│   └── california_housing.csv                        # California Housing Dataset (20,640 rows)
│
├── assets/                                           # High-Resolution Visualization Assets
│   ├── age_distribution_histogram.png                # Histogram: Age vs Survival
│   ├── fare_outliers_boxplot.png                     # Boxplot: Fare Outliers & Class
│   ├── survival_rate_barchart.png                    # Bar Chart: Gender & Class Survival
│   ├── correlation_heatmap.png                       # Heatmap: Feature Correlation Matrix
│   ├── predicted_vs_actual_housing.png               # Scatter Plot: Predicted vs Actual Prices
│   └── confusion_matrix.png                          # Heatmap: Confusion Matrix (Task 4)
│
├── src/                                              # Reusable Code & Build Pipeline
│   ├── generate_notebooks.py                         # Complete 4-task build pipeline
│   └── generate_task3.py                             # Task 3 pipeline script
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

### 3. Run Task Notebooks
```bash
jupyter notebook tasks/Task-03-Linear-Regression-House-Prices/Task_03_Linear_Regression.ipynb
jupyter notebook tasks/Task-04-Logistic-Regression-Titanic/Task_04_Logistic_Regression.ipynb
```

---

## 📈 Task 03: Linear Regression Model (House Price Prediction)

### 🛠️ Modeling Pipeline Overview
1. **Dataset Ingestion:** California Housing Dataset (`20,640` records).
2. **Predictor Selection (4 Features):** Selected `MedInc` (Median Neighborhood Income), `AveRooms` (Average Room Count), `Latitude`, and `Longitude`.
3. **Target Variable:** `MedHouseVal` (Median house value in $100,000s).
4. **Train-Test Split:** 80% Training (16,512 samples) / 20% Testing (4,128 samples) with `random_state=42`.
5. **Model:** Scikit-Learn `LinearRegression()`.

---

### 📊 Performance Metrics

- **$R^2$ Score:** **0.5724** (**57.24%** of house price variance explained).
- **Root Mean Squared Error (RMSE):** **0.7486** ($100,000 units), representing a typical estimation error of **$74,858.88**.

---

### 🖼️ Predicted vs. Actual Prices Scatter Plot
![Predicted vs Actual Prices](assets/predicted_vs_actual_housing.png)

---

### 💡 Plain English $R^2$ Explanation for Non-Technical Stakeholders

> Imagine trying to estimate the market price of a house before it goes up for sale. The **$R^2$ score (57.24%)** measures how much of the real-world variation in home prices our model can explain using basic neighborhood information like median income, room counts, and geographic location.
>
> Specifically, an **$R^2$ score of 0.57** means our model successfully accounts for **57.24%** of why home prices differ across neighborhoods. The remaining **42.76%** of price variation comes from unobserved factors outside our dataset, such as interior renovations, school district ratings, or recent market bidding wars.
>
> In practical business terms, our model provides a solid baseline automated valuation tool that estimates home prices within a typical margin of error (RMSE) of **$74,858.88**.

---

## 🤖 Task 04: Logistic Regression Model (Titanic Survival Classifier)

- **Test Accuracy:** **80.45%** ($144 / 179$ correct predictions).
- **Confusion Matrix:** $TN = 97$, $TP = 47$, $FP = 13$, $FN = 22$.
- **Precision for Survivors:** **78.33%** | **Recall for Survivors:** **68.12%**.

![Confusion Matrix](assets/confusion_matrix.png)

---

## 🎥 LinkedIn Presentation Guides

### Task 3 Walkthrough (2-3 mins):
- Introduce yourself and state your participation in the **Neurofive ML Track**.
- Explain why you selected `MedInc`, `AveRooms`, `Latitude`, and `Longitude` to predict house prices.
- Show the **Predicted vs. Actual Prices Scatter Plot** and explain the red ideal line ($y=x$).
- Share your **Plain English $R^2$ score explanation (57.24%)** and RMSE ($74,858).
- Post on LinkedIn tagging **@Neurofive Solutions**!
