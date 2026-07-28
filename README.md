# 🚢 Neurofive Machine Learning Track

Welcome to the official **Neurofive ML Track** repository maintained by **Rao Hamza Irshad**. This repository showcases complete, modular tasks covering Exploratory Data Analysis (EDA), Statistical Data Cleaning, Data Storytelling, **Linear Regression (Price Prediction)**, **Logistic Regression (Classification)**, **Hyperparameter Tuning (`GridSearchCV`)**, **Telco Customer Churn Prediction (Decision Trees)**, and **Scikit-Learn Production Pipelines & Feature Engineering**.

---

## 📌 Task Submission Matrix (Task-Wise Links)

| Track Task | Task Scope & Model Type | Key Deliverables & Performance | Direct Notebook Link |
| :--- | :--- | :--- | :--- |
| **Task 01** | **Baseline EDA & Setup** *(Exploratory Data Analysis)* | Structural inspection (`.info()`, `.describe()`, `.head()`), feature classification, 6-line executive Data Story. | 🔗 [**Task 1 Notebook**](tasks/Task-01-Baseline-EDA/Task_01_Titanic_EDA.ipynb) |
| **Task 02** | **Data Cleaning & Visual Storytelling** *(EDA & Visualization)* | Statistical `fillna()` justifications, IQR Outlier Detection, 4 Seaborn plots, written feature survival analysis. | 🔗 [**Task 2 Notebook**](tasks/Task-02-Cleaning-and-Visualization/Task_02_Data_Cleaning_Visualizations.ipynb) |
| **Task 03** | **Linear Regression Model** *(Regression - Housing Prices)* | Predict house values using 4 features (`MedInc`, `AveRooms`, `Latitude`, `Longitude`). **RMSE = $74,858.88**, **$R^2$ = 57.24%**, Scatter Plot. | 🔗 [**Task 3 Notebook**](tasks/Task-03-Linear-Regression-House-Prices/Task_03_Linear_Regression.ipynb) |
| **Task 04** | **Logistic Regression Model** *(Classification - Survival)* | Categorical One-Hot Encoding (`pd.get_dummies`), 80/20 train-test split, **Accuracy = 80.45%**, Confusion Matrix breakdown. | 🔗 [**Task 4 Notebook**](tasks/Task-04-Logistic-Regression-Titanic/Task_04_Logistic_Regression.ipynb) |
| **Task 05** | **Hyperparameter Tuning & Evaluation** *(Model Optimization)* | Precision/Recall/F1 evaluation, "Why Accuracy Lies" note, 5-Fold `GridSearchCV` tuning, Before/After performance comparison table. | 🔗 [**Task 5 Notebook**](tasks/Task-05-Hyperparameter-Tuning-Evaluation/Task_05_Hyperparameter_Tuning.ipynb) |
| **Task 06** | **Telco Customer Churn Prediction** *(Decision Trees & Business Pitch)* | IBM Telco Churn EDA, Decision Tree vs. Logistic Regression comparison, **Top 3 Churn Drivers** (`tenure`, `Fiber Optic`, `TotalCharges`), 5-sentence Business Summary. | 🔗 [**Task 6 Notebook**](tasks/Task-06-Telco-Customer-Churn/Task_06_Telco_Customer_Churn.ipynb) |
| **Task 07** | **Scikit-Learn Production Pipelines** *(Leak-Free ML & Feature Engineering)* | `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, custom `TitanicFeatureEngineer` (`FamilySize`, `IsAlone`, `FarePerPerson`), **Accuracy = 81.01%**, **ROC-AUC = 0.8524**, `joblib` serialization. | 🔗 [**Task 7 Notebook**](tasks/Task-07-Scikit-Learn-Pipeline/Task_07_Scikit_Learn_Pipeline.ipynb) |

---

## 📂 Professional Repository Architecture

```
neurofive-ml-track/
│
├── tasks/                                            # Modular Task Directory
│   ├── Task-01-Baseline-EDA/
│   │   └── Task_01_Titanic_EDA.ipynb                 # Task 1 Notebook (Baseline EDA)
│   ├── Task-02-Cleaning-and-Visualization/
│   │   └── Task_02_Data_Cleaning_Visualizations.ipynb# Task 2 Notebook (Data Cleaning)
│   ├── Task-03-Linear-Regression-House-Prices/
│   │   └── Task_03_Linear_Regression.ipynb           # Task 3 Notebook (Linear Regression)
│   ├── Task-04-Logistic-Regression-Titanic/
│   │   └── Task_04_Logistic_Regression.ipynb         # Task 4 Notebook (Logistic Regression)
│   ├── Task-05-Hyperparameter-Tuning-Evaluation/
│   │   └── Task_05_Hyperparameter_Tuning.ipynb       # Task 5 Notebook (GridSearchCV Tuning)
│   ├── Task-06-Telco-Customer-Churn/
│   │   └── Task_06_Telco_Customer_Churn.ipynb        # Task 6 Notebook (Decision Trees & Churn)
│   └── Task-07-Scikit-Learn-Pipeline/
│       ├── Task_07_Scikit_Learn_Pipeline.ipynb       # Task 7 Notebook (Scikit-Learn Pipelines)
│       └── titanic_pipeline.joblib                   # Production Pipeline Artifact
│
├── models/                                           # Saved Production Pipeline Artifacts
│   └── titanic_pipeline.joblib                       # Serialized End-to-End Model Pipeline
│
├── data/                                             # Raw Datasets
│   ├── titanic.csv                                   # Titanic Dataset (891 rows x 12 columns)
│   ├── california_housing.csv                        # California Housing Dataset (20,640 rows)
│   └── telco_customer_churn.csv                      # Telco Customer Churn Dataset (7,043 rows)
│
├── assets/                                           # High-Resolution Visualization Assets
│   ├── age_distribution_histogram.png                # Histogram: Age vs Survival
│   ├── fare_outliers_boxplot.png                     # Boxplot: Fare Outliers & Class
│   ├── survival_rate_barchart.png                    # Bar Chart: Gender & Class Survival
│   ├── correlation_heatmap.png                       # Heatmap: Feature Correlation Matrix
│   ├── predicted_vs_actual_housing.png               # Scatter Plot: Predicted vs Actual Prices
│   ├── confusion_matrix.png                          # Heatmap: Baseline Confusion Matrix
│   ├── confusion_matrix_tuned.png                    # Heatmap: Tuned Confusion Matrix (Task 5)
│   ├── telco_churn_eda.png                           # Bar Chart: Churn Rate by Contract Type (Task 6)
│   ├── churn_feature_importances.png                 # Bar Chart: Decision Tree Feature Importances (Task 6)
│   ├── pipeline_architecture_diagram.png             # Schematic: Scikit-Learn Pipeline Architecture (Task 7)
│   ├── pipeline_performance_comparison.png           # Bar Chart: Manual vs Pipeline Benchmark (Task 7)
│   └── pipeline_confusion_matrix.png                 # Heatmap: Final Pipeline Confusion Matrix (Task 7)
│
├── src/                                              # Reusable Code & Build Pipeline
│   ├── generate_notebooks.py                         # Baseline build script
│   ├── generate_task3.py                             # Task 3 generator
│   ├── generate_task5.py                             # Task 5 generator
│   ├── generate_task6.py                             # Task 6 generator
│   └── generate_task7.py                             # Task 7 pipeline generator
│
├── requirements.txt                                  # Project Dependencies (scikit-learn, joblib, etc.)
├── README.md                                         # Executive Repository Documentation
└── .gitignore                                        # Workspace Git Ignore Rules
```

---

## ⚙️ Environment Setup & Quickstart

```bash
git clone https://github.com/MrRaoHamza/neurofive-ml-track.git
cd neurofive-ml-track
pip install -r requirements.txt
python src/generate_task7.py
jupyter notebook tasks/Task-07-Scikit-Learn-Pipeline/Task_07_Scikit_Learn_Pipeline.ipynb
```

---

## ⚙️ Task 07: Scikit-Learn Production Pipelines & Leak-Free ML

### 🛠️ Modular Architecture & Data Leakage Prevention
1. **The Data Leakage Problem:** Manual preprocessing (e.g., fitting `StandardScaler` or imputing missing values on an entire dataset prior to splitting) leaks future test set information into training folds. This produces artificial validation scores that fail in production.
2. **Scikit-Learn `Pipeline` & `ColumnTransformer` Solution:** We chained all numeric imputation (`SimpleImputer(strategy='median')`) and scaling (`StandardScaler`), as well as categorical imputation (`SimpleImputer(strategy='most_frequent')`) and encoding (`OneHotEncoder(drop='first', handle_unknown='ignore')`), into a single callable object.
3. **Custom Feature Engineering Transformer:** Constructed `TitanicFeatureEngineer(BaseEstimator, TransformerMixin)` to dynamically create:
   - **`FamilySize`** = `SibSp` + `Parch` + 1
   - **`IsAlone`** = 1 if `FamilySize == 1` else 0
   - **`FarePerPerson`** = `Fare` / `FamilySize`
4. **End-to-End Integration:** The final model chains `TitanicFeatureEngineer` ➔ `ColumnTransformer` ➔ `LogisticRegression(C=1.5)`, ensuring 100% parameter isolation between train and evaluation splits.

![Pipeline Architecture](assets/pipeline_architecture_diagram.png)

---

### 📊 Benchmark Model Comparison: Manual vs. Standard Pipeline vs. Engineered Pipeline

| Evaluation Metric | Manual Baseline (Task 4/5) | Standard Pipeline Baseline | Engineered Pipeline (Final) | Performance Improvement |
| :--- | :---: | :---: | :---: | :--- |
| **Accuracy** | 79.89% | 79.89% | **81.01%** | **+1.12%** — Higher overall classification accuracy |
| **Precision (Survivors)** | 77.97% | 77.97% | **78.33%** | **+0.36%** — Reduced false positive survival predictions |
| **Recall (Survivors)** | 66.67% | 66.67% | **68.12%** | **+1.45%** — Correctly identified more actual survivors |
| **F1-Score (Survivors)** | 0.7188 | 0.7188 | **0.7287** | **+0.0099** — Superior harmonic mean of Precision & Recall |
| **ROC-AUC Score** | 0.8436 | 0.8436 | **0.8524** | **+0.0088** — Enhanced class probability separation |

![Performance Comparison](assets/pipeline_performance_comparison.png)

---

### 🖼️ Diagnostic Matrix & Production Deployment
![Pipeline Confusion Matrix](assets/pipeline_confusion_matrix.png)

**Model Serialization (`joblib`):**  
The full fitted pipeline was serialized to `models/titanic_pipeline.joblib`. Production inference requires just one line of code without reproducing any manual data transformation:
```python
import joblib
pipeline = joblib.load("models/titanic_pipeline.joblib")
predictions = pipeline.predict(raw_passenger_dataframe)
```

---

## 📉 Task 06: Telco Customer Churn Prediction & Decision Trees

### 🛠️ Modeling Pipeline & Data Preprocessing
1. **Dataset Ingestion:** Loaded **7,043 customer records** from the Telco Customer Churn dataset.
2. **Data Cleaning:** Converted whitespace strings in `TotalCharges` to floats (`pd.to_numeric`) and imputed 11 missing values with median. Dropped non-informative `customerID`.
3. **Class Imbalance Note:** Target variable `Churn` exhibits imbalance with **26.54% churners** (1,869 customers) vs **73.46% retained** (5,174 customers).
4. **Model Comparison:** Evaluated a **Decision Tree Classifier (`max_depth=5`)** against a **Logistic Regression (Class-Balanced)** model on an 80/20 train-test split (1,409 test samples).

---

### 📊 Decision Tree vs. Logistic Regression Model Comparison

| Evaluation Metric | Decision Tree (`max_depth=5`) | Logistic Regression (`class_weight='balanced'`) | Winning Model & Takeaway |
| :--- | :---: | :---: | :--- |
| **Accuracy** | **79.42%** | **73.88%** | **Decision Tree (+5.54%)** — Better overall classification precision |
| **Precision (Churners)** | **62.96%** | **50.52%** | **Decision Tree (+12.44%)** — Fewer false churn warnings |
| **Recall (Churners)** | **54.55%** | **78.07%** | **Logistic Regression (+23.52%)** — Catches significantly more churners |
| **F1-Score (Churners)** | **58.45%** | **61.34%** | **Logistic Regression (+2.89%)** — Superior balance of precision/recall |
| **ROC-AUC Score** | **0.8284** | **0.8412** | **Logistic Regression (+0.0128)** — Better probability discrimination |

---

### 🔍 Top 3 Features Driving Churn (`.feature_importances_`)

Using `dt_model.feature_importances_` from the Decision Tree Classifier, we identified the top 3 drivers of customer loss:

1. **`tenure` (42.14% Importance Weight):** Customer tenure in months is the single strongest indicator. Newer customers (0–12 months) display the highest vulnerability to churn.
2. **`InternetService_Fiber optic` (35.75% Importance Weight):** Customers subscribing to Fiber Optic internet churn at disproportionately higher rates due to premium monthly charges.
3. **`TotalCharges` (4.71% Importance Weight):** Accumulated billing total reflects long-term customer value and churn risk threshold.

![Feature Importances](assets/churn_feature_importances.png)

---

### 💼 Executive Business Summary (Presentation to Leadership)

> Our predictive analysis of **7,043 telecom customers** reveals an annual churn rate of **26.54%**, representing substantial recurring revenue loss. Using a Decision Tree classification model (79.42% accuracy), we identified the **top 3 primary drivers of customer departure** as: **(1) Customer Tenure** (42.1% of predictive weight), **(2) Fiber Optic Internet Service** (35.8% of weight), and **(3) Accumulated Total Charges**.
>
> Crucially, customers on month-to-month contracts churn at an alarming **42.7% rate**, compared to just **2.8%** for two-year contract holders. To immediately reduce churn and preserve annual recurring revenue, executive leadership should implement targeted multi-year contract upgrade discounts, offer bundled technical support incentives during a customer's first 12 months, and conduct pricing reviews on premium Fiber Optic packages.

---

## 🎥 LinkedIn Presentation Guides

### Task 7 Walkthrough (2-3 mins):
- **Hook & Core Problem:** Introduce yourself as a Data Scientist in the **Neurofive ML Track**. Explain that professional ML isn't a collection of disparate notebook cells, but a robust pipeline that prevents **data leakage**.
- **Visual Demo:** Show the **Scikit-Learn Pipeline Architecture Diagram** ([`assets/pipeline_architecture_diagram.png`](assets/pipeline_architecture_diagram.png)) and explain how `ColumnTransformer` handles `StandardScaler` for numeric features and `OneHotEncoder` for categorical features automatically.
- **Feature Engineering:** Highlight your 3 custom engineered features (`FamilySize`, `IsAlone`, `FarePerPerson`) created via a custom scikit-learn transformer, boosting model accuracy to **81.01%** and ROC-AUC to **0.8524**.
- **Production Export:** Demonstrate saving the complete pipeline to `models/titanic_pipeline.joblib` and performing single-line inference on unseen passenger data.
- **Call to Action:** Post on LinkedIn tagging **@Neurofive Solutions**!

---

### Task 6 Walkthrough (2-3 mins):
- Introduce yourself as a Data Scientist in the **Neurofive ML Track**.
- Present the business problem: **Telecom Customer Churn (26.54% loss rate)**.
- Show the **Decision Tree Feature Importances chart** ([`assets/churn_feature_importances.png`](assets/churn_feature_importances.png)) and explain the Top 3 drivers (`tenure`, `Fiber Optic`, `TotalCharges`).
- Pitch your **4-5 sentence Executive Business Summary** to non-technical client stakeholders.
- Post on LinkedIn tagging **@Neurofive Solutions**!
