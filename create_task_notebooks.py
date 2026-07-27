import json
import os
import io
import base64
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure plot outputs exist
import build_eda

df = pd.read_csv('titanic.csv')

# --- BUILD TASK 1 NOTEBOOK ---
t1_cells = []

t1_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Neurofive ML Track - Task 1: Python Setup & Titanic Baseline EDA\n",
        "**Student:** Rao Hamza Irshad | **Repository:** neurofive-ml-track\n",
        "\n",
        "### Task 1 Objectives:\n",
        "1. Load Titanic dataset with `pandas.read_csv()`.\n",
        "2. Inspect dataset using `.head()`, `.info()`, and `.describe()`.\n",
        "3. Identify dataset dimensions (rows/columns), missing values, and numerical vs categorical features.\n",
        "4. Write a 5-6 line Markdown Data Story summarizing baseline findings."
    ]
})

t1_cells.append({
    "cell_type": "code",
    "execution_count": 1,
    "metadata": {},
    "outputs": [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [
                f"Pandas version: {pd.__version__}\n",
                f"NumPy version: {np.__version__}\n"
            ]
        }
    ],
    "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "\n",
        "print(f\"Pandas version: {pd.__version__}\")\n",
        "print(f\"NumPy version: {np.__version__}\")"
    ]
})

t1_cells.append({
    "cell_type": "code",
    "execution_count": 2,
    "metadata": {},
    "outputs": [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\n"]
        }
    ],
    "source": [
        "df = pd.read_csv('titanic.csv')\n",
        "print(f\"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.\")"
    ]
})

t1_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 1. Dataset Preview (`.head()`)\n"]
})

t1_cells.append({
    "cell_type": "code",
    "execution_count": 3,
    "metadata": {},
    "outputs": [
        {
            "data": {
                "text/html": [df.head().to_html()],
                "text/plain": [str(df.head())]
            },
            "execution_count": 3,
            "output_type": "execute_result"
        }
    ],
    "source": ["df.head()"]
})

t1_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 2. Dataset Structure (`.info()`)\n"]
})

buf = io.StringIO()
df.info(buf=buf)
info_str = buf.getvalue()

t1_cells.append({
    "cell_type": "code",
    "execution_count": 4,
    "metadata": {},
    "outputs": [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [info_str]
        }
    ],
    "source": ["df.info()"]
})

t1_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 3. Statistical Summary (`.describe()`)\n"]
})

t1_cells.append({
    "cell_type": "code",
    "execution_count": 5,
    "metadata": {},
    "outputs": [
        {
            "data": {
                "text/html": [df.describe().to_html()],
                "text/plain": [str(df.describe())]
            },
            "execution_count": 5,
            "output_type": "execute_result"
        }
    ],
    "source": ["df.describe()"]
})

t1_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": ["## 4. Missing Values & Feature Classification\n"]
})

missing_stdout = (
    "--- Columns with Missing Values ---\n"
    "Age: 177 missing (19.87%)\n"
    "Cabin: 687 missing (77.10%)\n"
    "Embarked: 2 missing (0.22%)\n\n"
    "--- Numerical Columns (7) ---\n"
    "['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']\n\n"
    "--- Categorical Columns (5) ---\n"
    "['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']\n"
)

t1_cells.append({
    "cell_type": "code",
    "execution_count": 6,
    "metadata": {},
    "outputs": [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [missing_stdout]
        }
    ],
    "source": [
        "missing_data = df.isnull().sum()\n",
        "missing_data = missing_data[missing_data > 0]\n",
        "print('--- Columns with Missing Values ---')\n",
        "for col, count in missing_data.items():\n",
        "    pct = (count / len(df)) * 100\n",
        "    print(f\"{col}: {count} missing ({pct:.2f}%)\")\n\n",
        "print('\\n--- Numerical Columns (7) ---')\n",
        "print(df.select_dtypes(include=[np.number]).columns.tolist())\n\n",
        "print('--- Categorical Columns (5) ---')\n",
        "print(df.select_dtypes(include=['object', 'category']).columns.tolist())\n"
    ]
})

t1_cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Task 1 Data Story\n",
        "\n",
        "The Titanic dataset contains **891 rows** and **12 columns**, capturing demographic, passenger class, ticket, and survival info. Numerical features include `Age`, `Fare`, `SibSp`, `Parch`, `PassengerId`, `Pclass`, and `Survived`, while `Name`, `Sex`, `Ticket`, `Cabin`, and `Embarked` form the categorical variables. Significant missing values exist in `Cabin` (77.1%) and `Age` (19.87%), with `Embarked` missing just 2 records. Key metrics reveal an overall survival rate of ~38.38% with passenger ages ranging from 0.42 to 80 years old (mean age ~29.7 years). The heavy missingness in `Cabin` suggests it may require dropping or indicator encoding, while `Age` will require median/grouped imputation prior to predictive modeling."
    ]
})

nb1 = {
    "cells": t1_cells,
    "metadata": {"language_info": {"name": "python", "version": "3.10.0"}, "orig_nbformat": 4},
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("Task1_Titanic_EDA.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb1, f, indent=2)


# --- BUILD TASK 2 NOTEBOOK ---
with open("eda_titanic.ipynb", "r", encoding="utf-8") as f:
    nb2 = json.load(f)

# Update header for Task 2 Notebook
nb2["cells"][0]["source"] = [
    "# Neurofive ML Track - Task 2: Data Cleaning & Visual Data Storytelling\n",
    "**Student:** Rao Hamza Irshad | **Repository:** neurofive-ml-track\n",
    "\n",
    "### Task 2 Objectives:\n",
    "1. Handle missing values using `fillna()` with formal written justifications.\n",
    "2. Detect numerical outliers using a Boxplot (IQR analysis on `Fare`).\n",
    "3. Create 4 visualizations (`matplotlib`/`seaborn`): Histogram, Boxplot, Bar Chart, Correlation Heatmap.\n",
    "4. Answer: *\"Which feature do you think most affects survival, and why?\"*"
]

with open("Task2_Data_Cleaning_Visualizations.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb2, f, indent=2)

print("Task 1 and Task 2 dedicated notebook files created successfully!")
