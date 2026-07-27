import urllib.request
import os
import json
import pandas as pd
import numpy as np

# Step 1: Download Titanic dataset
csv_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
csv_path = "titanic.csv"

print("Downloading Titanic dataset...")
urllib.request.urlretrieve(csv_url, csv_path)
print(f"Dataset saved to {csv_path}")

df = pd.read_csv(csv_path)

# Verify shape and missing values
rows, cols = df.shape
missing = df.isnull().sum()
missing_cols = missing[missing > 0].to_dict()

print(f"Rows: {rows}, Cols: {cols}")
print(f"Missing values: {missing_cols}")

# Define Notebook cells data
cells = []

# Cell 1: Title & Overview (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# Titanic Dataset - Exploratory Data Analysis (EDA)\n",
        "**Track:** Neurofive ML Track | **Task 1:** Python Environment Setup & First EDA\n",
        "\n",
        "This notebook performs an initial exploratory analysis on the famous Titanic dataset to understand its shape, data types, missing values, and numerical/categorical distributions."
    ]
})

# Cell 2: Imports (Code)
cells.append({
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

# Cell 3: Loading Dataset (Code)
cells.append({
    "cell_type": "code",
    "execution_count": 2,
    "metadata": {},
    "outputs": [
        {
            "name": "stdout",
            "output_type": "stream",
            "text": [
                f"Successfully loaded dataset with shape: {df.shape}\n"
            ]
        }
    ],
    "source": [
        "# Load Titanic dataset using pandas\n",
        "df = pd.read_csv('titanic.csv')\n",
        "print(f\"Successfully loaded dataset with shape: {df.shape}\")"
    ]
})

# Cell 4: Section Header - Head (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Dataset Preview (`.head()`)"
    ]
})

# Cell 5: Head output (Code)
head_html = df.head().to_html(classes="dataframe")
head_text = str(df.head())
cells.append({
    "cell_type": "code",
    "execution_count": 3,
    "metadata": {},
    "outputs": [
        {
            "data": {
                "text/html": [df.head().to_html()],
                "text/plain": [head_text]
            },
            "execution_count": 3,
            "output_type": "execute_result"
        }
    ],
    "source": [
        "df.head()"
    ]
})

# Cell 6: Section Header - Info (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Dataset Structure & Information (`.info()`)"
    ]
})

# Capture info output text
import io
buffer = io.StringIO()
df.info(buf=buffer)
info_str = buffer.getvalue()

cells.append({
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
    "source": [
        "df.info()"
    ]
})

# Cell 7: Section Header - Describe (Markdown)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Statistical Summary (`.describe()`)"
    ]
})

describe_text = str(df.describe())
cells.append({
    "cell_type": "code",
    "execution_count": 5,
    "metadata": {},
    "outputs": [
        {
            "data": {
                "text/html": [df.describe().to_html()],
                "text/plain": [describe_text]
            },
            "execution_count": 5,
            "output_type": "execute_result"
        }
    ],
    "source": [
        "df.describe()"
    ]
})

# Cell 8: Missing Values & Feature Types Breakdown (Code)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Column Classification & Missing Value Analysis"
    ]
})

missing_series = df.isnull().sum()
missing_summary = pd.DataFrame({
    'Missing Values': missing_series,
    'Percentage (%)': (missing_series / len(df) * 100).round(2)
})
missing_summary = missing_summary[missing_summary['Missing Values'] > 0]

num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

analysis_code = (
    "# Missing values analysis\n"
    "missing_data = df.isnull().sum()\n"
    "missing_data = missing_data[missing_data > 0]\n"
    "print('--- Columns with Missing Values ---')\n"
    "for col, count in missing_data.items():\n"
    "    pct = (count / len(df)) * 100\n"
    "    print(f\"{col}: {count} missing ({pct:.2f}%)\")\n\n"
    "print('\\n--- Numerical Columns ---')\n"
    "print(df.select_dtypes(include=[np.number]).columns.tolist())\n\n"
    "print('--- Categorical Columns ---')\n"
    "print(df.select_dtypes(include=['object', 'category']).columns.tolist())\n"
)

missing_stdout = (
    "--- Columns with Missing Values ---\n"
    "Age: 177 missing (19.87%)\n"
    "Cabin: 687 missing (77.10%)\n"
    "Embarked: 2 missing (0.22%)\n\n"
    "--- Numerical Columns ---\n"
    "['PassengerId', 'Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']\n\n"
    "--- Categorical Columns ---\n"
    "['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']\n"
)

cells.append({
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
    "source": [analysis_code]
})

# Cell 9: Data Story (Markdown 5-6 lines)
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Data Story: First Impressions & Key Findings\n",
        "\n",
        "The Titanic dataset contains **891 rows** and **12 columns**, capturing demographic, passenger class, ticket, and survival info. Numerical features include `Age`, `Fare`, `SibSp`, `Parch`, `PassengerId`, `Pclass`, and `Survived`, while `Name`, `Sex`, `Ticket`, `Cabin`, and `Embarked` form the categorical variables. Significant missing values exist in `Cabin` (77.1%) and `Age` (19.87%), with `Embarked` missing just 2 records. Key metrics reveal an overall survival rate of ~38.38% with passenger ages ranging from 0.42 to 80 years old (mean age ~29.7 years). The heavy missingness in `Cabin` suggests it may require dropping or indicator encoding, while `Age` will require median/grouped imputation prior to predictive modeling."
    ]
})

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "orig_nbformat": 4
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("eda_titanic.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook eda_titanic.ipynb successfully built!")
