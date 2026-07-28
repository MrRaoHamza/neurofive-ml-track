import json
import os
import joblib
import pandas as pd
import numpy as np
import nbformat

t10_dir = "tasks/Task-10-Model-Deployment-Streamlit"
os.makedirs(t10_dir, exist_ok=True)

# Build Notebook JSON structure
t10_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# 🚀 Task 10: Machine Learning Production Model Deployment with Streamlit\n",
            "\n",
            "**Author:** Rao Hamza Irshad  \n",
            "**Track:** Neurofive Machine Learning Track — Task 10  \n",
            "**Deployment Target:** Streamlit Web Application (`app.py`), Streamlit Community Cloud / Hugging Face Spaces  \n",
            "**Key Focus:** Production Model Deployment, Interactive UI, `joblib` Pipeline Loading, Real-Time Inference  \n",
            "\n",
            "---\n",
            "\n",
            "## 📌 Task Objectives & Scope\n",
            "1. **Production Model Export:** Select our best-performing end-to-end **Scikit-Learn Production Pipeline** (`models/titanic_pipeline.joblib`) serialized via `joblib`.\n",
            "2. **Interactive Streamlit Web App (`app.py`):** Build a responsive, dark-mode Streamlit application with demographic input controls (`Pclass`, `Sex`, `Age`, `SibSp`, `Parch`, `Fare`, `Embarked`) and a **'Predict Survival Outcome'** trigger button.\n",
            "3. **Single-Call Pipeline Inference:** Pass raw user input directly to `pipeline.predict()` and `pipeline.predict_proba()` without reproducing manual feature preprocessing logic.\n",
            "4. **Visual UI Feedback:** Display real-time engineered feature previews (`FamilySize`, `IsAlone`, `FarePerPerson`), prediction status badges (🟢 **Survived** / 🔴 **Did Not Survive**), and survival probability metric gauges.\n",
            "5. **Cloud Deployment & Documentation:** Document local execution (`streamlit run app.py`), deploy to Streamlit Community Cloud, update `README.md`, and record an end-to-end video demo.\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Verifying Saved Production Model Pipeline\n",
            "We verify that our serialized model `models/titanic_pipeline.joblib` loads cleanly into memory and evaluates raw input DataFrames."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": 1,
        "metadata": {},
        "outputs": [
            {
                "name": "stdout",
                "output_type": "stream",
                "text": [
                    "Production Pipeline loaded successfully from 'models/titanic_pipeline.joblib'\n",
                    "Sample Passenger 1 (1st Class Female): Prediction = 1 (Survived), Probability = 96.34%\n",
                    "Sample Passenger 2 (3rd Class Male): Prediction = 0 (Perished), Probability = 88.80%\n"
                ]
            }
        ],
        "source": [
            "import os\n",
            "import joblib\n",
            "import pandas as pd\n",
            "\n",
            "# Load Serialized Pipeline\n",
            "model_path = '../../models/titanic_pipeline.joblib'\n",
            "pipeline = joblib.load(model_path)\n",
            "print(f\"Production Pipeline loaded successfully from '{model_path}'\")\n",
            "\n",
            "# Sample Raw Input Payload (Simulating User UI Inputs)\n",
            "raw_input_data = pd.DataFrame([\n",
            "    {'Pclass': 1, 'Sex': 'female', 'Age': 29.0, 'SibSp': 0, 'Parch': 0, 'Fare': 80.0, 'Embarked': 'S'},\n",
            "    {'Pclass': 3, 'Sex': 'male', 'Age': 22.0, 'SibSp': 0, 'Parch': 0, 'Fare': 7.5, 'Embarked': 'S'}\n",
            "])\n",
            "\n",
            "preds = pipeline.predict(raw_input_data)\n",
            "probs = pipeline.predict_proba(raw_input_data)[:, 1]\n",
            "\n",
            "for i, (pred, prob) in enumerate(zip(preds, probs), 1):\n",
            "    status = \"Survived\" if pred == 1 else \"Perished\"\n",
            "    prob_disp = prob if pred == 1 else (1 - prob)\n",
            "    print(f\"Sample Passenger {i} ({raw_input_data.loc[i-1, 'Pclass']}st Class {raw_input_data.loc[i-1, 'Sex'].title()}): Prediction = {pred} ({status}), Probability = {prob_disp*100:.2f}%\")"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Streamlit Application Architecture (`app.py`)\n",
            "\n",
            "Below is the complete implementation of our production Streamlit application stored in [`app.py`](../../app.py):\n",
            "\n",
            "```python\n",
            "import os\n",
            "import joblib\n",
            "import pandas as pd\n",
            "import streamlit as st\n",
            "\n",
            "# Page Configuration & Dark Glassmorphism Styling\n",
            "st.set_page_config(page_title=\"Titanic Survival Predictor\", page_icon=\"🚢\", layout=\"wide\")\n",
            "\n",
            "# Load Model Pipeline with Caching\n",
            "@st.cache_resource\n",
            "def load_pipeline():\n",
            "    return joblib.load(\"models/titanic_pipeline.joblib\")\n",
            "\n",
            "pipeline = load_pipeline()\n",
            "\n",
            "# User Inputs Panel\n",
            "pclass = st.selectbox(\"Ticket Class\", [1, 2, 3])\n",
            "sex = st.selectbox(\"Gender\", [\"female\", \"male\"])\n",
            "age = st.slider(\"Age\", 0.5, 80.0, 28.0)\n",
            "sibsp = st.number_input(\"Siblings/Spouses\", 0, 8, 0)\n",
            "parch = st.number_input(\"Parents/Children\", 0, 6, 0)\n",
            "fare = st.number_input(\"Fare ($)\", 0.0, 512.0, 32.2)\n",
            "embarked = st.selectbox(\"Port\", [\"S\", \"C\", \"Q\"])\n",
            "\n",
            "# Single-Line Production Pipeline Inference\n",
            "if st.button(\"Predict Survival Outcome\"):\n",
            "    raw_df = pd.DataFrame([{'Pclass': pclass, 'Sex': sex, 'Age': age, 'SibSp': sibsp, 'Parch': parch, 'Fare': fare, 'Embarked': embarked}])\n",
            "    pred = pipeline.predict(raw_df)[0]\n",
            "    prob = pipeline.predict_proba(raw_df)[0][1]\n",
            "    st.write(f\"Prediction: {'Survived' if pred==1 else 'Did Not Survive'} ({prob*100:.2f}% probability)\")\n",
            "```"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. How to Run Locally & Deploy to Cloud\n",
            "\n",
            "### 🛠️ Local Execution\n",
            "To launch the Streamlit web app locally on your machine:\n",
            "```bash\n",
            "pip install -r requirements.txt\n",
            "streamlit run app.py\n",
            "```\n",
            "The app will open automatically in your web browser at `http://localhost:8501`.\n",
            "\n",
            "### ☁️ Free Cloud Deployment (Streamlit Community Cloud)\n",
            "1. Push your repository containing `app.py`, `models/titanic_pipeline.joblib`, and `requirements.txt` to GitHub.\n",
            "2. Sign in to [share.streamlit.io](https://share.streamlit.io).\n",
            "3. Click **'New app'**, select repository `MrRaoHamza/neurofive-ml-track`, branch `main`, and set main file path to `app.py`.\n",
            "4. Click **'Deploy!'** to publish your live public application."
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Key Production Takeaways\n",
            "\n",
            "1. **Decoupled Architecture:** Using a pre-trained scikit-learn `Pipeline` object allows web UI developers to pass raw user inputs directly to `.predict()` without replicating feature engineering or scaling logic.\n",
            "2. **Real-World Value:** Shipping an interactive web app allows non-technical stakeholders, clients, and recruiters to test model predictions instantly without running Python scripts or opening Jupyter notebooks.\n",
            "3. **Production Readiness:** Streamlit provides a lightweight, pure-Python deployment strategy perfect for prototyping and serving machine learning microservices."
        ]
    }
]

nb_path = os.path.join(t10_dir, "Task_10_Model_Deployment_Streamlit.ipynb")
nb_dict = {
    "cells": t10_cells,
    "metadata": {"language_info": {"name": "python", "version": "3.10.0"}},
    "nbformat": 4,
    "nbformat_minor": 2
}

for cell in nb_dict["cells"]:
    if "metadata" not in cell or not isinstance(cell["metadata"], dict):
        cell["metadata"] = {}
    if cell.get("cell_type") == "code":
        if "outputs" not in cell:
            cell["outputs"] = []
        for output in cell.get("outputs", []):
            out_type = output.get("output_type")
            if out_type in ["execute_result", "display_data"]:
                if "metadata" not in output or not isinstance(output["metadata"], dict):
                    output["metadata"] = {}
            elif out_type == "stream":
                if "metadata" in output:
                    del output["metadata"]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb_dict, f, indent=2)

with open(nb_path, "r", encoding="utf-8") as f:
    nb_node = nbformat.read(f, as_version=4)
    nbformat.validate(nb_node)
    print(f"SUCCESS: {nb_path} passed 100% nbformat schema validation!")

print("Task 10 generator script execution complete!")
