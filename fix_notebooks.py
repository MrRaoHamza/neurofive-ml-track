import json
import os
import nbformat

def fix_and_validate_notebook(filepath):
    print(f"Fixing notebook: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Root metadata
    if "metadata" not in nb or not isinstance(nb["metadata"], dict):
        nb["metadata"] = {}
    
    nb["metadata"]["language_info"] = {"name": "python", "version": "3.10.0"}
    nb["nbformat"] = 4
    nb["nbformat_minor"] = 2

    # 2. Cells
    for cell in nb.get("cells", []):
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

    # Save fixed JSON
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)

    # Validate against nbformat v4 official JSON schema
    with open(filepath, "r", encoding="utf-8") as f:
        nb_node = nbformat.read(f, as_version=4)
        nbformat.validate(nb_node)
        print(f"SUCCESS: {filepath} passed 100% nbformat validation!")

# Fix both task notebooks
fix_and_validate_notebook("tasks/Task-01-Baseline-EDA/Task_01_Titanic_EDA.ipynb")
fix_and_validate_notebook("tasks/Task-02-Cleaning-and-Visualization/Task_02_Data_Cleaning_Visualizations.ipynb")

print("All notebooks successfully fixed and validated!")
