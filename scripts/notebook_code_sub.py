import os
import re
import nbformat
#from nbformat.reader import NotebookReader
from nbformat import read
from nbformat.v4.nbbase import new_code_cell, new_markdown_cell

def substitute_code_in_notebook(notebook_path, pattern, replacement):
    """Reads a Jupyter Notebook, replaces code matching a pattern, and saves it.

    Args:
        notebook_path (str): Path to the notebook file.
        pattern (str): Regular expression pattern to match code to be replaced.
        replacement (str): Replacement string.
    """

    with open(notebook_path, 'r') as f:
        nb = nbformat.read(f, as_version=4)

    for cell in nb.cells:
        if cell.cell_type == 'code':
            new_source = re.sub(pattern, replacement, cell.source)
            cell.source = new_source

    with open(notebook_path, 'w') as f:
        nbformat.write(nb, f)


models = [
    "linear_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "mlp",
    "transformer",
    "gru",
    "lstm",
    "bi-gru",
    "bi-lstm",
]

manufacturer = "204"
plant = "o"

root_dir = os.getcwd()

for model in models:
    print("============")
    print("Model:", model)
    print("============")
    os.chdir(root_dir)
    notebooks_path = "ccs28-ml-modelling/notebooks/modelling/" + manufacturer + "/" + model + "/" + plant + "/"
    notebooks = list(filter(lambda filename: filename.endswith(".ipynb"), os.listdir(notebooks_path)))
    os.chdir(notebooks_path)
    
    for notebook_path in notebooks:
        pattern = r'"Plant": "AQ"'  # Replace with your actual pattern
        replacement = '"Plant": "O"'  # Replace with your desired replacement
        substitute_code_in_notebook(notebook_path, pattern, replacement)
    
        pattern = r'204/aq\w*'  # Replace with your actual pattern
        replacement = '204/o'  # Replace with your desired replacement
        substitute_code_in_notebook(notebook_path, pattern, replacement)


        pattern = r'ch_features = \[\n\s*("MgO",\n\s*"Na2O",\n\s*"SO3",\n\s*"K2O",\n)\s*\]'

        # Replace with your actual pattern
        replacement = """ch_features = [
    "CaO",
    "MgO",
    "Na2O",
    "Al2O3",
    "SiO2",
    "SO3",
    "K2O",
    "Fe2O3",
]

df_copy["std_ch_feats"] = df_copy[ch_features].std(ddof=0, axis=1)

df_copy["ratio_CaO_to_SiO2"] = df_copy["CaO"] / df_copy["SiO2"]
df_copy["ratio_MgO_to_CaO"] = df_copy["MgO"] / df_copy["CaO"]
"""
        substitute_code_in_notebook(notebook_path, pattern, replacement)


        pattern = r'CEMENT_TYPES = \[(.*?)\]'

        #r"""CEMENT_TYPES = ["Cement_Type_CPII F40", "Cement_Type_CPV ARI"]\w*"""
        replacement = """CEMENT_TYPES = ["Cement_Type_CPII F40", "Cement_Type_Fibrocimento"]"""
        substitute_code_in_notebook(notebook_path, pattern, replacement)


# Example usage:
# notebook_path = 'my_notebook.ipynb'
# pattern = r'old_string_to_replace'  # Replace with your actual pattern
# replacement = 'new_string'  # Replace with your desired replacement
# substitute_code_in_notebook(notebook_path, pattern, replacement)
