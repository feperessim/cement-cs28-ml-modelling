import os
import papermill as pm


def execute_notebook_at(root_dir, manufacturer, plant, models, notebooks_file_names=[]):
    for model in models:
        python_dir = os.path.join(
            os.path.join(os.path.join(root_dir, manufacturer), model), plant
        )
        os.chdir(python_dir)
        if notebooks_file_names == []:
            notebooks_file_names = list(
                filter(lambda filename: filename.endswith(".ipynb"), os.listdir())
            )

        for notebook_filename in notebooks_file_names:
            print("\n\n=======================================")
            print("Running notebook:", notebook_filename)
            print("=======================================\n\n")
            input_notebook_file = os.path.join(python_dir, notebook_filename)
            output_notebook_file = input_notebook_file
            pm.execute_notebook(input_notebook_file, output_notebook_file)


root_dir = "/home/peressim/projects/ccs28-ml-modelling/notebooks/modelling/"

manufacturer = ""
plant = ""
models = []
notebooks_file_names = []
print("\n\n=======================================")
print("Manufacturer:", manufacturer)
print("Plant:", plant)
print("=======================================\n\n")

execute_notebook_at(root_dir, manufacturer, plant, models, notebooks_file_names)
