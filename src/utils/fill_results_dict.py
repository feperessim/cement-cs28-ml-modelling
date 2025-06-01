import numpy as np
import pandas as pd


def fill_results_dict(results_dict, scores):
    num_samples = len(scores["test_r2"])

    # Create a DataFrame by copying the results_dict for each sample
    df_results = pd.DataFrame([results_dict.copy()] * num_samples)

    # Define a mapping of metric names between the input scores and the DataFrame
    metric_mapping = {
        "train_neg_root_mean_squared_error": "RMSE Train",
        "test_neg_root_mean_squared_error": "RMSE Test",
        "train_neg_mean_absolute_error": "MAE Train",
        "test_neg_mean_absolute_error": "MAE Test",
        "train_neg_mean_absolute_percentage_error": "MAPE Train",
        "test_neg_mean_absolute_percentage_error": "MAPE Test",
        "train_r2": "R2 Train",
        "test_r2": "R2 Test",
    }

    # Iterate through the mapping and fill the DataFrame
    for score_key, df_column_name in metric_mapping.items():
        if score_key in scores:
            if "R2" not in df_column_name:
                df_results[df_column_name] = np.abs(scores[score_key])
            else:
                df_results[df_column_name] = scores[score_key]

    return df_results
