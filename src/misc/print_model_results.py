import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg

N_DECIMALS = 2

def map_strategy_name(strategy):
    return {
        "original": "Original",
        "top_5": "Top 5",
        "top_10": "Top 10",
        "random_25": "Random 25\\%"
    }.get(strategy, strategy)

def calculate_errors(results, dataset, strategy, model_names):
    # Metrics to compute
    metrics = ["MAEUncens", "MAEHinge", "MAEMargin", "MAEIPCWV1", "MAEIPCWV2", "MAEDepBG", "MAEDepIPCW"]
    true_metrics = {f"{metric}True": metric for metric in ["CI", "IBS", "MAE"]}
    
    # Initialize error storage
    mean_errors = {metric: [] for metric in metrics}
    std_errors = {metric: [] for metric in metrics}

    for model_name in model_names:
        # Extract true metrics
        true_values = {
            true_metric: results.loc[
                (results["Dataset"] == dataset) &
                (results["Strategy"] == strategy) &
                (results["ModelName"] == model_name), true_metric
            ].values for true_metric in true_metrics
        }

        # Extract predicted metrics
        for metric in metrics:
            predicted_values = results.loc[
                (results["Dataset"] == dataset) &
                (results["Strategy"] == strategy) &
                (results["ModelName"] == model_name), metric
            ].values
            
            # Match true metric and calculate errors
            true_metric_key = next((k for k, v in true_metrics.items() if metric.startswith(v)), None)
            if true_metric_key:
                true_values_for_metric = true_values[true_metric_key]
                errors = abs(true_values_for_metric - predicted_values)
                mean_errors[metric].append(np.mean(errors))
                std_errors[metric].append(np.std(errors))
    
    # Aggregate mean/std errors across models
    mean_errors = {k: np.mean(v) for k, v in mean_errors.items()}
    std_errors = {k: np.mean(v) for k, v in std_errors.items()}

    return mean_errors, std_errors

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "dependent.csv"))

    # Scale metrics by percentage
    cols_to_scale = ["CITrue", "HarrellCI", "CIDepBG", "IBSTrue", "IBSIPCW", "IBSDepBG", "IBSDepIPCW"]
    results[cols_to_scale] = results[cols_to_scale] * 100

    datasets = ["gbsg", "metabric", "mimic", "nacd", "whas", "seer_brain",
                "seer_breast", "seer_liver", "seer_prostate", "seer_stomach"]
    strategies = ["original", "top_5", "top_10", "random_25"]
    model_names = ["coxph"]

    for dataset in datasets:
        print(dataset)
        for strategy in strategies:
            mean_errors, std_errors = calculate_errors(results, dataset, strategy, model_names)

            # Format for printing
            formatted_errors = {
                k: f"%.{N_DECIMALS}f" % round(v, N_DECIMALS)
                for k, v in mean_errors.items()
            }

            text = f"& {map_strategy_name(strategy)}" + \
                   "".join(f" & {formatted_errors[metric]}" for metric in mean_errors) + " \\\\"
            print(text)
        print()
