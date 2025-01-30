import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg

N_DECIMALS = 2

def get_dataset_info(dataset_name):
    return {
        "whas": (215, "9.3"),
        "gbsg": (299, "4.0"),
        "seer_prostate": (1011, "16.4"),
        "metabric": (1102, "3.8"),
        "nacd": (1497, "4.7"),
        "seer_breast": (1685, "9.4"),
        "mimic": (3301, "11.2"),
        "seer_stomach": (5311, "16.4"),
        "seer_liver": (5557, "15.9"),
        "seer_brain": (5624, "12.1"),
        "support": (6036, "3.1")
    }.get(dataset_name, dataset_name)

def map_strategy_name(strategy):
    return {
        "original": "Original",
        "top_5": "Top 5",
        "top_10": "Top 10",
        "random_25": "Rand. 25\\%"
    }.get(strategy, strategy)
    
def map_dataset_name(dataset_name):
    return {
        "gbsg": "GBSG",
        "metabric": "METABRIC",
        "mimic": "MIMIC-IV",
        "nacd": "NACD",
        "support": "SUPPORT",
        "whas": "WHAS",
        "seer_brain": "SEER-brain",
        "seer_breast": "SEER-breast",
        "seer_liver": "SEER-liver",
        "seer_prostate": "SEER-prostate",
        "seer_stomach": "SEER-stomach",
    }.get(dataset_name, dataset_name)
                
def calculate_errors(results, dataset, strategy, model_names, metrics):
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
    metrics = ["CIHarrell", "CIUno", "CIDepIPCW", "IBSIPCW", "IBSDepBG",
               "MAEUncens", "MAEHinge", "MAEMargin" ,"MAEPseudo", "MAEDepBG"]
    
    # Scale metrics by percentage
    cols_to_scale = ["CITrue", "CIHarrell", "CIUno", "CIDepIPCW", "IBSTrue", "IBSIPCW", "IBSDepBG"]
    results[cols_to_scale] = results[cols_to_scale] * 100

    datasets = ["gbsg", "metabric", "mimic", "nacd", "support",
                "seer_brain", "seer_breast", "seer_liver", "seer_prostate", "seer_stomach"]
    strategies = ["original", "top_5", "top_10", "random_25"]
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

for idx, dataset in enumerate(datasets):
    n_samples, censoring_rate = get_dataset_info(dataset)
    print(r"\multirow{4}{*}{\makecell{" + f"{map_dataset_name(dataset)} \\\ ($N$={n_samples}, $C$={censoring_rate}\%)" + r"}}")
    for strategy in strategies:
        mean_errors, std_errors = calculate_errors(results, dataset, strategy, model_names, metrics)

        # Format for printing
        formatted_errors = {
            k: f"%.{N_DECIMALS}f" % round(v, N_DECIMALS)
            for k, v in mean_errors.items()
        }
        formatted_std_errors = {
            k: f"%.{N_DECIMALS}f" % round(v, N_DECIMALS)
            for k, v in std_errors.items()
        }
        
        # Construct the text with mean and std errors
        text = f"& {map_strategy_name(strategy)}" + \
            "".join(f" & {formatted_errors[metric]}$\pm$\\scriptsize" + r"{" + f"{formatted_std_errors[metric]}" + r"}" for metric in mean_errors) + " \\\\"
        print(text)
    
    if idx != len(datasets) - 1:
        print(r"\cmidrule(lr){1-1}")
