import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg
from utility.data import get_dataset_info, map_dataset_name, map_strategy_name
from utility.survival import theta_to_kendall_tau

N_DECIMALS = 3
SIGMA_LEVEL = 1
                
def calculate_errors(results, dataset, strategy, model_names, metrics):
    true_metrics = {f"{metric}True": metric for metric in ["CI", "IBS", "MAE"]}
    
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

        for metric in metrics:
            predicted_values = results.loc[
                (results["Dataset"] == dataset) &
                (results["Strategy"] == strategy) &
                (results["ModelName"] == model_name), metric
            ].values
            
            true_metric_key = next((k for k, v in true_metrics.items() if metric.startswith(v)), None)
            if true_metric_key:
                true_values_for_metric = true_values[true_metric_key]

                # only compute if both are non-empty and same length
                if len(true_values_for_metric) > 0 and len(predicted_values) > 0:
                    errors = abs(true_values_for_metric - predicted_values)
                    mean_errors[metric].append(np.mean(errors))
                    std_errors[metric].append(SIGMA_LEVEL * np.std(errors))
                else:
                    mean_errors[metric].append(np.nan)
                    std_errors[metric].append(np.nan)
    
    # Aggregate mean/std errors across models
    mean_errors = {k: np.nanmean(v) if len(v) > 0 else np.nan for k, v in mean_errors.items()}
    std_errors = {k: np.nanmean(v) if len(v) > 0 else np.nan for k, v in std_errors.items()}

    return mean_errors, std_errors

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))
    
    metrics = [
        "CIHarrell", "CIUno",
        "IBSIPCW", "MAEHinge", "MAEMargin",
        "CIIndepBG", "IBSIndepBG", "MAEIndepBG",
        "CIDepBG", "IBSDepBG", "MAEDepBG",
    ]
    
    datasets = [
        "metabric",
        "mimic_all",
        "mimic_hospital",
        "seer_brain",
        "seer_liver",
        "seer_stomach",
    ]
    strategies = ["original", "top_5", "random_25"]
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

    for dataset in datasets:
        n_samples, censoring_rate = get_dataset_info(dataset)
        print(f"\n=== Dataset: {map_dataset_name(dataset)} "
              f"(N={n_samples}, C={censoring_rate}%) ===")
        
        # header
        header = ["Metric"] + [map_strategy_name(s) for s in strategies]
        print("-" * (18 * (len(strategies) + 1)))
        print("".join(f"{h:<18}" for h in header))
        print("-" * (18 * (len(strategies) + 1)))
        
        # rows: one per metric
        for metric in metrics:
            row = [f"{metric:<18}"]
            for strategy in strategies:
                mean_errors, std_errors = calculate_errors(results, dataset, strategy, model_names, metrics)
                mean_val = round(mean_errors[metric], N_DECIMALS)
                std_val = round(std_errors[metric], N_DECIMALS)
                if np.isnan(mean_val) or np.isnan(std_val):
                    row.append("NA")
                else:
                    row.append(f"{round(mean_val, N_DECIMALS)} ± {round(std_val, N_DECIMALS)}")
            print("".join(f"{c:<18}" for c in row))
        
        print("-" * (18 * (len(strategies) + 1)))
