import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg
from utility.data import get_dataset_info, map_dataset_name, map_strategy_name
from utility.survival import theta_to_kendall_tau

N_DECIMALS = 3
SIGMA_LEVEL = 1

def percent_improvement(ipcw_vals, variant_vals):
    vals = []
    for b, v in zip(ipcw_vals, variant_vals):
        if np.isnan(b) or np.isnan(v):
            continue
        vals.append((b - v) / b * 100.0)
    if len(vals) == 0:
        return np.nan
    return np.mean(vals)
                
def calculate_errors(results, dataset, strategy, model_names, metrics):
    true_metrics = {f"{metric}True": metric for metric in ["IBS"]}
    
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
                std_errors[metric].append(SIGMA_LEVEL * np.std(errors))
    
    # Aggregate mean/std errors across models
    mean_errors = {k: np.nanmean(v) for k, v in mean_errors.items()}
    std_errors = {k: np.nanmean(v) for k, v in std_errors.items()}

    return mean_errors, std_errors

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))
    
    strategy_names = {
        "original": "Original",
        "top_5": "Top-5",
        "top_10": "Top-10",
        "random_25": "Rand. 25\\%",
    }
    
    metric_variants = [
        ("IBSIPCW", "IPCW"),
        ("IBSIndepBG", "Dep (KM)"),
        ("IBSDepBG", "Dep (CG)"),
        ("IBSDepBGUW", r"Dep (CG)$^{\text{UW}}$"),
    ]

    datasets = [
        "whas",
        "metabric",
        "churn",
        "gbsg",
        "nacd",
        "flchain",
        "support",
        "employee",
        "mimic_all",
        "seer_brain",
        "seer_liver",
        "seer_stomach",
    ]
    
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "weibullaft"]

    for strategy_i, (strategy_key, strategy_label) in enumerate(strategy_names.items()):
        print(r"\multirow{4}{*}{\rotatebox{90}{" + strategy_label + r"}}")

        ipcw_values = None  # will hold the per-dataset baseline for this strategy

        # Print rows for each metric variant
        for metric_key, metric_label in metric_variants:

            row_entries = [metric_label]
            current_values = []   # values across datasets for this method/strategy

            # Collect numbers across datasets
            for dataset in datasets:
                mean_errors, std_errors = calculate_errors(
                    results, dataset, strategy_key, model_names, [metric_key]
                )

                m = mean_errors[metric_key]
                s = std_errors[metric_key]

                current_values.append(m)
                row_entries.append(f"{m:.3f}")

            # Average column
            avg_m = np.nanmean(current_values)
            row_entries.append(f"{avg_m:.3f}")

            # Improvement vs IPCW
            if metric_key == "IBSIPCW":
                # Baseline row: store baseline values, no improvement text
                ipcw_values = current_values
                improvement_text = ""
            else:
                # Compute improvement using Average column only
                avg_ipcw = np.nanmean(ipcw_values)
                avg_impr = (avg_m - avg_ipcw) / avg_ipcw * 100.0

                sign = "+" if avg_impr >= 0 else ""
                color = "improvRed" if avg_impr >= 0 else "improvGreen"
                improvement_text = rf" \textcolor{{{color}}}{{({sign}{avg_impr:.1f}\%)}}"

            # Print row
            print("& " + " & ".join(row_entries) + improvement_text + r" \\")

        # Midrule between strategy blocks
        if strategy_i < len(strategy_names) - 1:
            print(r"\midrule")
        else:
            print(r"\bottomrule")