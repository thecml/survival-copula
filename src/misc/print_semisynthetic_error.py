import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg

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
        ("IBSIPCW", "IPCW (KM)"),
        ("IBSIPCW_CoxPH", "IPCW (CoxPH)"),
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

    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

    for strategy_i, (strategy_key, strategy_label) in enumerate(strategy_names.items()):
        print(r"\multirow{5}{*}{\rotatebox{90}{" + strategy_label + r"}}")

        # Calculate every metric × dataset value before printing so that
        # the minimum error can be bolded within each dataset and strategy.
        values = {}
        averages = {}

        for metric_key, _ in metric_variants:
            values[metric_key] = []

            for dataset in datasets:
                mean_errors, _ = calculate_errors(
                    results,
                    dataset,
                    strategy_key,
                    model_names,
                    [metric_key],
                )
                values[metric_key].append(mean_errors[metric_key])

            averages[metric_key] = np.nanmean(values[metric_key])

        # Minimum error in each dataset column. All numerical ties are bolded.
        value_matrix = np.asarray(
            [values[metric_key] for metric_key, _ in metric_variants],
            dtype=float,
        )
        best_by_dataset = np.nanmin(value_matrix, axis=0)

        ipcw_values = values["IBSIPCW"]
        avg_ipcw = averages["IBSIPCW"]

        for metric_key, metric_label in metric_variants:
            row_entries = [metric_label]

            for dataset_i, value in enumerate(values[metric_key]):
                formatted = f"{value:.3f}"
                if np.isclose(value, best_by_dataset[dataset_i]):
                    formatted = rf"\textbf{{{formatted}}}"
                row_entries.append(formatted)

            avg_m = averages[metric_key]
            row_entries.append(f"{avg_m:.3f}")

            if metric_key == "IBSIPCW":
                improvement_text = ""
            else:
                # Positive means more error than IPCW (worse);
                # negative means less error than IPCW (better).
                avg_impr = (avg_m - avg_ipcw) / avg_ipcw * 100.0
                sign = "+" if avg_impr >= 0 else ""
                color = "improvRed" if avg_impr >= 0 else "improvGreen"
                improvement_text = (
                    rf" \textcolor{{{color}}}"
                    rf"{{({sign}{avg_impr:.1f}\%)}}"
                )

            print("& " + " & ".join(row_entries) + improvement_text + r" \\")

        if strategy_i < len(strategy_names) - 1:
            print(r"\midrule")
        else:
            print(r"\bottomrule")
