import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg

N_DECIMALS = 3
SIGMA_LEVEL = 1

def calculate_errors(results, dataset, strategy, model_names, metrics):
    true_metrics = {f"{metric}True": metric for metric in ["IBS"]}

    mean_errors = {metric: [] for metric in metrics}
    std_errors = {metric: [] for metric in metrics}

    for model_name in model_names:
        true_values = {
            true_metric: results.loc[
                (results["Dataset"] == dataset) &
                (results["Strategy"] == strategy) &
                (results["ModelName"] == model_name),
                true_metric
            ].values
            for true_metric in true_metrics
        }

        for metric in metrics:
            predicted_values = results.loc[
                (results["Dataset"] == dataset) &
                (results["Strategy"] == strategy) &
                (results["ModelName"] == model_name),
                metric
            ].values

            true_metric_key = next((k for k, v in true_metrics.items() if metric.startswith(v)), None)
            if true_metric_key:
                tv = true_values[true_metric_key]
                errors = np.abs(tv - predicted_values)
                mean_errors[metric].append(np.mean(errors))
                std_errors[metric].append(SIGMA_LEVEL * np.std(errors))

    mean_errors = {k: np.nanmean(v) for k, v in mean_errors.items()}
    std_errors = {k: np.nanmean(v) for k, v in std_errors.items()}

    return mean_errors, std_errors


def fmt_pm(m, s, decimals=3):
    """Format mean ± std for LaTeX."""
    if np.isnan(m) or np.isnan(s):
        return r"--"
    return rf"{m:.{decimals}f} $\pm$ {s:.{decimals}f}"

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))

    strategy_names = {
        "original": "Original",
        "top_5": "Top-5",
        "top_10": "Top-10",
        "random_25": r"Rand. 25\%",
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

    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

    for strategy_i, (strategy_key, strategy_label) in enumerate(strategy_names.items()):
        print(r"\multirow{4}{*}{\rotatebox{90}{" + strategy_label + r"}}")

        for metric_key, metric_label in metric_variants:
            row_entries = [metric_label]
            means = []

            for dataset in datasets:
                mean_errors, std_errors = calculate_errors(
                    results, dataset, strategy_key, model_names, [metric_key]
                )
                m = mean_errors[metric_key]
                s = std_errors[metric_key]

                means.append(m)
                row_entries.append(fmt_pm(m, s, decimals=N_DECIMALS))

            print("& " + " & ".join(row_entries) + r" \\")

        if strategy_i < len(strategy_names) - 1:
            print(r"\midrule")
        else:
            print(r"\bottomrule")
