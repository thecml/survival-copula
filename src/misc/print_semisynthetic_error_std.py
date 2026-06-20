from pathlib import Path

import numpy as np
import pandas as pd

import config as cfg

N_DECIMALS = 3
SIGMA_LEVEL = 1

STRATEGIES = [
    ("original", "Original"),
    ("top_5", "Top-5"),
    ("top_10", "Top-10"),
    ("random_25", r"Rand.\ 25\%"),
]

METRICS = [
    ("IBSIPCW", "IPCW (KM)"),
    ("IBSIPCW_CoxPH", "IPCW (CoxPH)"),
    ("IBSIndepBG", "Dep (KM)"),
    ("IBSDepBG", "Dep (CG)"),
    ("IBSDepBGUW", r"Dep (CG)$^{\text{UW}}$"),
]

DATASET_PANELS = [
    ["whas", "metabric", "churn", "gbsg", "nacd", "flchain"],
    ["support", "employee", "mimic_all", "seer_brain", "seer_liver", "seer_stomach"],
]

MODEL_NAMES = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

def calculate_error(results, dataset, strategy, metric, model_names):
    """Return the mean error and mean within-model SD used by the old script."""
    model_means = []
    model_stds = []

    for model_name in model_names:
        mask = (
            (results["Dataset"] == dataset)
            & (results["Strategy"] == strategy)
            & (results["ModelName"] == model_name)
        )

        true_values = results.loc[mask, "IBSTrue"].to_numpy(dtype=float)
        predicted_values = results.loc[mask, metric].to_numpy(dtype=float)

        if len(true_values) != len(predicted_values):
            raise ValueError(
                f"Length mismatch for dataset={dataset}, strategy={strategy}, "
                f"model={model_name}, metric={metric}: "
                f"{len(true_values)} true values vs {len(predicted_values)} predictions"
            )

        if len(true_values) == 0:
            continue

        errors = np.abs(true_values - predicted_values)
        model_means.append(np.mean(errors))
        model_stds.append(SIGMA_LEVEL * np.std(errors))

    if not model_means:
        return np.nan, np.nan

    return np.mean(model_means), np.mean(model_stds)

def fmt_pm(mean, std, decimals=N_DECIMALS):
    if np.isnan(mean) or np.isnan(std):
        return "--"
    return rf"{mean:.{decimals}f} $\pm$ {std:.{decimals}f}"

def print_panel(results, datasets):
    """Print one directly pasteable LaTeX tabular body for six datasets."""
    n_metric_rows = len(METRICS)

    for strategy_index, (strategy_key, strategy_label) in enumerate(STRATEGIES):
        for metric_index, (metric_key, metric_label) in enumerate(METRICS):
            values = []
            for dataset in datasets:
                mean, std = calculate_error(
                    results=results,
                    dataset=dataset,
                    strategy=strategy_key,
                    metric=metric_key,
                    model_names=MODEL_NAMES,
                )
                values.append(fmt_pm(mean, std))

            if metric_index == 0:
                first_cell = rf"\multirow{{{n_metric_rows}}}{{*}}{{{strategy_label}}}"
            else:
                first_cell = ""

            print(
                f"{first_cell} & {metric_label} & "
                + " & ".join(values)
                + r" \\" 
            )

        print(r"\bottomrule" if strategy_index == len(STRATEGIES) - 1 else r"\midrule")

def main():
    results_path = Path(cfg.RESULTS_DIR) / "semisynthetic_results.csv"
    results = pd.read_csv(results_path)

    required_columns = {"Dataset", "Strategy", "ModelName", "IBSTrue"}
    required_columns.update(metric for metric, _ in METRICS)
    missing = sorted(required_columns.difference(results.columns))
    if missing:
        raise KeyError(f"Missing required columns in {results_path}: {missing}")

    for panel_index, datasets in enumerate(DATASET_PANELS, start=1):
        print(f"% ===== Subtable {panel_index} =====")
        print_panel(results, datasets)
        if panel_index < len(DATASET_PANELS):
            print()

if __name__ == "__main__":
    main()
