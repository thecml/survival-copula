import pandas as pd
from pathlib import Path
import numpy as np
import config as cfg
from utility.data import get_dataset_info, map_dataset_name, map_strategy_name
from utility.survival import theta_to_kendall_tau

N_DECIMALS = 3
SIGMA_LEVEL = 1
                
def calculate_errors(results, dataset, strategy, model_names, metrics,
                     true_col="IBSTrue", abs_error=True, per_model_avg=True):
    # filter once for dataset+strategy
    mask = (results["Dataset"] == dataset) & (results["Strategy"] == strategy)
    df = results.loc[mask].copy()
    if df.shape[0] == 0:
        return ({m: np.nan for m in metrics}, {m: np.nan for m in metrics})

    mean_errors = {}
    std_errors = {}

    for metric in metrics:
        if metric not in df.columns or true_col not in df.columns:
            mean_errors[metric] = np.nan
            std_errors[metric] = np.nan
            continue

        # keep only rows where both metric and true_col are non-null
        sub = df.loc[df[metric].notnull() & df[true_col].notnull(),
                     ["ModelName", metric, true_col]]
        if sub.shape[0] == 0:
            mean_errors[metric] = np.nan
            std_errors[metric] = np.nan
            continue

        # compute per-row differences
        diffs = (sub[true_col] - sub[metric]).values
        if abs_error:
            diffs = np.abs(diffs)

        if per_model_avg:
            # compute mean error per model, then average across the requested model_names
            per_model = sub.groupby("ModelName").apply(lambda g: np.mean(np.abs(g[true_col] - g[metric])
                                                                         if abs_error else (g[true_col] - g[metric])))
            # select only models in model_names (if present)
            per_model = per_model.reindex(model_names).dropna()
            if per_model.shape[0] == 0:
                mean_errors[metric] = np.nan
                std_errors[metric] = np.nan
            else:
                mean_errors[metric] = float(per_model.mean())
                std_errors[metric] = float(per_model.std(ddof=0))
        else:
            # pooled across all rows
            mean_errors[metric] = float(np.mean(diffs))
            std_errors[metric] = float(np.std(diffs, ddof=0))

    return mean_errors, std_errors

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))
    
    metrics = ["IBSUncensored", "IBSIPCW", "IBSIndepBGUW", "IBSDepBG", "IBSDepBGUW"]
    
    datasets = [
        "metabric",
        "mimic_all",
        "mimic_hospital",
        "seer_brain",
        "seer_liver",
        "seer_stomach",
    ]
    strategies = ["original", "top_1", "top_5", "top_10", "random_25"]
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
