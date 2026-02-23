import pandas as pd
from pathlib import Path
import config as cfg

N_DECIMALS = 3
sigma_level = 1
                
if __name__ == "__main__":
    df = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "synthetic_results_wrong_copula_gauss.csv"))
    
    # Filter for a consistent sample size and number of features if needed
    df_filtered = df[(df['n_samples'] == 10000) & (df['n_features'] == 10)]

    # Group by copula, k_tau and compute mean and std
    grouped = df_filtered.groupby(['copula_name', 'k_tau'])
    mean_errors = grouped.mean().reset_index()
    std_errors = grouped.std().reset_index()

    # Define the metrics to tabulate
    metrics = [
        ('ci_harrell_error', 'ci_dep_ipcw_error', 'CI-Harrell'),
        ('ci_uno_error', 'ci_dep_ipcw_error', 'CI-Uno'),
        ('ibs_ipcw_error', 'ibs_dep_bg_error', 'IBS'),
        ('mae_margin_error', 'mae_dep_bg_error', 'MAE')
    ]

    # Create tables for each copula and metric
    tables = []
    for copula in ['clayton']:
        for ind_col, dep_col, metric_name in metrics:
            mean_subset = mean_errors[mean_errors['copula_name'] == copula]
            std_subset = std_errors[std_errors['copula_name'] == copula]

            table = pd.DataFrame()
            table["Kendall's τ"] = mean_subset["k_tau"]
            table["Independent Metric (mean ± std)"] = mean_subset[ind_col].round(3).astype(str) + " ± " + std_subset[ind_col].round(3).astype(str)
            table["Dependent Metric (mean ± std)"] = mean_subset[dep_col].round(3).astype(str) + " ± " + std_subset[dep_col].round(3).astype(str)
            table.insert(0, 'Metric', metric_name)
            table.insert(1, 'Copula', copula.capitalize())
            tables.append(table)

    # Concatenate all into a single table
    final_table = pd.concat(tables, ignore_index=True)
    
    print(final_table)