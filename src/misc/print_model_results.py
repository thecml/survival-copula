import pandas as pd
from pathlib import Path
import glob
import os
import config as cfg
import numpy as np

N_DECIMALS = 2
ALPHA = 0.05

def map_strategy_name(strategy):
    if strategy == "original":
        return "Original"
    elif strategy == "top_5":
        return "Top 5"
    elif strategy == "top_10":
        return "Top 10"
    elif strategy == "random_25":
        return "Random 25\\%"

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "dependent.csv"))
    
    cols_to_scale = ["CITrue", "IBSTrue", "CI", "CIDep", "IBS", "IBSDep"]
    results[cols_to_scale] = results[cols_to_scale] * 100
    
    datasets = ["gbsg", "metabric", "whas", "seer_breast", "seer_liver", "seer_prostate", "seer_stomach"]
    strategies = ["original", "top_5", "top_10", "random_25"]
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]
    
    for dataset in datasets:
        for strategy in strategies:
            text = ""
            mean_harrell_ci_error, mean_ci_dep_error, mean_ibs_error, mean_ibs_dep_error, mean_mae_margin_error, mean_mae_dep_error = list(), list(), list(), list(), list(), list()
            std_harrell_ci_error, std_ci_dep_error, std_ibs_error, std_ibs_dep_error, std_mae_margin_error, std_mae_dep_error = list(), list(), list(), list(), list(), list()
            for model_name in model_names:
                # Get true metrics
                true_ci = results.loc[(results['Dataset'] == dataset)
                                       & (results['Strategy'] == strategy)
                                       & (results['ModelName'] == model_name)]['CITrue']
                true_ibs = results.loc[(results['Dataset'] == dataset)
                                       & (results['Strategy'] == strategy)
                                       & (results['ModelName'] == model_name)]['IBSTrue']
                true_mae = results.loc[(results['Dataset'] == dataset)
                                       & (results['Strategy'] == strategy)
                                       & (results['ModelName'] == model_name)]['MAETrue']
                 
                # Get model metrics
                harrell_ci = results.loc[(results['Dataset'] == dataset)
                                         & (results['Strategy'] == strategy)
                                         & (results['ModelName'] == model_name)]['CI']
                ci_dep = results.loc[(results['Dataset'] == dataset)
                                     & (results['Strategy'] == strategy)
                                     & (results['ModelName'] == model_name)]['CIDep']
                ibs = results.loc[(results['Dataset'] == dataset)
                                     & (results['Strategy'] == strategy)
                                     & (results['ModelName'] == model_name)]['IBS']
                ibs_dep = results.loc[(results['Dataset'] == dataset)
                                     & (results['Strategy'] == strategy)
                                     & (results['ModelName'] == model_name)]['IBSDep']
                mae_margin = results.loc[(results['Dataset'] == dataset)
                                     & (results['Strategy'] == strategy)
                                     & (results['ModelName'] == model_name)]['MAEMargin']
                mae_dep = results.loc[(results['Dataset'] == dataset)
                                     & (results['Strategy'] == strategy)
                                     & (results['ModelName'] == model_name)]['MAEDep']
                
                # Calculate mean model error
                mean_harrell_ci_error.append(np.mean(abs(true_ci - harrell_ci)))
                mean_ci_dep_error.append(np.mean(abs(true_ci - ci_dep)))
                mean_ibs_error.append(np.mean(abs(true_ibs - ibs)))
                mean_ibs_dep_error.append(np.mean(abs(true_ibs - ibs_dep)))
                mean_mae_margin_error.append(np.mean(abs(true_mae - mae_margin)))
                mean_mae_dep_error.append(np.mean(abs(true_mae - mae_dep)))
                
                # Calculate std model error
                std_harrell_ci_error.append(np.std(abs(true_ci - harrell_ci)))
                std_ci_dep_error.append(np.std(abs(true_ci - ci_dep)))
                std_ibs_error.append(np.std(abs(true_ibs - ibs)))
                std_ibs_dep_error.append(np.std(abs(true_ibs - ibs_dep)))
                std_mae_margin_error.append(np.std(abs(true_mae - mae_margin)))
                std_mae_dep_error.append(np.std(abs(true_mae - mae_dep)))
                 
            mean_harrell_ci_error = np.mean(mean_harrell_ci_error)
            mean_ci_dep_error = np.mean(mean_ci_dep_error)
            mean_ibs_error = np.mean(mean_ibs_error)
            mean_ibs_dep_error = np.mean(mean_ibs_dep_error)
            mean_mae_margin_error = np.mean(mean_mae_margin_error)
            mean_mae_dep_error = np.mean(mean_mae_dep_error)
            
            std_harrell_ci_error = np.mean(std_harrell_ci_error)
            std_ci_dep_error = np.mean(std_ci_dep_error)
            std_ibs_error = np.mean(std_ibs_error)
            std_ibs_dep_error = np.mean(std_ibs_dep_error)
            std_mae_margin_error = np.mean(std_mae_margin_error)
            std_mae_dep_error = np.mean(std_mae_dep_error)
            
            # Format for printing
            mean_harrell_ci_error = f"%.{N_DECIMALS}f" % round(mean_harrell_ci_error, N_DECIMALS)
            mean_ci_dep_error = f"%.{N_DECIMALS}f" % round(mean_ci_dep_error, N_DECIMALS)
            mean_ibs_error = f"%.{N_DECIMALS}f" % round(mean_ibs_error, N_DECIMALS)
            mean_ibs_dep_error = f"%.{N_DECIMALS}f" % round(mean_ibs_dep_error, N_DECIMALS)
            mean_mae_margin_error = f"%.{N_DECIMALS}f" % round(mean_mae_margin_error, N_DECIMALS)
            mean_mae_dep_error = f"%.{N_DECIMALS}f" % round(mean_mae_dep_error, N_DECIMALS)
            
            text += f"& {map_strategy_name(strategy)}" + \
                    f" & {mean_harrell_ci_error}" + \
                    f" & {mean_ci_dep_error}" + \
                    f" & {mean_ibs_error}" + \
                    f" & {mean_ibs_dep_error}" + \
                    f" & {mean_mae_margin_error}" + \
                    f" & {mean_mae_dep_error} \\\\"
            print(text)
        print()