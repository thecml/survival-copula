from pathlib import Path
import numpy as np
import pandas as pd
import config as cfg

# ======================================================
# CONFIG
# ======================================================
strategies = ["original", "top_5", "top_10", "random_25"]
datasets = [
    "whas","metabric","churn","gbsg","nacd","flchain",
    "support","employee","mimic_all","seer_brain",
    "seer_liver","seer_stomach"
]
metrics_to_eval = {
    "IBSIPCW"     : "IPCW",
    "IBSIndepBG"  : "Dep (KM)",
    "IBSDepBG"    : "Dep (CG)",
    "IBSDepBGUW"  : "Dep (CG)^{UW}",
}
seeds = list(range(10))
model_names = ["coxph","gbsa","rsf","deepsurv","mtlr"]

def top3_correct(true_vals, metric_vals):
    """Set-based top-3 match (order does not matter)."""
    true_rank   = np.argsort(true_vals)
    metric_rank = np.argsort(metric_vals)
    return int(set(true_rank[:3]) == set(metric_rank[:3]))

# ======================================================
# LOAD RESULTS
# ======================================================
results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))

# ======================================================
# COLLECT DIAGNOSTIC RESULTS
# ======================================================
records = []

for dataset in datasets:
    for strategy in strategies:
        for seed in seeds:

            df = results.loc[
                (results["Dataset"] == dataset)
                & (results["Strategy"] == strategy)
                & (results["Seed"] == seed)
            ].sort_values("ModelName")

            if len(df) != 5:
                continue

            true_vals = df["IBSTrue"].values
            spread = np.max(true_vals) - np.min(true_vals)   # ranking difficulty

            rec = {
                "Dataset": dataset,
                "Strategy": strategy,
                "Seed": seed,
                "Spread_IBSTrue": spread,
            }

            # Evaluate metrics
            for metric_key, label in metrics_to_eval.items():
                metric_vals = df[metric_key].values
                rec[f"Top3_{metric_key}"] = top3_correct(true_vals, metric_vals)

            records.append(rec)

# ======================================================
# BUILD FINAL DF
# ======================================================
diagnostics = pd.DataFrame(records)

# Save optional CSV (useful for plotting)
diagnostics.to_csv("ranking_difficulty_analysis.csv", index=False)

# ======================================================
# SUMMARY PRINT
# ======================================================
print("\n=== Ranking Difficulty Summary ===")
summary = diagnostics.groupby("Dataset")["Spread_IBSTrue"].agg(["mean", "min", "max"])
print(summary)

print("\n=== Top-3 Correct vs Spread (Correlation) ===")
for metric_key in metrics_to_eval.keys():
    corr = diagnostics["Spread_IBSTrue"].corr(diagnostics[f"Top3_{metric_key}"])
    print(f"{metric_key}: corr = {corr:.3f}")

print("\n=== Per-dataset Top-3 Correct (IPCW vs DepKM vs DepBG vs DepBGUW) ===")
table = diagnostics.groupby("Dataset")[
    ["Top3_IBSIPCW", "Top3_IBSIndepBG", "Top3_IBSDepBG", "Top3_IBSDepBGUW"]
].sum()
print(table)
