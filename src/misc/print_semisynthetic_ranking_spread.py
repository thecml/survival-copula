from pathlib import Path
import numpy as np
import pandas as pd
import config as cfg

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))

    ranking_metrics = [
        ("IBSIPCW", "IPCW"),
        ("IBSIndepBG", "Dep (KM)"),
        ("IBSDepBG", "Dep (CG)"),
        ("IBSDepBGUW", r"Dep (CG)$^{\text{UW}}$"),
    ]

    datasets = [
        "whas", "metabric", "churn", "gbsg", "nacd", "flchain",
        "support", "employee", "mimic_all", "seer_brain",
        "seer_liver", "seer_stomach"
    ]

    strategies = ["original", "top_5", "top_10", "random_25"]
    seeds = list(range(10))
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

    def topk_match_set(true_vals, metric_vals, k=3):
        """Set-based top-k matching (order ignored)."""
        true_rank = np.argsort(true_vals)
        metric_rank = np.argsort(metric_vals)
        return set(true_rank[:k]) == set(metric_rank[:k])

    for metric_key, metric_label in ranking_metrics:

        row = [metric_label]

        for dataset in datasets:

            weighted_score = 0.0
            total_spread = 0.0

            for strategy in strategies:
                for seed in seeds:

                    df = results.loc[
                        (results["Dataset"] == dataset) &
                        (results["Strategy"] == strategy) &
                        (results["Seed"] == seed)
                    ].sort_values("ModelName")

                    if len(df) != 5:
                        continue

                    true_vals = df["IBSTrue"].values
                    metric_vals = df[metric_key].values

                    # Spread of true IBS across models (ranking difficulty)
                    spread = float(np.max(true_vals) - np.min(true_vals))

                    # Accumulate total spread for normalization
                    total_spread += spread

                    # If correct top-3, add spread-weighted credit
                    if topk_match_set(true_vals, metric_vals, k=3):
                        weighted_score += spread

            # Normalize so the maximum possible score is 1.0
            if total_spread > 0:
                normalized_score = weighted_score / total_spread
            else:
                normalized_score = 0.0

            row.append(f"{normalized_score:.3f}")

        print(" & ".join(row) + r" \\")
