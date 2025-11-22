from pathlib import Path
import numpy as np
import pandas as pd
import config as cfg

if __name__ == "__main__":
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))

    # Metrics to evaluate
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

    # Use all strategies for the combined total (40 runs)
    strategies = ["original", "top_5", "top_10", "random_25"]

    seeds = list(range(10))
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

    def topk_correct(true_vals, metric_vals, k=3):
        """Return True if the *set* of top-k models matches (order ignored)."""
        true_rank = np.argsort(true_vals)
        metric_rank = np.argsort(metric_vals)
        return set(true_rank[:k]) == set(metric_rank[:k])

    # ==== Build the LaTeX table rows ====
    for metric_key, metric_label in ranking_metrics:

        row = [metric_label]

        for dataset in datasets:

            correct = 0
            total = 0

            for strategy in strategies:
                for seed in seeds:

                    df = results.loc[
                        (results["Dataset"] == dataset)
                        & (results["Strategy"] == strategy)
                        & (results["Seed"] == seed)
                    ].sort_values("ModelName")

                    if len(df) != 5:
                        continue  # skip incomplete runs

                    # Extract values
                    true_vals = df["IBSTrue"].values
                    metric_vals = df[metric_key].values

                    # Compare ranking sets
                    correct += int(topk_correct(true_vals, metric_vals, k=3))
                    total += 1

            row.append(f"{correct}/{total}")

        print(" & ".join(row) + r" \\")
