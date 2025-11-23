from pathlib import Path
import numpy as np
import pandas as pd
import config as cfg

if __name__ == "__main__":
    # Load results
    results = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results.csv"))

    # Normalize column names just in case
    results.columns = results.columns.str.strip()

    # Metrics to evaluate (added Smooth)
    ranking_metrics = [
        ("IBSIPCW", "IPCW"),
        ("IBSIndepBG", "Dep (KM)"),
        ("IBSDepBG", "Dep (CG)"),
        ("IBSDepBGUW", r"Dep (CG)$^{\text{UW}}$"),
        ("IBSDepBGSmooth", "Dep (CG) Smooth"),
    ]

    # Quick sanity check: required columns
    needed_cols = {"IBSTrue"} | {m for m, _ in ranking_metrics}
    missing = needed_cols - set(results.columns)
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    # Datasets to include (keep your list, but you can also infer from the file)
    datasets = [
        "whas", "metabric", "churn", "gbsg", "nacd", "flchain",
        "support", "employee", "mimic_all", "seer_brain",
        "seer_liver", "seer_stomach"
    ]

    # Use whatever is actually present in the file
    strategies = sorted(results["Strategy"].unique())
    seeds = sorted(results["Seed"].unique())

    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]

    def topk_correct(true_vals, metric_vals, k=3):
        """Return True if the *set* of top-k models matches (order ignored)."""
        true_rank = np.argsort(true_vals)
        metric_rank = np.argsort(metric_vals)
        return set(true_rank[:k]) == set(metric_rank[:k])

    # For overall stats
    global_correct = {key: 0 for key, _ in ranking_metrics}
    global_total = {key: 0 for key, _ in ranking_metrics}

    # ==== Build all rows ====
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

                    # Skip incomplete runs (need all 5 models)
                    if len(df) != len(model_names):
                        continue

                    true_vals = df["IBSTrue"].values
                    metric_vals = df[metric_key].values

                    hit = int(topk_correct(true_vals, metric_vals, k=3))
                    correct += hit
                    total += 1

            row.append(f"{correct}/{total}")
            global_correct[metric_key] += correct
            global_total[metric_key] += total

        # LaTeX row for this metric
        print(" & ".join(row) + r" \\")

    # ==== Overall summary ====
    print("\n% Overall top-3 ranking accuracy:")
    for metric_key, metric_label in ranking_metrics:
        c = global_correct[metric_key]
        t = global_total[metric_key]
        if t > 0:
            print(f"% {metric_label}: {c}/{t} = {c/t:.3f}")
        else:
            print(f"% {metric_label}: 0/0 (no runs)")
