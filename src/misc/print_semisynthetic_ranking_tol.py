from pathlib import Path
import numpy as np
import pandas as pd
import config as cfg

EPSILON = 0.002     # tolerance for IBS differences
KENDALL_THRESHOLD = 0.33


# ============================================================
# OPTION A: ε – TOLERANT TOP-3 SET MATCHING
# ============================================================
def tolerant_top3_correct(true_vals, metric_vals, epsilon=EPSILON):
    """
    Returns 1 if metric selects the same TOP-3 SET as the truth,
    where ties (differences < epsilon) form equivalence groups.
    Order is completely ignored.
    """

    # ---- Step 1: Rank true IBS ----
    true_rank = np.argsort(true_vals)
    sorted_true = true_vals[true_rank]

    # ---- Step 2: Determine true top-3 with tolerance ----
    top3_limit = sorted_true[2]   # IBS value of the 3rd ranked model

    # Allow all models with true IBS <= top3_limit + ε
    true_top_set = set(np.where(true_vals <= top3_limit + epsilon)[0])

    # If more than 3 satisfy this (ties), restrict to smallest 3 indices
    if len(true_top_set) > 3:
        true_top_set = set(sorted(list(true_top_set))[:3])

    # ---- Step 3: Metric top-3 the same way ----
    metric_rank = np.argsort(metric_vals)
    top3_metric = set(metric_rank[:3])

    # ---- Step 4: Evaluate set equality ----
    return int(true_top_set == top3_metric)


# ============================================================
# OPTION B: ε – MEANINGFUL KENDALL τ
# ============================================================
def kendall_tau_tolerant(true_vals, metric_vals, epsilon=EPSILON):
    """
    Kendall τ computed ONLY on pairs whose true IBS differ by ≥ epsilon.
    """
    n = len(true_vals)
    true_rank = np.argsort(true_vals)
    metric_rank = np.argsort(metric_vals)

    # Precompute metric order positions for speed
    metric_pos = {m: r for r, m in enumerate(metric_rank)}

    concordant = 0
    discordant = 0

    for a in range(n):
        for b in range(a + 1, n):
            i = true_rank[a]
            j = true_rank[b]

            # Ignore pairs too similar in true IBS
            if abs(true_vals[i] - true_vals[j]) < epsilon:
                continue

            # ranking direction
            direction_true = np.sign(true_vals[j] - true_vals[i])
            direction_metric = np.sign(metric_pos[j] - metric_pos[i])

            if direction_true == direction_metric:
                concordant += 1
            else:
                discordant += 1

    denom = concordant + discordant
    if denom == 0:
        return np.nan

    return (concordant - discordant) / denom


def kendall_top3_correct_tau(true_vals, metric_vals,
                             epsilon=EPSILON, threshold=KENDALL_THRESHOLD):
    """Returns 1 if τ_ε >= threshold."""
    tau = kendall_tau_tolerant(true_vals, metric_vals, epsilon)
    if np.isnan(tau):
        return 0
    return int(tau >= threshold)


# ============================================================
# EXAMPLE: Compute both Option A & B across datasets
# ============================================================
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
        "seer_liver", "seer_stomach",
    ]

    strategies = ["original", "top_5", "top_10", "random_25"]
    model_names = ["coxph", "gbsa", "rsf", "deepsurv", "mtlr"]
    seeds = list(range(10))

    for metric_key, metric_label in ranking_metrics:
        print("\n====", metric_label, "====")

        for dataset in datasets:
            correct_A = 0   # tolerant top-3 set match
            correct_B = 0   # kendall τ meaningful

            total = 0

            for strategy in strategies:
                for seed in seeds:
                    df = results[
                        (results["Dataset"] == dataset) &
                        (results["Strategy"] == strategy) &
                        (results["Seed"] == seed)
                    ].sort_values("ModelName")

                    if len(df) != 5:
                        continue

                    true_vals = df["IBSTrue"].values
                    metric_vals = df[metric_key].values

                    # Option A
                    correct_A += tolerant_top3_correct(true_vals, metric_vals)

                    # Option B
                    correct_B += kendall_top3_correct_tau(true_vals, metric_vals)

                    total += 1

            print(
                f"{dataset:12s} | "
                f"A (ε-top3): {correct_A}/{total} | "
                f"B (ε-τ): {correct_B}/{total}"
            )
