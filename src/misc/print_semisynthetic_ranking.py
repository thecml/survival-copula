from pathlib import Path
import sys
import numpy as np
import pandas as pd
import config as cfg


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DATASETS = [
    "whas", "metabric", "churn", "gbsg", "nacd", "flchain",
    "support", "employee", "mimic_all", "seer_brain",
    "seer_liver", "seer_stomach",
]

DATASET_LABELS = {
    "whas": "WHAS",
    "metabric": "METAB.",
    "churn": "Churn",
    "gbsg": "GBSG",
    "nacd": "NACD",
    "flchain": "FLC.",
    "support": "SUPP.",
    "employee": "Emp.",
    "mimic_all": "MIMIC (IV)",
    "seer_brain": "SEER (brain)",
    "seer_liver": "SEER (liver)",
    "seer_stomach": "SEER (stom.)",
}

# Use all strategies for the combined total, giving up to 40 runs per dataset.
STRATEGIES = ["original", "top_5", "top_10", "random_25"]
SEEDS = list(range(10))
EXPECTED_MODELS = 5

# Metric columns. Each entry is:
#   (candidate result columns, body label, header label)
# The CoxPH IPCW column name differs across experiments. If no separate CoxPH
# column is present, we fall back to IBSIPCW and print a warning.
METRICS = [
    (["IBSIPCW"],
     "IPCW (KM)",
     r"\shortstack{IPCW\\(KM)}"),

    (["IBSIPCWCoxPH", "IBSIPCW_CoxPH", "IBSIPCWCOX", "IBSIPCW_Cox", "IBSIPCW"],
     "IPCW (CoxPH)",
     r"\shortstack{IPCW\\(CoxPH)}"),

    (["IBSIndepBG"],
     "Dep (KM)",
     r"\shortstack{Dep\\(KM)}"),

    (["IBSDepBG"],
     "Dep (CG)",
     r"\shortstack{Dep\\(CG)}"),

    (["IBSDepBGUW"],
     r"Dep (CG)$^{\text{UW}}$",
     r"\shortstack{Dep\\(CG)$^{\text{UW}}$}"),
]


# ---------------------------------------------------------------------
# Ranking helpers
# ---------------------------------------------------------------------

def resolve_metric_columns(results: pd.DataFrame):
    """Resolve metric keys from candidate column names."""
    resolved = []
    for candidates, body_label, header_label in METRICS:
        chosen = next((c for c in candidates if c in results.columns), None)
        if chosen is None:
            raise KeyError(
                f"Could not find any result column for metric {body_label}. "
                f"Tried: {candidates}"
            )

        # Warn if CoxPH falls back to KM IPCW.
        if body_label == "IPCW (CoxPH)" and chosen == "IBSIPCW":
            print(
                "[warning] No separate CoxPH IPCW column found; "
                "using IBSIPCW for IPCW (CoxPH).",
                file=sys.stderr,
            )

        resolved.append((chosen, body_label, header_label))
    return resolved


def topk_soft(true_vals, metric_vals, k=3, min_hits=2):
    """Return True if at least min_hits of the true top-k are in the metric top-k."""
    true_rank = np.argsort(true_vals)
    metric_rank = np.argsort(metric_vals)

    true_topk = set(true_rank[:k])
    metric_topk = set(metric_rank[:k])

    hits = len(true_topk & metric_topk)
    return hits >= min_hits, hits


def ranking_count(results: pd.DataFrame, dataset: str, metric_key: str):
    correct = 0
    total = 0

    for strategy in STRATEGIES:
        for seed in SEEDS:
            df = results.loc[
                (results["Dataset"] == dataset)
                & (results["Strategy"] == strategy)
                & (results["Seed"] == seed)
            ].sort_values("ModelName")

            if len(df) != EXPECTED_MODELS:
                continue  # skip incomplete runs

            true_vals = df["IBSTrue"].values
            metric_vals = df[metric_key].values

            correct += int(topk_soft(true_vals, metric_vals)[0])
            total += 1

    return correct, total


def fmt_score(correct: int, total: int, is_best: bool):
    value = f"{correct}/{total}"
    if is_best:
        return rf"\textbf{{{value}}}"
    return value


# ---------------------------------------------------------------------
# LaTeX printing
# ---------------------------------------------------------------------

def build_scores(results: pd.DataFrame, metric_specs):
    rows = []
    for dataset in DATASETS:
        entries = []
        for metric_key, body_label, header_label in metric_specs:
            correct, total = ranking_count(results, dataset, metric_key)
            ratio = correct / total if total > 0 else np.nan
            entries.append({
                "metric_key": metric_key,
                "label": body_label,
                "correct": correct,
                "total": total,
                "ratio": ratio,
            })
        rows.append((dataset, entries))
    return rows


def print_latex_table(results: pd.DataFrame):
    metric_specs = resolve_metric_columns(results)
    rows = build_scores(results, metric_specs)

    print(r"\begin{table}[!t]")
    print(r"\centering")
    print(r"\caption{")
    print(r"Ranking performance of random, independent and proposed dependent metrics on 12 datasets using the \emph{Original} feature strategy. The results show how often each metric correctly identified the top-3 survival learners according to the oracle IBS metric across 10 experiments per dataset. Higher is better.}")
    print(r"\label{tab:ranking_results}")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{l|ccccc}")
    print(r"\toprule")

    header = ["Dataset"] + [header_label for _, _, header_label in metric_specs]
    print(" & ".join(header) + r" \\")
    print(r"\midrule")

    for dataset, entries in rows:
        ratios = np.array([e["ratio"] for e in entries], dtype=float)
        best = np.nanmax(ratios)
        out = [DATASET_LABELS.get(dataset, dataset)]
        for e in entries:
            out.append(fmt_score(e["correct"], e["total"], np.isclose(e["ratio"], best)))
        print(" & ".join(out) + r" \\")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")


if __name__ == "__main__":
    results_path = Path(cfg.RESULTS_DIR) / "semisynthetic_results.csv"
    results = pd.read_csv(results_path)
    print_latex_table(results)
