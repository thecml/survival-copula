import pandas as pd
import numpy as np
from pathlib import Path
import config as cfg

STRATEGIES = ["original"]

DATASET_ORDER = [
    "whas",
    "metabric",
    "churn",
    "gbsg",
    "nacd",
    "flchain",
    "support",
    "employee",
    "mimic_all",
    "seer_brain",
    "seer_liver",
    "seer_stomach",
]

DATASET_LABEL = {
    "whas": "WHAS",
    "metabric": "METABRIC",
    "churn": "Churn",
    "gbsg": "GBSG",
    "nacd": "NACD",
    "flchain": "FLChain",
    "support": "SUPPORT",
    "employee": "Employee",
    "mimic_all": "MIMIC-IV (all)",
    "seer_brain": "SEER (brain)",
    "seer_liver": "SEER (liver)",
    "seer_stomach": "SEER (stomach)",
}

# IBS variants only
METRIC_VARIANTS = [
    ("IBSIPCW", "IPCW", "IBSIPCWTime"),
    ("IBSIPCW", "IPCW", "IBSIPCWTime"),
    ("IBSIndepBG", "Dep (KM)", "IBSIndepBGTime"),
    ("IBSDepBG", "Dep (CG)", "IBSDepBGTime"),
    ("IBSDepBGUW", r"Dep (CG)$^{\text{UW}}$", "IBSDepBGUWTime"),
]

# Formatting
TIME_MIN_DECIMALS = 2
MEM_MIB_DECIMALS = 2
RUNTIME_SEC_DECIMALS = 3

# Print mean-only or mean ± std
PRINT_MEAN_PM_STD = False  # set True if you want mean ± std in the appendix

def std0(x: pd.Series) -> float:
    x = x.dropna().astype(float)
    return float(x.std(ddof=0)) if len(x) else np.nan

def fmt_mean(x: float, decimals: int) -> str:
    return "--" if np.isnan(x) else f"{x:.{decimals}f}"

def fmt_mean_pm_std(m: float, s: float, decimals: int) -> str:
    if np.isnan(m) or np.isnan(s):
        return "--"
    return f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"

def fmt_cell(m: float, s: float, decimals: int) -> str:
    return fmt_mean_pm_std(m, s, decimals) if PRINT_MEAN_PM_STD else fmt_mean(m, decimals)

def gib_to_mib(x: pd.Series) -> pd.Series:
    # Assumes CopulaMemoryUsed is in GiB in the CSV
    return x.astype(float) * 1024.0

# ----------------------------
# Load + filter
# ----------------------------
df = pd.read_csv(Path.joinpath(cfg.RESULTS_DIR, "semisynthetic_results_timing_org.csv"))

df = df[df["Strategy"].isin(STRATEGIES)].copy()

# ----------------------------
# Table 1: Copula fitting (deduplicate repeats across models)
# ----------------------------
required_cop_cols = {"Seed", "Dataset", "Strategy", "ModelName", "CopulaRuntime", "CopulaMemoryUsed"}
missing = required_cop_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns for copula table: {missing}")

cop = (
    df.sort_values(["Seed", "Dataset", "Strategy", "ModelName"])
      .groupby(["Seed", "Dataset", "Strategy"], as_index=False)
      .agg(
          CopulaRuntime=("CopulaRuntime", "first"),        # seconds
          CopulaMemoryUsed=("CopulaMemoryUsed", "first"),  # assumed GiB
      )
)

cop["CopulaTimeMin"] = cop["CopulaRuntime"].astype(float) / 60.0
cop["CopulaMemMiB"] = cop["CopulaMemoryUsed"].astype(float)  # already MiB

cop_ds = (
    cop.groupby("Dataset", as_index=False)
       .agg(
           time_mean=("CopulaTimeMin", "mean"),
           time_std=("CopulaTimeMin", std0),
           mem_mean=("CopulaMemMiB", "mean"),
           mem_std=("CopulaMemMiB", std0),
       )
)

print("% --- LaTeX rows: Copula fitting (Original only) ---")
for ds in DATASET_ORDER:
    row = cop_ds[cop_ds["Dataset"] == ds]
    if row.empty:
        print(f"% {ds} missing from CSV (Original)")
        continue
    t_m = float(row["time_mean"].iloc[0])
    t_s = float(row["time_std"].iloc[0])
    m_m = float(row["mem_mean"].iloc[0])
    m_s = float(row["mem_std"].iloc[0])

    time_cell = fmt_cell(t_m, t_s, TIME_MIN_DECIMALS)
    mem_cell = fmt_cell(m_m, m_s, MEM_MIB_DECIMALS)
    print(f"{DATASET_LABEL.get(ds, ds)} & {time_cell} & {mem_cell} \\\\")

# ----------------------------
# Table 2: IBS evaluation runtimes (seconds, Original only)
# ----------------------------
missing_time_cols = [col for _, _, col in METRIC_VARIANTS if col not in df.columns]
if missing_time_cols:
    raise ValueError("Missing IBS timing columns in CSV: " + ", ".join(missing_time_cols))

agg_dict = {}
for _, _, col in METRIC_VARIANTS:
    agg_dict[col] = (col, "mean")
    agg_dict[f"{col}_std"] = (col, std0)

eval_ds = df.groupby("Dataset", as_index=False).agg(**agg_dict)

print("\n% --- LaTeX rows: IBS runtime (Original only) ---")
for ds in DATASET_ORDER:
    row = eval_ds[eval_ds["Dataset"] == ds]
    if row.empty:
        print(f"% {ds} missing from CSV (Original)")
        continue

    cells = []
    for _, _, col in METRIC_VARIANTS:
        m = float(row[col].iloc[0])
        s = float(row[f"{col}_std"].iloc[0])
        cells.append(fmt_cell(m, s, RUNTIME_SEC_DECIMALS))

    print(f"{DATASET_LABEL.get(ds, ds)} & " + " & ".join(cells) + r" \\")
