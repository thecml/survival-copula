import numpy as np
import pandas as pd
import config as cfg

df = pd.read_csv(f"{cfg.RESULTS_DIR}/synthetic_results_wrong_copula.csv")

# --- Settings ---
z_threshold = 2
keep_exps = ["family", "dep", "gaussian"]
copulas_keep = ["clayton", "frank"]
tau_keep = [0.0, 0.25, 0.5, 0.75]
n_keep = 10000

m_ipcw = "ibs_ipcw_error"
m_km   = "ibs_indep_bg_error"
m_cg   = "ibs_dep_bg_error"

# --- Robust z-score filtering ---
cols_for_z = [
    "ibs_uncens_error", "ibs_ipcw_error", "ibs_indep_bg_error",
    "ibs_dep_bg_error", "ibs_indep_bguw_error", "ibs_dep_bguw_error"
]

X = df[cols_for_z].astype(float)
mu = X.mean(axis=0)
sd = X.std(axis=0, ddof=0).replace(0, np.nan)
Z = ((X - mu) / sd).fillna(0.0)
df = df[(Z.abs() < z_threshold).all(axis=1)].copy()

# --- Restrict ---
df = df[
    df["experiment"].isin(keep_exps) &
    df["copula_name"].isin(copulas_keep) &
    df["k_tau"].isin(tau_keep) &
    (df["n_samples"] == n_keep)
]
exp_order = ["family", "dep", "gaussian"]
df["experiment"] = pd.Categorical(df["experiment"], categories=exp_order, ordered=True)

# --- Mean/std ---
stats = (
    df.groupby(["copula_name", "k_tau", "experiment"])[[m_ipcw, m_km, m_cg]]
      .agg(["mean", "std"])
)
stats.columns = ["_".join(c) for c in stats.columns]
stats = stats.reset_index()

def fmt(metric):
    return stats.apply(lambda r: f"{r[f'{metric}_mean']:.4f} $\\pm$ {r[f'{metric}_std']:.4f}", axis=1)

stats["IBS-IPCW"] = fmt(m_ipcw)
stats["IBS-Dep (KM)"] = fmt(m_km)
stats["IBS-Dep (CG)"] = fmt(m_cg)

stats = stats.sort_values(["copula_name", "k_tau", "experiment"])

table = stats[["copula_name", "k_tau", "experiment",
               "IBS-IPCW", "IBS-Dep (KM)", "IBS-Dep (CG)"]]

table = table.rename(columns={
    "copula_name": "Copula",
    "k_tau": r"$\tau$",
    "experiment": "Experiment"
})
table["Copula"] = table["Copula"].str.capitalize()
table["Experiment"] = table["Experiment"].str.capitalize()

latex = table.to_latex(
    index=False,
    escape=False,
    column_format="ll lccc",
    caption="Mean $\\pm$ std of IBS variants under wrong copula specification ($n=10{,}000$).",
    label="tab:ibs_wrong_copula"
)

print("\n================= LATEX TABLE =================\n")
print(latex)
print("\n================================================\n")
