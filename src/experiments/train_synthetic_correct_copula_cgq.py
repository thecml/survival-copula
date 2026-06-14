"""Synthetic dependent-censoring experiment comparing BG_UW against CG_Q.

This is intentionally a minimal variant of train_synthetic_correct_copula.py:
the DGP, copula calibration, train/test split, CoxPH model, and true IBS
reference are unchanged. Only the dependent-evaluator methods and output files
are changed.
"""

import os
import random
import torch
import pandas as pd
import numpy as np
import config as cfg
from SurvivalEVAL import SurvivalEvaluator
from scipy.interpolate import interp1d

from dgp import DGP_Weibull_linear
from evaluators import DependentEvaluator
from sota.sksurv import make_cox_model
from utility.data import dotdict
from utility.experiment import _set_global_seeds, _simulate_uv_archimedean, _uv_seed
from utility.survival import convert_to_structured, kendall_tau_to_theta, make_time_bins

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_cfg = {
    "alpha_e1": 19,  # censor alpha base
    "alpha_e2": 17,  # event alpha
    "gamma_e1": 6,   # censor gamma
    "gamma_e2": 4,   # event gamma
    "n_samples": 10000,
    "n_features": 10,
}

def make_dep_censor_df_for_setting(
    X: torch.Tensor,
    dgp_event,
    dgp_cens,
    u: torch.Tensor,
    v: torch.Tensor,
):
    t_c = dgp_cens.rvs(X, u)  # numpy
    t_e = dgp_event.rvs(X, v)  # numpy

    T = np.minimum(t_e, t_c)
    E = (t_e < t_c).astype(int)

    n_features = X.shape[1]
    df = pd.DataFrame(X.detach().cpu().numpy(), columns=[f"X{i}" for i in range(n_features)])
    df["time"] = np.where(T <= 0, 1.0, T)
    df["event"] = E
    df["true_time"] = np.where(t_e <= 0, 1.0, t_e)
    df["true_censor"] = np.where(t_c <= 0, 1.0, t_c)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["time", "true_time"]).reset_index(drop=True)
    return df

def make_train_test_split_indices(n: int, train_frac: float, split_seed: int):
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    return perm[:n_train], perm[n_train:]

def calibrate_alpha_c_mults_by_tau(
    data_cfg,
    pilot_seeds,
    copula_names,
    k_taus,
    mult_grid,
    target_censoring,   # e.g. 0.50 or 0.70
    device,
    dtype,
    linear=True,
    hidden_dim=32,
):
    """
    For each (copula, k_tau), choose alpha_c_mult that makes censoring_rate close to target_censoring
    averaged across pilot_seeds.
    Returns:
      chosen[(copula, k_tau)] = best_mult
      calib_df with mean/std censoring per candidate mult
    """
    assert linear, "Nonlinear version not implemented here (same idea, but reuse weights and change alpha)."

    n_samples  = int(data_cfg["n_samples"])
    n_features = int(data_cfg["n_features"])

    alpha_c_base = float(data_cfg["alpha_e1"])
    gamma_c      = float(data_cfg["gamma_e1"])
    alpha_e      = float(data_cfg["alpha_e2"])
    gamma_e      = float(data_cfg["gamma_e2"])

    rows = []

    for seed in pilot_seeds:
        _set_global_seeds(int(seed))

        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        X = torch.rand((n_samples, n_features), generator=g, device=device, dtype=dtype)

        beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        beta_cens  = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        dgp_event = DGP_Weibull_linear(
            n_features, alpha_e, gamma_e, use_x=True,
            device=device, dtype=dtype, coeff=beta_event
        )

        for copula_name in copula_names:
            for k_tau in k_taus:
                # sample (u,v)
                if float(k_tau) == 0.0:
                    rng = np.random.default_rng(int(_uv_seed(int(seed), float(k_tau), str(copula_name))))
                    u_np = rng.uniform(0.0, 1.0, n_samples)
                    v_np = rng.uniform(0.0, 1.0, n_samples)
                else:
                    uv_seed = _uv_seed(int(seed), float(k_tau), str(copula_name))
                    u_np, v_np = _simulate_uv_archimedean(str(copula_name), n_samples, float(k_tau), uv_seed)

                u = torch.from_numpy(u_np).to(device=device, dtype=dtype)
                v = torch.from_numpy(v_np).to(device=device, dtype=dtype)

                # event times are fixed for this (seed,copula,tau)
                t_e = dgp_event.rvs(X, v)

                for mult in mult_grid:
                    alpha_c = alpha_c_base * float(mult)
                    dgp_cens = DGP_Weibull_linear(
                        n_features, alpha_c, gamma_c, use_x=True,
                        device=device, dtype=dtype, coeff=beta_cens
                    )
                    t_c = dgp_cens.rvs(X, u)

                    E = (t_e < t_c).astype(np.float64)
                    censoring_rate = float(1.0 - E.mean())

                    rows.append({
                        "seed": int(seed),
                        "copula_name": str(copula_name),
                        "k_tau": float(k_tau),
                        "alpha_c_mult": float(mult),
                        "alpha_c_used": float(alpha_c),
                        "censoring_rate": censoring_rate,
                        "target": float(target_censoring),
                        "abs_err_to_target": abs(censoring_rate - float(target_censoring)),
                    })

    calib = pd.DataFrame(rows)

    calib_df = (
        calib.groupby(["copula_name", "k_tau", "alpha_c_mult"], as_index=False)
             .agg(
                 censor_rate_mean=("censoring_rate", "mean"),
                 censor_rate_std=("censoring_rate", "std"),
                 abs_err_mean=("abs_err_to_target", "mean"),
                 alpha_c_used_mean=("alpha_c_used", "mean"),
             )
             .sort_values(["copula_name", "k_tau", "abs_err_mean", "alpha_c_mult"])
             .reset_index(drop=True)
    )

    # Select the censoring multiplier separately for each seed.
    # This keeps each experimental replicate close to the target censoring rate
    # instead of only matching the target on average across seeds.
    chosen = {}
    selected_rows = []
    for seed in pilot_seeds:
        for copula_name in copula_names:
            for k_tau in k_taus:
                sub = calib[
                    (calib["seed"] == int(seed)) &
                    (calib["copula_name"] == str(copula_name)) &
                    (calib["k_tau"] == float(k_tau))
                ].copy()
                best_row = sub.sort_values(["abs_err_to_target", "alpha_c_mult"]).iloc[0]
                best = float(best_row["alpha_c_mult"])
                chosen[(int(seed), str(copula_name), float(k_tau))] = best
                selected_rows.append(best_row.to_dict())

    selected_calib_df = pd.DataFrame(selected_rows)

    return chosen, calib_df, selected_calib_df

def run_bguw_vs_cgq_experiment(
    data_cfg,
    seeds,
    copula_names,
    k_taus,
    alpha_c_mult_by_setting,   # dict[(seed,copula,k_tau)] -> mult
    device,
    dtype,
    train_frac=0.7,
    split_seed=0,
    num_points=10,
    linear=True,
    hidden_dim=32,
):
    rows = []

    n_samples = int(data_cfg["n_samples"])
    n_features = int(data_cfg["n_features"])

    alpha_c_base = float(data_cfg["alpha_e1"])   # <-- base, will be multiplied
    gamma_c = float(data_cfg["gamma_e1"])
    alpha_e = float(data_cfg["alpha_e2"])
    gamma_e = float(data_cfg["gamma_e2"])

    features = [f"X{i}" for i in range(n_features)]
    train_idx, test_idx = make_train_test_split_indices(n_samples, train_frac, split_seed)

    for seed in seeds:
        _set_global_seeds(int(seed))

        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        X = torch.rand((n_samples, n_features), generator=g, device=device, dtype=dtype)

        if linear:
            beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
            beta_cens  = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

            dgp_event = DGP_Weibull_linear(
                n_features, alpha_e, gamma_e, use_x=True,
                device=device, dtype=dtype, coeff=beta_event
            )
        else:
            raise NotImplementedError("Add the nonlinear version later (same calibration idea).")

        for copula_name in copula_names:
            for k_tau in k_taus:
                # pick calibrated alpha_c for this (copula,tau)
                mult = float(alpha_c_mult_by_setting[(int(seed), str(copula_name), float(k_tau))])
                alpha_c = alpha_c_base * mult

                dgp_cens = DGP_Weibull_linear(
                    n_features, alpha_c, gamma_c, use_x=True,
                    device=device, dtype=dtype, coeff=beta_cens
                )

                # sample (u,v)
                if float(k_tau) == 0.0:
                    rng = np.random.default_rng(int(_uv_seed(int(seed), float(k_tau), str(copula_name))))
                    u_np = rng.uniform(0.0, 1.0, n_samples)
                    v_np = rng.uniform(0.0, 1.0, n_samples)
                else:
                    uv_seed = _uv_seed(int(seed), float(k_tau), str(copula_name))
                    u_np, v_np = _simulate_uv_archimedean(str(copula_name), n_samples, float(k_tau), uv_seed)

                u = torch.from_numpy(u_np).to(device=device, dtype=dtype)
                v = torch.from_numpy(v_np).to(device=device, dtype=dtype)

                df = make_dep_censor_df_for_setting(X=X, dgp_event=dgp_event, dgp_cens=dgp_cens, u=u, v=v)

                if len(df) != n_samples:
                    tr, te = make_train_test_split_indices(len(df), train_frac, split_seed)
                    df_train = df.iloc[tr].copy()
                    df_test  = df.iloc[te].copy()
                else:
                    df_train = df.iloc[train_idx].copy()
                    df_test  = df.iloc[test_idx].copy()

                censoring_rate = float(1.0 - df["event"].mean())

                true_test_time = df_test["true_time"].values
                true_test_event = np.ones(df_test.shape[0], dtype=int)

                y_train = convert_to_structured(df_train["time"], df_train["event"])
                X_train = df_train[features]
                X_test  = df_test[features]

                time_bins = make_time_bins(df_train["true_time"].values, event=None, dtype=dtype).to(device)
                time_bins = torch.cat((torch.tensor([0.0], device=device, dtype=dtype), time_bins)).cpu().numpy()
                t_star = np.quantile(df_test["true_time"].values, 0.9)
                time_bins = time_bins[time_bins <= t_star]

                config = dotdict(cfg.COXPH_PARAMS)
                model = make_cox_model(config)
                model.fit(X_train, y_train)

                surv_fns = model.predict_survival_function(X_test)
                surv = np.row_stack([fn(model.unique_times_) for fn in surv_fns])

                spline = interp1d(
                    model.unique_times_, surv, kind="linear",
                    bounds_error=False,
                    fill_value=(1.0, surv[:, -1])
                )
                S = np.clip(spline(time_bins), 0.0, 1.0)
                surv_on_grid = pd.DataFrame(S, columns=time_bins)
                surv_on_grid[0.0] = 1.0

                true_eval = SurvivalEvaluator(surv_on_grid, time_bins, true_test_time, true_test_event)
                ibs_true = float(true_eval.integrated_brier_score(IPCW_weighted=False, num_points=num_points))

                theta = kendall_tau_to_theta(str(copula_name), float(k_tau))
                dep_eval = DependentEvaluator(
                    surv_on_grid, time_bins,
                    df_test["time"].values, df_test["event"].values,
                    df_train["time"].values, df_train["event"].values,
                    copula_name=str(copula_name), alpha=theta
                )
                ibs_dep_bguw = float(dep_eval.integrated_brier_score(method="BG_UW", num_points=num_points))
                ibs_dep_cgq = float(dep_eval.integrated_brier_score(method="CG_Q", num_points=num_points))

                err_dep_bguw = abs(ibs_true - ibs_dep_bguw)
                err_dep_cgq = abs(ibs_true - ibs_dep_cgq)

                rows.append({
                    "seed": int(seed),
                    "copula_name": str(copula_name),
                    "k_tau": float(k_tau),
                    "alpha_c_mult": float(mult),
                    "alpha_c_used": float(alpha_c),
                    "censoring_rate": float(censoring_rate),
                    "ibs_true": ibs_true,
                    "ibs_dep_bguw": ibs_dep_bguw,
                    "ibs_dep_cgq": ibs_dep_cgq,
                    "err_dep_bguw": err_dep_bguw,
                    "err_dep_cgq": err_dep_cgq,
                    "delta_err_cgq_minus_bguw": err_dep_cgq - err_dep_bguw,
                })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    SEEDS = list(range(0, 10))
    PILOT_SEEDS = list(range(0, 10))
    COPULA_NAMES = ["clayton", "frank"]
    K_TAU = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # use a target censoring rate
    TARGET_CENSOR = 0.50

    # grid of multipliers to search over
    mult_grid = np.concatenate([
        np.linspace(0.05, 0.50, 40),
        np.linspace(0.50, 2.00, 60),
        np.linspace(2.00, 6.00, 40),
    ]).tolist()

    alpha_c_mult_by_setting, calib_df, selected_calib_df = calibrate_alpha_c_mults_by_tau(
        data_cfg=data_cfg,
        pilot_seeds=PILOT_SEEDS,
        copula_names=COPULA_NAMES,
        k_taus=K_TAU,
        mult_grid=mult_grid,
        target_censoring=TARGET_CENSOR,
        device=device,
        dtype=dtype,
        linear=True,
    )

    # optional sanity check
    print("Seed-specific censoring calibration summary:")
    print(
        selected_calib_df.groupby(["copula_name", "k_tau"], as_index=False)
        .agg(
            censor_rate_mean=("censoring_rate", "mean"),
            censor_rate_std=("censoring_rate", "std"),
            abs_err_mean=("abs_err_to_target", "mean"),
            alpha_c_mult_min=("alpha_c_mult", "min"),
            alpha_c_mult_max=("alpha_c_mult", "max"),
        )
        .to_string(index=False)
    )

    results_df = run_bguw_vs_cgq_experiment(
        data_cfg=data_cfg,
        seeds=SEEDS,
        copula_names=COPULA_NAMES,
        k_taus=K_TAU,
        alpha_c_mult_by_setting=alpha_c_mult_by_setting,
        device=device,
        dtype=dtype,
        train_frac=0.7,
        split_seed=0,
        num_points=10,
        linear=True,
    )

    print("Actual censoring rates in generated results:")
    print(
        results_df.groupby(["copula_name", "k_tau"], as_index=False)
        .agg(
            censoring_rate_mean=("censoring_rate", "mean"),
            censoring_rate_std=("censoring_rate", "std"),
            censoring_rate_min=("censoring_rate", "min"),
            censoring_rate_max=("censoring_rate", "max"),
        )
        .to_string(index=False)
    )
    
    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    filename = f"{cfg.RESULTS_DIR}/synthetic_results_bguw_vs_cgq.csv"
    results_df.to_csv(filename, index=False)