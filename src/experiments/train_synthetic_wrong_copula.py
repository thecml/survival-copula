import os
import random

import numpy as np
import pandas as pd
import torch
import config as cfg
from SurvivalEVAL import SurvivalEvaluator
from scipy.interpolate import interp1d
from scipy.stats import norm

from dgp import DGP_Weibull_linear
from evaluators import DependentEvaluator
from sota.sksurv import make_cox_model
from utility.data import dotdict
from utility.experiment import _set_global_seeds, _uv_seed
from utility.survival import convert_to_structured, kendall_tau_to_theta, make_time_bins

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RUN_TAG = "wrong_copula_fixed_event_fixed_model_seedcal_v5"
print(f"[RUN_TAG] {RUN_TAG}")

data_cfg = {
    "alpha_e1": 19,
    "alpha_e2": 17,
    "gamma_e1": 6,
    "gamma_e2": 4,
    "n_samples": 10000,
    "n_features": 10,
}


def tau_to_rho_gaussian(k_tau: float) -> float:
    """Kendall's tau to Gaussian copula Pearson rho."""
    k_tau = float(k_tau)
    rho = np.sin(np.pi * k_tau / 2.0)
    return float(np.clip(rho, -0.999, 0.999))


def make_train_test_split_indices(n: int, train_frac: float, split_seed: int):
    rng = np.random.default_rng(int(split_seed))
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    return perm[:n_train], perm[n_train:]


def make_event_uniforms(*, seed: int, n: int, device, dtype):
    """
    Fixed event uniforms per seed.

    These are reused across all copula families and all true Kendall tau values.
    Consequently, the realized event times and oracle IBS target are kept much
    more stable when varying dependence strength.
    """
    rng = np.random.default_rng(int(seed) + 123_456_789)
    v_np = rng.uniform(0.0, 1.0, int(n))
    v_np = np.clip(v_np, 1e-12, 1.0 - 1e-12)
    return torch.from_numpy(v_np).to(device=device, dtype=dtype)


def sample_censor_uniform_given_event_uniform(
    *,
    copula_name: str,
    k_tau: float,
    seed: int,
    event_v: torch.Tensor,
    device,
    dtype,
):
    """
    Sample censoring uniforms U conditional on fixed event uniforms V.

    This is the core redesign: V is fixed across tau, while U changes with the
    copula dependence. The resulting event times are fixed across tau, and only
    censoring becomes more/less dependent on those event times.

    Variables are the uniforms passed to DGP_Weibull_linear.rvs. In this codebase
    rvs(x, u) uses the survival-uniform inverse t = F^{-1}(u | x) in the sense
    S(t | x)=u, but the copula construction only needs them to be Uniform(0,1).
    """
    copula_name = str(copula_name)
    k_tau = float(k_tau)
    n = int(event_v.numel())

    v = event_v.detach().cpu().numpy().astype(float)
    v = np.clip(v, 1e-12, 1.0 - 1e-12)

    rng = np.random.default_rng(int(_uv_seed(int(seed), float(k_tau), copula_name)) + 987_654_321)
    w = rng.uniform(0.0, 1.0, n)
    w = np.clip(w, 1e-12, 1.0 - 1e-12)

    if k_tau == 0.0:
        u = w

    elif copula_name == "gaussian":
        rho = tau_to_rho_gaussian(k_tau)
        z_v = norm.ppf(v)
        z_w = norm.ppf(w)
        z_u = rho * z_v + np.sqrt(1.0 - rho * rho) * z_w
        u = norm.cdf(z_u)

    elif copula_name == "clayton":
        theta = float(kendall_tau_to_theta("clayton", k_tau))
        theta = max(theta, 1e-12)
        # w = dC(u,v)/dv = v^(-theta-1) * A^(-1/theta-1)
        # A = u^(-theta) + v^(-theta) - 1
        A = (w * (v ** (theta + 1.0))) ** (-theta / (1.0 + theta))
        u_neg_theta = A - (v ** (-theta)) + 1.0
        u_neg_theta = np.maximum(u_neg_theta, 1e-12)
        u = u_neg_theta ** (-1.0 / theta)

    elif copula_name == "frank":
        theta = float(kendall_tau_to_theta("frank", k_tau))
        theta = max(theta, 1e-12)
        # For Frank, solve w = dC(u,v)/dv for a = exp(-theta*u).
        # Let b=exp(-theta*v), d=exp(-theta)-1, y=a-1:
        # w = b*y / (d + y*(b-1)) => y = w*d / (b - w*(b-1)).
        b = np.exp(-theta * v)
        d = np.exp(-theta) - 1.0
        denom = b - w * (b - 1.0)
        denom = np.where(np.abs(denom) < 1e-12, np.sign(denom) * 1e-12, denom)
        a = 1.0 + (w * d / denom)
        a = np.clip(a, np.exp(-theta) + 1e-12, 1.0 - 1e-12)
        u = -np.log(a) / theta

    else:
        raise ValueError(f"Unknown copula_name={copula_name}")

    u = np.clip(u, 1e-12, 1.0 - 1e-12)
    return torch.from_numpy(u).to(device=device, dtype=dtype)


def make_dep_censor_df_for_setting(*, X, dgp_event, dgp_cens, u_censor, v_event):
    t_c = dgp_cens.rvs(X, u_censor)  # numpy
    t_e = dgp_event.rvs(X, v_event)  # numpy; fixed across tau for a seed

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


def assumed_settings_for_experiment(
    *,
    exp: str,
    dgp_copula: str,
    dgp_tau: float,
    dep_true_tau: float,
    dep_assumed_taus: list[float],
):
    exp = str(exp)
    dgp_copula = str(dgp_copula)
    dgp_tau = float(dgp_tau)

    if exp == "family":
        if dgp_copula == "clayton":
            return [("frank", dgp_tau)]
        if dgp_copula == "frank":
            return [("clayton", dgp_tau)]
        return []

    if exp == "dep_clayton":
        if dgp_copula == "clayton" and np.isclose(dgp_tau, float(dep_true_tau)):
            return [("clayton", float(tau_assumed)) for tau_assumed in dep_assumed_taus]
        return []

    if exp == "dep_frank":
        if dgp_copula == "frank" and np.isclose(dgp_tau, float(dep_true_tau)):
            return [("frank", float(tau_assumed)) for tau_assumed in dep_assumed_taus]
        return []

    if exp == "gaussian":
        if dgp_copula == "gaussian":
            return [("clayton", dgp_tau)]
        return []

    raise ValueError(exp)


def calibrate_alpha_c_mults_by_tau(
    *,
    data_cfg,
    pilot_seeds,
    copula_names,
    k_taus,
    mult_grid,
    target_censoring,
    device,
    dtype,
    linear=True,
    hidden_dim=32,
):
    """
    Seed-specific calibration with fixed event uniforms.

    Returns:
      chosen[(seed, copula, k_tau)] = best_mult
      calib_df = all calibration rows
    """
    assert linear, "Nonlinear version not implemented here."

    n_samples = int(data_cfg["n_samples"])
    n_features = int(data_cfg["n_features"])

    alpha_c_base = float(data_cfg["alpha_e1"])
    gamma_c = float(data_cfg["gamma_e1"])
    alpha_e = float(data_cfg["alpha_e2"])
    gamma_e = float(data_cfg["gamma_e2"])

    rows = []

    for seed in pilot_seeds:
        _set_global_seeds(int(seed))

        g = torch.Generator(device=device)
        g.manual_seed(int(seed))

        X = torch.rand((n_samples, n_features), generator=g, device=device, dtype=dtype)
        beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        beta_cens = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        dgp_event = DGP_Weibull_linear(
            n_features, alpha_e, gamma_e, use_x=True,
            device=device, dtype=dtype, coeff=beta_event,
        )

        v_event = make_event_uniforms(seed=int(seed), n=n_samples, device=device, dtype=dtype)
        t_e = dgp_event.rvs(X, v_event)

        for copula_name in copula_names:
            for k_tau in k_taus:
                u_censor = sample_censor_uniform_given_event_uniform(
                    copula_name=str(copula_name),
                    k_tau=float(k_tau),
                    seed=int(seed),
                    event_v=v_event,
                    device=device,
                    dtype=dtype,
                )

                for mult in mult_grid:
                    alpha_c = alpha_c_base * float(mult)
                    dgp_cens = DGP_Weibull_linear(
                        n_features, alpha_c, gamma_c, use_x=True,
                        device=device, dtype=dtype, coeff=beta_cens,
                    )
                    t_c = dgp_cens.rvs(X, u_censor)
                    E = (t_e < t_c).astype(np.float64)
                    censoring_rate = float(1.0 - E.mean())

                    rows.append({
                        "run_tag": RUN_TAG,
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

    chosen = {}
    selected_rows = []
    for seed in pilot_seeds:
        for copula_name in copula_names:
            for k_tau in k_taus:
                sub = calib[
                    (calib["seed"] == int(seed))
                    & (calib["copula_name"] == str(copula_name))
                    & (calib["k_tau"] == float(k_tau))
                ].copy()
                if sub.empty:
                    raise RuntimeError(
                        f"Missing calibration rows for seed={seed}, copula={copula_name}, tau={k_tau}"
                    )
                best_row = sub.sort_values(["abs_err_to_target", "alpha_c_mult"]).iloc[0]
                best_mult = float(best_row["alpha_c_mult"])
                chosen[(int(seed), str(copula_name), float(k_tau))] = best_mult
                selected_rows.append(best_row.to_dict())

    selected_calib_df = pd.DataFrame(selected_rows)
    return chosen, calib_df, selected_calib_df

def run_wrong_copula_experiment(
    *,
    data_cfg,
    seeds,
    dgp_copulas,
    k_taus,
    alpha_c_mult_by_setting,
    experiments,
    dep_true_tau,
    dep_assumed_taus,
    device,
    dtype,
    train_frac=0.7,
    split_seed=0,
    num_points=10,
):
    """Run misspecification experiment with fixed event times and fixed model predictions.

    For each seed, event times and the prediction model are fixed once. Only the
    observed censoring process changes with the DGP copula/tau. This isolates
    evaluation-metric behavior from changes in the fitted survival model.
    """
    rows = []

    n_samples = int(data_cfg["n_samples"])
    n_features = int(data_cfg["n_features"])

    alpha_c_base = float(data_cfg["alpha_e1"])
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
        beta_event = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1
        beta_cens = 2 * torch.rand((n_features,), generator=g, device=device, dtype=dtype) - 1

        dgp_event = DGP_Weibull_linear(
            n_features, alpha_e, gamma_e, use_x=True,
            device=device, dtype=dtype, coeff=beta_event,
        )

        # Fixed event uniforms and true event times for this seed.
        v_event = make_event_uniforms(seed=int(seed), n=n_samples, device=device, dtype=dtype)
        true_event_time = dgp_event.rvs(X, v_event)
        true_event_time = np.where(true_event_time <= 0, 1.0, true_event_time)

        X_np = X.detach().cpu().numpy()
        X_df = pd.DataFrame(X_np, columns=features)

        X_train_fixed = X_df.iloc[train_idx].copy()
        X_test_fixed = X_df.iloc[test_idx].copy()
        true_train_time = true_event_time[train_idx]
        true_test_time = true_event_time[test_idx]
        true_train_event = np.ones(len(train_idx), dtype=int)
        true_test_event = np.ones(len(test_idx), dtype=int)

        # Fixed prediction model: fit once per seed on uncensored/oracle event times.
        y_train_oracle = convert_to_structured(true_train_time, true_train_event)

        time_bins = make_time_bins(true_train_time, event=None, dtype=dtype).to(device)
        time_bins = torch.cat((torch.tensor([0.0], device=device, dtype=dtype), time_bins)).cpu().numpy()
        t_star = np.quantile(true_test_time, 0.9)
        time_bins = time_bins[time_bins <= t_star]

        config = dotdict(cfg.COXPH_PARAMS)
        model = make_cox_model(config)
        model.fit(X_train_fixed, y_train_oracle)

        surv_fns = model.predict_survival_function(X_test_fixed)
        surv = np.row_stack([fn(model.unique_times_) for fn in surv_fns])

        spline = interp1d(
            model.unique_times_, surv, kind="linear",
            bounds_error=False,
            fill_value=(1.0, surv[:, -1]),
        )
        S = np.clip(spline(time_bins), 0.0, 1.0)
        surv_on_grid = pd.DataFrame(S, columns=time_bins)
        surv_on_grid[0.0] = 1.0

        true_eval = SurvivalEvaluator(surv_on_grid, time_bins, true_test_time, true_test_event)
        ibs_true = float(true_eval.integrated_brier_score(IPCW_weighted=False, num_points=num_points))

        for dgp_copula in dgp_copulas:
            for k_tau in k_taus:
                dgp_copula = str(dgp_copula)
                k_tau = float(k_tau)

                mult = float(alpha_c_mult_by_setting[(int(seed), dgp_copula, k_tau)])
                alpha_c = alpha_c_base * mult

                dgp_cens = DGP_Weibull_linear(
                    n_features, alpha_c, gamma_c, use_x=True,
                    device=device, dtype=dtype, coeff=beta_cens,
                )

                u_censor = sample_censor_uniform_given_event_uniform(
                    copula_name=dgp_copula,
                    k_tau=k_tau,
                    seed=int(seed),
                    event_v=v_event,
                    device=device,
                    dtype=dtype,
                )

                censor_time = dgp_cens.rvs(X, u_censor)
                censor_time = np.where(censor_time <= 0, 1.0, censor_time)
                observed_time = np.minimum(true_event_time, censor_time)
                observed_event = (true_event_time < censor_time).astype(int)

                censoring_rate = float(1.0 - observed_event.mean())

                train_time = observed_time[train_idx]
                train_event = observed_event[train_idx]
                test_time = observed_time[test_idx]
                test_event = observed_event[test_idx]

                ipcw_eval = SurvivalEvaluator(
                    surv_on_grid, time_bins,
                    test_time, test_event,
                    train_time, train_event,
                )
                ibs_ipcw = float(ipcw_eval.integrated_brier_score(num_points=num_points))

                for exp in experiments:
                    assumed_settings = assumed_settings_for_experiment(
                        exp=exp,
                        dgp_copula=dgp_copula,
                        dgp_tau=k_tau,
                        dep_true_tau=dep_true_tau,
                        dep_assumed_taus=dep_assumed_taus,
                    )

                    for assumed_copula, assumed_tau in assumed_settings:
                        theta = kendall_tau_to_theta(str(assumed_copula), float(assumed_tau))

                        dep_eval = DependentEvaluator(
                            surv_on_grid, time_bins,
                            test_time, test_event,
                            train_time, train_event,
                            copula_name=str(assumed_copula),
                            alpha=theta,
                        )
                        ibs_dep_bguw = float(dep_eval.integrated_brier_score(method="BG_UW", num_points=num_points))

                        rows.append({
                            "run_tag": RUN_TAG,
                            "prediction_model": "oracle_fixed_per_seed",
                            "experiment": str(exp),
                            "seed": int(seed),
                            "dgp_copula": str(dgp_copula),
                            "true_k_tau": float(k_tau),
                            "assumed_copula": str(assumed_copula),
                            "assumed_k_tau": float(assumed_tau),
                            "tau_abs_error": abs(float(k_tau) - float(assumed_tau)),
                            "alpha_c_mult": float(mult),
                            "alpha_c_used": float(alpha_c),
                            "censoring_rate": float(censoring_rate),
                            "ibs_true": ibs_true,
                            "ibs_ipcw": ibs_ipcw,
                            "ibs_dep_bguw": ibs_dep_bguw,
                            "bias_ipcw": ibs_ipcw - ibs_true,
                            "bias_dep_bguw": ibs_dep_bguw - ibs_true,
                            "err_ipcw": abs(ibs_true - ibs_ipcw),
                            "err_dep_bguw": abs(ibs_true - ibs_dep_bguw),
                        })

    return pd.DataFrame(rows)
if __name__ == "__main__":
    SEEDS = list(range(0, 10))
    PILOT_SEEDS = list(range(0, 10))
    COPULA_NAMES = ["clayton", "frank", "gaussian"]
    K_TAU = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    DEP_TRUE_TAU = 0.5
    DEP_ASSUMED_TAUS = K_TAU

    DGP_COPULAS = ["clayton", "frank", "gaussian"]
    experiments = ["family", "dep_clayton", "dep_frank", "gaussian"]

    TARGET_CENSOR = 0.50

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

    print("Using fixed event uniforms and seed-specific censoring calibration.")
    print(
        selected_calib_df.groupby(["copula_name", "k_tau"], as_index=False)
        .agg(
            alpha_c_mult_mean=("alpha_c_mult", "mean"),
            alpha_c_mult_std=("alpha_c_mult", "std"),
            censoring_rate_mean=("censoring_rate", "mean"),
            censoring_rate_std=("censoring_rate", "std"),
        )
        .to_string(index=False)
    )

    results_df = run_wrong_copula_experiment(
        data_cfg=data_cfg,
        seeds=SEEDS,
        dgp_copulas=DGP_COPULAS,
        k_taus=K_TAU,
        alpha_c_mult_by_setting=alpha_c_mult_by_setting,
        experiments=experiments,
        dep_true_tau=DEP_TRUE_TAU,
        dep_assumed_taus=DEP_ASSUMED_TAUS,
        device=device,
        dtype=dtype,
        train_frac=0.7,
        split_seed=0,
        num_points=10,
    )

    print("Actual censoring rates in generated results:")
    print(
        results_df.groupby(["dgp_copula", "true_k_tau"], as_index=False)
        .agg(
            censoring_rate_mean=("censoring_rate", "mean"),
            censoring_rate_std=("censoring_rate", "std"),
            censoring_rate_min=("censoring_rate", "min"),
            censoring_rate_max=("censoring_rate", "max"),
            ibs_true_mean=("ibs_true", "mean"),
            ibs_true_std=("ibs_true", "std"),
        )
        .to_string(index=False)
    )

    print("Oracle IBS stability by seed/copuIa family:")
    print(
        results_df.groupby(["seed", "dgp_copula"], as_index=False)
        .agg(ibs_true_min=("ibs_true", "min"), ibs_true_max=("ibs_true", "max"))
        .assign(ibs_true_range=lambda d: d["ibs_true_max"] - d["ibs_true_min"])
        .groupby("dgp_copula", as_index=False)
        .agg(mean_within_seed_range=("ibs_true_range", "mean"), max_within_seed_range=("ibs_true_range", "max"))
        .to_string(index=False)
    )

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)

    filename = f"{cfg.RESULTS_DIR}/synthetic_results_wrong_copula_fixed_model.csv"
    results_df.to_csv(filename, index=False)
