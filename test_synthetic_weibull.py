import torch
from calculate_mae_dependent import mae_dependent
from copula import Clayton_Bivariate, Frank_Bivariate
from data_loader import MetabricDataLoader, SingleEventSyntheticDataLoader
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations.util import predict_median_survival_time
from sklearn.model_selection import train_test_split

from models import Weibull_log_linear, LogNormalCox_linear
from loss import loss_double
from make_semi_synthetic import combine_data_with_censor, make_synthetic_censoring
from plot import compare_km_curves
from utility import convert_to_structured, kendall_tau_to_theta, make_time_bins
from trainer import independent_train_loop_linear, dependent_train_loop_linear, predict_survival_curve

import config as cfg
# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CONSTS
K_TAU = 0.25
SEED = 0
LINEAR = True
COPULA_NAME = "clayton"

# Ktau/Theta (frank)
# 0 - 0
# 0.25 - 0.66
# 0.5 - 2.0

if __name__ == "__main__":    
    dl = SingleEventSyntheticDataLoader().load_data(cfg.data_cfg, k_tau=K_TAU, copula_name=COPULA_NAME,
                                                    linear=LINEAR, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=SEED)
    
    for dataset in [train_dict, valid_dict, test_dict]: # put on device
        for key in ['X', 'T', 'E']:
            dataset[key] = dataset[key].to(device)
    
    n_features = train_dict['X'].shape[1]
    dgps = dl.dgps
    n_events = 2
    
    dgp1 = dgps[0]
    dgp2 = dgps[1]
    
    theta_dgp = kendall_tau_to_theta(COPULA_NAME, K_TAU)
    print(f"Goal theta: {theta_dgp}")
    eps = 1e-4
    
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Train dependent model
    dep_model1 = Weibull_log_linear(n_features, dtype=dtype, device=device) # censoring model
    dep_model2 = Weibull_log_linear(n_features, dtype=dtype, device=device) # event model
    copula = Clayton_Bivariate(4.0, 1e-4, dtype=dtype, device=device) # copula model
    dep_model1, dep_model2, copula = dependent_train_loop_linear(dep_model1, dep_model2, train_dict,
                                                                 valid_dict, copula=copula, n_iter=200000,
                                                                 lr=1e-3, verbose=True)
    survival_outputs, _, _ = predict_survival_curve(dep_model1, test_dict['X'], time_bins)
    survival_outputs = pd.DataFrame(survival_outputs, columns=np.array(time_bins))
    dep_evaluator = SurvivalEvaluator(survival_outputs, time_bins, test_dict['T'], test_dict['E'],
                                      train_dict['T'], train_dict['E'])
    mae_margin = dep_evaluator.mae(method="Margin")
    copula_theta = float(copula.parameters()[0])
    print(f"Copula theta: {copula_theta}")
    print()
    