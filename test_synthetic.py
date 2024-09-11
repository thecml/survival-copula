from SurvivalEVAL import LifelinesEvaluator
import numpy as np
import pandas as pd 
import torch
import matplotlib.pyplot as plt
import random
import os
from model import CopulaMLP
from loss import loss_DGP_Triple

from data_loader import CompetingRiskSyntheticDataLoader
from copula import Nested_Convex_Copula, Clayton_Bivariate, Clayton_Triple, Frank_Triple, Frank_Bivariate
from distributions import Weibull_log_linear, Weibull_nonlinear, EXP_nonlinear
from utility import kendall_tau_to_theta, make_time_bins, compute_l1_difference
import config as cfg

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CONSTS
K_TAU = 0.5
SEED = 0
LINEAR = False
COPULA_NAME = "frank"

# Ktau/Theta (frank)
# 0 - 0
# 0.25 - 0.66
# 0.5 - 2.0

if __name__ == "__main__":    
    dl = CompetingRiskSyntheticDataLoader().load_data(cfg.data_cfg, k_tau=K_TAU, copula_name=COPULA_NAME,
                                                      linear=LINEAR, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=SEED)
    
    for dataset in [train_dict, valid_dict, test_dict]: # put on device
        for key in ['X', 'T', 'E']:
            dataset[key] = dataset[key].to(device)
    
    n_features = train_dict['X'].shape[1]
    n_events = dl.n_events
    dgps = dl.dgps
    
    dgp1 = dgps[0]
    dgp2 = dgps[1]
    dgp3 = dgps[2]
    print(f'minimum possibe loss train: {loss_DGP_Triple(train_dict, dgp1, dgp2, dgp3, None)}')#tau = 0 --> copula None
    print(f'minimum possibe loss val: {loss_DGP_Triple(valid_dict, dgp1, dgp2, dgp3, None)}')#tau = 0 --> copula None
    print(f'minimum possibe loss test: {loss_DGP_Triple(test_dict, dgp1, dgp2, dgp3, None)}')#tau = 0 --> copula None
    copula_test = Nested_Convex_Copula(['fr'], ['fr'], [2.0], [2.0], eps=1e-3, dtype=dtype, device=device)
    print(f'minimum possibe loss with copula train: {loss_DGP_Triple(train_dict, dgp1, dgp2, dgp3, copula_test)}')#tau = 0 --> best thing model can achieve
    print(f'minimum possibe loss with copula val: {loss_DGP_Triple(valid_dict, dgp1, dgp2, dgp3, copula_test)}')#tau = 0 --> best thing model can achieve
    print(f'minimum possibe loss with copula test: {loss_DGP_Triple(test_dict, dgp1, dgp2, dgp3, copula_test)}')#tau = 0 --> best thing model can achieve
    
    copula_dgp = 'frank'
    theta_dgp = kendall_tau_to_theta(copula_dgp, K_TAU)
    print(f"Goal theta: {theta_dgp}")
    eps = 1e-4
    
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    #remove the comment to check the percentage of each event
    #plt.hist(train_dict['E'].cpu().numpy())
    #plt.show()
    #assert 0

    # Make copula
    copula = Nested_Convex_Copula(['fr', 'fr', 'fr', 'fr'],
                                  ['fr', 'fr', 'fr', 'fr'],
                                  [0.01, 0.01, 0.01, 0.01],
                                  [0.01, 0.01, 0.01, 0.01],
                                  eps=1e-3, dtype=dtype, device=device)
    
    # Make and train model
    n_epochs = 10000 #10000
    n_dists = 3
    batch_size = 128 # High batch size (>1024) fails with NaN for the copula with k_tau=0.5 (linear)
    layers = [128, 128]
    lr_dict = {'network': 0.0001, 'copula': 0.001}
    model = CopulaMLP(n_features, layers=layers, n_events=n_events,
                      n_dists=n_dists, copula=copula, dgps=dgps,
                      time_bins=time_bins, device=device)
    model.fit(train_dict, valid_dict, lr_dict=lr_dict, n_epochs=n_epochs,
              patience=50, batch_size=batch_size, verbose=True, weight_decay=0.01)

    # Make predictions (include censoring)
    all_preds = []
    for i in range(n_events):
        model_preds = model.predict(test_dict['X'].to(device), time_bins, risk=i)
        model_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
        all_preds.append(model_preds)
    
    # Make evaluation for each event
    model_name = "mlp"
    for event_id, surv_preds in enumerate(all_preds):
        n_samples = test_dict['X'].shape[0]
        truth_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
        for i in range(time_bins.shape[0]):
            truth_preds[:,i] = dgps[event_id].survival(time_bins[i], test_dict['X'].to(device))
        model_preds_th = torch.tensor(surv_preds.values, device=device, dtype=dtype)
        survival_l1 = float(compute_l1_difference(truth_preds, model_preds_th,
                                                  n_samples, steps=time_bins))
        
        n_train_samples = len(train_dict['X'])
        n_test_samples= len(test_dict['X'])
        y_train_time = train_dict['T']
        y_train_event = (train_dict['E'] == event_id)*1.0
        y_test_time = test_dict['T']
        y_test_event = (test_dict['E'] == event_id)*1.0
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score()
        mae_hinge = lifelines_eval.mae(method="Hinge")
        mae_margin = lifelines_eval.mae(method="Margin")
        mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
        d_calib = lifelines_eval.d_calibration()[0]
        
        metrics = [ci, ibs, mae_hinge, mae_margin, mae_pseudo, survival_l1, d_calib]
        print(f'{model_name}: ' + f'{metrics}')