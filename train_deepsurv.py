import numpy as np
import pandas as pd 
import torch
import matplotlib.pyplot as plt
import random
import os
from model import CopulaMLP
from loss import loss_DGP_Triple

from data_loader import SeerCompetingDataLoader, MimicCompetingDataLoader
from copula import Nested_Convex_Copula
from utility import dotdict, kendall_tau_to_theta, make_time_bins, compute_l1_difference, preprocess_data
import config as cfg
from deepsurv import DeepSurv, train_deepsurv_model, make_deepsurv_prediction
from scipy.interpolate import interp1d

from SurvivalEVAL.Evaluator import LifelinesEvaluator

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CONSTS
SEED = 0

if __name__ == "__main__":    
    # Load and split data
    dl = MimicCompetingDataLoader()
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1,
                                                      test_size=0.2, random_state=SEED)
    n_events = dl.n_events - 1
    n_features = train_dict['X'].shape[1]

    # Preprocess data
    cat_features = dl.cat_features
    num_features = dl.num_features
    n_features = train_dict['X'].shape[1]
    X_train = pd.DataFrame(train_dict['X'], columns=dl.columns)
    X_valid = pd.DataFrame(valid_dict['X'], columns=dl.columns)
    X_test = pd.DataFrame(test_dict['X'], columns=dl.columns)
    X_train, X_valid, X_test= preprocess_data(X_train, X_valid, X_test, cat_features,
                                              num_features, as_array=True)
    train_dict['X'] = torch.tensor(X_train, device=device, dtype=dtype)
    valid_dict['X'] = torch.tensor(X_valid, device=device, dtype=dtype)
    test_dict['X'] = torch.tensor(X_test, device=device, dtype=dtype)
    n_samples = train_dict['X'].shape[0]
    
    # Put on device
    for dataset in [train_dict, valid_dict, test_dict]:
        for key in ['X', 'T', 'E']:
            dataset[key] = dataset[key].to(device)
    
    # Make time bins
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))

    config = dotdict(cfg.DEEPSURV_PARAMS)
    trained_models = []
    for i in range(n_events):
        model = DeepSurv(in_features=n_features, config=config)
        data_train = pd.DataFrame(train_dict['X'].cpu().numpy())
        data_train['time'] = train_dict['T'].cpu().numpy()
        data_train['event'] = (train_dict['E'].cpu().numpy() == i+1)*1.0
        data_valid = pd.DataFrame(valid_dict['X'].cpu().numpy())
        data_valid['time'] = valid_dict['T'].cpu().numpy()
        data_valid['event'] = (valid_dict['E'].cpu().numpy() == i+1)*1.0
        model = train_deepsurv_model(model, data_train, data_valid, time_bins, config=config,
                                     random_state=0, reset_model=True, device=device, dtype=dtype)
        trained_models.append(model)

    # Make predictions
    all_preds = []
    for trained_model in trained_models:
        preds, time_bins_model = make_deepsurv_prediction(trained_model, test_dict['X'].to(device),
                                                          config=config, dtype=dtype)
        spline = interp1d(time_bins_model.cpu().numpy(), preds.cpu().numpy(),
                          kind='linear', fill_value='extrapolate')
        preds = pd.DataFrame(spline(time_bins.cpu().numpy()),
                             columns=time_bins.cpu().numpy())
        all_preds.append(preds)

    # Make evaluation for each event
    model_name = "deepsurv"
    model_results = pd.DataFrame()
    for event_id, surv_preds in enumerate(all_preds):
        n_train_samples = len(train_dict['X'])
        n_test_samples= len(test_dict['X'])
        y_train_time = train_dict['T']
        y_train_event = (train_dict['E'] == event_id+1)*1.0
        y_test_time = test_dict['T']
        y_test_event = (test_dict['E'] == event_id+1)*1.0
        lifelines_eval = LifelinesEvaluator(surv_preds.T, y_test_time, y_test_event,
                                            y_train_time, y_train_event)
        
        ci = lifelines_eval.concordance()[0]
        ibs = lifelines_eval.integrated_brier_score()
        mae_hinge = lifelines_eval.mae(method="Hinge")
        mae_margin = lifelines_eval.mae(method="Margin")
        mae_pseudo = lifelines_eval.mae(method="Pseudo_obs")
        d_calib = lifelines_eval.d_calibration()[0]
        
        metrics = [ci, ibs, mae_hinge, mae_margin, mae_pseudo, d_calib]
        print(f'{model_name}: ' + f'{metrics}')