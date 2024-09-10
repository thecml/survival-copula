import numpy as np
import pandas as pd 
import torch
import matplotlib.pyplot as plt
import random
import os
#from model import CopulaMLP
from model import CopulaMLP
from loss import loss_DGP_Triple

from data_loader import SeerCompetingDataLoader, MimicCompetingDataLoader
from copula import Nested_Convex_Copula
from utility import kendall_tau_to_theta, make_time_bins, compute_l1_difference, preprocess_data
import config as cfg

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
    dl = SeerCompetingDataLoader()
    dl = dl.load_data()
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1,
                                                      test_size=0.2, random_state=SEED)
    n_events = dl.n_events
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

    # Make copula
    copula = Nested_Convex_Copula(['fr', 'fr', 'fr', 'fr', 'fr'],
                                  ['fr', 'fr', 'fr', 'fr', 'fr'],
                                  [0.01, 0.01, 0.01, 0.01, 0.01],
                                  [0.01, 0.01, 0.01, 0.01, 0.01],
                                  eps=1e-3, dtype=dtype, device=device)
    #from copula import Clayton_Triple, Frank_Triple
    #copula = Frank_Triple(0.01, eps=1e-3, dtype=dtype, device=device)
    
    # Make and train model
    n_epochs = 10000
    n_dists = 3 # 1 for MIMIC, 3 for SEER
    batch_size = 128 # High batch size (>1024) fails with NaN for the copula with k_tau=0.5 (linear)
    layers = [32]
    lr_dict = {'network': 0.0001, 'copula': 0.001}
    model = CopulaMLP(n_features, layers=layers, n_events=n_events,
                      n_dists=n_dists, copula=copula, dgps=None,
                      time_bins=time_bins, device=device)
    model.fit(train_dict, valid_dict, lr_dict=lr_dict, n_epochs=n_epochs,
              patience=50, batch_size=batch_size, verbose=False, weight_decay=0.01) # patience=50 for dgp/seer, patience=10 for mimic
    
    # Make predictions
    all_preds = []
    for i in range(n_events-1):
        model_preds = model.predict(test_dict['X'].to(device), time_bins, risk=i+1)
        model_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
        all_preds.append(model_preds)
    
    # Make evaluation for each event
    model_name = "mlp"
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