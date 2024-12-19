import random
import torch
from src.copula import Clayton_Bivariate
from src.data_loader import SingleEventSyntheticDataLoader
import pandas as pd
import numpy as np
import config as cfg
from SurvivalEVAL import SurvivalEvaluator

from src.models import Weibull_log_linear
from src.utility.survival import kendall_tau_to_theta, make_time_bins
from src.trainer import train_copula_model, predict_survival_curve

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

K_TAU = 0.25
SEED = 0
LINEAR = False
COPULA_NAME = "clayton"

if __name__ == "__main__":    
    dl = SingleEventSyntheticDataLoader().load_data(cfg.data_cfg, k_tau=K_TAU, copula_name=COPULA_NAME,
                                                    linear=LINEAR, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=SEED)
    
    for dataset in [train_dict, valid_dict, test_dict]: # put on device
        for key in ['X', 'T', 'E']:
            dataset[key] = dataset[key].to(device)
    
    theta_dgp = kendall_tau_to_theta(COPULA_NAME, K_TAU)
    print(f"Goal theta: {theta_dgp}")
    
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    # Train dependent model
    n_features = train_dict['X'].shape[1]
    dep_model1 = Weibull_log_linear(n_features, dtype=dtype, device=device) # censoring model
    dep_model2 = Weibull_log_linear(n_features, dtype=dtype, device=device) # event model
    copula = Clayton_Bivariate(2.0, 1e-4, dtype=dtype, device=device) # copula model
    dep_model1, dep_model2, copula = train_copula_model(dep_model1, dep_model2, train_dict,
                                                                 valid_dict, copula=copula, n_epochs=10000,
                                                                 lr=1e-3, batch_size=1024, verbose=True)
    survival_outputs, _, _ = predict_survival_curve(dep_model1, test_dict['X'], time_bins)
    survival_outputs = pd.DataFrame(survival_outputs.cpu().detach().numpy(), columns=np.array(time_bins.cpu()))
    dep_evaluator = SurvivalEvaluator(survival_outputs, time_bins, test_dict['T'], test_dict['E'],
                                      train_dict['T'], train_dict['E'])
    mae_margin = dep_evaluator.mae(method="Margin")
    copula_theta = float(copula.parameters()[0])
    print(f"Copula theta: {copula_theta}")
    print()
    