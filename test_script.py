import numpy as np
import pandas as pd 
import torch
import matplotlib.pyplot as plt
import random
from utility import make_time_bins, compute_l1_difference
from model import CopulaMLP
from loss import loss_DGP_Triple

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# Setup precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data_cfg = {
    "alpha_e1": 16,
    "alpha_e2": 18,
    "alpha_e3": 16,
    "gamma_e1": 4,
    "gamma_e2": 4,
    "gamma_e3": 4,
    "n_hidden": 8,
    "n_events": 3,
    "n_samples": 20000,
    "n_features": 10
}

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def loss_triple(model1, model2, model3, data, copula=None):#estimates the joint loss
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    s3 = model3.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    f3 = model3.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2) + LOG(s3)
        p2 = LOG(f2) + LOG(s1) + LOG(s3)
        p3 = LOG(f3) + LOG(s1) + LOG(s2)
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1), s3.reshape(-1,1)], dim=1) #.clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
        p3 = LOG(f3) + LOG(copula.conditional_cdf("w", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    p3[torch.isnan(p3)] = 0
    e1 = (data['E'] == 0)*1.0
    e2 = (data['E'] == 1)*1.0
    e3 = (data['E'] == 2)*1.0
    loss = torch.sum(p1 * e1) + torch.sum(p2*e2) + torch.sum(p3*e3)
    loss = -loss/data['E'].shape[0]
    return loss

def single_loss(model, data, event_name='t1'):#estimates loss assuming every thing is observed/no censoring for checking 
    f = model.PDF(data[event_name], data['X'])
    return -torch.mean(LOG(f))

if __name__ == "__main__":
    from data_loader import CompetingRiskSyntheticDataLoader
    from copula import Nested_Convex_Copula, Clayton_Bivariate, Clayton_Triple, Frank_Triple
    from distributions import Weibull_log_linear, Weibull_nonlinear, EXP_nonlinear
    from utility import kendall_tau_to_theta
    
    k_tau = 0.5 # 0.25
    dl = CompetingRiskSyntheticDataLoader().load_data(data_cfg, k_tau=k_tau, copula_name="frank",
                                                      linear=True, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=0)
    
    for dataset in [train_dict, valid_dict, test_dict]: # put on device
        for key in ['X', 'T', 'E']:
            dataset[key] = dataset[key].to(device)
    
    n_features = train_dict['X'].shape[1]
    n_events = dl.n_events
    dgps = dl.dgps
    
    # Ktau/Theta (frank)
    # 0 - 0
    # 0.25 - 0.66
    # 0.5 - 2.0
    
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
    
    copula_dgp = 'clayton'
    theta_dgp = kendall_tau_to_theta(copula_dgp, k_tau)
    print(f"Goal theta: {theta_dgp}")
    eps = 1e-4
    
    time_bins = make_time_bins(train_dict['T'].cpu(), event=None, dtype=dtype).to(device)
    time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
    
    #remove the comment to check the percentage of each event
    #plt.hist(train_dict['E'].cpu().numpy())
    #plt.show()
    #assert 0

    #copula for estimation
    copula_start_point = 2.0
    #copula = Clayton(torch.tensor([copula_start_point]),eps, DEVICE)
    #copula = Frank(torch.tensor([copula_start_point]),eps, DEVICE)
    #copula = Clayton(torch.tensor([copula_start_point]),eps, DEVICE)
    #copula = NestedClayton(torch.tensor([copula_start_point]),torch.tensor([copula_start_point]),eps,eps, DEVICE)
    copula = Nested_Convex_Copula(['fr'], ['fr'], [0.01], [0.01], eps=1e-3, dtype=dtype, device=device)
    #copula = Clayton_Triple(theta=2.0, eps=1e-3, dtype=dtype, device=device)
    
    # Make and train model
    n_epochs = 10000
    n_dists = 1
    batch_size = 128 # High batch size (>1024) fails with NaN for the copula with k_tau=0.5 (linear)
    layers = [128, 256]
    lr_dict = {'network': 0.0001, 'copula': 0.001} # 0.001
    model = CopulaMLP(n_features, layers=layers, n_events=n_events,
                      n_dists=n_dists, copula=copula, dgps=dgps,
                      time_bins=time_bins, device=device)
    model.fit(train_dict, valid_dict, lr_dict=lr_dict, n_epochs=n_epochs,
              patience=25, batch_size=batch_size, verbose=True, weight_decay=0.01)

    # Predict
    all_preds = []
    for i in range(n_events):
        model_preds = model.predict(test_dict['X'].to(device), time_bins, risk=i)
        model_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
        all_preds.append(model_preds)
    
    # Compute Survival-L1
    for event_id, surv_preds in enumerate(all_preds):
        n_samples = test_dict['X'].shape[0]
        truth_preds = torch.zeros((n_samples, time_bins.shape[0]), device=device)
        for i in range(time_bins.shape[0]):
            truth_preds[:,i] = dgps[event_id].survival(time_bins[i], test_dict['X'].to(device))
        model_preds_th = torch.tensor(surv_preds.values, device=device, dtype=dtype)
        survival_l1 = float(compute_l1_difference(truth_preds, model_preds_th,
                                                  n_samples, steps=time_bins))
        print(f'{event_id+1}: ' + f'{survival_l1}')
