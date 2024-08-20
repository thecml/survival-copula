import numpy as np 
import torch
import matplotlib.pyplot as plt
import random

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
    
    k_tau = 0.25

    dl = CompetingRiskSyntheticDataLoader().load_data(data_cfg, k_tau=k_tau, copula_name="frank",
                                                      linear=False, device=device, dtype=dtype)
    train_dict, valid_dict, test_dict = dl.split_data(train_size=0.7, valid_size=0.1, test_size=0.2,
                                                      random_state=0)
    
    for dataset in [train_dict, valid_dict, test_dict]: # put on device
        for key in ['X', 'T', 'E']:
            dataset[key] = dataset[key].to(device)
    
    n_features = train_dict['X'].shape[1]
    n_events = dl.n_events
    dgps = dl.dgps
    
    dgp1 = dgps[0]
    dgp2 = dgps[1]
    dgp3 = dgps[2]

    copula_dgp = 'clayton'
    theta_dgp = kendall_tau_to_theta(copula_dgp, k_tau)
    print(f"Goal theta: {theta_dgp}")
    eps = 1e-4
    
    #remove the comment to check the percentage of each event 
    #plt.hist(train_dict['E'])
    #plt.show()
    #assert 0

    #copula for estimation
    copula_start_point = 2.0
    #copula = Clayton(torch.tensor([copula_start_point]),eps, DEVICE)
    #copula = Frank(torch.tensor([copula_start_point]),eps, DEVICE)
    #copula = Clayton(torch.tensor([copula_start_point]),eps, DEVICE)
    #copula = NestedClayton(torch.tensor([copula_start_point]),torch.tensor([copula_start_point]),eps,eps, DEVICE)
    copula = Nested_Convex_Copula(['fr'], ['fr'], [2.0], [2.0], eps=1e-3, dtype=dtype, device=device)
    #copula = Clayton_Triple(theta=2.0, eps=1e-3, dtype=dtype, device=device)
    
    #indep_model1 = Weibull_log_linear(n_features, device=device)
    #indep_model2 = Weibull_log_linear(n_features, device=device)
    #indep_model3 = Weibull_log_linear(n_features, device=device)
    indep_model1 = Weibull_nonlinear(n_features, n_hidden=8, risk_function=torch.nn.ReLU(), device=device, dtype=dtype)
    indep_model2 = Weibull_nonlinear(n_features, n_hidden=8, risk_function=torch.nn.ReLU(), device=device, dtype=dtype)
    indep_model3 = Weibull_nonlinear(n_features, n_hidden=8, risk_function=torch.nn.ReLU(), device=device, dtype=dtype)
    
    indep_model1.enable_grad()
    indep_model2.enable_grad()
    indep_model3.enable_grad()
    copula.enable_grad()

    #training loop
    optimizer = torch.optim.Adam([{"params": indep_model1.parameters(), "lr": 1e-3, "weight_decay":0.001},
                                  {"params": indep_model2.parameters(), "lr": 1e-3, "weight_decay":0.001},
                                  {"params": indep_model3.parameters(), "lr": 1e-3, "weight_decay":0.001},
                                  {"params": copula.parameters(), "lr": 1e-3, "weight_decay":0.001}])
    n_epochs = 50000
    min_delta = 0.001
    best_val_loss = torch.tensor(float('inf')).to(device)
    epochs_no_improve = 0
    patience = 2500
    #copula = None
    
    for i in range(n_epochs):
        optimizer.zero_grad()
        loss = loss_triple(indep_model1, indep_model2, indep_model3, train_dict, copula)
        loss.backward()
        
        copula_grad_multiplier = 1.0
        copula_grad_clip = 1.0
        if (copula_grad_multiplier) and (copula is not None):
            if isinstance(copula, Nested_Convex_Copula):
                for p in copula.parameters()[:-2]:
                    p.grad = (p.grad * copula_grad_multiplier).clip(-1 * copula_grad_clip, 1 *copula_grad_clip)
            else:
                for p in copula.parameters():
                    p.grad = (p.grad * copula_grad_multiplier).clip(-1 * copula_grad_clip, 1 *copula_grad_clip)
                     
        optimizer.step()
        
        if copula is not None:
            if isinstance(copula, Nested_Convex_Copula):
                for p in copula.parameters()[-2]:
                    if p < 0.01:
                        with torch.no_grad():
                            p = torch.clamp(p, 0.01, 100)
            else:
                for p in copula.parameters():
                    if p < 0.01:
                        with torch.no_grad():
                            p = torch.clamp(p, 0.01, 100)
        if i % 100 == 0:
            val_loss = loss_triple(indep_model1, indep_model2, indep_model3, valid_dict, copula)
            if copula is not None:
                if isinstance(copula, Nested_Convex_Copula):
                    params = [np.around(float(param), 5) for param in copula.parameters()[:-2]]
                else:
                    params = [np.around(float(param), 5) for param in copula.parameters()]
                print(i, "/", n_epochs, "train_loss: ", round(loss.item(), 4),
                    "val_loss: ", round(best_val_loss.item(), 4),
                    "copula: ", params)
            else:
                print(i, "/", n_epochs, "train_loss: ", round(loss.item(), 4),
                    "val_loss: ", round(val_loss.item(), 4),
                    "min_val_loss: ", round(best_val_loss.item(), 4))

            
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if copula is not None:
                best_theta = [p.detach().clone().cpu() for p in copula.parameters()]
        else:
            epochs_no_improve += 1
            
        if epochs_no_improve >= patience:
            print(f"Early stopping at iteration {i}, best val loss: {best_val_loss}")
            break

    print("###############################################################")
    #NLL of all of the events together
    print(loss_triple(indep_model1, indep_model2, indep_model3, test_dict, copula))
    #check the dgp performance
    #copula.theta = torch.tensor([theta_dgp], device=device)
    copula = Clayton_Triple(theta=theta_dgp, eps=1e-3, dtype=dtype, device=device)
    print(loss_triple(dgp1, dgp2, dgp3, test_dict, copula))
    
    from utility import surv_diff
    print(surv_diff(dgp1, indep_model1, test_dict['X'], 200))
    print(surv_diff(dgp2, indep_model2, test_dict['X'], 200))
    print(surv_diff(dgp3, indep_model3, test_dict['X'], 200))
