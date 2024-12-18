import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pandas as pd
from typing import List, Tuple, Optional, Union

def LOG(x):
    return torch.log(x+1e-20*(x<1e-20))

def loss_function(model1, model2, data, copula=None):
    s1 = model1.survival(data['T'], data['X'])
    s2 = model2.survival(data['T'], data['X'])
    f1 = model1.PDF(data['T'], data['X'])
    f2 = model2.PDF(data['T'], data['X'])
    w = torch.mean(data['E'])
    if copula is None:
        p1 = LOG(f1) + LOG(s2)
        p2 = LOG(f2) + LOG(s1)
    else:
        S = torch.cat([s1.reshape(-1,1), s2.reshape(-1,1)], dim=1).clamp(0.001,0.999)
        p1 = LOG(f1) + LOG(copula.conditional_cdf("u", S))
        p2 = LOG(f2) + LOG(copula.conditional_cdf("v", S))
    p1[torch.isnan(p1)] = 0
    p2[torch.isnan(p2)] = 0
    return -torch.mean(p1 * data['E'] + (1-data['E'])*p2)

def predict_survival_curve(model, x_test, time_bins, truth=False):
    device = torch.device("cpu")
    surv_estimate = torch.zeros((x_test.shape[0], time_bins.shape[0]), device=device)
    x_test = torch.tensor(x_test)
    time_bins = torch.tensor(time_bins)
    for i in range(time_bins.shape[0]):
        surv_estimate[:,i] = model.survival(time_bins[i], x_test)
    return surv_estimate, time_bins, time_bins.max()

def dependent_train_loop_linear(model1, model2, train_data, val_data,
                                n_iter, optimizer1='Adam', lr=1e-4,
                                verbose=False, copula=None):
    model1.enable_grad()
    model2.enable_grad()
    copula.enable_grad()
    
    min_val_loss = 1000
    optimizer = torch.optim.Adam([{"params": model1.parameters(), "lr": lr},
                                  {"params": model2.parameters(), "lr": lr},
                                  {"params": copula.parameters(), "lr": lr}])
    for itr in range(n_iter):
        optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, copula)
        loss.backward()
        for p in copula.parameters():
            p.grad = p.grad * 100
            p.grad.clamp_(torch.tensor([-0.5]), torch.tensor([0.5]))
        
        optimizer.step()
        
        for p in copula.parameters():
            if p <= 0.01:
                with torch.no_grad():
                    p[:] = torch.clamp(p, 0.01, 100)
        
        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, copula)
            if verbose and itr % 100 == 0:
                print(f"{val_loss} - {copula.theta}")
            
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr = 0
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                best_theta = copula.theta.detach().clone()
                min_val_loss = val_loss.detach().clone()
            else:
                stop_itr += 1
                if stop_itr == 2000:
                    break
                
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    copula.theta = best_theta
    
    return model1, model2, copula

def independent_train_loop_linear(model1, model2, train_data, val_data,
                                  n_iter, optimizer1='Adam', optimizer2='Adam',
                                  lr=1e-3, verbose=False):
    train_loss_log = []
    val_loss_log = []
    copula_log = torch.zeros((n_iter,))
    model1.enable_grad()
    model2.enable_grad()
    
    copula_grad_log = []
    mu_grad_log = [[], []]
    sigma_grad_log = [[], []]
    coeff_grad_log = [[], []]
    train_loss = []
    val_loss = []
    min_val_loss = 1000
    stop_itr = 0
    if optimizer1 == 'Adam':
        model_optimizer = torch.optim.Adam(list(model1.parameters()) + list(model2.parameters()), lr=lr)
    
    for itr in range(n_iter):
        model_optimizer.zero_grad()
        loss = loss_function(model1, model2, train_data, None)
        loss.backward()
        model_optimizer.step() 
        train_loss_log.append(loss.detach().clone())
        with torch.no_grad():
            val_loss = loss_function(model1, model2, val_data, None)
            val_loss_log.append(val_loss.detach().clone())
            if not torch.isnan(val_loss) and val_loss < min_val_loss:
                stop_itr = 0
                best_c1 = model1.coeff.detach().clone()
                best_c2 = model2.coeff.detach().clone()
                best_mu1 = model1.mu.detach().clone()
                best_mu2 = model2.mu.detach().clone()
                best_sig1 = model1.sigma.detach().clone()
                best_sig2 = model2.sigma.detach().clone()
                min_val_loss = val_loss.detach().clone()
            else:
                stop_itr += 1
                if stop_itr == 2000:  
                    break
                
    model1.mu = best_mu1
    model2.mu = best_mu2
    model1.sigma = best_sig1
    model2.sigma = best_sig2
    model1.coeff = best_c1
    model2.coeff = best_c2
    return model1, model2