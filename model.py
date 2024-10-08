import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from copula import Nested_Convex_Copula
from utility import compute_l1_difference

from loss import conditional_weibull_loss, conditional_weibull_loss_multi

def create_representation(inputdim, layers, activation, bias=True):
    if activation == 'ReLU6':
        act = nn.ReLU6()
    elif activation == 'ReLU':
        act = nn.ReLU()
    elif activation == 'SeLU':
        act = nn.SELU()
    elif activation == 'Tanh':
        act = nn.Tanh()

    modules = []
    prevdim = inputdim

    for hidden in layers:
        modules.append(nn.Linear(prevdim, hidden, bias=bias))
        modules.append(act)
        prevdim = hidden

    return nn.Sequential(*modules)

class DeepSurvivalMachinesTorch(torch.nn.Module):
    """"
    k: number of distributions
    temp: 1000 default, temprature for softmax
    discount: not used yet 
    """
    def __init__(self, inputdim, k, layers, temp, risks, discount=1.0):
        super(DeepSurvivalMachinesTorch, self).__init__()

        self.k = k
        self.temp = float(temp)
        self.discount = float(discount)
        
        self.risks = risks

        if layers is None: layers = []
        self.layers = layers

        if len(layers) == 0: lastdim = inputdim
        else: lastdim = layers[-1]

        self.act = nn.SELU()
        self.shape = nn.Parameter(-torch.ones(self.k * risks)) #(k * risk)
        self.scale = nn.Parameter(-torch.ones(self.k * risks))

        self.gate = nn.Linear(lastdim, self.k * self.risks, bias=False)
        self.scaleg = nn.Linear(lastdim, self.k * self.risks, bias=True)
        self.shapeg = nn.Linear(lastdim, self.k * self.risks, bias=True)
        self.embedding = create_representation(inputdim, layers, 'ReLU6') # ReLU6

    def forward(self, x):
        xrep = self.embedding(x)
        dim = x.shape[0]
        shape = torch.clamp(self.act(self.shapeg(xrep)) + self.shape.expand(dim,-1), min=-10, max=10)
        scale = torch.clamp(self.act(self.scaleg(xrep)) + self.scale.expand(dim,-1), min=-10, max=10)
        gate = self.gate(xrep) / self.temp
        outcomes = []
        for i in range(self.risks):
            outcomes.append((shape[:,i*self.k:(i+1)* self.k], scale[:,i*self.k:(i+1)* self.k], gate[:,i*self.k:(i+1)* self.k]))
            
        return outcomes
        
class CopulaMLP:
    def __init__(self, n_features, n_events, n_dists=5, layers=[32, 32],
                 time_bins=None, dgps=None, copula=None, device='cuda'):
        
        self.n_features = n_features
        self.copula = copula
        self.time_bins = time_bins
        self.dgps = dgps
        
        self.n_events = n_events
        self.device = device
        self.model = DeepSurvivalMachinesTorch(n_features, n_dists, layers, 1000,
                                               risks=n_events)
        
    def get_model(self):
        return self.model
    
    def get_copula(self):
        return self.copula
    
    def fit(self, train_dict, valid_dict, batch_size=1024, n_epochs=20000, 
            copula_grad_multiplier=1.0, copula_grad_clip=1.0,
            patience=100, optimizer='adam', weight_decay=0.001,
            lr_dict={'network': 5e-4, 'copula': 0.005},
            betas=(0.9, 0.999), use_wandb=False, verbose=False, multi=False):

        optim_dict = [{'params': self.model.parameters(), 'lr': lr_dict['network']}]
        if self.copula is not None:
            self.copula.enable_grad()
            optim_dict.append({'params': self.copula.parameters(), 'lr': lr_dict['copula']})
        
        if optimizer == 'adam':
            optimizer = torch.optim.Adam(optim_dict, betas=betas, weight_decay=weight_decay)
        elif optimizer == 'adamw':
            optimizer = torch.optim.AdamW(optim_dict, betas=betas, weight_decay=weight_decay)
        
        X = train_dict['X'].to(self.device)
        T = train_dict['T'].to(self.device)
        E = train_dict['E'].to(self.device)
        train_loader = DataLoader(TensorDataset(X, T, E), batch_size=batch_size, shuffle=True)
        
        self.model.to(self.device)
        min_delta = 0.0001
        best_val_loss = torch.tensor(float('inf')).to(self.device)
        epochs_no_improve = 0

        # Training loop with early stopping
        for itr in range(n_epochs):
            self.model.train()
            total_train_loss = 0
            num_batches = 0
            
            for xi, ti, ei in train_loader:
                optimizer.zero_grad()
                if self.copula is not None:
                    if multi:
                        loss = conditional_weibull_loss_multi(self.model, xi, ti, ei, self.device, copula=self.copula)
                    else:
                        loss = conditional_weibull_loss(self.model, xi, ti, ei, copula=self.copula)
                else:
                    if multi:
                        loss = conditional_weibull_loss_multi(self.model, xi, ti, ei, self.device, copula=None)
                    else:
                        loss = conditional_weibull_loss(self.model, xi, ti, ei)
                        
                loss.backward()
                    
                if (copula_grad_multiplier) and (self.copula is not None):
                    if isinstance(self.copula, Nested_Convex_Copula):
                        for p in self.copula.parameters()[:-2]:
                            p.grad = (p.grad * copula_grad_multiplier).clip(-1 * copula_grad_clip, 1 *copula_grad_clip)
                    else:
                        for p in self.copula.parameters():
                            p.grad = (p.grad * copula_grad_multiplier).clip(-1 * copula_grad_clip, 1 *copula_grad_clip)
                
                optimizer.step()
                
                if self.copula is not None:
                    if isinstance(self.copula, Nested_Convex_Copula):
                        for p in self.copula.parameters()[-2]:
                            if p < 0.01:
                                with torch.no_grad():
                                    p = torch.clamp(p, 0.01, 100)
                    else:
                        for p in self.copula.parameters():
                            if p < 0.01:
                                with torch.no_grad():
                                    p = torch.clamp(p, 0.01, 100)
            
                total_train_loss += loss.item()
                num_batches += 1
        
            avg_train_loss = total_train_loss / num_batches
            
            self.model.eval()
            with torch.no_grad():
                # Compute survival L1
                if self.dgps is not None:
                    n_samples = valid_dict['X'].shape[0]
                    total_survival_l1 = 0
                    for i in range(self.n_events-1):
                        truth_preds = torch.zeros((n_samples, self.time_bins.shape[0]), device=self.device)
                        for j in range(self.time_bins.shape[0]):
                            truth_preds[:,j] = self.dgps[i+1].survival(self.time_bins[j], valid_dict['X'].to(self.device))
                        model_preds = self.predict(valid_dict['X'].to(self.device), self.time_bins, i+1)
                        model_preds_th = torch.tensor(model_preds, device=self.device, dtype=torch.float64)
                        survival_l1 = float(compute_l1_difference(truth_preds, model_preds_th,
                                                                n_samples, steps=self.time_bins))
                        total_survival_l1 += survival_l1
                    total_survival_l1 /= (self.n_events-1)
                else:
                    total_survival_l1 = 0.0
                
                if self.copula is not None:
                    if multi:
                        val_loss = conditional_weibull_loss_multi(self.model, valid_dict['X'].to(self.device),
                                                            valid_dict['T'].to(self.device), valid_dict['E'].to(self.device),
                                                            self.device, copula=self.copula)
                    else:
                        val_loss = conditional_weibull_loss(self.model, valid_dict['X'].to(self.device),
                                                            valid_dict['T'].to(self.device), valid_dict['E'].to(self.device),
                                                            elbo=True, copula=self.copula)
                else:
                    if multi:
                        val_loss = conditional_weibull_loss_multi(self.model, valid_dict['X'].to(self.device),
                                                            valid_dict['T'].to(self.device), valid_dict['E'].to(self.device),
                                                            self.device, copula=None)
                    else:
                        val_loss = conditional_weibull_loss(self.model, valid_dict['X'].to(self.device),
                                                            valid_dict['T'].to(self.device), valid_dict['E'].to(self.device))
                
                #scheduler.step(val_loss)

            if verbose:
                if self.copula is not None:
                    if isinstance(self.copula, Nested_Convex_Copula):
                        params = [np.around(float(param), 5) for param in self.copula.parameters()[:-2]]
                    else:
                        params = [np.around(float(param), 5) for param in self.copula.parameters()]
                    print(itr, "/", n_epochs, "train_loss: ", round(avg_train_loss, 4),
                        "val_loss: ", round(val_loss.item(), 4),
                        "val_l1: ", round(total_survival_l1, 4),
                        "min_val_loss: ", round(best_val_loss.item(), 4),
                        "copula: ", params)
                else:
                    print(itr, "/", n_epochs, "train_loss: ", round(avg_train_loss, 4),
                        "val_loss: ", round(val_loss.item(), 4),
                        "val_l1: ", round(total_survival_l1, 4),
                        "min_val_loss: ", round(best_val_loss.item(), 4))

            # Check for early stopping
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0
                if self.copula is not None:
                    best_theta = [p.detach().clone().cpu() for p in self.copula.parameters()]
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping at iteration {itr}, best val loss: {best_val_loss}")
                break
            
        if self.copula is not None:
            self.copula.set_params(best_theta)
                    
    def predict(self, x_test, time_bins, risk=0):
        t = list(time_bins.cpu().numpy())
        params = self.model.forward(x_test)
        
        shape, scale, logits = params[risk][0], params[risk][1], params[risk][2]
        k_ = shape
        b_ = scale

        squish = nn.LogSoftmax(dim=1)
        logits = squish(logits)
        
        t_horz = torch.tensor(time_bins).double().to(logits.device)
        t_horz = t_horz.repeat(shape.shape[0], 1)
        
        cdfs = []
        for j in range(len(time_bins)):

            t = t_horz[:, j]
            lcdfs = []

            for g in range(self.model.k):

                k = k_[:, g]
                b = b_[:, g]
                s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
                lcdfs.append(s)

            lcdfs = torch.stack(lcdfs, dim=1)
            lcdfs = lcdfs+logits
            lcdfs = torch.logsumexp(lcdfs, dim=1)
            cdfs.append(lcdfs.detach().cpu().numpy())
        
        return np.exp(np.array(cdfs)).T
                    