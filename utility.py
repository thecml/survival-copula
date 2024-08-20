import torch
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from skmultilearn.model_selection import iterative_train_test_split

def Survival(model, estimate, x, steps=200):
    device = x.device
    u = torch.ones((x.shape[0],), device=device)*0.001
    time_steps = torch.linspace(1e-4,1,steps,device=device).reshape(1,-1).repeat(x.shape[0],1)
    t_max_model = model.rvs(x, u)
    #t_max_estimate = estimate.rvs(x, u)#.reshape(-1,1).repeat(1,100)
    #e = (t_max_model < t_max_estimate).type(torch.float32)
    t_max = t_max_model#e * t_max_estimate + (1-e) * t_max_model
    
    t_max = t_max.reshape(-1,1).repeat(1,steps)
    
    time_steps = t_max * time_steps
    surv1 = torch.zeros((x.shape[0], steps),device=device)
    surv2 = torch.zeros((x.shape[0], steps),device=device)
    for i in range(steps):
        surv1[:,i] = model.survival(time_steps[:,i],x)
        surv2[:,i] = estimate.survival(time_steps[:,i], x)
    
    return surv1, surv2, time_steps, t_max_model

def surv_diff(model, estimate, x, steps):
    surv1, surv2, time_steps, t_m = Survival(model, estimate, x, steps)
    
    
    integ = torch.sum(torch.diff(torch.cat([torch.zeros((surv1.shape[0],1), device=x.device), time_steps], dim=1))*(torch.abs(surv1-surv2)), dim=1)
    #integ2 = torch.sum(torch.diff(torch.cat([torch.zeros(surv1.shape[0],1), time_steps], dim=1))*(torch.abs(surv1)), dim=1)
    #return torch.mean(integ/integ2)
    #print(torch.std(integ/t_m))
    #print(integ.shape)
    #print((integ/t_m).shape)
    return torch.mean(integ/t_m)#time_steps[:,-1])#add std

def kendall_tau_to_theta(copula_name, k_tau):
    if copula_name == "clayton":
        return 2 * k_tau / (1 - k_tau)
    elif copula_name == "frank":
        return -np.log(1 - k_tau) / k_tau
    elif copula_name == "gumbel":
        return 1 / (1 - k_tau)
    else:
        raise ValueError('Copula not implemented')
    
def theta_to_kendall_tau(copula_name, theta):
    if copula_name == "clayton":
        return theta / (theta + 2)
    elif copula_name == "frank":
        return 1 - 4 * ((1 - np.exp(-theta)) / theta)
    elif copula_name == "gumbel":
        return (theta - 1) / theta
    else:
        raise ValueError('Copula not implemented')
    
def multilabel_train_test_split(X, y, test_size, random_state=None):
    """Iteratively stratified train/test split
    (Add random_state to scikit-multilearn iterative_train_test_split function)
    See this paper for details: https://link.springer.com/chapter/10.1007/978-3-642-23808-6_10
    """
    X, y = shuffle(X, y, random_state=random_state)
    X_train, y_train, X_test, y_test = iterative_train_test_split(X, y, test_size=test_size)
    return X_train, y_train, X_test, y_test

def make_stratified_split(
        df: pd.DataFrame,
        stratify_colname: str = 'event',
        frac_train: float = 0.5,
        frac_valid: float = 0.0,
        frac_test: float = 0.5,
        random_state: int = None
) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
    '''Courtesy of https://github.com/shi-ang/BNN-ISD/tree/main'''
    assert frac_train >= 0 and frac_valid >= 0 and frac_test >= 0, "Check train validation test fraction."
    frac_sum = frac_train + frac_valid + frac_test
    frac_train = frac_train / frac_sum
    frac_valid = frac_valid / frac_sum
    frac_test = frac_test / frac_sum

    X = df.values  # Contains all columns.
    columns = df.columns
    if stratify_colname == 'event':
        stra_lab = df[stratify_colname]
    elif stratify_colname == 'time':
        stra_lab = df[stratify_colname]
        bins = np.linspace(start=stra_lab.min(), stop=stra_lab.max(), num=20)
        stra_lab = np.digitize(stra_lab, bins, right=True).reshape(-1, 1)
    elif stratify_colname == "both":
        t = df["time"]
        bins = np.linspace(start=t.min(), stop=t.max(), num=20)
        t = np.digitize(t, bins, right=True)
        e = df["event"]
        stra_lab = np.stack([t, e], axis=1)
    else:
        raise ValueError("unrecognized stratify policy")

    x_train, _, x_temp, y_temp = multilabel_train_test_split(X, y=stra_lab, test_size=(1.0 - frac_train),
                                                             random_state=random_state)
    if frac_valid == 0:
        x_val, x_test = [], x_temp
    else:
        x_val, _, x_test, _ = multilabel_train_test_split(x_temp, y=y_temp,
                                                          test_size=frac_test / (frac_valid + frac_test),
                                                          random_state=random_state)
    df_train = pd.DataFrame(data=x_train, columns=columns)
    df_val = pd.DataFrame(data=x_val, columns=columns)
    df_test = pd.DataFrame(data=x_test, columns=columns)
    assert len(df) == len(df_train) + len(df_val) + len(df_test)
    return df_train, df_val, df_test