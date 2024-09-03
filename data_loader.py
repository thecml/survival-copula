import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from pycop import simulation
import torch
from utility import kendall_tau_to_theta
from utility import make_stratified_split
from dgp import DGP_Weibull_linear, DGP_Weibull_nonlinear

class BaseDataLoader(ABC):
    """
    Base class for data loaders.
    """
    def __init__(self):
        """Initilizer method that takes a file path, file name,
        settings and optionally a converter"""
        self.X: pd.DataFrame = None
        self.y_t: List[np.ndarray] = None
        self.y_e: List[np.ndarray] = None
        self.num_features: List[str] = None
        self.cat_features: List[str] = None
        self.min_time = None
        self.max_time = None
        self.n_events = None
        self.params = None

    @abstractmethod
    def load_data(self, n_samples) -> None:
        """Loads the data from a data set at startup"""
        
    @abstractmethod
    def split_data(self) -> None:
        """Loads the data from a data set at startup"""

    def get_data(self) -> pd.DataFrame:
        """
        This method returns the features and targets
        :returns: X, y_t and y_e
        """
        return self.X, self.y_t, self.y_e

    def get_features(self) -> List[str]:
        """
        This method returns the names of numerical and categorial features
        :return: the columns of X as a list
        """
        return self.num_features, self.cat_features

    def _get_num_features(self, data) -> List[str]:
        return data.select_dtypes(include=np.number).columns.tolist()

    def _get_cat_features(self, data) -> List[str]:
        return data.select_dtypes(['object']).columns.tolist()

class SingleEventSyntheticDataLoader(BaseDataLoader):
    def load_data(self, data_config, copula_name='clayton', k_tau=0,
                  linear=True, device='cpu', dtype=torch.float64):
        """
        This method generates synthetic data for single event (and censoring)
        DGP1: Data generation process for event
        DGP2: Data generation process for censoring
        """
        alpha_e1 = data_config['alpha_e1']
        alpha_e2 = data_config['alpha_e2']
        gamma_e1 = data_config['gamma_e1']
        gamma_e2 = data_config['gamma_e2']
        n_hidden = data_config['n_hidden']
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)

        if linear:
            dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
            dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
        else:
            dgp1 = DGP_Weibull_nonlinear(n_features, n_hidden=n_hidden, alpha=[alpha_e1]*n_hidden,
                                         gamma=[gamma_e1]*n_hidden, device=device, dtype=dtype)
            dgp2 = DGP_Weibull_nonlinear(n_features, n_hidden=n_hidden, alpha=[alpha_e2]*n_hidden,
                                         gamma=[gamma_e2]*n_hidden, device=device, dtype=dtype)
            
        if copula_name is None or k_tau == 0:
            rng = np.random.default_rng(0)
            u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            uv = torch.stack([u, v], dim=1)
        else:
            theta = kendall_tau_to_theta(copula_name, k_tau)
            u, v = simulation.simu_archimedean(copula_name, 2, X.shape[0], theta=theta)
            u = torch.from_numpy(u).type(dtype).reshape(-1,1)
            v = torch.from_numpy(v).type(dtype).reshape(-1,1)
            uv = torch.cat([u, v], axis=1)
            
        t1_times = dgp1.rvs(X, uv[:,0].to(device)).cpu()
        t2_times = dgp2.rvs(X, uv[:,1].to(device)).cpu()
        
        observed_times = np.minimum(t1_times, t2_times)
        event_indicators = (t2_times < t1_times).type(torch.int)
        
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y_e = event_indicators
        self.y_t = observed_times
        self.dgps = [dgp1, dgp2]
        self.n_events = 2
        
        return self
    
    def split_data(self, train_size: float, valid_size: float,
                   test_size: float, dtype=torch.float64, random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
    
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
    
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = torch.tensor(dataframe.loc[:, 'X0':'X9'].to_numpy(), dtype=dtype)
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(), dtype=dtype)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(), dtype=dtype)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class CompetingRiskSyntheticDataLoader(BaseDataLoader):
    def load_data(self, data_config, copula_name='clayton', k_tau=0,
                  linear=True, device='cpu', dtype=torch.float64):
        """
        This method generates synthetic data for 2 competing risks (and censoring)
        DGP1: Data generation process for event 1
        DGP2: Data generation process for event 2
        DGP3: Data generation process for censoring
        """
        alpha_e1 = data_config['alpha_e1']
        alpha_e2 = data_config['alpha_e2']
        alpha_e3 = data_config['alpha_e3']
        gamma_e1 = data_config['gamma_e1']
        gamma_e2 = data_config['gamma_e2']
        gamma_e3 = data_config['gamma_e3']
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)

        if linear:
            dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
            dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
            dgp3 = DGP_Weibull_linear(n_features, alpha_e3, gamma_e3, device, dtype)
        else:
            dgp1 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e1, gamma=gamma_e1, device=device, dtype=dtype)
            dgp2 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e2, gamma=gamma_e2, device=device, dtype=dtype)
            dgp3 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e3, gamma=gamma_e3, device=device, dtype=dtype)
        
        if copula_name is None or k_tau == 0:
            rng = np.random.default_rng(0)
            u = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            v = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            w = torch.tensor(rng.uniform(0, 1, n_samples), device=device, dtype=dtype)
            uvw = torch.stack([u, v, w], dim=1)
        else:
            theta = kendall_tau_to_theta(copula_name, k_tau)
            u, v, w = simulation.simu_archimedean(copula_name, 3, X.shape[0], theta=theta)
            u = torch.from_numpy(u).type(dtype).reshape(-1,1)
            v = torch.from_numpy(v).type(dtype).reshape(-1,1)
            w = torch.from_numpy(w).type(dtype).reshape(-1,1)
            uvw = torch.cat([u,v,w], axis=1).to(device)
            
        t1_times = dgp1.rvs(X, uvw[:,0]).cpu()
        t2_times = dgp2.rvs(X, uvw[:,1]).cpu()
        t3_times = dgp3.rvs(X, uvw[:,2]).cpu()
        
        event_times = np.concatenate([t1_times.reshape(-1,1),
                                      t2_times.reshape(-1,1),
                                      t3_times.reshape(-1,1)], axis=1)
        event_indicators = np.argmin(event_times, axis=1)
        observed_times = event_times[np.arange(event_times.shape[0]), event_indicators]
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y_e = event_indicators
        self.y_t = observed_times
        self.y_t1 = t1_times
        self.y_t2 = t2_times
        self.y_t3 = t3_times
        
        self.n_events = 3
        self.dgps = [dgp1, dgp2, dgp3]
        
        return self
   
    def split_data(self,
                   train_size: float,
                   valid_size: float,
                   test_size: float,
                   dtype=torch.float64,
                   random_state=0):
        df = pd.DataFrame(self.X)
        df['event'] = self.y_e
        df['time'] = self.y_t
        df['t1'] = self.y_t1
        df['t2'] = self.y_t2
        df['t3'] = self.y_t3
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = torch.tensor(dataframe.loc[:, 'X0':'X9'].to_numpy(), dtype=dtype)
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(), dtype=dtype)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(), dtype=dtype)
            data_dict['T1'] = torch.tensor(dataframe['t1'].to_numpy(), dtype=dtype)
            data_dict['T2'] = torch.tensor(dataframe['t2'].to_numpy(), dtype=dtype)
            data_dict['T3'] = torch.tensor(dataframe['t3'].to_numpy(), dtype=dtype)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]
        