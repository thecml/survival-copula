import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import List
import numpy as np
from pycop import simulation
import torch
from utility.survival import convert_to_structured, kendall_tau_to_theta
from utility.survival import make_stratified_split
from dgp import DGP_Weibull_linear, DGP_Weibull_nonlinear
import config as cfg
from pathlib import Path
from sksurv.datasets import load_gbsg2, load_aids, load_whas500, load_flchain

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
        :return: df
        """
        df = pd.DataFrame(self.X)
        df['time'] = self.y['time']
        df['event'] = self.y['event']
        return df

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

def get_data_loader(dataset_name: str) -> BaseDataLoader:
    if dataset_name == "synthetic":
        return SingleEventSyntheticDataLoader()
    elif dataset_name == "seer":
        return SeerDataLoader()
    elif dataset_name == "mimic":
        return MimicDataLoader()
    elif dataset_name == "metabric":
        return MetabricDataLoader()
    elif dataset_name == "support":
        return SupportDataLoader()
    elif dataset_name == "aids":
        return AidsDataLoader()
    elif dataset_name == "gbsg":
        return GbsgDataLoader()
    elif dataset_name == "whas":
        return WhasDataLoader()
    elif dataset_name == "flchain":
        return FlchainDataLoader()
    else:
        raise NotImplementedError()

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
        n_samples = data_config['n_samples']
        n_features = data_config['n_features']
        
        X = torch.rand((n_samples, n_features), device=device, dtype=dtype)

        if linear:
            dgp1 = DGP_Weibull_linear(n_features, alpha_e1, gamma_e1, device, dtype)
            dgp2 = DGP_Weibull_linear(n_features, alpha_e2, gamma_e2, device, dtype)
        else:
            dgp1 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e1,
                                         gamma=gamma_e1, device=device, dtype=dtype)
            dgp2 = DGP_Weibull_nonlinear(n_features, alpha=alpha_e2,
                                         gamma=gamma_e2, device=device, dtype=dtype)
            
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
        
        t1_times = dgp1.rvs(X, uv[:,0].to(device))
        t2_times = dgp2.rvs(X, uv[:,1].to(device))
        
        observed_times = np.minimum(t1_times, t2_times)
        event_indicators = np.array((t2_times < t1_times), dtype=np.int32)
    
        columns = [f'X{i}' for i in range(n_features)]
        self.X = pd.DataFrame(X.cpu(), columns=columns)
        self.y_e = event_indicators
        self.y_t = observed_times
        self.dgps = [dgp1, dgp2]
        self.n_events = 1
        
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
        n_features = df.shape[1] - 2
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = torch.tensor(dataframe.loc[:, 'X0':f'X{n_features-1}'].to_numpy(), dtype=dtype)
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(), dtype=dtype)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(), dtype=dtype)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class SeerDataLoader(BaseDataLoader):
    """
    Data loader for SEER dataset
    """
    def load_data(self):
        data = pd.read_csv(Path.joinpath(cfg.DATA_DIR, 'seer.csv'))
        
        data = data.loc[data['Survival Months'] > 0]
        
        numeric_rows = pd.to_numeric(data["Grade"], errors='coerce').notna()
        data = data[numeric_rows]

        outcomes = data.copy()
        outcomes['event'] =  data['Status']
        outcomes['time'] = data['Survival Months']
        outcomes = outcomes[['event', 'time']]
        outcomes.loc[outcomes['event'] == 'Alive', ['event']] = 0
        outcomes.loc[outcomes['event'] == 'Dead', ['event']] = 1

        data = data.drop(['Status', "Survival Months"], axis=1)

        obj_cols = data.select_dtypes(['bool']).columns.tolist() \
                + data.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            data[col] = data[col].astype('object')

        self.X = pd.DataFrame(data)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

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
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.float64),dtype=dtype)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float64), dtype=dtype)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class MimicDataLoader(BaseDataLoader):
    """
    Data loader for MIMIC dataset
    """
    def load_data(self, n_samples:int = 10000):
        '''
        t and e order, followed by death
        '''
        df = pd.read_csv(Path.joinpath(cfg.DATA_DIR, 'mimic.csv.gz'), compression='gzip', index_col=0)
        df = df[cfg.mimic_features] # select only best features
        
        if n_samples:
            df = df.sample(n=n_samples, random_state=0)
            
        df = df[(df['Age'] >= 60) & (df['Age'] <= 65)] # select cohort ages 60-65
        df = df[df['death_time'] > 0]
  
        columns_to_drop = [col for col in df.columns if
                           any(substring in col for substring in ['_event', '_time', 'hadm_id'])]
        self.X = df.drop(columns_to_drop, axis=1)
        self.y = convert_to_structured(df['death_time'], df['death_event'])
        self.columns = list(self.X.columns)
        self.num_features = self._get_num_features(self.X)
        self.cat_features = self._get_cat_features(self.X)
        
        self.y_t = df[f'death_time'].values # use only death
        self.y_e = df[f'death_event'].values
        self.n_events = 1
        
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
        
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='time', frac_train=train_size,
                                                            frac_valid=valid_size, frac_test=test_size,
                                                            random_state=random_state)
        
        dataframes = [df_train, df_valid, df_test]
        dicts = []
        for dataframe in dataframes:
            data_dict = dict()
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.float64),dtype=dtype)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float64), dtype=dtype)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class MetabricDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        data = pd.read_feather(Path.joinpath(cfg.DATA_DIR, 'metabric.feather')) 
        data['duration'] = data['duration'].apply(round)

        data = data.loc[data['duration'] > 0]
        
        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['duration']
        outcomes = outcomes[['event', 'time']]

        num_feats =  ['x0', 'x1', 'x2', 'x3', 'x8'] \
                     + ['x4', 'x5', 'x6', 'x7']

        self.num_features = num_feats
        self.cat_features = []
                    
        self.X = pd.DataFrame(data[num_feats], dtype=np.float64)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])

        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class SupportDataLoader(BaseDataLoader):
    """
    Data loader for SUPPORT dataset
    """
    def load_data(self, n_samples:int = None) -> None:
        path = Path.joinpath(cfg.DATA_DIR, 'support.feather')
        data = pd.read_feather(path)

        if n_samples:
            data = data.sample(n=n_samples, random_state=0)

        data = data.loc[data['duration'] > 0]

        outcomes = data.copy()
        outcomes['event'] =  data['event']
        outcomes['time'] = data['duration']
        outcomes = outcomes[['event', 'time']]

        num_feats =  ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6',
                      'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13']

        self.num_features = num_feats
        self.cat_features = []
        self.X = pd.DataFrame(data[num_feats], dtype=np.float64)
        self.y = convert_to_structured(outcomes['time'], outcomes['event'])
        self.columns = self.X.columns
        self.n_events = 1
        
        self.y_e = outcomes['event']
        self.y_t = outcomes['time']

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
            data_dict['X'] = dataframe.drop(['event', 'time'], axis=1).to_numpy()
            data_dict['E'] = torch.tensor(dataframe['event'].to_numpy(dtype=np.float64),dtype=dtype)
            data_dict['T'] = torch.tensor(dataframe['time'].to_numpy(dtype=np.float64), dtype=dtype)
            dicts.append(data_dict)
            
        return dicts[0], dicts[1], dicts[2]

class AidsDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_aids()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')
        self.X = pd.DataFrame(X)

        self.y = convert_to_structured(y['time'], y['censor'])
        self.num_features = ['age', 'cd4', 'hemophil', 'ivdrug', 'karnof',
                             'priorzdv', 'raceth', 'sex', 'strat2', 'tx',
                             'txgrp']
        self.cat_features = []
        return self
    
    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class GbsgDataLoader(BaseDataLoader):
    def load_data(self) -> BaseDataLoader:
        X, y = load_gbsg2()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(y['time'], y['cens'])
        self.num_features = ['age', 'estrec', 'pnodes', 'progrec', 'tsize']
        self.cat_features = ['horTh', 'menostat', 'tgrade']
        return self
    
    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class WhasDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_whas500()

        obj_cols = X.select_dtypes(['bool']).columns.tolist() \
                   + X.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X[col] = X[col].astype('category')

        self.X = pd.DataFrame(X)
        self.y = convert_to_structured(y['lenfol'], y['fstat'])
        self.num_features = list(self.X.columns)
        self.cat_features = []
        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()

class FlchainDataLoader(BaseDataLoader):
    def load_data(self) -> None:
        X, y = load_flchain()
        X['event'] = y['death']
        X['time'] = y['futime']

        X = X.loc[X['time'] > 0]
        X_data = X.drop(['event', 'time'], axis=1).reset_index(drop=True)

        obj_cols = X_data.select_dtypes(['bool']).columns.tolist() \
                   + X_data.select_dtypes(['object']).columns.tolist()
        for col in obj_cols:
            X_data[col] = X_data[col].astype('object')

        self.X = pd.DataFrame(X_data)
        self.y = convert_to_structured(X['time'], X['event'])
        self.num_features = ['age', 'creatinine', 'kappa', 'lambda', 'sample.yr']
        self.cat_features = ['chapter', 'flc.grp', 'mgus', 'sex']
        return self

    def split_data(self,
                train_size: float,
                valid_size: float,
                test_size: float,
                dtype=torch.float64,
                random_state=0):
        raise NotImplementedError()
    