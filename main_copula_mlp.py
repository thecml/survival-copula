import torch
from calculate_mae_dependent import mae_dependent
from copula import Clayton_Bivariate, Frank_Bivariate
from data_loader import MetabricDataLoader
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from SurvivalEVAL import SurvivalEvaluator
from SurvivalEVAL.Evaluations.util import predict_median_survival_time
from sklearn.model_selection import train_test_split

from model import CopulaMLP
from models import Weibull_log_linear, LogNormalCox_linear
from loss import loss_double
from make_semi_synthetic import combine_data_with_censor, make_synthetic_censoring
from plot import compare_km_curves
from utility import convert_to_structured, make_time_bins
from trainer import independent_train_loop_linear, dependent_train_loop_linear, predict_survival_curve

# Set precision
dtype = torch.float64
torch.set_default_dtype(dtype)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Load data
    dl = MetabricDataLoader().load_data()
    num_features, cat_features = dl.get_features()
    df_full = dl.get_data()
    
    # Drop censored rows
    df = df_full.drop(df_full[df_full.event == 0].index)
    df.reset_index(drop=True, inplace=True)
    df.time = df.time.round().astype(int)
    
    # Make synthetic censoring time
    strategy = "original" # original, best, feature_importance
    censor_times = make_synthetic_censoring(strategy, df, df_full)
    censor_times = np.round(censor_times).astype(int)
    
    # Combine truth and censored data to make semi synth data
    df = combine_data_with_censor(df, censor_times)
    
    # Split data
    data_train_valid, data_test = train_test_split(df, test_size=0.3, random_state=0)
    data_train, data_valid = train_test_split(data_train_valid, test_size=0.2, random_state=0)
    
    # Process data.
    data_train = data_train.drop(columns=["true_time"])
    true_test_time = data_test.true_time.values
    true_test_event = np.ones(data_test.shape[0])
    data_valid = data_valid.drop(columns=["true_time"])
    data_test = data_test.drop(columns=["true_time"])
    time_bins = make_time_bins(data_train["time"].values, event=data_train["event"].values)
    
    # Train CoxPH model
    model = CoxPHFitter(penalizer=0.0001) 
    model.fit(data_train, duration_col='time', event_col='event')

    # Calculate true MAE
    survival_outputs = model.predict_survival_function(data_test, time_bins).T
    true_evaluator = SurvivalEvaluator(survival_outputs, time_bins, true_test_time, true_test_event)
    mae_true = true_evaluator.mae(method="Uncensored")
    
    # Calculate censored MAE
    coxph_evaluator = SurvivalEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                  data_train.time.values, data_train.event.values)
    mae_unc = coxph_evaluator.mae(method="Uncensored")
    mae_margin = coxph_evaluator.mae(method="Margin")
    print(f"CoxPH: MAE True={mae_true:.2f}, MAE Uncensored={mae_unc:.2f}, MAE Margin={mae_margin:.2f}")

    # Format data
    train_dict, valid_dict, test_dict = dict(), dict(), dict()
    train_dict['X'] = torch.tensor(data_train.drop(columns=["time", "event"]).values, device=device, dtype=dtype)
    train_dict['T'] = torch.tensor(data_train['time'].values, device=device, dtype=dtype)
    train_dict['E'] = torch.tensor(data_train['event'].values, device=device, dtype=dtype)
    valid_dict['X'] = torch.tensor(data_valid.drop(columns=["time", "event"]).values, device=device, dtype=dtype)
    valid_dict['T'] = torch.tensor(data_valid['time'].values, device=device, dtype=dtype)
    valid_dict['E'] = torch.tensor(data_valid['event'].values, device=device, dtype=dtype)
    test_dict['X'] = torch.tensor(data_test.drop(columns=["time", "event"]).values, device=device, dtype=dtype)
    test_dict['T'] = torch.tensor(data_test['time'].values, device=device, dtype=dtype)
    test_dict['E'] = torch.tensor(data_test['event'].values, device=device, dtype=dtype)
    
    # Train dependent model
    n_features = data_train.shape[1] - 2
    copula = Clayton_Bivariate(0.01, 1e-3, dtype=dtype, device=device)
    n_epochs = 10000
    n_dists = 1
    batch_size = 32
    layers = [32]
    lr_dict = {'network': 1e-3, 'copula': 1e-3}
    model = CopulaMLP(n_features, layers=layers, n_events=2,
                      n_dists=n_dists, copula=copula,
                      time_bins=time_bins, device=device)
    
    # Train model parameters
    model.fit(train_dict, valid_dict, lr_dict=lr_dict, n_epochs=n_epochs,
              patience=10, batch_size=batch_size, verbose=True, weight_decay=0,
              train_copula=False)

    # Make event predictions
    model_preds = model.predict(test_dict['X'].to(device), time_bins, risk=0)
    model_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
    true_evaluator = SurvivalEvaluator(survival_outputs, time_bins, true_test_time, true_test_event)
    mae_true = true_evaluator.mae(method="Uncensored")
    dep_evaluator = SurvivalEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                      data_train.time.values, data_train.event.values)
    mae_margin = dep_evaluator.mae(method="Margin")
    copula_theta = float(model.copula.parameters()[0][0])
    print(f"Copula theta: {copula_theta}")
    print(f"Dependent model: MAE True={mae_true:.2f} - MAE Margin={mae_margin:.2f}")
    print()
    
    # Fine-tune copula parameters
    model.fit(train_dict, valid_dict, lr_dict=lr_dict, n_epochs=10000,
            patience=10000, batch_size=batch_size, verbose=True, weight_decay=0,
            train_copula=True, stationarity_threshold=0.0005, stationarity_patience=5)

    # Make event predictions
    model_preds = model.predict(test_dict['X'].to(device), time_bins, risk=0)
    model_preds = pd.DataFrame(model_preds, columns=time_bins.cpu().numpy())
    true_evaluator = SurvivalEvaluator(survival_outputs, time_bins, true_test_time, true_test_event)
    mae_true = true_evaluator.mae(method="Uncensored")
    dep_evaluator = SurvivalEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                      data_train.time.values, data_train.event.values)
    mae_margin = dep_evaluator.mae(method="Margin")
    copula_theta = float(model.copula.parameters()[0][0])
    print(f"Copula theta: {copula_theta}")
    print(f"Dependent model: MAE True={mae_true:.2f} - MAE Margin={mae_margin:.2f}")
    print()
    
    # Calculate dependent MAE margin
    predicted_times = dep_evaluator.predict_time_from_curve(predict_median_survival_time)
    mae_dep = mae_dependent(predicted_times, data_test.time.values, data_test.event.values,
                            data_train.time.values, data_train.event.values, alpha=copula_theta) # dependent based on margin
    print(mae_dep)
        