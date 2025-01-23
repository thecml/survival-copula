import random
from SurvivalEVAL import SurvivalEvaluator
import torch
from metrics import ci_dependent, ibs_dependent, mae_dependent
from copula import Clayton_Bivariate, Frank_Bivariate
from data_loader import get_data_loader
import pandas as pd
import numpy as np
import config as cfg
from utility.data import fix_types
from SurvivalEVAL.Evaluations.util import predict_median_survival_time

from models import Weibull_log_linear
from strategies import combine_data_with_censor, make_synthetic_censoring_top_k
from utility.preprocessor import Preprocessor
from utility.survival import convert_to_structured, make_stratified_split, make_time_bins
from trainer import train_copula_model

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

dtype = torch.float64
torch.set_default_dtype(dtype)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEED = 0
DATASET_NAME = "whas"
COPULA_NAME = "frank"
N_FEATURES = list(range(14, 0, -1))

if __name__ == "__main__":
    results_dict = {}
    for top_k in N_FEATURES:
        results_dict[top_k] = {}
        # Load data
        dl = get_data_loader(DATASET_NAME).load_data()
        df_full = dl.get_data().reset_index(drop=True)
        num_features, cat_features = dl.get_features()
        
        # Preprocess full dataset
        preprocessor = Preprocessor(cat_feat_strat='mode', num_feat_strat='mean', scaling_strategy="minmax")
        transformer = preprocessor.fit(df_full.drop(['time', 'event'], axis=1),
                                    cat_feats=cat_features, num_feats=num_features,
                                    one_hot=True, fill_value=-1)
        X = transformer.transform(df_full.drop(['time', 'event'], axis=1)).reset_index(drop=True)
        df_full = pd.concat([X, df_full[['time', 'event']]], axis=1)
        
        # Drop censored rows
        df = df_full.drop(df_full[df_full.event == 0].index)
        df.reset_index(drop=True, inplace=True)
        df.time = df.time.round().astype(int)
    
        censor_times, selected_features = make_synthetic_censoring_top_k(df, df_full, top_k)
        censor_times = np.round(censor_times).astype(int)
        
        df = combine_data_with_censor(df, censor_times, selected_features)
    
        # Split data
        df_train, df_valid, df_test = make_stratified_split(df, stratify_colname='both', frac_train=0.7,
                                                            frac_valid=0.1, frac_test=0.2,
                                                            random_state=SEED)
        
        # Adjust types
        df_train, df_valid, df_test = fix_types(df_train, df_valid, df_test)
        
        # Process data
        data_train = df_train.drop(columns=["true_time"])
        true_test_time = df_test.true_time.values
        true_test_event = np.ones(df_test.shape[0])
        data_valid = df_valid.drop(columns=["true_time"])
        data_test = df_test.drop(columns=["true_time"])
        X_train = data_train.drop(columns=["time", "event"])
        X_valid = data_valid.drop(columns=["time", "event"])
        X_test = data_test.drop(columns=["time", "event"])
        
        # Format data
        train_dict, valid_dict, test_dict = dict(), dict(), dict()
        train_dict['X'] = torch.tensor(X_train.values, device=device, dtype=dtype)
        train_dict['T'] = torch.tensor(data_train['time'].values, device=device, dtype=dtype)
        train_dict['E'] = torch.tensor(data_train['event'].values, device=device, dtype=dtype)
        valid_dict['X'] = torch.tensor(X_valid.values, device=device, dtype=dtype)
        valid_dict['T'] = torch.tensor(data_valid['time'].values, device=device, dtype=dtype)
        valid_dict['E'] = torch.tensor(data_valid['event'].values, device=device, dtype=dtype)
        test_dict['X'] = torch.tensor(X_test.values, device=device, dtype=dtype)
        test_dict['T'] = torch.tensor(data_test['time'].values, device=device, dtype=dtype)
        test_dict['E'] = torch.tensor(data_test['event'].values, device=device, dtype=dtype)
        n_samples = train_dict['X'].shape[0]
        n_features = train_dict['X'].shape[1]
        X_train = pd.DataFrame(train_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
        X_valid = pd.DataFrame(valid_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
        X_test = pd.DataFrame(test_dict['X'].cpu().numpy(), columns=[f'X{i}' for i in range(n_features)])
        y_train = convert_to_structured(train_dict['T'].cpu().numpy(), train_dict['E'].cpu().numpy())
        y_valid = convert_to_structured(valid_dict['T'].cpu().numpy(), valid_dict['E'].cpu().numpy())
        y_test = convert_to_structured(test_dict['T'].cpu().numpy(), test_dict['E'].cpu().numpy())
        
        # Make time bins
        time_bins = make_time_bins(train_dict['T'].cpu(), event=train_dict['E'].cpu(), dtype=dtype).to(device)
        time_bins = torch.cat((torch.tensor([0]).to(device), time_bins))
        
        # Train indep model
        indep_model1 = Weibull_log_linear(n_features, dtype=dtype, device=device) # censoring model
        indep_model2 = Weibull_log_linear(n_features, dtype=dtype, device=device) # event model
        indep_model1, indep_model2, _, _ = train_copula_model(indep_model1, indep_model2, train_dict,
                                                              valid_dict, n_epochs=10000,
                                                              patience=100, lr=1e-3, batch_size=1024,
                                                              verbose=False)
        
        # Train dep model
        dep_model1 = Weibull_log_linear(n_features, dtype=dtype, device=device) # censoring model
        dep_model2 = Weibull_log_linear(n_features, dtype=dtype, device=device) # event model
        if COPULA_NAME == "clayton":
            copula = Clayton_Bivariate(2.0, 1e-4, dtype=dtype, device=device)
        elif COPULA_NAME == "frank":
            copula = Frank_Bivariate(2.0, 1e-4, dtype=dtype, device=device)
        dep_model1, dep_model2, copula, _ = train_copula_model(dep_model1, dep_model2, train_dict,
                                                               valid_dict, copula=copula, n_epochs=10000,
                                                               patience=100, lr=1e-3, batch_size=1024,
                                                               verbose=False)
        copula_theta = float(copula.parameters()[0][0])
        
        # Compute survival function
        survival_indep = torch.zeros((test_dict['X'].shape[0], len(time_bins)), device=device)
        survival_dep = torch.zeros((test_dict['X'].shape[0], len(time_bins)), device=device)
        for i in range(len(time_bins)):
            survival_indep[:,i] = indep_model1.survival(time_bins[i], test_dict['X'])
            survival_dep[:,i] = dep_model1.survival(time_bins[i], test_dict['X'])

        # Loop through each survival model
        survival_models = {"indep_model": survival_indep, "dep_model": survival_dep}
        for model_name, survival_outputs in survival_models.items():
            # Prepare the survival output as a DataFrame
            survival_outputs = pd.DataFrame(survival_outputs.cpu(), columns=time_bins.cpu().numpy())
            survival_outputs[0] = 1  # Ensure survival probability at time 0 is 1

            # Calculate true metrics
            true_evaluator = SurvivalEvaluator(survival_outputs, time_bins, true_test_time, true_test_event)
            ci_true = true_evaluator.concordance()[0]
            ibs_true = true_evaluator.integrated_brier_score(IPCW_weighted=False, num_points=10)
            mae_true = true_evaluator.mae(method="Uncensored")

            # Calculate dependent metrics
            dep_evaluator = SurvivalEvaluator(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                            data_train.time.values, data_train.event.values)
            predicted_times = dep_evaluator.predict_time_from_curve(predict_median_survival_time)
            ci_dep = ci_dependent(predicted_times, time_bins, data_test.time.values, data_test.event.values,
                                data_train.time.values, data_train.event.values, copula_name=COPULA_NAME,
                                alpha=copula_theta)
            ibs_dep = ibs_dependent(survival_outputs, time_bins, data_test.time.values, data_test.event.values,
                                    data_train.time.values, data_train.event.values, num_points=10, 
                                    copula_name=COPULA_NAME, alpha=copula_theta)
            mae_dep = mae_dependent(predicted_times, data_test.time.values, data_test.event.values,
                                    data_train.time.values, data_train.event.values, copula_name=COPULA_NAME,
                                    alpha=copula_theta)

            # Calculate errors
            ci_error = ci_true - ci_dep
            ibs_error = ibs_true - ibs_dep
            mae_error = mae_true - mae_dep

            # Store results in the dictionary
            results_dict[top_k][model_name]= {
                "ci_error": ci_error,
                "ibs_error": ibs_error,
                "mae_error": mae_error
            }
            
    # Convert nested results dictionary to a DataFrame for CSV saving
    flattened_results = []
    for top_k, models in results_dict.items():
        for model_name, metrics in models.items():
            flattened_results.append({
                "top_k": top_k,
                "model_name": model_name,
                **metrics})

    results_df = pd.DataFrame(flattened_results)

    # Save results to a CSV file
    filename = f"{cfg.RESULTS_DIR}/weibull_model_error_{COPULA_NAME.lower()}.csv"
    results_df.to_csv(filename, index=False)
        