from data_loader import MetabricDataLoader
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter, WeibullAFTFitter
from sksurv.ensemble import GradientBoostingSurvivalAnalysis

from plot import compare_km_curves
from utility import convert_to_structured

def combine_data_with_censor(
        df: pd.DataFrame,
        censor_time: np.ndarray,
) -> pd.DataFrame:
    event_status = df.event.values
    true_times = df.time.values
    times = np.copy(true_times)
    event_status[censor_time < true_times] = 0
    times[event_status == 0] = censor_time[event_status == 0]
    df.drop(columns=["time", "event"], inplace=True)
    df["time"] = times
    df["event"] = event_status
    df["true_time"] = true_times
    df = df[df.time != 0]  # Drop all patients with censor time 0
    df.reset_index(drop=True, inplace=True)
    return df

def calculate_pdf(cdf):
    cdf_extra = np.ones((cdf.shape[0], cdf.shape[1] + 1))
    cdf_extra[:, :-1] = cdf
    pdf = np.diff(cdf_extra, axis=1)
    pdf /= pdf.sum(axis=1)[:, None]
    return pdf

def make_synthetic_censoring(strategy: str,
                             df_event: pd.DataFrame,
                             df_all: pd.DataFrame):
    """
    Build synthetic dependent censoring times
    :param strategy: type of censoring strategy
    :param df_event: dataframe with all event patients
    :param df_all: dataframe with all patients
    :return: synthetic censoring times
    """
    if strategy == "original":
        # Use original censoring distribution from the dataset. Assumes cond. indep censoring
        df_all_copy = df_all.copy()  # Make a copy to avoid changing the original dataset
        df_all_copy.event = 1 - df_all_copy.event
        cph = CoxPHFitter(penalizer=0.0001)
        cph.fit(df_all_copy, duration_col='time', event_col='event')
        censor_curves = cph.predict_survival_function(df_event)
        uniq_times = censor_curves.index.values
        censor_cdf = 1 - censor_curves.values.T
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
    elif strategy == "best":
        # Use only the single best feature by hazard ratio
        df_all_copy = df_all.copy()
        df_all_copy.event = 1 - df_all_copy.event
        cph = CoxPHFitter(penalizer=0.0001) 
        cph.fit(df_all_copy, duration_col='time', event_col='event')
        hazard_ratios = cph.hazard_ratios_
        max_hr_ft = hazard_ratios.idxmax(axis=0)
        df_subset = df_all_copy[['time', 'event', max_hr_ft]]
        cph_new = CoxPHFitter()
        cph_new.fit(df_subset, duration_col='time', event_col='event')
        censor_curves = cph_new.predict_survival_function(df_event)
        uniq_times = censor_curves.index.values
        censor_cdf = 1 - censor_curves.values.T
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
    elif strategy == "high_corr":
        df_all_copy = df_all.copy()
        df_all_copy.event = 1 - df_all_copy.event
        df_ft_only = df_all_copy.drop(columns=['time', 'event'])
        corr_matrix = df_ft_only.corr(method='pearson')
        corr_threshold = 0.5
        high_corr_pairs = [(i, j, corr_matrix.loc[i, j]) 
                           for i in corr_matrix.columns 
                           for j in corr_matrix.columns 
                           if i != j and abs(corr_matrix.loc[i, j]) > corr_threshold]
        to_drop = set([i for i, j, _ in high_corr_pairs])
        df_subset = df_all_copy.drop(to_drop, axis=1)
        cph_new = CoxPHFitter()
        cph_new.fit(df_subset, duration_col='time', event_col='event')
        censor_curves = cph_new.predict_survival_function(df_event)
        uniq_times = censor_curves.index.values
        censor_cdf = 1 - censor_curves.values.T
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
    elif strategy == "weak_corr":
        df_all_copy = df_all.copy()
        df_all_copy.event = 1 - df_all_copy.event
        df_ft_only = df_all_copy.drop(columns=['time', 'event'])
        corr_matrix = df_ft_only.corr(method='pearson')
        corr_threshold = 0.5
        weak_corr_pairs = [(i, j, corr_matrix.loc[i, j]) 
                           for i in corr_matrix.columns 
                           for j in corr_matrix.columns 
                           if i != j and abs(corr_matrix.loc[i, j]) < corr_threshold]
        to_drop = set([i for i, j, _ in weak_corr_pairs])
        df_subset = df_all_copy.drop(to_drop, axis=1)
        cph_new = CoxPHFitter()
        cph_new.fit(df_subset, duration_col='time', event_col='event')
        censor_curves = cph_new.predict_survival_function(df_event)
        uniq_times = censor_curves.index.values
        censor_cdf = 1 - censor_curves.values.T
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
    elif strategy == "feature_importance":
        df_all_copy = df_all.copy()
        df_all_copy.event = 1 - df_all_copy.event
        X = df_all_copy[df_all_copy.columns].drop(['time', 'event'], axis=1)
        y = convert_to_structured(df_all_copy['time'], df_all_copy['event'])
        gbsa = GradientBoostingSurvivalAnalysis(random_state=0)
        gbsa.fit(X, y)
        importances = gbsa.feature_importances_
        feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
        top_5_fts = list(feature_importances.sort_values(by='Importance', ascending=False)[:5]['Feature'])
        df_subset = df_all_copy[['time', 'event'] + top_5_fts]
        cph = CoxPHFitter()
        cph.fit(df_subset, duration_col='time', event_col='event')
        censor_curves = cph.predict_survival_function(df_event)
        uniq_times = censor_curves.index.values
        censor_cdf = 1 - censor_curves.values.T
        censor_pdf = calculate_pdf(censor_cdf)
        censor_times = np.empty(censor_pdf.shape[0])
        for i in range(censor_pdf.shape[0]):
            censor_times[i] = uniq_times[np.argmax(censor_pdf[i, :])] # use the max prob
    elif strategy == "half":
        # Use half of the original features
        pass
    elif strategy == "top_10":
        # Use the ten best features
        pass
    elif strategy == "bottom_10":
        # Use the ten worst features
        pass
    
    return censor_times

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
    strategy = "feature_importance"
    censor_times = make_synthetic_censoring(strategy, df, df_full)
    censor_times = np.round(censor_times).astype(int)
    
    # Combine truth and censored data
    df = combine_data_with_censor(df, censor_times)
    compare_km_curves(df_full, df, show=True)
    