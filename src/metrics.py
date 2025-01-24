from SurvivalEVAL import mean_error
import numpy as np
import pandas as pd
from typing import Optional
import warnings
import torch
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import config as cfg
from dataclasses import InitVar, dataclass, field
from scipy.integrate import trapezoid
from utility.metrics import estimate_concordance_index

from utility.metrics import estimate_concordance_index, predict_multi_probabilities_from_curve
from SurvivalEVAL.Evaluations.util import check_and_convert

from SurvivalEVAL.Evaluations.util import (check_and_convert, KaplanMeierArea, km_mean,
                                           predict_mean_survival_time, predict_median_survival_time)

from utility.survival import convert_to_structured

pandas2ri.activate()

compound_cox = importr("compound.Cox")

class CopulaGraphic():
    def __init__(self, event_times, event_indicators,
                 copula_name="clayton", alpha=0) -> None:
        #index = np.lexsort((event_indicators, event_times))
        #unique_times = np.unique(event_times[index], return_counts=True)
        #self.survival_times = unique_times[0]
        
        if copula_name == "clayton":
            cg_result = compound_cox.CG_Clayton(event_times,
                                                event_indicators,
                                                alpha=alpha,
                                                S_plot=False)
        elif copula_name == "frank":
            cg_result = compound_cox.CG_Frank(event_times,
                                              event_indicators,
                                              alpha=alpha,
                                              S_plot=False)
        elif copula_name == "gumbel":
            cg_result = compound_cox.CG_Gumbel(event_times,
                                               event_indicators,
                                               alpha=alpha,
                                               S_plot=False)
        else:
            raise NotImplementedError()
        
        self.survival_probabilities = cg_result.rx2('surv')
        self.survival_times = cg_result.rx2('time')
        
        #area_probabilities = np.append(1, self.survival_probabilities)
        #area_times = np.append(0, self.survival_times)
        #area_times = cg_result.rx2('time')
        self.survival_probabilities[0] = 1
        area_probabilities = self.survival_probabilities
        area_times = self.survival_times
        
        self.cg_linear_zero = -1 / ((area_probabilities[-1] - 1) / area_times[-1])
        if self.survival_probabilities[-1] != 0:
            area_times = np.append(area_times, self.cg_linear_zero)
            area_probabilities = np.append(area_probabilities, 0)

        area_diff = np.diff(area_times, 1)
        average_probabilities = (area_probabilities[0:-1] + area_probabilities[1:]) / 2
        area = np.flip(np.flip(area_diff * average_probabilities).cumsum())

        self.area_times = np.append(area_times, np.inf)
        self.area_probabilities = area_probabilities
        self.area = np.append(area, 0)

    def predict(self, prediction_times: np.array):
        probability_index = np.digitize(prediction_times, self.survival_times)
        probability_index = np.where(
            probability_index == self.survival_times.size + 1,
            probability_index - 1,
            probability_index,
        )
        probabilities = np.append(1, self.survival_probabilities)[probability_index]
        return probabilities
    
    # Best guess based on the CG estimator probabilities
    def best_guess(self, censor_times: np.array):
        slope = (1 - min(self.survival_probabilities)) / (0 - max(self.survival_times))
        
        before_last_idx = censor_times <= max(self.survival_times)
        after_last_idx = censor_times > max(self.survival_times)
        surv_prob = np.empty_like(censor_times).astype(float)
        surv_prob[after_last_idx] = 1 + censor_times[after_last_idx] * slope
        surv_prob[before_last_idx] = self.predict(censor_times[before_last_idx])

        surv_prob = np.clip(surv_prob, a_min=1e-10, a_max=None)

        censor_indexes = np.digitize(censor_times, self.area_times)
        censor_indexes = np.where(
            censor_indexes == self.area_times.size + 1,
            censor_indexes - 1,
            censor_indexes,
        )

        beyond_idx = censor_indexes > len(self.area_times) - 2
        censor_area = np.zeros_like(censor_times).astype(float)

        censor_area[~beyond_idx] = ((self.area_times[censor_indexes[~beyond_idx]] - censor_times[~beyond_idx]) *
                                    (self.area_probabilities[censor_indexes[~beyond_idx]] + surv_prob[~beyond_idx])
                                    * 0.5)
        censor_area[~beyond_idx] += self.area[censor_indexes[~beyond_idx]]
        return censor_times + censor_area / surv_prob
    
# MAE Dependent under an assumed copula
def mae_dependent(predicted_times: np.ndarray,
                  event_times: np.ndarray,
                  event_indicators: np.ndarray,
                  train_event_times: Optional[np.ndarray] = None,
                  train_event_indicators: Optional[np.ndarray] = None,
                  copula_name: str = "clayton",
                  alpha: float = 0):
    event_indicators = event_indicators.astype(bool)
    n_test = event_times.size
    
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)
        
    cg_model = CopulaGraphic(train_event_times, train_event_indicators,
                             copula_name=copula_name, alpha=alpha)
    cg_linear_zero = cg_model.cg_linear_zero
    if np.isinf(cg_linear_zero):
        cg_linear_zero = max(cg_model.survival_times)
    
    censor_times = event_times[~event_indicators]
    weights = np.ones(n_test)
    
    # Weighted = True
    weights[~event_indicators] = 1 - cg_model.predict(censor_times)
    
    # IPCW-v1
    best_guesses = np.empty(shape=n_test)
    for i in range(n_test):
        if event_indicators[i] == 1:
            best_guesses[i] = event_times[i]
        else:
            # Numpy will throw a warning if afterward_event_times are all false. TODO: consider change the code.
            afterward_event_idx = train_event_times[train_event_indicators == 1] > event_times[i]
            best_guesses[i] = np.mean(train_event_times[train_event_indicators == 1][afterward_event_idx])
    nan_idx = np.argwhere(np.isnan(best_guesses))
    predicted_times = np.delete(predicted_times, nan_idx)
    best_guesses = np.delete(best_guesses, nan_idx)
    weights = np.delete(weights, nan_idx)
    
    errors = best_guesses - predicted_times
            
    #best_guesses = cg_model.best_guess(censor_times)
    #best_guesses[censor_times > cg_linear_zero] = censor_times[censor_times > cg_linear_zero]
    
    #errors = np.empty(predicted_times.size)
    #errors[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
    #errors[~event_indicators] = best_guesses - predicted_times[~event_indicators]
    
    return np.average(np.abs(errors), weights=weights)

# CI Dependent under an assumed copula (Margin pair method)
def ci_dependent(predicted_times: np.ndarray,
                 time_bins: np.ndarray,
                 event_times: np.ndarray,
                 event_indicators: np.ndarray,
                 train_event_times: Optional[np.ndarray] = None,
                 train_event_indicators: Optional[np.ndarray] = None,
                 copula_name: str = "clayton",
                 alpha: float = 0):
    event_indicators = event_indicators.astype(bool)
    train_event_indicators = train_event_indicators.astype(bool)
    
    method = "bg"
    if method == "bg":
        cg_model = CopulaGraphic(train_event_times, train_event_indicators,
                                 copula_name=copula_name, alpha=alpha)
        cg_linear_zero = cg_model.cg_linear_zero
        if np.isinf(cg_linear_zero):
            cg_linear_zero = max(cg_model.survival_times)
        predicted_times = np.clip(predicted_times, a_max=cg_linear_zero, a_min=None)
        risks = -1 * predicted_times

        censor_times = event_times[~event_indicators]
        partial_weights = np.ones_like(event_indicators, dtype=float)
        partial_weights[~event_indicators] = 1 - cg_model.predict(censor_times)

        best_guesses = cg_model.best_guess(censor_times)
        best_guesses[censor_times > cg_linear_zero] = censor_times[censor_times > cg_linear_zero]

        bg_event_times = np.copy(event_times)
        bg_event_times[~event_indicators] = best_guesses
        
        cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = estimate_concordance_index(
            event_indicators, event_times, estimate=risks, bg_event_time=bg_event_times, partial_weights=partial_weights)
        
        total_pairs = concordant_pairs + discordant_pairs + risk_ties # Ties = risk
        concordant_pairs = concordant_pairs + 0.5 * risk_ties
        cindex = concordant_pairs / total_pairs
        
        return cindex
        
    elif method == "ipcw":
        risks = -1 * predicted_times
        
        #estimate = _check_estimate_1d(estimate, test_time)
        tau = None
        
        if tau is not None:
            mask = event_times < tau
            survival_test = survival_test[mask]
        
        survival_train = convert_to_structured(train_event_times, train_event_indicators)
        survival_test = convert_to_structured(event_times, event_indicators)
        
        # Fit CG model
        inverse_train_event_indicators = 1 - train_event_indicators
        ipc_model = CopulaGraphic(train_event_times, inverse_train_event_indicators,
                                  copula_name=copula_name, alpha=alpha)
        #ipc_model = KaplanMeierArea(train_event_times, inverse_train_event_indicators
        # )
        #from sksurv.nonparametric import CensoringDistributionEstimator
        #cens = CensoringDistributionEstimator()
        #cens.fit(survival_train)
        
        # Predict
        ipc_pred = ipc_model.predict(event_times)
        unique_time_ = time_bins
        prob_ = ipc_pred
        
        # Calculate IPCW
        actual_event_time = event_times[event_indicators] # time
        extends = actual_event_time > unique_time_[-1]
        
        # beyond last time point is zero probability
        Shat = np.empty(actual_event_time.shape, dtype=float)
        Shat[extends] = 0.0

        valid = ~extends
        actual_event_time = actual_event_time[valid]
        idx = np.searchsorted(unique_time_, actual_event_time)
        # for non-exact matches, we need to shift the index to left
        eps = np.finfo(unique_time_.dtype).eps
        exact = np.absolute(unique_time_[idx] - actual_event_time) < eps
        idx[~exact] -= 1
        Shat[valid] = prob_[idx] #Ghat
        
        if (Shat == 0.0).any():
            raise ValueError("censoring survival function is zero at one or more time points")
        
        weights = np.zeros(event_times.shape[0])
        weights[event_indicators] = 1.0 / Shat
        ipcw_test = weights
        
        if tau is None:
            ipcw = ipcw_test
        else:
            ipcw = np.empty(risks.shape[0], dtype=ipcw_test.dtype)
            ipcw[mask] = ipcw_test
            ipcw[~mask] = 0

        tied_tol = 1e-8
        w = np.square(ipcw)

        from SurvivalEVAL.Evaluations.Concordance import _estimate_concordance_index
        cindex, concordant, discordant, tied_risk, tied_time = _estimate_concordance_index(event_indicators, event_times,
                                                                                           risks, w, tied_tol)
        
        return cindex
    else:
        raise NotImplementedError()

    #cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = estimate_concordance_index(
    #    event_indicators, event_times, estimate=risks, bg_event_time=bg_event_times, partial_weights=partial_weights)

# IBS IPCW Dependent under an assumed copula
def ibs_dependent(predicted_curves: np.ndarray,
                  time_bins: np.ndarray,
                  event_times: np.ndarray,
                  event_indicators: np.ndarray,
                  train_event_times: Optional[np.ndarray] = None,
                  train_event_indicators: Optional[np.ndarray] = None,
                  num_points: int = None,
                  copula_name: str = "clayton",
                  alpha: float = 0):
    
    predicted_curves = check_and_convert(predicted_curves)
    time_bins = check_and_convert(time_bins)
    
    event_indicators = event_indicators.astype(bool)
    train_event_indicators = train_event_indicators.astype(bool)
    
    max_target_time = np.max(np.concatenate((event_times, train_event_times))) if train_event_times \
                      is not None else np.max(event_times)
        
    if num_points is None:
        censored_times = event_times[event_indicators == 0]
        time_points = np.unique(censored_times)
        if time_points.size == 0:
            raise ValueError("You don't have censor data in the testset, "
                             "please provide \"num_points\" for calculating IBS")
        else:
            time_range = np.max(time_points) - np.min(time_points)
    else:
        time_points = np.linspace(0, max_target_time, num_points)
        time_range = max_target_time
    
    # Calculate multiple Brier scores at multiple specific times.
    #predict_probs_mat = predict_multi_probabilities_from_curve(predicted_curves, time_bins,
    #                                                           time_points, interpolation="Linear")   

    method = "bg" # bg using CG estimator
    if method == "ipcw":
        target_times_mat = np.repeat(time_points.reshape(1, -1), repeats=len(event_times), axis=0)
        event_times_mat = np.repeat(event_times.reshape(-1, 1), repeats=len(time_points), axis=1)
        event_indicators_mat = np.repeat(event_indicators.reshape(-1, 1), repeats=len(time_points), axis=1)
        event_indicators_mat = event_indicators_mat.astype(bool)    
    
        inverse_train_event_indicators = 1 - train_event_indicators

        # Use the CG estimator for IPCW
        #ipc_model = CopulaGraphic(train_event_times, inverse_train_event_indicators,
        #                          copula_name=copula_name, alpha=alpha)
        ipc_model = KaplanMeierArea(train_event_times, inverse_train_event_indicators)

        # Category one calculates IPCW weight at observed time point.
        # Category one is individuals with event time lower than the time of interest and were NOT censored.
        ipc_pred = ipc_model.predict(event_times_mat)
        # Catch if denominator is 0.
        ipc_pred[ipc_pred == 0] = np.inf
        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat) / ipc_pred
        # Catch if event times goes over max training event time, i.e. predict gives NA
        weight_cat1[np.isnan(weight_cat1)] = 0
        # Category 2 is individuals whose time was greater than the time of interest (singleBrierTime)
        # contain both censored and uncensored individuals.
        ipc_target_pred = ipc_model.predict(target_times_mat)
        # Catch if denominator is 0.
        ipc_target_pred[ipc_target_pred == 0] = np.inf
        weight_cat2 = (event_times_mat > target_times_mat) / ipc_target_pred
        # predict returns NA if the passed in time is greater than any of the times used to build
        # the inverse probability of censoring model.
        weight_cat2[np.isnan(weight_cat2)] = 0
        
        square_error_mat = np.square(predict_probs_mat) * weight_cat1 + np.square(1 - predict_probs_mat) * weight_cat2
        brier_scores = np.mean(square_error_mat, axis=0)
        
    elif method == "bg":
        censored_times = event_times[event_indicators == 0]
        cg_model = CopulaGraphic(train_event_times, train_event_indicators, copula_name=copula_name, alpha=alpha)
        
        
        censored_times_bg = cg_model.best_guess(censored_times)
        event_times_bg = event_times.copy()
        event_times_bg[event_indicators == 0] = censored_times_bg
        
        event_indicators = np.ones_like(event_indicators)

        target_times_mat = np.repeat(time_points.reshape(1, -1), repeats=len(event_times), axis=0)
        event_times_mat = np.repeat(event_times_bg.reshape(-1, 1), repeats=len(time_points), axis=1)
        event_indicators_mat = np.repeat(event_indicators.reshape(-1, 1), repeats=len(time_points), axis=1)
        event_indicators_mat = event_indicators_mat.astype(bool)
        
        from SurvivalEVAL.Evaluations.util import predict_multi_probs_from_curve
        predict_probs_mat = []
        for i in range(predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(predicted_curves[i, :],
                                                           time_bins,
                                                           time_points).tolist()
            predict_probs_mat.append(predict_probs)
        predict_probs_mat = np.array(predict_probs_mat)
    
        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat)
        weight_cat2 = (event_times_mat > target_times_mat)
        
        square_error_mat = np.square(predict_probs_mat) * weight_cat1 + np.square(1 - predict_probs_mat) * weight_cat2
        brier_scores = np.mean(square_error_mat, axis=0)
    else:
        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat)
        weight_cat2 = (event_times_mat > target_times_mat)
    
    if np.isnan(brier_scores).any():
        warnings.warn("Time-dependent Brier Score contains nan")
        bs_dict = {}
        for time_point, b_score in zip(time_points, brier_scores):
            bs_dict[time_point] = b_score
        print("Brier scores for multiple time points are".format(bs_dict))
        
    integral_value = trapezoid(brier_scores, time_points)
    ibs_score = integral_value / time_range
    
    return ibs_score
    