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

from utility.metrics import estimate_concordance_index, predict_multi_probabilities_from_curve
from SurvivalEVAL.Evaluations.util import check_and_convert

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
        
    censor_times = event_times[~event_indicators]
    weights = np.ones(n_test)
    
    cg_model = CopulaGraphic(train_event_times, train_event_indicators,
                             copula_name=copula_name, alpha=alpha)
    cg_linear_zero = cg_model.cg_linear_zero
    
    best_guesses = cg_model.best_guess(censor_times)
    best_guesses[censor_times > cg_linear_zero] = censor_times[censor_times > cg_linear_zero]
    
    errors = np.empty(predicted_times.size)
    errors[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
    errors[~event_indicators] = best_guesses - predicted_times[~event_indicators]
    
    return np.average(np.abs(errors), weights=weights)

# CI Dependent under an assumed copula (Margin pair method)
def ci_dependent(predicted_times: np.ndarray,
                 event_times: np.ndarray,
                 event_indicators: np.ndarray,
                 train_event_times: Optional[np.ndarray] = None,
                 train_event_indicators: Optional[np.ndarray] = None,
                 copula_name: str = "clayton",
                 alpha: float = 0):
    event_indicators = event_indicators.astype(bool)
    train_event_indicators = train_event_indicators.astype(bool)
    
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

    # Ties = none
    total_pairs = concordant_pairs + discordant_pairs
    cindex = concordant_pairs / total_pairs

    return cindex, concordant_pairs, total_pairs

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
    
    predict_probs_mat = predict_multi_probabilities_from_curve(predicted_curves, time_bins,
                                                               time_points, interpolation="Linear")
    target_times_mat = np.repeat(time_points.reshape(1, -1), repeats=len(event_times), axis=0)
    event_times_mat = np.repeat(event_times.reshape(-1, 1), repeats=len(time_points), axis=1)
    event_indicators_mat = np.repeat(event_indicators.reshape(-1, 1), repeats=len(time_points), axis=1)
    event_indicators_mat = event_indicators_mat.astype(bool)
    
    inverse_train_event_indicators = 1 - train_event_indicators

    ipc_model = CopulaGraphic(train_event_times, inverse_train_event_indicators,
                              copula_name=copula_name, alpha=alpha)

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

    ipcw_square_error_mat = np.square(predict_probs_mat) * weight_cat1 + np.square(1 - predict_probs_mat) * weight_cat2
    brier_scores = np.mean(ipcw_square_error_mat, axis=0)
    
    if np.isnan(brier_scores).any():
        warnings.warn("Time-dependent Brier Score contains nan")
        bs_dict = {}
        for time_point, b_score in zip(time_points, brier_scores):
            bs_dict[time_point] = b_score
        print("Brier scores for multiple time points are".format(bs_dict))
    integral_value = trapezoid(brier_scores, time_points)
    ibs_score = integral_value / time_range
    
    return ibs_score
    
    
    
    
    
    
    
    
    
    