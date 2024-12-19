from SurvivalEVAL import mean_error
import numpy as np
import pandas as pd
from typing import Optional
import warnings
import torch
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import src.config as cfg
from dataclasses import InitVar, dataclass, field

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
                  alpha: float = 0):
    event_indicators = event_indicators.astype(bool)
    n_test = event_times.size
    
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)
        
    censor_times = event_times[~event_indicators]
    weights = np.ones(n_test)
    
    cg_model = CopulaGraphic(train_event_times, train_event_indicators, alpha=alpha)
    cg_linear_zero = cg_model.cg_linear_zero
    
    best_guesses = cg_model.best_guess(censor_times)
    best_guesses[censor_times > cg_linear_zero] = censor_times[censor_times > cg_linear_zero]
    
    errors = np.empty(predicted_times.size)
    errors[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
    errors[~event_indicators] = best_guesses - predicted_times[~event_indicators]
    
    return np.average(np.abs(errors), weights=weights)