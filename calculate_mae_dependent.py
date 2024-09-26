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

pandas2ri.activate()

compound_cox = importr("compound.Cox")

class CopulaGraphic():
    def __init__(self, event_times, event_indicators) -> None:
        #index = np.lexsort((event_indicators, event_times))
        #unique_times = np.unique(event_times[index], return_counts=True)
        #self.survival_times = unique_times[0]
        
        cg_result = compound_cox.CG_Clayton(event_times,
                                            event_indicators,
                                            alpha=0,
                                            S_plot=False) # assumes Clayton(th=0)
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
                  train_event_indicators: Optional[np.ndarray] = None):
    event_indicators = event_indicators.astype(bool)
    n_test = event_times.size
    
    if train_event_indicators is not None:
        train_event_indicators = train_event_indicators.astype(bool)
        
    censor_times = event_times[~event_indicators]
    weights = np.ones(n_test)
    
    cg_model = CopulaGraphic(train_event_times, train_event_indicators)
    cg_linear_zero = cg_model.cg_linear_zero
    
    best_guesses = cg_model.best_guess(censor_times)
    best_guesses[censor_times > cg_linear_zero] = censor_times[censor_times > cg_linear_zero]
    
    errors = np.empty(predicted_times.size)
    errors[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
    errors[~event_indicators] = best_guesses - predicted_times[~event_indicators]
    
    return np.average(np.abs(errors), weights=weights)

if __name__ == "__main__":
    # Test the functions
    train_t = np.array([0, 1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                        26, 27, 28, 29, 30, 31, 32, 33, 34,  60, 61, 62, 63, 64, 65, 66, 67,
                        74, 75, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93,
                        98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                        117, 118, 119, 120, 120, 120, 121, 121, 124, 125, 126, 127, 128, 129,
                        136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149,
                        155, 156, 157, 158, 159, 161, 182, 183, 186, 190, 191, 192, 192, 192,
                        193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 202, 203,
                        204, 202, 203, 204, 212, 213, 214, 215, 216, 217, 222, 223, 224])
    train_e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                        0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,
                        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                        0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0,
                        1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                        0, 0, 1, 1, 0, 0, 0, 0, 0, 0])
    t = np.array([5, 10, 19, 31, 43, 59, 63, 75, 97, 113, 134, 151, 163, 176, 182, 195, 200, 210, 220])
    e = np.array([1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0])
    predict_time = np.array([18, 19, 5, 12, 75, 100, 120, 85, 36, 95, 170, 41, 200, 210, 260, 86, 100, 120, 140])
    
    mae_km = mean_error(predict_time, t, e, train_t, train_e, method='Margin', weighted=False)
    mae_dep = mae_dependent(predict_time, t, e, train_t, train_e) # dependent based on margin
    
    print(mae_km)
    print(mae_dep)