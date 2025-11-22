import numpy as np
from typing import Callable, Optional
import warnings
from functools import cached_property

from scipy.integrate import trapezoid
from estimators import CopulaGraphic, CopulaGraphicWrapper
from utility.metrics import estimate_concordance_index

from utility.metrics import estimate_concordance_index, predict_multi_probabilities_from_curve
from SurvivalEVAL.Evaluations.util import (check_and_convert, KaplanMeierArea, km_mean,
                                           predict_mean_survival_time, predict_median_survival_time,
                                           predict_multi_probs_from_curve)
from SurvivalEVAL.Evaluations.custom_types import NumericArrayLike

from scipy.ndimage import gaussian_filter1d

class DependentEvaluator:
    def __init__(self,
            predicted_survival_curves: NumericArrayLike,
            time_coordinates: NumericArrayLike,
            test_event_times: NumericArrayLike,
            test_event_indicators: NumericArrayLike,
            train_event_times: Optional[NumericArrayLike] = None,
            train_event_indicators: Optional[NumericArrayLike] = None,
            copula_name: str = None,
            alpha: float = None,
            predict_time_method: str = "Median",
            interpolation: str = "Linear"
    ):
        self._predicted_curves = check_and_convert(predicted_survival_curves)
        self._time_coordinates = check_and_convert(time_coordinates)

        if self._time_coordinates.ndim == 1:
            if self._time_coordinates[0] != 0:
                warnings.warn("The first time coordinate is not 0. A authentic survival curve should start from 0 "
                              "with 100% survival probability. Adding 0 to the beginning of the time coordinates and"
                              " 1 to the beginning of the predicted curves.")
                # Add 0 to the beginning of the time coordinates, and add the 100% survival probability to the
                # beginning of the predicted curves.
                self._time_coordinates = np.insert(self._time_coordinates, 0, 0)
                self._predicted_curves = np.insert(self._predicted_curves, 0, 1, axis=1)

        test_event_times, test_event_indicators = check_and_convert(test_event_times, test_event_indicators)
        self.event_times = test_event_times
        self.event_indicators = test_event_indicators

        if (train_event_times is not None) and (train_event_indicators is not None):
            train_event_times, train_event_indicators = check_and_convert(train_event_times, train_event_indicators)
        self.train_event_times = train_event_times
        self.train_event_indicators = train_event_indicators

        if predict_time_method == "Median":
            self.predict_time_method = predict_median_survival_time
        elif predict_time_method == "Mean":
            self.predict_time_method = predict_mean_survival_time
        else:
            error = "Please enter one of 'Median' or 'Mean' for calculating predicted survival time."
            raise TypeError(error)

        self.interpolation = interpolation
        
        self.copula_name = copula_name
        self.alpha = alpha

    def _error_trainset(self, method_name: str):
        if (self.train_event_times is None) or (self.train_event_indicators is None):
            raise TypeError("Train set information is missing. "
                            "Evaluator cannot perform {} evaluation.".format(method_name))

    @property
    def predicted_curves(self):
        return self._predicted_curves

    @predicted_curves.setter
    def predicted_curves(self, val: NumericArrayLike):
        print("Setter called. Resetting predicted curves for this evaluator.")
        self._predicted_curves = check_and_convert(val)
        self._clear_cache()

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @time_coordinates.setter
    def time_coordinates(self, val: NumericArrayLike):
        print("Setter called. Resetting time coordinates for this evaluator.")
        self._time_coordinates = check_and_convert(val)
        self._clear_cache()

    @cached_property
    def predicted_event_times(self):
        return self.predict_time_from_curve(self.predict_time_method)

    def _clear_cache(self):
        self.__dict__.pop('predicted_event_times', None)

    def predict_time_from_curve(self, predict_method: Callable) -> np.ndarray:
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            error = "Prediction method must be 'predict_mean_survival_time' or 'predict_median_survival_time', " \
                    "got '{}' instead".format(predict_method.__name__)
            raise TypeError(error)

        predicted_times = []
        for i in range(self.predicted_curves.shape[0]):
            predicted_time = predict_method(self.predicted_curves[i, :], self.time_coordinates, self.interpolation)
            predicted_times.append(predicted_time)
        predicted_times = np.array(predicted_times)
        return predicted_times
    
    def concordance(self, method: str):
        event_times = self.event_times.copy()
        event_indicators = self.event_indicators.copy()
        train_event_times = self.train_event_times
        train_event_indicators = self.train_event_indicators
        copula_name = self.copula_name
        alpha = self.alpha
        
        if method == "BG":
            cg_model = CopulaGraphicWrapper(train_event_times, train_event_indicators,
                                            copula_name=copula_name, alpha=alpha)
            
            cg_linear_zero = cg_model.cg_linear_zero
            if np.isinf(cg_linear_zero):
                cg_linear_zero = max(cg_model.survival_times)
            predicted_times = np.clip(self.predicted_event_times, a_max=cg_linear_zero, a_min=None)
            risks = -1 * predicted_times
            
            # Impute censored cases with CG best guesses
            event_times_bg = event_times.copy()
            cens_mask = (event_indicators == 0)
            event_times_bg[cens_mask] = cg_model.best_guess(event_times[cens_mask])
            
            # Calculate weights
            partial_weights = np.ones_like(event_indicators, dtype=float)
            use_cg_weights = True  
            
            if use_cg_weights:  
                # KM/CG-based weights
                censor_times = event_times[cens_mask]
                partial_weights[cens_mask] = 1 - cg_model.predict(censor_times)
            else:  
                # Distance-based weights
                distance = event_times_bg[cens_mask] - event_times[cens_mask] 
                t_max = event_times.max()
                partial_weights[cens_mask] = 1 - (distance / t_max)
                partial_weights[cens_mask] = np.clip(partial_weights[cens_mask], 0.0, 1.0)

            n = len(event_times)
            numerator = denominator = 0.0
            concordant = discordant = tied_risk = 0
            tied_time = None
            tied_tol = 1e-8

            # Calculate CI
            for i in range(n):
                for j in range(i + 1, n):
                    # skip censored–censored pairs
                    if event_indicators[i] == 0 and event_indicators[j] == 0:
                        continue

                    ti, tj = event_times_bg[i], event_times_bg[j]
                    ri, rj = risks[i], risks[j]
                    wi, wj = partial_weights[i], partial_weights[j]
                    
                    if ti == tj:
                        continue  # not comparable

                    tied_time = min(ti, tj)

                    # earlier time = "event" for the pair
                    if ti < tj:
                        if abs(ri - rj) <= tied_tol:
                            numerator += 0.5 * wi
                            tied_risk += 1
                        elif ri > rj:
                            numerator += 1 * wi
                            concordant += 1
                        else:
                            discordant += 1
                        denominator += 1 * wi

                    elif tj < ti:
                        if abs(ri - rj) <= tied_tol:
                            numerator += 0.5 * wj
                            tied_risk += 1
                        elif rj > ri:
                            numerator += 1 * wj
                            concordant += 1
                        else:
                            discordant += 1
                        denominator += 1 * wj

            cindex = numerator / denominator
            
            return cindex, concordant, discordant, tied_risk, tied_time
    
        elif method == "IPCW":
            time_bins = self.time_coordinates
            cg_model_event = CopulaGraphicWrapper(train_event_times, train_event_indicators,
                                                  copula_name=copula_name, alpha=alpha)
            cg_linear_zero = cg_model_event.cg_linear_zero
            if np.isinf(cg_linear_zero):
                cg_linear_zero = max(cg_model_event.survival_times)
            predicted_times = np.clip(self.predicted_event_times, a_max=cg_linear_zero, a_min=None)
            risks = -1 * predicted_times
            
            tau = max(train_event_times) # truncate
            
            if tau is not None:
                mask = event_times < tau
                event_times_mask = event_times[mask]
            
            inverse_train_event_indicators = 1 - train_event_indicators
            time_bins = self.time_coordinates
            cg_model_censor = CopulaGraphicWrapper(train_event_times, inverse_train_event_indicators,
                                                   copula_name=copula_name, alpha=alpha)
            ipcw_test = cg_model_censor.predict(event_times_mask)
            
            if tau is None:
                ipcw = ipcw_test
            else:
                ipcw = np.empty(risks.shape[0], dtype=ipcw_test.dtype)
                ipcw[mask] = ipcw_test
                ipcw[~mask] = 0

            w = np.square(ipcw)
            event_indicators = event_indicators.astype(bool)
            from sksurv.metrics import _estimate_concordance_index
            cindex, concordant_pairs, discordant_pairs, risk_ties, time_ties = _estimate_concordance_index(event_indicators,
                                                                                                           event_times,
                                                                                                           risks,
                                                                                                           weights=w,
                                                                                                           tied_tol=tied_tol)
        else:
            raise NotImplementedError()
        
        return cindex, concordant_pairs, (concordant_pairs+discordant_pairs)    
        
    def integrated_brier_score(self, method: str, num_points: int):
        predicted_curves = check_and_convert(self.predicted_curves)
        time_bins = check_and_convert(self.time_coordinates)

        event_times = self.event_times
        event_indicators = self.event_indicators
        train_event_times = self.train_event_times
        train_event_indicators = self.train_event_indicators
        copula_name = self.copula_name
        alpha = self.alpha
        
        event_indicators = self.event_indicators.astype(bool)
        train_event_indicators = self.train_event_indicators.astype(bool)
        
        max_target_time = np.max(np.concatenate((event_times, train_event_times))) if train_event_times \
                          is not None else np.max(event_times)
            
        time_points = np.linspace(0, max_target_time, num_points)
        time_range = max_target_time
        
        predict_probs_mat = []
        for i in range(predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(
                predicted_curves[i, :],
                time_bins,
                time_points
            ).tolist()
            predict_probs_mat.append(predict_probs)

        predict_probs_mat = np.array(predict_probs_mat)

        # If smoothing is requested, apply AFTER prediction
        do_smoothing = (method == "BG_smooth")

        if do_smoothing:
            predict_probs_mat = gaussian_filter1d(
                predict_probs_mat, sigma=1.0, axis=1
            )

        if method == "BG":
            censored_times = event_times[event_indicators == 0]
            cg_model = CopulaGraphicWrapper(
                train_event_times, train_event_indicators,
                copula_name=copula_name, alpha=alpha
            )

            censored_times_bg = cg_model.best_guess(censored_times)
            event_times_bg = event_times.copy()
            event_times_bg[event_indicators == 0] = censored_times_bg

            event_indicators_bg = np.ones_like(event_indicators, dtype=bool)

            target_times_mat = np.repeat(time_points.reshape(1, -1), repeats=len(event_times), axis=0)
            event_times_mat = np.repeat(event_times_bg.reshape(-1, 1), repeats=len(time_points), axis=1)
            event_indicators_mat = np.repeat(event_indicators_bg.reshape(-1, 1), repeats=len(time_points), axis=1)

            weight_cat1 = (event_times_mat <= target_times_mat) & event_indicators_mat
            weight_cat2 = (event_times_mat > target_times_mat)

        elif method == "BG_UW":

            censored_mask = (event_indicators == 0)
            censored_times = event_times[censored_mask]

            cg_model = CopulaGraphicWrapper(
                train_event_times, train_event_indicators,
                copula_name=copula_name, alpha=alpha
            )

            censored_times_bg = cg_model.best_guess(censored_times)
            event_times_bg = event_times.copy()
            event_times_bg[censored_mask] = censored_times_bg
            
            target_times_mat = np.repeat(time_points.reshape(1, -1), repeats=len(event_times), axis=0)
            event_times_mat  = np.repeat(event_times_bg.reshape(-1, 1), repeats=len(time_points), axis=1)

            # treat imputed censored as "events"
            event_indicators_mat = np.ones_like(event_times_mat, dtype=bool)

            weight_cat1 = (event_times_mat <= target_times_mat)
            weight_cat2 = (event_times_mat > target_times_mat)

            # Build uncertainty weights per patient
            if censored_times.size > 0:
                cg_model_uncert = CopulaGraphic(
                    train_event_times, train_event_indicators,
                    alpha=alpha, type=copula_name
                )
                S_e = cg_model_uncert.predict(censored_times)
                F_c = 1.0 - S_e

                gamma = 1.0
                w_c = np.clip(F_c, 0.0, 1.0) ** gamma
            else:
                w_c = np.array([])

            w_row = np.ones(len(event_times), dtype=float)
            if w_c.size > 0:
                w_row[censored_mask] = w_c

            w_row /= (w_row.mean() + 1e-12)
            w_mat = np.repeat(w_row.reshape(-1, 1), repeats=len(time_points), axis=1)

            weight_cat1 = weight_cat1 * w_mat
            weight_cat2 = weight_cat2 * w_mat

        elif method == "BG_smooth":

            censored_times = event_times[event_indicators == 0]
            cg_model = CopulaGraphicWrapper(
                train_event_times, train_event_indicators,
                copula_name=copula_name, alpha=alpha
            )

            censored_times_bg = cg_model.best_guess(censored_times)
            event_times_bg = event_times.copy()
            event_times_bg[event_indicators == 0] = censored_times_bg

            event_times_mat = np.repeat(event_times_bg.reshape(-1, 1), repeats=len(time_points), axis=1)
            target_times_mat = np.repeat(time_points.reshape(1, -1), repeats=len(event_times), axis=0)

            # Smooth transition between "before" and "after" event
            tau = max_target_time / 200.0
            diff = event_times_mat - target_times_mat

            weight_cat1 = 1.0 / (1.0 + np.exp(diff / (tau + 1e-12)))
            weight_cat2 = 1.0 - weight_cat1

        else:
            raise NotImplementedError()

        square_error_mat = np.square(predict_probs_mat) * weight_cat1 + np.square(1 - predict_probs_mat) * weight_cat2
        brier_scores = np.mean(square_error_mat, axis=0)
    
        if np.isnan(brier_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")
            bs_dict = {}
            for time_point, b_score in zip(time_points, brier_scores):
                bs_dict[time_point] = b_score
            print("Brier scores for multiple time points are".format(bs_dict))
            
        integral_value = trapezoid(brier_scores, time_points)
        ibs_score = integral_value / time_range
        
        return ibs_score
    
    def mae(self, method: str, weighted: bool=True):
        predicted_times = self.predict_time_from_curve(self.predict_time_method)
        time_bins = check_and_convert(self.time_coordinates)

        event_times = self.event_times
        event_indicators = self.event_indicators
        train_event_times = self.train_event_times
        train_event_indicators = self.train_event_indicators
        copula_name = self.copula_name
        alpha = self.alpha
        
        event_indicators = event_indicators.astype(bool)
        n_test = event_times.size
        if train_event_indicators is not None:
            train_event_indicators = train_event_indicators.astype(bool)
        
        # Calculate the weighting for each sample
        if method in ["BG", "IPCW"]:
            time_bins = self.time_coordinates
            cg_model = CopulaGraphicWrapper(train_event_times, train_event_indicators,
                                            copula_name=copula_name, alpha=alpha)
            cg_linear_zero = cg_model.cg_linear_zero
            if np.isinf(cg_linear_zero):
                cg_linear_zero = max(cg_model.survival_times)
            
            censor_times = event_times[~event_indicators]
            weights = np.ones(n_test)
            
            if weighted:
                weights[~event_indicators] = 1 - cg_model.predict(censor_times)

        # Set the error func
        error_func = np.abs
        
        # Calculate error
        if method == "BG":
            best_guesses = cg_model.best_guess(censor_times)
            best_guesses[censor_times > cg_linear_zero] = censor_times[censor_times > cg_linear_zero]
            
            errors = np.empty(predicted_times.size)
            
            errors[event_indicators] = event_times[event_indicators] - predicted_times[event_indicators]
            errors[~event_indicators] = best_guesses - predicted_times[~event_indicators]
            
        elif method == "IPCW": # CG with IPCW-V1 weighting
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
            
        return np.average(error_func(errors), weights=weights)
