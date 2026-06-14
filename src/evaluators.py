from lifelines import CoxPHFitter
import numpy as np
from typing import Callable, Optional
import warnings
from functools import cached_property

import pandas as pd
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

        elif method == "CG_Q":
            cg_model = CopulaGraphicWrapper(
                train_event_times,
                train_event_indicators,
                copula_name=copula_name,
                alpha=alpha,
            )

            target_times_mat = np.repeat(
                time_points.reshape(1, -1),
                repeats=len(event_times),
                axis=0,
            )

            event_times_mat = np.repeat(
                event_times.reshape(-1, 1),
                repeats=len(time_points),
                axis=1,
            )

            observed_mask = event_indicators.astype(bool)
            censored_mask = ~observed_mask

            q_mat = np.zeros_like(target_times_mat, dtype=float)

            q_mat[observed_mask, :] = (
                event_times_mat[observed_mask, :] > target_times_mat[observed_mask, :]
            ).astype(float)

            censored_times = event_times[censored_mask]

            if censored_times.size > 0:
                q_mat[censored_mask, :] = (
                    cg_model.conditional_survival_after_censoring(
                        censored_times,
                        time_points,
                    )
                )

            square_error_mat = (
                q_mat * np.square(1.0 - predict_probs_mat)
                + (1.0 - q_mat) * np.square(predict_probs_mat)
            )

            brier_scores = np.mean(square_error_mat, axis=0)

        elif method == "CG_Q_COPULA":
            cg_model = CopulaGraphicWrapper(
                train_event_times,
                train_event_indicators,
                copula_name=copula_name,
                alpha=alpha,
            )

            target_times_mat = np.repeat(
                time_points.reshape(1, -1),
                repeats=len(event_times),
                axis=0,
            )

            event_times_mat = np.repeat(
                event_times.reshape(-1, 1),
                repeats=len(time_points),
                axis=1,
            )

            observed_mask = event_indicators.astype(bool)
            censored_mask = ~observed_mask

            q_mat = np.zeros_like(target_times_mat, dtype=float)

            # Observed events have deterministic status.
            q_mat[observed_mask, :] = (
                event_times_mat[observed_mask, :] > target_times_mat[observed_mask, :]
            ).astype(float)

            censored_times = event_times[censored_mask]

            if censored_times.size > 0:
                q_mat[censored_mask, :] = (
                    cg_model.conditional_survival_after_censoring_copula(
                        censored_times,
                        time_points,
                    )
                )

            square_error_mat = (
                q_mat * np.square(1.0 - predict_probs_mat)
                + (1.0 - q_mat) * np.square(predict_probs_mat)
            )

            brier_scores = np.mean(square_error_mat, axis=0)
        else:
            raise NotImplementedError()

        if method in ["BG", "BG_UW"]:
            square_error_mat = (
                np.square(predict_probs_mat) * weight_cat1
                + np.square(1.0 - predict_probs_mat) * weight_cat2
            )
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
    
class IndependentEvaluator:
    """
    Integrated Brier score under conditionally independent censoring.

    This evaluator uses a Cox proportional hazards model for the censoring
    distribution G(t | X) = P(C > t | X), rather than a marginal Kaplan-Meier
    censoring curve. It is therefore the conditional/independent-censoring
    analogue of the usual KM-IPCW IBS.

    Parameters
    ----------
    predicted_survival_curves:
        Predicted event survival curves S_hat(t | x) for the test set, with
        shape (n_test, n_time_points). A pandas DataFrame is accepted.
    time_coordinates:
        Time grid corresponding to the columns of predicted_survival_curves.
    test_event_times, test_event_indicators:
        Observed test times and event indicators, where 1 means event and
        0 means censored.
    train_event_times, train_event_indicators:
        Observed train times and event indicators, where 1 means event and
        0 means censored.
    train_features, test_features:
        Covariate matrices used for the censoring CoxPH model. These should be
        the same covariates available to the event model after preprocessing.
    """

    def __init__(
        self,
        predicted_survival_curves: NumericArrayLike,
        time_coordinates: NumericArrayLike,
        test_event_times: NumericArrayLike,
        test_event_indicators: NumericArrayLike,
        train_event_times: NumericArrayLike,
        train_event_indicators: NumericArrayLike,
        train_features,
        test_features,
        predict_time_method: str = "Median",
        interpolation: str = "Linear",
        censor_penalizer: float = 0.01,
    ):
        self._predicted_curves = check_and_convert(predicted_survival_curves)
        self._time_coordinates = check_and_convert(time_coordinates)

        if self._time_coordinates.ndim == 1 and self._time_coordinates[0] != 0:
            warnings.warn(
                "The first time coordinate is not 0. Adding 0 to the time grid "
                "and 1 to the beginning of the predicted survival curves."
            )
            self._time_coordinates = np.insert(self._time_coordinates, 0, 0)
            self._predicted_curves = np.insert(self._predicted_curves, 0, 1, axis=1)

        test_event_times, test_event_indicators = check_and_convert(
            test_event_times, test_event_indicators
        )
        train_event_times, train_event_indicators = check_and_convert(
            train_event_times, train_event_indicators
        )

        self.event_times = test_event_times.astype(float)
        self.event_indicators = test_event_indicators.astype(bool)
        self.train_event_times = train_event_times.astype(float)
        self.train_event_indicators = train_event_indicators.astype(bool)

        self.train_features = self._as_feature_frame(train_features, prefix="X")
        self.test_features = self._as_feature_frame(test_features, prefix="X")

        if self.train_features.shape[0] != self.train_event_times.shape[0]:
            raise ValueError("train_features must have one row per training observation.")
        if self.test_features.shape[0] != self.event_times.shape[0]:
            raise ValueError("test_features must have one row per test observation.")

        if predict_time_method == "Median":
            self.predict_time_method = predict_median_survival_time
        elif predict_time_method == "Mean":
            self.predict_time_method = predict_mean_survival_time
        else:
            raise TypeError("Please enter one of 'Median' or 'Mean'.")

        self.interpolation = interpolation
        self.censor_penalizer = float(censor_penalizer)
        self._censor_model = None
        self._constant_censor_survival = None

        self._fit_censoring_model()

    @staticmethod
    def _as_feature_frame(features, prefix: str) -> pd.DataFrame:
        if isinstance(features, pd.DataFrame):
            X = features.copy()
        else:
            X = pd.DataFrame(np.asarray(features))
        X = X.reset_index(drop=True)
        X.columns = [str(c) if str(c) not in ["__time", "__censor_event"] else f"{prefix}{j}"
                     for j, c in enumerate(X.columns)]
        return X.astype(float)

    @property
    def predicted_curves(self):
        return self._predicted_curves

    @property
    def time_coordinates(self):
        return self._time_coordinates

    @cached_property
    def predicted_event_times(self):
        return self.predict_time_from_curve(self.predict_time_method)

    def predict_time_from_curve(self, predict_method: Callable) -> np.ndarray:
        if (predict_method is not predict_mean_survival_time) and (predict_method is not predict_median_survival_time):
            raise TypeError(
                "Prediction method must be 'predict_mean_survival_time' or "
                "'predict_median_survival_time'."
            )
        return np.array([
            predict_method(self.predicted_curves[i, :], self.time_coordinates, self.interpolation)
            for i in range(self.predicted_curves.shape[0])
        ])

    def predict_multi_probabilities_from_curve(self, target_times: np.ndarray) -> np.ndarray:
        predict_probs_mat = []
        for i in range(self.predicted_curves.shape[0]):
            predict_probs = predict_multi_probs_from_curve(
                self.predicted_curves[i, :],
                self.time_coordinates,
                target_times,
                self.interpolation,
            ).tolist()
            predict_probs_mat.append(predict_probs)
        return np.array(predict_probs_mat)

    def _fit_censoring_model(self):
        if CoxPHFitter is None:
            raise ImportError(
                "IndependentEvaluator requires lifelines. Install it with `pip install lifelines` "
                "or add it to your conda environment."
            )

        # For censoring model, censoring is the event of interest.
        censor_events = (~self.train_event_indicators).astype(int)

        if np.sum(censor_events) == 0:
            # No observed censoring in train: G(t | X) = 1 on the observed support.
            self._constant_censor_survival = 1.0
            return

        df_censor = self.train_features.copy()
        df_censor["__time"] = self.train_event_times
        df_censor["__censor_event"] = censor_events

        # Lifelines can be sensitive to separation/collinearity, so retry with
        # stronger ridge penalization before failing.
        last_error = None
        for penalizer in [self.censor_penalizer, 0.1, 1.0, 10.0]:
            try:
                model = CoxPHFitter(penalizer=penalizer)
                model.fit(
                    df_censor,
                    duration_col="__time",
                    event_col="__censor_event",
                    show_progress=False,
                )
                self._censor_model = model
                self.censor_penalizer = float(penalizer)
                return
            except Exception as exc:  # lifelines raises several convergence-related errors
                last_error = exc

        raise RuntimeError(
            "CoxPH censoring model failed to fit, even with stronger penalization. "
            f"Last error: {last_error}"
        )

    def _predict_censor_survival(self, X: pd.DataFrame, times: np.ndarray) -> np.ndarray:
        """Return G_hat(times_i | X_i) for matched rows/times."""
        times = np.asarray(times, dtype=float)
        X = X.reset_index(drop=True)

        if X.shape[0] != times.shape[0]:
            raise ValueError("X and times must have the same number of rows/elements.")

        if self._constant_censor_survival is not None:
            return np.full(X.shape[0], float(self._constant_censor_survival), dtype=float)

        # Use unique times to avoid relying on lifelines preserving duplicate
        # requested times. The result has shape (n_unique_times, n_rows).
        unique_times, inverse = np.unique(times, return_inverse=True)
        surv = self._censor_model.predict_survival_function(X, times=unique_times)
        values = np.asarray(surv.values, dtype=float)
        if values.shape[0] != unique_times.shape[0]:
            values = values.T

        out = values[inverse, np.arange(X.shape[0])]
        out = np.asarray(out, dtype=float)
        out[~np.isfinite(out)] = np.nan
        return out

    def _predict_censor_survival_matrix(self, target_times: np.ndarray) -> np.ndarray:
        """Return matrix G_hat(t_j | X_i), shape (n_test, n_times)."""
        target_times = np.asarray(target_times, dtype=float)

        if self._constant_censor_survival is not None:
            return np.full((self.test_features.shape[0], target_times.shape[0]),
                           float(self._constant_censor_survival), dtype=float)

        surv = self._censor_model.predict_survival_function(self.test_features, times=target_times)
        values = np.asarray(surv.values, dtype=float)
        if values.shape[0] == target_times.shape[0]:
            return values.T
        return values

    def brier_score_multiple_points(self, target_times: np.ndarray) -> np.ndarray:
        if target_times.ndim != 1:
            raise TypeError("target_times must be a one-dimensional array.")

        predict_probs_mat = self.predict_multi_probabilities_from_curve(target_times)

        target_times_mat = np.repeat(target_times.reshape(1, -1), repeats=len(self.event_times), axis=0)
        event_times_mat = np.repeat(self.event_times.reshape(-1, 1), repeats=len(target_times), axis=1)
        event_indicators_mat = np.repeat(self.event_indicators.reshape(-1, 1), repeats=len(target_times), axis=1)

        # G_hat(T_i | X_i), used for observed events before/equal t.
        G_at_event = self._predict_censor_survival(self.test_features, self.event_times)
        G_at_event_mat = np.repeat(G_at_event.reshape(-1, 1), repeats=len(target_times), axis=1)

        # G_hat(t | X_i), used for individuals known to be event-free at t.
        G_at_target_mat = self._predict_censor_survival_matrix(target_times)

        G_at_event_mat[G_at_event_mat <= 0] = np.inf
        G_at_target_mat[G_at_target_mat <= 0] = np.inf

        weight_cat1 = ((event_times_mat <= target_times_mat) & event_indicators_mat) / G_at_event_mat
        weight_cat2 = (event_times_mat > target_times_mat) / G_at_target_mat

        weight_cat1[~np.isfinite(weight_cat1)] = 0.0
        weight_cat2[~np.isfinite(weight_cat2)] = 0.0

        square_error_mat = (
            np.square(predict_probs_mat) * weight_cat1
            + np.square(1.0 - predict_probs_mat) * weight_cat2
        )
        return np.mean(square_error_mat, axis=0)

    def integrated_brier_score(self, num_points: int = None, draw_figure: bool = False) -> float:
        max_target_time = np.max(np.concatenate((self.event_times, self.train_event_times)))

        if num_points is None:
            censored_times = self.event_times[~self.event_indicators]
            time_points = np.unique(censored_times)
            if time_points.size == 0:
                raise ValueError(
                    "No censored observations in the test set; provide num_points for calculating IBS."
                )
            time_range = np.max(time_points) - np.min(time_points)
        else:
            time_points = np.linspace(0, max_target_time, num_points)
            time_range = max_target_time

        brier_scores = self.brier_score_multiple_points(time_points)
        if np.isnan(brier_scores).any():
            warnings.warn("Time-dependent Brier Score contains nan")

        integral_value = trapezoid(brier_scores, time_points)
        ibs_score = integral_value / time_range

        if draw_figure:
            import matplotlib.pyplot as plt
            plt.plot(time_points, brier_scores, 'bo-')
            plt.xlabel('Time')
            plt.ylabel('Brier Score')
            plt.show()

        return float(ibs_score)