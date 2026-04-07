from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from weightederm._fixed_werm import fit_fixed_werm_model
from weightederm._prediction import linear_predict
from weightederm._signal_fitting import fit_weighted_signals
from weightederm._smooth_optimization import fit_weighted_smooth_signal
from weightederm._validation import (
    prepare_fixed_fit_inputs,
    validate_alpha,
    validate_huber_epsilon,
    validate_penalty,
)


def _huber_loss(predictions: np.ndarray, targets: np.ndarray, *, epsilon: float) -> np.ndarray:
    residuals = predictions - targets
    absolute = np.abs(residuals)
    quadratic = 0.5 * residuals**2
    linear = epsilon * (absolute - 0.5 * epsilon)
    return np.where(absolute <= epsilon, quadratic, linear)


def _weighted_huber_objective(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    epsilon: float,
    fit_intercept: bool,
    penalty: str = "none",
    alpha: float = 0.0,
) -> tuple[float, np.ndarray]:
    if fit_intercept:
        coef = params[:-1]
        intercept = params[-1]
    else:
        coef = params
        intercept = 0.0

    predictions = X @ coef + intercept
    residuals = predictions - y
    losses = _huber_loss(predictions, y, epsilon=epsilon)
    objective = float(np.sum(weights * losses))

    score = np.where(np.abs(residuals) <= epsilon, residuals, epsilon * np.sign(residuals))
    weighted_score = weights * score
    grad_coef = X.T @ weighted_score

    if fit_intercept:
        gradient = np.concatenate([grad_coef, np.array([np.sum(weighted_score)])])
    else:
        gradient = np.asarray(grad_coef, dtype=float)

    if penalty == "l2":
        objective += alpha * float(np.sum(coef**2))
        penalty_grad = 2.0 * alpha * coef
        if fit_intercept:
            gradient[:-1] += penalty_grad
        else:
            gradient += penalty_grad
    elif penalty == "l1":
        objective += alpha * float(np.sum(np.abs(coef)))
        penalty_grad = alpha * np.sign(coef)
        if fit_intercept:
            gradient[:-1] += penalty_grad
        else:
            gradient += penalty_grad

    return objective, gradient


def _fit_weighted_huber_signal(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool,
    epsilon: float,
    max_iter: int,
    tol: float,
    penalty: str = "none",
    alpha: float = 0.0,
) -> tuple[np.ndarray, float | None]:
    def objective_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        return _weighted_huber_objective(
            params,
            X,
            y,
            weights,
            epsilon=epsilon,
            fit_intercept=fit_intercept,
            penalty=penalty,
            alpha=alpha,
        )

    return fit_weighted_smooth_signal(
        X,
        y,
        weights,
        fit_intercept=fit_intercept,
        objective_and_gradient=objective_and_gradient,
        max_iter=max_iter,
        tol=tol,
        estimator_name="Huber",
    )


def _fit_weighted_huber_signals(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool,
    epsilon: float,
    max_iter: int,
    tol: float,
    penalty: str = "none",
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    return fit_weighted_signals(
        X,
        y,
        weights,
        fit_signal=lambda X, y, sample_weights: _fit_weighted_huber_signal(
            X,
            y,
            sample_weights,
            fit_intercept=fit_intercept,
            epsilon=epsilon,
            max_iter=max_iter,
            tol=tol,
            penalty=penalty,
            alpha=alpha,
        ),
    )


class WERMHuber(RegressorMixin, BaseEstimator):
    """Fixed-changepoint WERM estimator with Huber loss.

    Parameters
    ----------
    num_chgpts : int
        Number of changepoints to estimate.
    delta : int, default=1
        Minimum spacing enforced during changepoint search.
    search_method : {"efficient", "brute_force"}, default="efficient"
        Second-stage changepoint search strategy.
    fit_intercept : bool, default=True
        Whether the signal models include an intercept.
    epsilon : float, default=1.35
        Huber transition parameter.
    max_iter : int, default=100
        Maximum number of optimizer iterations in first-stage signal fitting.
    tol : float, default=1e-5
        Optimizer tolerance.
    penalty : {"none", "l1", "l2"}, default="none"
        Penalty used in the first-stage signal fits.
    alpha : float, default=0.0
        Penalty strength.
    """

    def __init__(
        self,
        num_chgpts,
        *,
        delta=1,
        search_method="efficient",
        fit_intercept=True,
        epsilon=1.35,
        max_iter=100,
        tol=1e-5,
        penalty="none",
        alpha=0.0,
    ):
        self.num_chgpts = num_chgpts
        self.delta = delta
        self.search_method = search_method
        self.fit_intercept = fit_intercept
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.alpha = alpha

    def fit(self, X, y):
        """Fit the estimator on ordered observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Ordered covariates.
        y : array-like of shape (n_samples,)
            Ordered responses.

        Returns
        -------
        self : WERMHuber
            Fitted estimator.
        """
        epsilon = validate_huber_epsilon(self.epsilon)
        penalty = validate_penalty(self.penalty)
        alpha = validate_alpha(self.alpha)
        feature_names_in, num_chgpts, delta, X_array, y_array, _ = prepare_fixed_fit_inputs(
            X,
            y,
            num_chgpts=self.num_chgpts,
            delta=self.delta,
        )

        fitted = fit_fixed_werm_model(
            self,
            X_array,
            y_array,
            num_chgpts=num_chgpts,
            delta=delta,
            search_method=self.search_method,
            feature_names_in=feature_names_in,
            fit_signals=lambda X, y, weights: _fit_weighted_huber_signals(
                X,
                y,
                weights,
                fit_intercept=self.fit_intercept,
                epsilon=epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                penalty=penalty,
                alpha=alpha,
            ),
            fit_last_segment_signal=lambda X, y: _fit_weighted_huber_signal(
                X,
                y,
                np.ones(len(y), dtype=float),
                fit_intercept=self.fit_intercept,
                epsilon=epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                penalty=penalty,
                alpha=alpha,
            ),
            loss=lambda predictions, targets: _huber_loss(predictions, targets, epsilon=epsilon),
        )
        self.n_iter_ = np.array([max(1, self.max_iter)], dtype=int)
        return fitted

    def predict(self, X):
        """Predict with the last-segment refit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates to score with the final segment model.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted responses.
        """
        return linear_predict(self, X)
