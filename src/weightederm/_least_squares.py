from __future__ import annotations

import warnings

import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

from weightederm._fixed_werm import fit_fixed_werm_model
from weightederm._prediction import linear_predict
from weightederm._signal_fitting import fit_weighted_signals
from weightederm._validation import (
    prepare_fixed_fit_inputs,
    validate_alpha,
    validate_fit_solver,
    validate_penalty,
)


def _squared_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    residuals = predictions - targets
    return residuals**2


def _weighted_squared_loss_objective(
    params: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
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
    losses = _squared_loss(predictions, y)
    objective = float(np.sum(weights * losses))

    weighted_residuals = weights * residuals
    grad_coef = 2.0 * (X.T @ weighted_residuals)
    if fit_intercept:
        gradient = np.concatenate([grad_coef, np.array([2.0 * np.sum(weighted_residuals)])])
    else:
        gradient = grad_coef

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


def _fit_weighted_least_squares_signal(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool,
    fit_solver: str = "direct",
    penalty: str = "none",
    alpha: float = 0.0,
) -> tuple[np.ndarray, float | None]:
    if np.sum(weights) <= 0:
        raise ValueError("Weighted least-squares requires at least one positive sample weight.")

    validated_fit_solver = validate_fit_solver(fit_solver)

    effective_solver = validated_fit_solver
    if penalty == "l1" and validated_fit_solver == "direct":
        warnings.warn(
            "L1 penalty is not compatible with the direct solver; falling back to L-BFGS-B.",
            RuntimeWarning,
            stacklevel=2,
        )
        effective_solver = "lbfgsb"

    if effective_solver == "direct":
        sqrt_weights = np.sqrt(weights).reshape(-1, 1)
        design_matrix = X
        if fit_intercept:
            design_matrix = np.column_stack([X, np.ones(X.shape[0])])

        weighted_design = design_matrix * sqrt_weights
        weighted_response = y * sqrt_weights.ravel()

        if penalty == "l2" and alpha > 0.0:
            sqrt_alpha = np.sqrt(alpha)
            n_features = X.shape[1]
            if fit_intercept:
                aug_rows = np.hstack([sqrt_alpha * np.eye(n_features), np.zeros((n_features, 1))])
            else:
                aug_rows = sqrt_alpha * np.eye(n_features)
            weighted_design = np.vstack([weighted_design, aug_rows])
            weighted_response = np.concatenate([weighted_response, np.zeros(n_features)])

        solution, _, _, _ = lstsq(weighted_design, weighted_response)
    else:
        initial_params = 0.1 * np.ones(X.shape[1] + int(fit_intercept), dtype=float)
        result = minimize(
            lambda params: _weighted_squared_loss_objective(
                params,
                X,
                y,
                weights,
                fit_intercept=fit_intercept,
                penalty=penalty,
                alpha=alpha,
            ),
            initial_params,
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1_000, "gtol": 1e-6},
        )
        if not result.success:
            if (
                "TOTAL NO. OF ITERATIONS REACHED LIMIT" in str(result.message)
                and np.all(np.isfinite(result.x))
                and np.isfinite(result.fun)
            ):
                warnings.warn(
                    "Weighted least-squares L-BFGS-B reached the iteration limit; "
                    "using the finite iterate to stay closer to the reference implementation.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                raise ValueError(
                    f"Weighted least-squares fitting failed to converge: {result.message}"
                )
        solution = result.x

    if fit_intercept:
        coef = np.asarray(solution[:-1], dtype=float)
        intercept = float(solution[-1])
    else:
        coef = np.asarray(solution, dtype=float)
        intercept = None

    return coef, intercept


def _fit_weighted_least_squares_signals(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool,
    fit_solver: str = "direct",
    penalty: str = "none",
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    return fit_weighted_signals(
        X,
        y,
        weights,
        fit_signal=lambda X, y, sample_weights: _fit_weighted_least_squares_signal(
            X,
            y,
            sample_weights,
            fit_intercept=fit_intercept,
            fit_solver=fit_solver,
            penalty=penalty,
            alpha=alpha,
        ),
    )


class WERMLeastSquares(RegressorMixin, BaseEstimator):
    """Fixed-changepoint WERM estimator with squared loss.

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
    fit_solver : {"direct", "lbfgsb"}, default="direct"
        Solver used for the first-stage weighted least-squares fits.
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
        fit_solver="direct",
        penalty="none",
        alpha=0.0,
    ):
        self.num_chgpts = num_chgpts
        self.delta = delta
        self.search_method = search_method
        self.fit_intercept = fit_intercept
        self.fit_solver = fit_solver
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
        self : WERMLeastSquares
            Fitted estimator.
        """
        penalty = validate_penalty(self.penalty)
        alpha = validate_alpha(self.alpha)
        feature_names_in, num_chgpts, delta, X_array, y_array, _ = prepare_fixed_fit_inputs(
            X,
            y,
            num_chgpts=self.num_chgpts,
            delta=self.delta,
        )
        fit_solver = validate_fit_solver(self.fit_solver)

        return fit_fixed_werm_model(
            self,
            X_array,
            y_array,
            num_chgpts=num_chgpts,
            delta=delta,
            search_method=self.search_method,
            feature_names_in=feature_names_in,
            fit_signals=lambda X, y, weights: _fit_weighted_least_squares_signals(
                X,
                y,
                weights,
                fit_intercept=self.fit_intercept,
                fit_solver=fit_solver,
                penalty=penalty,
                alpha=alpha,
            ),
            fit_last_segment_signal=lambda X, y: _fit_weighted_least_squares_signal(
                X,
                y,
                np.ones(len(y), dtype=float),
                fit_intercept=self.fit_intercept,
                fit_solver=fit_solver,
                penalty=penalty,
                alpha=alpha,
            ),
            loss=_squared_loss,
        )

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
