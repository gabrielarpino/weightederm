from __future__ import annotations

import numpy as np
from scipy.optimize import linprog
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin

from weightederm._fixed_werm import fit_fixed_werm_model
from weightederm._prediction import logistic_predict_proba
from weightederm._signal_fitting import fit_weighted_signals
from weightederm._smooth_optimization import fit_weighted_smooth_signal
from weightederm._validation import (
    prepare_fixed_fit_inputs,
    validate_alpha,
    validate_penalty,
)


def _logistic_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, predictions) - targets * predictions


def _weighted_logistic_objective(
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

    linear_predictor = X @ coef + intercept
    losses = _logistic_loss(linear_predictor, y)
    objective = float(np.sum(weights * losses))

    score = weights * (expit(linear_predictor) - y)
    grad_coef = X.T @ score
    if fit_intercept:
        gradient = np.concatenate([grad_coef, np.array([np.sum(score)])])
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


def _check_logistic_subproblem_has_finite_optimum(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool,
    penalty: str = "none",
) -> None:
    if penalty != "none":
        return  # regularisation guarantees a finite optimum

    positive_weight_mask = weights > 0
    if not np.any(positive_weight_mask):
        raise ValueError("Weighted logistic fitting requires at least one positive sample weight.")

    X_positive = X[positive_weight_mask]
    y_positive = y[positive_weight_mask]

    unique_classes = np.unique(y_positive)
    if unique_classes.size < 2:
        raise ValueError(
            "Weighted logistic fitting has no finite optimum without penalization "
            "because the positive-weight data contain only one class."
        )

    signed_labels = np.where(y_positive == 1.0, 1.0, -1.0)
    design_matrix = X_positive
    if fit_intercept:
        design_matrix = np.column_stack([X_positive, np.ones(X_positive.shape[0])])

    # Feasibility of y_i * (x_i^T beta + b) >= 1 certifies linear separability.
    constraint_matrix = -(signed_labels[:, None] * design_matrix)
    rhs = -np.ones(X_positive.shape[0], dtype=float)
    feasibility = linprog(
        c=np.zeros(design_matrix.shape[1], dtype=float),
        A_ub=constraint_matrix,
        b_ub=rhs,
        bounds=[(None, None)] * design_matrix.shape[1],
        method="highs",
    )
    if feasibility.success:
        raise ValueError(
            "Weighted logistic fitting appears linearly separable, so the unpenalized "
            "problem has no finite optimum. Add penalization in a later version or "
            "modify the data/configuration."
        )


def _fit_weighted_logistic_signal(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool,
    max_iter: int,
    tol: float,
    penalty: str = "none",
    alpha: float = 0.0,
) -> tuple[np.ndarray, float | None]:
    _check_logistic_subproblem_has_finite_optimum(
        X,
        y,
        weights,
        fit_intercept=fit_intercept,
        penalty=penalty,
    )

    def objective_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        return _weighted_logistic_objective(
            params,
            X,
            y,
            weights,
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
        estimator_name="logistic",
    )


def _fit_weighted_logistic_signals(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool,
    max_iter: int,
    tol: float,
    penalty: str = "none",
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    return fit_weighted_signals(
        X,
        y,
        weights,
        fit_signal=lambda X, y, sample_weights: _fit_weighted_logistic_signal(
            X,
            y,
            sample_weights,
            fit_intercept=fit_intercept,
            max_iter=max_iter,
            tol=tol,
            penalty=penalty,
            alpha=alpha,
        ),
    )


class WERMLogistic(ClassifierMixin, BaseEstimator):
    """Fixed-changepoint WERM estimator with binary logistic loss.

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
    max_iter : int, default=100
        Maximum number of optimizer iterations in first-stage signal fitting.
    tol : float, default=1e-5
        Optimizer tolerance.
    penalty : {"none", "l1", "l2"}, default="l2"
        Penalty used in the first-stage signal fits.
    alpha : float, default=1.0
        Penalty strength.
    """

    def __init__(
        self,
        num_chgpts,
        *,
        delta=1,
        search_method="efficient",
        fit_intercept=True,
        max_iter=100,
        tol=1e-5,
        penalty="l2",
        alpha=1.0,
    ):
        self.num_chgpts = num_chgpts
        self.delta = delta
        self.search_method = search_method
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.penalty = penalty
        self.alpha = alpha

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        """Fit the estimator on ordered binary observations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Ordered covariates.
        y : array-like of shape (n_samples,)
            Binary labels.

        Returns
        -------
        self : WERMLogistic
            Fitted estimator.
        """
        penalty = validate_penalty(self.penalty)
        alpha = validate_alpha(self.alpha)
        feature_names_in, num_chgpts, delta, X_array, y_array, classes = prepare_fixed_fit_inputs(
            X,
            y,
            num_chgpts=self.num_chgpts,
            delta=self.delta,
            binary=True,
        )
        y_binary = (y_array == classes[1]).astype(float)
        fitted = fit_fixed_werm_model(
            self,
            X_array,
            y_binary,
            num_chgpts=num_chgpts,
            delta=delta,
            search_method=self.search_method,
            feature_names_in=feature_names_in,
            fit_signals=lambda X, y, weights: _fit_weighted_logistic_signals(
                X,
                y,
                weights,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                penalty=penalty,
                alpha=alpha,
            ),
            fit_last_segment_signal=lambda X, y: _fit_weighted_logistic_signal(
                X,
                y,
                np.ones(len(y), dtype=float),
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                penalty=penalty,
                alpha=alpha,
            ),
            loss=_logistic_loss,
            extra_attributes={"classes_": classes},
        )
        self.n_iter_ = np.array([max(1, self.max_iter)], dtype=int)
        return fitted

    def predict_proba(self, X):
        """Estimate class probabilities with the last-segment refit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates to score with the final segment model.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probabilities for ``classes_[0]`` and ``classes_[1]``.
        """
        return logistic_predict_proba(self, X)

    def predict(self, X):
        """Predict class labels with the last-segment refit.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariates to classify with the final segment model.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        probabilities = self.predict_proba(X)[:, 1]
        return np.where(probabilities >= 0.5, self.classes_[1], self.classes_[0])
