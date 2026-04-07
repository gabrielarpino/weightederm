from __future__ import annotations

import numpy as np

from weightederm._prediction import fit_last_segment_model
from weightederm._search import search_changepoints
from weightederm._weights import compute_exact_marginal_weights


def fit_fixed_werm_model(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_chgpts: int,
    delta: int,
    search_method: str,
    fit_signals,
    loss,
    fit_last_segment_signal=None,
    feature_names_in: np.ndarray | None = None,
    extra_attributes: dict | None = None,
):
    if search_method not in {"brute_force", "efficient"}:
        raise ValueError("search_method must be either 'efficient' or 'brute_force'.")

    num_signals = num_chgpts + 1
    weights = compute_exact_marginal_weights(
        num_signals=num_signals,
        n_samples=X.shape[0],
    )
    signal_coefs, signal_intercepts, theta_hat = fit_signals(X, y, weights)
    changepoints, objective = search_changepoints(
        theta_hat,
        y,
        loss=loss,
        num_chgpts=num_chgpts,
        delta=delta,
        search_method=search_method,
    )
    if fit_last_segment_signal is not None:
        last_segment_coef, last_segment_intercept = fit_last_segment_model(
            X,
            y,
            changepoints,
            fit_segment_signal=fit_last_segment_signal,
        )
        estimator.last_segment_coef_ = last_segment_coef
        estimator.last_segment_intercept_ = last_segment_intercept

    estimator.changepoints_ = changepoints
    estimator.num_chgpts_ = num_chgpts
    estimator.num_signals_ = num_signals
    estimator.objective_ = objective
    estimator.n_features_in_ = X.shape[1]
    if feature_names_in is None:
        if hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")
    else:
        estimator.feature_names_in_ = feature_names_in
    estimator._weights_ = weights
    estimator._signal_coefs_ = signal_coefs
    estimator._signal_intercepts_ = signal_intercepts
    estimator._theta_hat_ = theta_hat

    if extra_attributes is not None:
        for name, value in extra_attributes.items():
            setattr(estimator, name, value)

    return estimator
