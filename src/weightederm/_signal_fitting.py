from __future__ import annotations

import numpy as np


def fit_weighted_signals(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_signal,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    num_signals = weights.shape[0]
    coefs = []
    intercepts = []
    theta_hat = np.empty((X.shape[0], num_signals), dtype=float)

    for signal_idx in range(num_signals):
        coef, intercept = fit_signal(X, y, weights[signal_idx])
        coefs.append(coef)
        if intercept is None:
            theta_hat[:, signal_idx] = X @ coef
        else:
            intercepts.append(intercept)
            theta_hat[:, signal_idx] = X @ coef + intercept

    intercept_array = np.asarray(intercepts, dtype=float) if intercepts else None
    return np.vstack(coefs), intercept_array, theta_hat
