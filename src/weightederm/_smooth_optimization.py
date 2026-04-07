from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def fit_weighted_smooth_signal(
    X: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    *,
    fit_intercept: bool,
    objective_and_gradient,
    max_iter: int,
    tol: float,
    estimator_name: str,
) -> tuple[np.ndarray, float | None]:
    if np.sum(weights) <= 0:
        raise ValueError(
            f"Weighted {estimator_name} fitting requires at least one positive sample weight."
        )

    n_features = X.shape[1]
    del X, y

    initial_params = np.zeros(n_features + int(fit_intercept), dtype=float)
    result = minimize(
        objective_and_gradient,
        initial_params,
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": tol},
    )
    if not result.success:
        raise ValueError(f"Weighted {estimator_name} fitting failed to converge: {result.message}")

    if fit_intercept:
        coef = np.asarray(result.x[:-1], dtype=float)
        intercept = float(result.x[-1])
    else:
        coef = np.asarray(result.x, dtype=float)
        intercept = None

    return coef, intercept
