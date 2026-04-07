from __future__ import annotations

import numpy as np
from scipy.special import expit
from scipy.sparse import issparse
from sklearn.utils.validation import check_is_fitted


def fit_last_segment_model(
    X: np.ndarray,
    y: np.ndarray,
    changepoints: np.ndarray | list[int] | tuple[int, ...],
    *,
    fit_segment_signal,
) -> tuple[np.ndarray, float | None]:
    changepoints_array = np.asarray(changepoints, dtype=int)
    start = 0 if changepoints_array.size == 0 else int(changepoints_array[-1])
    coef, intercept = fit_segment_signal(X[start:], y[start:])
    return np.asarray(coef, dtype=float), None if intercept is None else float(intercept)


def prepare_prediction_features(estimator, X) -> np.ndarray:
    check_is_fitted(estimator, ["last_segment_coef_", "last_segment_intercept_", "n_features_in_"])

    if issparse(X):
        raise TypeError("Sparse data not supported.")
    X_array = np.asarray(X)
    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like object. Reshape your data.")
    if X_array.shape[1] != estimator.n_features_in_:
        raise ValueError(
            f"X has {X_array.shape[1]} features, but {estimator.__class__.__name__} "
            f"is expecting {estimator.n_features_in_} features as input"
        )
    X_float = X_array.astype(float, copy=False)
    if not np.isfinite(X_float).all():
        raise ValueError("Input contains NaN or inf.")
    return X_float


def linear_predict(estimator, X) -> np.ndarray:
    X_array = prepare_prediction_features(estimator, X)
    predictions = X_array @ estimator.last_segment_coef_
    if estimator.last_segment_intercept_ is not None:
        predictions = predictions + estimator.last_segment_intercept_
    return np.asarray(predictions, dtype=float)


def logistic_predict_proba(estimator, X) -> np.ndarray:
    linear_predictor = linear_predict(estimator, X)
    positive_class_probability = expit(linear_predictor)
    return np.column_stack([1.0 - positive_class_probability, positive_class_probability])
