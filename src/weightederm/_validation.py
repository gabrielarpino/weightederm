from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.sparse import issparse
from sklearn.exceptions import DataConversionWarning
from sklearn.utils.multiclass import type_of_target
import warnings


def validate_num_chgpts(num_chgpts: int) -> int:
    if isinstance(num_chgpts, bool) or not isinstance(num_chgpts, Integral):
        raise ValueError("num_chgpts must be a non-negative integer.")
    if num_chgpts < 0:
        raise ValueError("num_chgpts must be a non-negative integer.")
    return int(num_chgpts)


def validate_max_num_chgpts(max_num_chgpts: int) -> int:
    try:
        return validate_num_chgpts(max_num_chgpts)
    except ValueError as exc:
        raise ValueError("max_num_chgpts must be a non-negative integer.") from exc


def validate_delta(delta: int) -> int:
    if isinstance(delta, bool) or not isinstance(delta, Integral):
        raise ValueError("delta must be a positive integer.")
    if delta < 1:
        raise ValueError("delta must be a positive integer.")
    return int(delta)


def validate_num_chgpts_for_n_samples(num_chgpts: int, n_samples: int) -> int:
    if num_chgpts >= n_samples:
        raise ValueError("num_chgpts must be less than the number of samples.")
    return num_chgpts


def validate_max_num_chgpts_for_n_samples(max_num_chgpts: int, n_samples: int) -> int:
    if max_num_chgpts >= n_samples:
        raise ValueError("max_num_chgpts must be less than the number of samples.")
    return max_num_chgpts


def validate_cv(cv: int, n_samples: int) -> int:
    if isinstance(cv, bool) or not isinstance(cv, Integral):
        raise ValueError("cv must be an integer greater than or equal to 2.")
    if cv < 2:
        raise ValueError("cv must be an integer greater than or equal to 2.")
    if cv > n_samples:
        raise ValueError(f"cv cannot exceed the number of samples; got n_samples={n_samples}.")
    return int(cv)


def validate_huber_epsilon(epsilon: float) -> float:
    try:
        epsilon_value = float(epsilon)
    except (TypeError, ValueError) as exc:
        raise ValueError("epsilon must be a real number greater than or equal to 1.") from exc

    if epsilon_value < 1.0:
        raise ValueError("epsilon must be a real number greater than or equal to 1.")
    return epsilon_value


def validate_fit_solver(fit_solver: str) -> str:
    if fit_solver not in {"direct", "lbfgsb"}:
        raise ValueError("fit_solver must be either 'direct' or 'lbfgsb'.")
    return fit_solver


def validate_penalty(penalty: str) -> str:
    if penalty not in {"none", "l1", "l2"}:
        raise ValueError("penalty must be 'none', 'l1', or 'l2'.")
    return penalty


def validate_alpha(alpha: float) -> float:
    try:
        alpha_value = float(alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("alpha must be a non-negative real number.") from exc
    if alpha_value < 0.0:
        raise ValueError("alpha must be a non-negative real number.")
    return alpha_value


def extract_feature_names_in(X) -> np.ndarray | None:
    columns = getattr(X, "columns", None)
    if columns is None:
        return None

    feature_names = np.asarray(columns, dtype=object)
    if feature_names.ndim != 1:
        return None
    if not all(isinstance(name, str) for name in feature_names.tolist()):
        return None
    return feature_names


def validate_fit_data(X, y) -> tuple[np.ndarray, np.ndarray]:
    if issparse(X):
        raise TypeError("Sparse data not supported.")
    if y is None:
        raise ValueError("requires y to be passed, but the target y is None")
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like object. Reshape your data.")
    if X_array.shape[1] < 1:
        raise ValueError(
            f"0 feature(s) (shape=({X_array.shape[0]}, 0)) while a minimum of 1 is required."
        )
    if y_array.ndim == 2 and y_array.shape[1] == 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected. "
            "Please change the shape of y to (n_samples,).",
            DataConversionWarning,
            stacklevel=2,
        )
        y_array = y_array.ravel()
    elif y_array.ndim != 1:
        raise ValueError("y must be a 1D array-like object.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X_array.shape[0] < 1:
        raise ValueError("X and y must contain at least one sample.")
    if np.iscomplexobj(X_array) or np.iscomplexobj(y_array):
        raise ValueError("Complex data not supported")
    X_float = X_array.astype(float, copy=False)
    y_float = y_array.astype(float, copy=False)
    if not np.isfinite(X_float).all() or not np.isfinite(y_float).all():
        raise ValueError("Input contains NaN or inf.")
    return X_float, y_float


def validate_binary_classification_data(X, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if issparse(X):
        raise TypeError("Sparse data not supported.")
    if y is None:
        raise ValueError("requires y to be passed, but the target y is None")
    X_array = np.asarray(X)
    y_array = np.asarray(y)

    if X_array.ndim != 2:
        raise ValueError("X must be a 2D array-like object. Reshape your data.")
    if X_array.shape[1] < 1:
        raise ValueError(
            f"0 feature(s) (shape=({X_array.shape[0]}, 0)) while a minimum of 1 is required."
        )
    if y_array.ndim == 2 and y_array.shape[1] == 1:
        warnings.warn(
            "A column-vector y was passed when a 1d array was expected. "
            "Please change the shape of y to (n_samples,).",
            DataConversionWarning,
            stacklevel=2,
        )
        y_array = y_array.ravel()
    elif y_array.ndim != 1:
        raise ValueError("y must be a 1D array-like object.")
    if X_array.shape[0] != y_array.shape[0]:
        raise ValueError("X and y must contain the same number of samples.")
    if X_array.shape[0] < 1:
        raise ValueError("X and y must contain at least one sample.")
    if np.iscomplexobj(X_array) or np.iscomplexobj(y_array):
        raise ValueError("Complex data not supported")
    X_float = X_array.astype(float, copy=False)
    if not np.isfinite(X_float).all():
        raise ValueError("Input contains NaN or inf.")
    if np.issubdtype(y_array.dtype, np.number):
        y_numeric = y_array.astype(float, copy=False)
        if not np.isfinite(y_numeric).all():
            raise ValueError("Input contains NaN or inf.")

    target_type = type_of_target(y_array, raise_unknown=True)
    if target_type.startswith("continuous"):
        raise ValueError(f"Unknown label type: {target_type}")
    if target_type != "binary":
        raise ValueError(
            f"Only binary classification is supported. The type of the target is {target_type}."
        )

    classes = np.unique(y_array)
    if classes.size != 2:
        if classes.size == 1:
            raise ValueError("y must contain exactly two classes; got 1 class.")
        raise ValueError("y must contain exactly two classes.")

    return X_float, y_array, classes


def prepare_fixed_fit_inputs(
    X,
    y,
    *,
    num_chgpts: int,
    delta: int,
    binary: bool = False,
) -> tuple[np.ndarray | None, int, int, np.ndarray, np.ndarray, np.ndarray | None]:
    feature_names_in = extract_feature_names_in(X)
    validated_num_chgpts = validate_num_chgpts(num_chgpts)
    validated_delta = validate_delta(delta)

    if binary:
        X_array, y_array, classes = validate_binary_classification_data(X, y)
    else:
        X_array, y_array = validate_fit_data(X, y)
        classes = None

    validated_num_chgpts = validate_num_chgpts_for_n_samples(
        validated_num_chgpts,
        X_array.shape[0],
    )
    return feature_names_in, validated_num_chgpts, validated_delta, X_array, y_array, classes


def prepare_cv_fit_inputs(
    X,
    y,
    *,
    max_num_chgpts: int,
    delta: int,
    cv: int,
    binary: bool = False,
) -> tuple[np.ndarray | None, int, int, int, np.ndarray, np.ndarray, np.ndarray | None]:
    feature_names_in = extract_feature_names_in(X)
    validated_max_num_chgpts = validate_max_num_chgpts(max_num_chgpts)
    validated_delta = validate_delta(delta)

    if binary:
        X_array, y_array, classes = validate_binary_classification_data(X, y)
    else:
        X_array, y_array = validate_fit_data(X, y)
        classes = None

    validated_max_num_chgpts = validate_max_num_chgpts_for_n_samples(
        validated_max_num_chgpts,
        X_array.shape[0],
    )
    validated_cv = validate_cv(cv, X_array.shape[0])
    return (
        feature_names_in,
        validated_max_num_chgpts,
        validated_delta,
        validated_cv,
        X_array,
        y_array,
        classes,
    )
