from __future__ import annotations

import numpy as np

from weightederm._prediction import fit_last_segment_model


def make_interleaved_folds(n_samples: int, cv: int) -> list[np.ndarray]:
    folds = [np.arange(offset, n_samples, cv, dtype=int) for offset in range(cv)]
    if any(len(fold) == 0 for fold in folds):
        raise ValueError(f"Cannot create {cv} non-empty folds with {n_samples} samples.")
    return folds


def segment_bounds_from_changepoints(
    n_samples: int,
    changepoints: np.ndarray | list[int] | tuple[int, ...],
) -> list[tuple[int, int]]:
    changepoints_array = np.asarray(changepoints, dtype=int)
    boundaries = [0, *changepoints_array.tolist(), n_samples]
    return list(zip(boundaries[:-1], boundaries[1:]))


def fit_werm_cv_model(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_num_chgpts: int,
    cv: int,
    fit_fixed_model,
    fit_segmented_model,
    score_segmented_model,
    fit_last_segment_signal=None,
    feature_names_in: np.ndarray | None = None,
    extra_attributes: dict | None = None,
):
    num_chgpts_grid = np.arange(max_num_chgpts + 1, dtype=int)
    folds = make_interleaved_folds(X.shape[0], cv)
    mean_scores = np.array(
        [
            _mean_fold_score_for_num_chgpts(
                X,
                y,
                num_chgpts=num_chgpts,
                folds=folds,
                fit_fixed_model=fit_fixed_model,
                fit_segmented_model=fit_segmented_model,
                score_segmented_model=score_segmented_model,
            )
            for num_chgpts in num_chgpts_grid
        ],
        dtype=float,
    )

    best_idx = int(np.argmin(mean_scores))
    best_num_chgpts = int(num_chgpts_grid[best_idx])
    final_werm = fit_fixed_model(X, y, best_num_chgpts)
    segment_bounds = segment_bounds_from_changepoints(X.shape[0], final_werm.changepoints_)
    segmented_fit = fit_segmented_model(X, y, segment_bounds)
    if fit_last_segment_signal is not None:
        last_segment_coef, last_segment_intercept = fit_last_segment_model(
            X,
            y,
            final_werm.changepoints_,
            fit_segment_signal=fit_last_segment_signal,
        )

    estimator.best_index_ = best_idx
    estimator.best_num_chgpts_ = best_num_chgpts
    estimator.best_score_ = float(mean_scores[best_idx])
    estimator.cv_results_ = {
        "num_chgpts": num_chgpts_grid.copy(),
        "mean_test_score": mean_scores.copy(),
    }
    estimator.num_chgpts_grid_ = num_chgpts_grid
    estimator.changepoints_ = final_werm.changepoints_
    estimator.num_chgpts_ = final_werm.num_chgpts_
    estimator.num_signals_ = final_werm.num_signals_
    estimator.objective_ = final_werm.objective_
    estimator.n_features_in_ = final_werm.n_features_in_
    if feature_names_in is None:
        if hasattr(estimator, "feature_names_in_"):
            delattr(estimator, "feature_names_in_")
    else:
        estimator.feature_names_in_ = feature_names_in
    estimator.segment_bounds_ = segmented_fit.bounds
    estimator.segment_coefs_ = segmented_fit.coefs
    estimator.segment_intercepts_ = segmented_fit.intercepts
    if fit_last_segment_signal is not None:
        estimator.last_segment_coef_ = last_segment_coef
        estimator.last_segment_intercept_ = last_segment_intercept
    estimator._weights_ = final_werm._weights_
    estimator._signal_coefs_ = final_werm._signal_coefs_
    estimator._signal_intercepts_ = final_werm._signal_intercepts_
    estimator._theta_hat_ = final_werm._theta_hat_

    if extra_attributes is not None:
        for name, value in extra_attributes.items():
            setattr(estimator, name, value)

    return estimator


def _map_fold_changepoints_to_global(
    changepoints: np.ndarray,
    train_indices: np.ndarray,
) -> np.ndarray:
    if len(changepoints) == 0:
        return np.array([], dtype=int)
    if np.all(changepoints < len(train_indices)):
        return train_indices[changepoints]
    return np.asarray(changepoints, dtype=int)


def _mean_fold_score_for_num_chgpts(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_chgpts: int,
    folds: list[np.ndarray],
    fit_fixed_model,
    fit_segmented_model,
    score_segmented_model,
) -> float:
    fold_scores = []
    n_samples = X.shape[0]

    for fold_idx, test_indices in enumerate(folds):
        train_indices = np.concatenate(
            [folds[other_idx] for other_idx in range(len(folds)) if other_idx != fold_idx]
        )
        train_indices.sort()

        fitted = fit_fixed_model(X[train_indices], y[train_indices], num_chgpts)
        mapped_changepoints = _map_fold_changepoints_to_global(fitted.changepoints_, train_indices)
        bounds = segment_bounds_from_changepoints(n_samples, mapped_changepoints)
        segmented_fit = fit_segmented_model(X, y, bounds, train_indices=train_indices)
        fold_scores.append(float(score_segmented_model(segmented_fit, X, y, test_indices)))

    return float(np.mean(fold_scores))
