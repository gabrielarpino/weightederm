from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SegmentedFit:
    bounds: list[tuple[int, int]]
    coefs: np.ndarray
    intercepts: np.ndarray | None


def fit_segmented_model(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[int, int]],
    *,
    fit_segment_signal,
    train_indices: np.ndarray | None = None,
) -> SegmentedFit:
    coefs = []
    intercepts = []

    for start, stop in bounds:
        if train_indices is None:
            segment_indices = np.arange(start, stop, dtype=int)
        else:
            segment_mask = (train_indices >= start) & (train_indices < stop)
            segment_indices = train_indices[segment_mask]

        if len(segment_indices) == 0:
            coefs.append(np.zeros(X.shape[1], dtype=float))
            intercepts.append(None)
            continue

        coef, intercept = fit_segment_signal(X[segment_indices], y[segment_indices])
        coefs.append(coef)
        intercepts.append(intercept)

    intercept_array = (
        None
        if all(intercept is None for intercept in intercepts)
        else np.asarray(
            [0.0 if intercept is None else intercept for intercept in intercepts], dtype=float
        )
    )
    return SegmentedFit(bounds=bounds, coefs=np.vstack(coefs), intercepts=intercept_array)


def score_segmented_model(
    segmented_fit: SegmentedFit,
    X: np.ndarray,
    y: np.ndarray,
    test_indices: np.ndarray,
    *,
    loss,
) -> float:
    fold_score = 0.0

    for segment_idx, (start, stop) in enumerate(segmented_fit.bounds):
        segment_test_mask = (test_indices >= start) & (test_indices < stop)
        segment_test_indices = test_indices[segment_test_mask]
        if len(segment_test_indices) == 0:
            continue

        predictions = X[segment_test_indices] @ segmented_fit.coefs[segment_idx]
        if segmented_fit.intercepts is not None:
            predictions = predictions + segmented_fit.intercepts[segment_idx]

        fold_score += float(np.sum(loss(predictions, y[segment_test_indices])))

    return fold_score
