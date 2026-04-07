from __future__ import annotations

from math import comb

import numpy as np


def _safe_comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return comb(n, k)


def compute_exact_marginal_weights(*, num_signals: int, n_samples: int) -> np.ndarray:
    if num_signals < 1:
        raise ValueError("num_signals must be a positive integer.")
    if n_samples < 1:
        raise ValueError("n_samples must be a positive integer.")
    if num_signals > n_samples:
        raise ValueError("num_signals cannot exceed n_samples.")

    denominator = comb(n_samples - 1, num_signals - 1)
    weights = np.zeros((num_signals, n_samples), dtype=float)

    for signal_idx in range(1, num_signals + 1):
        for sample_idx in range(1, n_samples + 1):
            numerator = _safe_comb(sample_idx - 1, signal_idx - 1) * _safe_comb(
                n_samples - sample_idx,
                num_signals - signal_idx,
            )
            weights[signal_idx - 1, sample_idx - 1] = numerator / denominator

    return weights
