from __future__ import annotations

from itertools import combinations

import numpy as np


def _generate_changepoint_configs(
    n_samples: int,
    num_chgpts: int,
    delta: int,
):
    if num_chgpts == 0:
        yield ()
        return

    for changepoints in combinations(range(1, n_samples), num_chgpts):
        if all(right - left >= delta for left, right in zip(changepoints[:-1], changepoints[1:])):
            yield changepoints


def _objective_for_config(
    theta_hat: np.ndarray,
    y: np.ndarray,
    *,
    loss,
    changepoints: tuple[int, ...],
) -> float:
    boundaries = (0, *changepoints, len(y))
    objective = 0.0

    for signal_idx, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        losses = loss(theta_hat[start:stop, signal_idx], y[start:stop])
        objective += float(np.sum(losses))

    return objective


def _search_changepoints_brute_force(
    theta_hat: np.ndarray,
    y: np.ndarray,
    *,
    loss,
    num_chgpts: int,
    delta: int,
) -> tuple[np.ndarray, float]:
    best_changepoints = None
    best_objective = None

    for changepoints in _generate_changepoint_configs(len(y), num_chgpts, delta):
        objective = _objective_for_config(theta_hat, y, loss=loss, changepoints=changepoints)
        if best_objective is None or objective < best_objective:
            best_objective = objective
            best_changepoints = changepoints

    if best_changepoints is None or best_objective is None:
        raise ValueError("No valid changepoint configuration exists for the given delta.")

    return np.asarray(best_changepoints, dtype=int), float(best_objective)


def _local_refine_changepoints(
    theta_hat: np.ndarray,
    y: np.ndarray,
    changepoints: list[int],
    *,
    loss,
    delta: int,
    max_iterations: int = 5,
) -> list[int]:
    if not changepoints:
        return changepoints

    n_samples = len(y)
    refined = sorted(changepoints)

    for _ in range(max_iterations):
        improved = False
        for idx, current_pos in enumerate(refined):
            best_pos = current_pos
            best_objective = _objective_for_config(
                theta_hat,
                y,
                loss=loss,
                changepoints=tuple(refined),
            )

            for offset in (-2, -1, 1, 2):
                candidate = current_pos + offset
                if candidate < 1 or candidate >= n_samples:
                    continue

                candidate_changepoints = refined.copy()
                candidate_changepoints[idx] = candidate
                candidate_changepoints.sort()
                if any(
                    right - left < delta
                    for left, right in zip(candidate_changepoints[:-1], candidate_changepoints[1:])
                ):
                    continue

                objective = _objective_for_config(
                    theta_hat,
                    y,
                    loss=loss,
                    changepoints=tuple(candidate_changepoints),
                )
                if objective < best_objective:
                    best_objective = objective
                    best_pos = candidate

            if best_pos != current_pos:
                refined[idx] = best_pos
                refined.sort()
                improved = True

        if not improved:
            break

    return refined


def _search_changepoints_efficient(
    theta_hat: np.ndarray,
    y: np.ndarray,
    *,
    loss,
    num_chgpts: int,
    delta: int,
) -> tuple[np.ndarray, float]:
    if num_chgpts == 0:
        return np.array([], dtype=int), _objective_for_config(
            theta_hat[:, :1],
            y,
            loss=loss,
            changepoints=(),
        )

    changepoints: list[int] = []
    n_samples = len(y)

    for _ in range(num_chgpts):
        best_position = None
        best_objective = None

        for candidate in range(1, n_samples):
            if any(abs(candidate - existing) < delta for existing in changepoints):
                continue

            test_changepoints = sorted(changepoints + [candidate])
            objective = _objective_for_config(
                theta_hat[:, : len(test_changepoints) + 1],
                y,
                loss=loss,
                changepoints=tuple(test_changepoints),
            )
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_position = candidate

        if best_position is None or best_objective is None:
            raise ValueError("No valid changepoint configuration exists for the given delta.")

        changepoints.append(best_position)
        changepoints.sort()

    changepoints = _local_refine_changepoints(
        theta_hat,
        y,
        changepoints,
        loss=loss,
        delta=delta,
    )
    objective = _objective_for_config(theta_hat, y, loss=loss, changepoints=tuple(changepoints))
    return np.asarray(changepoints, dtype=int), float(objective)


def search_changepoints(
    theta_hat: np.ndarray,
    y: np.ndarray,
    *,
    loss,
    num_chgpts: int,
    delta: int,
    search_method: str,
) -> tuple[np.ndarray, float]:
    if search_method == "brute_force":
        return _search_changepoints_brute_force(
            theta_hat,
            y,
            loss=loss,
            num_chgpts=num_chgpts,
            delta=delta,
        )
    if search_method == "efficient":
        return _search_changepoints_efficient(
            theta_hat,
            y,
            loss=loss,
            num_chgpts=num_chgpts,
            delta=delta,
        )

    raise ValueError("search_method must be either 'efficient' or 'brute_force'.")
