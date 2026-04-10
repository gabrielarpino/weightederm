from __future__ import annotations

import csv
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

from joblib import Parallel, delayed

import numpy as np
from scipy.spatial.distance import directed_hausdorff

from weightederm._cv_estimators import WERMLeastSquaresCV, WERMLogisticCV
from weightederm._least_squares import WERMLeastSquares
from weightederm._logistic import WERMLogistic


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    p: int
    num_signals: int
    delta_ratios: tuple[float, ...]
    fractional_chgpt_locations: tuple[float, ...]
    delta_fraction: float


@dataclass(frozen=True)
class TrialData:
    X_fit: np.ndarray
    y_fit: np.ndarray
    true_changepoints: np.ndarray
    X_mcscan: np.ndarray | None = None
    y_mcscan: np.ndarray | None = None


def reference_like_benchmark_specs() -> dict[str, ExperimentSpec]:
    return {
        "M1": ExperimentSpec(
            name="M1",
            p=200,
            num_signals=3,
            delta_ratios=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
            fractional_chgpt_locations=(2 / 5, 7 / 10),
            delta_fraction=1.0 / 20.0,
        ),
        "M2": ExperimentSpec(
            name="M2",
            p=200,
            num_signals=4,
            delta_ratios=(0.5, 1.5, 2.5, 3.5, 4.5),
            fractional_chgpt_locations=(0.4, 0.7),
            delta_fraction=1.0 / 10.0,
        ),
        "M3": ExperimentSpec(
            name="M3",
            p=200,
            num_signals=2,
            delta_ratios=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0),
            fractional_chgpt_locations=(1.0 / 3.0,),
            delta_fraction=1.0 / 20.0,
        ),
    }


def hausdorff_distance(x1: list[int] | np.ndarray, x2: list[int] | np.ndarray) -> float:
    x1_arr = np.asarray(x1, dtype=float).reshape(-1, 1)
    x2_arr = np.asarray(x2, dtype=float).reshape(-1, 1)
    if x1_arr.size == 0 and x2_arr.size == 0:
        return 0.0
    if x1_arr.size == 0 or x2_arr.size == 0:
        return math.inf
    return max(
        directed_hausdorff(x1_arr, x2_arr)[0],
        directed_hausdorff(x2_arr, x1_arr)[0],
    )


def normalized_hausdorff_distance(
    x1: list[int] | np.ndarray,
    x2: list[int] | np.ndarray,
    *,
    n_samples: int,
) -> float:
    return hausdorff_distance(x1, x2) / float(n_samples)


def simulate_trial(spec: ExperimentSpec, delta_ratio: float, seed: int) -> TrialData:
    rng = np.random.default_rng(seed)
    n = int(delta_ratio * spec.p)
    true_changepoints = (n * np.asarray(spec.fractional_chgpt_locations)).astype(int)
    signal_ids = _segment_ids(n, true_changepoints)

    if spec.name == "M1":
        kappa = 1.0
        alpha = 0.5
        noise_std = np.sqrt(0.6)
        X = rng.normal(0.0, np.sqrt(1.0 / n), size=(n, spec.p))
        X = X @ np.linalg.cholesky(_ar1_covariance(spec.p, 0.2)).T
        B_tilde = _sparse_gaussian_signal(rng, spec.p, spec.num_signals, alpha, delta_ratio, kappa)
        theta = (X * np.sqrt(n)) @ (B_tilde / np.sqrt(n))
        y = theta[np.arange(n), signal_ids] + rng.normal(0.0, noise_std, size=n)
        y = y / (alpha * kappa**2 + noise_std**2)
        X_scaled = X * np.sqrt(n)
        return TrialData(
            X_fit=X_scaled,
            y_fit=y,
            true_changepoints=true_changepoints,
            X_mcscan=X_scaled,
            y_mcscan=y,
        )

    if spec.name == "M2":
        alpha = 0.5
        var_beta_1 = 8.0
        sigma_w = np.sqrt(400.0 * delta_ratio)
        noise_scale = np.sqrt(0.1)
        X = rng.normal(0.0, np.sqrt(1.0 / n), size=(n, spec.p))
        B_tilde = _sparse_diff_signal(rng, spec.p, spec.num_signals, alpha, var_beta_1, sigma_w)
        theta = (X * np.sqrt(n)) @ (B_tilde / np.sqrt(n))
        y = theta[np.arange(n), signal_ids] + rng.normal(0.0, noise_scale, size=n)
        return TrialData(
            X_fit=X,
            y_fit=y,
            true_changepoints=true_changepoints,
            X_mcscan=X * np.sqrt(n),
            y_mcscan=y,
        )

    if spec.name == "M3":
        kappa = 1.0
        alpha = 0.5
        X = rng.normal(0.0, np.sqrt(1.0 / n), size=(n, spec.p))
        B_tilde = _sparse_gaussian_signal(rng, spec.p, spec.num_signals, alpha, delta_ratio, kappa)
        theta = (X * np.sqrt(n)) @ (B_tilde / np.sqrt(n))
        eta = theta[np.arange(n), signal_ids]
        y = (rng.uniform(size=n) < _sigmoid(eta)).astype(int)
        return TrialData(X_fit=X, y_fit=y, true_changepoints=true_changepoints)

    raise ValueError(f"Unknown experiment name: {spec.name}")


def fit_werm_changepoints(
    spec: ExperimentSpec,
    trial: TrialData,
    *,
    least_squares_fit_solver: str = "direct",
) -> np.ndarray:
    delta = max(1, int(trial.X_fit.shape[0] * spec.delta_fraction))
    if spec.name in {"M1", "M2"}:
        estimator = WERMLeastSquares(
            num_chgpts=len(spec.fractional_chgpt_locations),
            delta=delta,
            search_method="efficient",
            fit_intercept=False,
            fit_solver=least_squares_fit_solver,
        )
        return np.asarray(estimator.fit(trial.X_fit, trial.y_fit).changepoints_, dtype=int)

    if spec.name == "M3":
        estimator = WERMLogistic(
            num_chgpts=spec.num_signals - 1,
            delta=delta,
            search_method="brute_force",
            fit_intercept=False,
            max_iter=500,
            tol=1e-6,
        )
        return np.asarray(estimator.fit(trial.X_fit, trial.y_fit).changepoints_, dtype=int)

    raise ValueError(f"Unknown experiment name: {spec.name}")


def fit_werm_unknown_changepoints(spec: ExperimentSpec, trial: TrialData) -> np.ndarray:
    delta = max(1, int(trial.X_fit.shape[0] * spec.delta_fraction))
    if spec.name in {"M1", "M2"}:
        estimator = WERMLeastSquaresCV(
            max_num_chgpts=spec.num_signals - 1,
            delta=delta,
            search_method="efficient",
            cv=5,
            fit_intercept=False,
        )
        return np.asarray(estimator.fit(trial.X_fit, trial.y_fit).changepoints_, dtype=int)

    if spec.name == "M3":
        estimator = WERMLogisticCV(
            max_num_chgpts=spec.num_signals - 1,
            delta=delta,
            search_method="efficient",
            cv=5,
            fit_intercept=False,
            max_iter=500,
            tol=1e-6,
        )
        return np.asarray(estimator.fit(trial.X_fit, trial.y_fit).changepoints_, dtype=int)

    raise ValueError(f"Unknown experiment name: {spec.name}")


def maybe_run_mcscan_changepoints(
    spec: ExperimentSpec,
    trial: TrialData,
    *,
    mode: str = "fixed",
) -> tuple[np.ndarray | None, str | None]:
    if trial.X_mcscan is None or trial.y_mcscan is None:
        return None, "McScan is only configured for the linear benchmark variants."

    if mode not in {"fixed", "auto"}:
        return None, f"Unsupported McScan mode: {mode}."

    try:
        if mode == "fixed":
            raw = _run_mcscan_backend(
                np.asarray(trial.X_mcscan),
                np.asarray(trial.y_mcscan).flatten(),
                num_chgpts=len(spec.fractional_chgpt_locations),
                method="not",
            )
        else:
            raw = _run_mcscan_backend(
                np.asarray(trial.X_mcscan),
                np.asarray(trial.y_mcscan).flatten(),
                num_chgpts=None,
                method="auto",
            )
        changepoints = raw[raw != -1]
        if changepoints.size > 0 and np.min(changepoints) >= 1:
            changepoints = changepoints - 1
        return changepoints.astype(int), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _run_mcscan_backend(
    X: np.ndarray,
    y: np.ndarray,
    *,
    num_chgpts: int | None,
    method: str,
) -> np.ndarray:
    import rpy2.robjects.numpy2ri
    from rpy2.robjects import conversion, default_converter
    from rpy2.robjects.packages import importr

    inferchange_package = importr("inferchange")
    converter = default_converter + rpy2.robjects.numpy2ri.converter

    with conversion.localconverter(converter):
        if num_chgpts is not None:
            result = inferchange_package.McScan(np.asarray(X), np.asarray(y), num_chgpts)
        else:
            result = inferchange_package.McScan(np.asarray(X), np.asarray(y), method=method)

    return np.asarray(result[0])


def _run_single_fixed_trial(
    spec: ExperimentSpec,
    delta_index: int,
    delta_ratio: float,
    trial_index: int,
    base_seed: int,
    include_mcscan: bool,
    mcscan_mode: str,
    least_squares_fit_solver: str,
) -> list[dict[str, object]]:
    seed = base_seed + 1_000 * delta_index + trial_index
    trial = simulate_trial(spec, delta_ratio, seed)
    rows: list[dict[str, object]] = []

    try:
        werm_cp = fit_werm_changepoints(
            spec,
            trial,
            least_squares_fit_solver=least_squares_fit_solver,
        )
        werm_status = "ok"
        werm_hausdorff = normalized_hausdorff_distance(
            trial.true_changepoints,
            werm_cp,
            n_samples=trial.X_fit.shape[0],
        )
    except Exception as exc:  # noqa: BLE001
        werm_cp = np.array([], dtype=int)
        werm_status = f"error: {exc}"
        werm_hausdorff = np.nan

    rows.append(
        {
            "experiment": spec.name,
            "method": "WERM",
            "delta_ratio": float(delta_ratio),
            "trial": trial_index,
            "hausdorff": float(werm_hausdorff),
            "status": werm_status,
            "predicted_num_chgpts": int(np.asarray(werm_cp, dtype=int).size),
            "estimated_changepoints": _serialize_changepoints(werm_cp),
            "true_changepoints": _serialize_changepoints(trial.true_changepoints),
        }
    )

    if include_mcscan and spec.name in {"M1", "M2"}:
        mcscan_cp, mcscan_error = maybe_run_mcscan_changepoints(spec, trial, mode=mcscan_mode)
        mcscan_hausdorff = (
            np.nan
            if mcscan_cp is None
            else normalized_hausdorff_distance(
                trial.true_changepoints,
                mcscan_cp,
                n_samples=trial.X_fit.shape[0],
            )
        )
        rows.append(
            {
                "experiment": spec.name,
                "method": "McScan",
                "delta_ratio": float(delta_ratio),
                "trial": trial_index,
                "hausdorff": float(mcscan_hausdorff),
                "status": "ok" if mcscan_error is None else f"error: {mcscan_error}",
                "predicted_num_chgpts": 0 if mcscan_cp is None else int(mcscan_cp.size),
                "estimated_changepoints": _serialize_changepoints(mcscan_cp),
                "true_changepoints": _serialize_changepoints(trial.true_changepoints),
            }
        )

    return rows


def run_benchmark(
    spec: ExperimentSpec,
    num_trials: int,
    include_mcscan: bool = False,
    base_seed: int = 42,
    mcscan_mode: str = "fixed",
    least_squares_fit_solver: str = "direct",
    n_jobs: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    task_args = [
        (
            spec,
            delta_index,
            delta_ratio,
            trial_index,
            base_seed,
            include_mcscan,
            mcscan_mode,
            least_squares_fit_solver,
        )
        for delta_index, delta_ratio in enumerate(spec.delta_ratios)
        for trial_index in range(num_trials)
    ]

    if n_jobs == 1:
        results = [_run_single_fixed_trial(*args) for args in task_args]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_fixed_trial)(*args) for args in task_args
        )

    rows = [row for trial_rows in results for row in trial_rows]
    return rows, summarize_trial_rows(rows)


def _run_single_unknown_trial(
    spec: ExperimentSpec,
    delta_index: int,
    delta_ratio: float,
    trial_index: int,
    base_seed: int,
    include_mcscan: bool,
) -> list[dict[str, object]]:
    seed = base_seed + 1_000 * delta_index + trial_index
    trial = simulate_trial(spec, delta_ratio, seed)
    rows: list[dict[str, object]] = []

    try:
        werm_cp = fit_werm_unknown_changepoints(spec, trial)
        werm_status = "ok"
        werm_hausdorff = normalized_hausdorff_distance(
            trial.true_changepoints,
            werm_cp,
            n_samples=trial.X_fit.shape[0],
        )
    except Exception as exc:  # noqa: BLE001
        werm_cp = np.array([], dtype=int)
        werm_status = f"error: {exc}"
        werm_hausdorff = np.nan

    rows.append(
        {
            "experiment": spec.name,
            "method": "WERM",
            "delta_ratio": float(delta_ratio),
            "trial": trial_index,
            "hausdorff": float(werm_hausdorff),
            "status": werm_status,
            "predicted_num_chgpts": int(np.asarray(werm_cp, dtype=int).size),
            "estimated_changepoints": _serialize_changepoints(werm_cp),
            "true_changepoints": _serialize_changepoints(trial.true_changepoints),
        }
    )

    if include_mcscan and spec.name in {"M1", "M2"}:
        mcscan_cp, mcscan_error = maybe_run_mcscan_changepoints(spec, trial, mode="auto")
        mcscan_hausdorff = (
            np.nan
            if mcscan_cp is None
            else normalized_hausdorff_distance(
                trial.true_changepoints,
                mcscan_cp,
                n_samples=trial.X_fit.shape[0],
            )
        )
        rows.append(
            {
                "experiment": spec.name,
                "method": "McScan",
                "delta_ratio": float(delta_ratio),
                "trial": trial_index,
                "hausdorff": float(mcscan_hausdorff),
                "status": "ok" if mcscan_error is None else f"error: {mcscan_error}",
                "predicted_num_chgpts": 0 if mcscan_cp is None else int(mcscan_cp.size),
                "estimated_changepoints": _serialize_changepoints(mcscan_cp),
                "true_changepoints": _serialize_changepoints(trial.true_changepoints),
            }
        )

    return rows


def run_benchmark_unknown(
    spec: ExperimentSpec,
    num_trials: int,
    include_mcscan: bool = False,
    base_seed: int = 42,
    n_jobs: int = 1,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    task_args = [
        (spec, delta_index, delta_ratio, trial_index, base_seed, include_mcscan)
        for delta_index, delta_ratio in enumerate(spec.delta_ratios)
        for trial_index in range(num_trials)
    ]

    if n_jobs == 1:
        results = [_run_single_unknown_trial(*args) for args in task_args]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_unknown_trial)(*args) for args in task_args
        )

    rows = [row for trial_rows in results for row in trial_rows]
    return rows, summarize_trial_rows(rows)


def summarize_trial_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, float], list[float]] = {}
    grouped_predicted_counts: dict[tuple[str, str, float], list[float]] = {}
    for row in rows:
        key = (
            str(row["experiment"]),
            str(row["method"]),
            float(row["delta_ratio"]),
        )
        grouped.setdefault(key, [])
        grouped_predicted_counts.setdefault(key, [])
        hausdorff = float(row["hausdorff"])
        grouped[key].append(hausdorff)
        predicted_num_chgpts = (
            float(row["predicted_num_chgpts"]) if "predicted_num_chgpts" in row else np.nan
        )
        grouped_predicted_counts[key].append(predicted_num_chgpts)

    summary: list[dict[str, object]] = []
    for experiment, method, delta_ratio in sorted(grouped):
        values = np.asarray(grouped[(experiment, method, delta_ratio)], dtype=float)
        predicted_counts = np.asarray(
            grouped_predicted_counts[(experiment, method, delta_ratio)],
            dtype=float,
        )
        summary_values = np.where(np.isinf(values), np.nan, values)
        num_infinite_hausdorff = int(np.isinf(values).sum())
        num_nan_hausdorff = int(np.isnan(values).sum())
        num_finite_hausdorff = int(np.isfinite(values).sum())
        mean_value = (
            float(np.nanmean(summary_values)) if not np.all(np.isnan(summary_values)) else np.nan
        )
        median_value = (
            float(np.nanmedian(summary_values)) if not np.all(np.isnan(summary_values)) else np.nan
        )
        mean_predicted_num_chgpts = (
            float(np.nanmean(predicted_counts))
            if not np.all(np.isnan(predicted_counts))
            else np.nan
        )
        median_predicted_num_chgpts = (
            float(np.nanmedian(predicted_counts))
            if not np.all(np.isnan(predicted_counts))
            else np.nan
        )
        p25_value = (
            float(np.nanpercentile(summary_values, 25))
            if not np.all(np.isnan(summary_values))
            else np.nan
        )
        p75_value = (
            float(np.nanpercentile(summary_values, 75))
            if not np.all(np.isnan(summary_values))
            else np.nan
        )
        p25_predicted = (
            float(np.nanpercentile(predicted_counts, 25))
            if not np.all(np.isnan(predicted_counts))
            else np.nan
        )
        p75_predicted = (
            float(np.nanpercentile(predicted_counts, 75))
            if not np.all(np.isnan(predicted_counts))
            else np.nan
        )
        if num_finite_hausdorff == 0:
            if (
                num_infinite_hausdorff > 0
                and not np.any(np.isnan(predicted_counts))
                and np.all(predicted_counts == 0)
            ):
                warnings.warn(
                    (
                        f"{experiment} {method} at delta_ratio={delta_ratio} predicted zero "
                        "changepoints across all trials, so mean_hausdorff and "
                        "median_hausdorff are NaN after reference-style inf filtering."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
            elif num_nan_hausdorff > 0:
                warnings.warn(
                    (
                        f"{experiment} {method} at delta_ratio={delta_ratio} has no finite "
                        "Hausdorff values; this summary group is NaN because all trials failed "
                        "or produced non-finite results."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
        summary.append(
            {
                "experiment": experiment,
                "method": method,
                "delta_ratio": delta_ratio,
                "mean_hausdorff": mean_value,
                "median_hausdorff": median_value,
                "p25_hausdorff": p25_value,
                "p75_hausdorff": p75_value,
                "num_trials": int(values.size),
                "mean_predicted_num_chgpts": mean_predicted_num_chgpts,
                "median_predicted_num_chgpts": median_predicted_num_chgpts,
                "p25_predicted_num_chgpts": p25_predicted,
                "p75_predicted_num_chgpts": p75_predicted,
                "num_infinite_hausdorff": num_infinite_hausdorff,
                "num_nan_hausdorff": num_nan_hausdorff,
                "num_finite_hausdorff": num_finite_hausdorff,
            }
        )
    return summary


def write_rows_to_csv(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must be non-empty.")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_hausdorff_summary(
    summary_rows: list[dict[str, object]],
    experiment_name: str,
):
    import matplotlib.pyplot as plt

    filtered = [row for row in summary_rows if row["experiment"] == experiment_name]
    if not filtered:
        raise ValueError(f"No summary rows found for experiment {experiment_name}.")

    fig, ax = plt.subplots(figsize=(7, 4))
    methods = sorted({str(row["method"]) for row in filtered})
    for method in methods:
        method_rows = [
            row
            for row in filtered
            if row["method"] == method and np.isfinite(row["mean_hausdorff"])
        ]
        method_rows.sort(key=lambda row: float(row["delta_ratio"]))
        if not method_rows:
            continue
        ax.plot(
            [float(row["delta_ratio"]) for row in method_rows],
            [float(row["mean_hausdorff"]) for row in method_rows],
            marker="o",
            label=method,
        )

    ax.set_title(f"{experiment_name}: mean Hausdorff distance")
    ax.set_xlabel(r"$\delta = n/p$")
    ax.set_ylabel("Hausdorff distance")
    ax.legend()
    ax.grid(alpha=0.25)
    return fig, ax


def plot_hausdorff_summary_with_percentiles(
    summary_rows: list[dict[str, object]],
    experiment_name: str,
):
    import matplotlib.pyplot as plt

    filtered = [row for row in summary_rows if row["experiment"] == experiment_name]
    if not filtered:
        raise ValueError(f"No summary rows found for experiment {experiment_name}.")

    fig, (ax_h, ax_n) = plt.subplots(1, 2, figsize=(12, 4))
    methods = sorted({str(row["method"]) for row in filtered})

    for method in methods:
        method_rows = [
            row
            for row in filtered
            if row["method"] == method and np.isfinite(row["mean_hausdorff"])
        ]
        method_rows.sort(key=lambda row: float(row["delta_ratio"]))
        if not method_rows:
            continue
        delta_ratios = [float(row["delta_ratio"]) for row in method_rows]
        means = [float(row["mean_hausdorff"]) for row in method_rows]
        yerr_lo = np.clip(
            np.nan_to_num(
                [float(row["mean_hausdorff"]) - float(row["p25_hausdorff"]) for row in method_rows],
                nan=0.0,
            ),
            0,
            None,
        )
        yerr_hi = np.clip(
            np.nan_to_num(
                [float(row["p75_hausdorff"]) - float(row["mean_hausdorff"]) for row in method_rows],
                nan=0.0,
            ),
            0,
            None,
        )
        ax_h.errorbar(
            delta_ratios,
            means,
            yerr=[yerr_lo, yerr_hi],
            marker="o",
            label=method,
            capsize=3,
        )

    ax_h.set_title(f"{experiment_name}: mean Hausdorff distance")
    ax_h.set_xlabel(r"$\delta = n/p$")
    ax_h.set_ylabel("Hausdorff distance")
    ax_h.legend()
    ax_h.grid(alpha=0.25)

    for method in methods:
        method_rows = [
            row
            for row in filtered
            if row["method"] == method
            and not np.isnan(float(row["mean_predicted_num_chgpts"]))
        ]
        method_rows.sort(key=lambda row: float(row["delta_ratio"]))
        if not method_rows:
            continue
        delta_ratios = [float(row["delta_ratio"]) for row in method_rows]
        means = [float(row["mean_predicted_num_chgpts"]) for row in method_rows]
        yerr_lo = np.clip(
            np.nan_to_num(
                [
                    float(row["mean_predicted_num_chgpts"]) - float(row["p25_predicted_num_chgpts"])
                    for row in method_rows
                ],
                nan=0.0,
            ),
            0,
            None,
        )
        yerr_hi = np.clip(
            np.nan_to_num(
                [
                    float(row["p75_predicted_num_chgpts"]) - float(row["mean_predicted_num_chgpts"])
                    for row in method_rows
                ],
                nan=0.0,
            ),
            0,
            None,
        )
        ax_n.errorbar(
            delta_ratios,
            means,
            yerr=[yerr_lo, yerr_hi],
            marker="o",
            label=method,
            capsize=3,
        )

    ax_n.set_title(f"{experiment_name}: estimated number of changepoints")
    ax_n.set_xlabel(r"$\delta = n/p$")
    ax_n.set_ylabel("number of changepoints")
    ax_n.legend()
    ax_n.grid(alpha=0.25)

    return fig, (ax_h, ax_n)


def _serialize_changepoints(changepoints: np.ndarray | None) -> str:
    if changepoints is None:
        return ""
    return ",".join(str(int(value)) for value in np.asarray(changepoints, dtype=int).tolist())


def _segment_ids(n: int, changepoints: np.ndarray) -> np.ndarray:
    return np.searchsorted(changepoints, np.arange(n), side="right")


def _ar1_covariance(p: int, r: float) -> np.ndarray:
    indices = np.arange(p)
    return r ** np.abs(indices[:, None] - indices[None, :])


def _sparse_gaussian_signal(
    rng: np.random.Generator,
    p: int,
    num_signals: int,
    alpha: float,
    delta_ratio: float,
    kappa: float,
) -> np.ndarray:
    sigma = kappa * np.sqrt(delta_ratio)
    support = rng.binomial(1, alpha, size=(p, num_signals))
    return support * rng.normal(scale=sigma, size=(p, num_signals))


def _sparse_diff_signal(
    rng: np.random.Generator,
    p: int,
    num_signals: int,
    alpha: float,
    var_beta_1: float,
    sigma_w: float,
) -> np.ndarray:
    eta = np.sqrt(var_beta_1 / (var_beta_1 + sigma_w**2))
    signal = np.zeros((p, num_signals))
    signal[:, 0] = rng.normal(scale=np.sqrt(var_beta_1), size=p)
    for signal_index in range(1, num_signals):
        signal[:, signal_index] = signal[:, signal_index - 1]
        changed = rng.binomial(1, alpha, size=p).astype(bool)
        signal[changed, signal_index] = eta * (
            signal[changed, signal_index] + rng.normal(scale=sigma_w, size=int(changed.sum()))
        )
    return signal


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))
