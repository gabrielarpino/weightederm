import numpy as np
import pytest

from dataclasses import replace

from weightederm._benchmark_examples import (
    ExperimentSpec,
    fit_werm_changepoints,
    fit_werm_unknown_changepoints,
    hausdorff_distance,
    maybe_run_mcscan_changepoints,
    normalized_hausdorff_distance,
    plot_hausdorff_summary_with_percentiles,
    reference_like_benchmark_specs,
    run_benchmark,
    run_benchmark_unknown,
    simulate_trial,
    summarize_trial_rows,
)


def test_reference_like_benchmark_specs_match_main_notebook_settings() -> None:
    specs = reference_like_benchmark_specs()

    assert set(specs) == {"M1", "M2", "M3"}

    m1 = specs["M1"]
    assert m1.p == 200
    assert m1.num_signals == 3
    assert m1.delta_ratios == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    assert m1.fractional_chgpt_locations == (0.4, 0.7)

    m2 = specs["M2"]
    assert m2.p == 200
    assert m2.num_signals == 4
    assert m2.delta_ratios == (0.5, 1.5, 2.5, 3.5, 4.5)
    assert m2.fractional_chgpt_locations == (0.4, 0.7)

    m3 = specs["M3"]
    assert m3.p == 200
    assert m3.num_signals == 2
    assert m3.delta_ratios == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0)
    assert m3.fractional_chgpt_locations == (1.0 / 3.0,)


def test_hausdorff_distance_handles_empty_sets() -> None:
    assert hausdorff_distance([], []) == 0.0
    assert np.isinf(hausdorff_distance([], [3]))
    assert hausdorff_distance([10, 20], [12, 18]) == 2.0


def test_normalized_hausdorff_distance_matches_reference_scaling() -> None:
    assert normalized_hausdorff_distance([10, 20], [12, 18], n_samples=100) == 0.02
    assert np.isinf(normalized_hausdorff_distance([], [3], n_samples=100))


def test_simulate_trial_returns_reference_like_shapes_for_m1() -> None:
    spec = reference_like_benchmark_specs()["M1"]

    trial = simulate_trial(spec, delta_ratio=2.0, seed=0)

    assert trial.X_fit.shape == (400, 200)
    assert trial.y_fit.shape == (400,)
    assert trial.X_mcscan.shape == (400, 200)
    assert trial.y_mcscan.shape == (400,)
    assert np.array_equal(trial.true_changepoints, np.array([160, 280]))


def test_simulate_trial_returns_reference_like_shapes_for_m3() -> None:
    spec = reference_like_benchmark_specs()["M3"]

    trial = simulate_trial(spec, delta_ratio=2.0, seed=0)

    assert trial.X_fit.shape == (400, 200)
    assert trial.y_fit.shape == (400,)
    assert set(np.unique(trial.y_fit)).issubset({0, 1})
    assert np.array_equal(trial.true_changepoints, np.array([133]))


def test_summarize_trial_rows_groups_by_experiment_method_and_delta() -> None:
    rows = [
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "hausdorff": 2.0,
            "predicted_num_chgpts": 2,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "hausdorff": 4.0,
            "predicted_num_chgpts": 1,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 2.0,
            "hausdorff": 1.0,
            "predicted_num_chgpts": 2,
        },
        {
            "experiment": "M1",
            "method": "McScan",
            "delta_ratio": 1.0,
            "hausdorff": 3.0,
            "predicted_num_chgpts": 2,
        },
    ]

    summary = summarize_trial_rows(rows)

    assert summary == [
        {
            "experiment": "M1",
            "method": "McScan",
            "delta_ratio": 1.0,
            "mean_hausdorff": 3.0,
            "median_hausdorff": 3.0,
            "p25_hausdorff": 3.0,
            "p75_hausdorff": 3.0,
            "num_trials": 1,
            "mean_predicted_num_chgpts": 2.0,
            "median_predicted_num_chgpts": 2.0,
            "p25_predicted_num_chgpts": 2.0,
            "p75_predicted_num_chgpts": 2.0,
            "num_infinite_hausdorff": 0,
            "num_nan_hausdorff": 0,
            "num_finite_hausdorff": 1,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "mean_hausdorff": 3.0,
            "median_hausdorff": 3.0,
            "p25_hausdorff": 2.5,
            "p75_hausdorff": 3.5,
            "num_trials": 2,
            "mean_predicted_num_chgpts": 1.5,
            "median_predicted_num_chgpts": 1.5,
            "p25_predicted_num_chgpts": 1.25,
            "p75_predicted_num_chgpts": 1.75,
            "num_infinite_hausdorff": 0,
            "num_nan_hausdorff": 0,
            "num_finite_hausdorff": 2,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 2.0,
            "mean_hausdorff": 1.0,
            "median_hausdorff": 1.0,
            "p25_hausdorff": 1.0,
            "p75_hausdorff": 1.0,
            "num_trials": 1,
            "mean_predicted_num_chgpts": 2.0,
            "median_predicted_num_chgpts": 2.0,
            "p25_predicted_num_chgpts": 2.0,
            "p75_predicted_num_chgpts": 2.0,
            "num_infinite_hausdorff": 0,
            "num_nan_hausdorff": 0,
            "num_finite_hausdorff": 1,
        },
    ]


def test_summarize_trial_rows_uses_nanmean_style_aggregation() -> None:
    rows = [
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "hausdorff": np.nan,
            "predicted_num_chgpts": 0,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "hausdorff": 4.0,
            "predicted_num_chgpts": 2,
        },
        {
            "experiment": "M1",
            "method": "McScan",
            "delta_ratio": 1.0,
            "hausdorff": np.inf,
            "predicted_num_chgpts": 0,
        },
    ]

    summary = summarize_trial_rows(rows)

    assert summary == [
        {
            "experiment": "M1",
            "method": "McScan",
            "delta_ratio": 1.0,
            "mean_hausdorff": np.nan,
            "median_hausdorff": np.nan,
            "p25_hausdorff": np.nan,
            "p75_hausdorff": np.nan,
            "num_trials": 1,
            "mean_predicted_num_chgpts": 0.0,
            "median_predicted_num_chgpts": 0.0,
            "p25_predicted_num_chgpts": 0.0,
            "p75_predicted_num_chgpts": 0.0,
            "num_infinite_hausdorff": 1,
            "num_nan_hausdorff": 0,
            "num_finite_hausdorff": 0,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "mean_hausdorff": 4.0,
            "median_hausdorff": 4.0,
            "p25_hausdorff": 4.0,
            "p75_hausdorff": 4.0,
            "num_trials": 2,
            "mean_predicted_num_chgpts": 1.0,
            "median_predicted_num_chgpts": 1.0,
            "p25_predicted_num_chgpts": 0.5,
            "p75_predicted_num_chgpts": 1.5,
            "num_infinite_hausdorff": 0,
            "num_nan_hausdorff": 1,
            "num_finite_hausdorff": 1,
        },
    ]


def test_summarize_trial_rows_ignores_infinite_hausdorff_values_like_reference_plots() -> None:
    rows = [
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "hausdorff": np.inf,
            "predicted_num_chgpts": 0,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "hausdorff": 0.2,
            "predicted_num_chgpts": 1,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "hausdorff": 0.4,
            "predicted_num_chgpts": 2,
        },
    ]

    summary = summarize_trial_rows(rows)

    assert summary == [
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 1.0,
            "mean_hausdorff": 0.30000000000000004,
            "median_hausdorff": 0.30000000000000004,
            "p25_hausdorff": 0.25,
            "p75_hausdorff": 0.35000000000000003,
            "num_trials": 3,
            "mean_predicted_num_chgpts": 1.0,
            "median_predicted_num_chgpts": 1.0,
            "p25_predicted_num_chgpts": 0.5,
            "p75_predicted_num_chgpts": 1.5,
            "num_infinite_hausdorff": 1,
            "num_nan_hausdorff": 0,
            "num_finite_hausdorff": 2,
        },
    ]


def test_summarize_trial_rows_warns_when_all_trials_predict_zero_changepoints() -> None:
    rows = [
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 8.0,
            "hausdorff": np.inf,
            "predicted_num_chgpts": 0,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 8.0,
            "hausdorff": np.inf,
            "predicted_num_chgpts": 0,
        },
    ]

    with pytest.warns(
        RuntimeWarning,
        match="predicted zero changepoints across all trials",
    ):
        summary = summarize_trial_rows(rows)

    assert np.isnan(summary[0]["mean_hausdorff"])
    assert np.isnan(summary[0]["p25_hausdorff"])
    assert np.isnan(summary[0]["p75_hausdorff"])
    assert summary[0]["mean_predicted_num_chgpts"] == 0.0
    assert summary[0]["p25_predicted_num_chgpts"] == 0.0
    assert summary[0]["p75_predicted_num_chgpts"] == 0.0


def test_fit_werm_changepoints_uses_fixed_least_squares_estimator(monkeypatch) -> None:
    spec = reference_like_benchmark_specs()["M1"]
    trial = simulate_trial(spec, delta_ratio=1.0, seed=0)
    observed = {}

    class FakeLeastSquares:
        def __init__(self, *, num_chgpts, delta, search_method, fit_intercept, fit_solver):
            observed["num_chgpts"] = num_chgpts
            observed["delta"] = delta
            observed["search_method"] = search_method
            observed["fit_intercept"] = fit_intercept
            observed["fit_solver"] = fit_solver

        def fit(self, X, y):
            observed["fit_shape"] = (X.shape, y.shape)
            self.changepoints_ = np.array([1, 2])
            return self

    monkeypatch.setattr("weightederm._benchmark_examples.WERMLeastSquares", FakeLeastSquares)

    result = fit_werm_changepoints(spec, trial)

    assert np.array_equal(result, np.array([1, 2]))
    assert observed == {
        "num_chgpts": 2,
        "delta": 10,
        "search_method": "efficient",
        "fit_intercept": False,
        "fit_solver": "direct",
        "fit_shape": ((200, 200), (200,)),
    }


def test_fit_werm_changepoints_allows_selecting_least_squares_fit_solver(monkeypatch) -> None:
    spec = reference_like_benchmark_specs()["M1"]
    trial = simulate_trial(spec, delta_ratio=1.0, seed=0)
    observed = {}

    class FakeLeastSquares:
        def __init__(self, *, num_chgpts, delta, search_method, fit_intercept, fit_solver):
            observed["num_chgpts"] = num_chgpts
            observed["delta"] = delta
            observed["search_method"] = search_method
            observed["fit_intercept"] = fit_intercept
            observed["fit_solver"] = fit_solver

        def fit(self, X, y):
            self.changepoints_ = np.array([1, 2])
            return self

    monkeypatch.setattr("weightederm._benchmark_examples.WERMLeastSquares", FakeLeastSquares)

    result = fit_werm_changepoints(spec, trial, least_squares_fit_solver="lbfgsb")

    assert np.array_equal(result, np.array([1, 2]))
    assert observed["fit_solver"] == "lbfgsb"


def test_fit_werm_unknown_changepoints_uses_least_squares_cv_estimator(monkeypatch) -> None:
    spec = reference_like_benchmark_specs()["M2"]
    trial = simulate_trial(spec, delta_ratio=1.5, seed=0)
    observed = {}

    class FakeLeastSquaresCV:
        def __init__(self, *, max_num_chgpts, delta, search_method, cv, fit_intercept):
            observed["max_num_chgpts"] = max_num_chgpts
            observed["delta"] = delta
            observed["search_method"] = search_method
            observed["cv"] = cv
            observed["fit_intercept"] = fit_intercept

        def fit(self, X, y):
            observed["fit_shape"] = (X.shape, y.shape)
            self.changepoints_ = np.array([10, 20])
            self.best_num_chgpts_ = 2
            return self

    monkeypatch.setattr("weightederm._benchmark_examples.WERMLeastSquaresCV", FakeLeastSquaresCV)

    result = fit_werm_unknown_changepoints(spec, trial)

    assert np.array_equal(result, np.array([10, 20]))
    assert observed == {
        "max_num_chgpts": 3,
        "delta": 30,
        "search_method": "efficient",
        "cv": 5,
        "fit_intercept": False,
        "fit_shape": ((300, 200), (300,)),
    }


def test_fit_werm_unknown_changepoints_uses_logistic_cv_estimator(monkeypatch) -> None:
    spec = reference_like_benchmark_specs()["M3"]
    trial = simulate_trial(spec, delta_ratio=3.0, seed=0)
    observed = {}

    class FakeLogisticCV:
        def __init__(
            self,
            *,
            max_num_chgpts,
            delta,
            search_method,
            cv,
            fit_intercept,
            max_iter,
            tol,
        ):
            observed["max_num_chgpts"] = max_num_chgpts
            observed["delta"] = delta
            observed["search_method"] = search_method
            observed["cv"] = cv
            observed["fit_intercept"] = fit_intercept
            observed["max_iter"] = max_iter
            observed["tol"] = tol

        def fit(self, X, y):
            observed["fit_shape"] = (X.shape, y.shape)
            self.changepoints_ = np.array([50])
            self.best_num_chgpts_ = 1
            return self

    monkeypatch.setattr("weightederm._benchmark_examples.WERMLogisticCV", FakeLogisticCV)

    result = fit_werm_unknown_changepoints(spec, trial)

    assert np.array_equal(result, np.array([50]))
    assert observed == {
        "max_num_chgpts": 1,
        "delta": 30,
        "search_method": "efficient",
        "cv": 5,
        "fit_intercept": False,
        "max_iter": 500,
        "tol": 1e-6,
        "fit_shape": ((600, 200), (600,)),
    }


def test_run_benchmark_unknown_records_zero_predicted_changepoints_and_raw_infinite_hausdorff(
    monkeypatch,
) -> None:
    spec = reference_like_benchmark_specs()["M1"]

    monkeypatch.setattr(
        "weightederm._benchmark_examples.fit_werm_unknown_changepoints",
        lambda spec, trial: np.array([], dtype=int),
    )

    rows, summary = run_benchmark_unknown(spec, num_trials=1, include_mcscan=False, base_seed=0)

    trial_row = rows[0]
    summary_row = summary[0]

    assert trial_row["predicted_num_chgpts"] == 0
    assert np.isinf(trial_row["hausdorff"])
    assert trial_row["estimated_changepoints"] == ""
    assert np.isnan(summary_row["mean_hausdorff"])
    assert np.isnan(summary_row["median_hausdorff"])


def test_maybe_run_mcscan_changepoints_uses_known_num_chgpts(monkeypatch) -> None:
    spec = reference_like_benchmark_specs()["M2"]
    trial = simulate_trial(spec, delta_ratio=0.5, seed=0)
    calls = {}

    def fake_backend(X, y, *, num_chgpts, method):
        calls["num_chgpts"] = num_chgpts
        calls["method"] = method
        return np.array([20, 40])

    monkeypatch.setattr("weightederm._benchmark_examples._run_mcscan_backend", fake_backend)

    result, error = maybe_run_mcscan_changepoints(spec, trial, mode="fixed")

    assert error is None
    assert np.array_equal(result, np.array([19, 39]))
    assert calls == {"num_chgpts": 2, "method": "not"}


def test_maybe_run_mcscan_changepoints_supports_auto_mode(monkeypatch) -> None:
    spec = reference_like_benchmark_specs()["M1"]
    trial = simulate_trial(spec, delta_ratio=1.0, seed=0)
    calls = {}

    def fake_backend(X, y, *, num_chgpts, method):
        calls["num_chgpts"] = num_chgpts
        calls["method"] = method
        return np.array([10, 20, -1])

    monkeypatch.setattr("weightederm._benchmark_examples._run_mcscan_backend", fake_backend)

    result, error = maybe_run_mcscan_changepoints(spec, trial, mode="auto")

    assert error is None
    assert np.array_equal(result, np.array([9, 19]))
    assert calls == {"num_chgpts": None, "method": "auto"}


# ---------------------------------------------------------------------------
# n_jobs parallelism tests
# Use a tiny real spec (p=10, 1 delta_ratio, 2 trials) so fitting is fast and
# no monkeypatching is needed — avoids the multiprocessing/monkeypatch conflict.
# ---------------------------------------------------------------------------

_TINY_SPEC = ExperimentSpec(
    name="M1",
    p=10,
    num_signals=2,
    delta_ratios=(2.0,),
    fractional_chgpt_locations=(0.5,),
    delta_fraction=0.1,
)

_TINY_SPEC_UNKNOWN = replace(
    reference_like_benchmark_specs()["M1"],
    delta_ratios=(2.0,),
)


def test_run_benchmark_accepts_n_jobs_parameter() -> None:
    rows, summary = run_benchmark(
        _TINY_SPEC, num_trials=2, include_mcscan=False, base_seed=0, n_jobs=1
    )
    assert len(rows) == 2
    assert len(summary) == 1


def test_run_benchmark_parallel_produces_same_rows_as_serial() -> None:
    rows_serial, summary_serial = run_benchmark(
        _TINY_SPEC, num_trials=2, include_mcscan=False, base_seed=0, n_jobs=1
    )
    rows_parallel, summary_parallel = run_benchmark(
        _TINY_SPEC, num_trials=2, include_mcscan=False, base_seed=0, n_jobs=2
    )
    assert rows_serial == rows_parallel
    assert summary_serial == summary_parallel


def test_run_benchmark_unknown_accepts_n_jobs_parameter() -> None:
    rows, summary = run_benchmark_unknown(
        _TINY_SPEC_UNKNOWN, num_trials=2, include_mcscan=False, base_seed=0, n_jobs=1
    )
    assert len(rows) == 2
    assert len(summary) == 1


def test_run_benchmark_unknown_parallel_produces_same_rows_as_serial() -> None:
    rows_serial, summary_serial = run_benchmark_unknown(
        _TINY_SPEC_UNKNOWN, num_trials=2, include_mcscan=False, base_seed=0, n_jobs=1
    )
    rows_parallel, summary_parallel = run_benchmark_unknown(
        _TINY_SPEC_UNKNOWN, num_trials=2, include_mcscan=False, base_seed=0, n_jobs=2
    )
    assert rows_serial == rows_parallel
    assert summary_serial == summary_parallel


# ---------------------------------------------------------------------------
# plot_hausdorff_summary_with_percentiles tests
# ---------------------------------------------------------------------------

def _make_summary_rows_with_percentiles() -> list[dict]:
    return [
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 2.0,
            "mean_hausdorff": 0.3,
            "median_hausdorff": 0.3,
            "p25_hausdorff": 0.2,
            "p75_hausdorff": 0.4,
            "num_trials": 5,
            "mean_predicted_num_chgpts": 1.8,
            "median_predicted_num_chgpts": 2.0,
            "p25_predicted_num_chgpts": 1.5,
            "p75_predicted_num_chgpts": 2.0,
            "num_infinite_hausdorff": 0,
            "num_nan_hausdorff": 0,
            "num_finite_hausdorff": 5,
        },
        {
            "experiment": "M1",
            "method": "WERM",
            "delta_ratio": 4.0,
            "mean_hausdorff": 0.1,
            "median_hausdorff": 0.1,
            "p25_hausdorff": 0.05,
            "p75_hausdorff": 0.15,
            "num_trials": 5,
            "mean_predicted_num_chgpts": 2.0,
            "median_predicted_num_chgpts": 2.0,
            "p25_predicted_num_chgpts": 2.0,
            "p75_predicted_num_chgpts": 2.0,
            "num_infinite_hausdorff": 0,
            "num_nan_hausdorff": 0,
            "num_finite_hausdorff": 5,
        },
        {
            "experiment": "M2",
            "method": "WERM",
            "delta_ratio": 2.0,
            "mean_hausdorff": 0.2,
            "median_hausdorff": 0.2,
            "p25_hausdorff": 0.1,
            "p75_hausdorff": 0.3,
            "num_trials": 5,
            "mean_predicted_num_chgpts": 3.0,
            "median_predicted_num_chgpts": 3.0,
            "p25_predicted_num_chgpts": 3.0,
            "p75_predicted_num_chgpts": 3.0,
            "num_infinite_hausdorff": 0,
            "num_nan_hausdorff": 0,
            "num_finite_hausdorff": 5,
        },
    ]


def test_plot_hausdorff_summary_with_percentiles_returns_fig_and_two_axes() -> None:
    import matplotlib
    matplotlib.use("Agg")

    summary_rows = _make_summary_rows_with_percentiles()
    fig, axes = plot_hausdorff_summary_with_percentiles(summary_rows, "M1")

    assert len(axes) == 2
    ax_hausdorff, ax_num_chgpts = axes
    assert ax_hausdorff.get_xlabel() != ""
    assert ax_num_chgpts.get_xlabel() != ""


def test_plot_hausdorff_summary_with_percentiles_raises_for_unknown_experiment() -> None:
    import matplotlib
    matplotlib.use("Agg")

    summary_rows = _make_summary_rows_with_percentiles()
    with pytest.raises(ValueError, match="No summary rows"):
        plot_hausdorff_summary_with_percentiles(summary_rows, "M9")
