import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from weightederm import WERMLeastSquares
from weightederm import _least_squares as least_squares_module
from weightederm._benchmark_examples import reference_like_benchmark_specs, simulate_trial
from weightederm._least_squares import (
    _fit_weighted_least_squares_signal,
    _fit_weighted_least_squares_signals,
    _weighted_squared_loss_objective,
)
from weightederm._weights import compute_exact_marginal_weights


def test_exact_marginal_weights_match_combinatorial_formula():
    weights = compute_exact_marginal_weights(num_signals=3, n_samples=5)

    expected = np.array(
        [
            [1.0, 0.5, 1.0 / 6.0, 0.0, 0.0],
            [0.0, 0.5, 2.0 / 3.0, 0.5, 0.0],
            [0.0, 0.0, 1.0 / 6.0, 0.5, 1.0],
        ]
    )

    np.testing.assert_allclose(weights, expected)


def test_weighted_signal_fits_use_global_weighted_least_squares():
    X = np.ones((4, 1))
    y = np.array([0.0, 0.0, 10.0, 10.0])
    weights = compute_exact_marginal_weights(num_signals=2, n_samples=4)

    coefs, intercepts, theta_hat = _fit_weighted_least_squares_signals(
        X,
        y,
        weights,
        fit_intercept=False,
    )

    np.testing.assert_allclose(coefs, np.array([[5.0 / 3.0], [25.0 / 3.0]]))
    assert intercepts is None
    np.testing.assert_allclose(
        theta_hat,
        np.array(
            [
                [5.0 / 3.0, 25.0 / 3.0],
                [5.0 / 3.0, 25.0 / 3.0],
                [5.0 / 3.0, 25.0 / 3.0],
                [5.0 / 3.0, 25.0 / 3.0],
            ]
        ),
    )


def test_weighted_signal_fits_handle_intercepts():
    X = np.zeros((4, 1))
    y = np.array([1.0, 1.0, 9.0, 9.0])
    weights = compute_exact_marginal_weights(num_signals=2, n_samples=4)

    coefs, intercepts, theta_hat = _fit_weighted_least_squares_signals(
        X,
        y,
        weights,
        fit_intercept=True,
    )

    np.testing.assert_allclose(coefs, np.array([[0.0], [0.0]]))
    np.testing.assert_allclose(intercepts, np.array([7.0 / 3.0, 23.0 / 3.0]))
    np.testing.assert_allclose(
        theta_hat,
        np.array(
            [
                [7.0 / 3.0, 23.0 / 3.0],
                [7.0 / 3.0, 23.0 / 3.0],
                [7.0 / 3.0, 23.0 / 3.0],
                [7.0 / 3.0, 23.0 / 3.0],
            ]
        ),
    )


def test_weighted_least_squares_signal_fit_reduces_same_explicit_objective():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 20.0])
    weights = np.array([1.0, 1.0, 1.0, 0.25])

    coef, intercept = _fit_weighted_least_squares_signal(
        X,
        y,
        weights,
        fit_intercept=True,
    )

    fitted_objective, _ = _weighted_squared_loss_objective(
        np.array([coef[0], intercept]),
        X,
        y,
        weights,
        fit_intercept=True,
    )
    baseline_objective, _ = _weighted_squared_loss_objective(
        np.array([0.0, 0.0]),
        X,
        y,
        weights,
        fit_intercept=True,
    )

    assert fitted_objective < baseline_objective


def test_weighted_least_squares_signal_fit_does_not_depend_on_sklearn_backend(monkeypatch):
    class ExplodingLinearRegression:
        def __init__(self, *args, **kwargs):
            raise AssertionError("sklearn LinearRegression should not be used")

    monkeypatch.setattr(
        least_squares_module, "LinearRegression", ExplodingLinearRegression, raising=False
    )

    coef, intercept = _fit_weighted_least_squares_signal(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([2.0, 4.0, 6.0]),
        np.ones(3),
        fit_intercept=True,
    )

    assert coef.shape == (1,)
    assert isinstance(intercept, float)


def test_weighted_least_squares_signal_fit_lbfgsb_reduces_same_explicit_objective():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 20.0])
    weights = np.array([1.0, 1.0, 1.0, 0.25])

    coef, intercept = least_squares_module._fit_weighted_least_squares_signal(
        X,
        y,
        weights,
        fit_intercept=True,
        fit_solver="lbfgsb",
    )

    fitted_objective, _ = _weighted_squared_loss_objective(
        np.array([coef[0], intercept]),
        X,
        y,
        weights,
        fit_intercept=True,
    )
    baseline_objective, _ = _weighted_squared_loss_objective(
        np.array([0.0, 0.0]),
        X,
        y,
        weights,
        fit_intercept=True,
    )

    assert fitted_objective < baseline_objective


def test_uniform_weights_match_unweighted_sklearn_linear_regression():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0])
    weights = np.ones(X.shape[0])

    coef, intercept = least_squares_module._fit_weighted_least_squares_signal(
        X,
        y,
        weights,
        fit_intercept=True,
    )

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y)

    np.testing.assert_allclose(coef, model.coef_)
    assert np.isclose(intercept, model.intercept_)


def test_fit_returns_changepoints_from_second_stage_brute_force_search():
    X = np.ones((4, 1))
    y = np.array([0.0, 0.0, 10.0, 10.0])

    estimator = WERMLeastSquares(num_chgpts=1, search_method="brute_force", fit_intercept=False)
    estimator.fit(X, y)

    assert estimator.num_chgpts_ == 1
    assert estimator.num_signals_ == 2
    assert estimator.changepoints_.tolist() == [2]
    assert np.isclose(estimator.objective_, 100.0 / 9.0)
    assert estimator.n_features_in_ == 1
    assert not hasattr(estimator, "segment_bounds_")
    assert not hasattr(estimator, "segment_coefs_")
    assert not hasattr(estimator, "segment_intercepts_")


def test_fit_returns_changepoints_from_second_stage_efficient_search():
    X = np.ones((4, 1))
    y = np.array([0.0, 0.0, 10.0, 10.0])

    estimator = WERMLeastSquares(num_chgpts=1, search_method="efficient", fit_intercept=False)
    estimator.fit(X, y)

    assert estimator.changepoints_.tolist() == [2]


def test_fit_preserves_public_behavior_with_intercepts_enabled():
    X = np.zeros((4, 1))
    y = np.array([1.0, 1.0, 9.0, 9.0])

    estimator = WERMLeastSquares(num_chgpts=1, search_method="brute_force", fit_intercept=True)
    estimator.fit(X, y)

    assert estimator.changepoints_.tolist() == [2]
    assert np.isclose(estimator.objective_, 64.0 / 9.0)


def test_fit_lbfgsb_preserves_public_changepoint_behavior():
    X = np.ones((4, 1))
    y = np.array([0.0, 0.0, 10.0, 10.0])

    estimator = WERMLeastSquares(
        num_chgpts=1,
        search_method="brute_force",
        fit_intercept=False,
        fit_solver="lbfgsb",
    )
    estimator.fit(X, y)

    assert estimator.changepoints_.tolist() == [2]
    assert np.isclose(estimator.objective_, 100.0 / 9.0)


def test_weighted_least_squares_lbfgsb_accepts_finite_iteration_limit_iterate(monkeypatch):
    class Result:
        success = False
        message = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
        x = np.array([2.0, 1.0])
        fun = 0.25

    def fake_minimize(*args, **kwargs):
        return Result()

    monkeypatch.setattr(least_squares_module, "minimize", fake_minimize)

    coef, intercept = least_squares_module._fit_weighted_least_squares_signal(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([3.0, 5.0, 7.0]),
        np.ones(3),
        fit_intercept=True,
        fit_solver="lbfgsb",
    )

    np.testing.assert_allclose(coef, np.array([2.0]))
    assert intercept == 1.0


def test_weighted_least_squares_lbfgsb_still_raises_for_nonfinite_iterate(monkeypatch):
    class Result:
        success = False
        message = "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"
        x = np.array([np.nan, 1.0])
        fun = np.nan

    def fake_minimize(*args, **kwargs):
        return Result()

    monkeypatch.setattr(least_squares_module, "minimize", fake_minimize)

    with pytest.raises(ValueError, match="failed to converge"):
        least_squares_module._fit_weighted_least_squares_signal(
            np.array([[1.0], [2.0], [3.0]]),
            np.array([3.0, 5.0, 7.0]),
            np.ones(3),
            fit_intercept=True,
            fit_solver="lbfgsb",
        )


def test_fit_lbfgsb_avoids_nan_on_reference_like_m1_delta_one_case():
    spec = reference_like_benchmark_specs()["M1"]
    trial = simulate_trial(spec, delta_ratio=1.0, seed=42)

    estimator = WERMLeastSquares(
        num_chgpts=len(spec.fractional_chgpt_locations),
        delta=max(1, int(trial.X_fit.shape[0] * spec.delta_fraction)),
        search_method="efficient",
        fit_intercept=False,
        fit_solver="lbfgsb",
    )
    estimator.fit(trial.X_fit, trial.y_fit)

    assert estimator.changepoints_.shape == (2,)
    assert np.isfinite(estimator.objective_)
