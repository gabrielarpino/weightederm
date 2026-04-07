import numpy as np

from weightederm import WERMHuber, WERMLeastSquares
from weightederm import _huber as huber_module
from weightederm._huber import (
    _fit_weighted_huber_signal,
    _fit_weighted_huber_signals,
    _weighted_huber_objective,
)
from weightederm._weights import compute_exact_marginal_weights


def test_weighted_huber_signal_fits_return_expected_shapes():
    X = np.ones((4, 1))
    y = np.array([0.0, 0.0, 10.0, 10.0])
    weights = compute_exact_marginal_weights(num_signals=2, n_samples=4)

    coefs, intercepts, theta_hat = _fit_weighted_huber_signals(
        X,
        y,
        weights,
        fit_intercept=False,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )

    assert coefs.shape == (2, 1)
    assert intercepts is None
    assert theta_hat.shape == (4, 2)


def test_weighted_huber_signal_fit_reduces_same_explicit_objective():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 50.0])
    weights = np.ones(X.shape[0])

    coef, intercept = _fit_weighted_huber_signal(
        X,
        y,
        weights,
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )

    fitted_objective, _ = _weighted_huber_objective(
        np.array([coef[0], intercept]),
        X,
        y,
        weights,
        epsilon=1.35,
        fit_intercept=True,
    )
    baseline_objective, _ = _weighted_huber_objective(
        np.array([0.0, 0.0]),
        X,
        y,
        weights,
        epsilon=1.35,
        fit_intercept=True,
    )

    assert fitted_objective < baseline_objective


def test_weighted_huber_signal_fit_does_not_depend_on_sklearn_backend(monkeypatch):
    class ExplodingHuberRegressor:
        def __init__(self, *args, **kwargs):
            raise AssertionError("sklearn HuberRegressor should not be used")

    monkeypatch.setattr(huber_module, "HuberRegressor", ExplodingHuberRegressor, raising=False)

    coef, intercept = _fit_weighted_huber_signal(
        np.array([[1.0], [2.0], [3.0]]),
        np.array([2.0, 4.0, 6.0]),
        np.ones(3),
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )

    assert coef.shape == (1,)
    assert isinstance(intercept, float)


def test_huber_fit_recovers_single_changepoint_with_brute_force():
    X = np.ones((4, 1))
    y = np.array([0.0, 0.0, 10.0, 10.0])

    estimator = WERMHuber(
        num_chgpts=1,
        search_method="brute_force",
        fit_intercept=False,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    assert estimator.changepoints_.tolist() == [2]
    assert estimator.num_chgpts_ == 1
    assert estimator.num_signals_ == 2


def test_huber_fit_recovers_single_changepoint_with_efficient_search():
    X = np.ones((4, 1))
    y = np.array([0.0, 0.0, 10.0, 10.0])

    estimator = WERMHuber(
        num_chgpts=1,
        search_method="efficient",
        fit_intercept=False,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    assert estimator.changepoints_.tolist() == [2]


def test_huber_matches_least_squares_on_clean_example():
    X = np.array([[1.0], [2.0], [3.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 3.0, 6.0, 9.0])

    huber = WERMHuber(
        num_chgpts=1,
        search_method="brute_force",
        fit_intercept=False,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )
    least_squares = WERMLeastSquares(
        num_chgpts=1,
        search_method="brute_force",
        fit_intercept=False,
    )

    huber.fit(X, y)
    least_squares.fit(X, y)

    assert huber.changepoints_.tolist() == least_squares.changepoints_.tolist()
