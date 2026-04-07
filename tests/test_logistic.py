import numpy as np
import pytest

from weightederm import WERMLogistic
from weightederm._logistic import (
    _fit_weighted_logistic_signal,
    _weighted_logistic_objective,
)


def test_werm_logistic_is_importable():
    estimator = WERMLogistic(num_chgpts=0)

    assert estimator.num_chgpts == 0


def test_weighted_logistic_signal_fit_reduces_same_explicit_objective():
    X = np.zeros((5, 1))
    y = np.array([0.0, 0.0, 1.0, 1.0, 1.0])
    weights = np.ones(X.shape[0])

    coef, intercept = _fit_weighted_logistic_signal(
        X,
        y,
        weights,
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )

    fitted_objective, _ = _weighted_logistic_objective(
        np.array([coef[0], intercept]),
        X,
        y,
        weights,
        fit_intercept=True,
    )
    baseline_objective, _ = _weighted_logistic_objective(
        np.array([0.0, 0.0]),
        X,
        y,
        weights,
        fit_intercept=True,
    )

    assert fitted_objective < baseline_objective


def test_logistic_accepts_general_binary_string_labels_and_exposes_classes():
    X = np.zeros((4, 1))
    y = np.array(["no", "no", "yes", "yes"], dtype=object)

    estimator = WERMLogistic(
        num_chgpts=1,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    np.testing.assert_array_equal(estimator.classes_, np.array(["no", "yes"], dtype=object))
    assert estimator.changepoints_.tolist() == [2]


def test_logistic_rejects_non_binary_labels():
    estimator = WERMLogistic(num_chgpts=0, search_method="brute_force")

    with pytest.raises(
        ValueError, match="Only binary classification is supported|exactly two classes"
    ):
        estimator.fit(np.zeros((3, 1)), np.array(["a", "b", "c"], dtype=object))


def test_logistic_fit_recovers_single_changepoint_with_brute_force():
    X = np.zeros((4, 1))
    y = np.array([0, 0, 1, 1])

    estimator = WERMLogistic(
        num_chgpts=1,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    assert estimator.changepoints_.tolist() == [2]
    assert estimator.num_chgpts_ == 1
    assert estimator.num_signals_ == 2


def test_logistic_fit_recovers_single_changepoint_with_efficient_search():
    X = np.zeros((4, 1))
    y = np.array([-1, -1, 1, 1])

    estimator = WERMLogistic(
        num_chgpts=1,
        search_method="efficient",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    np.testing.assert_array_equal(estimator.classes_, np.array([-1, 1]))
    assert estimator.changepoints_.tolist() == [2]


def test_separable_weighted_logistic_signal_fit_raises_helpful_error():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(X.shape[0])

    with pytest.raises(ValueError, match="separable|finite optimum|penalization"):
        _fit_weighted_logistic_signal(
            X,
            y,
            weights,
            fit_intercept=True,
            max_iter=200,
            tol=1e-8,
        )


def test_unpenalized_logistic_surfaces_separability_failure_cleanly():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    estimator = WERMLogistic(
        num_chgpts=0, search_method="brute_force", max_iter=200, tol=1e-8, penalty="none"
    )

    with pytest.raises(ValueError, match="separable|finite optimum|penalization"):
        estimator.fit(X, y)
