import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from weightederm import (
    WERMHuber,
    WERMHuberCV,
    WERMLeastSquares,
    WERMLeastSquaresCV,
    WERMLogistic,
    WERMLogisticCV,
)


@pytest.mark.parametrize(
    "estimator",
    [
        WERMLeastSquares(num_chgpts=1, search_method="brute_force", fit_intercept=True),
        WERMHuber(
            num_chgpts=1,
            search_method="brute_force",
            fit_intercept=True,
            epsilon=1.35,
            max_iter=200,
            tol=1e-6,
        ),
    ],
)
def test_fixed_regression_estimators_predict_from_eager_last_segment_refit(estimator):
    X = np.array([[0.0], [1.0], [0.0], [1.0]])
    y = np.array([0.0, 1.0, 10.0, 12.0])

    estimator.fit(X, y)

    assert estimator.changepoints_.tolist() == [2]
    np.testing.assert_allclose(estimator.last_segment_coef_, np.array([2.0]), atol=1e-5)
    assert estimator.last_segment_intercept_ == pytest.approx(10.0, abs=1e-5)
    np.testing.assert_allclose(estimator.predict(np.array([[2.0], [3.0]])), np.array([14.0, 16.0]))
    assert estimator.score(np.array([[0.0], [1.0]]), np.array([10.0, 12.0])) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "estimator",
    [
        WERMLeastSquaresCV(
            max_num_chgpts=1,
            cv=2,
            search_method="brute_force",
            fit_intercept=True,
        ),
        WERMHuberCV(
            max_num_chgpts=1,
            cv=2,
            search_method="brute_force",
            fit_intercept=True,
            epsilon=1.35,
            max_iter=200,
            tol=1e-6,
        ),
    ],
)
def test_cv_regression_estimators_predict_from_eager_last_segment_refit(estimator):
    X = np.arange(1.0, 9.0).reshape(-1, 1)
    y = 3.0 * X[:, 0] - 2.0

    estimator.fit(X, y)

    assert estimator.best_num_chgpts_ == 0
    np.testing.assert_allclose(estimator.last_segment_coef_, np.array([3.0]), atol=1e-4)
    assert estimator.last_segment_intercept_ == pytest.approx(-2.0, abs=1e-4)
    np.testing.assert_allclose(estimator.predict(np.array([[9.0], [10.0]])), np.array([25.0, 28.0]))
    assert estimator.score(X, y) == pytest.approx(1.0)


def test_fixed_logistic_exposes_predict_predict_proba_and_accuracy_score():
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

    assert estimator.changepoints_.tolist() == [2]
    assert estimator.last_segment_coef_.shape == (1,)
    assert estimator.last_segment_intercept_ > 0.0
    probabilities = estimator.predict_proba(np.zeros((3, 1)))
    assert probabilities.shape == (3, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(3))
    predictions = estimator.predict(np.zeros((3, 1)))
    np.testing.assert_array_equal(predictions, np.array(["yes", "yes", "yes"], dtype=object))
    assert estimator.score(
        np.zeros((2, 1)), np.array(["yes", "yes"], dtype=object)
    ) == pytest.approx(1.0)


def test_cv_logistic_exposes_predict_predict_proba_and_accuracy_score():
    X = np.zeros((12, 1))
    y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])

    estimator = WERMLogisticCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    assert estimator.best_num_chgpts_ == 1
    probabilities = estimator.predict_proba(np.zeros((2, 1)))
    assert probabilities.shape == (2, 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
    np.testing.assert_array_equal(estimator.predict(np.zeros((3, 1))), np.array([0, 0, 0]))
    assert estimator.score(np.zeros((7, 1)), np.array([0, 0, 0, 1, 1, 0, 0])) == pytest.approx(
        5.0 / 7.0
    )


def test_predict_before_fit_raises_not_fitted_error():
    least_squares = WERMLeastSquares(num_chgpts=0, search_method="brute_force")
    logistic = WERMLogistic(num_chgpts=0, search_method="brute_force")

    with pytest.raises(NotFittedError):
        least_squares.predict(np.zeros((1, 1)))

    with pytest.raises(NotFittedError):
        logistic.predict_proba(np.zeros((1, 1)))
