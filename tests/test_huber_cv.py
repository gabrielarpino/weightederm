import numpy as np
import pytest

from weightederm import WERMHuberCV


def test_werm_huber_cv_is_importable():
    estimator = WERMHuberCV(max_num_chgpts=1)

    assert estimator.max_num_chgpts == 1


def test_huber_cv_search_grid_is_zero_through_max_num_chgpts():
    X = np.arange(1.0, 7.0).reshape(-1, 1)
    y = 2.0 * X[:, 0] + 1.0

    estimator = WERMHuberCV(
        max_num_chgpts=2,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    np.testing.assert_array_equal(estimator.num_chgpts_grid_, np.array([0, 1, 2]))
    assert len(estimator.cv_results_["mean_test_score"]) == 3


def test_huber_cv_selects_zero_changepoints_on_clean_linear_data():
    X = np.arange(1.0, 9.0).reshape(-1, 1)
    y = 3.0 * X[:, 0] - 2.0

    estimator = WERMHuberCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    assert estimator.best_num_chgpts_ == 0
    assert estimator.changepoints_.tolist() == []


def test_huber_cv_selects_one_changepoint_and_exposes_segmented_refit():
    X = np.zeros((8, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0])

    estimator = WERMHuberCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    assert estimator.best_num_chgpts_ == 1
    assert estimator.num_chgpts_ == 1
    assert estimator.changepoints_.tolist() == [4]
    assert estimator.segment_bounds_ == [(0, 4), (4, 8)]
    assert estimator.segment_coefs_.shape == (2, 1)
    np.testing.assert_allclose(estimator.segment_coefs_, np.zeros((2, 1)))
    np.testing.assert_allclose(estimator.segment_intercepts_, np.array([0.0, 10.0]), atol=1e-4)


def test_huber_cv_rejects_invalid_epsilon():
    estimator = WERMHuberCV(max_num_chgpts=0, epsilon=0.9, cv=2, search_method="brute_force")

    with pytest.raises(ValueError, match="epsilon"):
        estimator.fit(np.arange(4.0).reshape(-1, 1), np.arange(4.0))
