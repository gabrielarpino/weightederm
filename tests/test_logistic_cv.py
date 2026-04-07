import numpy as np

from weightederm import WERMLogisticCV


def test_werm_logistic_cv_is_importable():
    estimator = WERMLogisticCV(max_num_chgpts=1)

    assert estimator.max_num_chgpts == 1


def test_logistic_cv_search_grid_and_classes_are_exposed():
    X = np.zeros((12, 1))
    y = np.array(
        ["no", "no", "yes", "yes", "yes", "no", "no", "no", "yes", "yes", "no", "no"], dtype=object
    )

    estimator = WERMLogisticCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    np.testing.assert_array_equal(estimator.num_chgpts_grid_, np.array([0, 1]))
    np.testing.assert_array_equal(estimator.classes_, np.array(["no", "yes"], dtype=object))
    assert len(estimator.cv_results_["mean_test_score"]) == 2


def test_logistic_cv_selects_one_changepoint_and_exposes_segmented_refit():
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
    assert estimator.num_chgpts_ == 1
    assert estimator.changepoints_.tolist() == [5]
    assert estimator.segment_bounds_ == [(0, 5), (5, 12)]
    assert estimator.segment_coefs_.shape == (2, 1)
    np.testing.assert_allclose(estimator.segment_coefs_, np.zeros((2, 1)))
    np.testing.assert_allclose(
        estimator.segment_intercepts_,
        np.array([np.log(3.0 / 2.0), np.log(2.0 / 5.0)]),
        atol=1e-4,
    )


def test_logistic_cv_does_not_raise_on_separable_data_with_default_l2_penalty():
    # WERMLogistic defaults to penalty="l2" — separable fold-level fits no longer raise.
    X = np.array([[-2.0], [-1.0], [1.0], [2.0], [-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1, 1, 1, 0, 0])

    estimator = WERMLogisticCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )

    estimator.fit(X, y)  # must not raise
    assert hasattr(estimator, "changepoints_")
