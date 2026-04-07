import numpy as np

from weightederm import WERMHuberCV, WERMLeastSquaresCV, WERMLogisticCV


def test_all_cv_estimators_expose_consistent_result_attrs():
    least_squares = WERMLeastSquaresCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
    )
    least_squares.fit(np.arange(1.0, 9.0).reshape(-1, 1), 3.0 * np.arange(1.0, 9.0) - 2.0)

    huber = WERMHuberCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )
    huber.fit(np.arange(1.0, 9.0).reshape(-1, 1), 3.0 * np.arange(1.0, 9.0) - 2.0)

    logistic_X = np.zeros((12, 1))
    logistic_y = np.array([0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0])
    logistic = WERMLogisticCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )
    logistic.fit(logistic_X, logistic_y)

    for estimator in (least_squares, huber, logistic):
        assert isinstance(estimator.best_index_, int)
        assert (
            estimator.best_score_ == estimator.cv_results_["mean_test_score"][estimator.best_index_]
        )
        np.testing.assert_array_equal(
            estimator.num_chgpts_grid_, estimator.cv_results_["num_chgpts"]
        )
