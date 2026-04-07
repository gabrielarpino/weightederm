import numpy as np

from weightederm import WERMHuberCV, WERMLeastSquaresCV, WERMLogisticCV
from weightederm._huber import _huber_loss


def test_least_squares_cv_defaults_to_absolute_value_scoring():
    X = np.zeros((6, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0])

    estimator = WERMLeastSquaresCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
    )
    estimator.fit(X, y)

    assert estimator.best_score_ == 3.0
    np.testing.assert_allclose(estimator.cv_results_["mean_test_score"], np.array([3.0]))
    np.testing.assert_allclose(estimator.segment_intercepts_, np.array([0.0]))


def test_least_squares_cv_uses_squared_loss_when_flag_enabled():
    X = np.zeros((6, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0])

    estimator = WERMLeastSquaresCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        use_base_loss_for_cv=True,
    )
    estimator.fit(X, y)

    assert np.isclose(estimator.best_score_, 24.0)
    np.testing.assert_allclose(estimator.cv_results_["mean_test_score"], np.array([24.0]))
    np.testing.assert_allclose(estimator.segment_intercepts_, np.array([1.0]))


def test_least_squares_cv_uses_user_supplied_m_scorer():
    X = np.zeros((6, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0])

    estimator = WERMLeastSquaresCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        m_scorer=lambda predictions, targets: (predictions - targets) ** 2,
    )
    estimator.fit(X, y)

    assert np.isclose(estimator.best_score_, 24.0)
    np.testing.assert_allclose(estimator.cv_results_["mean_test_score"], np.array([24.0]))
    np.testing.assert_allclose(estimator.segment_intercepts_, np.array([1.0]))


def test_least_squares_cv_base_loss_flag_overrides_custom_cv_scorer():
    X = np.zeros((6, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0])

    estimator = WERMLeastSquaresCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        m_scorer=lambda predictions, targets: np.abs(predictions - targets),
        use_base_loss_for_cv=True,
    )
    estimator.fit(X, y)

    assert np.isclose(estimator.best_score_, 24.0)
    np.testing.assert_allclose(estimator.segment_intercepts_, np.array([1.0]))


def test_huber_cv_defaults_to_absolute_value_scoring():
    X = np.zeros((6, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0])

    estimator = WERMHuberCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    assert estimator.best_score_ == 3.0
    np.testing.assert_allclose(estimator.cv_results_["mean_test_score"], np.array([3.0]))
    np.testing.assert_allclose(estimator.segment_intercepts_, np.array([0.0]))


def test_huber_cv_uses_huber_loss_when_flag_enabled():
    X = np.zeros((6, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 6.0])

    estimator = WERMHuberCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
        use_base_loss_for_cv=True,
    )
    estimator.fit(X, y)

    explicit_base_loss_estimator = WERMHuberCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        epsilon=1.35,
        max_iter=200,
        tol=1e-6,
        m_scorer=lambda predictions, targets: _huber_loss(predictions, targets, epsilon=1.35),
    )
    explicit_base_loss_estimator.fit(X, y)

    assert np.isclose(estimator.best_score_, explicit_base_loss_estimator.best_score_)
    np.testing.assert_allclose(
        estimator.cv_results_["mean_test_score"],
        explicit_base_loss_estimator.cv_results_["mean_test_score"],
    )
    np.testing.assert_allclose(
        estimator.segment_intercepts_,
        explicit_base_loss_estimator.segment_intercepts_,
    )


def test_logistic_cv_defaults_to_logistic_scoring():
    X = np.zeros((4, 1))
    y = np.array([0, 0, 1, 1])

    estimator = WERMLogisticCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )
    estimator.fit(X, y)

    expected_fold_loss = 2.0 * np.log(2.0)
    assert np.isclose(estimator.best_score_, expected_fold_loss)
    np.testing.assert_allclose(
        estimator.cv_results_["mean_test_score"],
        np.array([expected_fold_loss]),
    )


def test_logistic_cv_base_loss_flag_matches_default_logistic_scoring():
    X = np.zeros((4, 1))
    y = np.array([0, 0, 1, 1])

    default_estimator = WERMLogisticCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
    )
    default_estimator.fit(X, y)

    base_loss_estimator = WERMLogisticCV(
        max_num_chgpts=0,
        cv=2,
        search_method="brute_force",
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
        m_scorer=lambda predictions, targets: np.abs(predictions - targets),
        use_base_loss_for_cv=True,
    )
    base_loss_estimator.fit(X, y)

    assert np.isclose(base_loss_estimator.best_score_, default_estimator.best_score_)
    np.testing.assert_allclose(
        base_loss_estimator.cv_results_["mean_test_score"],
        default_estimator.cv_results_["mean_test_score"],
    )
