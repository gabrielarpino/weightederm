import numpy as np
import pytest

from weightederm import (
    WERMHuber,
    WERMHuberCV,
    WERMLeastSquares,
    WERMLeastSquaresCV,
    WERMLogistic,
    WERMLogisticCV,
)


class ArrayWithColumns:
    def __init__(self, data, columns):
        self._data = np.asarray(data, dtype=float)
        self.columns = np.asarray(columns, dtype=object)

    def __array__(self, dtype=None):
        return np.asarray(self._data, dtype=dtype)


@pytest.mark.parametrize(
    ("estimator", "X", "y", "expects_classes"),
    [
        (
            WERMLeastSquares(num_chgpts=0, search_method="brute_force", fit_intercept=True),
            ArrayWithColumns(
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
                ["feature_a", "feature_b"],
            ),
            np.array([1.0, 3.0, 5.0, 7.0]),
            False,
        ),
        (
            WERMHuber(
                num_chgpts=0,
                search_method="brute_force",
                fit_intercept=True,
                epsilon=1.35,
                max_iter=200,
                tol=1e-6,
            ),
            ArrayWithColumns(
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
                ["feature_a", "feature_b"],
            ),
            np.array([1.0, 3.0, 5.0, 7.0]),
            False,
        ),
        (
            WERMLogistic(
                num_chgpts=0,
                search_method="brute_force",
                fit_intercept=True,
                max_iter=200,
                tol=1e-6,
            ),
            ArrayWithColumns(
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ["feature_a", "feature_b"],
            ),
            np.array(["no", "no", "yes", "yes"], dtype=object),
            True,
        ),
    ],
)
def test_fixed_estimators_expose_locked_public_fitted_attrs(estimator, X, y, expects_classes):
    estimator.fit(X, y)

    assert hasattr(estimator, "changepoints_")
    assert hasattr(estimator, "num_chgpts_")
    assert hasattr(estimator, "num_signals_")
    assert hasattr(estimator, "objective_")
    assert hasattr(estimator, "n_features_in_")
    assert hasattr(estimator, "feature_names_in_")
    assert hasattr(estimator, "last_segment_coef_")
    assert hasattr(estimator, "last_segment_intercept_")
    np.testing.assert_array_equal(estimator.feature_names_in_, np.array(["feature_a", "feature_b"]))

    assert not hasattr(estimator, "best_index_")
    assert not hasattr(estimator, "best_num_chgpts_")
    assert not hasattr(estimator, "best_score_")
    assert not hasattr(estimator, "cv_results_")
    assert not hasattr(estimator, "num_chgpts_grid_")
    assert not hasattr(estimator, "segment_bounds_")
    assert not hasattr(estimator, "segment_coefs_")
    assert not hasattr(estimator, "segment_intercepts_")

    if expects_classes:
        assert hasattr(estimator, "classes_")
    else:
        assert not hasattr(estimator, "classes_")


@pytest.mark.parametrize(
    ("estimator", "X", "y", "expects_classes"),
    [
        (
            WERMLeastSquaresCV(
                max_num_chgpts=0,
                cv=2,
                search_method="brute_force",
                fit_intercept=True,
            ),
            ArrayWithColumns(
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
                ["feature_a", "feature_b"],
            ),
            np.array([1.0, 3.0, 5.0, 7.0]),
            False,
        ),
        (
            WERMHuberCV(
                max_num_chgpts=0,
                cv=2,
                search_method="brute_force",
                fit_intercept=True,
                epsilon=1.35,
                max_iter=200,
                tol=1e-6,
            ),
            ArrayWithColumns(
                [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0], [3.0, 1.0]],
                ["feature_a", "feature_b"],
            ),
            np.array([1.0, 3.0, 5.0, 7.0]),
            False,
        ),
        (
            WERMLogisticCV(
                max_num_chgpts=0,
                cv=2,
                search_method="brute_force",
                fit_intercept=True,
                max_iter=200,
                tol=1e-6,
            ),
            ArrayWithColumns(
                [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
                ["feature_a", "feature_b"],
            ),
            np.array(["no", "no", "yes", "yes"], dtype=object),
            True,
        ),
    ],
)
def test_cv_estimators_expose_locked_public_fitted_attrs(estimator, X, y, expects_classes):
    estimator.fit(X, y)

    assert hasattr(estimator, "best_index_")
    assert hasattr(estimator, "best_num_chgpts_")
    assert hasattr(estimator, "best_score_")
    assert hasattr(estimator, "cv_results_")
    assert hasattr(estimator, "num_chgpts_grid_")
    assert hasattr(estimator, "changepoints_")
    assert hasattr(estimator, "num_chgpts_")
    assert hasattr(estimator, "num_signals_")
    assert hasattr(estimator, "objective_")
    assert hasattr(estimator, "n_features_in_")
    assert hasattr(estimator, "feature_names_in_")
    assert hasattr(estimator, "segment_bounds_")
    assert hasattr(estimator, "segment_coefs_")
    assert hasattr(estimator, "segment_intercepts_")
    assert hasattr(estimator, "last_segment_coef_")
    assert hasattr(estimator, "last_segment_intercept_")
    np.testing.assert_array_equal(estimator.feature_names_in_, np.array(["feature_a", "feature_b"]))

    if expects_classes:
        assert hasattr(estimator, "classes_")
    else:
        assert not hasattr(estimator, "classes_")


def test_feature_names_in_is_not_set_for_plain_numpy_input():
    X = np.arange(8.0).reshape(-1, 2)
    y = np.array([1.0, 3.0, 5.0, 7.0])

    fixed = WERMLeastSquares(num_chgpts=0, search_method="brute_force", fit_intercept=True)
    fixed.fit(X, y)
    assert not hasattr(fixed, "feature_names_in_")

    cv = WERMLeastSquaresCV(max_num_chgpts=0, cv=2, search_method="brute_force", fit_intercept=True)
    cv.fit(X, y)
    assert not hasattr(cv, "feature_names_in_")
