import numpy as np
import pytest

from weightederm import WERMHuber, WERMLeastSquares, WERMLogistic
from weightederm._validation import (
    prepare_cv_fit_inputs,
    prepare_fixed_fit_inputs,
)
from weightederm._weights import compute_exact_marginal_weights


def test_fit_raises_for_negative_num_chgpts():
    estimator = WERMLeastSquares(num_chgpts=-1, search_method="brute_force")

    with pytest.raises(ValueError, match="num_chgpts"):
        estimator.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))


def test_fit_raises_for_delta_below_one():
    estimator = WERMLeastSquares(num_chgpts=0, delta=0, search_method="brute_force")

    with pytest.raises(ValueError, match="delta"):
        estimator.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))


def test_fit_raises_for_mismatched_sample_lengths():
    estimator = WERMLeastSquares(num_chgpts=0, search_method="brute_force")

    with pytest.raises(ValueError, match="same number of samples"):
        estimator.fit(np.array([[1.0], [2.0], [3.0]]), np.array([1.0, 2.0]))


def test_fit_raises_for_empty_input_data():
    estimator = WERMLeastSquares(num_chgpts=0, search_method="brute_force")

    with pytest.raises(ValueError, match="at least one sample"):
        estimator.fit(np.empty((0, 1)), np.empty((0,)))


def test_fit_raises_for_non_numeric_input():
    estimator = WERMLeastSquares(num_chgpts=0, search_method="brute_force")

    with pytest.raises((TypeError, ValueError), match="convert|string|number"):
        estimator.fit([["a"], ["b"]], [1.0, 2.0])


def test_fit_raises_for_zero_feature_input():
    estimator = WERMLeastSquares(num_chgpts=0, search_method="brute_force")

    with pytest.raises(ValueError, match="0 feature\\(s\\).*minimum of 1 is required"):
        estimator.fit(np.empty((4, 0)), np.arange(4.0))


def test_fit_raises_for_invalid_least_squares_fit_solver():
    estimator = WERMLeastSquares(
        num_chgpts=0,
        search_method="brute_force",
        fit_solver="not-a-solver",
    )

    with pytest.raises(ValueError, match="fit_solver"):
        estimator.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))


def test_logistic_fit_raises_for_empty_input_data():
    estimator = WERMLogistic(num_chgpts=0, search_method="brute_force")

    with pytest.raises(ValueError, match="at least one sample"):
        estimator.fit(np.empty((0, 1)), np.empty((0,)))


def test_fit_raises_when_num_chgpts_is_not_less_than_number_of_samples():
    estimator = WERMLeastSquares(num_chgpts=4, search_method="brute_force")

    with pytest.raises(ValueError, match="num_chgpts must be less than the number of samples"):
        estimator.fit(np.arange(4.0).reshape(-1, 1), np.arange(4.0))


def test_fit_raises_when_delta_and_num_chgpts_make_segmentation_impossible():
    estimator = WERMLeastSquares(num_chgpts=3, delta=2, search_method="brute_force")

    with pytest.raises(ValueError, match="No valid changepoint configuration"):
        estimator.fit(np.arange(4.0).reshape(-1, 1), np.arange(4.0))


def test_exact_marginal_weights_raise_for_more_signals_than_samples():
    with pytest.raises(ValueError, match="num_signals"):
        compute_exact_marginal_weights(num_signals=5, n_samples=4)


def test_huber_fit_raises_for_invalid_epsilon():
    estimator = WERMHuber(num_chgpts=0, epsilon=0.9, search_method="brute_force")

    with pytest.raises(ValueError, match="epsilon"):
        estimator.fit(np.array([[1.0], [2.0]]), np.array([1.0, 2.0]))


def test_prepare_fixed_fit_inputs_extracts_feature_names_and_validates_regression_inputs():
    class ArrayWithColumns:
        def __init__(self):
            self.columns = np.array(["feature_a", "feature_b"], dtype=object)

        def __array__(self, dtype=None):
            data = np.array([[1.0, 2.0], [3.0, 4.0]])
            return np.asarray(data, dtype=dtype)

    feature_names, num_chgpts, delta, X_array, y_array, classes = prepare_fixed_fit_inputs(
        ArrayWithColumns(),
        np.array([1.0, 2.0]),
        num_chgpts=0,
        delta=1,
    )

    np.testing.assert_array_equal(feature_names, np.array(["feature_a", "feature_b"]))
    assert num_chgpts == 0
    assert delta == 1
    np.testing.assert_allclose(X_array, np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(y_array, np.array([1.0, 2.0]))
    assert classes is None


def test_prepare_fixed_fit_inputs_validates_binary_targets_when_requested():
    feature_names, num_chgpts, delta, X_array, y_array, classes = prepare_fixed_fit_inputs(
        np.zeros((4, 1)),
        np.array(["no", "no", "yes", "yes"], dtype=object),
        num_chgpts=1,
        delta=2,
        binary=True,
    )

    assert feature_names is None
    assert num_chgpts == 1
    assert delta == 2
    np.testing.assert_allclose(X_array, np.zeros((4, 1)))
    np.testing.assert_array_equal(y_array, np.array(["no", "no", "yes", "yes"], dtype=object))
    np.testing.assert_array_equal(classes, np.array(["no", "yes"], dtype=object))


def test_prepare_cv_fit_inputs_extracts_feature_names_and_validates_cv_inputs():
    class ArrayWithColumns:
        def __init__(self):
            self.columns = np.array(["feature_a"], dtype=object)

        def __array__(self, dtype=None):
            data = np.array([[1.0], [2.0], [3.0], [4.0]])
            return np.asarray(data, dtype=dtype)

    (
        feature_names,
        max_num_chgpts,
        delta,
        cv,
        X_array,
        y_array,
        classes,
    ) = prepare_cv_fit_inputs(
        ArrayWithColumns(),
        np.array([1.0, 2.0, 3.0, 4.0]),
        max_num_chgpts=1,
        delta=2,
        cv=2,
    )

    np.testing.assert_array_equal(feature_names, np.array(["feature_a"], dtype=object))
    assert max_num_chgpts == 1
    assert delta == 2
    assert cv == 2
    np.testing.assert_allclose(X_array, np.array([[1.0], [2.0], [3.0], [4.0]]))
    np.testing.assert_allclose(y_array, np.array([1.0, 2.0, 3.0, 4.0]))
    assert classes is None
