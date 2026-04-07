import pytest
from sklearn.utils.estimator_checks import (
    check_classifiers_regression_target,
    check_complex_data,
    check_estimators_nan_inf,
    check_fit_score_takes_y,
    check_fit1d,
    check_fit2d_1sample,
    check_n_features_in_after_fitting,
    check_non_transformer_estimators_n_iter,
    check_requires_y_none,
    check_supervised_y_2d,
)

from weightederm import (
    WERMHuber,
    WERMHuberCV,
    WERMLeastSquares,
    WERMLeastSquaresCV,
    WERMLogistic,
    WERMLogisticCV,
)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLeastSquares", WERMLeastSquares(num_chgpts=0, search_method="brute_force")),
        ("WERMHuber", WERMHuber(num_chgpts=0, search_method="brute_force")),
        (
            "WERMLeastSquaresCV",
            WERMLeastSquaresCV(max_num_chgpts=0, cv=2, search_method="brute_force"),
        ),
        ("WERMHuberCV", WERMHuberCV(max_num_chgpts=0, cv=2, search_method="brute_force")),
    ],
)
def test_regression_estimators_pass_n_features_in_check(name, estimator):
    check_n_features_in_after_fitting(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLogistic", WERMLogistic(num_chgpts=0, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=2, search_method="brute_force")),
    ],
)
def test_logistic_estimators_pass_fit_score_takes_y_check(name, estimator):
    check_fit_score_takes_y(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLeastSquares", WERMLeastSquares(num_chgpts=0, search_method="brute_force")),
        ("WERMHuber", WERMHuber(num_chgpts=0, search_method="brute_force")),
        ("WERMLogistic", WERMLogistic(num_chgpts=0, search_method="brute_force")),
        (
            "WERMLeastSquaresCV",
            WERMLeastSquaresCV(max_num_chgpts=0, cv=2, search_method="brute_force"),
        ),
        ("WERMHuberCV", WERMHuberCV(max_num_chgpts=0, cv=2, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
    ],
)
def test_estimators_reject_complex_data(name, estimator):
    check_complex_data(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLeastSquares", WERMLeastSquares(num_chgpts=0, search_method="brute_force")),
        ("WERMHuber", WERMHuber(num_chgpts=0, search_method="brute_force")),
        ("WERMLogistic", WERMLogistic(num_chgpts=0, search_method="brute_force")),
        (
            "WERMLeastSquaresCV",
            WERMLeastSquaresCV(max_num_chgpts=0, cv=2, search_method="brute_force"),
        ),
        ("WERMHuberCV", WERMHuberCV(max_num_chgpts=0, cv=2, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
    ],
)
def test_estimators_reject_nan_and_inf(name, estimator):
    check_estimators_nan_inf(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLeastSquares", WERMLeastSquares(num_chgpts=0, search_method="brute_force")),
        ("WERMHuber", WERMHuber(num_chgpts=0, search_method="brute_force")),
        (
            "WERMLeastSquaresCV",
            WERMLeastSquaresCV(max_num_chgpts=0, cv=2, search_method="brute_force"),
        ),
        ("WERMHuberCV", WERMHuberCV(max_num_chgpts=0, cv=2, search_method="brute_force")),
    ],
)
def test_regression_estimators_handle_column_vector_y_like_sklearn(name, estimator):
    check_supervised_y_2d(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLogistic", WERMLogistic(num_chgpts=0, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
    ],
)
def test_logistic_estimators_reject_continuous_targets_like_sklearn(name, estimator):
    check_classifiers_regression_target(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLogistic", WERMLogistic(num_chgpts=0, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
    ],
)
def test_logistic_estimators_handle_column_vector_y_like_sklearn(name, estimator):
    check_supervised_y_2d(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMHuber", WERMHuber(num_chgpts=0, search_method="brute_force")),
        ("WERMLogistic", WERMLogistic(num_chgpts=0, search_method="brute_force")),
        ("WERMHuberCV", WERMHuberCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
    ],
)
def test_iterative_estimators_expose_n_iter_like_sklearn(name, estimator):
    check_non_transformer_estimators_n_iter(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLeastSquares", WERMLeastSquares(num_chgpts=0, search_method="brute_force")),
        ("WERMHuber", WERMHuber(num_chgpts=0, search_method="brute_force")),
        ("WERMLogistic", WERMLogistic(num_chgpts=0, search_method="brute_force")),
        (
            "WERMLeastSquaresCV",
            WERMLeastSquaresCV(max_num_chgpts=0, cv=5, search_method="brute_force"),
        ),
        ("WERMHuberCV", WERMHuberCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
    ],
)
def test_estimators_reject_1d_X_like_sklearn(name, estimator):
    check_fit1d(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        (
            "WERMLeastSquaresCV",
            WERMLeastSquaresCV(max_num_chgpts=0, cv=5, search_method="brute_force"),
        ),
        ("WERMHuberCV", WERMHuberCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
    ],
)
def test_cv_estimators_surface_one_sample_messages_like_sklearn(name, estimator):
    check_fit2d_1sample(name, estimator)


@pytest.mark.parametrize(
    "name, estimator",
    [
        ("WERMLeastSquares", WERMLeastSquares(num_chgpts=0, search_method="brute_force")),
        ("WERMHuber", WERMHuber(num_chgpts=0, search_method="brute_force")),
        ("WERMLogistic", WERMLogistic(num_chgpts=0, search_method="brute_force")),
        (
            "WERMLeastSquaresCV",
            WERMLeastSquaresCV(max_num_chgpts=0, cv=5, search_method="brute_force"),
        ),
        ("WERMHuberCV", WERMHuberCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
        ("WERMLogisticCV", WERMLogisticCV(max_num_chgpts=0, cv=5, search_method="brute_force")),
    ],
)
def test_estimators_require_y_with_sklearn_style_error(name, estimator):
    check_requires_y_none(name, estimator)
