from __future__ import annotations

import numpy as np
import pytest

from weightederm._huber import (
    _fit_weighted_huber_signal,
    _weighted_huber_objective,
)
from weightederm._logistic import (
    _fit_weighted_logistic_signal,
    _weighted_logistic_objective,
)
from weightederm._least_squares import (
    _fit_weighted_least_squares_signal,
    _weighted_squared_loss_objective,
)
from weightederm._validation import validate_alpha, validate_penalty
from weightederm import (
    WERMHuber,
    WERMHuberCV,
    WERMLeastSquares,
    WERMLeastSquaresCV,
    WERMLogistic,
    WERMLogisticCV,
)


# ---------------------------------------------------------------------------
# validate_penalty
# ---------------------------------------------------------------------------


def test_validate_penalty_accepts_none():
    assert validate_penalty("none") == "none"


def test_validate_penalty_accepts_l1():
    assert validate_penalty("l1") == "l1"


def test_validate_penalty_accepts_l2():
    assert validate_penalty("l2") == "l2"


def test_validate_penalty_rejects_invalid_string():
    with pytest.raises(ValueError, match="penalty"):
        validate_penalty("ridge")


def test_validate_penalty_rejects_none_type():
    with pytest.raises(ValueError, match="penalty"):
        validate_penalty(None)


# ---------------------------------------------------------------------------
# validate_alpha
# ---------------------------------------------------------------------------


def test_validate_alpha_accepts_zero():
    assert validate_alpha(0.0) == pytest.approx(0.0)


def test_validate_alpha_accepts_positive_float():
    assert validate_alpha(1.5) == pytest.approx(1.5)


def test_validate_alpha_accepts_integer():
    assert validate_alpha(2) == pytest.approx(2.0)


def test_validate_alpha_rejects_negative():
    with pytest.raises(ValueError, match="alpha"):
        validate_alpha(-0.1)


def test_validate_alpha_rejects_non_numeric():
    with pytest.raises(ValueError, match="alpha"):
        validate_alpha("big")


# ---------------------------------------------------------------------------
# WERMLogistic defaults
# ---------------------------------------------------------------------------


def test_werm_logistic_defaults_to_l2_penalty_with_alpha_one():
    est = WERMLogistic(num_chgpts=0)

    assert est.penalty == "l2"
    assert est.alpha == 1.0


def test_werm_logistic_penalty_and_alpha_appear_in_get_params():
    est = WERMLogistic(num_chgpts=0, penalty="l1", alpha=0.5)

    params = est.get_params()
    assert params["penalty"] == "l1"
    assert params["alpha"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# WERMHuber defaults
# ---------------------------------------------------------------------------


def test_werm_huber_defaults_to_no_penalty():
    est = WERMHuber(num_chgpts=0)

    assert est.penalty == "none"
    assert est.alpha == 0.0


def test_werm_huber_penalty_and_alpha_appear_in_get_params():
    est = WERMHuber(num_chgpts=0, penalty="l2", alpha=2.0)

    params = est.get_params()
    assert params["penalty"] == "l2"
    assert params["alpha"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Logistic objective: L2 penalty
# ---------------------------------------------------------------------------


def test_l2_logistic_objective_adds_ridge_term():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(4)
    params = np.array([0.5])  # fit_intercept=False
    alpha = 2.0

    obj_none, _ = _weighted_logistic_objective(
        params, X, y, weights, fit_intercept=False, penalty="none", alpha=0.0
    )
    obj_l2, _ = _weighted_logistic_objective(
        params, X, y, weights, fit_intercept=False, penalty="l2", alpha=alpha
    )

    expected_penalty = alpha * float(np.sum(params**2))
    assert obj_l2 == pytest.approx(obj_none + expected_penalty)


def test_l2_logistic_gradient_adds_ridge_to_coef_not_intercept():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(4)
    params = np.array([0.5, 0.1])  # fit_intercept=True: [coef, intercept]
    alpha = 2.0

    _, grad_none = _weighted_logistic_objective(
        params, X, y, weights, fit_intercept=True, penalty="none", alpha=0.0
    )
    _, grad_l2 = _weighted_logistic_objective(
        params, X, y, weights, fit_intercept=True, penalty="l2", alpha=alpha
    )

    # coef position gets 2*alpha*coef; intercept position unchanged
    expected_delta = np.array([2.0 * alpha * params[0], 0.0])
    np.testing.assert_allclose(grad_l2, grad_none + expected_delta)


# ---------------------------------------------------------------------------
# Logistic objective: L1 penalty
# ---------------------------------------------------------------------------


def test_l1_logistic_objective_adds_lasso_term():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(4)
    params = np.array([0.5])
    alpha = 1.5

    obj_none, _ = _weighted_logistic_objective(
        params, X, y, weights, fit_intercept=False, penalty="none", alpha=0.0
    )
    obj_l1, _ = _weighted_logistic_objective(
        params, X, y, weights, fit_intercept=False, penalty="l1", alpha=alpha
    )

    expected_penalty = alpha * float(np.sum(np.abs(params)))
    assert obj_l1 == pytest.approx(obj_none + expected_penalty)


def test_l1_logistic_gradient_adds_subgradient_to_coef_not_intercept():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(4)
    params = np.array([0.5, 0.1])  # coef > 0, so sign = +1
    alpha = 1.5

    _, grad_none = _weighted_logistic_objective(
        params, X, y, weights, fit_intercept=True, penalty="none", alpha=0.0
    )
    _, grad_l1 = _weighted_logistic_objective(
        params, X, y, weights, fit_intercept=True, penalty="l1", alpha=alpha
    )

    expected_delta = np.array([alpha * np.sign(params[0]), 0.0])
    np.testing.assert_allclose(grad_l1, grad_none + expected_delta)


# ---------------------------------------------------------------------------
# Logistic signal fitter: shrinkage + separability skip
# ---------------------------------------------------------------------------


def test_l2_logistic_larger_alpha_shrinks_coef():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    weights = np.ones(6)

    coef_weak, _ = _fit_weighted_logistic_signal(
        X,
        y,
        weights,
        fit_intercept=False,
        max_iter=500,
        tol=1e-7,
        penalty="l2",
        alpha=0.001,
    )
    coef_strong, _ = _fit_weighted_logistic_signal(
        X,
        y,
        weights,
        fit_intercept=False,
        max_iter=500,
        tol=1e-7,
        penalty="l2",
        alpha=100.0,
    )

    assert np.linalg.norm(coef_strong) < np.linalg.norm(coef_weak)


def test_penalized_logistic_skips_separability_check():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    weights = np.ones(4)

    coef, intercept = _fit_weighted_logistic_signal(
        X,
        y,
        weights,
        fit_intercept=True,
        max_iter=200,
        tol=1e-6,
        penalty="l2",
        alpha=1.0,
    )

    assert coef.shape == (1,)
    assert isinstance(intercept, float)


def test_werm_logistic_default_penalty_fits_separable_data():
    X = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])

    estimator = WERMLogistic(num_chgpts=0, search_method="brute_force", max_iter=200, tol=1e-6)
    estimator.fit(X, y)  # default penalty="l2" — must not raise

    assert hasattr(estimator, "changepoints_")


# ---------------------------------------------------------------------------
# Huber objective: L2 penalty
# ---------------------------------------------------------------------------


def test_l2_huber_objective_adds_ridge_term():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    params = np.array([0.5])
    alpha = 2.0

    obj_none, _ = _weighted_huber_objective(
        params, X, y, weights, epsilon=1.35, fit_intercept=False, penalty="none", alpha=0.0
    )
    obj_l2, _ = _weighted_huber_objective(
        params, X, y, weights, epsilon=1.35, fit_intercept=False, penalty="l2", alpha=alpha
    )

    expected_penalty = alpha * float(np.sum(params**2))
    assert obj_l2 == pytest.approx(obj_none + expected_penalty)


def test_l2_huber_gradient_adds_ridge_to_coef_not_intercept():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    params = np.array([0.5, 0.1])  # fit_intercept=True
    alpha = 2.0

    _, grad_none = _weighted_huber_objective(
        params, X, y, weights, epsilon=1.35, fit_intercept=True, penalty="none", alpha=0.0
    )
    _, grad_l2 = _weighted_huber_objective(
        params, X, y, weights, epsilon=1.35, fit_intercept=True, penalty="l2", alpha=alpha
    )

    expected_delta = np.array([2.0 * alpha * params[0], 0.0])
    np.testing.assert_allclose(grad_l2, grad_none + expected_delta)


# ---------------------------------------------------------------------------
# Huber objective: L1 penalty
# ---------------------------------------------------------------------------


def test_l1_huber_objective_adds_lasso_term():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    params = np.array([0.5])
    alpha = 1.5

    obj_none, _ = _weighted_huber_objective(
        params, X, y, weights, epsilon=1.35, fit_intercept=False, penalty="none", alpha=0.0
    )
    obj_l1, _ = _weighted_huber_objective(
        params, X, y, weights, epsilon=1.35, fit_intercept=False, penalty="l1", alpha=alpha
    )

    expected_penalty = alpha * float(np.sum(np.abs(params)))
    assert obj_l1 == pytest.approx(obj_none + expected_penalty)


# ---------------------------------------------------------------------------
# Huber signal fitter: shrinkage
# ---------------------------------------------------------------------------


def test_l2_huber_larger_alpha_shrinks_coef():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    weights = np.ones(6)

    coef_weak, _ = _fit_weighted_huber_signal(
        X,
        y,
        weights,
        fit_intercept=False,
        epsilon=1.35,
        max_iter=500,
        tol=1e-7,
        penalty="l2",
        alpha=0.001,
    )
    coef_strong, _ = _fit_weighted_huber_signal(
        X,
        y,
        weights,
        fit_intercept=False,
        epsilon=1.35,
        max_iter=500,
        tol=1e-7,
        penalty="l2",
        alpha=100.0,
    )

    assert np.linalg.norm(coef_strong) < np.linalg.norm(coef_weak)


# ---------------------------------------------------------------------------
# Validation errors bubble up through estimator fit()
# ---------------------------------------------------------------------------


def test_werm_logistic_fit_raises_for_invalid_penalty():
    est = WERMLogistic(num_chgpts=0, penalty="elasticnet")

    with pytest.raises(ValueError, match="penalty"):
        est.fit(np.zeros((4, 1)), np.array([0, 0, 1, 1]))


def test_werm_huber_fit_raises_for_negative_alpha():
    est = WERMHuber(num_chgpts=0, penalty="l2", alpha=-1.0)

    with pytest.raises(ValueError, match="alpha"):
        est.fit(np.zeros((4, 1)), np.array([1.0, 2.0, 3.0, 4.0]))


# ---------------------------------------------------------------------------
# WERMLeastSquares defaults
# ---------------------------------------------------------------------------


def test_werm_least_squares_defaults_to_no_penalty():
    est = WERMLeastSquares(num_chgpts=0)

    assert est.penalty == "none"
    assert est.alpha == 0.0


def test_werm_least_squares_penalty_and_alpha_appear_in_get_params():
    est = WERMLeastSquares(num_chgpts=0, penalty="l2", alpha=3.0)

    params = est.get_params()
    assert params["penalty"] == "l2"
    assert params["alpha"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# LeastSquares objective: L2 penalty
# ---------------------------------------------------------------------------


def test_l2_least_squares_objective_adds_ridge_term():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    params = np.array([0.5])
    alpha = 2.0

    obj_none, _ = _weighted_squared_loss_objective(
        params, X, y, weights, fit_intercept=False, penalty="none", alpha=0.0
    )
    obj_l2, _ = _weighted_squared_loss_objective(
        params, X, y, weights, fit_intercept=False, penalty="l2", alpha=alpha
    )

    expected_penalty = alpha * float(np.sum(params**2))
    assert obj_l2 == pytest.approx(obj_none + expected_penalty)


def test_l2_least_squares_gradient_adds_ridge_to_coef_not_intercept():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    params = np.array([0.5, 0.1])  # fit_intercept=True: [coef, intercept]
    alpha = 2.0

    _, grad_none = _weighted_squared_loss_objective(
        params, X, y, weights, fit_intercept=True, penalty="none", alpha=0.0
    )
    _, grad_l2 = _weighted_squared_loss_objective(
        params, X, y, weights, fit_intercept=True, penalty="l2", alpha=alpha
    )

    expected_delta = np.array([2.0 * alpha * params[0], 0.0])
    np.testing.assert_allclose(grad_l2, grad_none + expected_delta)


# ---------------------------------------------------------------------------
# LeastSquares objective: L1 penalty
# ---------------------------------------------------------------------------


def test_l1_least_squares_objective_adds_lasso_term():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)
    params = np.array([0.5])
    alpha = 1.5

    obj_none, _ = _weighted_squared_loss_objective(
        params, X, y, weights, fit_intercept=False, penalty="none", alpha=0.0
    )
    obj_l1, _ = _weighted_squared_loss_objective(
        params, X, y, weights, fit_intercept=False, penalty="l1", alpha=alpha
    )

    expected_penalty = alpha * float(np.sum(np.abs(params)))
    assert obj_l1 == pytest.approx(obj_none + expected_penalty)


# ---------------------------------------------------------------------------
# LeastSquares signal fitter: shrinkage + L1 direct solver fallback
# ---------------------------------------------------------------------------


def test_l2_least_squares_larger_alpha_shrinks_coef():
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    weights = np.ones(6)

    coef_weak, _ = _fit_weighted_least_squares_signal(
        X,
        y,
        weights,
        fit_intercept=False,
        penalty="l2",
        alpha=0.001,
    )
    coef_strong, _ = _fit_weighted_least_squares_signal(
        X,
        y,
        weights,
        fit_intercept=False,
        penalty="l2",
        alpha=100.0,
    )

    assert np.linalg.norm(coef_strong) < np.linalg.norm(coef_weak)


def test_l1_direct_least_squares_issues_warning_and_falls_back_to_lbfgsb():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    y = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.ones(4)

    with pytest.warns(RuntimeWarning, match="L1"):
        coef, _ = _fit_weighted_least_squares_signal(
            X,
            y,
            weights,
            fit_intercept=False,
            fit_solver="direct",
            penalty="l1",
            alpha=1.0,
        )

    assert coef.shape == (1,)


# ---------------------------------------------------------------------------
# Validation errors bubble up through WERMLeastSquares.fit()
# ---------------------------------------------------------------------------


def test_werm_least_squares_fit_raises_for_invalid_penalty():
    est = WERMLeastSquares(num_chgpts=0, penalty="elasticnet")

    with pytest.raises(ValueError, match="penalty"):
        est.fit(np.zeros((4, 1)), np.array([1.0, 2.0, 3.0, 4.0]))


def test_werm_least_squares_fit_raises_for_negative_alpha():
    est = WERMLeastSquares(num_chgpts=0, penalty="l2", alpha=-1.0)

    with pytest.raises(ValueError, match="alpha"):
        est.fit(np.zeros((4, 1)), np.array([1.0, 2.0, 3.0, 4.0]))


# ---------------------------------------------------------------------------
# CV estimators: penalty/alpha defaults and pass-through
# ---------------------------------------------------------------------------


def test_werm_least_squares_cv_defaults_to_no_penalty():
    est = WERMLeastSquaresCV(max_num_chgpts=1)

    assert est.penalty == "none"
    assert est.alpha == 0.0


def test_werm_huber_cv_defaults_to_no_penalty():
    est = WERMHuberCV(max_num_chgpts=1)

    assert est.penalty == "none"
    assert est.alpha == 0.0


def test_werm_logistic_cv_defaults_to_l2_penalty_with_alpha_one():
    est = WERMLogisticCV(max_num_chgpts=1)

    assert est.penalty == "l2"
    assert est.alpha == 1.0


def test_cv_penalty_and_alpha_appear_in_get_params():
    est = WERMLeastSquaresCV(max_num_chgpts=1, penalty="l2", alpha=2.0)

    params = est.get_params()
    assert params["penalty"] == "l2"
    assert params["alpha"] == pytest.approx(2.0)


def test_least_squares_cv_with_l2_penalty_fits_without_error():
    X = np.arange(1.0, 9.0).reshape(-1, 1)
    y = np.array([1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0])

    est = WERMLeastSquaresCV(
        max_num_chgpts=1,
        cv=2,
        search_method="brute_force",
        fit_intercept=False,
        penalty="l2",
        alpha=1.0,
    )
    est.fit(X, y)

    assert hasattr(est, "changepoints_")


def test_cv_fit_raises_for_invalid_penalty():
    est = WERMLeastSquaresCV(max_num_chgpts=1, penalty="ridge")

    with pytest.raises(ValueError, match="penalty"):
        est.fit(np.zeros((8, 1)), np.ones(8))
