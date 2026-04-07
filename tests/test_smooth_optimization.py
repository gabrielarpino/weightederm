import numpy as np

from weightederm._smooth_optimization import fit_weighted_smooth_signal


def test_shared_smooth_optimizer_returns_expected_shape_with_intercept():
    target = np.array([2.5, -1.0])

    def objective_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        residual = params - target
        return float(np.sum(residual**2)), 2.0 * residual

    coef, intercept = fit_weighted_smooth_signal(
        np.ones((4, 1)),
        np.arange(4.0),
        np.ones(4),
        fit_intercept=True,
        objective_and_gradient=objective_and_gradient,
        max_iter=100,
        tol=1e-8,
        estimator_name="test",
    )

    assert coef.shape == (1,)
    assert isinstance(intercept, float)
    np.testing.assert_allclose(coef, np.array([2.5]), atol=1e-6)
    assert np.isclose(intercept, -1.0, atol=1e-6)


def test_shared_smooth_optimizer_returns_expected_shape_without_intercept():
    target = np.array([3.0, -2.0])

    def objective_and_gradient(params: np.ndarray) -> tuple[float, np.ndarray]:
        residual = params - target
        return float(np.sum(residual**2)), 2.0 * residual

    coef, intercept = fit_weighted_smooth_signal(
        np.ones((4, 2)),
        np.arange(4.0),
        np.ones(4),
        fit_intercept=False,
        objective_and_gradient=objective_and_gradient,
        max_iter=100,
        tol=1e-8,
        estimator_name="test",
    )

    assert coef.shape == (2,)
    assert intercept is None
    np.testing.assert_allclose(coef, target, atol=1e-6)
