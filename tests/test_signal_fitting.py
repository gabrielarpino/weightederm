import numpy as np

from weightederm._signal_fitting import fit_weighted_signals


def test_shared_signal_fitting_helper_returns_expected_shapes_with_intercepts():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([10.0, 20.0, 30.0])
    weights = np.array([[1.0, 0.5, 0.25], [0.25, 0.5, 1.0]])

    def fit_signal(X, y, sample_weights):
        level = float(np.average(y, weights=sample_weights))
        return np.array([0.0]), level

    coefs, intercepts, theta_hat = fit_weighted_signals(
        X,
        y,
        weights,
        fit_signal=fit_signal,
    )

    assert coefs.shape == (2, 1)
    assert intercepts.shape == (2,)
    assert theta_hat.shape == (3, 2)
    np.testing.assert_allclose(coefs, np.zeros((2, 1)))
    np.testing.assert_allclose(theta_hat[:, 0], np.repeat(intercepts[0], 3))
    np.testing.assert_allclose(theta_hat[:, 1], np.repeat(intercepts[1], 3))


def test_shared_signal_fitting_helper_returns_expected_shapes_without_intercepts():
    X = np.array([[1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0])
    weights = np.array([[1.0, 1.0, 1.0], [1.0, 0.5, 0.25]])

    def fit_signal(X, y, sample_weights):
        slope = float(np.sum(sample_weights * X[:, 0] * y) / np.sum(sample_weights * X[:, 0] ** 2))
        return np.array([slope]), None

    coefs, intercepts, theta_hat = fit_weighted_signals(
        X,
        y,
        weights,
        fit_signal=fit_signal,
    )

    assert coefs.shape == (2, 1)
    assert intercepts is None
    assert theta_hat.shape == (3, 2)
    np.testing.assert_allclose(theta_hat[:, 0], X[:, 0] * coefs[0, 0])
    np.testing.assert_allclose(theta_hat[:, 1], X[:, 0] * coefs[1, 0])
