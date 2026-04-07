import numpy as np
import pytest

from weightederm._fixed_werm import fit_fixed_werm_model


def test_shared_fixed_werm_engine_populates_common_outputs():
    X = np.ones((4, 1))
    y = np.array([0.0, 0.0, 10.0, 10.0])

    estimator = type("DummyEstimator", (), {})()

    def fit_signals(X, y, weights):
        theta_hat = np.column_stack(
            [
                np.full(X.shape[0], 5.0 / 3.0),
                np.full(X.shape[0], 25.0 / 3.0),
            ]
        )
        signal_coefs = np.array([[5.0 / 3.0], [25.0 / 3.0]])
        signal_intercepts = None
        return signal_coefs, signal_intercepts, theta_hat

    def squared_loss(predictions, targets):
        residuals = predictions - targets
        return residuals**2

    fit_fixed_werm_model(
        estimator,
        X,
        y,
        num_chgpts=1,
        delta=1,
        search_method="brute_force",
        fit_signals=fit_signals,
        loss=squared_loss,
    )

    assert estimator.changepoints_.tolist() == [2]
    assert estimator.num_chgpts_ == 1
    assert estimator.num_signals_ == 2
    assert estimator.n_features_in_ == 1
    assert np.isclose(estimator.objective_, 100.0 / 9.0)
    assert estimator._weights_.shape == (2, 4)
    assert estimator._signal_coefs_.shape == (2, 1)
    assert estimator._signal_intercepts_ is None
    assert estimator._theta_hat_.shape == (4, 2)


def test_shared_fixed_werm_engine_rejects_invalid_search_method():
    estimator = type("DummyEstimator", (), {})()

    with pytest.raises(ValueError, match="search_method"):
        fit_fixed_werm_model(
            estimator,
            np.ones((2, 1)),
            np.array([0.0, 1.0]),
            num_chgpts=0,
            delta=1,
            search_method="not-a-method",
            fit_signals=lambda X, y, weights: (
                np.zeros((1, X.shape[1])),
                None,
                np.zeros((X.shape[0], 1)),
            ),
            loss=lambda predictions, targets: (predictions - targets) ** 2,
        )
