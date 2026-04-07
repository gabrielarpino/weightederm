import numpy as np

from weightederm._segmented_cv import fit_segmented_model, score_segmented_model


def test_shared_segmented_fit_builds_expected_outputs_with_intercepts():
    X = np.zeros((6, 1))
    y = np.array([1.0, 1.0, 1.0, 9.0, 9.0, 9.0])
    bounds = [(0, 3), (3, 6)]

    def fit_segment_signal(X_seg, y_seg):
        return np.array([0.0]), float(np.mean(y_seg))

    segmented_fit = fit_segmented_model(X, y, bounds, fit_segment_signal=fit_segment_signal)

    assert segmented_fit.bounds == bounds
    assert segmented_fit.coefs.shape == (2, 1)
    np.testing.assert_allclose(segmented_fit.coefs, np.zeros((2, 1)))
    np.testing.assert_allclose(segmented_fit.intercepts, np.array([1.0, 9.0]))


def test_shared_segmented_fit_uses_train_indices_and_scores_with_loss():
    X = np.array([[1.0], [2.0], [3.0], [1.0], [2.0], [3.0]])
    y = np.array([1.0, 2.0, 3.0, 3.0, 6.0, 9.0])
    bounds = [(0, 3), (3, 6)]
    train_indices = np.array([0, 2, 3, 5])
    test_indices = np.array([1, 4])

    def fit_segment_signal(X_seg, y_seg):
        slope = float(np.sum(X_seg[:, 0] * y_seg) / np.sum(X_seg[:, 0] ** 2))
        return np.array([slope]), None

    segmented_fit = fit_segmented_model(
        X,
        y,
        bounds,
        fit_segment_signal=fit_segment_signal,
        train_indices=train_indices,
    )

    def loss(predictions, targets):
        return (predictions - targets) ** 2

    score = score_segmented_model(segmented_fit, X, y, test_indices, loss=loss)

    assert segmented_fit.intercepts is None
    assert segmented_fit.coefs.shape == (2, 1)
    assert np.isclose(score, 0.0)
