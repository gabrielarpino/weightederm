import numpy as np

from weightederm._cv import (
    fit_werm_cv_model,
    make_interleaved_folds,
    segment_bounds_from_changepoints,
)


def test_make_interleaved_folds_matches_reference_pattern():
    folds = make_interleaved_folds(n_samples=8, cv=3)

    assert [fold.tolist() for fold in folds] == [[0, 3, 6], [1, 4, 7], [2, 5]]


def test_segment_bounds_from_changepoints_uses_half_open_intervals():
    bounds = segment_bounds_from_changepoints(8, np.array([2, 5]))

    assert bounds == [(0, 2), (2, 5), (5, 8)]


def test_shared_cv_engine_populates_common_outputs():
    estimator = type("DummyEstimator", (), {})()
    X = np.zeros((8, 1))
    y = np.array([0.0, 0.0, 0.0, 0.0, 10.0, 10.0, 10.0, 10.0])

    class DummyFixedModel:
        def __init__(self, num_chgpts: int):
            self.changepoints_ = (
                np.array([4], dtype=int) if num_chgpts == 1 else np.array([], dtype=int)
            )
            self.num_chgpts_ = num_chgpts
            self.num_signals_ = num_chgpts + 1
            self.objective_ = float(num_chgpts)
            self.n_features_in_ = 1
            self._weights_ = np.zeros((num_chgpts + 1, 8))
            self._signal_coefs_ = np.zeros((num_chgpts + 1, 1))
            self._signal_intercepts_ = np.zeros(num_chgpts + 1)
            self._theta_hat_ = np.zeros((8, num_chgpts + 1))

    class DummySegmentedFit:
        def __init__(self, bounds):
            self.bounds = bounds
            self.coefs = np.zeros((len(bounds), 1))
            self.intercepts = np.array([0.0, 10.0]) if len(bounds) == 2 else np.array([5.0])

    def fit_fixed_model(X, y, num_chgpts):
        return DummyFixedModel(num_chgpts)

    def fit_segmented_model(X, y, bounds, *, train_indices=None):
        return DummySegmentedFit(bounds)

    def score_segmented_model(segmented_fit, X, y, test_indices):
        return 10.0 if len(segmented_fit.bounds) == 1 else 0.0

    fit_werm_cv_model(
        estimator,
        X,
        y,
        max_num_chgpts=1,
        cv=2,
        fit_fixed_model=fit_fixed_model,
        fit_segmented_model=fit_segmented_model,
        score_segmented_model=score_segmented_model,
    )

    assert estimator.best_num_chgpts_ == 1
    assert estimator.num_chgpts_grid_.tolist() == [0, 1]
    assert estimator.changepoints_.tolist() == [4]
    assert estimator.segment_bounds_ == [(0, 4), (4, 8)]
    assert estimator.segment_coefs_.shape == (2, 1)


def test_shared_cv_engine_breaks_score_ties_in_favor_of_smaller_num_chgpts():
    estimator = type("DummyEstimator", (), {})()
    X = np.zeros((6, 1))
    y = np.zeros(6)

    class DummyFixedModel:
        def __init__(self, num_chgpts: int):
            self.changepoints_ = np.array([], dtype=int)
            self.num_chgpts_ = num_chgpts
            self.num_signals_ = num_chgpts + 1
            self.objective_ = 0.0
            self.n_features_in_ = 1
            self._weights_ = np.zeros((num_chgpts + 1, 6))
            self._signal_coefs_ = np.zeros((num_chgpts + 1, 1))
            self._signal_intercepts_ = np.zeros(num_chgpts + 1)
            self._theta_hat_ = np.zeros((6, num_chgpts + 1))

    class DummySegmentedFit:
        def __init__(self):
            self.bounds = [(0, 6)]
            self.coefs = np.zeros((1, 1))
            self.intercepts = np.array([0.0])

    fit_werm_cv_model(
        estimator,
        X,
        y,
        max_num_chgpts=2,
        cv=2,
        fit_fixed_model=lambda X, y, num_chgpts: DummyFixedModel(num_chgpts),
        fit_segmented_model=lambda X, y, bounds, train_indices=None: DummySegmentedFit(),
        score_segmented_model=lambda segmented_fit, X, y, test_indices: 1.0,
    )

    assert estimator.best_index_ == 0
    assert estimator.best_num_chgpts_ == 0
    assert estimator.best_score_ == estimator.cv_results_["mean_test_score"][0]
