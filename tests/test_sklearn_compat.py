import numpy as np
import pytest
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from weightederm import (
    WERMHuber,
    WERMHuberCV,
    WERMLeastSquares,
    WERMLeastSquaresCV,
    WERMLogistic,
    WERMLogisticCV,
)


@pytest.mark.parametrize(
    ("estimator", "expected_params"),
    [
        (
            WERMLeastSquares(
                num_chgpts=1, delta=2, search_method="brute_force", fit_intercept=False
            ),
            {"num_chgpts": 1, "delta": 2, "search_method": "brute_force", "fit_intercept": False},
        ),
        (
            WERMHuber(
                num_chgpts=1,
                delta=2,
                search_method="brute_force",
                fit_intercept=False,
                epsilon=2.0,
                max_iter=50,
                tol=1e-4,
            ),
            {
                "num_chgpts": 1,
                "delta": 2,
                "search_method": "brute_force",
                "fit_intercept": False,
                "epsilon": 2.0,
                "max_iter": 50,
                "tol": 1e-4,
            },
        ),
        (
            WERMLogistic(
                num_chgpts=1,
                delta=2,
                search_method="brute_force",
                fit_intercept=False,
                max_iter=50,
                tol=1e-4,
            ),
            {
                "num_chgpts": 1,
                "delta": 2,
                "search_method": "brute_force",
                "fit_intercept": False,
                "max_iter": 50,
                "tol": 1e-4,
            },
        ),
        (
            WERMLeastSquaresCV(
                max_num_chgpts=2,
                delta=2,
                search_method="brute_force",
                cv=3,
                fit_intercept=False,
            ),
            {
                "max_num_chgpts": 2,
                "delta": 2,
                "search_method": "brute_force",
                "cv": 3,
                "fit_intercept": False,
                "m_scorer": None,
                "use_base_loss_for_cv": False,
            },
        ),
        (
            WERMHuberCV(
                max_num_chgpts=2,
                delta=2,
                search_method="brute_force",
                cv=3,
                fit_intercept=False,
                epsilon=2.0,
                max_iter=50,
                tol=1e-4,
            ),
            {
                "max_num_chgpts": 2,
                "delta": 2,
                "search_method": "brute_force",
                "cv": 3,
                "fit_intercept": False,
                "epsilon": 2.0,
                "max_iter": 50,
                "tol": 1e-4,
                "m_scorer": None,
                "use_base_loss_for_cv": False,
            },
        ),
        (
            WERMLogisticCV(
                max_num_chgpts=2,
                delta=2,
                search_method="brute_force",
                cv=3,
                fit_intercept=False,
                max_iter=50,
                tol=1e-4,
            ),
            {
                "max_num_chgpts": 2,
                "delta": 2,
                "search_method": "brute_force",
                "cv": 3,
                "fit_intercept": False,
                "max_iter": 50,
                "tol": 1e-4,
                "m_scorer": None,
                "use_base_loss_for_cv": False,
            },
        ),
    ],
)
def test_estimators_expose_sklearn_style_params(estimator, expected_params):
    params = estimator.get_params()

    for name, value in expected_params.items():
        assert params[name] == value


def test_clone_preserves_estimator_type_and_params():
    estimator = WERMHuberCV(
        max_num_chgpts=2,
        delta=2,
        search_method="brute_force",
        cv=3,
        fit_intercept=False,
        epsilon=2.0,
        max_iter=50,
        tol=1e-4,
    )

    cloned = clone(estimator)

    assert isinstance(cloned, WERMHuberCV)
    assert cloned is not estimator
    assert cloned.get_params() == estimator.get_params()


def test_set_params_updates_estimator_attributes():
    estimator = WERMLeastSquares(
        num_chgpts=0, delta=1, search_method="efficient", fit_intercept=True
    )

    returned = estimator.set_params(num_chgpts=1, delta=2, search_method="brute_force")

    assert returned is estimator
    assert estimator.num_chgpts == 1
    assert estimator.delta == 2
    assert estimator.search_method == "brute_force"


def test_pipeline_fit_works_with_fixed_estimator():
    X = np.arange(1.0, 9.0).reshape(-1, 1)
    y = 3.0 * X[:, 0] - 2.0

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", WERMLeastSquares(num_chgpts=0, search_method="brute_force")),
        ]
    )
    pipeline.fit(X, y)

    assert pipeline.named_steps["model"].num_chgpts_ == 0


def test_pipeline_fit_works_with_cv_estimator():
    X = np.arange(1.0, 9.0).reshape(-1, 1)
    y = 3.0 * X[:, 0] - 2.0

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", WERMLeastSquaresCV(max_num_chgpts=1, cv=2, search_method="brute_force")),
        ]
    )
    pipeline.fit(X, y)

    assert pipeline.named_steps["model"].best_num_chgpts_ == 0
