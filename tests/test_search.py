import numpy as np

from weightederm._search import search_changepoints


def test_shared_search_brute_force_recovers_single_changepoint():
    theta_hat = np.array(
        [
            [1.0, 9.0],
            [1.0, 9.0],
            [1.0, 9.0],
            [1.0, 9.0],
        ]
    )
    y = np.array([1.0, 1.0, 9.0, 9.0])

    changepoints, objective = search_changepoints(
        theta_hat,
        y,
        loss=lambda predictions, targets: (predictions - targets) ** 2,
        num_chgpts=1,
        delta=1,
        search_method="brute_force",
    )

    assert changepoints.tolist() == [2]
    assert objective == 0.0


def test_shared_search_efficient_recovers_single_changepoint():
    theta_hat = np.array(
        [
            [1.0, 9.0],
            [1.0, 9.0],
            [1.0, 9.0],
            [1.0, 9.0],
        ]
    )
    y = np.array([1.0, 1.0, 9.0, 9.0])

    changepoints, objective = search_changepoints(
        theta_hat,
        y,
        loss=lambda predictions, targets: (predictions - targets) ** 2,
        num_chgpts=1,
        delta=1,
        search_method="efficient",
    )

    assert changepoints.tolist() == [2]
    assert objective == 0.0


def test_shared_search_handles_zero_changepoints():
    theta_hat = np.array([[2.0], [2.0], [2.0]])
    y = np.array([2.0, 2.0, 2.0])

    changepoints, objective = search_changepoints(
        theta_hat,
        y,
        loss=lambda predictions, targets: (predictions - targets) ** 2,
        num_chgpts=0,
        delta=1,
        search_method="brute_force",
    )

    assert changepoints.tolist() == []
    assert objective == 0.0
