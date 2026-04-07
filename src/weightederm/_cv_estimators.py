from __future__ import annotations

import numpy as np
from scipy.optimize import linprog, minimize
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

from weightederm._cv import fit_werm_cv_model
from weightederm._huber import WERMHuber, _fit_weighted_huber_signal, _huber_loss
from weightederm._least_squares import (
    WERMLeastSquares,
    _fit_weighted_least_squares_signal,
    _squared_loss,
)
from weightederm._logistic import WERMLogistic, _fit_weighted_logistic_signal, _logistic_loss
from weightederm._prediction import linear_predict, logistic_predict_proba
from weightederm._segmented_cv import fit_segmented_model, score_segmented_model
from weightederm._validation import (
    validate_alpha,
    validate_huber_epsilon,
    validate_penalty,
    prepare_cv_fit_inputs,
)


def _absolute_error_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.abs(predictions - targets)


def _split_params(params: np.ndarray, *, fit_intercept: bool) -> tuple[np.ndarray, float | None]:
    if fit_intercept:
        return np.asarray(params[:-1], dtype=float), float(params[-1])
    return np.asarray(params, dtype=float), None


def _fit_segment_signal_with_absolute_loss(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_intercept: bool,
) -> tuple[np.ndarray, float | None]:
    n_samples, n_features = X.shape
    design_matrix = X
    if fit_intercept:
        design_matrix = np.column_stack([X, np.ones(n_samples, dtype=float)])

    num_param_vars = design_matrix.shape[1]
    objective = np.concatenate(
        [np.zeros(num_param_vars, dtype=float), np.ones(n_samples, dtype=float)]
    )
    A_ub = np.vstack(
        [
            np.hstack([design_matrix, -np.eye(n_samples, dtype=float)]),
            np.hstack([-design_matrix, -np.eye(n_samples, dtype=float)]),
        ]
    )
    b_ub = np.concatenate([y, -y])
    bounds = [(None, None)] * num_param_vars + [(0.0, None)] * n_samples

    result = linprog(
        c=objective,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"Segment fitting with absolute-value loss failed: {result.message}")

    return _split_params(result.x[:num_param_vars], fit_intercept=fit_intercept)


def _fit_segment_signal_with_custom_loss(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_intercept: bool,
    m_scorer,
) -> tuple[np.ndarray, float | None]:
    n_features = X.shape[1]
    initial_params = np.zeros(n_features + int(fit_intercept), dtype=float)

    def objective(params: np.ndarray) -> float:
        coef, intercept = _split_params(params, fit_intercept=fit_intercept)
        predictions = X @ coef
        if intercept is not None:
            predictions = predictions + intercept
        return float(np.sum(m_scorer(predictions, y)))

    result = minimize(
        objective,
        initial_params,
        method="Powell",
        options={"maxiter": 1000, "xtol": 1e-6, "ftol": 1e-6},
    )
    if not result.success:
        raise ValueError(f"Segment fitting with custom CV scorer failed: {result.message}")

    return _split_params(result.x, fit_intercept=fit_intercept)


def _fit_segment_signal_for_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_intercept: bool,
    loss_kind: str,
    custom_loss=None,
    huber_epsilon: float | None = None,
    logistic_max_iter: int | None = None,
    logistic_tol: float | None = None,
) -> tuple[np.ndarray, float | None]:
    if loss_kind == "absolute":
        return _fit_segment_signal_with_absolute_loss(
            X,
            y,
            fit_intercept=fit_intercept,
        )
    if loss_kind == "squared":
        return _fit_weighted_least_squares_signal(
            X,
            y,
            np.ones(len(y), dtype=float),
            fit_intercept=fit_intercept,
            fit_solver="direct",
            penalty="none",
            alpha=0.0,
        )
    if loss_kind == "huber":
        if huber_epsilon is None:
            raise ValueError("Huber CV segment fitting requires epsilon.")
        if logistic_max_iter is None or logistic_tol is None:
            raise ValueError("Huber CV segment fitting requires max_iter and tol.")
        return _fit_weighted_huber_signal(
            X,
            y,
            np.ones(len(y), dtype=float),
            fit_intercept=fit_intercept,
            epsilon=huber_epsilon,
            max_iter=logistic_max_iter,
            tol=logistic_tol,
            penalty="none",
            alpha=0.0,
        )
    if loss_kind == "logistic":
        if logistic_max_iter is None or logistic_tol is None:
            raise ValueError("Logistic CV segment fitting requires max_iter and tol.")
        return _fit_weighted_logistic_signal(
            X,
            y,
            np.ones(len(y), dtype=float),
            fit_intercept=fit_intercept,
            max_iter=logistic_max_iter,
            tol=logistic_tol,
            penalty="l2",
            alpha=1.0,
        )
    if custom_loss is None:
        raise ValueError("Custom CV segment fitting requires a custom loss callable.")

    return _fit_segment_signal_with_custom_loss(
        X,
        y,
        fit_intercept=fit_intercept,
        m_scorer=custom_loss,
    )


def _fit_segmented_least_squares(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[int, int]],
    *,
    fit_intercept: bool,
    fit_loss_kind: str,
    score_loss,
    train_indices: np.ndarray | None = None,
) -> object:
    return fit_segmented_model(
        X,
        y,
        bounds,
        fit_segment_signal=lambda X_seg, y_seg: _fit_segment_signal_for_cv(
            X_seg,
            y_seg,
            fit_intercept=fit_intercept,
            loss_kind=fit_loss_kind,
            custom_loss=score_loss,
        ),
        train_indices=train_indices,
    )


def _score_segmented_least_squares_fit(
    segmented_fit,
    X: np.ndarray,
    y: np.ndarray,
    test_indices: np.ndarray,
    *,
    score_loss,
) -> float:
    return score_segmented_model(
        segmented_fit,
        X,
        y,
        test_indices,
        loss=score_loss,
    )


def _fit_segmented_huber(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[int, int]],
    *,
    fit_intercept: bool,
    fit_loss_kind: str,
    score_loss,
    epsilon: float,
    max_iter: int,
    tol: float,
    train_indices: np.ndarray | None = None,
) -> object:
    return fit_segmented_model(
        X,
        y,
        bounds,
        fit_segment_signal=lambda X_seg, y_seg: _fit_segment_signal_for_cv(
            X_seg,
            y_seg,
            fit_intercept=fit_intercept,
            loss_kind=fit_loss_kind,
            custom_loss=score_loss,
            huber_epsilon=epsilon,
            logistic_max_iter=max_iter,
            logistic_tol=tol,
        ),
        train_indices=train_indices,
    )


def _score_segmented_huber_fit(
    segmented_fit,
    X: np.ndarray,
    y: np.ndarray,
    test_indices: np.ndarray,
    *,
    score_loss,
) -> float:
    return score_segmented_model(
        segmented_fit,
        X,
        y,
        test_indices,
        loss=score_loss,
    )


def _make_huber_cv_loss(*, epsilon: float):
    def huber_cv_loss(predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        return _huber_loss(predictions, targets, epsilon=epsilon)

    return huber_cv_loss


def _fit_segmented_logistic(
    X: np.ndarray,
    y: np.ndarray,
    bounds: list[tuple[int, int]],
    *,
    fit_intercept: bool,
    fit_loss_kind: str,
    score_loss,
    max_iter: int,
    tol: float,
    train_indices: np.ndarray | None = None,
):
    return fit_segmented_model(
        X,
        y,
        bounds,
        fit_segment_signal=lambda X_seg, y_seg: _fit_segment_signal_for_cv(
            X_seg,
            y_seg,
            fit_intercept=fit_intercept,
            loss_kind=fit_loss_kind,
            custom_loss=score_loss,
            logistic_max_iter=max_iter,
            logistic_tol=tol,
        ),
        train_indices=train_indices,
    )


def _score_segmented_logistic_fit(
    segmented_fit,
    X: np.ndarray,
    y: np.ndarray,
    test_indices: np.ndarray,
    *,
    score_loss,
) -> float:
    return score_segmented_model(
        segmented_fit,
        X,
        y,
        test_indices,
        loss=score_loss,
    )


class WERMLeastSquaresCV(RegressorMixin, BaseEstimator):
    """Cross-validated WERM estimator with squared-loss fixed-model fits.

    Parameters
    ----------
    max_num_chgpts : int
        Largest changepoint count considered during CV.
    delta : int, default=1
        Minimum spacing enforced during changepoint search.
    search_method : {"efficient", "brute_force"}, default="efficient"
        Second-stage changepoint search strategy.
    cv : int, default=5
        Number of interleaved folds.
    fit_intercept : bool, default=True
        Whether the signal models include an intercept.
    m_scorer : callable, default=None
        Optional held-out scoring loss. If ``None``, absolute loss is used unless
        ``use_base_loss_for_cv=True``.
    use_base_loss_for_cv : bool, default=False
        If ``True``, use squared loss for the segment refits and held-out scoring.
    penalty : {"none", "l1", "l2"}, default="none"
        Penalty passed through to the inner fixed model.
    alpha : float, default=0.0
        Penalty strength passed through to the inner fixed model.
    """

    def __init__(
        self,
        max_num_chgpts,
        *,
        delta=1,
        search_method="efficient",
        cv=5,
        fit_intercept=True,
        m_scorer=None,
        use_base_loss_for_cv=False,
        penalty="none",
        alpha=0.0,
    ):
        self.max_num_chgpts = max_num_chgpts
        self.delta = delta
        self.search_method = search_method
        self.cv = cv
        self.fit_intercept = fit_intercept
        self.m_scorer = m_scorer
        self.use_base_loss_for_cv = use_base_loss_for_cv
        self.penalty = penalty
        self.alpha = alpha

    def fit(self, X, y):
        """Select the changepoint count by cross-validation and refit on full data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Ordered covariates.
        y : array-like of shape (n_samples,)
            Ordered responses.

        Returns
        -------
        self : WERMLeastSquaresCV
            Fitted estimator.
        """
        penalty = validate_penalty(self.penalty)
        alpha = validate_alpha(self.alpha)
        feature_names_in, max_num_chgpts, delta, cv, X_array, y_array, _ = prepare_cv_fit_inputs(
            X,
            y,
            max_num_chgpts=self.max_num_chgpts,
            delta=self.delta,
            cv=self.cv,
        )
        score_loss = (
            _squared_loss
            if self.use_base_loss_for_cv
            else (_absolute_error_loss if self.m_scorer is None else self.m_scorer)
        )
        fit_loss_kind = (
            "squared"
            if self.use_base_loss_for_cv
            else ("absolute" if score_loss is _absolute_error_loss else "custom")
        )

        fitted = fit_werm_cv_model(
            self,
            X_array,
            y_array,
            max_num_chgpts=max_num_chgpts,
            cv=cv,
            feature_names_in=feature_names_in,
            fit_fixed_model=lambda X, y, num_chgpts: WERMLeastSquares(
                num_chgpts=num_chgpts,
                delta=delta,
                search_method=self.search_method,
                fit_intercept=self.fit_intercept,
                penalty=penalty,
                alpha=alpha,
            ).fit(X, y),
            fit_segmented_model=lambda X, y, bounds, train_indices=None: (
                _fit_segmented_least_squares(
                    X,
                    y,
                    bounds,
                    fit_intercept=self.fit_intercept,
                    fit_loss_kind=fit_loss_kind,
                    score_loss=score_loss,
                    train_indices=train_indices,
                )
            ),
            score_segmented_model=lambda segmented_fit, X, y, test_indices: (
                _score_segmented_least_squares_fit(
                    segmented_fit,
                    X,
                    y,
                    test_indices,
                    score_loss=score_loss,
                )
            ),
            fit_last_segment_signal=lambda X, y: _fit_weighted_least_squares_signal(
                X,
                y,
                np.ones(len(y), dtype=float),
                fit_intercept=self.fit_intercept,
                fit_solver="direct",
                penalty=penalty,
                alpha=alpha,
            ),
        )
        return fitted

    def predict(self, X):
        """Predict with the final last-segment refit."""
        return linear_predict(self, X)


class WERMHuberCV(RegressorMixin, BaseEstimator):
    """Cross-validated WERM estimator with Huber-loss fixed-model fits.

    Parameters
    ----------
    max_num_chgpts : int
        Largest changepoint count considered during CV.
    delta : int, default=1
        Minimum spacing enforced during changepoint search.
    search_method : {"efficient", "brute_force"}, default="efficient"
        Second-stage changepoint search strategy.
    cv : int, default=5
        Number of interleaved folds.
    fit_intercept : bool, default=True
        Whether the signal models include an intercept.
    epsilon : float, default=1.35
        Huber transition parameter.
    max_iter : int, default=100
        Maximum number of optimizer iterations in first-stage signal fitting.
    tol : float, default=1e-5
        Optimizer tolerance.
    m_scorer : callable, default=None
        Optional held-out scoring loss. If ``None``, absolute loss is used unless
        ``use_base_loss_for_cv=True``.
    use_base_loss_for_cv : bool, default=False
        If ``True``, use Huber loss for the segment refits and held-out scoring.
    penalty : {"none", "l1", "l2"}, default="none"
        Penalty passed through to the inner fixed model.
    alpha : float, default=0.0
        Penalty strength passed through to the inner fixed model.
    """

    def __init__(
        self,
        max_num_chgpts,
        *,
        delta=1,
        search_method="efficient",
        cv=5,
        fit_intercept=True,
        epsilon=1.35,
        max_iter=100,
        tol=1e-5,
        m_scorer=None,
        use_base_loss_for_cv=False,
        penalty="none",
        alpha=0.0,
    ):
        self.max_num_chgpts = max_num_chgpts
        self.delta = delta
        self.search_method = search_method
        self.cv = cv
        self.fit_intercept = fit_intercept
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tol = tol
        self.m_scorer = m_scorer
        self.use_base_loss_for_cv = use_base_loss_for_cv
        self.penalty = penalty
        self.alpha = alpha

    def fit(self, X, y):
        """Select the changepoint count by cross-validation and refit on full data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Ordered covariates.
        y : array-like of shape (n_samples,)
            Ordered responses.

        Returns
        -------
        self : WERMHuberCV
            Fitted estimator.
        """
        penalty = validate_penalty(self.penalty)
        alpha = validate_alpha(self.alpha)
        epsilon = validate_huber_epsilon(self.epsilon)
        feature_names_in, max_num_chgpts, delta, cv, X_array, y_array, _ = prepare_cv_fit_inputs(
            X,
            y,
            max_num_chgpts=self.max_num_chgpts,
            delta=self.delta,
            cv=self.cv,
        )
        base_huber_loss = _make_huber_cv_loss(epsilon=epsilon)
        score_loss = (
            base_huber_loss
            if self.use_base_loss_for_cv
            else (_absolute_error_loss if self.m_scorer is None else self.m_scorer)
        )
        fit_loss_kind = (
            "huber"
            if self.use_base_loss_for_cv
            else ("absolute" if score_loss is _absolute_error_loss else "custom")
        )

        fitted = fit_werm_cv_model(
            self,
            X_array,
            y_array,
            max_num_chgpts=max_num_chgpts,
            cv=cv,
            feature_names_in=feature_names_in,
            fit_fixed_model=lambda X, y, num_chgpts: WERMHuber(
                num_chgpts=num_chgpts,
                delta=delta,
                search_method=self.search_method,
                fit_intercept=self.fit_intercept,
                epsilon=epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                penalty=penalty,
                alpha=alpha,
            ).fit(X, y),
            fit_segmented_model=lambda X, y, bounds, train_indices=None: _fit_segmented_huber(
                X,
                y,
                bounds,
                fit_intercept=self.fit_intercept,
                fit_loss_kind=fit_loss_kind,
                score_loss=score_loss,
                epsilon=epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                train_indices=train_indices,
            ),
            score_segmented_model=lambda segmented_fit, X, y, test_indices: (
                _score_segmented_huber_fit(
                    segmented_fit,
                    X,
                    y,
                    test_indices,
                    score_loss=score_loss,
                )
            ),
            fit_last_segment_signal=lambda X, y: _fit_weighted_huber_signal(
                X,
                y,
                np.ones(len(y), dtype=float),
                fit_intercept=self.fit_intercept,
                epsilon=epsilon,
                max_iter=self.max_iter,
                tol=self.tol,
                penalty=penalty,
                alpha=alpha,
            ),
        )
        self.n_iter_ = np.array([max(1, self.max_iter)], dtype=int)
        return fitted

    def predict(self, X):
        """Predict with the final last-segment refit."""
        return linear_predict(self, X)


class WERMLogisticCV(ClassifierMixin, BaseEstimator):
    """Cross-validated WERM estimator with binary logistic fixed-model fits.

    Parameters
    ----------
    max_num_chgpts : int
        Largest changepoint count considered during CV.
    delta : int, default=1
        Minimum spacing enforced during changepoint search.
    search_method : {"efficient", "brute_force"}, default="efficient"
        Second-stage changepoint search strategy.
    cv : int, default=5
        Number of interleaved folds.
    fit_intercept : bool, default=True
        Whether the signal models include an intercept.
    max_iter : int, default=100
        Maximum number of optimizer iterations in first-stage signal fitting.
    tol : float, default=1e-5
        Optimizer tolerance.
    m_scorer : callable, default=None
        Optional held-out scoring loss. If ``None``, logistic loss is used unless
        ``use_base_loss_for_cv=True``.
    use_base_loss_for_cv : bool, default=False
        If ``True``, use logistic loss for the segment refits and held-out scoring.
    penalty : {"none", "l1", "l2"}, default="l2"
        Penalty passed through to the inner fixed model.
    alpha : float, default=1.0
        Penalty strength passed through to the inner fixed model.
    """

    def __init__(
        self,
        max_num_chgpts,
        *,
        delta=1,
        search_method="efficient",
        cv=5,
        fit_intercept=True,
        max_iter=100,
        tol=1e-5,
        m_scorer=None,
        use_base_loss_for_cv=False,
        penalty="l2",
        alpha=1.0,
    ):
        self.max_num_chgpts = max_num_chgpts
        self.delta = delta
        self.search_method = search_method
        self.cv = cv
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.m_scorer = m_scorer
        self.use_base_loss_for_cv = use_base_loss_for_cv
        self.penalty = penalty
        self.alpha = alpha

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        return tags

    def fit(self, X, y):
        """Select the changepoint count by cross-validation and refit on full data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Ordered covariates.
        y : array-like of shape (n_samples,)
            Binary labels.

        Returns
        -------
        self : WERMLogisticCV
            Fitted estimator.
        """
        penalty = validate_penalty(self.penalty)
        alpha = validate_alpha(self.alpha)
        feature_names_in, max_num_chgpts, delta, cv, X_array, y_array, classes = (
            prepare_cv_fit_inputs(
                X,
                y,
                max_num_chgpts=self.max_num_chgpts,
                delta=self.delta,
                cv=self.cv,
                binary=True,
            )
        )
        y_binary = (y_array == classes[1]).astype(float)
        score_loss = (
            _logistic_loss if self.use_base_loss_for_cv or self.m_scorer is None else self.m_scorer
        )
        fit_loss_kind = (
            "logistic" if (self.use_base_loss_for_cv or score_loss is _logistic_loss) else "custom"
        )

        fitted = fit_werm_cv_model(
            self,
            X_array,
            y_binary,
            max_num_chgpts=max_num_chgpts,
            cv=cv,
            feature_names_in=feature_names_in,
            fit_fixed_model=lambda X, y, num_chgpts: WERMLogistic(
                num_chgpts=num_chgpts,
                delta=delta,
                search_method=self.search_method,
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                penalty=penalty,
                alpha=alpha,
            ).fit(X, y),
            fit_segmented_model=lambda X, y, bounds, train_indices=None: _fit_segmented_logistic(
                X,
                y,
                bounds,
                fit_intercept=self.fit_intercept,
                fit_loss_kind=fit_loss_kind,
                score_loss=score_loss,
                max_iter=self.max_iter,
                tol=self.tol,
                train_indices=train_indices,
            ),
            score_segmented_model=lambda segmented_fit, X, y, test_indices: (
                _score_segmented_logistic_fit(
                    segmented_fit,
                    X,
                    y,
                    test_indices,
                    score_loss=score_loss,
                )
            ),
            fit_last_segment_signal=lambda X, y: _fit_weighted_logistic_signal(
                X,
                y,
                np.ones(len(y), dtype=float),
                fit_intercept=self.fit_intercept,
                max_iter=self.max_iter,
                tol=self.tol,
                penalty=penalty,
                alpha=alpha,
            ),
            extra_attributes={"classes_": classes},
        )
        self.n_iter_ = np.array([max(1, self.max_iter)], dtype=int)
        return fitted

    def predict_proba(self, X):
        """Estimate class probabilities with the final last-segment refit."""
        return logistic_predict_proba(self, X)

    def predict(self, X):
        """Predict class labels with the final last-segment refit."""
        probabilities = self.predict_proba(X)[:, 1]
        return np.where(probabilities >= 0.5, self.classes_[1], self.classes_[0])
