# Fitted Attributes

All attributes listed here are set by `fit()` and are available immediately
afterwards. Attributes prefixed with `_` are internal but accessible for
debugging and research.

---

## All estimators

### `changepoints_`

```
ndarray of shape (num_chgpts_,), dtype int
```

0-indexed sample positions of the detected changepoints, sorted in ascending
order. A changepoint at index `k` means the model parameters shift **between**
`y[k-1]` and `y[k]` — that is, samples `0..k-1` belong to the preceding
segment and samples `k..` belong to the following segment.

**Example:**

```python
# ATTRS_TEST: changepoints
import numpy as np
from weightederm import WERMLeastSquares

X = np.ones((60, 1))
y = np.concatenate([np.zeros(30), np.ones(30)])
model = WERMLeastSquares(num_chgpts=1, delta=4, fit_intercept=False).fit(X, y)

cp = model.changepoints_[0]
print(f"changepoint at index {cp}: y[cp-1]={y[cp-1]}, y[cp]={y[cp]}")
assert y[cp - 1] == 0.0 and y[cp] == 1.0
```

### `num_chgpts_`

```
int
```

Number of detected changepoints. For fixed estimators this always equals the
constructor argument `num_chgpts`. For CV estimators this equals
`best_num_chgpts_`.

### `num_signals_`

```
int
```

Number of segments (`num_chgpts_ + 1`).

### `objective_`

```
float
```

Minimised WERM objective value at `changepoints_`. Lower values indicate a
better fit to the data under the current signal weights. Not comparable across
different values of `num_chgpts` without adjusting for model complexity.

### `n_features_in_`

```
int
```

Number of features seen during `fit()`. `predict()` raises a `ValueError` if
`X` has a different number of columns.

### `feature_names_in_`

```
ndarray of shape (n_features_in_,), dtype object   [present only when X is a DataFrame]
```

Column names of the input DataFrame, if `X` was passed as a
`pandas.DataFrame` with string column names.

### `last_segment_coef_`

```
ndarray of shape (n_features_in_,)
```

Coefficient vector of an **unweighted** base-loss fit on the last detected
segment `X[changepoints_[-1]:]`. Used by `predict()`. When `num_chgpts_=0`
this is the coefficient from a single global fit on all training data.

### `last_segment_intercept_`

```
float or None
```

Intercept of the last-segment fit. `None` when `fit_intercept=False`.

---

## Internal attributes

These are set on all estimators but are primarily intended for research and
debugging.

### `_weights_`

```
ndarray of shape (num_signals_, n_samples)
```

Marginal prior weights `w[l, i]`. Row `l` contains the probability that
sample `i` belongs to signal `l` under a uniform prior over all valid
configurations with exactly `num_chgpts_` changepoints. Each column sums to 1
across rows.

### `_signal_coefs_`

```
ndarray of shape (num_signals_, n_features_in_)
```

Stage 1 coefficient vectors, one per signal. These are estimated under the
WERM marginal weights, not from plain regression on the final segments.
They are **not** the same as `last_segment_coef_`.

### `_signal_intercepts_`

```
ndarray of shape (num_signals_,) or None
```

Stage 1 intercepts, one per signal. `None` when `fit_intercept=False`.

### `_theta_hat_`

```
ndarray of shape (n_samples, n_features_in_)  [regression]
ndarray of shape (n_samples,)                 [logistic, binary probabilities]
```

In-sample predictions derived from the Stage 1 signal fits and the optimal
changepoint configuration. For regression estimators this is the design-matrix
projection; for the logistic estimator these are the linear predictors at the
optimal changepoint location.

---

## CV estimators only

### `best_num_chgpts_`

```
int
```

Number of changepoints selected by cross-validation.

### `best_index_`

```
int
```

Index into `num_chgpts_grid_` corresponding to `best_num_chgpts_`.

### `best_score_`

```
float
```

Mean held-out CV score for `best_num_chgpts_`. Lower is better (it is a
loss, not a metric).

### `cv_results_`

```
dict with keys:
  "num_chgpts"     : ndarray of shape (max_num_chgpts + 1,)
  "mean_test_score": ndarray of shape (max_num_chgpts + 1,)
```

Full cross-validation score grid. Useful for visualising the model-selection
curve:

```python
# ATTRS_TEST: cv_results
import numpy as np
from weightederm import WERMLeastSquaresCV

X = np.ones((80, 2))
y = np.concatenate([np.zeros(40), np.ones(40)])
model = WERMLeastSquaresCV(max_num_chgpts=3, cv=4, delta=5, fit_intercept=False).fit(X, y)

for k, s in zip(model.cv_results_["num_chgpts"], model.cv_results_["mean_test_score"]):
    print(f"  num_chgpts={k}: {s:.4f}")

assert model.best_num_chgpts_ in model.cv_results_["num_chgpts"]
```

### `num_chgpts_grid_`

```
ndarray of shape (max_num_chgpts + 1,)
```

The full candidate grid `[0, 1, …, max_num_chgpts]`.

### `segment_bounds_`

```
list of (int, int) tuples, length num_signals_
```

Half-open `[start, stop)` index pairs for each detected segment, derived from
the final refit's `changepoints_`. Equivalent to `[(0, cp0), (cp0, cp1), …, (cpL, n)]`.

### `segment_coefs_`

```
ndarray of shape (num_signals_, n_features_in_)
```

Per-segment coefficient vectors from an unweighted refit on each segment using
the CV scoring loss (`m_scorer`, absolute error by default). These differ from
`_signal_coefs_` (which are WERM-weighted) and from `last_segment_coef_` (which
uses the base loss and is used by `predict()`).

### `segment_intercepts_`

```
ndarray of shape (num_signals_,) or None
```

Per-segment intercepts from the same unweighted refit. `None` when
`fit_intercept=False`.

---

## Logistic estimators only

### `classes_`

```
ndarray of shape (2,)
```

The two unique class labels observed during `fit()`, in sorted order. The
second element `classes_[1]` is the positive class whose probability is
returned in `predict_proba(X)[:, 1]`.

### `n_iter_`

```
ndarray of shape (1,), dtype int
```

Reported number of iterations for sklearn compatibility (`max_iter` is stored
here since L-BFGS-B does not expose a true iteration count per signal).
