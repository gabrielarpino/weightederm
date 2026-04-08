# User Guide

## What `weightederm` does

`weightederm` detects **changepoints** in ordered (e.g. time-series) regression and
classification data. A changepoint is a sample index where the underlying model
parameters shift abruptly. The package estimates those indices under a **Weighted
Empirical Risk Minimization (WERM)** objective that marginalises over possible
changepoint configurations using a combinatorial prior on signal membership.

The result of `fit()` is always a set of estimated changepoint indices
(`changepoints_`), not a prediction rule for new data. `predict()` is provided for
convenience and forecasts using the **most recent detected segment** — see
[Prediction](#prediction) below.

---

## How the algorithm works

WERM changepoint detection proceeds in two stages.

### Stage 1 — signal fitting

For a candidate set of `L` changepoints the data is assigned to `L + 1` latent
**signals** (segments). Rather than making a hard assignment, each observation `i`
is associated with signal `l` via a marginal prior weight `w[l, i]` derived from a
uniform prior over all valid configurations with exactly `L` changepoints.  These
weights are computed analytically from combinatorial formulae and concentrate near
the true segment boundaries.

For each signal `l`, a weighted regression is solved using those weights:

```
minimise  sum_i  w[l, i] * loss(X[i] @ theta_l, y[i])   +   penalty(theta_l)
```

This produces `L + 1` coefficient vectors (`_signal_coefs_`) and optionally
`L + 1` intercepts (`_signal_intercepts_`).

### Stage 2 — changepoint search

The goodness-of-fit (GoF) score for a specific configuration `eta = (eta_1, ..., eta_L)`
is a weighted combination of segment-level losses evaluated at the Stage 1
coefficients. The optimal configuration

```
eta* = argmin  GoF(eta)
```

is found either by **brute-force** enumeration (exact, slow) or by an
**efficient** greedy + local-refinement search (fast, default). This gives
`changepoints_`.

---

## A minimal workflow

```python
# DOCS_TEST: minimal_workflow
import numpy as np
from weightederm import WERMLeastSquares

rng = np.random.default_rng(0)
n, p = 80, 4
true_cp = 40

X = rng.normal(size=(n, p))
y = np.empty(n)
y[:true_cp] = X[:true_cp] @ [2, -1, 0, 0] + 0.1 * rng.normal(size=true_cp)
y[true_cp:] = X[true_cp:] @ [-1, 2, 0, 0] + 0.1 * rng.normal(size=n - true_cp)

model = WERMLeastSquares(num_chgpts=1, delta=5, fit_intercept=False)
model.fit(X, y)

print("estimated changepoint:", model.changepoints_[0])
print("true changepoint:     ", true_cp)
assert abs(int(model.changepoints_[0]) - true_cp) <= 5
```

After `fit()`, `model.changepoints_` is a 0-indexed integer array of length
`num_chgpts`. The data is implicitly partitioned into the following segments:

| Segment | Samples |
|---------|---------|
| 0 | `[0, changepoints_[0])` |
| 1 | `[changepoints_[0], changepoints_[1])` |
| … | … |
| L | `[changepoints_[-1], n)` |

---

## Fixed vs CV estimators

### Fixed (`WERMLeastSquares`, `WERMHuber`, `WERMLogistic`)

Use when the number of changepoints is known from domain knowledge or theory.
Set `num_chgpts` directly.

```python
# DOCS_TEST: fixed_estimator
import numpy as np
from weightederm import WERMLeastSquares

X = np.ones((60, 2))
y = np.concatenate([np.zeros(30), np.ones(30)])
model = WERMLeastSquares(num_chgpts=1, delta=4, fit_intercept=False).fit(X, y)
assert model.num_chgpts_ == 1
assert len(model.changepoints_) == 1
```

### CV (`WERMLeastSquaresCV`, `WERMHuberCV`, `WERMLogisticCV`)

Use when the number of changepoints is unknown. The CV estimator:

1. Tries every `num_chgpts` in `{0, 1, …, max_num_chgpts}`.
2. For each candidate, runs `cv`-fold interleaved cross-validation.
3. Selects the candidate with the lowest mean held-out loss.
4. Refits the selected model on the full dataset.

```python
# DOCS_TEST: cv_estimator
import numpy as np
from weightederm import WERMLeastSquaresCV

X = np.ones((80, 2))
y = np.concatenate([np.zeros(40), np.ones(40)])
model = WERMLeastSquaresCV(max_num_chgpts=3, cv=4, delta=5, fit_intercept=False).fit(X, y)
assert model.best_num_chgpts_ == 1
assert "mean_test_score" in model.cv_results_
```

#### Reading CV results

```python
for k, score in zip(model.num_chgpts_grid_, model.cv_results_["mean_test_score"]):
    print(f"  num_chgpts={k}: mean CV loss = {score:.4f}")
```

#### CV scoring loss

By default, least-squares and Huber CV estimate held-out scores using **absolute
error** (more robust than squared error for model selection). To use the
estimator's base loss instead:

```python
from weightederm import WERMLeastSquaresCV
model = WERMLeastSquaresCV(max_num_chgpts=2, cv=3, use_base_loss_for_cv=True)
```

For the logistic CV estimator the base loss (logistic) is always used regardless
of this flag.

---

## Choosing `delta`

`delta` sets the **minimum gap** between adjacent candidate changepoints during
the Stage 2 search. It does **not** affect the Stage 1 signal weights.

A practical rule of thumb:

```
delta = max(1, int(0.05 * n))   # 5 % of the sample size
```

Setting `delta` too small allows the search to detect spurious micro-changepoints.
Setting it too large prevents the detection of closely-spaced true changepoints.
For the reference benchmark experiments a `delta` of roughly `n / (20 * num_signals)`
works well.

---

## Penalties

All six estimators accept `penalty` and `alpha` parameters. The penalty is applied
to the **coefficient vector only** (never the intercept), matching the sklearn
convention.

| `penalty` | Applied to Stage 1 fits | Default |
|-----------|-------------------------|---------|
| `"none"` | nothing | LS, Huber |
| `"l2"` | `alpha * ‖coef‖²` | Logistic |
| `"l1"` | `alpha * ‖coef‖₁` | — |

```python
# DOCS_TEST: penalty_usage
import numpy as np
from weightederm import WERMLeastSquares

X = np.ones((60, 3))
y = np.concatenate([np.zeros(30), np.ones(30)])
model_none = WERMLeastSquares(num_chgpts=1, delta=4, penalty="none", fit_intercept=False).fit(X, y)
model_l2   = WERMLeastSquares(num_chgpts=1, delta=4, penalty="l2", alpha=1.0, fit_intercept=False).fit(X, y)
assert model_none.changepoints_.shape == (1,)
assert model_l2.changepoints_.shape == (1,)
# L2 shrinks the last-segment coefficient toward zero
assert np.linalg.norm(model_l2.last_segment_coef_) <= np.linalg.norm(model_none.last_segment_coef_) + 1e-9
```

**Logistic default:** `penalty="l2"`, `alpha=1.0`. Unpenalised logistic fits can
diverge when a segment is linearly separable; the default L2 penalty prevents
this.

**L1 + direct solver:** combining `penalty="l1"` with `fit_solver="direct"` (the
default for `WERMLeastSquares`) triggers a `RuntimeWarning` and automatically
falls back to the L-BFGS-B solver.

---

## Prediction

`predict(X)` forecasts by fitting a **fresh, unweighted regression on the last
detected segment** using the estimator's base loss, then evaluating it on `X`.

The reasoning: after changepoints are detected, the most recent segment represents
the current regime. New observations are assumed to come from this regime.

```python
# DOCS_TEST: predict_usage
import numpy as np
from weightederm import WERMLeastSquares

X_train = np.ones((60, 2))
y_train = np.concatenate([np.zeros(30), np.ones(30)])
model = WERMLeastSquares(num_chgpts=1, delta=4, fit_intercept=False).fit(X_train, y_train)

X_new = np.ones((10, 2))
y_pred = model.predict(X_new)
assert y_pred.shape == (10,)
```

The last-segment coefficients are stored as `last_segment_coef_` and
`last_segment_intercept_` and are available immediately after `fit()`.

For logistic estimators, `predict(X)` returns class labels and `predict_proba(X)`
returns an `(n, 2)` probability matrix.

> **Note:** `predict()` on a changepoint detector has a different meaning than on
> a standard supervised model. The returned values are *forecasts under the
> assumption that the most recent regime continues* — not predictions derived from
> the full training-set structure.

---

## sklearn compatibility

All six estimators are sklearn-compatible:

- `get_params()` / `set_params()` / `clone()`
- Usable inside `sklearn.pipeline.Pipeline`
- Pass `sklearn.utils.estimator_checks.check_estimator()`

```python
# DOCS_TEST: sklearn_pipeline
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from weightederm import WERMLeastSquares

X = np.arange(1.0, 61.0).reshape(-1, 1)
y = np.concatenate([np.zeros(30), np.ones(30)])
pipe = Pipeline([("scale", StandardScaler()), ("model", WERMLeastSquares(num_chgpts=1, delta=4))])
pipe.fit(X, y)
assert hasattr(pipe.named_steps["model"], "changepoints_")
```
