# Parameter Reference

Complete parameter descriptions with practical guidance for all six estimators.

Parameters are grouped by role. All estimators share the [Common parameters](#common-parameters);
additional parameters are listed per estimator class.

---

## Common parameters

These appear on all six estimators.

### `delta` — minimum changepoint spacing

| | |
|---|---|
| Type | `int` |
| Default | `1` |
| Applies to | all estimators |

Minimum number of samples between adjacent candidate changepoints during the
Stage 2 search. A value of `1` imposes no spacing constraint.

**How to choose:**

- Rule of thumb: `delta = max(1, int(0.05 * n))` (5 % of sample size).
- For very large `n` with few expected changepoints, `delta` can be set to
  `n // (4 * max_expected_num_signals)`.
- Setting `delta` too small permits spurious micro-changepoints. Setting it
  too large prevents detecting closely-spaced true changepoints.
- `delta` **does not** affect the Stage 1 marginal prior weights.

### `search_method` — second-stage search strategy

| | |
|---|---|
| Type | `{"efficient", "brute_force"}` |
| Default | `"efficient"` |
| Applies to | all estimators |

- `"efficient"` — greedy forward search followed by local-refinement moves.
  Scales well with `n` and `num_chgpts`. Recommended for all practical use.
- `"brute_force"` — evaluates every valid configuration. Exact but
  `O(n^num_chgpts)`. Use only for small problems or to verify `"efficient"` on
  toy data.

### `fit_intercept` — include an intercept term

| | |
|---|---|
| Type | `bool` |
| Default | `True` |
| Applies to | all estimators |

When `True`, each signal model includes a free intercept. Set to `False`
when the covariates already include a constant column or theory dictates
signals passing through the origin.

### `penalty` — regularisation type

| | |
|---|---|
| Type | `{"none", "l1", "l2"}` |
| Default | `"none"` for LS / Huber; `"l2"` for Logistic |
| Applies to | all estimators |

Applied to the coefficient vector in every Stage 1 signal fit. Never applied to
the intercept.

- `"none"` — no regularisation. Default for `WERMLeastSquares` and `WERMHuber`.
- `"l2"` — ridge penalty `alpha * ‖coef‖²`. Default for `WERMLogistic` to avoid
  divergence on separable segments.
- `"l1"` — lasso penalty `alpha * ‖coef‖₁`. Uses L-BFGS-B regardless of
  `fit_solver`; a `RuntimeWarning` is emitted if `fit_solver="direct"` was
  requested.

### `alpha` — regularisation strength

| | |
|---|---|
| Type | `float` |
| Default | `0.0` for LS / Huber; `1.0` for Logistic |
| Applies to | all estimators |

Ignored when `penalty="none"`. Larger values produce stronger shrinkage. The
scale is the same as in sklearn: `alpha` multiplies the penalty term directly,
so sensible values depend on the scale of `X` and `y`.

---

## `WERMLeastSquares`

```
WERMLeastSquares(num_chgpts, *, delta=1, search_method="efficient",
                 fit_intercept=True, fit_solver="direct",
                 penalty="none", alpha=0.0)
```

### `num_chgpts`

| | |
|---|---|
| Type | `int ≥ 0` |
| Required | yes |

Number of changepoints to detect. `num_chgpts=0` fits a single global model
with no changepoints.

### `fit_solver` — Stage 1 solver

| | |
|---|---|
| Type | `{"direct", "lbfgsb"}` |
| Default | `"direct"` |

- `"direct"` — solves the weighted normal equations via `scipy.linalg.lstsq`.
  Fast and exact for L2-penalised or unpenalised problems. Automatically
  augments the normal equations for L2 regularisation.
- `"lbfgsb"` — gradient-based optimisation via L-BFGS-B. Useful for very
  large `p` where the direct solve is slow, or when `penalty="l1"` is used.

---

## `WERMHuber`

```
WERMHuber(num_chgpts, *, delta=1, search_method="efficient",
          fit_intercept=True, epsilon=1.35, max_iter=100, tol=1e-5,
          penalty="none", alpha=0.0)
```

### `num_chgpts`

Same as `WERMLeastSquares.num_chgpts`.

### `epsilon` — Huber transition parameter

| | |
|---|---|
| Type | `float > 1.0` |
| Default | `1.35` |

Controls the boundary between the quadratic (small residuals) and linear
(large residuals) regions of the Huber loss. The sklearn default `1.35`
corresponds to 95 % efficiency under a standard normal error distribution.
Reduce toward `1.0` for heavier-tailed data (more robustness, less efficiency);
increase toward `∞` to approach squared loss.

### `max_iter` / `tol` — optimiser convergence

| | |
|---|---|
| `max_iter` | `int`, default `100` |
| `tol` | `float`, default `1e-5` |

Maximum number of L-BFGS-B iterations and gradient-norm tolerance for the
Stage 1 signal fits. Increase `max_iter` or tighten `tol` if you see
convergence warnings.

---

## `WERMLogistic`

```
WERMLogistic(num_chgpts, *, delta=1, search_method="efficient",
             fit_intercept=True, max_iter=100, tol=1e-5,
             penalty="l2", alpha=1.0)
```

### `num_chgpts`

Same as `WERMLeastSquares.num_chgpts`.

### `max_iter` / `tol`

Same as `WERMHuber`. Logistic fits often need more iterations on hard or
high-dimensional problems; consider `max_iter=300` or more.

### `penalty` / `alpha`

The **default** is `penalty="l2"`, `alpha=1.0`. This matches sklearn's
`LogisticRegression` default and prevents numerical divergence when a segment
is perfectly linearly separable. Setting `penalty="none"` is safe when segments
are not separable, but will raise a `ValueError` if separability is detected.

---

## `WERMLeastSquaresCV`

```
WERMLeastSquaresCV(max_num_chgpts, *, delta=1, search_method="efficient",
                   cv=5, fit_intercept=True, m_scorer=None,
                   use_base_loss_for_cv=False, penalty="none", alpha=0.0)
```

### `max_num_chgpts`

| | |
|---|---|
| Type | `int ≥ 0` |
| Required | yes |

Upper bound of the CV search grid `{0, 1, …, max_num_chgpts}`. Set this to
the maximum plausible number of changepoints; larger values increase compute
time linearly.

### `cv` — number of folds

| | |
|---|---|
| Type | `int ≥ 2` |
| Default | `5` |

Number of interleaved (systematic) cross-validation folds. Interleaved folds
preserve temporal ordering, unlike random splits. Use `cv=3` for small samples
(`n < 60`).

### `m_scorer` — custom CV scoring loss

| | |
|---|---|
| Type | `callable(predictions, targets) → ndarray` or `None` |
| Default | `None` |

Custom held-out loss function. If `None`, absolute error is used (unless
`use_base_loss_for_cv=True`). Signature must match
`loss(predictions: ndarray, targets: ndarray) → ndarray` returning
per-sample losses.

### `use_base_loss_for_cv`

| | |
|---|---|
| Type | `bool` |
| Default | `False` |

When `True`, the segment fits and held-out scoring during CV both use the
**squared loss** (the base loss of this estimator) instead of absolute error.
Absolute error is the default because it is more robust for model selection
under outliers or non-Gaussian noise, but squared-loss CV may give sharper
selection in clean settings.

### `penalty` / `alpha`

Passed through to the inner `WERMLeastSquares` final refit. CV does **not**
tune over penalty or alpha.

---

## `WERMHuberCV`

```
WERMHuberCV(max_num_chgpts, *, delta=1, search_method="efficient",
            cv=5, fit_intercept=True, epsilon=1.35, max_iter=100,
            tol=1e-5, m_scorer=None, use_base_loss_for_cv=False,
            penalty="none", alpha=0.0)
```

All parameters are as described above. `use_base_loss_for_cv=True` switches
CV scoring to Huber loss (using the same `epsilon`).

---

## `WERMLogisticCV`

```
WERMLogisticCV(max_num_chgpts, *, delta=1, search_method="efficient",
               cv=5, fit_intercept=True, max_iter=100, tol=1e-5,
               m_scorer=None, penalty="l2", alpha=1.0)
```

`WERMLogisticCV` always uses logistic loss for CV scoring regardless of any
`use_base_loss_for_cv` flag (logistic loss *is* the base loss). The default
`penalty="l2"`, `alpha=1.0` is passed through to each inner `WERMLogistic`
fold fit and the final refit.
