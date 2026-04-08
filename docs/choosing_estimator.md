# Choosing an Estimator

Use this page to decide which estimator class and parameter settings are
appropriate for your problem.

---

## Step 1 — What kind of response variable do you have?

```
Response is continuous  ──────────────────────────────┐
                                                       ▼
                              Are outliers likely or is noise heavy-tailed?
                              ├── No  → WERMLeastSquares / WERMLeastSquaresCV
                              └── Yes → WERMHuber / WERMHuberCV

Response is binary (0/1)  ──► WERMLogistic / WERMLogisticCV
```

### Least squares (`WERMLeastSquares`, `WERMLeastSquaresCV`)

- Minimises the sum of squared residuals within each signal.
- Optimal when noise is approximately Gaussian.
- Fast: the Stage 1 fits have a direct closed-form solution (`fit_solver="direct"`).
- Sensitive to outliers.

### Huber (`WERMHuber`, `WERMHuberCV`)

- Interpolates between squared loss (small residuals) and absolute loss (large
  residuals) via the `epsilon` parameter.
- Appropriate when the noise distribution is heavier-tailed than Gaussian or
  when occasional gross outliers are expected.
- Slightly slower than least squares (requires L-BFGS-B optimisation).
- The default `epsilon=1.35` gives 95 % Gaussian efficiency.

### Logistic (`WERMLogistic`, `WERMLogisticCV`)

- Minimises the binary cross-entropy within each signal.
- Use whenever `y` consists of two distinct class labels.
- Requires `penalty="l2"` (the default) when segments may be linearly separable.

---

## Step 2 — Do you know the number of changepoints?

```
Number of changepoints is known  ──► Fixed estimator  (WERMLeastSquares, etc.)
Number of changepoints is unknown ─► CV estimator     (WERMLeastSquaresCV, etc.)
```

### Fixed estimators

Set `num_chgpts` from domain knowledge (e.g. "one regulatory regime change",
"two known phase transitions"). The algorithm will always return exactly
`num_chgpts` changepoints.

```python
# CHOOSING_TEST: fixed_known
import numpy as np
from weightederm import WERMLeastSquares

X = np.ones((60, 1))
y = np.concatenate([np.zeros(30), np.ones(30)])
# We know there is exactly one changepoint
model = WERMLeastSquares(num_chgpts=1, delta=5, fit_intercept=False).fit(X, y)
assert model.num_chgpts_ == 1
```

### CV estimators

Use when the number of changepoints is unknown. Set `max_num_chgpts` to a
reasonable upper bound and let the CV procedure select from
`{0, 1, …, max_num_chgpts}`.

```python
# CHOOSING_TEST: cv_unknown
import numpy as np
from weightederm import WERMLeastSquaresCV

X = np.ones((90, 1))
y = np.concatenate([np.zeros(30), 2 * np.ones(30), np.zeros(30)])
# We do not know the number of changepoints; set an upper bound of 3
model = WERMLeastSquaresCV(max_num_chgpts=3, cv=3, delta=5, fit_intercept=False).fit(X, y)
assert model.best_num_chgpts_ == 2
```

---

## Step 3 — How should I set `delta`?

`delta` is the most practically important tuning parameter. Too small → spurious
changepoints; too large → missed changepoints.

| Situation | Recommended `delta` |
|-----------|---------------------|
| No prior knowledge | `max(1, n // 20)` |
| Minimum segment size known (e.g. 10 samples) | set `delta` to that minimum |
| Very large n (> 10 000) | `n // 50` to `n // 20` |
| Small n (< 100) | `3` to `max(1, n // 10)` |

---

## Step 4 — Should I use `penalty`?

| Estimator | Default | When to change |
|-----------|---------|----------------|
| `WERMLeastSquares` / `WERMHuber` | `"none"` | Add `"l2"` only if `p` is large relative to segment size or you want explicit shrinkage |
| `WERMLogistic` | `"l2"` (alpha=1.0) | Reduce `alpha` if segments are large and well-separated; set `penalty="none"` only if you are certain no segment is separable |

**L1 penalty** is available on all three losses but requires L-BFGS-B and is
generally slower. It can be useful for variable selection within segments in
high-dimensional settings.

---

## Step 5 — Which CV scoring loss should I use?

For `WERMLeastSquaresCV` and `WERMHuberCV`, the default held-out scoring loss is
**absolute error**, which is more robust for model selection than squared error.

Switch to the base loss with `use_base_loss_for_cv=True` when:

- The data is clean (Gaussian-ish noise, no outliers).
- You want the model-selection criterion to be consistent with the estimation
  criterion.
- You observe that the absolute-error CV selects a sub-optimal `num_chgpts`
  on held-out evaluation data.

---

## Quick-reference decision table

| Response | Outliers? | `num_chgpts` | Recommended estimator |
|----------|-----------|--------------|----------------------|
| Continuous | No | Known | `WERMLeastSquares` |
| Continuous | No | Unknown | `WERMLeastSquaresCV` |
| Continuous | Yes | Known | `WERMHuber` |
| Continuous | Yes | Unknown | `WERMHuberCV` |
| Binary | — | Known | `WERMLogistic` |
| Binary | — | Unknown | `WERMLogisticCV` |
