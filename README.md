# 𐄷 Weighted ERM - fast & accurate change point regression

<p align="center">
  <img src="assets/raw_marginal_plot_5.svg" alt="weightederm" style="width: 100%; max-width: 800px; height: auto;">
</p>

<p align="center">
  <a href="https://doi.org/10.5281/zenodo.19462775">
    <img src="https://zenodo.org/badge/1204283258.svg" alt="DOI">
  </a>
</p>

`weightederm` is a scikit-learn-style package for fast and accurate (offline) change point detection and estimation in regression settings via weighted empirical risk minimization (WERM).
This is a robust, flexible implementation of the original research code (https://github.com/gabrielarpino/WeightedERM-reference.git) that is better suited for practical use. 

It currently provides fixed- and CV-based estimators for:

| Loss | Fixed `num_chgpts` | Unknown `num_chgpts` via CV |
| --- | --- | --- |
| Least squares | `WERMLeastSquares` | `WERMLeastSquaresCV` |
| Huber | `WERMHuber` | `WERMHuberCV` |
| Logistic (binary) | `WERMLogistic` | `WERMLogisticCV` |

## Installation

### Using pip

```bash
pip install weightederm
```

or

```bash
pip install git+https://github.com/gabrielarpino/weightederm.git
```

### From root after cloning

Install from the repository root with standard Python packaging tools:

```bash
pip install .
```

For editable local development:

```bash
pip install -e .
```

If you are using `uv`, the equivalent setup command is:

```bash
uv sync
```

This repository is already source-installable: building with `uv build` produces both an sdist and a wheel.

## Documentation

| Page | Contents |
|------|----------|
| [User Guide](docs/user_guide.md) | Algorithm overview, fixed vs CV workflow, penalties, `predict()` semantics, sklearn integration |
| [Choosing an Estimator](docs/choosing_estimator.md) | Decision guide: which loss, fixed vs CV, `delta`, `penalty` |
| [Parameter Reference](docs/parameter_reference.md) | Every parameter for every class with practical guidance |
| [Fitted Attributes](docs/fitted_attributes.md) | Every post-`fit()` attribute explained |

## Quick Notes

- Observations are assumed to be ordered.
- `num_signals = num_chgpts + 1`.
- `delta` constrains the changepoint search only; it does not currently change the prior weights.
- CV defaults:
  - least squares / Huber: absolute-loss model selection
  - logistic: logistic-loss model selection
- Set `use_base_loss_for_cv=True` on a CV estimator to score folds with the estimator's base loss instead.
- Unpenalized logistic fits can fail on separable data. In that case `weightederm` raises rather than silently regularizing.

## Minimum Working Examples

These examples are intentionally small, deterministic, and tested. They are not full reproductions of the benchmark notebook, but they follow the same spirit as minimal M1/M2/M3-style runs.

### M1-style: fixed number of changepoints, sparse linear regression

```python
# README_TEST_M1
import numpy as np
from weightederm import WERMLeastSquares

rng = np.random.default_rng(0)
n, p = 120, 20
true_cp = 60

X = rng.normal(size=(n, p))
beta_left = np.zeros(p)
beta_left[[0, 3]] = [2.0, -1.5]
beta_right = np.zeros(p)
beta_right[[0, 3]] = [-1.0, 2.5]

y = np.empty(n)
y[:true_cp] = X[:true_cp] @ beta_left + 0.2 * rng.normal(size=true_cp)
y[true_cp:] = X[true_cp:] @ beta_right + 0.2 * rng.normal(size=n - true_cp)

model = WERMLeastSquares(
    num_chgpts=1,
    delta=5,
    search_method="efficient",
    fit_intercept=False,
)
model.fit(X, y)

print("true changepoint:", true_cp)
print("estimated changepoint:", model.changepoints_[0])
assert abs(int(model.changepoints_[0]) - true_cp) <= 5
```

### M2-style: unknown number of changepoints, sparse linear regression with CV

```python
# README_TEST_M2
import numpy as np
from weightederm import WERMLeastSquaresCV

rng = np.random.default_rng(1)
n, p = 180, 10
true_cps = np.array([60, 120])

X = rng.normal(size=(n, p))
beta_1 = np.zeros(p)
beta_1[[0]] = [3.5]
beta_2 = np.zeros(p)
beta_2[[0, 1]] = [-3.0, 3.0]
beta_3 = np.zeros(p)
beta_3[[0, 1, 2]] = [2.5, -2.5, 2.5]

y = np.empty(n)
y[:60] = X[:60] @ beta_1 + 0.05 * rng.normal(size=60)
y[60:120] = X[60:120] @ beta_2 + 0.05 * rng.normal(size=60)
y[120:] = X[120:] @ beta_3 + 0.05 * rng.normal(size=60)

model = WERMLeastSquaresCV(
    max_num_chgpts=2,
    delta=5,
    cv=3,
    search_method="efficient",
    fit_intercept=False,
)
model.fit(X, y)

print("true changepoints:", true_cps.tolist())
print("selected num_chgpts:", model.best_num_chgpts_)
print("estimated changepoints:", model.changepoints_.tolist())
assert model.best_num_chgpts_ == 2
assert np.max(np.abs(model.changepoints_ - true_cps)) <= 5
```

### M3-style: fixed number of changepoints, sparse logistic regression

```python
# README_TEST_M3
import numpy as np
from weightederm import WERMLogistic

rng = np.random.default_rng(2)
n, p = 160, 12
true_cp = 80

X = rng.normal(size=(n, p))
beta_left = np.zeros(p)
beta_left[[0, 2]] = [2.5, -2.0]
beta_right = np.zeros(p)
beta_right[[0, 2]] = [-2.5, 2.0]

eta = np.empty(n)
eta[:true_cp] = X[:true_cp] @ beta_left
eta[true_cp:] = X[true_cp:] @ beta_right
prob = 1.0 / (1.0 + np.exp(-eta))
y = rng.binomial(1, prob)

model = WERMLogistic(
    num_chgpts=1,
    delta=5,
    search_method="efficient",
    fit_intercept=False,
    max_iter=300,
    tol=1e-6,
)
model.fit(X, y)

print("true changepoint:", true_cp)
print("estimated changepoint:", model.changepoints_[0])
assert abs(int(model.changepoints_[0]) - true_cp) <= 5
```

For fuller benchmark-style experiments, including normalized Hausdorff summaries and optional McScan comparison, see [notebooks/reference_like_m123_benchmarks.ipynb](notebooks/reference_like_m123_benchmarks.ipynb).
