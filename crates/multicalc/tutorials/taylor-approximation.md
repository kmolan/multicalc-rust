# Taylor approximation

Linear and quadratic Taylor models of a multi-variable function around a point, with
goodness-of-fit metrics.

- `LinearApproximator`: `f(x) ≈ f(a) + ∇f(a)·(x − a)`.
- `QuadraticApproximator`: the same API, plus the Hessian term `½ (x − a)ᵀ H(a) (x − a)`.
- `approximate` builds the model from the derivator's partials (`AutoDiffMulti` by default).
  `predict` evaluates it.
- `prediction_metrics(&points, &f)` returns the mean absolute error, MSE, RMSE, R² and adjusted R²
  against `f` at the sample points. R² is `NaN` if `f` is constant over the points. Adjusted R²
  is `NaN` if there are too few points.
- The metrics use pairwise summation by default. Chain `.with_kahan_summation()` on the
  approximator for Kahan summation.

```rust
use multicalc::LinearApproximator;
use multicalc::scalar_fn;

let f = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] * v[1] + v[2] * v[2] * v[2]);
let base_point = [1.0, 2.0, 3.0];        // where the model is anchored
let linear: LinearApproximator = LinearApproximator::default();
let model = linear.approximate(&f, &base_point).unwrap();

let nearby = [1.1, 2.1, 3.1];
let y = model.predict(&nearby);          // 35.2, against f = 35.301
assert!((y - 35.2).abs() < 1e-12);

let samples = [[1.1, 2.1, 3.1], [0.9, 1.9, 2.9], [1.0, 2.2, 3.0]];
let metrics = model.prediction_metrics(&samples, &f);
let rmse = metrics.root_mean_squared_error;
```

Errors: `approximate` returns [`DiffError`](error-handling.md) from the derivator. Full demo:
[approximation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/approximation.rs).


---

[Back to the tutorial index](README.md)
