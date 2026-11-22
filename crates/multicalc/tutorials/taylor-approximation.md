# Taylor approximation

Local Taylor models of a function around a point (linear and quadratic) with goodness-of-fit
metrics.

- `LinearApproximator`: first-order model.
- `QuadraticApproximator`: same API, also captures curvature.
- `approximate` builds the model; `predict` evaluates it; `prediction_metrics` returns MAE, MSE,
  RMSE, R², and adjusted R² against sample points.
- Metrics use pairwise summation by default; chain `.with_kahan_summation()` to opt into Kahan.

```rust
use multicalc::LinearApproximator;
use multicalc::scalar_fn;

let f = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] * v[1] + v[2] * v[2] * v[2]);
let base_point = [1.0, 2.0, 3.0];        // where the model is anchored
let linear: LinearApproximator = LinearApproximator::default();
let model = linear.approximate(&f, &base_point).unwrap();

let nearby = [1.1, 2.1, 3.1];
let y = model.predict(&nearby);
// model.prediction_metrics(&samples, &f) returns RMSE, R^2, and more
```

`QuadraticApproximator` works the same way and captures curvature as well.

Errors: the underlying derivatives return [`DiffError`](error-handling.md). Full demo:
[approximation.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/approximation.rs).


---

[Back to the tutorial index](README.md)
