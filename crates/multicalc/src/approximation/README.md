# approximation

Local Taylor models of a function around a point — linear and quadratic — with goodness-of-fit
metrics.

- [`linear_approximation::LinearApproximator`](linear_approximation.rs) — first-order model.
- [`quadratic_approximation::QuadraticApproximator`](quadratic_approximation.rs) — same API, also
  captures curvature.
- `get` builds the model; `predict` evaluates it; `get_prediction_metrics` returns MAE, MSE, RMSE,
  R², and adjusted R² against sample points.
- Metrics use pairwise summation by default; chain `.with_kahan_summation()` to opt into Kahan.

```rust
use multicalc::approximation::linear_approximation::LinearApproximator;
use multicalc::scalar_fn;

let f = scalar_fn!(|v: &[f64; 3]| v[0] + v[1] * v[1] + v[2] * v[2] * v[2]);
let linear: LinearApproximator = LinearApproximator::default();
let model = linear.get(&f, &[1.0, 2.0, 3.0]).unwrap();

let y = model.predict(&[1.1, 2.1, 3.1]);
// model.get_prediction_metrics(&samples, &f) returns RMSE, R^2, and more
```
