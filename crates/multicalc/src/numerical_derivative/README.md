# numerical_derivative

Derivatives of any order, total and partial — exact via forward-mode autodiff, or by finite
differences for black-box functions — plus Jacobian and Hessian matrices.

- [`autodiff`](autodiff.rs)`::{AutoDiffSingle, AutoDiffMulti}` — exact derivatives.
- [`finite_difference`](finite_difference.rs)`::{FiniteDifferenceSingle, FiniteDifferenceMulti}` — for
  functions you cannot author with `scalar_fn!`.
- Both implement the [`derivator`](derivator.rs)`::DerivatorSingleVariable` / `DerivatorMultiVariable`
  traits (`get`, `get_single`, `get_double`, `get_single_partial`).
- [`jacobian::Jacobian`](jacobian.rs) and [`hessian::Hessian`](hessian.rs) build the matrices.

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::numerical_derivative::derivator::DerivatorMultiVariable;
use multicalc::scalar_fn;

// g(x, y, z) = y*sin(x) + x*cos(y) + x*y*e^z; order = number of indices passed
let g = scalar_fn!(|v: &[f64; 3]| v[1] * v[0].sin() + v[0] * v[1].cos() + v[0] * v[1] * v[2].exp());
let d = AutoDiffMulti::default();
let point = [1.0, 2.0, 3.0];

let dx    = d.get_single_partial(&g, 0, &point).unwrap();  // dg/dx
let mixed = d.get(&g, &[0, 1], &point).unwrap();           // d(dg/dx)/dy
```

The Jacobian and Hessian take a `scalar_fn_vec!` / `scalar_fn!` function; see the root
[README](../../README.md) for those snippets.
