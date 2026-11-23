# Vector calculus

Curl and divergence of a vector field, and line and flux integrals along a parametric curve.

- `curl_2d` / `curl_3d` and `divergence_2d` / `divergence_3d` take a derivator and a
  `scalar_fn_vec!` field. Pass `AutoDiffMulti::default()` for exact results.
- `line_integral_2d` / `line_integral_3d` and `flux_integral_2d` / `flux_integral_3d` sample the
  field, so they take plain closures for the field components and the curve `t ↦ (x(t), y(t))`.
  The `_custom` forms take the iteration count. `line_integral_partial_2d` / `_3d` integrate one
  component.

```rust
use multicalc::AutoDiffMulti;
use multicalc::constant;
use multicalc::scalar_fn_vec;
use multicalc::vector_field::{curl_2d, divergence_2d, flux_integral_2d, line_integral_2d};

// field (2xy, 3cos y)
let field = scalar_fn_vec!(|v: &[f64; 2]| [constant(2.0) * v[0] * v[1], constant(3.0) * v[1].cos()]);
let curl = curl_2d(AutoDiffMulti::default(), &field, &[1.0, 3.14]).unwrap();             // -2x = -2
let divergence = divergence_2d(AutoDiffMulti::default(), &field, &[1.0, 3.14]).unwrap(); // 2y - 3sin y

// field (y, -x) along the unit circle (cos t, sin t)
let g: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&(|v: &[f64; 2]| v[1]), &(|v: &[f64; 2]| -v[0])];
let curve: [&dyn Fn(f64) -> f64; 2] = [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
let limit = [0.0, 2.0 * std::f64::consts::PI];
let line = line_integral_2d(&g, &curve, &limit).unwrap();   // -2*pi
let flux = flux_integral_2d(&g, &curve, &limit).unwrap();   //  0
```

`curl_2d` returns `∂Vy/∂x − ∂Vx/∂y`. `curl_3d` returns
`(∂Vz/∂y − ∂Vy/∂z, ∂Vx/∂z − ∂Vz/∂x, ∂Vy/∂x − ∂Vx/∂y)`.

Errors: the operators return [`DiffError`](error-handling.md) from differentiation, and the
integrals return [`IntegrateError`](error-handling.md). Full demo:
[vector_field.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/vector_field.rs).


---

[Back to the tutorial index](README.md)
