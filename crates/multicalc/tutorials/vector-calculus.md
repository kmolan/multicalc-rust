# Vector calculus

Curl and divergence via autodiff, plus line and flux integrals sampled along a curve.

- `curl_2d` / `curl_3d` and `divergence_2d` / `divergence_3d` take an explicit derivator (pass
  `AutoDiffMulti::default()` for exact results) and a `scalar_fn_vec!` field.
- `line_integral_2d` and `flux_integral_2d`, with their 3D and `_custom` forms, sample the field,
  so they take plain closures for the field and the parametric curve.

```rust
use multicalc::AutoDiffMulti;
use multicalc::constant;
use multicalc::scalar_fn_vec;
use multicalc::vector_field::{curl_2d, divergence_2d, flux_integral_2d, line_integral_2d};

// field (2xy, 3cos y)
let field = scalar_fn_vec!(|v: &[f64; 2]| [constant(2.0) * v[0] * v[1], constant(3.0) * v[1].cos()]);
let curl = curl_2d(AutoDiffMulti::default(), &field, &[1.0, 3.14]).unwrap();
let divergence = divergence_2d(AutoDiffMulti::default(), &field, &[1.0, 3.14]).unwrap();

// field (y, -x) along the unit circle (cos t, sin t)
let g: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&(|v: &[f64; 2]| v[1]), &(|v: &[f64; 2]| -v[0])];
let curve: [&dyn Fn(f64) -> f64; 2] = [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
let limit = [0.0, 2.0 * std::f64::consts::PI];
let line = line_integral_2d(&g, &curve, &limit).unwrap();   // -2*pi
let flux = flux_integral_2d(&g, &curve, &limit).unwrap();   //  0
```

The 3D curl is `(dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy)`.

Errors: the operators return [`DiffError`](error-handling.md) from differentiation, and the
integrals return [`IntegrateError`](error-handling.md). Full demo:
[vector_field.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/vector_field.rs).


---

[Back to the tutorial index](README.md)
