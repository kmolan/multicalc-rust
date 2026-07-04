# vector_field

Vector calculus: curl and divergence via autodiff, plus line and flux integrals sampled along a
curve.

- [`curl`](curl.rs)`::{get_2d, get_3d}` and [`divergence`](divergence.rs)`::{get_2d, get_3d}` take an
  explicit derivator — pass `AutoDiffMulti::default()` for exact results — and a `scalar_fn_vec!`
  field.
- [`line_integral`](line_integral.rs) and [`flux_integral`](flux_integral.rs) sample the field, so
  they take plain closures for the field and the parametric curve.

```rust
use multicalc::numerical_derivative::autodiff::AutoDiffMulti;
use multicalc::scalar::c;
use multicalc::scalar_fn_vec;
use multicalc::vector_field::{curl, divergence, line_integral, flux_integral};

// field (2xy, 3cos y)
let field = scalar_fn_vec!(|v: &[f64; 2]| [c(2.0) * v[0] * v[1], c(3.0) * v[1].cos()]);
let curl_2d = curl::get_2d(AutoDiffMulti::default(), &field, &[1.0, 3.14]).unwrap();
let div_2d = divergence::get_2d(AutoDiffMulti::default(), &field, &[1.0, 3.14]).unwrap();

// field (y, -x) along the unit circle (cos t, sin t)
let g: [&dyn Fn(&[f64; 2]) -> f64; 2] = [&(|v: &[f64; 2]| v[1]), &(|v: &[f64; 2]| -v[0])];
let curve: [&dyn Fn(f64) -> f64; 2] = [&(|t: f64| t.cos()), &(|t: f64| t.sin())];
let limit = [0.0, 2.0 * std::f64::consts::PI];
let line = line_integral::get_2d(&g, &curve, &limit).unwrap();   // -2*pi
let flux = flux_integral::get_2d(&g, &curve, &limit).unwrap();   //  0
```

Credits: 3D curl `(dVz/dy - dVy/dz, dVx/dz - dVz/dx, dVy/dx - dVx/dy)` — see [`curl.rs`](curl.rs).
