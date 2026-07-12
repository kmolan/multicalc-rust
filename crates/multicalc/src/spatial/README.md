# spatial

Rotations, Lie groups and rigid-body transforms for 2D and 3D. Fixed-size, stack-allocated, no
panics, and generic over the `Numeric` scalar, so `f32`, `f64`, and the autodiff duals all work.

- [`Quaternion`](quaternion.rs) — Hamilton quaternion, stored scalar-first `[w, x, y, z]`: the raw
  algebra plus axis-angle / rotation-matrix / ZYX-Euler conversions, `slerp`, and `exp`/`ln`.
- [`SO2`](lie/so2.rs) / [`SE2`](lie/se2.rs) — 2D rotation and rigid-body transform.
- [`SO3`](lie/so3.rs) / [`SE3`](lie/se3.rs) — 3D rotation (wrapping a unit `Quaternion`, which
  carries the unit-rotation invariant) and rigid-body transform.

Every group provides `identity`, `compose` (also `*`), `inverse`, `act` on a point, `exp`/`log`,
`hat`/`vee`, `adjoint`, geodesic `interpolate`, and matrix conversions. Conventions: the tangent
ordering is `[v; ω]` (linear part first) for `SE2`/`SE3`; the retract is right-perturbation
`X · exp(ξ)`; angles are radians. `exp`/`log` Taylor-continue near θ = 0 so derivatives stay finite
at rest.

```rust
use multicalc::spatial::{SE3, SO3};
use multicalc::linear_algebra::Vector;

// A 90° rotation about z, applied to a point.
let r = SO3::<f64>::exp(Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]));
let p = r.act(Vector::new([1.0, 0.0, 0.0]));         // ≈ (0, 1, 0)

// A rigid transform: rotate, then translate.
let g = SE3::from_parts(r, Vector::new([1.0, 2.0, 3.0]));
let q = g.act(Vector::new([1.0, 0.0, 0.0]));         // ≈ (1, 3, 3)

// exp/log round trip on the tangent twist [v; ω].
let xi = g.log();
let g2 = SE3::exp(xi);
```
