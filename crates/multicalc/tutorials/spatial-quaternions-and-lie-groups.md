# Spatial: quaternions and Lie groups

Rotations, Lie groups, and rigid-body transforms for 2D and 3D. Fixed-size, stack-allocated, no
panics, and generic over the `Numeric` scalar, so `f32`, `f64`, and the autodiff duals all work.

- `Quaternion`: Hamilton quaternion, stored scalar-first `[w, x, y, z]`: the raw algebra plus
  axis-angle / rotation-matrix / ZYX-Euler conversions, `slerp`, and `exp`/`ln`.
- `SO2` / `SE2`: 2D rotation and rigid-body transform.
- `SO3` / `SE3`: 3D rotation (wrapping a unit `Quaternion`, which carries the unit-rotation
  invariant) and rigid-body transform.
- `Twist` / `Wrench`: typed spatial velocity and force in the linear-first `[v; ω]` /
  `[force; torque]` ordering.

Every group provides `identity`, `compose` (also `*`), `inverse`, `act` on a point, `exp`/`log`,
`hat`/`vee`, `adjoint`, geodesic `interpolate`, and matrix conversions. Conventions: the tangent
ordering is `[v; ω]` (linear part first) for `SE2`/`SE3`; the retract is right-perturbation
`X · exp(ξ)`; angles are radians. `exp`/`log` Taylor-continue near θ = 0 so derivatives stay
finite at rest.

```rust
use multicalc::{SE3, SO3};
use multicalc::Vector;

// A 90° rotation about z, applied to a point.
let quarter_turn_about_z = Vector::new([0.0, 0.0, core::f64::consts::FRAC_PI_2]);
let r = SO3::<f64>::exp(quarter_turn_about_z);

let point = Vector::new([1.0, 0.0, 0.0]);
let p = r.act(point);                                // ≈ (0, 1, 0)

// A rigid transform: rotate, then translate.
let translation = Vector::new([1.0, 2.0, 3.0]);
let g = SE3::from_parts(r, translation);
let q = g.act(point);                                // ≈ (1, 3, 3)

// exp/log round trip on the tangent twist [v; ω].
let xi = g.log();
let g2 = SE3::exp(xi);
```

`SO3::from_two_direction_pairs` reads an orientation straight off two directions the body can see.
Standing still, a drone's push sensor says which way is down and its compass says roughly which way
is north; both directions are also known in world terms, and two directions that are not parallel
fix the orientation completely. The first pair is trusted exactly and the second only settles the
leftover spin about it, so the noisier reading belongs second. It returns `None` when a direction has
no length or when the pair are parallel. This is how an error-state filter gets its starting
attitude — and it only works while the body is still, because a body in free fall feels no push at
all and has no "down" to read.

Because everything is generic over the scalar, a derivative with respect to a joint angle or
pose parameter flows through `act`, `compose`, and `exp`/`log` under autodiff. That is what the
inverse-kinematics showcases are built on. Full demo:
[lie_groups.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/lie_groups.rs);
worked application:
[3d_arm_ik.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/3d_arm_ik.rs).


---

[Back to the tutorial index](README.md)
