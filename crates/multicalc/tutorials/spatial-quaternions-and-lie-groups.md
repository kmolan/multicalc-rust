# Spatial: quaternions and Lie groups

Rotations, Lie groups, and rigid-body transforms for 2D and 3D. Fixed-size, stack-allocated, no
panics, and generic over the `Numeric` scalar, so `f32`, `f64`, and the autodiff duals all work.

- `Quaternion`: Hamilton quaternion, stored scalar-first `[w, x, y, z]`: the raw algebra plus
  axis-angle / rotation-matrix / ZYX-Euler conversions, `slerp`, and `exp`/`log`.
- `SO2` / `SE2`: 2D rotation and rigid-body transform.
- `SO3` / `SE3`: 3D rotation (wrapping a unit `Quaternion`, which carries the unit-rotation
  invariant) and rigid-body transform.
- `Twist` / `Wrench`: typed spatial velocity and force in the linear-first `[v; ω]` /
  `[force; torque]` ordering, and the algebra that goes with them — see below. Both convert from
  and to `[T; 6]` and `Vector6D`, and `*` by a scalar scales both parts. `Quaternion` converts from
  and to `[w, x, y, z]`.
- `SpatialInertia`: mass, center of mass `c`, and rotational inertia `I_c` about `c`, with the
  spatial products below.

Every group provides `identity`, `compose` (also `*`), `inverse`, `act` on a point, `exp`/`log`,
`hat`/`vee`, `adjoint`, geodesic `interpolate`, left and right Jacobians, and matrix conversions.
Conventions:

- The `SE2`/`SE3` tangent order is `[v; ω]`, linear part first.
- The retraction is right-perturbation `X · exp(ξ)`.
- Angles are in radians.
- `exp`/`log` switch to Taylor series near θ = 0, so derivatives stay finite at θ = 0.

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
let twist = g.log();
let g2 = SE3::exp(twist);
```

`SO3::from_two_direction_pairs` gives the orientation from two direction pairs, each observed in
the body frame and known in the world frame. Two non-parallel directions fix the orientation.

- The orientation maps the primary observed direction onto its reference exactly.
- The secondary pair sets only the remaining rotation about the primary direction, so put the
  noisier reading second.
- Only the directions are used; the lengths do not matter.
- It returns `None` if a direction has zero or non-finite length, or if the two observed
  directions or the two reference directions are parallel.

An error-state filter uses it for the initial attitude from accelerometer and magnetometer
readings. The accelerometer gives the gravity direction only at rest; in free fall the specific
force is zero.

All types are generic over the scalar, so `act`, `compose` and `exp`/`log` propagate autodiff
derivatives with respect to joint angles or pose parameters. The inverse-kinematics showcases use
this. Demo:
[lie_groups.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/lie_groups.rs);
application:
[3d_arm_ik.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/showcase/3d_arm_ik.rs).

## Spatial algebra

| Operation | Formula or meaning |
|---|---|
| `SE3::act_twist` / `act_wrench` | Frame change of a twist / wrench; `inverse_act_twist` / `inverse_act_wrench` for the inverse. No 6×6 is formed. |
| `SE3::act_inertia` / `inverse_act_inertia` | Frame change of a `SpatialInertia`. |
| `SE3::adjoint` / `force_adjoint` | The 6×6 motion and force adjoints. |
| `Twist::cross` | Motion cross product `v × v'`: `[ω×v' + v×ω' ; ω×ω']`. |
| `Twist::cross_wrench` | Force cross product `v ×* f`: `[ω×f ; ω×τ + v×f]`. |
| `Twist::dot_wrench` | Power `v·f + ω·τ`. |
| `SpatialInertia::momentum` | Spatial momentum `I·v` about the body-frame origin. |
| `SpatialInertia::bias_wrench` | The velocity-product term `v ×* (I·v)`, as RNEA needs it. |
| `SpatialInertia::kinetic_energy` | `½·vᵀ·I·v`. |
| `SpatialInertia::combined` | The inertia of two rigidly attached bodies. |

```rust
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3, SpatialInertia, Twist};

// A 3 kg body with its center of mass 10 cm above its origin.
let body = SpatialInertia::from_diagonal_inertia(
    3.0_f64,
    Vector::new([0.0, 0.0, 0.1]),
    Vector::new([0.05, 0.05, 0.08]),
)?;

// Translation along x, rotation about z.
let motion = Twist::new(Vector::new([1.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 2.0]));
let doubled = motion * 2.0;
assert_eq!(doubled, motion.scale(2.0));
let flat: [f64; 6] = motion.into();
assert_eq!(flat, [1.0, 0.0, 0.0, 0.0, 0.0, 2.0]);

let carried = body.momentum(motion);
let energy = body.kinetic_energy(motion);

// The same body and twist in a frame 0.5 m along x.
let elsewhere = SE3::from_parts(SO3::<f64>::identity(), Vector::new([0.5, 0.0, 0.0]));
let moved_body = elsewhere.act_inertia(body);
let moved_motion = elsewhere.act_twist(motion);

// Kinetic energy is frame-invariant.
assert!((moved_body.kinetic_energy(moved_motion) - energy).abs() < 1e-12);

// Momentum transforms as a wrench.
let moved_momentum = elsewhere.act_wrench(carried);
assert!((moved_momentum.to_vector() - moved_body.momentum(moved_motion).to_vector()).norm() < 1e-12);

// Two rigidly attached bodies.
let whole = body.combined(moved_body);
assert!((whole.mass() - 6.0).abs() < 1e-12);
# Ok::<(), multicalc::CalcError>(())
```

---

[Back to the tutorial index](README.md)
