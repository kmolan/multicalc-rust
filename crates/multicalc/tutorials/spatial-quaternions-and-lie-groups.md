# Spatial: quaternions and Lie groups

Rotations, Lie groups, and rigid-body transforms for 2D and 3D. Fixed-size, stack-allocated, no
panics, and generic over the `Numeric` scalar, so `f32`, `f64`, and the autodiff duals all work.

- `Quaternion`: Hamilton quaternion, stored scalar-first `[w, x, y, z]`: the raw algebra plus
  axis-angle / rotation-matrix / ZYX-Euler conversions, `slerp`, and `exp`/`log`.
- `SO2` / `SE2`: 2D rotation and rigid-body transform.
- `SO3` / `SE3`: 3D rotation (wrapping a unit `Quaternion`, which carries the unit-rotation
  invariant) and rigid-body transform.
- `Twist` / `Wrench`: typed spatial velocity and force in the linear-first `[v; ω]` /
  `[force; torque]` ordering, and the algebra that goes with them — see below.
- `SpatialInertia`: a body's mass, the point it balances on, and how hard it is to spin, plus what
  its motion carries.

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
let twist = g.log();
let g2 = SE3::exp(twist);
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

## Spatial algebra

A motion and a force are each six numbers, and almost everything a robot's dynamics needs is one of
three operations on them: reading one of them from a different frame, working out what a moving frame
does to one of them, and asking what a body's mass distribution makes of a motion.

Reading in another frame is the transform itself. An `SE3` that carries points from one frame to
another carries motions and forces too — `act_twist` and `act_wrench` going one way,
`inverse_act_twist` and `inverse_act_wrench` coming back. Nothing builds a 6×6 to do it; the 6×6
forms (`adjoint` for motions, `force_adjoint` for forces) are there when a plain matrix is what you
want.

`Twist::cross` is what one motion does to another read from a frame riding along with it, and
`Twist::cross_wrench` is the same idea for a force. `Twist::dot_wrench` pairs a motion with a force
and gives the rate work is being done.

`SpatialInertia` answers the rest: `momentum` for how much motion a body carries, `bias_wrench` for
what it takes to hold the motion it already has, `kinetic_energy` for the energy in it, and `combined`
for two bodies stuck rigidly together.

```rust
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SE3, SO3, SpatialInertia, Twist};

// A 3 kg body that balances 10 cm above its own origin.
let body = SpatialInertia::from_diagonal_inertia(
    3.0_f64,
    Vector::new([0.0, 0.0, 0.1]),
    Vector::new([0.05, 0.05, 0.08]),
)?;

// Sliding along x while turning about z.
let motion = Twist::new(Vector::new([1.0, 0.0, 0.0]), Vector::new([0.0, 0.0, 2.0]));

// What that motion carries, and the energy in it.
let carried = body.momentum(motion);
let energy = body.kinetic_energy(motion);

// The same body and the same motion, read from a frame half a metre along x.
let elsewhere = SE3::from_parts(SO3::<f64>::identity(), Vector::new([0.5, 0.0, 0.0]));
let moved_body = elsewhere.act_inertia(body);
let moved_motion = elsewhere.act_twist(motion);

// Energy does not care which frame it is read in.
assert!((moved_body.kinetic_energy(moved_motion) - energy).abs() < 1e-12);

// Neither does the momentum, once it is carried across too.
let moved_momentum = elsewhere.act_wrench(carried);
assert!((moved_momentum.to_vector() - moved_body.momentum(moved_motion).to_vector()).norm() < 1e-12);

// Two bodies stuck together.
let whole = body.combined(moved_body);
assert!((whole.mass() - 6.0).abs() < 1e-12);
# Ok::<(), multicalc::CalcError>(())
```

---

[Back to the tutorial index](README.md)
