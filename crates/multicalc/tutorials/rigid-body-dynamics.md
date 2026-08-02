# Rigid-body dynamics

How a single body's mass is spread out, where that body is and how it is moving when nothing holds
it in place, and how quickly its motion changes under the forces on it.

- `SpatialInertia`: a body's mass, the point it balances about, and how it resists being spun about
  that point. `inertia_about` asks the same question about a different reference point — moving away
  from the balance point always makes the body harder to spin, by the mass times how far the point
  moved. Building one is fallible: the mass has to be positive and finite and the resistance has to
  read the same across the diagonal, so a body that cannot exist is rejected at construction.
- `FreeJointState`: the pose and velocity of a body free to move in all six directions.
- `RigidBody`: takes an inertia and gravity and answers the question an integrator keeps asking —
  given which way the body is facing, how fast it is turning, and what is pushing on it, how quickly
  is its motion changing?

Both `SpatialInertia` and `FreeJointState` hand their numbers back as plain arrays rather than
wrapper types, so the conventions matter. The seven place numbers are position first, then
orientation, as `[x, y, z, w, qx, qy, qz]` — matching how MuJoCo writes a free joint. The six motion
numbers are `[v; ω]`, linear first, the ordering the rest of the crate uses.

```rust
use multicalc::{FreeJointState, SpatialInertia};
use multicalc::{Matrix, SE3, Twist, Vector};

// A 2 kg body balancing at its origin.
let inertia = SpatialInertia::new(
    2.0_f64,
    Vector::new([0.0, 0.0, 0.0]),
    Matrix::from_diagonal([1.0, 1.0, 1.0]),
)?;

// Spinning it about a point one metre away is harder, by mass times distance squared.
let about_offset = inertia.inertia_about(Vector::new([1.0, 0.0, 0.0]));
assert_eq!(about_offset[(1, 1)], 3.0);

// Where the body is and how it is moving.
let state = FreeJointState::new(SE3::<f64>::identity(), Twist::zeros());
assert_eq!(state.generalized_position(), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
# Ok::<(), multicalc::CalcError>(())
```

## The body's motion

Everything in `RigidBody` that can fail — inverting the body's resistance to spinning — happens once
when the body is built, so each later call is a fixed handful of small products.

Straight-line motion is in world axes, turning is in the body's own axes, and so are the forces
applied to it. That is the same split a flying machine already has: a position estimate in world
axes, a turn rate straight off the gyro.

`RigidBody::stepped` closes the loop: hand it the state, the wrench, and how long the tick lasts,
and it hands back the state a tick later. It reads the accelerations once at the start of the tick
and once half way through and moves the whole state with the half-way values, and it carries the
direction the body faces forward as a turn rather than as four loose numbers — so the orientation
stays a true rotation with nothing to scale back. The error shrinks with the square of the tick
length, coarser than handing `state_derivative` to `Rk4`, which shrinks with the fourth power but
lets the orientation drift.

```rust
use multicalc::dynamics::{RigidBody, state_vector_from_free_joint};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::ode::Rk4;
use multicalc::spatial::{FreeJointState, SE3, SpatialInertia, Twist, Wrench};

// A small flying machine: 0.8 kg, harder to spin about its up axis than the other two.
let mass = 0.8_f64;
let balance_point = Vector::new([0.0, 0.0, 0.0]);
let resistance_to_spinning = Matrix::from_diagonal([0.005, 0.005, 0.009]);
let gravity_strength = 9.81;
let earth_gravity = Vector::new([0.0, 0.0, -gravity_strength]);

let inertia = SpatialInertia::new(mass, balance_point, resistance_to_spinning)?;
let body = RigidBody::new(inertia, earth_gravity)?;

// Pushed straight up hard enough to carry its own weight, and turned not at all.
let weight = mass * gravity_strength;
let no_turn = Vector::new([0.0, 0.0, 0.0]);
let carrying_its_own_weight = Wrench::new(Vector::new([0.0, 0.0, weight]), no_turn);

// Held there for a second, the machine has not moved.
let at_rest = FreeJointState::new(SE3::identity(), Twist::zeros());
let start = state_vector_from_free_joint(at_rest);
let rate = |_time: f64, state: &Vector<13, f64>| {
    body.state_derivative(state, carrying_its_own_weight)
};

let start_time = 0.0;
let step = 0.001;
let step_count = 1000;
let after = Rk4::integrate(&rate, start_time, &start, step, step_count, |_time, _state| {});
assert!(after[2].abs() < 1e-9);
# Ok::<(), multicalc::CalcError>(())
```

```rust
use multicalc::dynamics::RigidBody;
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{FreeJointState, SE3, SpatialInertia, Twist, Wrench};

let mass = 0.8_f64;
let balance_point = Vector::new([0.0, 0.0, 0.0]);
let resistance_to_spinning = Vector::new([0.005, 0.007, 0.009]);
let earth_gravity = Vector::new([0.0, 0.0, -9.81]);

let inertia =
    SpatialInertia::from_diagonal_inertia(mass, balance_point, resistance_to_spinning)?;
let body = RigidBody::new(inertia, earth_gravity)?;

// Thrown while tumbling, with nothing pushing on it, and followed for a second.
let thrown = Twist::new(Vector::new([1.5, 0.0, 2.0]), Vector::new([7.0, 3.0, 5.0]));
let mut state = FreeJointState::new(SE3::identity(), thrown);
let nothing_applied = Wrench::zeros();

let tick = 0.001;
for _ in 0..1000 {
    state = body.stepped(state, nothing_applied, tick);
}

// A second of tumbling and the direction it faces is still a true rotation.
let facing = state.pose().rotation().quaternion();
assert!((facing.norm() - 1.0).abs() < 1e-13);
# Ok::<(), multicalc::CalcError>(())
```

Errors: `SpatialInertia::new` returns [`SpatialError`](error-handling.md): `NonPositiveMass`,
`NonFinite`, `NotSymmetric`, or `NonPositiveInertia`. `RigidBody::new` returns
[`DynamicsError`](error-handling.md): `NonFinite`, `NonPositiveInertia`, or `Linalg`. Everything on
the per-tick path — `accelerations`, `state_derivative`, and `stepped` — is infallible.

`SpatialInertia` and `FreeJointState` are what a model file loads into. The separate
`multicalc-mjcf` crate reads one rigid body out of a MuJoCo MJCF file — working its mass out from
the shapes it is built from where the file does not state it — and is checked against MuJoCo's own
compile of the same file. Full demo:
[model_ingestion.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/model_ingestion.rs).

The wrench a body is handed usually comes from somewhere — see [Plant](plant.md) for the rotors
that produce it. Full demo:
[rigid_body_dynamics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/rigid_body_dynamics.rs).


---

[Back to the tutorial index](README.md)
