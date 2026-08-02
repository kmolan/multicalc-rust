# Rigid-body dynamics and rotor mixing

`SpatialInertia` says how a body's mass is spread out. `RigidBody` takes that and gravity and
answers the question an integrator keeps asking: given which way the body is facing, how fast it is
turning, and what is pushing on it, how quickly is its motion changing? Everything that can fail —
inverting the body's resistance to spinning — happens once when the body is built, so each later
call is a fixed handful of small products.

Straight-line motion is in world axes, turning is in the body's own axes, and so are the forces
applied to it. That is the same split a flying machine already has: a position estimate in world
axes, a turn rate straight off the gyro.

`MultirotorMixer` sits on the other side of the same loop. Each rotor pushes along the body's z
axis; where it sits decides how much it tips the body, and which way it turns decides how much it
twists it about z. The mixer holds that relation and the way back through it, both worked out
once, so asking for a push and a turn costs one small matrix product. Its answer comes back as a
`Wrench`, which is exactly what `RigidBody` takes — so the two join up with nothing in between.

`RigidBody::stepped` closes the loop the other way: hand it the state, the wrench, and how long the
tick lasts, and it hands back the state a tick later. It reads the accelerations once at the start
of the tick and once half way through and moves the whole state with the half-way values, and it
carries the direction the body faces forward as a turn rather than as four loose numbers — so the
orientation stays a true rotation with nothing to scale back. The error shrinks with the square of
the tick length, coarser than handing `state_derivative` to `Rk4`, which shrinks with the fourth
power but lets the orientation drift.

```rust
use multicalc::dynamics::{RigidBody, state_vector_from_free_joint};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::ode::Rk4;
use multicalc::plant::MultirotorMixer;
use multicalc::spatial::{FreeJointState, SE3, SpatialInertia, Twist};

// A small flying machine: 0.8 kg, harder to spin about its up axis than the other two.
let mass = 0.8_f64;
let balance_point = Vector::new([0.0, 0.0, 0.0]);
let resistance_to_spinning = Matrix::from_diagonal([0.005, 0.005, 0.009]);
let earth_gravity = Vector::new([0.0, 0.0, -9.81]);

let inertia = SpatialInertia::new(mass, balance_point, resistance_to_spinning)?;
let body = RigidBody::new(inertia, earth_gravity)?;

// Four rotors 15 cm out, each twisting the body 1.6 cm-worth per newton of push, able to give
// between nothing and 5 N.
let arm_length = 0.15;
let torque_per_thrust = 0.016;
let minimum_thrust = 0.0;
let maximum_thrust = 5.0;
let mixer = MultirotorMixer::<4, f64>::quadrotor_x(
    arm_length,
    torque_per_thrust,
    minimum_thrust,
    maximum_thrust,
)?;

// Asked to carry its own weight and turn not at all, it shares the push out evenly.
let weight = mass * 9.81;
let no_turn = Vector::new([0.0, 0.0, 0.0]);
let commands = mixer.rotor_thrusts(weight, no_turn);
assert!(!commands.saturated());
for rotor in 0..4 {
    assert!((commands.thrusts()[rotor] - weight / 4.0).abs() < 1e-12);
}

// Those thrusts add back up to the push that was asked for, in the form the body takes.
let wrench = mixer.wrench(commands.thrusts());

// Held there for a second, the machine has not moved.
let at_rest = FreeJointState::new(SE3::identity(), Twist::zeros());
let start = state_vector_from_free_joint(at_rest);
let rate = |_time: f64, state: &Vector<13, f64>| body.state_derivative(state, wrench);

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

Errors: `RigidBody::new` returns [`DynamicsError`](error-handling.md): `NonFinite`,
`NonPositiveInertia`, or `Linalg`. `MultirotorMixer::new` and `quadrotor_x` return
[`PlantError`](error-handling.md): `NonFinite`, `NonPositiveArmLength`, `NonPositiveTorqueRatio`,
`InvalidThrustLimits`, `RotorLayoutNotIndependent`, or `Linalg`. Everything on the per-tick path —
`accelerations`, `state_derivative`, `stepped`, `rotor_thrusts`, and `wrench` — is infallible.

A saturated command does not produce the push and turn that was asked for, because what was left
over has nowhere to go. `RotorCommands::saturated` says when that happened, so an outer loop can
stop winding up against a limit it cannot reach past. Full demo:
[rigid_body_dynamics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/rigid_body_dynamics.rs).


---

[Back to the tutorial index](README.md)
