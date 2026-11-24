# Rigid-body dynamics

Mass properties, free-body state, and dynamics for a single rigid body and for a fixed-base jointed
robot.

- `SpatialInertia`: mass `m`, center of mass `c`, and rotational inertia `I_c` about `c`.
  `inertia_about(p)` gives the inertia about another point by the parallel-axis theorem,
  `I_p = I_c + m·(|d|²·1₃ − d·dᵀ)` with `d = c − p`. Construction is fallible: the mass must be
  positive and finite, and `I_c` must be symmetric with a positive diagonal.
- `FreeJointState`: the pose and twist of a 6-DOF free body.
- `RigidBody`: an inertia and gravity. It gives the accelerations from the orientation, the angular
  rate, and the applied wrench, as an ODE right-hand side or a one-tick step.
- `ArticulatedBody`: a fixed-base jointed robot with mass: a `KinematicTree`, one spatial inertia
  per joint slot, and gravity.
  - `inverse_dynamics`: joint torque for a given acceleration, by the recursive Newton-Euler
    algorithm (RNEA).
  - `forward_dynamics`: acceleration for a given torque, by the articulated-body algorithm (ABA).
  - `joint_space_inertia`: `H(q)`, by the composite-rigid-body algorithm (CRBA).
  - `kinetic_energy` and `potential_energy`, for passivity checks.
  - `generalized_momentum_and_bias`: `p = H(q)·q̇` and its bias `β`, for the momentum observer.
  - Per-joint armature, viscous damping and Coulomb friction come from the model.
- `MomentumObserver`: sensorless contact detection.

`FreeJointState` returns plain arrays:

- `generalized_position`: `[x, y, z, w, qx, qy, qz]`, position first, then a scalar-first
  quaternion, as MuJoCo orders a free joint.
- `generalized_velocity`: `[v; ω]`, linear first, as in the rest of the crate.

```rust
use multicalc::{FreeJointState, SpatialInertia};
use multicalc::{Matrix, SE3, Twist, Vector};

// A 2 kg body with its center of mass at its origin.
let inertia = SpatialInertia::new(
    2.0_f64,
    Vector::new([0.0, 0.0, 0.0]),
    Matrix::from_diagonal([1.0, 1.0, 1.0]),
)?;

// Parallel-axis theorem, 1 m along x: I_yy = 1 + 2·1².
let about_offset = inertia.inertia_about(Vector::new([1.0, 0.0, 0.0]));
assert_eq!(about_offset[(1, 1)], 3.0);

let state = FreeJointState::new(SE3::<f64>::identity(), Twist::zeros());
assert_eq!(state.generalized_position(), [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
# Ok::<(), multicalc::CalcError>(())
```

## Single-body motion

`RigidBody::new` checks that `I_c` is positive-definite and inverts it once. Each later call is a
fixed number of 3×3 products and does not fail.

- The linear velocity and acceleration are in world axes.
- The angular rate, the angular acceleration and the applied wrench are in body axes. The applied
  torque is about the body-frame origin.

This split matches a flight stack: world-frame position estimate, body-frame gyro rates.

`RigidBody::stepped(state, wrench, timestep)` advances one tick by the explicit midpoint rule. It
evaluates the accelerations at the start and at the midpoint, and advances the state with the
midpoint values. It composes the rotation on `SO3`, so the orientation stays a unit quaternion
with no renormalization. The global error is `O(Δt²)`. `Rk4` on `state_derivative` gives
`O(Δt⁴)`, but its quaternion drifts off the unit sphere.

The ODE state is a `Vector<13, T>`: `generalized_position` then `generalized_velocity`.
`state_vector_from_free_joint` and `free_joint_from_state_vector` convert to and from
`FreeJointState`. `free_joint_from_state_vector` normalizes the quaternion and returns `None` if
it is zero.

```rust
use multicalc::dynamics::{RigidBody, state_vector_from_free_joint};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::ode::Rk4;
use multicalc::spatial::{FreeJointState, SE3, SpatialInertia, Twist, Wrench};

// A 0.8 kg quadrotor airframe, I_zz > I_xx = I_yy.
let mass = 0.8_f64;
let balance_point = Vector::new([0.0, 0.0, 0.0]);
let resistance_to_spinning = Matrix::from_diagonal([0.005, 0.005, 0.009]);
let gravity_strength = 9.81;
let earth_gravity = Vector::new([0.0, 0.0, -gravity_strength]);

let inertia = SpatialInertia::new(mass, balance_point, resistance_to_spinning)?;
let body = RigidBody::new(inertia, earth_gravity)?;

// Body-frame thrust equal to the weight, zero torque.
let weight = mass * gravity_strength;
let no_turn = Vector::new([0.0, 0.0, 0.0]);
let carrying_its_own_weight = Wrench::new(Vector::new([0.0, 0.0, weight]), no_turn);

// After 1 s of hover from rest, the altitude is unchanged.
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

`RigidBody::state_derivative` does not fail. If the four quaternion entries of the state do not
normalize, it returns a zero derivative, so an integrator holds the body in place and continues.
`RigidBody::try_state_derivative` returns `DynamicsError::DegenerateOrientation` instead.

```rust
use multicalc::dynamics::RigidBody;
use multicalc::error::DynamicsError;
use multicalc::linear_algebra::Vector;
use multicalc::spatial::{SpatialInertia, Wrench};

let inertia = SpatialInertia::from_diagonal_inertia(
    0.8_f64,
    Vector::new([0.0, 0.0, 0.0]),
    Vector::new([0.005, 0.007, 0.009]),
)?;
let body = RigidBody::new(inertia, Vector::new([0.0, 0.0, -9.81]))?;

// The quaternion entries (indices 3 to 6) are all zero.
let no_direction = Vector::new([
    1.0, -2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.1, -0.2, 0.3, -0.4, 0.5,
]);
assert_eq!(
    body.try_state_derivative(&no_direction, Wrench::zeros()),
    Err(DynamicsError::DegenerateOrientation),
);
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

// Free flight from a tumbling twist, with no applied wrench, for 1 s.
let thrown = Twist::new(Vector::new([1.5, 0.0, 2.0]), Vector::new([7.0, 3.0, 5.0]));
let mut state = FreeJointState::new(SE3::identity(), thrown);
let nothing_applied = Wrench::zeros();

let tick = 0.001;
for _ in 0..1000 {
    state = body.stepped(state, nothing_applied, tick);
}

// The orientation is still a unit quaternion.
let facing = state.pose().rotation().quaternion();
assert!((facing.norm() - 1.0).abs() < 1e-13);
# Ok::<(), multicalc::CalcError>(())
```

## A jointed robot

`ArticulatedBody::new(tree, inertias, gravity)` takes one `Option<SpatialInertia>` per joint slot.
A slot with no mass carries `None`; a weld carries its own inertia. Each inertia is stated in the
frame that `KinematicTreeState::pose` returns for that slot: the body frame after the joint
motion, which is the frame of MJCF `<inertial>`. `new` rejects a floating base.

Joint torques include armature, viscous damping and Coulomb friction from the model. The Coulomb
term is `friction_loss · sign(q̇)`, with `sign(0) = 0`. It is not differentiable at `q̇ = 0`, so
under `Dual` it adds nothing to `∂τ/∂q̇`.

`inverse_dynamics` gives
`τ = H(q)·q̈ + C(q,q̇)·q̇ + G(q) + damping⊙q̇ + friction_loss⊙sign(q̇)`. `joint_space_inertia`
returns the same `H(q)`, with each armature on its diagonal entry. `gravity_torque` gives `G(q)`, and
`bias_torque` gives the torque at `q̈ = 0`. The `_at` forms take a configuration and solve the
forward kinematics first. The plain forms take a solved `KinematicTreeState`.

```rust
use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{SE3, SO3, SpatialInertia};

// Two hinges about y, 1 m apart, each link with its center of mass 0.5 m out.
let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
let link_length = 1.0;
let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([link_length, 0.0, 0.0]));
let tree = KinematicTree::<2, 2, f64>::try_from_joints(
    &[
        Joint::revolute(hinge_axis, SE3::identity()),
        Joint::revolute(hinge_axis, link),
    ],
    &[JointParent::World, JointParent::Joint(0)],
)?;

let mass = 1.5;
let balance_point = Vector::new([0.5, 0.0, 0.0]);
let resistance_to_spinning = Matrix::from_diagonal([0.02, 0.02, 0.02]);
let earth_gravity = Vector::new([0.0, 0.0, -9.81]);

let arm = SpatialInertia::new(mass, balance_point, resistance_to_spinning)?;
let body = ArticulatedBody::new(tree, &[Some(arm), Some(arm)], earth_gravity)?;

let folded = Vector::new([0.3, -0.9]);
let still = Vector::zeros();
let speeding_up = Vector::new([1.0, 0.5]);

// τ = G(q) + H(q)·q̈ at q̇ = 0.
let holding = body.gravity_torque_at(&folded)?;
let driving = body.inverse_dynamics_at(&folded, &still, &speeding_up)?;
let inertial = body.joint_space_inertia_at(&folded)? * speeding_up;
for joint in 0..2 {
    assert!((driving[joint] - holding[joint] - inertial[joint]).abs() < 1e-10);
}
# Ok::<(), multicalc::CalcError>(())
```

`forward_dynamics` inverts `inverse_dynamics`. It is `O(n)` in the joint count and does not form
`H`. `forward_dynamics_with_workspace` takes a caller-owned `DynamicsWorkspace`, for a control
loop that cannot hold a model-sized frame on the stack.

```rust
use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{SE3, SO3, SpatialInertia};

let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));

// Per-joint armature, viscous damping and Coulomb friction loss.
let armature = [0.05, 0.03];
let damping = [0.7, 0.4];
let friction_loss = [0.2, 0.1];
let tree = KinematicTree::<2, 2, f64>::try_from_joints(
    &[
        Joint::revolute(hinge_axis, SE3::identity())
            .with_armature(armature[0])
            .with_damping(damping[0])
            .with_friction_loss(friction_loss[0]),
        Joint::revolute(hinge_axis, link)
            .with_armature(armature[1])
            .with_damping(damping[1])
            .with_friction_loss(friction_loss[1]),
    ],
    &[JointParent::World, JointParent::Joint(0)],
)?;

let mass = 1.5;
let balance_point = Vector::new([0.5, 0.0, 0.0]);
let resistance_to_spinning = Matrix::from_diagonal([0.02, 0.02, 0.02]);
let earth_gravity = Vector::new([0.0, 0.0, -9.81]);

let arm = SpatialInertia::new(mass, balance_point, resistance_to_spinning)?;
let body = ArticulatedBody::new(tree, &[Some(arm), Some(arm)], earth_gravity)?;

let folded = Vector::new([0.3, -0.9]);
let moving = Vector::new([1.1, -0.7]);
let wanted = Vector::new([0.8, 1.4]);

// ABA(RNEA(q̈)) = q̈.
let torque = body.inverse_dynamics_at(&folded, &moving, &wanted)?;
let produced = body.forward_dynamics_at(&folded, &moving, &torque)?;
for joint in 0..2 {
    assert!((produced[joint] - wanted[joint]).abs() < 1e-10);
}
# Ok::<(), multicalc::CalcError>(())
```

With no dissipation, `kinetic_energy + potential_energy` stays constant along a trajectory released
from rest.

```rust
use multicalc::dynamics::ArticulatedBody;
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::ode::Rk4;
use multicalc::spatial::{SE3, SO3, SpatialInertia};

let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
let tree = KinematicTree::<2, 2, f64>::try_from_joints(
    &[
        Joint::revolute(hinge_axis, SE3::identity()),
        Joint::revolute(hinge_axis, link),
    ],
    &[JointParent::World, JointParent::Joint(0)],
)?;

let mass = 1.5;
let balance_point = Vector::new([0.5, 0.0, 0.0]);
let resistance_to_spinning = Matrix::from_diagonal([0.02, 0.02, 0.02]);
let earth_gravity = Vector::new([0.0, 0.0, -9.81]);

let arm = SpatialInertia::new(mass, balance_point, resistance_to_spinning)?;
let body = ArticulatedBody::new(tree, &[Some(arm), Some(arm)], earth_gravity)?;

// The integrator state is [q; q̇].
let total_energy = |state: &Vector<4, f64>| -> Result<f64, multicalc::CalcError> {
    let solved = body.tree().forward_kinematics(&Vector::new([state[0], state[1]]))?;
    let rates = Vector::new([state[2], state[3]]);
    Ok(body.kinetic_energy(&solved, &rates)? + body.potential_energy(&solved)?)
};

let start = Vector::new([1.2, -0.5, 0.0, 0.0]);
let rate = |_time: f64, state: &Vector<4, f64>| {
    let position = Vector::new([state[0], state[1]]);
    let velocity = Vector::new([state[2], state[3]]);
    let acceleration = body
        .forward_dynamics_at(&position, &velocity, &Vector::zeros())
        .unwrap_or_else(|_| Vector::zeros());
    Vector::new([velocity[0], velocity[1], acceleration[0], acceleration[1]])
};

let start_time = 0.0;
let step = 1e-4;
let step_count = 1000;
let after = Rk4::integrate(&rate, start_time, &start, step, step_count, |_time, _state| {});
let held = (total_energy(&after)? - total_energy(&start)?).abs() / total_energy(&start)?.abs();
assert!(held < 1e-8);
# Ok::<(), multicalc::CalcError>(())
```

`multicalc-robot-model` loads an `ArticulatedBody` from a model file. It is not a dependency of
this crate, so this block does not compile here. The `articulated_dynamics` demo runs it.

```rust,ignore
use multicalc::linear_algebra::Vector;

let earth_gravity = Vector::new([0.0, 0.0, -9.81]);
let model = multicalc_robot_model::mjcf::load_path(std::path::Path::new(
    "third_party/menagerie/franka_emika_panda/panda.xml",
))?;
let panda = model.articulated_body::<16, 16>(earth_gravity)?;

// G(q) at q = 0. Slot 0 is the welded base. The arm joints in odd slots have axes parallel to
// gravity, so their torque is zero.
let holding = panda.gravity_torque_at(&Vector::zeros())?;
// holding[2] = -4.0399 N·m, holding[4] = -3.2669 N·m, holding[6] = 2.2997 N·m
```

## Contact detection without sensors

`MomentumObserver` estimates the external joint torque `τ_ext` from `q`, `q̇` and the applied
torque `τ`, with no force sensor and no `q̈`. It integrates `ṗ = τ + τ_ext − β`, with `p = H(q)·q̇`
and `β = G(q) − C(q,q̇)ᵀ·q̇ + damping⊙q̇ + friction_loss⊙sign(q̇)` from
`ArticulatedBody::generalized_momentum_and_bias`. The residual follows `ṙ = K ⊙ (τ_ext − r)`.

- Pass the torque applied over the tick that ends at the measured `(q, q̇)`.
- The first update after `new` or `reset` latches the start state and returns zero.
- A step in `τ_ext` reaches `r` as `1 − (1 + K·dt)^−k` after `k` ticks.
- `sign(0) = 0`, so stiction shows in `r`. Set the contact threshold above the friction loss.

```rust
use multicalc::dynamics::{ArticulatedBody, MomentumObserver};
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::spatial::{SE3, SO3, SpatialInertia};

// The two-hinge arm from above.
let hinge_axis = Vector::new([0.0, 1.0, 0.0]);
let link = SE3::from_parts(SO3::<f64>::identity(), Vector::new([1.0, 0.0, 0.0]));
let tree = KinematicTree::<2, 2, f64>::try_from_joints(
    &[
        Joint::revolute(hinge_axis, SE3::identity()),
        Joint::revolute(hinge_axis, link),
    ],
    &[JointParent::World, JointParent::Joint(0)],
)?;
let arm = SpatialInertia::new(
    1.5,
    Vector::new([0.5, 0.0, 0.0]),
    Matrix::from_diagonal([0.02, 0.02, 0.02]),
)?;
let body = ArticulatedBody::new(tree, &[Some(arm), Some(arm)], Vector::new([0.0, 0.0, -9.81]))?;

// K = 50 rad/s at 1 kHz; contact is |r| > 0.3 N·m on any joint.
let timestep = 0.001;
let mut observer = MomentumObserver::new(Vector::new([50.0, 50.0]), timestep)?;
let contact_threshold = 0.3;

// A driven swing, and an unmodeled push over ticks 500 to 699.
let push = Vector::new([0.8, -0.5]);
let mut position = Vector::new([1.2, -0.5]);
let mut velocity = Vector::zeros();
let mut torque = Vector::zeros();
let mut first_contact = None;
let mut last_contact = None;
for tick in 0..1000 {
    // `torque` acted over the tick that ends here.
    let residual = observer.update_at(&body, &position, &velocity, &torque)?;
    if (0..2).any(|joint| residual[joint].abs() > contact_threshold) {
        first_contact.get_or_insert(tick);
        last_contact = Some(tick);
    }

    let time = tick as f64 * timestep;
    torque = Vector::new([2.0 * (3.0 * time).sin(), (2.0 * time).cos()]);
    let external = if (500..700).contains(&tick) { push } else { Vector::zeros() };
    let acceleration = body.forward_dynamics_at(&position, &velocity, &(torque + external))?;
    velocity = velocity + acceleration.scale(timestep);
    position = position + velocity.scale(timestep);
}

// The flag rises about 10 ms after the push starts and falls about 30 ms after it stops.
assert!(first_contact.is_some_and(|tick| (501..=520).contains(&tick)));
assert!(last_contact.is_some_and(|tick| (700..=750).contains(&tick)));
# Ok::<(), multicalc::CalcError>(())
```

## Errors

- `SpatialInertia::new` returns [`SpatialError`](error-handling.md): `NonPositiveMass`,
  `NonFinite`, `NotSymmetric`, or `NonPositiveInertia`.
- `RigidBody::new` returns [`DynamicsError`](error-handling.md): `NonFinite`,
  `NonPositiveInertia`, or `Linalg`.
- `RigidBody::try_state_derivative` returns `DegenerateOrientation`.
- `RigidBody::accelerations`, `state_derivative` and `stepped` are infallible.
- `ArticulatedBody::new` returns `FloatingBaseUnsupported`, `InertiaCountMismatch`, or
  `NonFinite`.
- `ArticulatedBody` queries return `StateShapeMismatch` (poses solved for a different joint
  count), `NonFinite`, or `Kinematics`. `forward_dynamics` also returns
  `NonPositiveArticulatedInertia`: a movable joint with no inertia behind it, such as a massless
  body on a joint with no armature.
- `MomentumObserver::new` returns `NonFinite`, `NonPositiveTimestep`, or `NonPositiveGain`.
  `update` returns `NonFinite` and the `ArticulatedBody` query errors.

## Model files

The separate `multicalc-robot-model` crate reads rigid bodies from MuJoCo MJCF and URDF files into
`SpatialInertia` and `FreeJointState`. For MJCF, it computes the mass properties from the geoms
where the file does not state them. Its output is checked against the MuJoCo compile of the same
file. Demo:
[mjcf_model_ingestion.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/mjcf_model_ingestion.rs).

[Plant](plant.md) models the rotors that produce the applied wrench. Demos:
[rigid_body_dynamics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/rigid_body_dynamics.rs),
[articulated_dynamics.rs](https://github.com/kmolan/multicalc-rust/blob/main/demos/examples/basics/articulated_dynamics.rs).

---

[Back to the tutorial index](README.md)
