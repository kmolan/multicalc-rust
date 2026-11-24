//! The Franka Panda checks: the eleven-slot model of `dynamics/articulated_franka_panda` against
//! Pinocchio, at state 1.

use core::hint::black_box;

use multicalc::control::{ComputedTorqueController, JointReference};
use multicalc::dynamics::{ArticulatedBody, MomentumObserver};
use multicalc::kinematics::{Joint, JointParent, KinematicTree};
use multicalc::linear_algebra::{Matrix, Vector};
use multicalc::scalar::Numeric;
use multicalc::spatial::{Quaternion, SE3, SO3, SpatialInertia};

use super::{CheckOutcome, close, failed};
use crate::cycles;
use crate::fixtures;

/// The base weld, seven revolute joints, the hand weld and two prismatic fingers.
const FRANKA_SLOTS: usize = 11;

/// The gains and the tick of the `3d_arm_torque` showcase.
const NATURAL_FREQUENCY: f32 = 20.0;
const DAMPING_RATIO: f32 = 1.0;
const OBSERVER_GAIN: f32 = 50.0;
const TICK: f32 = 1e-3;

/// The fixture model in `T`. With `friction` off, damping and friction loss are zero: the model
/// of Pinocchio's `rnea`, `crba` and `aba`.
fn franka_body<T: Numeric>(
    friction: bool,
) -> Option<ArticulatedBody<FRANKA_SLOTS, FRANKA_SLOTS, T>> {
    let vector3 = |row: [f64; 3]| Vector::new(row.map(T::from_f64));
    let mut joints = [Joint::fixed(SE3::identity()); FRANKA_SLOTS];
    let mut parents = [JointParent::World; FRANKA_SLOTS];
    let mut inertias = [None; FRANKA_SLOTS];
    for (slot, kind) in fixtures::FRANKA_DYNAMICS_JOINT_KINDS
        .bytes()
        .take(FRANKA_SLOTS)
        .enumerate()
    {
        let [w, x, y, z] = fixtures::FRANKA_DYNAMICS_ORIGIN_QUATERNIONS[slot].map(T::from_f64);
        let rotation = SO3::from_quaternion(Quaternion::new(w, x, y, z).try_normalized()?);
        let origin = SE3::from_parts(
            rotation,
            vector3(fixtures::FRANKA_DYNAMICS_ORIGIN_POSITIONS[slot]),
        );
        let axis = vector3(fixtures::FRANKA_DYNAMICS_AXES[slot]);
        let joint = match kind {
            b'R' => Joint::revolute(axis, origin)
                .with_anchor(vector3(fixtures::FRANKA_DYNAMICS_ANCHORS[slot])),
            b'P' => Joint::prismatic(axis, origin),
            _ => Joint::fixed(origin),
        };
        let (damping, friction_loss) = if friction {
            (
                fixtures::FRANKA_DYNAMICS_DAMPINGS[slot],
                fixtures::FRANKA_DYNAMICS_FRICTION_LOSSES[slot],
            )
        } else {
            (0.0, 0.0)
        };
        joints[slot] = joint
            .with_zero_offset(T::from_f64(fixtures::FRANKA_DYNAMICS_ZERO_OFFSETS[slot]))
            .with_armature(T::from_f64(fixtures::FRANKA_DYNAMICS_ARMATURES[slot]))
            .with_damping(T::from_f64(damping))
            .with_friction_loss(T::from_f64(friction_loss));
        let parent = fixtures::FRANKA_DYNAMICS_PARENTS[slot];
        if parent >= 0.0 {
            parents[slot] = JointParent::Joint(parent as usize);
        }
        let mass = fixtures::FRANKA_DYNAMICS_MASSES[slot];
        if mass > 0.0 {
            let rotational = fixtures::FRANKA_DYNAMICS_ROTATIONAL_INERTIAS[slot];
            inertias[slot] = Some(
                SpatialInertia::new(
                    T::from_f64(mass),
                    vector3(fixtures::FRANKA_DYNAMICS_CENTERS_OF_MASS[slot]),
                    Matrix::from_fn(|row, column| T::from_f64(rotational[row * 3 + column])),
                )
                .ok()?,
            );
        }
    }
    let tree = KinematicTree::try_from_joints(&joints, &parents).ok()?;
    let gravity = Vector::new([T::ZERO, T::ZERO, T::from_f64(-9.81)]);
    ArticulatedBody::new(tree, &inertias, gravity).ok()
}

/// One state row of the fixture in `T`.
fn franka_row<T: Numeric>(row: [f64; FRANKA_SLOTS]) -> Vector<FRANKA_SLOTS, T> {
    Vector::new(row.map(T::from_f64))
}

/// Golden: RNEA of the Franka at state 1 against Pinocchio's `rnea`.
pub fn franka_inverse_dynamics() -> CheckOutcome {
    const NAME: &str = "franka_inverse_dynamics";
    let Some(body) = franka_body::<f64>(false) else {
        return failed(NAME, 0);
    };
    let body = black_box(body);
    let position = black_box(franka_row::<f64>(fixtures::FRANKA_DYNAMICS_POSITIONS));
    let velocity = black_box(franka_row::<f64>(fixtures::FRANKA_DYNAMICS_VELOCITIES));
    let acceleration = black_box(franka_row::<f64>(fixtures::FRANKA_DYNAMICS_ACCELERATIONS));

    let start = cycles::read();
    let torque = black_box(body.inverse_dynamics_at(&position, &velocity, &acceleration));
    let elapsed = cycles::read().wrapping_sub(start);

    let Ok(torque) = torque else {
        return failed(NAME, elapsed);
    };
    let passed = (0..FRANKA_SLOTS).all(|slot| {
        close(
            torque[slot],
            fixtures::FRANKA_DYNAMICS_RIGID_BODY_TORQUES[slot],
            fixtures::FRANKA_DYNAMICS_ABS,
            fixtures::FRANKA_DYNAMICS_REL,
        )
    });
    CheckOutcome {
        name: NAME,
        passed,
        cycles: elapsed,
        value: torque[2],
    }
}

/// Golden: CRBA of the Franka at state 1 against Pinocchio's `crba`.
pub fn franka_mass_matrix() -> CheckOutcome {
    const NAME: &str = "franka_mass_matrix";
    let Some(body) = franka_body::<f64>(false) else {
        return failed(NAME, 0);
    };
    let body = black_box(body);
    let position = black_box(franka_row::<f64>(fixtures::FRANKA_DYNAMICS_POSITIONS));

    let start = cycles::read();
    let inertia = black_box(body.joint_space_inertia_at(&position));
    let elapsed = cycles::read().wrapping_sub(start);

    let Ok(inertia) = inertia else {
        return failed(NAME, elapsed);
    };
    let passed = (0..FRANKA_SLOTS).all(|row| {
        (0..FRANKA_SLOTS).all(|column| {
            close(
                inertia[(row, column)],
                fixtures::FRANKA_DYNAMICS_JOINT_SPACE_INERTIA[row][column],
                fixtures::FRANKA_DYNAMICS_ABS,
                fixtures::FRANKA_DYNAMICS_REL,
            )
        })
    });
    CheckOutcome {
        name: NAME,
        passed,
        cycles: elapsed,
        value: inertia[(1, 1)],
    }
}

/// Golden: ABA of the Franka at state 1, given Pinocchio's rigid-body torques, against
/// Pinocchio's `aba`.
pub fn franka_forward_dynamics() -> CheckOutcome {
    const NAME: &str = "franka_forward_dynamics";
    let Some(body) = franka_body::<f64>(false) else {
        return failed(NAME, 0);
    };
    let body = black_box(body);
    let position = black_box(franka_row::<f64>(fixtures::FRANKA_DYNAMICS_POSITIONS));
    let velocity = black_box(franka_row::<f64>(fixtures::FRANKA_DYNAMICS_VELOCITIES));
    let torque = black_box(franka_row::<f64>(
        fixtures::FRANKA_DYNAMICS_RIGID_BODY_TORQUES,
    ));

    let start = cycles::read();
    let acceleration = black_box(body.forward_dynamics_at(&position, &velocity, &torque));
    let elapsed = cycles::read().wrapping_sub(start);

    let Ok(acceleration) = acceleration else {
        return failed(NAME, elapsed);
    };
    let passed = (0..FRANKA_SLOTS).all(|slot| {
        close(
            acceleration[slot],
            fixtures::FRANKA_DYNAMICS_FORWARD_ACCELERATIONS[slot],
            fixtures::FRANKA_DYNAMICS_ABS,
            fixtures::FRANKA_DYNAMICS_REL,
        )
    });
    CheckOutcome {
        name: NAME,
        passed,
        cycles: elapsed,
        value: acceleration[1],
    }
}

/// Identity: one 1 kHz computed-torque tick of the Franka with the momentum observer, in `f32`.
/// At zero error `τ` is RNEA of the reference (the fixture's `τ`). The arm does not move, so
/// `τ_ext = β − τ`, and the first tick after the latch gives `r = K·dt/(1 + K·dt) ⊙ τ_ext`.
pub fn arm_control_tick_f32() -> CheckOutcome {
    const NAME: &str = "arm_control_tick_f32";
    let Some(body) = franka_body::<f32>(true) else {
        return failed(NAME, 0);
    };
    let (Ok(controller), Ok(mut observer)) = (
        ComputedTorqueController::<FRANKA_SLOTS, f32>::from_natural_frequency(
            NATURAL_FREQUENCY,
            DAMPING_RATIO,
        ),
        MomentumObserver::<FRANKA_SLOTS, f32>::new(Vector::from_fn(|_| OBSERVER_GAIN), TICK),
    ) else {
        return failed(NAME, 0);
    };
    let position = black_box(franka_row::<f32>(fixtures::FRANKA_DYNAMICS_POSITIONS));
    let velocity = black_box(franka_row::<f32>(fixtures::FRANKA_DYNAMICS_VELOCITIES));
    let acceleration = franka_row::<f32>(fixtures::FRANKA_DYNAMICS_ACCELERATIONS);
    let reference = JointReference::new(position, velocity, acceleration);
    // The first update latches `p₀` and `β₀`.
    if observer
        .update_at(&body, &position, &velocity, &Vector::zeros())
        .is_err()
    {
        return failed(NAME, 0);
    }

    let start = cycles::read();
    let tick = body
        .tree()
        .forward_kinematics(&position)
        .ok()
        .and_then(|state| {
            let torque = controller
                .torque(&body, &state, &position, &velocity, &reference)
                .ok()?;
            let residual = observer.update(&body, &state, &velocity, &torque).ok()?;
            Some((torque, residual))
        });
    let tick = black_box(tick);
    let elapsed = cycles::read().wrapping_sub(start);

    let Some((torque, residual)) = tick else {
        return failed(NAME, elapsed);
    };
    let gain_step = f64::from(OBSERVER_GAIN) * f64::from(TICK);
    let factor = gain_step / (1.0 + gain_step);
    let passed = (0..FRANKA_SLOTS).all(|slot| {
        let applied = fixtures::FRANKA_DYNAMICS_TORQUES[slot];
        let external = factor * (fixtures::FRANKA_DYNAMICS_MOMENTUM_BIASES[slot] - applied);
        close(
            f64::from(torque[slot]),
            applied,
            fixtures::FRANKA_DYNAMICS_F32_ABS,
            fixtures::FRANKA_DYNAMICS_F32_REL,
        ) && close(
            f64::from(residual[slot]),
            external,
            fixtures::FRANKA_DYNAMICS_F32_ABS,
            fixtures::FRANKA_DYNAMICS_F32_REL,
        )
    });
    CheckOutcome {
        name: NAME,
        passed,
        cycles: elapsed,
        value: f64::from(residual[2]),
    }
}
